from __future__ import annotations

import numpy as np


def is_numba_available() -> bool:
    try:
        import numba  # noqa: F401

        return True
    except Exception:
        return False


def _require_numba():
    try:
        from numba import njit  # type: ignore

        return njit
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Numba is not available. Install it (conda-forge: numba) or use --engine python. "
            f"({type(e).__name__}: {e})"
        )


def _build_pfd_kernel():
    njit = _require_numba()

    @njit(cache=True)
    def pfd_keep_mask(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        width: int,
        height: int,
        show_on: bool,
        show_off: bool,
        radius_px: int,
        win_ticks: int,
        min_neighbors: float,
        stage1_var: int,
        mode_b: bool,
        last_on: np.ndarray,
        last_off: np.ndarray,
        last_pol: np.ndarray,
        last_evt: np.ndarray,
        flip_buf: np.ndarray,
        flip_head: np.ndarray,
        flip_count: np.ndarray,
    ) -> np.ndarray:
        n = int(t.shape[0])
        keep = np.zeros((n,), dtype=np.uint8)
        w = int(width)
        h = int(height)

        rr = int(radius_px)
        if rr < 1:
            rr = 1
        if rr > 8:
            rr = 8

        win = int(win_ticks)
        if win <= 0:
            win = 1

        neigh_thr = float(min_neighbors)
        var = int(stage1_var)
        maxn = (2 * rr + 1) * (2 * rr + 1) - 1
        if var < 1:
            var = 1
        if var > maxn:
            var = maxn

        fifo_size = int(flip_buf.shape[1])

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            if xi < 0 or xi >= w or yi < 0 or yi >= h:
                continue

            pi = 1 if int(p[i]) > 0 else -1
            if pi > 0:
                if not show_on:
                    continue
            else:
                if not show_off:
                    continue

            ti = int(t[i])
            idx0 = yi * w + xi

            # polarity flip update
            lp = int(last_pol[idx0])
            if lp != 0 and lp != pi:
                c = int(flip_count[idx0])
                h0 = int(flip_head[idx0])
                if c < fifo_size:
                    pos = (h0 + c) % fifo_size
                    flip_buf[idx0, pos] = np.uint64(ti)
                    flip_count[idx0] = np.int32(c + 1)
                else:
                    flip_buf[idx0, h0] = np.uint64(ti)
                    flip_head[idx0] = np.int32((h0 + 1) % fifo_size)
            last_pol[idx0] = np.int8(pi)
            last_evt[idx0] = np.uint64(ti)

            # stage-1
            if pi > 0:
                last_on[idx0] = np.uint64(ti)
                same = last_on
            else:
                last_off[idx0] = np.uint64(ti)
                same = last_off

            y0 = yi - rr
            if y0 < 0:
                y0 = 0
            y1 = yi + rr
            if y1 >= h:
                y1 = h - 1
            x0 = xi - rr
            if x0 < 0:
                x0 = 0
            x1 = xi + rr
            if x1 >= w:
                x1 = w - 1

            support = 0
            for yy in range(y0, y1 + 1):
                base = yy * w
                for xx in range(x0, x1 + 1):
                    if xx == xi and yy == yi:
                        continue
                    ts = int(same[base + xx])
                    if ts == 0:
                        continue
                    dt = ti - ts
                    if dt < 0:
                        dt = -dt
                    if dt < win:
                        support += 1

            if support < var:
                continue

            # current flip count
            cur_flip = 0
            c0 = int(flip_count[idx0])
            h0 = int(flip_head[idx0])
            for k in range(c0):
                pos = (h0 + k) % fifo_size
                ft = int(flip_buf[idx0, pos])
                if ft == 0:
                    continue
                dt = ti - ft
                if dt < 0:
                    dt = -dt
                if dt <= win:
                    cur_flip += 1

            neigh_active = 0
            neigh_flip_sum = 0
            for yy in range(y0, y1 + 1):
                base = yy * w
                for xx in range(x0, x1 + 1):
                    if xx == xi and yy == yi:
                        continue
                    idx = base + xx
                    tev = int(last_evt[idx])
                    if tev != 0:
                        dt = ti - tev
                        if dt < 0:
                            dt = -dt
                        if dt <= win:
                            neigh_active += 1

                    cc = int(flip_count[idx])
                    hh = int(flip_head[idx])
                    for k in range(cc):
                        pos = (hh + k) % fifo_size
                        ft = int(flip_buf[idx, pos])
                        if ft == 0:
                            continue
                        dt = ti - ft
                        if dt < 0:
                            dt = -dt
                        if dt <= win:
                            neigh_flip_sum += 1

            if float(neigh_active) <= neigh_thr:
                continue

            neigh_flip_mean = float(neigh_flip_sum) / float(neigh_active)
            if mode_b:
                score = abs(neigh_flip_mean)
            else:
                score = abs(float(cur_flip) - neigh_flip_mean)
            if score <= 1.0:
                keep[i] = 1

        return keep

    return pfd_keep_mask


_PFD_KEEP_MASK = None


def pfd_keep_mask_numba(
    *,
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    width: int,
    height: int,
    show_on: bool,
    show_off: bool,
    radius_px: int,
    win_ticks: int,
    min_neighbors: float,
    stage1_var: int,
    mode_b: bool,
    last_on: np.ndarray,
    last_off: np.ndarray,
    last_pol: np.ndarray,
    last_evt: np.ndarray,
    flip_buf: np.ndarray,
    flip_head: np.ndarray,
    flip_count: np.ndarray,
) -> np.ndarray:
    global _PFD_KEEP_MASK
    if _PFD_KEEP_MASK is None:
        _PFD_KEEP_MASK = _build_pfd_kernel()

    return _PFD_KEEP_MASK(
        t,
        x,
        y,
        p,
        int(width),
        int(height),
        bool(show_on),
        bool(show_off),
        int(radius_px),
        int(win_ticks),
        float(min_neighbors),
        int(stage1_var),
        bool(mode_b),
        last_on,
        last_off,
        last_pol,
        last_evt,
        flip_buf,
        flip_head,
        flip_count,
    )


def pfd_state_init(width: int, height: int, fifo_size: int = 5):
    n = int(width) * int(height)
    return (
        np.zeros((n,), dtype=np.uint64),  # last_on
        np.zeros((n,), dtype=np.uint64),  # last_off
        np.zeros((n,), dtype=np.int8),  # last_pol
        np.zeros((n,), dtype=np.uint64),  # last_evt
        np.zeros((n, int(fifo_size)), dtype=np.uint64),  # flip_buf
        np.zeros((n,), dtype=np.int32),  # flip_head
        np.zeros((n,), dtype=np.int32),  # flip_count
    )

