from __future__ import annotations

from typing import Tuple

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


def _build_stc_kernel():
    njit = _require_numba()

    @njit(cache=True)
    def stc_keep_mask(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        width: int,
        height: int,
        show_on: bool,
        show_off: bool,
        r: int,
        need: int,
        win_ticks: int,
        last_on: np.ndarray,
        last_off: np.ndarray,
    ) -> np.ndarray:
        n = t.shape[0]
        keep = np.zeros((n,), dtype=np.uint8)

        w = int(width)
        h = int(height)
        rr = int(r)
        if rr < 0:
            rr = 0
        if rr > 8:
            rr = 8

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            ti = int(t[i])
            pi = 1 if int(p[i]) > 0 else -1

            # Bounds check
            if xi < 0 or xi >= w or yi < 0 or yi >= h:
                continue

            # Polarity visibility
            if pi > 0:
                if not show_on:
                    continue
                last = last_on
            else:
                if not show_off:
                    continue
                last = last_off

            idx0 = yi * w + xi

            # IMPORTANT: always update, even if we drop
            last[idx0] = np.uint64(ti)

            # Fast paths match Python version
            if need <= 1:
                keep[i] = 1
                continue
            if win_ticks <= 0:
                continue

            t0 = ti - win_ticks if ti > win_ticks else 0

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

            cnt = 0
            for yy in range(y0, y1 + 1):
                base = yy * w
                for xx in range(x0, x1 + 1):
                    ts = int(last[base + xx])
                    if ts != 0 and ts >= t0:
                        cnt += 1
                        if cnt >= need:
                            keep[i] = 1
                            break
                if keep[i] == 1:
                    break

        return keep

    return stc_keep_mask


_STC_KEEP_MASK = None


def stc_keep_mask_numba(
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
    min_neighbors: int,
    win_ticks: int,
    last_on: np.ndarray,
    last_off: np.ndarray,
) -> np.ndarray:
    """Numba-accelerated STC keep mask.

    Returns uint8 keep mask (0/1) and updates last_on/last_off in place.
    """

    global _STC_KEEP_MASK
    if _STC_KEEP_MASK is None:
        _STC_KEEP_MASK = _build_stc_kernel()

    return _STC_KEEP_MASK(
        t,
        x,
        y,
        p,
        int(width),
        int(height),
        bool(show_on),
        bool(show_off),
        int(radius_px),
        int(min_neighbors),
        int(win_ticks),
        last_on,
        last_off,
    )


def stc_state_init(width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    n = int(width) * int(height)
    return np.zeros((n,), dtype=np.uint64), np.zeros((n,), dtype=np.uint64)
