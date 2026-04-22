from __future__ import annotations

import os

import numpy as np

from ....timebase import TimeBase

try:
    import numba
except Exception:  # pragma: no cover
    numba = None


def _read_float_env(name: str, default: float) -> float:
    try:
        v = float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)
    if not bool(np.isfinite(v)):
        return float(default)
    return float(v)


def _read_int_env(name: str, default: int) -> int:
    try:
        v = int(str(os.environ.get(name, str(default))).strip())
    except Exception:
        return int(default)
    return int(v)


def _require_numba() -> None:
    if numba is None:
        raise RuntimeError("n112 requires numba, but import failed")


def _try_build_n112_kernel():
    _require_numba()

    @numba.njit(cache=True)
    def _kernel(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        width: int,
        height: int,
        radius_large_px: int,
        radius_small_px: int,
        tau_ticks: int,
        purity_thresh: float,
        out: np.ndarray,
    ) -> None:
        n = int(t.shape[0])
        npx = int(width) * int(height)

        last_ts = np.zeros((npx,), dtype=np.uint64)
        last_pol = np.zeros((npx,), dtype=np.int8)

        rr_large = int(radius_large_px)
        if rr_large < 0:
            rr_large = 0
        if rr_large > 8:
            rr_large = 8

        rr_small = int(radius_small_px)
        if rr_small < 0:
            rr_small = 0
        if rr_small > rr_large:
            rr_small = rr_large

        tau = int(tau_ticks)
        if tau <= 0:
            tau = 1
        inv_tau = 1.0 / float(tau)

        thresh = float(purity_thresh)
        if thresh <= 1e-3:
            thresh = 1e-3

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            if xi < 0 or xi >= width or yi < 0 or yi >= height:
                out[i] = np.float32(0.0)
                continue

            ti = np.uint64(t[i])
            pi = 1 if int(p[i]) > 0 else -1

            x0 = xi - rr_large
            if x0 < 0:
                x0 = 0
            x1 = xi + rr_large
            if x1 >= width:
                x1 = width - 1
            y0 = yi - rr_large
            if y0 < 0:
                y0 = 0
            y1 = yi + rr_large
            if y1 >= height:
                y1 = height - 1

            sector_same_large = np.zeros((4,), dtype=np.float64)
            e_small_same = 0.0
            e_small_opp = 0.0

            for yy in range(y0, y1 + 1):
                base = yy * width
                for xx in range(x0, x1 + 1):
                    if xx == xi and yy == yi:
                        continue

                    idx_nb = base + xx
                    ts = last_ts[idx_nb]
                    if ts == 0:
                        continue
                    if ti <= ts:
                        continue

                    dt_ticks = int(ti - ts)
                    if dt_ticks <= 0 or dt_ticks > tau:
                        continue

                    w_age = 1.0 - float(dt_ticks) * inv_tau
                    if w_age <= 0.0:
                        continue

                    dx = xx - xi
                    if dx < 0:
                        dx = -dx
                    dy = yy - yi
                    if dy < 0:
                        dy = -dy
                    in_small = (dx <= rr_small and dy <= rr_small)

                    pj = int(last_pol[idx_nb])
                    if pj == pi:
                        k = 0
                        if yy > yi:
                            k += 2
                        if xx > xi:
                            k += 1
                        sector_same_large[k] += w_age
                        if in_small:
                            e_small_same += w_age
                    else:
                        if in_small:
                            e_small_opp += w_age

            score_base = 0.0
            for k in range(4):
                score_base += sector_same_large[k]

            sum_small = e_small_same + e_small_opp
            mult = 1.0
            if sum_small > 1e-3:
                purity = e_small_same / sum_small
                if purity < thresh:
                    mult = purity / thresh

            out[i] = np.float32(score_base * mult)

            idx0 = yi * width + xi
            last_ts[idx0] = ti
            last_pol[idx0] = np.int8(pi)

    return _kernel


def score_stream_n112(
    ev,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_us: int,
    tb: TimeBase,
    scores_out: np.ndarray | None = None,
) -> np.ndarray:
    """N112 (7.59-3): dual-scale polarity purity gate."""

    n = int(ev.t.shape[0])
    if scores_out is None:
        scores_out = np.empty((n,), dtype=np.float32)
    if n <= 0:
        return scores_out

    tau_ticks = int(tb.us_to_ticks(int(tau_us)))
    radius_small_px = int(_read_int_env("MYEVS_N112_RSMALL", 2))
    purity_thresh = float(_read_float_env("MYEVS_N112_THRESH", 0.6))

    ker = _try_build_n112_kernel()
    ker(
        ev.t.astype(np.uint64, copy=False),
        ev.x.astype(np.int32, copy=False),
        ev.y.astype(np.int32, copy=False),
        ev.p,
        int(width),
        int(height),
        int(radius_px),
        int(radius_small_px),
        int(tau_ticks),
        float(purity_thresh),
        scores_out,
    )
    return scores_out
