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


def _require_numba() -> None:
    if numba is None:
        raise RuntimeError("n99 requires numba, but import failed")


def _try_build_n99_kernel():
    _require_numba()

    @numba.njit(cache=True)
    def _kernel(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        width: int,
        height: int,
        radius_px: int,
        tau_ticks: int,
        tick_us: float,
        d0_px: float,
        t0_us: float,
        sigma_d_px: float,
        sigma_t_us: float,
        out: np.ndarray,
    ) -> None:
        n = int(t.shape[0])
        npx = int(width) * int(height)

        pos_ts = np.zeros((npx,), dtype=np.uint64)
        neg_ts = np.zeros((npx,), dtype=np.uint64)

        rr = int(radius_px)
        if rr < 0:
            rr = 0
        if rr > 8:
            rr = 8

        tau = int(tau_ticks)
        if tau <= 0:
            tau = 1

        d0 = float(d0_px)
        t0 = float(t0_us)
        sd = float(sigma_d_px)
        st = float(sigma_t_us)
        if sd <= 0.0:
            sd = 1e-6
        if st <= 0.0:
            st = 1.0

        inv_2sd2 = 1.0 / (2.0 * sd * sd)
        inv_2st2 = 1.0 / (2.0 * st * st)

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            if xi < 0 or xi >= width or yi < 0 or yi >= height:
                out[i] = np.float32(0.0)
                continue

            ti = np.uint64(t[i])
            pi = 1 if int(p[i]) > 0 else -1

            x0 = xi - rr
            if x0 < 0:
                x0 = 0
            x1 = xi + rr
            if x1 >= width:
                x1 = width - 1
            y0 = yi - rr
            if y0 < 0:
                y0 = 0
            y1 = yi + rr
            if y1 >= height:
                y1 = height - 1

            # Track top-3 nearest-in-time same-polarity supports.
            bdt0 = 2147483647
            bdt1 = 2147483647
            bdt2 = 2147483647
            bdx0 = 0.0
            bdx1 = 0.0
            bdx2 = 0.0
            bdy0 = 0.0
            bdy1 = 0.0
            bdy2 = 0.0

            for yy in range(y0, y1 + 1):
                base = yy * width
                dyf = float(yi - yy)
                for xx in range(x0, x1 + 1):
                    if xx == xi and yy == yi:
                        continue

                    idx_nb = base + xx
                    ts = np.uint64(0)
                    if pi > 0:
                        ts = pos_ts[idx_nb]
                    else:
                        ts = neg_ts[idx_nb]
                    if ts == 0:
                        continue
                    if ti <= ts:
                        continue

                    dt_ticks = int(ti - ts)
                    if dt_ticks <= 0 or dt_ticks > tau:
                        continue

                    dx = float(xi - xx)
                    if dt_ticks < bdt0:
                        bdt2 = bdt1
                        bdx2 = bdx1
                        bdy2 = bdy1
                        bdt1 = bdt0
                        bdx1 = bdx0
                        bdy1 = bdy0
                        bdt0 = dt_ticks
                        bdx0 = dx
                        bdy0 = dyf
                    elif dt_ticks < bdt1:
                        bdt2 = bdt1
                        bdx2 = bdx1
                        bdy2 = bdy1
                        bdt1 = dt_ticks
                        bdx1 = dx
                        bdy1 = dyf
                    elif dt_ticks < bdt2:
                        bdt2 = dt_ticks
                        bdx2 = dx
                        bdy2 = dyf

            score = 0.0

            if bdt0 < 2147483647:
                d = np.sqrt(bdx0 * bdx0 + bdy0 * bdy0)
                dt_us = float(bdt0) * tick_us
                dd = d - d0
                dt = dt_us - t0
                g = np.exp(-(dd * dd) * inv_2sd2 - (dt * dt) * inv_2st2)
                if g > score:
                    score = g

            if bdt1 < 2147483647:
                d = np.sqrt(bdx1 * bdx1 + bdy1 * bdy1)
                dt_us = float(bdt1) * tick_us
                dd = d - d0
                dt = dt_us - t0
                g = np.exp(-(dd * dd) * inv_2sd2 - (dt * dt) * inv_2st2)
                if g > score:
                    score = g

            if bdt2 < 2147483647:
                d = np.sqrt(bdx2 * bdx2 + bdy2 * bdy2)
                dt_us = float(bdt2) * tick_us
                dd = d - d0
                dt = dt_us - t0
                g = np.exp(-(dd * dd) * inv_2sd2 - (dt * dt) * inv_2st2)
                if g > score:
                    score = g

            out[i] = np.float32(score)

            idx0 = yi * width + xi
            if pi > 0:
                pos_ts[idx0] = ti
            else:
                neg_ts[idx0] = ti

    return _kernel


def score_stream_n99(
    ev,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_us: int,
    tb: TimeBase,
    scores_out: np.ndarray | None = None,
) -> np.ndarray:
    """N99: top3 2D (d, dt) compatibility with max aggregation."""

    n = int(ev.t.shape[0])
    if scores_out is None:
        scores_out = np.empty((n,), dtype=np.float32)
    if n <= 0:
        return scores_out

    tau_ticks = int(tb.us_to_ticks(int(tau_us)))
    tick_us = float(tb.ticks_to_us(1))

    d0_px = float(_read_float_env("MYEVS_N99_D0_PX", 2.2))
    t0_us = float(_read_float_env("MYEVS_N99_T0_US", 1400.0))
    sigma_d_px = float(_read_float_env("MYEVS_N99_SIGMA_D_PX", 1.2))
    sigma_t_us = float(_read_float_env("MYEVS_N99_SIGMA_T_US", 5000.0))

    ker = _try_build_n99_kernel()
    ker(
        ev.t.astype(np.uint64, copy=False),
        ev.x.astype(np.int32, copy=False),
        ev.y.astype(np.int32, copy=False),
        ev.p,
        int(width),
        int(height),
        int(radius_px),
        int(tau_ticks),
        float(tick_us),
        float(d0_px),
        float(t0_us),
        float(sigma_d_px),
        float(sigma_t_us),
        scores_out,
    )
    return scores_out
