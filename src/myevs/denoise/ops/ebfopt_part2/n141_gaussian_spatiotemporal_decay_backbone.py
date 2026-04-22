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
        raise RuntimeError("n141 requires numba, but import failed")


def _try_build_n141_kernel():
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
        tau_base_ticks: int,
        sigma_space: float,
        sigma_time_ratio: float,
        out: np.ndarray,
    ) -> None:
        n = int(t.shape[0])
        npx = int(width) * int(height)

        last_ts = np.zeros((npx,), dtype=np.uint64)
        last_pol = np.zeros((npx,), dtype=np.int8)

        rr = int(radius_px)
        if rr < 0:
            rr = 0
        if rr > 8:
            rr = 8

        tau_base = int(tau_base_ticks)
        if tau_base <= 0:
            tau_base = 1

        sig_s = float(sigma_space)
        if sig_s <= 1e-6:
            sig_s = 1e-6
        inv_2sig_s2 = 1.0 / (2.0 * sig_s * sig_s)

        tsr = float(sigma_time_ratio)
        if tsr <= 1e-6:
            tsr = 1e-6
        sig_t = float(tau_base) * tsr
        if sig_t <= 1e-6:
            sig_t = 1e-6
        inv_2sig_t2 = 1.0 / (2.0 * sig_t * sig_t)

        # Truncate very old neighbors for speed; 4*sigma keeps >99.99% Gaussian mass.
        dt_cut = int(4.0 * sig_t)
        if dt_cut < 1:
            dt_cut = 1

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

            score = 0.0

            for yy in range(y0, y1 + 1):
                base = yy * width
                dy = yy - yi
                for xx in range(x0, x1 + 1):
                    if xx == xi and yy == yi:
                        continue

                    idx_nb = base + xx
                    if int(last_pol[idx_nb]) != pi:
                        continue

                    ts = last_ts[idx_nb]
                    if ts == 0 or ti <= ts:
                        continue

                    dt_ticks = int(ti - ts)
                    if dt_ticks > dt_cut:
                        continue

                    dt2 = float(dt_ticks * dt_ticks)
                    w_time = np.exp(-dt2 * inv_2sig_t2)
                    if w_time <= 0.0:
                        continue

                    dx = xx - xi
                    d2 = float(dx * dx + dy * dy)
                    w_space = np.exp(-d2 * inv_2sig_s2)
                    if w_space <= 0.0:
                        continue

                    score += w_time * w_space

            out[i] = np.float32(score)

            idx0 = yi * width + xi
            last_ts[idx0] = ti
            last_pol[idx0] = np.int8(pi)

    return _kernel


def score_stream_n141(
    ev,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_us: int,
    tb: TimeBase,
    scores_out: np.ndarray | None = None,
) -> np.ndarray:
    """N141 (7.86 extension): Gaussian spatial + Gaussian temporal decay."""

    n = int(ev.t.shape[0])
    if scores_out is None:
        scores_out = np.empty((n,), dtype=np.float32)
    if n <= 0:
        return scores_out

    tau_base_ticks = int(tb.us_to_ticks(int(tau_us)))
    sigma_space = float(_read_float_env("MYEVS_N141_SIGMA", 3.0))
    sigma_time_ratio = float(_read_float_env("MYEVS_N141_TSIGMA_RATIO", 1.0))

    ker = _try_build_n141_kernel()
    ker(
        ev.t.astype(np.uint64, copy=False),
        ev.x.astype(np.int32, copy=False),
        ev.y.astype(np.int32, copy=False),
        ev.p,
        int(width),
        int(height),
        int(radius_px),
        int(tau_base_ticks),
        float(sigma_space),
        float(sigma_time_ratio),
        scores_out,
    )
    return scores_out
