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
        raise RuntimeError("n96 requires numba, but import failed")


def _try_build_n96_kernel():
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
        v0_px_per_us: float,
        sigma_v_px_per_us: float,
        eps_us: float,
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

        v0 = float(v0_px_per_us)
        sigma_v = float(sigma_v_px_per_us)
        if sigma_v <= 0.0:
            sigma_v = 1e-6

        eps = float(eps_us)
        if eps <= 0.0:
            eps = 1.0

        inv_sigma_v = 1.0 / sigma_v

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

            # Top1 support: same polarity history support with minimal dt (max ts).
            found = False
            best_dt_ticks = 0
            best_dx = 0.0
            best_dy = 0.0

            for yy in range(y0, y1 + 1):
                base = yy * width
                dyf = float(yi - yy)
                for xx in range(x0, x1 + 1):
                    # Exclude center pixel to avoid same-pixel repeat dominating.
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

                    if (not found) or (dt_ticks < best_dt_ticks):
                        found = True
                        best_dt_ticks = dt_ticks
                        best_dx = float(xi - xx)
                        best_dy = dyf

            score = 0.0
            if found:
                dist = np.sqrt(best_dx * best_dx + best_dy * best_dy)
                dt_us = float(best_dt_ticks) * tick_us
                v = dist / (dt_us + eps)
                score = np.exp(-np.abs(v - v0) * inv_sigma_v)

            out[i] = np.float32(score)

            idx0 = yi * width + xi
            if pi > 0:
                pos_ts[idx0] = ti
            else:
                neg_ts[idx0] = ti

    return _kernel


def score_stream_n96(
    ev,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_us: int,
    tb: TimeBase,
    scores_out: np.ndarray | None = None,
) -> np.ndarray:
    """N96: top1-v backbone score.

    Score definition:
      score_i = exp(-|v_i - v0| / sigma_v), if top1 support exists, else 0.

    where:
      top1 support = most recent same-polarity support in local window (excluding center pixel),
      v_i = d_i / (dt_i + eps).
    """

    n = int(ev.t.shape[0])
    if scores_out is None:
        scores_out = np.empty((n,), dtype=np.float32)
    if n <= 0:
        return scores_out

    tau_ticks = int(tb.us_to_ticks(int(tau_us)))
    tick_us = float(tb.ticks_to_us(1))

    v0_px_per_us = float(_read_float_env("MYEVS_N96_V0_PX_PER_US", 0.0020))
    sigma_v_px_per_us = float(_read_float_env("MYEVS_N96_SIGMA_V_PX_PER_US", 0.0010))
    eps_us = float(_read_float_env("MYEVS_N96_EPS_US", 1.0))

    ker = _try_build_n96_kernel()
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
        float(v0_px_per_us),
        float(sigma_v_px_per_us),
        float(eps_us),
        scores_out,
    )
    return scores_out
