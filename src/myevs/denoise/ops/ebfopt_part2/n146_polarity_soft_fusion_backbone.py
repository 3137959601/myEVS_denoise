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
        raise RuntimeError("n146 requires numba, but import failed")


def _build_space_lut(radius_px: int, sigma_space: float) -> np.ndarray:
    rr = int(radius_px)
    if rr < 0:
        rr = 0
    if rr > 8:
        rr = 8

    sig = float(sigma_space)
    if sig <= 1e-6:
        sig = 1e-6

    lut = np.zeros((rr + 1,), dtype=np.float32)
    inv_2sig2 = 1.0 / (2.0 * sig * sig)
    for d in range(rr + 1):
        lut[d] = np.float32(np.exp(-float(d * d) * inv_2sig2))
    return lut


def _try_build_n146_kernel():
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
        lut_space: np.ndarray,
        gamma: float,
        eps: float,
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
        inv_tau_base = 1.0 / float(tau_base)

        g = float(gamma)
        if g < 0.0:
            g = 0.0

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

            e_same = 0.0
            e_opp = 0.0

            for yy in range(y0, y1 + 1):
                base = yy * width
                dy = yy - yi
                ady = dy if dy >= 0 else -dy
                for xx in range(x0, x1 + 1):
                    if xx == xi and yy == yi:
                        continue

                    idx_nb = base + xx
                    ts = last_ts[idx_nb]
                    if ts == 0 or ti <= ts:
                        continue

                    dt_ticks = int(ti - ts)
                    if dt_ticks > tau_base:
                        continue

                    # n145 temporal backbone: per-neighbor (1 - dt/tau)^2.
                    ratio = float(dt_ticks) * inv_tau_base
                    base_time = 1.0 - ratio
                    if base_time <= 0.0:
                        continue
                    w_time = base_time * base_time

                    dx = xx - xi
                    adx = dx if dx >= 0 else -dx
                    d = adx if adx >= ady else ady
                    if d > rr:
                        d = rr
                    w_space = float(lut_space[d])
                    if w_space <= 0.0:
                        continue

                    w = w_time * w_space

                    if int(last_pol[idx_nb]) == pi:
                        e_same += w
                    else:
                        e_opp += w

            e_total = e_same + e_opp
            if e_total <= 0.0:
                out[i] = np.float32(0.0)
            else:
                mix = e_opp / (e_total + eps)
                one_minus_mix = 1.0 - mix
                if one_minus_mix < 0.0:
                    one_minus_mix = 0.0
                w_purity = one_minus_mix**g
                out[i] = np.float32(e_same + w_purity * e_opp)

            idx0 = yi * width + xi
            last_ts[idx0] = ti
            last_pol[idx0] = np.int8(pi)

    return _kernel


def score_stream_n146(
    ev,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_us: int,
    tb: TimeBase,
    scores_out: np.ndarray | None = None,
) -> np.ndarray:
    """N146: n145 时空底盘 + S52 极性软融合。

    temporal: max(0, 1 - dt/tau)^2
    spatial:  LUT-based Gaussian over Chebyshev distance d
    fusion:   score = e_same + (1 - mix)^gamma * e_opp
    """

    n = int(ev.t.shape[0])
    if scores_out is None:
        scores_out = np.empty((n,), dtype=np.float32)
    if n <= 0:
        return scores_out

    tau_base_ticks = int(tb.us_to_ticks(int(tau_us)))
    sigma_space = float(_read_float_env("MYEVS_N146_SIGMA", 2.5))
    gamma = float(_read_float_env("MYEVS_N146_GAMMA", 2.0))
    eps = float(_read_float_env("MYEVS_N146_EPS", 1e-3))
    if eps <= 0.0:
        eps = 1e-9

    lut_space = _build_space_lut(int(radius_px), float(sigma_space))
    ker = _try_build_n146_kernel()
    ker(
        ev.t.astype(np.uint64, copy=False),
        ev.x.astype(np.int32, copy=False),
        ev.y.astype(np.int32, copy=False),
        ev.p,
        int(width),
        int(height),
        int(radius_px),
        int(tau_base_ticks),
        lut_space,
        float(gamma),
        float(eps),
        scores_out,
    )
    return scores_out