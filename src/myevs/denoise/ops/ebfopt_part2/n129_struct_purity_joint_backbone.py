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
        raise RuntimeError("n129 requires numba, but import failed")


def _try_build_n129_kernel():
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
        eta: float,
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

        e = float(eta)
        if e < 0.0:
            e = 0.0
        if e > 1.0:
            e = 1.0

        g = float(gamma)
        if g <= 0.0:
            g = 1.0

        epsv = float(eps)
        if epsv <= 0.0:
            epsv = 1e-3

        # Use current radius as R_max; fallback to 1 avoids divide-by-zero for rr=0.
        rmax = float(rr if rr > 0 else 1)
        inv_rmax2 = 1.0 / (rmax * rmax)

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

            e_same_in = 0.0
            e_same_out = 0.0
            e_opp_in = 0.0
            e_opp_out = 0.0

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

                    w_time = 1.0 - float(dt_ticks) * inv_tau_base
                    if w_time <= 0.0:
                        continue

                    dx = xx - xi
                    adx = dx if dx >= 0 else -dx
                    d = adx if adx >= ady else ady

                    ratio2 = float(d * d) * inv_rmax2
                    w_space = 1.0 - e * ratio2
                    if w_space <= 0.0:
                        continue

                    w_joint = w_time * w_space
                    is_inner = d <= 2

                    if int(last_pol[idx_nb]) == pi:
                        if is_inner:
                            e_same_in += w_joint
                        else:
                            e_same_out += w_joint
                    else:
                        if is_inner:
                            e_opp_in += w_joint
                        else:
                            e_opp_out += w_joint

            e_total = e_same_in + e_same_out + e_opp_in + e_opp_out
            if e_total <= 0.0:
                score = 0.0
            else:
                mix = (e_opp_in + e_opp_out) / (e_total + epsv)
                if mix < 0.0:
                    mix = 0.0
                if mix > 1.0:
                    mix = 1.0
                # 7.76 purity weight.
                w_purity = (1.0 - mix) ** g

                outer_ratio = (e_same_out + e_opp_out) / (e_total + epsv)
                if outer_ratio < 0.0:
                    outer_ratio = 0.0
                if outer_ratio > 1.0:
                    outer_ratio = 1.0
                # 7.76 structure band-pass (parabola centered at 0.5).
                w_struct = 4.0 * outer_ratio * (1.0 - outer_ratio)
                if w_struct < 0.0:
                    w_struct = 0.0

                score = e_total * w_purity * w_struct

            out[i] = np.float32(score)

            idx0 = yi * width + xi
            last_ts[idx0] = ti
            last_pol[idx0] = np.int8(pi)

    return _kernel


def score_stream_n129(
    ev,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_us: int,
    tb: TimeBase,
    scores_out: np.ndarray | None = None,
) -> np.ndarray:
    """N129 (7.76): joint spatiotemporal energy with purity and structure weights."""

    n = int(ev.t.shape[0])
    if scores_out is None:
        scores_out = np.empty((n,), dtype=np.float32)
    if n <= 0:
        return scores_out

    tau_base_ticks = int(tb.us_to_ticks(int(tau_us)))
    eta = float(_read_float_env("MYEVS_N129_ETA", 0.5))
    gamma = float(_read_float_env("MYEVS_N129_GAMMA", 2.0))
    eps = float(_read_float_env("MYEVS_N129_EPS", 1e-3))

    ker = _try_build_n129_kernel()
    ker(
        ev.t.astype(np.uint64, copy=False),
        ev.x.astype(np.int32, copy=False),
        ev.y.astype(np.int32, copy=False),
        ev.p,
        int(width),
        int(height),
        int(radius_px),
        int(tau_base_ticks),
        float(eta),
        float(gamma),
        float(eps),
        scores_out,
    )
    return scores_out
