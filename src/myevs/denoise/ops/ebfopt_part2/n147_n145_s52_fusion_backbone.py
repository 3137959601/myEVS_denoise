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
        raise RuntimeError("n147 requires numba, but import failed")


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


def _try_build_n147_kernel():
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
        hot_state: np.ndarray,
        beta_state: np.ndarray,
        mix_state: np.ndarray,
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

        # Keep s52's fixed adaptation horizon.
        N = 4096.0
        tr = tau_base // 2
        if tr <= 0:
            tr = 1
        eps = 1e-6
        acc_max = 2147483647

        b = float(beta_state[0])
        if b < 0.0:
            b = 0.0
        if b > 1.0:
            b = 1.0

        mstate = float(mix_state[0])
        if mstate < 0.0:
            mstate = 0.0
        if mstate > 1.0:
            mstate = 1.0

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            if xi < 0 or xi >= width or yi < 0 or yi >= height:
                out[i] = np.float32(0.0)
                continue

            ti = int(t[i])
            pi = 1 if int(p[i]) > 0 else -1

            idx0 = yi * width + xi

            ts0 = int(last_ts[idx0])
            h0 = int(hot_state[idx0])

            dt0 = tau_base if ts0 == 0 else (ti - ts0)
            if dt0 < 0:
                dt0 = -dt0
            if dt0 != 0:
                h0 = h0 - dt0
                if h0 < 0:
                    h0 = 0

            inc = tau_base - dt0
            if inc > 0:
                h0 = h0 + inc
                if h0 > acc_max:
                    h0 = acc_max

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

            raw_same = 0.0
            raw_opp = 0.0
            cnt_support = 0

            for yy in range(y0, y1 + 1):
                base = yy * width
                dy = yy - yi
                ady = dy if dy >= 0 else -dy
                for xx in range(x0, x1 + 1):
                    if xx == xi and yy == yi:
                        continue

                    idx_nb = base + xx
                    pol_nb = int(last_pol[idx_nb])
                    if pol_nb != pi and pol_nb != -pi:
                        continue

                    ts = int(last_ts[idx_nb])
                    if ts == 0 or ti <= ts:
                        continue

                    dt_ticks = ti - ts
                    if dt_ticks > tau_base:
                        continue

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

                    wst = w_time * w_space
                    if pol_nb == pi:
                        raw_same += wst
                        cnt_support += 1
                    else:
                        raw_opp += wst

            last_ts[idx0] = np.uint64(ti)
            last_pol[idx0] = np.int8(pi)
            hot_state[idx0] = np.int32(h0)

            u_self = float(h0) / (float(h0) + float(tr) + eps)
            if u_self < 0.0:
                u_self = 0.0
            if u_self > 1.0:
                u_self = 1.0

            b = b + (u_self - b) / N
            if b < 0.0:
                b = 0.0
            if b > 1.0:
                b = 1.0

            denom_mix = raw_same + raw_opp
            mix = 0.0
            if denom_mix > 0.0:
                mix = raw_opp / (denom_mix + eps)
                if mix < 0.0:
                    mix = 0.0
                if mix > 1.0:
                    mix = 1.0

            mstate = mstate + (mix - mstate) / N
            if mstate < 0.0:
                mstate = 0.0
            if mstate > 1.0:
                mstate = 1.0

            alpha_eff = 1.0 - mstate
            if alpha_eff < 0.0:
                alpha_eff = 0.0
            alpha_eff = alpha_eff * alpha_eff

            raw_gated = raw_same + alpha_eff * raw_opp
            base_score = raw_gated / (1.0 + (u_self * u_self))

            cnt_possible = (x1 - x0 + 1) * (y1 - y0 + 1) - 1
            if cnt_possible <= 0:
                sfrac = 0.0
            else:
                sfrac = float(cnt_support) / float(cnt_possible)
                if sfrac < 0.0:
                    sfrac = 0.0
                if sfrac > 1.0:
                    sfrac = 1.0

            out[i] = np.float32(base_score * (1.0 + b * sfrac))

        beta_state[0] = np.float32(b)
        mix_state[0] = np.float32(mstate)

    return _kernel


def score_stream_n147(
    ev,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_us: int,
    tb: TimeBase,
    scores_out: np.ndarray | None = None,
) -> np.ndarray:
    """N147: n145 raw_same + s52 online fusion.

    - raw_same/opp: n145-style spatiotemporal weights (time squared + spatial Gaussian LUT)
    - fusion: s52-style adaptive mix-gate + self-occupancy + support boost
    """

    n = int(ev.t.shape[0])
    if scores_out is None:
        scores_out = np.empty((n,), dtype=np.float32)
    if n <= 0:
        return scores_out

    tau_base_ticks = int(tb.us_to_ticks(int(tau_us)))
    sigma_space = float(_read_float_env("MYEVS_N147_SIGMA", 2.5))
    lut_space = _build_space_lut(int(radius_px), float(sigma_space))

    npx = int(width) * int(height)
    hot_state = np.zeros((npx,), dtype=np.int32)
    beta_state = np.zeros((1,), dtype=np.float32)
    mix_state = np.zeros((1,), dtype=np.float32)

    ker = _try_build_n147_kernel()
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
        hot_state,
        beta_state,
        mix_state,
        scores_out,
    )
    return scores_out