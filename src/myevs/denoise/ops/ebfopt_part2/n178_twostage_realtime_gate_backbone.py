from __future__ import annotations

import numpy as np

from ....timebase import TimeBase

try:
    import numba
except Exception:  # pragma: no cover
    numba = None


N178_SIGMA = 2.8
N178_BETA_INIT = 0.65
N178_K_SFRAC = 4.0 / 5.0
N178_K_MIX = 1.0 / 8.0
N178_RSTATE_INIT = 0.10

_N178_KERNEL = None
_N178_SPACE_CACHE: dict[tuple[int, float], np.ndarray] = {}


def _require_numba() -> None:
    if numba is None:
        raise RuntimeError("n178 requires numba, but import failed")


def _space_lut(radius_px: int, sigma_space: float) -> np.ndarray:
    rr = max(0, min(8, int(radius_px)))
    sig = max(1e-6, float(sigma_space))
    key = (rr, sig)
    cached = _N178_SPACE_CACHE.get(key)
    if cached is not None:
        return cached
    max_d2 = 2 * rr * rr
    lut = np.empty((max_d2 + 1,), dtype=np.float32)
    inv_2sig2 = 1.0 / (2.0 * sig * sig)
    for d2 in range(max_d2 + 1):
        lut[d2] = np.float32(np.exp(-float(d2) * inv_2sig2))
    _N178_SPACE_CACHE[key] = lut
    return lut


def _try_build_n178_kernel():
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
        space_lut: np.ndarray,
        out: np.ndarray,
    ) -> None:
        n = int(t.shape[0])
        npx = int(width) * int(height)
        last_ts = np.zeros((npx,), dtype=np.uint64)
        last_pol = np.zeros((npx,), dtype=np.int8)

        rr = max(0, min(8, int(radius_px)))
        tau_base = int(tau_base_ticks)
        if tau_base <= 0:
            tau_base = 1
        inv_tau = 1.0 / float(tau_base)
        eps = 1e-6

        b = float(N178_BETA_INIT)
        mstate = 0.0
        rstate = float(N178_RSTATE_INIT)

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            if xi < 0 or xi >= width or yi < 0 or yi >= height:
                out[i] = np.float32(0.0)
                continue

            ti = int(t[i])
            pi = 1 if int(p[i]) > 0 else -1
            idx0 = yi * width + xi

            # Stage 1: extremely cheap local confirmation. If no same-polarity
            # event exists in the immediate 3x3 window, dense n176 scoring is
            # skipped. This preserves the full score for confirmed candidates.
            local_same = 0
            for ddy in range(-1, 2):
                ny = yi + ddy
                if ny < 0 or ny >= height:
                    continue
                for ddx in range(-1, 2):
                    if ddx == 0 and ddy == 0:
                        continue
                    nx = xi + ddx
                    if nx < 0 or nx >= width:
                        continue
                    idxn = ny * width + nx
                    if int(last_pol[idxn]) != pi:
                        continue
                    tsn = int(last_ts[idxn])
                    if tsn == 0 or ti <= tsn:
                        continue
                    if ti - tsn <= tau_base:
                        local_same = 1
                        break
                if local_same == 1:
                    break

            prev_pol0 = int(last_pol[idx0])
            ts0 = int(last_ts[idx0])
            dt0 = tau_base
            if ts0 != 0:
                dt0 = ti - ts0
                if dt0 < 0:
                    dt0 = -dt0
                if dt0 > tau_base:
                    dt0 = tau_base

            if local_same == 0:
                # Keep state updates causal but avoid the expensive dense window.
                u_lite0 = 1.0 - float(dt0) * inv_tau
                if u_lite0 < 0.0:
                    u_lite0 = 0.0
                if u_lite0 > 1.0:
                    u_lite0 = 1.0
                rhythm_bad0 = u_lite0 if prev_pol0 == -pi else 0.0
                rstate = rstate + (rhythm_bad0 - rstate) / 4096.0
                if rstate < 0.0:
                    rstate = 0.0
                if rstate > 1.0:
                    rstate = 1.0
                out[i] = np.float32(0.0)
                last_ts[idx0] = np.uint64(ti)
                last_pol[idx0] = np.int8(pi)
                continue

            x0 = max(0, xi - rr)
            x1 = min(width - 1, xi + rr)
            y0 = max(0, yi - rr)
            y1 = min(height - 1, yi + rr)

            raw_same = 0.0
            raw_opp = 0.0
            cnt_support = 0

            for ny in range(y0, y1 + 1):
                dy = ny - yi
                for nx in range(x0, x1 + 1):
                    dx = nx - xi
                    if dx == 0 and dy == 0:
                        continue
                    idx = ny * width + nx
                    pol_nb = int(last_pol[idx])
                    if pol_nb != pi and pol_nb != -pi:
                        continue
                    ts = int(last_ts[idx])
                    if ts == 0 or ti <= ts:
                        continue
                    dt = ti - ts
                    if dt > tau_base:
                        continue
                    base_time = 1.0 - float(dt) * inv_tau
                    if base_time <= 0.0:
                        continue
                    d2 = dx * dx + dy * dy
                    wst = base_time * base_time * float(space_lut[d2])
                    if pol_nb == pi:
                        raw_same += wst
                        cnt_support += 1
                    else:
                        raw_opp += wst

            mix = 0.0
            denom_mix = raw_same + raw_opp
            if denom_mix > 0.0:
                mix = raw_opp / (denom_mix + eps)
                if mix < 0.0:
                    mix = 0.0
                if mix > 1.0:
                    mix = 1.0

            mstate = mstate + (mix - mstate) / 4096.0
            if mstate < 0.0:
                mstate = 0.0
            if mstate > 1.0:
                mstate = 1.0
            alpha_eff = 1.0 - mstate
            if alpha_eff < 0.0:
                alpha_eff = 0.0
            alpha_eff = alpha_eff * alpha_eff

            u_lite = 1.0 - float(dt0) * inv_tau
            if u_lite < 0.0:
                u_lite = 0.0
            if u_lite > 1.0:
                u_lite = 1.0

            rhythm_bad = 0.0
            rhythm_good = 0.0
            if prev_pol0 == -pi:
                rhythm_bad = u_lite
            elif prev_pol0 == pi:
                rhythm_good = u_lite

            rstate = rstate + (rhythm_bad - rstate) / 4096.0
            if rstate < 0.0:
                rstate = 0.0
            if rstate > 1.0:
                rstate = 1.0
            rhythm_pressure = 0.5 * (rhythm_bad + rstate)
            if rhythm_pressure < 0.0:
                rhythm_pressure = 0.0
            if rhythm_pressure > 1.0:
                rhythm_pressure = 1.0

            cnt_possible = (x1 - x0 + 1) * (y1 - y0 + 1) - 1
            sfrac = float(cnt_support) / float(cnt_possible) if cnt_possible > 0 else 0.0
            if sfrac < 0.0:
                sfrac = 0.0
            if sfrac > 1.0:
                sfrac = 1.0

            alpha_eff = alpha_eff * (1.0 - rhythm_pressure / 3.0)
            if alpha_eff < 0.0:
                alpha_eff = 0.0

            relief = 1.0 - N178_K_SFRAC * sfrac
            if relief < 0.25:
                relief = 0.25
            if relief > 1.0:
                relief = 1.0
            mix_gain = 1.0 + N178_K_MIX * mix
            if mix_gain < 0.5:
                mix_gain = 0.5
            if mix_gain > 2.0:
                mix_gain = 2.0
            rhythm_scale = 1.0 + 0.5 * rhythm_pressure - 0.25 * rhythm_good
            if rhythm_scale < 0.5:
                rhythm_scale = 0.5
            if rhythm_scale > 1.75:
                rhythm_scale = 1.75
            u_eff = u_lite * relief * mix_gain * rhythm_scale
            if u_eff < 0.0:
                u_eff = 0.0
            if u_eff > 1.0:
                u_eff = 1.0

            b = b + (u_eff - b) / 4096.0
            if b < 0.0:
                b = 0.0
            if b > 1.0:
                b = 1.0

            raw_gated = raw_same + alpha_eff * raw_opp
            base_score = raw_gated / (1.0 + u_eff * u_eff)
            support_scale = 1.0 + b * sfrac * (1.0 + 0.25 * rhythm_good)
            if support_scale < 1.0:
                support_scale = 1.0
            if support_scale > 2.0:
                support_scale = 2.0
            out[i] = np.float32(base_score * support_scale)

            last_ts[idx0] = np.uint64(ti)
            last_pol[idx0] = np.int8(pi)

    return _kernel


def _get_n178_kernel():
    global _N178_KERNEL
    if _N178_KERNEL is None:
        _N178_KERNEL = _try_build_n178_kernel()
    return _N178_KERNEL


def score_stream_n178(
    ev,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_us: int,
    tb: TimeBase,
    scores_out: np.ndarray | None = None,
) -> np.ndarray:
    """N178: two-stage realtime gate.

    Stage 1 uses a 3x3 same-polarity confirmation check. Only confirmed
    candidates receive the full dense n176-style score.
    """

    n = int(ev.t.shape[0])
    if scores_out is None:
        scores_out = np.empty((n,), dtype=np.float32)
    if n <= 0:
        return scores_out

    tau_base_ticks = int(tb.us_to_ticks(int(tau_us)))
    lut = _space_lut(int(radius_px), float(N178_SIGMA))
    ker = _get_n178_kernel()
    ker(
        ev.t.astype(np.uint64, copy=False),
        ev.x.astype(np.int32, copy=False),
        ev.y.astype(np.int32, copy=False),
        ev.p,
        int(width),
        int(height),
        int(radius_px),
        int(tau_base_ticks),
        lut,
        scores_out,
    )
    return scores_out
