from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class S40NoiseSurpriseZScoreStateOccNbOccMixFuseGeomParams:
    """Parameters for Part2 s40 (s39 with conservative fusion).

    This variant keeps s39's features:
    - u_self: per-pixel state occupancy
    - u_nb_mix: neighborhood occupancy weighted by polarity-mix

    But changes the fusion from union (aggressive) to geometric mean (conservative):

        u_eff = sqrt(u_self * u_nb_mix)

    Motivation
    - s39's union fusion can turn two weak cues into a strong background uplift,
      which may hurt light/mid. Geometric mean requires both cues to be high.

    Hyperparameters (env)
    - tau_rate_us: EMA time constant (microseconds) for global rate estimate; 0 means auto (use tau_us).
    - k_nbmix: strength multiplier for the neighborhood-mix occupancy term (>=0).
    """

    tau_rate_us: int = 0
    k_nbmix: float = 1.0


def _env_int(env: dict[str, str], name: str, default: int) -> int:
    s = (env.get(name, "") or "").strip()
    if not s:
        return int(default)
    try:
        v = int(float(s))
    except Exception:
        return int(default)
    if v < 0:
        v = 0
    return int(v)


def _env_float(env: dict[str, str], name: str, default: float) -> float:
    s = (env.get(name, "") or "").strip()
    if not s:
        return float(default)
    try:
        v = float(s)
    except Exception:
        return float(default)
    if not np.isfinite(v):
        return float(default)
    if v < 0.0:
        v = 0.0
    return float(v)


def s40_noise_surprise_zscore_stateocc_nbocc_mix_fuse_geom_params_from_env(
    env: dict[str, str] | None = None,
) -> S40NoiseSurpriseZScoreStateOccNbOccMixFuseGeomParams:
    """Read s40 parameters from environment.

    - MYEVS_EBF_S40_TAU_RATE_US (default 0; 0 means auto)
    - MYEVS_EBF_S40_K_NBMIX (default 1.0)
    """

    if env is None:
        env = os.environ  # type: ignore[assignment]

    tau_rate_us = _env_int(env, "MYEVS_EBF_S40_TAU_RATE_US", 0)
    k_nbmix = _env_float(env, "MYEVS_EBF_S40_K_NBMIX", 1.0)
    return S40NoiseSurpriseZScoreStateOccNbOccMixFuseGeomParams(tau_rate_us=int(tau_rate_us), k_nbmix=float(k_nbmix))


def try_build_s40_noise_surprise_zscore_stateocc_nbocc_mix_fuse_geom_scores_kernel():
    """Build and return Numba kernel for s40 score streaming.

    Returns None if numba is unavailable.
    """

    try:
        from numba import njit  # type: ignore
    except Exception:
        return None

    @njit(cache=True)
    def ebf_s40_scores_stream(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        width: int,
        height: int,
        radius_px: int,
        tau_ticks: int,
        tau_rate_ticks: int,
        k_nbmix: float,
        last_ts: np.ndarray,
        last_pol: np.ndarray,
        hot_state: np.ndarray,
        rate_ema: np.ndarray,
        scores_out: np.ndarray,
    ) -> None:
        n = int(t.shape[0])
        w = int(width)
        h = int(height)

        rr = int(radius_px)
        if rr < 0:
            rr = 0
        if rr > 8:
            rr = 8

        tau = int(tau_ticks)
        if tau <= 0:
            tau = 1

        tr = int(tau_rate_ticks)
        if tr <= 0:
            tr = tau

        k = float(k_nbmix)
        if not np.isfinite(k) or k < 0.0:
            k = 0.0

        r_ema = float(rate_ema[0])
        if not np.isfinite(r_ema) or r_ema < 0.0:
            r_ema = 0.0

        inv_tau = 1.0 / float(tau)
        n_pix = float(w * h)
        eps = 1e-6

        acc_max = 2147483647

        if rr <= 0:
            prev_t = 0
            for i in range(n):
                xi = int(x[i])
                yi = int(y[i])
                ti = int(t[i])

                if i > 0:
                    dtg = ti - prev_t
                    if dtg > 0:
                        inst = 1.0 / float(dtg)
                        a_rate = 1.0 - np.exp(-float(dtg) / float(tr))
                        r_ema = r_ema + a_rate * (inst - r_ema)
                prev_t = ti

                if xi < 0 or xi >= w or yi < 0 or yi >= h:
                    scores_out[i] = 0.0
                    continue

                pi = 1 if int(p[i]) > 0 else -1
                idx0 = yi * w + xi

                ts0 = int(last_ts[idx0])
                h0 = int(hot_state[idx0])

                dt0 = tau if ts0 == 0 else (ti - ts0)
                if dt0 < 0:
                    dt0 = -dt0

                if dt0 != 0:
                    h0 = h0 - dt0
                    if h0 < 0:
                        h0 = 0

                inc = tau - dt0
                if inc > 0:
                    h0 = h0 + inc
                    if h0 > acc_max:
                        h0 = acc_max

                last_ts[idx0] = np.uint64(ti)
                last_pol[idx0] = np.int8(pi)
                hot_state[idx0] = np.int32(h0)

                scores_out[i] = np.inf

            rate_ema[0] = r_ema
            return

        prev_t = 0
        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            ti = int(t[i])

            if i > 0:
                dtg = ti - prev_t
                if dtg > 0:
                    inst = 1.0 / float(dtg)
                    a_rate = 1.0 - np.exp(-float(dtg) / float(tr))
                    r_ema = r_ema + a_rate * (inst - r_ema)
            prev_t = ti

            if xi < 0 or xi >= w or yi < 0 or yi >= h:
                scores_out[i] = 0.0
                continue

            pi = 1 if int(p[i]) > 0 else -1
            idx0 = yi * w + xi

            ts0 = int(last_ts[idx0])
            h0 = int(hot_state[idx0])

            dt0 = tau if ts0 == 0 else (ti - ts0)
            if dt0 < 0:
                dt0 = -dt0

            if dt0 != 0:
                h0 = h0 - dt0
                if h0 < 0:
                    h0 = 0

            inc = tau - dt0
            if inc > 0:
                h0 = h0 + inc
                if h0 > acc_max:
                    h0 = acc_max

            raw_w = 0
            raw_all_w = 0

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

            m = (x1 - x0 + 1) * (y1 - y0 + 1) - 1
            if m < 1:
                m = 1

            for yy in range(y0, y1 + 1):
                base = yy * w
                for xx in range(x0, x1 + 1):
                    if xx == xi and yy == yi:
                        continue

                    idx = base + xx
                    ts = int(last_ts[idx])
                    if ts == 0:
                        continue

                    dt = ti - ts
                    if dt < 0:
                        dt = -dt
                    if dt > tau:
                        continue

                    rec = (tau - dt)
                    raw_all_w += rec
                    if int(last_pol[idx]) == pi:
                        raw_w += rec

            last_ts[idx0] = np.uint64(ti)
            last_pol[idx0] = np.int8(pi)
            hot_state[idx0] = np.int32(h0)

            raw = float(raw_w) * inv_tau

            r_pix = r_ema / n_pix
            if r_pix < 0.0:
                r_pix = 0.0

            hf = float(h0)
            u_self = hf / (hf + float(tr) + eps)
            if u_self < 0.0:
                u_self = 0.0
            if u_self > 1.0:
                u_self = 1.0

            raw_all = float(raw_all_w) * inv_tau
            u_nb = raw_all / (raw_all + float(m) + eps)
            if u_nb < 0.0:
                u_nb = 0.0
            if u_nb > 1.0:
                u_nb = 1.0

            raw_opp_w = raw_all_w - raw_w
            if raw_opp_w < 0:
                raw_opp_w = 0
            mix = float(raw_opp_w) / (float(raw_all_w) + eps)
            if mix < 0.0:
                mix = 0.0
            if mix > 1.0:
                mix = 1.0

            u_nb_mix = k * (u_nb * mix)
            if u_nb_mix < 0.0:
                u_nb_mix = 0.0
            if u_nb_mix > 1.0:
                u_nb_mix = 1.0

            # Conservative fusion: requires both cues to be high.
            u = np.sqrt(u_self * u_nb_mix)
            if u < 0.0:
                u = 0.0
            if u > 1.0:
                u = 1.0

            s = 1.0 + u
            r_eff = r_pix * (s * s)
            if r_eff < 0.0:
                r_eff = 0.0

            a = r_eff * float(tau)

            if a < 1e-3:
                ew = 0.5 * a - (a * a) / 6.0
                ew2 = (a / 3.0) - (a * a) / 12.0
            else:
                ea = np.exp(-a)
                ew = 1.0 - (1.0 - ea) / a
                ew2 = (a * a - 2.0 * a + 2.0 - 2.0 * ea) / (a * a)

            mu_per = 0.5 * ew
            e2_per = 0.5 * ew2
            var_per = e2_per - mu_per * mu_per
            if var_per < 0.0:
                var_per = 0.0

            mu = float(m) * mu_per
            var = float(m) * var_per

            denom = np.sqrt(var + eps)
            scores_out[i] = float((raw - mu) / denom)

        rate_ema[0] = r_ema

    return ebf_s40_scores_stream


__all__ = [
    "S40NoiseSurpriseZScoreStateOccNbOccMixFuseGeomParams",
    "s40_noise_surprise_zscore_stateocc_nbocc_mix_fuse_geom_params_from_env",
    "try_build_s40_noise_surprise_zscore_stateocc_nbocc_mix_fuse_geom_scores_kernel",
]
