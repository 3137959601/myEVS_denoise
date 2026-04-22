from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class S34NoiseSurpriseZScoreSelfRateMaxParams:
    """Parameters for Part2 s34 (s28 z-score + pixel self-rate max correction).

        Core idea:
        - Keep s28 raw support + z-score as backbone.
        - Use pixel self inter-arrival dt0 as a *continuous* short-dt indicator.
        - Apply a clipped, conditional penalty only on a small subset:
                score = z28 - k_self * clip(tau/dt0, 0, 8)
            but only when raw >= raw_thr and dt0 <= tau/8.

    Hyperparameters:
    - tau_rate_us: EMA time constant (us) for global rate estimate (same as s28).
      0 means auto (use current tau_us).
    - k_self: scale for self-rate correction (>=0). Typical range 0~1.

    Notes:
    - This is intended as a *local fix* that doesn't change the main ordering
      everywhere; it should only affect events that already look suspiciously
      strong under raw support.
    """

    tau_rate_us: int = 0
    k_self: float = 0.25


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
    return float(v)


def s34_noise_surprise_zscore_selfrate_max_params_from_env(
    env: dict[str, str] | None = None,
) -> S34NoiseSurpriseZScoreSelfRateMaxParams:
    """Read s34 parameters from environment.

    - MYEVS_EBF_S34_TAU_RATE_US (default 0; 0 means auto)
    - MYEVS_EBF_S34_K_SELF      (default 0.25)
    """

    if env is None:
        env = os.environ  # type: ignore[assignment]

    tau_rate_us = _env_int(env, "MYEVS_EBF_S34_TAU_RATE_US", 0)
    k_self = float(max(0.0, _env_float(env, "MYEVS_EBF_S34_K_SELF", 0.25)))

    return S34NoiseSurpriseZScoreSelfRateMaxParams(
        tau_rate_us=int(tau_rate_us),
        k_self=float(k_self),
    )


def try_build_s34_noise_surprise_zscore_selfrate_max_scores_kernel():
    """Build and return Numba kernel for s34 score streaming.

    Returns None if numba is unavailable.

    Kernel signature:
    (t,x,y,p,width,height,radius_px,tau_ticks,tau_rate_ticks,k_self,
     last_ts,last_pol,rate_ema,scores_out) -> None

    Arrays:
    - last_ts: uint64 per pixel
    - last_pol: int8 per pixel
    - rate_ema: float64 array (1,) global rate EMA (events/tick)

    Output:
    - scores_out: float32 score per event
    """

    try:
        from numba import njit  # type: ignore
    except Exception:
        return None

    @njit(cache=True)
    def ebf_s34_scores_stream(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        width: int,
        height: int,
        radius_px: int,
        tau_ticks: int,
        tau_rate_ticks: int,
        k_self: float,
        last_ts: np.ndarray,
        last_pol: np.ndarray,
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

        # global rate EMA in events/tick
        r_ema = float(rate_ema[0])
        if not np.isfinite(r_ema) or r_ema < 0.0:
            r_ema = 0.0

        inv_tau = 1.0 / float(tau)
        n_pix = float(w * h)
        eps = 1e-6

        ks = float(k_self)
        if not np.isfinite(ks) or ks < 0.0:
            ks = 0.0

        raw_thr = 3.0
        self_ratio_clip = 8.0

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
                last_ts[idx0] = np.uint64(ti)
                last_pol[idx0] = np.int8(pi)

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

            # Neighborhood raw support
            raw_w = 0

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
                    if int(last_pol[idx]) != pi:
                        continue
                    ts = int(last_ts[idx])
                    if ts == 0:
                        continue
                    dt = ti - ts
                    if dt < 0:
                        dt = -dt
                    if dt <= tau:
                        raw_w += (tau - dt)

            # self dt0 before we overwrite last_ts
            ts0 = int(last_ts[idx0])
            if ts0 == 0:
                dt0 = 0
            else:
                dt0 = ti - ts0
                if dt0 < 0:
                    dt0 = -dt0

            # update state
            last_ts[idx0] = np.uint64(ti)
            last_pol[idx0] = np.int8(pi)

            raw = float(raw_w) * inv_tau

            # s28 backbone z-score
            r_pix = r_ema / n_pix
            if r_pix < 0.0:
                r_pix = 0.0

            a_g = r_pix * float(tau)
            a = a_g

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
            z = float((raw - mu) / np.sqrt(var + eps))

            score = z
            if ks > 0.0 and raw >= raw_thr and dt0 > 0:
                # Only treat as bursty/hot when dt0 is very short.
                if dt0 <= (tau // 8):
                    ratio = float(tau) / float(dt0)
                    if ratio > self_ratio_clip:
                        ratio = self_ratio_clip
                    score = z - ks * ratio

            scores_out[i] = float(score)

        rate_ema[0] = r_ema

    return ebf_s34_scores_stream


__all__ = [
    "S34NoiseSurpriseZScoreSelfRateMaxParams",
    "s34_noise_surprise_zscore_selfrate_max_params_from_env",
    "try_build_s34_noise_surprise_zscore_selfrate_max_scores_kernel",
]
