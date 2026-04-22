from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class S31NoiseSurpriseZScorePolBiasParams:
    """Parameters for Part2 s31 (s28 surprise z-score with polarity-bias correction).

    Motivation (lesson from s29 failure):
    - Both s28 and s29 assume noise polarity is random with P(match)=0.5.
      In practice, heavy hot pixels often have strong polarity bias (or the
      stream has global imbalance), which makes same-polarity matches more
      likely under noise than 0.5.
    - If we keep using 0.5, the noise-model mean mu is underestimated and z is
      over-optimistic -> more FP, especially for hotmask-dominated heavy.

    Core idea:
    - Keep s28's global-rate noise model for the recency weight distribution.
    - Replace the fixed match probability 0.5 by an online estimate derived from
      global polarity mean m = E[p] (p in {-1,+1}):
        q = P(match) = P(+,+)+P(-,-) = ((1+m)/2)^2 + ((1-m)/2)^2 = (1+m^2)/2
      Under independence between polarity and recency weights:
        mu_per  = q * E[w]
        E2_per  = q * E[w^2]
        var_per = E2_per - mu_per^2

    Constraints: streaming, single-pass, O(r^2), Numba-only.

    Hyperparameters:
    - tau_rate_us: EMA time constant (microseconds) for global rate/polarity estimate.
      0 means "auto" (use tau_us for the current sweep point).
    """

    tau_rate_us: int = 0


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


def s31_noise_surprise_zscore_polbias_params_from_env(
    env: dict[str, str] | None = None,
) -> S31NoiseSurpriseZScorePolBiasParams:
    """Read s31 parameters from environment.

    - MYEVS_EBF_S31_TAU_RATE_US (default 0; 0 means auto)
    """

    if env is None:
        env = os.environ  # type: ignore[assignment]

    tau_rate_us = _env_int(env, "MYEVS_EBF_S31_TAU_RATE_US", 0)
    return S31NoiseSurpriseZScorePolBiasParams(tau_rate_us=int(tau_rate_us))


def try_build_s31_noise_surprise_zscore_polbias_scores_kernel():
    """Build and return Numba kernel for s31 score streaming.

    Returns None if numba is unavailable.

    Kernel signature:
      (t,x,y,p,width,height,radius_px,tau_ticks,tau_rate_ticks,last_ts,last_pol,rate_pol_ema,scores_out) -> None

    Arrays:
    - last_ts: uint64 per pixel
    - last_pol: int8 per pixel
    - rate_pol_ema: float64 array of shape (2,)
        rate_pol_ema[0] = global event rate EMA (events/tick)
        rate_pol_ema[1] = global polarity mean EMA in [-1,+1]

    Output:
    - scores_out: float32 z-score per event
    """

    try:
        from numba import njit  # type: ignore
    except Exception:
        return None

    @njit(cache=True)
    def ebf_s31_scores_stream(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        width: int,
        height: int,
        radius_px: int,
        tau_ticks: int,
        tau_rate_ticks: int,
        last_ts: np.ndarray,
        last_pol: np.ndarray,
        rate_pol_ema: np.ndarray,
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

        r_ema = float(rate_pol_ema[0])
        if not np.isfinite(r_ema) or r_ema < 0.0:
            r_ema = 0.0

        m_ema = float(rate_pol_ema[1])
        if not np.isfinite(m_ema):
            m_ema = 0.0
        if m_ema < -1.0:
            m_ema = -1.0
        if m_ema > 1.0:
            m_ema = 1.0

        inv_tau = 1.0 / float(tau)
        n_pix = float(w * h)
        eps = 1e-6

        # Pass-through behavior (still updates state): score is +inf.
        if rr <= 0:
            prev_t = 0
            for i in range(n):
                xi = int(x[i])
                yi = int(y[i])
                ti = int(t[i])

                pi = 1 if int(p[i]) > 0 else -1

                if i > 0:
                    dtg = ti - prev_t
                    if dtg > 0:
                        inst = 1.0 / float(dtg)
                        a_rate = 1.0 - np.exp(-float(dtg) / float(tr))
                        r_ema = r_ema + a_rate * (inst - r_ema)
                        m_ema = m_ema + a_rate * (float(pi) - m_ema)
                prev_t = ti

                if xi < 0 or xi >= w or yi < 0 or yi >= h:
                    scores_out[i] = 0.0
                    continue

                idx0 = yi * w + xi
                last_ts[idx0] = np.uint64(ti)
                last_pol[idx0] = np.int8(pi)
                scores_out[i] = np.inf

            rate_pol_ema[0] = r_ema
            rate_pol_ema[1] = m_ema
            return

        prev_t = 0
        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            ti = int(t[i])

            pi = 1 if int(p[i]) > 0 else -1

            # Update global rate + global polarity mean using inter-event dt.
            if i > 0:
                dtg = ti - prev_t
                if dtg > 0:
                    inst = 1.0 / float(dtg)
                    a_rate = 1.0 - np.exp(-float(dtg) / float(tr))
                    r_ema = r_ema + a_rate * (inst - r_ema)
                    m_ema = m_ema + a_rate * (float(pi) - m_ema)
            prev_t = ti

            if xi < 0 or xi >= w or yi < 0 or yi >= h:
                scores_out[i] = 0.0
                continue

            idx0 = yi * w + xi

            # Baseline raw support (same polarity) as in EBF:
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

            # geometric neighbor count
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
                    if dt > tau:
                        continue

                    raw_w += (tau - dt)

            # Always update self state
            last_ts[idx0] = np.uint64(ti)
            last_pol[idx0] = np.int8(pi)

            raw = float(raw_w) * inv_tau

            # Global noise rate proxy.
            r_pix = r_ema / n_pix
            if r_pix < 0.0:
                r_pix = 0.0

            a = r_pix * float(tau)  # dimensionless

            if a < 1e-3:
                ew = 0.5 * a - (a * a) / 6.0
                ew2 = (a / 3.0) - (a * a) / 12.0
            else:
                ea = np.exp(-a)
                ew = 1.0 - (1.0 - ea) / a
                ew2 = (a * a - 2.0 * a + 2.0 - 2.0 * ea) / (a * a)

            # Polarity match probability under (biased) random polarity model.
            # q = (1 + m^2)/2
            if m_ema < -1.0:
                m_ema = -1.0
            if m_ema > 1.0:
                m_ema = 1.0
            q = 0.5 * (1.0 + m_ema * m_ema)
            if q < 0.5:
                q = 0.5
            if q > 1.0:
                q = 1.0

            mu_per = q * ew
            e2_per = q * ew2
            var_per = e2_per - mu_per * mu_per
            if var_per < 0.0:
                var_per = 0.0

            mu = float(m) * mu_per
            var = float(m) * var_per

            denom = np.sqrt(var + eps)
            scores_out[i] = float((raw - mu) / denom)

        rate_pol_ema[0] = r_ema
        rate_pol_ema[1] = m_ema

    return ebf_s31_scores_stream


__all__ = [
    "S31NoiseSurpriseZScorePolBiasParams",
    "s31_noise_surprise_zscore_polbias_params_from_env",
    "try_build_s31_noise_surprise_zscore_polbias_scores_kernel",
]
