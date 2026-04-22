from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class S28NoiseSurpriseZScoreParams:
    """Parameters for Part2 s28 (noise-model surprise / z-score).

    Motivation:
    - Raw EBF support (same-pol neighborhood sum) increases with event rate.
      Under heavy noise, many hot/high-rate pixels get high raw scores.
    - We want a *dimensionless* score that roughly measures "how surprising" the
      observed raw support is under a simple noise-rate model.

    Core idea (sketch):
    - Maintain a global EMA estimate of sensor event rate (events/tick).
    - Convert to per-pixel noise rate: r_pix = rate_ema / (W*H).
    - Under a Poisson noise model (random polarity, independent pixels), derive
      E[raw] and Var[raw] for the baseline raw support:
        raw = sum_{neighbors} 1{pol match} * max(0, 1 - dt/tau)
      where dt is the backward recurrence time (Exp(r_pix)).
    - Output a z-score:
        z = (raw - mu) / sqrt(var + eps)

    Constraints: streaming, single-pass, O(r^2), Numba-only.

    Hyperparameters:
    - tau_rate_us: EMA time constant (microseconds) for global rate estimate.
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


def s28_noise_surprise_zscore_params_from_env(
    env: dict[str, str] | None = None,
) -> S28NoiseSurpriseZScoreParams:
    """Read s28 parameters from environment.

    - MYEVS_EBF_S28_TAU_RATE_US (default 0; 0 means auto)

    Notes:
    - Keeping this as the only explicit hyperparameter is intentional: the
      standardization should mainly be determined by (s, tau_us) and the
      observed event rate.
    """

    if env is None:
        env = os.environ  # type: ignore[assignment]

    tau_rate_us = _env_int(env, "MYEVS_EBF_S28_TAU_RATE_US", 0)

    return S28NoiseSurpriseZScoreParams(tau_rate_us=int(tau_rate_us))


def try_build_s28_noise_surprise_zscore_scores_kernel():
    """Build and return Numba kernel for s28 score streaming.

    Returns None if numba is unavailable.

    Kernel signature:
      (t,x,y,p,width,height,radius_px,tau_ticks,tau_rate_ticks,last_ts,last_pol,rate_ema,scores_out) -> None

    Arrays:
    - last_ts: uint64 per pixel
    - last_pol: int8 per pixel
    - rate_ema: float64 array of shape (1,) storing global event rate (events/tick)

    Output:
    - scores_out: float32 z-score per event
    """

    try:
        from numba import njit  # type: ignore
    except Exception:
        return None

    @njit(cache=True)
    def ebf_s28_scores_stream(
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

        # Pass-through behavior (still updates state): score is +inf.
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

            # Update global rate EMA (events/tick) using inter-event dt.
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

            # Noise-model expectation for raw.
            # r_pix: per-pixel event rate (events/tick)
            r_pix = r_ema / n_pix
            if r_pix < 0.0:
                r_pix = 0.0

            a = r_pix * float(tau)  # dimensionless

            if a < 1e-3:
                # series expansions for stability
                ew = 0.5 * a - (a * a) / 6.0
                ew2 = (a / 3.0) - (a * a) / 12.0
            else:
                ea = np.exp(-a)
                ew = 1.0 - (1.0 - ea) / a
                ew2 = (a * a - 2.0 * a + 2.0 - 2.0 * ea) / (a * a)

            # Mark (polarity) match probability is ~0.5 under noise.
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

    return ebf_s28_scores_stream


__all__ = [
    "S28NoiseSurpriseZScoreParams",
    "s28_noise_surprise_zscore_params_from_env",
    "try_build_s28_noise_surprise_zscore_scores_kernel",
]
