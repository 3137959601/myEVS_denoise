from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class S30SurpriseZScoreLocalRateMaxParams:
    """Parameters for Part2 s30 (s28 with local-rate max correction).

    Motivation (lessons from s29 failure):
    - Dominant FP in ED24 heavy is *local* hotmask / near-hot. A pure global-rate
      normalization (s28) underestimates the noise rate for those pixels.
    - s29 tried to use local polarity-consistency as the score and failed:
      it penalized signal-like neighborhoods too aggressively.

    Core idea:
    - Keep s28's score definition: z = (raw - mu(r)) / sqrt(var(r) + eps), where
      raw is same-polarity neighborhood support.
    - But replace the per-pixel noise rate proxy with a *conservative local
      correction* estimated on-the-fly from the current neighborhood:
        r_eff = r_global + max(0, r_local - r_global) * conf
      where conf is derived from the observed neighborhood recency weights.

    Properties:
    - Streaming, single-pass, O(r^2), Numba-only.
    - No extra per-pixel state beyond baseline `last_ts/last_pol`.
    - Only explicit hyperparameter is the global rate EMA time constant.

    Hyperparameters:
    - tau_rate_us: EMA time constant (microseconds) for global rate estimate.
      0 means "auto" (use tau_us of the current sweep point).
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


def s30_surprise_zscore_localrate_max_params_from_env(
    env: dict[str, str] | None = None,
) -> S30SurpriseZScoreLocalRateMaxParams:
    """Read s30 parameters from environment.

    - MYEVS_EBF_S30_TAU_RATE_US (default 0; 0 means auto)
    """

    if env is None:
        env = os.environ  # type: ignore[assignment]

    tau_rate_us = _env_int(env, "MYEVS_EBF_S30_TAU_RATE_US", 0)
    return S30SurpriseZScoreLocalRateMaxParams(tau_rate_us=int(tau_rate_us))


def try_build_s30_surprise_zscore_localrate_max_scores_kernel():
    """Build and return Numba kernel for s30 score streaming.

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
    def ebf_s30_scores_stream(
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

            # Baseline raw support (same polarity) + local recency mass (all pol).
            raw_w = 0
            all_w = 0

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

                    ts = int(last_ts[idx])
                    if ts == 0:
                        continue

                    dt = ti - ts
                    if dt < 0:
                        dt = -dt
                    if dt > tau:
                        continue

                    wgt = tau - dt
                    all_w += wgt
                    if int(last_pol[idx]) == pi:
                        raw_w += wgt

            # Always update self state
            last_ts[idx0] = np.uint64(ti)
            last_pol[idx0] = np.int8(pi)

            raw = float(raw_w) * inv_tau

            # Global noise proxy.
            r_pix_g = r_ema / n_pix
            if r_pix_g < 0.0:
                r_pix_g = 0.0
            a_g = r_pix_g * float(tau)

            # Local correction: estimate a from observed mean recency weight.
            # wbar in [0,1], per-neighbor mean of max(0,1-dt/tau).
            wbar = float(all_w) / (float(tau) * float(m))
            if wbar < 0.0:
                wbar = 0.0
            if wbar > 1.0:
                wbar = 1.0

            # Invert ew(a)=1-(1-exp(-a))/a to estimate a_local.
            a_l = 0.0
            if wbar > 0.0:
                if wbar < 0.25:
                    a_l = 2.0 * wbar + (4.0 / 3.0) * wbar * wbar
                elif wbar > 0.85:
                    denom = 1.0 - wbar
                    if denom < 1e-6:
                        denom = 1e-6
                    a_l = 1.0 / denom
                else:
                    a_l = 2.0 * wbar / (1.0 - wbar)

                if a_l < 1e-6:
                    a_l = 1e-6
                if a_l > 50.0:
                    a_l = 50.0

                # Two Newton steps.
                for _ in range(2):
                    ea = np.exp(-a_l)
                    ew = 1.0 - (1.0 - ea) / a_l
                    f = ew - wbar
                    fp = ((1.0 - ea) - a_l * ea) / (a_l * a_l)
                    if fp < 1e-6:
                        fp = 1e-6
                    a_l = a_l - f / fp
                    if a_l < 1e-6:
                        a_l = 1e-6
                    if a_l > 50.0:
                        a_l = 50.0

            # Only allow local estimate to increase effective noise level.
            # Use wbar as a confidence weight so sparse neighborhoods keep global behavior.
            da = a_l - a_g
            if da > 0.0:
                a = a_g + da * wbar
            else:
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

            denom = np.sqrt(var + eps)
            scores_out[i] = float((raw - mu) / denom)

        rate_ema[0] = r_ema

    return ebf_s30_scores_stream


__all__ = [
    "S30SurpriseZScoreLocalRateMaxParams",
    "s30_surprise_zscore_localrate_max_params_from_env",
    "try_build_s30_surprise_zscore_localrate_max_scores_kernel",
]
