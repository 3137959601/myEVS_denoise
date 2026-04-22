from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class S36NoiseSurpriseZScoreStateOccupancyParams:
    """Parameters for Part2 s36 (state-occupancy-conditioned noise surprise z-score).

    Motivation (builds on s28 + s35):
    - s28 uses a global-rate Poisson null model; hot pixels violate it.
    - s35 improves heavy noise, but introduces 3 sensitive hyperparameters
      (tau_rate_us, gamma, hmax) and can over-intervene in light/mid.

    Core idea:
    - Keep s28 backbone (global rate EMA -> per-pixel r_pix -> mu/var -> z-score).
    - Maintain a per-pixel "hotness" state H as a leaky accumulator (tick units),
      but make the *increment* depend on self inter-event time dt0:

        H <- max(0, H - dt0) + max(0, tau - dt0)

      This avoids giving every event an unconditional +tau boost (which can hurt
      light/mid by treating normal activity as "hot" too early).

    - Convert H into a dimensionless occupancy u in [0,1):

        u = H / (H + tau_r)

      where tau_r is the same time constant used for the global rate EMA.

    - Modulate the null-model rate smoothly without extra gain/clamp knobs:

        r_eff = r_pix * (1 + u)^2

      This is bounded (<=4x) and parameter-free beyond tau_r.

    Constraints: streaming, single-pass, O(r^2), Numba-only.

    Hyperparameters:
    - tau_rate_us: EMA time constant (microseconds) for global rate estimate.
      0 means auto (use tau_us for the current sweep point).
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


def s36_noise_surprise_zscore_stateoccupancy_params_from_env(
    env: dict[str, str] | None = None,
) -> S36NoiseSurpriseZScoreStateOccupancyParams:
    """Read s36 parameters from environment.

    - MYEVS_EBF_S36_TAU_RATE_US (default 0; 0 means auto)
    """

    if env is None:
        env = os.environ  # type: ignore[assignment]

    tau_rate_us = _env_int(env, "MYEVS_EBF_S36_TAU_RATE_US", 0)
    return S36NoiseSurpriseZScoreStateOccupancyParams(tau_rate_us=int(tau_rate_us))


def try_build_s36_noise_surprise_zscore_stateoccupancy_scores_kernel():
    """Build and return Numba kernel for s36 score streaming.

    Returns None if numba is unavailable.

    Kernel signature:
      (t,x,y,p,width,height,radius_px,tau_ticks,tau_rate_ticks,last_ts,last_pol,hot_state,rate_ema,scores_out) -> None

    Arrays:
    - last_ts: uint64 per pixel
    - last_pol: int8 per pixel
    - hot_state: int32 per pixel (leaky accumulator in tick units)
    - rate_ema: float64 array of shape (1,) storing global event rate (events/tick)

    Output:
    - scores_out: float32 z-score per event
    """

    try:
        from numba import njit  # type: ignore
    except Exception:
        return None

    @njit(cache=True)
    def ebf_s36_scores_stream(
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

        r_ema = float(rate_ema[0])
        if not np.isfinite(r_ema) or r_ema < 0.0:
            r_ema = 0.0

        inv_tau = 1.0 / float(tau)
        n_pix = float(w * h)
        eps = 1e-6

        acc_max = 2147483647

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

                ts0 = int(last_ts[idx0])
                h0 = int(hot_state[idx0])

                # Treat unseen pixel as having a large dt0 so inc=0.
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

            # Update global rate EMA (events/tick)
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

            # Update hot_state from self dt0 (dt-dependent increment)
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

            # Baseline raw support (same polarity) as in EBF
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
                    if dt > tau:
                        continue

                    raw_w += (tau - dt)

            # Always update self state
            last_ts[idx0] = np.uint64(ti)
            last_pol[idx0] = np.int8(pi)
            hot_state[idx0] = np.int32(h0)

            raw = float(raw_w) * inv_tau

            r_pix = r_ema / n_pix
            if r_pix < 0.0:
                r_pix = 0.0

            # occupancy u in [0,1)
            hf = float(h0)
            u = hf / (hf + float(tr) + eps)
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

    return ebf_s36_scores_stream


__all__ = [
    "S36NoiseSurpriseZScoreStateOccupancyParams",
    "s36_noise_surprise_zscore_stateoccupancy_params_from_env",
    "try_build_s36_noise_surprise_zscore_stateoccupancy_scores_kernel",
]
