from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class S35NoiseSurpriseZScorePixelStateParams:
    """Parameters for Part2 s35 (pixel-state-conditioned noise surprise z-score).

    Motivation:
    - s28 standardizes raw EBF support using a global noise-rate model.
    - Under heavy noise, *hot pixels* violate the global-rate assumption:
      they behave as if their local noise rate is much higher than average.

    Core idea:
    - Keep s28 backbone: z-score under a simple Poisson noise model.
    - Maintain a cheap per-pixel "hot state" H (single leaky accumulator) as an
      online proxy of abnormal persistence at that pixel:

        H <- max(0, H - dt0) + tau

      where dt0 is the time since the last event at this pixel.
    - Use H to modulate the per-pixel effective noise rate:

        r_eff = r_pix * (1 + gamma * clip(H/tau, 0, hmax))

      then compute mu/var with r_eff instead of r_pix.

    This keeps a unified "adaptive null model" story (no posterior penalty term)
    while adding only O(1) work per event and one int32 state per pixel.

    Constraints: streaming, single-pass, O(r^2), Numba-only.

    Hyperparameters:
    - tau_rate_us: EMA time constant (microseconds) for global rate estimate.
      0 means auto (use tau_us for the current sweep point).
    - gamma: strength of pixel-state modulation (>=0).
    - hmax: clamp for H/tau to avoid extreme scaling (>=0).
    """

    tau_rate_us: int = 0
    gamma: float = 1.0
    hmax: float = 8.0


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


def s35_noise_surprise_zscore_pixelstate_params_from_env(
    env: dict[str, str] | None = None,
) -> S35NoiseSurpriseZScorePixelStateParams:
    """Read s35 parameters from environment.

    - MYEVS_EBF_S35_TAU_RATE_US (default 0; 0 means auto)
    - MYEVS_EBF_S35_GAMMA (default 1.0; clamped to >=0)
    - MYEVS_EBF_S35_HMAX (default 8.0; clamped to >=0)
    """

    if env is None:
        env = os.environ  # type: ignore[assignment]

    tau_rate_us = _env_int(env, "MYEVS_EBF_S35_TAU_RATE_US", 0)
    gamma = float(max(0.0, _env_float(env, "MYEVS_EBF_S35_GAMMA", 1.0)))
    hmax = float(max(0.0, _env_float(env, "MYEVS_EBF_S35_HMAX", 8.0)))

    return S35NoiseSurpriseZScorePixelStateParams(
        tau_rate_us=int(tau_rate_us),
        gamma=float(gamma),
        hmax=float(hmax),
    )


def try_build_s35_noise_surprise_zscore_pixelstate_scores_kernel():
    """Build and return Numba kernel for s35 score streaming.

    Returns None if numba is unavailable.

    Kernel signature:
      (t,x,y,p,width,height,radius_px,tau_ticks,tau_rate_ticks,gamma,hmax,last_ts,last_pol,hot_state,rate_ema,scores_out) -> None

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
    def ebf_s35_scores_stream(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        width: int,
        height: int,
        radius_px: int,
        tau_ticks: int,
        tau_rate_ticks: int,
        gamma: float,
        hmax: float,
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

        g = float(gamma)
        if not np.isfinite(g) or g < 0.0:
            g = 0.0

        hm = float(hmax)
        if not np.isfinite(hm) or hm < 0.0:
            hm = 0.0

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

                # Update hot_state from self dt0
                ts0 = int(last_ts[idx0])
                h0 = int(hot_state[idx0])
                if ts0 == 0:
                    dt0 = 0
                else:
                    dt0 = ti - ts0
                    if dt0 < 0:
                        dt0 = -dt0

                if dt0 != 0:
                    h0 = h0 - dt0
                    if h0 < 0:
                        h0 = 0

                h0 = h0 + tau
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

            # Update hot_state from self dt0
            ts0 = int(last_ts[idx0])
            h0 = int(hot_state[idx0])

            if ts0 == 0:
                dt0 = 0
            else:
                dt0 = ti - ts0
                if dt0 < 0:
                    dt0 = -dt0

            if dt0 != 0:
                h0 = h0 - dt0
                if h0 < 0:
                    h0 = 0

            h0 = h0 + tau
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

            # pixel-state-conditioned effective per-pixel rate
            h_norm = float(h0) * inv_tau  # H/tau
            if h_norm < 0.0:
                h_norm = 0.0
            if hm > 0.0 and h_norm > hm:
                h_norm = hm
            if hm <= 0.0:
                h_norm = 0.0

            r_eff = r_pix * (1.0 + g * h_norm)
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

    return ebf_s35_scores_stream


__all__ = [
    "S35NoiseSurpriseZScorePixelStateParams",
    "s35_noise_surprise_zscore_pixelstate_params_from_env",
    "try_build_s35_noise_surprise_zscore_pixelstate_scores_kernel",
]
