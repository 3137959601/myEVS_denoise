from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class S38NoiseSurpriseZScoreStateOccNbOccFusionParams:
    """Parameters for Part2 s38 (state-occupancy + neighborhood-occupancy fusion).

    Motivation (builds on s36/s37 introspection):
    - s36/s37 derive occupancy u solely from per-pixel self-history (hot_state H).
      Many hard FP modes (hotmask / bursty blobs) can keep slipping through when
      the *kept* FP events tend to have u below the strong-suppression regime.
    - s38 keeps the same s28 backbone and the same self hot_state update as s36,
      but additionally estimates a *neighborhood occupancy* from recent activity
      (any polarity) within the same O(r^2) neighborhood traversal.

    Core idea:
    - Keep global rate EMA -> r_pix.
    - Keep self hot_state H update (tick units):
        H <- max(0, H - dt0) + max(0, tau - dt0)
      and self occupancy:
        u_self = H / (H + tau_r)

    - In the same neighbor loop used for raw support, accumulate a recency mass
      for *any polarity* neighbors:
        raw_all = sum_j max(0, 1 - dt_j/tau)   (scaled by tau in the kernel)
      and map it to a neighborhood occupancy (dimensionless, in [0,1)):
        u_nb = raw_all / (raw_all + m)
      where m is the number of neighbors in the (2r+1)x(2r+1) window.

    - Fuse occupancies as a probability-union (parameter-free):
        u = 1 - (1 - u_self) * (1 - u_nb)

    - Modulate null-model rate smoothly as in s36:
        r_eff = r_pix * (1 + u)^2

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


def s38_noise_surprise_zscore_stateocc_nbocc_fusion_params_from_env(
    env: dict[str, str] | None = None,
) -> S38NoiseSurpriseZScoreStateOccNbOccFusionParams:
    """Read s38 parameters from environment.

    - MYEVS_EBF_S38_TAU_RATE_US (default 0; 0 means auto)
    """

    if env is None:
        env = os.environ  # type: ignore[assignment]

    tau_rate_us = _env_int(env, "MYEVS_EBF_S38_TAU_RATE_US", 0)
    return S38NoiseSurpriseZScoreStateOccNbOccFusionParams(tau_rate_us=int(tau_rate_us))


def try_build_s38_noise_surprise_zscore_stateocc_nbocc_fusion_scores_kernel():
    """Build and return Numba kernel for s38 score streaming.

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
    def ebf_s38_scores_stream(
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

            # Update self hot_state from dt0
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

            # Neighborhood recency masses
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

            # Always update self state
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

            raw_all = float(raw_all_w) * inv_tau  # in [0, m]
            u_nb = raw_all / (raw_all + float(m) + eps)
            if u_nb < 0.0:
                u_nb = 0.0
            if u_nb > 1.0:
                u_nb = 1.0

            u = 1.0 - (1.0 - u_self) * (1.0 - u_nb)
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

    return ebf_s38_scores_stream


__all__ = [
    "S38NoiseSurpriseZScoreStateOccNbOccFusionParams",
    "s38_noise_surprise_zscore_stateocc_nbocc_fusion_params_from_env",
    "try_build_s38_noise_surprise_zscore_stateocc_nbocc_fusion_scores_kernel",
]
