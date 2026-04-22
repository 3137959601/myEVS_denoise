from __future__ import annotations

import numpy as np


def try_build_s74_ebf_labelscore_surprise_adaptive_null_fixed_scores_kernel():
    """Build and return Numba kernel for s74 score streaming.

    s74 goal (new base model, minimal knobs):
    - Replace the baseline "raw support" axis with a *noise-surprise z-score*
      under an adaptive null model that accounts for per-pixel hotness.

    Key lessons applied from README (s28-s35 chain):
    - Global-rate-only standardization (s28) is insufficient for heavy hotmask FP.
    - Injecting hotness into the null model (s35) can suppress hot pixels without
      a blunt global multiplier, but we want a minimal version for slim sweeps.

    Definition (streaming, single-pass, O(r^2)):
    - Baseline raw support (same polarity, triangular recency weight):
        raw = sum_{neighbors} 1{pol match} * max(0, 1 - dt/tau)
    - Global event-rate EMA (events/tick) with time constant tr:
        tr = max(1, tau/2)    (fixed, no sweep hyperparameter)
    - Per-pixel hot_state H (int32, tick units):
        H <- max(0, H - dt0) + tau
      where dt0 is the self inter-event time at this pixel.
    - Effective per-pixel rate:
        r_pix = rate_ema / (W*H)
        h = clip(H/tau, 0, hmax)
        r_eff = r_pix * (1 + gamma * h)
      with fixed constants: gamma=0.3, hmax=4.
    - Under Exp(r_eff) backward recurrence time and random polarity
      P(match)=0.5, we reuse s28's mu/var derivation and output:
        score = (raw - mu) / sqrt(var + eps)

    Returns None if numba is unavailable.

    Kernel signature:
      (t,x,y,p,width,height,radius_px,tau_ticks,last_ts,last_pol,hot_state,rate_ema,scores_out) -> None

    Arrays:
    - last_ts: uint64 per pixel
    - last_pol: int8 per pixel
    - hot_state: int32 per pixel
    - rate_ema: float64 array of shape (1,) storing global event rate (events/tick)
    """

    try:
        from numba import njit  # type: ignore
    except Exception:
        return None

    @njit(cache=False)
    def ebf_s74_scores_stream(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        width: int,
        height: int,
        radius_px: int,
        tau_ticks: int,
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

        # Fixed internal time constant for global-rate EMA: tr = tau/2
        tr = tau // 2
        if tr <= 0:
            tr = 1

        gamma = 0.3
        hmax = 4.0

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
            h_norm = float(h0) * inv_tau
            if h_norm < 0.0:
                h_norm = 0.0
            if h_norm > hmax:
                h_norm = hmax

            r_eff = r_pix * (1.0 + gamma * h_norm)
            if r_eff < 0.0:
                r_eff = 0.0

            a = r_eff * float(tau)  # dimensionless

            if a < 1e-3:
                # series expansions for stability
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

    return ebf_s74_scores_stream


__all__ = [
    "try_build_s74_ebf_labelscore_surprise_adaptive_null_fixed_scores_kernel",
]
