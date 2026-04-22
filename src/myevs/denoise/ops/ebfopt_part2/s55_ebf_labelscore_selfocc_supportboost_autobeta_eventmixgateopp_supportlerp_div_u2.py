from __future__ import annotations

import math

import numpy as np


def try_build_s55_ebf_labelscore_selfocc_supportboost_autobeta_eventmixgateopp_supportlerp_div_u2_scores_kernel():
    """Build and return Numba kernel for s55 score streaming.

    s55 is a shape tweak of s53 (no new hyperparameters).

    Background (s53):
        mix = raw_opp / (raw_same + raw_opp)
        alpha_eff = (1-mix)^2 * sqrt(sfrac)
        raw = raw_same + alpha_eff * raw_opp

    Issue observed:
        s54 (fourth-root on sfrac) can slightly improve mid but tends to hurt heavy,
        suggesting that boosting opp weight for *narrow-support* events is risky.

    s55 idea:
        Loosen the mix gate only when spatial support is wide (large sfrac), while
        keeping the original sqrt(sfrac) suppression for narrow-support patterns.

        Let gate = (1-mix)^2.
        Support-adaptive gate (linear lerp toward 1 when sfrac->1):
            gate_eff = gate + (1-gate) * sfrac
        Then:
            alpha_eff = gate_eff * sqrt(sfrac)

    Returns None if numba is unavailable.
    """

    try:
        from numba import njit  # type: ignore
    except Exception:
        return None

    @njit(cache=False)
    def ebf_s55_scores_stream(
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
        beta_state: np.ndarray,
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

        # Fixed (non-exposed) adaptation horizon (events).
        N = 4096.0

        # Tie tau_r to tau without exposing a knob.
        tr = tau // 2
        if tr <= 0:
            tr = 1

        inv_tau = 1.0 / float(tau)
        eps = 1e-6
        acc_max = 2147483647

        if rr <= 0:
            for i in range(n):
                xi = int(x[i])
                yi = int(y[i])
                ti = int(t[i])
                if xi < 0 or xi >= w or yi < 0 or yi >= h:
                    scores_out[i] = 0.0
                    continue
                pi = 1 if int(p[i]) > 0 else -1
                idx0 = yi * w + xi
                last_ts[idx0] = np.uint64(ti)
                last_pol[idx0] = np.int8(pi)
                scores_out[i] = np.inf
            return

        b = float(beta_state[0])
        if b < 0.0:
            b = 0.0
        if b > 1.0:
            b = 1.0

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            ti = int(t[i])

            if xi < 0 or xi >= w or yi < 0 or yi >= h:
                scores_out[i] = 0.0
                continue

            pi = 1 if int(p[i]) > 0 else -1
            idx0 = yi * w + xi

            # Update hot_state (same linear-decay accumulator as s36+ family).
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
            opp_w = 0
            cnt_support = 0

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

            for yy in range(y0, y1 + 1):
                base = yy * w
                for xx in range(x0, x1 + 1):
                    if xx == xi and yy == yi:
                        continue

                    idx = base + xx
                    pol = int(last_pol[idx])
                    if pol != pi and pol != -pi:
                        continue

                    ts = int(last_ts[idx])
                    if ts == 0:
                        continue

                    dt = ti - ts
                    if dt < 0:
                        dt = -dt
                    if dt > tau:
                        continue

                    w_age = tau - dt
                    if w_age <= 0:
                        continue

                    if pol == pi:
                        raw_w += w_age
                        cnt_support += 1
                    else:
                        opp_w += w_age

            # Always update self state.
            last_ts[idx0] = np.uint64(ti)
            last_pol[idx0] = np.int8(pi)
            hot_state[idx0] = np.int32(h0)

            # u_self from hot_state.
            hf = float(h0)
            u_self = hf / (hf + float(tr) + eps)
            if u_self < 0.0:
                u_self = 0.0
            if u_self > 1.0:
                u_self = 1.0

            # Update adaptive beta_state as running mean of u_self.
            b = b + (u_self - b) / N
            if b < 0.0:
                b = 0.0
            if b > 1.0:
                b = 1.0

            cnt_possible = (x1 - x0 + 1) * (y1 - y0 + 1) - 1
            if cnt_possible <= 0:
                sfrac = 0.0
            else:
                sfrac = float(cnt_support) / float(cnt_possible)
                if sfrac < 0.0:
                    sfrac = 0.0
                if sfrac > 1.0:
                    sfrac = 1.0

            # Per-event polarity mix.
            denom = float(raw_w + opp_w)
            mix = 0.0
            if denom > 0.0:
                mix = float(opp_w) / denom
                if mix < 0.0:
                    mix = 0.0
                if mix > 1.0:
                    mix = 1.0

            purity = 1.0 - mix
            if purity < 0.0:
                purity = 0.0

            gate = purity * purity
            if gate < 0.0:
                gate = 0.0
            if gate > 1.0:
                gate = 1.0

            if sfrac > 0.0:
                # gate_eff = gate + (1-gate)*sfrac
                gate_eff = gate + (1.0 - gate) * sfrac
                alpha_eff = gate_eff * math.sqrt(sfrac)
            else:
                alpha_eff = 0.0

            raw_gated = (float(raw_w) + float(alpha_eff) * float(opp_w)) * inv_tau
            base_score = float(raw_gated / (1.0 + (u_self * u_self)))
            scores_out[i] = base_score * (1.0 + b * sfrac)

        beta_state[0] = b

    return ebf_s55_scores_stream


__all__ = [
    "try_build_s55_ebf_labelscore_selfocc_supportboost_autobeta_eventmixgateopp_supportlerp_div_u2_scores_kernel",
]
