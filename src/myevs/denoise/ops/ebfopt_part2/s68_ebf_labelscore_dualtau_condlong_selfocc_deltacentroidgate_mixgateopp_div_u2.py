from __future__ import annotations

import numpy as np


def try_build_s68_ebf_labelscore_dualtau_condlong_selfocc_deltacentroidgate_mixgateopp_div_u2_scores_kernel():
    """Build and return Numba kernel for s68 score streaming.

    s68: conditional long-delta (short=tau, long=2*tau, delta-only boost).

    Trajectory proxy focuses on *long-only* same-pol support (dt in (tau, tau_long])
    to avoid being dominated by dense local short-window clusters:

        c_delta = weighted centroid of same-pol neighbors with dt>tau
        c_traj = clip(||c_delta - current|| / r, 0, 1)
        g_long = (1 - u_self) * c_traj

    Still streaming, single-pass, O(r^2), no new sweep hyperparameters.

    Returns None if numba is unavailable.
    """

    try:
        from numba import njit  # type: ignore
    except Exception:
        return None

    @njit(cache=False)
    def ebf_s68_scores_stream(
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
        mix_state: np.ndarray,
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

        tau_long = tau * 2
        if tau_long <= 0:
            tau_long = tau

        N = 4096.0

        tr = tau // 2
        if tr <= 0:
            tr = 1

        inv_tau = 1.0 / float(tau)
        inv_tau_long = 1.0 / float(tau_long)
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

        mstate = float(mix_state[0])
        if mstate < 0.0:
            mstate = 0.0
        if mstate > 1.0:
            mstate = 1.0

        denom_r = float(rr) + eps

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            ti = int(t[i])

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
            opp_w = 0
            raw_w_long = 0
            opp_w_long = 0

            w_delta = 0.0
            sumx_delta = 0.0
            sumy_delta = 0.0

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
                    if dt > tau_long:
                        continue

                    w_age_long = tau_long - dt
                    if w_age_long > 0:
                        if pol == pi:
                            raw_w_long += w_age_long
                            if dt > tau:
                                wf = float(w_age_long)
                                w_delta += wf
                                sumx_delta += wf * float(xx)
                                sumy_delta += wf * float(yy)
                        else:
                            opp_w_long += w_age_long

                    if dt > tau:
                        continue

                    w_age = tau - dt
                    if w_age <= 0:
                        continue

                    if pol == pi:
                        raw_w += w_age
                    else:
                        opp_w += w_age

            last_ts[idx0] = np.uint64(ti)
            last_pol[idx0] = np.int8(pi)
            hot_state[idx0] = np.int32(h0)

            hf = float(h0)
            u_self = hf / (hf + float(tr) + eps)
            if u_self < 0.0:
                u_self = 0.0
            if u_self > 1.0:
                u_self = 1.0

            denom = float(raw_w + opp_w)
            mix = 0.0
            if denom > 0.0:
                mix = float(opp_w) / denom
                if mix < 0.0:
                    mix = 0.0
                if mix > 1.0:
                    mix = 1.0

            mstate = mstate + (mix - mstate) / N
            if mstate < 0.0:
                mstate = 0.0
            if mstate > 1.0:
                mstate = 1.0

            alpha_eff = 1.0 - mstate
            if alpha_eff < 0.0:
                alpha_eff = 0.0
            alpha_eff = alpha_eff * alpha_eff

            raw_short = (float(raw_w) + float(alpha_eff) * float(opp_w)) * inv_tau
            raw_long = (float(raw_w_long) + float(alpha_eff) * float(opp_w_long)) * inv_tau_long

            delta = raw_long - raw_short
            if delta < 0.0:
                delta = 0.0
            if delta > raw_short:
                delta = raw_short

            c_traj = 0.0
            if w_delta > 0.0:
                cx = float(sumx_delta) / float(w_delta)
                cy = float(sumy_delta) / float(w_delta)
                dx = float(cx - float(xi))
                dy = float(cy - float(yi))
                dist = (dx * dx + dy * dy) ** 0.5
                c_traj = dist / denom_r
                if c_traj < 0.0:
                    c_traj = 0.0
                if c_traj > 1.0:
                    c_traj = 1.0

            g_long = (1.0 - u_self) * c_traj
            if g_long < 0.0:
                g_long = 0.0
            if g_long > 1.0:
                g_long = 1.0

            raw_cond = raw_short + g_long * delta
            scores_out[i] = float(raw_cond / (1.0 + (u_self * u_self)))

        mix_state[0] = mstate
        beta_state[0] = float(beta_state[0])

    return ebf_s68_scores_stream


__all__ = [
    "try_build_s68_ebf_labelscore_dualtau_condlong_selfocc_deltacentroidgate_mixgateopp_div_u2_scores_kernel",
]
