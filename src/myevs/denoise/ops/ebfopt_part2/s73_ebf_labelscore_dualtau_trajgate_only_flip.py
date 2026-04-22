from __future__ import annotations

import numpy as np


def try_build_s73_ebf_labelscore_dualtau_trajgate_only_flip_scores_kernel():
    r"""Build and return Numba kernel for s73 score streaming.

    s73 goal:
    - Same as s72 (extremely simplified), but **flip** the trajectory gate direction.

    Definitions (same as s72):
    - Short window: tau
    - Long window: 2*tau
    - Same-polarity evidence (baseline-style triangular age weight):
        raw_short = \sum_{dt<=tau} (tau-dt)/tau
        raw_long  = \sum_{dt<=2tau} (2tau-dt)/(2tau)
    - Centroid drift ratio:
        c_traj = clip( ||c_long - c_short|| / radius_px, 0, 1 )

    Flip:
    - Use a *stability* gate instead:
        g = 1 - c_traj

    Score:
    - Delta-only long boost gated by g:
        score = raw_short + g * max(0, raw_long - raw_short)

    Returns None if numba is unavailable.
    """

    try:
        from numba import njit  # type: ignore
    except Exception:
        return None

    @njit(cache=False)
    def ebf_s73_scores_stream(
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

        eps = 1e-6
        inv_tau = 1.0 / float(tau)
        inv_tau_long = 1.0 / float(tau_long)
        denom_r = float(rr) + eps

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

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            if xi < 0 or xi >= w or yi < 0 or yi >= h:
                scores_out[i] = 0.0
                continue

            ti = int(t[i])
            pi = 1 if int(p[i]) > 0 else -1

            idx0 = yi * w + xi

            raw_w = 0
            raw_w_long = 0

            sum_w_s = 0.0
            sum_x_s = 0.0
            sum_y_s = 0.0
            sum_w_l = 0.0
            sum_x_l = 0.0
            sum_y_l = 0.0

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

                    if int(last_pol[idx]) != pi:
                        continue

                    ts = int(last_ts[idx])
                    if ts == 0:
                        continue

                    dt = (ti - ts) if ti >= ts else (ts - ti)
                    if dt > tau_long:
                        continue

                    w_age_long = tau_long - dt
                    if w_age_long > 0:
                        raw_w_long += w_age_long
                        wf_l = float(w_age_long)
                        sum_w_l += wf_l
                        sum_x_l += wf_l * float(xx)
                        sum_y_l += wf_l * float(yy)

                    if dt > tau:
                        continue

                    w_age = tau - dt
                    if w_age <= 0:
                        continue

                    raw_w += w_age
                    wf_s = float(w_age)
                    sum_w_s += wf_s
                    sum_x_s += wf_s * float(xx)
                    sum_y_s += wf_s * float(yy)

            last_ts[idx0] = np.uint64(ti)
            last_pol[idx0] = np.int8(pi)

            raw_short = float(raw_w) * inv_tau
            raw_long = float(raw_w_long) * inv_tau_long

            delta = raw_long - raw_short
            if delta < 0.0:
                delta = 0.0

            c_traj = 0.0
            if sum_w_s > 0.0 and sum_w_l > 0.0:
                cx_s = sum_x_s / sum_w_s
                cy_s = sum_y_s / sum_w_s
                cx_l = sum_x_l / sum_w_l
                cy_l = sum_y_l / sum_w_l
                dx = cx_l - cx_s
                dy = cy_l - cy_s
                d = (dx * dx + dy * dy) ** 0.5
                c_traj = d / denom_r
                if c_traj < 0.0:
                    c_traj = 0.0
                if c_traj > 1.0:
                    c_traj = 1.0

            g = 1.0 - c_traj
            if g < 0.0:
                g = 0.0
            if g > 1.0:
                g = 1.0

            scores_out[i] = raw_short + g * delta

    return ebf_s73_scores_stream
