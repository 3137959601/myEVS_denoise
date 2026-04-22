from __future__ import annotations

"""Part2 s78: s77 + cross-pol local activity penalty (still no EBF).

Motivation
----------
7.21 notes point out a key failure mode for s76: ignoring polarity lets ON/OFF
mix into pseudo-edges, especially under mid/heavy flicker/hot regions.

s77 addressed this partially by scoring only within the event's polarity
channel, plus local normalization and a 2x multi-scale consistency term.

s78 adds one more *targeted* ingredient without introducing new sweep dims:
- penalize events whose *opposite polarity* channel is also locally active,
  which is a proxy for cross-pol flicker / alternating noise.

Definition (per event)
----------------------
Base score is s77:

    C_s = |∇A_s| / (mean_3x3(A_s) + eps)
    C_l = |∇A_l| / (mean_3x3(A_l) + eps)
    S_cont = (C_s + C_l) / (C_s + C_l + |C_s - C_l| + eps)
    base = (C_s + C_l) * S_cont

Cross-pol penalty uses the opposite polarity short-scale local mean:

    m_opp = mean_3x3(A_opp_s)
    score = base / (1 + beta * m_opp)

beta is fixed (not swept) to avoid new hyper-parameters.
"""

import numpy as np


def try_build_s78_aocc_polnorm_multiscale_crosspol_penalty_scores_kernel():
    """Build and return Numba kernel for s78 streaming scores.

    Returns None if numba is unavailable.

    Kernel signature:
        (t,x,y,p,width,height,radius_px,tau_ticks,
         last_ts_pos_s,last_as_pos_s,last_ts_pos_l,last_as_pos_l,
         last_ts_neg_s,last_as_neg_s,last_ts_neg_l,last_as_neg_l,
         scores_out) -> None

    - tau_l is internally fixed to 2*tau.
    - beta is fixed constant.
    - For each event, compute s77 on the matching polarity and penalize by
      opposite-pol short-scale local mean activity.
    """

    try:
        from numba import njit  # type: ignore
    except Exception:
        return None

    @njit(cache=True)
    def _activity_at(ti: int, idx: int, tau: float, last_t: np.ndarray, last_a: np.ndarray) -> float:
        ts = int(last_t[idx])
        if ts <= 0:
            return 0.0
        dt = ti - ts
        if dt <= 0:
            return float(last_a[idx])
        return float(last_a[idx]) * float(np.exp(-float(dt) / tau))

    @njit(cache=True)
    def ebf_s78_scores_stream(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        width: int,
        height: int,
        radius_px: int,
        tau_ticks: int,
        last_ts_pos_s: np.ndarray,
        last_as_pos_s: np.ndarray,
        last_ts_pos_l: np.ndarray,
        last_as_pos_l: np.ndarray,
        last_ts_neg_s: np.ndarray,
        last_as_neg_s: np.ndarray,
        last_ts_neg_l: np.ndarray,
        last_as_neg_l: np.ndarray,
        scores_out: np.ndarray,
    ) -> None:
        n = int(t.shape[0])
        w = int(width)
        h = int(height)

        step = int(radius_px)
        if step < 1:
            step = 1
        if step > 8:
            step = 8

        tau_s_i = int(tau_ticks)
        if tau_s_i < 1:
            tau_s_i = 1
        tau_l_i = tau_s_i * 2
        tau_s = float(tau_s_i)
        tau_l = float(tau_l_i)

        eps = 1e-3
        a_cap = 50.0
        beta = 0.75

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            if xi < 0 or xi >= w or yi < 0 or yi >= h:
                scores_out[i] = 0.0
                continue

            ti = int(t[i])
            pi = int(p[i])

            if pi > 0:
                # same polarity
                last_ts_s = last_ts_pos_s
                last_as_s = last_as_pos_s
                last_ts_l = last_ts_pos_l
                last_as_l = last_as_pos_l
                # opposite polarity (short only)
                opp_ts_s = last_ts_neg_s
                opp_as_s = last_as_neg_s
            else:
                last_ts_s = last_ts_neg_s
                last_as_s = last_as_neg_s
                last_ts_l = last_ts_neg_l
                last_as_l = last_as_neg_l
                opp_ts_s = last_ts_pos_s
                opp_as_s = last_as_pos_s

            idx0 = yi * w + xi

            # Update center activity for both scales.
            a0s = _activity_at(ti, idx0, tau_s, last_ts_s, last_as_s) + 1.0
            if a0s > a_cap:
                a0s = a_cap
            last_ts_s[idx0] = np.uint64(ti)
            last_as_s[idx0] = np.float32(a0s)

            a0l = _activity_at(ti, idx0, tau_l, last_ts_l, last_as_l) + 1.0
            if a0l > a_cap:
                a0l = a_cap
            last_ts_l[idx0] = np.uint64(ti)
            last_as_l[idx0] = np.float32(a0l)

            # 3x3 sample grid bounds.
            x0 = xi - step
            x1 = xi
            x2 = xi + step
            if x0 < 0:
                x0 = 0
            if x2 >= w:
                x2 = w - 1

            y0 = yi - step
            y1 = yi
            y2 = yi + step
            if y0 < 0:
                y0 = 0
            if y2 >= h:
                y2 = h - 1

            idx00 = y0 * w + x0
            idx01 = y0 * w + x1
            idx02 = y0 * w + x2
            idx10 = y1 * w + x0
            idx11 = y1 * w + x1
            idx12 = y1 * w + x2
            idx20 = y2 * w + x0
            idx21 = y2 * w + x1
            idx22 = y2 * w + x2

            # Same-pol short samples.
            s00 = _activity_at(ti, idx00, tau_s, last_ts_s, last_as_s)
            s01 = _activity_at(ti, idx01, tau_s, last_ts_s, last_as_s)
            s02 = _activity_at(ti, idx02, tau_s, last_ts_s, last_as_s)
            s10 = _activity_at(ti, idx10, tau_s, last_ts_s, last_as_s)
            s11 = a0s if (idx11 == idx0) else _activity_at(ti, idx11, tau_s, last_ts_s, last_as_s)
            s12 = _activity_at(ti, idx12, tau_s, last_ts_s, last_as_s)
            s20 = _activity_at(ti, idx20, tau_s, last_ts_s, last_as_s)
            s21 = _activity_at(ti, idx21, tau_s, last_ts_s, last_as_s)
            s22 = _activity_at(ti, idx22, tau_s, last_ts_s, last_as_s)

            # Same-pol long samples.
            l00 = _activity_at(ti, idx00, tau_l, last_ts_l, last_as_l)
            l01 = _activity_at(ti, idx01, tau_l, last_ts_l, last_as_l)
            l02 = _activity_at(ti, idx02, tau_l, last_ts_l, last_as_l)
            l10 = _activity_at(ti, idx10, tau_l, last_ts_l, last_as_l)
            l11 = a0l if (idx11 == idx0) else _activity_at(ti, idx11, tau_l, last_ts_l, last_as_l)
            l12 = _activity_at(ti, idx12, tau_l, last_ts_l, last_as_l)
            l20 = _activity_at(ti, idx20, tau_l, last_ts_l, last_as_l)
            l21 = _activity_at(ti, idx21, tau_l, last_ts_l, last_as_l)
            l22 = _activity_at(ti, idx22, tau_l, last_ts_l, last_as_l)

            # Opp-pol short local mean (penalty only).
            o00 = _activity_at(ti, idx00, tau_s, opp_ts_s, opp_as_s)
            o01 = _activity_at(ti, idx01, tau_s, opp_ts_s, opp_as_s)
            o02 = _activity_at(ti, idx02, tau_s, opp_ts_s, opp_as_s)
            o10 = _activity_at(ti, idx10, tau_s, opp_ts_s, opp_as_s)
            o11 = _activity_at(ti, idx11, tau_s, opp_ts_s, opp_as_s)
            o12 = _activity_at(ti, idx12, tau_s, opp_ts_s, opp_as_s)
            o20 = _activity_at(ti, idx20, tau_s, opp_ts_s, opp_as_s)
            o21 = _activity_at(ti, idx21, tau_s, opp_ts_s, opp_as_s)
            o22 = _activity_at(ti, idx22, tau_s, opp_ts_s, opp_as_s)

            # s77 base.
            gxs = (s02 + 2.0 * s12 + s22) - (s00 + 2.0 * s10 + s20)
            gys = (s20 + 2.0 * s21 + s22) - (s00 + 2.0 * s01 + s02)
            gxl = (l02 + 2.0 * l12 + l22) - (l00 + 2.0 * l10 + l20)
            gyl = (l20 + 2.0 * l21 + l22) - (l00 + 2.0 * l01 + l02)

            grad_s = float(np.hypot(gxs, gys))
            grad_l = float(np.hypot(gxl, gyl))

            mean_s = (s00 + s01 + s02 + s10 + s11 + s12 + s20 + s21 + s22) / 9.0
            mean_l = (l00 + l01 + l02 + l10 + l11 + l12 + l20 + l21 + l22) / 9.0

            cs = grad_s / (mean_s + eps)
            cl = grad_l / (mean_l + eps)

            denom = (cs + cl) + abs(cs - cl) + eps
            s_cont = (cs + cl) / denom
            base = (cs + cl) * s_cont

            mean_opp = (o00 + o01 + o02 + o10 + o11 + o12 + o20 + o21 + o22) / 9.0
            scores_out[i] = base / (1.0 + beta * mean_opp)

    return ebf_s78_scores_stream
