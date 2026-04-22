from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class S47EbfLabelscoreSelfOccAbnDivU2Params:
    """Parameters for Part2 s47 (s44 with an *abnormal activity* self-state).

    Summary
    - Keep baseline EBF raw labelscore unchanged.
    - Keep the s44 outer score shape unchanged:

        score = raw / (1 + u_self^2)

    - Only change how self hot_state (H) accumulates: scale the increment by how
      "abnormally fast" this pixel fires compared to its local neighborhood.

    Motivation (based on s45/s46 evidence)
    - s45 shows a hard gate on low u is not helpful.
    - s46 shows a much steeper high-u penalty can hurt mid/heavy.
    - So the likely bottleneck is not the outer penalty shape, but the purity of u_self.

    Abnormal-activity scaling
    - Let dt0 be the time since last event at this pixel.
    - Let dt_nb be the minimum dt to any neighbor's last timestamp in the window.
    - Define a cheap abnormality factor:

        g = clip((dt_nb - dt0) / (dt_nb + eps), 0, 1)

      Intuition:
      - If this pixel fires much faster than its neighbors (dt0 << dt_nb), g ~ 1.
      - If neighbors are equally fast or faster (dt0 >= dt_nb), g ~ 0.

    Then use inc = max(0, tau - dt0) * g when updating hot_state.

    Hyperparameters (env)
    - tau_rate_us: time constant (microseconds) used to normalize self hot_state.
      0 means auto (use current tau_us).
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


def s47_ebf_labelscore_selfocc_abn_div_u2_params_from_env(
    env: dict[str, str] | None = None,
) -> S47EbfLabelscoreSelfOccAbnDivU2Params:
    """Read s47 parameters from environment.

    - MYEVS_EBF_S47_TAU_RATE_US (default 0; 0 means auto)
    """

    if env is None:
        env = os.environ  # type: ignore[assignment]

    tau_rate_us = _env_int(env, "MYEVS_EBF_S47_TAU_RATE_US", 0)
    return S47EbfLabelscoreSelfOccAbnDivU2Params(tau_rate_us=int(tau_rate_us))


def try_build_s47_ebf_labelscore_selfocc_abn_div_u2_scores_kernel():
    """Build and return Numba kernel for s47 score streaming.

    Returns None if numba is unavailable.
    """

    try:
        from numba import njit  # type: ignore
    except Exception:
        return None

    @njit(cache=True)
    def ebf_s47_scores_stream(
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

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            ti = int(t[i])

            if xi < 0 or xi >= w or yi < 0 or yi >= h:
                scores_out[i] = 0.0
                continue

            pi = 1 if int(p[i]) > 0 else -1
            idx0 = yi * w + xi

            # Update self hot_state (s44-like) but with abnormal-activity scaling.
            ts0 = int(last_ts[idx0])
            h0 = int(hot_state[idx0])

            dt0 = tau if ts0 == 0 else (ti - ts0)
            if dt0 < 0:
                dt0 = -dt0

            if dt0 != 0:
                h0 = h0 - dt0
                if h0 < 0:
                    h0 = 0

            # Neighborhood min dt (any polarity) within the same window we already scan.
            dt_nb_min = tau

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

                    if dt < dt_nb_min:
                        dt_nb_min = dt

                    if dt > tau:
                        continue

                    if int(last_pol[idx]) != pi:
                        continue

                    raw_w += (tau - dt)

            # Abnormality factor g in [0,1].
            g = (float(dt_nb_min) - float(dt0)) / (float(dt_nb_min) + eps)
            if g < 0.0:
                g = 0.0
            if g > 1.0:
                g = 1.0

            inc0 = tau - dt0
            if inc0 > 0:
                inc = int(float(inc0) * g)
                if inc > 0:
                    h0 = h0 + inc
                    if h0 > acc_max:
                        h0 = acc_max

            # Always update self state.
            last_ts[idx0] = np.uint64(ti)
            last_pol[idx0] = np.int8(pi)
            hot_state[idx0] = np.int32(h0)

            raw = float(raw_w) * inv_tau

            hf = float(h0)
            u_self = hf / (hf + float(tr) + eps)
            if u_self < 0.0:
                u_self = 0.0
            if u_self > 1.0:
                u_self = 1.0

            scores_out[i] = float(raw / (1.0 + (u_self * u_self)))

    return ebf_s47_scores_stream


__all__ = [
    "S47EbfLabelscoreSelfOccAbnDivU2Params",
    "s47_ebf_labelscore_selfocc_abn_div_u2_params_from_env",
    "try_build_s47_ebf_labelscore_selfocc_abn_div_u2_scores_kernel",
]
