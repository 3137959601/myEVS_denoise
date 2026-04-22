from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class S44EbfLabelscoreSelfOccDivU2Params:
    """Parameters for Part2 s44 (baseline EBF labelscore with self-occupancy penalty).

    Motivation
    - Baseline EBF labelscore (raw same-polarity recency sum) is much better on light.
    - s39+ (z-score/null-model modulation) can help heavy, but tends to hurt light ranking.

    Core idea (no new hyperparams)
    - Keep baseline score `raw` exactly as EBF:

        raw = sum_{nei same-pol} (tau - dt)/tau

    - Compute a self-occupancy hotness proxy u_self in [0,1] (same construction as s36/s39),
      then apply a gentle penalty:

        score = raw / (1 + u_self^2)

      This preserves baseline ranking for normal pixels (u_self≈0) but suppresses persistent
      hot pixels (u_self→1) which are dominant in heavy noise.

    Hyperparameters (env)
    - tau_rate_us: EMA / decay time constant (microseconds) used for self-occupancy normalization.
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


def s44_ebf_labelscore_selfocc_div_u2_params_from_env(
    env: dict[str, str] | None = None,
) -> S44EbfLabelscoreSelfOccDivU2Params:
    """Read s44 parameters from environment.

    - MYEVS_EBF_S44_TAU_RATE_US (default 0; 0 means auto)
    """

    if env is None:
        env = os.environ  # type: ignore[assignment]

    tau_rate_us = _env_int(env, "MYEVS_EBF_S44_TAU_RATE_US", 0)
    return S44EbfLabelscoreSelfOccDivU2Params(tau_rate_us=int(tau_rate_us))


def try_build_s44_ebf_labelscore_selfocc_div_u2_scores_kernel():
    """Build and return Numba kernel for s44 score streaming.

    Returns None if numba is unavailable.
    """

    try:
        from numba import njit  # type: ignore
    except Exception:
        return None

    @njit(cache=True)
    def ebf_s44_scores_stream(
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

            # Update hot_state with linear-decay accumulator (same as s36+ family).
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

                    dt = ti - ts
                    if dt < 0:
                        dt = -dt
                    if dt > tau:
                        continue

                    raw_w += (tau - dt)

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

            d = 1.0 + (u_self * u_self)
            scores_out[i] = float(raw / d)

    return ebf_s44_scores_stream


__all__ = [
    "S44EbfLabelscoreSelfOccDivU2Params",
    "s44_ebf_labelscore_selfocc_div_u2_params_from_env",
    "try_build_s44_ebf_labelscore_selfocc_div_u2_scores_kernel",
]
