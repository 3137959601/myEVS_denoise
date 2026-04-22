from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class S50EbfLabelscoreSelfOccSupportBoostDivU2Params:
    """Parameters for Part2 s50 (s44 + gentle support-breadth boost).

    Motivation
    - s44 preserves baseline EBF labelscore raw ordering and applies a mild self-occupancy penalty:

        score = raw / (1 + u_self^2)

      It improves heavy mostly by allowing a lower global threshold (better recall) but does not
      substantially reduce remaining hotmask false positives.

    Hypothesis
    - Remaining FP-kept events tend to have high raw but narrower spatial support (fewer same-pol
      neighbors contributing within the (s,tau) window).

    Core idea
    - Keep baseline raw identical to EBF labelscore:

        raw = sum_{nei same-pol} (tau - dt)/tau

    - Keep s44 self-occupancy penalty:

        base = raw / (1 + u_self^2)

    - Add a gentle multiplicative boost based on the number of supporting same-pol neighbors:

        g = 1 + beta * min(1, cnt_support / cnt0)
        score = base * g

      This is intended to slightly favor broad-supported events (more likely signal) over
      narrow/coincidental support (more likely residual FP) while staying close to the raw axis.

    Hyperparameters (env)
    - MYEVS_EBF_S50_TAU_RATE_US: time constant for self-occupancy normalization (us). 0 means auto.
    - MYEVS_EBF_S50_BETA: boost strength (>=0). 0 disables boost (reduces to s44).
    - MYEVS_EBF_S50_CNT0: saturation support count (>0). Larger -> weaker boost.
    """

    tau_rate_us: int = 0
    beta: float = 0.0
    cnt0: int = 8


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


def s50_ebf_labelscore_selfocc_supportboost_div_u2_params_from_env(
    env: dict[str, str] | None = None,
) -> S50EbfLabelscoreSelfOccSupportBoostDivU2Params:
    """Read s50 parameters from environment."""

    if env is None:
        env = os.environ  # type: ignore[assignment]

    tau_rate_us = _env_int(env, "MYEVS_EBF_S50_TAU_RATE_US", 0)
    beta = _env_float(env, "MYEVS_EBF_S50_BETA", 0.0)
    if beta < 0.0:
        beta = 0.0
    cnt0 = _env_int(env, "MYEVS_EBF_S50_CNT0", 8)
    if cnt0 <= 0:
        cnt0 = 1

    return S50EbfLabelscoreSelfOccSupportBoostDivU2Params(
        tau_rate_us=int(tau_rate_us),
        beta=float(beta),
        cnt0=int(cnt0),
    )


def try_build_s50_ebf_labelscore_selfocc_supportboost_div_u2_scores_kernel():
    """Build and return Numba kernel for s50 score streaming.

    Returns None if numba is unavailable.
    """

    try:
        from numba import njit  # type: ignore
    except Exception:
        return None

    @njit(cache=True)
    def ebf_s50_scores_stream(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        width: int,
        height: int,
        radius_px: int,
        tau_ticks: int,
        tau_rate_ticks: int,
        beta: float,
        cnt0: int,
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

        b = float(beta)
        if b < 0.0:
            b = 0.0
        c0 = int(cnt0)
        if c0 <= 0:
            c0 = 1

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
                    cnt_support += 1

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

            base_score = float(raw / (1.0 + (u_self * u_self)))

            g = 1.0
            if b > 0.0:
                sfrac = float(cnt_support) / float(c0)
                if sfrac < 0.0:
                    sfrac = 0.0
                if sfrac > 1.0:
                    sfrac = 1.0
                g = 1.0 + b * sfrac

            scores_out[i] = base_score * g

    return ebf_s50_scores_stream


__all__ = [
    "S50EbfLabelscoreSelfOccSupportBoostDivU2Params",
    "s50_ebf_labelscore_selfocc_supportboost_div_u2_params_from_env",
    "try_build_s50_ebf_labelscore_selfocc_supportboost_div_u2_scores_kernel",
]
