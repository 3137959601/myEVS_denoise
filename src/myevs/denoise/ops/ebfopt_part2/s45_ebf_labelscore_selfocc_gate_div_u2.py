from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class S45EbfLabelscoreSelfOccGateDivU2Params:
    """Parameters for Part2 s45 (s44 + gated self-occupancy penalty).

    Motivation
    - s44 keeps baseline EBF labelscore ranking (raw same-pol recency sum), and only applies a
      gentle self-occupancy penalty, which fixes the "light regression" issue.
    - For further optimization, we want an even *more conservative* behavior on normal pixels
      (low self-occupancy), while keeping (or strengthening) suppression on persistent hot pixels
      (high self-occupancy) that dominate heavy false positives.

    Core idea
    - Same baseline raw as EBF (identical to s44):

        raw = sum_{nei same-pol} (tau - dt)/tau

    - Same self-occupancy proxy u_self in [0,1] derived from the linear-decay hot_state.

    - Apply a *gated* penalty:

        u_gate = max(0, (u_self - u0) / (1 - u0))
        score  = raw / (1 + u_gate^2)

      When u0=0, this reduces exactly to s44.

    Hyperparameters (env)
    - tau_rate_us: time constant (microseconds) used to normalize self hot_state.
      0 means auto (use current tau_us).
    - u0: occupancy gate threshold in [0,1). Below u0, score==raw (no penalty).
    """

    tau_rate_us: int = 0
    u0: float = 0.0


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
    return float(v)


def s45_ebf_labelscore_selfocc_gate_div_u2_params_from_env(
    env: dict[str, str] | None = None,
) -> S45EbfLabelscoreSelfOccGateDivU2Params:
    """Read s45 parameters from environment.

    - MYEVS_EBF_S45_TAU_RATE_US (default 0; 0 means auto)
    - MYEVS_EBF_S45_U0 (default 0.0; clamp to [0, 0.999])
    """

    if env is None:
        env = os.environ  # type: ignore[assignment]

    tau_rate_us = _env_int(env, "MYEVS_EBF_S45_TAU_RATE_US", 0)
    u0 = _env_float(env, "MYEVS_EBF_S45_U0", 0.0)
    if u0 < 0.0:
        u0 = 0.0
    if u0 >= 1.0:
        u0 = 0.999
    return S45EbfLabelscoreSelfOccGateDivU2Params(tau_rate_us=int(tau_rate_us), u0=float(u0))


def try_build_s45_ebf_labelscore_selfocc_gate_div_u2_scores_kernel():
    """Build and return Numba kernel for s45 score streaming.

    Returns None if numba is unavailable.
    """

    try:
        from numba import njit  # type: ignore
    except Exception:
        return None

    @njit(cache=True)
    def ebf_s45_scores_stream(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        width: int,
        height: int,
        radius_px: int,
        tau_ticks: int,
        tau_rate_ticks: int,
        u0: float,
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

        u0f = float(u0)
        if u0f < 0.0:
            u0f = 0.0
        if u0f >= 1.0:
            u0f = 0.999

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

            # Update hot_state with linear-decay accumulator.
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

            if u_self <= u0f:
                scores_out[i] = float(raw)
            else:
                denom = (1.0 - u0f) + eps
                u_gate = (u_self - u0f) / denom
                if u_gate < 0.0:
                    u_gate = 0.0
                if u_gate > 1.0:
                    u_gate = 1.0
                d = 1.0 + (u_gate * u_gate)
                scores_out[i] = float(raw / d)

    return ebf_s45_scores_stream


__all__ = [
    "S45EbfLabelscoreSelfOccGateDivU2Params",
    "s45_ebf_labelscore_selfocc_gate_div_u2_params_from_env",
    "try_build_s45_ebf_labelscore_selfocc_gate_div_u2_scores_kernel",
]
