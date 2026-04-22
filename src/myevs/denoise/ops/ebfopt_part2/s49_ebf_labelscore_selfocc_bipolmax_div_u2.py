from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class S49EbfLabelscoreSelfOccBiPolMaxDivU2Params:
    """Parameters for Part2 s49 (bipolar self-state, u_self=max(H+ , H-)).

    Summary
    - Keep baseline EBF raw labelscore unchanged.
    - Keep the s44 outer score shape unchanged:

        score = raw / (1 + u_self^2)

    - Only change the self-state definition:
      maintain two per-pixel accumulators (pos/neg) instead of a single hot_state.

    State update (per pixel)
    - For each event at the pixel, linearly decay both accumulators by dt0.
    - Add inc=max(0, tau-dt0) only to the accumulator matching the current polarity.

    Self-occupancy
    - Use the stronger channel as the hotness proxy:

        H = max(H_pos, H_neg)
        u_self = H / (H + tau_r + eps)

    Motivation
    - Persistent hot pixels tend to have long same-polarity runs.
    - Real signals more often alternate polarity; splitting channels reduces their
      effective H, making u_self more selective.

    Hyperparameters (env)
    - tau_rate_us: time constant (microseconds) used to normalize self hotness.
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


def s49_ebf_labelscore_selfocc_bipolmax_div_u2_params_from_env(
    env: dict[str, str] | None = None,
) -> S49EbfLabelscoreSelfOccBiPolMaxDivU2Params:
    """Read s49 parameters from environment.

    - MYEVS_EBF_S49_TAU_RATE_US (default 0; 0 means auto)
    """

    if env is None:
        env = os.environ  # type: ignore[assignment]

    tau_rate_us = _env_int(env, "MYEVS_EBF_S49_TAU_RATE_US", 0)
    return S49EbfLabelscoreSelfOccBiPolMaxDivU2Params(tau_rate_us=int(tau_rate_us))


def try_build_s49_ebf_labelscore_selfocc_bipolmax_div_u2_scores_kernel():
    """Build and return Numba kernel for s49 score streaming.

    Returns None if numba is unavailable.
    """

    try:
        from numba import njit  # type: ignore
    except Exception:
        return None

    @njit(cache=True)
    def ebf_s49_scores_stream(
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
        hot_pos: np.ndarray,
        hot_neg: np.ndarray,
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

            ts0 = int(last_ts[idx0])
            hp = int(hot_pos[idx0])
            hn = int(hot_neg[idx0])

            dt0 = tau if ts0 == 0 else (ti - ts0)
            if dt0 < 0:
                dt0 = -dt0

            if dt0 != 0:
                hp = hp - dt0
                if hp < 0:
                    hp = 0
                hn = hn - dt0
                if hn < 0:
                    hn = 0

            inc = tau - dt0
            if inc > 0:
                if pi > 0:
                    hp = hp + inc
                    if hp > acc_max:
                        hp = acc_max
                else:
                    hn = hn + inc
                    if hn > acc_max:
                        hn = acc_max

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
            hot_pos[idx0] = np.int32(hp)
            hot_neg[idx0] = np.int32(hn)

            raw = float(raw_w) * inv_tau

            hmax = float(hp) if hp >= hn else float(hn)
            u_self = hmax / (hmax + float(tr) + eps)
            if u_self < 0.0:
                u_self = 0.0
            if u_self > 1.0:
                u_self = 1.0

            scores_out[i] = float(raw / (1.0 + (u_self * u_self)))

    return ebf_s49_scores_stream


__all__ = [
    "S49EbfLabelscoreSelfOccBiPolMaxDivU2Params",
    "s49_ebf_labelscore_selfocc_bipolmax_div_u2_params_from_env",
    "try_build_s49_ebf_labelscore_selfocc_bipolmax_div_u2_scores_kernel",
]
