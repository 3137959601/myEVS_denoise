from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class S22AnyPolBurstGateParams:
    """Parameters for Part2 s22 (same-pixel any-polarity burst gate).

    Intuition
    - Hot pixels / burst noise often fires repeatedly at the same pixel with tiny dt.
    - Unlike s9 (which only gates when previous same-pixel event has SAME polarity),
      s22 gates on *any* previous event at the same pixel. This targets alternating
      flicker (+ - + -) and other rapid toggles without adding any new per-pixel state.

    Gate rule (conservative)
    - Compute baseline raw_score (same-pol neighborhood evidence).
    - Update last_ts/last_pol like baseline.
    - Apply penalty only when raw_score >= raw_thr and dt_self <= dt_thr_us.

    Penalty
        ratio = clamp(dt_self / dt_thr_us, 0..1)
        score = raw_score * ratio^gamma

    Notes
    - dt_thr_us is absolute (microseconds), not normalized to tau.
      This makes the gate behavior more stable when sweeping tau.
    """

    dt_thr_us: float = 4096.0
    raw_thr: float = 0.0
    gamma: float = 1.0


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


def s22_anypol_burst_gate_params_from_env(env: dict[str, str] | None = None) -> S22AnyPolBurstGateParams:
    """Read s22 parameters from environment.

    - MYEVS_EBF_S22_DT_THR_US in [0, inf)
    - MYEVS_EBF_S22_RAW_THR   in [0, inf)
    - MYEVS_EBF_S22_GAMMA     in [0, inf)

    Defaults are conservative.
    """

    if env is None:
        env = os.environ  # type: ignore[assignment]

    dt_thr_us = float(max(0.0, _env_float(env, "MYEVS_EBF_S22_DT_THR_US", 4096.0)))
    raw_thr = float(max(0.0, _env_float(env, "MYEVS_EBF_S22_RAW_THR", 0.0)))
    gamma = float(max(0.0, _env_float(env, "MYEVS_EBF_S22_GAMMA", 1.0)))

    return S22AnyPolBurstGateParams(dt_thr_us=dt_thr_us, raw_thr=raw_thr, gamma=gamma)


def try_build_s22_anypol_burst_gate_scores_kernel():
    """Build and return Numba kernel for s22 score streaming.

    Returns None if numba is unavailable.

    Kernel signature:
        (t,x,y,p,width,height,radius_px,tau_ticks,dt_thr_ticks,raw_thr,gamma,last_ts,last_pol,scores_out) -> None

    Notes
    - dt_thr_ticks is absolute (ticks), converted from dt_thr_us via TimeBase.
    - Uses no additional per-pixel state beyond last_ts/last_pol.
    """

    try:
        from numba import njit  # type: ignore
    except Exception:
        return None

    @njit(cache=True)
    def ebf_s22_scores_stream(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        width: int,
        height: int,
        radius_px: int,
        tau_ticks: int,
        dt_thr_ticks: int,
        raw_thr: float,
        gamma: float,
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

        dthr = int(dt_thr_ticks)
        if dthr < 0:
            dthr = 0

        if raw_thr < 0.0:
            raw_thr = 0.0
        if gamma < 0.0:
            gamma = 0.0

        if rr <= 0:
            for i in range(n):
                xi = int(x[i])
                yi = int(y[i])
                if xi < 0 or xi >= w or yi < 0 or yi >= h:
                    scores_out[i] = 0.0
                    continue
                ti = int(t[i])
                pi = 1 if int(p[i]) > 0 else -1
                idx0 = yi * w + xi
                last_ts[idx0] = np.uint64(ti)
                last_pol[idx0] = np.int8(pi)
                scores_out[i] = np.inf
            return

        inv_tau = 1.0 / float(tau)
        eps = 1e-12

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            if xi < 0 or xi >= w or yi < 0 or yi >= h:
                scores_out[i] = 0.0
                continue

            ti = int(t[i])
            pi = 1 if int(p[i]) > 0 else -1

            idx0 = yi * w + xi
            prev_ts = int(last_ts[idx0])

            raw_score = 0.0

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

                    wt = float(tau - dt) * inv_tau
                    raw_score += wt

            # update state
            last_ts[idx0] = np.uint64(ti)
            last_pol[idx0] = np.int8(pi)

            if raw_score < raw_thr or dthr <= 0 or prev_ts == 0:
                scores_out[i] = raw_score
                continue

            dt0 = ti - prev_ts
            if dt0 < 0:
                dt0 = -dt0
            if dt0 >= dthr:
                scores_out[i] = raw_score
                continue

            ratio = float(dt0) / (float(dthr) + eps)
            if ratio < 0.0:
                ratio = 0.0
            if ratio > 1.0:
                ratio = 1.0

            if gamma <= 1e-12:
                pen = ratio
            else:
                pen = ratio ** gamma

            scores_out[i] = raw_score * pen

    return ebf_s22_scores_stream
