from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class S24S14BurstinessGateParams:
    """Parameters for Part2 s24 (s14 backbone + burstiness penalty).

    Motivation:
    - s14 improves separability by adding a conservative cross-polarity boost:
        score0 = raw_same (+ alpha * raw_opp when raw_same >= raw_thr)
    - Remaining FP often comes from bursty noise / hot clusters that create an
      unusually high fraction of *very recent* neighbors.

    We define a burstiness ratio b in one neighborhood scan:
        total_w = sum_{nei}(w_age)
        burst_w = sum_{nei}(w_age * 1(dt <= burst_dt))
        b = burst_w / (total_w + eps)

    Only when raw_same is already high AND b is high, apply a conservative
    penalty to score0:
        if b > b_thr:  score = score0 * ((1-b)/(1-b_thr))^gamma
        else:          score = score0

    This remains streaming-friendly and O(r^2), and does NOT require extra
    per-pixel state arrays beyond last_ts/last_pol.
    """

    alpha: float = 0.25
    raw_thr: float = 3.0

    burst_dt_us: float = 2048.0
    b_thr: float = 0.85
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


def s24_s14_burstiness_gate_params_from_env(env: dict[str, str] | None = None) -> S24S14BurstinessGateParams:
    """Read s24 parameters from environment.

    - MYEVS_EBF_S24_ALPHA
    - MYEVS_EBF_S24_RAW_THR
    - MYEVS_EBF_S24_BURST_DT_US
    - MYEVS_EBF_S24_B_THR
    - MYEVS_EBF_S24_GAMMA
    """

    if env is None:
        env = os.environ  # type: ignore[assignment]

    alpha = float(max(0.0, _env_float(env, "MYEVS_EBF_S24_ALPHA", 0.25)))
    raw_thr = float(max(0.0, _env_float(env, "MYEVS_EBF_S24_RAW_THR", 3.0)))

    burst_dt_us = float(max(0.0, _env_float(env, "MYEVS_EBF_S24_BURST_DT_US", 2048.0)))
    b_thr = float(_env_float(env, "MYEVS_EBF_S24_B_THR", 0.85))
    # Keep denominator (1-b_thr) valid.
    if b_thr < 0.0:
        b_thr = 0.0
    if b_thr > 0.999:
        b_thr = 0.999
    gamma = float(max(0.0, _env_float(env, "MYEVS_EBF_S24_GAMMA", 1.0)))

    return S24S14BurstinessGateParams(
        alpha=alpha,
        raw_thr=raw_thr,
        burst_dt_us=burst_dt_us,
        b_thr=b_thr,
        gamma=gamma,
    )


def try_build_s24_s14_burstiness_gate_scores_kernel():
    """Build and return Numba kernel for s24 score streaming.

    Returns None if numba is unavailable.

    Kernel signature:
        (t,x,y,p,width,height,radius_px,tau_ticks,alpha,raw_thr,burst_dt_ticks,b_thr,gamma,last_ts,last_pol,scores_out) -> None
    """

    try:
        from numba import njit  # type: ignore
    except Exception:
        return None

    @njit(cache=True)
    def ebf_s24_scores_stream(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        width: int,
        height: int,
        radius_px: int,
        tau_ticks: int,
        alpha: float,
        raw_thr: float,
        burst_dt_ticks: int,
        b_thr: float,
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

        bdt = int(burst_dt_ticks)
        if bdt < 0:
            bdt = 0
        if bdt > tau:
            bdt = tau

        if alpha < 0.0:
            alpha = 0.0
        if raw_thr < 0.0:
            raw_thr = 0.0

        if b_thr < 0.0:
            b_thr = 0.0
        if b_thr > 0.999:
            b_thr = 0.999
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
        inv_1mb = 1.0 / max(1.0 - float(b_thr), 1e-6)

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            if xi < 0 or xi >= w or yi < 0 or yi >= h:
                scores_out[i] = 0.0
                continue

            ti = int(t[i])
            pi = 1 if int(p[i]) > 0 else -1
            idx0 = yi * w + xi

            raw = 0.0
            opp = 0.0
            tot_w = 0.0
            burst_w = 0.0

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
                    if pol == 0:
                        continue
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

                    wt = float(tau - dt) * inv_tau
                    tot_w += wt
                    if dt <= bdt:
                        burst_w += wt

                    if pol == pi:
                        raw += wt
                    else:
                        opp += wt

            last_ts[idx0] = np.uint64(ti)
            last_pol[idx0] = np.int8(pi)

            if raw < raw_thr:
                scores_out[i] = raw
                continue

            score0 = raw + alpha * opp

            b = 0.0
            if tot_w > 0.0:
                b = burst_w / (tot_w + eps)
                if b < 0.0:
                    b = 0.0
                if b > 1.0:
                    b = 1.0

            if b > b_thr:
                base = (1.0 - b) * inv_1mb
                if base < 0.0:
                    base = 0.0
                if base > 1.0:
                    base = 1.0
                if gamma == 1.0:
                    pen = base
                elif gamma == 0.0:
                    pen = 1.0
                else:
                    pen = base ** gamma
                score0 = score0 * pen

            scores_out[i] = score0

    return ebf_s24_scores_stream
