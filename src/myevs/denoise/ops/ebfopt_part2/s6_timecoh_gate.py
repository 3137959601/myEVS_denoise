from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class S6TimeCohGateParams:
    """Parameters for Part2 s6: temporal-structure consistency gate.

    Compute the weighted variance of neighbor dt in the tau window.
    Structured motion tends to have more consistent dt patterns; noise tends to
    have more scattered dt.

    time_coh = 1 / (1 + var_dt / tau^2)

    Gate like s2: penalize high-raw events with low time_coh.
    """

    timecoh_thr: float = 0.55
    raw_thr: float = 3.0
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


def s6_timecoh_gate_params_from_env(env: dict[str, str] | None = None) -> S6TimeCohGateParams:
    if env is None:
        env = os.environ  # type: ignore[assignment]

    timecoh_thr = float(max(1e-6, min(1.0, _env_float(env, "MYEVS_EBF_S6_TIMECOH_THR", 0.55))))
    raw_thr = float(max(0.0, _env_float(env, "MYEVS_EBF_S6_RAW_THR", 3.0)))
    gamma = float(max(0.0, _env_float(env, "MYEVS_EBF_S6_GAMMA", 1.0)))

    return S6TimeCohGateParams(timecoh_thr=timecoh_thr, raw_thr=raw_thr, gamma=gamma)


def try_build_s6_timecoh_gate_scores_kernel():
    """Kernel signature:
        (t,x,y,p,width,height,radius_px,tau_ticks,timecoh_thr,raw_thr,gamma,last_ts,last_pol,scores_out) -> None
    """

    try:
        from numba import njit  # type: ignore
    except Exception:
        return None

    @njit(cache=True)
    def ebf_s6_scores_stream(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        width: int,
        height: int,
        radius_px: int,
        tau_ticks: int,
        timecoh_thr: float,
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
        if timecoh_thr <= 1e-6:
            timecoh_thr = 1e-6
        if timecoh_thr > 1.0:
            timecoh_thr = 1.0
        if raw_thr < 0.0:
            raw_thr = 0.0
        if gamma < 0.0:
            gamma = 0.0

        if rr <= 0 or tau <= 0:
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
        tau_f = float(tau)

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            if xi < 0 or xi >= w or yi < 0 or yi >= h:
                scores_out[i] = 0.0
                continue

            ti = int(t[i])
            pi = 1 if int(p[i]) > 0 else -1

            idx0 = yi * w + xi

            raw_score = 0.0
            sum_w = 0.0
            m1 = 0.0
            m2 = 0.0

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
                    if dt > tau:
                        continue

                    wt = (float(tau - dt) * inv_tau)
                    raw_score += wt

                    sum_w += wt
                    dtf = float(dt)
                    m1 += wt * dtf
                    m2 += wt * (dtf * dtf)

            last_ts[idx0] = np.uint64(ti)
            last_pol[idx0] = np.int8(pi)

            if raw_score < raw_thr or sum_w <= eps:
                scores_out[i] = raw_score
                continue

            mean_dt = m1 / (sum_w + eps)
            var_dt = (m2 / (sum_w + eps)) - mean_dt * mean_dt
            if var_dt < 0.0:
                var_dt = 0.0

            timecoh = 1.0 / (1.0 + (var_dt / (tau_f * tau_f + eps)))
            if timecoh < 0.0:
                timecoh = 0.0
            if timecoh > 1.0:
                timecoh = 1.0

            if timecoh >= timecoh_thr:
                scores_out[i] = raw_score
                continue

            ratio = timecoh / (timecoh_thr + eps)
            if ratio < 0.0:
                ratio = 0.0
            if ratio > 1.0:
                ratio = 1.0

            if gamma <= 1e-12:
                pen = ratio
            else:
                pen = ratio ** gamma

            scores_out[i] = raw_score * pen

    return ebf_s6_scores_stream
