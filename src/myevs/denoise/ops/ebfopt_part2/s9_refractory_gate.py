from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class S9RefractoryGateParams:
    """Parameters for Part2 s9 (same-pixel refractory/burst gate).

    Intuition: hot pixels / burst noise often fires repeatedly at the *same pixel*
    with extremely small inter-event dt, and often with the same polarity.

    We only apply a penalty when:
      - baseline raw_score is already high (raw >= raw_thr)
      - previous event at the same pixel has the same polarity
      - dt_self / tau < dt_thr (dt_thr is normalized to tau)

    This is intentionally "conservative": it targets a very specific failure
    mode and aims to avoid global rescaling or geometry assumptions.
    """

    dt_thr: float = 0.004  # normalized to tau (e.g. 0.004 * 128ms ~= 512us)
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


def s9_refractory_gate_params_from_env(env: dict[str, str] | None = None) -> S9RefractoryGateParams:
    """Read s9 parameters from environment.

    - MYEVS_EBF_S9_DT_THR  in (0, 1]   (normalized to tau)
    - MYEVS_EBF_S9_RAW_THR in [0, inf)
    - MYEVS_EBF_S9_GAMMA   in [0, inf)

    Defaults are intentionally conservative.
    """

    if env is None:
        env = os.environ  # type: ignore[assignment]

    dt_thr = float(max(1e-6, min(1.0, _env_float(env, "MYEVS_EBF_S9_DT_THR", 0.004))))
    raw_thr = float(max(0.0, _env_float(env, "MYEVS_EBF_S9_RAW_THR", 3.0)))
    gamma = float(max(0.0, _env_float(env, "MYEVS_EBF_S9_GAMMA", 1.0)))

    return S9RefractoryGateParams(dt_thr=dt_thr, raw_thr=raw_thr, gamma=gamma)


def try_build_s9_refractory_gate_scores_kernel():
    """Build and return Numba kernel for s9 score streaming.

    Returns None if numba is unavailable.

    Kernel signature:
        (t,x,y,p,width,height,radius_px,tau_ticks,dt_thr,raw_thr,gamma,last_ts,last_pol,scores_out) -> None

    Notes:
    - dt_thr is normalized (0..1] relative to tau_ticks.
    - The gate uses same-pixel previous event only (idx0).
    """

    try:
        from numba import njit  # type: ignore
    except Exception:
        return None

    @njit(cache=True)
    def ebf_s9_scores_stream(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        width: int,
        height: int,
        radius_px: int,
        tau_ticks: int,
        dt_thr: float,
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
        if dt_thr <= 1e-6:
            dt_thr = 1e-6
        if dt_thr > 1.0:
            dt_thr = 1.0
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

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            if xi < 0 or xi >= w or yi < 0 or yi >= h:
                scores_out[i] = 0.0
                continue

            ti = int(t[i])
            pi = 1 if int(p[i]) > 0 else -1

            idx0 = yi * w + xi
            ts0 = int(last_ts[idx0])
            pol0 = int(last_pol[idx0])

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

                    dt = (ti - ts) if ti >= ts else (ts - ti)
                    if dt > tau:
                        continue

                    wt = float(tau - dt) * inv_tau
                    raw_score += wt

            # update state
            last_ts[idx0] = np.uint64(ti)
            last_pol[idx0] = np.int8(pi)

            if raw_score < raw_thr:
                scores_out[i] = raw_score
                continue

            # Only gate if we have a previous same-polarity event at the same pixel.
            if ts0 == 0 or pol0 != pi:
                scores_out[i] = raw_score
                continue

            dt0 = (ti - ts0) if ti >= ts0 else (ts0 - ti)
            # normalize to tau
            dt_norm = float(dt0) * inv_tau

            if dt_norm >= dt_thr:
                scores_out[i] = raw_score
                continue

            ratio = dt_norm / (dt_thr + eps)
            if ratio < 0.0:
                ratio = 0.0
            if ratio > 1.0:
                ratio = 1.0

            if gamma <= 1e-12:
                pen = ratio
            else:
                pen = ratio ** gamma

            scores_out[i] = raw_score * pen

    return ebf_s9_scores_stream
