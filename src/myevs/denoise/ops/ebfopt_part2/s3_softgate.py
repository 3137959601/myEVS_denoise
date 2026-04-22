from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class S3SoftGateParams:
    """Parameters for Part2 s3: smooth/robust gating based on coherence.

    Goal: reduce the mis-harm of hard thresholds in s2 by replacing the sharp
    (raw>=raw_thr and coh<coh_thr) condition with smooth sigmoid gates.
    """

    coh_thr: float = 0.4
    raw_thr: float = 3.0
    gamma: float = 1.0
    alpha: float = 1.0
    k_raw: float = 2.0
    k_coh: float = 12.0


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


def s3_softgate_params_from_env(env: dict[str, str] | None = None) -> S3SoftGateParams:
    if env is None:
        env = os.environ  # type: ignore[assignment]

    coh_thr = float(max(1e-6, min(1.0, _env_float(env, "MYEVS_EBF_S3_COH_THR", 0.4))))
    raw_thr = float(max(0.0, _env_float(env, "MYEVS_EBF_S3_RAW_THR", 3.0)))
    gamma = float(max(0.0, _env_float(env, "MYEVS_EBF_S3_GAMMA", 1.0)))
    alpha = float(max(0.0, min(1.0, _env_float(env, "MYEVS_EBF_S3_ALPHA", 1.0))))
    k_raw = float(max(1e-6, _env_float(env, "MYEVS_EBF_S3_K_RAW", 2.0)))
    k_coh = float(max(1e-6, _env_float(env, "MYEVS_EBF_S3_K_COH", 12.0)))

    return S3SoftGateParams(
        coh_thr=coh_thr,
        raw_thr=raw_thr,
        gamma=gamma,
        alpha=alpha,
        k_raw=k_raw,
        k_coh=k_coh,
    )


def try_build_s3_softgate_scores_kernel():
    """Build and return Numba kernel for s3 score streaming.

    Returns None if numba is unavailable.

    Kernel signature:
        (t,x,y,p,width,height,radius_px,tau_ticks,coh_thr,raw_thr,gamma,alpha,k_raw,k_coh,last_ts,last_pol,scores_out) -> None
    """

    try:
        from numba import njit  # type: ignore
    except Exception:
        return None

    @njit(cache=True)
    def ebf_s3_scores_stream(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        width: int,
        height: int,
        radius_px: int,
        tau_ticks: int,
        coh_thr: float,
        raw_thr: float,
        gamma: float,
        alpha: float,
        k_raw: float,
        k_coh: float,
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
        if coh_thr <= 1e-6:
            coh_thr = 1e-6
        if coh_thr > 1.0:
            coh_thr = 1.0
        if raw_thr < 0.0:
            raw_thr = 0.0
        if gamma < 0.0:
            gamma = 0.0
        if alpha < 0.0:
            alpha = 0.0
        if alpha > 1.0:
            alpha = 1.0
        if k_raw <= 1e-12:
            k_raw = 1e-12
        if k_coh <= 1e-12:
            k_coh = 1e-12

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

            raw_score = 0.0
            sxx = 0.0
            syy = 0.0
            sxy = 0.0

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
                dy = float(yy - yi)
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

                    dx = float(xx - xi)
                    sxx += wt * (dx * dx)
                    syy += wt * (dy * dy)
                    sxy += wt * (dx * dy)

            last_ts[idx0] = np.uint64(ti)
            last_pol[idx0] = np.int8(pi)

            trace = sxx + syy
            det = sxx * syy - sxy * sxy
            disc = trace * trace - 4.0 * det
            if disc < 0.0:
                disc = 0.0
            coh = (np.sqrt(disc) / (trace + eps)) if trace > 0.0 else 0.0
            if coh < 0.0:
                coh = 0.0
            if coh > 1.0:
                coh = 1.0

            # Smooth gates
            # w_raw ~ 0 when raw << raw_thr; ~1 when raw >> raw_thr
            w_raw = 1.0 / (1.0 + np.exp(-k_raw * (raw_score - raw_thr)))
            # w_coh ~ 0 when coh >= coh_thr (safe); ~1 when coh << coh_thr
            w_coh = 1.0 / (1.0 + np.exp(-k_coh * (coh_thr - coh)))

            # magnitude term (0..1): how far below coh_thr
            ratio = coh / (coh_thr + eps)
            if ratio < 0.0:
                ratio = 0.0
            if ratio > 1.0:
                ratio = 1.0
            if gamma <= 1e-12:
                mag = 1.0 - ratio
            else:
                mag = 1.0 - (ratio ** gamma)

            pen = 1.0 - alpha * w_raw * w_coh * mag
            if pen < 0.0:
                pen = 0.0
            scores_out[i] = raw_score * pen

    return ebf_s3_scores_stream
