from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class S4ResidualGateParams:
    """Parameters for Part2 s4: first-moment (resultant) consistency gate.

    Idea: instead of 2nd-moment anisotropy (coh), use the resultant vector of
    neighbor offsets. For structured motion/edges, weighted offsets tend to
    align, producing a strong resultant; for isotropic noise, the resultant
    cancels out.

    alignment = ||sum(w*dx, w*dy)||^2 / (sum_w * sum(w*(dx^2+dy^2)) + eps)

    Then gate like s2: penalize high-raw events with low alignment.
    """

    align_thr: float = 0.25
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


def s4_residual_gate_params_from_env(env: dict[str, str] | None = None) -> S4ResidualGateParams:
    if env is None:
        env = os.environ  # type: ignore[assignment]

    align_thr = float(max(1e-6, min(1.0, _env_float(env, "MYEVS_EBF_S4_ALIGN_THR", 0.25))))
    raw_thr = float(max(0.0, _env_float(env, "MYEVS_EBF_S4_RAW_THR", 3.0)))
    gamma = float(max(0.0, _env_float(env, "MYEVS_EBF_S4_GAMMA", 1.0)))

    return S4ResidualGateParams(align_thr=align_thr, raw_thr=raw_thr, gamma=gamma)


def try_build_s4_residual_gate_scores_kernel():
    """Kernel signature:
        (t,x,y,p,width,height,radius_px,tau_ticks,align_thr,raw_thr,gamma,last_ts,last_pol,scores_out) -> None
    """

    try:
        from numba import njit  # type: ignore
    except Exception:
        return None

    @njit(cache=True)
    def ebf_s4_scores_stream(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        width: int,
        height: int,
        radius_px: int,
        tau_ticks: int,
        align_thr: float,
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
        if align_thr <= 1e-6:
            align_thr = 1e-6
        if align_thr > 1.0:
            align_thr = 1.0
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

            raw_score = 0.0
            sum_w = 0.0
            sx = 0.0
            sy = 0.0
            sr2 = 0.0

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
                    sum_w += wt

                    dx = float(xx - xi)
                    sx += wt * dx
                    sy += wt * dy
                    sr2 += wt * (dx * dx + dy * dy)

            last_ts[idx0] = np.uint64(ti)
            last_pol[idx0] = np.int8(pi)

            if raw_score < raw_thr or sum_w <= eps or sr2 <= eps:
                scores_out[i] = raw_score
                continue

            # alignment in [0,1]
            align = (sx * sx + sy * sy) / (sum_w * sr2 + eps)
            if align < 0.0:
                align = 0.0
            if align > 1.0:
                align = 1.0

            if align >= align_thr:
                scores_out[i] = raw_score
                continue

            ratio = align / (align_thr + eps)
            if ratio < 0.0:
                ratio = 0.0
            if ratio > 1.0:
                ratio = 1.0

            if gamma <= 1e-12:
                pen = ratio
            else:
                pen = ratio ** gamma

            scores_out[i] = raw_score * pen

    return ebf_s4_scores_stream
