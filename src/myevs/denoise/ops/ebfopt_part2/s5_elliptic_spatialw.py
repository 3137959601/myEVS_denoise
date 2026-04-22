from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class S5EllipticSpatialWParams:
    """Parameters for Part2 s5: elliptic / directional (global) spatial weighting.

    This is a low-latency variant inspired by V10's spatial distance weighting,
    but uses an oriented ellipse instead of a circle.

    Notes:
    - This is *global* directionality (theta is constant), not per-event adaptive.
    - Kept as a Part2 candidate because it is streaming-friendly and cheap.
    """

    ax: float = 1.0
    ay: float = 0.6
    theta_deg: float = 0.0


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


def s5_elliptic_spatialw_params_from_env(env: dict[str, str] | None = None) -> S5EllipticSpatialWParams:
    if env is None:
        env = os.environ  # type: ignore[assignment]

    ax = float(max(1e-3, _env_float(env, "MYEVS_EBF_S5_AX", 1.0)))
    ay = float(max(1e-3, _env_float(env, "MYEVS_EBF_S5_AY", 0.6)))
    theta_deg = float(_env_float(env, "MYEVS_EBF_S5_THETA_DEG", 0.0))

    return S5EllipticSpatialWParams(ax=ax, ay=ay, theta_deg=theta_deg)


def try_build_s5_elliptic_spatialw_scores_kernel():
    """Kernel signature:
        (t,x,y,p,width,height,radius_px,tau_ticks,ax,ay,cos_t,sin_t,last_ts,last_pol,scores_out) -> None

    Weight definition:
      d = sqrt((dx'/ax)^2 + (dy'/ay)^2)
      w_sp = max(0, 1 - d/r)
    where (dx',dy') is (dx,dy) rotated by theta.
    """

    try:
        from numba import njit  # type: ignore
    except Exception:
        return None

    @njit(cache=True)
    def ebf_s5_scores_stream(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        width: int,
        height: int,
        radius_px: int,
        tau_ticks: int,
        ax: float,
        ay: float,
        cos_t: float,
        sin_t: float,
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
        if ax <= 1e-6:
            ax = 1e-6
        if ay <= 1e-6:
            ay = 1e-6

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
        inv_r = 1.0 / float(rr)

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            if xi < 0 or xi >= w or yi < 0 or yi >= h:
                scores_out[i] = 0.0
                continue

            ti = int(t[i])
            pi = 1 if int(p[i]) > 0 else -1

            idx0 = yi * w + xi
            score = 0.0

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

                    wt = float(tau - dt) * inv_tau

                    dx = float(xx - xi)
                    # rotate
                    dxp = cos_t * dx + sin_t * dy
                    dyp = -sin_t * dx + cos_t * dy

                    d = np.sqrt((dxp / ax) * (dxp / ax) + (dyp / ay) * (dyp / ay))
                    w_sp = 1.0 - d * inv_r
                    if w_sp <= 0.0:
                        continue

                    score += wt * w_sp

            last_ts[idx0] = np.uint64(ti)
            last_pol[idx0] = np.int8(pi)
            scores_out[i] = score

    return ebf_s5_scores_stream
