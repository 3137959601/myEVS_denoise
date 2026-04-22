from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class S8PlaneR2GateParams:
    """Parameters for Part2 s8 (plane explained-variance / R^2 gating on dt surface).

    Hypothesis:
    - For true structured motion/edges, neighborhood dt forms an approximate plane over (dx,dy).
    - For noise bursts/hot pixels, dt over (dx,dy) is less planar -> lower explained variance.

    We compute baseline EBF raw_score, then (optionally) penalize it when:
    - raw_score is high (raw_thr) AND
    - plane fit quality (R^2) is low (r2_thr) AND
    - enough neighbor points exist (min_pts).

    All computations are streaming-friendly and O(r^2).
    """

    r2_thr: float = 0.60  # gate threshold on R^2 in (0,1]
    raw_thr: float = 3.0  # only gate when raw_score >= raw_thr
    gamma: float = 1.0  # penalty exponent >=0
    min_pts: int = 6  # minimum neighbor count for plane fit


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


def _env_int(env: dict[str, str], name: str, default: int) -> int:
    s = (env.get(name, "") or "").strip()
    if not s:
        return int(default)
    try:
        v = int(float(s))
    except Exception:
        return int(default)
    return int(v)


def s8_plane_r2_gate_params_from_env(env: dict[str, str] | None = None) -> S8PlaneR2GateParams:
    """Read s8 parameters from environment.

    - MYEVS_EBF_S8_R2_THR   in (0,1]
    - MYEVS_EBF_S8_RAW_THR  in [0,inf)
    - MYEVS_EBF_S8_GAMMA    in [0,inf)
    - MYEVS_EBF_S8_MIN_PTS  in [3, inf)
    """

    if env is None:
        env = os.environ  # type: ignore[assignment]

    r2_thr = float(max(1e-6, min(1.0, _env_float(env, "MYEVS_EBF_S8_R2_THR", 0.60))))
    raw_thr = float(max(0.0, _env_float(env, "MYEVS_EBF_S8_RAW_THR", 3.0)))
    gamma = float(max(0.0, _env_float(env, "MYEVS_EBF_S8_GAMMA", 1.0)))
    min_pts = int(max(3, _env_int(env, "MYEVS_EBF_S8_MIN_PTS", 6)))

    return S8PlaneR2GateParams(r2_thr=r2_thr, raw_thr=raw_thr, gamma=gamma, min_pts=min_pts)


def try_build_s8_plane_r2_gate_scores_kernel():
    """Build and return Numba kernel for s8 score streaming.

    Returns None if numba is unavailable.

    Kernel signature:
        (t,x,y,p,width,height,radius_px,tau_ticks,r2_thr,raw_thr,gamma,min_pts,last_ts,last_pol,scores_out) -> None
    """

    try:
        from numba import njit  # type: ignore
    except Exception:
        return None

    @njit(cache=True)
    def _solve_3x3(
        a00: float,
        a01: float,
        a02: float,
        a10: float,
        a11: float,
        a12: float,
        a20: float,
        a21: float,
        a22: float,
        b0: float,
        b1: float,
        b2: float,
        eps: float,
    ) -> tuple[bool, float, float, float]:
        # Gaussian elimination on 3x3 system.
        m00 = a00
        m01 = a01
        m02 = a02
        m10 = a10
        m11 = a11
        m12 = a12
        m20 = a20
        m21 = a21
        m22 = a22
        y0 = b0
        y1 = b1
        y2 = b2

        # pivot 0
        piv = m00
        if abs(piv) < eps:
            if abs(m10) > abs(piv):
                m00, m01, m02, y0, m10, m11, m12, y1 = m10, m11, m12, y1, m00, m01, m02, y0
                piv = m00
            elif abs(m20) > abs(piv):
                m00, m01, m02, y0, m20, m21, m22, y2 = m20, m21, m22, y2, m00, m01, m02, y0
                piv = m00
        if abs(piv) < eps:
            return False, 0.0, 0.0, 0.0

        inv = 1.0 / piv
        m00 *= inv
        m01 *= inv
        m02 *= inv
        y0 *= inv

        f10 = m10
        m10 -= f10 * m00
        m11 -= f10 * m01
        m12 -= f10 * m02
        y1 -= f10 * y0

        f20 = m20
        m20 -= f20 * m00
        m21 -= f20 * m01
        m22 -= f20 * m02
        y2 -= f20 * y0

        # pivot 1
        piv = m11
        if abs(piv) < eps:
            if abs(m21) > abs(piv):
                m10, m11, m12, y1, m20, m21, m22, y2 = m20, m21, m22, y2, m10, m11, m12, y1
                piv = m11
        if abs(piv) < eps:
            return False, 0.0, 0.0, 0.0

        inv = 1.0 / piv
        m10 *= inv
        m11 *= inv
        m12 *= inv
        y1 *= inv

        f01 = m01
        m00 -= f01 * m10
        m01 -= f01 * m11
        m02 -= f01 * m12
        y0 -= f01 * y1

        f21 = m21
        m20 -= f21 * m10
        m21 -= f21 * m11
        m22 -= f21 * m12
        y2 -= f21 * y1

        # pivot 2
        piv = m22
        if abs(piv) < eps:
            return False, 0.0, 0.0, 0.0

        inv = 1.0 / piv
        m20 *= inv
        m21 *= inv
        m22 *= inv
        y2 *= inv

        f02 = m02
        m00 -= f02 * m20
        m01 -= f02 * m21
        m02 -= f02 * m22
        y0 -= f02 * y2

        f12 = m12
        m10 -= f12 * m20
        m11 -= f12 * m21
        m12 -= f12 * m22
        y1 -= f12 * y2

        return True, y0, y1, y2

    @njit(cache=True)
    def ebf_s8_scores_stream(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        width: int,
        height: int,
        radius_px: int,
        tau_ticks: int,
        r2_thr: float,
        raw_thr: float,
        gamma: float,
        min_pts: int,
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
        if r2_thr <= 1e-6:
            r2_thr = 1e-6
        if r2_thr > 1.0:
            r2_thr = 1.0
        if raw_thr < 0.0:
            raw_thr = 0.0
        if gamma < 0.0:
            gamma = 0.0
        if min_pts < 3:
            min_pts = 3

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

            # Weighted plane fit sums for z=dt as function of (dx,dy)
            sw = 0.0
            sx = 0.0
            sy = 0.0
            sxx = 0.0
            syy = 0.0
            sxy = 0.0
            sz = 0.0
            sxz = 0.0
            syz = 0.0
            sz2 = 0.0
            cnt = 0

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

                    z = float(dt)
                    dx = float(xx - xi)

                    sw += wt
                    sx += wt * dx
                    sy += wt * dy
                    sxx += wt * (dx * dx)
                    syy += wt * (dy * dy)
                    sxy += wt * (dx * dy)
                    sz += wt * z
                    sxz += wt * (dx * z)
                    syz += wt * (dy * z)
                    sz2 += wt * (z * z)
                    cnt += 1

            last_ts[idx0] = np.uint64(ti)
            last_pol[idx0] = np.int8(pi)

            if raw_score < raw_thr or cnt < min_pts or sw <= eps:
                scores_out[i] = raw_score
                continue

            ok, a, b, c = _solve_3x3(
                sxx,
                sxy,
                sx,
                sxy,
                syy,
                sy,
                sx,
                sy,
                sw,
                sxz,
                syz,
                sz,
                1e-9,
            )
            if not ok:
                scores_out[i] = raw_score
                continue

            # SSE = sum w*(z - a*dx - b*dy - c)^2 via expanded sums
            sse = sz2
            sse -= 2.0 * a * sxz
            sse -= 2.0 * b * syz
            sse -= 2.0 * c * sz
            sse += (a * a) * sxx
            sse += (b * b) * syy
            sse += (c * c) * sw
            sse += 2.0 * a * b * sxy
            sse += 2.0 * a * c * sx
            sse += 2.0 * b * c * sy
            if sse < 0.0:
                sse = 0.0

            # SST = sum w*(z - zbar)^2 = sz2 - (sz^2)/sw
            sst = sz2 - (sz * sz) / (sw + eps)
            if sst <= eps:
                # Too little variation in z; skip gating to avoid numeric instability.
                scores_out[i] = raw_score
                continue

            r2 = 1.0 - (sse / (sst + eps))
            if r2 < 0.0:
                r2 = 0.0
            if r2 > 1.0:
                r2 = 1.0

            if r2 >= r2_thr:
                scores_out[i] = raw_score
                continue

            ratio = r2 / (r2_thr + eps)
            if ratio < 0.0:
                ratio = 0.0
            if ratio > 1.0:
                ratio = 1.0

            if gamma <= 1e-12:
                pen = ratio
            else:
                pen = ratio ** gamma
            scores_out[i] = raw_score * pen

    return ebf_s8_scores_stream
