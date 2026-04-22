from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class S1DirCohParams:
    """Parameters for Part2 s1 (directional coherence modulation)."""

    eta: float = 0.2


def s1_dircoh_params_from_env(env: dict[str, str] | None = None, *, default_eta: float = 0.2) -> S1DirCohParams:
    """Read s1 parameters from environment.

    - MYEVS_EBF_S1_ETA in [0,1]
    """

    if env is None:
        env = os.environ  # type: ignore[assignment]

    s = (env.get("MYEVS_EBF_S1_ETA", "") or "").strip()
    try:
        eta = float(s) if s else float(default_eta)
    except Exception:
        eta = float(default_eta)

    if not np.isfinite(eta):
        eta = float(default_eta)

    eta = float(max(0.0, min(1.0, eta)))
    return S1DirCohParams(eta=eta)


def try_build_s1_dircoh_scores_kernel():
    """Build and return Numba kernel for s1 score streaming.

    Returns None if numba is unavailable.

    Kernel signature:
        (t,x,y,p,width,height,radius_px,tau_ticks,eta,last_ts,last_pol,scores_out) -> None
    """

    try:
        from numba import njit  # type: ignore
    except Exception:
        return None

    @njit(cache=True)
    def ebf_s1_scores_stream(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        width: int,
        height: int,
        radius_px: int,
        tau_ticks: int,
        eta: float,
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
        if eta < 0.0:
            eta = 0.0
        if eta > 1.0:
            eta = 1.0

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

            scores_out[i] = raw_score * (eta + (1.0 - eta) * coh)

    return ebf_s1_scores_stream
