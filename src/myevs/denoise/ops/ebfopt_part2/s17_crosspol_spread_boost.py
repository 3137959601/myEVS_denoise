from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class S17CrossPolSpreadBoostParams:
    """Parameters for Part2 s17 (s14 boost with opposite-support spread trust).

    Motivation:
    - s14 boosts using opp = sum_opp(w_age).
    - Some FP may come from "accidental" opposite-polarity support that is highly
      concentrated (e.g., a single neighbor pixel), which may be less reliable
      than spatially spread opposite evidence.

    We compute a cheap spread proxy for opposite-polarity neighbors in one
    neighborhood scan:
      opp = sum(w)
      mx = sum(w*dx), my = sum(w*dy)
      m2 = sum(w*(dx^2+dy^2))
      var = E[r^2] - ||E[r]||^2

    Then we down-weight the opp boost when var is small:
      trust = 1                               if var >= var_thr
              beta + (1-beta)*(var/var_thr)^gamma   otherwise

    Score:
      if raw < raw_thr: score = raw
      else: score = raw + alpha * opp * trust

    This stays streaming-friendly and O(r^2).
    """

    alpha: float = 0.2
    raw_thr: float = 0.0
    var_thr: float = 2.0
    beta: float = 0.2
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


def s17_crosspol_spread_boost_params_from_env(env: dict[str, str] | None = None) -> S17CrossPolSpreadBoostParams:
    """Read s17 parameters from environment.

    - MYEVS_EBF_S17_ALPHA
    - MYEVS_EBF_S17_RAW_THR
    - MYEVS_EBF_S17_VAR_THR
    - MYEVS_EBF_S17_BETA
    - MYEVS_EBF_S17_GAMMA
    """

    if env is None:
        env = os.environ  # type: ignore[assignment]

    alpha = float(max(0.0, _env_float(env, "MYEVS_EBF_S17_ALPHA", 0.2)))
    raw_thr = float(max(0.0, _env_float(env, "MYEVS_EBF_S17_RAW_THR", 0.0)))
    var_thr = float(max(1e-6, _env_float(env, "MYEVS_EBF_S17_VAR_THR", 2.0)))

    beta = float(_env_float(env, "MYEVS_EBF_S17_BETA", 0.2))
    if beta < 0.0:
        beta = 0.0
    if beta > 1.0:
        beta = 1.0

    gamma = float(max(0.0, _env_float(env, "MYEVS_EBF_S17_GAMMA", 1.0)))

    return S17CrossPolSpreadBoostParams(alpha=alpha, raw_thr=raw_thr, var_thr=var_thr, beta=beta, gamma=gamma)


def try_build_s17_crosspol_spread_boost_scores_kernel():
    """Build and return Numba kernel for s17 score streaming.

    Returns None if numba is unavailable.

    Kernel signature:
        (t,x,y,p,width,height,radius_px,tau_ticks,alpha,raw_thr,var_thr,beta,gamma,last_ts,last_pol,scores_out) -> None
    """

    try:
        from numba import njit  # type: ignore
    except Exception:
        return None

    @njit(cache=True)
    def ebf_s17_scores_stream(
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
        var_thr: float,
        beta: float,
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

        if alpha < 0.0:
            alpha = 0.0
        if raw_thr < 0.0:
            raw_thr = 0.0
        if var_thr <= 1e-6:
            var_thr = 1e-6
        if beta < 0.0:
            beta = 0.0
        if beta > 1.0:
            beta = 1.0
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

            raw = 0.0
            opp = 0.0
            mx = 0.0
            my = 0.0
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
                dy = yy - yi
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
                    if pol == pi:
                        raw += wt
                    else:
                        opp += wt
                        dx = xx - xi
                        mx += wt * float(dx)
                        my += wt * float(dy)
                        m2 += wt * float(dx * dx + dy * dy)

            last_ts[idx0] = np.uint64(ti)
            last_pol[idx0] = np.int8(pi)

            if raw < raw_thr:
                scores_out[i] = raw
                continue

            trust = 1.0
            if opp > eps:
                ex = mx / opp
                ey = my / opp
                er2 = m2 / opp
                var = er2 - (ex * ex + ey * ey)
                if var < var_thr:
                    if var < 0.0:
                        var = 0.0
                    xnorm = var / var_thr
                    if xnorm < 0.0:
                        xnorm = 0.0
                    if xnorm > 1.0:
                        xnorm = 1.0
                    if gamma <= 1e-12:
                        smooth = xnorm
                    else:
                        smooth = xnorm ** gamma
                    trust = beta + (1.0 - beta) * smooth

            scores_out[i] = raw + alpha * opp * trust

    return ebf_s17_scores_stream
