from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class S13CrossPolSupportGateParams:
    """Parameters for Part2 s13 (cross-polarity support gate).

    Hypothesis (cross-polarity structural evidence, conservative):
    - Hot pixels / burst noise tends to produce strong same-polarity recency
      support (high raw_score) but little opposite-polarity activity nearby.
    - Real edges / motion often brings both polarities into the local
      neighborhood within the time window.

    We compute (in the same O(r^2) neighborhood scan):
      raw_score = sum_same(w_age)
      opp_score = sum_opp(w_age)
      balance   = opp_score / (raw_score + opp_score + eps)   in [0,1]

    Gate rule:
      if raw_score >= raw_thr and balance < bal_thr:
          score = raw_score * (balance / bal_thr) ** gamma
      else:
          score = raw_score
    """

    bal_thr: float = 0.05
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


def s13_crosspol_support_gate_params_from_env(env: dict[str, str] | None = None) -> S13CrossPolSupportGateParams:
    """Read s13 parameters from environment.

    - MYEVS_EBF_S13_BAL_THR  in (0, 1]
    - MYEVS_EBF_S13_RAW_THR  in [0, inf)
    - MYEVS_EBF_S13_GAMMA    in [0, inf)
    """

    if env is None:
        env = os.environ  # type: ignore[assignment]

    bal_thr = float(max(1e-6, min(1.0, _env_float(env, "MYEVS_EBF_S13_BAL_THR", 0.05))))
    raw_thr = float(max(0.0, _env_float(env, "MYEVS_EBF_S13_RAW_THR", 3.0)))
    gamma = float(max(0.0, _env_float(env, "MYEVS_EBF_S13_GAMMA", 1.0)))

    return S13CrossPolSupportGateParams(bal_thr=bal_thr, raw_thr=raw_thr, gamma=gamma)


def try_build_s13_crosspol_support_gate_scores_kernel():
    """Build and return Numba kernel for s13 score streaming.

    Returns None if numba is unavailable.

    Kernel signature:
        (t,x,y,p,width,height,radius_px,tau_ticks,bal_thr,raw_thr,gamma,last_ts,last_pol,scores_out) -> None
    """

    try:
        from numba import njit  # type: ignore
    except Exception:
        return None

    @njit(cache=True)
    def ebf_s13_scores_stream(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        width: int,
        height: int,
        radius_px: int,
        tau_ticks: int,
        bal_thr: float,
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

        if bal_thr <= 1e-6:
            bal_thr = 1e-6
        if bal_thr > 1.0:
            bal_thr = 1.0
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

            raw_score = 0.0
            opp_score = 0.0

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

                    dt = (ti - ts) if ti >= ts else (ts - ti)
                    if dt > tau:
                        continue

                    wt = float(tau - dt) * inv_tau
                    if pol == pi:
                        raw_score += wt
                    else:
                        opp_score += wt

            # update state
            last_ts[idx0] = np.uint64(ti)
            last_pol[idx0] = np.int8(pi)

            if raw_score < raw_thr:
                scores_out[i] = raw_score
                continue

            denom = raw_score + opp_score + eps
            bal = opp_score / denom
            if bal < 0.0:
                bal = 0.0
            if bal > 1.0:
                bal = 1.0

            if bal >= bal_thr:
                scores_out[i] = raw_score
                continue

            ratio = bal / (bal_thr + eps)
            if ratio < 0.0:
                ratio = 0.0
            if ratio > 1.0:
                ratio = 1.0

            if gamma <= 1e-12:
                pen = ratio
            else:
                pen = ratio ** gamma
            scores_out[i] = raw_score * pen

    return ebf_s13_scores_stream
