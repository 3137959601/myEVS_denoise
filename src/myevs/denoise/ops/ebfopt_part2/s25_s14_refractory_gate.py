from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class S25S14RefractoryGateParams:
    """Parameters for Part2 s25 (s14 backbone + same-pixel refractory penalty).

    Motivation (README-aligned):
    - s14 (cross-pol boost) is strong and also improves light.
    - heavy still contains hot-pixel / burst noise that often repeats at the
      *same pixel* with extremely small dt and often the same polarity.

    Design goal:
    - Keep s14 behavior unchanged for most events.
    - Only apply a conservative penalty when BOTH:
        (1) local same-pol evidence is already high (raw >= ref_raw_thr)
        (2) previous event at the same pixel has the same polarity AND
            dt_self / tau < dt_thr  (dt_thr is normalized to tau)

    This reuses only last_ts/last_pol, remains streaming-friendly and O(r^2),
    and aims to improve heavy F1 without dropping light.

    Score rule:
      score0 = raw              (raw < s14_raw_thr)
            = raw + alpha*opp   (raw >= s14_raw_thr)

      if refractory gate triggers:
          score = score0 * ( (dt_norm / dt_thr) ** gamma )
      else:
          score = score0

    Notes:
    - dt_thr is normalized in (0, 1] relative to tau_ticks (same convention as s9).
    - penalty is in [0,1], so it only suppresses suspicious bursty events.
    """

    alpha: float = 0.25
    s14_raw_thr: float = 3.0

    dt_thr: float = 0.004  # normalized to tau (e.g. 0.004 * 128ms ~= 512us)
    ref_raw_thr: float = 3.0
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


def s25_s14_refractory_gate_params_from_env(env: dict[str, str] | None = None) -> S25S14RefractoryGateParams:
    """Read s25 parameters from environment.

    - MYEVS_EBF_S25_ALPHA
    - MYEVS_EBF_S25_RAW_THR         (s14 boost threshold)
    - MYEVS_EBF_S25_DT_THR          (normalized to tau)
    - MYEVS_EBF_S25_REF_RAW_THR     (refractory gate requires raw >= this)
    - MYEVS_EBF_S25_GAMMA
    """

    if env is None:
        env = os.environ  # type: ignore[assignment]

    alpha = float(max(0.0, _env_float(env, "MYEVS_EBF_S25_ALPHA", 0.25)))
    s14_raw_thr = float(max(0.0, _env_float(env, "MYEVS_EBF_S25_RAW_THR", 3.0)))

    dt_thr = float(max(1e-6, min(1.0, _env_float(env, "MYEVS_EBF_S25_DT_THR", 0.004))))
    ref_raw_thr = float(max(0.0, _env_float(env, "MYEVS_EBF_S25_REF_RAW_THR", 3.0)))
    gamma = float(max(0.0, _env_float(env, "MYEVS_EBF_S25_GAMMA", 1.0)))

    return S25S14RefractoryGateParams(
        alpha=alpha,
        s14_raw_thr=s14_raw_thr,
        dt_thr=dt_thr,
        ref_raw_thr=ref_raw_thr,
        gamma=gamma,
    )


def try_build_s25_s14_refractory_gate_scores_kernel():
    """Build and return Numba kernel for s25 score streaming.

    Returns None if numba is unavailable.

    Kernel signature:
        (t,x,y,p,width,height,radius_px,tau_ticks,alpha,s14_raw_thr,dt_thr,ref_raw_thr,gamma,last_ts,last_pol,scores_out) -> None
    """

    try:
        from numba import njit  # type: ignore
    except Exception:
        return None

    @njit(cache=True)
    def ebf_s25_scores_stream(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        width: int,
        height: int,
        radius_px: int,
        tau_ticks: int,
        alpha: float,
        s14_raw_thr: float,
        dt_thr: float,
        ref_raw_thr: float,
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
        if s14_raw_thr < 0.0:
            s14_raw_thr = 0.0

        if dt_thr <= 1e-6:
            dt_thr = 1e-6
        if dt_thr > 1.0:
            dt_thr = 1.0
        if ref_raw_thr < 0.0:
            ref_raw_thr = 0.0
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
            ts0 = int(last_ts[idx0])
            pol0 = int(last_pol[idx0])

            raw = 0.0
            opp = 0.0

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
                    if pol == pi:
                        raw += wt
                    else:
                        opp += wt

            # update state
            last_ts[idx0] = np.uint64(ti)
            last_pol[idx0] = np.int8(pi)

            if raw < s14_raw_thr:
                score0 = raw
            else:
                score0 = raw + alpha * opp

            # Conservative refractory gate: only when raw is already high.
            if raw < ref_raw_thr:
                scores_out[i] = score0
                continue

            # Only gate if we have a previous same-polarity event at the same pixel.
            if ts0 == 0 or pol0 != pi:
                scores_out[i] = score0
                continue

            dt0 = ti - ts0
            if dt0 < 0:
                dt0 = -dt0
            dt_norm = float(dt0) * inv_tau

            if dt_norm >= dt_thr:
                scores_out[i] = score0
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

            scores_out[i] = score0 * pen

    return ebf_s25_scores_stream
