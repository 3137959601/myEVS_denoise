from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class S15FlipFlickerGateParams:
    """Parameters for Part2 s15 (pixel-level polarity flip flicker gate).

    Motivation:
    - Some sensor noise manifests as very fast polarity alternation at the SAME pixel
      (e.g., + - + -) within an extremely small dt.
    - This is a more explicit, localized noise hypothesis than generic cross-polarity
      neighborhood logic.

    In one O(r^2) neighborhood scan:
      raw = sum_same(w_age)
      opp = sum_opp(w_age)

    Extra center-pixel test (O(1)):
      flicker = (prev_pol == -p) and (|t - prev_ts| <= flip_dt_ticks)

    Score rule (conservative):
      if raw < raw_thr:
          score = raw
      else:
          score = raw + alpha * opp
          if flicker:
              score *= (1 - beta)

    Notes:
    - Keep flip_dt_us small (tens of microseconds) to avoid hurting true signal.
    - Keep beta small (e.g. 0.05~0.3) to limit rank distortion.
    """

    alpha: float = 0.2
    raw_thr: float = 0.0
    flip_dt_us: float = 50.0
    beta: float = 0.2


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


def s15_flip_flicker_gate_params_from_env(env: dict[str, str] | None = None) -> S15FlipFlickerGateParams:
    """Read s15 parameters from environment.

    - MYEVS_EBF_S15_ALPHA
    - MYEVS_EBF_S15_RAW_THR
    - MYEVS_EBF_S15_FLIP_DT_US
    - MYEVS_EBF_S15_BETA
    """

    if env is None:
        env = os.environ  # type: ignore[assignment]

    alpha = float(max(0.0, _env_float(env, "MYEVS_EBF_S15_ALPHA", 0.2)))
    raw_thr = float(max(0.0, _env_float(env, "MYEVS_EBF_S15_RAW_THR", 0.0)))
    flip_dt_us = float(max(0.0, _env_float(env, "MYEVS_EBF_S15_FLIP_DT_US", 50.0)))
    beta = float(_env_float(env, "MYEVS_EBF_S15_BETA", 0.2))
    if beta < 0.0:
        beta = 0.0
    if beta > 0.95:
        beta = 0.95

    return S15FlipFlickerGateParams(alpha=alpha, raw_thr=raw_thr, flip_dt_us=flip_dt_us, beta=beta)


def try_build_s15_flip_flicker_gate_scores_kernel():
    """Build and return Numba kernel for s15 score streaming.

    Returns None if numba is unavailable.

    Kernel signature:
        (t,x,y,p,width,height,radius_px,tau_ticks,alpha,raw_thr,flip_dt_ticks,beta,last_ts,last_pol,scores_out) -> None
    """

    try:
        from numba import njit  # type: ignore
    except Exception:
        return None

    @njit(cache=True)
    def ebf_s15_scores_stream(
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
        flip_dt_ticks: int,
        beta: float,
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

        fdt = int(flip_dt_ticks)
        if fdt < 0:
            fdt = 0

        if beta < 0.0:
            beta = 0.0
        if beta > 0.95:
            beta = 0.95

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

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            if xi < 0 or xi >= w or yi < 0 or yi >= h:
                scores_out[i] = 0.0
                continue

            ti = int(t[i])
            pi = 1 if int(p[i]) > 0 else -1
            idx0 = yi * w + xi

            prev_ts = int(last_ts[idx0])
            prev_pol = int(last_pol[idx0])
            flicker = 0
            if fdt > 0 and prev_ts != 0 and prev_pol == -pi:
                dt0 = ti - prev_ts
                if dt0 < 0:
                    dt0 = -dt0
                if dt0 <= fdt:
                    flicker = 1

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

            last_ts[idx0] = np.uint64(ti)
            last_pol[idx0] = np.int8(pi)

            if raw < raw_thr:
                scores_out[i] = raw
            else:
                score = raw + alpha * opp
                if flicker != 0:
                    score *= 1.0 - beta
                scores_out[i] = score

    return ebf_s15_scores_stream
