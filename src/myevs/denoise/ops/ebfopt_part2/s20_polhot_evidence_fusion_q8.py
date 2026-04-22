from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class S20PolHotEvidenceFusionQ8Params:
    """Parameters for Part2 s20 (polarity-aware hotness, Q8 evidence fusion).

    Streaming, single-pass, O(r^2), Numba-only.

    Compared to s19, the only architectural change is the same-pixel hotness
    term: keep two leaky accumulators per pixel (one per polarity), and penalize
    only the accumulator of the current event's polarity.
    """

    alpha: float = 0.2
    beta: float = 0.5


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


def s20_polhot_evidence_fusion_q8_params_from_env(env: dict[str, str] | None = None) -> S20PolHotEvidenceFusionQ8Params:
    """Read s20 parameters from environment.

    - MYEVS_EBF_S20_ALPHA (default 0.2)
    - MYEVS_EBF_S20_BETA  (default 0.5)

    Notes:
    - Values are clamped to >=0.
    """

    if env is None:
        env = os.environ  # type: ignore[assignment]

    alpha = float(max(0.0, _env_float(env, "MYEVS_EBF_S20_ALPHA", 0.2)))
    beta = float(max(0.0, _env_float(env, "MYEVS_EBF_S20_BETA", 0.5)))
    return S20PolHotEvidenceFusionQ8Params(alpha=alpha, beta=beta)


def _to_q8(x: float) -> int:
    v = int(x * 256.0 + 0.5)
    if v < 0:
        v = 0
    if v > 32767:
        v = 32767
    return int(v)


def try_build_s20_polhot_evidence_fusion_q8_scores_kernel():
    """Build and return Numba kernel for s20 score streaming.

    Returns None if numba is unavailable.

    Kernel signature:
      (t,x,y,p,width,height,radius_px,tau_ticks,alpha_q8,beta_q8,last_ts,last_pol,acc_neg,acc_pos,scores_out) -> None

    Arrays:
    - last_ts: uint64 per pixel
    - last_pol: int8 per pixel (for neighborhood evidence only)
    - acc_neg/acc_pos: int32 per pixel (saturating), leaky accumulators
    """

    try:
        from numba import njit  # type: ignore
    except Exception:
        return None

    @njit(cache=True)
    def ebf_s20_scores_stream(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        width: int,
        height: int,
        radius_px: int,
        tau_ticks: int,
        alpha_q8: int,
        beta_q8: int,
        last_ts: np.ndarray,
        last_pol: np.ndarray,
        acc_neg: np.ndarray,
        acc_pos: np.ndarray,
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

        a_q8 = int(alpha_q8)
        if a_q8 < 0:
            a_q8 = 0
        b_q8 = int(beta_q8)
        if b_q8 < 0:
            b_q8 = 0

        inv_scale = 1.0 / (float(tau) * 256.0)
        acc_max = 2147483647

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
                if pi > 0:
                    acc_pos[idx0] = np.int32(tau)
                else:
                    acc_neg[idx0] = np.int32(tau)

                scores_out[i] = np.inf
            return

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            if xi < 0 or xi >= w or yi < 0 or yi >= h:
                scores_out[i] = 0.0
                continue

            ti = int(t[i])
            pi = 1 if int(p[i]) > 0 else -1
            idx0 = yi * w + xi

            # Same-pixel polarity-aware hotness update:
            # leak BOTH channels by dt0 since last event at this pixel,
            # then add tau to the current polarity channel.
            ts0 = int(last_ts[idx0])
            accn = int(acc_neg[idx0])
            accp = int(acc_pos[idx0])

            if ts0 == 0:
                dt0 = 0
            else:
                dt0 = ti - ts0
                if dt0 < 0:
                    dt0 = -dt0

            if dt0 != 0:
                accn = accn - dt0
                if accn < 0:
                    accn = 0
                accp = accp - dt0
                if accp < 0:
                    accp = 0

            if pi > 0:
                accp = accp + tau
                if accp > acc_max:
                    accp = acc_max
                acc_same = accp
            else:
                accn = accn + tau
                if accn > acc_max:
                    accn = acc_max
                acc_same = accn

            # Neighborhood evidence (integer weights)
            raw_w = 0
            opp_w = 0

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

                    w_age = tau - dt
                    if pol == pi:
                        raw_w += w_age
                    else:
                        opp_w += w_age

            # Always update self state
            last_ts[idx0] = np.uint64(ti)
            last_pol[idx0] = np.int8(pi)
            acc_neg[idx0] = np.int32(accn)
            acc_pos[idx0] = np.int32(accp)

            score_q8 = (int(raw_w) << 8) + int(a_q8) * int(opp_w) - int(b_q8) * int(acc_same)
            if score_q8 <= 0:
                scores_out[i] = 0.0
            else:
                scores_out[i] = float(score_q8) * inv_scale

    return ebf_s20_scores_stream


__all__ = [
    "S20PolHotEvidenceFusionQ8Params",
    "s20_polhot_evidence_fusion_q8_params_from_env",
    "try_build_s20_polhot_evidence_fusion_q8_scores_kernel",
    "_to_q8",
]
