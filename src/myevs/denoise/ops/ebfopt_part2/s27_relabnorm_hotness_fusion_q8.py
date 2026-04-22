from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class S27RelAbNormHotnessFusionQ8Params:
    """Parameters for Part2 s27 (relative abnormal hotness penalty, Q8).

    Motivation:
    - Heavy FP is dominated by hot/near-hot/high-rate pixels.
    - Penalizing absolute same-pixel hotness (acc_mix) can increase FP/TP together.

    Core idea:
    - Penalize only the *abnormal* part of same-pixel hotness relative to its neighborhood.
    - Use the neighborhood mean of *decayed* hotness accumulators as a local baseline.

    Definitions (tick units):
      acc_mix = acc_same + kappa * acc_opp
      mean_nb = mean_{neighbors within tau} (decay(acc_pos)+decay(acc_neg))
      acc_pen = max(0, acc_mix - lambda_nb * mean_nb)

    Score (Q8 fixed-point):
      score_q8 = (raw_w<<8) + alpha_q8*opp_w - beta_q8*acc_pen
      score = score_q8 / (tau * 256)

    Constraints: streaming, single-pass, O(r^2), Numba-only.
    """

    alpha: float = 0.2
    beta: float = 0.8
    kappa: float = 1.0
    lambda_nb: float = 1.0


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


def s27_relabnorm_hotness_fusion_q8_params_from_env(
    env: dict[str, str] | None = None,
) -> S27RelAbNormHotnessFusionQ8Params:
    """Read s27 parameters from environment.

    - MYEVS_EBF_S27_ALPHA    (default 0.2, clamped to >=0)
    - MYEVS_EBF_S27_BETA     (default 0.8, clamped to >=0)
    - MYEVS_EBF_S27_KAPPA    (default 1.0, clamped to [0,1])
    - MYEVS_EBF_S27_LAMBDA_NB (default 1.0, clamped to [0,2])

    Notes:
    - lambda_nb controls how strongly neighborhood baseline cancels hotness penalty.
      Larger lambda_nb => more pixels treated as "normal" hotness => less penalty.
    """

    if env is None:
        env = os.environ  # type: ignore[assignment]

    alpha = float(max(0.0, _env_float(env, "MYEVS_EBF_S27_ALPHA", 0.2)))
    beta = float(max(0.0, _env_float(env, "MYEVS_EBF_S27_BETA", 0.8)))

    kappa = float(_env_float(env, "MYEVS_EBF_S27_KAPPA", 1.0))
    if not np.isfinite(kappa):
        kappa = 1.0
    kappa = float(max(0.0, min(1.0, kappa)))

    lambda_nb = float(_env_float(env, "MYEVS_EBF_S27_LAMBDA_NB", 1.0))
    if not np.isfinite(lambda_nb):
        lambda_nb = 1.0
    lambda_nb = float(max(0.0, min(2.0, lambda_nb)))

    return S27RelAbNormHotnessFusionQ8Params(
        alpha=alpha,
        beta=beta,
        kappa=kappa,
        lambda_nb=lambda_nb,
    )


def _to_q8(x: float) -> int:
    v = int(x * 256.0 + 0.5)
    if v < 0:
        v = 0
    if v > 32767:
        v = 32767
    return int(v)


def try_build_s27_relabnorm_hotness_fusion_q8_scores_kernel():
    """Build and return Numba kernel for s27 score streaming.

    Returns None if numba is unavailable.

    Kernel signature:
      (t,x,y,p,width,height,radius_px,tau_ticks,alpha_q8,beta_q8,kappa_q8,lambda_q8,last_ts,last_pol,acc_neg,acc_pos,scores_out) -> None

    Arrays:
    - last_ts: uint64 per pixel
    - last_pol: int8 per pixel (for neighborhood evidence only)
    - acc_neg/acc_pos: int32 per pixel (saturating), leaky accumulators

    Important: neighbor accumulators are *locally decayed* by dt within the kernel
    (without mutating neighbor states) to approximate same-time baseline.
    """

    try:
        from numba import njit  # type: ignore
    except Exception:
        return None

    @njit(cache=True)
    def ebf_s27_scores_stream(
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
        kappa_q8: int,
        lambda_q8: int,
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
        k_q8 = int(kappa_q8)
        if k_q8 < 0:
            k_q8 = 0
        if k_q8 > 256:
            k_q8 = 256
        l_q8 = int(lambda_q8)
        if l_q8 < 0:
            l_q8 = 0
        if l_q8 > 512:
            l_q8 = 512

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
                acc_opp = accn
            else:
                accn = accn + tau
                if accn > acc_max:
                    accn = acc_max
                acc_same = accn
                acc_opp = accp

            acc_mix = int(acc_same) + ((int(k_q8) * int(acc_opp) + 128) >> 8)

            # Neighborhood evidence (integer weights) + neighborhood baseline for hotness
            raw_w = 0
            opp_w = 0
            sum_nb = 0  # int64 in numba
            cnt_nb = 0

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

                    ts = int(last_ts[idx])
                    if ts == 0:
                        continue

                    dt = ti - ts
                    if dt < 0:
                        dt = -dt
                    if dt > tau:
                        continue

                    # Evidence uses neighbor polarity
                    pol = int(last_pol[idx])
                    if pol != 0:
                        w_age = tau - dt
                        if pol == pi:
                            raw_w += w_age
                        elif pol == -pi:
                            opp_w += w_age

                    # Neighborhood baseline uses decayed total hotness (both channels)
                    an = int(acc_neg[idx]) - dt
                    if an < 0:
                        an = 0
                    ap = int(acc_pos[idx]) - dt
                    if ap < 0:
                        ap = 0
                    sum_nb += int(an + ap)
                    cnt_nb += 1

            # Always update self state
            last_ts[idx0] = np.uint64(ti)
            last_pol[idx0] = np.int8(pi)
            acc_neg[idx0] = np.int32(accn)
            acc_pos[idx0] = np.int32(accp)

            if cnt_nb > 0:
                mean_nb = int(sum_nb // cnt_nb)
            else:
                mean_nb = 0

            baseline = (int(l_q8) * int(mean_nb) + 128) >> 8
            acc_pen = int(acc_mix) - int(baseline)
            if acc_pen < 0:
                acc_pen = 0

            score_q8 = (int(raw_w) << 8) + int(a_q8) * int(opp_w) - int(b_q8) * int(acc_pen)
            if score_q8 <= 0:
                scores_out[i] = 0.0
            else:
                scores_out[i] = float(score_q8) * inv_scale

    return ebf_s27_scores_stream


__all__ = [
    "S27RelAbNormHotnessFusionQ8Params",
    "s27_relabnorm_hotness_fusion_q8_params_from_env",
    "try_build_s27_relabnorm_hotness_fusion_q8_scores_kernel",
    "_to_q8",
]
