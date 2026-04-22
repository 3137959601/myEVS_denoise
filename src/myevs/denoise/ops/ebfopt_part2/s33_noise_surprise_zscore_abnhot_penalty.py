from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class S33NoiseSurpriseZScoreAbnHotPenaltyParams:
    """Parameters for Part2 s33 (s28 z-score + abnormal-hotness penalty).

    Motivation (from Part2 README lessons):
    - s28 (noise surprise z-score) is a good *global* standardizer, but heavy FP
      is dominated by hotmask/near-hot.
    - "Local background" fixes (s30/s32) can easily become correlated with raw
      and hurt overall ranking.
    - A safer pattern is a *clipped, conditional penalty* that only applies to a
      small, high-risk subset (raw already high) and uses *relative abnormality*
      (center vs neighborhood baseline), not absolute activity.

    Core idea:
    - Keep s28 z-score as the backbone: z28(raw, r_global).
    - Maintain per-pixel leaky hotness accumulators (pos/neg) as in s27.
    - Compute neighborhood baseline as mean(decayed(acc_pos+acc_neg)).
    - Define abnormality (dimensionless):
        abn = max(0, (acc_tot_center/tau) - (mean_nb_tot/tau))
    - Apply only when raw >= raw_thr (fixed constant):
        score = z28 - beta * abn

    Hyperparameters:
    - tau_rate_us: EMA time constant (us) for global rate estimate (same as s28).
      0 means auto (use current tau_us).
    - beta: penalty strength (>=0).

    Constraints: streaming, single-pass, O(r^2), Numba-only.
    """

    tau_rate_us: int = 0
    beta: float = 0.5


def _env_int(env: dict[str, str], name: str, default: int) -> int:
    s = (env.get(name, "") or "").strip()
    if not s:
        return int(default)
    try:
        v = int(float(s))
    except Exception:
        return int(default)
    if v < 0:
        v = 0
    return int(v)


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


def s33_noise_surprise_zscore_abnhot_penalty_params_from_env(
    env: dict[str, str] | None = None,
) -> S33NoiseSurpriseZScoreAbnHotPenaltyParams:
    """Read s33 parameters from environment.

    - MYEVS_EBF_S33_TAU_RATE_US (default 0; 0 means auto)
    - MYEVS_EBF_S33_BETA       (default 0.5)
    """

    if env is None:
        env = os.environ  # type: ignore[assignment]

    tau_rate_us = _env_int(env, "MYEVS_EBF_S33_TAU_RATE_US", 0)
    beta = float(max(0.0, _env_float(env, "MYEVS_EBF_S33_BETA", 0.5)))

    return S33NoiseSurpriseZScoreAbnHotPenaltyParams(
        tau_rate_us=int(tau_rate_us),
        beta=float(beta),
    )


def try_build_s33_noise_surprise_zscore_abnhot_penalty_scores_kernel():
    """Build and return Numba kernel for s33 score streaming.

    Returns None if numba is unavailable.

    Kernel signature:
      (t,x,y,p,width,height,radius_px,tau_ticks,tau_rate_ticks,beta,
       last_ts,last_pol,rate_ema,acc_neg,acc_pos,scores_out) -> None

    Arrays:
    - last_ts: uint64 per pixel
    - last_pol: int8 per pixel
    - rate_ema: float64 array (1,) global rate EMA (events/tick)
    - acc_neg/acc_pos: int32 per pixel leaky hotness accumulators

    Output:
    - scores_out: float32 score per event
    """

    try:
        from numba import njit  # type: ignore
    except Exception:
        return None

    @njit(cache=True)
    def ebf_s33_scores_stream(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        width: int,
        height: int,
        radius_px: int,
        tau_ticks: int,
        tau_rate_ticks: int,
        beta: float,
        last_ts: np.ndarray,
        last_pol: np.ndarray,
        rate_ema: np.ndarray,
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

        tr = int(tau_rate_ticks)
        if tr <= 0:
            tr = tau

        # global rate EMA in events/tick
        r_ema = float(rate_ema[0])
        if not np.isfinite(r_ema) or r_ema < 0.0:
            r_ema = 0.0

        inv_tau = 1.0 / float(tau)
        n_pix = float(w * h)
        eps = 1e-6

        b = float(beta)
        if not np.isfinite(b) or b < 0.0:
            b = 0.0

        raw_thr = 3.0
        acc_max = 2147483647

        if rr <= 0:
            prev_t = 0
            for i in range(n):
                xi = int(x[i])
                yi = int(y[i])
                ti = int(t[i])

                if i > 0:
                    dtg = ti - prev_t
                    if dtg > 0:
                        inst = 1.0 / float(dtg)
                        a_rate = 1.0 - np.exp(-float(dtg) / float(tr))
                        r_ema = r_ema + a_rate * (inst - r_ema)
                prev_t = ti

                if xi < 0 or xi >= w or yi < 0 or yi >= h:
                    scores_out[i] = 0.0
                    continue

                pi = 1 if int(p[i]) > 0 else -1
                idx0 = yi * w + xi

                # Update per-pixel states
                last_ts[idx0] = np.uint64(ti)
                last_pol[idx0] = np.int8(pi)
                if pi > 0:
                    acc_pos[idx0] = np.int32(tau)
                else:
                    acc_neg[idx0] = np.int32(tau)

                scores_out[i] = np.inf

            rate_ema[0] = r_ema
            return

        prev_t = 0
        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            ti = int(t[i])

            if i > 0:
                dtg = ti - prev_t
                if dtg > 0:
                    inst = 1.0 / float(dtg)
                    a_rate = 1.0 - np.exp(-float(dtg) / float(tr))
                    r_ema = r_ema + a_rate * (inst - r_ema)
            prev_t = ti

            if xi < 0 or xi >= w or yi < 0 or yi >= h:
                scores_out[i] = 0.0
                continue

            pi = 1 if int(p[i]) > 0 else -1
            idx0 = yi * w + xi

            # ---- Hotness accumulator update (same as s27 style) ----
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
            else:
                accn = accn + tau
                if accn > acc_max:
                    accn = acc_max

            acc_neg[idx0] = np.int32(accn)
            acc_pos[idx0] = np.int32(accp)

            acc_tot_center = float(accn + accp)

            # ---- Baseline raw support (same-pol) + neighborhood baseline hotness ----
            raw_w = 0
            sum_nb = 0.0
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

            # geometric neighbor count
            m = (x1 - x0 + 1) * (y1 - y0 + 1) - 1
            if m < 1:
                m = 1

            for yy in range(y0, y1 + 1):
                base = yy * w
                for xx in range(x0, x1 + 1):
                    if xx == xi and yy == yi:
                        continue

                    idx = base + xx

                    # baseline raw support
                    if int(last_pol[idx]) == pi:
                        ts = int(last_ts[idx])
                        if ts != 0:
                            dt = ti - ts
                            if dt < 0:
                                dt = -dt
                            if dt <= tau:
                                raw_w += (tau - dt)

                    # neighborhood baseline hotness (decayed acc_tot)
                    tsn = int(last_ts[idx])
                    if tsn == 0:
                        continue
                    dtn = ti - tsn
                    if dtn < 0:
                        dtn = -dtn
                    if dtn > tau:
                        continue

                    an = int(acc_neg[idx])
                    ap = int(acc_pos[idx])
                    an = an - dtn
                    if an < 0:
                        an = 0
                    ap = ap - dtn
                    if ap < 0:
                        ap = 0
                    sum_nb += float(an + ap)
                    cnt_nb += 1

            # Always update last_ts/last_pol
            last_ts[idx0] = np.uint64(ti)
            last_pol[idx0] = np.int8(pi)

            raw = float(raw_w) * inv_tau

            # ---- s28 z-score backbone (global rate) ----
            r_pix = r_ema / n_pix
            if r_pix < 0.0:
                r_pix = 0.0

            a = r_pix * float(tau)

            if a < 1e-3:
                ew = 0.5 * a - (a * a) / 6.0
                ew2 = (a / 3.0) - (a * a) / 12.0
            else:
                ea = np.exp(-a)
                ew = 1.0 - (1.0 - ea) / a
                ew2 = (a * a - 2.0 * a + 2.0 - 2.0 * ea) / (a * a)

            mu_per = 0.5 * ew
            e2_per = 0.5 * ew2
            var_per = e2_per - mu_per * mu_per
            if var_per < 0.0:
                var_per = 0.0

            mu = float(m) * mu_per
            var = float(m) * var_per
            z = float((raw - mu) / np.sqrt(var + eps))

            # ---- Abnormal hotness penalty (relative to neighborhood baseline) ----
            score = z
            if b > 0.0 and raw >= raw_thr and cnt_nb > 0:
                mean_nb = sum_nb / float(cnt_nb)
                abn = (acc_tot_center - mean_nb) * inv_tau
                if abn > 0.0:
                    score = z - b * abn

            scores_out[i] = float(score)

        rate_ema[0] = r_ema

    return ebf_s33_scores_stream


__all__ = [
    "S33NoiseSurpriseZScoreAbnHotPenaltyParams",
    "s33_noise_surprise_zscore_abnhot_penalty_params_from_env",
    "try_build_s33_noise_surprise_zscore_abnhot_penalty_scores_kernel",
]
