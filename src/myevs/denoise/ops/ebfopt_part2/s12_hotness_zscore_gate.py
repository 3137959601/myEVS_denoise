from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class S12HotnessZScoreGateParams:
    """Parameters for Part2 s12 (neighbor z-score hotness anomaly gate).

    This variant keeps the s10/s11 same-pixel leaky accumulator `acc` for same
    polarity, but replaces s11's ratio-to-neighbor-mean with a more robust
    anomaly measure using neighbor mean+variance:

        z = (acc_center - mean_acc_neighbors) / (std_acc_neighbors + eps)

    Only when raw is already high AND acc is high AND z is high do we penalize.

    Complexity: single-pass streaming, O(r^2), Numba required.
    """

    acc_thr: float = 3.0
    z_thr: float = 3.0
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


def s12_hotness_zscore_gate_params_from_env(env: dict[str, str] | None = None) -> S12HotnessZScoreGateParams:
    """Read s12 parameters from environment.

    - MYEVS_EBF_S12_ACC_THR in (0, inf)
    - MYEVS_EBF_S12_Z_THR   in (0, inf)
    - MYEVS_EBF_S12_RAW_THR in [0, inf)
    - MYEVS_EBF_S12_GAMMA   in [0, inf)
    """

    if env is None:
        env = os.environ  # type: ignore[assignment]

    acc_thr = float(max(1e-6, _env_float(env, "MYEVS_EBF_S12_ACC_THR", 3.0)))
    z_thr = float(max(1e-6, _env_float(env, "MYEVS_EBF_S12_Z_THR", 3.0)))
    raw_thr = float(max(0.0, _env_float(env, "MYEVS_EBF_S12_RAW_THR", 3.0)))
    gamma = float(max(0.0, _env_float(env, "MYEVS_EBF_S12_GAMMA", 1.0)))

    return S12HotnessZScoreGateParams(acc_thr=acc_thr, z_thr=z_thr, raw_thr=raw_thr, gamma=gamma)


def try_build_s12_hotness_zscore_gate_scores_kernel():
    """Build and return Numba kernel for s12 score streaming.

    Returns None if numba is unavailable.

    Kernel signature:
        (t,x,y,p,width,height,radius_px,tau_ticks,acc_thr,z_thr,raw_thr,gamma,last_ts,last_pol,self_acc,scores_out) -> None

    Notes:
    - self_acc is a per-pixel float accumulator for SAME polarity.
    - For neighbor statistics, we use an effective leaky value:
        acc_eff = max(0, self_acc - dt/tau)
      using dt = |t_i - last_ts(neighbor)|.
    """

    try:
        from numba import njit  # type: ignore
    except Exception:
        return None

    @njit(cache=True)
    def ebf_s12_scores_stream(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        width: int,
        height: int,
        radius_px: int,
        tau_ticks: int,
        acc_thr: float,
        z_thr: float,
        raw_thr: float,
        gamma: float,
        last_ts: np.ndarray,
        last_pol: np.ndarray,
        self_acc: np.ndarray,
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

        if acc_thr <= 1e-6:
            acc_thr = 1e-6
        if z_thr <= 1e-6:
            z_thr = 1e-6
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
                self_acc[idx0] = np.float32(1.0)
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
            acc0 = float(self_acc[idx0])

            raw_score = 0.0
            sum_acc = 0.0
            sum_acc2 = 0.0
            cnt_acc = 0

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
                    if int(last_pol[idx]) != pi:
                        continue

                    ts = int(last_ts[idx])
                    if ts == 0:
                        continue

                    dt = (ti - ts) if ti >= ts else (ts - ti)
                    if dt > tau:
                        continue

                    wt = float(tau - dt) * inv_tau
                    raw_score += wt

                    acc_eff = float(self_acc[idx]) - float(dt) * inv_tau
                    if acc_eff < 0.0:
                        acc_eff = 0.0
                    sum_acc += acc_eff
                    sum_acc2 += acc_eff * acc_eff
                    cnt_acc += 1

            # update same-polarity leaky accumulator (center)
            if ts0 == 0 or pol0 != pi:
                acc = 1.0
            else:
                dt0 = (ti - ts0) if ti >= ts0 else (ts0 - ti)
                acc = acc0 - float(dt0) * inv_tau
                if acc < 0.0:
                    acc = 0.0
                acc = acc + 1.0

            last_ts[idx0] = np.uint64(ti)
            last_pol[idx0] = np.int8(pi)
            self_acc[idx0] = np.float32(acc)

            if raw_score < raw_thr or acc <= acc_thr or cnt_acc <= 1:
                scores_out[i] = raw_score
                continue

            mean_acc = sum_acc / float(cnt_acc)
            var_acc = sum_acc2 / float(cnt_acc) - mean_acc * mean_acc
            if var_acc < 0.0:
                var_acc = 0.0
            std_acc = (var_acc + eps) ** 0.5

            z = (acc - mean_acc) / std_acc
            if z < z_thr:
                scores_out[i] = raw_score
                continue

            r = z_thr / (z + eps)
            if r < 0.0:
                r = 0.0
            if r > 1.0:
                r = 1.0

            if gamma <= 1e-12:
                pen = r
            else:
                pen = r ** gamma

            scores_out[i] = raw_score * pen

    return ebf_s12_scores_stream
