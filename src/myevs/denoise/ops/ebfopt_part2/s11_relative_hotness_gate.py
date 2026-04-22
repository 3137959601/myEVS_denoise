from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class S11RelativeHotnessGateParams:
    """Parameters for Part2 s11 (relative-hotness anomaly gate).

    Motivation:
    - s10 uses an absolute same-pixel leaky-rate accumulator (acc) to detect hot
      pixels/burst noise, but absolute thresholds can still mis-penalize busy
      edge pixels when the whole neighborhood is active.

    s11 adds a *relative* condition:
      ratio = acc_center / (mean_acc_neighbors + eps)

    Only when raw is already high AND acc is high AND ratio is high (anomaly)
    do we apply a penalty.

    This stays streaming-friendly and O(r^2): we reuse the raw neighborhood
    scan and (cheaply) accumulate neighbor acc statistics.
    """

    acc_thr: float = 3.0
    ratio_thr: float = 2.0
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


def s11_relative_hotness_gate_params_from_env(env: dict[str, str] | None = None) -> S11RelativeHotnessGateParams:
    """Read s11 parameters from environment.

    - MYEVS_EBF_S11_ACC_THR    in (0, inf)
    - MYEVS_EBF_S11_RATIO_THR  in (0, inf)
    - MYEVS_EBF_S11_RAW_THR    in [0, inf)
    - MYEVS_EBF_S11_GAMMA      in [0, inf)

    Defaults are conservative and intended for prescreen.
    """

    if env is None:
        env = os.environ  # type: ignore[assignment]

    acc_thr = float(max(1e-6, _env_float(env, "MYEVS_EBF_S11_ACC_THR", 3.0)))
    ratio_thr = float(max(1e-6, _env_float(env, "MYEVS_EBF_S11_RATIO_THR", 2.0)))
    raw_thr = float(max(0.0, _env_float(env, "MYEVS_EBF_S11_RAW_THR", 3.0)))
    gamma = float(max(0.0, _env_float(env, "MYEVS_EBF_S11_GAMMA", 1.0)))

    return S11RelativeHotnessGateParams(acc_thr=acc_thr, ratio_thr=ratio_thr, raw_thr=raw_thr, gamma=gamma)


def try_build_s11_relative_hotness_gate_scores_kernel():
    """Build and return Numba kernel for s11 score streaming.

    Returns None if numba is unavailable.

    Kernel signature:
        (t,x,y,p,width,height,radius_px,tau_ticks,acc_thr,ratio_thr,raw_thr,gamma,last_ts,last_pol,self_acc,scores_out) -> None

    Notes:
    - self_acc is a per-pixel float accumulator for SAME polarity (as in s10).
    - We apply an approximate continuous decay when reading neighbor acc:
        acc_eff = max(0, self_acc - dt/tau)
      using dt = |t_i - last_ts(neighbor)|.
    """

    try:
        from numba import njit  # type: ignore
    except Exception:
        return None

    @njit(cache=True)
    def ebf_s11_scores_stream(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        width: int,
        height: int,
        radius_px: int,
        tau_ticks: int,
        acc_thr: float,
        ratio_thr: float,
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
        if ratio_thr <= 1e-6:
            ratio_thr = 1e-6
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

            if raw_score < raw_thr or acc <= acc_thr:
                scores_out[i] = raw_score
                continue

            mean_acc = sum_acc / float(cnt_acc) if cnt_acc > 0 else 0.0
            ratio = acc / (mean_acc + eps)

            if ratio < ratio_thr:
                scores_out[i] = raw_score
                continue

            # penalize by relative-hotness ratio
            r = ratio_thr / (ratio + eps)
            if r < 0.0:
                r = 0.0
            if r > 1.0:
                r = 1.0

            if gamma <= 1e-12:
                pen = r
            else:
                pen = r ** gamma

            scores_out[i] = raw_score * pen

    return ebf_s11_scores_stream
