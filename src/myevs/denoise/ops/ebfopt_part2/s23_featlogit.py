from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class S23FeatLogitParams:
    """Parameters for Part2 s23 (feature + linear logit fusion).

    Motivation
    - Prior Part2 variants mostly apply a *single* heuristic gate/penalty to baseline raw.
      Those tend to saturate at ~1e-3 level gains on ED24.
    - s23 reframes scoring as: extract a small set of mechanistic features per event
      (still online, single-pass, O(r^2)), then fuse them via a learnable linear model.

    Score
        logit = bias
                + w_same * raw_same
                + w_opp  * raw_opp
                + w_oppr * (raw_opp / (raw_same + eps))
                + w_toggle * toggle
                + w_dtsmall * (1 - clamp(dt0 / dt_thr, 0..1))
                + w_sameburst * (dtsmall * (1 - toggle))
                + w_selfacc * selfacc
                + w_hot * ishot
                + w_hotnbr * hotnbr

    Where
    - raw_same/raw_opp: tau-weighted neighborhood evidence split by neighbor polarity.
    - toggle: 1 if previous same-pixel event exists and polarity toggled, else 0.
    - dt0: dt to previous same-pixel event (any polarity), absolute ticks.

    Notes
    - dt_thr_us is absolute microseconds (converted to ticks); not normalized to tau.
    - Output is a real-valued score (logit). Thresholding is handled by the existing sweep.
        - By default, no extra per-pixel arrays beyond last_ts/last_pol (baseline state).
            Optionally, when w_selfacc != 0, enables a tiny per-pixel state array:
                - self_acc_q8: uint16 (Q8 fixed-point), 2 bytes per pixel.
            This captures same-pixel same-polarity leaky "hotness/rate" similar in spirit to s19,
            and is especially useful for heavy noise.
    """

    dt_thr_us: float = 4096.0

    bias: float = 0.0
    w_same: float = 1.0
    w_opp: float = 0.0
    w_oppr: float = 0.0
    w_toggle: float = 0.0
    w_dtsmall: float = 0.0
    w_sameburst: float = 0.0
    w_selfacc: float = 0.0
    w_hot: float = 0.0
    w_hotnbr: float = 0.0


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


def _env_str(env: dict[str, str], name: str, default: str = "") -> str:
    s = (env.get(name, "") or "").strip()
    if not s:
        return str(default)
    return str(s)


def s23_featlogit_params_from_env(env: dict[str, str] | None = None) -> S23FeatLogitParams:
    """Read s23 parameters from environment.

    Hyper-params
    - MYEVS_EBF_S23_DT_THR_US

    Linear weights (logit)
    - MYEVS_EBF_S23_BIAS
    - MYEVS_EBF_S23_W_SAME
    - MYEVS_EBF_S23_W_OPP
    - MYEVS_EBF_S23_W_OPPR
    - MYEVS_EBF_S23_W_TOGGLE
    - MYEVS_EBF_S23_W_DTSMALL
    - MYEVS_EBF_S23_W_SAMEBURST
    - MYEVS_EBF_S23_W_SELFACC
    - MYEVS_EBF_S23_W_HOT
    - MYEVS_EBF_S23_W_HOTNBR
    """

    if env is None:
        env = os.environ  # type: ignore[assignment]

    dt_thr_us = float(max(0.0, _env_float(env, "MYEVS_EBF_S23_DT_THR_US", 4096.0)))

    bias = _env_float(env, "MYEVS_EBF_S23_BIAS", 0.0)
    w_same = _env_float(env, "MYEVS_EBF_S23_W_SAME", 1.0)
    w_opp = _env_float(env, "MYEVS_EBF_S23_W_OPP", 0.0)
    w_oppr = _env_float(env, "MYEVS_EBF_S23_W_OPPR", 0.0)
    w_toggle = _env_float(env, "MYEVS_EBF_S23_W_TOGGLE", 0.0)
    w_dtsmall = _env_float(env, "MYEVS_EBF_S23_W_DTSMALL", 0.0)
    w_sameburst = _env_float(env, "MYEVS_EBF_S23_W_SAMEBURST", 0.0)
    w_selfacc = _env_float(env, "MYEVS_EBF_S23_W_SELFACC", 0.0)
    w_hot = _env_float(env, "MYEVS_EBF_S23_W_HOT", 0.0)
    w_hotnbr = _env_float(env, "MYEVS_EBF_S23_W_HOTNBR", 0.0)

    return S23FeatLogitParams(
        dt_thr_us=dt_thr_us,
        bias=bias,
        w_same=w_same,
        w_opp=w_opp,
        w_oppr=w_oppr,
        w_toggle=w_toggle,
        w_dtsmall=w_dtsmall,
        w_sameburst=w_sameburst,
        w_selfacc=w_selfacc,
        w_hot=w_hot,
        w_hotnbr=w_hotnbr,
    )


def try_build_s23_featlogit_scores_kernel(*, with_selfacc: bool = False, with_hotmask: bool = False):
    """Build and return Numba kernel for s23 score streaming.

    Returns None if numba is unavailable.

    Kernel signature:
        (t,x,y,p,width,height,radius_px,tau_ticks,dt_thr_ticks,
         bias,w_same,w_opp,w_oppr,w_toggle,w_dtsmall,w_sameburst,
            last_ts,last_pol,scores_out) -> None

        If with_selfacc=True:
           (...,
             bias,w_same,w_opp,w_oppr,w_toggle,w_dtsmall,w_sameburst,w_selfacc,
            last_ts,last_pol,self_acc_q8,scores_out) -> None

                If with_hotmask=True:
                     (...,
                         bias,w_same,w_opp,w_oppr,w_toggle,w_dtsmall,w_sameburst,w_hot,w_hotnbr,
                        last_ts,last_pol,hotmask_u8,scores_out) -> None

                If with_selfacc=True and with_hotmask=True:
                     (...,
                         bias,w_same,w_opp,w_oppr,w_toggle,w_dtsmall,w_sameburst,w_selfacc,w_hot,w_hotnbr,
                        last_ts,last_pol,self_acc_q8,hotmask_u8,scores_out) -> None
    """

    try:
        from numba import njit  # type: ignore
    except Exception:
        return None

    @njit(cache=True)
    def ebf_s23_scores_stream(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        width: int,
        height: int,
        radius_px: int,
        tau_ticks: int,
        dt_thr_ticks: int,
        bias: float,
        w_same: float,
        w_opp: float,
        w_oppr: float,
        w_toggle: float,
        w_dtsmall: float,
        w_sameburst: float,
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

        dthr = int(dt_thr_ticks)
        if dthr < 0:
            dthr = 0

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
                scores_out[i] = bias
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
            prev_ts = int(last_ts[idx0])
            prev_pol = int(last_pol[idx0])

            raw_same = 0.0
            raw_opp = 0.0

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

                    wt = float(tau - dt) * inv_tau
                    if int(last_pol[idx]) == pi:
                        raw_same += wt
                    else:
                        raw_opp += wt

            # update state
            last_ts[idx0] = np.uint64(ti)
            last_pol[idx0] = np.int8(pi)

            toggle = 0.0
            if prev_ts != 0 and prev_pol != 0 and prev_pol != pi:
                toggle = 1.0

            dt_ratio = 1.0
            if dthr > 0 and prev_ts != 0:
                dt0 = ti - prev_ts
                if dt0 < 0:
                    dt0 = -dt0
                r = float(dt0) / (float(dthr) + eps)
                if r < 0.0:
                    r = 0.0
                if r > 1.0:
                    r = 1.0
                dt_ratio = r

            dtsmall = 1.0 - dt_ratio
            oppr = raw_opp / (raw_same + eps)
            sameburst = dtsmall * (1.0 - toggle)

            scores_out[i] = (
                bias
                + w_same * raw_same
                + w_opp * raw_opp
                + w_oppr * oppr
                + w_toggle * toggle
                + w_dtsmall * dtsmall
                + w_sameburst * sameburst
            )

    @njit(cache=True)
    def ebf_s23_scores_stream_hotmask(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        width: int,
        height: int,
        radius_px: int,
        tau_ticks: int,
        dt_thr_ticks: int,
        bias: float,
        w_same: float,
        w_opp: float,
        w_oppr: float,
        w_toggle: float,
        w_dtsmall: float,
        w_sameburst: float,
        w_hot: float,
        w_hotnbr: float,
        last_ts: np.ndarray,
        last_pol: np.ndarray,
        hotmask_u8: np.ndarray,
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

        dthr = int(dt_thr_ticks)
        if dthr < 0:
            dthr = 0

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
                ishot = 1.0 if int(hotmask_u8[idx0]) != 0 else 0.0
                scores_out[i] = bias + w_hot * ishot
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
            prev_ts = int(last_ts[idx0])
            prev_pol = int(last_pol[idx0])

            raw_same = 0.0
            raw_opp = 0.0
            hot_cnt = 0
            nbr_cnt = 0

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
                    nbr_cnt += 1
                    if int(hotmask_u8[idx]) != 0:
                        hot_cnt += 1
                    ts = int(last_ts[idx])
                    if ts == 0:
                        continue

                    dt = ti - ts
                    if dt < 0:
                        dt = -dt
                    if dt > tau:
                        continue

                    wt = float(tau - dt) * inv_tau
                    if int(last_pol[idx]) == pi:
                        raw_same += wt
                    else:
                        raw_opp += wt

            # update state
            last_ts[idx0] = np.uint64(ti)
            last_pol[idx0] = np.int8(pi)

            toggle = 0.0
            if prev_ts != 0 and prev_pol != 0 and prev_pol != pi:
                toggle = 1.0

            dt_ratio = 1.0
            if dthr > 0 and prev_ts != 0:
                dt0 = ti - prev_ts
                if dt0 < 0:
                    dt0 = -dt0
                r = float(dt0) / (float(dthr) + eps)
                if r < 0.0:
                    r = 0.0
                if r > 1.0:
                    r = 1.0
                dt_ratio = r

            dtsmall = 1.0 - dt_ratio
            oppr = raw_opp / (raw_same + eps)
            sameburst = dtsmall * (1.0 - toggle)
            ishot = 1.0 if int(hotmask_u8[idx0]) != 0 else 0.0
            hotnbr = 0.0
            if nbr_cnt > 0:
                hotnbr = float(hot_cnt) / float(nbr_cnt)

            scores_out[i] = (
                bias
                + w_same * raw_same
                + w_opp * raw_opp
                + w_oppr * oppr
                + w_toggle * toggle
                + w_dtsmall * dtsmall
                + w_sameburst * sameburst
                + w_hot * ishot
                + w_hotnbr * hotnbr
            )

    @njit(cache=True)
    def ebf_s23_scores_stream_selfacc(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        width: int,
        height: int,
        radius_px: int,
        tau_ticks: int,
        dt_thr_ticks: int,
        bias: float,
        w_same: float,
        w_opp: float,
        w_oppr: float,
        w_toggle: float,
        w_dtsmall: float,
        w_sameburst: float,
        w_selfacc: float,
        last_ts: np.ndarray,
        last_pol: np.ndarray,
        self_acc_q8: np.ndarray,
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

        dthr = int(dt_thr_ticks)
        if dthr < 0:
            dthr = 0

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
                # init self-acc to 0.0 in Q8
                self_acc_q8[idx0] = np.uint16(0)
                scores_out[i] = bias
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
            prev_ts = int(last_ts[idx0])
            prev_pol = int(last_pol[idx0])

            raw_same = 0.0
            raw_opp = 0.0

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

                    wt = float(tau - dt) * inv_tau
                    if int(last_pol[idx]) == pi:
                        raw_same += wt
                    else:
                        raw_opp += wt

            dt0 = 0
            if prev_ts != 0:
                dt0 = ti - prev_ts
                if dt0 < 0:
                    dt0 = -dt0

            # --- per-pixel leaky burst accumulator (Q8, uint16) ---
            # Target "burst" noise: accumulate when dt0 <= dt_thr, regardless of polarity.
            acc1 = 0
            if prev_ts != 0:
                dec = (int(dt0) << 8) // tau
                if dec < 0:
                    dec = 0
                acc0 = int(self_acc_q8[idx0])
                acc1 = acc0 - int(dec)
                if acc1 < 0:
                    acc1 = 0
                if dthr > 0 and dt0 <= dthr:
                    acc1 = acc1 + 256
                if acc1 > 65535:
                    acc1 = 65535
            else:
                acc1 = 0
            self_acc_q8[idx0] = np.uint16(acc1)
            selfacc = float(acc1) * (1.0 / 256.0)

            # update baseline state
            last_ts[idx0] = np.uint64(ti)
            last_pol[idx0] = np.int8(pi)

            toggle = 0.0
            if prev_ts != 0 and prev_pol != 0 and prev_pol != pi:
                toggle = 1.0

            dt_ratio = 1.0
            if dthr > 0 and prev_ts != 0:
                r = float(dt0) / (float(dthr) + eps)
                if r < 0.0:
                    r = 0.0
                if r > 1.0:
                    r = 1.0
                dt_ratio = r

            dtsmall = 1.0 - dt_ratio
            oppr = raw_opp / (raw_same + eps)
            sameburst = dtsmall * (1.0 - toggle)

            scores_out[i] = (
                bias
                + w_same * raw_same
                + w_opp * raw_opp
                + w_oppr * oppr
                + w_toggle * toggle
                + w_dtsmall * dtsmall
                + w_sameburst * sameburst
                + w_selfacc * selfacc
            )

    @njit(cache=True)
    def ebf_s23_scores_stream_selfacc_hotmask(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        width: int,
        height: int,
        radius_px: int,
        tau_ticks: int,
        dt_thr_ticks: int,
        bias: float,
        w_same: float,
        w_opp: float,
        w_oppr: float,
        w_toggle: float,
        w_dtsmall: float,
        w_sameburst: float,
        w_selfacc: float,
        w_hot: float,
        w_hotnbr: float,
        last_ts: np.ndarray,
        last_pol: np.ndarray,
        self_acc_q8: np.ndarray,
        hotmask_u8: np.ndarray,
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

        dthr = int(dt_thr_ticks)
        if dthr < 0:
            dthr = 0

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
                self_acc_q8[idx0] = np.uint16(0)
                ishot = 1.0 if int(hotmask_u8[idx0]) != 0 else 0.0
                scores_out[i] = bias + w_hot * ishot
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
            prev_ts = int(last_ts[idx0])
            prev_pol = int(last_pol[idx0])

            raw_same = 0.0
            raw_opp = 0.0
            hot_cnt = 0
            nbr_cnt = 0

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
                    nbr_cnt += 1
                    if int(hotmask_u8[idx]) != 0:
                        hot_cnt += 1
                    ts = int(last_ts[idx])
                    if ts == 0:
                        continue

                    dt = ti - ts
                    if dt < 0:
                        dt = -dt
                    if dt > tau:
                        continue

                    wt = float(tau - dt) * inv_tau
                    if int(last_pol[idx]) == pi:
                        raw_same += wt
                    else:
                        raw_opp += wt

            dt0 = 0
            if prev_ts != 0:
                dt0 = ti - prev_ts
                if dt0 < 0:
                    dt0 = -dt0

            # self-acc update (Q8): accumulate when dt0 <= dt_thr, regardless of polarity
            acc1 = 0
            if prev_ts != 0:
                dec = (int(dt0) << 8) // tau
                if dec < 0:
                    dec = 0
                acc0 = int(self_acc_q8[idx0])
                acc1 = acc0 - int(dec)
                if acc1 < 0:
                    acc1 = 0
                if dthr > 0 and dt0 <= dthr:
                    acc1 = acc1 + 256
                if acc1 > 65535:
                    acc1 = 65535
            else:
                acc1 = 0
            self_acc_q8[idx0] = np.uint16(acc1)
            selfacc = float(acc1) * (1.0 / 256.0)

            # update baseline state
            last_ts[idx0] = np.uint64(ti)
            last_pol[idx0] = np.int8(pi)

            toggle = 0.0
            if prev_ts != 0 and prev_pol != 0 and prev_pol != pi:
                toggle = 1.0

            dt_ratio = 1.0
            if dthr > 0 and prev_ts != 0:
                r = float(dt0) / (float(dthr) + eps)
                if r < 0.0:
                    r = 0.0
                if r > 1.0:
                    r = 1.0
                dt_ratio = r

            dtsmall = 1.0 - dt_ratio
            oppr = raw_opp / (raw_same + eps)
            sameburst = dtsmall * (1.0 - toggle)
            ishot = 1.0 if int(hotmask_u8[idx0]) != 0 else 0.0
            hotnbr = 0.0
            if nbr_cnt > 0:
                hotnbr = float(hot_cnt) / float(nbr_cnt)

            scores_out[i] = (
                bias
                + w_same * raw_same
                + w_opp * raw_opp
                + w_oppr * oppr
                + w_toggle * toggle
                + w_dtsmall * dtsmall
                + w_sameburst * sameburst
                + w_selfacc * selfacc
                + w_hot * ishot
                + w_hotnbr * hotnbr
            )

    if bool(with_selfacc) and bool(with_hotmask):
        return ebf_s23_scores_stream_selfacc_hotmask
    if bool(with_selfacc):
        return ebf_s23_scores_stream_selfacc
    if bool(with_hotmask):
        return ebf_s23_scores_stream_hotmask
    return ebf_s23_scores_stream


def try_build_s23_featlogit_features_kernel(*, with_selfacc: bool = False, with_hotmask: bool = False):
    """Build and return Numba kernel that streams and outputs s23 features.

    This is intended for *offline training* (still streaming, uses last_ts/last_pol).
    Returns None if numba is unavailable.

    Kernel signature:
        (t,x,y,p,width,height,radius_px,tau_ticks,dt_thr_ticks,last_ts,last_pol,
         raw_same_out,raw_opp_out,oppr_out,toggle_out,dtsmall_out,sameburst_out) -> None

    If with_selfacc=True:
        (..., last_ts,last_pol,self_acc_q8,
         raw_same_out,raw_opp_out,oppr_out,toggle_out,dtsmall_out,sameburst_out,selfacc_out) -> None

    If with_hotmask=True:
        (..., last_ts,last_pol,hotmask_u8,
         raw_same_out,raw_opp_out,oppr_out,toggle_out,dtsmall_out,sameburst_out,hotnbr_out) -> None

    If with_selfacc=True and with_hotmask=True:
        (..., last_ts,last_pol,self_acc_q8,hotmask_u8,
         raw_same_out,raw_opp_out,oppr_out,toggle_out,dtsmall_out,sameburst_out,selfacc_out,hotnbr_out) -> None
    """

    try:
        from numba import njit  # type: ignore
    except Exception:
        return None

    @njit(cache=True)
    def ebf_s23_features_stream(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        width: int,
        height: int,
        radius_px: int,
        tau_ticks: int,
        dt_thr_ticks: int,
        last_ts: np.ndarray,
        last_pol: np.ndarray,
        raw_same_out: np.ndarray,
        raw_opp_out: np.ndarray,
        oppr_out: np.ndarray,
        toggle_out: np.ndarray,
        dtsmall_out: np.ndarray,
        sameburst_out: np.ndarray,
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

        dthr = int(dt_thr_ticks)
        if dthr < 0:
            dthr = 0

        inv_tau = 1.0 / float(tau)
        eps = 1e-12

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            if xi < 0 or xi >= w or yi < 0 or yi >= h:
                raw_same_out[i] = 0.0
                raw_opp_out[i] = 0.0
                oppr_out[i] = 0.0
                toggle_out[i] = 0.0
                dtsmall_out[i] = 0.0
                continue

            ti = int(t[i])
            pi = 1 if int(p[i]) > 0 else -1

            idx0 = yi * w + xi
            prev_ts = int(last_ts[idx0])
            prev_pol = int(last_pol[idx0])

            raw_same = 0.0
            raw_opp = 0.0

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

                    wt = float(tau - dt) * inv_tau
                    if int(last_pol[idx]) == pi:
                        raw_same += wt
                    else:
                        raw_opp += wt

            # update state
            last_ts[idx0] = np.uint64(ti)
            last_pol[idx0] = np.int8(pi)

            toggle = 0.0
            if prev_ts != 0 and prev_pol != 0 and prev_pol != pi:
                toggle = 1.0

            dt_ratio = 1.0
            if dthr > 0 and prev_ts != 0:
                dt0 = ti - prev_ts
                if dt0 < 0:
                    dt0 = -dt0
                r = float(dt0) / (float(dthr) + eps)
                if r < 0.0:
                    r = 0.0
                if r > 1.0:
                    r = 1.0
                dt_ratio = r
            dtsmall = 1.0 - dt_ratio
            sameburst = dtsmall * (1.0 - toggle)

            raw_same_out[i] = raw_same
            raw_opp_out[i] = raw_opp
            oppr_out[i] = raw_opp / (raw_same + eps)
            toggle_out[i] = toggle
            dtsmall_out[i] = dtsmall
            sameburst_out[i] = sameburst

    @njit(cache=True)
    def ebf_s23_features_stream_hotmask(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        width: int,
        height: int,
        radius_px: int,
        tau_ticks: int,
        dt_thr_ticks: int,
        last_ts: np.ndarray,
        last_pol: np.ndarray,
        hotmask_u8: np.ndarray,
        raw_same_out: np.ndarray,
        raw_opp_out: np.ndarray,
        oppr_out: np.ndarray,
        toggle_out: np.ndarray,
        dtsmall_out: np.ndarray,
        sameburst_out: np.ndarray,
        hotnbr_out: np.ndarray,
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

        dthr = int(dt_thr_ticks)
        if dthr < 0:
            dthr = 0

        inv_tau = 1.0 / float(tau)
        eps = 1e-12

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            if xi < 0 or xi >= w or yi < 0 or yi >= h:
                raw_same_out[i] = 0.0
                raw_opp_out[i] = 0.0
                oppr_out[i] = 0.0
                toggle_out[i] = 0.0
                dtsmall_out[i] = 0.0
                sameburst_out[i] = 0.0
                hotnbr_out[i] = 0.0
                continue

            ti = int(t[i])
            pi = 1 if int(p[i]) > 0 else -1

            idx0 = yi * w + xi
            prev_ts = int(last_ts[idx0])
            prev_pol = int(last_pol[idx0])

            raw_same = 0.0
            raw_opp = 0.0
            hot_cnt = 0
            nbr_cnt = 0

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
                    nbr_cnt += 1
                    if int(hotmask_u8[idx]) != 0:
                        hot_cnt += 1

                    ts = int(last_ts[idx])
                    if ts == 0:
                        continue

                    dt = ti - ts
                    if dt < 0:
                        dt = -dt
                    if dt > tau:
                        continue

                    wt = float(tau - dt) * inv_tau
                    if int(last_pol[idx]) == pi:
                        raw_same += wt
                    else:
                        raw_opp += wt

            # update state
            last_ts[idx0] = np.uint64(ti)
            last_pol[idx0] = np.int8(pi)

            toggle = 0.0
            if prev_ts != 0 and prev_pol != 0 and prev_pol != pi:
                toggle = 1.0

            dt_ratio = 1.0
            if dthr > 0 and prev_ts != 0:
                dt0 = ti - prev_ts
                if dt0 < 0:
                    dt0 = -dt0
                r = float(dt0) / (float(dthr) + eps)
                if r < 0.0:
                    r = 0.0
                if r > 1.0:
                    r = 1.0
                dt_ratio = r
            dtsmall = 1.0 - dt_ratio
            sameburst = dtsmall * (1.0 - toggle)

            hotnbr = 0.0
            if nbr_cnt > 0:
                hotnbr = float(hot_cnt) / float(nbr_cnt)

            raw_same_out[i] = raw_same
            raw_opp_out[i] = raw_opp
            oppr_out[i] = raw_opp / (raw_same + eps)
            toggle_out[i] = toggle
            dtsmall_out[i] = dtsmall
            sameburst_out[i] = sameburst
            hotnbr_out[i] = hotnbr

    @njit(cache=True)
    def ebf_s23_features_stream_selfacc(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        width: int,
        height: int,
        radius_px: int,
        tau_ticks: int,
        dt_thr_ticks: int,
        last_ts: np.ndarray,
        last_pol: np.ndarray,
        self_acc_q8: np.ndarray,
        raw_same_out: np.ndarray,
        raw_opp_out: np.ndarray,
        oppr_out: np.ndarray,
        toggle_out: np.ndarray,
        dtsmall_out: np.ndarray,
        sameburst_out: np.ndarray,
        selfacc_out: np.ndarray,
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

        dthr = int(dt_thr_ticks)
        if dthr < 0:
            dthr = 0

        inv_tau = 1.0 / float(tau)
        eps = 1e-12

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            if xi < 0 or xi >= w or yi < 0 or yi >= h:
                raw_same_out[i] = 0.0
                raw_opp_out[i] = 0.0
                oppr_out[i] = 0.0
                toggle_out[i] = 0.0
                dtsmall_out[i] = 0.0
                selfacc_out[i] = 0.0
                continue

            ti = int(t[i])
            pi = 1 if int(p[i]) > 0 else -1

            idx0 = yi * w + xi
            prev_ts = int(last_ts[idx0])
            prev_pol = int(last_pol[idx0])

            raw_same = 0.0
            raw_opp = 0.0

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

                    wt = float(tau - dt) * inv_tau
                    if int(last_pol[idx]) == pi:
                        raw_same += wt
                    else:
                        raw_opp += wt

            dt0 = 0
            if prev_ts != 0:
                dt0 = ti - prev_ts
                if dt0 < 0:
                    dt0 = -dt0

            # self-acc update (Q8): accumulate when dt0 <= dt_thr, regardless of polarity
            acc1 = 0
            if prev_ts != 0:
                dec = (int(dt0) << 8) // tau
                if dec < 0:
                    dec = 0
                acc0 = int(self_acc_q8[idx0])
                acc1 = acc0 - int(dec)
                if acc1 < 0:
                    acc1 = 0
                if dthr > 0 and dt0 <= dthr:
                    acc1 = acc1 + 256
                if acc1 > 65535:
                    acc1 = 65535
            else:
                acc1 = 0
            self_acc_q8[idx0] = np.uint16(acc1)
            selfacc = float(acc1) * (1.0 / 256.0)

            # update state
            last_ts[idx0] = np.uint64(ti)
            last_pol[idx0] = np.int8(pi)

            toggle = 0.0
            if prev_ts != 0 and prev_pol != 0 and prev_pol != pi:
                toggle = 1.0

            dt_ratio = 1.0
            if dthr > 0 and prev_ts != 0:
                r = float(dt0) / (float(dthr) + eps)
                if r < 0.0:
                    r = 0.0
                if r > 1.0:
                    r = 1.0
                dt_ratio = r
            dtsmall = 1.0 - dt_ratio
            sameburst = dtsmall * (1.0 - toggle)

            raw_same_out[i] = raw_same
            raw_opp_out[i] = raw_opp
            oppr_out[i] = raw_opp / (raw_same + eps)
            toggle_out[i] = toggle
            dtsmall_out[i] = dtsmall
            sameburst_out[i] = sameburst
            selfacc_out[i] = selfacc

    @njit(cache=True)
    def ebf_s23_features_stream_selfacc_hotmask(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        width: int,
        height: int,
        radius_px: int,
        tau_ticks: int,
        dt_thr_ticks: int,
        last_ts: np.ndarray,
        last_pol: np.ndarray,
        self_acc_q8: np.ndarray,
        hotmask_u8: np.ndarray,
        raw_same_out: np.ndarray,
        raw_opp_out: np.ndarray,
        oppr_out: np.ndarray,
        toggle_out: np.ndarray,
        dtsmall_out: np.ndarray,
        sameburst_out: np.ndarray,
        selfacc_out: np.ndarray,
        hotnbr_out: np.ndarray,
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

        dthr = int(dt_thr_ticks)
        if dthr < 0:
            dthr = 0

        inv_tau = 1.0 / float(tau)
        eps = 1e-12

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            if xi < 0 or xi >= w or yi < 0 or yi >= h:
                raw_same_out[i] = 0.0
                raw_opp_out[i] = 0.0
                oppr_out[i] = 0.0
                toggle_out[i] = 0.0
                dtsmall_out[i] = 0.0
                sameburst_out[i] = 0.0
                selfacc_out[i] = 0.0
                hotnbr_out[i] = 0.0
                continue

            ti = int(t[i])
            pi = 1 if int(p[i]) > 0 else -1

            idx0 = yi * w + xi
            prev_ts = int(last_ts[idx0])
            prev_pol = int(last_pol[idx0])

            raw_same = 0.0
            raw_opp = 0.0
            hot_cnt = 0
            nbr_cnt = 0

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
                    nbr_cnt += 1
                    if int(hotmask_u8[idx]) != 0:
                        hot_cnt += 1

                    ts = int(last_ts[idx])
                    if ts == 0:
                        continue

                    dt = ti - ts
                    if dt < 0:
                        dt = -dt
                    if dt > tau:
                        continue

                    wt = float(tau - dt) * inv_tau
                    if int(last_pol[idx]) == pi:
                        raw_same += wt
                    else:
                        raw_opp += wt

            dt0 = 0
            if prev_ts != 0:
                dt0 = ti - prev_ts
                if dt0 < 0:
                    dt0 = -dt0

            # self-acc update (Q8): accumulate when dt0 <= dt_thr, regardless of polarity
            acc1 = 0
            if prev_ts != 0:
                dec = (int(dt0) << 8) // tau
                if dec < 0:
                    dec = 0
                acc0 = int(self_acc_q8[idx0])
                acc1 = acc0 - int(dec)
                if acc1 < 0:
                    acc1 = 0
                if dthr > 0 and dt0 <= dthr:
                    acc1 = acc1 + 256
                if acc1 > 65535:
                    acc1 = 65535
            else:
                acc1 = 0
            self_acc_q8[idx0] = np.uint16(acc1)
            selfacc = float(acc1) * (1.0 / 256.0)

            # update state
            last_ts[idx0] = np.uint64(ti)
            last_pol[idx0] = np.int8(pi)

            toggle = 0.0
            if prev_ts != 0 and prev_pol != 0 and prev_pol != pi:
                toggle = 1.0

            dt_ratio = 1.0
            if dthr > 0 and prev_ts != 0:
                r = float(dt0) / (float(dthr) + eps)
                if r < 0.0:
                    r = 0.0
                if r > 1.0:
                    r = 1.0
                dt_ratio = r
            dtsmall = 1.0 - dt_ratio
            sameburst = dtsmall * (1.0 - toggle)

            hotnbr = 0.0
            if nbr_cnt > 0:
                hotnbr = float(hot_cnt) / float(nbr_cnt)

            raw_same_out[i] = raw_same
            raw_opp_out[i] = raw_opp
            oppr_out[i] = raw_opp / (raw_same + eps)
            toggle_out[i] = toggle
            dtsmall_out[i] = dtsmall
            sameburst_out[i] = sameburst
            selfacc_out[i] = selfacc
            hotnbr_out[i] = hotnbr

    if bool(with_selfacc) and bool(with_hotmask):
        return ebf_s23_features_stream_selfacc_hotmask
    if bool(with_selfacc):
        return ebf_s23_features_stream_selfacc
    if bool(with_hotmask):
        return ebf_s23_features_stream_hotmask
    return ebf_s23_features_stream
