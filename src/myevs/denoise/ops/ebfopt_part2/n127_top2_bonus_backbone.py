from __future__ import annotations

import os

import numpy as np

from ....timebase import TimeBase

try:
    import numba
except Exception:  # pragma: no cover
    numba = None


def _read_float_env(name: str, default: float) -> float:
    try:
        v = float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)
    if not bool(np.isfinite(v)):
        return float(default)
    return float(v)


def _read_int_env(name: str, default: int) -> int:
    try:
        v = int(os.environ.get(name, str(default)))
    except Exception:
        return int(default)
    return int(v)


def _require_numba() -> None:
    if numba is None:
        raise RuntimeError("n127 requires numba, but import failed")


def _try_build_n127_kernel():
    _require_numba()

    @numba.njit(cache=True)
    def _kernel(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        width: int,
        height: int,
        radius_px: int,
        tau_base_ticks: int,
        tau1_ticks: int,
        tau2_ticks: int,
        alpha1: float,
        alpha2: float,
        raw_thr: float,
        out: np.ndarray,
    ) -> None:
        n = int(t.shape[0])
        npx = int(width) * int(height)

        last_ts = np.zeros((npx,), dtype=np.uint64)
        last_pol = np.zeros((npx,), dtype=np.int8)

        rr = int(radius_px)
        if rr < 0:
            rr = 0
        if rr > 8:
            rr = 8

        tau_base = int(tau_base_ticks)
        if tau_base <= 0:
            tau_base = 1
        inv_tau_base = 1.0 / float(tau_base)

        tau1 = int(tau1_ticks)
        if tau1 <= 0:
            tau1 = 1
        inv_tau1 = 1.0 / float(tau1)

        tau2 = int(tau2_ticks)
        if tau2 <= 0:
            tau2 = 1
        inv_tau2 = 1.0 / float(tau2)

        a1 = float(alpha1)
        if a1 < 0.0:
            a1 = 0.0

        a2 = float(alpha2)
        if a2 < 0.0:
            a2 = 0.0

        thr = float(raw_thr)
        if thr < 0.0:
            thr = 0.0

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            if xi < 0 or xi >= width or yi < 0 or yi >= height:
                out[i] = np.float32(0.0)
                continue

            ti = np.uint64(t[i])
            pi = 1 if int(p[i]) > 0 else -1

            x0 = xi - rr
            if x0 < 0:
                x0 = 0
            x1 = xi + rr
            if x1 >= width:
                x1 = width - 1
            y0 = yi - rr
            if y0 < 0:
                y0 = 0
            y1 = yi + rr
            if y1 >= height:
                y1 = height - 1

            raw_score = 0.0
            top1_ts = np.uint64(0)
            top2_ts = np.uint64(0)

            for yy in range(y0, y1 + 1):
                base = yy * width
                for xx in range(x0, x1 + 1):
                    if xx == xi and yy == yi:
                        continue

                    idx_nb = base + xx
                    if int(last_pol[idx_nb]) != pi:
                        continue

                    ts = last_ts[idx_nb]
                    if ts == 0 or ti <= ts:
                        continue

                    if ts > top1_ts:
                        top2_ts = top1_ts
                        top1_ts = ts
                    elif ts > top2_ts:
                        top2_ts = ts

                    dt_ticks = int(ti - ts)
                    if dt_ticks > tau_base:
                        continue

                    w_age = 1.0 - float(dt_ticks) * inv_tau_base
                    if w_age > 0.0:
                        raw_score += w_age

            score = raw_score
            if raw_score >= thr:
                bonus = 0.0

                if top1_ts > 0:
                    dt1 = int(ti - top1_ts)
                    if dt1 > 0 and dt1 <= tau1:
                        w1 = 1.0 - float(dt1) * inv_tau1
                        if w1 > 0.0:
                            bonus += a1 * w1

                if top2_ts > 0:
                    dt2 = int(ti - top2_ts)
                    if dt2 > 0 and dt2 <= tau2:
                        w2 = 1.0 - float(dt2) * inv_tau2
                        if w2 > 0.0:
                            bonus += a2 * w2

                score += bonus

            out[i] = np.float32(score)

            idx0 = yi * width + xi
            last_ts[idx0] = ti
            last_pol[idx0] = np.int8(pi)

    return _kernel


def score_stream_n127(
    ev,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_us: int,
    tb: TimeBase,
    scores_out: np.ndarray | None = None,
) -> np.ndarray:
    """N127 (7.73): baseline plus top1/top2 same-polarity soft bonus."""

    n = int(ev.t.shape[0])
    if scores_out is None:
        scores_out = np.empty((n,), dtype=np.float32)
    if n <= 0:
        return scores_out

    tau_base_ticks = int(tb.us_to_ticks(int(tau_us)))
    tau1_us = int(_read_int_env("MYEVS_N127_TAU1_US", 5000))
    tau2_us = int(_read_int_env("MYEVS_N127_TAU2_US", 10000))
    tau1_ticks = int(tb.us_to_ticks(int(tau1_us)))
    tau2_ticks = int(tb.us_to_ticks(int(tau2_us)))
    alpha1 = float(_read_float_env("MYEVS_N127_ALPHA1", 0.2))
    alpha2 = float(_read_float_env("MYEVS_N127_ALPHA2", 0.1))
    raw_thr = float(_read_float_env("MYEVS_N127_RAW_THR", 2.0))

    ker = _try_build_n127_kernel()
    ker(
        ev.t.astype(np.uint64, copy=False),
        ev.x.astype(np.int32, copy=False),
        ev.y.astype(np.int32, copy=False),
        ev.p,
        int(width),
        int(height),
        int(radius_px),
        int(tau_base_ticks),
        int(tau1_ticks),
        int(tau2_ticks),
        float(alpha1),
        float(alpha2),
        float(raw_thr),
        scores_out,
    )
    return scores_out
