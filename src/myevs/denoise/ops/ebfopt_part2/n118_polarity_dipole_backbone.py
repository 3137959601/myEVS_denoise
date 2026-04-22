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
        raise RuntimeError("n118 requires numba, but import failed")


def _try_build_n118_kernel():
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
        echo_min_ticks: int,
        echo_max_ticks: int,
        alpha_boost: float,
        dist_th_sq: float,
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

        e_min = int(echo_min_ticks)
        e_max = int(echo_max_ticks)
        if e_min < 0:
            e_min = 0
        if e_max < e_min:
            e_max = e_min
        if e_max <= 0:
            e_max = 1

        alpha = float(alpha_boost)
        if alpha < 0.0:
            alpha = 0.0

        d2_th = float(dist_th_sq)
        if d2_th < 0.0:
            d2_th = 0.0

        inv_tau_base = 1.0 / float(tau_base)
        inv_echo_max = 1.0 / float(e_max)

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

            w_same = 0.0
            x_same = 0.0
            y_same = 0.0

            w_opp = 0.0
            x_opp = 0.0
            y_opp = 0.0

            for yy in range(y0, y1 + 1):
                base = yy * width
                yyf = float(yy)
                for xx in range(x0, x1 + 1):
                    if xx == xi and yy == yi:
                        continue

                    idx_nb = base + xx
                    ts = last_ts[idx_nb]
                    if ts == 0:
                        continue
                    if ti <= ts:
                        continue

                    dt_ticks = int(ti - ts)
                    if dt_ticks <= 0:
                        continue

                    pj = int(last_pol[idx_nb])
                    xxf = float(xx)

                    if pj == pi:
                        if dt_ticks > tau_base:
                            continue
                        wt = 1.0 - float(dt_ticks) * inv_tau_base
                        if wt <= 0.0:
                            continue
                        w_same += wt
                        x_same += wt * xxf
                        y_same += wt * yyf
                    else:
                        if dt_ticks < e_min or dt_ticks > e_max:
                            continue
                        wt = 1.0 - float(dt_ticks) * inv_echo_max
                        if wt <= 0.0:
                            continue
                        w_opp += wt
                        x_opp += wt * xxf
                        y_opp += wt * yyf

            score = w_same

            if w_same > 1e-6 and w_opp > 1e-6 and alpha > 0.0:
                cx_same = x_same / w_same
                cy_same = y_same / w_same
                cx_opp = x_opp / w_opp
                cy_opp = y_opp / w_opp

                dx = cx_same - cx_opp
                dy = cy_same - cy_opp
                d2 = dx * dx + dy * dy
                if d2 >= d2_th:
                    score *= (1.0 + alpha)

            out[i] = np.float32(score)

            idx0 = yi * width + xi
            last_ts[idx0] = ti
            last_pol[idx0] = np.int8(pi)

    return _kernel


def score_stream_n118(
    ev,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_us: int,
    tb: TimeBase,
    scores_out: np.ndarray | None = None,
) -> np.ndarray:
    """N118 (7.64): polarity spatial dipole via CoM offset."""

    n = int(ev.t.shape[0])
    if scores_out is None:
        scores_out = np.empty((n,), dtype=np.float32)
    if n <= 0:
        return scores_out

    tau_base_ticks = int(tb.us_to_ticks(int(tau_us)))
    echo_min_us = int(_read_int_env("MYEVS_N118_ECHO_MIN_US", 1000))
    echo_max_us = int(_read_int_env("MYEVS_N118_ECHO_MAX_US", 60000))
    echo_min_ticks = int(tb.us_to_ticks(int(echo_min_us)))
    echo_max_ticks = int(tb.us_to_ticks(int(echo_max_us)))

    alpha_boost = float(_read_float_env("MYEVS_N118_ALPHA", 0.5))
    dist_th = float(_read_float_env("MYEVS_N118_DIST_TH", 1.5))
    if dist_th < 0.0:
        dist_th = 0.0
    dist_th_sq = float(dist_th * dist_th)

    ker = _try_build_n118_kernel()
    ker(
        ev.t.astype(np.uint64, copy=False),
        ev.x.astype(np.int32, copy=False),
        ev.y.astype(np.int32, copy=False),
        ev.p,
        int(width),
        int(height),
        int(radius_px),
        int(tau_base_ticks),
        int(echo_min_ticks),
        int(echo_max_ticks),
        float(alpha_boost),
        float(dist_th_sq),
        scores_out,
    )
    return scores_out
