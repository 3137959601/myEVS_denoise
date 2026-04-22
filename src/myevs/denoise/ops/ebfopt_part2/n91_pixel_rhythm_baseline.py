from __future__ import annotations

import os

import numpy as np

from ....timebase import TimeBase

try:
    import numba
except Exception:  # pragma: no cover
    numba = None


def _read_int_env(name: str, default: int) -> int:
    try:
        v = int(float(os.environ.get(name, str(default))))
    except Exception:
        return int(default)
    return int(v)


def _read_float_env(name: str, default: float) -> float:
    try:
        v = float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)
    if not bool(np.isfinite(v)):
        return float(default)
    return float(v)


def _require_numba() -> None:
    if numba is None:
        raise RuntimeError("n91 requires numba, but import failed")


def _try_build_n91_kernel():
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
        tau_ticks: int,
        tr_ticks: int,
        g_min: float,
        out: np.ndarray,
    ) -> None:
        n = int(t.shape[0])
        npx = int(width) * int(height)

        last_ts_pos = np.zeros((npx,), dtype=np.uint64)
        prev_ts_pos = np.zeros((npx,), dtype=np.uint64)
        last_ts_neg = np.zeros((npx,), dtype=np.uint64)
        prev_ts_neg = np.zeros((npx,), dtype=np.uint64)

        tau = int(tau_ticks)
        if tau <= 0:
            for i in range(n):
                xi = int(x[i])
                yi = int(y[i])
                if xi < 0 or xi >= width or yi < 0 or yi >= height:
                    out[i] = 0.0
                    continue
                out[i] = np.inf
                idx = yi * width + xi
                ti = np.uint64(t[i])
                if int(p[i]) > 0:
                    prev_ts_pos[idx] = last_ts_pos[idx]
                    last_ts_pos[idx] = ti
                else:
                    prev_ts_neg[idx] = last_ts_neg[idx]
                    last_ts_neg[idx] = ti
            return

        inv_tau = 1.0 / float(tau)
        tr = int(tr_ticks)
        if tr <= 0:
            tr = tau
        inv_tr = 1.0 / float(tr)

        rr = int(radius_px)
        if rr < 0:
            rr = 0
        if rr > 8:
            rr = 8

        g0 = float(g_min)
        if g0 < 0.0:
            g0 = 0.0
        if g0 > 1.0:
            g0 = 1.0

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            if xi < 0 or xi >= width or yi < 0 or yi >= height:
                out[i] = 0.0
                continue

            ti = np.uint64(t[i])
            pi = 1 if int(p[i]) > 0 else -1
            idx0 = yi * width + xi

            if rr <= 0:
                out[i] = np.inf
                if pi > 0:
                    prev_ts_pos[idx0] = last_ts_pos[idx0]
                    last_ts_pos[idx0] = ti
                else:
                    prev_ts_neg[idx0] = last_ts_neg[idx0]
                    last_ts_neg[idx0] = ti
                continue

            score = 0.0

            y0 = yi - rr
            if y0 < 0:
                y0 = 0
            y1 = yi + rr
            if y1 >= height:
                y1 = height - 1

            x0 = xi - rr
            if x0 < 0:
                x0 = 0
            x1 = xi + rr
            if x1 >= width:
                x1 = width - 1

            for yy in range(y0, y1 + 1):
                base = yy * width
                for xx in range(x0, x1 + 1):
                    if xx == xi and yy == yi:
                        continue

                    idx = base + xx
                    if pi > 0:
                        ts1 = last_ts_pos[idx]
                        ts2 = prev_ts_pos[idx]
                    else:
                        ts1 = last_ts_neg[idx]
                        ts2 = prev_ts_neg[idx]

                    if ts1 == 0:
                        continue

                    dt1 = int(ti - ts1) if ti >= ts1 else int(ts1 - ti)
                    if dt1 <= 0 or dt1 > tau:
                        continue

                    w_age = float(tau - dt1) * inv_tau

                    g = 1.0
                    if ts2 > 0 and ts1 > ts2:
                        dt12 = float(int(ts1 - ts2))
                        g = dt12 * inv_tr
                        if g < g0:
                            g = g0
                        elif g > 1.0:
                            g = 1.0

                    score += w_age * g

            out[i] = np.float32(score)

            if pi > 0:
                prev_ts_pos[idx0] = last_ts_pos[idx0]
                last_ts_pos[idx0] = ti
            else:
                prev_ts_neg[idx0] = last_ts_neg[idx0]
                last_ts_neg[idx0] = ti

    return _kernel


def score_stream_n91(
    ev,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_us: int,
    tb: TimeBase,
    scores_out: np.ndarray | None = None,
) -> np.ndarray:
    """N91: baseline full-neighborhood sum with per-pixel rhythm gating.

    For each neighbor pixel, use the latest same-polarity timestamp as the
    baseline support term, and attenuate it by a per-pixel rhythm confidence
    derived from the interval between the last two same-polarity events at
    that pixel.
    """

    n = int(ev.t.shape[0])
    if scores_out is None:
        scores_out = np.empty((n,), dtype=np.float32)
    if n <= 0:
        return scores_out

    tau_ticks = int(tb.us_to_ticks(int(tau_us)))
    tr_us = int(_read_int_env("MYEVS_N91_TR_US", 1000))
    g_min = float(_read_float_env("MYEVS_N91_GMIN", 0.4))
    tr_ticks = int(tb.us_to_ticks(max(1, tr_us)))

    ker = _try_build_n91_kernel()
    ker(
        ev.t.astype(np.uint64, copy=False),
        ev.x.astype(np.int32, copy=False),
        ev.y.astype(np.int32, copy=False),
        ev.p,
        int(width),
        int(height),
        int(radius_px),
        int(tau_ticks),
        int(tr_ticks),
        float(g_min),
        scores_out,
    )
    return scores_out
