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


def _require_numba() -> None:
    if numba is None:
        raise RuntimeError("n135 requires numba, but import failed")


def _try_build_n135_kernel():
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
        inner_radius: int,
        s_act: float,
        s_trust: float,
        h_thresh: float,
        eps: float,
        out: np.ndarray,
    ) -> None:
        n = int(t.shape[0])
        npx = int(width) * int(height)

        last_ts = np.zeros((npx,), dtype=np.uint64)
        last_pol = np.zeros((npx,), dtype=np.int8)
        last_score = np.zeros((npx,), dtype=np.float32)

        rr = int(radius_px)
        if rr < 0:
            rr = 0
        if rr > 8:
            rr = 8

        rc = int(inner_radius)
        if rc < 0:
            rc = 0
        if rc > rr:
            rc = rr
        has_outer_ring = rr > rc

        tau_base = int(tau_base_ticks)
        if tau_base <= 0:
            tau_base = 1
        inv_tau_base = 1.0 / float(tau_base)

        sact_v = float(s_act)
        if sact_v < 0.0:
            sact_v = 0.0

        strust_v = float(s_trust)
        if strust_v < 0.0:
            strust_v = 0.0

        h_v = float(h_thresh)
        if h_v < 0.5:
            h_v = 0.5
        if h_v >= 0.999:
            h_v = 0.999
        inv_h_tail = 1.0 / (1.0 - h_v)

        epsv = float(eps)
        if epsv <= 0.0:
            epsv = 1e-3

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

            # Stage-0 baseline score: all previous neighbors are eligible.
            s_base = 0.0
            for yy in range(y0, y1 + 1):
                base = yy * width
                for xx in range(x0, x1 + 1):
                    if xx == xi and yy == yi:
                        continue

                    idx_nb = base + xx
                    ts = last_ts[idx_nb]
                    if ts == 0 or ti <= ts:
                        continue
                    if int(last_pol[idx_nb]) != pi:
                        continue

                    dt_ticks = int(ti - ts)
                    if dt_ticks > tau_base:
                        continue

                    w_time = 1.0 - float(dt_ticks) * inv_tau_base
                    if w_time > 0.0:
                        s_base += w_time

            # Always keep first-stage memory, and also store first-stage score.
            idx0 = yi * width + xi
            last_ts[idx0] = ti
            last_pol[idx0] = np.int8(pi)
            last_score[idx0] = np.float32(s_base)

            # Stage-1 only for low-score events.
            if s_base < sact_v or (not has_outer_ring):
                out[i] = np.float32(s_base)
                continue

            # Stage-2 review: ignore low-confidence neighbors by last_score.
            e_in = 0.0
            e_out = 0.0
            for yy in range(y0, y1 + 1):
                base = yy * width
                dy = yy - yi
                ady = dy if dy >= 0 else -dy
                for xx in range(x0, x1 + 1):
                    if xx == xi and yy == yi:
                        continue

                    idx_nb = base + xx
                    ts = last_ts[idx_nb]
                    if ts == 0 or ti <= ts:
                        continue
                    if int(last_pol[idx_nb]) != pi:
                        continue

                    if float(last_score[idx_nb]) < strust_v:
                        continue

                    dt_ticks = int(ti - ts)
                    if dt_ticks > tau_base:
                        continue

                    w_time = 1.0 - float(dt_ticks) * inv_tau_base
                    if w_time <= 0.0:
                        continue

                    dx = xx - xi
                    adx = dx if dx >= 0 else -dx
                    d = adx if adx >= ady else ady
                    if d <= rc:
                        e_in += w_time
                    else:
                        e_out += w_time

            e_tot = e_in + e_out
            if e_tot <= 0.0:
                out[i] = np.float32(0.0)
                continue

            ratio = e_out / (e_tot + epsv)
            if ratio <= h_v:
                score = s_base
            else:
                w_struct = (1.0 - ratio) * inv_h_tail
                if w_struct < 0.0:
                    w_struct = 0.0
                if w_struct > 1.0:
                    w_struct = 1.0
                score = s_base * w_struct

            out[i] = np.float32(score)

    return _kernel


def score_stream_n135(
    ev,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_us: int,
    tb: TimeBase,
    scores_out: np.ndarray | None = None,
) -> np.ndarray:
    """N135 (7.81): confidence-map dual-track cascade.

    Track-A (memory): keep all first-stage events and their first-stage score.
    Track-B (review): for high-score events, second-stage structure check only
    uses trusted neighbors whose first-stage score >= S_trust.
    """

    n = int(ev.t.shape[0])
    if scores_out is None:
        scores_out = np.empty((n,), dtype=np.float32)
    if n <= 0:
        return scores_out

    tau_base_ticks = int(tb.us_to_ticks(int(tau_us)))
    inner_radius = int(_read_float_env("MYEVS_N135_INNER_RADIUS", 2.0))
    s_act = float(_read_float_env("MYEVS_N135_S_ACT", 3.0))
    s_trust = float(_read_float_env("MYEVS_N135_S_TRUST", 2.0))
    h_thresh = float(_read_float_env("MYEVS_N135_H_THRESH", 0.90))
    eps = float(_read_float_env("MYEVS_N135_EPS", 1e-3))

    ker = _try_build_n135_kernel()
    ker(
        ev.t.astype(np.uint64, copy=False),
        ev.x.astype(np.int32, copy=False),
        ev.y.astype(np.int32, copy=False),
        ev.p,
        int(width),
        int(height),
        int(radius_px),
        int(tau_base_ticks),
        int(inner_radius),
        float(s_act),
        float(s_trust),
        float(h_thresh),
        float(eps),
        scores_out,
    )
    return scores_out
