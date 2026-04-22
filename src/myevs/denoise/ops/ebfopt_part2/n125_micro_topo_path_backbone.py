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
        v = int(os.environ.get(name, str(default)))
    except Exception:
        return int(default)
    return int(v)


def _require_numba() -> None:
    if numba is None:
        raise RuntimeError("n125 requires numba, but import failed")


def _try_build_n125_kernel():
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
        path_depth: int,
        tau_step_ticks: int,
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

        depth = int(path_depth)
        if depth < 1:
            depth = 1
        if depth > 8:
            depth = 8

        tau_step = int(tau_step_ticks)
        if tau_step <= 0:
            tau_step = 1

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            if xi < 0 or xi >= width or yi < 0 or yi >= height:
                out[i] = np.float32(0.0)
                continue

            ti = np.uint64(t[i])
            pi = 1 if int(p[i]) > 0 else -1

            # Stage-1: monotonic greedy path tracing in 8-neighborhood.
            curr_x = xi
            curr_y = yi
            curr_t = ti
            path_valid = True

            for _ in range(depth):
                best_t = np.uint64(0)
                best_x = -1
                best_y = -1

                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        if dx == 0 and dy == 0:
                            continue

                        nx = curr_x + dx
                        ny = curr_y + dy
                        if nx < 0 or nx >= width or ny < 0 or ny >= height:
                            continue

                        idx_nb = ny * width + nx
                        if int(last_pol[idx_nb]) != pi:
                            continue

                        ts_nb = last_ts[idx_nb]
                        if ts_nb == 0 or ts_nb >= curr_t:
                            continue

                        dt_step = int(curr_t - ts_nb)
                        if dt_step > tau_step:
                            continue

                        if ts_nb > best_t:
                            best_t = ts_nb
                            best_x = nx
                            best_y = ny

                if best_t == 0:
                    path_valid = False
                    break

                curr_x = best_x
                curr_y = best_y
                curr_t = best_t

            idx0 = yi * width + xi
            if not path_valid:
                out[i] = np.float32(0.0)
                last_ts[idx0] = ti
                last_pol[idx0] = np.int8(pi)
                continue

            # Stage-2: baseline same-polarity recency sum.
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

            score = 0.0
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

                    dt_ticks = int(ti - ts)
                    if dt_ticks > tau_base:
                        continue

                    w_age = 1.0 - float(dt_ticks) * inv_tau_base
                    if w_age > 0.0:
                        score += w_age

            out[i] = np.float32(score)
            last_ts[idx0] = ti
            last_pol[idx0] = np.int8(pi)

    return _kernel


def score_stream_n125(
    ev,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_us: int,
    tb: TimeBase,
    scores_out: np.ndarray | None = None,
) -> np.ndarray:
    """N125 (7.71): micro-topological monotonic path gate + baseline score."""

    n = int(ev.t.shape[0])
    if scores_out is None:
        scores_out = np.empty((n,), dtype=np.float32)
    if n <= 0:
        return scores_out

    tau_base_ticks = int(tb.us_to_ticks(int(tau_us)))
    path_depth = int(_read_int_env("MYEVS_N125_PATH_DEPTH", 4))
    tau_step_us = int(_read_int_env("MYEVS_N125_TAU_STEP_US", 5000))
    tau_step_ticks = int(tb.us_to_ticks(int(tau_step_us)))

    ker = _try_build_n125_kernel()
    ker(
        ev.t.astype(np.uint64, copy=False),
        ev.x.astype(np.int32, copy=False),
        ev.y.astype(np.int32, copy=False),
        ev.p,
        int(width),
        int(height),
        int(radius_px),
        int(tau_base_ticks),
        int(path_depth),
        int(tau_step_ticks),
        scores_out,
    )
    return scores_out
