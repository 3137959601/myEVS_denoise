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


def _require_numba() -> None:
    if numba is None:
        raise RuntimeError("n90 requires numba, but import failed")


@numba.njit(cache=True)
def _update_best2(cand_t: np.uint64, best_t1: np.uint64, best_t2: np.uint64):
    if cand_t == 0:
        return best_t1, best_t2
    if best_t1 == 0 or cand_t > best_t1:
        return cand_t, best_t1
    if best_t2 == 0 or cand_t > best_t2:
        return best_t1, cand_t
    return best_t1, best_t2


def _try_build_n90_kernel():
    _require_numba()

    @numba.njit(cache=True)
    def _kernel(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        width: int,
        height: int,
        tau_ticks: int,
        window_size: int,
        out: np.ndarray,
    ) -> None:
        n = int(t.shape[0])

        npx = width * height
        last_t = np.zeros((npx,), dtype=np.uint64)
        last_p = np.zeros((npx,), dtype=np.int8)

        tau = int(tau_ticks)
        tau_f = float(max(1, tau))

        w = int(window_size)
        if w < 1:
            w = 1
        if (w % 2) == 0:
            w = w + 1
        r = w // 2

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            if xi < 0 or xi >= width or yi < 0 or yi >= height:
                out[i] = 0.0
                continue

            ti = np.uint64(t[i])
            pi = np.int8(1 if int(p[i]) > 0 else -1)

            idx_i = yi * width + xi

            if tau <= 0:
                last_t[idx_i] = ti
                last_p[idx_i] = pi
                out[i] = np.inf
                continue

            best_t1 = np.uint64(0)
            best_t2 = np.uint64(0)

            x0 = xi - r
            x1 = xi + r
            y0 = yi - r
            y1 = yi + r
            if x0 < 0:
                x0 = 0
            if y0 < 0:
                y0 = 0
            if x1 >= width:
                x1 = width - 1
            if y1 >= height:
                y1 = height - 1

            for yy in range(y0, y1 + 1):
                base = yy * width
                for xx in range(x0, x1 + 1):
                    idx = base + xx
                    tj = last_t[idx]
                    if tj == 0:
                        continue
                    if int(last_p[idx]) != int(pi):
                        continue
                    dt = int(ti - tj) if ti >= tj else int(tj - ti)
                    if dt <= 0 or dt > tau:
                        continue
                    best_t1, best_t2 = _update_best2(tj, best_t1, best_t2)

            score = 0.0
            if best_t1 > 0 and best_t2 > 0:
                dt1 = float(int(ti - best_t1) if ti >= best_t1 else int(best_t1 - ti))
                dt2 = float(int(ti - best_t2) if ti >= best_t2 else int(best_t2 - ti))
                score = np.exp(-dt1 / tau_f) + np.exp(-dt2 / tau_f)

            out[i] = np.float32(score)
            last_t[idx_i] = ti
            last_p[idx_i] = pi

    return _kernel


def score_stream_n90(
    ev,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_us: int,
    tb: TimeBase,
    scores_out: np.ndarray | None = None,
) -> np.ndarray:
    """N90: two-neighbor same-polarity exponential score.

    For each event, in a local window (default 9x9), choose the two most
    recent past neighbors that are both same-polarity as current and within
    dt <= tau. Score is:

        score = exp(-dt1/tau) + exp(-dt2/tau)

    If fewer than 2 valid neighbors exist, score is 0.
    """

    n = int(ev.t.shape[0])
    if scores_out is None:
        scores_out = np.empty((n,), dtype=np.float32)
    if n <= 0:
        return scores_out

    tau_ticks = int(tb.us_to_ticks(int(tau_us)))

    window_size = int(_read_int_env("MYEVS_N90_WINDOW", 9))

    ker = _try_build_n90_kernel()
    ker(
        ev.t.astype(np.uint64, copy=False),
        ev.x.astype(np.int32, copy=False),
        ev.y.astype(np.int32, copy=False),
        ev.p,
        int(width),
        int(height),
        int(tau_ticks),
        int(max(1, window_size)),
        scores_out,
    )
    return scores_out
