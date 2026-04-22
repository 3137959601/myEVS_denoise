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
        raise RuntimeError("n123 requires numba, but import failed")


def _try_build_n123_kernel():
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
        lambda_noise: float,
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
        inv_tau = 1.0 / float(tau_base)

        lam = float(lambda_noise)
        if lam < 0.0:
            lam = 0.0

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

            e = np.zeros((8,), dtype=np.float64)

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
                    if dt_ticks <= 0 or dt_ticks > tau_base:
                        continue

                    w_age = 1.0 - float(dt_ticks) * inv_tau
                    if w_age <= 0.0:
                        continue

                    dx = xx - xi
                    dy = yy - yi
                    adx = dx if dx >= 0 else -dx
                    ady = dy if dy >= 0 else -dy

                    if dx > 0 and dy >= 0:
                        k = 0 if adx > ady else 1
                    elif dx <= 0 and dy > 0:
                        k = 2 if ady > adx else 3
                    elif dx < 0 and dy <= 0:
                        k = 4 if adx > ady else 5
                    else:
                        k = 6 if ady > adx else 7

                    e[k] += w_age

            # insertion sort ascending
            for j in range(1, 8):
                key = e[j]
                m = j - 1
                while m >= 0 and e[m] > key:
                    e[m + 1] = e[m]
                    m -= 1
                e[m + 1] = key

            noise_floor = (e[0] + e[1] + e[2]) / 3.0

            s = 0.0
            net1 = e[7] - lam * noise_floor
            if net1 > 0.0:
                s += net1
            net2 = e[6] - lam * noise_floor
            if net2 > 0.0:
                s += net2

            out[i] = np.float32(s)

            idx0 = yi * width + xi
            last_ts[idx0] = ti
            last_pol[idx0] = np.int8(pi)

    return _kernel


def score_stream_n123(
    ev,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_us: int,
    tb: TimeBase,
    scores_out: np.ndarray | None = None,
) -> np.ndarray:
    """N123 (7.68): 8-sector isotropic subtraction with top-2 sectors."""

    n = int(ev.t.shape[0])
    if scores_out is None:
        scores_out = np.empty((n,), dtype=np.float32)
    if n <= 0:
        return scores_out

    tau_base_ticks = int(tb.us_to_ticks(int(tau_us)))
    lambda_noise = float(_read_float_env("MYEVS_N123_LAMBDA", 0.5))

    ker = _try_build_n123_kernel()
    ker(
        ev.t.astype(np.uint64, copy=False),
        ev.x.astype(np.int32, copy=False),
        ev.y.astype(np.int32, copy=False),
        ev.p,
        int(width),
        int(height),
        int(radius_px),
        int(tau_base_ticks),
        float(lambda_noise),
        scores_out,
    )
    return scores_out
