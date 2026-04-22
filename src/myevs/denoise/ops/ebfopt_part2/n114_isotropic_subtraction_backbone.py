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
        raise RuntimeError("n114 requires numba, but import failed")


def _try_build_n114_kernel():
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

        tau = int(tau_ticks)
        if tau <= 0:
            tau = 1
        inv_tau = 1.0 / float(tau)

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

            E0 = 0.0
            E1 = 0.0
            E2 = 0.0
            E3 = 0.0

            for yy in range(y0, y1 + 1):
                base = yy * width
                for xx in range(x0, x1 + 1):
                    if xx == xi and yy == yi:
                        continue

                    idx_nb = base + xx
                    if int(last_pol[idx_nb]) != pi:
                        continue

                    ts = last_ts[idx_nb]
                    if ts == 0:
                        continue
                    if ti <= ts:
                        continue

                    dt_ticks = int(ti - ts)
                    if dt_ticks <= 0 or dt_ticks > tau:
                        continue

                    w_age = 1.0 - float(dt_ticks) * inv_tau
                    if w_age <= 0.0:
                        continue

                    k = 0
                    if yy > yi:
                        k += 2
                    if xx > xi:
                        k += 1

                    if k == 0:
                        E0 += w_age
                    elif k == 1:
                        E1 += w_age
                    elif k == 2:
                        E2 += w_age
                    else:
                        E3 += w_age

            min1 = E0
            min2 = E1
            if min1 > min2:
                tmp = min1
                min1 = min2
                min2 = tmp

            if E2 < min1:
                min2 = min1
                min1 = E2
            elif E2 < min2:
                min2 = E2

            if E3 < min1:
                min2 = min1
                min1 = E3
            elif E3 < min2:
                min2 = E3

            noise_floor = 0.5 * (min1 + min2)

            score = 0.0
            n0 = E0 - lam * noise_floor
            if n0 > 0.0:
                score += n0
            n1 = E1 - lam * noise_floor
            if n1 > 0.0:
                score += n1
            n2 = E2 - lam * noise_floor
            if n2 > 0.0:
                score += n2
            n3 = E3 - lam * noise_floor
            if n3 > 0.0:
                score += n3

            out[i] = np.float32(score)

            idx0 = yi * width + xi
            last_ts[idx0] = ti
            last_pol[idx0] = np.int8(pi)

    return _kernel


def score_stream_n114(
    ev,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_us: int,
    tb: TimeBase,
    scores_out: np.ndarray | None = None,
) -> np.ndarray:
    """N114 (7.61): adaptive isotropic noise-floor subtraction."""

    n = int(ev.t.shape[0])
    if scores_out is None:
        scores_out = np.empty((n,), dtype=np.float32)
    if n <= 0:
        return scores_out

    tau_ticks = int(tb.us_to_ticks(int(tau_us)))
    lambda_noise = float(_read_float_env("MYEVS_N114_LAMBDA", 0.5))

    ker = _try_build_n114_kernel()
    ker(
        ev.t.astype(np.uint64, copy=False),
        ev.x.astype(np.int32, copy=False),
        ev.y.astype(np.int32, copy=False),
        ev.p,
        int(width),
        int(height),
        int(radius_px),
        int(tau_ticks),
        float(lambda_noise),
        scores_out,
    )
    return scores_out
