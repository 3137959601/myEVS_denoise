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
        raise RuntimeError("n4 requires numba, but import failed")


def _try_build_n4_kernel():
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
        support_dt_ticks: int,
        support_gain: float,
        momentum_gain: float,
    ) -> np.ndarray:
        n = int(t.shape[0])
        out = np.zeros((n,), dtype=np.float32)

        npx = width * height
        last_pos = np.zeros((npx,), dtype=np.uint64)
        last_neg = np.zeros((npx,), dtype=np.uint64)

        r = int(radius_px)
        r2 = int(r * r)

        eps = 1e-6
        inv_support_dt = 1.0 / float(max(1, support_dt_ticks))

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            if xi < 0 or xi >= width or yi < 0 or yi >= height:
                continue

            ti = np.uint64(t[i])
            arr = last_pos if int(p[i]) > 0 else last_neg

            support = 0.0
            sum_w = 0.0
            sum_ux = 0.0
            sum_uy = 0.0

            for dy in range(-r, r + 1):
                yy = yi + dy
                if yy < 0 or yy >= height:
                    continue
                for dx in range(-r, r + 1):
                    d2 = dx * dx + dy * dy
                    if d2 > r2:
                        continue
                    if d2 == 0:
                        continue

                    xx = xi + dx
                    if xx < 0 or xx >= width:
                        continue

                    jdx = yy * width + xx
                    tj = arr[jdx]
                    if tj == 0:
                        continue

                    dtj = np.uint64(ti - tj)
                    if dtj > np.uint64(support_dt_ticks):
                        continue

                    # Triangular recency term times inverse-distance spatial term.
                    w_t = 1.0 - float(dtj) * inv_support_dt
                    if w_t <= 0.0:
                        continue
                    w_sp = 1.0 / (1.0 + float(d2))
                    w = w_t * w_sp

                    support += w
                    sum_w += w

                    # Motion proxy: unit vector from neighbor to current event.
                    ux = float(dx)
                    uy = float(dy)
                    norm = np.sqrt(ux * ux + uy * uy)
                    if norm > eps:
                        sum_ux += w * (ux / norm)
                        sum_uy += w * (uy / norm)

            momentum_consistency = 0.0
            if sum_w > eps:
                vec_norm = np.sqrt(sum_ux * sum_ux + sum_uy * sum_uy)
                momentum_consistency = vec_norm / (sum_w + eps)
                if momentum_consistency < 0.0:
                    momentum_consistency = 0.0
                if momentum_consistency > 1.0:
                    momentum_consistency = 1.0

            score = support_gain * support * (1.0 + momentum_gain * momentum_consistency)
            out[i] = np.float32(score)

            idx0 = yi * width + xi
            arr[idx0] = ti

        return out

    return _kernel


def score_stream_n4(
    ev,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_us: int,
    tb: TimeBase,
    scores_out: np.ndarray | None = None,
) -> np.ndarray:
    """N4: local momentum-consistency weighted support.

    Real-time, one-pass, O(r^2) per event.

    Let S_i be same-polarity local support in a short window, and c_i be
    neighborhood direction-consistency in [0,1] from weighted unit vectors.

    score_i = gain_support * S_i * (1 + gain_momentum * c_i)

    This keeps baseline density evidence as the main term, and uses momentum
    consistency as a mild multiplicative boost.

    Env vars:
    - MYEVS_N4_SUPPORT_DT_US (default min(tau_us, 32000))
    - MYEVS_N4_SUPPORT_GAIN (default 1.00)
    - MYEVS_N4_MOMENTUM_GAIN (default 0.60)
    """

    n = int(ev.t.shape[0])
    if scores_out is None:
        scores_out = np.empty((n,), dtype=np.float32)
    if n <= 0:
        return scores_out

    support_dt_us = int(_read_int_env("MYEVS_N4_SUPPORT_DT_US", min(int(tau_us), 32000)))
    support_dt_ticks = int(tb.us_to_ticks(max(1, support_dt_us)))

    support_gain = float(_read_float_env("MYEVS_N4_SUPPORT_GAIN", 1.00))
    momentum_gain = float(_read_float_env("MYEVS_N4_MOMENTUM_GAIN", 0.60))
    ker = _try_build_n4_kernel()
    scores = ker(
        ev.t.astype(np.uint64, copy=False),
        ev.x.astype(np.int32, copy=False),
        ev.y.astype(np.int32, copy=False),
        ev.p,
        int(width),
        int(height),
        int(radius_px),
        int(max(1, support_dt_ticks)),
        float(max(0.0, support_gain)),
        float(max(0.0, momentum_gain)),
    )
    scores_out[:] = scores
    return scores_out
