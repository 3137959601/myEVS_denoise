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
        raise RuntimeError("n6 requires numba, but import failed")


def _try_build_n6_kernel():
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
        tau_inhib_ticks: float,
        inhib_gain: float,
        lateral_gain: float,
        self_gain: float,
        support_gain: float,
        momentum_gain: float,
        escape_gain: float,
        escape_floor: float,
    ) -> np.ndarray:
        n = int(t.shape[0])
        out = np.zeros((n,), dtype=np.float32)

        npx = width * height
        inhib = np.zeros((npx,), dtype=np.float32)
        last_update = np.zeros((npx,), dtype=np.uint64)
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
            idx = yi * width + xi

            # Decay center inhibition.
            t_prev = last_update[idx]
            if t_prev > 0:
                dt = float(np.uint64(ti - t_prev))
                if tau_inhib_ticks > 1.0:
                    inhib[idx] = inhib[idx] * np.exp(-dt / tau_inhib_ticks)

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
                    xx = xi + dx
                    if xx < 0 or xx >= width:
                        continue

                    jdx = yy * width + xx
                    tj = arr[jdx]
                    if tj == 0:
                        continue

                    dtj = np.uint64(ti - tj)
                    if dtj <= np.uint64(support_dt_ticks):
                        # N2 support + N4-style recency-weighted direction evidence.
                        w_sp = 1.0 / (1.0 + float(d2))
                        w_t = 1.0 - float(dtj) * inv_support_dt
                        if w_t < 0.0:
                            w_t = 0.0
                        support += w_sp

                        if d2 > 0 and w_t > 0.0:
                            w = w_t * w_sp
                            sum_w += w
                            ux = float(dx)
                            uy = float(dy)
                            norm = np.sqrt(ux * ux + uy * uy)
                            if norm > eps:
                                sum_ux += w * (ux / norm)
                                sum_uy += w * (uy / norm)

                    # Lateral inhibition update for neighborhood.
                    t_up = last_update[jdx]
                    if t_up > 0:
                        dtu = float(np.uint64(ti - t_up))
                        if tau_inhib_ticks > 1.0:
                            inhib[jdx] = inhib[jdx] * np.exp(-dtu / tau_inhib_ticks)
                    inhib[jdx] += np.float32(inhib_gain * lateral_gain / (1.0 + float(d2)))
                    last_update[jdx] = ti

            # Momentum consistency in [0,1].
            c = 0.0
            if sum_w > eps:
                vec_norm = np.sqrt(sum_ux * sum_ux + sum_uy * sum_uy)
                c = vec_norm / (sum_w + eps)
                if c < 0.0:
                    c = 0.0
                if c > 1.0:
                    c = 1.0

            # Escape gate: high consistency => lower effective inhibition.
            kappa = 1.0 - escape_gain * c
            if kappa < escape_floor:
                kappa = escape_floor
            if kappa > 1.0:
                kappa = 1.0

            # Keep self-inhibition, but allow consistency-based escape.
            inhib[idx] += np.float32(inhib_gain * self_gain * kappa)
            last_update[idx] = ti

            num = support_gain * support * (1.0 + momentum_gain * c)
            den = 1.0 + float(inhib[idx]) * kappa
            out[i] = np.float32(num / den)

            arr[idx] = ti

        return out

    return _kernel


def score_stream_n6(
    ev,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_us: int,
    tb: TimeBase,
    scores_out: np.ndarray | None = None,
) -> np.ndarray:
    """N6: dynamic inhibition + momentum-consistency escape gate.

    Real-time, one-pass, O(r^2) per event.

    score_i = [g_s * S_i * (1 + g_m * c_i)] / [1 + kappa_i * I(x_i, y_i)]
    kappa_i = clip(1 - g_e * c_i, kappa_min, 1)

    where:
    - S_i: same-polarity local short-term support (N2 style)
    - I: dynamic lateral inhibition field (N2 style)
    - c_i in [0,1]: local momentum consistency (N4 style)

    Env vars:
    - MYEVS_N6_TAU_INHIB_US (default 32000)
    - MYEVS_N6_SUPPORT_DT_US (default min(tau_us, 32000))
    - MYEVS_N6_INHIB_GAIN (default 0.35)
    - MYEVS_N6_LATERAL_GAIN (default 0.60)
    - MYEVS_N6_SELF_GAIN (default 1.60)
    - MYEVS_N6_SUPPORT_GAIN (default 1.00)
    - MYEVS_N6_MOMENTUM_GAIN (default 0.60)
    - MYEVS_N6_ESCAPE_GAIN (default 0.70)
    - MYEVS_N6_ESCAPE_FLOOR (default 0.25)
    """

    n = int(ev.t.shape[0])
    if scores_out is None:
        scores_out = np.empty((n,), dtype=np.float32)
    if n <= 0:
        return scores_out

    tau_inhib_us = int(_read_int_env("MYEVS_N6_TAU_INHIB_US", 32000))
    support_dt_us = int(_read_int_env("MYEVS_N6_SUPPORT_DT_US", min(int(tau_us), 32000)))

    tau_inhib_ticks = float(tb.us_to_ticks(max(1, tau_inhib_us)))
    support_dt_ticks = int(tb.us_to_ticks(max(1, support_dt_us)))

    inhib_gain = float(_read_float_env("MYEVS_N6_INHIB_GAIN", 0.35))
    lateral_gain = float(_read_float_env("MYEVS_N6_LATERAL_GAIN", 0.60))
    self_gain = float(_read_float_env("MYEVS_N6_SELF_GAIN", 1.60))
    support_gain = float(_read_float_env("MYEVS_N6_SUPPORT_GAIN", 1.00))
    momentum_gain = float(_read_float_env("MYEVS_N6_MOMENTUM_GAIN", 0.60))
    escape_gain = float(_read_float_env("MYEVS_N6_ESCAPE_GAIN", 0.70))
    escape_floor = float(_read_float_env("MYEVS_N6_ESCAPE_FLOOR", 0.25))

    ker = _try_build_n6_kernel()
    scores = ker(
        ev.t.astype(np.uint64, copy=False),
        ev.x.astype(np.int32, copy=False),
        ev.y.astype(np.int32, copy=False),
        ev.p,
        int(width),
        int(height),
        int(radius_px),
        int(max(1, support_dt_ticks)),
        float(max(1.0, tau_inhib_ticks)),
        float(max(0.0, inhib_gain)),
        float(max(0.0, lateral_gain)),
        float(max(0.0, self_gain)),
        float(max(0.0, support_gain)),
        float(max(0.0, momentum_gain)),
        float(max(0.0, escape_gain)),
        float(min(1.0, max(0.0, escape_floor))),
    )
    scores_out[:] = scores
    return scores_out
