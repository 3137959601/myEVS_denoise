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
        raise RuntimeError("n2 requires numba, but import failed")


def _try_build_n2_kernel():
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

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            if xi < 0 or xi >= width or yi < 0 or yi >= height:
                continue

            ti = np.uint64(t[i])
            idx = yi * width + xi

            # Time decay for center inhibition value.
            t_prev = last_update[idx]
            if t_prev > 0:
                dt = float(np.uint64(ti - t_prev))
                if tau_inhib_ticks > 1.0:
                    inhib[idx] = inhib[idx] * np.exp(-dt / tau_inhib_ticks)

            arr = last_pos if int(p[i]) > 0 else last_neg

            support = 0.0
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
                        # Nearer neighbors contribute more support.
                        w = 1.0 / (1.0 + float(d2))
                        support += w

                    # Lateral inhibition update for neighborhood.
                    t_up = last_update[jdx]
                    if t_up > 0:
                        dtu = float(np.uint64(ti - t_up))
                        if tau_inhib_ticks > 1.0:
                            inhib[jdx] = inhib[jdx] * np.exp(-dtu / tau_inhib_ticks)
                    inhib[jdx] += np.float32(inhib_gain * lateral_gain / (1.0 + float(d2)))
                    last_update[jdx] = ti

            # Stronger self inhibition for repeated same-pixel firing.
            inhib[idx] += np.float32(inhib_gain * self_gain)
            last_update[idx] = ti

            # Score: support evidence divided by local inhibition.
            s = (support_gain * support) / (1.0 + float(inhib[idx]))
            out[i] = np.float32(s)

            arr[idx] = ti

        return out

    return _kernel


def score_stream_n2(
    ev,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_us: int,
    tb: TimeBase,
    scores_out: np.ndarray | None = None,
) -> np.ndarray:
    """N2: dynamic lateral inhibition field with local support.

    Real-time, one-pass, O(r^2) per event.

    For each event i, we accumulate local same-polarity short-term support S_i,
    and maintain an inhibition field I(x,y) with exponential decay and lateral
    increments. Final score is:

    score_i = (gain_support * S_i) / (1 + I(x_i, y_i)).

    Env vars:
    - MYEVS_N2_TAU_INHIB_US (default 32000)
    - MYEVS_N2_SUPPORT_DT_US (default min(tau_us, 32000))
    - MYEVS_N2_INHIB_GAIN (default 0.35)
    - MYEVS_N2_LATERAL_GAIN (default 0.60)
    - MYEVS_N2_SELF_GAIN (default 1.60)
    - MYEVS_N2_SUPPORT_GAIN (default 1.00)
    """

    n = int(ev.t.shape[0])
    if scores_out is None:
        scores_out = np.empty((n,), dtype=np.float32)
    if n <= 0:
        return scores_out

    tau_inhib_us = int(_read_int_env("MYEVS_N2_TAU_INHIB_US", 32000))
    support_dt_us = int(_read_int_env("MYEVS_N2_SUPPORT_DT_US", min(int(tau_us), 32000)))

    tau_inhib_ticks = float(tb.us_to_ticks(max(1, tau_inhib_us)))
    support_dt_ticks = int(tb.us_to_ticks(max(1, support_dt_us)))

    inhib_gain = float(_read_float_env("MYEVS_N2_INHIB_GAIN", 0.35))
    lateral_gain = float(_read_float_env("MYEVS_N2_LATERAL_GAIN", 0.60))
    self_gain = float(_read_float_env("MYEVS_N2_SELF_GAIN", 1.60))
    support_gain = float(_read_float_env("MYEVS_N2_SUPPORT_GAIN", 1.00))
    ker = _try_build_n2_kernel()
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
    )
    scores_out[:] = scores
    return scores_out
