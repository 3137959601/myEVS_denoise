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
        raise RuntimeError("n71 requires numba, but import failed")


def _try_build_n71_kernel():
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
        tau_inhib_ticks: float,
        tau_fast_ticks: float,
        tau_slow_ticks: float,
        foot_dt_max_ticks: int,
        self_gain: float,
        lambda_inhib: float,
        sigma_spatial: float,
        rho: float,
    ) -> np.ndarray:
        n = int(t.shape[0])
        out = np.zeros((n,), dtype=np.float32)

        npx = width * height
        inhib_self = np.zeros((npx,), dtype=np.float32)
        last_inhib_update = np.zeros((npx,), dtype=np.uint64)
        last_pos = np.zeros((npx,), dtype=np.uint64)
        last_neg = np.zeros((npx,), dtype=np.uint64)

        r = int(radius_px)
        r2 = int(r * r)
        eps = 1e-6
        inv_two_sigma2 = 1.0 / max(eps, 2.0 * sigma_spatial * sigma_spatial)

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            if xi < 0 or xi >= width or yi < 0 or yi >= height:
                continue

            ti = np.uint64(t[i])
            idx = yi * width + xi

            t_prev = last_inhib_update[idx]
            if t_prev > 0:
                dt = float(np.uint64(ti - t_prev))
                inhib_self[idx] = inhib_self[idx] * np.exp(-dt / max(1.0, tau_inhib_ticks))

            arr = last_pos if int(p[i]) > 0 else last_neg

            e_fast = 0.0
            e_slow = 0.0
            for dy in range(-r, r + 1):
                yy = yi + dy
                if yy < 0 or yy >= height:
                    continue
                for dx in range(-r, r + 1):
                    d2 = dx * dx + dy * dy
                    if d2 > r2 or d2 == 0:
                        continue

                    xx = xi + dx
                    if xx < 0 or xx >= width:
                        continue

                    jdx = yy * width + xx
                    tj = arr[jdx]
                    if tj == 0:
                        continue

                    dtj = np.uint64(ti - tj)
                    if dtj > np.uint64(foot_dt_max_ticks):
                        continue

                    w_s = np.exp(-float(d2) * inv_two_sigma2)
                    e_fast += w_s * np.exp(-float(dtj) / max(1.0, tau_fast_ticks))
                    e_slow += w_s * np.exp(-float(dtj) / max(1.0, tau_slow_ticks))

            delta = e_slow - e_fast
            if delta < 0.0:
                delta = 0.0
            e_dual = e_fast + rho * delta

            score = e_dual / (1.0 + lambda_inhib * float(inhib_self[idx]))
            out[i] = np.float32(score)

            inhib_self[idx] += np.float32(self_gain)
            last_inhib_update[idx] = ti
            arr[idx] = ti

        return out

    return _kernel


def score_stream_n71(
    ev,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_us: int,
    tb: TimeBase,
    scores_out: np.ndarray | None = None,
) -> np.ndarray:
    """N7.1: dual-timescale footprint with self-inhibition.

    score_i = E_dual / (1 + lambda * I_self)
    E_dual = E_fast + rho * max(0, E_slow - E_fast)

    Env vars:
    - MYEVS_N71_TAU0_US (default 16000)
    - MYEVS_N71_TAU_INHIB_US (default 32000)
    - MYEVS_N71_SLOW_RATIO (default 4.0)
    - MYEVS_N71_FOOT_DT_MAX_RATIO (default 8.0)
    - MYEVS_N71_RHO (default 0.60)
    - MYEVS_N71_SELF_GAIN (default 0.55)
    - MYEVS_N71_LAMBDA_INHIB (default 0.30)
    - MYEVS_N71_SIGMA_SPATIAL (default 2.20)
    """

    n = int(ev.t.shape[0])
    if scores_out is None:
        scores_out = np.empty((n,), dtype=np.float32)
    if n <= 0:
        return scores_out

    # Bind n71 timescales to caller tau_us by default so sweep tau-us-list is effective.
    tau0_default_us = max(1000, int(tau_us) // 8)
    tau_inhib_default_us = max(1000, int(tau_us) // 4)

    tau0_env = os.environ.get("MYEVS_N71_TAU0_US", "").strip()
    tau_inhib_env = os.environ.get("MYEVS_N71_TAU_INHIB_US", "").strip()

    tau0_us = int(_read_int_env("MYEVS_N71_TAU0_US", tau0_default_us)) if tau0_env else int(tau0_default_us)
    tau_inhib_us = (
        int(_read_int_env("MYEVS_N71_TAU_INHIB_US", tau_inhib_default_us))
        if tau_inhib_env
        else int(tau_inhib_default_us)
    )
    slow_ratio = float(_read_float_env("MYEVS_N71_SLOW_RATIO", 4.0))
    dt_max_ratio = float(_read_float_env("MYEVS_N71_FOOT_DT_MAX_RATIO", 8.0))

    tau_fast_ticks = float(tb.us_to_ticks(max(1, tau0_us)))
    tau_slow_ticks = float(tb.us_to_ticks(max(1, int(tau0_us * max(1.0, slow_ratio)))))
    tau_inhib_ticks = float(tb.us_to_ticks(max(1, tau_inhib_us)))
    foot_dt_max_ticks = int(tb.us_to_ticks(max(1, int(tau0_us * max(1.0, dt_max_ratio)))))

    rho = float(_read_float_env("MYEVS_N71_RHO", 0.60))
    self_gain = float(_read_float_env("MYEVS_N71_SELF_GAIN", 0.55))
    lambda_inhib = float(_read_float_env("MYEVS_N71_LAMBDA_INHIB", 0.30))
    sigma_spatial = float(_read_float_env("MYEVS_N71_SIGMA_SPATIAL", 2.20))

    ker = _try_build_n71_kernel()
    scores = ker(
        ev.t.astype(np.uint64, copy=False),
        ev.x.astype(np.int32, copy=False),
        ev.y.astype(np.int32, copy=False),
        ev.p,
        int(width),
        int(height),
        int(radius_px),
        float(max(1.0, tau_inhib_ticks)),
        float(max(1.0, tau_fast_ticks)),
        float(max(1.0, tau_slow_ticks)),
        int(max(1, foot_dt_max_ticks)),
        float(max(0.0, self_gain)),
        float(max(0.0, lambda_inhib)),
        float(max(0.2, sigma_spatial)),
        float(max(0.0, rho)),
    )
    scores_out[:] = scores
    return scores_out
