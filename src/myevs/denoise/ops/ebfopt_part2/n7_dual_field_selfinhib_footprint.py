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
        raise RuntimeError("n7 requires numba, but import failed")


def _try_build_n7_kernel():
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
        tau_foot_ticks: float,
        foot_dt_max_ticks: int,
        self_gain: float,
        lambda_inhib: float,
        sigma_spatial: float,
        escape_ref_ticks: float,
        escape_eta: float,
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

            # Self-inhibition decay on current pixel.
            t_prev = last_inhib_update[idx]
            if t_prev > 0:
                dt = float(np.uint64(ti - t_prev))
                inhib_self[idx] = inhib_self[idx] * np.exp(-dt / max(1.0, tau_inhib_ticks))

            arr = last_pos if int(p[i]) > 0 else last_neg

            # Footprint field: same-polarity nearby trail evidence (excluding center pixel).
            e_foot = 0.0
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
                    w_t = np.exp(-float(dtj) / max(1.0, tau_foot_ticks))
                    e_foot += w_s * w_t

            # Escape reward based on center same-polarity freshness.
            t0 = arr[idx]
            if t0 > 0:
                dt0 = float(np.uint64(ti - t0))
                u0 = dt0 / max(1.0, escape_ref_ticks)
                if u0 < 0.0:
                    u0 = 0.0
                if u0 > 1.0:
                    u0 = 1.0
                r_escape = np.power(u0, max(0.0, escape_eta))
            else:
                r_escape = 1.0

            e_star = e_foot * r_escape
            score = e_star / (1.0 + lambda_inhib * float(inhib_self[idx]))
            out[i] = np.float32(score)

            # State updates.
            inhib_self[idx] += np.float32(self_gain)
            last_inhib_update[idx] = ti
            arr[idx] = ti

        return out

    return _kernel


def score_stream_n7(
    ev,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_us: int,
    tb: TimeBase,
    scores_out: np.ndarray | None = None,
) -> np.ndarray:
    """N7 (7.35 related): dual-field self-inhibition and footprint model.

    score_i = E_foot*(x_i, y_i, t_i) / (1 + lambda * I_self(x_i, y_i, t_i))
    E_foot* = E_foot * R_escape,
    R_escape = clip(dt0 / T_ref, 0, 1)^eta.

    Env vars:
    - MYEVS_N7_TAU_INHIB_US (default 32000)
    - MYEVS_N7_TAU_FOOT_US (default 16000)
    - MYEVS_N7_FOOT_DT_MAX_US (default 32000)
    - MYEVS_N7_SELF_GAIN (default 1.00)
    - MYEVS_N7_LAMBDA_INHIB (default 0.80)
    - MYEVS_N7_SIGMA_SPATIAL (default 1.50)
    - MYEVS_N7_ESCAPE_REF_US (default 32000)
    - MYEVS_N7_ESCAPE_ETA (default 1.00)
    """

    n = int(ev.t.shape[0])
    if scores_out is None:
        scores_out = np.empty((n,), dtype=np.float32)
    if n <= 0:
        return scores_out

    tau_inhib_us = int(_read_int_env("MYEVS_N7_TAU_INHIB_US", 32000))
    tau_foot_us = int(_read_int_env("MYEVS_N7_TAU_FOOT_US", 16000))
    foot_dt_max_us = int(_read_int_env("MYEVS_N7_FOOT_DT_MAX_US", 32000))

    tau_inhib_ticks = float(tb.us_to_ticks(max(1, tau_inhib_us)))
    tau_foot_ticks = float(tb.us_to_ticks(max(1, tau_foot_us)))
    foot_dt_max_ticks = int(tb.us_to_ticks(max(1, foot_dt_max_us)))

    self_gain = float(_read_float_env("MYEVS_N7_SELF_GAIN", 1.00))
    lambda_inhib = float(_read_float_env("MYEVS_N7_LAMBDA_INHIB", 0.80))
    sigma_spatial = float(_read_float_env("MYEVS_N7_SIGMA_SPATIAL", 1.50))
    escape_ref_us = int(_read_int_env("MYEVS_N7_ESCAPE_REF_US", 32000))
    escape_ref_ticks = float(tb.us_to_ticks(max(1, escape_ref_us)))
    escape_eta = float(_read_float_env("MYEVS_N7_ESCAPE_ETA", 1.00))

    ker = _try_build_n7_kernel()
    scores = ker(
        ev.t.astype(np.uint64, copy=False),
        ev.x.astype(np.int32, copy=False),
        ev.y.astype(np.int32, copy=False),
        ev.p,
        int(width),
        int(height),
        int(radius_px),
        float(max(1.0, tau_inhib_ticks)),
        float(max(1.0, tau_foot_ticks)),
        int(max(1, foot_dt_max_ticks)),
        float(max(0.0, self_gain)),
        float(max(0.0, lambda_inhib)),
        float(max(0.2, sigma_spatial)),
        float(max(1.0, escape_ref_ticks)),
        float(max(0.0, escape_eta)),
    )
    scores_out[:] = scores
    return scores_out
