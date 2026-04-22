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
        raise RuntimeError("n8 requires numba, but import failed")


def _try_build_n8_kernel():
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
        lambda0: float,
        sigma_spatial: float,
        rho0: float,
        tau_traj_ticks: float,
        tau_traj_cons_ticks: float,
        traj_dt_max_ticks: int,
        gamma_traj0: float,
        g_tau_short_ticks: float,
        g_tau_long_ticks: float,
        g_alpha: float,
        g_beta: float,
        g_traj_beta: float,
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

        t_prev_global = np.uint64(0)
        rate_short = 0.0
        rate_long = 0.0

        dxs = (-1, 0, 1, -1, 1, -1, 0, 1)
        dys = (-1, -1, -1, 0, 0, 1, 1, 1)

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            if xi < 0 or xi >= width or yi < 0 or yi >= height:
                continue

            ti = np.uint64(t[i])
            idx = yi * width + xi

            if t_prev_global > 0:
                dtg = float(np.uint64(ti - t_prev_global))
                inv_dt = 1.0 / max(1.0, dtg)
                ds = np.exp(-dtg / max(1.0, g_tau_short_ticks))
                dl = np.exp(-dtg / max(1.0, g_tau_long_ticks))
                rate_short = rate_short * ds + (1.0 - ds) * inv_dt
                rate_long = rate_long * dl + (1.0 - dl) * inv_dt
            t_prev_global = ti

            g = 0.0
            if rate_long > 0.0:
                g = rate_short / (rate_long + eps) - 1.0
            if g < 0.0:
                g = 0.0
            if g > 1.0:
                g = 1.0

            lambda_eff = lambda0 * (1.0 + g_alpha * g)
            rho_eff = rho0 * (1.0 - g_beta * g)
            gamma_traj_eff = gamma_traj0 * (1.0 - g_traj_beta * g)
            if rho_eff < 0.0:
                rho_eff = 0.0
            if gamma_traj_eff < 0.0:
                gamma_traj_eff = 0.0

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
            e_dual = e_fast + rho_eff * delta

            traj = 0.0
            for k in range(8):
                dx = dxs[k]
                dy = dys[k]

                xb = xi - dx
                yb = yi - dy
                xf = xi + dx
                yf = yi + dy

                wb = 0.0
                wf = 0.0
                if xb >= 0 and xb < width and yb >= 0 and yb < height:
                    ib = yb * width + xb
                    tbk = arr[ib]
                    if tbk > 0:
                        dtb = np.uint64(ti - tbk)
                        if dtb <= np.uint64(traj_dt_max_ticks):
                            wb = np.exp(-float(dtb) / max(1.0, tau_traj_ticks))

                if xf >= 0 and xf < width and yf >= 0 and yf < height:
                    iff = yf * width + xf
                    tfw = arr[iff]
                    if tfw > 0:
                        dtf = np.uint64(ti - tfw)
                        if dtf <= np.uint64(traj_dt_max_ticks):
                            wf = np.exp(-float(dtf) / max(1.0, tau_traj_ticks))

                wdir = 0.0
                if wb > 0.0 and wf > 0.0:
                    # Favor causal chains that are recent on both sides and temporally symmetric.
                    ib = yb * width + xb
                    iff = yf * width + xf
                    dtb = float(np.uint64(ti - arr[ib]))
                    dtf = float(np.uint64(ti - arr[iff]))
                    w_cons = np.exp(-abs(dtb - dtf) / max(1.0, tau_traj_cons_ticks))
                    wdir = wb * wf * w_cons
                elif wb > 0.0:
                    wdir = 0.5 * wb
                elif wf > 0.0:
                    wdir = 0.5 * wf

                if wdir > traj:
                    traj = wdir

            score_num = e_dual + gamma_traj_eff * traj
            score = score_num / (1.0 + lambda_eff * float(inhib_self[idx]))
            out[i] = np.float32(score)

            inhib_self[idx] += np.float32(self_gain)
            last_inhib_update[idx] = ti
            arr[idx] = ti

        return out

    return _kernel


def score_stream_n8(
    ev,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_us: int,
    tb: TimeBase,
    scores_out: np.ndarray | None = None,
) -> np.ndarray:
    """N8: causal trajectory evidence on top of dual-field model.

    score_i = (E_dual + gamma_traj * C_traj) / (1 + lambda_eff * I_self)

    E_dual uses N7.1 dual-timescale footprint.
    C_traj is 8-direction causal continuity evidence from same-polarity timestamps.

    Env vars:
    - MYEVS_N8_TAU0_US (default 16000)
    - MYEVS_N8_TAU_INHIB_US (default 32000)
    - MYEVS_N8_SLOW_RATIO (default 4.0)
    - MYEVS_N8_FOOT_DT_MAX_RATIO (default 8.0)
    - MYEVS_N8_RHO0 (default 0.60)
    - MYEVS_N8_SELF_GAIN (default 0.55)
    - MYEVS_N8_LAMBDA0 (default 0.30)
    - MYEVS_N8_SIGMA_SPATIAL (default 2.20)
    - MYEVS_N8_TAU_TRAJ_US (default 12000)
    - MYEVS_N8_TAU_TRAJ_CONS_US (default 8000)
    - MYEVS_N8_TRAJ_DT_MAX_US (default 32000)
    - MYEVS_N8_GAMMA_TRAJ0 (default 1.20)
    - MYEVS_N8_G_TAU_SHORT_US (default 2000)
    - MYEVS_N8_G_TAU_LONG_US (default 20000)
    - MYEVS_N8_G_ALPHA (default 0.80)
    - MYEVS_N8_G_BETA (default 0.30)
    - MYEVS_N8_G_TRAJ_BETA (default 0.20)
    """

    n = int(ev.t.shape[0])
    if scores_out is None:
        scores_out = np.empty((n,), dtype=np.float32)
    if n <= 0:
        return scores_out

    tau0_us = int(_read_int_env("MYEVS_N8_TAU0_US", 16000))
    tau_inhib_us = int(_read_int_env("MYEVS_N8_TAU_INHIB_US", 32000))
    slow_ratio = float(_read_float_env("MYEVS_N8_SLOW_RATIO", 4.0))
    dt_max_ratio = float(_read_float_env("MYEVS_N8_FOOT_DT_MAX_RATIO", 8.0))

    tau_fast_ticks = float(tb.us_to_ticks(max(1, tau0_us)))
    tau_slow_ticks = float(tb.us_to_ticks(max(1, int(tau0_us * max(1.0, slow_ratio)))))
    tau_inhib_ticks = float(tb.us_to_ticks(max(1, tau_inhib_us)))
    foot_dt_max_ticks = int(tb.us_to_ticks(max(1, int(tau0_us * max(1.0, dt_max_ratio)))))

    rho0 = float(_read_float_env("MYEVS_N8_RHO0", 0.60))
    self_gain = float(_read_float_env("MYEVS_N8_SELF_GAIN", 0.55))
    lambda0 = float(_read_float_env("MYEVS_N8_LAMBDA0", 0.30))
    sigma_spatial = float(_read_float_env("MYEVS_N8_SIGMA_SPATIAL", 2.20))

    tau_traj_us = int(_read_int_env("MYEVS_N8_TAU_TRAJ_US", 12000))
    tau_traj_cons_us = int(_read_int_env("MYEVS_N8_TAU_TRAJ_CONS_US", 8000))
    traj_dt_max_us = int(_read_int_env("MYEVS_N8_TRAJ_DT_MAX_US", 32000))
    tau_traj_ticks = float(tb.us_to_ticks(max(1, tau_traj_us)))
    tau_traj_cons_ticks = float(tb.us_to_ticks(max(1, tau_traj_cons_us)))
    traj_dt_max_ticks = int(tb.us_to_ticks(max(1, traj_dt_max_us)))
    gamma_traj0 = float(_read_float_env("MYEVS_N8_GAMMA_TRAJ0", 1.20))

    g_tau_short_us = int(_read_int_env("MYEVS_N8_G_TAU_SHORT_US", 2000))
    g_tau_long_us = int(_read_int_env("MYEVS_N8_G_TAU_LONG_US", 20000))
    g_tau_short_ticks = float(tb.us_to_ticks(max(1, g_tau_short_us)))
    g_tau_long_ticks = float(tb.us_to_ticks(max(1, g_tau_long_us)))
    g_alpha = float(_read_float_env("MYEVS_N8_G_ALPHA", 0.80))
    g_beta = float(_read_float_env("MYEVS_N8_G_BETA", 0.30))
    g_traj_beta = float(_read_float_env("MYEVS_N8_G_TRAJ_BETA", 0.20))

    ker = _try_build_n8_kernel()
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
        float(max(0.0, lambda0)),
        float(max(0.2, sigma_spatial)),
        float(max(0.0, rho0)),
        float(max(1.0, tau_traj_ticks)),
        float(max(1.0, tau_traj_cons_ticks)),
        int(max(1, traj_dt_max_ticks)),
        float(max(0.0, gamma_traj0)),
        float(max(1.0, g_tau_short_ticks)),
        float(max(1.0, g_tau_long_ticks)),
        float(max(0.0, g_alpha)),
        float(max(0.0, g_beta)),
        float(max(0.0, g_traj_beta)),
    )
    scores_out[:] = scores
    return scores_out
