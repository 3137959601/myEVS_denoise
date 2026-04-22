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
        raise RuntimeError("n84 requires numba, but import failed")


def _try_build_n84_kernel():
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
        tau_fast_ticks: float,
        tau_slow_ticks: float,
        tau_inhib_ticks: float,
        tau_confirm_ticks: float,
        tau_kill_ticks: float,
        self_gain: float,
        foot_sigma: float,
        foot_rho: float,
        foot_dt_max_ticks: float,
        c_hot: float,
        c_foot: float,
        r0_px: float,
        kv_px_per_tick: float,
        d_new: float,
        alpha_slow: float,
        w_x: float,
        w_v: float,
        w_h: float,
        w_g: float,
        w_f: float,
        w_t: float,
        alpha_score: float,
        confirm_link_min: float,
        track_capacity: int,
        v_ema: float,
        sigma_ema: float,
        k_len: float,
        beta_g: float,
        g_tau_short_ticks: float,
        g_tau_long_ticks: float,
        lmax: float,
    ) -> np.ndarray:
        n = int(t.shape[0])
        out = np.zeros((n,), dtype=np.float32)

        npx = width * height

        inhib_self = np.zeros((npx,), dtype=np.float32)
        last_inhib_update = np.zeros((npx,), dtype=np.uint64)

        last_pos = np.zeros((npx,), dtype=np.uint64)
        last_neg = np.zeros((npx,), dtype=np.uint64)

        max_tracks = int(track_capacity)
        if max_tracks < 32:
            max_tracks = 32

        tr_state = np.zeros((max_tracks,), dtype=np.int8)  # 0 dead, 1 tentative, 2 confirmed
        tr_pol = np.zeros((max_tracks,), dtype=np.int8)
        tr_x_last = np.zeros((max_tracks,), dtype=np.float32)
        tr_y_last = np.zeros((max_tracks,), dtype=np.float32)
        tr_t_last = np.zeros((max_tracks,), dtype=np.uint64)
        tr_x_prev = np.zeros((max_tracks,), dtype=np.float32)
        tr_y_prev = np.zeros((max_tracks,), dtype=np.float32)
        tr_t_prev = np.zeros((max_tracks,), dtype=np.uint64)
        tr_t_birth = np.zeros((max_tracks,), dtype=np.uint64)
        tr_vx = np.zeros((max_tracks,), dtype=np.float32)
        tr_vy = np.zeros((max_tracks,), dtype=np.float32)
        tr_sigma_v = np.zeros((max_tracks,), dtype=np.float32)
        tr_sigma_theta = np.zeros((max_tracks,), dtype=np.float32)
        tr_len = np.zeros((max_tracks,), dtype=np.float32)
        tr_mean_hot = np.zeros((max_tracks,), dtype=np.float32)
        tr_mean_foot = np.zeros((max_tracks,), dtype=np.float32)
        tr_last_link = np.zeros((max_tracks,), dtype=np.float32)
        tr_pending0 = np.full((max_tracks,), -1, dtype=np.int32)
        tr_pending1 = np.full((max_tracks,), -1, dtype=np.int32)
        tr_pending_cnt = np.zeros((max_tracks,), dtype=np.int8)

        t_prev_global = np.uint64(0)
        rate_short = 0.0
        rate_long = 0.0

        r = int(radius_px)
        if r < 1:
            r = 1
        r2 = int(r * r)
        inv_two_sigma2 = 1.0 / max(1e-6, 2.0 * foot_sigma * foot_sigma)

        eps = 1e-6

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

            t_prev = last_inhib_update[idx]
            if t_prev > 0:
                dt_i = float(np.uint64(ti - t_prev))
                inhib_self[idx] = inhib_self[idx] * np.exp(-dt_i / max(1.0, tau_inhib_ticks))
            u_hot_evt = float(inhib_self[idx]) / (float(inhib_self[idx]) + max(eps, c_hot))

            arr = last_pos if int(p[i]) > 0 else last_neg

            # Dual-footprint support (kept as reward signal, not main score axis).
            e_fast = 0.0
            e_slow = 0.0
            for dy in range(-r, r + 1):
                yy = yi + dy
                if yy < 0 or yy >= height:
                    continue
                for dx in range(-r, r + 1):
                    d2 = dx * dx + dy * dy
                    if d2 == 0 or d2 > r2:
                        continue
                    xx = xi + dx
                    if xx < 0 or xx >= width:
                        continue
                    jdx = yy * width + xx
                    tj = arr[jdx]
                    if tj == 0:
                        continue
                    dtj = float(np.uint64(ti - tj))
                    if dtj <= 0.0 or dtj > foot_dt_max_ticks:
                        continue
                    ws = np.exp(-float(d2) * inv_two_sigma2)
                    e_fast += ws * np.exp(-dtj / max(1.0, tau_fast_ticks))
                    e_slow += ws * np.exp(-dtj / max(1.0, tau_slow_ticks))

            delta = e_slow - e_fast
            if delta < 0.0:
                delta = 0.0
            e_dual = e_fast + foot_rho * delta
            u_foot_evt = e_dual / (e_dual + max(eps, c_foot))

            # Track matching with predicted position and time-adaptive radius.
            best_k = -1
            best_d = 1e12
            best_link = 0.0
            best_dt = 0.0

            for k in range(max_tracks):
                st = int(tr_state[k])
                if st == 0:
                    continue

                tlast = tr_t_last[k]
                if tlast == 0:
                    tr_state[k] = 0
                    continue

                dt_ticks = float(np.uint64(ti - tlast))
                if dt_ticks <= 0.0:
                    continue
                if dt_ticks > tau_kill_ticks:
                    tr_state[k] = 0
                    tr_pending0[k] = -1
                    tr_pending1[k] = -1
                    tr_pending_cnt[k] = np.int8(0)
                    continue

                if int(tr_pol[k]) != int(p[i]):
                    continue
                if dt_ticks > tau_slow_ticks:
                    continue

                pred_x = float(tr_x_last[k]) + float(tr_vx[k]) * dt_ticks
                pred_y = float(tr_y_last[k]) + float(tr_vy[k]) * dt_ticks

                r_pred = r0_px + kv_px_per_tick * dt_ticks
                if r_pred < 1.0:
                    r_pred = 1.0

                ex = float(xi) - pred_x
                ey = float(yi) - pred_y
                dist = np.sqrt(ex * ex + ey * ey)
                if dist > r_pred:
                    continue

                dt_cost = 0.0
                if dt_ticks > tau_fast_ticks:
                    dt_cost = alpha_slow * (dt_ticks - tau_fast_ticks) / max(1.0, tau_slow_ticks - tau_fast_ticks)

                vix = (float(xi) - float(tr_x_last[k])) / max(1.0, dt_ticks)
                viy = (float(yi) - float(tr_y_last[k])) / max(1.0, dt_ticks)
                vh = np.sqrt(float(tr_vx[k]) * float(tr_vx[k]) + float(tr_vy[k]) * float(tr_vy[k]))
                dv = np.sqrt((vix - float(tr_vx[k])) * (vix - float(tr_vx[k])) + (viy - float(tr_vy[k])) * (viy - float(tr_vy[k])))

                dx_n = dist / max(1.0, r_pred)
                dv_n = dv / (vh + eps)

                d = (
                    w_x * dx_n
                    + w_v * dv_n
                    + w_h * u_hot_evt
                    + w_g * g
                    - w_f * u_foot_evt
                    + w_t * dt_cost
                )

                if d < best_d:
                    best_d = d
                    best_k = k
                    best_dt = dt_ticks

            if best_k >= 0 and best_d < d_new:
                best_link = np.exp(-best_d)

            # Attach to existing track if reliable enough; otherwise spawn tentative track.
            if best_k >= 0 and best_d < d_new:
                k = best_k

                old_vx = float(tr_vx[k])
                old_vy = float(tr_vy[k])

                vix = (float(xi) - float(tr_x_last[k])) / max(1.0, best_dt)
                viy = (float(yi) - float(tr_y_last[k])) / max(1.0, best_dt)
                new_vx = (1.0 - v_ema) * old_vx + v_ema * vix
                new_vy = (1.0 - v_ema) * old_vy + v_ema * viy

                old_vmag = np.sqrt(old_vx * old_vx + old_vy * old_vy)
                dvn = np.sqrt((vix - old_vx) * (vix - old_vx) + (viy - old_vy) * (viy - old_vy)) / (old_vmag + eps)

                new_vmag = np.sqrt(new_vx * new_vx + new_vy * new_vy)
                cosang = 1.0
                if old_vmag > eps and new_vmag > eps:
                    cosang = (old_vx * new_vx + old_vy * new_vy) / (old_vmag * new_vmag + eps)
                    if cosang < -1.0:
                        cosang = -1.0
                    if cosang > 1.0:
                        cosang = 1.0
                dang = (1.0 - cosang) * 0.5

                tr_sigma_v[k] = np.float32((1.0 - sigma_ema) * float(tr_sigma_v[k]) + sigma_ema * dvn)
                tr_sigma_theta[k] = np.float32((1.0 - sigma_ema) * float(tr_sigma_theta[k]) + sigma_ema * dang)

                old_len = float(tr_len[k])
                new_len = old_len + 1.0
                if new_len > lmax:
                    new_len = lmax
                blend = 1.0 / max(1.0, new_len)

                tr_mean_hot[k] = np.float32((1.0 - blend) * float(tr_mean_hot[k]) + blend * u_hot_evt)
                tr_mean_foot[k] = np.float32((1.0 - blend) * float(tr_mean_foot[k]) + blend * u_foot_evt)
                tr_len[k] = np.float32(new_len)
                tr_last_link[k] = np.float32(best_link)

                tr_x_prev[k] = tr_x_last[k]
                tr_y_prev[k] = tr_y_last[k]
                tr_t_prev[k] = tr_t_last[k]
                tr_x_last[k] = np.float32(xi)
                tr_y_last[k] = np.float32(yi)
                tr_t_last[k] = ti
                tr_vx[k] = np.float32(new_vx)
                tr_vy[k] = np.float32(new_vy)

                st = int(tr_state[k])
                if st == 1:
                    age = float(np.uint64(ti - tr_t_birth[k]))
                    if new_len >= 2.0 and best_link >= confirm_link_min:
                        tr_state[k] = np.int8(2)

                        q_l = new_len / (new_len + max(1.0, k_len))
                        q_v = 1.0 / (1.0 + float(tr_sigma_v[k]))
                        q_th = 1.0 / (1.0 + float(tr_sigma_theta[k]))
                        q_h = 1.0 - float(tr_mean_hot[k])
                        if q_h < 0.0:
                            q_h = 0.0
                        q_f = float(tr_mean_foot[k])
                        if q_f < 0.0:
                            q_f = 0.0
                        if q_f > 1.0:
                            q_f = 1.0
                        q_g = 1.0 - beta_g * g
                        if q_g < 0.2:
                            q_g = 0.2

                        q = q_l * q_v * q_th * q_h * q_f * q_g
                        s_evt = alpha_score * q + (1.0 - alpha_score) * best_link
                        if s_evt < 0.0:
                            s_evt = 0.0
                        out[i] = np.float32(s_evt)

                        p0 = int(tr_pending0[k])
                        p1 = int(tr_pending1[k])
                        if p0 >= 0 and p0 < n:
                            out[p0] = np.float32(s_evt)
                        if p1 >= 0 and p1 < n:
                            out[p1] = np.float32(s_evt)
                        tr_pending0[k] = -1
                        tr_pending1[k] = -1
                        tr_pending_cnt[k] = np.int8(0)
                    else:
                        if age > tau_confirm_ticks:
                            # Expire unconfirmed tentative and restart from current event.
                            tr_state[k] = np.int8(1)
                            tr_t_birth[k] = ti
                            tr_x_prev[k] = np.float32(xi)
                            tr_y_prev[k] = np.float32(yi)
                            tr_t_prev[k] = ti
                            tr_x_last[k] = np.float32(xi)
                            tr_y_last[k] = np.float32(yi)
                            tr_t_last[k] = ti
                            tr_vx[k] = np.float32(0.0)
                            tr_vy[k] = np.float32(0.0)
                            tr_sigma_v[k] = np.float32(1.0)
                            tr_sigma_theta[k] = np.float32(1.0)
                            tr_len[k] = np.float32(1.0)
                            tr_mean_hot[k] = np.float32(u_hot_evt)
                            tr_mean_foot[k] = np.float32(u_foot_evt)
                            tr_last_link[k] = np.float32(0.0)
                            tr_pending0[k] = np.int32(i)
                            tr_pending1[k] = np.int32(-1)
                            tr_pending_cnt[k] = np.int8(1)
                        else:
                            cnt = int(tr_pending_cnt[k])
                            if cnt <= 0:
                                tr_pending0[k] = np.int32(i)
                                tr_pending_cnt[k] = np.int8(1)
                            elif cnt == 1:
                                tr_pending1[k] = np.int32(i)
                                tr_pending_cnt[k] = np.int8(2)
                else:
                    q_l = new_len / (new_len + max(1.0, k_len))
                    q_v = 1.0 / (1.0 + float(tr_sigma_v[k]))
                    q_th = 1.0 / (1.0 + float(tr_sigma_theta[k]))
                    q_h = 1.0 - float(tr_mean_hot[k])
                    if q_h < 0.0:
                        q_h = 0.0
                    q_f = float(tr_mean_foot[k])
                    if q_f < 0.0:
                        q_f = 0.0
                    if q_f > 1.0:
                        q_f = 1.0
                    q_g = 1.0 - beta_g * g
                    if q_g < 0.2:
                        q_g = 0.2

                    q = q_l * q_v * q_th * q_h * q_f * q_g
                    s_evt = alpha_score * q + (1.0 - alpha_score) * best_link
                    if s_evt < 0.0:
                        s_evt = 0.0
                    out[i] = np.float32(s_evt)
            else:
                slot = -1
                oldest_t = np.uint64(0)

                for k in range(max_tracks):
                    if int(tr_state[k]) == 0:
                        slot = k
                        break

                if slot < 0:
                    for k in range(max_tracks):
                        if int(tr_state[k]) == 1:
                            tk = tr_t_last[k]
                            if oldest_t == 0 or tk < oldest_t:
                                oldest_t = tk
                                slot = k

                if slot < 0:
                    for k in range(max_tracks):
                        tk = tr_t_last[k]
                        if oldest_t == 0 or tk < oldest_t:
                            oldest_t = tk
                            slot = k

                if slot >= 0:
                    tr_state[slot] = np.int8(1)
                    tr_pol[slot] = np.int8(p[i])
                    tr_x_last[slot] = np.float32(xi)
                    tr_y_last[slot] = np.float32(yi)
                    tr_t_last[slot] = ti
                    tr_x_prev[slot] = np.float32(xi)
                    tr_y_prev[slot] = np.float32(yi)
                    tr_t_prev[slot] = ti
                    tr_t_birth[slot] = ti
                    tr_vx[slot] = np.float32(0.0)
                    tr_vy[slot] = np.float32(0.0)
                    tr_sigma_v[slot] = np.float32(1.0)
                    tr_sigma_theta[slot] = np.float32(1.0)
                    tr_len[slot] = np.float32(1.0)
                    tr_mean_hot[slot] = np.float32(u_hot_evt)
                    tr_mean_foot[slot] = np.float32(u_foot_evt)
                    tr_last_link[slot] = np.float32(0.0)
                    tr_pending0[slot] = np.int32(i)
                    tr_pending1[slot] = np.int32(-1)
                    tr_pending_cnt[slot] = np.int8(1)

            inhib_self[idx] += np.float32(self_gain)
            last_inhib_update[idx] = ti
            arr[idx] = ti

        return out

    return _kernel


def score_stream_n84(
    ev,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_us: int,
    tb: TimeBase,
    scores_out: np.ndarray | None = None,
) -> np.ndarray:
    """N84-R (7.38): explicit tracklet ECSM with delayed confirmation.

    Core changes vs old n84:
    - Explicit tracklet states (tentative / confirmed / dead).
    - Predicted-position matching with adaptive radius (not fixed 8-neighbor).
    - New-chain delayed confirmation to avoid killing chain births.
    - Event score mainly from confirmed-chain quality, with small link correction.

    Env vars:
    - MYEVS_N84_TAU_FAST_US (default tau_us/8)
    - MYEVS_N84_TAU_SLOW_US (default 4*tau_fast)
    - MYEVS_N84_TAU_INHIB_US (default tau_us/4)
    - MYEVS_N84_TAU_CONFIRM_US (default tau_slow)
    - MYEVS_N84_TAU_KILL_US (default 2*tau_slow)
    - MYEVS_N84_SELF_GAIN (default 0.55)
    - MYEVS_N84_FOOT_SIGMA (default 2.2)
    - MYEVS_N84_FOOT_RHO (default 0.60)
    - MYEVS_N84_FOOT_DT_MAX_RATIO (default 2.0)
    - MYEVS_N84_C_HOT (default 1.0)
    - MYEVS_N84_C_FOOT (default 0.8)
    - MYEVS_N84_R0_PX (default 1.8)
    - MYEVS_N84_KV_PX_PER_MS (default 0.06)
    - MYEVS_N84_D_NEW (default 1.35)
    - MYEVS_N84_ALPHA_SLOW (default 0.80)
    - MYEVS_N84_W_X / W_V / W_H / W_G / W_F / W_T
    - MYEVS_N84_ALPHA_SCORE (default 0.80)
    - MYEVS_N84_CONFIRM_LINK_MIN (default 0.20)
    - MYEVS_N84_TRACK_CAPACITY (default 1024)
    - MYEVS_N84_V_EMA (default 0.35)
    - MYEVS_N84_SIGMA_EMA (default 0.15)
    - MYEVS_N84_K_LEN (default 6.0)
    - MYEVS_N84_LMAX (default 24)
    - MYEVS_N84_BETA_G (default 0.35)
    - MYEVS_N84_G_TAU_SHORT_US (default 2000)
    - MYEVS_N84_G_TAU_LONG_US (default 20000)
    """

    n = int(ev.t.shape[0])
    if scores_out is None:
        scores_out = np.empty((n,), dtype=np.float32)
    if n <= 0:
        return scores_out

    tau_fast_default = max(1000, int(tau_us) // 8)
    tau_slow_default = max(2000, tau_fast_default * 4)
    tau_inhib_default = max(1000, int(tau_us) // 4)
    tau_confirm_default = max(1000, tau_slow_default)
    tau_kill_default = max(2000, tau_slow_default * 2)

    tau_fast_us = int(_read_int_env("MYEVS_N84_TAU_FAST_US", tau_fast_default))
    tau_slow_us = int(_read_int_env("MYEVS_N84_TAU_SLOW_US", tau_slow_default))
    tau_inhib_us = int(_read_int_env("MYEVS_N84_TAU_INHIB_US", tau_inhib_default))
    tau_confirm_us = int(_read_int_env("MYEVS_N84_TAU_CONFIRM_US", tau_confirm_default))
    tau_kill_us = int(_read_int_env("MYEVS_N84_TAU_KILL_US", tau_kill_default))

    tau_fast_ticks = float(tb.us_to_ticks(max(1, tau_fast_us)))
    tau_slow_ticks = float(tb.us_to_ticks(max(1, tau_slow_us)))
    if tau_slow_ticks < tau_fast_ticks:
        tau_slow_ticks = tau_fast_ticks
    tau_inhib_ticks = float(tb.us_to_ticks(max(1, tau_inhib_us)))
    tau_confirm_ticks = float(tb.us_to_ticks(max(1, tau_confirm_us)))
    tau_kill_ticks = float(tb.us_to_ticks(max(1, tau_kill_us)))

    alpha_slow = float(_read_float_env("MYEVS_N84_ALPHA_SLOW", 0.80))
    self_gain = float(_read_float_env("MYEVS_N84_SELF_GAIN", 0.55))
    foot_sigma = float(_read_float_env("MYEVS_N84_FOOT_SIGMA", 2.2))
    foot_rho = float(_read_float_env("MYEVS_N84_FOOT_RHO", 0.60))
    foot_dt_max_ratio = float(_read_float_env("MYEVS_N84_FOOT_DT_MAX_RATIO", 2.0))
    foot_dt_max_ticks = float(max(1.0, tau_slow_ticks * max(1.0, foot_dt_max_ratio)))

    c_hot = float(_read_float_env("MYEVS_N84_C_HOT", 1.0))
    c_foot = float(_read_float_env("MYEVS_N84_C_FOOT", 0.8))

    r0_px = float(_read_float_env("MYEVS_N84_R0_PX", 1.8))
    kv_px_per_ms = float(_read_float_env("MYEVS_N84_KV_PX_PER_MS", 0.06))
    kv_px_per_tick = kv_px_per_ms / max(1e-6, float(tb.us_to_ticks(1000.0)))
    d_new = float(_read_float_env("MYEVS_N84_D_NEW", 1.35))

    w_x = float(_read_float_env("MYEVS_N84_W_X", 1.00))
    w_v = float(_read_float_env("MYEVS_N84_W_V", 0.80))
    w_h = float(_read_float_env("MYEVS_N84_W_H", 0.55))
    w_g = float(_read_float_env("MYEVS_N84_W_G", 0.35))
    w_f = float(_read_float_env("MYEVS_N84_W_F", 0.60))
    w_t = float(_read_float_env("MYEVS_N84_W_T", 0.65))

    alpha_score = float(_read_float_env("MYEVS_N84_ALPHA_SCORE", 0.80))
    confirm_link_min = float(_read_float_env("MYEVS_N84_CONFIRM_LINK_MIN", 0.20))
    track_capacity = int(_read_int_env("MYEVS_N84_TRACK_CAPACITY", 1024))
    v_ema = float(_read_float_env("MYEVS_N84_V_EMA", 0.35))
    sigma_ema = float(_read_float_env("MYEVS_N84_SIGMA_EMA", 0.15))
    k_len = float(_read_float_env("MYEVS_N84_K_LEN", 6.0))
    lmax = float(_read_float_env("MYEVS_N84_LMAX", 24.0))

    beta_g = float(_read_float_env("MYEVS_N84_BETA_G", 0.35))
    g_tau_short_us = int(_read_int_env("MYEVS_N84_G_TAU_SHORT_US", 2000))
    g_tau_long_us = int(_read_int_env("MYEVS_N84_G_TAU_LONG_US", 20000))
    g_tau_short_ticks = float(tb.us_to_ticks(max(1, g_tau_short_us)))
    g_tau_long_ticks = float(tb.us_to_ticks(max(1, g_tau_long_us)))

    ker = _try_build_n84_kernel()
    scores = ker(
        ev.t.astype(np.uint64, copy=False),
        ev.x.astype(np.int32, copy=False),
        ev.y.astype(np.int32, copy=False),
        ev.p,
        int(width),
        int(height),
        int(radius_px),
        float(max(1.0, tau_fast_ticks)),
        float(max(1.0, tau_slow_ticks)),
        float(max(1.0, tau_inhib_ticks)),
        float(max(1.0, tau_confirm_ticks)),
        float(max(1.0, tau_kill_ticks)),
        float(max(0.0, self_gain)),
        float(max(0.2, foot_sigma)),
        float(max(0.0, foot_rho)),
        float(max(1.0, foot_dt_max_ticks)),
        float(max(1e-4, c_hot)),
        float(max(1e-4, c_foot)),
        float(max(1.0, r0_px)),
        float(max(0.0, kv_px_per_tick)),
        float(max(0.1, d_new)),
        float(max(0.0, alpha_slow)),
        float(max(0.0, w_x)),
        float(max(0.0, w_v)),
        float(max(0.0, w_h)),
        float(max(0.0, w_g)),
        float(max(0.0, w_f)),
        float(max(0.0, w_t)),
        float(min(1.0, max(0.0, alpha_score))),
        float(min(1.0, max(0.0, confirm_link_min))),
        int(max(32, track_capacity)),
        float(min(1.0, max(0.01, v_ema))),
        float(min(1.0, max(0.01, sigma_ema))),
        float(max(1.0, k_len)),
        float(max(0.0, beta_g)),
        float(max(1.0, g_tau_short_ticks)),
        float(max(1.0, g_tau_long_ticks)),
        float(max(4.0, lmax)),
    )
    scores_out[:] = scores
    return scores_out
