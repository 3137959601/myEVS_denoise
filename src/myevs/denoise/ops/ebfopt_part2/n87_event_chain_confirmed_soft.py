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
        raise RuntimeError("n87 requires numba, but import failed")


def _try_build_n87_kernel():
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
        w_t: float,
        w_h: float,
        w_f: float,
        track_capacity: int,
        track_buf_max: int,
        confirm_len_min: int,
        confirm_disp_min: float,
        confirm_hot_max: float,
        k_len: float,
        q_l_w: float,
        q_p_w: float,
        q_h_w: float,
        q_f_w: float,
        q_min: float,
        q_max: float,
        link_blend: float,
        beta_g: float,
        g_tau_short_ticks: float,
        g_tau_long_ticks: float,
        confirm_len_add_max: int,
        confirm_disp_scale_max: float,
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

        buf_max = int(track_buf_max)
        if buf_max < 8:
            buf_max = 8

        # 0 dead, 1 tentative, 2 confirmed
        tr_state = np.zeros((max_tracks,), dtype=np.int8)
        tr_pol = np.zeros((max_tracks,), dtype=np.int8)

        tr_t_birth = np.zeros((max_tracks,), dtype=np.uint64)
        tr_t_last = np.zeros((max_tracks,), dtype=np.uint64)

        tr_x_birth = np.zeros((max_tracks,), dtype=np.float32)
        tr_y_birth = np.zeros((max_tracks,), dtype=np.float32)
        tr_x_last = np.zeros((max_tracks,), dtype=np.float32)
        tr_y_last = np.zeros((max_tracks,), dtype=np.float32)

        tr_len = np.zeros((max_tracks,), dtype=np.float32)
        tr_path = np.zeros((max_tracks,), dtype=np.float32)
        tr_mean_hot = np.zeros((max_tracks,), dtype=np.float32)
        tr_mean_foot = np.zeros((max_tracks,), dtype=np.float32)

        tr_buf_count = np.zeros((max_tracks,), dtype=np.int32)
        tr_buf = np.full((max_tracks * buf_max,), -1, dtype=np.int32)
        tr_buf_link = np.zeros((max_tracks * buf_max,), dtype=np.float32)

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

            best_k = -1
            best_d = 1e12
            best_link = 0.0

            for k in range(max_tracks):
                st = int(tr_state[k])
                if st == 0:
                    continue

                tlast = tr_t_last[k]
                if tlast == 0:
                    tr_state[k] = 0
                    tr_buf_count[k] = 0
                    continue

                dt_ticks = float(np.uint64(ti - tlast))
                if dt_ticks <= 0.0:
                    continue

                if dt_ticks > tau_kill_ticks:
                    tr_state[k] = 0
                    tr_buf_count[k] = 0
                    continue

                if st == 1:
                    age = float(np.uint64(ti - tr_t_birth[k]))
                    if age > tau_confirm_ticks:
                        tr_state[k] = 0
                        tr_buf_count[k] = 0
                        continue

                if int(tr_pol[k]) != int(p[i]):
                    continue
                if dt_ticks > tau_slow_ticks:
                    continue

                pred_x = float(tr_x_last[k])
                pred_y = float(tr_y_last[k])
                r_pred = r0_px + kv_px_per_tick * dt_ticks
                if r_pred < 1.0:
                    r_pred = 1.0

                ex = float(xi) - pred_x
                ey = float(yi) - pred_y
                dist = np.sqrt(ex * ex + ey * ey)
                if dist > r_pred:
                    continue

                d_t = 0.0
                if dt_ticks > tau_fast_ticks:
                    d_t = alpha_slow * (dt_ticks - tau_fast_ticks) / max(1.0, tau_slow_ticks - tau_fast_ticks)
                    if d_t > 1.0:
                        d_t = 1.0

                d_x = dist / max(1.0, r_pred)
                d = w_x * d_x + w_t * d_t + w_h * u_hot_evt - w_f * u_foot_evt

                if d < best_d:
                    best_d = d
                    best_k = k

            if best_k >= 0 and best_d < d_new:
                best_link = np.exp(-best_d)
                k = best_k

                old_x_last = float(tr_x_last[k])
                old_y_last = float(tr_y_last[k])

                step_dx = float(xi) - old_x_last
                step_dy = float(yi) - old_y_last
                step = np.sqrt(step_dx * step_dx + step_dy * step_dy)

                new_len = float(tr_len[k]) + 1.0
                blend = 1.0 / max(1.0, new_len)

                tr_len[k] = np.float32(new_len)
                tr_path[k] = np.float32(float(tr_path[k]) + step)
                tr_mean_hot[k] = np.float32((1.0 - blend) * float(tr_mean_hot[k]) + blend * u_hot_evt)
                tr_mean_foot[k] = np.float32((1.0 - blend) * float(tr_mean_foot[k]) + blend * u_foot_evt)

                tr_x_last[k] = np.float32(xi)
                tr_y_last[k] = np.float32(yi)
                tr_t_last[k] = ti

                st = int(tr_state[k])
                if st == 1:
                    c = int(tr_buf_count[k])
                    base = k * buf_max
                    if c < buf_max:
                        tr_buf[base + c] = i
                        tr_buf_link[base + c] = np.float32(best_link)
                        tr_buf_count[k] = c + 1
                    else:
                        for b in range(1, buf_max):
                            tr_buf[base + b - 1] = tr_buf[base + b]
                            tr_buf_link[base + b - 1] = tr_buf_link[base + b]
                        tr_buf[base + buf_max - 1] = i
                        tr_buf_link[base + buf_max - 1] = np.float32(best_link)

                    disp_x = float(tr_x_last[k]) - float(tr_x_birth[k])
                    disp_y = float(tr_y_last[k]) - float(tr_y_birth[k])
                    disp = np.sqrt(disp_x * disp_x + disp_y * disp_y)

                    cond_hot = True
                    if confirm_hot_max < 1.0:
                        cond_hot = float(tr_mean_hot[k]) <= confirm_hot_max

                    len_eff = int(confirm_len_min) + int(round(float(confirm_len_add_max) * beta_g * g))
                    if len_eff < 2:
                        len_eff = 2
                    disp_eff = confirm_disp_min * (1.0 + max(0.0, confirm_disp_scale_max) * beta_g * g)

                    if int(new_len) >= len_eff and disp >= disp_eff and cond_hot:
                        tr_state[k] = np.int8(2)

                        q_l = new_len / (new_len + max(1.0, k_len))
                        q_p = disp / (float(tr_path[k]) + eps)
                        if q_p < 0.0:
                            q_p = 0.0
                        if q_p > 1.0:
                            q_p = 1.0
                        q_h = 1.0 - float(tr_mean_hot[k])
                        if q_h < 0.0:
                            q_h = 0.0
                        if q_h > 1.0:
                            q_h = 1.0
                        q_f = float(tr_mean_foot[k])
                        if q_f < 0.0:
                            q_f = 0.0
                        if q_f > 1.0:
                            q_f = 1.0

                        q = q_l_w * q_l + q_p_w * q_p + q_h_w * q_h + q_f_w * q_f
                        if q < q_min:
                            q = q_min
                        if q > q_max:
                            q = q_max
                        if q < 0.0:
                            q = 0.0
                        if q > 1.0:
                            q = 1.0

                        c2 = int(tr_buf_count[k])
                        for b in range(c2):
                            ei = tr_buf[base + b]
                            if ei >= 0 and ei < n:
                                lq = float(tr_buf_link[base + b])
                                s = (1.0 - link_blend) * q + link_blend * lq
                                if s < 0.0:
                                    s = 0.0
                                if s > 1.0:
                                    s = 1.0
                                if s > float(out[ei]):
                                    out[ei] = np.float32(s)
                        tr_buf_count[k] = 0

                else:
                    disp_x = float(tr_x_last[k]) - float(tr_x_birth[k])
                    disp_y = float(tr_y_last[k]) - float(tr_y_birth[k])
                    disp = np.sqrt(disp_x * disp_x + disp_y * disp_y)

                    q_l = new_len / (new_len + max(1.0, k_len))
                    q_p = disp / (float(tr_path[k]) + eps)
                    if q_p < 0.0:
                        q_p = 0.0
                    if q_p > 1.0:
                        q_p = 1.0
                    q_h = 1.0 - float(tr_mean_hot[k])
                    if q_h < 0.0:
                        q_h = 0.0
                    if q_h > 1.0:
                        q_h = 1.0
                    q_f = float(tr_mean_foot[k])
                    if q_f < 0.0:
                        q_f = 0.0
                    if q_f > 1.0:
                        q_f = 1.0

                    q = q_l_w * q_l + q_p_w * q_p + q_h_w * q_h + q_f_w * q_f
                    if q < q_min:
                        q = q_min
                    if q > q_max:
                        q = q_max
                    if q < 0.0:
                        q = 0.0
                    if q > 1.0:
                        q = 1.0
                    s = (1.0 - link_blend) * q + link_blend * best_link
                    if s < 0.0:
                        s = 0.0
                    if s > 1.0:
                        s = 1.0
                    out[i] = np.float32(s)

            else:
                slot = -1
                oldest_t = np.uint64(0)

                for k in range(max_tracks):
                    if int(tr_state[k]) == 0:
                        slot = k
                        break

                if slot < 0:
                    for k in range(max_tracks):
                        tk = tr_t_last[k]
                        if oldest_t == 0 or tk < oldest_t:
                            oldest_t = tk
                            slot = k

                if slot >= 0:
                    tr_state[slot] = np.int8(1)
                    tr_pol[slot] = np.int8(p[i])

                    tr_t_birth[slot] = ti
                    tr_t_last[slot] = ti

                    tr_x_birth[slot] = np.float32(xi)
                    tr_y_birth[slot] = np.float32(yi)
                    tr_x_last[slot] = np.float32(xi)
                    tr_y_last[slot] = np.float32(yi)

                    tr_len[slot] = np.float32(1.0)
                    tr_path[slot] = np.float32(0.0)
                    tr_mean_hot[slot] = np.float32(u_hot_evt)
                    tr_mean_foot[slot] = np.float32(u_foot_evt)

                    base = slot * buf_max
                    for b in range(buf_max):
                        tr_buf[base + b] = -1
                        tr_buf_link[base + b] = 0.0
                    tr_buf[base] = i
                    tr_buf_link[base] = np.float32(1.0)
                    tr_buf_count[slot] = 1

            inhib_self[idx] += np.float32(self_gain)
            last_inhib_update[idx] = ti
            arr[idx] = ti

        return out

    return _kernel


def score_stream_n87(
    ev,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_us: int,
    tb: TimeBase,
    scores_out: np.ndarray | None = None,
) -> np.ndarray:
    """N87 (7.40+): soft-confirmed chain model.

    Key design points:
    - Tentative chain events keep score 0 and do not participate in ranking.
    - A chain is confirmed by minimal length and minimal propagation distance.
    - Once confirmed, cached chain events are released with chain quality score.
    """

    n = int(ev.t.shape[0])
    if scores_out is None:
        scores_out = np.empty((n,), dtype=np.float32)
    if n <= 0:
        return scores_out

    tau_fast_default = max(1000, int(tau_us) // 16)
    tau_slow_default = max(2000, tau_fast_default * 4)
    tau_inhib_default = max(1000, int(tau_us) // 4)

    tau_fast_us = int(_read_int_env("MYEVS_N87_TF_US", tau_fast_default))
    tau_slow_us = int(_read_int_env("MYEVS_N87_TS_US", tau_slow_default))
    tau_inhib_us = int(_read_int_env("MYEVS_N87_TAU_INHIB_US", tau_inhib_default))

    tau_confirm_us = int(_read_int_env("MYEVS_N87_T_CONFIRM_US", tau_slow_us))
    tau_kill_us = int(_read_int_env("MYEVS_N87_T_KILL_US", max(2000, 2 * tau_slow_us)))

    tau_fast_ticks = float(tb.us_to_ticks(max(1, tau_fast_us)))
    tau_slow_ticks = float(tb.us_to_ticks(max(1, tau_slow_us)))
    if tau_slow_ticks < tau_fast_ticks:
        tau_slow_ticks = tau_fast_ticks

    tau_inhib_ticks = float(tb.us_to_ticks(max(1, tau_inhib_us)))
    tau_confirm_ticks = float(tb.us_to_ticks(max(1, tau_confirm_us)))
    tau_kill_ticks = float(tb.us_to_ticks(max(1, tau_kill_us)))

    self_gain = float(_read_float_env("MYEVS_N87_SELF_GAIN", 0.55))

    foot_sigma = float(_read_float_env("MYEVS_N87_FOOT_SIGMA", 2.2))
    foot_rho = float(_read_float_env("MYEVS_N87_FOOT_RHO", 0.60))
    foot_dt_max_ratio = float(_read_float_env("MYEVS_N87_FOOT_DT_MAX_RATIO", 2.0))
    foot_dt_max_ticks = float(max(1.0, tau_slow_ticks * max(1.0, foot_dt_max_ratio)))

    c_hot = float(_read_float_env("MYEVS_N87_C_HOT", 1.0))
    c_foot = float(_read_float_env("MYEVS_N87_C_FOOT", 0.8))

    r0_px = float(_read_float_env("MYEVS_N87_R0_PX", 2.0))
    kv_px_per_ms = float(_read_float_env("MYEVS_N87_KV_PX_PER_MS", 0.10))
    kv_px_per_tick = kv_px_per_ms / max(1e-6, float(tb.us_to_ticks(1000.0)))

    d_new = float(_read_float_env("MYEVS_N87_D_NEW", 1.2))
    alpha_slow = float(_read_float_env("MYEVS_N87_ALPHA_SLOW", 0.80))

    w_x = float(_read_float_env("MYEVS_N87_W_X", 1.0))
    w_t = float(_read_float_env("MYEVS_N87_W_T", 0.6))
    w_h = float(_read_float_env("MYEVS_N87_W_H", 0.5))
    w_f = float(_read_float_env("MYEVS_N87_W_F", 0.7))

    track_capacity = int(_read_int_env("MYEVS_N87_TRACK_CAPACITY", 1024))
    track_buf_max = int(_read_int_env("MYEVS_N87_TRACK_BUF_MAX", 32))

    confirm_len_min = int(_read_int_env("MYEVS_N87_CONFIRM_LEN_MIN", 3))
    confirm_disp_min = float(_read_float_env("MYEVS_N87_CONFIRM_DISP_MIN", 2.0))
    confirm_hot_max = float(_read_float_env("MYEVS_N87_CONFIRM_HOT_MAX", 1.0))
    confirm_len_add_max = int(_read_int_env("MYEVS_N87_CONFIRM_LEN_ADD_MAX", 1))
    confirm_disp_scale_max = float(_read_float_env("MYEVS_N87_CONFIRM_DISP_SCALE_MAX", 0.6))

    k_len = float(_read_float_env("MYEVS_N87_K_LEN", 3.0))
    q_l_w = float(_read_float_env("MYEVS_N87_Q_L_W", 0.25))
    q_p_w = float(_read_float_env("MYEVS_N87_Q_P_W", 0.40))
    q_h_w = float(_read_float_env("MYEVS_N87_Q_H_W", 0.20))
    q_f_w = float(_read_float_env("MYEVS_N87_Q_F_W", 0.15))
    q_min = float(_read_float_env("MYEVS_N87_Q_MIN", 0.0))
    q_max = float(_read_float_env("MYEVS_N87_Q_MAX", 1.0))
    link_blend = float(_read_float_env("MYEVS_N87_LINK_BLEND", 0.35))

    beta_g = float(_read_float_env("MYEVS_N87_BETA_G", 0.8))
    g_tau_short_us = int(_read_int_env("MYEVS_N87_G_TAU_SHORT_US", 2000))
    g_tau_long_us = int(_read_int_env("MYEVS_N87_G_TAU_LONG_US", 20000))
    g_tau_short_ticks = float(tb.us_to_ticks(max(1, g_tau_short_us)))
    g_tau_long_ticks = float(tb.us_to_ticks(max(1, g_tau_long_us)))

    ker = _try_build_n87_kernel()
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
        float(max(0.0, w_t)),
        float(max(0.0, w_h)),
        float(max(0.0, w_f)),
        int(max(32, track_capacity)),
        int(max(8, track_buf_max)),
        int(max(2, confirm_len_min)),
        float(max(0.0, confirm_disp_min)),
        float(max(0.0, min(1.0, confirm_hot_max))),
        float(max(1.0, k_len)),
        float(max(0.0, q_l_w)),
        float(max(0.0, q_p_w)),
        float(max(0.0, q_h_w)),
        float(max(0.0, q_f_w)),
        float(max(0.0, min(1.0, q_min))),
        float(max(0.0, min(1.0, q_max))),
        float(max(0.0, min(1.0, link_blend))),
        float(max(0.0, beta_g)),
        float(max(1.0, g_tau_short_ticks)),
        float(max(1.0, g_tau_long_ticks)),
        int(max(0, confirm_len_add_max)),
        float(max(0.0, confirm_disp_scale_max)),
    )
    scores_out[:] = scores
    return scores_out
