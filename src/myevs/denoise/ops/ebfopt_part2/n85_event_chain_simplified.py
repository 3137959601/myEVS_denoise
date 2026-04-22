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
        raise RuntimeError("n85 requires numba, but import failed")


def _try_build_n85_kernel():
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
        gamma_birth: float,
        alpha_score: float,
        confirm_link_min: float,
        track_capacity: int,
        lmax: float,
        k_len: float,
        a_l: float,
        a_p: float,
        a_n: float,
        a_h: float,
        a_g: float,
        beta_g: float,
        g_tau_short_ticks: float,
        g_tau_long_ticks: float,
    ) -> np.ndarray:
        n = int(t.shape[0])
        out = np.zeros((n,), dtype=np.float32)

        npx = width * height

        # Pixel-wise self-hotness state.
        inhib_self = np.zeros((npx,), dtype=np.float32)
        last_inhib_update = np.zeros((npx,), dtype=np.uint64)

        # Same-pol recent timestamps for dual-footprint support.
        last_pos = np.zeros((npx,), dtype=np.uint64)
        last_neg = np.zeros((npx,), dtype=np.uint64)

        max_tracks = int(track_capacity)
        if max_tracks < 32:
            max_tracks = 32

        # 0 dead, 1 tentative, 2 confirmed
        tr_state = np.zeros((max_tracks,), dtype=np.int8)
        tr_pol = np.zeros((max_tracks,), dtype=np.int8)

        tr_t_birth = np.zeros((max_tracks,), dtype=np.uint64)
        tr_t_last = np.zeros((max_tracks,), dtype=np.uint64)

        tr_x_birth = np.zeros((max_tracks,), dtype=np.float32)
        tr_y_birth = np.zeros((max_tracks,), dtype=np.float32)
        tr_x_last = np.zeros((max_tracks,), dtype=np.float32)
        tr_y_last = np.zeros((max_tracks,), dtype=np.float32)
        tr_x_prev = np.zeros((max_tracks,), dtype=np.float32)
        tr_y_prev = np.zeros((max_tracks,), dtype=np.float32)
        tr_t_prev = np.zeros((max_tracks,), dtype=np.uint64)

        tr_len = np.zeros((max_tracks,), dtype=np.float32)
        tr_path = np.zeros((max_tracks,), dtype=np.float32)
        tr_novel = np.zeros((max_tracks,), dtype=np.float32)
        tr_mean_hot = np.zeros((max_tracks,), dtype=np.float32)
        tr_mean_foot = np.zeros((max_tracks,), dtype=np.float32)

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

            # Global burst proxy.
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

            # Self-hotness decay and readout at current event pixel.
            t_prev = last_inhib_update[idx]
            if t_prev > 0:
                dt_i = float(np.uint64(ti - t_prev))
                inhib_self[idx] = inhib_self[idx] * np.exp(-dt_i / max(1.0, tau_inhib_ticks))
            u_hot_evt = float(inhib_self[idx]) / (float(inhib_self[idx]) + max(eps, c_hot))

            arr = last_pos if int(p[i]) > 0 else last_neg

            # Dual-footprint support (same-pol neighborhood).
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

            # Match to local active tracklets.
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
                    continue

                dt_ticks = float(np.uint64(ti - tlast))
                if dt_ticks <= 0.0:
                    continue

                # Lifecycle pruning.
                if dt_ticks > tau_kill_ticks:
                    tr_state[k] = 0
                    continue

                if st == 1:
                    age = float(np.uint64(ti - tr_t_birth[k]))
                    if age > tau_confirm_ticks:
                        tr_state[k] = 0
                        continue

                if int(tr_pol[k]) != int(p[i]):
                    continue
                if dt_ticks > tau_slow_ticks:
                    continue

                # Predict by constant velocity from last two points when possible.
                pred_x = float(tr_x_last[k])
                pred_y = float(tr_y_last[k])
                if float(tr_len[k]) >= 2.0 and tr_t_prev[k] > 0:
                    dprev = float(np.uint64(tr_t_last[k] - tr_t_prev[k]))
                    if dprev > 0.0:
                        vx = (float(tr_x_last[k]) - float(tr_x_prev[k])) / dprev
                        vy = (float(tr_y_last[k]) - float(tr_y_prev[k])) / dprev
                        pred_x = pred_x + vx * dt_ticks
                        pred_y = pred_y + vy * dt_ticks

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

                d_x = dist / max(1.0, r_pred)
                d = w_x * d_x + w_t * d_t + w_h * u_hot_evt - w_f * u_foot_evt

                if d < best_d:
                    best_d = d
                    best_k = k

            if best_k >= 0 and best_d < d_new:
                best_link = np.exp(-best_d)

            if best_k >= 0 and best_d < d_new:
                k = best_k

                old_x_last = float(tr_x_last[k])
                old_y_last = float(tr_y_last[k])
                old_x_prev = float(tr_x_prev[k])
                old_y_prev = float(tr_y_prev[k])

                # Step update.
                step_dx = float(xi) - old_x_last
                step_dy = float(yi) - old_y_last
                step = np.sqrt(step_dx * step_dx + step_dy * step_dy)

                new_len = float(tr_len[k]) + 1.0
                if new_len > lmax:
                    new_len = lmax
                blend = 1.0 / max(1.0, new_len)

                tr_len[k] = np.float32(new_len)
                tr_path[k] = np.float32(float(tr_path[k]) + step)
                tr_mean_hot[k] = np.float32((1.0 - blend) * float(tr_mean_hot[k]) + blend * u_hot_evt)
                tr_mean_foot[k] = np.float32((1.0 - blend) * float(tr_mean_foot[k]) + blend * u_foot_evt)

                # Spatial novelty proxy: move and avoid immediate two-step bounce.
                is_moved = 1.0 if (int(round(old_x_last)) != xi or int(round(old_y_last)) != yi) else 0.0
                is_bounce = 1.0 if (int(round(old_x_prev)) == xi and int(round(old_y_prev)) == yi) else 0.0
                novel_inc = is_moved * (1.0 - is_bounce)
                tr_novel[k] = np.float32(float(tr_novel[k]) + novel_inc)

                tr_x_prev[k] = np.float32(old_x_last)
                tr_y_prev[k] = np.float32(old_y_last)
                tr_t_prev[k] = tr_t_last[k]
                tr_x_last[k] = np.float32(xi)
                tr_y_last[k] = np.float32(yi)
                tr_t_last[k] = ti

                st = int(tr_state[k])
                if st == 1:
                    if (new_len >= 2.0 and best_link >= confirm_link_min) or new_len >= 3.0:
                        tr_state[k] = np.int8(2)
                        st = 2

                if st == 2:
                    # Chain quality focuses on propagation + novelty.
                    q_l = new_len / (new_len + max(1.0, k_len))

                    disp_x = float(tr_x_last[k]) - float(tr_x_birth[k])
                    disp_y = float(tr_y_last[k]) - float(tr_y_birth[k])
                    disp = np.sqrt(disp_x * disp_x + disp_y * disp_y)
                    q_p = disp / (float(tr_path[k]) + eps)
                    if q_p < 0.0:
                        q_p = 0.0
                    if q_p > 1.0:
                        q_p = 1.0

                    q_n = float(tr_novel[k]) / max(1.0, new_len)
                    if q_n < 0.0:
                        q_n = 0.0
                    if q_n > 1.0:
                        q_n = 1.0

                    q_h = 1.0 - float(tr_mean_hot[k])
                    if q_h < 0.0:
                        q_h = 0.0
                    if q_h > 1.0:
                        q_h = 1.0

                    q_g = 1.0 - beta_g * g
                    if q_g < 0.0:
                        q_g = 0.0
                    if q_g > 1.0:
                        q_g = 1.0

                    q = a_l * q_l + a_p * q_p + a_n * q_n + a_h * q_h + a_g * q_g
                    if q < 0.0:
                        q = 0.0
                    if q > 1.0:
                        q = 1.0

                    s_evt = alpha_score * q + (1.0 - alpha_score) * best_link
                    if s_evt < 0.0:
                        s_evt = 0.0
                    out[i] = np.float32(s_evt)
                else:
                    # Tentative events keep a low but non-zero birth score.
                    s_birth = gamma_birth * u_foot_evt * (1.0 - u_hot_evt)
                    if s_birth < 0.0:
                        s_birth = 0.0
                    out[i] = np.float32(s_birth)
            else:
                # Spawn new tentative track.
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
                    tr_t_prev[slot] = ti

                    tr_x_birth[slot] = np.float32(xi)
                    tr_y_birth[slot] = np.float32(yi)
                    tr_x_last[slot] = np.float32(xi)
                    tr_y_last[slot] = np.float32(yi)
                    tr_x_prev[slot] = np.float32(xi)
                    tr_y_prev[slot] = np.float32(yi)

                    tr_len[slot] = np.float32(1.0)
                    tr_path[slot] = np.float32(0.0)
                    tr_novel[slot] = np.float32(1.0)
                    tr_mean_hot[slot] = np.float32(u_hot_evt)
                    tr_mean_foot[slot] = np.float32(u_foot_evt)

                s_birth = gamma_birth * u_foot_evt * (1.0 - u_hot_evt)
                if s_birth < 0.0:
                    s_birth = 0.0
                out[i] = np.float32(s_birth)

            # Update pixel states after scoring.
            inhib_self[idx] += np.float32(self_gain)
            last_inhib_update[idx] = ti
            arr[idx] = ti

        return out

    return _kernel


def score_stream_n85(
    ev,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_us: int,
    tb: TimeBase,
    scores_out: np.ndarray | None = None,
) -> np.ndarray:
    """N85 (7.39 ECSM-S): simplified event-chain model.

    Key design points:
    - Tentative chain events are non-zero by birth score.
    - Link cost only keeps x/t/hot/foot terms (drop velocity term).
    - Confirmed chain quality emphasizes propagation and spatial novelty.
    """

    n = int(ev.t.shape[0])
    if scores_out is None:
        scores_out = np.empty((n,), dtype=np.float32)
    if n <= 0:
        return scores_out

    tau_fast_default = max(1000, int(tau_us) // 16)
    tau_slow_default = max(2000, tau_fast_default * 4)
    tau_inhib_default = max(1000, int(tau_us) // 4)

    tau_fast_us = int(_read_int_env("MYEVS_N85_TF_US", tau_fast_default))
    tau_slow_us = int(_read_int_env("MYEVS_N85_TS_US", tau_slow_default))
    tau_inhib_us = int(_read_int_env("MYEVS_N85_TAU_INHIB_US", tau_inhib_default))

    tau_confirm_us = int(_read_int_env("MYEVS_N85_T_CONFIRM_US", tau_slow_us))
    tau_kill_us = int(_read_int_env("MYEVS_N85_T_KILL_US", max(2000, 2 * tau_slow_us)))

    tau_fast_ticks = float(tb.us_to_ticks(max(1, tau_fast_us)))
    tau_slow_ticks = float(tb.us_to_ticks(max(1, tau_slow_us)))
    if tau_slow_ticks < tau_fast_ticks:
        tau_slow_ticks = tau_fast_ticks

    tau_inhib_ticks = float(tb.us_to_ticks(max(1, tau_inhib_us)))
    tau_confirm_ticks = float(tb.us_to_ticks(max(1, tau_confirm_us)))
    tau_kill_ticks = float(tb.us_to_ticks(max(1, tau_kill_us)))

    self_gain = float(_read_float_env("MYEVS_N85_SELF_GAIN", 0.55))

    foot_sigma = float(_read_float_env("MYEVS_N85_FOOT_SIGMA", 2.2))
    foot_rho = float(_read_float_env("MYEVS_N85_FOOT_RHO", 0.60))
    foot_dt_max_ratio = float(_read_float_env("MYEVS_N85_FOOT_DT_MAX_RATIO", 2.0))
    foot_dt_max_ticks = float(max(1.0, tau_slow_ticks * max(1.0, foot_dt_max_ratio)))

    c_hot = float(_read_float_env("MYEVS_N85_C_HOT", 1.0))
    c_foot = float(_read_float_env("MYEVS_N85_C_FOOT", 0.8))

    r0_px = float(_read_float_env("MYEVS_N85_R0_PX", 2.0))
    kv_px_per_ms = float(_read_float_env("MYEVS_N85_KV_PX_PER_MS", 0.10))
    kv_px_per_tick = kv_px_per_ms / max(1e-6, float(tb.us_to_ticks(1000.0)))

    d_new = float(_read_float_env("MYEVS_N85_D_NEW", 1.2))
    alpha_slow = float(_read_float_env("MYEVS_N85_ALPHA_SLOW", 0.80))

    w_x = float(_read_float_env("MYEVS_N85_W_X", 1.0))
    w_t = float(_read_float_env("MYEVS_N85_W_T", 0.6))
    w_h = float(_read_float_env("MYEVS_N85_W_H", 0.5))
    w_f = float(_read_float_env("MYEVS_N85_W_F", 0.7))

    gamma_birth = float(_read_float_env("MYEVS_N85_GAMMA_BIRTH", 0.35))
    alpha_score = float(_read_float_env("MYEVS_N85_ALPHA_SCORE", 0.75))
    confirm_link_min = float(_read_float_env("MYEVS_N85_CONFIRM_LINK_MIN", 0.25))

    track_capacity = int(_read_int_env("MYEVS_N85_TRACK_CAPACITY", 1024))
    lmax = float(_read_float_env("MYEVS_N85_LMAX", 48.0))
    k_len = float(_read_float_env("MYEVS_N85_K_LEN", 3.0))

    a_l = float(_read_float_env("MYEVS_N85_A_L", 0.20))
    a_p = float(_read_float_env("MYEVS_N85_A_P", 0.30))
    a_n = float(_read_float_env("MYEVS_N85_A_N", 0.25))
    a_h = float(_read_float_env("MYEVS_N85_A_H", 0.20))
    a_g = float(_read_float_env("MYEVS_N85_A_G", 0.05))

    beta_g = float(_read_float_env("MYEVS_N85_BETA_G", 0.20))
    g_tau_short_us = int(_read_int_env("MYEVS_N85_G_TAU_SHORT_US", 2000))
    g_tau_long_us = int(_read_int_env("MYEVS_N85_G_TAU_LONG_US", 20000))
    g_tau_short_ticks = float(tb.us_to_ticks(max(1, g_tau_short_us)))
    g_tau_long_ticks = float(tb.us_to_ticks(max(1, g_tau_long_us)))

    ker = _try_build_n85_kernel()
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
        float(max(0.0, gamma_birth)),
        float(min(1.0, max(0.0, alpha_score))),
        float(min(1.0, max(0.0, confirm_link_min))),
        int(max(32, track_capacity)),
        float(max(4.0, lmax)),
        float(max(1.0, k_len)),
        float(max(0.0, a_l)),
        float(max(0.0, a_p)),
        float(max(0.0, a_n)),
        float(max(0.0, a_h)),
        float(max(0.0, a_g)),
        float(max(0.0, beta_g)),
        float(max(1.0, g_tau_short_ticks)),
        float(max(1.0, g_tau_long_ticks)),
    )
    scores_out[:] = scores
    return scores_out
