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
        raise RuntimeError("n88 requires numba, but import failed")


def _try_build_n88_kernel():
    _require_numba()

    @numba.njit(cache=True)
    def _kernel(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        width: int,
        height: int,
        t_link_ticks: float,
        t_kill_ticks: float,
        r_link_px: float,
        w_x: float,
        w_t: float,
        track_capacity: int,
        k_len: float,
        a_l: float,
        a_p: float,
        a_n: float,
    ) -> np.ndarray:
        n = int(t.shape[0])
        out = np.zeros((n,), dtype=np.float32)

        max_tracks = int(track_capacity)
        if max_tracks < 32:
            max_tracks = 32

        tr_alive = np.zeros((max_tracks,), dtype=np.int8)
        tr_pol = np.zeros((max_tracks,), dtype=np.int8)

        tr_t_last = np.zeros((max_tracks,), dtype=np.uint64)
        tr_x_birth = np.zeros((max_tracks,), dtype=np.float32)
        tr_y_birth = np.zeros((max_tracks,), dtype=np.float32)
        tr_x_last = np.zeros((max_tracks,), dtype=np.float32)
        tr_y_last = np.zeros((max_tracks,), dtype=np.float32)
        tr_x_prev = np.zeros((max_tracks,), dtype=np.float32)
        tr_y_prev = np.zeros((max_tracks,), dtype=np.float32)

        tr_len = np.zeros((max_tracks,), dtype=np.float32)
        tr_path = np.zeros((max_tracks,), dtype=np.float32)
        tr_unique = np.zeros((max_tracks,), dtype=np.float32)

        eps = 1e-6

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            if xi < 0 or xi >= width or yi < 0 or yi >= height:
                continue

            ti = np.uint64(t[i])

            best_k = -1
            best_d = 1e12

            for k in range(max_tracks):
                if int(tr_alive[k]) == 0:
                    continue

                dt_ticks = float(np.uint64(ti - tr_t_last[k]))
                if dt_ticks <= 0.0:
                    continue
                if dt_ticks > t_kill_ticks:
                    tr_alive[k] = 0
                    continue

                if int(tr_pol[k]) != int(p[i]):
                    continue
                if dt_ticks > t_link_ticks:
                    continue

                dx = float(xi) - float(tr_x_last[k])
                dy = float(yi) - float(tr_y_last[k])
                dist = np.sqrt(dx * dx + dy * dy)
                if dist > r_link_px:
                    continue

                d_x = dist / max(1.0, r_link_px)
                d_t = dt_ticks / max(1.0, t_link_ticks)
                d = w_x * d_x + w_t * d_t

                if d < best_d:
                    best_d = d
                    best_k = k

            if best_k >= 0:
                k = best_k

                old_x_last = float(tr_x_last[k])
                old_y_last = float(tr_y_last[k])
                old_x_prev = float(tr_x_prev[k])
                old_y_prev = float(tr_y_prev[k])

                step_dx = float(xi) - old_x_last
                step_dy = float(yi) - old_y_last
                step = np.sqrt(step_dx * step_dx + step_dy * step_dy)

                new_len = float(tr_len[k]) + 1.0
                tr_len[k] = np.float32(new_len)
                tr_path[k] = np.float32(float(tr_path[k]) + step)

                moved = 1.0 if (int(round(old_x_last)) != xi or int(round(old_y_last)) != yi) else 0.0
                bounce = 1.0 if (int(round(old_x_prev)) == xi and int(round(old_y_prev)) == yi) else 0.0
                uniq_inc = moved * (1.0 - bounce)
                tr_unique[k] = np.float32(float(tr_unique[k]) + uniq_inc)

                tr_x_prev[k] = np.float32(old_x_last)
                tr_y_prev[k] = np.float32(old_y_last)
                tr_x_last[k] = np.float32(xi)
                tr_y_last[k] = np.float32(yi)
                tr_t_last[k] = ti

                q_l = new_len / (new_len + max(1.0, k_len))

                disp_x = float(tr_x_last[k]) - float(tr_x_birth[k])
                disp_y = float(tr_y_last[k]) - float(tr_y_birth[k])
                disp = np.sqrt(disp_x * disp_x + disp_y * disp_y)
                q_p = disp / (float(tr_path[k]) + eps)
                if q_p < 0.0:
                    q_p = 0.0
                if q_p > 1.0:
                    q_p = 1.0

                q_n = 0.0
                if new_len > 1.0:
                    q_n = max(0.0, float(tr_unique[k])) / max(1.0, new_len - 1.0)
                if q_n < 0.0:
                    q_n = 0.0
                if q_n > 1.0:
                    q_n = 1.0

                q = a_l * q_l + a_p * q_p + a_n * q_n
                if q < 0.0:
                    q = 0.0
                if q > 1.0:
                    q = 1.0
                out[i] = np.float32(q)
            else:
                slot = -1
                oldest_t = np.uint64(0)

                for k in range(max_tracks):
                    if int(tr_alive[k]) == 0:
                        slot = k
                        break

                if slot < 0:
                    for k in range(max_tracks):
                        tk = tr_t_last[k]
                        if oldest_t == 0 or tk < oldest_t:
                            oldest_t = tk
                            slot = k

                if slot >= 0:
                    tr_alive[slot] = np.int8(1)
                    tr_pol[slot] = np.int8(p[i])
                    tr_t_last[slot] = ti

                    tr_x_birth[slot] = np.float32(xi)
                    tr_y_birth[slot] = np.float32(yi)
                    tr_x_last[slot] = np.float32(xi)
                    tr_y_last[slot] = np.float32(yi)
                    tr_x_prev[slot] = np.float32(xi)
                    tr_y_prev[slot] = np.float32(yi)

                    tr_len[slot] = np.float32(1.0)
                    tr_path[slot] = np.float32(0.0)
                    tr_unique[slot] = np.float32(0.0)

                out[i] = np.float32(0.0)

        return out

    return _kernel


def score_stream_n88(
    ev,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_us: int,
    tb: TimeBase,
    scores_out: np.ndarray | None = None,
) -> np.ndarray:
    """N88-lite (7.41): minimal propagation-chain scoring.

    Design constraints from 7.41:
    - Only spatiotemporal chain linking (no hotness/footprint/global terms).
    - Event score is chain quality from length, propagation efficiency and novelty.
    """

    n = int(ev.t.shape[0])
    if scores_out is None:
        scores_out = np.empty((n,), dtype=np.float32)
    if n <= 0:
        return scores_out

    t_link_us = int(_read_int_env("MYEVS_N88_T_LINK_US", 16000))
    t_kill_us = int(_read_int_env("MYEVS_N88_T_KILL_US", 32000))
    r_link_px = float(_read_float_env("MYEVS_N88_R_LINK_PX", 3.0))

    t_link_ticks = float(tb.us_to_ticks(max(1, t_link_us)))
    t_kill_ticks = float(tb.us_to_ticks(max(1, t_kill_us)))

    w_x = float(_read_float_env("MYEVS_N88_W_X", 1.0))
    w_t = float(_read_float_env("MYEVS_N88_W_T", 0.5))

    track_capacity = int(_read_int_env("MYEVS_N88_TRACK_CAPACITY", 1024))

    k_len = float(_read_float_env("MYEVS_N88_K_LEN", 3.0))
    a_l = float(_read_float_env("MYEVS_N88_A_L", 0.25))
    a_p = float(_read_float_env("MYEVS_N88_A_P", 0.40))
    a_n = float(_read_float_env("MYEVS_N88_A_N", 0.35))

    ker = _try_build_n88_kernel()
    scores = ker(
        ev.t.astype(np.uint64, copy=False),
        ev.x.astype(np.int32, copy=False),
        ev.y.astype(np.int32, copy=False),
        ev.p,
        int(width),
        int(height),
        float(max(1.0, t_link_ticks)),
        float(max(1.0, t_kill_ticks)),
        float(max(1.0, r_link_px)),
        float(max(0.0, w_x)),
        float(max(0.0, w_t)),
        int(max(32, track_capacity)),
        float(max(1.0, k_len)),
        float(max(0.0, a_l)),
        float(max(0.0, a_p)),
        float(max(0.0, a_n)),
    )
    scores_out[:] = scores
    return scores_out
