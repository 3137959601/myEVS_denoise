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
        raise RuntimeError("N1 requires numba, but import failed")


def _try_build_chain_indicator_kernel():
    _require_numba()

    @numba.njit(cache=True)
    def _chain_indicator(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        width: int,
        height: int,
        radius_px: int,
        dt_max_ticks: int,
        speed_ratio_sq: float,
    ) -> np.ndarray:
        n = int(t.shape[0])
        out = np.zeros((n,), dtype=np.float32)

        last_pos = np.zeros((width * height,), dtype=np.uint64)
        last_neg = np.zeros((width * height,), dtype=np.uint64)

        r = int(radius_px)
        r2 = int(r * r)

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            if xi < 0 or xi >= width or yi < 0 or yi >= height:
                continue

            ti = np.uint64(t[i])

            arr = last_pos if int(p[i]) > 0 else last_neg

            best_t1 = np.uint64(0)
            best_t2 = np.uint64(0)
            best_idx1 = -1
            best_idx2 = -1

            for dy in range(-r, r + 1):
                yy = yi + dy
                if yy < 0 or yy >= height:
                    continue
                for dx in range(-r, r + 1):
                    if (dx * dx + dy * dy) > r2:
                        continue
                    xx = xi + dx
                    if xx < 0 or xx >= width:
                        continue
                    idx = yy * width + xx
                    tj = arr[idx]
                    if tj == 0:
                        continue

                    if tj > best_t1:
                        best_t2 = best_t1
                        best_idx2 = best_idx1
                        best_t1 = tj
                        best_idx1 = idx
                    elif tj > best_t2 and idx != best_idx1:
                        best_t2 = tj
                        best_idx2 = idx

            li = np.float32(0.0)

            # 1-step continuation (weak evidence): i has a recent same-pol neighbor j.
            if best_idx1 != -1:
                dt1 = np.uint64(ti - best_t1)
                if dt1 > 0 and dt1 <= np.uint64(dt_max_ticks):
                    x1 = int(best_idx1 % width)
                    y1 = int(best_idx1 // width)
                    d1x = xi - x1
                    d1y = yi - y1
                    d1sq = int(d1x * d1x + d1y * d1y)
                    if d1sq > 0:
                        li = np.float32(0.2)

            # 2-step chain (stronger evidence): k -> j -> i with weak speed consistency.
            if li > 0.0 and best_idx2 != -1:
                dt1 = np.uint64(ti - best_t1)
                dt2 = np.uint64(best_t1 - best_t2)

                if dt2 > 0 and dt2 <= np.uint64(dt_max_ticks):
                    x1 = int(best_idx1 % width)
                    y1 = int(best_idx1 // width)
                    x2 = int(best_idx2 % width)
                    y2 = int(best_idx2 // width)

                    d1x = xi - x1
                    d1y = yi - y1
                    d2x = x1 - x2
                    d2y = y1 - y2
                    d1sq = int(d1x * d1x + d1y * d1y)
                    d2sq = int(d2x * d2x + d2y * d2y)

                    # Reject degenerate chains (hotpixel-like) and enforce k within r of j.
                    if d1sq > 0 and d2sq > 0 and d2sq <= r2:
                        a = float(d1sq) * float(dt2) * float(dt2)
                        b = float(d2sq) * float(dt1) * float(dt1)
                        if a > 0.0 and b > 0.0 and (a <= speed_ratio_sq * b) and (b <= speed_ratio_sq * a):
                            li = np.float32(1.0)

            out[i] = li

            arr[yi * width + xi] = ti

        return out

    return _chain_indicator


def score_stream_n1(
    ev,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_us: int,
    tb: TimeBase,
    scores_out: np.ndarray | None = None,
) -> np.ndarray:
    """N1 (per 7.31): block confidence × event local chain consistency.

    This addresses the known failure mode of the earlier N1: giving every event
    in a block the same score lets noise "hitch a ride" (precision collapses).

    Dual layer:
    - C_b: block confidence from short-window statistics (context only)
    - L_i: per-event minimal two-step chain consistency (local evidence)
    Final Score_i = C_{b(i)}(prev_window) * L_i

    Env vars (minimal)
    - MYEVS_N1_BLOCK_PX (default 32)
    - MYEVS_N1_WIN_US (0=auto, default auto=max(20000, tau/4))
    - MYEVS_N1_CHAIN_DT_MAX_US (0=auto, default=win_us)
    - MYEVS_N1_CHAIN_SPEED_RATIO (default 2.0)
    """

    n = int(ev.t.shape[0])
    if scores_out is None:
        scores_out = np.empty((n,), dtype=np.float32)
    if n <= 0:
        return scores_out

    w = int(width)
    h = int(height)

    # Spatial blocks
    block_px = int(_read_int_env("MYEVS_N1_BLOCK_PX", 32))
    if block_px <= 4:
        block_px = 4
    nbx = int((w + block_px - 1) // block_px)
    nby = int((h + block_px - 1) // block_px)
    n_blocks = int(nbx * nby)
    block_area = float(block_px * block_px)

    # Windowing
    win_us_env = _read_int_env("MYEVS_N1_WIN_US", 0)
    if win_us_env > 0:
        win_us = int(win_us_env)
    else:
        win_us = max(20000, int(tau_us) // 4)
    win_ticks = int(tb.us_to_ticks(int(win_us)))
    if win_ticks <= 0:
        win_ticks = 1

    t0 = int(ev.t[0])

    # Heuristic thresholds (kept internal for now)
    quiet_cnt_thr = 6
    burst_cnt_thr = 80
    act_frac_thr = 0.04
    act_frac_low = 0.02
    hotness_thr = 6.0
    pol_bias_thr = 0.98

    # Assign each event a (win_id, block_id)
    t_i64 = ev.t.astype(np.int64, copy=False)
    win_id = ((t_i64 - np.int64(t0)) // np.int64(win_ticks)).astype(np.int32, copy=False)
    n_win = int(win_id[-1]) + 1
    if n_win <= 1:
        scores_out.fill(np.float32(0.0))
        return scores_out

    x_i = ev.x.astype(np.int32, copy=False)
    y_i = ev.y.astype(np.int32, copy=False)
    ok = (x_i >= 0) & (x_i < w) & (y_i >= 0) & (y_i < h)

    bx = (x_i // int(block_px)).astype(np.int32, copy=False)
    by = (y_i // int(block_px)).astype(np.int32, copy=False)
    bid = (by * int(nbx) + bx).astype(np.int32, copy=False)
    bid = np.where(ok, bid, -1).astype(np.int32, copy=False)

    # Per-window block statistics
    cnt = np.zeros((n_win, n_blocks), dtype=np.int32)
    pol_sum = np.zeros((n_win, n_blocks), dtype=np.int32)
    act = np.zeros((n_win, n_blocks), dtype=np.int32)
    last_seen_win = np.full((h * w,), -1, dtype=np.int32)

    for i in range(n):
        b = int(bid[i])
        if b < 0:
            continue
        wi = int(win_id[i])

        cnt[wi, b] += 1
        pol_sum[wi, b] += 1 if int(ev.p[i]) > 0 else -1

        pix = int(y_i[i] * w + x_i[i])
        if last_seen_win[pix] != wi:
            last_seen_win[pix] = wi
            act[wi, b] += 1

    # Compute block confidence per window (C_b in [0,1])
    conf = np.zeros((n_win, n_blocks), dtype=np.float32)
    struct_run = np.zeros((n_blocks,), dtype=np.uint8)

    f_act = np.float32(act_frac_thr)
    f_hot = np.float32(hotness_thr)
    f_pol = np.float32(pol_bias_thr)

    for wi in range(n_win):
        c = cnt[wi].astype(np.float32, copy=False)
        a = act[wi].astype(np.float32, copy=False)
        n_nonzero = np.maximum(c, 1.0)
        act_frac = a / np.float32(block_area)
        hotness = c / np.maximum(a, 1.0)
        pol_bias = np.abs(pol_sum[wi].astype(np.float32, copy=False)) / n_nonzero

        is_quiet = c < np.float32(quiet_cnt_thr)
        is_burst = (c >= np.float32(burst_cnt_thr)) & ((hotness >= f_hot) | (act_frac <= np.float32(act_frac_low)))

        s_act = np.clip(act_frac / (f_act + 1e-6), 0.0, 1.0)
        s_hot = np.clip((f_hot - hotness) / (f_hot + 1e-6), 0.0, 1.0)
        s_pol = np.clip((f_pol - pol_bias) / (f_pol + 1e-6), 0.0, 1.0)
        structure = (s_act * s_hot * s_pol).astype(np.float32, copy=False)

        # Persistence: only count truly structural windows.
        is_struct = structure >= np.float32(0.60)
        struct_run[is_struct] = np.minimum(struct_run[is_struct] + 1, 255).astype(np.uint8, copy=False)
        struct_run[~is_struct] = 0

        persist = (np.minimum(struct_run.astype(np.float32, copy=False), 3.0) / 3.0).astype(np.float32, copy=False)
        conf_w = structure * (np.float32(0.70) + np.float32(0.30) * persist)

        # Quiet blocks are downweighted, burst blocks suppressed.
        conf_w[is_quiet] *= np.float32(0.50)
        conf_w[is_burst] = np.float32(0.0)

        conf[wi] = np.clip(conf_w, 0.0, 1.0).astype(np.float32, copy=False)

    # One-window latency: use previous window confidence.
    wi_prev = np.maximum(win_id - 1, 0)
    conf_per_event = np.zeros((n,), dtype=np.float32)
    for i in range(n):
        b = int(bid[i])
        if b < 0:
            continue
        conf_per_event[i] = conf[int(wi_prev[i]), b]

    # Event-level minimal local consistency (two-step chain)
    chain_dt_max_us = int(_read_int_env("MYEVS_N1_CHAIN_DT_MAX_US", 0))
    if chain_dt_max_us > 0:
        dt_max_ticks = int(tb.us_to_ticks(int(chain_dt_max_us)))
    else:
        dt_max_ticks = int(win_ticks)
    if dt_max_ticks <= 0:
        dt_max_ticks = 1

    speed_ratio = float(_read_float_env("MYEVS_N1_CHAIN_SPEED_RATIO", 2.0))
    if speed_ratio < 1.0:
        speed_ratio = 1.0
    speed_ratio_sq = float(speed_ratio * speed_ratio)

    ker = _try_build_chain_indicator_kernel()
    li = ker(
        ev.t.astype(np.uint64, copy=False),
        ev.x.astype(np.int32, copy=False),
        ev.y.astype(np.int32, copy=False),
        ev.p,
        int(w),
        int(h),
        int(radius_px),
        int(dt_max_ticks),
        float(speed_ratio_sq),
    )

    scores_out[:] = conf_per_event * li
    return scores_out
