from __future__ import annotations

import os

import numpy as np

from ....denoise.numba_ebf import ebf_scores_stream_numba, ebf_state_init
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
        raise RuntimeError("n89 requires numba, but import failed")


def _try_build_n89_shift_kernel():
    _require_numba()

    @numba.njit(cache=True)
    def _kernel(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        base_scores: np.ndarray,
        width: int,
        height: int,
        block_size: int,
        t_state_ticks: float,
        t_active_ticks: float,
        lambda_shift: float,
        h_low: float,
        h_high: float,
    ) -> np.ndarray:
        n = int(t.shape[0])
        out = np.empty((n,), dtype=np.float32)

        npx = width * height
        nbx = (width + block_size - 1) // block_size
        nby = (height + block_size - 1) // block_size
        nb = nbx * nby

        blk_n = np.zeros((nb,), dtype=np.float32)
        blk_a = np.zeros((nb,), dtype=np.float32)
        blk_last_t = np.zeros((nb,), dtype=np.uint64)

        pix_last_seen = np.zeros((npx,), dtype=np.uint64)

        eps = 1e-6
        denom_h = max(eps, h_high - h_low)

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            if xi < 0 or xi >= width or yi < 0 or yi >= height:
                out[i] = np.float32(base_scores[i])
                continue

            ti = np.uint64(t[i])
            bx = xi // block_size
            by = yi // block_size
            bid = by * nbx + bx

            t_prev_b = blk_last_t[bid]
            if t_prev_b > 0:
                dtb = float(np.uint64(ti - t_prev_b))
                decay = np.exp(-dtb / max(1.0, t_state_ticks))
                blk_n[bid] = blk_n[bid] * decay
                blk_a[bid] = blk_a[bid] * decay

            blk_n[bid] += np.float32(1.0)

            pidx = yi * width + xi
            t_prev_p = pix_last_seen[pidx]
            is_new = False
            if t_prev_p == 0:
                is_new = True
            else:
                if float(np.uint64(ti - t_prev_p)) > t_active_ticks:
                    is_new = True

            if is_new:
                blk_a[bid] += np.float32(1.0)
            pix_last_seen[pidx] = ti
            blk_last_t[bid] = ti

            h = float(blk_n[bid]) / (float(blk_a[bid]) + eps)
            z = (h - h_low) / denom_h
            if z < 0.0:
                z = 0.0
            if z > 1.0:
                z = 1.0

            s = float(base_scores[i]) - lambda_shift * z
            out[i] = np.float32(s)

        return out

    return _kernel


def score_stream_n89(
    ev,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_us: int,
    tb: TimeBase,
    scores_out: np.ndarray | None = None,
) -> np.ndarray:
    """N89 (7.42): baseline score with single block-hotness threshold shift.

    Formula:
      S'(e) = S_base(e) - lambda * Z_b(t)
      Z_b(t) = clip((H_b(t)-H_low)/(H_high-H_low), 0, 1)
      H_b(t) ~ N_b/A_b with exponential-window approximation.

    This keeps baseline ranking backbone and only applies one external-state
    decision-boundary shift.
    """

    n = int(ev.t.shape[0])
    if scores_out is None:
        scores_out = np.empty((n,), dtype=np.float32)
    if n <= 0:
        return scores_out

    tau_ticks = int(tb.us_to_ticks(int(tau_us)))

    base_scores = np.empty((n,), dtype=np.float32)
    last_ts, last_pol = ebf_state_init(int(width), int(height))
    ebf_scores_stream_numba(
        t=ev.t,
        x=ev.x,
        y=ev.y,
        p=ev.p,
        width=int(width),
        height=int(height),
        radius_px=int(radius_px),
        tau_ticks=int(tau_ticks),
        last_ts=last_ts,
        last_pol=last_pol,
        scores_out=base_scores,
    )

    block_size = int(_read_int_env("MYEVS_N89_BLOCK_SIZE", 16))
    if block_size < 4:
        block_size = 4

    t_state_us = int(_read_int_env("MYEVS_N89_T_STATE_US", 25000))
    t_active_us = int(_read_int_env("MYEVS_N89_T_ACTIVE_US", 25000))
    t_state_ticks = float(tb.us_to_ticks(max(1, t_state_us)))
    t_active_ticks = float(tb.us_to_ticks(max(1, t_active_us)))

    lambda_shift = float(_read_float_env("MYEVS_N89_LAMBDA", 0.20))
    h_low = float(_read_float_env("MYEVS_N89_H_LOW", 1.20))
    h_high = float(_read_float_env("MYEVS_N89_H_HIGH", 4.00))
    if h_high <= h_low:
        h_high = h_low + 1.0

    ker = _try_build_n89_shift_kernel()
    shifted = ker(
        ev.t.astype(np.uint64, copy=False),
        ev.x.astype(np.int32, copy=False),
        ev.y.astype(np.int32, copy=False),
        base_scores,
        int(width),
        int(height),
        int(block_size),
        float(max(1.0, t_state_ticks)),
        float(max(1.0, t_active_ticks)),
        float(max(0.0, lambda_shift)),
        float(h_low),
        float(h_high),
    )

    scores_out[:] = shifted
    return scores_out
