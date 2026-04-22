from __future__ import annotations

import os

import numpy as np

from ....timebase import TimeBase
from ...numba_ebf import ebf_scores_stream_numba, ebf_state_init


def _sobel_contrast_std_binary(image_u8: np.ndarray) -> float:
    """Sobel-gradient-magnitude standard deviation for a binary image.

    Lightweight AOCC-lite proxy used by the block controller.
    - Input expected uint8 with values {0,1}.
    - Uses 1-pixel zero padding.
    """

    img = np.asarray(image_u8)
    if img.ndim != 2:
        raise ValueError("image_u8 must be 2D")

    f = img.astype(np.float32, copy=False)
    p = np.pad(f, pad_width=((1, 1), (1, 1)), mode="constant", constant_values=0.0)

    gx = (
        p[0:-2, 2:]
        + 2.0 * p[1:-1, 2:]
        + p[2:, 2:]
        - (p[0:-2, 0:-2] + 2.0 * p[1:-1, 0:-2] + p[2:, 0:-2])
    )
    gy = (
        p[0:-2, 0:-2]
        + 2.0 * p[0:-2, 1:-1]
        + p[0:-2, 2:]
        - (p[2:, 0:-2] + 2.0 * p[2:, 1:-1] + p[2:, 2:])
    )

    mag = np.hypot(gx, gy)
    return float(np.std(mag, dtype=np.float64))


def score_stream_s81(
    ev,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_us: int,
    tb: TimeBase,
    scores_out: np.ndarray | None = None,
) -> np.ndarray:
    """Part2 s81: block-level AOCC-lite continuity controller × baseline EBF.

    Idea
    - Baseline: standard EBF score.
    - Controller: per-block structure visibility proxy = std(|∇I|) on a binary event frame.
    - Continuity: use min over 3 consecutive blocks.
    - Normalize: EMA normalization per sequence to avoid brittle absolute scales.
        - Fusion: multiplicative factor in [0.75, 1.25] per block (block-adaptive thresholding).
            The strength can be tuned via env vars:
                - MYEVS_S81_FACTOR_MIN (default 0.75)
                - MYEVS_S81_FACTOR_MAX (default 1.25)

    This function is intentionally called by slim sweep scripts; it is not a Numba kernel.
    """

    def _read_float_env(name: str, default: float) -> float:
        try:
            v = float(os.environ.get(name, str(default)))
        except Exception:
            return float(default)
        if not bool(np.isfinite(v)):
            return float(default)
        return float(v)

    n = int(ev.t.shape[0])
    if scores_out is None:
        scores_out = np.empty((n,), dtype=np.float32)

    # 1) baseline EBF scores
    tau_ticks = int(tb.us_to_ticks(int(tau_us)))
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
        scores_out=scores_out,
    )

    if n <= 0:
        return scores_out

    # Block length: proportional to tau (short window). Keep a small floor for stability.
    block_us = max(2000, int(tau_us) // 4)
    block_ticks = int(tb.us_to_ticks(int(block_us)))
    if block_ticks <= 0:
        block_ticks = 1

    t0 = int(ev.t[0])
    win_id = ((ev.t.astype(np.int64, copy=False) - np.int64(t0)) // np.int64(block_ticks)).astype(
        np.int64, copy=False
    )

    n_blocks = int(win_id[-1]) + 1
    if n_blocks <= 1:
        return scores_out

    # Find contiguous ranges per block (win_id is non-decreasing).
    change = np.empty((n,), dtype=bool)
    change[0] = True
    change[1:] = win_id[1:] != win_id[:-1]
    starts = np.nonzero(change)[0]
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = n

    # 2) compute per-block contrast
    img = np.zeros((int(height), int(width)), dtype=np.uint8)
    c_block = np.zeros((n_blocks,), dtype=np.float32)

    w = int(width)
    h = int(height)
    for bi in range(int(starts.shape[0])):
        i0 = int(starts[bi])
        i1 = int(ends[bi])
        b = int(win_id[i0])
        if i1 <= i0 or b < 0 or b >= n_blocks:
            continue

        img.fill(0)
        xs = ev.x[i0:i1]
        ys = ev.y[i0:i1]
        ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        if not bool(np.all(ok)):
            xs = xs[ok]
            ys = ys[ok]
        if xs.size <= 0:
            c_block[b] = 0.0
            continue

        img[ys, xs] = 1
        c_block[b] = np.float32(_sobel_contrast_std_binary(img))

    # 3) continuity over 3 consecutive blocks (AOCC-style "continuous" idea)
    c_cont = np.zeros_like(c_block)
    for b in range(n_blocks):
        if b == 0:
            c_cont[b] = c_block[b]
        elif b == 1:
            c_cont[b] = np.minimum(c_block[b], c_block[b - 1])
        else:
            c_cont[b] = np.minimum(np.minimum(c_block[b], c_block[b - 1]), c_block[b - 2])

    # 4) normalize with an EMA; map to bounded factor
    alpha = 0.1
    eps = 1e-6

    factor_min = _read_float_env("MYEVS_S81_FACTOR_MIN", 0.75)
    factor_max = _read_float_env("MYEVS_S81_FACTOR_MAX", 1.25)
    if factor_max < factor_min:
        factor_min, factor_max = factor_max, factor_min
    span = float(factor_max - factor_min)

    ema = 0.0
    factor_block = np.empty((n_blocks,), dtype=np.float32)
    for b in range(n_blocks):
        c = float(max(float(c_cont[b]), 0.0))
        ema = (1.0 - alpha) * float(ema) + alpha * c
        r = c / (float(ema) + eps) if ema > 0.0 else 0.0
        gate01 = r / (r + 1.0) if r > 0.0 else 0.0
        factor_block[b] = np.float32(float(factor_min) + span * float(gate01))

    scores_out *= factor_block[win_id]
    return scores_out
