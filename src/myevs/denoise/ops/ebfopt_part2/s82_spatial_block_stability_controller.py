from __future__ import annotations

import os

import numpy as np

from ....timebase import TimeBase
from ...numba_ebf import ebf_scores_stream_numba, ebf_state_init


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


def _block_edges_total_variation(img_u8: np.ndarray) -> np.ndarray:
    """Compute a cheap structure proxy: total variation edge-count per pixel.

    For a binary frame I in {0,1}, we compute
      tv = |I[x+1]-I[x]| + |I[y+1]-I[y]|
    aggregated into an edge map (uint16/int32), then sum within spatial blocks.

    This is much cheaper than Sobel and is stable for binary frames.
    """

    img = np.asarray(img_u8)
    if img.ndim != 2:
        raise ValueError("img_u8 must be 2D")

    # Use int16 to avoid uint8 underflow.
    a = img.astype(np.int16, copy=False)

    # Horizontal and vertical differences.
    dx = np.abs(a[:, 1:] - a[:, :-1])
    dy = np.abs(a[1:, :] - a[:-1, :])

    # Accumulate into a per-pixel edge map.
    edge = np.zeros_like(a, dtype=np.int32)
    edge[:, 1:] += dx
    edge[1:, :] += dy
    return edge


def score_stream_s82(
    ev,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_us: int,
    tb: TimeBase,
    scores_out: np.ndarray | None = None,
) -> np.ndarray:
    """Part2 s82: spatial-block 3-window stability controller × baseline EBF.

    Implements the "7.22 route B" idea more faithfully than s81:
    - Partition the frame into spatial blocks (e.g. 32x32).
    - For each short time window, build a binary event frame.
    - For each spatial block, compute a cheap contrast proxy C (total variation).
    - Keep 3-window history (C0,C1,C2) and compute stability index S in [0,1]:
        S = (C0+C1+C2) / (C0+C1+C2 + |C0-C1| + |C1-C2| + eps)
      High S means structure is present and stable across windows.
    - Use S as a weak control quantity to modulate baseline scores per event block.

    Defaults are intentionally conservative to avoid global re-ranking.

    Env vars
    - MYEVS_S82_BLOCK_PX: spatial block size in pixels (default 32)
    - MYEVS_S82_WIN_US: time window size in microseconds (0=auto, default auto=max(20000, tau/4))
    - MYEVS_S82_FACTOR_MIN / MYEVS_S82_FACTOR_MAX: factor range (default 0.90..1.10)
    """

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

    # 2) time windowing
    win_us_env = _read_int_env("MYEVS_S82_WIN_US", 0)
    if win_us_env > 0:
        win_us = int(win_us_env)
    else:
        win_us = max(20000, int(tau_us) // 4)

    win_ticks = int(tb.us_to_ticks(int(win_us)))
    if win_ticks <= 0:
        win_ticks = 1

    t0 = int(ev.t[0])
    win_id = ((ev.t.astype(np.int64, copy=False) - np.int64(t0)) // np.int64(win_ticks)).astype(
        np.int64, copy=False
    )
    n_win = int(win_id[-1]) + 1
    if n_win <= 1:
        return scores_out

    # Find contiguous ranges per window (win_id is non-decreasing).
    change = np.empty((n,), dtype=bool)
    change[0] = True
    change[1:] = win_id[1:] != win_id[:-1]
    starts = np.nonzero(change)[0]
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = n

    # 3) spatial blocks
    block_px = int(_read_int_env("MYEVS_S82_BLOCK_PX", 32))
    if block_px <= 4:
        block_px = 4

    w = int(width)
    h = int(height)
    nbx = int((w + block_px - 1) // block_px)
    nby = int((h + block_px - 1) // block_px)
    n_blocks = int(nbx * nby)

    factor_min = _read_float_env("MYEVS_S82_FACTOR_MIN", 0.90)
    factor_max = _read_float_env("MYEVS_S82_FACTOR_MAX", 1.10)
    if factor_max < factor_min:
        factor_min, factor_max = factor_max, factor_min
    span = float(factor_max - factor_min)

    eps = 1e-6

    # Contrast history per block.
    c1 = np.zeros((n_blocks,), dtype=np.float32)
    c2 = np.zeros((n_blocks,), dtype=np.float32)

    # Prealloc window frame.
    img = np.zeros((h, w), dtype=np.uint8)

    for wi in range(int(starts.shape[0])):
        i0 = int(starts[wi])
        i1 = int(ends[wi])
        if i1 <= i0:
            continue

        # Build binary frame for this window.
        img.fill(0)
        xs = ev.x[i0:i1]
        ys = ev.y[i0:i1]
        ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        if not bool(np.all(ok)):
            xs = xs[ok]
            ys = ys[ok]
        if xs.size > 0:
            img[ys, xs] = 1

        edge = _block_edges_total_variation(img)

        # Sum edges per spatial block.
        c0 = np.zeros((n_blocks,), dtype=np.float32)
        b = 0
        for by in range(nby):
            y0 = int(by * block_px)
            y1 = int(min(h, (by + 1) * block_px))
            for bx in range(nbx):
                x0 = int(bx * block_px)
                x1 = int(min(w, (bx + 1) * block_px))
                c0[b] = np.float32(np.sum(edge[y0:y1, x0:x1], dtype=np.int64))
                b += 1

        # Stability index S in [0,1].
        num = c0 + c1 + c2
        den = num + np.abs(c0 - c1) + np.abs(c1 - c2) + eps
        s = (num / den).astype(np.float32, copy=False)

        factor = (float(factor_min) + span * s.astype(np.float64, copy=False)).astype(np.float32, copy=False)

        # Apply per-event factor by spatial block.
        xw = ev.x[i0:i1].astype(np.int32, copy=False)
        yw = ev.y[i0:i1].astype(np.int32, copy=False)
        ok2 = (xw >= 0) & (xw < w) & (yw >= 0) & (yw < h)
        if not bool(np.all(ok2)):
            xw = xw[ok2]
            yw = yw[ok2]
            # When filtering coordinates, we must filter corresponding scores view.
            # To avoid complex scatter, fall back to per-event loop in this rare case.
            idx = np.nonzero(ok2)[0]
            for jj, ii in enumerate(idx):
                bx = int(xw[jj]) // int(block_px)
                by = int(yw[jj]) // int(block_px)
                bid = int(by * nbx + bx)
                scores_out[i0 + int(ii)] *= float(factor[bid])
        else:
            bx = (xw // int(block_px)).astype(np.int32, copy=False)
            by = (yw // int(block_px)).astype(np.int32, copy=False)
            bid = (by * int(nbx) + bx).astype(np.int32, copy=False)
            scores_out[i0:i1] *= factor[bid]

        # Shift history.
        c2 = c1
        c1 = c0

    return scores_out
