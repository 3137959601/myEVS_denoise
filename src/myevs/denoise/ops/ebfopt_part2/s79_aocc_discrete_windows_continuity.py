from __future__ import annotations

"""Part2 s79: discrete 3-window structural continuity (AOCC-inspired, no EBF).

Motivation
----------
Both s76..s78 used exponentially decayed activity surfaces. That is convenient
but still not the same object as AOCC's multi-Δt accumulation and the idea of
*consecutive time windows*.

s79 implements the 7.21 Route-B idea more literally:
- Maintain per-pixel activity counts in three *consecutive* windows of length τ:

    W1=[t-τ, t]     (current)
    W2=[t-2τ, t-τ]  (previous)
    W3=[t-3τ, t-2τ] (previous-previous)

- Compute a local structure strength Ck in each window k, then compute a
  continuity score:

    S_cont = (C1+C2+C3) / (C1+C2+C3 + |C1-C2| + |C2-C3| + eps)

- Use score = (C1+C2+C3) * S_cont

Implementation notes
--------------------
We avoid per-event exp() by using a discrete bin index:
    bin = floor(t / τ)
Each pixel stores:
- last_bin (int64)
- c0,c1,c2: counts for bins last_bin, last_bin-1, last_bin-2

When reading a pixel at current bin, we *virtually* align its stored bins to the
current bin (0..2 lag); if lag >= 3, counts are treated as 0.

We split polarity channels by maintaining separate states for + and -.
Per event, we score using the current polarity channel only.

Complexity
----------
Per event: fixed 3 windows × 3x3 sampling = 27 reads + Sobel/hypot + arithmetic.
State per polarity: last_bin:int64 + 3 float32 counts.
"""

import numpy as np


def try_build_s79_aocc_discrete_windows_continuity_scores_kernel():
    """Build and return Numba kernel for s79 streaming scores.

    Returns None if numba is unavailable.

    Kernel signature:
        (t,x,y,p,width,height,radius_px,tau_ticks,
         last_bin_pos,c0_pos,c1_pos,c2_pos,
         last_bin_neg,c0_neg,c1_neg,c2_neg,
         scores_out) -> None

    - tau_ticks is the window length for binning (>=1).
    - For each event, only the matching polarity state is used/updated.
    """

    try:
        from numba import njit  # type: ignore
    except Exception:
        return None

    @njit(cache=True)
    def _get_c_aligned(cur_bin: int, idx: int, last_bin: np.ndarray, c0: np.ndarray, c1: np.ndarray, c2: np.ndarray):
        lb = int(last_bin[idx])
        if lb == -1:
            return 0.0, 0.0, 0.0
        lag = cur_bin - lb
        if lag <= 0:
            # same bin (or time went backwards)
            return float(c0[idx]), float(c1[idx]), float(c2[idx])
        if lag == 1:
            return 0.0, float(c0[idx]), float(c1[idx])
        if lag == 2:
            return 0.0, 0.0, float(c0[idx])
        return 0.0, 0.0, 0.0

    @njit(cache=True)
    def _advance_and_add(cur_bin: int, idx: int, last_bin: np.ndarray, c0: np.ndarray, c1: np.ndarray, c2: np.ndarray):
        lb = int(last_bin[idx])
        if lb == -1:
            last_bin[idx] = np.int64(cur_bin)
            c0[idx] = np.float32(1.0)
            c1[idx] = np.float32(0.0)
            c2[idx] = np.float32(0.0)
            return

        lag = cur_bin - lb
        if lag <= 0:
            # same bin
            c0[idx] = np.float32(float(c0[idx]) + 1.0)
            return

        if lag >= 3:
            # too old, reset
            last_bin[idx] = np.int64(cur_bin)
            c0[idx] = np.float32(1.0)
            c1[idx] = np.float32(0.0)
            c2[idx] = np.float32(0.0)
            return

        # shift by lag (1 or 2)
        if lag == 1:
            c2[idx] = c1[idx]
            c1[idx] = c0[idx]
            c0[idx] = np.float32(1.0)
        else:
            # lag == 2
            c2[idx] = c0[idx]
            c1[idx] = np.float32(0.0)
            c0[idx] = np.float32(1.0)

        last_bin[idx] = np.int64(cur_bin)

    @njit(cache=True)
    def ebf_s79_scores_stream(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        width: int,
        height: int,
        radius_px: int,
        tau_ticks: int,
        last_bin_pos: np.ndarray,
        c0_pos: np.ndarray,
        c1_pos: np.ndarray,
        c2_pos: np.ndarray,
        last_bin_neg: np.ndarray,
        c0_neg: np.ndarray,
        c1_neg: np.ndarray,
        c2_neg: np.ndarray,
        scores_out: np.ndarray,
    ) -> None:
        n = int(t.shape[0])
        w = int(width)
        h = int(height)

        step = int(radius_px)
        if step < 1:
            step = 1
        if step > 8:
            step = 8

        tau_i = int(tau_ticks)
        if tau_i < 1:
            tau_i = 1

        eps = 1e-3

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            if xi < 0 or xi >= w or yi < 0 or yi >= h:
                scores_out[i] = 0.0
                continue

            ti = int(t[i])
            cur_bin = ti // tau_i
            pi = int(p[i])

            if pi > 0:
                last_bin = last_bin_pos
                c0 = c0_pos
                c1 = c1_pos
                c2 = c2_pos
            else:
                last_bin = last_bin_neg
                c0 = c0_neg
                c1 = c1_neg
                c2 = c2_neg

            idx0 = yi * w + xi

            # Always update the center pixel state for the current event.
            _advance_and_add(cur_bin, idx0, last_bin, c0, c1, c2)

            # 3x3 bounds.
            x0 = xi - step
            x1 = xi
            x2 = xi + step
            if x0 < 0:
                x0 = 0
            if x2 >= w:
                x2 = w - 1

            y0 = yi - step
            y1 = yi
            y2 = yi + step
            if y0 < 0:
                y0 = 0
            if y2 >= h:
                y2 = h - 1

            idx00 = y0 * w + x0
            idx01 = y0 * w + x1
            idx02 = y0 * w + x2
            idx10 = y1 * w + x0
            idx11 = y1 * w + x1
            idx12 = y1 * w + x2
            idx20 = y2 * w + x0
            idx21 = y2 * w + x1
            idx22 = y2 * w + x2

            # Read aligned counts for each sample.
            a00 = _get_c_aligned(cur_bin, idx00, last_bin, c0, c1, c2)
            a01 = _get_c_aligned(cur_bin, idx01, last_bin, c0, c1, c2)
            a02 = _get_c_aligned(cur_bin, idx02, last_bin, c0, c1, c2)
            a10 = _get_c_aligned(cur_bin, idx10, last_bin, c0, c1, c2)
            a11 = _get_c_aligned(cur_bin, idx11, last_bin, c0, c1, c2)
            a12 = _get_c_aligned(cur_bin, idx12, last_bin, c0, c1, c2)
            a20 = _get_c_aligned(cur_bin, idx20, last_bin, c0, c1, c2)
            a21 = _get_c_aligned(cur_bin, idx21, last_bin, c0, c1, c2)
            a22 = _get_c_aligned(cur_bin, idx22, last_bin, c0, c1, c2)

            # Window-1 (current): take component 0.
            w1_00 = a00[0]; w1_01 = a01[0]; w1_02 = a02[0]
            w1_10 = a10[0]; w1_11 = a11[0]; w1_12 = a12[0]
            w1_20 = a20[0]; w1_21 = a21[0]; w1_22 = a22[0]

            # Window-2 (previous): component 1.
            w2_00 = a00[1]; w2_01 = a01[1]; w2_02 = a02[1]
            w2_10 = a10[1]; w2_11 = a11[1]; w2_12 = a12[1]
            w2_20 = a20[1]; w2_21 = a21[1]; w2_22 = a22[1]

            # Window-3 (previous-previous): component 2.
            w3_00 = a00[2]; w3_01 = a01[2]; w3_02 = a02[2]
            w3_10 = a10[2]; w3_11 = a11[2]; w3_12 = a12[2]
            w3_20 = a20[2]; w3_21 = a21[2]; w3_22 = a22[2]

            # Structure strength per window: Sobel grad / local mean.
            # W1
            gx1 = (w1_02 + 2.0 * w1_12 + w1_22) - (w1_00 + 2.0 * w1_10 + w1_20)
            gy1 = (w1_20 + 2.0 * w1_21 + w1_22) - (w1_00 + 2.0 * w1_01 + w1_02)
            g1 = float(np.hypot(gx1, gy1))
            m1 = (w1_00 + w1_01 + w1_02 + w1_10 + w1_11 + w1_12 + w1_20 + w1_21 + w1_22) / 9.0
            c1v = g1 / (m1 + eps)

            # W2
            gx2 = (w2_02 + 2.0 * w2_12 + w2_22) - (w2_00 + 2.0 * w2_10 + w2_20)
            gy2 = (w2_20 + 2.0 * w2_21 + w2_22) - (w2_00 + 2.0 * w2_01 + w2_02)
            g2 = float(np.hypot(gx2, gy2))
            m2 = (w2_00 + w2_01 + w2_02 + w2_10 + w2_11 + w2_12 + w2_20 + w2_21 + w2_22) / 9.0
            c2v = g2 / (m2 + eps)

            # W3
            gx3 = (w3_02 + 2.0 * w3_12 + w3_22) - (w3_00 + 2.0 * w3_10 + w3_20)
            gy3 = (w3_20 + 2.0 * w3_21 + w3_22) - (w3_00 + 2.0 * w3_01 + w3_02)
            g3 = float(np.hypot(gx3, gy3))
            m3 = (w3_00 + w3_01 + w3_02 + w3_10 + w3_11 + w3_12 + w3_20 + w3_21 + w3_22) / 9.0
            c3v = g3 / (m3 + eps)

            num = (c1v + c2v + c3v)
            den = num + abs(c1v - c2v) + abs(c2v - c3v) + eps
            s_cont = num / den
            scores_out[i] = num * s_cont

    return ebf_s79_scores_stream
