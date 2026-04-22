from __future__ import annotations

"""Part2 s76: AOCC-inspired online structural score (no EBF).

Motivation (from 7.20 discussion)
--------------------------------
AOCC measures how "structure becomes visible" across time scales via contrast
curves on accumulated event frames. AOCC itself is an offline metric and is too
heavy to compute per-event in real time.

This s76 variant is a streaming approximation of "local structural visibility":

1) Maintain an *activity surface* A(x,y,t) with exponential time decay.
   - Each incoming event adds +1 at its pixel.
   - Between events, A decays with time constant tau.

2) Use spatial gradient magnitude of A as a per-event score:
   - We compute a Sobel-like gradient on a 3x3 stencil.
   - The spatial step is controlled by `radius_px` (derived from sweep `s`).

The resulting score is NOT an EBF neighbor count; it's an edge/structure
strength proxy on a temporally smoothed event-activity surface.

Complexity
----------
Per event: O(1) reads (9 samples) + a few exp() + arithmetic.
State: 2 arrays of size W*H: last_t (uint64), last_a (float32).

Notes
-----
- Polarity is intentionally ignored for the first clean implementation.
- This module is designed for the Part2 slim sweep scripts.
"""

import numpy as np


def try_build_s76_aocc_activity_sobel_gradmag_scores_kernel():
    """Build and return Numba kernel for s76 streaming scores.

    Returns None if numba is unavailable.

    Kernel signature:
        (t,x,y,p,width,height,radius_px,tau_ticks,last_t,last_a,scores_out) -> None

    Where:
    - radius_px is the spatial step of the Sobel stencil (>=1).
    - tau_ticks is the exponential decay time constant in ticks (>=1).
    """

    try:
        from numba import njit  # type: ignore
    except Exception:
        return None

    @njit(cache=True)
    def _activity_at(
        ti: int,
        idx: int,
        tau: float,
        last_t: np.ndarray,
        last_a: np.ndarray,
    ) -> float:
        ts = int(last_t[idx])
        if ts <= 0:
            return 0.0
        dt = ti - ts
        if dt <= 0:
            return float(last_a[idx])
        # exponential decay
        return float(last_a[idx]) * float(np.exp(-float(dt) / tau))

    @njit(cache=True)
    def ebf_s76_scores_stream(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        width: int,
        height: int,
        radius_px: int,
        tau_ticks: int,
        last_t: np.ndarray,
        last_a: np.ndarray,
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
        tau = float(tau_i)

        # numeric/scale safety: cap activity to avoid runaway growth
        a_cap = 50.0

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            if xi < 0 or xi >= w or yi < 0 or yi >= h:
                scores_out[i] = 0.0
                continue

            ti = int(t[i])

            # Update center activity with current event (+1).
            idx0 = yi * w + xi
            a0 = _activity_at(ti, idx0, tau, last_t, last_a) + 1.0
            if a0 > a_cap:
                a0 = a_cap
            last_t[idx0] = np.uint64(ti)
            last_a[idx0] = np.float32(a0)

            # Sample a 3x3 grid with spatial step.
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

            # Note: for the center (x1,y1), we use the updated activity a0.
            # For other samples, use decayed activity at time ti.
            idx00 = y0 * w + x0
            idx01 = y0 * w + x1
            idx02 = y0 * w + x2
            idx10 = y1 * w + x0
            idx11 = y1 * w + x1
            idx12 = y1 * w + x2
            idx20 = y2 * w + x0
            idx21 = y2 * w + x1
            idx22 = y2 * w + x2

            a00 = _activity_at(ti, idx00, tau, last_t, last_a)
            a01 = _activity_at(ti, idx01, tau, last_t, last_a)
            a02 = _activity_at(ti, idx02, tau, last_t, last_a)

            a10 = _activity_at(ti, idx10, tau, last_t, last_a)
            a11 = a0 if (idx11 == idx0) else _activity_at(ti, idx11, tau, last_t, last_a)
            a12 = _activity_at(ti, idx12, tau, last_t, last_a)

            a20 = _activity_at(ti, idx20, tau, last_t, last_a)
            a21 = _activity_at(ti, idx21, tau, last_t, last_a)
            a22 = _activity_at(ti, idx22, tau, last_t, last_a)

            # Sobel gradient (right-left, bottom-top)
            gx = (a02 + 2.0 * a12 + a22) - (a00 + 2.0 * a10 + a20)
            gy = (a20 + 2.0 * a21 + a22) - (a00 + 2.0 * a01 + a02)

            scores_out[i] = float(np.hypot(gx, gy))

    return ebf_s76_scores_stream
