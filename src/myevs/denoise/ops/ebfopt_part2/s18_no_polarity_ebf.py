from __future__ import annotations

import numpy as np


def try_build_s18_no_polarity_ebf_scores_kernel():
    """Build and return Numba kernel for Part2 s18 (EBF without polarity check).

    Motivation:
    - Baseline EBF only counts same-polarity neighbors: pol_nei == pol_center.
    - s18 removes polarity gating and counts all recent neighbors regardless of polarity.

    This isolates whether polarity consistency is a key source of separability on ED24.

    Returns None if numba is unavailable.

    Kernel signature:
        (t,x,y,p,width,height,radius_px,tau_ticks,last_ts,scores_out) -> None

    Notes:
    - Streaming / single-pass: updates per-pixel last_ts for every event.
    - Complexity: O(r^2) neighborhood scan, same as baseline.
    - Polarity p is ignored for scoring but is kept in signature for API symmetry.
    """

    try:
        from numba import njit  # type: ignore
    except Exception:
        return None

    @njit(cache=True)
    def ebf_s18_scores_stream(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        width: int,
        height: int,
        radius_px: int,
        tau_ticks: int,
        last_ts: np.ndarray,
        scores_out: np.ndarray,
    ) -> None:
        n = int(t.shape[0])
        w = int(width)
        h = int(height)

        rr = int(radius_px)
        if rr < 0:
            rr = 0
        if rr > 8:
            rr = 8

        tau = int(tau_ticks)

        # Pass-through (still updates state): score is effectively +inf.
        if rr <= 0 or tau <= 0:
            for i in range(n):
                xi = int(x[i])
                yi = int(y[i])
                if xi < 0 or xi >= w or yi < 0 or yi >= h:
                    scores_out[i] = 0.0
                    continue
                ti = int(t[i])
                idx0 = yi * w + xi
                last_ts[idx0] = np.uint64(ti)
                scores_out[i] = np.inf
            return

        inv_tau = 1.0 / float(tau)

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            if xi < 0 or xi >= w or yi < 0 or yi >= h:
                scores_out[i] = 0.0
                continue

            ti = int(t[i])
            idx0 = yi * w + xi

            score = 0.0

            y0 = yi - rr
            if y0 < 0:
                y0 = 0
            y1 = yi + rr
            if y1 >= h:
                y1 = h - 1

            x0 = xi - rr
            if x0 < 0:
                x0 = 0
            x1 = xi + rr
            if x1 >= w:
                x1 = w - 1

            for yy in range(y0, y1 + 1):
                base = yy * w
                for xx in range(x0, x1 + 1):
                    if xx == xi and yy == yi:
                        continue

                    idx = base + xx
                    ts = int(last_ts[idx])
                    if ts == 0:
                        continue

                    dt = ti - ts
                    if dt < 0:
                        dt = -dt
                    if dt > tau:
                        continue

                    score += float(tau - dt) * inv_tau

            last_ts[idx0] = np.uint64(ti)
            scores_out[i] = score

    return ebf_s18_scores_stream
