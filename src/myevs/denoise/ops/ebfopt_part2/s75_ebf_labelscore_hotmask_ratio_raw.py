from __future__ import annotations

import numpy as np


def try_build_s75_ebf_labelscore_hotmask_ratio_raw_scores_kernel():
    """Build and return Numba kernel for s75 score streaming.

    s75 goal (replace raw, target seg1 failure mode):
    - Replace baseline neighborhood support (raw sum) with a hotmask-aware variant
      that suppresses evidence dominated by hot pixels.
    - Use the precomputed ED24 hotmask (uint8 {0,1} per pixel), so we do not
      introduce any new sweep dimensions.

    Core definition (streaming, single-pass, O(r^2)):
    - Same-polarity neighbors within radius r.
    - Triangular age weight for dt<=tau:
        w = max(0, (tau - dt) / tau)
    - Split support into non-hot and hot contributions using hot_mask[idx]:
        raw_nonhot = sum_{hot_mask==0} w
        raw_hot    = sum_{hot_mask==1} w
    - Final score (hot-aware but not anti-hot):
        score = raw_nonhot / (1 + raw_hot)

    Notes:
    - hot_mask is provided as a flattened uint8 array of shape (W*H,).
    - Returns None if numba is unavailable.

    Kernel signature:
      (t,x,y,p,width,height,radius_px,tau_ticks,last_ts,last_pol,hot_mask,scores_out) -> None
    """

    try:
        from numba import njit  # type: ignore
    except Exception:
        return None

    @njit(cache=False)
    def ebf_s75_scores_stream(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        width: int,
        height: int,
        radius_px: int,
        tau_ticks: int,
        last_ts: np.ndarray,
        last_pol: np.ndarray,
        hot_mask: np.ndarray,
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
        if tau <= 0:
            tau = 1

        if rr <= 0:
            for i in range(n):
                xi = int(x[i])
                yi = int(y[i])
                if xi < 0 or xi >= w or yi < 0 or yi >= h:
                    scores_out[i] = 0.0
                    continue

                ti = int(t[i])
                pi = 1 if int(p[i]) > 0 else -1
                idx0 = yi * w + xi
                last_ts[idx0] = np.uint64(ti)
                last_pol[idx0] = np.int8(pi)
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
            pi = 1 if int(p[i]) > 0 else -1
            idx0 = yi * w + xi

            raw_nonhot = 0.0
            raw_hot = 0.0

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
                    if int(last_pol[idx]) != pi:
                        continue

                    ts = int(last_ts[idx])
                    if ts == 0:
                        continue

                    dt = (ti - ts) if ti >= ts else (ts - ti)
                    if dt > tau:
                        continue

                    w_age = float(tau - dt) * inv_tau
                    if w_age <= 0.0:
                        continue

                    if int(hot_mask[idx]) != 0:
                        raw_hot += w_age
                    else:
                        raw_nonhot += w_age

            last_ts[idx0] = np.uint64(ti)
            last_pol[idx0] = np.int8(pi)

            scores_out[i] = float(raw_nonhot / (1.0 + raw_hot))

    return ebf_s75_scores_stream


__all__ = [
    "try_build_s75_ebf_labelscore_hotmask_ratio_raw_scores_kernel",
]
