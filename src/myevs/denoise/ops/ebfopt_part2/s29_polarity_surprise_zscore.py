from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class S29PolaritySurpriseZScoreParams:
    """Parameters for Part2 s29 (local polarity-surprise z-score).

    Motivation (from s28 lessons):
    - In ED24/myPedestrain_06 heavy, dominant FP is local hotmask / near-hot.
      A global-rate normalization (s28) is an information mismatch: it tries to
      explain a local failure mode with a global statistic.

    Core idea:
    - In one neighborhood scan (same as baseline EBF), accumulate *local*
      background activity and compare the observed same-polarity support to what
      we'd expect under a simple noise model:
        * Neighbor polarities are random (P(match)=0.5).
        * Recency weights w_j = max(0, 1 - dt/tau) are treated as fixed.

      Define (using the same tau-weight as baseline, but in integer ticks):
        S_all  = sum_j w_j
        S_same = sum_j w_j * 1{pol_j == pol_i}
        S_sq   = sum_j w_j^2

      Under the noise polarity model:
        E[S_same] = 0.5 * S_all
        Var[S_same] = 0.25 * S_sq

      So a natural z-score is:
        z = (S_same - 0.5*S_all) / sqrt(0.25*S_sq + eps)
          = (2*S_same - S_all) / sqrt(S_sq + eps)

    Properties:
    - Streaming, single-pass, O(r^2), Numba-only.
    - No extra per-pixel state beyond baseline `last_ts/last_pol`.
    - No extra hyperparameters (uses tau_us of the sweep point).

    Note:
    - This score measures *local polarity-consistency surprise* rather than
      absolute density; it is designed to be more robust to local rate/hotspots.
    """


def try_build_s29_polarity_surprise_zscore_scores_kernel():
    """Build and return Numba kernel for s29 score streaming.

    Returns None if numba is unavailable.

    Kernel signature:
      (t,x,y,p,width,height,radius_px,tau_ticks,last_ts,last_pol,scores_out) -> None

    Arrays:
    - last_ts: uint64 per pixel
    - last_pol: int8 per pixel

    Output:
    - scores_out: float32 z-score per event
    """

    try:
        from numba import njit  # type: ignore
    except Exception:
        return None

    @njit(cache=True)
    def ebf_s29_scores_stream(
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

        # Same convention as other Part2 kernels: rr<=0 is a pass-through (still
        # updates state), and returns +inf so ROC treats it as always-positive.
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

        eps = 1e-6

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            if xi < 0 or xi >= w or yi < 0 or yi >= h:
                scores_out[i] = 0.0
                continue

            ti = int(t[i])
            pi = 1 if int(p[i]) > 0 else -1
            idx0 = yi * w + xi

            sum_all_w = 0
            sum_same_w = 0
            sum_sq_w2 = 0

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
                    pol = int(last_pol[idx])
                    if pol == 0:
                        continue
                    if pol != pi and pol != -pi:
                        continue

                    ts = int(last_ts[idx])
                    if ts == 0:
                        continue

                    dt = ti - ts
                    if dt < 0:
                        dt = -dt
                    if dt > tau:
                        continue

                    wj = tau - dt
                    if wj <= 0:
                        continue

                    sum_all_w += wj
                    if pol == pi:
                        sum_same_w += wj

                    # wj is int in [1,tau]; wj^2 fits in 64-bit for typical tau.
                    sum_sq_w2 += wj * wj

            # Always update self state
            last_ts[idx0] = np.uint64(ti)
            last_pol[idx0] = np.int8(pi)

            if sum_sq_w2 <= 0:
                scores_out[i] = 0.0
                continue

            num = 2.0 * float(sum_same_w) - float(sum_all_w)
            denom = np.sqrt(float(sum_sq_w2) + eps)
            scores_out[i] = float(num / denom)

    return ebf_s29_scores_stream


__all__ = [
    "S29PolaritySurpriseZScoreParams",
    "try_build_s29_polarity_surprise_zscore_scores_kernel",
]
