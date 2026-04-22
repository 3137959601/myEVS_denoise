from __future__ import annotations

import numpy as np


def event_structural_ratio_for_counts(
    counts_1d: np.ndarray,
    *,
    width: int,
    height: int,
) -> float:
    """Compute Event Structural Ratio (ESR) from a per-pixel event count map.

    This is a no-reference metric adapted from the E-MLB benchmark implementation
    (EventStructuralRatio in cuke-emlb).

    Definitions:
    - counts_1d: flattened array of length K = width*height, counts >= 0
    - N: total number of events in the slice
    - M: int(2/3 * N)

    Formula (matching cuke-emlb/python/src/utils/metric.py::EventStructuralRatio):
      ntss = sum(n*(n-1)) / (N*(N-1))
      ln   = K - sum((1 - M/N)**n)
      esr  = sqrt(ntss * ln)

    Returns 0.0 for N < 2.
    """

    w = int(width)
    h = int(height)
    if w <= 0 or h <= 0:
        raise ValueError("width/height must be positive")

    n = np.asarray(counts_1d)
    if n.ndim != 1 or int(n.shape[0]) != w * h:
        raise ValueError("counts_1d must be 1D with length width*height")

    # N is total events in this slice.
    N = int(np.sum(n))
    if N < 2:
        return 0.0

    M = int(N * 2 / 3)
    if M < 0:
        M = 0
    if M > N:
        M = N

    # ntss
    n_i64 = n.astype(np.int64, copy=False)
    ntss_num = int(np.sum(n_i64 * (n_i64 - 1)))
    ntss_den = float(N) * float(N - 1)
    ntss = float(ntss_num) / (ntss_den + float(np.spacing(1)))

    # ln
    K = int(w * h)
    q = 1.0 - (float(M) / float(N))
    # q in [0,1]. For q==0, define 0**0 -> 1 (numpy does that for integer exponent 0).
    ln = float(K) - float(np.sum(np.power(q, n_i64, dtype=np.float64)))

    v = ntss * ln
    if v <= 0.0:
        return 0.0
    return float(np.sqrt(v))


def event_structural_ratio_mean_from_xy(
    x: np.ndarray,
    y: np.ndarray,
    *,
    width: int,
    height: int,
    chunk_size: int = 30000,
) -> float:
    """Compute mean ESR over fixed-size event chunks.

    Mirrors the benchmark usage pattern:
    EventStructuralRatio.evalEventStorePerNumber(interval=30000) then mean().

    - Uses only full chunks; remainder is ignored.
    - Ignores polarity (ESR is based on event count surface).
    """

    w = int(width)
    h = int(height)
    if w <= 0 or h <= 0:
        raise ValueError("width/height must be positive")

    xs = np.asarray(x).astype(np.int32, copy=False)
    ys = np.asarray(y).astype(np.int32, copy=False)
    if xs.ndim != 1 or ys.ndim != 1 or xs.shape[0] != ys.shape[0]:
        raise ValueError("x and y must be 1D arrays of the same length")

    n_total = int(xs.shape[0])
    if n_total <= 0:
        return 0.0

    m = int(chunk_size)
    if m <= 0:
        m = 30000

    k = int(w * h)
    n_full = (n_total // m) * m
    if n_full < m:
        return 0.0

    out: list[float] = []
    for i0 in range(0, n_full, m):
        xi = xs[i0 : i0 + m]
        yi = ys[i0 : i0 + m]

        # Filter out-of-bounds (should be rare)
        ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        if not bool(np.all(ok)):
            xi = xi[ok]
            yi = yi[ok]

        if xi.size < 2:
            out.append(0.0)
            continue

        idx = (yi.astype(np.int64) * int(w) + xi.astype(np.int64)).astype(np.int64, copy=False)
        counts = np.bincount(idx, minlength=k)
        out.append(event_structural_ratio_for_counts(counts, width=w, height=h))

    if not out:
        return 0.0
    return float(np.mean(np.asarray(out, dtype=np.float64)))
