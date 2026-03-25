from __future__ import annotations

import numpy as np


def roc_curve_from_scores(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    max_points: int = 5000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ROC curve from binary labels and continuous scores.

    This mirrors the standard ROC construction:
    - Sort by score descending
    - Sweep threshold at each distinct score
    - Compute (FPR, TPR)

    Returns fpr, tpr, thresholds where thresholds are the distinct score values
    at which the step changes (descending).

    The returned points are optionally downsampled for plotting.
    """

    y = np.asarray(y_true).astype(np.int8, copy=False)
    s = np.asarray(y_score).astype(np.float64, copy=False)
    if y.ndim != 1 or s.ndim != 1 or y.shape[0] != s.shape[0]:
        raise ValueError("y_true and y_score must be 1D arrays of the same length")

    n = int(y.shape[0])
    if n == 0:
        return (
            np.asarray([0.0, 1.0], dtype=np.float64),
            np.asarray([0.0, 1.0], dtype=np.float64),
            np.asarray([np.inf, -np.inf], dtype=np.float64),
        )

    # Ensure y in {0,1}
    if not bool(np.all((y == 0) | (y == 1))):
        raise ValueError("y_true must be binary (0/1)")

    pos = int(np.sum(y))
    neg = int(n - pos)
    if pos == 0 or neg == 0:
        # Degenerate: ROC is a point + endpoints
        fpr = np.asarray([0.0, 1.0], dtype=np.float64)
        tpr = np.asarray([0.0, 1.0], dtype=np.float64)
        thr = np.asarray([np.inf, -np.inf], dtype=np.float64)
        return fpr, tpr, thr

    order = np.argsort(-s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order]

    # Cumulative TP/FP at each index.
    tp_cum = np.cumsum(y_sorted, dtype=np.int64)
    fp_cum = np.cumsum(1 - y_sorted, dtype=np.int64)

    # Take indices where score changes (end of each group).
    # Example: scores [0.9,0.9,0.7,...] -> take last index of each run.
    change = np.empty((n,), dtype=bool)
    change[:-1] = s_sorted[:-1] != s_sorted[1:]
    change[-1] = True
    idx = np.nonzero(change)[0]

    tp = tp_cum[idx]
    fp = fp_cum[idx]

    tpr = tp.astype(np.float64) / float(pos)
    fpr = fp.astype(np.float64) / float(neg)
    thresholds = s_sorted[idx].astype(np.float64, copy=False)

    # Add (0,0) at threshold=+inf
    fpr = np.concatenate([np.asarray([0.0], dtype=np.float64), fpr])
    tpr = np.concatenate([np.asarray([0.0], dtype=np.float64), tpr])
    thresholds = np.concatenate([np.asarray([np.inf], dtype=np.float64), thresholds])

    # Downsample for plotting if needed (keep endpoints).
    if max_points is not None and int(max_points) > 0 and fpr.shape[0] > int(max_points):
        m = int(max_points)
        keep = np.unique(
            np.concatenate(
                [
                    np.asarray([0, fpr.shape[0] - 1], dtype=np.int64),
                    np.linspace(0, fpr.shape[0] - 1, num=m, dtype=np.int64),
                ]
            )
        )
        fpr = fpr[keep]
        tpr = tpr[keep]
        thresholds = thresholds[keep]

    return fpr, tpr, thresholds


def auc_trapz_sorted(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Trapezoidal AUC assuming fpr is sorted ascending."""
    x = np.asarray(fpr, dtype=np.float64)
    y = np.asarray(tpr, dtype=np.float64)
    if x.size < 2:
        return 0.0
    # np.trapz is deprecated in newer NumPy; use trapezoid if available.
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(y=y, x=x))


def auc_from_scores(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Exact ROC-AUC from scores via full ROC construction + trapezoid."""
    fpr, tpr, _thr = roc_curve_from_scores(y_true, y_score, max_points=0)
    return auc_trapz_sorted(fpr, tpr)
