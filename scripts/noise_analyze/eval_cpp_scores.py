from __future__ import annotations

import argparse
import csv
import struct
from pathlib import Path

import numpy as np


def _read_labels(bin_path: Path) -> np.ndarray:
    with bin_path.open("rb") as f:
        magic, n, _width, _height = struct.unpack("<8sQII", f.read(24))
        if magic != b"MYEVSBIN":
            raise SystemExit(f"bad binary magic: {bin_path}")
        n = int(n)
        f.seek(24 + n * 8 + n * 2 + n * 2 + n)
        return np.fromfile(f, dtype=np.uint8, count=n)


def _best_f1_auc(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float, float, float, float]:
    y = labels.astype(np.uint8, copy=False)
    s = scores.astype(np.float64, copy=False)
    order = np.argsort(-s, kind="mergesort")
    ys = y[order]
    ss = s[order]
    pos = float(np.sum(y > 0))
    neg = float(y.shape[0] - pos)
    if pos <= 0.0 or neg <= 0.0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    tp = np.cumsum(ys > 0, dtype=np.float64)
    fp = np.cumsum(ys == 0, dtype=np.float64)
    recall = tp / pos
    precision = tp / np.maximum(tp + fp, 1.0)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    best_idx = int(np.argmax(f1))

    # ROC AUC from rank order, trapezoid on all operating points.
    tpr = np.concatenate(([0.0], recall, [1.0]))
    fpr = np.concatenate(([0.0], fp / neg, [1.0]))
    auc = float(np.trapezoid(tpr, fpr))
    return float(auc), float(f1[best_idx]), float(precision[best_idx]), float(recall[best_idx]), float(ss[best_idx])


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate C++ scorer float32 scores against labels stored in exported bin.")
    ap.add_argument("--bin", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    labels = _read_labels(Path(str(args.bin)))
    scores = np.fromfile(str(args.scores), dtype=np.float32)
    if scores.shape[0] != labels.shape[0]:
        raise SystemExit(f"score count mismatch: scores={scores.shape[0]} labels={labels.shape[0]}")

    auc, f1, precision, recall, thr = _best_f1_auc(labels, scores)
    out = Path(str(args.out_csv))
    out.parent.mkdir(parents=True, exist_ok=True)
    exists = out.exists()
    with out.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["tag", "events", "auc", "best_f1", "precision", "recall", "threshold"])
        w.writerow([str(args.tag), int(labels.shape[0]), f"{auc:.9f}", f"{f1:.9f}", f"{precision:.9f}", f"{recall:.9f}", f"{thr:.9f}"])
    print(f"{args.tag}: auc={auc:.9f} best_f1={f1:.9f} precision={precision:.9f} recall={recall:.9f} thr={thr:.9f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
