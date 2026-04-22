from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LabeledEvents:
    x: np.ndarray
    y: np.ndarray
    label: np.ndarray


def load_labeled_npy_xylabel(path: str, *, max_events: int = 0) -> LabeledEvents:
    arr = np.load(path, mmap_mode="r", allow_pickle=True)
    if max_events > 0:
        arr = arr[:max_events]

    if getattr(arr, "dtype", None) is not None and getattr(arr.dtype, "names", None):
        names = set(arr.dtype.names)
        need = {"x", "y", "label"}
        if not need.issubset(names):
            missing = sorted(need - names)
            raise SystemExit(f"input structured npy missing fields: {missing}")
        x = arr["x"].astype(np.int32, copy=False)
        y = arr["y"].astype(np.int32, copy=False)
        label = arr["label"].astype(np.int8, copy=False)
    else:
        a2 = np.asarray(arr)
        if a2.ndim != 2 or a2.shape[1] < 5:
            raise SystemExit("input must be structured (t/x/y/p/label) or 2D array with >=5 columns")

        c0 = a2[: min(10000, a2.shape[0]), 0]
        is_bin0 = bool(np.all((c0 == 0) | (c0 == 1)))
        if is_bin0:
            # [label, t, y, x, p]
            label = a2[:, 0].astype(np.int8, copy=False)
            y = a2[:, 2].astype(np.int32, copy=False)
            x = a2[:, 3].astype(np.int32, copy=False)
        else:
            # [t, x, y, p, label]
            x = a2[:, 1].astype(np.int32, copy=False)
            y = a2[:, 2].astype(np.int32, copy=False)
            label = a2[:, 4].astype(np.int8, copy=False)

    label = (label > 0).astype(np.int8, copy=False)
    return LabeledEvents(
        x=np.ascontiguousarray(x),
        y=np.ascontiguousarray(y),
        label=np.ascontiguousarray(label),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build a hotpixel mask (.npy) from labeled ED24 npy by counting NEGATIVE events per pixel."
    )
    ap.add_argument(
        "--input",
        required=True,
        help="labeled .npy (structured with x/y/label or 2D array with t/x/y/p/label columns)",
    )
    ap.add_argument("--width", type=int, default=346)
    ap.add_argument("--height", type=int, default=260)
    ap.add_argument("--max-events", type=int, default=0, help="0 means all")
    ap.add_argument("--topk", type=int, default=4096, help="number of hot pixels to mark")
    ap.add_argument("--min-neg", type=int, default=0, help="ignore pixels with neg_count < min_neg")
    ap.add_argument(
        "--pos-exclude",
        type=int,
        default=0,
        help="if >0, exclude pixels with pos_count >= pos_exclude (avoid masking true positives)",
    )
    ap.add_argument(
        "--pos-weight",
        type=float,
        default=0.0,
        help="score = neg_count - pos_weight*pos_count (default 0 => pure neg_count)",
    )
    ap.add_argument(
        "--dilate-r",
        type=int,
        default=0,
        help="optional: dilate selected hot pixels by radius r (Chebyshev/square neighborhood). 0 disables.",
    )
    ap.add_argument(
        "--output",
        required=True,
        help="output .npy path (uint8 mask). Saved as (H,W).",
    )
    args = ap.parse_args(argv)

    ev = load_labeled_npy_xylabel(str(args.input), max_events=int(args.max_events))

    w = int(args.width)
    h = int(args.height)
    if w <= 0 or h <= 0:
        raise SystemExit("width/height must be positive")

    x = ev.x.astype(np.int64, copy=False)
    y = ev.y.astype(np.int64, copy=False)
    ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    if not bool(np.all(ok)):
        x = x[ok]
        y = y[ok]
        label = ev.label[ok]
    else:
        label = ev.label

    pix = y * w + x
    is_pos = label.astype(bool, copy=False)
    is_neg = ~is_pos

    neg_counts = np.bincount(pix[is_neg], minlength=w * h).astype(np.int64, copy=False)
    pos_counts = np.bincount(pix[is_pos], minlength=w * h).astype(np.int64, copy=False)

    score = neg_counts.astype(np.float64, copy=False)
    if float(args.pos_weight) != 0.0:
        score = score - float(args.pos_weight) * pos_counts.astype(np.float64, copy=False)

    if int(args.min_neg) > 0:
        score = np.where(neg_counts >= int(args.min_neg), score, -np.inf)

    if int(args.pos_exclude) > 0:
        score = np.where(pos_counts < int(args.pos_exclude), score, -np.inf)

    k = int(args.topk)
    k = max(1, min(k, w * h))

    finite = np.isfinite(score)
    n_finite = int(np.sum(finite))
    if n_finite <= 0:
        raise SystemExit("no valid pixels remain after constraints (all scores are -inf)")
    if k > n_finite:
        print(f"warning: requested topk={k} but only {n_finite} pixels are valid; shrinking k")
        k = n_finite

    valid_idx = np.flatnonzero(finite)
    valid_score = score[valid_idx]
    cand_local = np.argpartition(valid_score, -k)[-k:]
    cand_local = cand_local[np.argsort(valid_score[cand_local])][::-1]
    cand = valid_idx[cand_local]

    mask = np.zeros((w * h,), dtype=np.uint8)
    mask[cand] = 1
    mask2d = mask.reshape((h, w))

    dr = int(args.dilate_r)
    if dr > 0:
        src = mask2d.astype(bool, copy=False)
        out = src.copy()
        for dy in range(-dr, dr + 1):
            y0s = max(0, -dy)
            y1s = min(h, h - dy)
            y0d = y0s + dy
            y1d = y1s + dy
            for dx in range(-dr, dr + 1):
                x0s = max(0, -dx)
                x1s = min(w, w - dx)
                x0d = x0s + dx
                x1d = x1s + dx
                out[y0d:y1d, x0d:x1d] |= src[y0s:y1s, x0s:x1s]
        mask2d = out.astype(np.uint8)

    out_path = str(args.output)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.save(out_path, mask2d)

    thr = float(score[cand[-1]]) if cand.size > 0 else float("nan")
    print(f"saved: {out_path}")
    print(f"marked: {int(cand.size)} / {w*h} pixels")
    if dr > 0:
        print(f"after dilation r={dr}: {int(np.sum(mask2d != 0))} / {w*h} pixels")
    print(f"score threshold (kth): {thr}")
    print(
        "top1:",
        f"pix={int(cand[0])}",
        f"neg={int(neg_counts[cand[0]])}",
        f"pos={int(pos_counts[cand[0]])}",
        f"score={float(score[cand[0]]):.3f}",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
