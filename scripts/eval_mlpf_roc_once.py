from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from myevs.events import filter_visibility_batches, unwrap_tick_batches
from myevs.io.auto import open_events
from myevs.metrics.roc_auc import build_clean_index, signal_mask
from myevs.timebase import TimeBase


def load_events(path: str, *, width: int, height: int, tick_ns: float, batch_events: int):
    reader = open_events(path, width=width, height=height, batch_events=batch_events, tick_ns=tick_ns, assume="npy")
    batches = filter_visibility_batches(unwrap_tick_batches(reader.batches, bits=None), show_on=True, show_off=True)
    ts: list[np.ndarray] = []
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    for b in batches:
        if len(b) == 0:
            continue
        ts.append(np.asarray(b.t, dtype=np.uint64))
        xs.append(np.asarray(b.x, dtype=np.int32))
        ys.append(np.asarray(b.y, dtype=np.int32))
        ps.append(np.asarray(b.p, dtype=np.int8))
    if not ts:
        raise RuntimeError(f"No events loaded: {path}")
    return np.concatenate(ts), np.concatenate(xs), np.concatenate(ys), np.concatenate(ps)


def build_clean_labels(clean_path: str, t, x, y, p, *, width: int, height: int, tick_ns: float, batch_events: int):
    reader = open_events(clean_path, width=width, height=height, batch_events=batch_events, tick_ns=tick_ns, assume="npy")
    clean_batches = unwrap_tick_batches(reader.batches, bits=None)
    clean_keys, packer = build_clean_index(
        reader.meta,
        clean_batches,
        show_on=True,
        show_off=True,
        unwrap_ts=False,
        ts_bits=None,
        match_ticks=0,
        match_bin_radius=0,
    )
    return signal_mask(
        clean_keys=clean_keys,
        packer=packer,
        t=t,
        x=x,
        y=y,
        p=p,
        match_ticks=0,
        match_bin_radius=0,
    ).astype(bool)


def mlpf_scores(
    *,
    model_path: str,
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    width: int,
    height: int,
    patch: int,
    duration_ticks: int,
    infer_batch: int,
    output_type: str,
) -> np.ndarray:
    model = torch.jit.load(model_path, map_location="cpu")
    model.eval()

    radius = patch // 2
    area = patch * patch
    in_dim = 2 * area
    last_ts = np.zeros((height, width), dtype=np.uint64)
    probs = np.zeros((t.shape[0],), dtype=np.float32)
    feats: list[np.ndarray] = []
    idxs: list[int] = []
    inv_win = 1.0 / float(max(1, duration_ticks))

    def flush() -> None:
        if not feats:
            return
        arr = np.stack(feats, axis=0).astype(np.float32, copy=False)
        with torch.no_grad():
            out = model(torch.from_numpy(arr)).reshape(-1)
            vals = out.cpu().numpy().astype(np.float64, copy=False)
        if output_type == "logit":
            vals = 1.0 / (1.0 + np.exp(-vals))
        elif output_type == "prob":
            vals = vals
        else:
            # External model fallback: preserve historical auto behavior.
            vals = np.where((0.0 <= vals) & (vals <= 1.0), vals, 1.0 / (1.0 + np.exp(-vals)))
        probs[np.asarray(idxs, dtype=np.int64)] = vals.astype(np.float32)
        feats.clear()
        idxs.clear()

    for i in range(t.shape[0]):
        xi = int(x[i])
        yi = int(y[i])
        ti = int(t[i])
        pol = 1.0 if int(p[i]) > 0 else -1.0
        feat = np.zeros((in_dim,), dtype=np.float32)
        k = 0
        for dy in range(-radius, radius + 1):
            yy = yi + dy
            for dx in range(-radius, radius + 1):
                xx = xi + dx
                if 0 <= xx < width and 0 <= yy < height:
                    prev = int(last_ts[yy, xx])
                    feat[k] = 1.0 - float(ti - prev) * inv_win
                    feat[k + area] = pol
                k += 1
        last_ts[yi, xi] = np.uint64(ti)
        feats.append(feat)
        idxs.append(i)
        if len(feats) >= infer_batch:
            flush()
    flush()
    return probs


def auc_from_points(fpr: np.ndarray, tpr: np.ndarray) -> float:
    order = np.argsort(fpr)
    x = fpr[order]
    y = tpr[order]
    return float(np.trapezoid(y, x))


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate MLPF ROC in one streaming pass, then sweep thresholds offline.")
    ap.add_argument("--clean", required=True)
    ap.add_argument("--noisy", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--width", type=int, default=346)
    ap.add_argument("--height", type=int, default=260)
    ap.add_argument("--tick-ns", type=float, default=1000.0)
    ap.add_argument("--batch-events", type=int, default=1_000_000)
    ap.add_argument("--infer-batch", type=int, default=8192)
    ap.add_argument("--values", default=",".join(f"{i/100:.2f}" for i in range(1, 100)))
    ap.add_argument("--tag", default="mlpf_once")
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    meta_path = Path(args.model).with_suffix(".json")
    if not meta_path.exists():
        raise SystemExit(f"Missing MLPF metadata json: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    patch = int(meta["patch"])
    duration_ticks = int(meta.get("duration_ticks") or TimeBase(float(args.tick_ns)).us_to_ticks(int(meta["duration_us"])))
    output_type = str(meta.get("output_type", "logit")).lower()

    print("[1/4] load noisy events")
    t, x, y, p = load_events(
        args.noisy,
        width=int(args.width),
        height=int(args.height),
        tick_ns=float(args.tick_ns),
        batch_events=int(args.batch_events),
    )
    print(f"events={t.shape[0]}")
    print("[2/4] build labels")
    y_sig = build_clean_labels(
        args.clean,
        t,
        x,
        y,
        p,
        width=int(args.width),
        height=int(args.height),
        tick_ns=float(args.tick_ns),
        batch_events=int(args.batch_events),
    )
    print(f"signal={int(y_sig.sum())} noise={int((~y_sig).sum())}")
    print("[3/4] model scores")
    prob = mlpf_scores(
        model_path=args.model,
        t=t,
        x=x,
        y=y,
        p=p,
        width=int(args.width),
        height=int(args.height),
        patch=patch,
        duration_ticks=duration_ticks,
        infer_batch=int(args.infer_batch),
        output_type=output_type,
    )
    print(f"score min={float(prob.min()):.6f} max={float(prob.max()):.6f} mean={float(prob.mean()):.6f}")

    thresholds = [float(v) for v in str(args.values).split(",") if str(v).strip()]
    total = int(prob.shape[0])
    sig_total = int(y_sig.sum())
    noise_total = int(total - sig_total)
    rows = []
    fprs = []
    tprs = []
    print("[4/4] sweep thresholds")
    for thr in thresholds:
        keep = prob >= float(thr)
        tp = int(np.logical_and(keep, y_sig).sum())
        fp = int(np.logical_and(keep, ~y_sig).sum())
        fn = int(sig_total - tp)
        tn = int(noise_total - fp)
        tpr = tp / sig_total if sig_total else 0.0
        fpr = fp / noise_total if noise_total else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2.0 * prec * tpr / (prec + tpr) if (prec + tpr) else 0.0
        acc = (tp + tn) / total if total else 0.0
        fprs.append(fpr)
        tprs.append(tpr)
        rows.append(
            {
                "tag": args.tag,
                "method": "mlpf",
                "param": "min-neighbors",
                "value": f"{thr:.6g}",
                "roc_convention": "paper",
                "match_us": "0",
                "events_total": total,
                "signal_total": sig_total,
                "noise_total": noise_total,
                "events_kept": int(keep.sum()),
                "signal_kept": tp,
                "noise_kept": fp,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "tpr": tpr,
                "fpr": fpr,
                "precision": prec,
                "accuracy": acc,
                "f1": f1,
            }
        )
    auc = auc_from_points(np.asarray(fprs, dtype=np.float64), np.asarray(tprs, dtype=np.float64))
    for row in rows:
        row["auc"] = auc

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    best_auc_row = max(rows, key=lambda r: float(r["auc"]))
    best_f1_row = max(rows, key=lambda r: float(r["f1"]))
    print(f"saved: {out}")
    print(f"AUC={auc:.6f}")
    print(f"best_f1={float(best_f1_row['f1']):.6f} threshold={best_f1_row['value']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
