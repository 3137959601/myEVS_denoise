from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from myevs.denoise.numba_ebf import ebf_scores_stream_numba, ebf_state_init
from myevs.denoise.ops.ebfopt_part2.n149_n145_s52_euclid_compactlut_backbone import score_stream_n149
from myevs.metrics.roc_auc import auc_trapz
from myevs.metrics.roc_score_label import auc_from_scores
from myevs.timebase import TimeBase


@dataclass(frozen=True)
class LabeledEvents:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    p: np.ndarray
    label: np.ndarray


def parse_int_list(text: str) -> list[int]:
    return [int(float(x.strip())) for x in str(text).split(",") if x.strip()]


def load_events(path: Path, max_events: int = 0) -> LabeledEvents:
    arr = np.load(path, mmap_mode="r")
    if max_events > 0:
        arr = arr[:max_events]
    return LabeledEvents(
        t=np.ascontiguousarray(arr["t"].astype(np.uint64, copy=False)),
        x=np.ascontiguousarray(arr["x"].astype(np.int32, copy=False)),
        y=np.ascontiguousarray(arr["y"].astype(np.int32, copy=False)),
        p=np.ascontiguousarray(arr["p"].astype(np.int8, copy=False)),
        label=np.ascontiguousarray((arr["label"] > 0).astype(np.uint8, copy=False)),
    )


def roc_from_scores_thresholds(
    labels: np.ndarray,
    scores: np.ndarray,
    thresholds: np.ndarray,
    *,
    tag: str,
    method: str,
    param: str,
) -> tuple[list[dict], dict]:
    n = int(labels.shape[0])
    pos = int(labels.sum())
    neg = int(n - pos)
    rows: list[dict] = []
    fprs: list[float] = []
    tprs: list[float] = []
    best: dict | None = None
    for th in thresholds:
        keep = scores > float(th)
        tp = int(np.count_nonzero(keep & (labels != 0)))
        fp = int(np.count_nonzero(keep & (labels == 0)))
        tn = int(neg - fp)
        fn = int(pos - tp)
        tpr = tp / pos if pos else 0.0
        fpr = fp / neg if neg else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        accuracy = (tp + tn) / n if n else 0.0
        f1 = 2.0 * precision * tpr / (precision + tpr) if (precision + tpr) else 0.0
        row = {
            "tag": tag,
            "method": method,
            "param": param,
            "value": float(th),
            "events_total": n,
            "signal_total": pos,
            "noise_total": neg,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "tpr": tpr,
            "fpr": fpr,
            "precision": precision,
            "accuracy": accuracy,
            "f1": f1,
        }
        rows.append(row)
        fprs.append(fpr)
        tprs.append(tpr)
        if best is None or (f1, tpr, precision, -fpr) > (
            float(best["f1"]),
            float(best["tpr"]),
            float(best["precision"]),
            -float(best["fpr"]),
        ):
            best = row
    auc = auc_trapz(np.asarray(fprs), np.asarray(tprs))
    for row in rows:
        row["auc"] = auc
    assert best is not None
    best = dict(best)
    best["auc"] = auc
    return rows, best


def roc_exact_rows(labels: np.ndarray, scores: np.ndarray, *, tag: str, method: str, max_points: int = 4000) -> tuple[list[dict], dict]:
    y = labels.astype(np.int8, copy=False)
    s = scores.astype(np.float64, copy=False)
    n = int(y.shape[0])
    pos = int(y.sum())
    neg = int(n - pos)
    order = np.argsort(-s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order]
    tp_cum = np.cumsum(y_sorted, dtype=np.int64)
    fp_cum = np.cumsum(1 - y_sorted, dtype=np.int64)
    change = np.empty((n,), dtype=bool)
    change[:-1] = s_sorted[:-1] != s_sorted[1:]
    change[-1] = True
    idx = np.nonzero(change)[0]
    tp = np.concatenate([np.asarray([0], dtype=np.int64), tp_cum[idx]])
    fp = np.concatenate([np.asarray([0], dtype=np.int64), fp_cum[idx]])
    thr = np.concatenate([np.asarray([np.inf], dtype=np.float64), s_sorted[idx]])
    tpr = tp.astype(np.float64) / float(pos) if pos else np.zeros_like(tp, dtype=np.float64)
    fpr = fp.astype(np.float64) / float(neg) if neg else np.zeros_like(fp, dtype=np.float64)
    auc = float((getattr(np, "trapezoid", None) or np.trapz)(y=tpr, x=fpr))

    keep_idx = np.arange(thr.shape[0])
    if max_points > 0 and keep_idx.shape[0] > max_points:
        keep_idx = np.unique(
            np.concatenate(
                [
                    np.asarray([0, thr.shape[0] - 1], dtype=np.int64),
                    np.linspace(0, thr.shape[0] - 1, max_points, dtype=np.int64),
                ]
            )
        )
    rows: list[dict] = []
    best: dict | None = None
    for i in keep_idx:
        tp_i = int(tp[i])
        fp_i = int(fp[i])
        tn_i = int(neg - fp_i)
        fn_i = int(pos - tp_i)
        precision = tp_i / (tp_i + fp_i) if (tp_i + fp_i) else 0.0
        f1 = 2.0 * precision * float(tpr[i]) / (precision + float(tpr[i])) if (precision + float(tpr[i])) else 0.0
        row = {
            "tag": tag,
            "method": method,
            "param": "score-threshold",
            "value": float(thr[i]),
            "events_total": n,
            "signal_total": pos,
            "noise_total": neg,
            "tp": tp_i,
            "fp": fp_i,
            "tn": tn_i,
            "fn": fn_i,
            "tpr": float(tpr[i]),
            "fpr": float(fpr[i]),
            "precision": precision,
            "accuracy": (tp_i + tn_i) / n if n else 0.0,
            "f1": f1,
            "auc": auc,
        }
        rows.append(row)
        if best is None or (f1, float(tpr[i]), precision, -float(fpr[i])) > (
            float(best["f1"]),
            float(best["tpr"]),
            float(best["precision"]),
            -float(best["fpr"]),
        ):
            best = row
    assert best is not None
    return rows, dict(best)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run EBF and N149 on converted DVSCLEAN data.")
    ap.add_argument("--npy-root", default=r"D:\hjx_workspace\scientific_reserach\dataset\DVSCLEAN\converted_npy")
    ap.add_argument("--out-root", default="data/DVSCLEAN")
    ap.add_argument("--max-events", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--tick-ns", type=float, default=1000.0)
    ap.add_argument("--ebf-radius", type=int, default=2)
    ap.add_argument("--ebf-tau-us", type=int, default=32000)
    ap.add_argument("--n149-radius-list", default="2,3,4,5")
    ap.add_argument("--n149-tau-us-list", default="16000,32000,64000,128000,256000,512000")
    args = ap.parse_args()

    npy_root = Path(args.npy_root)
    out_root = Path(args.out_root)
    tb = TimeBase(tick_ns=float(args.tick_ns))
    ebf_thresholds = np.round(np.arange(0.0, 15.0 + 1e-9, 0.1), 10)
    n149_radii = parse_int_list(args.n149_radius_list)
    n149_taus = parse_int_list(args.n149_tau_us_list)

    summary: list[dict] = []
    files = sorted(npy_root.glob("*/*/*_labeled.npy"))
    if not files:
        raise SystemExit(f"No converted labeled npy found under: {npy_root}")

    for npy_path in files:
        scene = npy_path.parents[1].name
        level = npy_path.parent.name
        meta_path = npy_path.with_name(f"{scene}_{level}_meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        ev = load_events(npy_path, max_events=int(args.max_events))
        print(f"[DVSCLEAN] {scene} {level}: n={ev.label.shape[0]}")

        scene_root = out_root / scene
        ebf_dir = scene_root / "EBF"
        n149_dir = scene_root / "N149"
        ebf_dir.mkdir(parents=True, exist_ok=True)
        n149_dir.mkdir(parents=True, exist_ok=True)

        last_ts, last_pol = ebf_state_init(int(args.width), int(args.height))
        ebf_scores = np.empty(ev.label.shape[0], dtype=np.float64)
        ebf_tau_ticks = int(tb.us_to_ticks(int(args.ebf_tau_us)))
        for i0 in range(0, ev.label.shape[0], 1_000_000):
            i1 = min(ev.label.shape[0], i0 + 1_000_000)
            out = np.empty(i1 - i0, dtype=np.float64)
            ebf_scores_stream_numba(
                t=ev.t[i0:i1],
                x=ev.x[i0:i1],
                y=ev.y[i0:i1],
                p=ev.p[i0:i1],
                width=int(args.width),
                height=int(args.height),
                radius_px=int(args.ebf_radius),
                tau_ticks=ebf_tau_ticks,
                last_ts=last_ts,
                last_pol=last_pol,
                scores_out=out,
            )
            ebf_scores[i0:i1] = out
        ebf_rows, ebf_best = roc_from_scores_thresholds(
            ev.label,
            ebf_scores,
            ebf_thresholds,
            tag=f"ebf_r{args.ebf_radius}_tau{args.ebf_tau_us}_{scene}_{level}",
            method="ebf",
            param="min-neighbors",
        )
        ebf_exact = auc_from_scores(ev.label, ebf_scores)
        ebf_csv = ebf_dir / f"roc_ebf_{scene}_{level}.csv"
        pd.DataFrame(ebf_rows).to_csv(ebf_csv, index=False)

        best_auc = None
        best_f1 = None
        n149_all_rows: list[dict] = []
        for r in n149_radii:
            for tau_us in n149_taus:
                tag = f"n149_r{r}_tau{tau_us}_{scene}_{level}"
                scores = score_stream_n149(
                    ev,
                    width=int(args.width),
                    height=int(args.height),
                    radius_px=int(r),
                    tau_us=int(tau_us),
                    tb=tb,
                )
                rows, best = roc_exact_rows(ev.label, scores, tag=tag, method="n149", max_points=4000)
                n149_all_rows.extend(rows)
                if best_auc is None or float(best["auc"]) > float(best_auc["auc"]):
                    best_auc = dict(best)
                    best_auc["radius"] = r
                    best_auc["tau_us"] = tau_us
                if best_f1 is None or float(best["f1"]) > float(best_f1["f1"]):
                    best_f1 = dict(best)
                    best_f1["radius"] = r
                    best_f1["tau_us"] = tau_us
        assert best_auc is not None and best_f1 is not None
        n149_csv = n149_dir / f"roc_n149_{scene}_{level}.csv"
        pd.DataFrame(n149_all_rows).to_csv(n149_csv, index=False)

        summary.append(
            {
                "scene": scene,
                "level": level,
                "events": int(ev.label.shape[0]),
                "signal": int(ev.label.sum()),
                "noise": int(ev.label.shape[0] - ev.label.sum()),
                "noise_per_signal": meta.get("noise_per_signal", ""),
                "estimated_noise_hz_per_pixel": meta.get("estimated_noise_hz_per_pixel", ""),
                "ebf_auc": ebf_best["auc"],
                "ebf_exact_auc": ebf_exact,
                "ebf_best_threshold": ebf_best["value"],
                "ebf_best_f1": ebf_best["f1"],
                "n149_best_auc": best_auc["auc"],
                "n149_best_auc_radius": best_auc["radius"],
                "n149_best_auc_tau_us": best_auc["tau_us"],
                "n149_best_auc_point_f1": best_auc["f1"],
                "n149_best_f1": best_f1["f1"],
                "n149_best_f1_radius": best_f1["radius"],
                "n149_best_f1_tau_us": best_f1["tau_us"],
                "n149_best_f1_auc": best_f1["auc"],
                "auc_delta_n149_minus_ebf": float(best_auc["auc"]) - float(ebf_best["auc"]),
                "f1_delta_n149_minus_ebf": float(best_f1["f1"]) - float(ebf_best["f1"]),
                "ebf_csv": str(ebf_csv),
                "n149_csv": str(n149_csv),
            }
        )
        print(summary[-1])

    out_root.mkdir(parents=True, exist_ok=True)
    summary_csv = out_root / "dvsclean_ebf_n149_summary.csv"
    pd.DataFrame(summary).to_csv(summary_csv, index=False)
    print(f"saved: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
