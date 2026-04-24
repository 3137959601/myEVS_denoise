from __future__ import annotations

import argparse
import csv
import os
import re
import time
from dataclasses import dataclass

import numpy as np

from myevs.denoise.ops.ebfopt_part2.n149_n145_s52_euclid_compactlut_backbone import score_stream_n149
from myevs.timebase import TimeBase


@dataclass(frozen=True)
class LabeledEvents:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    p: np.ndarray
    label: np.ndarray


def _parse_int_list(text: str) -> list[int]:
    out: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(float(part)))
    return out


def load_labeled_npy(path: str, *, max_events: int = 0) -> LabeledEvents:
    arr = np.load(path, mmap_mode="r", allow_pickle=True)
    if max_events > 0:
        arr = arr[:max_events]

    if getattr(arr, "dtype", None) is not None and getattr(arr.dtype, "names", None):
        names = set(arr.dtype.names)
        need = {"t", "x", "y", "p", "label"}
        if not need.issubset(names):
            raise SystemExit(f"missing fields in {path}: {sorted(need - names)}")
        t = arr["t"].astype(np.uint64, copy=False)
        x = arr["x"].astype(np.int32, copy=False)
        y = arr["y"].astype(np.int32, copy=False)
        p = arr["p"].astype(np.int8, copy=False)
        label = arr["label"].astype(np.int8, copy=False)
    else:
        a2 = np.asarray(arr)
        if a2.ndim != 2 or a2.shape[1] < 5:
            raise SystemExit(f"invalid npy shape: {a2.shape}")
        c0 = a2[: min(10000, a2.shape[0]), 0]
        is_bin0 = bool(np.all((c0 == 0) | (c0 == 1)))
        if is_bin0:
            label = a2[:, 0].astype(np.int8, copy=False)
            t = a2[:, 1].astype(np.uint64, copy=False)
            y = a2[:, 2].astype(np.int32, copy=False)
            x = a2[:, 3].astype(np.int32, copy=False)
            p = a2[:, 4].astype(np.int8, copy=False)
        else:
            t = a2[:, 0].astype(np.uint64, copy=False)
            x = a2[:, 1].astype(np.int32, copy=False)
            y = a2[:, 2].astype(np.int32, copy=False)
            p = a2[:, 3].astype(np.int8, copy=False)
            label = a2[:, 4].astype(np.int8, copy=False)

    label = (label > 0).astype(np.int8, copy=False)
    return LabeledEvents(
        t=np.ascontiguousarray(t),
        x=np.ascontiguousarray(x),
        y=np.ascontiguousarray(y),
        p=np.ascontiguousarray(p),
        label=np.ascontiguousarray(label),
    )


def roc_points(y_true01: np.ndarray, y_score: np.ndarray, *, max_points: int = 4000):
    y = np.asarray(y_true01).astype(np.int8, copy=False)
    s = np.asarray(y_score).astype(np.float64, copy=False)

    n = int(y.shape[0])
    pos = int(np.sum(y))
    neg = int(n - pos)
    if n == 0 or pos == 0 or neg == 0:
        thr = np.asarray([np.inf, -np.inf], dtype=np.float64)
        tp = np.asarray([0, pos], dtype=np.int64)
        fp = np.asarray([0, neg], dtype=np.int64)
        fpr = np.asarray([0.0, 1.0], dtype=np.float64)
        tpr = np.asarray([0.0, 1.0], dtype=np.float64)
        return 0.0, thr, tp, fp, fpr, tpr

    order = np.argsort(-s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order]
    tp_cum = np.cumsum(y_sorted, dtype=np.int64)
    fp_cum = np.cumsum(1 - y_sorted, dtype=np.int64)

    change = np.empty((n,), dtype=bool)
    change[:-1] = s_sorted[:-1] != s_sorted[1:]
    change[-1] = True
    idx = np.nonzero(change)[0]

    tp_u = np.concatenate([np.asarray([0], dtype=np.int64), tp_cum[idx]])
    fp_u = np.concatenate([np.asarray([0], dtype=np.int64), fp_cum[idx]])
    thr_u = np.concatenate([np.asarray([np.inf], dtype=np.float64), s_sorted[idx].astype(np.float64, copy=False)])

    tpr_u = tp_u.astype(np.float64) / float(pos)
    fpr_u = fp_u.astype(np.float64) / float(neg)
    auc = float((getattr(np, "trapezoid", None) or np.trapz)(y=tpr_u, x=fpr_u))

    if max_points > 0 and thr_u.shape[0] > int(max_points):
        keep = np.unique(
            np.concatenate(
                [
                    np.asarray([0, thr_u.shape[0] - 1], dtype=np.int64),
                    np.linspace(0, thr_u.shape[0] - 1, num=int(max_points), dtype=np.int64),
                ]
            )
        )
        thr_u = thr_u[keep]
        tp_u = tp_u[keep]
        fp_u = fp_u[keep]
        fpr_u = fpr_u[keep]
        tpr_u = tpr_u[keep]

    return auc, thr_u, tp_u, fp_u, fpr_u, tpr_u


def _best_f1_index(tp: np.ndarray, fp: np.ndarray, *, pos: int, neg: int) -> int:
    best_i = 0
    best_key = (-1.0, 0.0, 0.0, -1.0)
    for i in range(int(tp.shape[0])):
        tp_i = int(tp[i])
        fp_i = int(fp[i])
        tpr = (tp_i / pos) if pos > 0 else 0.0
        fpr = (fp_i / neg) if neg > 0 else 0.0
        prec_den = tp_i + fp_i
        precision = (tp_i / prec_den) if prec_den > 0 else 0.0
        f1_den = precision + tpr
        f1 = (2.0 * precision * tpr / f1_den) if f1_den > 0 else 0.0
        key = (float(f1), float(tpr), float(precision), -float(fpr))
        if key > best_key:
            best_key = key
            best_i = i
    return int(best_i)


def _plot_roc(csv_path: str, png_path: str, title: str) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return
    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    tags = sorted({(r.get("tag") or "") for r in rows if (r.get("tag") or "")})
    if not tags:
        return

    def _tag_without_env(tag_text: str) -> str:
        return re.sub(r"_(light|mid|heavy)$", "", str(tag_text), flags=re.IGNORECASE)

    auc_by_tag: dict[str, float] = {}
    for r in rows:
        t = str(r.get("tag", ""))
        if not t or t in auc_by_tag:
            continue
        try:
            auc_by_tag[t] = float(r.get("auc", "nan"))
        except Exception:
            auc_by_tag[t] = float("nan")

    plt.figure(figsize=(8, 6), dpi=160)
    for tag in tags:
        fpr = [float(r["fpr"]) for r in rows if r.get("tag") == tag]
        tpr = [float(r["tpr"]) for r in rows if r.get("tag") == tag]
        if fpr:
            auc = auc_by_tag.get(tag, float("nan"))
            short_tag = _tag_without_env(tag)
            if np.isfinite(auc):
                label = f"{short_tag} | AUC={auc:.6f}"
            else:
                label = short_tag
            plt.plot(fpr, tpr, linewidth=1.0, label=label)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(title)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(False)
    plt.legend(fontsize=7, ncol=1)
    os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="ED24 N149 ROC sweep (label exact).")
    ap.add_argument("--max-events", type=int, default=0, help="0=all")
    ap.add_argument("--width", type=int, default=346)
    ap.add_argument("--height", type=int, default=260)
    ap.add_argument("--tick-ns", type=float, default=1000.0)
    ap.add_argument("--radius-list", default="1,2,3,4")
    ap.add_argument("--tau-us-list", default="1000,2000,4000")
    ap.add_argument("--out-dir", default="data/ED24/myPedestrain_06/N149")
    ap.add_argument("--light", default=r"D:\hjx_workspace\scientific_reserach\dataset\ED24\myPedestrain_06\Pedestrain_06_1.8.npy")
    ap.add_argument("--mid", default=r"D:\hjx_workspace\scientific_reserach\dataset\ED24\myPedestrain_06\Pedestrain_06_2.5.npy")
    ap.add_argument("--heavy", default=r"D:\hjx_workspace\scientific_reserach\dataset\ED24\myPedestrain_06\Pedestrain_06_3.3.npy")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tb = TimeBase(tick_ns=float(args.tick_ns))
    radius_list = _parse_int_list(args.radius_list)
    tau_list = _parse_int_list(args.tau_us_list)

    env_inputs = {"light": args.light, "mid": args.mid, "heavy": args.heavy}
    runtime_rows: list[dict[str, float | str]] = []

    header = [
        "tag",
        "method",
        "param",
        "value",
        "events_total",
        "signal_total",
        "noise_total",
        "tp",
        "fp",
        "tn",
        "fn",
        "tpr",
        "fpr",
        "precision",
        "accuracy",
        "f1",
        "auc",
    ]

    for env, path in env_inputs.items():
        t0 = time.perf_counter()
        ev = load_labeled_npy(path, max_events=int(args.max_events))
        n = int(ev.label.shape[0])
        pos = int(np.sum(ev.label))
        neg = int(n - pos)
        print(f"[n149] env={env} n={n} pos={pos} neg={neg}")

        out_csv = os.path.join(args.out_dir, f"roc_n149_{env}.csv")
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)

            for r in radius_list:
                for tau_us in tau_list:
                    tag = f"n149_r{int(r)}_tau{int(tau_us)}_{env}"
                    scores = score_stream_n149(
                        ev,
                        width=int(args.width),
                        height=int(args.height),
                        radius_px=int(r),
                        tau_us=int(tau_us),
                        tb=tb,
                    )
                    auc, thr, tp, fp, _fpr, _tpr = roc_points(ev.label, scores, max_points=4000)
                    for i in range(int(thr.shape[0])):
                        thr_i = float(thr[i])
                        tp_i = int(tp[i])
                        fp_i = int(fp[i])
                        tn_i = int(neg - fp_i)
                        fn_i = int(pos - tp_i)
                        tpr = (tp_i / pos) if pos > 0 else 0.0
                        fpr = (fp_i / neg) if neg > 0 else 0.0
                        prec_den = tp_i + fp_i
                        precision = (tp_i / prec_den) if prec_den > 0 else 0.0
                        acc = ((tp_i + tn_i) / n) if n > 0 else 0.0
                        f1_den = precision + tpr
                        f1 = (2.0 * precision * tpr / f1_den) if f1_den > 0 else 0.0
                        w.writerow(
                            [
                                tag,
                                "n149",
                                "score-threshold",
                                thr_i,
                                n,
                                pos,
                                neg,
                                tp_i,
                                fp_i,
                                tn_i,
                                fn_i,
                                tpr,
                                fpr,
                                precision,
                                acc,
                                f1,
                                auc,
                            ]
                        )

                    best_i = _best_f1_index(tp, fp, pos=pos, neg=neg)
                    tp_b = int(tp[best_i])
                    fp_b = int(fp[best_i])
                    rec_b = (tp_b / pos) if pos > 0 else 0.0
                    prec_b_den = tp_b + fp_b
                    prec_b = (tp_b / prec_b_den) if prec_b_den > 0 else 0.0
                    f1_b_den = prec_b + rec_b
                    f1_b = (2.0 * prec_b * rec_b / f1_b_den) if f1_b_den > 0 else 0.0
                    print(
                        f"  tag={tag} auc={auc:.6f} best_f1={f1_b:.6f}"
                    )

        out_png = os.path.join(args.out_dir, f"roc_n149_{env}.png")
        _plot_roc(out_csv, out_png, "N149 ROC")
        print(f"saved: {out_csv}")
        print(f"saved: {out_png}")
        t1 = time.perf_counter()
        runtime_rows.append(
            {
                "algorithm": "n149",
                "level": env,
                "elapsed_sec": float(t1 - t0),
            }
        )

    runtime_csv = os.path.join(args.out_dir, "runtime_n149.csv")
    with open(runtime_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["algorithm", "level", "elapsed_sec"])
        w.writeheader()
        for row in runtime_rows:
            w.writerow(row)
    print(f"saved: {runtime_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
