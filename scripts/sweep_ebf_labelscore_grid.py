from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
import re

import numpy as np

from myevs.timebase import TimeBase


@dataclass(frozen=True)
class LabeledEvents:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    p: np.ndarray
    label: np.ndarray


def _parse_int_list(s: str) -> list[int]:
    out: list[int] = []
    for part in str(s).replace(" ", "").split(","):
        if not part:
            continue
        out.append(int(part))
    return out


def load_labeled_npy(path: str, *, max_events: int = 0) -> LabeledEvents:
    arr = np.load(path, mmap_mode="r", allow_pickle=True)
    if max_events > 0:
        arr = arr[:max_events]

    if getattr(arr, "dtype", None) is not None and getattr(arr.dtype, "names", None):
        names = set(arr.dtype.names)
        need = {"t", "x", "y", "p", "label"}
        if not need.issubset(names):
            missing = sorted(need - names)
            raise SystemExit(f"input structured npy missing fields: {missing}")
        t = arr["t"].astype(np.uint64, copy=False)
        x = arr["x"].astype(np.int32, copy=False)
        y = arr["y"].astype(np.int32, copy=False)
        p = arr["p"].astype(np.int8, copy=False)
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
            t = a2[:, 1].astype(np.uint64, copy=False)
            y = a2[:, 2].astype(np.int32, copy=False)
            x = a2[:, 3].astype(np.int32, copy=False)
            p = a2[:, 4].astype(np.int8, copy=False)
        else:
            # [t, x, y, p, label]
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


def _try_build_numba_kernel():
    try:
        from numba import njit  # type: ignore
    except Exception:
        return None

    @njit(cache=True)
    def ebf_scores_stream(
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

        if rr <= 0 or tau <= 0:
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

                    if int(last_pol[idx]) != pi:
                        continue

                    ts = int(last_ts[idx])
                    if ts == 0:
                        continue

                    dt = (ti - ts) if ti >= ts else (ts - ti)
                    if dt > tau:
                        continue

                    score += (float(tau - dt) * inv_tau)

            last_ts[idx0] = np.uint64(ti)
            last_pol[idx0] = np.int8(pi)
            scores_out[i] = score

    return ebf_scores_stream


def score_stream_ebf(
    ev: LabeledEvents,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_us: int,
    tb: TimeBase,
    _kernel_cache: dict[str, object],
) -> np.ndarray:
    scores = np.empty((ev.t.shape[0],), dtype=np.float32)

    ker = _kernel_cache.get("ker")
    if ker is None:
        ker = _try_build_numba_kernel()
        _kernel_cache["ker"] = ker

    if ker is not None:
        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            last_ts,
            last_pol,
            scores,
        )
        return scores

    from myevs.denoise.ops.base import Dims
    from myevs.denoise.ops.ebf import EbfOp
    from myevs.denoise.types import DenoiseConfig

    cfg = DenoiseConfig(
        method="ebf",
        pipeline=None,
        time_window_us=int(tau_us),
        radius_px=int(radius_px),
        min_neighbors=0.0,
        refractory_us=0,
        show_on=True,
        show_off=True,
    )
    op = EbfOp(Dims(width=int(width), height=int(height)), cfg, tb)
    n = int(ev.t.shape[0])
    for i in range(n):
        scores[i] = float(op.score(int(ev.x[i]), int(ev.y[i]), int(ev.p[i]), int(ev.t[i])))
        if (i + 1) % 500000 == 0:
            print(f"scored: {i+1}/{n} (r={radius_px}, tau_us={tau_us})")
    return scores


def _read_existing_tags(out_path: str) -> set[str]:
    if not os.path.exists(out_path):
        return set()
    try:
        with open(out_path, "r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            if r.fieldnames is None:
                return set()
            tags: set[str] = set()
            for row in r:
                t = (row.get("tag") or "").strip()
                auc = (row.get("auc") or "").strip()
                if t and auc:
                    tags.add(t)
            return tags
    except Exception:
        return set()


ROC_HEADER = [
    "tag",
    "method",
    "param",
    "value",
    "roc_convention",
    "match_us",
    "events_total",
    "signal_total",
    "noise_total",
    "events_kept",
    "signal_kept",
    "noise_kept",
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


def _roc_points_from_scores(
    y_true01: np.ndarray,
    y_score: np.ndarray,
    *,
    max_points: int,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (auc, thresholds, tp, fp, fpr, tpr) using standard ROC construction.

    Convention here matches myevs.metrics.roc_score_label:
    - Sort by score descending
    - Predicted positive for threshold thr: score >= thr
    - Adds the conventional starting point (0,0) at threshold=+inf
    """

    y = np.asarray(y_true01).astype(np.int8, copy=False)
    s = np.asarray(y_score).astype(np.float64, copy=False)
    if y.ndim != 1 or s.ndim != 1 or y.shape[0] != s.shape[0]:
        raise ValueError("y_true and y_score must be 1D arrays of the same length")

    n = int(y.shape[0])
    pos = int(np.sum(y))
    neg = int(n - pos)
    if n == 0 or pos == 0 or neg == 0:
        thr = np.asarray([np.inf, -np.inf], dtype=np.float64)
        tp = np.asarray([0, pos], dtype=np.int64)
        fp = np.asarray([0, neg], dtype=np.int64)
        fpr = np.asarray([0.0, 1.0], dtype=np.float64)
        tpr = np.asarray([0.0, 1.0], dtype=np.float64)
        auc = 0.0
        return auc, thr, tp, fp, fpr, tpr

    order = np.argsort(-s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order]

    tp_cum = np.cumsum(y_sorted, dtype=np.int64)
    fp_cum = np.cumsum(1 - y_sorted, dtype=np.int64)

    change = np.empty((n,), dtype=bool)
    change[:-1] = s_sorted[:-1] != s_sorted[1:]
    change[-1] = True
    idx = np.nonzero(change)[0]

    tp_u = tp_cum[idx]
    fp_u = fp_cum[idx]
    thr_u = s_sorted[idx].astype(np.float64, copy=False)

    # add start point (0,0) at +inf
    tp_u = np.concatenate([np.asarray([0], dtype=np.int64), tp_u])
    fp_u = np.concatenate([np.asarray([0], dtype=np.int64), fp_u])
    thr_u = np.concatenate([np.asarray([np.inf], dtype=np.float64), thr_u])

    tpr_u = tp_u.astype(np.float64) / float(pos)
    fpr_u = fp_u.astype(np.float64) / float(neg)

    # exact AUC from full curve
    auc = float((getattr(np, "trapezoid", None) or np.trapz)(y=tpr_u, x=fpr_u))

    # downsample for output if needed
    if max_points is not None and int(max_points) > 0 and fpr_u.shape[0] > int(max_points):
        m = int(max_points)
        keep = np.unique(
            np.concatenate(
                [
                    np.asarray([0, fpr_u.shape[0] - 1], dtype=np.int64),
                    np.linspace(0, fpr_u.shape[0] - 1, num=m, dtype=np.int64),
                ]
            )
        )
        thr_u = thr_u[keep]
        tp_u = tp_u[keep]
        fp_u = fp_u[keep]
        fpr_u = fpr_u[keep]
        tpr_u = tpr_u[keep]

    return auc, thr_u, tp_u, fp_u, fpr_u, tpr_u


def _write_roc_rows(
    writer: csv.writer,
    *,
    tag: str,
    method: str,
    param: str,
    thresholds: np.ndarray,
    tp: np.ndarray,
    fp: np.ndarray,
    pos: int,
    neg: int,
    auc: float,
) -> None:
    n = int(pos + neg)
    for i in range(int(thresholds.shape[0])):
        thr = float(thresholds[i])
        tp_i = int(tp[i])
        fp_i = int(fp[i])
        tn_i = int(neg - fp_i)
        fn_i = int(pos - tp_i)

        events_kept = tp_i + fp_i
        signal_kept = tp_i
        noise_kept = fp_i

        tpr = (tp_i / pos) if pos > 0 else 0.0
        fpr = (fp_i / neg) if neg > 0 else 0.0

        prec_den = tp_i + fp_i
        precision = (tp_i / prec_den) if prec_den > 0 else 0.0
        acc = ((tp_i + tn_i) / n) if n > 0 else 0.0
        f1_den = precision + tpr
        f1 = (2.0 * precision * tpr / f1_den) if f1_den > 0 else 0.0

        writer.writerow(
            [
                tag,
                method,
                param,
                thr,
                "paper",
                0,
                n,
                int(pos),
                int(neg),
                int(events_kept),
                int(signal_kept),
                int(noise_kept),
                int(tp_i),
                int(fp_i),
                int(tn_i),
                int(fn_i),
                float(tpr),
                float(fpr),
                float(precision),
                float(acc),
                float(f1),
                float(auc),
            ]
        )


def _plot_roc_png(*, csv_path: str, png_path: str, title: str) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        print(f"skip plot (matplotlib unavailable): {type(e).__name__}: {e}")
        return

    # Load ROC CSV and plot by tag
    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    if not rows:
        print(f"skip plot (empty csv): {csv_path}")
        return

    tags = sorted({(row.get("tag") or "") for row in rows if (row.get("tag") or "")})
    if not tags:
        print(f"skip plot (no tags): {csv_path}")
        return

    def _legend_label(tag: str) -> str:
        # tag example: ebf_labelscore_r2_tau16000_light
        m = re.search(r"_r(\d+)_tau(\d+)_", tag)
        r_s = m.group(1) if m else "?"
        tau_s = m.group(2) if m else "?"

        auc_val: float | None = None
        for row in rows:
            if row.get("tag") != tag:
                continue
            a = (row.get("auc") or "").strip()
            if a:
                try:
                    auc_val = float(a)
                except Exception:
                    auc_val = None
                break

        if auc_val is None:
            return f"ebf_r{r_s} tau{tau_s}"
        return f"ebf_r{r_s} tau{tau_s} (AUC={auc_val:.4f})"

    plt.figure(figsize=(8, 6), dpi=160)
    for tag in tags:
        fpr = [float(row["fpr"]) for row in rows if row.get("tag") == tag]
        tpr = [float(row["tpr"]) for row in rows if row.get("tag") == tag]
        if not fpr:
            continue
        plt.plot(fpr, tpr, linewidth=1.0, label=_legend_label(tag))

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
    ap = argparse.ArgumentParser(description="Sweep EBF AUC(score+label) on labeled .npy (no CLI modifications).")
    ap.add_argument("--max-events", type=int, default=int(os.environ.get("EBF_MAX_EVENTS", "0")), help="0=all")
    ap.add_argument("--out-dir", default="data/mydriving/EBF", help="output directory")
    ap.add_argument(
        "--plot-only",
        action="store_true",
        help="Only regenerate PNG from existing ROC CSV (useful when CSV is open/locked on Windows).",
    )
    ap.add_argument("--width", type=int, default=346)
    ap.add_argument("--height", type=int, default=260)
    ap.add_argument("--tick-ns", type=float, default=1000.0)
    ap.add_argument("--r-list", default="1,2,3")
    ap.add_argument("--tau-us-list", default="8000,16000,32000,64000,128000")
    ap.add_argument("--roc-max-points", type=int, default=5000)

    ap.add_argument(
        "--light",
        default=r"D:\hjx_workspace\scientific_reserach\dataset\DND21\mydriving\driving_noise_light_slomo_shot_withlabel\driving_noise_light_labeled.npy",
    )
    ap.add_argument(
        "--mid",
        default=r"D:\hjx_workspace\scientific_reserach\dataset\DND21\mydriving\driving_noise_mid_slomo_shot_withlabel\driving_noise_mid_labeled.npy",
    )
    ap.add_argument(
        "--heavy",
        default=r"D:\hjx_workspace\scientific_reserach\dataset\DND21\mydriving\driving_noise_heavy_slomo_shot_withlabel\driving_noise_heavy_labeled.npy",
    )

    args = ap.parse_args()

    tb = TimeBase(tick_ns=float(args.tick_ns))
    radius_list = _parse_int_list(args.r_list)
    tau_us_list = _parse_int_list(args.tau_us_list)

    env_inputs = {
        "light": str(args.light),
        "mid": str(args.mid),
        "heavy": str(args.heavy),
    }

    out_dir = str(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Per-env ROC outputs (same header/format as data/mydriving/*/roc_*.csv)
    roc_csv = {
        env: os.path.join(out_dir, f"roc_ebf_{env}_labelscore_r123_tau8_16_32_64_128ms.csv") for env in env_inputs
    }
    roc_png = {
        env: os.path.join(out_dir, f"roc_ebf_{env}_labelscore_r123_tau8_16_32_64_128ms.png") for env in env_inputs
    }

    if bool(args.plot_only):
        for env in ("light", "mid", "heavy"):
            if not os.path.exists(roc_csv[env]):
                print(f"skip plot-only (missing csv): env={env} path={roc_csv[env]}")
                continue
            _plot_roc_png(
                csv_path=roc_csv[env],
                png_path=roc_png[env],
                title=f"EBF ROC ({env}) labelscore: r in {radius_list}, tau in {tau_us_list}us",
            )
            print(f"saved: {roc_png[env]}")
        return 0

    kernel_cache: dict[str, object] = {}

    best_global = ("", -1.0)
    best_by_env: dict[str, tuple[str, float]] = {"light": ("", -1.0), "mid": ("", -1.0), "heavy": ("", -1.0)}

    # Start fresh ROC csv (these files are large; resuming is less useful)
    write_enabled: dict[str, bool] = {}
    for env, p in roc_csv.items():
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
        try:
            with open(p, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(ROC_HEADER)
            write_enabled[env] = True
        except PermissionError:
            if os.path.exists(p):
                print(f"warn: cannot write (locked). env={env} path={p} -> skip recompute, plot only")
                write_enabled[env] = False
            else:
                raise

    for env, in_path in env_inputs.items():
        if not write_enabled.get(env, True):
            _plot_roc_png(
                csv_path=roc_csv[env],
                png_path=roc_png[env],
                title=f"EBF ROC ({env}) labelscore: r in {radius_list}, tau in {tau_us_list}us",
            )
            print(f"saved: {roc_png[env]}")
            continue

        ev = load_labeled_npy(in_path, max_events=int(args.max_events))
        n = int(ev.label.shape[0])
        pos = int(np.sum(ev.label))
        neg = int(n - pos)
        print(f"loaded: env={env} n={n} pos={pos} neg={neg} in={in_path}")

        with open(roc_csv[env], "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)

            for r in radius_list:
                for tau_us in tau_us_list:
                    tag = f"ebf_labelscore_r{r}_tau{tau_us}_{env}"

                    scores = score_stream_ebf(
                        ev,
                        width=int(args.width),
                        height=int(args.height),
                        radius_px=int(r),
                        tau_us=int(tau_us),
                        tb=tb,
                        _kernel_cache=kernel_cache,
                    )

                    auc, thr, tp, fp, _fpr, _tpr = _roc_points_from_scores(
                        ev.label,
                        scores,
                        max_points=int(args.roc_max_points),
                    )

                    if auc > best_global[1]:
                        best_global = (tag, float(auc))
                    if auc > best_by_env[env][1]:
                        best_by_env[env] = (tag, float(auc))

                    _write_roc_rows(
                        w,
                        tag=tag,
                        method="ebf",
                        param="min-neighbors",
                        thresholds=thr,
                        tp=tp,
                        fp=fp,
                        pos=pos,
                        neg=neg,
                        auc=float(auc),
                    )

                    print(f"auc={auc:.6f} env={env} r={r} tau_us={tau_us} points={int(thr.shape[0])}")

        # plot per-env png
        _plot_roc_png(
            csv_path=roc_csv[env],
            png_path=roc_png[env],
            title=f"EBF ROC ({env}) labelscore: r in {radius_list}, tau in {tau_us_list}us",
        )

    print("=== BEST (by env) ===")
    for env in ("light", "mid", "heavy"):
        tag, auc = best_by_env[env]
        if auc >= 0:
            print(f"{env}: {tag} auc={auc:.6f}")
    print("=== BEST (global) ===")
    print(f"{best_global[0]} auc={best_global[1]:.6f}")
    for env in ("light", "mid", "heavy"):
        print(f"saved: {roc_csv[env]}")
        print(f"saved: {roc_png[env]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
