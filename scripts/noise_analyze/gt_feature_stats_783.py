from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass

import numpy as np

try:
    import numba
except Exception:  # pragma: no cover
    numba = None


@dataclass(frozen=True)
class LabeledEvents:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    p: np.ndarray
    label: np.ndarray


def _require_numba() -> None:
    if numba is None:
        raise SystemExit("gt_feature_stats_783 requires numba")


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


@numba.njit(cache=True)
def _kernel_783(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    label: np.ndarray,
    width: int,
    height: int,
    tau_ticks: int,
    radius: int,
    score_thr: float,
    out_a: np.ndarray,
    out_cls: np.ndarray,
) -> int:
    n = int(t.shape[0])
    w = int(width)
    h = int(height)
    tau = int(tau_ticks)
    rr = int(radius)
    if rr < 0:
        rr = 0
    if rr > 8:
        rr = 8
    if tau <= 0:
        tau = 1

    pix = h * w
    last_ts = np.zeros((pix,), dtype=np.uint64)
    last_pol = np.zeros((pix,), dtype=np.int8)

    k = 0
    inv_tau = 1.0 / float(tau)
    for i in range(n):
        xi = int(x[i])
        yi = int(y[i])
        if xi < 0 or xi >= w or yi < 0 or yi >= h:
            continue

        ti = int(t[i])
        pi = int(p[i])

        x0 = xi - rr
        if x0 < 0:
            x0 = 0
        x1 = xi + rr
        if x1 >= w:
            x1 = w - 1
        y0 = yi - rr
        if y0 < 0:
            y0 = 0
        y1 = yi + rr
        if y1 >= h:
            y1 = h - 1

        s_base = 0.0
        e0 = 0.0
        e90 = 0.0
        e45 = 0.0
        e135 = 0.0

        for yy in range(y0, y1 + 1):
            base = yy * w
            dy = yy - yi
            for xx in range(x0, x1 + 1):
                if xx == xi and yy == yi:
                    continue

                idx = base + xx
                ts = int(last_ts[idx])
                if ts == 0:
                    continue
                if int(last_pol[idx]) != pi:
                    continue

                dt = ti - ts
                if dt < 0:
                    dt = -dt
                if dt > tau:
                    continue

                wt = 1.0 - float(dt) * inv_tau
                if wt <= 0.0:
                    continue
                s_base += wt

                dx = xx - xi
                if dy == 0:
                    e0 += wt
                if dx == 0:
                    e90 += wt
                if dx == dy:
                    e45 += wt
                if dx == -dy:
                    e135 += wt

        if s_base >= score_thr:
            emax = e0
            if e90 > emax:
                emax = e90
            if e45 > emax:
                emax = e45
            if e135 > emax:
                emax = e135
            esum = e0 + e90 + e45 + e135
            a = (4.0 * emax) / (esum + 1e-3)
            if a < 0.0:
                a = 0.0
            if a > 4.0:
                a = 4.0
            out_a[k] = a
            out_cls[k] = int(label[i])
            k += 1

        idx0 = yi * w + xi
        last_ts[idx0] = np.uint64(ti)
        last_pol[idx0] = np.int8(pi)

    return k


def _safe_stats(v: np.ndarray) -> tuple[float, float, float, float]:
    if v.size <= 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    return float(np.mean(v)), float(np.std(v)), float(np.quantile(v, 0.5)), float(np.quantile(v, 0.9))


def run_783(
    *,
    labeled_npy: str,
    out_dir: str,
    max_events: int,
    tau_us: int,
    radius: int,
    score_thr: float,
) -> None:
    _require_numba()
    ev = load_labeled_npy(labeled_npy, max_events=max_events)

    w = int(ev.x.max()) + 1
    h = int(ev.y.max()) + 1

    n = int(ev.t.shape[0])
    out_a = np.zeros((n,), dtype=np.float32)
    out_cls = np.zeros((n,), dtype=np.int8)

    k = _kernel_783(
        ev.t,
        ev.x,
        ev.y,
        ev.p,
        ev.label,
        w,
        h,
        int(tau_us),
        int(radius),
        float(score_thr),
        out_a,
        out_cls,
    )

    a = out_a[:k].astype(np.float64)
    cls = out_cls[:k].astype(np.int8)

    os.makedirs(out_dir, exist_ok=True)
    summary_csv = os.path.join(out_dir, "summary.csv")
    hist_csv = os.path.join(out_dir, "hist.csv")
    summary_json = os.path.join(out_dir, "summary.json")

    rows_summary: list[dict[str, object]] = []
    rows_hist: list[dict[str, object]] = []

    bins = np.linspace(0.0, 4.0, 21)

    for cls_name, cval in (("noise", 0), ("signal", 1)):
        m = cls == cval
        v = a[m]
        cnt = int(v.size)
        mean, std, p50, p90 = _safe_stats(v)
        l1_2 = float(np.mean((v >= 1.0) & (v < 2.0))) if cnt > 0 else float("nan")
        r2p5_4 = float(np.mean((v >= 2.5) & (v <= 4.0))) if cnt > 0 else float("nan")

        rows_summary.append(
            {
                "class": cls_name,
                "events_highscore": cnt,
                "a_score_mean": mean,
                "a_score_std": std,
                "a_score_p50": p50,
                "a_score_p90": p90,
                "a_score_1_2_ratio": l1_2,
                "a_score_2p5_4_ratio": r2p5_4,
            }
        )

        hist, edges = np.histogram(v, bins=bins)
        for i in range(hist.size):
            c = int(hist[i])
            rows_hist.append(
                {
                    "feature": "a_score",
                    "class": cls_name,
                    "bin_lo": float(edges[i]),
                    "bin_hi": float(edges[i + 1]),
                    "count": c,
                    "ratio": (float(c) / float(cnt)) if cnt > 0 else float("nan"),
                }
            )

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows_summary[0].keys()))
        wcsv.writeheader()
        for r in rows_summary:
            wcsv.writerow(r)

    with open(hist_csv, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows_hist[0].keys()))
        wcsv.writeheader()
        for r in rows_hist:
            wcsv.writerow(r)

    payload = {
        "config": {
            "labeled_npy": labeled_npy,
            "max_events": int(max_events),
            "tau_us": int(tau_us),
            "radius": int(radius),
            "score_thr": float(score_thr),
            "width": int(w),
            "height": int(h),
            "events_total": int(n),
            "events_highscore_total": int(k),
        },
        "summary": rows_summary,
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"saved: {summary_csv}")
    print(f"saved: {hist_csv}")
    print(f"saved: {summary_json}")


def main() -> int:
    ap = argparse.ArgumentParser(description="7.83 stats: high-score anisotropy A_score histogram")
    ap.add_argument(
        "--labeled-npy",
        default="D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3.npy",
    )
    ap.add_argument("--out-dir", default="data/ED24/myPedestrain_06/EBF_Part2/_gt_feature_stats_783_heavy")
    ap.add_argument("--max-events", type=int, default=500000)
    ap.add_argument("--tau-us", type=int, default=30000)
    ap.add_argument("--radius", type=int, default=4)
    ap.add_argument("--score-thr", type=float, default=3.0)
    args = ap.parse_args()

    run_783(
        labeled_npy=str(args.labeled_npy),
        out_dir=str(args.out_dir),
        max_events=int(args.max_events),
        tau_us=int(args.tau_us),
        radius=int(args.radius),
        score_thr=float(args.score_thr),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
