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
        raise SystemExit("gt_feature_stats_779_score_bandpass requires numba")


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


def _parse_edges(text: str) -> np.ndarray:
    parts = [x.strip() for x in str(text).split(",") if x.strip()]
    vals = []
    for p in parts:
        vals.append(float(p))
    if len(vals) < 2:
        raise SystemExit("--score-edges requires at least 2 numbers")
    arr = np.asarray(vals, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise SystemExit("--score-edges contains non-finite value")
    if np.any(arr[1:] <= arr[:-1]):
        raise SystemExit("--score-edges must be strictly increasing")
    if arr[0] < 0.0:
        raise SystemExit("--score-edges must start from >= 0")
    return arr


@numba.njit(cache=True)
def _fill_hist_edges(v: float, edges: np.ndarray, hist: np.ndarray) -> None:
    n = int(hist.shape[0])
    if n <= 0:
        return
    if not (v >= 0.0):
        return
    b = n - 1
    for i in range(n):
        lo = edges[i]
        hi = edges[i + 1]
        if v >= lo and v < hi:
            b = i
            break
    hist[b] += 1


@numba.njit(cache=True)
def _kernel_sbase_hist(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    label: np.ndarray,
    width: int,
    height: int,
    tau_ticks: int,
    score_edges: np.ndarray,
    hist_noise: np.ndarray,
    hist_signal: np.ndarray,
    sum_score: np.ndarray,
    sqsum_score: np.ndarray,
    count_score: np.ndarray,
):
    n = int(t.shape[0])
    w = int(width)
    h = int(height)
    tau = int(tau_ticks)

    last_ts = np.zeros((h * w,), dtype=np.uint64)
    last_pol = np.zeros((h * w,), dtype=np.int8)

    for i in range(n):
        xi = int(x[i])
        yi = int(y[i])
        if xi < 0 or xi >= w or yi < 0 or yi >= h:
            continue

        ti = int(t[i])
        pi = int(p[i])
        cls = int(label[i])

        x0 = xi - 4
        if x0 < 0:
            x0 = 0
        x1 = xi + 4
        if x1 >= w:
            x1 = w - 1
        y0 = yi - 4
        if y0 < 0:
            y0 = 0
        y1 = yi + 4
        if y1 >= h:
            y1 = h - 1

        s_base = 0.0
        for yy in range(y0, y1 + 1):
            base = yy * w
            for xx in range(x0, x1 + 1):
                if xx == xi and yy == yi:
                    continue
                idx = base + xx
                ts = int(last_ts[idx])
                if ts == 0 or ts >= ti:
                    continue
                if int(last_pol[idx]) != pi:
                    continue
                dt_ticks = ti - ts
                if dt_ticks <= tau:
                    w_time = 1.0 - float(dt_ticks) / float(tau)
                    if w_time > 0.0:
                        s_base += w_time

        if cls == 0:
            _fill_hist_edges(s_base, score_edges, hist_noise)
        else:
            _fill_hist_edges(s_base, score_edges, hist_signal)

        count_score[cls] += 1
        sum_score[cls] += s_base
        sqsum_score[cls] += s_base * s_base

        idx0 = yi * w + xi
        last_ts[idx0] = np.uint64(ti)
        last_pol[idx0] = np.int8(pi)


def _safe_ratio(a: float, b: float) -> float:
    if b <= 0:
        return float("nan")
    return float(a) / float(b)


def _mean_std(sum_v: float, sqsum_v: float, n: int) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    m = sum_v / float(n)
    v = sqsum_v / float(n) - m * m
    if np.isfinite(v) and v < 0.0:
        v = 0.0
    s = float(np.sqrt(v)) if np.isfinite(v) else float("nan")
    return float(m), float(s)


def run_stats(
    *,
    labeled_npy: str,
    width: int,
    height: int,
    tick_ns: float,
    max_events: int,
    tau_us: int,
    score_edges: np.ndarray,
) -> tuple[list[dict], list[dict], dict]:
    ev = load_labeled_npy(str(labeled_npy), max_events=int(max_events))
    tick_us = float(tick_ns) / 1000.0
    if tick_us <= 0:
        tick_us = 1.0
    tau_ticks = int(round(float(tau_us) / tick_us))

    bins = int(score_edges.shape[0] - 1)
    h_noise = np.zeros((bins,), dtype=np.int64)
    h_signal = np.zeros((bins,), dtype=np.int64)
    sum_score = np.zeros((2,), dtype=np.float64)
    sqsum_score = np.zeros((2,), dtype=np.float64)
    count_score = np.zeros((2,), dtype=np.int64)

    _kernel_sbase_hist(
        ev.t,
        ev.x,
        ev.y,
        ev.p,
        ev.label,
        int(width),
        int(height),
        int(tau_ticks),
        score_edges,
        h_noise,
        h_signal,
        sum_score,
        sqsum_score,
        count_score,
    )

    summary_rows: list[dict] = []
    for cidx, cname in ((0, "noise"), (1, "signal")):
        n = int(count_score[cidx])
        m, s = _mean_std(float(sum_score[cidx]), float(sqsum_score[cidx]), n)
        summary_rows.append(
            {
                "class": cname,
                "count": n,
                "mean_s_base": m,
                "std_s_base": s,
            }
        )

    hist_rows: list[dict] = []
    for cname, hh in (("noise", h_noise), ("signal", h_signal)):
        n_all = int(np.sum(hh))
        for bi in range(int(hh.shape[0])):
            lo = float(score_edges[bi])
            hi = float(score_edges[bi + 1])
            c = int(hh[bi])
            hist_rows.append(
                {
                    "feature": "s_base",
                    "class": cname,
                    "bin_index": int(bi),
                    "bin_lo": lo,
                    "bin_hi": hi,
                    "count": c,
                    "ratio": _safe_ratio(c, n_all),
                }
            )

    meta = {
        "input": str(labeled_npy),
        "events": int(ev.t.shape[0]),
        "tau_us": int(tau_us),
        "window": "9x9",
        "score_edges": [float(x) for x in np.asarray(score_edges, dtype=np.float64)],
        "score_formula": "S_base=sum(max(0,1-dt/tau)) over same-pol neighbors in 9x9",
    }
    return summary_rows, hist_rows, meta


def _write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def main() -> int:
    _require_numba()

    ap = argparse.ArgumentParser(description="7.79 baseline score bandpass stats (GT noise/signal)")
    ap.add_argument("--labeled-npy", default=r"D:\hjx_workspace\scientific_reserach\dataset\ED24\myPedestrain_06\Pedestrain_06_3.3.npy")
    ap.add_argument("--width", type=int, default=346)
    ap.add_argument("--height", type=int, default=260)
    ap.add_argument("--tick-ns", type=float, default=1000.0)
    ap.add_argument("--max-events", type=int, default=500000)
    ap.add_argument("--tau-us", type=int, default=30000)
    ap.add_argument(
        "--score-edges",
        default="0,1,2,3,4,5,7,10,15,20,30,50,1e18",
        help="Comma-separated increasing edges. Last edge should be large for >= tail bin.",
    )
    ap.add_argument("--out-dir", default="data/ED24/myPedestrain_06/EBF_Part2/_gt_feature_stats_779")
    args = ap.parse_args()

    score_edges = _parse_edges(str(args.score_edges))
    summary_rows, hist_rows, meta = run_stats(
        labeled_npy=str(args.labeled_npy),
        width=int(args.width),
        height=int(args.height),
        tick_ns=float(args.tick_ns),
        max_events=int(args.max_events),
        tau_us=int(args.tau_us),
        score_edges=score_edges,
    )

    out_dir = os.path.abspath(str(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)
    out_summary = os.path.join(out_dir, "summary.csv")
    out_hist = os.path.join(out_dir, "hist_sbase.csv")
    out_json = os.path.join(out_dir, "summary.json")

    _write_csv(out_summary, summary_rows, ["class", "count", "mean_s_base", "std_s_base"])
    _write_csv(out_hist, hist_rows, ["feature", "class", "bin_index", "bin_lo", "bin_hi", "count", "ratio"])

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "summary": summary_rows}, f, ensure_ascii=False, indent=2)

    print(f"Saved: {out_summary}")
    print(f"Saved: {out_hist}")
    print(f"Saved: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
