from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
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
        raise SystemExit("gt_feature_stats_n139 requires numba")


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
def _fill_hist(a: float, edges: np.ndarray, hist: np.ndarray) -> None:
    n = int(hist.shape[0])
    if n <= 0:
        return
    if not (a >= 0.0):
        return
    if a >= edges[n]:
        hist[n - 1] += 1
        return
    for i in range(n):
        lo = edges[i]
        hi = edges[i + 1]
        if a >= lo and a < hi:
            hist[i] += 1
            return


@numba.njit(cache=True)
def _kernel_n139_a_true(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    label: np.ndarray,
    width: int,
    height: int,
    tau_ticks: int,
    radius: int,
    low_thresh: float,
    high_thresh: float,
    eps: float,
    edges: np.ndarray,
    hist_b_noise: np.ndarray,
    hist_b_signal: np.ndarray,
    hist_a_noise: np.ndarray,
    hist_a_signal: np.ndarray,
    count_in: np.ndarray,
    count_kept: np.ndarray,
    sum_before: np.ndarray,
    sqsum_before: np.ndarray,
    sum_after: np.ndarray,
    sqsum_after: np.ndarray,
) -> None:
    n = int(t.shape[0])
    w = int(width)
    h = int(height)
    rr = int(radius)
    if rr < 0:
        rr = 0
    if rr > 8:
        rr = 8

    tau = int(tau_ticks)
    if tau <= 0:
        tau = 1
    inv_tau = 1.0 / float(tau)

    pix = w * h
    last_ts = np.zeros((pix,), dtype=np.uint64)
    last_pol = np.zeros((pix,), dtype=np.int8)

    for i in range(n):
        xi = int(x[i])
        yi = int(y[i])
        if xi < 0 or xi >= w or yi < 0 or yi >= h:
            continue

        ti = np.uint64(t[i])
        pi = 1 if int(p[i]) > 0 else -1
        cls = int(label[i])

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

        e0 = 0.0
        e90 = 0.0
        e45 = 0.0
        e135 = 0.0
        s_base = 0.0

        for yy in range(y0, y1 + 1):
            base = yy * w
            dy = yy - yi
            for xx in range(x0, x1 + 1):
                if xx == xi and yy == yi:
                    continue
                idx_nb = base + xx
                ts = last_ts[idx_nb]
                if ts == 0 or ti <= ts:
                    continue
                if int(last_pol[idx_nb]) != pi:
                    continue

                dt_ticks = int(ti - ts)
                if dt_ticks > tau:
                    continue
                wt = 1.0 - float(dt_ticks) * inv_tau
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

        emax = e0
        if e90 > emax:
            emax = e90
        if e45 > emax:
            emax = e45
        if e135 > emax:
            emax = e135

        a_true = 0.0
        passed = False
        if s_base > 0.0:
            a_true = emax / (s_base + eps)
            if a_true < 0.0:
                a_true = 0.0
            if a_true > 1.0:
                a_true = 1.0
            passed = (a_true >= low_thresh and a_true <= high_thresh)

        count_in[cls] += 1
        sum_before[cls] += a_true
        sqsum_before[cls] += a_true * a_true
        if cls == 0:
            _fill_hist(a_true, edges, hist_b_noise)
        else:
            _fill_hist(a_true, edges, hist_b_signal)

        if passed:
            count_kept[cls] += 1
            sum_after[cls] += a_true
            sqsum_after[cls] += a_true * a_true
            if cls == 0:
                _fill_hist(a_true, edges, hist_a_noise)
            else:
                _fill_hist(a_true, edges, hist_a_signal)

        idx0 = yi * w + xi
        last_ts[idx0] = ti
        last_pol[idx0] = np.int8(pi)


def _mean_std(sum_v: float, sqsum_v: float, n: int) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    m = float(sum_v) / float(n)
    v = float(sqsum_v) / float(n) - m * m
    if np.isfinite(v) and v < 0.0:
        v = 0.0
    s = float(np.sqrt(v)) if np.isfinite(v) else float("nan")
    return float(m), float(s)


def _write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _build_hist_rows(
    edges: np.ndarray,
    hist_b_noise: np.ndarray,
    hist_b_signal: np.ndarray,
    hist_a_noise: np.ndarray,
    hist_a_signal: np.ndarray,
) -> list[dict]:
    out: list[dict] = []
    data = (
        ("before", "noise", hist_b_noise),
        ("before", "signal", hist_b_signal),
        ("after", "noise", hist_a_noise),
        ("after", "signal", hist_a_signal),
    )
    for stage, cls, h in data:
        total = int(np.sum(h))
        for i in range(int(h.shape[0])):
            c = int(h[i])
            out.append(
                {
                    "feature": "a_true",
                    "stage": stage,
                    "class": cls,
                    "bin_index": int(i),
                    "bin_lo": float(edges[i]),
                    "bin_hi": float(edges[i + 1]),
                    "count": c,
                    "ratio": (float(c) / float(total)) if total > 0 else float("nan"),
                }
            )
    return out


def _plot_hist(
    *,
    edges: np.ndarray,
    hist_b_noise: np.ndarray,
    hist_b_signal: np.ndarray,
    hist_a_noise: np.ndarray,
    hist_a_signal: np.ndarray,
    out_png: str,
    title: str,
) -> None:
    n = int(edges.shape[0] - 1)
    x = np.arange(n, dtype=float)
    labels = [f"[{edges[i]:.2f},{edges[i+1]:.2f})" for i in range(n)]

    def _ratio(h: np.ndarray) -> np.ndarray:
        s = float(np.sum(h))
        if s <= 0.0:
            return np.zeros_like(h, dtype=np.float64)
        return h.astype(np.float64) / s

    rb_n = _ratio(hist_b_noise)
    rb_s = _ratio(hist_b_signal)
    ra_n = _ratio(hist_a_noise)
    ra_s = _ratio(hist_a_signal)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    ax0, ax1 = axes

    ax0.bar(x - 0.18, rb_n, width=0.36, color="#d62728", alpha=0.84, label="noise")
    ax0.bar(x + 0.18, rb_s, width=0.36, color="#2ca02c", alpha=0.84, label="signal")
    ax0.set_title("Before filter")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=30)
    ax0.set_xlabel("A_true bins")
    ax0.set_ylabel("ratio")
    ax0.grid(axis="y", linestyle="--", alpha=0.35)
    ax0.legend(fontsize=10)

    ax1.bar(x - 0.18, ra_n, width=0.36, color="#d62728", alpha=0.84, label="noise")
    ax1.bar(x + 0.18, ra_s, width=0.36, color="#2ca02c", alpha=0.84, label="signal")
    ax1.set_title("After filter (kept only)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30)
    ax1.set_xlabel("A_true bins")
    ax1.grid(axis="y", linestyle="--", alpha=0.35)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def run_stats(
    *,
    labeled_npy: str,
    out_dir: str,
    max_events: int,
    tau_us: int,
    radius: int,
    low: float,
    high: float,
    eps: float,
    n_bins: int,
) -> None:
    _require_numba()

    ev = load_labeled_npy(str(labeled_npy), max_events=int(max_events))
    width = int(ev.x.max()) + 1
    height = int(ev.y.max()) + 1

    edges = np.linspace(0.0, 1.0, int(n_bins) + 1, dtype=np.float64)
    bins = int(n_bins)

    hist_b_noise = np.zeros((bins,), dtype=np.int64)
    hist_b_signal = np.zeros((bins,), dtype=np.int64)
    hist_a_noise = np.zeros((bins,), dtype=np.int64)
    hist_a_signal = np.zeros((bins,), dtype=np.int64)

    count_in = np.zeros((2,), dtype=np.int64)
    count_kept = np.zeros((2,), dtype=np.int64)
    sum_before = np.zeros((2,), dtype=np.float64)
    sqsum_before = np.zeros((2,), dtype=np.float64)
    sum_after = np.zeros((2,), dtype=np.float64)
    sqsum_after = np.zeros((2,), dtype=np.float64)

    _kernel_n139_a_true(
        ev.t,
        ev.x,
        ev.y,
        ev.p,
        ev.label,
        int(width),
        int(height),
        int(tau_us),
        int(radius),
        float(low),
        float(high),
        float(eps),
        edges,
        hist_b_noise,
        hist_b_signal,
        hist_a_noise,
        hist_a_signal,
        count_in,
        count_kept,
        sum_before,
        sqsum_before,
        sum_after,
        sqsum_after,
    )

    rows_summary: list[dict] = []
    for cls_idx, cls_name in ((0, "noise"), (1, "signal")):
        n_in = int(count_in[cls_idx])
        n_keep = int(count_kept[cls_idx])
        m_b, s_b = _mean_std(float(sum_before[cls_idx]), float(sqsum_before[cls_idx]), n_in)
        m_a, s_a = _mean_std(float(sum_after[cls_idx]), float(sqsum_after[cls_idx]), n_keep)
        rows_summary.append(
            {
                "class": cls_name,
                "events_before": n_in,
                "events_after": n_keep,
                "keep_rate": (float(n_keep) / float(n_in)) if n_in > 0 else float("nan"),
                "a_true_mean_before": m_b,
                "a_true_std_before": s_b,
                "a_true_mean_after": m_a,
                "a_true_std_after": s_a,
            }
        )

    hist_rows = _build_hist_rows(
        edges,
        hist_b_noise,
        hist_b_signal,
        hist_a_noise,
        hist_a_signal,
    )

    os.makedirs(out_dir, exist_ok=True)
    out_summary = os.path.join(out_dir, "summary.csv")
    out_hist = os.path.join(out_dir, "hist_a_true_bins.csv")
    out_json = os.path.join(out_dir, "summary.json")
    out_png = os.path.join(out_dir, "hist_a_true_before_after.png")

    _write_csv(
        out_summary,
        rows_summary,
        [
            "class",
            "events_before",
            "events_after",
            "keep_rate",
            "a_true_mean_before",
            "a_true_std_before",
            "a_true_mean_after",
            "a_true_std_after",
        ],
    )
    _write_csv(
        out_hist,
        hist_rows,
        ["feature", "stage", "class", "bin_index", "bin_lo", "bin_hi", "count", "ratio"],
    )

    meta = {
        "config": {
            "labeled_npy": str(labeled_npy),
            "max_events": int(max_events),
            "tau_us": int(tau_us),
            "radius": int(radius),
            "low": float(low),
            "high": float(high),
            "eps": float(eps),
            "n_bins": int(n_bins),
            "width": int(width),
            "height": int(height),
        },
        "summary": rows_summary,
        "global": {
            "events_before_total": int(np.sum(count_in)),
            "events_after_total": int(np.sum(count_kept)),
            "overall_keep_rate": (float(np.sum(count_kept)) / float(np.sum(count_in))) if int(np.sum(count_in)) > 0 else float("nan"),
            "noise_keep_rate": (float(count_kept[0]) / float(count_in[0])) if int(count_in[0]) > 0 else float("nan"),
            "signal_keep_rate": (float(count_kept[1]) / float(count_in[1])) if int(count_in[1]) > 0 else float("nan"),
        },
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    title = (
        f"n139 heavy A_true bins | s={radius}, tau={tau_us}us, "
        f"L={low:.2f}, H={high:.2f}"
    )
    _plot_hist(
        edges=edges,
        hist_b_noise=hist_b_noise,
        hist_b_signal=hist_b_signal,
        hist_a_noise=hist_a_noise,
        hist_a_signal=hist_a_signal,
        out_png=out_png,
        title=title,
    )

    print(f"saved: {out_summary}")
    print(f"saved: {out_hist}")
    print(f"saved: {out_json}")
    print(f"saved: {out_png}")


def main() -> int:
    ap = argparse.ArgumentParser(description="n139 A_true bins stats (before/after filter)")
    ap.add_argument(
        "--labeled-npy",
        default="D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3.npy",
    )
    ap.add_argument("--out-dir", default="data/ED24/myPedestrain_06/EBF_Part2/_gt_feature_stats_n139_heavy_best")
    ap.add_argument("--max-events", type=int, default=400000)
    ap.add_argument("--tau-us", type=int, default=32000)
    ap.add_argument("--radius", type=int, default=5)
    ap.add_argument("--low", type=float, default=0.10)
    ap.add_argument("--high", type=float, default=0.80)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--n-bins", type=int, default=20)
    args = ap.parse_args()

    run_stats(
        labeled_npy=str(args.labeled_npy),
        out_dir=str(args.out_dir),
        max_events=int(args.max_events),
        tau_us=int(args.tau_us),
        radius=int(args.radius),
        low=float(args.low),
        high=float(args.high),
        eps=float(args.eps),
        n_bins=int(args.n_bins),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
