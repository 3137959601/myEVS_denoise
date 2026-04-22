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
        raise SystemExit("gt_feature_stats_777 requires numba")


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
def _fill_hist_uniform(v: float, hist: np.ndarray) -> None:
    n = int(hist.shape[0])
    if n <= 0:
        return
    if not (v >= 0.0):
        return
    if v >= 1.0:
        b = n - 1
    else:
        b = int(v * float(n))
        if b < 0:
            b = 0
        elif b >= n:
            b = n - 1
    hist[b] += 1


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
def _kernel_stats(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    label: np.ndarray,
    width: int,
    height: int,
    tau_ticks: int,
    inner_radius: int,
    dt_edges_us: np.ndarray,
    hist_dt_noise: np.ndarray,
    hist_dt_signal: np.ndarray,
    hist_ratio_noise: np.ndarray,
    hist_ratio_signal: np.ndarray,
    hist_ratio_high_noise: np.ndarray,
    hist_ratio_high_signal: np.ndarray,
    nearest_counts: np.ndarray,
    stats_count: np.ndarray,
    stats_sum: np.ndarray,
    stats_sqsum: np.ndarray,
    high_count: np.ndarray,
):
    n = int(t.shape[0])
    w = int(width)
    h = int(height)
    tau = int(tau_ticks)
    rc = int(inner_radius)
    if rc < 0:
        rc = 0
    if rc > 4:
        rc = 4
    eps = 1e-12

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

        nearest_dt = 1 << 62
        nearest_d = 0
        has_nearest = 0

        e_in = 0.0
        e_out = 0.0

        for yy in range(y0, y1 + 1):
            base = yy * w
            dy = yy - yi
            ady = dy if dy >= 0 else -dy
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
                dx = xx - xi
                adx = dx if dx >= 0 else -dx
                d = adx if adx >= ady else ady

                if dt_ticks < nearest_dt:
                    nearest_dt = dt_ticks
                    nearest_d = d
                    has_nearest = 1

                if dt_ticks <= tau:
                    w_time = 1.0 - float(dt_ticks) / float(tau)
                    if w_time > 0.0:
                        if d <= rc:
                            e_in += w_time
                        else:
                            e_out += w_time

        if has_nearest == 1:
            dt_us = float(nearest_dt)
            if cls == 0:
                _fill_hist_edges(dt_us, dt_edges_us, hist_dt_noise)
            else:
                _fill_hist_edges(dt_us, dt_edges_us, hist_dt_signal)

            stats_count[cls, 0] += 1
            stats_sum[cls, 0] += dt_us
            stats_sqsum[cls, 0] += dt_us * dt_us

            if nearest_d <= rc:
                nearest_counts[cls, 0] += 1
            else:
                nearest_counts[cls, 1] += 1

        s_base = e_in + e_out
        ratio_pure = e_out / (s_base + eps)
        if ratio_pure < 0.0:
            ratio_pure = 0.0
        if ratio_pure > 1.0:
            ratio_pure = 1.0

        if cls == 0:
            _fill_hist_uniform(ratio_pure, hist_ratio_noise)
        else:
            _fill_hist_uniform(ratio_pure, hist_ratio_signal)

        stats_count[cls, 1] += 1
        stats_sum[cls, 1] += ratio_pure
        stats_sqsum[cls, 1] += ratio_pure * ratio_pure

        stats_count[cls, 2] += 1
        stats_sum[cls, 2] += s_base
        stats_sqsum[cls, 2] += s_base * s_base

        if s_base > 3.0:
            high_count[cls] += 1
            if cls == 0:
                _fill_hist_uniform(ratio_pure, hist_ratio_high_noise)
            else:
                _fill_hist_uniform(ratio_pure, hist_ratio_high_signal)

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
    inner_radius: int,
    bins_uniform: int,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    ev = load_labeled_npy(str(labeled_npy), max_events=int(max_events))
    tick_us = float(tick_ns) / 1000.0
    if tick_us <= 0:
        tick_us = 1.0
    tau_ticks = int(round(float(tau_us) / tick_us))

    bu = int(bins_uniform)
    if bu < 5:
        bu = 5

    dt_edges_us = np.asarray([0.0, 100.0, 1000.0, 5000.0, 10000.0, 30000.0, 1e18], dtype=np.float64)

    h_dt_noise = np.zeros((6,), dtype=np.int64)
    h_dt_signal = np.zeros((6,), dtype=np.int64)
    h_ratio_noise = np.zeros((bu,), dtype=np.int64)
    h_ratio_signal = np.zeros((bu,), dtype=np.int64)
    h_ratio_high_noise = np.zeros((bu,), dtype=np.int64)
    h_ratio_high_signal = np.zeros((bu,), dtype=np.int64)

    # nearest_counts[class, side], side: 0=inner, 1=outer
    nearest_counts = np.zeros((2, 2), dtype=np.int64)

    # stats feature order: dt_min_us, ratio_pure, s_base
    cnt = np.zeros((2, 3), dtype=np.int64)
    sm = np.zeros((2, 3), dtype=np.float64)
    sq = np.zeros((2, 3), dtype=np.float64)
    high_count = np.zeros((2,), dtype=np.int64)

    _kernel_stats(
        ev.t,
        ev.x,
        ev.y,
        ev.p,
        ev.label,
        int(width),
        int(height),
        int(tau_ticks),
        int(inner_radius),
        dt_edges_us,
        h_dt_noise,
        h_dt_signal,
        h_ratio_noise,
        h_ratio_signal,
        h_ratio_high_noise,
        h_ratio_high_signal,
        nearest_counts,
        cnt,
        sm,
        sq,
        high_count,
    )

    features = [
        ("dt_min_us", 0),
        ("ratio_pure", 1),
        ("s_base", 2),
    ]
    classes = [(0, "noise"), (1, "signal")]

    summary_rows: list[dict] = []
    for cidx, cname in classes:
        for fname, fidx in features:
            n = int(cnt[cidx, fidx])
            m, s = _mean_std(float(sm[cidx, fidx]), float(sq[cidx, fidx]), n)
            summary_rows.append(
                {
                    "class": cname,
                    "feature": fname,
                    "count": n,
                    "mean": m,
                    "std": s,
                }
            )
        summary_rows.append(
            {
                "class": cname,
                "feature": "ratio_pure_highscore_count",
                "count": int(high_count[cidx]),
                "mean": float("nan"),
                "std": float("nan"),
            }
        )

    hist_rows: list[dict] = []

    dt_specs = [
        ("dt_min_us", h_dt_noise, h_dt_signal),
    ]
    for fname, h0, h1 in dt_specs:
        for cname, hh in (("noise", h0), ("signal", h1)):
            n_all = int(np.sum(hh))
            for bi in range(int(hh.shape[0])):
                lo = float(dt_edges_us[bi])
                hi = float(dt_edges_us[bi + 1])
                c = int(hh[bi])
                hist_rows.append(
                    {
                        "feature": fname,
                        "class": cname,
                        "bin_index": int(bi),
                        "bin_lo": lo,
                        "bin_hi": hi,
                        "count": c,
                        "ratio": _safe_ratio(c, n_all),
                    }
                )

    uniform_specs = [
        ("ratio_pure", h_ratio_noise, h_ratio_signal),
        ("ratio_pure_highscore", h_ratio_high_noise, h_ratio_high_signal),
    ]
    for fname, h0, h1 in uniform_specs:
        for cname, hh in (("noise", h0), ("signal", h1)):
            n_all = int(np.sum(hh))
            for bi in range(int(hh.shape[0])):
                lo = float(bi) / float(hh.shape[0])
                hi = float(bi + 1) / float(hh.shape[0])
                c = int(hh[bi])
                hist_rows.append(
                    {
                        "feature": fname,
                        "class": cname,
                        "bin_index": int(bi),
                        "bin_lo": lo,
                        "bin_hi": hi,
                        "count": c,
                        "ratio": _safe_ratio(c, n_all),
                    }
                )

    nearest_rows: list[dict] = []
    for cidx, cname in classes:
        inner = int(nearest_counts[cidx, 0])
        outer = int(nearest_counts[cidx, 1])
        tot = inner + outer
        nearest_rows.append(
            {
                "class": cname,
                "total_with_nearest": tot,
                "inner_count": inner,
                "outer_count": outer,
                "inner_ratio": _safe_ratio(inner, tot),
                "outer_ratio": _safe_ratio(outer, tot),
            }
        )

    meta = {
        "input": str(labeled_npy),
        "events": int(ev.t.shape[0]),
        "tau_us": int(tau_us),
        "window": "9x9",
        "inner_radius": int(inner_radius),
        "bins_uniform": int(bu),
        "score_threshold": 3.0,
    }
    return summary_rows, hist_rows, nearest_rows, meta


def _write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def main() -> int:
    _require_numba()

    ap = argparse.ArgumentParser(description="7.77 baseline time/structure GT feature stats.")
    ap.add_argument("--labeled-npy", default=r"D:\hjx_workspace\scientific_reserach\dataset\ED24\myPedestrain_06\Pedestrain_06_3.3.npy")
    ap.add_argument("--width", type=int, default=346)
    ap.add_argument("--height", type=int, default=260)
    ap.add_argument("--tick-ns", type=float, default=1000.0)
    ap.add_argument("--max-events", type=int, default=500000)
    ap.add_argument("--tau-us", type=int, default=30000)
    ap.add_argument("--inner-radius", type=int, default=2)
    ap.add_argument("--bins-uniform", type=int, default=20)
    ap.add_argument("--out-dir", default="data/ED24/myPedestrain_06/EBF_Part2/_gt_feature_stats_777")
    args = ap.parse_args()

    summary_rows, hist_rows, nearest_rows, meta = run_stats(
        labeled_npy=str(args.labeled_npy),
        width=int(args.width),
        height=int(args.height),
        tick_ns=float(args.tick_ns),
        max_events=int(args.max_events),
        tau_us=int(args.tau_us),
        inner_radius=int(args.inner_radius),
        bins_uniform=int(args.bins_uniform),
    )

    out_dir = os.path.abspath(str(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    out_summary = os.path.join(out_dir, "summary.csv")
    out_hist = os.path.join(out_dir, "hist.csv")
    out_nearest = os.path.join(out_dir, "nearest_side.csv")
    out_json = os.path.join(out_dir, "summary.json")

    _write_csv(
        out_summary,
        summary_rows,
        ["class", "feature", "count", "mean", "std"],
    )
    _write_csv(
        out_hist,
        hist_rows,
        ["feature", "class", "bin_index", "bin_lo", "bin_hi", "count", "ratio"],
    )
    _write_csv(
        out_nearest,
        nearest_rows,
        ["class", "total_with_nearest", "inner_count", "outer_count", "inner_ratio", "outer_ratio"],
    )

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": meta,
                "summary": summary_rows,
                "nearest_side": nearest_rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved: {out_summary}")
    print(f"Saved: {out_hist}")
    print(f"Saved: {out_nearest}")
    print(f"Saved: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
