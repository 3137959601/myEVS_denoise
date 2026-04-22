from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class LabeledEvents:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    p: np.ndarray
    label: np.ndarray


def load_labeled_npy(path: str, *, start_events: int = 0, max_events: int = 0) -> LabeledEvents:
    arr = np.load(path, mmap_mode="r", allow_pickle=True)

    s0 = int(start_events)
    if s0 < 0:
        s0 = 0
    if s0 > 0:
        arr = arr[s0:]

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


def _load_hotmask(hotmask_npy: str, *, width: int, height: int) -> np.ndarray:
    m = np.load(hotmask_npy)
    a = np.asarray(m)
    if a.ndim == 2:
        if a.shape != (int(height), int(width)):
            raise SystemExit(f"hotmask shape mismatch: got {a.shape}, expect {(int(height), int(width))}")
        return (a != 0).reshape(-1)
    if a.ndim == 1:
        if a.shape[0] != int(width) * int(height):
            raise SystemExit(
                f"hotmask shape mismatch: got {a.shape}, expect ({int(width) * int(height)},)"
            )
        return (a != 0).reshape(-1)
    raise SystemExit(f"hotmask must be 1D or 2D array, got ndim={a.ndim}")


def _run_lengths_of_sorted(a: np.ndarray) -> tuple[int, float]:
    """Return (max_run, mean_run) for consecutive equal values in sorted array."""
    if a.size == 0:
        return 0, 0.0
    diff = a[1:] != a[:-1]
    idx = np.flatnonzero(diff)
    ends = np.concatenate([idx, np.asarray([a.size - 1], dtype=np.int64)])
    starts = np.concatenate([np.asarray([0], dtype=np.int64), ends[:-1] + 1])
    runs = (ends - starts + 1).astype(np.int64)
    return int(runs.max(initial=0)), float(runs.mean())


def _quantiles(a: np.ndarray, ps: list[float]) -> dict[str, float]:
    out: dict[str, float] = {}
    if a.size == 0:
        for p in ps:
            out[f"p{int(round(p * 100)):02d}"] = float("nan")
        return out

    q = np.quantile(a, ps)
    for p, v in zip(ps, q):
        out[f"p{int(round(p * 100)):02d}"] = float(v)
    return out


def noise_structure_stats(
    *,
    labeled_npy: str,
    out_csv: str,
    width: int,
    height: int,
    tick_ns: float,
    start_events: int,
    max_events: int,
    window_events: int,
    tau_us: int,
    cluster_radius: int,
    cluster_win_us: int,
    cluster_k: int,
    hotmask_npy: str,
    topk_pixels: int,
    topk_pixels_csv: str,
) -> None:
    ev = load_labeled_npy(labeled_npy, start_events=int(start_events), max_events=int(max_events))

    w = int(width)
    h = int(height)
    if w <= 0 or h <= 0:
        raise SystemExit(f"invalid dims: {w}x{h}")

    n = int(ev.t.size)
    if n <= 0:
        raise SystemExit("no events loaded")

    # Convert ticks -> microseconds (float), but keep integer microseconds for grouping.
    tick_ns_f = float(tick_ns)
    if tick_ns_f <= 0:
        tick_ns_f = 1000.0
    t_us = (ev.t.astype(np.float64) * (tick_ns_f / 1000.0)).astype(np.float64, copy=False)

    # Optional hotmask.
    hotmask = None
    if str(hotmask_npy).strip():
        hotmask = _load_hotmask(str(hotmask_npy), width=w, height=h)

    # Per-pixel state for burst/cluster stats.
    n_pix = w * h
    last_t_us = np.full((n_pix,), np.nan, dtype=np.float64)
    last_t_us_samepol = np.full((n_pix,), np.nan, dtype=np.float64)
    last_pol = np.zeros((n_pix,), dtype=np.int8)

    # For cluster: last timestamp per pixel (any pol) in microseconds.
    last_t_us_any = np.full((n_pix,), np.nan, dtype=np.float64)

    # Pixelwise noise-only counts for spatial structure.
    noise_cnt = np.zeros((n_pix,), dtype=np.int32)

    rr = int(cluster_radius)
    if rr < 0:
        rr = 0
    if rr > 4:
        rr = 4
    win_us = float(max(0, int(cluster_win_us)))

    # Quantile probs.
    ps = [0.1, 0.5, 0.9, 0.99]

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)

    fieldnames = [
        "seg_start_events",
        "seg_max_events",
        "win_start",
        "win_end",
        "t_us_start",
        "t_us_end",
        "dur_us",
        "events",
        "events_per_ms",
        "signal",
        "noise",
        "signal_frac",
        "on_frac",
        "hotmask_frac",
        "unique_pixels",
        "unique_pixels_frac",
        "max_events_same_us",
        "mean_events_same_us",
        "max_events_same_10us",
        "mean_events_same_10us",
        "dt_any_us_mean",
        "dt_samepol_us_mean",
        "dt_any_norm_p10",
        "dt_any_norm_p50",
        "dt_any_norm_p90",
        "dt_any_norm_p99",
        "dt_samepol_norm_p10",
        "dt_samepol_norm_p50",
        "dt_samepol_norm_p90",
        "dt_samepol_norm_p99",
        "nb_recent_cnt_mean",
        "nb_recent_cnt_p90",
        "nb_recent_frac_ge_k",
    ]

    tau_us_f = float(max(1, int(tau_us)))
    k_thr = int(max(1, int(cluster_k)))

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.DictWriter(f, fieldnames=fieldnames)
        wcsv.writeheader()

        win = int(window_events)
        if win <= 0:
            win = n

        for i0 in range(0, n, win):
            i1 = min(n, i0 + win)
            if i1 <= i0:
                continue

            xw = ev.x[i0:i1]
            yw = ev.y[i0:i1]
            pw = ev.p[i0:i1]
            lw = ev.label[i0:i1]
            tw_us = t_us[i0:i1]

            # Filter in-bounds (should be, but defensive).
            inb = (xw >= 0) & (xw < w) & (yw >= 0) & (yw < h)
            if not bool(np.all(inb)):
                xw = xw[inb]
                yw = yw[inb]
                pw = pw[inb]
                lw = lw[inb]
                tw_us = tw_us[inb]

            m = int(xw.size)
            if m <= 0:
                continue

            idx = (yw.astype(np.int64) * w + xw.astype(np.int64)).astype(np.int64, copy=False)

            # Basic counts.
            sig = int(np.count_nonzero(lw > 0))
            noi = int(m - sig)
            on_frac = float(np.count_nonzero(pw > 0)) / float(max(1, m))

            # Hotmask coverage (by events).
            hotmask_frac = float("nan")
            if hotmask is not None:
                hotmask_frac = float(np.count_nonzero(hotmask[idx])) / float(max(1, m))

            # Unique pixels.
            uniq_px = int(np.unique(idx).size)
            uniq_px_frac = float(uniq_px) / float(n_pix)

            # Time ranges and rates.
            t0 = float(tw_us[0])
            t1 = float(tw_us[-1])
            dur_us = float(max(1.0, t1 - t0))
            events_per_ms = float(m) / (dur_us / 1000.0)

            # Global sync proxy: run-lengths for equal timestamps.
            t_us_i = np.floor(tw_us + 1e-9).astype(np.int64)
            max_same_us, mean_same_us = _run_lengths_of_sorted(t_us_i)
            t_10us = (t_us_i // 10).astype(np.int64)
            max_same_10us, mean_same_10us = _run_lengths_of_sorted(t_10us)

            # Per-event dt (any-pol, same-pol) + neighbor recent count.
            dt_any_norm = np.full((m,), np.nan, dtype=np.float64)
            dt_samepol_norm = np.full((m,), np.nan, dtype=np.float64)
            nb_cnt = np.zeros((m,), dtype=np.int16)

            for j in range(m):
                idx0 = int(idx[j])
                tj = float(tw_us[j])
                pj = 1 if int(pw[j]) > 0 else -1

                # dt_any
                tprev = last_t_us[idx0]
                if np.isfinite(tprev):
                    dt_any_norm[j] = float(max(0.0, tj - float(tprev))) / tau_us_f
                last_t_us[idx0] = tj

                # dt_samepol
                if int(last_pol[idx0]) == pj and np.isfinite(last_t_us_samepol[idx0]):
                    dt_samepol_norm[j] = float(max(0.0, tj - float(last_t_us_samepol[idx0]))) / tau_us_f
                last_t_us_samepol[idx0] = tj
                last_pol[idx0] = np.int8(pj)

                # Neighbor recent activity count (cluster proxy)
                if rr > 0 and win_us > 0:
                    xi = int(xw[j])
                    yi = int(yw[j])

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

                    c = 0
                    for yy in range(y0, y1 + 1):
                        base = yy * w
                        for xx in range(x0, x1 + 1):
                            if xx == xi and yy == yi:
                                continue
                            idn = base + xx
                            tnb = last_t_us_any[idn]
                            if np.isfinite(tnb) and (tj - float(tnb)) <= win_us:
                                c += 1
                    nb_cnt[j] = np.int16(c)

                last_t_us_any[idx0] = tj

                # Noise-only pixel count.
                if int(lw[j]) == 0:
                    noise_cnt[idx0] += 1

            dt_any_f = dt_any_norm[np.isfinite(dt_any_norm)]
            dt_same_f = dt_samepol_norm[np.isfinite(dt_samepol_norm)]

            dt_any_us_mean = float(np.nanmean(dt_any_f) * tau_us_f) if dt_any_f.size else float("nan")
            dt_same_us_mean = float(np.nanmean(dt_same_f) * tau_us_f) if dt_same_f.size else float("nan")

            q_any = _quantiles(dt_any_f, ps)
            q_same = _quantiles(dt_same_f, ps)

            nb_cnt_f = nb_cnt.astype(np.float64)
            nb_mean = float(nb_cnt_f.mean()) if nb_cnt_f.size else float("nan")
            nb_p90 = float(np.quantile(nb_cnt_f, 0.9)) if nb_cnt_f.size else float("nan")
            nb_frac_ge_k = float(np.count_nonzero(nb_cnt >= k_thr)) / float(max(1, m))

            row = {
                "seg_start_events": int(start_events),
                "seg_max_events": int(max_events),
                "win_start": int(i0),
                "win_end": int(i1),
                "t_us_start": float(t0),
                "t_us_end": float(t1),
                "dur_us": float(dur_us),
                "events": int(m),
                "events_per_ms": float(events_per_ms),
                "signal": int(sig),
                "noise": int(noi),
                "signal_frac": float(sig) / float(max(1, m)),
                "on_frac": float(on_frac),
                "hotmask_frac": float(hotmask_frac),
                "unique_pixels": int(uniq_px),
                "unique_pixels_frac": float(uniq_px_frac),
                "max_events_same_us": int(max_same_us),
                "mean_events_same_us": float(mean_same_us),
                "max_events_same_10us": int(max_same_10us),
                "mean_events_same_10us": float(mean_same_10us),
                "dt_any_us_mean": float(dt_any_us_mean),
                "dt_samepol_us_mean": float(dt_same_us_mean),
                "dt_any_norm_p10": float(q_any["p10"]),
                "dt_any_norm_p50": float(q_any["p50"]),
                "dt_any_norm_p90": float(q_any["p90"]),
                "dt_any_norm_p99": float(q_any["p99"]),
                "dt_samepol_norm_p10": float(q_same["p10"]),
                "dt_samepol_norm_p50": float(q_same["p50"]),
                "dt_samepol_norm_p90": float(q_same["p90"]),
                "dt_samepol_norm_p99": float(q_same["p99"]),
                "nb_recent_cnt_mean": float(nb_mean),
                "nb_recent_cnt_p90": float(nb_p90),
                "nb_recent_frac_ge_k": float(nb_frac_ge_k),
            }
            wcsv.writerow(row)

    # Optional: top-K noise pixels.
    if str(topk_pixels_csv).strip() and int(topk_pixels) > 0:
        k = int(topk_pixels)
        idxs = np.argsort(-noise_cnt, kind="mergesort")
        idxs = idxs[:k]

        os.makedirs(os.path.dirname(os.path.abspath(str(topk_pixels_csv))), exist_ok=True)
        with open(str(topk_pixels_csv), "w", newline="", encoding="utf-8") as f:
            wcsv = csv.DictWriter(f, fieldnames=["rank", "x", "y", "noise_events"])
            wcsv.writeheader()
            for r, idx0 in enumerate(idxs.tolist(), start=1):
                c = int(noise_cnt[int(idx0)])
                if c <= 0:
                    continue
                x = int(idx0 % w)
                y = int(idx0 // w)
                wcsv.writerow({"rank": int(r), "x": x, "y": y, "noise_events": c})


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Noise structure stats for a slice (default: seg1 [200k,400k)). "
            "Outputs per-window summary CSV for diagnosing non-stationarity / burstiness / clustering."
        )
    )

    ap.add_argument("--labeled-npy", required=True, help="Input labeled .npy with fields t/x/y/p/label")
    ap.add_argument("--out-csv", required=True, help="Output per-window CSV path")

    ap.add_argument("--width", type=int, default=346)
    ap.add_argument("--height", type=int, default=260)
    ap.add_argument("--tick-ns", type=float, default=1000.0)

    ap.add_argument(
        "--start-events",
        type=int,
        default=200000,
        help="Skip first N events (default=200k => focus on seg1: [200k,400k))",
    )
    ap.add_argument(
        "--max-events",
        type=int,
        default=200000,
        help="Max events to load after start-events (default=200k => one segment)",
    )

    ap.add_argument(
        "--window-events",
        type=int,
        default=20000,
        help="Window size in events for per-window stats",
    )

    ap.add_argument(
        "--tau-us",
        type=int,
        default=128000,
        help="Tau (us) used for dt/tau normalization in burst stats",
    )

    ap.add_argument("--cluster-radius", type=int, default=1, help="Neighbor radius (px) for cluster proxy")
    ap.add_argument("--cluster-win-us", type=int, default=2000, help="Neighbor recency window (us) for cluster proxy")
    ap.add_argument("--cluster-k", type=int, default=3, help="Threshold k for nb_recent_cnt >= k")

    ap.add_argument("--hotmask-npy", default="", help="Optional hotmask .npy")

    ap.add_argument("--topk-pixels", type=int, default=30)
    ap.add_argument("--topk-pixels-csv", default="", help="Optional output CSV for top-K noise pixels")

    args = ap.parse_args()

    noise_structure_stats(
        labeled_npy=str(args.labeled_npy),
        out_csv=str(args.out_csv),
        width=int(args.width),
        height=int(args.height),
        tick_ns=float(args.tick_ns),
        start_events=int(args.start_events),
        max_events=int(args.max_events),
        window_events=int(args.window_events),
        tau_us=int(args.tau_us),
        cluster_radius=int(args.cluster_radius),
        cluster_win_us=int(args.cluster_win_us),
        cluster_k=int(args.cluster_k),
        hotmask_npy=str(args.hotmask_npy).strip(),
        topk_pixels=int(args.topk_pixels),
        topk_pixels_csv=str(args.topk_pixels_csv).strip(),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
