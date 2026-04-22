from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

try:
    import numba
except Exception:  # pragma: no cover
    numba = None

SCOPE_NAMES = ("all", "top1", "top3")


@dataclass(frozen=True)
class LabeledEvents:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    p: np.ndarray
    label: np.ndarray


def _require_numba() -> None:
    if numba is None:
        raise SystemExit("center_relative_support_stats requires numba")


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


@numba.njit(cache=True)
def _bin_index(v: float, bins: np.ndarray) -> int:
    if v < bins[0] or v >= bins[-1]:
        return -1
    lo = 0
    hi = int(bins.shape[0]) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if v < bins[mid]:
            hi = mid
        else:
            lo = mid
    return lo


@numba.njit(cache=True)
def _accumulate_sample(
    scope_idx: int,
    cls: int,
    dist: float,
    dt_us: float,
    v: float,
    support_cnt: np.ndarray,
    sum_d: np.ndarray,
    sum_dt: np.ndarray,
    sum_v: np.ndarray,
    hist_d: np.ndarray,
    hist_dt: np.ndarray,
    hist_v: np.ndarray,
    hist_2d: np.ndarray,
    d_bins: np.ndarray,
    dt_bins: np.ndarray,
    v_bins: np.ndarray,
) -> None:
    support_cnt[scope_idx, cls] += 1
    sum_d[scope_idx, cls] += dist
    sum_dt[scope_idx, cls] += dt_us
    sum_v[scope_idx, cls] += v

    bd = _bin_index(dist, d_bins)
    if bd >= 0:
        hist_d[scope_idx, cls, bd] += 1

    bdt = _bin_index(dt_us, dt_bins)
    if bdt >= 0:
        hist_dt[scope_idx, cls, bdt] += 1

    bv = _bin_index(v, v_bins)
    if bv >= 0:
        hist_v[scope_idx, cls, bv] += 1

    if bd >= 0 and bdt >= 0:
        hist_2d[scope_idx, cls, bd, bdt] += 1


@numba.njit(cache=True)
def _compute_center_relative_stats(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    label: np.ndarray,
    width: int,
    height: int,
    radius_px: int,
    tau_us: float,
    tick_us: float,
    include_center: int,
    d_bins: np.ndarray,
    dt_bins: np.ndarray,
    v_bins: np.ndarray,
) -> tuple[np.ndarray, ...]:
    n = int(t.shape[0])
    npx = int(width) * int(height)

    # scope: 0=all, 1=top1, 2=top3
    event_cnt = np.zeros((3, 2), dtype=np.int64)
    support_cnt = np.zeros((3, 2), dtype=np.int64)
    support_per_event_sum = np.zeros((3, 2), dtype=np.float64)
    max_supports = np.zeros((3, 2), dtype=np.int64)

    sum_d = np.zeros((3, 2), dtype=np.float64)
    sum_dt = np.zeros((3, 2), dtype=np.float64)
    sum_v = np.zeros((3, 2), dtype=np.float64)

    nd = int(d_bins.shape[0]) - 1
    ndt = int(dt_bins.shape[0]) - 1
    nv = int(v_bins.shape[0]) - 1

    hist_d = np.zeros((3, 2, nd), dtype=np.int64)
    hist_dt = np.zeros((3, 2, ndt), dtype=np.int64)
    hist_v = np.zeros((3, 2, nv), dtype=np.int64)
    hist_2d = np.zeros((3, 2, nd, ndt), dtype=np.int64)

    pos_ts = np.zeros((npx,), dtype=np.uint64)
    neg_ts = np.zeros((npx,), dtype=np.uint64)

    rr = int(radius_px)
    if rr < 0:
        rr = 0
    if rr > 8:
        rr = 8

    tau_lim = float(tau_us)
    if tau_lim <= 0.0:
        tau_lim = 1.0

    for i in range(n):
        xi = int(x[i])
        yi = int(y[i])
        if xi < 0 or xi >= width or yi < 0 or yi >= height:
            continue

        cls = 1 if int(label[i]) > 0 else 0
        event_cnt[0, cls] += 1
        event_cnt[1, cls] += 1
        event_cnt[2, cls] += 1

        ti = np.uint64(t[i])
        pi = 1 if int(p[i]) > 0 else -1

        x0 = xi - rr
        if x0 < 0:
            x0 = 0
        x1 = xi + rr
        if x1 >= width:
            x1 = width - 1
        y0 = yi - rr
        if y0 < 0:
            y0 = 0
        y1 = yi + rr
        if y1 >= height:
            y1 = height - 1

        ncur = 0

        # Keep the 3 most recent supports (smallest dt).
        best_dt = np.empty((3,), dtype=np.float64)
        best_d = np.empty((3,), dtype=np.float64)
        best_v = np.empty((3,), dtype=np.float64)
        best_dt[0] = 1e30
        best_dt[1] = 1e30
        best_dt[2] = 1e30
        best_d[0] = 0.0
        best_d[1] = 0.0
        best_d[2] = 0.0
        best_v[0] = 0.0
        best_v[1] = 0.0
        best_v[2] = 0.0

        for yy in range(y0, y1 + 1):
            base = yy * width
            dy = float(yy - yi)
            for xx in range(x0, x1 + 1):
                if include_center == 0 and xx == xi and yy == yi:
                    continue

                idx = base + xx
                ts = pos_ts[idx] if pi > 0 else neg_ts[idx]
                if ts == 0:
                    continue
                if ti <= ts:
                    continue

                dt_us = float(ti - ts) * tick_us
                if dt_us <= 0.0 or dt_us > tau_lim:
                    continue

                dx = float(xx - xi)
                dist = np.sqrt(dx * dx + dy * dy)
                v = dist / dt_us

                _accumulate_sample(
                    0,
                    cls,
                    dist,
                    dt_us,
                    v,
                    support_cnt,
                    sum_d,
                    sum_dt,
                    sum_v,
                    hist_d,
                    hist_dt,
                    hist_v,
                    hist_2d,
                    d_bins,
                    dt_bins,
                    v_bins,
                )

                # Insert into top3 by dt (ascending).
                if dt_us < best_dt[2]:
                    pos = 2
                    while pos > 0 and dt_us < best_dt[pos - 1]:
                        best_dt[pos] = best_dt[pos - 1]
                        best_d[pos] = best_d[pos - 1]
                        best_v[pos] = best_v[pos - 1]
                        pos -= 1
                    best_dt[pos] = dt_us
                    best_d[pos] = dist
                    best_v[pos] = v

                ncur += 1

        m = ncur
        if m > 3:
            m = 3

        # top1
        if m >= 1:
            _accumulate_sample(
                1,
                cls,
                best_d[0],
                best_dt[0],
                best_v[0],
                support_cnt,
                sum_d,
                sum_dt,
                sum_v,
                hist_d,
                hist_dt,
                hist_v,
                hist_2d,
                d_bins,
                dt_bins,
                v_bins,
            )

        # top3
        for k in range(m):
            _accumulate_sample(
                2,
                cls,
                best_d[k],
                best_dt[k],
                best_v[k],
                support_cnt,
                sum_d,
                sum_dt,
                sum_v,
                hist_d,
                hist_dt,
                hist_v,
                hist_2d,
                d_bins,
                dt_bins,
                v_bins,
            )

        # per-event support stats (scope dependent)
        support_per_event_sum[0, cls] += float(ncur)
        support_per_event_sum[1, cls] += 1.0 if ncur > 0 else 0.0
        support_per_event_sum[2, cls] += float(m)

        if ncur > max_supports[0, cls]:
            max_supports[0, cls] = ncur
        if (1 if ncur > 0 else 0) > max_supports[1, cls]:
            max_supports[1, cls] = 1 if ncur > 0 else 0
        if m > max_supports[2, cls]:
            max_supports[2, cls] = m

        idx0 = yi * width + xi
        if pi > 0:
            pos_ts[idx0] = ti
        else:
            neg_ts[idx0] = ti

    return (
        event_cnt,
        support_cnt,
        support_per_event_sum,
        sum_d,
        sum_dt,
        sum_v,
        max_supports,
        hist_d,
        hist_dt,
        hist_v,
        hist_2d,
    )


def _midpoints(bins: np.ndarray) -> np.ndarray:
    return 0.5 * (bins[:-1] + bins[1:])


def _quantile_from_hist(bins: np.ndarray, counts: np.ndarray, q: float) -> float:
    total = int(np.sum(counts))
    if total <= 0:
        return float("nan")
    target = q * float(total - 1)
    cdf = np.cumsum(counts.astype(np.float64))
    idx = int(np.searchsorted(cdf, target, side="left"))
    if idx < 0:
        idx = 0
    if idx >= counts.shape[0]:
        idx = counts.shape[0] - 1
    mids = _midpoints(bins)
    return float(mids[idx])


def _write_summary_csv(
    out_csv: str,
    *,
    event_cnt: np.ndarray,
    support_cnt: np.ndarray,
    support_per_event_sum: np.ndarray,
    sum_d: np.ndarray,
    sum_dt: np.ndarray,
    sum_v: np.ndarray,
    max_supports: np.ndarray,
    hist_d: np.ndarray,
    hist_dt: np.ndarray,
    hist_v: np.ndarray,
    d_bins: np.ndarray,
    dt_bins: np.ndarray,
    v_bins: np.ndarray,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)

    fields = [
        "scope",
        "class",
        "events",
        "supports",
        "supports_per_event_mean",
        "supports_per_event_max",
        "d_mean",
        "d_p50",
        "d_p90",
        "dt_us_mean",
        "dt_us_p50",
        "dt_us_p90",
        "v_px_per_us_mean",
        "v_px_per_us_p50",
        "v_px_per_us_p90",
    ]

    rows: list[dict[str, object]] = []
    for scope_idx, scope_name in enumerate(SCOPE_NAMES):
        for cls_name, cls_idx in (("noise", 0), ("signal", 1)):
            events = int(event_cnt[scope_idx, cls_idx])
            supports = int(support_cnt[scope_idx, cls_idx])
            row = {
                "scope": scope_name,
                "class": cls_name,
                "events": events,
                "supports": supports,
                "supports_per_event_mean": float(support_per_event_sum[scope_idx, cls_idx] / float(max(1, events))),
                "supports_per_event_max": int(max_supports[scope_idx, cls_idx]),
                "d_mean": float(sum_d[scope_idx, cls_idx] / float(max(1, supports))),
                "d_p50": _quantile_from_hist(d_bins, hist_d[scope_idx, cls_idx], 0.50),
                "d_p90": _quantile_from_hist(d_bins, hist_d[scope_idx, cls_idx], 0.90),
                "dt_us_mean": float(sum_dt[scope_idx, cls_idx] / float(max(1, supports))),
                "dt_us_p50": _quantile_from_hist(dt_bins, hist_dt[scope_idx, cls_idx], 0.50),
                "dt_us_p90": _quantile_from_hist(dt_bins, hist_dt[scope_idx, cls_idx], 0.90),
                "v_px_per_us_mean": float(sum_v[scope_idx, cls_idx] / float(max(1, supports))),
                "v_px_per_us_p50": _quantile_from_hist(v_bins, hist_v[scope_idx, cls_idx], 0.50),
                "v_px_per_us_p90": _quantile_from_hist(v_bins, hist_v[scope_idx, cls_idx], 0.90),
            }
            rows.append(row)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _write_hist_csv(
    out_csv: str,
    *,
    hist_d: np.ndarray,
    hist_dt: np.ndarray,
    hist_v: np.ndarray,
    hist_2d: np.ndarray,
    d_bins: np.ndarray,
    dt_bins: np.ndarray,
    v_bins: np.ndarray,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "scope",
                "metric",
                "class",
                "bin_lo",
                "bin_hi",
                "bin_lo_2",
                "bin_hi_2",
                "count",
                "ratio",
            ],
        )
        w.writeheader()

        for scope_idx, scope_name in enumerate(SCOPE_NAMES):
            for metric, hist, bins in (
                ("d", hist_d, d_bins),
                ("dt_us", hist_dt, dt_bins),
                ("v_px_per_us", hist_v, v_bins),
            ):
                for cls_name, cls_idx in (("noise", 0), ("signal", 1)):
                    total = float(np.sum(hist[scope_idx, cls_idx]))
                    for bi in range(int(hist.shape[2])):
                        cnt = int(hist[scope_idx, cls_idx, bi])
                        w.writerow(
                            {
                                "scope": scope_name,
                                "metric": metric,
                                "class": cls_name,
                                "bin_lo": float(bins[bi]),
                                "bin_hi": float(bins[bi + 1]),
                                "bin_lo_2": "",
                                "bin_hi_2": "",
                                "count": cnt,
                                "ratio": float(cnt / total) if total > 0 else float("nan"),
                            }
                        )

            metric = "d_dt_2d"
            for cls_name, cls_idx in (("noise", 0), ("signal", 1)):
                total = float(np.sum(hist_2d[scope_idx, cls_idx]))
                for bi in range(int(d_bins.shape[0]) - 1):
                    for bj in range(int(dt_bins.shape[0]) - 1):
                        cnt = int(hist_2d[scope_idx, cls_idx, bi, bj])
                        w.writerow(
                            {
                                "scope": scope_name,
                                "metric": metric,
                                "class": cls_name,
                                "bin_lo": float(d_bins[bi]),
                                "bin_hi": float(d_bins[bi + 1]),
                                "bin_lo_2": float(dt_bins[bj]),
                                "bin_hi_2": float(dt_bins[bj + 1]),
                                "count": cnt,
                                "ratio": float(cnt / total) if total > 0 else float("nan"),
                            }
                        )


def _plot_1d_hist(
    *,
    out_png: str,
    title: str,
    xlabel: str,
    bins: np.ndarray,
    hist: np.ndarray,
    logx: bool = False,
) -> None:
    mids = _midpoints(bins)
    noise = hist[0].astype(np.float64)
    signal = hist[1].astype(np.float64)
    noise = noise / max(1.0, float(np.sum(noise)))
    signal = signal / max(1.0, float(np.sum(signal)))

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    ax.plot(mids, noise, label="noise", color="#d62728", linewidth=1.8)
    ax.plot(mids, signal, label="signal", color="#2ca02c", linewidth=1.8)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("ratio")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _plot_2d_heatmap(
    *,
    out_png: str,
    title: str,
    d_bins: np.ndarray,
    dt_bins: np.ndarray,
    hist_2d_scope: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharex=True, sharey=True, constrained_layout=True)
    mesh_args = dict(shading="auto", cmap="magma")

    pcm = None
    for ax, cls_name, cls_idx in zip(axes, ("noise", "signal"), (0, 1)):
        data = hist_2d_scope[cls_idx].astype(np.float64)
        pcm = ax.pcolormesh(d_bins, dt_bins, np.log1p(data).T, **mesh_args)
        ax.set_xscale("linear")
        ax.set_yscale("log")
        ax.set_title(f"{title} - {cls_name}")
        ax.set_xlabel("d (px)")
        ax.set_ylabel("dt (us)")
        ax.grid(False)

    if pcm is not None:
        cbar = fig.colorbar(pcm, ax=axes.ravel().tolist(), shrink=0.92)
        cbar.set_label("log1p(count)")
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def center_relative_support_stats(
    *,
    labeled_npy: str,
    out_dir: str,
    width: int,
    height: int,
    tick_ns: float,
    start_events: int,
    max_events: int,
    s: int,
    tau_us: int,
    include_center: bool,
    d_bins_n: int,
    dt_bins_n: int,
    v_bins_n: int,
) -> None:
    _require_numba()
    ev = load_labeled_npy(labeled_npy, start_events=int(start_events), max_events=int(max_events))

    w = int(width)
    h = int(height)
    if w <= 0 or h <= 0:
        raise SystemExit(f"invalid dims: {w}x{h}")

    radius_px = max(0, (int(s) - 1) // 2)
    tau_us_f = float(tau_us)
    if tau_us_f <= 0.0:
        tau_us_f = 1.0

    tick_ns_f = float(tick_ns)
    if tick_ns_f <= 0.0:
        tick_ns_f = 1000.0
    tick_us = tick_ns_f / 1000.0

    d_max = np.sqrt(2.0) * float(radius_px)
    if d_max <= 0.0:
        d_max = 1.0
    d_bins = np.linspace(0.0, d_max, int(d_bins_n) + 1, dtype=np.float64)
    dt_bins = np.logspace(0.0, np.log10(tau_us_f), int(dt_bins_n) + 1, dtype=np.float64)
    v_bins = np.logspace(-6.0, 1.0, int(v_bins_n) + 1, dtype=np.float64)

    stats = _compute_center_relative_stats(
        ev.t,
        ev.x,
        ev.y,
        ev.p,
        ev.label,
        w,
        h,
        radius_px,
        tau_us_f,
        tick_us,
        1 if include_center else 0,
        d_bins,
        dt_bins,
        v_bins,
    )

    (
        event_cnt,
        support_cnt,
        support_per_event_sum,
        sum_d,
        sum_dt,
        sum_v,
        max_supports,
        hist_d,
        hist_dt,
        hist_v,
        hist_2d,
    ) = stats

    os.makedirs(out_dir, exist_ok=True)
    summary_csv = os.path.join(out_dir, "summary_center_relative.csv")
    hist_csv = os.path.join(out_dir, "hist_center_relative.csv")

    _write_summary_csv(
        summary_csv,
        event_cnt=event_cnt,
        support_cnt=support_cnt,
        support_per_event_sum=support_per_event_sum,
        sum_d=sum_d,
        sum_dt=sum_dt,
        sum_v=sum_v,
        max_supports=max_supports,
        hist_d=hist_d,
        hist_dt=hist_dt,
        hist_v=hist_v,
        d_bins=d_bins,
        dt_bins=dt_bins,
        v_bins=v_bins,
    )
    _write_hist_csv(
        hist_csv,
        hist_d=hist_d,
        hist_dt=hist_dt,
        hist_v=hist_v,
        hist_2d=hist_2d,
        d_bins=d_bins,
        dt_bins=dt_bins,
        v_bins=v_bins,
    )

    for scope_idx, scope_name in enumerate(SCOPE_NAMES):
        suffix = scope_name
        _plot_1d_hist(
            out_png=os.path.join(out_dir, f"figure_1_d_distribution_{suffix}.png"),
            title=f"Center-relative distance distribution ({scope_name})",
            xlabel="d (px)",
            bins=d_bins,
            hist=hist_d[scope_idx],
            logx=False,
        )
        _plot_1d_hist(
            out_png=os.path.join(out_dir, f"figure_2_dt_distribution_{suffix}.png"),
            title=f"Center-relative time-lag distribution ({scope_name})",
            xlabel="dt (us)",
            bins=dt_bins,
            hist=hist_dt[scope_idx],
            logx=True,
        )
        _plot_1d_hist(
            out_png=os.path.join(out_dir, f"figure_3_v_distribution_{suffix}.png"),
            title=f"Center-relative apparent speed distribution ({scope_name})",
            xlabel="v = d / dt (px/us)",
            bins=v_bins,
            hist=hist_v[scope_idx],
            logx=True,
        )
        _plot_2d_heatmap(
            out_png=os.path.join(out_dir, f"figure_4_d_dt_heatmap_{suffix}.png"),
            title=f"Center-relative (d, dt) heatmap ({scope_name})",
            d_bins=d_bins,
            dt_bins=dt_bins,
            hist_2d_scope=hist_2d[scope_idx],
        )

    # Backward-compatible aliases for the all-support scope.
    for i in range(1, 5):
        src = os.path.join(out_dir, f"figure_{i}_{'d_distribution' if i == 1 else 'dt_distribution' if i == 2 else 'v_distribution' if i == 3 else 'd_dt_heatmap'}_all.png")
        dst = os.path.join(out_dir, f"figure_{i}_{'d_distribution' if i == 1 else 'dt_distribution' if i == 2 else 'v_distribution' if i == 3 else 'd_dt_heatmap'}.png")
        if os.path.exists(src):
            with open(src, "rb") as rf, open(dst, "wb") as wf:
                wf.write(rf.read())

    print(f"saved: {summary_csv}")
    print(f"saved: {hist_csv}")
    print("saved: per-scope figures for all/top1/top3")


def main() -> int:
    ap = argparse.ArgumentParser(description="Center-relative support statistics for 7.47-style analysis.")
    ap.add_argument("--labeled-npy", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--width", type=int, default=346)
    ap.add_argument("--height", type=int, default=260)
    ap.add_argument("--tick-ns", type=float, default=1000.0)
    ap.add_argument("--start-events", type=int, default=0)
    ap.add_argument("--max-events", type=int, default=400000)
    ap.add_argument("--s", type=int, default=9)
    ap.add_argument("--tau-us", type=int, default=128000)
    ap.add_argument("--include-center", action="store_true", help="Include same-pixel historical support (default: exclude).")
    ap.add_argument("--d-bins", type=int, default=48)
    ap.add_argument("--dt-bins", type=int, default=48)
    ap.add_argument("--v-bins", type=int, default=48)
    args = ap.parse_args()

    center_relative_support_stats(
        labeled_npy=str(args.labeled_npy),
        out_dir=str(args.out_dir),
        width=int(args.width),
        height=int(args.height),
        tick_ns=float(args.tick_ns),
        start_events=int(args.start_events),
        max_events=int(args.max_events),
        s=int(args.s),
        tau_us=int(args.tau_us),
        include_center=bool(args.include_center),
        d_bins_n=int(args.d_bins),
        dt_bins_n=int(args.dt_bins),
        v_bins_n=int(args.v_bins),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
