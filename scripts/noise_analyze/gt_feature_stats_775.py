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
        raise SystemExit("gt_feature_stats_775 requires numba")


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
def _fill_hist_velocity(v: float, hist: np.ndarray) -> None:
    # bins: [0,0.1), [0.1,0.5), [0.5,1), [1,5), [5,+inf)
    if not (v >= 0.0):
        return
    if v < 0.1:
        hist[0] += 1
    elif v < 0.5:
        hist[1] += 1
    elif v < 1.0:
        hist[2] += 1
    elif v < 5.0:
        hist[3] += 1
    else:
        hist[4] += 1


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
    eta_mix: float,
    hist_outer_noise: np.ndarray,
    hist_outer_signal: np.ndarray,
    hist_mix_noise: np.ndarray,
    hist_mix_signal: np.ndarray,
    hist_aniso_noise: np.ndarray,
    hist_aniso_signal: np.ndarray,
    hist_vel_noise: np.ndarray,
    hist_vel_signal: np.ndarray,
    stats_count: np.ndarray,
    stats_sum: np.ndarray,
    stats_sqsum: np.ndarray,
) -> None:
    n = int(t.shape[0])
    w = int(width)
    h = int(height)
    npx = w * h
    tau = int(tau_ticks)
    if tau <= 0:
        tau = 1

    eps = 1e-3
    rmax = 4.0
    inv_rmax2 = 1.0 / (rmax * rmax)

    eta = float(eta_mix)
    if eta < 0.0:
        eta = 0.0
    if eta > 1.0:
        eta = 1.0

    last_ts = np.zeros((npx,), dtype=np.uint64)
    last_pol = np.zeros((npx,), dtype=np.int8)

    for i in range(n):
        xi = int(x[i])
        yi = int(y[i])
        if xi < 0 or xi >= w or yi < 0 or yi >= h:
            continue

        ti = np.uint64(t[i])
        pi = 1 if int(p[i]) > 0 else -1
        cls = 1 if int(label[i]) > 0 else 0

        y0 = yi - 4
        if y0 < 0:
            y0 = 0
        y1 = yi + 4
        if y1 >= h:
            y1 = h - 1
        x0 = xi - 4
        if x0 < 0:
            x0 = 0
        x1 = xi + 4
        if x1 >= w:
            x1 = w - 1

        # feature 1: outer-ring energy ratio
        e_in = 0.0
        e_out = 0.0

        # feature 2: smooth polarity mix
        e_same_mix = 0.0
        e_opp_mix = 0.0

        # feature 3: anisotropy tensor
        m20 = 0.0
        m02 = 0.0
        m11 = 0.0
        sum_w_aniso = 0.0

        # feature 4: mean apparent velocity
        vel_wsum = 0.0
        vel_sum = 0.0

        for yy in range(y0, y1 + 1):
            base = yy * w
            dy = yy - yi
            ady = dy if dy >= 0 else -dy

            for xx in range(x0, x1 + 1):
                if xx == xi and yy == yi:
                    continue
                idx = base + xx
                ts = last_ts[idx]
                if ts == 0 or ts >= ti:
                    continue
                dt_ticks = int(ti - ts)
                if dt_ticks > tau:
                    continue
                w_time = 1.0 - float(dt_ticks) / float(tau)
                if w_time <= 0.0:
                    continue

                dx = xx - xi
                adx = dx if dx >= 0 else -dx
                d = adx if adx >= ady else ady

                same_pol = int(last_pol[idx]) == pi

                if same_pol:
                    if d <= 2:
                        e_in += w_time
                    else:
                        e_out += w_time

                    dxf = float(dx)
                    dyf = float(dy)
                    m20 += w_time * dxf * dxf
                    m02 += w_time * dyf * dyf
                    m11 += w_time * dxf * dyf
                    sum_w_aniso += w_time

                    dt_ms = float(dt_ticks) / 1000.0
                    if dt_ms > 0.0:
                        vj = float(d) / dt_ms
                        vel_wsum += w_time * vj
                        vel_sum += w_time

                w_space = 1.0 - eta * (float(d * d) * inv_rmax2)
                if w_space <= 0.0:
                    continue
                w_joint = w_time * w_space
                if same_pol:
                    e_same_mix += w_joint
                else:
                    e_opp_mix += w_joint

        # Outer-ring ratio
        den_outer = e_in + e_out + eps
        outer_ratio = e_out / den_outer
        if cls == 0:
            _fill_hist_uniform(outer_ratio, hist_outer_noise)
            stats_count[0, 0] += 1
            stats_sum[0, 0] += outer_ratio
            stats_sqsum[0, 0] += outer_ratio * outer_ratio
        else:
            _fill_hist_uniform(outer_ratio, hist_outer_signal)
            stats_count[1, 0] += 1
            stats_sum[1, 0] += outer_ratio
            stats_sqsum[1, 0] += outer_ratio * outer_ratio

        # Smooth mix (skip sparse support)
        den_mix = e_same_mix + e_opp_mix
        if den_mix >= 1.0:
            mix = e_opp_mix / (den_mix + eps)
            if cls == 0:
                _fill_hist_uniform(mix, hist_mix_noise)
                stats_count[0, 1] += 1
                stats_sum[0, 1] += mix
                stats_sqsum[0, 1] += mix * mix
            else:
                _fill_hist_uniform(mix, hist_mix_signal)
                stats_count[1, 1] += 1
                stats_sum[1, 1] += mix
                stats_sqsum[1, 1] += mix * mix

        # Anisotropy
        if sum_w_aniso >= 1.5:
            tr = m20 + m02
            det = m20 * m02 - m11 * m11
            if det < 0.0:
                det = 0.0
            aniso = 1.0 - (4.0 * det) / (tr * tr + eps)
            if aniso < 0.0:
                aniso = 0.0
            if aniso > 1.0:
                aniso = 1.0
            if cls == 0:
                _fill_hist_uniform(aniso, hist_aniso_noise)
                stats_count[0, 2] += 1
                stats_sum[0, 2] += aniso
                stats_sqsum[0, 2] += aniso * aniso
            else:
                _fill_hist_uniform(aniso, hist_aniso_signal)
                stats_count[1, 2] += 1
                stats_sum[1, 2] += aniso
                stats_sqsum[1, 2] += aniso * aniso

        # Mean velocity
        if vel_sum > 0.0:
            vmean = vel_wsum / vel_sum
            if cls == 0:
                _fill_hist_velocity(vmean, hist_vel_noise)
                stats_count[0, 3] += 1
                stats_sum[0, 3] += vmean
                stats_sqsum[0, 3] += vmean * vmean
            else:
                _fill_hist_velocity(vmean, hist_vel_signal)
                stats_count[1, 3] += 1
                stats_sum[1, 3] += vmean
                stats_sqsum[1, 3] += vmean * vmean

        idx0 = yi * w + xi
        last_ts[idx0] = ti
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
    eta_mix: float,
    bins_uniform: int,
) -> tuple[list[dict], list[dict], dict]:
    ev = load_labeled_npy(str(labeled_npy), max_events=int(max_events))
    tick_us = float(tick_ns) / 1000.0
    if tick_us <= 0:
        tick_us = 1.0
    tau_ticks = int(round(float(tau_us) / tick_us))

    bu = int(bins_uniform)
    if bu < 5:
        bu = 5

    h_outer_noise = np.zeros((bu,), dtype=np.int64)
    h_outer_signal = np.zeros((bu,), dtype=np.int64)
    h_mix_noise = np.zeros((bu,), dtype=np.int64)
    h_mix_signal = np.zeros((bu,), dtype=np.int64)
    h_aniso_noise = np.zeros((bu,), dtype=np.int64)
    h_aniso_signal = np.zeros((bu,), dtype=np.int64)
    h_vel_noise = np.zeros((5,), dtype=np.int64)
    h_vel_signal = np.zeros((5,), dtype=np.int64)

    # stats arrays dims: [class(0=noise,1=signal), feature]
    # feature order: outer, mix, anisotropy, velocity
    cnt = np.zeros((2, 4), dtype=np.int64)
    sm = np.zeros((2, 4), dtype=np.float64)
    sq = np.zeros((2, 4), dtype=np.float64)

    _kernel_stats(
        ev.t,
        ev.x,
        ev.y,
        ev.p,
        ev.label,
        int(width),
        int(height),
        int(tau_ticks),
        float(eta_mix),
        h_outer_noise,
        h_outer_signal,
        h_mix_noise,
        h_mix_signal,
        h_aniso_noise,
        h_aniso_signal,
        h_vel_noise,
        h_vel_signal,
        cnt,
        sm,
        sq,
    )

    features = [
        ("outer_ratio", 0),
        ("mix_smooth", 1),
        ("anisotropy", 2),
        ("velocity_mean", 3),
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

    hist_rows: list[dict] = []
    uniform_specs = [
        ("outer_ratio", h_outer_noise, h_outer_signal),
        ("mix_smooth", h_mix_noise, h_mix_signal),
        ("anisotropy", h_aniso_noise, h_aniso_signal),
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

    vel_edges = [0.0, 0.1, 0.5, 1.0, 5.0, float("inf")]
    for cname, hh in (("noise", h_vel_noise), ("signal", h_vel_signal)):
        n_all = int(np.sum(hh))
        for bi in range(5):
            c = int(hh[bi])
            hist_rows.append(
                {
                    "feature": "velocity_mean",
                    "class": cname,
                    "bin_index": int(bi),
                    "bin_lo": float(vel_edges[bi]),
                    "bin_hi": float(vel_edges[bi + 1]),
                    "count": c,
                    "ratio": _safe_ratio(c, n_all),
                }
            )

    meta = {
        "input": str(labeled_npy),
        "events": int(ev.t.shape[0]),
        "tau_us": int(tau_us),
        "window": "9x9",
        "eta_mix": float(eta_mix),
        "bins_uniform": int(bu),
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

    ap = argparse.ArgumentParser(description="7.75 GT-based feature statistics in Heavy env.")
    ap.add_argument("--labeled-npy", default=r"D:\hjx_workspace\scientific_reserach\dataset\ED24\myPedestrain_06\Pedestrain_06_3.3.npy")
    ap.add_argument("--width", type=int, default=346)
    ap.add_argument("--height", type=int, default=260)
    ap.add_argument("--tick-ns", type=float, default=1000.0)
    ap.add_argument("--max-events", type=int, default=500000)
    ap.add_argument("--tau-us", type=int, default=30000)
    ap.add_argument("--eta-mix", type=float, default=0.5)
    ap.add_argument("--bins-uniform", type=int, default=20)
    ap.add_argument("--out-dir", default="data/ED24/myPedestrain_06/EBF_Part2/_gt_feature_stats_775")
    args = ap.parse_args()

    summary_rows, hist_rows, meta = run_stats(
        labeled_npy=str(args.labeled_npy),
        width=int(args.width),
        height=int(args.height),
        tick_ns=float(args.tick_ns),
        max_events=int(args.max_events),
        tau_us=int(args.tau_us),
        eta_mix=float(args.eta_mix),
        bins_uniform=int(args.bins_uniform),
    )

    out_dir = os.path.abspath(str(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    out_summary = os.path.join(out_dir, "summary.csv")
    out_hist = os.path.join(out_dir, "hist.csv")
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
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "summary": summary_rows}, f, ensure_ascii=False, indent=2)

    print(f"Saved: {out_summary}")
    print(f"Saved: {out_hist}")
    print(f"Saved: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
