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
        raise SystemExit("n125_path_failure_stats_773 requires numba")


def _parse_env_list(s: str) -> list[str]:
    out: list[str] = []
    for it in str(s).split(","):
        v = it.strip().lower()
        if not v:
            continue
        if v not in {"light", "mid", "heavy"}:
            raise SystemExit(f"invalid env: {v}")
        out.append(v)
    if not out:
        raise SystemExit("empty --env-list")
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
def _kernel_anisotropy(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    width: int,
    height: int,
    tau_ticks: int,
    min_pts: int,
    hist: np.ndarray,
    stats_cnt: np.ndarray,
    stats_sum: np.ndarray,
) -> None:
    """
    hist: anisotropy histogram in [0,1], n_bins bins.
    stats_cnt:
      [0]=valid_events
      [1]=eligible_events (same-pol points >= min_pts)
    stats_sum:
      [0]=score_sum
      [1]=score_sq_sum
    """

    n = int(t.shape[0])
    w = int(width)
    h = int(height)
    npx = w * h
    tau = int(tau_ticks)
    if tau <= 0:
        tau = 1
    mmin = int(min_pts)
    if mmin < 1:
        mmin = 1

    n_bins = int(hist.shape[0])
    eps = 1e-3

    last_ts = np.zeros((npx,), dtype=np.uint64)
    last_pol = np.zeros((npx,), dtype=np.int8)

    for i in range(n):
        xi = int(x[i])
        yi = int(y[i])
        if xi < 0 or xi >= w or yi < 0 or yi >= h:
            continue

        ti = np.uint64(t[i])
        pi = 1 if int(p[i]) > 0 else -1
        stats_cnt[0] += 1

        m20 = 0.0
        m02 = 0.0
        m11 = 0.0
        n_same = 0

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

        for yy in range(y0, y1 + 1):
            base = yy * w
            dyi = yy - yi
            for xx in range(x0, x1 + 1):
                if xx == xi and yy == yi:
                    continue
                idx = base + xx
                if int(last_pol[idx]) != pi:
                    continue
                ts = last_ts[idx]
                if ts == 0 or ts >= ti:
                    continue
                dt = int(ti - ts)
                if dt > tau:
                    continue

                wj = 1.0 - float(dt) / float(tau)
                if wj <= 0.0:
                    continue

                dxi = xx - xi
                dxf = float(dxi)
                dyf = float(dyi)
                m20 += wj * dxf * dxf
                m02 += wj * dyf * dyf
                m11 += wj * dxf * dyf
                n_same += 1

        if n_same >= mmin:
            ttrace = m20 + m02
            det = m20 * m02 - m11 * m11
            if det < 0.0:
                det = 0.0
            score = 1.0 - (4.0 * det) / (ttrace * ttrace + eps)
            if score < 0.0:
                score = 0.0
            if score > 1.0:
                score = 1.0

            stats_cnt[1] += 1
            stats_sum[0] += score
            stats_sum[1] += score * score

            b = int(score * float(n_bins))
            if b >= n_bins:
                b = n_bins - 1
            if b < 0:
                b = 0
            hist[b] += 1

        idx0 = yi * w + xi
        last_ts[idx0] = ti
        last_pol[idx0] = np.int8(pi)


def _safe_ratio(a: float, b: float) -> float:
    if b <= 0:
        return float("nan")
    return float(a) / float(b)


def run_one_env(
    *,
    env_name: str,
    labeled_npy: str,
    width: int,
    height: int,
    tick_ns: float,
    max_events: int,
    tau_us: int,
    min_pts: int,
    bins: int,
) -> dict:
    ev = load_labeled_npy(str(labeled_npy), max_events=int(max_events))
    tick_us = float(tick_ns) / 1000.0
    if tick_us <= 0:
        tick_us = 1.0
    tau_ticks = int(round(float(tau_us) / tick_us))

    hist = np.zeros((int(bins),), dtype=np.int64)
    stats_cnt = np.zeros((2,), dtype=np.int64)
    stats_sum = np.zeros((2,), dtype=np.float64)

    _kernel_anisotropy(
        ev.t,
        ev.x,
        ev.y,
        ev.p,
        int(width),
        int(height),
        int(tau_ticks),
        int(min_pts),
        hist,
        stats_cnt,
        stats_sum,
    )

    n_valid = int(stats_cnt[0])
    n_elig = int(stats_cnt[1])
    mean = _safe_ratio(stats_sum[0], n_elig)
    var = _safe_ratio(stats_sum[1], n_elig) - mean * mean if n_elig > 0 else float("nan")
    if np.isfinite(var) and var < 0.0:
        var = 0.0
    std = float(np.sqrt(var)) if np.isfinite(var) else float("nan")

    return {
        "env": str(env_name),
        "input": str(labeled_npy),
        "events": n_valid,
        "tau_us": int(tau_us),
        "window": "9x9",
        "min_pts": int(min_pts),
        "eligible_events": n_elig,
        "eligible_rate": _safe_ratio(n_elig, n_valid),
        "anisotropy_mean": mean,
        "anisotropy_std": std,
        "hist_bins": int(bins),
        "hist_range": [0.0, 1.0],
        "hist_count": hist.tolist(),
        "hist_ratio": [_safe_ratio(int(v), n_elig) for v in hist.tolist()],
        "high_08_10_ratio": _safe_ratio(int(np.sum(hist[int(0.8 * bins) :])), n_elig),
        "low_00_03_ratio": _safe_ratio(int(np.sum(hist[: int(0.3 * bins)])), n_elig),
    }


def write_summary_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields = [
        "env",
        "events",
        "eligible_events",
        "eligible_rate",
        "anisotropy_mean",
        "anisotropy_std",
        "high_08_10_ratio",
        "low_00_03_ratio",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})


def main() -> int:
    _require_numba()
    ap = argparse.ArgumentParser(description="7.73 anisotropy tensor statistics for Light/Heavy.")
    ap.add_argument("--env-list", default="light,heavy")
    ap.add_argument("--light", default=r"D:\hjx_workspace\scientific_reserach\dataset\ED24\myPedestrain_06\Pedestrain_06_1.8.npy")
    ap.add_argument("--mid", default=r"D:\hjx_workspace\scientific_reserach\dataset\ED24\myPedestrain_06\Pedestrain_06_2.5.npy")
    ap.add_argument("--heavy", default=r"D:\hjx_workspace\scientific_reserach\dataset\ED24\myPedestrain_06\Pedestrain_06_3.3.npy")
    ap.add_argument("--width", type=int, default=346)
    ap.add_argument("--height", type=int, default=260)
    ap.add_argument("--tick-ns", type=float, default=1000.0)
    ap.add_argument("--max-events", type=int, default=500000)
    ap.add_argument("--tau-us", type=int, default=30000)
    ap.add_argument("--min-pts", type=int, default=4)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument(
        "--out-dir",
        default="data/ED24/myPedestrain_06/EBF_Part2/_n125_path_failure_stats_773",
    )
    args = ap.parse_args()

    env_list = _parse_env_list(args.env_list)
    in_map = {
        "light": str(args.light),
        "mid": str(args.mid),
        "heavy": str(args.heavy),
    }

    out_dir = os.path.abspath(str(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    results: list[dict] = []
    for env_name in env_list:
        in_path = in_map[env_name]
        if not os.path.exists(in_path):
            raise SystemExit(f"missing input for {env_name}: {in_path}")

        print(f"[RUN] {env_name} input={in_path}")
        rec = run_one_env(
            env_name=env_name,
            labeled_npy=in_path,
            width=int(args.width),
            height=int(args.height),
            tick_ns=float(args.tick_ns),
            max_events=int(args.max_events),
            tau_us=int(args.tau_us),
            min_pts=int(args.min_pts),
            bins=int(args.bins),
        )
        results.append(rec)
        print(
            f"[OK] {env_name} events={rec['events']} eligible={rec['eligible_rate']:.4f} "
            f"anis_mean={rec['anisotropy_mean']:.4f} high[0.8,1]={rec['high_08_10_ratio']:.4f}"
        )

    out_json = os.path.join(out_dir, "summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    out_csv = os.path.join(out_dir, "summary.csv")
    write_summary_csv(out_csv, results)

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
