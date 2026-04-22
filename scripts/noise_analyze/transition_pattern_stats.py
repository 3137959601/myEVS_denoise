from __future__ import annotations

import argparse
import csv
import math
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
        raise SystemExit("transition_pattern_stats requires numba")


def _parse_int_list(s: str) -> list[int]:
    out: list[int] = []
    for it in str(s).split(","):
        it = it.strip()
        if not it:
            continue
        out.append(int(float(it)))
    if not out:
        raise SystemExit("empty integer list")
    return out


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


def _safe_div(a: float, b: float) -> float:
    if b == 0:
        return float("nan")
    return float(a) / float(b)


@numba.njit(cache=True)
def _update_best2(cand: int, t: np.ndarray, best1: int, best2: int) -> tuple[int, int]:
    if cand < 0:
        return best1, best2
    if best1 < 0:
        return cand, best2
    if cand == best1:
        return best1, best2
    # For prev stats we need the most recent history events (largest timestamp).
    if t[cand] > t[best1]:
        return cand, best1
    if best2 < 0:
        return best1, cand
    if cand == best2:
        return best1, best2
    if t[cand] > t[best2]:
        return best1, cand
    return best1, best2


@numba.njit(cache=True)
def _compute_prev_stats(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    label: np.ndarray,
    width: int,
    height: int,
    windows: np.ndarray,
    very_fast_ticks: int,
    fast_ticks: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = int(t.shape[0])
    npx = int(width) * int(height)
    nw = int(windows.shape[0])

    same_dt1 = np.full((n,), -1, dtype=np.int64)
    same_dt2 = np.full((n,), -1, dtype=np.int64)
    same_pol1 = np.full((n,), -1, dtype=np.int8)
    same_pol2 = np.full((n,), -1, dtype=np.int8)

    neigh_dt1 = np.full((nw, n), -1, dtype=np.int64)
    neigh_dt2 = np.full((nw, n), -1, dtype=np.int64)
    neigh_pol1 = np.full((nw, n), -1, dtype=np.int8)
    neigh_pol2 = np.full((nw, n), -1, dtype=np.int8)
    neigh_joint_mode = np.full((nw, n), -1, dtype=np.int8)
    neigh_joint_bucket = np.full((nw, n), 3, dtype=np.int8)  # 0=very_fast,1=fast,2=medium,3=miss

    prev1_noise = np.full((npx,), -1, dtype=np.int32)
    prev2_noise = np.full((npx,), -1, dtype=np.int32)
    prev1_sig = np.full((npx,), -1, dtype=np.int32)
    prev2_sig = np.full((npx,), -1, dtype=np.int32)

    for i in range(n):
        xi = int(x[i])
        yi = int(y[i])
        if xi < 0 or xi >= width or yi < 0 or yi >= height:
            continue
        cls_sig = int(label[i])
        idx = yi * width + xi

        if cls_sig > 0:
            j1 = int(prev1_sig[idx])
            j2 = int(prev2_sig[idx])
        else:
            j1 = int(prev1_noise[idx])
            j2 = int(prev2_noise[idx])

        if j1 >= 0:
            same_dt1[i] = np.int64(t[i] - t[j1])
            same_pol1[i] = np.int8(1 if int(p[j1]) == int(p[i]) else 0)
        if j2 >= 0:
            same_dt2[i] = np.int64(t[i] - t[j2])
            same_pol2[i] = np.int8(1 if int(p[j2]) == int(p[i]) else 0)

        for wi in range(nw):
            win = int(windows[wi])
            r = win // 2
            x0 = xi - r
            x1 = xi + r
            y0 = yi - r
            y1 = yi + r
            if x0 < 0:
                x0 = 0
            if y0 < 0:
                y0 = 0
            if x1 >= width:
                x1 = width - 1
            if y1 >= height:
                y1 = height - 1

            best1 = -1
            best2 = -1
            for yy in range(y0, y1 + 1):
                base = yy * width
                for xx in range(x0, x1 + 1):
                    jdx = base + xx
                    if cls_sig > 0:
                        a = int(prev1_sig[jdx])
                        b = int(prev2_sig[jdx])
                    else:
                        a = int(prev1_noise[jdx])
                        b = int(prev2_noise[jdx])
                    best1, best2 = _update_best2(a, t, best1, best2)
                    best1, best2 = _update_best2(b, t, best1, best2)

            if best1 >= 0:
                neigh_dt1[wi, i] = np.int64(t[i] - t[best1])
                neigh_pol1[wi, i] = np.int8(1 if int(p[best1]) == int(p[i]) else 0)
            if best2 >= 0:
                neigh_dt2[wi, i] = np.int64(t[i] - t[best2])
                neigh_pol2[wi, i] = np.int8(1 if int(p[best2]) == int(p[i]) else 0)

            # Joint pattern is fixed in older->newer order: prev2 -> prev1.
            if best1 >= 0 and best2 >= 0:
                s2 = 1 if int(p[best2]) == int(p[i]) else 0
                s1 = 1 if int(p[best1]) == int(p[i]) else 0
                mode = 0
                if s2 == 1 and s1 == 1:
                    mode = 0  # same->same
                elif s2 == 1 and s1 == 0:
                    mode = 1  # same->opp
                elif s2 == 0 and s1 == 1:
                    mode = 2  # opp->same
                else:
                    mode = 3  # opp->opp
                neigh_joint_mode[wi, i] = np.int8(mode)

                dt1 = int(t[i] - t[best1])
                if dt1 <= int(very_fast_ticks):
                    neigh_joint_bucket[wi, i] = np.int8(0)
                elif dt1 <= int(fast_ticks):
                    neigh_joint_bucket[wi, i] = np.int8(1)
                else:
                    neigh_joint_bucket[wi, i] = np.int8(2)

        if cls_sig > 0:
            prev2_sig[idx] = prev1_sig[idx]
            prev1_sig[idx] = i
        else:
            prev2_noise[idx] = prev1_noise[idx]
            prev1_noise[idx] = i

    return (
        same_dt1,
        same_dt2,
        same_pol1,
        same_pol2,
        neigh_dt1,
        neigh_dt2,
        neigh_pol1,
        neigh_pol2,
        neigh_joint_mode,
        neigh_joint_bucket,
    )


def _dt_quantiles_us(dt_ticks: np.ndarray, tick_ns: float) -> tuple[float, float, float, float, float]:
    if dt_ticks.size <= 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    us = dt_ticks.astype(np.float64) * (float(tick_ns) / 1000.0)
    q = np.quantile(us, [0.5, 0.9, 0.99])
    return float(us.mean()), float(np.median(us)), float(q[0]), float(q[1]), float(q[2])


def _write_summary_csv(
    out_csv: str,
    *,
    label: np.ndarray,
    same_dt1: np.ndarray,
    same_dt2: np.ndarray,
    same_pol1: np.ndarray,
    same_pol2: np.ndarray,
    neigh_dt1: np.ndarray,
    neigh_dt2: np.ndarray,
    neigh_pol1: np.ndarray,
    neigh_pol2: np.ndarray,
    neigh_joint_mode: np.ndarray,
    neigh_joint_bucket: np.ndarray,
    windows: list[int],
    tick_ns: float,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)

    rows: list[dict[str, object]] = []

    cls_defs = [(0, "noise"), (1, "signal")]
    base_scopes = [
        ("pixel", same_dt1, same_pol1, "pre1"),
        ("pixel", same_dt2, same_pol2, "pre2"),
    ]

    for cls_v, cls_name in cls_defs:
        m_cls = label == cls_v
        n_cls = int(m_cls.sum())

        for scope, dt_arr, pol_arr, step in base_scopes:
            dt = dt_arr[m_cls]
            pol = pol_arr[m_cls]
            valid = dt >= 0
            dt_v = dt[valid]
            pol_v = pol[valid]

            same_n = int((pol_v == 1).sum())
            opp_n = int((pol_v == 0).sum())
            hit_n = int(valid.sum())

            mean_us, med_us, p50_us, p90_us, p99_us = _dt_quantiles_us(dt_v, tick_ns)
            rows.append(
                {
                    "class": cls_name,
                    "scope": scope,
                    "step": step,
                    "window": "",
                    "events": n_cls,
                    "hits": hit_n,
                    "hit_rate": _safe_div(hit_n, n_cls),
                    "same_pol_rate": _safe_div(same_n, hit_n),
                    "opp_pol_rate": _safe_div(opp_n, hit_n),
                    "dt_mean_us": mean_us,
                    "dt_median_us": med_us,
                    "dt_p50_us": p50_us,
                    "dt_p90_us": p90_us,
                    "dt_p99_us": p99_us,
                }
            )

        for wi, win in enumerate(windows):
            for step, dt_arr, pol_arr in (
                ("pre1", neigh_dt1[wi], neigh_pol1[wi]),
                ("pre2", neigh_dt2[wi], neigh_pol2[wi]),
            ):
                dt = dt_arr[m_cls]
                pol = pol_arr[m_cls]
                valid = dt >= 0
                dt_v = dt[valid]
                pol_v = pol[valid]

                same_n = int((pol_v == 1).sum())
                opp_n = int((pol_v == 0).sum())
                hit_n = int(valid.sum())

                mean_us, med_us, p50_us, p90_us, p99_us = _dt_quantiles_us(dt_v, tick_ns)
                rows.append(
                    {
                        "class": cls_name,
                        "scope": "neighborhood",
                        "step": step,
                        "window": int(win),
                        "events": n_cls,
                        "hits": hit_n,
                        "hit_rate": _safe_div(hit_n, n_cls),
                        "same_pol_rate": _safe_div(same_n, hit_n),
                        "opp_pol_rate": _safe_div(opp_n, hit_n),
                        "dt_mean_us": mean_us,
                        "dt_median_us": med_us,
                        "dt_p50_us": p50_us,
                        "dt_p90_us": p90_us,
                        "dt_p99_us": p99_us,
                        "joint_pattern": "",
                        "dt_bucket": "",
                        "pattern_count": "",
                        "pattern_ratio": "",
                    }
                )

            # C + D: neighborhood joint past pattern statistics and dt bucket distribution.
            mode_arr = neigh_joint_mode[wi][m_cls]
            bucket_arr = neigh_joint_bucket[wi][m_cls]
            valid_joint = mode_arr >= 0
            joint_hits = int(valid_joint.sum())

            mode_names = ("same->same", "same->opp", "opp->same", "opp->opp")
            bucket_names = ("very_fast", "fast", "medium", "miss")

            for mi, mname in enumerate(mode_names):
                mm = mode_arr == mi
                mcnt = int(mm.sum())
                rows.append(
                    {
                        "class": cls_name,
                        "scope": "neighborhood",
                        "step": "joint_prev2_prev1",
                        "window": int(win),
                        "events": n_cls,
                        "hits": joint_hits,
                        "hit_rate": _safe_div(joint_hits, n_cls),
                        "same_pol_rate": "",
                        "opp_pol_rate": "",
                        "dt_mean_us": "",
                        "dt_median_us": "",
                        "dt_p50_us": "",
                        "dt_p90_us": "",
                        "dt_p99_us": "",
                        "joint_pattern": mname,
                        "dt_bucket": "",
                        "pattern_count": mcnt,
                        "pattern_ratio": _safe_div(mcnt, joint_hits),
                    }
                )

                for bi, bname in enumerate(bucket_names):
                    if bname == "miss":
                        bcnt = 0
                    else:
                        bcnt = int(np.logical_and(mm, bucket_arr == bi).sum())
                    rows.append(
                        {
                            "class": cls_name,
                            "scope": "neighborhood",
                            "step": "joint_prev2_prev1_dt_bucket",
                            "window": int(win),
                            "events": n_cls,
                            "hits": joint_hits,
                            "hit_rate": _safe_div(joint_hits, n_cls),
                            "same_pol_rate": "",
                            "opp_pol_rate": "",
                            "dt_mean_us": "",
                            "dt_median_us": "",
                            "dt_p50_us": "",
                            "dt_p90_us": "",
                            "dt_p99_us": "",
                            "joint_pattern": mname,
                            "dt_bucket": bname,
                            "pattern_count": bcnt,
                            "pattern_ratio": _safe_div(bcnt, mcnt),
                        }
                    )

            # Explicit miss rows (not enough prev2/prev1).
            miss_cnt = int((mode_arr < 0).sum())
            rows.append(
                {
                    "class": cls_name,
                    "scope": "neighborhood",
                    "step": "joint_prev2_prev1_dt_bucket",
                    "window": int(win),
                    "events": n_cls,
                    "hits": joint_hits,
                    "hit_rate": _safe_div(joint_hits, n_cls),
                    "same_pol_rate": "",
                    "opp_pol_rate": "",
                    "dt_mean_us": "",
                    "dt_median_us": "",
                    "dt_p50_us": "",
                    "dt_p90_us": "",
                    "dt_p99_us": "",
                    "joint_pattern": "all",
                    "dt_bucket": "miss",
                    "pattern_count": miss_cnt,
                    "pattern_ratio": _safe_div(miss_cnt, n_cls),
                }
            )

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "class",
                "scope",
                "step",
                "window",
                "events",
                "hits",
                "hit_rate",
                "same_pol_rate",
                "opp_pol_rate",
                "dt_mean_us",
                "dt_median_us",
                "dt_p50_us",
                "dt_p90_us",
                "dt_p99_us",
                "joint_pattern",
                "dt_bucket",
                "pattern_count",
                "pattern_ratio",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _build_log_bins_us() -> np.ndarray:
    # 1us to ~1s with log-like spacing.
    return np.asarray([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000, 1000000], dtype=np.float64)


def _write_hist_csv(
    out_csv: str,
    *,
    label: np.ndarray,
    same_dt1: np.ndarray,
    same_dt2: np.ndarray,
    neigh_dt1: np.ndarray,
    neigh_dt2: np.ndarray,
    windows: list[int],
    tick_ns: float,
) -> None:
    bins_us = _build_log_bins_us()
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)

    rows: list[dict[str, object]] = []
    cls_defs = [(0, "noise"), (1, "signal")]

    items: list[tuple[str, str, str, np.ndarray]] = [
        ("pixel", "", "pre1", same_dt1),
        ("pixel", "", "pre2", same_dt2),
    ]
    for wi, win in enumerate(windows):
        items.append(("neighborhood", str(int(win)), "pre1", neigh_dt1[wi]))
        items.append(("neighborhood", str(int(win)), "pre2", neigh_dt2[wi]))

    for cls_v, cls_name in cls_defs:
        m_cls = label == cls_v
        for scope, win, step, dt_arr in items:
            dt_ticks = dt_arr[m_cls]
            dt_ticks = dt_ticks[dt_ticks >= 0]
            if dt_ticks.size <= 0:
                continue
            dt_us = dt_ticks.astype(np.float64) * (float(tick_ns) / 1000.0)
            h, edges = np.histogram(dt_us, bins=bins_us)
            total = float(h.sum())
            for bi in range(h.size):
                cnt = int(h[bi])
                rows.append(
                    {
                        "class": cls_name,
                        "scope": scope,
                        "window": win,
                        "step": step,
                        "bin_lo_us": float(edges[bi]),
                        "bin_hi_us": float(edges[bi + 1]),
                        "count": cnt,
                        "ratio": _safe_div(cnt, total),
                    }
                )

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["class", "scope", "window", "step", "bin_lo_us", "bin_hi_us", "count", "ratio"],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)


def transition_pattern_stats(
    *,
    labeled_npy: str,
    out_summary_csv: str,
    out_hist_csv: str,
    width: int,
    height: int,
    tick_ns: float,
    start_events: int,
    max_events: int,
    windows: list[int],
    joint_very_fast_us: float,
    joint_fast_us: float,
) -> None:
    _require_numba()

    ev = load_labeled_npy(labeled_npy, start_events=int(start_events), max_events=int(max_events))

    windows = [int(w) for w in windows if int(w) > 0 and int(w) % 2 == 1]
    if not windows:
        raise SystemExit("--windows must contain odd positive ints, e.g. 7,9")

    tick_us = float(tick_ns) / 1000.0
    vf_ticks = int(max(1, round(float(joint_very_fast_us) / max(tick_us, 1e-9))))
    f_ticks = int(max(vf_ticks + 1, round(float(joint_fast_us) / max(tick_us, 1e-9))))

    (
        same_dt1,
        same_dt2,
        same_pol1,
        same_pol2,
        neigh_dt1,
        neigh_dt2,
        neigh_pol1,
        neigh_pol2,
        neigh_joint_mode,
        neigh_joint_bucket,
    ) = _compute_prev_stats(
        ev.t,
        ev.x,
        ev.y,
        ev.p,
        ev.label,
        int(width),
        int(height),
        np.asarray(windows, dtype=np.int32),
        int(vf_ticks),
        int(f_ticks),
    )

    _write_summary_csv(
        out_summary_csv,
        label=ev.label,
        same_dt1=same_dt1,
        same_dt2=same_dt2,
        same_pol1=same_pol1,
        same_pol2=same_pol2,
        neigh_dt1=neigh_dt1,
        neigh_dt2=neigh_dt2,
        neigh_pol1=neigh_pol1,
        neigh_pol2=neigh_pol2,
        neigh_joint_mode=neigh_joint_mode,
        neigh_joint_bucket=neigh_joint_bucket,
        windows=windows,
        tick_ns=float(tick_ns),
    )

    _write_hist_csv(
        out_hist_csv,
        label=ev.label,
        same_dt1=same_dt1,
        same_dt2=same_dt2,
        neigh_dt1=neigh_dt1,
        neigh_dt2=neigh_dt2,
        windows=windows,
        tick_ns=float(tick_ns),
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Compute previous-event transition statistics for same-pixel and neighborhood events. "
            "Outputs summary CSV and interval histogram CSV."
        )
    )
    ap.add_argument("--labeled-npy", required=True)
    ap.add_argument("--out-summary-csv", required=True)
    ap.add_argument("--out-hist-csv", required=True)
    ap.add_argument("--width", type=int, default=346)
    ap.add_argument("--height", type=int, default=260)
    ap.add_argument("--tick-ns", type=float, default=1000.0)
    ap.add_argument("--start-events", type=int, default=0)
    ap.add_argument("--max-events", type=int, default=400000)
    ap.add_argument("--windows", default="7,9", help="odd window side lengths, e.g. 7,9")
    ap.add_argument("--joint-very-fast-us", type=float, default=2000.0)
    ap.add_argument("--joint-fast-us", type=float, default=8000.0)

    args = ap.parse_args()

    transition_pattern_stats(
        labeled_npy=str(args.labeled_npy),
        out_summary_csv=str(args.out_summary_csv),
        out_hist_csv=str(args.out_hist_csv),
        width=int(args.width),
        height=int(args.height),
        tick_ns=float(args.tick_ns),
        start_events=int(args.start_events),
        max_events=int(args.max_events),
        windows=_parse_int_list(str(args.windows)),
        joint_very_fast_us=float(args.joint_very_fast_us),
        joint_fast_us=float(args.joint_fast_us),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
