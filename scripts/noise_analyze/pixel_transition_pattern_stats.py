from __future__ import annotations

import argparse
import csv
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


HIST_BINS_US = np.asarray(
    [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000, 1000000],
    dtype=np.float64,
)


def _require_numba() -> None:
    if numba is None:
        raise SystemExit("pixel_transition_pattern_stats requires numba")


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


@numba.njit(cache=False)
def _bin_index(dt_us: float, bins_us: np.ndarray) -> int:
    lo = 0
    hi = int(bins_us.shape[0]) - 1
    if dt_us < bins_us[0] or dt_us >= bins_us[-1]:
        return -1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if dt_us < bins_us[mid]:
            hi = mid
        else:
            lo = mid
    return lo


@numba.njit(cache=False)
def _update_best2(cand: int, t: np.ndarray, best1: int, best2: int) -> tuple[int, int]:
    if cand < 0:
        return best1, best2
    if best1 < 0:
        return cand, best2
    if cand == best1:
        return best1, best2
    if t[cand] > t[best1]:
        return cand, best1
    if best2 < 0:
        return best1, cand
    if cand == best2:
        return best1, best2
    if t[cand] > t[best2]:
        return best1, cand
    return best1, best2


@numba.njit(cache=False)
def _compute_stats(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    label: np.ndarray,
    width: int,
    height: int,
    windows: np.ndarray,
    bins_us: np.ndarray,
    tick_us: float,
) -> tuple[np.ndarray, ...]:
    n = int(t.shape[0])
    npx = int(width) * int(height)
    nw = int(windows.shape[0])
    ncls = 2
    nsteps = 2
    nbins = int(bins_us.shape[0]) - 1

    pixel_hits = np.zeros((ncls, nsteps), dtype=np.int64)
    pixel_same = np.zeros((ncls, nsteps), dtype=np.int64)
    pixel_opp = np.zeros((ncls, nsteps), dtype=np.int64)
    pixel_dt_sum = np.zeros((ncls, nsteps), dtype=np.float64)
    pixel_hist = np.zeros((ncls, nsteps, nbins), dtype=np.int64)

    nb_hits = np.zeros((ncls, nw, nsteps), dtype=np.int64)
    nb_same = np.zeros((ncls, nw, nsteps), dtype=np.int64)
    nb_opp = np.zeros((ncls, nw, nsteps), dtype=np.int64)
    nb_dt_sum = np.zeros((ncls, nw, nsteps), dtype=np.float64)
    nb_hist = np.zeros((ncls, nw, nsteps, nbins), dtype=np.int64)
    nb_joint_hits = np.zeros((ncls, nw), dtype=np.int64)
    nb_joint_mode = np.zeros((ncls, nw, 4), dtype=np.int64)

    prev1 = np.full((npx,), -1, dtype=np.int32)
    prev2 = np.full((npx,), -1, dtype=np.int32)

    for i in range(n):
        xi = int(x[i])
        yi = int(y[i])
        if xi < 0 or xi >= width or yi < 0 or yi >= height:
            continue

        cls = int(label[i])
        if cls < 0 or cls >= ncls:
            continue

        idx = yi * width + xi
        j1 = int(prev1[idx])
        j2 = int(prev2[idx])

        if j1 >= 0:
            dt_us = float(t[i] - t[j1]) * tick_us
            pixel_hits[cls, 0] += 1
            pixel_dt_sum[cls, 0] += dt_us
            if int(p[j1]) == int(p[i]):
                pixel_same[cls, 0] += 1
            else:
                pixel_opp[cls, 0] += 1
            bi = _bin_index(dt_us, bins_us)
            if bi >= 0:
                pixel_hist[cls, 0, bi] += 1

        if j2 >= 0:
            dt_us = float(t[i] - t[j2]) * tick_us
            pixel_hits[cls, 1] += 1
            pixel_dt_sum[cls, 1] += dt_us
            if int(p[j2]) == int(p[i]):
                pixel_same[cls, 1] += 1
            else:
                pixel_opp[cls, 1] += 1
            bi = _bin_index(dt_us, bins_us)
            if bi >= 0:
                pixel_hist[cls, 1, bi] += 1

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

            for yy in range(y0, y1 + 1):
                base = yy * width
                for xx in range(x0, x1 + 1):
                    jdx = base + xx
                    a = int(prev1[jdx])
                    b = int(prev2[jdx])

                    if a >= 0:
                        dt_us = float(t[i] - t[a]) * tick_us
                        nb_hits[cls, wi, 0] += 1
                        nb_dt_sum[cls, wi, 0] += dt_us
                        if int(p[a]) == int(p[i]):
                            nb_same[cls, wi, 0] += 1
                        else:
                            nb_opp[cls, wi, 0] += 1
                        bi = _bin_index(dt_us, bins_us)
                        if bi >= 0:
                            nb_hist[cls, wi, 0, bi] += 1

                    if b >= 0:
                        dt_us = float(t[i] - t[b]) * tick_us
                        nb_hits[cls, wi, 1] += 1
                        nb_dt_sum[cls, wi, 1] += dt_us
                        if int(p[b]) == int(p[i]):
                            nb_same[cls, wi, 1] += 1
                        else:
                            nb_opp[cls, wi, 1] += 1
                        bi = _bin_index(dt_us, bins_us)
                        if bi >= 0:
                            nb_hist[cls, wi, 1, bi] += 1

                    if a >= 0 and b >= 0:
                        s2 = 1 if int(p[b]) == int(p[i]) else 0
                        s1 = 1 if int(p[a]) == int(p[i]) else 0
                        mode = 0
                        if s2 == 1 and s1 == 1:
                            mode = 0
                        elif s2 == 1 and s1 == 0:
                            mode = 1
                        elif s2 == 0 and s1 == 1:
                            mode = 2
                        else:
                            mode = 3
                        nb_joint_hits[cls, wi] += 1
                        nb_joint_mode[cls, wi, mode] += 1

        prev2[idx] = prev1[idx]
        prev1[idx] = i

    return (
        pixel_hits,
        pixel_same,
        pixel_opp,
        pixel_dt_sum,
        pixel_hist,
        nb_hits,
        nb_same,
        nb_opp,
        nb_dt_sum,
        nb_hist,
        nb_joint_hits,
        nb_joint_mode,
    )


def _write_summary_csv(
    out_csv: str,
    *,
    pixel_hits: np.ndarray,
    pixel_same: np.ndarray,
    pixel_opp: np.ndarray,
    pixel_dt_sum: np.ndarray,
    nb_hits: np.ndarray,
    nb_same: np.ndarray,
    nb_opp: np.ndarray,
    nb_dt_sum: np.ndarray,
    nb_joint_hits: np.ndarray,
    nb_joint_mode: np.ndarray,
    windows: list[int],
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    rows: list[dict[str, object]] = []
    cls_defs = [(0, "noise"), (1, "signal")]

    for cls_v, cls_name in cls_defs:
        events = int(pixel_hits[cls_v, 0] + pixel_hits[cls_v, 1])
        for step_i, step in enumerate(("pre1", "pre2")):
            hits = int(pixel_hits[cls_v, step_i])
            rows.append(
                {
                    "class": cls_name,
                    "scope": "pixel",
                    "step": step,
                    "window": "",
                    "events": events,
                    "hits": hits,
                    "hit_rate": _safe_div(hits, events),
                    "same_pol_rate": _safe_div(int(pixel_same[cls_v, step_i]), hits),
                    "opp_pol_rate": _safe_div(int(pixel_opp[cls_v, step_i]), hits),
                    "dt_mean_us": _safe_div(float(pixel_dt_sum[cls_v, step_i]), hits),
                    "joint_pattern": "",
                    "pattern_count": "",
                    "pattern_ratio": "",
                }
            )

        for wi, win in enumerate(windows):
            for step_i, step in enumerate(("pre1", "pre2")):
                hits = int(nb_hits[cls_v, wi, step_i])
                rows.append(
                    {
                        "class": cls_name,
                        "scope": "neighborhood_pixel",
                        "step": step,
                        "window": int(win),
                        "events": events,
                        "hits": hits,
                        "hit_rate": _safe_div(hits, events),
                        "same_pol_rate": _safe_div(int(nb_same[cls_v, wi, step_i]), hits),
                        "opp_pol_rate": _safe_div(int(nb_opp[cls_v, wi, step_i]), hits),
                        "dt_mean_us": _safe_div(float(nb_dt_sum[cls_v, wi, step_i]), hits),
                        "joint_pattern": "",
                        "pattern_count": "",
                        "pattern_ratio": "",
                    }
                )

            joint_hits = int(nb_joint_hits[cls_v, wi])
            for mi, name in enumerate(("same->same", "same->opp", "opp->same", "opp->opp")):
                cnt = int(nb_joint_mode[cls_v, wi, mi])
                rows.append(
                    {
                        "class": cls_name,
                        "scope": "neighborhood_pixel",
                        "step": "joint_prev2_prev1",
                        "window": int(win),
                        "events": events,
                        "hits": joint_hits,
                        "hit_rate": _safe_div(joint_hits, events),
                        "same_pol_rate": "",
                        "opp_pol_rate": "",
                        "dt_mean_us": "",
                        "joint_pattern": name,
                        "pattern_count": cnt,
                        "pattern_ratio": _safe_div(cnt, joint_hits),
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
                "joint_pattern",
                "pattern_count",
                "pattern_ratio",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_hist_csv(
    out_csv: str,
    *,
    pixel_hist: np.ndarray,
    nb_hist: np.ndarray,
    windows: list[int],
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    rows: list[dict[str, object]] = []
    cls_defs = [(0, "noise"), (1, "signal")]
    items: list[tuple[str, str, str, np.ndarray]] = [
        ("pixel", "", "pre1", pixel_hist[:, 0, :]),
        ("pixel", "", "pre2", pixel_hist[:, 1, :]),
    ]
    for wi, win in enumerate(windows):
        items.append(("neighborhood_pixel", str(int(win)), "pre1", nb_hist[:, wi, 0, :]))
        items.append(("neighborhood_pixel", str(int(win)), "pre2", nb_hist[:, wi, 1, :]))

    for cls_v, cls_name in cls_defs:
        for scope, win, step, hist_arr in items:
            counts = hist_arr[cls_v]
            total = float(np.sum(counts))
            for bi in range(counts.shape[0]):
                rows.append(
                    {
                        "class": cls_name,
                        "scope": scope,
                        "window": win,
                        "step": step,
                        "bin_lo_us": float(HIST_BINS_US[bi]),
                        "bin_hi_us": float(HIST_BINS_US[bi + 1]),
                        "count": int(counts[bi]),
                        "ratio": _safe_div(int(counts[bi]), total),
                    }
                )

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["class", "scope", "window", "step", "bin_lo_us", "bin_hi_us", "count", "ratio"])
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
) -> None:
    _require_numba()
    ev = load_labeled_npy(labeled_npy, start_events=int(start_events), max_events=int(max_events))

    windows = [int(w) for w in windows if int(w) > 0 and int(w) % 2 == 1]
    if not windows:
        raise SystemExit("--windows must contain odd positive ints, e.g. 7,9")

    (
        pixel_hits,
        pixel_same,
        pixel_opp,
        pixel_dt_sum,
        pixel_hist,
        nb_hits,
        nb_same,
        nb_opp,
        nb_dt_sum,
        nb_hist,
        nb_joint_hits,
        nb_joint_mode,
    ) = _compute_stats(
        ev.t,
        ev.x,
        ev.y,
        ev.p,
        ev.label,
        int(width),
        int(height),
        np.asarray(windows, dtype=np.int32),
        HIST_BINS_US,
        float(tick_ns) / 1000.0,
    )

    _write_summary_csv(
        out_summary_csv,
        pixel_hits=pixel_hits,
        pixel_same=pixel_same,
        pixel_opp=pixel_opp,
        pixel_dt_sum=pixel_dt_sum,
        nb_hits=nb_hits,
        nb_same=nb_same,
        nb_opp=nb_opp,
        nb_dt_sum=nb_dt_sum,
        nb_joint_hits=nb_joint_hits,
        nb_joint_mode=nb_joint_mode,
        windows=windows,
    )

    _write_hist_csv(
        out_hist_csv,
        pixel_hist=pixel_hist,
        nb_hist=nb_hist,
        windows=windows,
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Compute previous-event transition statistics for same-pixel and per-neighborhood-pixel events. "
            "This version uses each neighbor pixel's own prev1/prev2 history, not the whole-neighborhood top2."
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())