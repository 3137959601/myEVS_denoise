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


def _require_numba() -> None:
    if numba is None:
        raise SystemExit("block_transition_stats requires numba")


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


@numba.njit(cache=True)
def _update_top3(cand: int, t: np.ndarray, top_idx: np.ndarray) -> None:
    if cand < 0:
        return
    if top_idx[0] == cand or top_idx[1] == cand or top_idx[2] == cand:
        return

    tc = t[cand]
    if top_idx[0] < 0:
        top_idx[0] = cand
        return
    if tc > t[top_idx[0]]:
        top_idx[2] = top_idx[1]
        top_idx[1] = top_idx[0]
        top_idx[0] = cand
        return

    if top_idx[1] < 0:
        top_idx[1] = cand
        return
    if tc > t[top_idx[1]]:
        top_idx[2] = top_idx[1]
        top_idx[1] = cand
        return

    if top_idx[2] < 0 or tc > t[top_idx[2]]:
        top_idx[2] = cand


@numba.njit(cache=True)
def _block_id(part: int, dx: int, dy: int) -> int:
    # part 0: UDLR (overlap), part 1: quadrants (non-overlap)
    if part == 0:
        ax = abs(dx)
        ay = abs(dy)
        if ay >= ax:
            if dy < 0:
                return 0  # up
            if dy > 0:
                return 1  # down
            return -1
        if dx < 0:
            return 2  # left
        if dx > 0:
            return 3  # right
        return -1
    if dx < 0 and dy < 0:
        return 0  # ul
    if dx > 0 and dy < 0:
        return 1  # ur
    if dx < 0 and dy > 0:
        return 2  # ll
    if dx > 0 and dy > 0:
        return 3  # lr
    return -1


@numba.njit(cache=True)
def _compute_block_transition_stats(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    label: np.ndarray,
    width: int,
    height: int,
    windows: np.ndarray,
    tick_ns: float,
) -> tuple[np.ndarray, ...]:
    n = int(t.shape[0])
    npx = int(width) * int(height)
    nw = int(windows.shape[0])

    # seg: 0=all,1=seg0,2=seg1
    seg_cnt = np.zeros((3, 2), dtype=np.int64)

    block_nonempty = np.zeros((3, 2, nw, 2, 4), dtype=np.int64)
    block_cnt_sum = np.zeros((3, 2, nw, 2, 4), dtype=np.int64)
    block_same_sum = np.zeros((3, 2, nw, 2, 4), dtype=np.int64)
    block_opp_sum = np.zeros((3, 2, nw, 2, 4), dtype=np.int64)

    top_hit = np.zeros((3, 2, nw, 2, 4, 3), dtype=np.int64)
    top_same = np.zeros((3, 2, nw, 2, 4, 3), dtype=np.int64)
    top_signal = np.zeros((3, 2, nw, 2, 4, 3), dtype=np.int64)
    top_dt_sum = np.zeros((3, 2, nw, 2, 4, 3), dtype=np.float64)

    joint_mode = np.zeros((3, 2, nw, 2, 4, 4), dtype=np.int64)

    nonempty_blocks_sum = np.zeros((3, 2, nw, 2), dtype=np.int64)
    samepol_blocks_sum = np.zeros((3, 2, nw, 2), dtype=np.int64)

    prev1_any = np.full((npx,), -1, dtype=np.int32)
    prev2_any = np.full((npx,), -1, dtype=np.int32)

    n_half = n // 2
    tick_to_us = tick_ns / 1000.0

    for i in range(n):
        xi = int(x[i])
        yi = int(y[i])
        if xi < 0 or xi >= width or yi < 0 or yi >= height:
            continue

        cls = 1 if int(label[i]) > 0 else 0
        pi = 1 if int(p[i]) > 0 else -1
        ti = t[i]

        seg_idx = 1 if i < n_half else 2

        seg_cnt[0, cls] += 1
        seg_cnt[seg_idx, cls] += 1

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

            for part in range(2):
                cnt = np.zeros((4,), dtype=np.int64)
                same_cnt = np.zeros((4,), dtype=np.int64)
                opp_cnt = np.zeros((4,), dtype=np.int64)

                top_idx = np.full((4, 3), -1, dtype=np.int32)

                for yy in range(y0, y1 + 1):
                    base = yy * width
                    dy = yy - yi
                    for xx in range(x0, x1 + 1):
                        dx = xx - xi
                        if dx == 0 and dy == 0:
                            continue

                        bid = _block_id(part, dx, dy)
                        if bid < 0:
                            continue

                        pix = base + xx
                        a = int(prev1_any[pix])
                        b = int(prev2_any[pix])

                        if a >= 0:
                            cnt[bid] += 1
                            pa = 1 if int(p[a]) > 0 else -1
                            if pa == pi:
                                same_cnt[bid] += 1
                            else:
                                opp_cnt[bid] += 1
                            _update_top3(a, t, top_idx[bid])

                        if b >= 0:
                            cnt[bid] += 1
                            pb = 1 if int(p[b]) > 0 else -1
                            if pb == pi:
                                same_cnt[bid] += 1
                            else:
                                opp_cnt[bid] += 1
                            _update_top3(b, t, top_idx[bid])

                nonempty_blocks = 0
                samepol_blocks = 0
                for b in range(4):
                    if cnt[b] > 0:
                        nonempty_blocks += 1
                    if same_cnt[b] > 0:
                        samepol_blocks += 1

                for s in (0, seg_idx):
                    nonempty_blocks_sum[s, cls, wi, part] += nonempty_blocks
                    samepol_blocks_sum[s, cls, wi, part] += samepol_blocks

                    for b in range(4):
                        block_cnt_sum[s, cls, wi, part, b] += cnt[b]
                        block_same_sum[s, cls, wi, part, b] += same_cnt[b]
                        block_opp_sum[s, cls, wi, part, b] += opp_cnt[b]
                        if cnt[b] > 0:
                            block_nonempty[s, cls, wi, part, b] += 1

                        idx1 = int(top_idx[b, 0])
                        idx2 = int(top_idx[b, 1])
                        idx3 = int(top_idx[b, 2])

                        if idx1 >= 0:
                            top_hit[s, cls, wi, part, b, 0] += 1
                            p1 = 1 if int(p[idx1]) > 0 else -1
                            if p1 == pi:
                                top_same[s, cls, wi, part, b, 0] += 1
                            if int(label[idx1]) > 0:
                                top_signal[s, cls, wi, part, b, 0] += 1
                            dt1 = float(ti - t[idx1]) * tick_to_us
                            top_dt_sum[s, cls, wi, part, b, 0] += dt1

                        if idx2 >= 0:
                            top_hit[s, cls, wi, part, b, 1] += 1
                            p2 = 1 if int(p[idx2]) > 0 else -1
                            if p2 == pi:
                                top_same[s, cls, wi, part, b, 1] += 1
                            if int(label[idx2]) > 0:
                                top_signal[s, cls, wi, part, b, 1] += 1
                            dt2 = float(ti - t[idx2]) * tick_to_us
                            top_dt_sum[s, cls, wi, part, b, 1] += dt2

                        if idx3 >= 0:
                            top_hit[s, cls, wi, part, b, 2] += 1
                            p3 = 1 if int(p[idx3]) > 0 else -1
                            if p3 == pi:
                                top_same[s, cls, wi, part, b, 2] += 1
                            if int(label[idx3]) > 0:
                                top_signal[s, cls, wi, part, b, 2] += 1
                            dt3 = float(ti - t[idx3]) * tick_to_us
                            top_dt_sum[s, cls, wi, part, b, 2] += dt3

                        if idx1 >= 0 and idx2 >= 0:
                            s2 = 1 if (1 if int(p[idx2]) > 0 else -1) == pi else 0
                            s1 = 1 if (1 if int(p[idx1]) > 0 else -1) == pi else 0
                            mode = 0
                            if s2 == 1 and s1 == 1:
                                mode = 0
                            elif s2 == 1 and s1 == 0:
                                mode = 1
                            elif s2 == 0 and s1 == 1:
                                mode = 2
                            else:
                                mode = 3
                            joint_mode[s, cls, wi, part, b, mode] += 1

        idx0 = yi * width + xi
        prev2_any[idx0] = prev1_any[idx0]
        prev1_any[idx0] = i

    return (
        seg_cnt,
        block_nonempty,
        block_cnt_sum,
        block_same_sum,
        block_opp_sum,
        top_hit,
        top_same,
        top_signal,
        top_dt_sum,
        joint_mode,
        nonempty_blocks_sum,
        samepol_blocks_sum,
    )


def _safe_div(a: float, b: float) -> float:
    if b == 0:
        return float("nan")
    return float(a) / float(b)


def _write_summary_csv(
    out_csv: str,
    *,
    env: str,
    windows: list[int],
    seg_cnt: np.ndarray,
    block_nonempty: np.ndarray,
    block_cnt_sum: np.ndarray,
    block_same_sum: np.ndarray,
    block_opp_sum: np.ndarray,
    top_hit: np.ndarray,
    top_same: np.ndarray,
    top_signal: np.ndarray,
    top_dt_sum: np.ndarray,
    nonempty_blocks_sum: np.ndarray,
    samepol_blocks_sum: np.ndarray,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)

    seg_names = ["all", "seg0", "seg1"]
    cls_names = ["noise", "signal"]
    part_names = ["udlr", "quadrant"]
    block_names = [
        ["up", "down", "left", "right"],
        ["ul", "ur", "ll", "lr"],
    ]

    fields = [
        "env",
        "segment",
        "class",
        "window",
        "partition",
        "block",
        "center_events",
        "block_nonempty_rate",
        "block_count_mean",
        "block_samepol_rate",
        "block_opppol_rate",
        "top1_hit_rate",
        "top1_samepol_rate",
        "top1_signal_rate",
        "top1_dt_mean_us",
        "top2_hit_rate",
        "top2_samepol_rate",
        "top2_signal_rate",
        "top2_dt_mean_us",
        "top3_hit_rate",
        "top3_samepol_rate",
        "top3_signal_rate",
        "top3_dt_mean_us",
        "nonempty_blocks_mean",
        "samepol_blocks_mean",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for s in range(3):
            for c in range(2):
                ce = int(seg_cnt[s, c])
                if ce <= 0:
                    continue
                for wi, win in enumerate(windows):
                    for part in range(2):
                        nbm = _safe_div(float(nonempty_blocks_sum[s, c, wi, part]), float(ce))
                        sbm = _safe_div(float(samepol_blocks_sum[s, c, wi, part]), float(ce))

                        for b in range(4):
                            row = {
                                "env": env,
                                "segment": seg_names[s],
                                "class": cls_names[c],
                                "window": int(win),
                                "partition": part_names[part],
                                "block": block_names[part][b],
                                "center_events": ce,
                                "block_nonempty_rate": _safe_div(float(block_nonempty[s, c, wi, part, b]), float(ce)),
                                "block_count_mean": _safe_div(float(block_cnt_sum[s, c, wi, part, b]), float(ce)),
                                "block_samepol_rate": _safe_div(
                                    float(block_same_sum[s, c, wi, part, b]),
                                    float(block_cnt_sum[s, c, wi, part, b]),
                                ),
                                "block_opppol_rate": _safe_div(
                                    float(block_opp_sum[s, c, wi, part, b]),
                                    float(block_cnt_sum[s, c, wi, part, b]),
                                ),
                                "nonempty_blocks_mean": nbm,
                                "samepol_blocks_mean": sbm,
                            }

                            for k in range(3):
                                hk = float(top_hit[s, c, wi, part, b, k])
                                row[f"top{k+1}_hit_rate"] = _safe_div(hk, float(ce))
                                row[f"top{k+1}_samepol_rate"] = _safe_div(float(top_same[s, c, wi, part, b, k]), hk)
                                row[f"top{k+1}_signal_rate"] = _safe_div(float(top_signal[s, c, wi, part, b, k]), hk)
                                row[f"top{k+1}_dt_mean_us"] = _safe_div(float(top_dt_sum[s, c, wi, part, b, k]), hk)

                            w.writerow(row)


def _write_joint_csv(
    out_csv: str,
    *,
    env: str,
    windows: list[int],
    seg_cnt: np.ndarray,
    joint_mode: np.ndarray,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)

    seg_names = ["all", "seg0", "seg1"]
    cls_names = ["noise", "signal"]
    part_names = ["udlr", "quadrant"]
    block_names = [
        ["up", "down", "left", "right"],
        ["ul", "ur", "ll", "lr"],
    ]
    mode_names = ["same_to_same", "same_to_opp", "opp_to_same", "opp_to_opp"]

    fields = [
        "env",
        "segment",
        "class",
        "window",
        "partition",
        "block",
        "joint_hits",
        "same_to_same_rate",
        "same_to_opp_rate",
        "opp_to_same_rate",
        "opp_to_opp_rate",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for s in range(3):
            for c in range(2):
                ce = int(seg_cnt[s, c])
                if ce <= 0:
                    continue
                for wi, win in enumerate(windows):
                    for part in range(2):
                        for b in range(4):
                            hits = float(np.sum(joint_mode[s, c, wi, part, b, :]))
                            row = {
                                "env": env,
                                "segment": seg_names[s],
                                "class": cls_names[c],
                                "window": int(win),
                                "partition": part_names[part],
                                "block": block_names[part][b],
                                "joint_hits": int(hits),
                            }
                            for mi, mn in enumerate(mode_names):
                                row[f"{mn}_rate"] = _safe_div(float(joint_mode[s, c, wi, part, b, mi]), hits)
                            w.writerow(row)


def block_transition_stats(
    *,
    labeled_npy: str,
    out_summary_csv: str,
    out_joint_csv: str,
    env: str,
    width: int,
    height: int,
    tick_ns: float,
    start_events: int,
    max_events: int,
    windows: list[int],
) -> None:
    _require_numba()

    windows = [int(w) for w in windows if int(w) > 0 and int(w) % 2 == 1]
    if not windows:
        raise SystemExit("--windows must contain odd positive ints, e.g. 7,9")

    ev = load_labeled_npy(labeled_npy, start_events=int(start_events), max_events=int(max_events))

    (
        seg_cnt,
        block_nonempty,
        block_cnt_sum,
        block_same_sum,
        block_opp_sum,
        top_hit,
        top_same,
        top_signal,
        top_dt_sum,
        joint_mode,
        nonempty_blocks_sum,
        samepol_blocks_sum,
    ) = _compute_block_transition_stats(
        ev.t,
        ev.x,
        ev.y,
        ev.p,
        ev.label,
        int(width),
        int(height),
        np.asarray(windows, dtype=np.int32),
        float(tick_ns),
    )

    _write_summary_csv(
        out_summary_csv,
        env=env,
        windows=windows,
        seg_cnt=seg_cnt,
        block_nonempty=block_nonempty,
        block_cnt_sum=block_cnt_sum,
        block_same_sum=block_same_sum,
        block_opp_sum=block_opp_sum,
        top_hit=top_hit,
        top_same=top_same,
        top_signal=top_signal,
        top_dt_sum=top_dt_sum,
        nonempty_blocks_sum=nonempty_blocks_sum,
        samepol_blocks_sum=samepol_blocks_sum,
    )

    _write_joint_csv(
        out_joint_csv,
        env=env,
        windows=windows,
        seg_cnt=seg_cnt,
        joint_mode=joint_mode,
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Compute block-wise transition statistics for labeled events. "
            "Includes UDLR/quadrants, top1/2/3 recent events, seg0/seg1 breakdown."
        )
    )
    ap.add_argument("--labeled-npy", required=True)
    ap.add_argument("--out-summary-csv", required=True)
    ap.add_argument("--out-joint-csv", required=True)
    ap.add_argument("--env", default="unknown")
    ap.add_argument("--width", type=int, default=346)
    ap.add_argument("--height", type=int, default=260)
    ap.add_argument("--tick-ns", type=float, default=1000.0)
    ap.add_argument("--start-events", type=int, default=0)
    ap.add_argument("--max-events", type=int, default=400000)
    ap.add_argument("--windows", default="7,9")

    args = ap.parse_args()

    block_transition_stats(
        labeled_npy=str(args.labeled_npy),
        out_summary_csv=str(args.out_summary_csv),
        out_joint_csv=str(args.out_joint_csv),
        env=str(args.env),
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
