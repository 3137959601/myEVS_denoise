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
        raise SystemExit("n125_path_failure_stats_771 requires numba")


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
def _metric_dt_jitter(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    width: int,
    height: int,
    dt_cap_ticks: int,
    jitter_ticks: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (dt_hist5, [collision_cnt, jitter_cnt, valid_events])."""

    n = int(t.shape[0])
    w = int(width)
    h = int(height)
    npx = w * h

    last_ts = np.zeros((npx,), dtype=np.uint64)
    last_pol = np.zeros((npx,), dtype=np.int8)

    hist = np.zeros((5,), dtype=np.int64)
    collision_cnt = 0
    jitter_cnt = 0
    valid_events = 0

    for i in range(n):
        xi = int(x[i])
        yi = int(y[i])
        if xi < 0 or xi >= w or yi < 0 or yi >= h:
            continue

        ti = np.uint64(t[i])
        pi = 1 if int(p[i]) > 0 else -1

        valid_events += 1
        latest_older = np.uint64(0)
        has_equal = False

        y0 = yi - 1
        if y0 < 0:
            y0 = 0
        y1 = yi + 1
        if y1 >= h:
            y1 = h - 1

        x0 = xi - 1
        if x0 < 0:
            x0 = 0
        x1 = xi + 1
        if x1 >= w:
            x1 = w - 1

        for yy in range(y0, y1 + 1):
            base = yy * w
            for xx in range(x0, x1 + 1):
                if xx == xi and yy == yi:
                    continue
                idx = base + xx
                if int(last_pol[idx]) != pi:
                    continue
                ts = last_ts[idx]
                if ts == 0:
                    continue
                if ts == ti:
                    has_equal = True
                elif ts < ti and ts > latest_older:
                    latest_older = ts

        if has_equal:
            collision_cnt += 1

        if latest_older != 0:
            dt = int(ti - latest_older)
            if dt <= jitter_ticks:
                jitter_cnt += 1

            if dt <= dt_cap_ticks:
                if dt <= 100:
                    hist[0] += 1
                elif dt <= 1000:
                    hist[1] += 1
                elif dt <= 5000:
                    hist[2] += 1
                elif dt <= 10000:
                    hist[3] += 1
                else:
                    hist[4] += 1

        idx0 = yi * w + xi
        last_ts[idx0] = ti
        last_pol[idx0] = np.int8(pi)

    return hist, np.asarray([collision_cnt, jitter_cnt, valid_events], dtype=np.int64)


@numba.njit(cache=True)
def _metric_path_death(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    width: int,
    height: int,
    depth: int,
    tau_step_ticks: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (step_hist_0to4, reason_hist_ABC)."""

    n = int(t.shape[0])
    w = int(width)
    h = int(height)
    npx = w * h

    last_ts = np.zeros((npx,), dtype=np.uint64)
    last_pol = np.zeros((npx,), dtype=np.int8)

    step_hist = np.zeros((5,), dtype=np.int64)
    reason_hist = np.zeros((3,), dtype=np.int64)  # A,B,C

    dmax = int(depth)
    if dmax < 1:
        dmax = 1
    if dmax > 4:
        dmax = 4

    tau_step = int(tau_step_ticks)
    if tau_step <= 0:
        tau_step = 1

    for i in range(n):
        xi = int(x[i])
        yi = int(y[i])
        if xi < 0 or xi >= w or yi < 0 or yi >= h:
            continue

        ti = np.uint64(t[i])
        pi = 1 if int(p[i]) > 0 else -1

        curr_x = xi
        curr_y = yi
        curr_t = ti
        steps_ok = 0
        failed = False
        fail_reason = -1

        for _ in range(dmax):
            has_older = False
            has_equal = False
            latest_older = np.uint64(0)
            best_x = -1
            best_y = -1

            y0 = curr_y - 1
            if y0 < 0:
                y0 = 0
            y1 = curr_y + 1
            if y1 >= h:
                y1 = h - 1

            x0 = curr_x - 1
            if x0 < 0:
                x0 = 0
            x1 = curr_x + 1
            if x1 >= w:
                x1 = w - 1

            for yy in range(y0, y1 + 1):
                base = yy * w
                for xx in range(x0, x1 + 1):
                    if xx == curr_x and yy == curr_y:
                        continue
                    idx = base + xx
                    if int(last_pol[idx]) != pi:
                        continue
                    ts = last_ts[idx]
                    if ts == 0:
                        continue
                    if ts == curr_t:
                        has_equal = True
                    elif ts < curr_t:
                        has_older = True
                        if ts > latest_older:
                            latest_older = ts
                            best_x = xx
                            best_y = yy

            if not has_older:
                failed = True
                if has_equal:
                    fail_reason = 1  # B
                else:
                    fail_reason = 0  # A
                break

            dt = int(curr_t - latest_older)
            if dt > tau_step:
                failed = True
                fail_reason = 2  # C
                break

            curr_t = latest_older
            curr_x = best_x
            curr_y = best_y
            steps_ok += 1

        if steps_ok >= 4:
            step_hist[4] += 1
        else:
            step_hist[steps_ok] += 1

        if failed and fail_reason >= 0:
            reason_hist[fail_reason] += 1

        idx0 = yi * w + xi
        last_ts[idx0] = ti
        last_pol[idx0] = np.int8(pi)

    return step_hist, reason_hist


def _fmt_ratio(cnt: int, den: int) -> float:
    if den <= 0:
        return float("nan")
    return float(cnt) / float(den)


def run_one_env(
    *,
    env_name: str,
    labeled_npy: str,
    width: int,
    height: int,
    tick_ns: float,
    max_events: int,
    depth: int,
    tau_step_us: int,
    dt_cap_us: int,
    jitter_us: int,
) -> dict:
    ev = load_labeled_npy(str(labeled_npy), max_events=int(max_events))

    tick_us = float(tick_ns) / 1000.0
    if tick_us <= 0:
        tick_us = 1.0

    dt_cap_ticks = int(round(float(dt_cap_us) / tick_us))
    jitter_ticks = int(round(float(jitter_us) / tick_us))
    tau_step_ticks = int(round(float(tau_step_us) / tick_us))

    dt_hist, jitter_stats = _metric_dt_jitter(
        ev.t,
        ev.x,
        ev.y,
        ev.p,
        int(width),
        int(height),
        int(dt_cap_ticks),
        int(jitter_ticks),
    )

    step_hist, reason_hist = _metric_path_death(
        ev.t,
        ev.x,
        ev.y,
        ev.p,
        int(width),
        int(height),
        int(depth),
        int(tau_step_ticks),
    )

    total_events = int(jitter_stats[2])
    hist_total = int(np.sum(dt_hist))
    fail_total = int(np.sum(step_hist[:4]))

    return {
        "env": str(env_name),
        "input": str(labeled_npy),
        "events": total_events,
        "dt_hist_edges_us": [
            "[1,100]",
            "(100,1000]",
            "(1000,5000]",
            "(5000,10000]",
            "(10000,50000]",
        ],
        "dt_hist_count": [int(v) for v in dt_hist.tolist()],
        "dt_hist_ratio": [_fmt_ratio(int(v), hist_total) for v in dt_hist.tolist()],
        "collision_count": int(jitter_stats[0]),
        "collision_rate": _fmt_ratio(int(jitter_stats[0]), total_events),
        "micro_jitter_count": int(jitter_stats[1]),
        "micro_jitter_rate": _fmt_ratio(int(jitter_stats[1]), total_events),
        "path_depth_hist_count": {
            "0": int(step_hist[0]),
            "1": int(step_hist[1]),
            "2": int(step_hist[2]),
            "3": int(step_hist[3]),
            "4": int(step_hist[4]),
        },
        "path_depth_hist_ratio": {
            "0": _fmt_ratio(int(step_hist[0]), total_events),
            "1": _fmt_ratio(int(step_hist[1]), total_events),
            "2": _fmt_ratio(int(step_hist[2]), total_events),
            "3": _fmt_ratio(int(step_hist[3]), total_events),
            "4": _fmt_ratio(int(step_hist[4]), total_events),
        },
        "death_reason_count": {
            "A_no_older_neighbor": int(reason_hist[0]),
            "B_equal_time_blocked": int(reason_hist[1]),
            "C_time_gap_too_large": int(reason_hist[2]),
        },
        "death_reason_ratio_among_failed": {
            "A_no_older_neighbor": _fmt_ratio(int(reason_hist[0]), fail_total),
            "B_equal_time_blocked": _fmt_ratio(int(reason_hist[1]), fail_total),
            "C_time_gap_too_large": _fmt_ratio(int(reason_hist[2]), fail_total),
        },
    }


def write_csv_summary(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields = [
        "env",
        "events",
        "collision_rate",
        "micro_jitter_rate",
        "depth0_ratio",
        "depth1_ratio",
        "depth2_ratio",
        "depth3_ratio",
        "depth4_ratio",
        "fail_A_ratio",
        "fail_B_ratio",
        "fail_C_ratio",
        "dt_1_100_ratio",
        "dt_100_1000_ratio",
        "dt_1000_5000_ratio",
        "dt_5000_10000_ratio",
        "dt_10000_50000_ratio",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "env": r["env"],
                    "events": r["events"],
                    "collision_rate": r["collision_rate"],
                    "micro_jitter_rate": r["micro_jitter_rate"],
                    "depth0_ratio": r["path_depth_hist_ratio"]["0"],
                    "depth1_ratio": r["path_depth_hist_ratio"]["1"],
                    "depth2_ratio": r["path_depth_hist_ratio"]["2"],
                    "depth3_ratio": r["path_depth_hist_ratio"]["3"],
                    "depth4_ratio": r["path_depth_hist_ratio"]["4"],
                    "fail_A_ratio": r["death_reason_ratio_among_failed"]["A_no_older_neighbor"],
                    "fail_B_ratio": r["death_reason_ratio_among_failed"]["B_equal_time_blocked"],
                    "fail_C_ratio": r["death_reason_ratio_among_failed"]["C_time_gap_too_large"],
                    "dt_1_100_ratio": r["dt_hist_ratio"][0],
                    "dt_100_1000_ratio": r["dt_hist_ratio"][1],
                    "dt_1000_5000_ratio": r["dt_hist_ratio"][2],
                    "dt_5000_10000_ratio": r["dt_hist_ratio"][3],
                    "dt_10000_50000_ratio": r["dt_hist_ratio"][4],
                }
            )


def main() -> int:
    _require_numba()

    ap = argparse.ArgumentParser(description="7.71 path-failure diagnostics: dt/jitter/path-death stats.")
    ap.add_argument("--env-list", default="light,heavy")
    ap.add_argument("--light", default=r"D:\hjx_workspace\scientific_reserach\dataset\ED24\myPedestrain_06\Pedestrain_06_1.8.npy")
    ap.add_argument("--mid", default=r"D:\hjx_workspace\scientific_reserach\dataset\ED24\myPedestrain_06\Pedestrain_06_2.5.npy")
    ap.add_argument("--heavy", default=r"D:\hjx_workspace\scientific_reserach\dataset\ED24\myPedestrain_06\Pedestrain_06_3.3.npy")

    ap.add_argument("--width", type=int, default=346)
    ap.add_argument("--height", type=int, default=260)
    ap.add_argument("--tick-ns", type=float, default=1000.0)
    ap.add_argument("--max-events", type=int, default=500000)

    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--tau-step-us", type=int, default=10000)
    ap.add_argument("--dt-cap-us", type=int, default=50000)
    ap.add_argument("--jitter-us", type=int, default=10)

    ap.add_argument(
        "--out-dir",
        default="data/ED24/myPedestrain_06/EBF_Part2/_n125_path_failure_stats_771",
    )

    args = ap.parse_args()

    env_list = _parse_env_list(args.env_list)
    path_map = {
        "light": str(args.light),
        "mid": str(args.mid),
        "heavy": str(args.heavy),
    }

    out_dir = os.path.abspath(str(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    results: list[dict] = []
    for env_name in env_list:
        in_path = path_map[env_name]
        if not os.path.exists(in_path):
            raise SystemExit(f"missing input file for {env_name}: {in_path}")

        print(f"[RUN] {env_name}  input={in_path}")
        rec = run_one_env(
            env_name=env_name,
            labeled_npy=in_path,
            width=int(args.width),
            height=int(args.height),
            tick_ns=float(args.tick_ns),
            max_events=int(args.max_events),
            depth=int(args.depth),
            tau_step_us=int(args.tau_step_us),
            dt_cap_us=int(args.dt_cap_us),
            jitter_us=int(args.jitter_us),
        )
        results.append(rec)
        print(
            f"[OK] {env_name} events={rec['events']} collision={rec['collision_rate']:.4f} "
            f"jitter={rec['micro_jitter_rate']:.4f} depth4={rec['path_depth_hist_ratio']['4']:.4f}"
        )

    out_json = os.path.join(out_dir, "summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    out_csv = os.path.join(out_dir, "summary.csv")
    write_csv_summary(out_csv, results)

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
