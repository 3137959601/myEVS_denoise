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
        raise SystemExit("n125_path_failure_stats_772 requires numba")


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
def _kernel_772(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    width: int,
    height: int,
    tau_base_ticks: int,
    tau_step_ticks: int,
    density_r4_hist: np.ndarray,
    mix_hist: np.ndarray,
    dt_bin5: np.ndarray,
    jump_bin5: np.ndarray,
    vel_bin5: np.ndarray,
    angle_hist: np.ndarray,
    stats_cnt: np.ndarray,
    stats_sum: np.ndarray,
) -> None:
    """
    stats_cnt:
      [0]=valid_events
      [1]=latest_support_events
      [2]=angle_valid_events

    stats_sum:
      [0..3]=same_mean_acc(R1..R4)
      [4..7]=opp_mean_acc(R1..R4)
      [8]=mix_mean_acc(R4)
      [9]=jump_mean_acc
      [10]=dt_mean_ticks_acc (latest support)
      [11]=vel_mean_acc
      [12]=angle_var_mean_acc
    """

    n = int(t.shape[0])
    w = int(width)
    h = int(height)
    npx = w * h

    tau_base = int(tau_base_ticks)
    if tau_base <= 0:
        tau_base = 1
    tau_step = int(tau_step_ticks)
    if tau_step <= 0:
        tau_step = 1

    mix_bins = int(mix_hist.shape[0])
    angle_bins = int(angle_hist.shape[0])

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

        # ring counts by Chebyshev distance d in {1,2,3,4}
        same_ring = np.zeros((5,), dtype=np.int32)
        opp_ring = np.zeros((5,), dtype=np.int32)

        latest_ts = np.uint64(0)
        latest_d = 0

        # for angle coherence within tau_step and same-pol
        n_ang = 0
        sum_cos = 0.0
        sum_sin = 0.0

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
            dy = yy - yi
            ady = dy if dy >= 0 else -dy
            for xx in range(x0, x1 + 1):
                if xx == xi and yy == yi:
                    continue
                dx = xx - xi
                adx = dx if dx >= 0 else -dx
                d = adx if adx >= ady else ady
                if d < 1 or d > 4:
                    continue

                idx = base + xx
                ts = last_ts[idx]
                if ts == 0:
                    continue
                if ts >= ti:
                    continue

                dt = int(ti - ts)
                pol = int(last_pol[idx])

                if pol == pi:
                    if ts > latest_ts:
                        latest_ts = ts
                        latest_d = d

                    if dt <= tau_base:
                        same_ring[d] += 1

                    if dt <= tau_step:
                        norm = np.sqrt(float(dx * dx + dy * dy))
                        if norm > 0.0:
                            sum_cos += float(dx) / norm
                            sum_sin += float(dy) / norm
                            n_ang += 1
                elif dt <= tau_base:
                    opp_ring[d] += 1

        s1 = same_ring[1]
        s2 = s1 + same_ring[2]
        s3 = s2 + same_ring[3]
        s4 = s3 + same_ring[4]

        o1 = opp_ring[1]
        o2 = o1 + opp_ring[2]
        o3 = o2 + opp_ring[3]
        o4 = o3 + opp_ring[4]

        stats_sum[0] += float(s1)
        stats_sum[1] += float(s2)
        stats_sum[2] += float(s3)
        stats_sum[3] += float(s4)
        stats_sum[4] += float(o1)
        stats_sum[5] += float(o2)
        stats_sum[6] += float(o3)
        stats_sum[7] += float(o4)

        if s4 < 0:
            s4 = 0
        if s4 > 80:
            s4 = 80
        density_r4_hist[s4] += 1

        mix = float(o4) / float(s4 + o4 + 1)
        if mix < 0.0:
            mix = 0.0
        if mix > 1.0:
            mix = 1.0
        stats_sum[8] += mix
        ib = int(mix * float(mix_bins))
        if ib >= mix_bins:
            ib = mix_bins - 1
        if ib < 0:
            ib = 0
        mix_hist[ib] += 1

        if latest_ts != 0:
            stats_cnt[1] += 1
            dtl = int(ti - latest_ts)
            stats_sum[9] += float(latest_d)
            stats_sum[10] += float(dtl)

            if dtl <= 1000:
                dt_bin5[0] += 1
            elif dtl <= 5000:
                dt_bin5[1] += 1
            elif dtl <= 10000:
                dt_bin5[2] += 1
            elif dtl <= 30000:
                dt_bin5[3] += 1
            else:
                dt_bin5[4] += 1

            if latest_d < 1:
                latest_d = 1
            if latest_d > 4:
                latest_d = 4
            jump_bin5[latest_d] += 1

            if dtl > 0:
                v = float(latest_d) / (float(dtl) / 1000.0)
                stats_sum[11] += v
                if v < 0.1:
                    vel_bin5[0] += 1
                elif v < 0.5:
                    vel_bin5[1] += 1
                elif v < 1.0:
                    vel_bin5[2] += 1
                elif v < 5.0:
                    vel_bin5[3] += 1
                else:
                    vel_bin5[4] += 1

        if n_ang >= 3:
            stats_cnt[2] += 1
            c = sum_cos / float(n_ang)
            s = sum_sin / float(n_ang)
            r = np.sqrt(c * c + s * s)
            if r < 0.0:
                r = 0.0
            if r > 1.0:
                r = 1.0
            var_angle = 1.0 - r
            stats_sum[12] += var_angle

            ia = int(var_angle * float(angle_bins))
            if ia >= angle_bins:
                ia = angle_bins - 1
            if ia < 0:
                ia = 0
            angle_hist[ia] += 1

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
    tau_base_us: int,
    tau_step_us: int,
    mix_bins: int,
    angle_bins: int,
) -> dict:
    ev = load_labeled_npy(str(labeled_npy), max_events=int(max_events))

    tick_us = float(tick_ns) / 1000.0
    if tick_us <= 0:
        tick_us = 1.0
    tau_base_ticks = int(round(float(tau_base_us) / tick_us))
    tau_step_ticks = int(round(float(tau_step_us) / tick_us))

    density_r4_hist = np.zeros((81,), dtype=np.int64)
    mix_hist = np.zeros((int(mix_bins),), dtype=np.int64)
    dt_bin5 = np.zeros((5,), dtype=np.int64)
    jump_bin5 = np.zeros((5,), dtype=np.int64)
    vel_bin5 = np.zeros((5,), dtype=np.int64)
    angle_hist = np.zeros((int(angle_bins),), dtype=np.int64)

    stats_cnt = np.zeros((3,), dtype=np.int64)
    stats_sum = np.zeros((13,), dtype=np.float64)

    _kernel_772(
        ev.t,
        ev.x,
        ev.y,
        ev.p,
        int(width),
        int(height),
        int(tau_base_ticks),
        int(tau_step_ticks),
        density_r4_hist,
        mix_hist,
        dt_bin5,
        jump_bin5,
        vel_bin5,
        angle_hist,
        stats_cnt,
        stats_sum,
    )

    n_valid = int(stats_cnt[0])
    n_latest = int(stats_cnt[1])
    n_angle = int(stats_cnt[2])

    return {
        "env": str(env_name),
        "input": str(labeled_npy),
        "events": n_valid,
        "tau_base_us": int(tau_base_us),
        "tau_step_us": int(tau_step_us),
        "density_same_mean": {
            "R1": _safe_ratio(stats_sum[0], n_valid),
            "R2": _safe_ratio(stats_sum[1], n_valid),
            "R3": _safe_ratio(stats_sum[2], n_valid),
            "R4": _safe_ratio(stats_sum[3], n_valid),
        },
        "density_opp_mean": {
            "R1": _safe_ratio(stats_sum[4], n_valid),
            "R2": _safe_ratio(stats_sum[5], n_valid),
            "R3": _safe_ratio(stats_sum[6], n_valid),
            "R4": _safe_ratio(stats_sum[7], n_valid),
        },
        "mix_r4_mean": _safe_ratio(stats_sum[8], n_valid),
        "density_r4_same_hist": density_r4_hist.tolist(),
        "mix_hist_bins": int(mix_bins),
        "mix_hist": mix_hist.tolist(),
        "latest_support_events": n_latest,
        "latest_support_rate": _safe_ratio(n_latest, n_valid),
        "dt_bins": ["[0,1ms]", "(1,5ms]", "(5,10ms]", "(10,30ms]", ">30ms"],
        "dt_hist": dt_bin5.tolist(),
        "dt_hist_ratio": [_safe_ratio(int(v), n_latest) for v in dt_bin5.tolist()],
        "jump_bins": ["none", "R1", "R2", "R3", "R4"],
        "jump_hist": jump_bin5.tolist(),
        "jump_hist_ratio": [_safe_ratio(int(v), n_latest) for v in jump_bin5.tolist()],
        "jump_mean": _safe_ratio(stats_sum[9], n_latest),
        "dt_mean_us": _safe_ratio(stats_sum[10] * tick_us, n_latest),
        "velocity_bins": ["[0,0.1)", "[0.1,0.5)", "[0.5,1.0)", "[1.0,5.0)", ">=5.0"],
        "velocity_hist": vel_bin5.tolist(),
        "velocity_hist_ratio": [_safe_ratio(int(v), n_latest) for v in vel_bin5.tolist()],
        "velocity_mean": _safe_ratio(stats_sum[11], n_latest),
        "angle_valid_events": n_angle,
        "angle_valid_rate": _safe_ratio(n_angle, n_valid),
        "angle_var_mean": _safe_ratio(stats_sum[12], n_angle),
        "angle_hist_bins": int(angle_bins),
        "angle_hist": angle_hist.tolist(),
        "angle_hist_ratio": [_safe_ratio(int(v), n_angle) for v in angle_hist.tolist()],
    }


def write_summary_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields = [
        "env",
        "events",
        "same_r4_mean",
        "opp_r4_mean",
        "mix_r4_mean",
        "latest_support_rate",
        "jump_mean",
        "dt_mean_us",
        "velocity_mean",
        "angle_valid_rate",
        "angle_var_mean",
        "dt_gt30ms_ratio",
        "jump_r3_r4_ratio",
        "vel_ge1_ratio",
        "angle_low_var_ratio",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            dt_ratio = r["dt_hist_ratio"]
            jump_ratio = r["jump_hist_ratio"]
            vel_ratio = r["velocity_hist_ratio"]
            ang_ratio = r["angle_hist_ratio"]
            # angle_low_var: var < 0.3
            bins = int(r["angle_hist_bins"])
            cut = int(0.3 * float(bins))
            if cut < 0:
                cut = 0
            if cut > bins:
                cut = bins
            low_var_ratio = 0.0
            for i in range(cut):
                v = ang_ratio[i]
                if np.isfinite(v):
                    low_var_ratio += float(v)
            w.writerow(
                {
                    "env": r["env"],
                    "events": r["events"],
                    "same_r4_mean": r["density_same_mean"]["R4"],
                    "opp_r4_mean": r["density_opp_mean"]["R4"],
                    "mix_r4_mean": r["mix_r4_mean"],
                    "latest_support_rate": r["latest_support_rate"],
                    "jump_mean": r["jump_mean"],
                    "dt_mean_us": r["dt_mean_us"],
                    "velocity_mean": r["velocity_mean"],
                    "angle_valid_rate": r["angle_valid_rate"],
                    "angle_var_mean": r["angle_var_mean"],
                    "dt_gt30ms_ratio": dt_ratio[4],
                    "jump_r3_r4_ratio": (jump_ratio[3] if np.isfinite(jump_ratio[3]) else 0.0)
                    + (jump_ratio[4] if np.isfinite(jump_ratio[4]) else 0.0),
                    "vel_ge1_ratio": (vel_ratio[3] if np.isfinite(vel_ratio[3]) else 0.0)
                    + (vel_ratio[4] if np.isfinite(vel_ratio[4]) else 0.0),
                    "angle_low_var_ratio": low_var_ratio,
                }
            )


def main() -> int:
    _require_numba()

    ap = argparse.ArgumentParser(description="7.72 multi-scale diagnostics for path failure analysis.")
    ap.add_argument("--env-list", default="light,heavy")
    ap.add_argument("--light", default=r"D:\hjx_workspace\scientific_reserach\dataset\ED24\myPedestrain_06\Pedestrain_06_1.8.npy")
    ap.add_argument("--mid", default=r"D:\hjx_workspace\scientific_reserach\dataset\ED24\myPedestrain_06\Pedestrain_06_2.5.npy")
    ap.add_argument("--heavy", default=r"D:\hjx_workspace\scientific_reserach\dataset\ED24\myPedestrain_06\Pedestrain_06_3.3.npy")
    ap.add_argument("--width", type=int, default=346)
    ap.add_argument("--height", type=int, default=260)
    ap.add_argument("--tick-ns", type=float, default=1000.0)
    ap.add_argument("--max-events", type=int, default=500000)
    ap.add_argument("--tau-base-us", type=int, default=30000)
    ap.add_argument("--tau-step-us", type=int, default=10000)
    ap.add_argument("--mix-bins", type=int, default=20)
    ap.add_argument("--angle-bins", type=int, default=20)
    ap.add_argument(
        "--out-dir",
        default="data/ED24/myPedestrain_06/EBF_Part2/_n125_path_failure_stats_772",
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
            tau_base_us=int(args.tau_base_us),
            tau_step_us=int(args.tau_step_us),
            mix_bins=int(args.mix_bins),
            angle_bins=int(args.angle_bins),
        )
        results.append(rec)
        print(
            f"[OK] {env_name} events={rec['events']} sameR4={rec['density_same_mean']['R4']:.3f} "
            f"mixR4={rec['mix_r4_mean']:.3f} dt>30ms={rec['dt_hist_ratio'][4]:.3f}"
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
