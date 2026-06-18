"""Coordinate sweep N149 v2.2 parameters on ED24 Bicycle_01.

The sweep optimizes one parameter at a time using the mean AUC over
Bicycle_01 1.8/2.5/3.3V, while keeping alpha fixed at the ED24 setting.
It is intended as a targeted follow-up to the Chapter-7 Bicycle_01 table.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PY = Path(r"D:/software/Anaconda_envs/envs/myEVS/python.exe")
DATA_DIR = Path(r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_01")
OUT_DIR = ROOT / "data" / "Hyperparameter ablation_study" / "bicycle01_n149_coord_20260608"

WIDTH = 346
HEIGHT = 260
LEVELS = ("1.8", "2.5", "3.3")
THR = "0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,8"

R_VALUES = (1, 2, 3, 4, 5, 7, 9)
TAU_VALUES = (8000, 16000, 32000, 64000, 128000, 256000, 384000, 512000, 768000)
SIGMA_VALUES = (1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 4.0, 5.0)

START_R = 5
START_TAU = 256000
START_SIGMA = 2.75
ALPHA = "0.25"


@dataclass(frozen=True)
class Params:
    r: int
    tau_us: int
    sigma: float


@dataclass(frozen=True)
class Sample:
    level: str
    clean: Path
    noisy: Path


def safe_float(value: float) -> str:
    return ("%g" % float(value)).replace(".", "p")


def sample_for(level: str) -> Sample:
    prefix = f"Bicycle_01_{level}"
    sample = Sample(
        level=level,
        clean=DATA_DIR / f"{prefix}_signal_only.npy",
        noisy=DATA_DIR / f"{prefix}.npy",
    )
    for path in (sample.clean, sample.noisy):
        if not path.exists():
            raise FileNotFoundError(path)
    return sample


def metric_csv(sample: Sample, params: Params, force: bool) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = (
        f"b01_n149_lv{sample.level.replace('.', 'p')}"
        f"_r{params.r}_tau{params.tau_us}_s{safe_float(params.sigma)}"
    )
    out_csv = OUT_DIR / f"{tag}.csv"
    if force and out_csv.exists():
        out_csv.unlink()
    if out_csv.exists():
        return out_csv

    env = os.environ.copy()
    env.update(
        {
            "MYEVS_N149_HOT_BITS": "8",
            "MYEVS_N149_HOT_INT_BITS": "3",
            "MYEVS_N149_HOT_DECAY_K": "2",
            "MYEVS_N149_HOT_K": "2",
            "MYEVS_N149_HOT_FUNC": "rational",
            "MYEVS_N149_SIGMA": str(float(params.sigma)),
            "MYEVS_N149_ALPHA_FIXED": ALPHA,
        }
    )
    cmd = [
        str(PY),
        "-m",
        "myevs.cli",
        "roc",
        "--clean",
        str(sample.clean),
        "--noisy",
        str(sample.noisy),
        "--assume",
        "npy",
        "--width",
        str(WIDTH),
        "--height",
        str(HEIGHT),
        "--tick-ns",
        "1000",
        "--engine",
        "cpp",
        "--method",
        "n149",
        "--radius-px",
        str(params.r),
        "--time-us",
        str(params.tau_us),
        "--param",
        "min-neighbors",
        "--values",
        THR,
        "--match-us",
        "0",
        "--match-bin-radius",
        "0",
        "--tag",
        tag,
        "--out-csv",
        str(out_csv),
    ]
    subprocess.run(cmd, cwd=ROOT, env=env, check=True, timeout=1200, capture_output=True, text=True)
    return out_csv


def run_one(sample: Sample, params: Params, stage: str, value: object, force: bool) -> dict[str, object]:
    csv_path = metric_csv(sample, params, force)
    df = pd.read_csv(csv_path)
    for col in ("auc", "f1", "value", "tpr"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    best_f1 = df.sort_values(["f1", "tpr"], ascending=[False, False]).iloc[0]
    return {
        "stage": stage,
        "value": value,
        "level": sample.level,
        "r": params.r,
        "tau_us": params.tau_us,
        "sigma": params.sigma,
        "auc": float(df["auc"].max()),
        "f1_at_auc_best": float(best_f1["f1"]),
        "f1_threshold": float(best_f1["value"]),
        "roc_csv": str(csv_path),
    }


def summarize_stage(rows: list[dict[str, object]], stage: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    stage_df = df[df["stage"] == stage].copy()
    summary = (
        stage_df.groupby(["stage", "value", "r", "tau_us", "sigma"], as_index=False)
        .agg(mean_auc=("auc", "mean"), mean_f1=("f1_at_auc_best", "mean"))
        .sort_values(["mean_auc", "mean_f1"], ascending=[False, False])
    )
    return summary


def choose_best(rows: list[dict[str, object]], stage: str) -> Params:
    summary = summarize_stage(rows, stage)
    best = summary.iloc[0]
    return Params(r=int(best["r"]), tau_us=int(best["tau_us"]), sigma=float(best["sigma"]))


def candidates(stage: str, current: Params) -> list[tuple[object, Params]]:
    if stage == "r":
        return [(r, Params(int(r), current.tau_us, current.sigma)) for r in R_VALUES]
    if stage == "tau":
        return [(tau, Params(current.r, int(tau), current.sigma)) for tau in TAU_VALUES]
    if stage == "sigma":
        return [(sigma, Params(current.r, current.tau_us, float(sigma))) for sigma in SIGMA_VALUES]
    raise ValueError(stage)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--passes", type=int, default=2)
    args = ap.parse_args()

    samples = [sample_for(level) for level in LEVELS]
    current = Params(START_R, START_TAU, START_SIGMA)
    all_rows: list[dict[str, object]] = []
    trajectory: list[dict[str, object]] = [
        {"step": "start", "r": current.r, "tau_us": current.tau_us, "sigma": current.sigma}
    ]

    t0 = time.time()
    for pass_idx in range(1, args.passes + 1):
        for axis in ("r", "tau", "sigma"):
            stage = f"p{pass_idx}_{axis}"
            jobs = [(sample, params, stage, value) for value, params in candidates(axis, current) for sample in samples]
            print(f"[b01-coord] {stage}: jobs={len(jobs)} current={current}", flush=True)
            stage_rows: list[dict[str, object]] = []
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futures = {ex.submit(run_one, sample, params, stage, value, args.force): (sample, params, value) for sample, params, stage, value in jobs}
                for i, fut in enumerate(as_completed(futures), 1):
                    row = fut.result()
                    stage_rows.append(row)
                    all_rows.append(row)
                    rate = len(all_rows) / max(time.time() - t0, 1e-6)
                    print(
                        f"[{stage} {i:02d}/{len(jobs)} | {rate:.2f} t/s] "
                        f"lv={row['level']} r={row['r']} tau={row['tau_us']} "
                        f"sigma={row['sigma']} AUC={row['auc']:.6f} F1={row['f1_at_auc_best']:.6f}",
                        flush=True,
                    )
            summary = summarize_stage(stage_rows, stage)
            summary.to_csv(OUT_DIR / f"{stage}_summary.csv", index=False, encoding="utf-8-sig")
            current = choose_best(stage_rows, stage)
            best = summary.iloc[0]
            trajectory.append(
                {
                    "step": stage,
                    "r": current.r,
                    "tau_us": current.tau_us,
                    "sigma": current.sigma,
                    "mean_auc": float(best["mean_auc"]),
                    "mean_f1": float(best["mean_f1"]),
                }
            )
            print(f"[b01-coord] {stage} best={current} mean_auc={best['mean_auc']:.6f}", flush=True)

    raw = pd.DataFrame(all_rows).sort_values(["stage", "value", "level"])
    raw.to_csv(OUT_DIR / "bicycle01_n149_coord_raw.csv", index=False, encoding="utf-8-sig")

    summaries = []
    for stage in raw["stage"].drop_duplicates():
        summaries.append(summarize_stage(all_rows, str(stage)))
    pd.concat(summaries, ignore_index=True).to_csv(OUT_DIR / "bicycle01_n149_coord_stage_summary.csv", index=False, encoding="utf-8-sig")

    traj = pd.DataFrame(trajectory)
    traj.to_csv(OUT_DIR / "bicycle01_n149_coord_trajectory.csv", index=False, encoding="utf-8-sig")

    final_rows = [run_one(sample, current, "final", "final", False) for sample in samples]
    final = pd.DataFrame(final_rows).sort_values("level")
    final.to_csv(OUT_DIR / "bicycle01_n149_coord_final.csv", index=False, encoding="utf-8-sig")

    print("\n[b01-coord] trajectory:")
    print(traj.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\n[b01-coord] final:")
    print(final[["level", "r", "tau_us", "sigma", "auc", "f1_at_auc_best", "f1_threshold"]].to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
