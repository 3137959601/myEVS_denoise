"""Run final ED24 SPIF/N149 v2.2 single-point evaluation.

Final ED24 deployment parameters:
  r=3, tau=256ms, sigma=2.75, alpha=0.25

The script reports AUC and the max F1 on the AUC-best threshold curve for
Chapter 7 tables.
"""
from __future__ import annotations

import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PY = Path(r"D:/software/Anaconda_envs/envs/myEVS/python.exe")
OUT_DIR = ROOT / "data" / "Hyperparameter ablation_study" / "ed24_final_n149_v22_20260608"

THR = "0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,8"
RADIUS = 3
TAU_US = 256_000
SIGMA = "2.75"
ALPHA = "0.25"


@dataclass(frozen=True)
class Sample:
    kind: str
    level: str
    label: str
    clean: Path
    noisy: Path
    width: int = 346
    height: int = 260


def samples() -> list[Sample]:
    ped_root = Path(r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06")
    bike_root = Path(r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02")
    out: list[Sample] = []
    for level in ("1.8", "2.1", "2.5", "3.3"):
        ped_prefix = f"Pedestrain_06_{level}"
        out.append(
            Sample(
                "ped",
                level,
                f"ED24 Ped {level}V",
                ped_root / f"{ped_prefix}_signal_only.npy",
                ped_root / f"{ped_prefix}.npy",
            )
        )
    for level in ("1.8", "2.1", "2.5", "3.3"):
        bike_prefix = f"Bicycle_02_{level}"
        out.append(
            Sample(
                "bike",
                level,
                f"ED24 Bike {level}V",
                bike_root / f"{bike_prefix}_signal_only.npy",
                bike_root / f"{bike_prefix}.npy",
            )
        )
    return out


def run_one(sample: Sample, force: bool) -> dict[str, object]:
    for path in (sample.clean, sample.noisy):
        if not path.exists():
            raise FileNotFoundError(path)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"n149_v22_final_{sample.kind}_{sample.level.replace('.', 'p')}_r{RADIUS}_tau{TAU_US}_s{SIGMA.replace('.', 'p')}"
    out_csv = OUT_DIR / f"{tag}.csv"
    if force and out_csv.exists():
        out_csv.unlink()

    if not out_csv.exists():
        env = os.environ.copy()
        env.update(
            {
                "MYEVS_N149_HOT_BITS": "8",
                "MYEVS_N149_HOT_INT_BITS": "3",
                "MYEVS_N149_HOT_DECAY_K": "2",
                "MYEVS_N149_HOT_K": "2",
                "MYEVS_N149_HOT_FUNC": "rational",
                "MYEVS_N149_SIGMA": SIGMA,
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
            str(sample.width),
            "--height",
            str(sample.height),
            "--tick-ns",
            "1000",
            "--engine",
            "cpp",
            "--method",
            "n149",
            "--radius-px",
            str(RADIUS),
            "--time-us",
            str(TAU_US),
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

    df = pd.read_csv(out_csv)
    for col in ("auc", "f1", "value", "tpr", "fpr", "precision"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    best_f1 = df.sort_values(["f1", "tpr"], ascending=[False, False]).iloc[0]
    return {
        "kind": sample.kind,
        "level": sample.level,
        "label": sample.label,
        "r": RADIUS,
        "tau_us": TAU_US,
        "sigma": float(SIGMA),
        "alpha": float(ALPHA),
        "auc": float(df["auc"].max()),
        "f1_at_auc_best": float(best_f1["f1"]),
        "f1_threshold": float(best_f1["value"]),
        "roc_csv": str(out_csv),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    t0 = time.time()
    rows: list[dict[str, object]] = []
    todo = samples()
    print(f"[ed24-final-n149] tasks={len(todo)} out={OUT_DIR}")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_one, sample, args.force): sample for sample in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            rows.append(row)
            rate = i / max(time.time() - t0, 1e-6)
            print(
                f"[{i:02d}/{len(todo)} | {rate:.2f} t/s] {row['label']} "
                f"AUC={row['auc']:.6f} F1={row['f1_at_auc_best']:.6f}",
                flush=True,
            )

    df = pd.DataFrame(rows).sort_values(["kind", "level"])
    df.to_csv(OUT_DIR / "ed24_final_n149_v22_summary.csv", index=False, encoding="utf-8-sig")
    print("\n[ed24-final-n149] summary:")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
