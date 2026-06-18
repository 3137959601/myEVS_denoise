"""Point-check current N149 v2.2 on Bicycle_01 and Bicycle_02."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PY = Path(r"D:/software/Anaconda_envs/envs/myEVS/python.exe")
OUT = ROOT / "data" / "Hyperparameter ablation_study" / "bicycle01_02_pointcheck_20260608"
THR = "0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,8"

SAMPLES = [
    ("B01", "1.8", Path(r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_01"), "Bicycle_01"),
    ("B01", "2.5", Path(r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_01"), "Bicycle_01"),
    ("B01", "3.3", Path(r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_01"), "Bicycle_01"),
    ("B02", "1.8", Path(r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02"), "Bicycle_02"),
    ("B02", "2.1", Path(r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02"), "Bicycle_02"),
    ("B02", "2.5", Path(r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02"), "Bicycle_02"),
    ("B02", "3.3", Path(r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02"), "Bicycle_02"),
]

POINTS = [
    ("b01_best", 3, 128000, 1.75),
    ("ed24_final", 3, 256000, 2.75),
    ("b02_hist", 5, 256000, 2.75),
]


def run_one(ds: str, lv: str, root: Path, prefix: str, point: str, r: int, tau: int, sigma: float) -> dict[str, object]:
    OUT.mkdir(parents=True, exist_ok=True)
    clean = root / f"{prefix}_{lv}_signal_only.npy"
    noisy = root / f"{prefix}_{lv}.npy"
    tag = f"{ds}_{lv.replace('.', 'p')}_{point}_r{r}_t{tau}_s{str(sigma).replace('.', 'p')}"
    out_csv = OUT / f"{tag}.csv"
    if not out_csv.exists():
        env = os.environ.copy()
        env.update(
            {
                "MYEVS_N149_HOT_BITS": "8",
                "MYEVS_N149_HOT_INT_BITS": "3",
                "MYEVS_N149_HOT_DECAY_K": "2",
                "MYEVS_N149_HOT_K": "2",
                "MYEVS_N149_HOT_FUNC": "rational",
                "MYEVS_N149_SIGMA": str(sigma),
                "MYEVS_N149_ALPHA_FIXED": "0.25",
            }
        )
        cmd = [
            str(PY),
            "-m",
            "myevs.cli",
            "roc",
            "--clean",
            str(clean),
            "--noisy",
            str(noisy),
            "--assume",
            "npy",
            "--width",
            "346",
            "--height",
            "260",
            "--tick-ns",
            "1000",
            "--engine",
            "cpp",
            "--method",
            "n149",
            "--radius-px",
            str(r),
            "--time-us",
            str(tau),
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
    for col in ("auc", "f1", "value"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    best_f1 = df.sort_values("f1", ascending=False).iloc[0]
    return {
        "dataset": ds,
        "level": lv,
        "point": point,
        "r": r,
        "tau_us": tau,
        "sigma": sigma,
        "auc": float(df["auc"].max()),
        "f1": float(best_f1["f1"]),
        "threshold": float(best_f1["value"]),
    }


def main() -> None:
    rows = []
    for sample in SAMPLES:
        for point in POINTS:
            row = run_one(*sample, *point)
            rows.append(row)
            print(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "bicycle01_02_pointcheck_summary.csv", index=False, encoding="utf-8-sig")
    print(df.pivot_table(index=["dataset", "level"], columns="point", values="auc").to_string())


if __name__ == "__main__":
    main()
