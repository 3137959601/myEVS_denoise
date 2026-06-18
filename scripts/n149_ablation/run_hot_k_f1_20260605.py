"""Sweep integer hot discount K for N149 v2.2 on Ped/Bike 3.3V.

This validates f(H)=(H+Q)/(K*H+Q) with current normalized hot_state.
For each dataset and K, report AUC and max F1 on the AUC-best curve.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PY = Path(r"D:/software/Anaconda_envs/envs/myEVS/python.exe")
OUT_DIR = ROOT / "data" / "Hyperparameter ablation_study" / "hot_k_f1_20260605"
THR = "0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,8"


@dataclass(frozen=True)
class Sample:
    key: str
    label: str
    clean: Path
    noisy: Path
    width: int = 346
    height: int = 260
    radius: int = 5
    tau_us: int = 256000
    sigma: str = "2.75"
    alpha: str = "0.25"


SAMPLES = (
    Sample(
        key="ped_3p3",
        label="Ped 3.3V",
        clean=Path(r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3_signal_only.npy"),
        noisy=Path(r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3.npy"),
    ),
    Sample(
        key="bike_3p3",
        label="Bike 3.3V",
        clean=Path(r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_3.3_signal_only.npy"),
        noisy=Path(r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_3.3.npy"),
    ),
)


def run_one(sample: Sample, k: int, force: bool) -> dict[str, object]:
    for path in (sample.clean, sample.noisy):
        if not path.exists():
            raise FileNotFoundError(path)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"hotk_{sample.key}_K{k}"
    out_csv = OUT_DIR / f"{tag}.csv"
    if force and out_csv.exists():
        out_csv.unlink()

    if not out_csv.exists():
        env = os.environ.copy()
        env.update(
            {
                "MYEVS_N149_HOT_K": str(k),
                "MYEVS_N149_HOT_BITS": "8",
                "MYEVS_N149_HOT_INT_BITS": "3",
                "MYEVS_N149_HOT_DECAY_K": "2",
                "MYEVS_N149_SIGMA": sample.sigma,
                "MYEVS_N149_ALPHA_FIXED": sample.alpha,
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
            str(sample.radius),
            "--time-us",
            str(sample.tau_us),
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
    auc = float(df["auc"].max())
    best_f1 = df.sort_values(["f1", "tpr"], ascending=[False, False]).iloc[0]
    return {
        "sample": sample.key,
        "sample_label": sample.label,
        "K": k,
        "f_range": f"[{1.0 / k:.3f},1]" if k > 0 else "",
        "auc": auc,
        "best_f1": float(best_f1["f1"]),
        "f1_threshold": best_f1.get("value", ""),
        "tpr_at_f1": best_f1.get("tpr", ""),
        "fpr_at_f1": best_f1.get("fpr", ""),
        "precision_at_f1": best_f1.get("precision", ""),
        "roc_csv": str(out_csv),
    }


def write_outputs(rows: list[dict[str, object]]) -> None:
    raw = pd.DataFrame(rows).sort_values(["sample", "K"])
    raw.to_csv(OUT_DIR / "hot_k_f1_raw.csv", index=False, encoding="utf-8-sig")
    auc_p = raw.pivot(index="K", columns="sample_label", values="auc")
    f1_p = raw.pivot(index="K", columns="sample_label", values="best_f1")
    thr_p = raw.pivot(index="K", columns="sample_label", values="f1_threshold")
    out = pd.DataFrame({"K": sorted(raw["K"].unique())})
    for label in [s.label for s in SAMPLES]:
        out[f"{label} AUC"] = out["K"].map(auc_p[label])
        out[f"{label} F1"] = out["K"].map(f1_p[label])
        out[f"{label} F1 thr"] = out["K"].map(thr_p[label])
    out.to_csv(OUT_DIR / "hot_k_f1_summary.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    global OUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--k-list", default="1,2,3,4,5,6,7,8")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    OUT_DIR = Path(args.out_dir)
    ks = [int(x.strip()) for x in args.k_list.split(",") if x.strip()]
    (OUT_DIR.parent / "logs").mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR.parent / "logs" / f"hot_lambda_sweep_{time.strftime('%Y%m%d_%H%M%S')}.json").open("w", encoding="utf-8") as f:
        json.dump({"script": __file__, "out_dir": str(OUT_DIR), "args": vars(args), "k_list": ks}, f, indent=2)

    tasks = [(sample, k) for sample in SAMPLES for k in ks]
    print(f"[hot-k] tasks={len(tasks)} out={OUT_DIR}")
    rows: list[dict[str, object]] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_one, sample, k, args.force): (sample, k) for sample, k in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            rows.append(row)
            rate = i / max(time.time() - t0, 1e-6)
            print(
                f"[{i:02d}/{len(tasks)} | {rate:.2f} t/s] "
                f"{row['sample_label']} K={row['K']} auc={row['auc']:.6f} f1={row['best_f1']:.6f}",
                flush=True,
            )

    write_outputs(rows)
    summary = pd.read_csv(OUT_DIR / "hot_k_f1_summary.csv")
    print("\n[hot-k] summary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
