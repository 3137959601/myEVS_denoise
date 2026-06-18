"""R=2 tau-sweep degradation for SPIF(N149 v2.2) and EBF.

This supplements README2 Section 8.9 with ED24 Ped/Bike 1.8/2.5/3.3V and
DVSCLEAN MAH00447 ratio50/ratio100.  For each dataset and algorithm, radius is
fixed to r=2 and tau is swept; threshold is swept by the standard ROC grid.
The reference "best" AUCs are the current full-sweep best values documented in
README2.
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
OUT_DIR = ROOT / "data" / "Hyperparameter ablation_study" / "r2_tau_degradation_20260605"
THR = "0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,8"
TAU_ED24 = (16000, 32000, 64000, 128000, 256000, 384000, 512000)
TAU_DVSCLEAN = (16000, 32000, 64000, 128000, 256000)


@dataclass(frozen=True)
class Sample:
    key: str
    label: str
    clean: Path
    noisy: Path
    width: int
    height: int
    tau_values: tuple[int, ...]
    spif_sigma: str
    spif_alpha: str
    spif_best_param: str
    spif_best_auc: float
    ebf_best_param: str
    ebf_best_auc: float


def ed24_sample(kind: str, voltage: str, spif_best_auc: float, ebf_best_auc: float) -> Sample:
    if kind == "ped":
        root = Path(r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06")
        prefix = f"Pedestrain_06_{voltage}"
        label = f"ED24 Ped {voltage}V"
    elif kind == "bike":
        root = Path(r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02")
        prefix = f"Bicycle_02_{voltage}"
        label = f"ED24 Bike {voltage}V"
    else:
        raise ValueError(kind)

    return Sample(
        key=f"{kind}_{voltage.replace('.', 'p')}",
        label=label,
        clean=root / f"{prefix}_signal_only.npy",
        noisy=root / f"{prefix}.npy",
        width=346,
        height=260,
        tau_values=TAU_ED24,
        spif_sigma="2.75",
        spif_alpha="0.25",
        spif_best_param="(5,256ms)",
        spif_best_auc=spif_best_auc,
        ebf_best_param="scene-wise",
        ebf_best_auc=ebf_best_auc,
    )


SAMPLES = (
    ed24_sample("ped", "1.8", 0.9563, 0.9454),
    ed24_sample("ped", "2.5", 0.9443, 0.9189),
    ed24_sample("ped", "3.3", 0.9375, 0.9078),
    ed24_sample("bike", "1.8", 0.9865, 0.9802),
    ed24_sample("bike", "2.5", 0.9800, 0.9686),
    ed24_sample("bike", "3.3", 0.9746, 0.9604),
    Sample(
        key="mah447_r50",
        label="MAH00447 ratio50",
        clean=Path(r"D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy/MAH00447/ratio50/MAH00447_ratio50_signal_only.npy"),
        noisy=Path(r"D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy/MAH00447/ratio50/MAH00447_ratio50_labeled.npy"),
        width=1280,
        height=720,
        tau_values=TAU_DVSCLEAN,
        spif_sigma="2.5",
        spif_alpha="0.25",
        spif_best_param="(5,128ms)",
        spif_best_auc=0.9959,
        ebf_best_param="(3,64ms)",
        ebf_best_auc=0.9941,
    ),
    Sample(
        key="mah447_r100",
        label="MAH00447 ratio100",
        clean=Path(r"D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy/MAH00447/ratio100/MAH00447_ratio100_signal_only.npy"),
        noisy=Path(r"D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy/MAH00447/ratio100/MAH00447_ratio100_labeled.npy"),
        width=1280,
        height=720,
        tau_values=TAU_DVSCLEAN,
        spif_sigma="2.5",
        spif_alpha="0.25",
        spif_best_param="(5,128ms)",
        spif_best_auc=0.9947,
        ebf_best_param="(4,32ms)",
        ebf_best_auc=0.9932,
    ),
)


def run_one(sample: Sample, algo: str, tau_us: int, force: bool) -> dict[str, object]:
    for path in (sample.clean, sample.noisy):
        if not path.exists():
            raise FileNotFoundError(path)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"r2tau_{algo}_{sample.key}_tau{tau_us}"
    out_csv = OUT_DIR / f"{tag}.csv"
    if force and out_csv.exists():
        out_csv.unlink()

    if not out_csv.exists():
        env = os.environ.copy()
        method = "n149" if algo == "spif" else "ebf"
        if algo == "spif":
            env.update(
                {
                    "MYEVS_N149_HOT_BITS": "8",
                    "MYEVS_N149_HOT_INT_BITS": "3",
                    "MYEVS_N149_HOT_DECAY_K": "2",
                    "MYEVS_N149_SIGMA": sample.spif_sigma,
                    "MYEVS_N149_ALPHA_FIXED": sample.spif_alpha,
                    "MYEVS_N149_HOT_K": "2",
                    "MYEVS_N149_HOT_FUNC": "rational",
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
            method,
            "--radius-px",
            "2",
            "--time-us",
            str(tau_us),
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
    best_f1 = df.sort_values(["f1"], ascending=False).iloc[0]
    return {
        "sample": sample.key,
        "sample_label": sample.label,
        "algo": algo,
        "tau_us": tau_us,
        "r2_auc": float(df["auc"].max()),
        "best_f1": float(best_f1["f1"]),
        "best_f1_threshold": float(best_f1["value"]),
        "roc_csv": str(out_csv),
    }


def write_outputs(rows: list[dict[str, object]]) -> pd.DataFrame:
    raw = pd.DataFrame(rows).sort_values(["sample", "algo", "tau_us"])
    raw.to_csv(OUT_DIR / "r2_tau_degradation_raw.csv", index=False, encoding="utf-8-sig")

    summary_rows: list[dict[str, object]] = []
    for sample in SAMPLES:
        for algo in ("spif", "ebf"):
            sub = raw[(raw["sample"] == sample.key) & (raw["algo"] == algo)]
            best = sub.sort_values(["r2_auc", "best_f1"], ascending=[False, False]).iloc[0]
            ref_auc = sample.spif_best_auc if algo == "spif" else sample.ebf_best_auc
            ref_param = sample.spif_best_param if algo == "spif" else sample.ebf_best_param
            summary_rows.append(
                {
                    "sample": sample.key,
                    "sample_label": sample.label,
                    "algo": algo,
                    "self_best_param": ref_param,
                    "self_best_auc": ref_auc,
                    "r2_best_tau_us": int(best["tau_us"]),
                    "r2_best_auc": float(best["r2_auc"]),
                    "delta_r2": float(best["r2_auc"]) - float(ref_auc),
                    "r2_best_f1": float(best["best_f1"]),
                    "r2_best_f1_threshold": float(best["best_f1_threshold"]),
                }
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "r2_tau_degradation_summary.csv", index=False, encoding="utf-8-sig")
    wide = summary.pivot(index="sample_label", columns="algo", values=["self_best_auc", "r2_best_tau_us", "r2_best_auc", "delta_r2"])
    wide.to_csv(OUT_DIR / "r2_tau_degradation_wide.csv", encoding="utf-8-sig")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    tasks = [(sample, algo, tau) for sample in SAMPLES for algo in ("spif", "ebf") for tau in sample.tau_values]
    print(f"[r2-tau] tasks={len(tasks)} out={OUT_DIR}")
    rows: list[dict[str, object]] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_one, sample, algo, tau, args.force): (sample, algo, tau) for sample, algo, tau in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            rows.append(row)
            rate = i / max(time.time() - t0, 1e-6)
            print(
                f"[{i:03d}/{len(tasks)} | {rate:.2f} t/s] "
                f"{row['sample_label']} {row['algo']} tau={row['tau_us']} auc={row['r2_auc']:.6f}",
                flush=True,
            )

    summary = write_outputs(rows)
    print("\n[r2-tau] summary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
