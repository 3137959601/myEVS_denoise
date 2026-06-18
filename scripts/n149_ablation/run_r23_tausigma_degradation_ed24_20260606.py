"""R=2/R=3 constrained sweep for SPIF(N149 v2.2) and EBF on ED24.

Compared with the earlier r=2-only script, SPIF re-sweeps both tau and sigma
for each constrained radius because the best spatial decay scale changes when
the support radius changes.  Alpha is kept at the dataset-level optimum.
EBF has no sigma parameter, so it sweeps tau only.
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
OUT_DIR = ROOT / "data" / "Hyperparameter ablation_study" / "r23_tausigma_ed24_20260606"
THR = "0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,8"
TAU_ED24 = (16000, 32000, 64000, 128000, 256000, 384000, 512000)
SIGMA_ED24 = (1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0)
RADII = (2, 3)


@dataclass(frozen=True)
class Sample:
    key: str
    label: str
    clean: Path
    noisy: Path
    width: int
    height: int
    tau_values: tuple[int, ...]
    sigma_values: tuple[float, ...]
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
        sigma_values=SIGMA_ED24,
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
)


def safe_float(v: float) -> str:
    return ("%g" % float(v)).replace(".", "p")


def run_one(sample: Sample, algo: str, radius: int, tau_us: int, sigma: float | None, force: bool) -> dict[str, object]:
    for path in (sample.clean, sample.noisy):
        if not path.exists():
            raise FileNotFoundError(path)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sigma_tag = "na" if sigma is None else safe_float(sigma)
    tag = f"r23_{algo}_{sample.key}_r{radius}_tau{tau_us}_s{sigma_tag}"
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
                    "MYEVS_N149_SIGMA": str(float(sigma)),
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
            str(radius),
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
    best_f1 = df.sort_values("f1", ascending=False).iloc[0]
    return {
        "sample": sample.key,
        "sample_label": sample.label,
        "algo": algo,
        "radius": radius,
        "tau_us": tau_us,
        "sigma": "" if sigma is None else float(sigma),
        "auc": float(df["auc"].max()),
        "best_f1": float(best_f1["f1"]),
        "best_f1_threshold": float(best_f1["value"]),
        "roc_csv": str(out_csv),
    }


def build_tasks() -> list[tuple[Sample, str, int, int, float | None]]:
    tasks: list[tuple[Sample, str, int, int, float | None]] = []
    for sample in SAMPLES:
        for radius in RADII:
            for tau in sample.tau_values:
                tasks.append((sample, "ebf", radius, tau, None))
            for sigma in sample.sigma_values:
                for tau in sample.tau_values:
                    tasks.append((sample, "spif", radius, tau, sigma))
    return tasks


def write_outputs(rows: list[dict[str, object]]) -> pd.DataFrame:
    raw = pd.DataFrame(rows).sort_values(["sample", "algo", "radius", "sigma", "tau_us"])
    raw.to_csv(OUT_DIR / "r23_tausigma_degradation_raw.csv", index=False, encoding="utf-8-sig")

    summary_rows: list[dict[str, object]] = []
    for sample in SAMPLES:
        for algo in ("spif", "ebf"):
            ref_auc = sample.spif_best_auc if algo == "spif" else sample.ebf_best_auc
            ref_param = sample.spif_best_param if algo == "spif" else sample.ebf_best_param
            for radius in RADII:
                sub = raw[(raw["sample"] == sample.key) & (raw["algo"] == algo) & (raw["radius"] == radius)]
                best = sub.sort_values(["auc", "best_f1"], ascending=[False, False]).iloc[0]
                best_param = f"(r={radius},tau={int(best['tau_us'])//1000}ms"
                if algo == "spif":
                    best_param += f",sigma={float(best['sigma']):g}"
                best_param += ")"
                summary_rows.append(
                    {
                        "sample": sample.key,
                        "sample_label": sample.label,
                        "algo": algo,
                        "radius": radius,
                        "self_best_param": ref_param,
                        "self_best_auc": ref_auc,
                        "best_param": best_param,
                        "best_tau_us": int(best["tau_us"]),
                        "best_sigma": "" if algo == "ebf" else float(best["sigma"]),
                        "auc": float(best["auc"]),
                        "delta": float(best["auc"]) - float(ref_auc),
                        "f1_at_auc_best": float(best["best_f1"]),
                        "f1_threshold": float(best["best_f1_threshold"]),
                    }
                )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "r23_tausigma_degradation_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    tasks = build_tasks()
    if args.limit > 0:
        tasks = tasks[: args.limit]

    print(f"[r23-tausigma] tasks={len(tasks)} out={OUT_DIR}")
    rows: list[dict[str, object]] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_one, *task, args.force): task for task in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            rows.append(row)
            rate = i / max(time.time() - t0, 1e-6)
            sigma = row["sigma"] if row["sigma"] != "" else "-"
            print(
                f"[{i:04d}/{len(tasks)} | {rate:.2f} t/s] "
                f"{row['sample_label']} {row['algo']} r={row['radius']} "
                f"tau={row['tau_us']} sigma={sigma} auc={row['auc']:.6f}",
                flush=True,
            )

    summary = write_outputs(rows)
    print("\n[r23-tausigma] summary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
