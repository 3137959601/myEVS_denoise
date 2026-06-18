"""R=3 sigma sweep after fixing tau from the Ped component ablation.

For each ED24 Ped sample and component variant, this script reads the best tau
from component_ped_r3_tau_20260608/ped_r3_component_tau_summary.csv, then
sweeps sigma only with r=3 and alpha=0.25.
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
TAU_SUMMARY = ROOT / "data" / "Hyperparameter ablation_study" / "component_ped_r3_tau_20260608" / "ped_r3_component_tau_summary.csv"
OUT_DIR = ROOT / "data" / "Hyperparameter ablation_study" / "component_ped_r3_sigma_20260608"

THR = "0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,8"
SIGMA_VALUES = (1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0)
RADIUS = 3
ALPHA = "0.25"


@dataclass(frozen=True)
class Sample:
    key: str
    label: str
    clean: Path
    noisy: Path
    width: int = 346
    height: int = 260


def samples() -> tuple[Sample, ...]:
    root = Path(r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06")
    out = []
    for level in ("1.8", "2.5", "3.3"):
        prefix = f"Pedestrain_06_{level}"
        out.append(
            Sample(
                key=level.replace(".", "p"),
                label=f"ED24 Ped {level}V",
                clean=root / f"{prefix}_signal_only.npy",
                noisy=root / f"{prefix}.npy",
            )
        )
    return tuple(out)


VARIANTS: dict[str, dict[str, str]] = {
    "baseline": {},
    "no_spatial": {"MYEVS_N149_NO_SPATIAL": "1"},
    "no_opp": {"MYEVS_N149_NO_OPP": "1"},
    "no_hot": {"MYEVS_N149_NO_HOT": "1"},
    "no_polarity": {"MYEVS_N149_BLIND": "1"},
    "time_only": {
        "MYEVS_N149_NO_SPATIAL": "1",
        "MYEVS_N149_NO_HOT": "1",
        "MYEVS_N149_BLIND": "1",
    },
}


def sigma_tag(sigma: float) -> str:
    return ("%g" % sigma).replace(".", "p")


def load_fixed_tau() -> dict[tuple[str, str], int]:
    df = pd.read_csv(TAU_SUMMARY)
    out: dict[tuple[str, str], int] = {}
    for _, row in df.iterrows():
        out[(str(row["sample"]), str(row["variant"]))] = int(row["best_tau_us"])
    return out


def run_one(sample: Sample, variant: str, tau_us: int, sigma: float, force: bool) -> dict[str, object]:
    for path in (sample.clean, sample.noisy):
        if not path.exists():
            raise FileNotFoundError(path)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"ped_r3_compsigma_{sample.key}_{variant}_tau{tau_us}_s{sigma_tag(sigma)}"
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
                "MYEVS_N149_SIGMA": str(float(sigma)),
                "MYEVS_N149_ALPHA_FIXED": ALPHA,
            }
        )
        env.update(VARIANTS[variant])
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
    for col in ("auc", "f1", "value", "tpr"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    best_f1 = df.sort_values(["f1", "tpr"], ascending=[False, False]).iloc[0]
    return {
        "sample": sample.key,
        "sample_label": sample.label,
        "variant": variant,
        "radius": RADIUS,
        "fixed_tau_us": tau_us,
        "sigma": float(sigma),
        "auc": float(df["auc"].max()),
        "best_f1": float(best_f1["f1"]),
        "best_f1_threshold": float(best_f1["value"]),
        "roc_csv": str(out_csv),
    }


def summarize(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for sample in samples():
        sub_sample = raw[raw["sample"] == sample.key]
        base_best = sub_sample[sub_sample["variant"] == "baseline"].sort_values(
            ["auc", "best_f1"], ascending=[False, False]
        ).iloc[0]
        base_auc = float(base_best["auc"])
        for variant in VARIANTS:
            sub = sub_sample[sub_sample["variant"] == variant]
            best = sub.sort_values(["auc", "best_f1"], ascending=[False, False]).iloc[0]
            rows.append(
                {
                    "sample": sample.key,
                    "sample_label": sample.label,
                    "variant": variant,
                    "fixed_tau_us": int(best["fixed_tau_us"]),
                    "best_sigma": float(best["sigma"]),
                    "auc": float(best["auc"]),
                    "delta_vs_baseline": float(best["auc"]) - base_auc,
                    "f1_at_auc_best": float(best["best_f1"]),
                    "f1_threshold": float(best["best_f1_threshold"]),
                    "baseline_auc": base_auc,
                }
            )
    summary = pd.DataFrame(rows)
    mean = (
        summary.groupby("variant", as_index=False)
        .agg(
            auc=("auc", "mean"),
            delta_vs_baseline=("delta_vs_baseline", "mean"),
            f1_at_auc_best=("f1_at_auc_best", "mean"),
        )
        .sort_values("variant")
    )
    return summary, mean


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    fixed_tau = load_fixed_tau()
    tasks = []
    for sample in samples():
        for variant in VARIANTS:
            tau = fixed_tau[(sample.key, variant)]
            for sigma in SIGMA_VALUES:
                tasks.append((sample, variant, tau, sigma))

    print(f"[ped-r3-compsigma] tasks={len(tasks)} out={OUT_DIR}")
    t0 = time.time()
    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_one, sample, variant, tau, sigma, args.force): (sample, variant, tau, sigma) for sample, variant, tau, sigma in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            rows.append(row)
            rate = i / max(time.time() - t0, 1e-6)
            print(
                f"[{i:03d}/{len(tasks)} | {rate:.2f} t/s] {row['sample_label']} "
                f"{row['variant']} tau={row['fixed_tau_us']} sigma={row['sigma']} "
                f"auc={row['auc']:.6f}",
                flush=True,
            )

    raw = pd.DataFrame(rows).sort_values(["sample", "variant", "sigma"])
    raw.to_csv(OUT_DIR / "ped_r3_component_sigma_raw.csv", index=False, encoding="utf-8-sig")
    summary, mean = summarize(raw)
    summary.to_csv(OUT_DIR / "ped_r3_component_sigma_summary.csv", index=False, encoding="utf-8-sig")
    mean.to_csv(OUT_DIR / "ped_r3_component_sigma_mean.csv", index=False, encoding="utf-8-sig")

    print("\n[ped-r3-compsigma] summary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\n[ped-r3-compsigma] mean:")
    print(mean.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
