"""R=3 tau-only component ablation for ED24 Pedestrian.

Scope:
  ED24 Ped 1.8V / 2.5V / 3.3V
  r=3, sigma=2.75, alpha=0.25
  sweep tau only for each ablation variant

Variants match README2 section 11.2E:
  baseline, no_spatial, no_opp, no_hot, no_polarity, time_only
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
OUT_DIR = ROOT / "data" / "Hyperparameter ablation_study" / "component_ped_r3_tau_20260608"

THR = "0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,8"
TAU_VALUES = (16_000, 32_000, 64_000, 128_000, 256_000, 384_000, 512_000)
RADIUS = 3
SIGMA = "2.75"
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


def run_one(sample: Sample, variant: str, tau_us: int, force: bool) -> dict[str, object]:
    for path in (sample.clean, sample.noisy):
        if not path.exists():
            raise FileNotFoundError(path)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"ped_r3_comp_{sample.key}_{variant}_tau{tau_us}"
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
        "tau_us": tau_us,
        "sigma": float(SIGMA),
        "alpha": float(ALPHA),
        "auc": float(df["auc"].max()),
        "best_f1": float(best_f1["f1"]),
        "best_f1_threshold": float(best_f1["value"]),
        "roc_csv": str(out_csv),
    }


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
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
                    "best_tau_us": int(best["tau_us"]),
                    "auc": float(best["auc"]),
                    "delta_vs_baseline": float(best["auc"]) - base_auc,
                    "f1_at_auc_best": float(best["best_f1"]),
                    "f1_threshold": float(best["best_f1_threshold"]),
                    "baseline_auc": base_auc,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    tasks = [(sample, variant, tau) for sample in samples() for variant in VARIANTS for tau in TAU_VALUES]
    print(f"[ped-r3-component] tasks={len(tasks)} out={OUT_DIR}")
    t0 = time.time()
    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_one, sample, variant, tau, args.force): (sample, variant, tau) for sample, variant, tau in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            rows.append(row)
            rate = i / max(time.time() - t0, 1e-6)
            print(
                f"[{i:03d}/{len(tasks)} | {rate:.2f} t/s] {row['sample_label']} "
                f"{row['variant']} tau={row['tau_us']} auc={row['auc']:.6f} f1={row['best_f1']:.6f}",
                flush=True,
            )

    raw = pd.DataFrame(rows).sort_values(["sample", "variant", "tau_us"])
    raw.to_csv(OUT_DIR / "ped_r3_component_tau_raw.csv", index=False, encoding="utf-8-sig")

    summary = summarize(raw)
    summary.to_csv(OUT_DIR / "ped_r3_component_tau_summary.csv", index=False, encoding="utf-8-sig")
    mean = (
        summary.groupby("variant", as_index=False)
        .agg(
            auc=("auc", "mean"),
            delta_vs_baseline=("delta_vs_baseline", "mean"),
            f1_at_auc_best=("f1_at_auc_best", "mean"),
        )
        .sort_values("variant")
    )
    mean.to_csv(OUT_DIR / "ped_r3_component_tau_mean.csv", index=False, encoding="utf-8-sig")

    print("\n[ped-r3-component] summary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\n[ped-r3-component] mean:")
    print(mean.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
