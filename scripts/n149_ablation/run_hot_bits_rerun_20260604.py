"""Rerun N149 hot_state bit-width ablation for README2 section 10.6.

The experiment fixes the current N149 v2.2 parameters and varies only
MYEVS_N149_HOT_BITS.  "int32" is represented by HOT_BITS=31, matching the
native implementation's positive int32 hot-mask branch.
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
OUT_DIR = ROOT / "data" / "Hyperparameter ablation_study" / "hot_bits_rerun_20260604"

THR = "0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,8"
BITS = (31, 16, 14, 12, 10, 8, 6, 5, 4, 3, 2)


@dataclass(frozen=True)
class Sample:
    key: str
    label: str
    clean: Path
    noisy: Path
    width: int
    height: int
    radius: int
    tau_us: int
    sigma: str
    alpha: str


SAMPLES = (
    Sample(
        key="drive_10hz",
        label="Drive 10hz",
        clean=Path(r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_10hz_ed24_withlabel/driving_noise_10hz_signal_only.npy"),
        noisy=Path(r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_10hz_ed24_withlabel/driving_noise_10hz_labeled.npy"),
        width=346,
        height=260,
        radius=2,
        tau_us=32000,
        sigma="1.75",
        alpha="0.05",
    ),
    Sample(
        key="ped_1p8",
        label="Ped 1.8V",
        clean=Path(r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_1.8_signal_only.npy"),
        noisy=Path(r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_1.8.npy"),
        width=346,
        height=260,
        radius=5,
        tau_us=256000,
        sigma="2.75",
        alpha="0.25",
    ),
    Sample(
        key="ped_3p3",
        label="Ped 3.3V",
        clean=Path(r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3_signal_only.npy"),
        noisy=Path(r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3.npy"),
        width=346,
        height=260,
        radius=5,
        tau_us=256000,
        sigma="2.75",
        alpha="0.25",
    ),
    Sample(
        key="ped_2p5",
        label="Ped 2.5V",
        clean=Path(r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_2.5_signal_only.npy"),
        noisy=Path(r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_2.5.npy"),
        width=346,
        height=260,
        radius=5,
        tau_us=256000,
        sigma="2.75",
        alpha="0.25",
    ),
    Sample(
        key="bike_2p5",
        label="Bike 2.5V",
        clean=Path(r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_2.5_signal_only.npy"),
        noisy=Path(r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_2.5.npy"),
        width=346,
        height=260,
        radius=5,
        tau_us=256000,
        sigma="2.75",
        alpha="0.25",
    ),
    Sample(
        key="bike_1p8",
        label="Bike 1.8V",
        clean=Path(r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_1.8_signal_only.npy"),
        noisy=Path(r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_1.8.npy"),
        width=346,
        height=260,
        radius=5,
        tau_us=256000,
        sigma="2.75",
        alpha="0.25",
    ),
    Sample(
        key="bike_3p3",
        label="Bike 3.3V",
        clean=Path(r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_3.3_signal_only.npy"),
        noisy=Path(r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_3.3.npy"),
        width=346,
        height=260,
        radius=5,
        tau_us=256000,
        sigma="2.75",
        alpha="0.25",
    ),
    Sample(
        key="dvsclean_mah00447_ratio50",
        label="MAH00447 r50",
        clean=Path(r"D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy/MAH00447/ratio50/MAH00447_ratio50_signal_only.npy"),
        noisy=Path(r"D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy/MAH00447/ratio50/MAH00447_ratio50_labeled.npy"),
        width=1280,
        height=720,
        radius=5,
        tau_us=128000,
        sigma="2.5",
        alpha="0.25",
    ),
    Sample(
        key="dvsclean_mah00447_ratio100",
        label="MAH00447 r100",
        clean=Path(r"D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy/MAH00447/ratio100/MAH00447_ratio100_signal_only.npy"),
        noisy=Path(r"D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy/MAH00447/ratio100/MAH00447_ratio100_labeled.npy"),
        width=1280,
        height=720,
        radius=5,
        tau_us=128000,
        sigma="2.5",
        alpha="0.25",
    ),
)


def bit_label(bits: int) -> str:
    return "int32" if bits >= 31 else str(bits)


def run_one(sample: Sample, bits: int, force: bool) -> dict[str, object]:
    for path in (sample.clean, sample.noisy):
        if not path.exists():
            raise FileNotFoundError(path)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"hotbits_{sample.key}_{bit_label(bits)}"
    out_csv = OUT_DIR / f"{tag}.csv"
    if force and out_csv.exists():
        out_csv.unlink()

    if not out_csv.exists():
        env = os.environ.copy()
        env.update(
            {
                "MYEVS_N149_HOT_BITS": str(bits),
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
    for col in ("auc", "f1", "value"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    best = df.sort_values(["auc", "f1"], ascending=[False, False]).iloc[0]
    return {
        "sample": sample.key,
        "sample_label": sample.label,
        "hot_bits": bits,
        "hot_bits_label": bit_label(bits),
        "auc": float(best["auc"]),
        "best_threshold": best.get("value", ""),
        "roc_csv": str(out_csv),
    }


def write_outputs(rows: list[dict[str, object]]) -> None:
    raw = pd.DataFrame(rows).sort_values(["sample", "hot_bits"], ascending=[True, False])
    raw.to_csv(OUT_DIR / "hot_bits_raw.csv", index=False, encoding="utf-8-sig")

    base = raw[raw["hot_bits"] == 31][["sample", "auc"]].rename(columns={"auc": "auc_int32"})
    delta = raw.merge(base, on="sample", how="left")
    delta["delta_vs_int32"] = delta["auc"] - delta["auc_int32"]
    delta.to_csv(OUT_DIR / "hot_bits_delta.csv", index=False, encoding="utf-8-sig")

    order = ["int32", "16", "14", "12", "10", "8", "6", "5", "4", "3", "2"]
    auc_pivot = delta.pivot(index="hot_bits_label", columns="sample_label", values="auc").reindex(order)
    d_pivot = delta.pivot(index="hot_bits_label", columns="sample_label", values="delta_vs_int32").reindex(order)

    summary_rows = []
    for bit in order:
        row = {"HOT_BITS": bit}
        for label in [s.label for s in SAMPLES]:
            row[label] = auc_pivot.loc[bit, label]
            row[f"{label} Delta"] = d_pivot.loc[bit, label]
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "hot_bits_summary.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="rerun existing per-case CSVs")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    tasks = [(sample, bits) for sample in SAMPLES for bits in BITS]
    print(f"[hot-bits] tasks={len(tasks)} out={OUT_DIR}")
    t0 = time.time()
    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_one, sample, bits, args.force): (sample, bits) for sample, bits in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            rows.append(row)
            rate = i / max(time.time() - t0, 1e-6)
            print(
                f"[{i:02d}/{len(tasks)} | {rate:.2f} t/s] "
                f"{row['sample_label']} bits={row['hot_bits_label']} auc={row['auc']:.6f}",
                flush=True,
            )

    write_outputs(rows)
    summary = pd.read_csv(OUT_DIR / "hot_bits_summary.csv")
    print("\n[hot-bits] summary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
