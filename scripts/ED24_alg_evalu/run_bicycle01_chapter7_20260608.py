"""Convert ED24 Bicycle_01 and run Chapter-7 style algorithm comparison.

Scope requested:
  Bicycle_01 levels: 1.8V, 2.5V, 3.3V
  Algorithms: N149_v2.2, EBF, PFD, STCF_orig, BAF, TS

The converted npy files are written to:
  D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_01

Input CSV labels follow the ED24 convention used elsewhere in this project:
  label=0 means signal, label=1 means noise.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PY = Path(r"D:/software/Anaconda_envs/envs/myEVS/python.exe")
RAW_DIR = Path(r"D:/hjx_workspace/scientific_reserach/dataset/ED24/Bicycle_01")
NPY_DIR = Path(r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_01")
OUT_DIR = ROOT / "data" / "ED24" / "myBicycle_01" / "chapter7_20260608"

LEVELS = ("1.8", "2.5", "3.3")
WIDTH = 346
HEIGHT = 260
THR_STD = "0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,8"
THR_TS = "0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5"
TAU_UNIFIED = "2000,5000,10000,15000,20000,25000,30000,40000,50000,64000,80000,100000,128000,160000,200000"


@dataclass(frozen=True)
class Sample:
    level: str
    clean: Path
    noisy: Path


def convert_level(level: str, force: bool) -> Sample:
    src = RAW_DIR / f"Bicycle_01_{level}.csv"
    noisy = NPY_DIR / f"Bicycle_01_{level}.npy"
    clean = NPY_DIR / f"Bicycle_01_{level}_signal_only.npy"
    if not src.exists():
        raise FileNotFoundError(src)

    if force or not noisy.exists():
        cmd = [
            str(PY),
            str(ROOT / "scripts" / "ED24csv_to_npy.py"),
            "--in",
            str(src),
            "--out",
            str(noisy),
            "--tick-ns",
            "1000",
            "--timestamp-unit",
            "us",
            "--width",
            str(WIDTH),
            "--height",
            str(HEIGHT),
            "--signal-label-value",
            "0",
            "--overwrite",
        ]
        subprocess.run(cmd, cwd=ROOT, check=True, timeout=600, capture_output=True, text=True)

    if force or not clean.exists():
        arr = np.load(noisy, mmap_mode="r")
        sig = np.asarray(arr[arr["label"] == 1]).copy()
        NPY_DIR.mkdir(parents=True, exist_ok=True)
        np.save(clean, sig)

    return Sample(level=level, clean=clean, noisy=noisy)


def compute_stats(samples: list[Sample]) -> pd.DataFrame:
    rows = []
    for s in samples:
        arr = np.load(s.noisy, mmap_mode="r")
        events = int(arr.shape[0])
        signal = int(np.sum(arr["label"] == 1))
        noise = events - signal
        duration_s = float((int(arr["t"][-1]) - int(arr["t"][0])) * 1e-6) if events > 1 else 0.0
        mev_s = events / duration_s / 1e6 if duration_s > 0 else 0.0
        hz_per_px = events / duration_s / (WIDTH * HEIGHT) if duration_s > 0 else 0.0
        signal_ratio = signal / events if events > 0 else 0.0
        noise_ratio = noise / events if events > 0 else 0.0
        sn = signal / noise if noise > 0 else float("inf")
        rows.append(
            {
                "level": f"{s.level}V",
                "events": events,
                "duration_s": duration_s,
                "mev_s": mev_s,
                "hz_per_px": hz_per_px,
                "signal_ratio": signal_ratio,
                "noise_ratio": noise_ratio,
                "sn_ratio": sn,
            }
        )
    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "bicycle01_event_stats.csv", index=False, encoding="utf-8-sig")
    return df


def cases_for(alg: str) -> list[tuple[str, list[str], dict[str, str]]]:
    if alg == "baf":
        return [("baf_r1", ["--method", "baf", "--engine", "cpp", "--radius-px", "1", "--min-neighbors", "1", "--param", "time-us", "--values", TAU_UNIFIED], {})]

    if alg == "stcf_orig":
        return [
            (
                f"stcf_orig_k{k}",
                ["--method", "stcf_original", "--engine", "cpp", "--radius-px", "1", "--min-neighbors", str(k), "--param", "time-us", "--values", TAU_UNIFIED],
                {},
            )
            for k in (1, 2, 3, 4, 5, 6)
        ]

    if alg == "pfd":
        out = []
        taus = "1000,2000,4000,8000,16000,32000,64000,128000,256000,512000"
        for m in (1, 2):
            for k in (1, 2):
                out.append(
                    (
                        f"pfd_r1_m{m}_k{k}",
                        ["--method", "pfd", "--engine", "cpp", "--radius-px", "1", "--min-neighbors", str(k), "--refractory-us", str(m), "--pfd-mode", "a", "--param", "time-us", "--values", taus],
                        {},
                    )
                )
        return out

    if alg == "ebf":
        grid = [(r, t) for r in (2, 3, 4, 5) for t in (16000, 32000, 64000, 128000, 256000)]
        return [(f"ebf_r{r}_tau{t}", ["--method", "ebf", "--engine", "cpp", "--radius-px", str(r), "--time-us", str(t), "--param", "min-neighbors", "--values", THR_STD], {}) for r, t in grid]

    if alg == "ts":
        grid = [(r, t) for r in (1, 2, 3) for t in (16000, 32000, 64000, 128000)]
        return [(f"ts_r{r}_tau{t}", ["--method", "ts", "--engine", "cpp", "--radius-px", str(r), "--time-us", str(t), "--param", "min-neighbors", "--values", THR_TS], {}) for r, t in grid]

    if alg == "n149_v22":
        env = {
            "MYEVS_N149_HOT_BITS": "8",
            "MYEVS_N149_HOT_INT_BITS": "3",
            "MYEVS_N149_HOT_DECAY_K": "2",
            "MYEVS_N149_HOT_K": "2",
            "MYEVS_N149_HOT_FUNC": "rational",
            "MYEVS_N149_SIGMA": "2.75",
            "MYEVS_N149_ALPHA_FIXED": "0.25",
        }
        # Chapter 7 ED24 Bicycle v2.2 fixed point, not the later r=3 deployment ablation.
        return [("n149_v22_r5_tau256000", ["--method", "n149", "--engine", "cpp", "--radius-px", "5", "--time-us", "256000", "--param", "min-neighbors", "--values", THR_STD], env)]

    raise ValueError(alg)


def run_case(sample: Sample, alg: str, tag: str, args: list[str], env_extra: dict[str, str], out_csv: Path) -> None:
    env = os.environ.copy()
    env.update(env_extra)
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
        "--match-us",
        "0",
        "--match-bin-radius",
        "0",
        "--tag",
        tag,
        "--out-csv",
        str(out_csv),
        "--append",
    ] + args
    subprocess.run(cmd, cwd=ROOT, env=env, check=True, timeout=1200, capture_output=True, text=True)


def best_auc_tag_max_f1(csv_path: Path) -> dict[str, object]:
    df = pd.read_csv(csv_path)
    for col in ("auc", "f1", "value", "tpr", "fpr", "precision"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    tag_auc = df.groupby("tag", as_index=False)["auc"].max().sort_values("auc", ascending=False)
    best_tag = str(tag_auc.iloc[0]["tag"])
    best_auc = float(tag_auc.iloc[0]["auc"])
    sub = df[df["tag"] == best_tag].copy()
    best_f1 = sub.sort_values(["f1", "tpr"], ascending=[False, False]).iloc[0]
    return {
        "best_tag": best_tag,
        "auc": best_auc,
        "f1_at_auc_best": float(best_f1["f1"]),
        "f1_threshold": float(best_f1["value"]),
        "roc_csv": str(csv_path),
    }


def run_one(sample: Sample, alg: str, force: bool) -> dict[str, object]:
    out_dir = OUT_DIR / alg
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"roc_{alg}_{sample.level}.csv"
    if force and out_csv.exists():
        out_csv.unlink()
    if not out_csv.exists():
        for tag, args, env in cases_for(alg):
            run_case(sample, alg, tag, args, env, out_csv)
    best = best_auc_tag_max_f1(out_csv)
    return {"level": sample.level, "algorithm": alg, **best}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-convert", action="store_true")
    ap.add_argument("--force-eval", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    samples = [convert_level(level, args.force_convert) for level in LEVELS]
    stats = compute_stats(samples)
    print("[bicycle01] stats:")
    print(stats.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    algorithms = ("n149_v22", "ebf", "pfd", "stcf_orig", "baf", "ts")
    jobs = [(s, alg) for s in samples for alg in algorithms]
    rows: list[dict[str, object]] = []
    t0 = time.time()
    print(f"[bicycle01] eval jobs={len(jobs)} out={OUT_DIR}")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_one, s, alg, args.force_eval): (s, alg) for s, alg in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            rows.append(row)
            rate = i / max(time.time() - t0, 1e-6)
            print(
                f"[{i:02d}/{len(jobs)} | {rate:.2f} j/s] {row['level']} {row['algorithm']} "
                f"AUC={row['auc']:.6f} F1={row['f1_at_auc_best']:.6f} tag={row['best_tag']}",
                flush=True,
            )

    summary = pd.DataFrame(rows).sort_values(["level", "algorithm"])
    summary.to_csv(OUT_DIR / "bicycle01_chapter7_summary.csv", index=False, encoding="utf-8-sig")
    wide_auc = summary.pivot(index="algorithm", columns="level", values="auc")
    wide_f1 = summary.pivot(index="algorithm", columns="level", values="f1_at_auc_best")
    print("\n[bicycle01] AUC:")
    print(wide_auc.to_string(float_format=lambda x: f"{x:.6f}"))
    print("\n[bicycle01] F1:")
    print(wide_f1.to_string(float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
