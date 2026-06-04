"""Fine sweep for k = 2*sigma^2 / r^2 around 8/9 and 1.0.

Goal: test whether using exp(-d^2/(k*r^2)) with a fixed k is better explained
and whether k=1.0 (i.e., 2*sigma^2=r^2) can outperform k=8/9.
"""
import math
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

PY = r"D:/software/Anaconda_envs/envs/myEVS/python.exe"
THR = "0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,8"
OUT_DIR = r"D:/hjx_workspace/scientific_reserach/projects/myEVS/data/Hyperparameter ablation_study/rsigma_k_sweep"

# k = 2*sigma^2/r^2
K_LIST = [0.75, 0.80, 0.85, 8.0 / 9.0, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20]

DATASETS = {
    "drive": {
        "w": 346,
        "h": 260,
        "r": 2,
        "tau": 32000,
        "alpha": "0.05",
        "sigma_star": 1.75,
        "levels": {
            "1hz": (
                r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_1hz_ed24_withlabel/driving_noise_1hz_signal_only.npy",
                r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_1hz_ed24_withlabel/driving_noise_1hz_labeled.npy",
            ),
            "3hz": (
                r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_3hz_ed24_withlabel/driving_noise_3hz_signal_only.npy",
                r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_3hz_ed24_withlabel/driving_noise_3hz_labeled.npy",
            ),
            "5hz": (
                r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_5hz_ed24_withlabel/driving_noise_5hz_signal_only.npy",
                r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_5hz_ed24_withlabel/driving_noise_5hz_labeled.npy",
            ),
            "7hz": (
                r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_7hz_ed24_withlabel/driving_noise_7hz_signal_only.npy",
                r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_7hz_ed24_withlabel/driving_noise_7hz_labeled.npy",
            ),
            "10hz": (
                r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_10hz_ed24_withlabel/driving_noise_10hz_signal_only.npy",
                r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_10hz_ed24_withlabel/driving_noise_10hz_labeled.npy",
            ),
        },
    },
    "ped": {
        "w": 346,
        "h": 260,
        "r": 5,
        "tau": 256000,
        "alpha": "0.25",
        "sigma_star": 2.75,
        "levels": {
            "1.8": (
                r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_1.8_signal_only.npy",
                r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_1.8.npy",
            ),
            "2.1": (
                r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_2.1_signal_only.npy",
                r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_2.1.npy",
            ),
            "2.5": (
                r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_2.5_signal_only.npy",
                r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_2.5.npy",
            ),
            "3.3": (
                r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3_signal_only.npy",
                r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3.npy",
            ),
        },
    },
    "bike": {
        "w": 346,
        "h": 260,
        "r": 5,
        "tau": 256000,
        "alpha": "0.25",
        "sigma_star": 2.75,
        "levels": {
            "1.8": (
                r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_1.8_signal_only.npy",
                r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_1.8.npy",
            ),
            "2.1": (
                r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_2.1_signal_only.npy",
                r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_2.1.npy",
            ),
            "2.5": (
                r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_2.5_signal_only.npy",
                r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_2.5.npy",
            ),
            "3.3": (
                r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_3.3_signal_only.npy",
                r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_3.3.npy",
            ),
        },
    },
}


def sigma_from_k(k: float, r: int) -> float:
    return float(r * math.sqrt(k / 2.0))


def run_one(ds_key: str, level: str, k: float, cfg: dict, clean: str, noisy: str) -> tuple:
    sigma = sigma_from_k(k, cfg["r"])
    env = os.environ.copy()
    env["MYEVS_N149_HOT_BITS"] = "8"
    env["MYEVS_N149_SIGMA"] = f"{sigma:.6f}"
    env["MYEVS_N149_ALPHA_FIXED"] = cfg["alpha"]
    env["MYEVS_N149_HOT_DECAY_K"] = "2"

    tag = f"rk_{ds_key}_{level}_k{str(round(k,6)).replace('.', 'p')}"
    csv = os.path.join(OUT_DIR, f"{tag}.csv")
    cmd = [
        PY,
        "-m",
        "myevs.cli",
        "roc",
        "--clean",
        clean,
        "--noisy",
        noisy,
        "--assume",
        "npy",
        "--width",
        str(cfg["w"]),
        "--height",
        str(cfg["h"]),
        "--tick-ns",
        "1000",
        "--engine",
        "cpp",
        "--method",
        "n149",
        "--radius-px",
        str(cfg["r"]),
        "--time-us",
        str(cfg["tau"]),
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
        csv,
    ]
    subprocess.run(cmd, env=env, check=True, timeout=900, capture_output=True)
    df = pd.read_csv(csv)
    auc = float(df["auc"].max())
    return ds_key, level, float(k), sigma, auc


def run_free_sigma(ds_key: str, level: str, cfg: dict, clean: str, noisy: str) -> tuple:
    sigma = float(cfg["sigma_star"])
    env = os.environ.copy()
    env["MYEVS_N149_HOT_BITS"] = "8"
    env["MYEVS_N149_SIGMA"] = f"{sigma:.6f}"
    env["MYEVS_N149_ALPHA_FIXED"] = cfg["alpha"]
    env["MYEVS_N149_HOT_DECAY_K"] = "2"

    tag = f"rk_{ds_key}_{level}_free_sigma"
    csv = os.path.join(OUT_DIR, f"{tag}.csv")
    cmd = [
        PY, "-m", "myevs.cli", "roc",
        "--clean", clean, "--noisy", noisy, "--assume", "npy",
        "--width", str(cfg["w"]), "--height", str(cfg["h"]), "--tick-ns", "1000",
        "--engine", "cpp", "--method", "n149",
        "--radius-px", str(cfg["r"]), "--time-us", str(cfg["tau"]),
        "--param", "min-neighbors", "--values", THR,
        "--match-us", "0", "--match-bin-radius", "0",
        "--tag", tag, "--out-csv", csv,
    ]
    subprocess.run(cmd, env=env, check=True, timeout=900, capture_output=True)
    df = pd.read_csv(csv)
    auc = float(df["auc"].max())
    return ds_key, level, "free_sigma", sigma, auc


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    tasks = []
    for ds_key, cfg in DATASETS.items():
        for level, (clean, noisy) in cfg["levels"].items():
            tasks.append(("free", ds_key, level, None, cfg, clean, noisy))
            for k in K_LIST:
                tasks.append(("k", ds_key, level, k, cfg, clean, noisy))

    print(f"[r-k] tasks={len(tasks)}")
    t0 = time.time()
    rows = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {}
        for t in tasks:
            kind, ds_key, level, k, cfg, clean, noisy = t
            if kind == "free":
                fut = ex.submit(run_free_sigma, ds_key, level, cfg, clean, noisy)
            else:
                fut = ex.submit(run_one, ds_key, level, k, cfg, clean, noisy)
            futures[fut] = t

        done = 0
        for fut in as_completed(futures):
            kind, ds_key, level, *_ = futures[fut]
            res = fut.result()
            rows.append(res)
            done += 1
            rate = done / max(1e-6, (time.time() - t0))
            print(f"[{done}/{len(tasks)} | {rate:.2f} t/s] {kind} {ds_key}/{level}", flush=True)

    out_rows = []
    for ds_key, level, kval, sigma, auc in rows:
        out_rows.append((ds_key, level, kval, sigma, auc))
    df = pd.DataFrame(out_rows, columns=["dataset", "level", "k", "sigma", "auc"])
    df.to_csv(os.path.join(OUT_DIR, "r_sigma_k_raw.csv"), index=False)

    base = df[df["k"] == "free_sigma"][["dataset", "level", "auc"]].rename(columns={"auc": "auc_base"})
    merged = df.merge(base, on=["dataset", "level"], how="left")
    merged["delta_vs_base"] = merged["auc"] - merged["auc_base"]
    merged.to_csv(os.path.join(OUT_DIR, "r_sigma_k_delta.csv"), index=False)

    sweep = merged[merged["k"] != "free_sigma"].copy()
    sweep["k"] = sweep["k"].astype(float)
    summary = sweep.groupby(["dataset", "k"], as_index=False)[["auc", "delta_vs_base"]].mean()
    summary.to_csv(os.path.join(OUT_DIR, "r_sigma_k_summary.csv"), index=False)

    best = summary.sort_values(["dataset", "auc"], ascending=[True, False]).groupby("dataset", as_index=False).first()
    best.to_csv(os.path.join(OUT_DIR, "r_sigma_k_best.csv"), index=False)

    print("\n[r-k] best-by-dataset:")
    print(best.to_string(index=False))

    for ds in ["drive", "ped", "bike"]:
        sub = summary[summary["dataset"] == ds].sort_values("k")
        row_8_9 = sub.iloc[(sub["k"] - (8.0 / 9.0)).abs().argmin()]
        row_1 = sub.iloc[(sub["k"] - 1.0).abs().argmin()]
        print(
            f"[r-k] {ds}: k=8/9 auc={row_8_9['auc']:.6f}, "
            f"k=1 auc={row_1['auc']:.6f}, delta(k=1-8/9)={row_1['auc'] - row_8_9['auc']:+.6f}"
        )


if __name__ == "__main__":
    main()

