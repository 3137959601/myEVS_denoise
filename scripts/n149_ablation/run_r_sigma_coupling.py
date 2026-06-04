"""Validate whether sigma can be coupled to radius r in N149 v2.2.

Compares 4 settings at fixed per-dataset optimal (r, tau, alpha):
1) free_sigma: use current best fixed sigma per dataset
2) sigma_eq_r: sigma = r
3) sigma_2r_div_3: sigma = 2r/3
4) sigma_r_div_2: sigma = r/2
"""
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

PY = r"D:/software/Anaconda_envs/envs/myEVS/python.exe"
THR = "0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,8"
OUT_DIR = r"D:/hjx_workspace/scientific_reserach/projects/myEVS/data/Hyperparameter ablation_study/rsigma_coupling"


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


def sigma_from_mode(mode: str, r: int, sigma_star: float) -> float:
    if mode == "free_sigma":
        return float(sigma_star)
    if mode == "sigma_eq_r":
        return float(r)
    if mode == "sigma_2r_div_3":
        return float(2.0 * r / 3.0)
    if mode == "sigma_r_div_2":
        return float(r / 2.0)
    raise ValueError(mode)


def run_one(ds_key: str, level: str, mode: str, cfg: dict, clean: str, noisy: str) -> tuple:
    sigma = sigma_from_mode(mode, cfg["r"], cfg["sigma_star"])
    env = os.environ.copy()
    env["MYEVS_N149_HOT_BITS"] = "8"
    env["MYEVS_N149_SIGMA"] = f"{sigma:.6f}"
    env["MYEVS_N149_ALPHA_FIXED"] = cfg["alpha"]
    env["MYEVS_N149_HOT_DECAY_K"] = "2"

    tag = f"rsig_{ds_key}_{level}_{mode}"
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
    return ds_key, level, mode, sigma, auc


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    modes = ["free_sigma", "sigma_eq_r", "sigma_2r_div_3", "sigma_r_div_2"]

    tasks = []
    for ds_key, cfg in DATASETS.items():
        for level, (clean, noisy) in cfg["levels"].items():
            for mode in modes:
                tasks.append((ds_key, level, mode, cfg, clean, noisy))

    print(f"[r-sigma] tasks={len(tasks)} datasets={len(DATASETS)} modes={len(modes)}")
    t0 = time.time()
    rows = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(run_one, *t): t for t in tasks}
        done = 0
        for fut in as_completed(futures):
            ds_key, level, mode, sigma, auc = fut.result()
            rows.append((ds_key, level, mode, sigma, auc))
            done += 1
            rate = done / max(1e-6, (time.time() - t0))
            print(f"[{done}/{len(tasks)} | {rate:.2f} t/s] {ds_key}/{level}/{mode} sigma={sigma:.4f} auc={auc:.6f}", flush=True)

    out = pd.DataFrame(rows, columns=["dataset", "level", "mode", "sigma", "auc"])
    out.to_csv(os.path.join(OUT_DIR, "r_sigma_coupling_raw.csv"), index=False)

    base = out[out["mode"] == "free_sigma"][["dataset", "level", "auc"]].rename(columns={"auc": "auc_base"})
    merged = out.merge(base, on=["dataset", "level"], how="left")
    merged["delta_vs_base"] = merged["auc"] - merged["auc_base"]
    merged.to_csv(os.path.join(OUT_DIR, "r_sigma_coupling_delta.csv"), index=False)

    summary = (
        merged.groupby(["dataset", "mode"], as_index=False)[["auc", "delta_vs_base"]]
        .mean()
        .sort_values(["dataset", "mode"])
    )
    summary.to_csv(os.path.join(OUT_DIR, "r_sigma_coupling_summary.csv"), index=False)
    print("\n[r-sigma] summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

