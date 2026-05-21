"""Phase 2: sigma fine-sweep on all datasets. Multi-threaded."""
import subprocess, os, time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

PY = r"D:/software/Anaconda_envs/envs/myEVS/python.exe"
THR = "0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,8"
OUT_BASE = r"D:/hjx_workspace/scientific_reserach/projects/myEVS/data/Hyperparameter ablation_study"

DATASETS = {
    "drive": {
        "levels": {
            "1hz": (r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_1hz_ed24_withlabel/driving_noise_1hz_signal_only.npy",
                    r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_1hz_ed24_withlabel/driving_noise_1hz_labeled.npy"),
            "3hz": (r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_3hz_ed24_withlabel/driving_noise_3hz_signal_only.npy",
                    r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_3hz_ed24_withlabel/driving_noise_3hz_labeled.npy"),
            "5hz": (r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_5hz_ed24_withlabel/driving_noise_5hz_signal_only.npy",
                    r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_5hz_ed24_withlabel/driving_noise_5hz_labeled.npy"),
            "7hz": (r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_7hz_ed24_withlabel/driving_noise_7hz_signal_only.npy",
                    r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_7hz_ed24_withlabel/driving_noise_7hz_labeled.npy"),
            "10hz": (r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_10hz_ed24_withlabel/driving_noise_10hz_signal_only.npy",
                     r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_10hz_ed24_withlabel/driving_noise_10hz_labeled.npy"),
        },
        "w": 346, "h": 260,
        "r": 2, "tau": 32000, "alpha": "0",
        "sigmas": [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0],
    },
    "ped": {
        "levels": {
            "1.8": (r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_1.8_signal_only.npy",
                    r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_1.8.npy"),
            "2.1": (r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_2.1_signal_only.npy",
                    r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_2.1.npy"),
            "2.5": (r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_2.5_signal_only.npy",
                    r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_2.5.npy"),
            "3.3": (r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3_signal_only.npy",
                    r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3.npy"),
        },
        "w": 346, "h": 260,
        "r": 5, "tau": 256000, "alpha": "0.25",
        "sigmas": [2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 4.0],
    },
    "bike": {
        "levels": {
            "1.8": (r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_1.8_signal_only.npy",
                    r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_1.8.npy"),
            "2.1": (r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_2.1_signal_only.npy",
                    r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_2.1.npy"),
            "2.5": (r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_2.5_signal_only.npy",
                    r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_2.5.npy"),
            "3.3": (r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_3.3_signal_only.npy",
                    r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_3.3.npy"),
        },
        "w": 346, "h": 260,
        "r": 5, "tau": 256000, "alpha": "0.25",
        "sigmas": [2.0, 2.25, 2.5, 2.75, 3.0, 3.5],
    },
    "dvsclean": {
        "levels": {},
        "w": 1280, "h": 720,
        "r": 5, "tau": 128000, "alpha": "0.25",
        "sigmas": [2.0, 2.25, 2.5, 2.75, 3.0, 3.5],
    },
    "led": {
        "levels": {},
        "w": 1280, "h": 720,
        "r": 2, "tau": 8000, "alpha": "1.0",
        "sigmas": [1.0, 1.25, 1.5, 1.75, 2.0, 2.5],
    },
}

# Build DVSCLEAN levels
for scene in ["MAH00444","MAH00446","MAH00447","MAH00448","MAH00449"]:
    for ratio in ["ratio50","ratio100"]:
        k = f"{scene}_{ratio}"
        DATASETS["dvsclean"]["levels"][k] = (
            f"D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy/{scene}/{ratio}/{scene}_{ratio}_signal_only.npy",
            f"D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy/{scene}/{ratio}/{scene}_{ratio}_labeled.npy")

# Build LED levels
for s in ["scene_100","scene_1004","scene_1018","scene_1028","scene_1032","scene_1033","scene_1034","scene_1043","scene_1045","scene_1046"]:
    DATASETS["led"]["levels"][s] = (
        f"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/{s}/slices_00031_00040_100ms/{s}_100ms_signal_only.npy",
        f"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/{s}/slices_00031_00040_100ms/{s}_100ms_labeled.npy")

def run_one(ds_key, level, clean, noisy, sigma_val, ds):
    env = os.environ.copy()
    env["MYEVS_N149_HOT_BITS"] = "16"
    env["MYEVS_N149_SIGMA"] = str(sigma_val)
    if ds["alpha"] != "0.25":
        env["MYEVS_N149_ALPHA_FIXED"] = ds["alpha"]
    tag = f"p2_sigma_{ds_key}_{level}_{sigma_val}".replace(".","p")
    out_dir = f"{OUT_BASE}/{ds_key}"
    csv = f"{out_dir}/{tag}.csv"
    cmd = [PY,'-m','myevs.cli','roc','--clean',clean,'--noisy',noisy,'--assume','npy',
           '--width',str(ds["w"]),'--height',str(ds["h"]),'--tick-ns','1000',
           '--engine','cpp','--method','n149',
           '--radius-px',str(ds["r"]),'--time-us',str(ds["tau"]),
           '--param','min-neighbors','--values',THR,
           '--match-us','0','--match-bin-radius','0',
           '--tag',tag,'--out-csv',csv,'--append']
    subprocess.run(cmd, check=True, timeout=900, env=env, capture_output=True)
    df = pd.read_csv(csv)
    return ds_key, level, sigma_val, float(df.loc[df["auc"].idxmax()]["auc"])

# Build all tasks
all_tasks = []
for key, ds in DATASETS.items():
    for lv, (cl, ny) in ds["levels"].items():
        for s in ds["sigmas"]:
            all_tasks.append((key, lv, cl, ny, s, ds))

total = len(all_tasks)
print(f"Phase 2 sigma fine: {total} tasks across {len(DATASETS)} datasets")
t0 = time.time()

results = []
with ThreadPoolExecutor(max_workers=12) as ex:
    futures = {ex.submit(run_one, *t): t for t in all_tasks}
    done = 0
    for f in as_completed(futures):
        key, lv, sv, auc = f.result()
        results.append((key, lv, sv, auc))
        done += 1
        rate = done / (time.time() - t0) if time.time() > t0 else 0
        eta = (total - done) / rate / 60 if rate > 0 else 0
        print(f"[{done}/{total} | {rate:.1f}t/s | ETA {eta:.0f}m] {key}/{lv} sigma={sv} AUC={auc:.4f}", flush=True)

# Save per-dataset summaries
df = pd.DataFrame(results, columns=["dataset","level","sigma","auc"])
for key in DATASETS:
    sub = df[df.dataset == key]
    sub.to_csv(f"{OUT_BASE}/{key}/phase2_sigma.csv", index=False)
    # Show best per level
    print(f"\n-- {key} sigma best --")
    for lv in sorted(sub.level.unique()):
        sl = sub[sub.level == lv]
        best = sl.loc[sl.auc.idxmax()]
        print(f"  {lv}: sigma={best.sigma} AUC={best.auc:.4f}")
    # Mean best
    mean_aucs = {s: sub[sub.sigma == s]["auc"].mean() for s in sub.sigma.unique()}
    best_s = max(mean_aucs, key=mean_aucs.get)
    print(f"  MEAN best: sigma={best_s} AUC={mean_aucs[best_s]:.4f}")

df.to_csv(f"{OUT_BASE}/phase2_sigma_all.csv", index=False)
print("\nDONE")
