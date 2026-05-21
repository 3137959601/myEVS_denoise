"""Run all datasets with finalized optimal hyperparameters. Multi-threaded."""
import subprocess, os, time, json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

PY = r"D:/software/Anaconda_envs/envs/myEVS/python.exe"
THR = "0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,8"
OUT_BASE = r"D:/hjx_workspace/scientific_reserach/projects/myEVS/data/Hyperparameter ablation_study"

CONFIGS = {
    "drive": {
        "r": 2, "tau": 32000, "sigma": 1.75, "alpha": "0",
        "w": 346, "h": 260,
        "levels": {
            "1hz":  ("driving_noise_1hz_ed24_withlabel",),
            "3hz":  ("driving_noise_3hz_ed24_withlabel",),
            "5hz":  ("driving_noise_5hz_ed24_withlabel",),
            "7hz":  ("driving_noise_7hz_ed24_withlabel",),
            "10hz": ("driving_noise_10hz_ed24_withlabel",),
        },
        "base": r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24",
    },
    "ped": {
        "r": 5, "tau": 256000, "sigma": 2.75, "alpha": "0.25",
        "w": 346, "h": 260,
        "levels": {
            "1.8": ("Pedestrain_06_1.8",),
            "2.1": ("Pedestrain_06_2.1",),
            "2.5": ("Pedestrain_06_2.5",),
            "3.3": ("Pedestrain_06_3.3",),
        },
        "base": r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06",
    },
    "bike": {
        "r": 5, "tau": 256000, "sigma": 2.75, "alpha": "0.25",
        "w": 346, "h": 260,
        "levels": {
            "1.8": ("Bicycle_02_1.8",),
            "2.1": ("Bicycle_02_2.1",),
            "2.5": ("Bicycle_02_2.5",),
            "3.3": ("Bicycle_02_3.3",),
        },
        "base": r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02",
    },
    "dvsclean": {
        "r": 5, "tau": 128000, "sigma": 2.5, "alpha": "0.25",
        "w": 1280, "h": 720,
        "levels": {},
        "base": r"D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy",
    },
    "led": {
        "r": 2, "tau": 8000, "sigma": 2.0, "alpha": "1.0",
        "w": 1280, "h": 720,
        "levels": {},
        "base": r"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy",
    },
}

# Build DVSCLEAN levels
for scene in ["MAH00444","MAH00446","MAH00447","MAH00448","MAH00449"]:
    for ratio in ["ratio50","ratio100"]:
        k = f"{scene}_{ratio}"
        CONFIGS["dvsclean"]["levels"][k] = (scene, ratio)

# Build LED levels
for s in ["scene_100","scene_1004","scene_1018","scene_1028","scene_1032","scene_1033","scene_1034","scene_1043","scene_1045","scene_1046"]:
    CONFIGS["led"]["levels"][s] = (s,)

def get_paths(ds_key, lv, parts):
    if ds_key == "drive":
        d = parts[0]
        return (f"{CONFIGS[ds_key]['base']}/{d}/driving_noise_{lv}_signal_only.npy",
                f"{CONFIGS[ds_key]['base']}/{d}/driving_noise_{lv}_labeled.npy")
    elif ds_key in ("ped", "bike"):
        fname = parts[0]
        return (f"{CONFIGS[ds_key]['base']}/{fname}_signal_only.npy",
                f"{CONFIGS[ds_key]['base']}/{fname}.npy")
    elif ds_key == "dvsclean":
        scene, ratio = parts
        return (f"{CONFIGS[ds_key]['base']}/{scene}/{ratio}/{scene}_{ratio}_signal_only.npy",
                f"{CONFIGS[ds_key]['base']}/{scene}/{ratio}/{scene}_{ratio}_labeled.npy")
    elif ds_key == "led":
        s = parts[0]
        return (f"{CONFIGS[ds_key]['base']}/{s}/slices_00031_00040_100ms/{s}_100ms_signal_only.npy",
                f"{CONFIGS[ds_key]['base']}/{s}/slices_00031_00040_100ms/{s}_100ms_labeled.npy")

def run_one(ds_key, lv, cfg):
    clean, noisy = get_paths(ds_key, lv, cfg["levels"][lv])
    env = os.environ.copy()
    env["MYEVS_N149_HOT_BITS"] = "16"
    env["MYEVS_N149_SIGMA"] = str(cfg["sigma"])
    if cfg["alpha"] != "0.25":
        env["MYEVS_N149_ALPHA_FIXED"] = cfg["alpha"]
    out_dir = f"{OUT_BASE}/{ds_key}"
    tag = f"final_{ds_key}_{lv}"
    csv = f"{out_dir}/{tag}.csv"
    cmd = [PY,'-m','myevs.cli','roc','--clean',clean,'--noisy',noisy,'--assume','npy',
           '--width',str(cfg["w"]),'--height',str(cfg["h"]),'--tick-ns','1000',
           '--engine','cpp','--method','n149',
           '--radius-px',str(cfg["r"]),'--time-us',str(cfg["tau"]),
           '--param','min-neighbors','--values',THR,
           '--match-us','0','--match-bin-radius','0',
           '--tag',tag,'--out-csv',csv,'--append']
    subprocess.run(cmd, check=True, timeout=900, env=env, capture_output=True)
    df = pd.read_csv(csv)
    best = df.loc[df["auc"].idxmax()]
    best_f1 = df.loc[df["f1"].idxmax()]
    return ds_key, lv, float(best["auc"]), float(best_f1["f1"])

all_tasks = []
for key, cfg in CONFIGS.items():
    for lv in cfg["levels"]:
        all_tasks.append((key, lv, cfg))

total = len(all_tasks)
print(f"Final run: {total} tasks across {len(CONFIGS)} datasets")
t0 = time.time()

results = []
with ThreadPoolExecutor(max_workers=12) as ex:
    futures = {ex.submit(run_one, *t): t for t in all_tasks}
    done = 0
    for f in as_completed(futures):
        key, lv, auc, f1 = f.result()
        results.append((key, lv, auc, f1))
        done += 1
        rate = done / (time.time() - t0) if time.time() > t0 else 0
        eta = (total - done) / rate / 60 if rate > 0 else 0
        print(f"[{done}/{total} | {rate:.1f}t/s | ETA {eta:.0f}m] {key}/{lv} AUC={auc:.4f} F1={f1:.4f}", flush=True)

df = pd.DataFrame(results, columns=["dataset","level","auc","f1"])
df.to_csv(f"{OUT_BASE}/final_all.csv", index=False)

# Per-dataset summary
for key in CONFIGS:
    sub = df[df.dataset == key]
    print(f"\n{key}: {len(sub)} levels, mean AUC={sub.auc.mean():.4f}")
    for _, r in sub.iterrows():
        print(f"  {r.level}: AUC={r.auc:.4f} F1={r.f1:.4f}")

print("\nDONE")
