"""Supplement alpha=2.0, 3.0 for Drive/Ped/Bike/DVSCLEAN at optimal r/tau/sigma."""
import subprocess, os, time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

PY = r"D:/software/Anaconda_envs/envs/myEVS/python.exe"
THR = "0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,8"
OUT = r"D:/hjx_workspace/scientific_reserach/projects/myEVS/data/Hyperparameter ablation_study"

# Optimal params per dataset (from README2 §11.1)
CONFIGS = {
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
            "10hz":(r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_10hz_ed24_withlabel/driving_noise_10hz_signal_only.npy",
                    r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_10hz_ed24_withlabel/driving_noise_10hz_labeled.npy"),
        },
        "w": 346, "h": 260, "r": 2, "tau": 32000, "sigma": 1.75,
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
        "w": 346, "h": 260, "r": 5, "tau": 256000, "sigma": 2.75,
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
        "w": 346, "h": 260, "r": 5, "tau": 256000, "sigma": 2.75,
    },
}

# DVSCLEAN: 10 scenes
DVSCLEAN_SCENES = ["MAH00444","MAH00446","MAH00447","MAH00448","MAH00449"]
DVSCLEAN_RATIOS = ["ratio50","ratio100"]

def run_one(ds, lv, clean, noisy, w, h, r, tau, sigma, alpha):
    tag = f"alpha_supp_{ds}_{lv}_a{str(alpha).replace('.','p')}"
    out_dir = f"{OUT}/{ds}"
    csv = f"{out_dir}/{tag}.csv"
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(csv):
        os.remove(csv)
    env = os.environ.copy()
    env["MYEVS_N149_HOT_BITS"] = "16"
    env["MYEVS_N149_SIGMA"] = str(sigma)
    env["MYEVS_N149_ALPHA_FIXED"] = str(alpha)
    cmd = [PY, '-m', 'myevs.cli', 'roc',
           '--clean', clean, '--noisy', noisy,
           '--assume', 'npy', '--width', str(w), '--height', str(h),
           '--tick-ns', '1000', '--engine', 'cpp', '--method', 'n149',
           '--radius-px', str(r), '--time-us', str(tau),
           '--param', 'min-neighbors', '--values', THR,
           '--match-us', '0', '--match-bin-radius', '0',
           '--tag', tag, '--out-csv', csv, '--append']
    subprocess.run(cmd, check=True, timeout=900, env=env, capture_output=True)
    df = pd.read_csv(csv)
    return ds, lv, alpha, float(df.loc[df["auc"].idxmax()]["auc"])

# Build tasks
tasks = []
for ds, cfg in CONFIGS.items():
    for lv, (clean, noisy) in cfg["levels"].items():
        for alpha in [2.0, 3.0]:
            tasks.append((ds, lv, clean, noisy, cfg["w"], cfg["h"],
                         cfg["r"], cfg["tau"], cfg["sigma"], alpha))

# DVSCLEAN
for scene in DVSCLEAN_SCENES:
    for ratio in DVSCLEAN_RATIOS:
        lv = f"{scene}_{ratio}"
        clean = f"D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy/{scene}/{ratio}/{scene}_{ratio}_signal_only.npy"
        noisy = f"D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy/{scene}/{ratio}/{scene}_{ratio}_labeled.npy"
        for alpha in [2.0, 3.0]:
            tasks.append(("dvsclean", lv, clean, noisy, 1280, 720, 5, 128000, 2.5, alpha))

total = len(tasks)
print(f"Alpha supplement: {total} tasks")
print(f"  Drive: 5lv x2 = 10, Ped: 4lv x2 = 8, Bike: 4lv x2 = 8, DVSCLEAN: 10sc x2 = 20")

t0 = time.time()
results = []
with ThreadPoolExecutor(max_workers=12) as ex:
    futures = {ex.submit(run_one, *t): t for t in tasks}
    done = 0
    for f in as_completed(futures):
        ds, lv, alpha, auc = f.result()
        results.append((ds, lv, alpha, auc))
        done += 1
        rate = done/(time.time()-t0) if time.time()>t0 else 0
        eta = (total-done)/rate/60 if rate>0 else 0
        print(f"[{done}/{total} | {rate:.1f}t/s | ETA {eta:.0f}m] {ds}/{lv} α={alpha} AUC={auc:.4f}", flush=True)

df = pd.DataFrame(results, columns=["dataset","level","alpha","auc"])
df.to_csv(f"{OUT}/_alpha_supplement.csv", index=False)

# Summary: mean per dataset per alpha
print("\n" + "=" * 70)
print("SUPPLEMENT RESULTS (mean per dataset)")
print("=" * 70)
for ds in ["drive", "ped", "bike", "dvsclean"]:
    sub = df[df.dataset == ds]
    for alpha in [2.0, 3.0]:
        mean_auc = sub[sub.alpha == alpha]["auc"].mean()
        print(f"  {ds} α={alpha}: mean AUC={mean_auc:.4f}")

# Also show per-level
print("\n" + "=" * 70)
print("PER-LEVEL DETAIL")
print("=" * 70)
for ds in ["drive", "ped", "bike", "dvsclean"]:
    sub = df[df.dataset == ds].sort_values(["level", "alpha"])
    for _, row in sub.iterrows():
        print(f"  {ds}/{row.level} α={row.alpha}: AUC={row.auc:.4f}")

print("\nDONE")
