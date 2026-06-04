"""Supplement: add no_polarity variant to component ablation on all datasets. 12-threaded."""
import subprocess, os, time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

PY = r"D:/software/Anaconda_envs/envs/myEVS/python.exe"
THR = "0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,8"
OUT = r"D:/hjx_workspace/scientific_reserach/projects/myEVS/data/Hyperparameter ablation_study"

DATASETS = [
    # Drive-ED24
    ("drive", "1hz",  r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_1hz_ed24_withlabel/driving_noise_1hz_signal_only.npy", r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_1hz_ed24_withlabel/driving_noise_1hz_labeled.npy", 346,260,2,32000,1.75,0.05),
    ("drive", "3hz",  r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_3hz_ed24_withlabel/driving_noise_3hz_signal_only.npy", r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_3hz_ed24_withlabel/driving_noise_3hz_labeled.npy", 346,260,2,32000,1.75,0.05),
    ("drive", "5hz",  r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_5hz_ed24_withlabel/driving_noise_5hz_signal_only.npy", r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_5hz_ed24_withlabel/driving_noise_5hz_labeled.npy", 346,260,2,32000,1.75,0.05),
    ("drive", "7hz",  r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_7hz_ed24_withlabel/driving_noise_7hz_signal_only.npy", r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_7hz_ed24_withlabel/driving_noise_7hz_labeled.npy", 346,260,2,32000,1.75,0.05),
    ("drive", "10hz", r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_10hz_ed24_withlabel/driving_noise_10hz_signal_only.npy", r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_10hz_ed24_withlabel/driving_noise_10hz_labeled.npy", 346,260,2,32000,1.75,0.05),
    # ED24 Pedestrian
    ("ped", "1.8", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_1.8_signal_only.npy", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_1.8.npy", 346,260,5,256000,2.75,0.25),
    ("ped", "2.1", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_2.1_signal_only.npy", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_2.1.npy", 346,260,5,256000,2.75,0.25),
    ("ped", "2.5", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_2.5_signal_only.npy", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_2.5.npy", 346,260,5,256000,2.75,0.25),
    ("ped", "3.3", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3_signal_only.npy", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3.npy", 346,260,5,256000,2.75,0.25),
    # ED24 Bicycle
    ("bike", "1.8", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_1.8_signal_only.npy", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_1.8.npy", 346,260,5,256000,2.75,0.25),
    ("bike", "2.1", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_2.1_signal_only.npy", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_2.1.npy", 346,260,5,256000,2.75,0.25),
    ("bike", "2.5", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_2.5_signal_only.npy", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_2.5.npy", 346,260,5,256000,2.75,0.25),
    ("bike", "3.3", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_3.3_signal_only.npy", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_3.3.npy", 346,260,5,256000,2.75,0.25),
]
# DVSCLEAN
for scene in ["MAH00444","MAH00446","MAH00447","MAH00448","MAH00449"]:
    for ratio in ["ratio50","ratio100"]:
        DATASETS.append(("dvsclean", f"{scene}_{ratio}",
            f"D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy/{scene}/{ratio}/{scene}_{ratio}_signal_only.npy",
            f"D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy/{scene}/{ratio}/{scene}_{ratio}_labeled.npy",
            1280,720,5,128000,2.5,0.25))
# LED
for s in ["scene_100","scene_1004","scene_1018","scene_1028","scene_1032","scene_1033","scene_1034","scene_1043","scene_1045","scene_1046"]:
    DATASETS.append(("led", s,
        f"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/{s}/slices_00031_00040_100ms/{s}_100ms_signal_only.npy",
        f"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/{s}/slices_00031_00040_100ms/{s}_100ms_labeled.npy",
        1280,720,2,8000,2.0,1.0))

def run_one(ds, lv, clean, noisy, w, h, r, tau, sigma, alpha):
    csv = f"{OUT}/{ds}/_comp_ab/comp_ab_{ds}_{lv}_no_polarity.csv"
    os.makedirs(os.path.dirname(csv), exist_ok=True)
    if os.path.exists(csv): os.remove(csv)
    env = os.environ.copy()
    env["MYEVS_N149_HOT_BITS"] = "16"
    env["MYEVS_N149_SIGMA"] = str(sigma)
    env["MYEVS_N149_ALPHA_FIXED"] = str(alpha)
    env["MYEVS_N149_BLIND"] = "1"
    cmd = [PY, '-m', 'myevs.cli', 'roc',
           '--clean', clean, '--noisy', noisy,
           '--assume', 'npy', '--width', str(w), '--height', str(h),
           '--tick-ns', '1000', '--engine', 'cpp', '--method', 'n149',
           '--radius-px', str(r), '--time-us', str(tau),
           '--param', 'min-neighbors', '--values', THR,
           '--match-us', '0', '--match-bin-radius', '0',
           '--tag', f'comp_ab_{ds}_{lv}_no_pol', '--out-csv', csv, '--append']
    r = subprocess.run(cmd, timeout=600, env=env, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"CLI failed for {ds}/{lv}: {r.stderr.decode()[:300]}")
    df = pd.read_csv(csv)
    return ds, lv, float(df["auc"].iloc[0])

# Pre-create all dirs
for ds in set(d[0] for d in DATASETS):
    os.makedirs(f"{OUT}/{ds}/_comp_ab", exist_ok=True)

tasks = [(ds, lv, clean, noisy, w, h, r, tau, sigma, alpha) for ds, lv, clean, noisy, w, h, r, tau, sigma, alpha in DATASETS]
total = len(tasks)
print(f"no_polarity supplement: {total} tasks")
t0 = time.time()

results = []
with ThreadPoolExecutor(max_workers=12) as ex:
    futures = {ex.submit(run_one, *t): t for t in tasks}
    done = 0
    for f in as_completed(futures):
        ds, lv, auc = f.result()
        results.append((ds, lv, auc))
        done += 1
        rate = done/(time.time()-t0) if time.time()>t0 else 0
        eta = (total-done)/rate/60 if rate>0 else 0
        print(f"[{done}/{total} | {rate:.1f}t/s | ETA {eta:.0f}m] {ds}/{lv} no_polarity AUC={auc:.4f}", flush=True)

df = pd.DataFrame(results, columns=["dataset","level","auc"])
df.to_csv(f"{OUT}/component_ablation_no_polarity.csv", index=False)

# Load existing baseline data to compute delta
base_df = pd.read_csv(f"{OUT}/component_ablation.csv")
base_df = base_df[base_df.variant=="baseline"][["dataset","level","auc"]].rename(columns={"auc":"baseline_auc"})
merged = df.merge(base_df, on=["dataset","level"])
merged["delta"] = merged["auc"] - merged["baseline_auc"]

print("\n=== no_polarity MEAN per dataset ===")
for ds in ["drive","ped","bike","dvsclean","led"]:
    sub = merged[merged.dataset==ds]
    print(f"  {ds}: mean AUC={sub.auc.mean():.4f}  Δ={sub.delta.mean():+.4f}")

print("\nDONE")
