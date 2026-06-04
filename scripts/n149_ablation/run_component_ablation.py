"""Component ablation: remove spatial/opp/hot from N149 v2.2, all datasets. 12-threaded."""
import subprocess, os, time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

PY = r"D:/software/Anaconda_envs/envs/myEVS/python.exe"
THR = "0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,8"
OUT = r"D:/hjx_workspace/scientific_reserach/projects/myEVS/data/Hyperparameter ablation_study"

DATASETS = [
    # Drive-ED24
    ("drive", "1hz",  r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_1hz_ed24_withlabel/driving_noise_1hz_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_1hz_ed24_withlabel/driving_noise_1hz_labeled.npy",
     346, 260, 2, 32000, 1.75, 0.05),
    ("drive", "3hz",  r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_3hz_ed24_withlabel/driving_noise_3hz_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_3hz_ed24_withlabel/driving_noise_3hz_labeled.npy",
     346, 260, 2, 32000, 1.75, 0.05),
    ("drive", "5hz",  r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_5hz_ed24_withlabel/driving_noise_5hz_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_5hz_ed24_withlabel/driving_noise_5hz_labeled.npy",
     346, 260, 2, 32000, 1.75, 0.05),
    ("drive", "7hz",  r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_7hz_ed24_withlabel/driving_noise_7hz_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_7hz_ed24_withlabel/driving_noise_7hz_labeled.npy",
     346, 260, 2, 32000, 1.75, 0.05),
    ("drive", "10hz", r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_10hz_ed24_withlabel/driving_noise_10hz_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_10hz_ed24_withlabel/driving_noise_10hz_labeled.npy",
     346, 260, 2, 32000, 1.75, 0.05),
    # ED24 Pedestrian
    ("ped", "1.8", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_1.8_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_1.8.npy",
     346, 260, 5, 256000, 2.75, 0.25),
    ("ped", "2.1", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_2.1_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_2.1.npy",
     346, 260, 5, 256000, 2.75, 0.25),
    ("ped", "2.5", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_2.5_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_2.5.npy",
     346, 260, 5, 256000, 2.75, 0.25),
    ("ped", "3.3", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3.npy",
     346, 260, 5, 256000, 2.75, 0.25),
    # ED24 Bicycle
    ("bike", "1.8", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_1.8_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_1.8.npy",
     346, 260, 5, 256000, 2.75, 0.25),
    ("bike", "2.1", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_2.1_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_2.1.npy",
     346, 260, 5, 256000, 2.75, 0.25),
    ("bike", "2.5", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_2.5_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_2.5.npy",
     346, 260, 5, 256000, 2.75, 0.25),
    ("bike", "3.3", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_3.3_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_3.3.npy",
     346, 260, 5, 256000, 2.75, 0.25),
]

# DVSCLEAN 10 sub-scenes
for scene in ["MAH00444","MAH00446","MAH00447","MAH00448","MAH00449"]:
    for ratio in ["ratio50","ratio100"]:
        lv = f"{scene}_{ratio}"
        DATASETS.append(("dvsclean", lv,
            f"D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy/{scene}/{ratio}/{scene}_{ratio}_signal_only.npy",
            f"D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy/{scene}/{ratio}/{scene}_{ratio}_labeled.npy",
            1280, 720, 5, 128000, 2.5, 0.25))

# LED 10 scenes
for s in ["scene_100","scene_1004","scene_1018","scene_1028","scene_1032",
          "scene_1033","scene_1034","scene_1043","scene_1045","scene_1046"]:
    DATASETS.append(("led", s,
        f"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/{s}/slices_00031_00040_100ms/{s}_100ms_signal_only.npy",
        f"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/{s}/slices_00031_00040_100ms/{s}_100ms_labeled.npy",
        1280, 720, 2, 8000, 2.0, 1.0))

VARIANTS = {
    "baseline":   {},
    "no_spatial": {"MYEVS_N149_NO_SPATIAL": "1"},
    "no_opp":     {"MYEVS_N149_NO_OPP": "1"},
    "no_hot":     {"MYEVS_N149_NO_HOT": "1"},
}

def run_one(ds, lv, clean, noisy, w, h, r, tau, sigma, alpha, vn, env_extra):
    tag = f"comp_ab_{ds}_{lv}_{vn}"
    csv = f"{OUT}/{ds}/_comp_ab/{tag}.csv"
    os.makedirs(os.path.dirname(csv), exist_ok=True)
    if os.path.exists(csv): os.remove(csv)
    env = os.environ.copy()
    env["MYEVS_N149_HOT_BITS"] = "16"
    env["MYEVS_N149_SIGMA"] = str(sigma)
    env["MYEVS_N149_ALPHA_FIXED"] = str(alpha)
    for k, v in env_extra.items(): env[k] = v
    cmd = [PY, '-m', 'myevs.cli', 'roc',
           '--clean', clean, '--noisy', noisy,
           '--assume', 'npy', '--width', str(w), '--height', str(h),
           '--tick-ns', '1000', '--engine', 'cpp', '--method', 'n149',
           '--radius-px', str(r), '--time-us', str(tau),
           '--param', 'min-neighbors', '--values', THR,
           '--match-us', '0', '--match-bin-radius', '0',
           '--tag', tag, '--out-csv', csv, '--append']
    subprocess.run(cmd, check=True, timeout=600, env=env, capture_output=True)
    df = pd.read_csv(csv)
    return ds, lv, vn, float(df.loc[df["auc"].idxmax()]["auc"])

tasks = [(ds, lv, clean, noisy, w, h, r, tau, sigma, alpha, vn, cfg)
         for ds, lv, clean, noisy, w, h, r, tau, sigma, alpha in DATASETS
         for vn, cfg in VARIANTS.items()]
total = len(tasks)
print(f"Component ablation: {total} tasks ({len(DATASETS)} levels x {len(VARIANTS)} variants)")
t0 = time.time()

results = []
with ThreadPoolExecutor(max_workers=12) as ex:
    futures = {ex.submit(run_one, *t): t for t in tasks}
    done = 0
    for f in as_completed(futures):
        ds, lv, vn, auc = f.result()
        results.append((ds, lv, vn, auc))
        done += 1
        rate = done/(time.time()-t0) if time.time()>t0 else 0
        eta = (total-done)/rate/60 if rate>0 else 0
        print(f"[{done}/{total} | {rate:.1f}t/s | ETA {eta:.0f}m] {ds}/{lv} {vn} AUC={auc:.4f}", flush=True)

df = pd.DataFrame(results, columns=["dataset","level","variant","auc"])
df.to_csv(f"{OUT}/component_ablation.csv", index=False)

# Summary: mean per dataset
print("\n" + "=" * 70)
print("COMPONENT ABLATION: DATASET MEANS")
print("=" * 70)
for ds in ["drive","ped","bike","dvsclean","led"]:
    sub = df[df.dataset==ds]
    base_mean = sub[sub.variant=="baseline"]["auc"].mean()
    print(f"\n{ds} (n={len(sub['level'].unique())}):")
    for vn in ["baseline","no_spatial","no_opp","no_hot"]:
        mean_auc = sub[sub.variant==vn]["auc"].mean()
        delta = mean_auc - base_mean
        print(f"  {vn}: mean={mean_auc:.4f}  Δ={delta:+.4f}")

# Per-level detail
print("\n" + "=" * 70)
print("PER-LEVEL DETAIL")
print("=" * 70)
for ds in ["drive","ped","bike","dvsclean","led"]:
    sub = df[df.dataset==ds]
    base_df = sub[sub.variant=="baseline"]
    print(f"\n{ds}:")
    for _, brow in base_df.iterrows():
        lv = brow["level"]
        base = brow["auc"]
        parts = [f"base={base:.4f}"]
        for vn in ["no_spatial","no_opp","no_hot"]:
            a = sub[(sub.level==lv)&(sub.variant==vn)]["auc"].values[0]
            parts.append(f"{vn}={a:.4f}({a-base:+.4f})")
        print(f"  {lv}: {' | '.join(parts)}")

print("\nDONE")
