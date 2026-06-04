"""PFD r=1 tau-sweep ROC on all datasets (paper methodology). 12-threaded."""
import subprocess, os, time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

PY = r"D:/software/Anaconda_envs/envs/myEVS/python.exe"
OUT = r"D:/hjx_workspace/scientific_reserach/projects/myEVS/data/missing_alg"

# BAF tau ranges per dataset (microseconds)
TAU_RANGES = {
    "drive":    "1000,2000,4000,8000,16000,32000",
    "ped":      "16000,32000,64000,128000,256000,512000",
    "dvsclean": "1000,2000,4000,8000,16000,32000,64000",
    "led":      "1000,2000,4000,8000,16000,32000",
}

# m={1,2}, k={1,2} grid
M_LIST = [1, 2]
K_LIST = [1, 2]

# ======== Driving 5 levels ========
DRIVE_LEVELS = {
    "1hz":  (r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_1hz_ed24_withlabel/driving_noise_1hz_signal_only.npy",
             r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_1hz_ed24_withlabel/driving_noise_1hz_labeled.npy", 346, 260),
    "3hz":  (r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_3hz_ed24_withlabel/driving_noise_3hz_signal_only.npy",
             r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_3hz_ed24_withlabel/driving_noise_3hz_labeled.npy", 346, 260),
    "5hz":  (r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_5hz_ed24_withlabel/driving_noise_5hz_signal_only.npy",
             r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_5hz_ed24_withlabel/driving_noise_5hz_labeled.npy", 346, 260),
    "7hz":  (r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_7hz_ed24_withlabel/driving_noise_7hz_signal_only.npy",
             r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_7hz_ed24_withlabel/driving_noise_7hz_labeled.npy", 346, 260),
    "10hz": (r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_10hz_ed24_withlabel/driving_noise_10hz_signal_only.npy",
             r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_10hz_ed24_withlabel/driving_noise_10hz_labeled.npy", 346, 260),
}

# ======== Ped 4 levels ========
PED_LEVELS = {
    "1.8": (r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_1.8_signal_only.npy",
            r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_1.8.npy", 346, 260),
    "2.1": (r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_2.1_signal_only.npy",
            r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_2.1.npy", 346, 260),
    "2.5": (r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_2.5_signal_only.npy",
            r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_2.5.npy", 346, 260),
    "3.3": (r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3_signal_only.npy",
            r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3.npy", 346, 260),
}

# ======== DVSCLEAN 10 scenes ========
DVSCLEAN_SCENES = []
for scene in ["MAH00444","MAH00446","MAH00447","MAH00448","MAH00449"]:
    for ratio in ["ratio50","ratio100"]:
        lv = f"{scene}_{ratio}"
        DVSCLEAN_SCENES.append((lv,
            f"D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy/{scene}/{ratio}/{scene}_{ratio}_signal_only.npy",
            f"D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy/{scene}/{ratio}/{scene}_{ratio}_labeled.npy",
            1280, 720))

# ======== LED 10 scenes ========
LED_SCENES = []
for s in ["scene_100","scene_1004","scene_1018","scene_1028","scene_1032",
          "scene_1033","scene_1034","scene_1043","scene_1045","scene_1046"]:
    LED_SCENES.append((s,
        f"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/{s}/slices_00031_00040_100ms/{s}_100ms_signal_only.npy",
        f"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/{s}/slices_00031_00040_100ms/{s}_100ms_labeled.npy",
        1280, 720))

def run_one(ds, lv, clean, noisy, w, h, tau_list, m, k):
    csv = f"{OUT}/{ds}/_pfd_tau/pfd_{lv}_m{m}_k{k}.csv"
    os.makedirs(os.path.dirname(csv), exist_ok=True)
    if os.path.exists(csv): os.remove(csv)
    cmd = [PY, '-m', 'myevs.cli', 'roc',
           '--clean', clean, '--noisy', noisy,
           '--assume', 'npy', '--width', str(w), '--height', str(h),
           '--tick-ns', '1000', '--engine', 'cpp', '--method', 'pfd',
           '--radius-px', '1', '--min-neighbors', str(k),
           '--refractory-us', str(m), '--pfd-mode', 'a',
           '--param', 'time-us', '--values', tau_list,
           '--match-us', '0', '--match-bin-radius', '0',
           '--tag', f'pfd_{ds}_{lv}_m{m}_k{k}', '--out-csv', csv, '--append']
    env = os.environ.copy()
    subprocess.run(cmd, check=True, timeout=600, env=env, capture_output=True)
    df = pd.read_csv(csv)
    return ds, lv, m, k, float(df["auc"].iloc[0])

tasks = []
for lv, (clean, noisy, w, h) in DRIVE_LEVELS.items():
    for m in M_LIST:
        for k in K_LIST:
            tasks.append(("drive", lv, clean, noisy, w, h, TAU_RANGES["drive"], m, k))
for lv, (clean, noisy, w, h) in PED_LEVELS.items():
    for m in M_LIST:
        for k in K_LIST:
            tasks.append(("ped", lv, clean, noisy, w, h, TAU_RANGES["ped"], m, k))
for lv, clean, noisy, w, h in DVSCLEAN_SCENES:
    for m in M_LIST:
        for k in K_LIST:
            tasks.append(("dvsclean", lv, clean, noisy, w, h, TAU_RANGES["dvsclean"], m, k))
for lv, clean, noisy, w, h in LED_SCENES:
    for m in M_LIST:
        for k in K_LIST:
            tasks.append(("led", lv, clean, noisy, w, h, TAU_RANGES["led"], m, k))

total = len(tasks)
print(f"PFD tau-sweep (r=1): {total} tasks ({len(M_LIST)}m x {len(K_LIST)}k per level)")
t0 = time.time()

results = []
with ThreadPoolExecutor(max_workers=12) as ex:
    futures = {ex.submit(run_one, *t): t for t in tasks}
    done = 0
    for f in as_completed(futures):
        ds, lv, m, k, auc = f.result()
        results.append((ds, lv, m, k, auc))
        done += 1
        rate = done/(time.time()-t0) if time.time()>t0 else 0
        eta = (total-done)/rate/60 if rate>0 else 0
        print(f"[{done}/{total} | {rate:.1f}t/s | ETA {eta:.0f}m] {ds}/{lv} m={m} k={k} AUC={auc:.4f}", flush=True)

df = pd.DataFrame(results, columns=["dataset","level","m","k","auc"])
df.to_csv(f"{OUT}/pfd_tau_sweep_all.csv", index=False)

# Summary: per dataset, per level best
print("\n=== PFD r=1 TAU-SWEEP SUMMARY ===")
for ds in ["drive","ped","dvsclean","led"]:
    sub = df[df.dataset==ds]
    print(f"\n{ds}:")
    for lv in sorted(sub["level"].unique()):
        sl = sub[sub.level==lv]
        best = sl.loc[sl.auc.idxmax()]
        print(f"  {lv}: best AUC={best.auc:.4f} m={int(best.m)} k={int(best.k)}")
        # All combos
        for _, row in sl.iterrows():
            marker = " ***" if (row.m==best.m and row.k==best.k) else ""
            print(f"    m={int(row.m)} k={int(row.k)}: AUC={row.auc:.4f}{marker}")
    # Mean
    best_per_lv = sub.groupby("level")["auc"].max()
    print(f"  MEAN: {best_per_lv.mean():.4f}")

print("\nDONE")
