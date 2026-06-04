"""no_polarity component ablation: MYEVS_N149_BLIND=1, all 33 levels, 8 threads."""
import subprocess, os, time, traceback
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

PY = r"D:/software/Anaconda_envs/envs/myEVS/python.exe"
THR = "0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,8"
OUT = r"D:/hjx_workspace/scientific_reserach/projects/myEVS/data/Hyperparameter ablation_study"

def make_entry(ds, lv, clean, noisy, w, h, r, tau, sigma, alpha):
    return (ds, lv, clean, noisy, w, h, r, tau, sigma, alpha)

DATASETS = [
    make_entry("drive","1hz", r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_1hz_ed24_withlabel/driving_noise_1hz_signal_only.npy", r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_1hz_ed24_withlabel/driving_noise_1hz_labeled.npy", 346,260,2,32000,1.75,0.05),
    make_entry("drive","3hz", r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_3hz_ed24_withlabel/driving_noise_3hz_signal_only.npy", r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_3hz_ed24_withlabel/driving_noise_3hz_labeled.npy", 346,260,2,32000,1.75,0.05),
    make_entry("drive","5hz", r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_5hz_ed24_withlabel/driving_noise_5hz_signal_only.npy", r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_5hz_ed24_withlabel/driving_noise_5hz_labeled.npy", 346,260,2,32000,1.75,0.05),
    make_entry("drive","7hz", r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_7hz_ed24_withlabel/driving_noise_7hz_signal_only.npy", r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_7hz_ed24_withlabel/driving_noise_7hz_labeled.npy", 346,260,2,32000,1.75,0.05),
    make_entry("drive","10hz",r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_10hz_ed24_withlabel/driving_noise_10hz_signal_only.npy", r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_10hz_ed24_withlabel/driving_noise_10hz_labeled.npy", 346,260,2,32000,1.75,0.05),
    make_entry("ped","1.8",r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_1.8_signal_only.npy", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_1.8.npy", 346,260,5,256000,2.75,0.25),
    make_entry("ped","2.1",r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_2.1_signal_only.npy", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_2.1.npy", 346,260,5,256000,2.75,0.25),
    make_entry("ped","2.5",r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_2.5_signal_only.npy", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_2.5.npy", 346,260,5,256000,2.75,0.25),
    make_entry("ped","3.3",r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3_signal_only.npy", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3.npy", 346,260,5,256000,2.75,0.25),
    make_entry("bike","1.8",r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_1.8_signal_only.npy", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_1.8.npy", 346,260,5,256000,2.75,0.25),
    make_entry("bike","2.1",r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_2.1_signal_only.npy", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_2.1.npy", 346,260,5,256000,2.75,0.25),
    make_entry("bike","2.5",r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_2.5_signal_only.npy", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_2.5.npy", 346,260,5,256000,2.75,0.25),
    make_entry("bike","3.3",r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_3.3_signal_only.npy", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_3.3.npy", 346,260,5,256000,2.75,0.25),
]
for scene in ["MAH00444","MAH00446","MAH00447","MAH00448","MAH00449"]:
    for ratio in ["ratio50","ratio100"]:
        lv = f"{scene}_{ratio}"
        DATASETS.append(make_entry("dvsclean",lv, f"D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy/{scene}/{ratio}/{scene}_{ratio}_signal_only.npy", f"D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy/{scene}/{ratio}/{scene}_{ratio}_labeled.npy", 1280,720,5,128000,2.5,0.25))
for s in ["scene_100","scene_1004","scene_1018","scene_1028","scene_1032","scene_1033","scene_1034","scene_1043","scene_1045","scene_1046"]:
    DATASETS.append(make_entry("led",s, f"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/{s}/slices_00031_00040_100ms/{s}_100ms_signal_only.npy", f"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/{s}/slices_00031_00040_100ms/{s}_100ms_labeled.npy", 1280,720,2,8000,2.0,1.0))

# Pre-create dirs
for ds in set(d[0] for d in DATASETS):
    os.makedirs(f"{OUT}/{ds}/_comp_ab", exist_ok=True)

def run_one(entry):
    ds, lv, clean, noisy, w, h, r, tau, sigma, alpha = entry
    csv = f"{OUT}/{ds}/_comp_ab/comp_ab_{ds}_{lv}_no_polarity.csv"
    try:
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
        r = subprocess.run(cmd, timeout=900, env=env, capture_output=True, text=True)
        if r.returncode != 0:
            return ds, lv, None, f"RC={r.returncode}: {r.stderr[:200]}"
        df = pd.read_csv(csv)
        return ds, lv, float(df["auc"].iloc[0]), None
    except Exception as e:
        return ds, lv, None, str(e)[:200]

total = len(DATASETS)
print(f"no_polarity (MYEVS_N149_BLIND=1): {total} tasks, 8 threads")
t0 = time.time()

results = []
errors = []
with ThreadPoolExecutor(max_workers=8) as ex:
    futures = {ex.submit(run_one, d): d for d in DATASETS}
    done = 0
    for f in as_completed(futures):
        ds, lv, auc, err = f.result()
        done += 1
        rate = done/(time.time()-t0) if time.time()>t0 else 0
        if err:
            errors.append((ds, lv, err))
            print(f"[{done}/{total} | {rate:.1f}t/s] {ds}/{lv} ERROR: {err[:100]}", flush=True)
        else:
            results.append((ds, lv, auc))
            print(f"[{done}/{total} | {rate:.1f}t/s] {ds}/{lv} AUC={auc:.4f}", flush=True)

if errors:
    print(f"\n{len(errors)} ERRORS:")
    for ds, lv, err in errors:
        print(f"  {ds}/{lv}: {err}")

if results:
    df = pd.DataFrame(results, columns=["dataset","level","auc"])
    df.to_csv(f"{OUT}/component_ablation_no_polarity.csv", index=False)

    # Load baseline
    try:
        base_df = pd.read_csv(f"{OUT}/component_ablation.csv")
        base_df = base_df[base_df.variant=="baseline"][["dataset","level","auc"]].rename(columns={"auc":"baseline_auc"})
        merged = df.merge(base_df, on=["dataset","level"])
        merged["delta"] = merged["auc"] - merged["baseline_auc"]

        print(f"\n=== no_polarity vs baseline mean per dataset ===")
        for ds in ["drive","ped","bike","dvsclean","led"]:
            sub = merged[merged.dataset==ds]
            if len(sub)==0: continue
            print(f"  {ds}: mean AUC={sub.auc.mean():.4f}  baseline={sub.baseline_auc.mean():.4f}  Δ={sub.delta.mean():+.4f}")
    except Exception as e:
        print(f"Could not load baseline: {e}")

print("\nDONE")
