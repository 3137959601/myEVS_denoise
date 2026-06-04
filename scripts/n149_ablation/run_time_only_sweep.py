"""time_only: re-sweep r & tau for optimal params. 8 threads."""
import subprocess, os, time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

PY = r"D:/software/Anaconda_envs/envs/myEVS/python.exe"
THR = "0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,8"
OUT = r"D:/hjx_workspace/scientific_reserach/projects/myEVS/data/Hyperparameter ablation_study"

# Dataset representatives with r & tau grids
# r range wider since spatial kernel is gone
DATASETS = [
    # Drive: one level first
    ("drive", "1hz", r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_1hz_ed24_withlabel/driving_noise_1hz_signal_only.npy", r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_1hz_ed24_withlabel/driving_noise_1hz_labeled.npy", 346,260, [1,2,3,5,7,9], [2000,4000,8000,16000,32000,64000,128000]),
    ("drive", "10hz",r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_10hz_ed24_withlabel/driving_noise_10hz_signal_only.npy", r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_10hz_ed24_withlabel/driving_noise_10hz_labeled.npy", 346,260, [1,2,3,5,7,9], [2000,4000,8000,16000,32000,64000,128000]),
    # Ped: light and heavy
    ("ped", "1.8", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_1.8_signal_only.npy", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_1.8.npy", 346,260, [1,2,3,5,7,9], [8000,16000,32000,64000,128000,256000,512000]),
    ("ped", "3.3", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3_signal_only.npy", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3.npy", 346,260, [1,2,3,5,7,9], [8000,16000,32000,64000,128000,256000,512000]),
    # Bike: light and heavy
    ("bike", "1.8", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_1.8_signal_only.npy", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_1.8.npy", 346,260, [1,2,3,5,7,9], [8000,16000,32000,64000,128000,256000,512000]),
    ("bike", "3.3", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_3.3_signal_only.npy", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_3.3.npy", 346,260, [1,2,3,5,7,9], [8000,16000,32000,64000,128000,256000,512000]),
]

os.makedirs(f"{OUT}/_time_only_sweep", exist_ok=True)

def run_one(ds, lv, clean, noisy, w, h, r, tau):
    csv = f"{OUT}/_time_only_sweep/to_{ds}_{lv}_r{r}_tau{tau}.csv"
    try:
        if os.path.exists(csv): os.remove(csv)
        env = os.environ.copy()
        env["MYEVS_N149_HOT_BITS"] = "16"
        env["MYEVS_N149_NO_SPATIAL"] = "1"
        env["MYEVS_N149_NO_HOT"] = "1"
        env["MYEVS_N149_BLIND"] = "1"
        cmd = [PY, '-m', 'myevs.cli', 'roc',
               '--clean', clean, '--noisy', noisy,
               '--assume', 'npy', '--width', str(w), '--height', str(h),
               '--tick-ns', '1000', '--engine', 'cpp', '--method', 'n149',
               '--radius-px', str(r), '--time-us', str(tau),
               '--param', 'min-neighbors', '--values', THR,
               '--match-us', '0', '--match-bin-radius', '0',
               '--tag', f'to_{ds}_{lv}_r{r}_tau{tau}', '--out-csv', csv, '--append']
        r2 = subprocess.run(cmd, timeout=600, env=env, capture_output=True, text=True)
        if r2.returncode != 0:
            return ds, lv, r, tau, None, f"RC={r2.returncode}"
        df = pd.read_csv(csv)
        return ds, lv, r, tau, float(df["auc"].iloc[0]), None
    except Exception as e:
        return ds, lv, r, tau, None, str(e)[:100]

tasks = []
for ds, lv, clean, noisy, w, h, r_list, tau_list in DATASETS:
    for r in r_list:
        for tau in tau_list:
            tasks.append((ds, lv, clean, noisy, w, h, r, tau))

total = len(tasks)
print(f"time_only sweep: {total} tasks, 8 threads")
t0 = time.time()
results, errors = [], []
with ThreadPoolExecutor(max_workers=8) as ex:
    futures = {ex.submit(run_one, *t): t for t in tasks}
    done = 0
    for f in as_completed(futures):
        ds, lv, r, tau, auc, err = f.result()
        done += 1; rate = done/(time.time()-t0) if time.time()>t0 else 0
        if err: errors.append((ds,lv,r,tau,err)); print(f"[{done}/{total}] {ds}/{lv} r={r} tau={tau} ERR", flush=True)
        else: results.append((ds,lv,r,tau,auc)); print(f"[{done}/{total}] {ds}/{lv} r={r} tau={tau//1000}ms AUC={auc:.4f}", flush=True)

if results:
    df = pd.DataFrame(results, columns=["dataset","level","r","tau","auc"])
    df.to_csv(f"{OUT}/time_only_sweep.csv", index=False)
    print("\n=== time_only BEST per level ===")
    for ds, lv, _, _, _, _, r_list, tau_list in DATASETS:
        sub = df[(df.dataset==ds)&(df.level==lv)]
        best = sub.loc[sub.auc.idxmax()]
        # Load baseline for comparison
        base_df = pd.read_csv(f"{OUT}/component_ablation.csv")
        base_auc = base_df[(base_df.dataset==ds)&(base_df.level==lv)&(base_df.variant=="baseline")]["auc"].values
        base = base_auc[0] if len(base_auc)>0 else 0

        # Old time_only (at baseline r,tau)
        old_to = pd.read_csv(f"{OUT}/component_ablation_time_only.csv")
        old_auc = old_to[(old_to.dataset==ds)&(old_to.level==lv)]["auc"].values
        old = old_auc[0] if len(old_auc)>0 else 0

        print(f"  {ds}/{lv}: best r={int(best.r)} tau={int(best.tau)//1000}ms AUC={best.auc:.4f} (baseline={base:.4f} Δ={best.auc-base:+.4f}) old_time_only={old:.4f}")
        # Check boundary
        r_max = max(r_list); tau_max = max(tau_list); tau_min = min(tau_list)
        boundary = []
        if int(best.r) == r_max: boundary.append("r_MAX")
        if int(best.tau) == tau_max: boundary.append("tau_MAX")
        if int(best.tau) == tau_min: boundary.append("tau_MIN")
        if boundary: print(f"    *** BOUNDARY: {boundary}")

print("\nDONE")
