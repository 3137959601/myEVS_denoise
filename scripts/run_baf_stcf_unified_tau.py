"""BAF & STCF_orig: unified tau [2,200]ms on ED24 Ped/Bike/DVSCLEAN + Drive STCF_orig. 12-threaded."""
import subprocess, os, time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

PY = r"D:/software/Anaconda_envs/envs/myEVS/python.exe"
OUT = r"D:/hjx_workspace/scientific_reserach/projects/myEVS/data/missing_alg"

# Unified tau: [2, 200]ms
TAU_UNIFIED = "2000,5000,10000,15000,20000,25000,30000,40000,50000,64000,80000,100000,128000,160000,200000"

# ============ BAF tasks ============
# ED24 Ped 4 levels
PED_BAF = []
for lv, npy_name in [("1.8","Pedestrain_06_1.8"),("2.1","Pedestrain_06_2.1"),("2.5","Pedestrain_06_2.5"),("3.3","Pedestrain_06_3.3")]:
    PED_BAF.append(("ped", lv,
        f"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/{npy_name}_signal_only.npy",
        f"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/{npy_name}.npy", 346, 260))

# ED24 Bike 4 levels
BIKE_BAF = []
for lv, npy_name in [("1.8","Bicycle_02_1.8"),("2.1","Bicycle_02_2.1"),("2.5","Bicycle_02_2.5"),("3.3","Bicycle_02_3.3")]:
    BIKE_BAF.append(("bike", lv,
        f"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/{npy_name}_signal_only.npy",
        f"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/{npy_name}.npy", 346, 260))

# DVSCLEAN 10 scenes
DVSCLEAN_BAF = []
for scene in ["MAH00444","MAH00446","MAH00447","MAH00448","MAH00449"]:
    for ratio in ["ratio50","ratio100"]:
        lv = f"{scene}_{ratio}"
        DVSCLEAN_BAF.append(("dvsclean", lv,
            f"D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy/{scene}/{ratio}/{scene}_{ratio}_signal_only.npy",
            f"D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy/{scene}/{ratio}/{scene}_{ratio}_labeled.npy", 1280, 720))

# ============ STCF_orig tasks (k=1..6, each sweeps tau) ============
# Drive 5 levels
DRIVE_STCF = []
for lv in ["1hz","3hz","5hz","7hz","10hz"]:
    DRIVE_STCF.append(("drive", lv,
        f"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_{lv}_ed24_withlabel/driving_noise_{lv}_signal_only.npy",
        f"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_{lv}_ed24_withlabel/driving_noise_{lv}_labeled.npy", 346, 260))

# ED24 Ped/Bike STCF_orig
PED_STCF = [(ds, lv, cl, ny, w, h) for ds, lv, cl, ny, w, h in PED_BAF]
BIKE_STCF = [(ds, lv, cl, ny, w, h) for ds, lv, cl, ny, w, h in BIKE_BAF]
DVSCLEAN_STCF = [(ds, lv, cl, ny, w, h) for ds, lv, cl, ny, w, h in DVSCLEAN_BAF]

STCF_K = [1, 2, 3, 4, 5, 6]

def run_baf(ds, lv, clean, noisy, w, h):
    csv = f"{OUT}/{ds}/_unified_tau/baf_{lv}_tau2-200.csv"
    os.makedirs(os.path.dirname(csv), exist_ok=True)
    if os.path.exists(csv): os.remove(csv)
    cmd = [PY, '-m', 'myevs.cli', 'roc',
           '--clean', clean, '--noisy', noisy,
           '--assume', 'npy', '--width', str(w), '--height', str(h),
           '--tick-ns', '1000', '--engine', 'cpp', '--method', 'baf',
           '--radius-px', '1', '--min-neighbors', '1',
           '--param', 'time-us', '--values', TAU_UNIFIED,
           '--match-us', '0', '--match-bin-radius', '0',
           '--tag', f'baf_{ds}_{lv}_t2-200', '--out-csv', csv, '--append']
    subprocess.run(cmd, check=True, timeout=600, env=os.environ.copy(), capture_output=True)
    df = pd.read_csv(csv)
    return ds, lv, "baf", float(df["auc"].iloc[0])

def run_stcf(ds, lv, clean, noisy, w, h, k):
    csv = f"{OUT}/{ds}/_unified_tau/stcf_orig_{lv}_k{k}_tau2-200.csv"
    os.makedirs(os.path.dirname(csv), exist_ok=True)
    if os.path.exists(csv): os.remove(csv)
    cmd = [PY, '-m', 'myevs.cli', 'roc',
           '--clean', clean, '--noisy', noisy,
           '--assume', 'npy', '--width', str(w), '--height', str(h),
           '--tick-ns', '1000', '--engine', 'cpp', '--method', 'stcf_original',
           '--radius-px', '1', '--min-neighbors', str(k),
           '--param', 'time-us', '--values', TAU_UNIFIED,
           '--match-us', '0', '--match-bin-radius', '0',
           '--tag', f'stcf_{ds}_{lv}_k{k}_t2-200', '--out-csv', csv, '--append']
    subprocess.run(cmd, check=True, timeout=600, env=os.environ.copy(), capture_output=True)
    df = pd.read_csv(csv)
    return ds, lv, k, float(df["auc"].iloc[0])

tasks = []

# BAF: Ped + Bike + DVSCLEAN (Drive uses cited values)
for t in PED_BAF + BIKE_BAF + DVSCLEAN_BAF:
    tasks.append(("baf", run_baf, t))

# STCF_orig: Drive + Ped + Bike + DVSCLEAN (all need re-run)
for t in DRIVE_STCF + PED_STCF + BIKE_STCF + DVSCLEAN_STCF:
    for k in STCF_K:
        tasks.append(("stcf", run_stcf, (*t, k)))

total = len(tasks)
baf_n = sum(1 for t in tasks if t[0]=="baf")
stcf_n = sum(1 for t in tasks if t[0]=="stcf")
print(f"Unified tau [2,200]ms: BAF={baf_n} + STCF_orig={stcf_n} = {total} tasks")
t0 = time.time()

results_baf = []
results_stcf = {}  # (ds, lv) -> [(k, auc)]
with ThreadPoolExecutor(max_workers=12) as ex:
    futures = {}
    for task_type, fn, args in tasks:
        futures[ex.submit(fn, *args)] = (task_type, args)
    done = 0
    for f in as_completed(futures):
        task_type, args = futures[f]
        if task_type == "baf":
            ds, lv, algo, auc = f.result()
            results_baf.append((ds, lv, auc))
            done += 1
        else:
            ds, lv, k, auc = f.result()
            key = (ds, lv)
            if key not in results_stcf: results_stcf[key] = []
            results_stcf[key].append((k, auc))
            done += 1
        rate = done/(time.time()-t0) if time.time()>t0 else 0
        eta = (total-done)/rate/60 if rate>0 else 0
        print(f"[{done}/{total} | {rate:.1f}t/s | ETA {eta:.0f}m] {task_type} {args[0]}/{args[1]}", flush=True)

# Summarize
print("\n=== BAF [2,200]ms RESULTS ===")
for ds in ["ped","bike","dvsclean"]:
    sub = [(lv, auc) for d, lv, auc in results_baf if d==ds]
    if not sub: continue
    print(f"\n{ds}:")
    old_vals = {
        "ped": {"1.8":0.8030,"2.1":0.7972,"2.5":0.7932,"3.3":0.7856},
        "bike": {"1.8":0.9489,"2.1":0.9352,"2.5":0.9149,"3.3":0.8960},
        "dvsclean": {"MAH00444_ratio50":0.9828,"MAH00444_ratio100":0.9833,
                     "MAH00446_ratio50":0.9719,"MAH00446_ratio100":0.9721,
                     "MAH00447_ratio50":0.9599,"MAH00447_ratio100":0.9600,
                     "MAH00448_ratio50":0.9489,"MAH00448_ratio100":0.9488,
                     "MAH00449_ratio50":0.9495,"MAH00449_ratio100":0.9489},
    }
    for lv, auc in sorted(sub):
        old = old_vals.get(ds, {}).get(lv, 0)
        print(f"  {lv}: AUC={auc:.4f} (old={old:.4f}, delta={auc-old:+.4f})")

print("\n=== STCF_orig [2,200]ms BEST PER K ===")
for ds_name, ds_levels in [("drive", DRIVE_STCF), ("ped", PED_STCF), ("bike", BIKE_STCF), ("dvsclean", DVSCLEAN_STCF)]:
    print(f"\n{ds_name}:")
    for _, lv, _, _, _, _ in ds_levels:
        key = (ds_name, lv)
        if key in results_stcf:
            best = max(results_stcf[key], key=lambda x: x[1])
            print(f"  {lv}: best k={best[0]} AUC={best[1]:.4f}")

print("\nDONE")
