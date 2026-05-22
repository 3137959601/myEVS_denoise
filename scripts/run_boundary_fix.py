"""Fix comparison algorithm threshold boundaries. 12-threaded."""
import subprocess, os, time, glob
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

PY = r"D:/software/Anaconda_envs/envs/myEVS/python.exe"
THR = "0,0.1,0.2,0.3,0.4,0.5,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,8,10,12,15"
OUT = r"D:/hjx_workspace/scientific_reserach/projects/myEVS/data/missing_alg"

# Scan all existing ROC CSVs for boundary hits
def check_boundary(csv_path):
    try:
        df = pd.read_csv(csv_path)
        if len(df) == 0: return None
        best = df.loc[df["auc"].idxmax()]
        val = str(best.get("value", ""))
        # Check if optimal threshold is at boundary (0 or 8 in original sweep)
        if val in ("0", "0.0") or val in ("8", "8.0"):
            return val
    except:
        pass
    return None

def rerun_with_extended_thr(csv_path, ds, lv, alg, clean, noisy, w, h, r, tau, method, extra_args=None):
    """Re-run with extended threshold range."""
    env = os.environ.copy()
    env["MYEVS_N149_HOT_BITS"] = "16"  # not used for non-N149 but harmless
    out_csv = csv_path.replace(".csv", "_ext.csv")
    cmd = [PY, '-m', 'myevs.cli', 'roc',
           '--clean', clean, '--noisy', noisy,
           '--assume', 'npy', '--width', str(w), '--height', str(h),
           '--tick-ns', '1000', '--engine', 'cpp', '--method', method,
           '--radius-px', str(r), '--time-us', str(tau),
           '--param', 'min-neighbors', '--values', THR,
           '--match-us', '0', '--match-bin-radius', '0',
           '--tag', f'bfix_{ds}_{lv}_{alg}', '--out-csv', out_csv, '--append']
    if extra_args: cmd.extend(extra_args)
    subprocess.run(cmd, check=True, timeout=900, env=env, capture_output=True)
    df = pd.read_csv(out_csv)
    best = df.loc[df["auc"].idxmax()]
    return float(best["auc"]), str(best.get("value", ""))

# Collect tasks: all non-N149 algorithms on non-Drive datasets
DATASETS = {
    "ped": [
        ("2.1", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_2.1_signal_only.npy",
         r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_2.1.npy", 346, 260),
    ],
    "bike": [
        ("2.1", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_2.1_signal_only.npy",
         r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_2.1.npy", 346, 260),
        ("3.3", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_3.3_signal_only.npy",
         r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_3.3.npy", 346, 260),
    ],
    "dvsclean": [],
    "led": [],
}

# Add DVSCLEAN scenes
for scene in ["MAH00444","MAH00446","MAH00447","MAH00448","MAH00449"]:
    for ratio in ["ratio50","ratio100"]:
        lv = f"{scene}_{ratio}"
        DATASETS["dvsclean"].append((lv,
            f"D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy/{scene}/{ratio}/{scene}_{ratio}_signal_only.npy",
            f"D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy/{scene}/{ratio}/{scene}_{ratio}_labeled.npy",
            1280, 720))

# Add LED scenes
for s in ["scene_100","scene_1004","scene_1018","scene_1028","scene_1032","scene_1033","scene_1034","scene_1043","scene_1045","scene_1046"]:
    DATASETS["led"].append((s,
        f"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/{s}/slices_00031_00040_100ms/{s}_100ms_signal_only.npy",
        f"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/{s}/slices_00031_00040_100ms/{s}_100ms_labeled.npy",
        1280, 720))

# Algorithm configs used in missing_alg
ALGS = {
    "baf": ("baf", 1, 8000, []),
    "stcf": ("stcf", 2, 32000, []),
    "ebf": ("ebf", 2, 32000, []),
    "knoise": ("knoise", 0, 8000, []),
    "ynoise": ("ynoise", 2, 16000, []),
    "ts": ("ts", 2, 32000, []),
    "pfd": ("pfd", 1, 8000, ["--refractory-us","2","--pfd-mode","a"]),
    "stcf_orig": ("stcf_original", 1, 16000, []),
}

# Build task list: check boundary, re-run if hit
tasks = []
for ds_key, levels in DATASETS.items():
    for lv, clean, noisy, w, h in levels:
        for alg_key, (method, r, tau, extra) in ALGS.items():
            csv_path = f"{OUT}/{ds_key}/miss_{ds_key}_{lv}_{alg_key}.csv"
            if os.path.exists(csv_path):
                boundary = check_boundary(csv_path)
                if boundary:
                    tasks.append((ds_key, lv, alg_key, clean, noisy, w, h, r, tau, method, csv_path, extra, boundary))
            # Also check evflow
            csv_path2 = f"{OUT}/{ds_key}/miss_{ds_key}_{lv}_evflow.csv"
            if os.path.exists(csv_path2):
                boundary = check_boundary(csv_path2)
                if boundary:
                    tasks.append((ds_key, lv, "evflow", clean, noisy, w, h, 2, 16000, "evflow", csv_path2, [], boundary))

print(f"Boundary fix: {len(tasks)} tasks with threshold boundary hits")
if len(tasks) == 0:
    print("No boundary hits found. All thresholds within optimal range.")
    exit(0)

t0 = time.time()
results = []
with ThreadPoolExecutor(max_workers=12) as ex:
    futures = {}
    for t in tasks:
        ds_key, lv, alg_key, clean, noisy, w, h, r, tau, method, csv_path, extra, boundary = t
        futures[ex.submit(rerun_with_extended_thr, csv_path, ds_key, lv, alg_key, clean, noisy, w, h, r, tau, method, extra)] = (ds_key, lv, alg_key, boundary)

    done = 0; total = len(tasks)
    for f in as_completed(futures):
        new_auc, new_val = f.result()
        ds_key, lv, alg_key, old_boundary = futures[f]
        done += 1
        rate = done/(time.time()-t0) if time.time()>t0 else 0
        eta = (total-done)/rate/60 if rate>0 else 0
        flag = " *** still boundary!" if new_val in ("0","0.0","15","15.0") else ""
        print(f"[{done}/{total} | {rate:.1f}t/s | ETA {eta:.0f}m] {ds_key}/{lv}/{alg_key} old_boundary={old_boundary} new_val={new_val} AUC={new_auc:.4f}{flag}", flush=True)
        results.append((ds_key, lv, alg_key, old_boundary, new_val, new_auc))

df = pd.DataFrame(results, columns=["dataset","level","algorithm","old_boundary","new_opt_val","auc"])
df.to_csv(f"{OUT}/boundary_fix_results.csv", index=False)

print(f"\nDone. {len(df)} fixed.")
still_boundary = df[df["new_opt_val"].isin(["0","0.0","15","15.0"])]
if len(still_boundary) > 0:
    print(f"WARNING: {len(still_boundary)} tasks still at boundary after extension:")
    for _,r in still_boundary.iterrows():
        print(f"  {r.dataset}/{r.level}/{r.algorithm}: val={r.new_opt_val} AUC={r.auc:.4f}")
