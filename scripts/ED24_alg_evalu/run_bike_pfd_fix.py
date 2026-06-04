"""Fix PFD on Bike: sweep tau per (r,m,k) combo, like STCF_orig methodology. 12-threaded."""
import subprocess, os, time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

PY = r"D:/software/Anaconda_envs/envs/myEVS/python.exe"
OUT = r"D:/hjx_workspace/scientific_reserach/projects/myEVS/data/missing_alg/bike"

LEVELS = {
    "1.8": (r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_1.8_signal_only.npy",
            r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_1.8.npy"),
    "2.1": (r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_2.1_signal_only.npy",
            r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_2.1.npy"),
    "2.5": (r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_2.5_signal_only.npy",
            r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_2.5.npy"),
    "3.3": (r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_3.3_signal_only.npy",
            r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_3.3.npy"),
}
W, H = 346, 260

# PFD hyperparameter grid (like STCF_orig sweeps per k)
R_LIST = [1, 2, 3]
M_LIST = [1, 2]  # refractory_us → stage1_var
K_LIST = [1, 2]  # min_neighbors

# tau sweep for ROC (like BAF tau sweep on ED24)
TAU_LIST = "200,500,1000,2000,4000,8000,16000,32000,64000"

def run_one(lv, clean, noisy, r, m, k):
    tag = f"pfd_fix_r{r}_m{m}_k{k}"
    csv = f"{OUT}/_pfd_cache/pfd_{lv}_r{r}_m{m}_k{k}.csv"
    os.makedirs(os.path.dirname(csv), exist_ok=True)
    if os.path.exists(csv): os.remove(csv)
    cmd = [PY, '-m', 'myevs.cli', 'roc',
           '--clean', clean, '--noisy', noisy,
           '--assume', 'npy', '--width', str(W), '--height', str(H),
           '--tick-ns', '1000', '--engine', 'cpp', '--method', 'pfd',
           '--radius-px', str(r), '--min-neighbors', str(k),
           '--refractory-us', str(m), '--pfd-mode', 'a',
           '--param', 'time-us', '--values', TAU_LIST,
           '--match-us', '0', '--match-bin-radius', '0',
           '--tag', tag, '--out-csv', csv, '--append']
    env = os.environ.copy()
    subprocess.run(cmd, check=True, timeout=600, env=env, capture_output=True)
    df = pd.read_csv(csv)
    best = df.loc[df["auc"].idxmax()]
    opt_tau = int(float(best.get("value", 0)))
    return lv, r, m, k, opt_tau, float(best["auc"]), float(best["f1"])

tasks = [(lv, clean, noisy, r, m, k)
         for lv, (clean, noisy) in LEVELS.items()
         for r in R_LIST for m in M_LIST for k in K_LIST]
total = len(tasks)
print(f"PFD Bike fix (tau-sweep ROC): {total} combos x {len(TAU_LIST.split(','))} tau pts each")
print(f"  r={R_LIST}, m={M_LIST}, k={K_LIST}")
t0 = time.time()

results = []
with ThreadPoolExecutor(max_workers=12) as ex:
    futures = {ex.submit(run_one, *t): t for t in tasks}
    done = 0
    for f in as_completed(futures):
        lv, r, m, k, opt_tau, auc, f1 = f.result()
        results.append((lv, r, m, k, opt_tau, auc, f1))
        done += 1
        rate = done/(time.time()-t0) if time.time()>t0 else 0
        eta = (total-done)/rate/60 if rate>0 else 0
        print(f"[{done}/{total} | {rate:.1f}t/s | ETA {eta:.0f}m] {lv} r={r} m={m} k={k} AUC={auc:.4f} opt_tau={opt_tau}", flush=True)

df = pd.DataFrame(results, columns=["level","r","m","k","opt_tau","auc","f1"])
df.to_csv(f"{OUT}/pfd_bike_v2_tau_sweep.csv", index=False)

# Per-level best
print("\n=== PFD BIKE PER-LEVEL BEST (tau-sweep ROC) ===")
for lv in ["1.8","2.1","2.5","3.3"]:
    sub = df[df.level==lv]
    best = sub.loc[sub.auc.idxmax()]
    print(f"  {lv}: best AUC={best.auc:.4f} r={int(best.r)} m={int(best.m)} k={int(best.k)} opt_tau={int(best.opt_tau)}")
    for _, row in sub.nlargest(3, 'auc').iterrows():
        print(f"    r={int(row.r)} m={int(row.m)} k={int(row.k)} tau={int(row.opt_tau)} AUC={row.auc:.4f}")

# Compare with old
print("\n=== COMPARISON OLD vs NEW ===")
old = {"1.8":0.8853, "2.1":0.8759, "2.5":0.8682, "3.3":0.8606}
for lv in ["1.8","2.1","2.5","3.3"]:
    sub = df[df.level==lv]
    new = sub.auc.max()
    print(f"  {lv}: OLD={old[lv]:.4f}  NEW={new:.4f}  Δ={new-old[lv]:+.4f}")

# Consistency
print("\n=== CONSISTENCY ===")
for r in R_LIST:
    for m in M_LIST:
        for k in K_LIST:
            a18 = df[(df.level=="1.8")&(df.r==r)&(df.m==m)&(df.k==k)]["auc"].values
            a33 = df[(df.level=="3.3")&(df.r==r)&(df.m==m)&(df.k==k)]["auc"].values
            if len(a18) and len(a33) and a33[0] > a18[0]:
                print(f"  ANOMALY r={r} m={m} k={k}: light={a18[0]:.4f} heavy={a33[0]:.4f}")
print("DONE")
