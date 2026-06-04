"""Fix EBF on ED24 Bicycle: r×tau sweep per level, standard 17-pt threshold, 12-threaded."""
import subprocess, os, time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

PY = r"D:/software/Anaconda_envs/envs/myEVS/python.exe"
THR = "0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,8"
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
R_LIST = [2, 3, 4]
TAU_LIST = [8000, 16000, 32000, 64000, 128000]

def run_one(lv, clean, noisy, r, tau):
    csv = f"{OUT}/_ebf_cache/ebf_{lv}_r{r}_tau{tau}.csv"
    os.makedirs(os.path.dirname(csv), exist_ok=True)
    if os.path.exists(csv): os.remove(csv)
    cmd = [PY, '-m', 'myevs.cli', 'roc',
           '--clean', clean, '--noisy', noisy,
           '--assume', 'npy', '--width', str(W), '--height', str(H),
           '--tick-ns', '1000', '--engine', 'cpp', '--method', 'ebf',
           '--radius-px', str(r), '--time-us', str(tau),
           '--param', 'min-neighbors', '--values', THR,
           '--match-us', '0', '--match-bin-radius', '0',
           '--tag', f'ebf_fix_r{r}_tau{tau}', '--out-csv', csv, '--append']
    env = os.environ.copy()
    subprocess.run(cmd, check=True, timeout=600, env=env, capture_output=True)
    df = pd.read_csv(csv)
    best = df.loc[df["auc"].idxmax()]
    return lv, r, tau, float(best["auc"]), float(best["f1"]), str(best.get("value",""))

tasks = [(lv, clean, noisy, r, tau)
         for lv, (clean, noisy) in LEVELS.items()
         for r in R_LIST for tau in TAU_LIST]
total = len(tasks)
print(f"EBF Bike fix: {total} tasks ({len(LEVELS)}lv x {len(R_LIST)}r x {len(TAU_LIST)}tau)")
t0 = time.time()

results = []
with ThreadPoolExecutor(max_workers=12) as ex:
    futures = {ex.submit(run_one, *t): t for t in tasks}
    done = 0
    for f in as_completed(futures):
        lv, r, tau, auc, f1, thr = f.result()
        results.append((lv, r, tau, auc, f1, thr))
        done += 1
        rate = done/(time.time()-t0) if time.time()>t0 else 0
        eta = (total-done)/rate/60 if rate>0 else 0
        print(f"[{done}/{total} | {rate:.1f}t/s | ETA {eta:.0f}m] Bike {lv} r={r} tau={tau} AUC={auc:.4f}", flush=True)

df = pd.DataFrame(results, columns=["level","r","tau","auc","f1","best_thr"])
df.to_csv(f"{OUT}/ebf_bike_v2_all.csv", index=False)

print("\n=== EBF BIKE PER-LEVEL BEST ===")
for lv in ["1.8","2.1","2.5","3.3"]:
    sub = df[df.level==lv]
    best = sub.loc[sub.auc.idxmax()]
    print(f"  {lv}: best AUC={best.auc:.4f} r={int(best.r)} tau={int(best.tau)} thr={best.best_thr}")
    for _, row in sub.nlargest(3, 'auc').iterrows():
        print(f"    r={int(row.r)} tau={int(row.tau)} AUC={row.auc:.4f}")

# Consistency
print("\n=== CONSISTENCY ===")
anomalies = 0
for r in R_LIST:
    for tau in TAU_LIST:
        a18 = df[(df.level=="1.8")&(df.r==r)&(df.tau==tau)]["auc"].values
        a33 = df[(df.level=="3.3")&(df.r==r)&(df.tau==tau)]["auc"].values
        if len(a18) and len(a33) and a33[0] > a18[0]:
            anomalies += 1
            print(f"  ANOMALY r={r} tau={tau}: light={a18[0]:.4f} heavy={a33[0]:.4f}")
print(f"  {anomalies} anomalies total")
print("DONE")
