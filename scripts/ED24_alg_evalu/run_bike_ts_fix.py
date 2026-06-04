"""Fix TS on ED24 Bicycle: proper (r,tau) sweep per level, 12-threaded."""
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
TS_R = [1, 2, 3, 4]
TS_TAU = [16000, 32000, 64000, 128000]
TS_THR = "0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5"

def run_one(lv, clean, noisy, r, tau):
    tag = f"ts_fix_r{r}_tau{tau}"
    csv = f"{OUT}/_ts_cache/ts_{lv}_r{r}_tau{tau}.csv"
    os.makedirs(os.path.dirname(csv), exist_ok=True)
    if os.path.exists(csv): os.remove(csv)
    cmd = [PY, '-m', 'myevs.cli', 'roc',
           '--clean', clean, '--noisy', noisy,
           '--assume', 'npy', '--width', str(W), '--height', str(H),
           '--tick-ns', '1000', '--engine', 'cpp', '--method', 'ts',
           '--radius-px', str(r), '--time-us', str(tau),
           '--param', 'min-neighbors', '--values', TS_THR,
           '--match-us', '0', '--match-bin-radius', '0',
           '--tag', tag, '--out-csv', csv, '--append']
    env = os.environ.copy()
    subprocess.run(cmd, check=True, timeout=300, env=env, capture_output=True)
    df = pd.read_csv(csv)
    best = df.loc[df["auc"].idxmax()]
    return lv, r, tau, float(best["auc"]), float(best["f1"]), str(best.get("value",""))

# Build tasks: per-level, sweep (r,tau)
tasks = []
for lv, (clean, noisy) in LEVELS.items():
    for r in TS_R:
        for tau in TS_TAU:
            tasks.append((lv, clean, noisy, r, tau))

total = len(tasks)
print(f"TS Bike fix: {total} tasks ({len(LEVELS)} levels x {len(TS_R)}r x {len(TS_TAU)}tau)")
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
df.to_csv(f"{OUT}/ts_bike_v2_all.csv", index=False)

# Per-level best
print("\n" + "=" * 70)
print("TS BIKE PER-LEVEL BEST")
print("=" * 70)
for lv in ["1.8","2.1","2.5","3.3"]:
    sub = df[df.level==lv]
    best = sub.loc[sub.auc.idxmax()]
    print(f"  {lv}: best AUC={best.auc:.4f} F1={best.f1:.4f} r={int(best.r)} tau={int(best.tau)} thr={best.best_thr}")
    # Top 3
    for _, row in sub.nlargest(3, 'auc').iterrows():
        print(f"    r={int(row.r)} tau={int(row.tau)} AUC={row.auc:.4f}")

# Check: heavy > light?
print("\n" + "=" * 70)
print("CONSISTENCY CHECK: heavy vs light")
print("=" * 70)
for r in TS_R:
    for tau in TS_TAU:
        row_18 = df[(df.level=="1.8")&(df.r==r)&(df.tau==tau)]
        row_33 = df[(df.level=="3.3")&(df.r==r)&(df.tau==tau)]
        if len(row_18) and len(row_33):
            a18 = row_18["auc"].values[0]
            a33 = row_33["auc"].values[0]
            flag = " *** ANOMALY!" if a33 > a18 else ""
            print(f"  r={r} tau={tau:>6}: light={a18:.4f} heavy={a33:.4f} Δ={a33-a18:+.4f}{flag}")

print("\nDONE")
