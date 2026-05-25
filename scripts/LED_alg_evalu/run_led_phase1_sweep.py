"""Phase 1: Full (r,tau) sweep for EBF/YNoise/TS/PFD/KNoise/STCF on LED scene_100.
Following Driving methodology — each (r,tau) combination gets its own threshold sweep."""
import subprocess, os, time, re
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

PY = r"D:/software/Anaconda_envs/envs/myEVS/python.exe"
THR = "0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,8"
OUT = r"D:/hjx_workspace/scientific_reserach/projects/myEVS/data/missing_alg/led"

clean = r"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/scene_100/slices_00031_00040_100ms/scene_100_100ms_signal_only.npy"
noisy = r"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/scene_100/slices_00031_00040_100ms/scene_100_100ms_labeled.npy"
W, H = 1280, 720

os.makedirs(f"{OUT}/_cache", exist_ok=True)

def run_one(method, tag, r, tau, thr, extra_args=None):
    csv = f"{OUT}/_cache/phase1_{tag}.csv"
    if os.path.exists(csv):
        os.remove(csv)
    cmd = [PY, '-m', 'myevs.cli', 'roc',
           '--clean', clean, '--noisy', noisy,
           '--assume', 'npy', '--width', str(W), '--height', str(H),
           '--tick-ns', '1000', '--engine', 'cpp', '--method', method,
           '--radius-px', str(r), '--time-us', str(tau),
           '--param', 'min-neighbors', '--values', thr,
           '--match-us', '0', '--match-bin-radius', '0',
           '--tag', tag, '--out-csv', csv, '--append']
    if extra_args:
        cmd.extend(extra_args)
    env = os.environ.copy()
    subprocess.run(cmd, check=True, timeout=600, env=env, capture_output=True)
    df = pd.read_csv(csv)
    best = df.loc[df["auc"].idxmax()]
    return float(best["auc"]), float(best["f1"]), str(best.get("value",""))

# Define sweep spaces (following Driving methodology)
TASKS = []

# EBF: r={2,3}, tau={8K,16K,32K,64K}
for r in [2, 3]:
    for tau in [8000, 16000, 32000, 64000]:
        TASKS.append(('ebf', f'ebf_r{r}_tau{tau}', r, tau, THR, None))

# YNoise: r={1,2,3}, tau={8K,16K,32K,64K}
for r in [1, 2, 3]:
    for tau in [8000, 16000, 32000, 64000]:
        TASKS.append(('ynoise', f'ynoise_r{r}_tau{tau}', r, tau, "1,2,3,4,6,8", None))

# TS: r={1,2,3}, tau={8K,16K,32K,64K}
for r in [1, 2, 3]:
    for tau in [8000, 16000, 32000, 64000]:
        TASKS.append(('ts', f'ts_r{r}_tau{tau}', r, tau, "0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5", None))

# PFD: r=1, m={1,2,3}, tau={8K,16K,32K}
for m in [1, 2, 3]:
    for tau in [8000, 16000, 32000]:
        TASKS.append(('pfd', f'pfd_r1_tau{tau}_m{m}', 1, tau, "1,2,3,4,6",
                      ['--refractory-us', str(m), '--pfd-mode', 'a']))

# KNoise: tau={1K,2K,4K,8K,16K,32K}, each with threshold sweep
for tau in [1000, 2000, 4000, 8000, 16000, 32000]:
    TASKS.append(('knoise', f'knoise_tau{tau}', 1, tau, "0,1,2,3,4,5,6", None))

# STCF: r={1,2}, tau={2K,4K,8K,16K,32K}
for r in [1, 2]:
    for tau in [2000, 4000, 8000, 16000, 32000]:
        TASKS.append(('stcf', f'stcf_r{r}_tau{tau}', r, tau, THR, None))

total = len(TASKS)
print(f"Phase 1: {total} tasks on LED scene_100")
for alg in ['ebf','ynoise','ts','pfd','knoise','stcf']:
    n = sum(1 for t in TASKS if t[0]==alg)
    print(f"  {alg}: {n} tasks")

t0 = time.time()
results = []
with ThreadPoolExecutor(max_workers=12) as ex:
    futures = {ex.submit(run_one, *t): t for t in TASKS}
    done = 0
    for f in as_completed(futures):
        method, tag, r, tau, thr, extra = futures[f]
        auc, f1, best_val = f.result()
        results.append((method, tag, r, tau, auc, f1, best_val))
        done += 1
        rate = done/(time.time()-t0) if time.time()>t0 else 0
        eta = (total-done)/rate/60 if rate>0 else 0
        print(f"[{done}/{total} | {rate:.1f}t/s | ETA {eta:.0f}m] {tag}: AUC={auc:.4f} F1={f1:.4f} best_thr={best_val}", flush=True)

df = pd.DataFrame(results, columns=["method","tag","r","tau","auc","f1","best_thr"])
df.to_csv(f"{OUT}/phase1_full_sweep.csv", index=False)

# Best per algorithm
print("\n" + "=" * 70)
print("PHASE 1 RESULTS: Best (r,tau) per algorithm on LED scene_100")
print("=" * 70)
for alg in ['ebf','ynoise','ts','pfd','knoise','stcf']:
    sub = df[df.method == alg]
    best = sub.loc[sub.auc.idxmax()]
    print(f"  {alg}: best AUC={best.auc:.4f} r={int(best.r)} tau={int(best.tau)} thr={best.best_thr}")
    # Show top 3
    top3 = sub.nlargest(3, 'auc')
    for _, r in top3.iterrows():
        print(f"    r={int(r.r)} tau={int(r.tau)} AUC={r.auc:.4f}")

# Compare with old values
print("\n" + "=" * 70)
print("COMPARISON (scene_100 only)")
print("=" * 70)
old_vals = {'ebf': 0.8100, 'ynoise': 0.8068, 'ts': 0.7540, 'pfd': 0.8248, 'knoise': 0.5322, 'stcf': 0.7293}
for alg in ['ebf','ynoise','ts','pfd','knoise','stcf']:
    sub = df[df.method == alg]
    new_best = sub.auc.max()
    old = old_vals.get(alg, 0)
    print(f"  {alg}: OLD={old:.4f} → NEW={new_best:.4f} (Δ={new_best-old:+.4f})")

print("\nDONE")
