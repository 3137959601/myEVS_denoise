"""LED: BAF & STCF_orig at OWN optimal params, following Driving methodology.
BAF: sweep tau (param=time-us), r=1 → multi-point ROC from tau variation.
STCF_orig: for each k, sweep tau (param=time-us) → k×tau ROC points."""
import subprocess, os, time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

PY = r"D:/software/Anaconda_envs/envs/myEVS/python.exe"
OUT = r"D:/hjx_workspace/scientific_reserach/projects/myEVS/data/missing_alg"

LED_SCENES = ["scene_100","scene_1004","scene_1018","scene_1028","scene_1032",
              "scene_1033","scene_1034","scene_1043","scene_1045","scene_1046"]

# BAF: sweep tau (same as Driving coarse)
BAF_TAUS = "1000,2000,4000,8000,16000,32000"

# STCF_orig: for each k, sweep tau (same k list as Driving coarse)
STCF_KS = [1, 2, 3, 4, 5, 6]
STCF_TAUS = "2000,4000,8000,16000,32000"

def run_baf(scene, clean, noisy, w, h):
    """BAF: sweep tau for multi-point ROC."""
    csv = f"{OUT}/led/miss_led_{scene}_baf_opt.csv"
    os.makedirs(os.path.dirname(csv), exist_ok=True)
    if os.path.exists(csv):
        os.remove(csv)
    cmd = [PY, '-m', 'myevs.cli', 'roc',
           '--clean', clean, '--noisy', noisy,
           '--assume', 'npy', '--width', str(w), '--height', str(h),
           '--tick-ns', '1000', '--engine', 'cpp', '--method', 'baf',
           '--radius-px', '1', '--min-neighbors', '1',
           '--param', 'time-us', '--values', BAF_TAUS,
           '--match-us', '0', '--match-bin-radius', '0',
           '--tag', 'baf_r1', '--out-csv', csv, '--append']
    env = os.environ.copy()
    subprocess.run(cmd, check=True, timeout=1200, env=env, capture_output=True)
    df = pd.read_csv(csv)
    best = df.loc[df["auc"].idxmax()]
    return scene, 'baf', float(best["auc"]), float(best["f1"]), int(best["time_us"]) if "time_us" in best else int(best.get("value",0))

def run_stcf_orig(scene, clean, noisy, w, h):
    """STCF_orig: for each k, sweep tau."""
    csv = f"{OUT}/led/miss_led_{scene}_stcf_orig_opt.csv"
    os.makedirs(os.path.dirname(csv), exist_ok=True)
    if os.path.exists(csv):
        os.remove(csv)
    env = os.environ.copy()
    for k in STCF_KS:
        cmd = [PY, '-m', 'myevs.cli', 'roc',
               '--clean', clean, '--noisy', noisy,
               '--assume', 'npy', '--width', str(w), '--height', str(h),
               '--tick-ns', '1000', '--engine', 'cpp', '--method', 'stcf_original',
               '--radius-px', '1', '--min-neighbors', str(k),
               '--param', 'time-us', '--values', STCF_TAUS,
               '--match-us', '0', '--match-bin-radius', '0',
               '--tag', f'stcf_orig_k{k}', '--out-csv', csv, '--append']
        subprocess.run(cmd, check=True, timeout=600, env=env, capture_output=True)
    df = pd.read_csv(csv)
    best = df.loc[df["auc"].idxmax()]
    k_opt = int(best["min_neighbors"]) if "min_neighbors" in best else int(best.get("value",0))
    tau_opt = int(best["time_us"]) if "time_us" in best else 0
    return scene, 'stcf_orig', float(best["auc"]), float(best["f1"]), k_opt, tau_opt

# Build tasks
baf_tasks = []
stcf_tasks = []
for s in LED_SCENES:
    clean = f"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/{s}/slices_00031_00040_100ms/{s}_100ms_signal_only.npy"
    noisy = f"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/{s}/slices_00031_00040_100ms/{s}_100ms_labeled.npy"
    baf_tasks.append((s, clean, noisy, 1280, 720))
    stcf_tasks.append((s, clean, noisy, 1280, 720))

total = len(baf_tasks) + len(stcf_tasks)
print(f"LED BAF({len(baf_tasks)}) + STCF_orig({len(stcf_tasks)}) = {total} tasks")
t0 = time.time()

results = []
with ThreadPoolExecutor(max_workers=12) as ex:
    futures = {}
    for t in baf_tasks:
        futures[ex.submit(run_baf, *t)] = ('baf', t[0])
    for t in stcf_tasks:
        futures[ex.submit(run_stcf_orig, *t)] = ('stcf_orig', t[0])

    done = 0
    for f in as_completed(futures):
        result = f.result()
        alg_type = futures[f][0]
        if alg_type == 'baf':
            scene, alg, auc, f1, tau_opt = result
            results.append({'scene': scene, 'algorithm': alg, 'auc': auc, 'f1': f1,
                          'opt_tau': tau_opt, 'opt_k': 1})
            done += 1
            rate = done/(time.time()-t0) if time.time()>t0 else 0
            eta = (total-done)/rate/60 if rate>0 else 0
            print(f"[{done}/{total} | {rate:.1f}t/s | ETA {eta:.0f}m] {scene} BAF AUC={auc:.4f} F1={f1:.4f} opt_tau={tau_opt}", flush=True)
        else:
            scene, alg, auc, f1, k_opt, tau_opt = result
            results.append({'scene': scene, 'algorithm': alg, 'auc': auc, 'f1': f1,
                          'opt_tau': tau_opt, 'opt_k': k_opt})
            done += 1
            rate = done/(time.time()-t0) if time.time()>t0 else 0
            eta = (total-done)/rate/60 if rate>0 else 0
            print(f"[{done}/{total} | {rate:.1f}t/s | ETA {eta:.0f}m] {scene} STCF_orig AUC={auc:.4f} F1={f1:.4f} opt_tau={tau_opt} opt_k={k_opt}", flush=True)

df = pd.DataFrame(results)
df.to_csv(f"{OUT}/led/baf_stcf_opt_final.csv", index=False)

# Summary
print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)
for alg in ['baf', 'stcf_orig']:
    sub = df[df.algorithm == alg]
    print(f"\n{alg}:")
    for _, r in sub.iterrows():
        if alg == 'baf':
            print(f"  {r.scene}: AUC={r.auc:.4f} F1={r.f1:.4f} opt_tau={r.opt_tau}")
        else:
            print(f"  {r.scene}: AUC={r.auc:.4f} F1={r.f1:.4f} opt_tau={r.opt_tau} opt_k={r.opt_k}")
    print(f"  MEAN AUC: {sub.auc.mean():.4f}")

# Compare with old values
print("\n" + "=" * 70)
print("COMPARISON: NEW vs OLD")
print("=" * 70)
try:
    old_baf = pd.read_csv(f"{OUT}/led/_cache/miss_led_scene_100_baf.csv") if False else None
except: pass
print(f"BAF  LED mean: {df[df.algorithm=='baf'].auc.mean():.4f} (OLD was ~0.7488)")
print(f"STCF_orig LED mean: {df[df.algorithm=='stcf_orig'].auc.mean():.4f} (OLD was ~0.8633)")

print("\nDONE")
