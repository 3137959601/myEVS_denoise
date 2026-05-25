"""Phase 2: Run EBF/YNoise/TS/PFD/KNoise/STCF on ALL 10 LED scenes with optimal (r,tau) from Phase 1."""
import subprocess, os, time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

PY = r"D:/software/Anaconda_envs/envs/myEVS/python.exe"
OUT = r"D:/hjx_workspace/scientific_reserach/projects/myEVS/data/missing_alg/led"

LED_SCENES = ["scene_100","scene_1004","scene_1018","scene_1028","scene_1032",
              "scene_1033","scene_1034","scene_1043","scene_1045","scene_1046"]

# Optimal (r, tau) per algorithm from Phase 1 on scene_100
OPTIMAL = {
    'ebf':    ('ebf',    2, 8000,  "0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,8", None),
    'ynoise': ('ynoise', 2, 8000,  "1,2,3,4,6,8", None),
    'ts':     ('ts',     1, 8000,  "0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5", None),
    'pfd':    ('pfd',    1, 8000,  "1,2,3,4,6", ['--refractory-us','1','--pfd-mode','a']),
    'knoise': ('knoise', 1, 1000,  "0,1,2,3,4,5,6", None),
    'stcf':   ('stcf',   2, 4000,  "0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,8", None),
}

def run_one(alg_key, scene, clean, noisy, w, h):
    method, r, tau, thr, extra = OPTIMAL[alg_key]
    tag = f"led_v2_{alg_key}"
    csv = f"{OUT}/miss_led_{scene}_{alg_key}_v2.csv"
    os.makedirs(os.path.dirname(csv), exist_ok=True)
    if os.path.exists(csv):
        os.remove(csv)
    cmd = [PY, '-m', 'myevs.cli', 'roc',
           '--clean', clean, '--noisy', noisy,
           '--assume', 'npy', '--width', str(w), '--height', str(h),
           '--tick-ns', '1000', '--engine', 'cpp', '--method', method,
           '--radius-px', str(r), '--time-us', str(tau),
           '--param', 'min-neighbors', '--values', thr,
           '--match-us', '0', '--match-bin-radius', '0',
           '--tag', tag, '--out-csv', csv, '--append']
    if extra:
        cmd.extend(extra)
    env = os.environ.copy()
    subprocess.run(cmd, check=True, timeout=600, env=env, capture_output=True)
    df = pd.read_csv(csv)
    best = df.loc[df["auc"].idxmax()]
    return scene, alg_key, float(best["auc"]), float(best["f1"]), str(best.get("value",""))

# Build tasks
tasks = []
for s in LED_SCENES:
    clean = f"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/{s}/slices_00031_00040_100ms/{s}_100ms_signal_only.npy"
    noisy = f"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/{s}/slices_00031_00040_100ms/{s}_100ms_labeled.npy"
    for alg in OPTIMAL:
        tasks.append((alg, s, clean, noisy, 1280, 720))

total = len(tasks)
print(f"Phase 2: {total} tasks ({len(LED_SCENES)} scenes x {len(OPTIMAL)} algs)")
t0 = time.time()

results = []
with ThreadPoolExecutor(max_workers=12) as ex:
    futures = {ex.submit(run_one, *t): t for t in tasks}
    done = 0
    for f in as_completed(futures):
        scene, alg, auc, f1, best_val = f.result()
        results.append((scene, alg, auc, f1, best_val))
        done += 1
        rate = done/(time.time()-t0) if time.time()>t0 else 0
        eta = (total-done)/rate/60 if rate>0 else 0
        print(f"[{done}/{total} | {rate:.1f}t/s | ETA {eta:.0f}m] {scene} {alg} AUC={auc:.4f} F1={f1:.4f}", flush=True)

df = pd.DataFrame(results, columns=["scene","algorithm","auc","f1","best_thr"])
df.to_csv(f"{OUT}/phase2_all_scenes.csv", index=False)

# Summary
print("\n" + "=" * 70)
print("PHASE 2 FINAL: 10-scene mean AUC")
print("=" * 70)

# Also load BAF and STCF_orig results from earlier run
baf_stcf = pd.read_csv(f"{OUT}/baf_stcf_opt_final.csv")
# Rename stcf_orig
baf_stcf_renamed = baf_stcf.copy()

all_summary = []
for alg in ['ebf','ynoise','ts','pfd','knoise','stcf']:
    sub = df[df.algorithm == alg]
    mean_auc = sub.auc.mean()
    method, r, tau, thr, extra = OPTIMAL[alg]
    print(f"  {alg}: mean AUC={mean_auc:.4f} (r={r}, tau={tau})")
    for _, row in sub.iterrows():
        print(f"    {row.scene}: AUC={row.auc:.4f}")
    all_summary.append({'algorithm': alg, 'auc': mean_auc, 'r': r, 'tau': tau, 'params': f'r={r} tau={tau}'})

# Add BAF and STCF_orig
baf_mean = baf_stcf[baf_stcf['algorithm']=='baf']['auc'].mean()
stcf_orig_mean = baf_stcf[baf_stcf['algorithm']=='stcf_orig']['auc'].mean()
print(f"  baf (from earlier): mean AUC={baf_mean:.4f} (r=1, tau=1ms)")
print(f"  stcf_orig (from earlier): mean AUC={stcf_orig_mean:.4f} (r=1, tau=2ms, k=2)")
all_summary.append({'algorithm': 'baf', 'auc': baf_mean, 'r': 1, 'tau': 1000, 'params': 'r=1 tau=1ms'})
all_summary.append({'algorithm': 'stcf_orig', 'auc': stcf_orig_mean, 'r': 1, 'tau': 2000, 'params': 'r=1 tau=2ms k=2'})

# Store final summary
summary_df = pd.DataFrame(all_summary)
summary_df = summary_df.sort_values('auc', ascending=False)
summary_df.to_csv(f"{OUT}/final_summary.csv", index=False)

print("\n" + "=" * 70)
print("RANKED FINAL SUMMARY (10-scene mean AUC)")
print("=" * 70)
for _, row in summary_df.iterrows():
    print(f"  {row.algorithm:>10}: {row.auc:.4f}  ({row.params})")

print("\nDONE")
