"""DVSCLEAN polarity ablation: baseline(α=0.25) vs blind vs same vs opp. 12-threaded."""
import subprocess, os, time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

PY = r"D:/software/Anaconda_envs/envs/myEVS/python.exe"
THR = "0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,8"
OUT = r"D:/hjx_workspace/scientific_reserach/projects/myEVS/data/Hyperparameter ablation_study/dvsclean"

R, TAU, SIGMA = 5, 128000, 2.5
W, H = 1280, 720

SCENES = ["MAH00444","MAH00446","MAH00447","MAH00448","MAH00449"]
RATIOS = ["ratio50","ratio100"]

VARIANTS = {
    "baseline": {"MYEVS_N149_ALPHA_FIXED": "0.25"},
    "blind":    {"MYEVS_N149_BLIND": "1"},
    "same":     {"MYEVS_N149_NO_OPP": "1"},
    "opp":      {"MYEVS_N149_NO_SAME": "1", "MYEVS_N149_ALPHA_FIXED": "1.0"},
}

def run_one(scene, ratio, vn, env_extra):
    lv = f"{scene}_{ratio}"
    clean = f"D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy/{scene}/{ratio}/{scene}_{ratio}_signal_only.npy"
    noisy = f"D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy/{scene}/{ratio}/{scene}_{ratio}_labeled.npy"
    csv = f"{OUT}/_pol_ablation/dvsclean_pol_{vn}_{lv}.csv"
    os.makedirs(os.path.dirname(csv), exist_ok=True)
    if os.path.exists(csv): os.remove(csv)
    env = os.environ.copy()
    env["MYEVS_N149_HOT_BITS"] = "16"
    env["MYEVS_N149_SIGMA"] = str(SIGMA)
    for k, v in env_extra.items(): env[k] = v
    cmd = [PY, '-m', 'myevs.cli', 'roc',
           '--clean', clean, '--noisy', noisy,
           '--assume', 'npy', '--width', str(W), '--height', str(H),
           '--tick-ns', '1000', '--engine', 'cpp', '--method', 'n149',
           '--radius-px', str(R), '--time-us', str(TAU),
           '--param', 'min-neighbors', '--values', THR,
           '--match-us', '0', '--match-bin-radius', '0',
           '--tag', f'dvsclean_pol_{vn}_{lv}', '--out-csv', csv, '--append']
    subprocess.run(cmd, check=True, timeout=600, env=env, capture_output=True)
    df = pd.read_csv(csv)
    return lv, vn, float(df.loc[df["auc"].idxmax()]["auc"])

tasks = [(s, r, vn, cfg) for s in SCENES for r in RATIOS for vn, cfg in VARIANTS.items()]
total = len(tasks)
print(f"DVSCLEAN polarity ablation: {total} tasks ({len(SCENES)*len(RATIOS)} scenes x {len(VARIANTS)} variants)")
t0 = time.time()

results = []
with ThreadPoolExecutor(max_workers=12) as ex:
    futures = {ex.submit(run_one, *t): t for t in tasks}
    done = 0
    for f in as_completed(futures):
        lv, vn, auc = f.result()
        results.append((lv, vn, auc))
        done += 1
        rate = done/(time.time()-t0) if time.time()>t0 else 0
        eta = (total-done)/rate/60 if rate>0 else 0
        print(f"[{done}/{total} | {rate:.1f}t/s | ETA {eta:.0f}m] {lv} {vn} AUC={auc:.4f}", flush=True)

df = pd.DataFrame(results, columns=["level","variant","auc"])
df.to_csv(f"{OUT}/polarity_ablation.csv", index=False)

print("\n=== DVSCLEAN POLARITY ABLATION MEAN ===")
for vn in ["baseline","blind","same","opp"]:
    sub = df[df.variant == vn]
    print(f"  {vn}: mean AUC={sub.auc.mean():.4f}")

# Per-scene
print("\nPer-scene:")
for lv in sorted(df["level"].unique()):
    sub = df[df["level"] == lv]
    vals = " | ".join(f"{vn}={sub[sub.variant==vn]['auc'].values[0]:.4f}" for vn in ["baseline","blind","same","opp"])
    print(f"  {lv}: {vals}")

print("\nDONE")
