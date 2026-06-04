"""LED polarity ablation: blind (no polarity) vs opposite-only. 12-threaded."""
import subprocess, os, time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

PY = r"D:/software/Anaconda_envs/envs/myEVS/python.exe"
THR = "0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,8"
OUT = r"D:/hjx_workspace/scientific_reserach/projects/myEVS/data/Hyperparameter ablation_study/led"

R, TAU, SIGMA = 2, 8000, 2.0

SCENES = ["scene_100","scene_1004","scene_1018","scene_1028","scene_1032",
          "scene_1033","scene_1034","scene_1043","scene_1045","scene_1046"]

VARIANTS = {
    "n149_blind": {"env": {"MYEVS_N149_BLIND": "1"}},
    "n149_opp":   {"env": {"MYEVS_N149_NO_SAME": "1", "MYEVS_N149_ALPHA_FIXED": "1.0"}},
}

def run_one(scene, variant_name, cfg):
    clean = f"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/{scene}/slices_00031_00040_100ms/{scene}_100ms_signal_only.npy"
    noisy = f"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/{scene}/slices_00031_00040_100ms/{scene}_100ms_labeled.npy"
    tag = f"led_pol_{variant_name}_{scene}"
    csv = f"{OUT}/_pol_ablation/{tag}.csv"
    os.makedirs(os.path.dirname(csv), exist_ok=True)
    if os.path.exists(csv): os.remove(csv)

    env = os.environ.copy()
    env["MYEVS_N149_HOT_BITS"] = "16"
    env["MYEVS_N149_SIGMA"] = str(SIGMA)
    for k, v in cfg.get("env", {}).items():
        env[k] = v

    cmd = [PY, '-m', 'myevs.cli', 'roc',
           '--clean', clean, '--noisy', noisy,
           '--assume', 'npy', '--width', '1280', '--height', '720',
           '--tick-ns', '1000', '--engine', 'cpp', '--method', 'n149',
           '--radius-px', str(R), '--time-us', str(TAU),
           '--param', 'min-neighbors', '--values', THR,
           '--match-us', '0', '--match-bin-radius', '0',
           '--tag', tag, '--out-csv', csv, '--append']
    subprocess.run(cmd, check=True, timeout=600, env=env, capture_output=True)
    df = pd.read_csv(csv)
    return scene, variant_name, float(df.loc[df["auc"].idxmax()]["auc"])

tasks = [(s, vn, cfg) for s in SCENES for vn, cfg in VARIANTS.items()]
total = len(tasks)
print(f"LED polarity ablation: {total} tasks ({len(SCENES)} scenes x {len(VARIANTS)} variants)")
t0 = time.time()

results = []
with ThreadPoolExecutor(max_workers=12) as ex:
    futures = {ex.submit(run_one, *t): t for t in tasks}
    done = 0
    for f in as_completed(futures):
        scene, variant_name, auc = f.result()
        results.append((scene, variant_name, auc))
        done += 1
        rate = done/(time.time()-t0) if time.time()>t0 else 0
        eta = (total-done)/rate/60 if rate>0 else 0
        print(f"[{done}/{total} | {rate:.1f}t/s | ETA {eta:.0f}m] {scene} {variant_name} AUC={auc:.4f}", flush=True)

df = pd.DataFrame(results, columns=["scene","variant","auc"])
df.to_csv(f"{OUT}/polarity_ablation_v2.csv", index=False)

# Summary
print("\n" + "=" * 70)
print("LED POLARITY ABLATION (r=2, tau=8K, sigma=2.0)")
print("=" * 70)
for vn in ["n149_blind", "n149_opp"]:
    sub = df[df.variant == vn]
    mean_auc = sub.auc.mean()
    print(f"\n{vn}: mean AUC={mean_auc:.4f}")
    for _, row in sub.iterrows():
        print(f"  {row.scene}: AUC={row.auc:.4f}")

# Load baseline from earlier alpha sweep
try:
    base_df = pd.read_csv(f"{OUT}/led_alpha_full_optimal.csv")
    base_mean = base_df[base_df["alpha"]==1.0]["auc"].mean()
    print(f"\nbaseline N149 (a=1.0): mean AUC={base_mean:.4f}")
    blind_mean = df[df.variant=="n149_blind"]["auc"].mean()
    opp_mean = df[df.variant=="n149_opp"]["auc"].mean()
    print(f"blind (S=R_all*f):       mean AUC={blind_mean:.4f} (vs baseline Δ={blind_mean-base_mean:+.4f})")
    print(f"opp   (S=R-*f):          mean AUC={opp_mean:.4f} (vs baseline Δ={opp_mean-base_mean:+.4f})")
except: pass

print("\nDONE")
