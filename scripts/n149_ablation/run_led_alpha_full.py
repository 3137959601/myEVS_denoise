"""LED full alpha sweep at OPTIMAL params (r=2, tau=8K, sigma=2.0), all 10 scenes."""
import subprocess, os, time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

PY = r"D:/software/Anaconda_envs/envs/myEVS/python.exe"
THR = "0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,8"
OUT = r"D:/hjx_workspace/scientific_reserach/projects/myEVS/data/Hyperparameter ablation_study/led"

R, TAU, SIGMA = 2, 8000, 2.0

SCENES = ["scene_100","scene_1004","scene_1018","scene_1028","scene_1032",
          "scene_1033","scene_1034","scene_1043","scene_1045","scene_1046"]
ALPHAS = [0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0]

def run_one(scene, alpha):
    clean = f"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/{scene}/slices_00031_00040_100ms/{scene}_100ms_signal_only.npy"
    noisy = f"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/{scene}/slices_00031_00040_100ms/{scene}_100ms_labeled.npy"
    tag = f"led_alpha_opt_{scene}_a{str(alpha).replace('.','p')}"
    csv = f"{OUT}/{tag}.csv"
    if os.path.exists(csv): os.remove(csv)
    env = os.environ.copy()
    env["MYEVS_N149_HOT_BITS"] = "16"
    env["MYEVS_N149_SIGMA"] = str(SIGMA)
    env["MYEVS_N149_ALPHA_FIXED"] = str(alpha)
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
    return scene, alpha, float(df.loc[df["auc"].idxmax()]["auc"])

tasks = [(s, a) for s in SCENES for a in ALPHAS]
total = len(tasks)
print(f"LED alpha full sweep (optimal params): {total} tasks ({len(SCENES)} scenes x {len(ALPHAS)} alphas)")
t0 = time.time()

results = []
with ThreadPoolExecutor(max_workers=12) as ex:
    futures = {ex.submit(run_one, *t): t for t in tasks}
    done = 0
    for f in as_completed(futures):
        scene, alpha, auc = f.result()
        results.append((scene, alpha, auc))
        done += 1
        rate = done/(time.time()-t0) if time.time()>t0 else 0
        eta = (total-done)/rate/60 if rate>0 else 0
        print(f"[{done}/{total} | {rate:.1f}t/s | ETA {eta:.0f}m] {scene} α={alpha} AUC={auc:.4f}", flush=True)

df = pd.DataFrame(results, columns=["scene","alpha","auc"])
df.to_csv(f"{OUT}/led_alpha_full_optimal.csv", index=False)

# Summary: mean per alpha
print("\n" + "=" * 70)
print("LED 10-SCENE MEAN AUC per α (r=2, tau=8K, sigma=2.0)")
print("=" * 70)
for alpha in ALPHAS:
    mean_auc = df[df.alpha == alpha]["auc"].mean()
    marker = " *** BEST" if mean_auc == df.groupby("alpha")["auc"].mean().max() else ""
    print(f"  α={alpha:>4}: mean AUC={mean_auc:.4f}{marker}")

# Show per-scene detail
print("\n" + "=" * 70)
print("PER-SCENE")
print("=" * 70)
for s in SCENES:
    sub = df[df.scene == s]
    best_a = sub.loc[sub.auc.idxmax()]
    print(f"  {s}: best α={best_a.alpha} AUC={best_a.auc:.4f}")
    # Show all alpha values for this scene
    vals = ", ".join(f"α{a}={sub[sub.alpha==a].auc.values[0]:.4f}" for a in ALPHAS if len(sub[sub.alpha==a])>0)
    print(f"       {vals}")

print("\nDONE")
