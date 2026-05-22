"""Fix: MLPF LED + MLPF 3hz retry + threshold boundary detection."""
import subprocess, os, time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

PY = r"D:/software/Anaconda_envs/envs/myEVS/python.exe"
THR = "0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,8"
THR_EXT = "0,0.1,0.2,0.3,0.4,0.5,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,8,10,12,15"
MLPF_THR = "0.01,0.02,0.03,0.04,0.05,0.1,0.14,0.2,0.3,0.5"

OUT = r"D:/hjx_workspace/scientific_reserach/projects/myEVS/data/missing_alg"

def run_cli(method, clean, noisy, w, h, r, tau, thr, tag, csv, extra_args=None, env_extra=None, timeout=900, model=None):
    env = os.environ.copy()
    if env_extra: env.update(env_extra)
    cmd = [PY,'-m','myevs.cli','roc','--clean',clean,'--noisy',noisy,
           '--assume','npy','--width',str(w),'--height',str(h),
           '--tick-ns','1000','--engine','cpp','--method',method,
           '--radius-px',str(r),'--time-us',str(tau),
           '--param','min-neighbors','--values',thr,
           '--match-us','0','--match-bin-radius','0',
           '--tag',tag,'--out-csv',csv,'--append']
    if model: cmd.extend(['--mlpf-model', model])
    if extra_args: cmd.extend(extra_args)
    subprocess.run(cmd, check=True, timeout=timeout, env=env, capture_output=True)
    df = pd.read_csv(csv)
    best = df.loc[df["auc"].idxmax()]
    return float(best["auc"]), float(best["f1"]), str(best.get("value",""))

# ====== Part 1: MLPF LED 10 scenes ======
LED_SCENES = ["scene_100","scene_1004","scene_1018","scene_1028","scene_1032","scene_1033","scene_1034","scene_1043","scene_1045","scene_1046"]
mlpf_tasks = []
for s in LED_SCENES:
    clean = f"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/{s}/slices_00031_00040_100ms/{s}_100ms_signal_only.npy"
    noisy = f"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/{s}/slices_00031_00040_100ms/{s}_100ms_labeled.npy"
    model = f"D:/hjx_workspace/scientific_reserach/projects/myEVS/data/LED/models/mlpf_torch_{s}.pt"
    mlpf_tasks.append(('mlpf','led',s,clean,noisy,1280,720,3,100000,model))

# ====== Part 2: MLPF 3hz retry ======
mlpf_tasks.append(('mlpf','drive','3hz',
    r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_3hz_ed24_withlabel/driving_noise_3hz_signal_only.npy",
    r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_3hz_ed24_withlabel/driving_noise_3hz_labeled.npy",
    346,260,3,100000,
    r"D:/hjx_workspace/scientific_reserach/projects/myEVS/data/DND21/mydriving_ED24/MLPF/mlpf_torch_3hz.pt"))

def run_mlpf(typ, ds, lv, clean, noisy, w, h, r, tau, model):
    csv = f"{OUT}/{ds}/miss_{ds}_{lv}_mlpf_v2.csv"
    auc, f1, _ = run_cli('mlpf', clean, noisy, w, h, r, tau, MLPF_THR, f'mlpf_{ds}_{lv}', csv, model=model, timeout=900)
    return ds, lv, 'mlpf', auc, f1

# ====== Main ======
total = len(mlpf_tasks)
print(f"Boundary fix: {total} tasks (MLPF LED:{len(LED_SCENES)} + 3hz:1)")
t0 = time.time()

results = []
with ThreadPoolExecutor(max_workers=8) as ex:
    futures = {ex.submit(run_mlpf, *t): t for t in mlpf_tasks}
    done = 0
    for f in as_completed(futures):
        ds, lv, alg, auc, f1 = f.result()
        results.append((ds, lv, alg, auc, f1))
        done += 1
        rate = done/(time.time()-t0) if time.time()>t0 else 0
        print(f"[{done}/{total} | {rate:.1f}t/s] {ds}/{lv} {alg} AUC={auc:.4f} F1={f1:.4f}", flush=True)

df = pd.DataFrame(results, columns=["dataset","level","algorithm","auc","f1"])
df.to_csv(f"{OUT}/boundary_fix.csv", index=False)

# Summary
for alg in df.algorithm.unique():
    sub = df[df.algorithm==alg]
    print(f"\n{alg}: mean AUC={sub.auc.mean():.4f} (n={len(sub)})")
    for _,r in sub.iterrows():
        print(f"  {r.dataset}/{r.level}: AUC={r.auc:.4f}")

print("\nDONE")
