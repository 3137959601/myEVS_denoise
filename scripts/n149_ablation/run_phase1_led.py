"""Phase 1: Driving-ED24 all levels (1/3/5/7/10hz) coarse sweep.
Usage: python run_phase1_drive.py
Output: data/n149_ablation/phase1_led.csv
"""
import subprocess, os, time
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

PY = r"D:/software/Anaconda_envs/envs/myEVS/python.exe"
THR = "0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,8"
OUT = r"D:/hjx_workspace/scientific_reserach/projects/myEVS/data/Hyperparameter ablation_study/led"

LEVELS = {
    "scene_100": (r"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/scene_100/slices_00031_00040_100ms/scene_100_100ms_signal_only.npy",
            r"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/scene_100/slices_00031_00040_100ms/scene_100_100ms_labeled.npy"),
    "scene_1004": (r"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/scene_1004/slices_00031_00040_100ms/scene_1004_100ms_signal_only.npy",
            r"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/scene_1004/slices_00031_00040_100ms/scene_1004_100ms_labeled.npy"),
    "scene_1018": (r"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/scene_1018/slices_00031_00040_100ms/scene_1018_100ms_signal_only.npy",
            r"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/scene_1018/slices_00031_00040_100ms/scene_1018_100ms_labeled.npy"),
    "scene_1028": (r"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/scene_1028/slices_00031_00040_100ms/scene_1028_100ms_signal_only.npy",
            r"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/scene_1028/slices_00031_00040_100ms/scene_1028_100ms_labeled.npy"),
    "scene_1032": (r"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/scene_1032/slices_00031_00040_100ms/scene_1032_100ms_signal_only.npy",
            r"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/scene_1032/slices_00031_00040_100ms/scene_1032_100ms_labeled.npy"),
    "scene_1033": (r"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/scene_1033/slices_00031_00040_100ms/scene_1033_100ms_signal_only.npy",
            r"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/scene_1033/slices_00031_00040_100ms/scene_1033_100ms_labeled.npy"),
    "scene_1034": (r"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/scene_1034/slices_00031_00040_100ms/scene_1034_100ms_signal_only.npy",
            r"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/scene_1034/slices_00031_00040_100ms/scene_1034_100ms_labeled.npy"),
    "scene_1043": (r"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/scene_1043/slices_00031_00040_100ms/scene_1043_100ms_signal_only.npy",
            r"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/scene_1043/slices_00031_00040_100ms/scene_1043_100ms_labeled.npy"),
    "scene_1045": (r"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/scene_1045/slices_00031_00040_100ms/scene_1045_100ms_signal_only.npy",
            r"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/scene_1045/slices_00031_00040_100ms/scene_1045_100ms_labeled.npy"),
    "scene_1046": (r"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/scene_1046/slices_00031_00040_100ms/scene_1046_100ms_signal_only.npy",
            r"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/scene_1046/slices_00031_00040_100ms/scene_1046_100ms_labeled.npy"),
}

W, H = 1280, 720
R_DEF, TAU_DEF, SIGMA_DEF, ALPHA_DEF = 2, 16000, 3.0, "1.0"

SWEEPS = {
    "r": [1,2,3,5,7,9],
    "tau":   [4000, 8000, 16000, 32000, 64000, 128000, 256000],
    "sigma": [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0],
    "alpha": [0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, "ema"],
}

def run(level, clean, noisy, param, value):
    env = os.environ.copy()
    env["MYEVS_N149_HOT_BITS"] = "16"
    r, tau, sigma, alpha = R_DEF, TAU_DEF, SIGMA_DEF, ALPHA_DEF
    if param == "r":      r = int(value)
    elif param == "tau":  tau = int(value)
    elif param == "sigma": sigma = float(value); env["MYEVS_N149_SIGMA"] = str(sigma)
    elif param == "alpha":
        if value == "ema": env["MYEVS_N149_USE_EMA"] = "1"
        else: env["MYEVS_N149_ALPHA_FIXED"] = str(float(value))
    tag = "p1_led_%s_%s_%s" % (level, param, str(value).replace("/","_").replace(".","p"))
    csv = "%s/%s.csv" % (OUT, tag)
    cmd = [PY,'-m','myevs.cli','roc','--clean',clean,'--noisy',noisy,'--assume','npy',
           '--width',str(W),'--height',str(H),'--tick-ns','1000','--engine','cpp','--method','n149',
           '--radius-px',str(r),'--time-us',str(tau),'--param','min-neighbors','--values',THR,
           '--match-us','0','--match-bin-radius','0','--tag',tag,'--out-csv',csv,'--append']
    subprocess.run(cmd, check=True, timeout=900, env=env, capture_output=True)
    df = pd.read_csv(csv)
    return level, param, value, float(df.loc[df["auc"].idxmax()]["auc"]), float(df.loc[df["auc"].idxmax()]["f1"])

tasks = [(lv, cl, ny, p, v) for lv, (cl, ny) in LEVELS.items() for p, vals in SWEEPS.items() for v in vals]
total = len(tasks)
print("[%s] LED Phase1: %d levels x %d param-sweeps = %d tasks" % (
    datetime.now().strftime("%H:%M:%S"), len(LEVELS), sum(len(v) for v in SWEEPS.values()), total))

results = []
with ThreadPoolExecutor(max_workers=8) as ex:
    futures = {ex.submit(run, *t): t for t in tasks}
    done = 0; t0 = time.time()
    for f in as_completed(futures):
        lv, p, v, auc, f1 = f.result()
        results.append((lv, p, v, auc, f1))
        done += 1
        rate = done / (time.time() - t0) if time.time() > t0 else 0
        eta = (total - done) / rate / 60 if rate > 0 else 0
        print("[%d/%d %d%% | %.1ft/s | ETA %.0fm] %s/%s=%s AUC=%.4f" % (
            done, total, done*100/total, rate, eta, lv, p, v, auc), flush=True)

df = pd.DataFrame(results, columns=["level","param","value","auc","f1"])
df.to_csv("%s/phase1_led.csv" % OUT, index=False)

# Show per-level bests
for p in ["r","tau","sigma","alpha"]:
    print("\n-- Best %s per level --" % p)
    for lv in LEVELS:
        sub = df[(df["level"]==lv)&(df["param"]==p)]
        if len(sub)==0: continue
        best = sub.loc[sub["auc"].idxmax()]
        print("  %s: %s=%-6s AUC=%.4f" % (lv, p, str(best["value"]), best["auc"]))
    # Mean across levels
    mean_aucs = {v: df[(df["param"]==p)&(df["value"]==v)]["auc"].mean() for v in SWEEPS[p]}
    best_mean = max(mean_aucs, key=mean_aucs.get)
    print("  MEAN best: %s=%-6s AUC=%.4f" % (p, str(best_mean), mean_aucs[best_mean]))

print("\nDONE")
