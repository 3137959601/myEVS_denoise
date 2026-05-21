"""补跑缺失算法：Drive 7hz/10hz + ED24 2.1 + LED 10 scenes. Multi-threaded."""
import subprocess, os, time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

PY = r"D:/software/Anaconda_envs/envs/myEVS/python.exe"
THR = "0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,8"
OUT = r"D:/hjx_workspace/scientific_reserach/projects/myEVS/data"

# Algorithm configs: (method, radius, tau, extra_args)
ALGS = {
    "baf":     ("baf", 1, 8000, []),
    "stcf":    ("stcf", 2, 32000, []),
    "ebf":     ("ebf", 2, 32000, []),
    "knoise":  ("knoise", 0, 8000, []),  # radius ignored, tau used
    "ynoise":  ("ynoise", 2, 16000, []),
    "ts":      ("ts", 2, 32000, []),
    "evflow":  ("evflow", 2, 16000, []),
    "pfd":     ("pfd", 3, 8000, ["--refractory-us","2","--pfd-mode","a"]),
    "stcf_orig": ("stcf_original", 1, 16000, []),
}

DATASETS = [
    # Drive 7hz/10hz
    ("drive", "7hz", r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_7hz_ed24_withlabel/driving_noise_7hz_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_7hz_ed24_withlabel/driving_noise_7hz_labeled.npy", 346, 260),
    ("drive", "10hz", r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_10hz_ed24_withlabel/driving_noise_10hz_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_10hz_ed24_withlabel/driving_noise_10hz_labeled.npy", 346, 260),
    # ED24 Ped 2.1
    ("ped", "2.1", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_2.1_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_2.1.npy", 346, 260),
    # ED24 Bike 2.1 + 3.3(heavy)
    ("bike", "2.1", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_2.1_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_2.1.npy", 346, 260),
    ("bike", "3.3", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_3.3_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_3.3.npy", 346, 260),
]

# LED 10 scenes
LED_SCENES = ["scene_100","scene_1004","scene_1018","scene_1028","scene_1032","scene_1033","scene_1034","scene_1043","scene_1045","scene_1046"]
for s in LED_SCENES:
    DATASETS.append(("led", s,
        f"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/{s}/slices_00031_00040_100ms/{s}_100ms_signal_only.npy",
        f"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/{s}/slices_00031_00040_100ms/{s}_100ms_labeled.npy",
        1280, 720))

def run_one(ds_key, lv, clean, noisy, w, h, alg_key, alg):
    method, r, tau, extra = alg
    tag = f"miss_{ds_key}_{lv}_{alg_key}"
    out_dir = f"{OUT}/missing_alg/{ds_key}"
    os.makedirs(out_dir, exist_ok=True)
    csv = f"{out_dir}/{tag}.csv"
    cmd = [PY, '-m', 'myevs.cli', 'roc',
           '--clean', clean, '--noisy', noisy,
           '--assume', 'npy', '--width', str(w), '--height', str(h),
           '--tick-ns', '1000', '--engine', 'cpp', '--method', method,
           '--radius-px', str(r), '--time-us', str(tau),
           '--param', 'min-neighbors', '--values', THR,
           '--match-us', '0', '--match-bin-radius', '0',
           '--tag', tag, '--out-csv', csv, '--append'] + extra
    subprocess.run(cmd, check=True, timeout=900, capture_output=True)
    df = pd.read_csv(csv)
    best = df.loc[df["auc"].idxmax()]
    return ds_key, lv, alg_key, float(best["auc"]), float(best["f1"])

# Build tasks
tasks = []
for ds_key, lv, clean, noisy, w, h in DATASETS:
    for alg_key, alg in ALGS.items():
        tasks.append((ds_key, lv, clean, noisy, w, h, alg_key, alg))

total = len(tasks)
print(f"Missing alg sweep: {total} tasks ({len(DATASETS)} datasets x {len(ALGS)} algs)")
t0 = time.time()

results = []
with ThreadPoolExecutor(max_workers=12) as ex:
    futures = {ex.submit(run_one, *t): t for t in tasks}
    done = 0
    for f in as_completed(futures):
        ds_key, lv, alg_key, auc, f1 = f.result()
        results.append((ds_key, lv, alg_key, auc, f1))
        done += 1
        rate = done / (time.time() - t0) if time.time() > t0 else 0
        eta = (total - done) / rate / 60 if rate > 0 else 0
        print(f"[{done}/{total} | {rate:.1f}t/s | ETA {eta:.0f}m] {ds_key}/{lv}/{alg_key} AUC={auc:.4f}", flush=True)

df = pd.DataFrame(results, columns=["dataset","level","algorithm","auc","f1"])
df.to_csv(f"{OUT}/missing_alg/all.csv", index=False)

# Summary
for ds_key in sorted(df.dataset.unique()):
    sub = df[df.dataset == ds_key]
    print(f"\n{ds_key}:")
    for alg in ALGS:
        sl = sub[(sub.algorithm == alg) & (sub.dataset == ds_key)]
        if len(sl) > 0:
            mean_auc = sl.auc.mean()
            print(f"  {alg}: n={len(sl)} mean AUC={mean_auc:.4f}")

print("\nDONE")
