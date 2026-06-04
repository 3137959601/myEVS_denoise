"""no_polarity component ablation via Python API (no subprocess). 12-threaded."""
import sys, os, time, numpy as np
sys.path.insert(0, r'D:/hjx_workspace/scientific_reserach/projects/myEVS/src')
from concurrent.futures import ThreadPoolExecutor, as_completed
from myevs.denoise.pipeline import DenoiseConfig, denoise_stream
from myevs.timebase import TimeBase
from myevs.events import EventBatch
from myevs.metrics.roc import compute_roc_auc

THR = [0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,8]
OUT = r"D:/hjx_workspace/scientific_reserach/projects/myEVS/data/Hyperparameter ablation_study"

DATASETS = [
    ("drive","1hz", r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_1hz_ed24_withlabel/driving_noise_1hz_signal_only.npy", r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_1hz_ed24_withlabel/driving_noise_1hz_labeled.npy", 346,260,2,32000,1.75,0.05),
    ("drive","3hz", r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_3hz_ed24_withlabel/driving_noise_3hz_signal_only.npy", r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_3hz_ed24_withlabel/driving_noise_3hz_labeled.npy", 346,260,2,32000,1.75,0.05),
    ("drive","5hz", r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_5hz_ed24_withlabel/driving_noise_5hz_signal_only.npy", r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_5hz_ed24_withlabel/driving_noise_5hz_labeled.npy", 346,260,2,32000,1.75,0.05),
    ("drive","7hz", r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_7hz_ed24_withlabel/driving_noise_7hz_signal_only.npy", r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_7hz_ed24_withlabel/driving_noise_7hz_labeled.npy", 346,260,2,32000,1.75,0.05),
    ("drive","10hz",r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_10hz_ed24_withlabel/driving_noise_10hz_signal_only.npy", r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_10hz_ed24_withlabel/driving_noise_10hz_labeled.npy", 346,260,2,32000,1.75,0.05),
    ("ped","1.8",r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_1.8_signal_only.npy", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_1.8.npy", 346,260,5,256000,2.75,0.25),
    ("ped","2.1",r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_2.1_signal_only.npy", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_2.1.npy", 346,260,5,256000,2.75,0.25),
    ("ped","2.5",r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_2.5_signal_only.npy", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_2.5.npy", 346,260,5,256000,2.75,0.25),
    ("ped","3.3",r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3_signal_only.npy", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3.npy", 346,260,5,256000,2.75,0.25),
    ("bike","1.8",r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_1.8_signal_only.npy", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_1.8.npy", 346,260,5,256000,2.75,0.25),
    ("bike","2.1",r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_2.1_signal_only.npy", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_2.1.npy", 346,260,5,256000,2.75,0.25),
    ("bike","2.5",r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_2.5_signal_only.npy", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_2.5.npy", 346,260,5,256000,2.75,0.25),
    ("bike","3.3",r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_3.3_signal_only.npy", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_3.3.npy", 346,260,5,256000,2.75,0.25),
]
for scene in ["MAH00444","MAH00446","MAH00447","MAH00448","MAH00449"]:
    for ratio in ["ratio50","ratio100"]:
        DATASETS.append(("dvsclean",f"{scene}_{ratio}", f"D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy/{scene}/{ratio}/{scene}_{ratio}_signal_only.npy", f"D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy/{scene}/{ratio}/{scene}_{ratio}_labeled.npy", 1280,720,5,128000,2.5,0.25))
for s in ["scene_100","scene_1004","scene_1018","scene_1028","scene_1032","scene_1033","scene_1034","scene_1043","scene_1045","scene_1046"]:
    DATASETS.append(("led",s, f"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/{s}/slices_00031_00040_100ms/{s}_100ms_signal_only.npy", f"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/{s}/slices_00031_00040_100ms/{s}_100ms_labeled.npy", 1280,720,2,8000,2.0,1.0))

tb = TimeBase(tick_ns=1000.0)

def run_one(ds, lv, clean_path, noisy_path, w, h, r, tau, sigma, alpha):
    os.environ["MYEVS_N149_HOT_BITS"] = "16"
    os.environ["MYEVS_N149_SIGMA"] = str(sigma)
    os.environ["MYEVS_N149_ALPHA_FIXED"] = str(alpha)
    os.environ["MYEVS_N149_BLIND"] = "1"

    clean = np.load(clean_path)
    noisy = np.load(noisy_path)
    labels = noisy['label'].astype(np.uint8)

    meta = type('Meta',(),{'width':w,'height':h})()
    cfg = DenoiseConfig(method="18", time_window_us=tau, radius_px=r, min_neighbors=0, show_on=True, show_off=True)

    tpr_list, fpr_list = [], []
    for th in THR:
        cfg2 = DenoiseConfig(method="18", time_window_us=tau, radius_px=r, min_neighbors=th, show_on=True, show_off=True)
        batch = EventBatch(t=noisy['t'].astype(np.uint64), x=noisy['x'].astype(np.uint16),
                           y=noisy['y'].astype(np.uint16), p=noisy['p'].astype(np.int8))
        den = list(denoise_stream(meta, [batch], cfg2, timebase=tb, engine="cpp"))
        if not den: continue
        kept = set(zip(den[0].t.tolist(), den[0].x.tolist(), den[0].y.tolist(), den[0].p.tolist()))
        noisy_set = set(zip(noisy['t'].tolist(), noisy['x'].tolist(), noisy['y'].tolist(), noisy['p'].tolist()))
        # Compute TPR/FPR using labels
        tp = sum(1 for i in range(len(noisy)) if (noisy['t'][i],noisy['x'][i],noisy['y'][i],noisy['p'][i]) in kept and labels[i]>0)
        fp = sum(1 for i in range(len(noisy)) if (noisy['t'][i],noisy['x'][i],noisy['y'][i],noisy['p'][i]) in kept and labels[i]==0)
        pos = int((labels>0).sum()); neg = int((labels==0).sum())
        tpr_list.append(tp/pos if pos else 0)
        fpr_list.append(fp/neg if neg else 0)

    # Simple trapezoidal AUC
    points = sorted(zip(fpr_list, tpr_list))
    auc = 0.0
    for i in range(len(points)-1):
        auc += (points[i+1][0]-points[i][0]) * (points[i][1]+points[i+1][1]) / 2
    return ds, lv, auc

total = len(DATASETS)
print(f"no_polarity API: {total} tasks")
t0 = time.time()
results = []
with ThreadPoolExecutor(max_workers=12) as ex:
    futures = {ex.submit(run_one, *d): d for d in DATASETS}
    done = 0
    for f in as_completed(futures):
        ds, lv, auc = f.result()
        results.append((ds, lv, auc))
        done += 1
        rate = done/(time.time()-t0) if time.time()>t0 else 0
        print(f"[{done}/{total} | {rate:.1f}t/s] {ds}/{lv} no_pol AUC={auc:.4f}", flush=True)

import pandas as pd
df = pd.DataFrame(results, columns=["dataset","level","auc"])
df.to_csv(f"{OUT}/component_ablation_no_polarity.csv", index=False)
base_df = pd.read_csv(f"{OUT}/component_ablation.csv")
base_df = base_df[base_df.variant=="baseline"][["dataset","level","auc"]].rename(columns={"auc":"baseline_auc"})
merged = df.merge(base_df, on=["dataset","level"])
merged["delta"] = merged["auc"] - merged["baseline_auc"]
print("\n=== no_polarity MEAN per dataset ===")
for ds in ["drive","ped","bike","dvsclean","led"]:
    sub = merged[merged.dataset==ds]
    print(f"  {ds}: mean AUC={sub.auc.mean():.4f}  Δ={sub.delta.mean():+.4f}")
print("DONE")
