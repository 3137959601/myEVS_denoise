"""Fix PFD (r=1) for DVSCLEAN/LED + EvFlow fine-sweep + apply."""
import subprocess, os, time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

PY = r"D:/software/Anaconda_envs/envs/myEVS/python.exe"
THR = "0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,8"
THR_LITE = "0,2,4,6,8"

OUT = r"D:/hjx_workspace/scientific_reserach/projects/myEVS/data/missing_alg"

# ===== Part 1: PFD r=1 for DVSCLEAN and LED =====
def pfd_task(ds_key, lv, clean, noisy, w, h, tau):
    csv = f"{OUT}/{ds_key}/miss_{ds_key}_{lv}_pfd_r1.csv"
    cmd = [PY,'-m','myevs.cli','roc','--clean',clean,'--noisy',noisy,
           '--assume','npy','--width',str(w),'--height',str(h),
           '--tick-ns','1000','--engine','cpp','--method','pfd',
           '--radius-px','1','--time-us',str(tau),
           '--param','min-neighbors','--values',THR,
           '--match-us','0','--match-bin-radius','0',
           '--refractory-us','2','--pfd-mode','a',
           '--tag',f'pfd_r1_{ds_key}_{lv}','--out-csv',csv,'--append']
    subprocess.run(cmd, check=True, timeout=900, capture_output=True)
    df = pd.read_csv(csv)
    return ds_key, lv, 'pfd_r1', float(df.loc[df.auc.idxmax()].auc)

# ===== Part 2: EvFlow fine-sweep (tau sweep on one representative) =====
def evflow_sweep(clean, noisy, w, h, r, tau_list, tag_prefix, out_csv):
    best_auc, best_tau = 0, 0
    for tau in tau_list:
        csv = out_csv.replace('.csv', f'_tau{tau}.csv')
        cmd = [PY,'-m','myevs.cli','roc','--clean',clean,'--noisy',noisy,
               '--assume','npy','--width',str(w),'--height',str(h),
               '--tick-ns','1000','--engine','cpp','--method','evflow',
               '--radius-px',str(r),'--time-us',str(tau),
               '--param','min-neighbors','--values',THR_LITE,
               '--match-us','0','--match-bin-radius','0',
               '--tag',f'{tag_prefix}_tau{tau}','--out-csv',csv,'--append']
        try:
            subprocess.run(cmd, check=True, timeout=600, capture_output=True)
            df = pd.read_csv(csv); auc = float(df.loc[df.auc.idxmax()].auc)
            print(f'  EvFlow tau={tau}: AUC={auc:.4f}')
            if auc > best_auc: best_auc, best_tau = auc, tau
        except: pass
    return best_tau, best_auc

def evflow_task(ds_key, lv, clean, noisy, w, h, r, tau):
    csv = f"{OUT}/{ds_key}/miss_{ds_key}_{lv}_evflow_v2.csv"
    cmd = [PY,'-m','myevs.cli','roc','--clean',clean,'--noisy',noisy,
           '--assume','npy','--width',str(w),'--height',str(h),
           '--tick-ns','1000','--engine','cpp','--method','evflow',
           '--radius-px',str(r),'--time-us',str(tau),
           '--param','min-neighbors','--values',THR,
           '--match-us','0','--match-bin-radius','0',
           '--tag',f'evflow_v2_{ds_key}_{lv}','--out-csv',csv,'--append']
    subprocess.run(cmd, check=True, timeout=600, capture_output=True)
    df = pd.read_csv(csv)
    return ds_key, lv, 'evflow_v2', float(df.loc[df.auc.idxmax()].auc)

# === Main ===
tasks = []

# PFD r=1: DVSCLEAN 10 scenes
for scene in ["MAH00444","MAH00446","MAH00447","MAH00448","MAH00449"]:
    for ratio in ["ratio50","ratio100"]:
        lv = f"{scene}_{ratio}"
        clean = f"D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy/{scene}/{ratio}/{scene}_{ratio}_signal_only.npy"
        noisy = f"D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy/{scene}/{ratio}/{scene}_{ratio}_labeled.npy"
        tasks.append(('pfd','dvsclean',lv,clean,noisy,1280,720,8000))

# PFD r=1: LED 10 scenes
for s in ["scene_100","scene_1004","scene_1018","scene_1028","scene_1032","scene_1033","scene_1034","scene_1043","scene_1045","scene_1046"]:
    clean = f"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/{s}/slices_00031_00040_100ms/{s}_100ms_signal_only.npy"
    noisy = f"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/{s}/slices_00031_00040_100ms/{s}_100ms_labeled.npy"
    tasks.append(('pfd','led',s,clean,noisy,1280,720,8000))

# EvFlow: Driving 7hz/10hz (use tau=16K like 8hz table)
for lv in ["7hz","10hz"]:
    clean = f"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_{lv}_ed24_withlabel/driving_noise_{lv}_signal_only.npy"
    noisy = f"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_{lv}_ed24_withlabel/driving_noise_{lv}_labeled.npy"
    tasks.append(('evflow','drive',lv,clean,noisy,346,260,2,16000))

# EvFlow: LED 10 scenes (r=2, tau=8K like LED optimal)
for s in ["scene_100","scene_1004","scene_1018","scene_1028","scene_1032","scene_1033","scene_1034","scene_1043","scene_1045","scene_1046"]:
    clean = f"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/{s}/slices_00031_00040_100ms/{s}_100ms_signal_only.npy"
    noisy = f"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/{s}/slices_00031_00040_100ms/{s}_100ms_labeled.npy"
    tasks.append(('evflow','led',s,clean,noisy,1280,720,2,8000))

total = len(tasks)
print(f"Fix tasks: {total} (PFD:{sum(1 for t in tasks if t[0]=='pfd')} EvFlow:{sum(1 for t in tasks if t[0]=='evflow')})")
t0 = time.time()

results = []
with ThreadPoolExecutor(max_workers=12) as ex:
    futures = {}
    for t in tasks:
        if t[0] == 'pfd':
            futures[ex.submit(pfd_task, t[1], t[2], t[3], t[4], t[5], t[6], t[7])] = t
        else:
            futures[ex.submit(evflow_task, t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8])] = t

    done = 0
    for f in as_completed(futures):
        ds_key, lv, typ, auc = f.result()
        results.append((ds_key, lv, typ, auc))
        done += 1
        rate = done / (time.time()-t0) if time.time()>t0 else 0
        eta = (total-done)/rate/60 if rate>0 else 0
        print(f"[{done}/{total} | {rate:.1f}t/s | ETA {eta:.0f}m] {typ} {ds_key}/{lv} AUC={auc:.4f}", flush=True)

df = pd.DataFrame(results, columns=['dataset','level','type','auc'])
df.to_csv(f"{OUT}/fix_pfd_evflow.csv", index=False)

for typ in ['pfd_r1','evflow_v2']:
    sub = df[df.type==typ]
    if len(sub)==0: continue
    print(f"\n{typ}:")
    for ds in sub.dataset.unique():
        sl = sub[sub.dataset==ds]
        print(f"  {ds}: mean AUC={sl.auc.mean():.4f} (n={len(sl)})")

print("\nDONE")
