"""N149 v2.1 full run on all datasets, 16-bit hot_state."""
import subprocess, os
import pandas as pd

PY = r"D:/software/Anaconda_envs/envs/myEVS/python.exe"
THR = "0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,8"
HOT_BITS = 16

TASKS = [
    ("Drive_1hz",  r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_1hz_ed24_withlabel/driving_noise_1hz_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_1hz_ed24_withlabel/driving_noise_1hz_labeled.npy", 346,260, 2, 32000, 0.9381),
    ("Drive_2hz",  r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_2hz_ed24_withlabel/driving_noise_2hz_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_2hz_ed24_withlabel/driving_noise_2hz_labeled.npy", 346,260, 2, 32000, 0.9386),
    ("Drive_5hz",  r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_5hz_ed24_withlabel/driving_noise_5hz_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_5hz_ed24_withlabel/driving_noise_5hz_labeled.npy", 346,260, 2, 32000, 0.9416),
    ("Drive_8hz",  r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_8hz_ed24_withlabel/driving_noise_8hz_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_8hz_ed24_withlabel/driving_noise_8hz_labeled.npy", 346,260, 2, 32000, 0.9418),
    ("Ped_light",  r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_1.8_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_1.8.npy", 346,260, 5, 256000, 0.9565),
    ("Ped_heavy",  r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3.npy", 346,260, 5, 256000, 0.9406),
    ("Bike_light", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_1.8_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_1.8.npy", 346,260, 5, 512000, 0.9845),
    ("Bike_lmid",  r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_2.1_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_2.1.npy", 346,260, 5, 512000, 0.9827),
    ("Bike_mid",   r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_2.5_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_2.5.npy", 346,260, 5, 512000, 0.9787),
    ("DVSCLEAN",   r"D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy/MAH00444/ratio100/MAH00444_ratio100_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy/MAH00444/ratio100/MAH00444_ratio100_labeled.npy", 1280,720, 5, 128000, 0.9978),
    ("LED",        r"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/scene_100/slices_00031_00040_100ms/scene_100_100ms_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/scene_100/slices_00031_00040_100ms/scene_100_100ms_labeled.npy", 1280,720, 2, 16000, 0.9133),
]

BASE = [PY, '-m', 'myevs.cli', 'roc', '--assume', 'npy', '--tick-ns', '1000',
        '--engine', 'cpp', '--method', 'n149',
        '--param', 'min-neighbors', '--values', THR,
        '--match-us', '0', '--match-bin-radius', '0', '--append']

results = []
for label, clean, noisy, w, h, r, tau, old_auc in TASKS:
    env = os.environ.copy()
    env["MYEVS_N149_HOT_BITS"] = str(HOT_BITS)
    tag = f"v21_16b_{label}"
    out_csv = f"data/_v21full_{tag}.csv"
    cmd = BASE + ['--clean', clean, '--noisy', noisy,
                  '--width', str(w), '--height', str(h),
                  '--radius-px', str(r), '--time-us', str(tau),
                  '--tag', tag, '--out-csv', out_csv]
    print(f"{label}...", end=" ", flush=True)
    subprocess.run(cmd, check=True, timeout=600, env=env)
    df = pd.read_csv(out_csv)
    best = df.loc[df['auc'].idxmax()]
    auc = float(best['auc']); f1 = float(best['f1'])
    delta = auc - old_auc
    flag = " ***" if abs(delta) > 0.002 else ""
    print(f"AUC={auc:.4f} F1={f1:.4f} Δ={delta:+.4f}{flag} (old={old_auc:.4f})")
    results.append((label, old_auc, auc, f1, delta, flag))

print(f"\n{'='*60}")
print(f"  N149 v2.1 (16-bit) vs Original N149")
print(f"{'='*60}")
print(f"{'Dataset':<14} {'Old AUC':>8} {'v2.1 AUC':>10} {'Δ':>10}")
print("-" * 46)
for label, old, new, f1, delta, flag in results:
    print(f"{label:<14} {old:>8.4f} {new:>10.4f} {delta:>+10.4f}{flag}")
print("\nDONE")
pd.DataFrame(results, columns=["dataset","old_auc","new_auc","f1","delta","flag"]).to_csv("data/_n149v21_16b_summary.csv", index=False)
