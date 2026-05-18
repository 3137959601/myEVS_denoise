"""N149 v2.1 comparison: 32-bit + 16-bit-optimal vs original N149."""
import subprocess, os
import pandas as pd

PY = r"D:/software/Anaconda_envs/envs/myEVS/python.exe"
THR = "0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,8"

# (label, clean, noisy, w, h, r, tau, old_auc, Tr_factor)
# Tr_factor: 1.0 for short tau, 0.25 for long tau
TASKS = [
    ("Drive_1hz",  r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_1hz_ed24_withlabel/driving_noise_1hz_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_1hz_ed24_withlabel/driving_noise_1hz_labeled.npy", 346,260, 2, 32000, 0.9381, 1.0),
    ("Drive_2hz",  r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_2hz_ed24_withlabel/driving_noise_2hz_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_2hz_ed24_withlabel/driving_noise_2hz_labeled.npy", 346,260, 2, 32000, 0.9386, 1.0),
    ("Drive_5hz",  r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_5hz_ed24_withlabel/driving_noise_5hz_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_5hz_ed24_withlabel/driving_noise_5hz_labeled.npy", 346,260, 2, 32000, 0.9416, 1.0),
    ("Drive_8hz",  r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_8hz_ed24_withlabel/driving_noise_8hz_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_8hz_ed24_withlabel/driving_noise_8hz_labeled.npy", 346,260, 2, 32000, 0.9418, 1.0),
    ("Ped_light",  r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_1.8_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_1.8.npy", 346,260, 5, 256000, 0.9565, 0.25),
    ("Ped_heavy",  r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3.npy", 346,260, 5, 256000, 0.9406, 0.25),
    ("Bike_light", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_1.8_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_1.8.npy", 346,260, 5, 512000, 0.9845, 0.25),
    ("Bike_lmid",  r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_2.1_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_2.1.npy", 346,260, 5, 512000, 0.9827, 0.25),
    ("Bike_mid",   r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_2.5_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_2.5.npy", 346,260, 5, 512000, 0.9787, 0.25),
    ("DVSCLEAN",   r"D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy/MAH00444/ratio100/MAH00444_ratio100_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy/MAH00444/ratio100/MAH00444_ratio100_labeled.npy", 1280,720, 5, 128000, 0.9978, 1.0),
    ("LED",        r"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/scene_100/slices_00031_00040_100ms/scene_100_100ms_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/scene_100/slices_00031_00040_100ms/scene_100_100ms_labeled.npy", 1280,720, 2, 16000, 0.9133, 1.0),
]

BASE = [PY, '-m', 'myevs.cli', 'roc', '--assume', 'npy', '--tick-ns', '1000',
        '--engine', 'cpp', '--method', 'n149',
        '--param', 'min-neighbors', '--values', THR,
        '--match-us', '0', '--match-bin-radius', '0', '--append']

results = []
for label, clean, noisy, w, h, r, tau, old_auc, tr_factor in TASKS:
    row = [label, old_auc]

    # (1) v2.1 32-bit (same as original bit width)
    env = os.environ.copy()
    env["MYEVS_N149_HOT_BITS"] = "31"
    if tr_factor != 1.0: env["MYEVS_N149_U_DENOM"] = str(tr_factor)
    csv = f"data/_v21cmp_32b_{label}.csv"
    cmd = BASE + ['--clean', clean, '--noisy', noisy, '--width', str(w), '--height', str(h),
                  '--radius-px', str(r), '--time-us', str(tau),
                  '--tag', f'v21_32b_{label}', '--out-csv', csv]
    subprocess.run(cmd, check=True, timeout=600, env=env, capture_output=True)
    auc32 = float(pd.read_csv(csv).loc[pd.read_csv(csv)['auc'].idxmax()]['auc'])
    row.append(auc32); row.append(auc32 - old_auc)

    # (2) v2.1 16-bit with optimal Tr
    env = os.environ.copy()
    env["MYEVS_N149_HOT_BITS"] = "16"
    if tr_factor != 1.0: env["MYEVS_N149_U_DENOM"] = str(tr_factor)
    csv = f"data/_v21cmp_16b_{label}.csv"
    cmd = BASE + ['--clean', clean, '--noisy', noisy, '--width', str(w), '--height', str(h),
                  '--radius-px', str(r), '--time-us', str(tau),
                  '--tag', f'v21_16b_{label}', '--out-csv', csv]
    subprocess.run(cmd, check=True, timeout=600, env=env, capture_output=True)
    auc16 = float(pd.read_csv(csv).loc[pd.read_csv(csv)['auc'].idxmax()]['auc'])
    row.append(auc16); row.append(auc16 - old_auc)

    # (3) v2.1 16-bit Tr=τ (baseline)
    env = os.environ.copy()
    env["MYEVS_N149_HOT_BITS"] = "16"
    csv = f"data/_v21cmp_16b_tr1_{label}.csv"
    cmd = BASE + ['--clean', clean, '--noisy', noisy, '--width', str(w), '--height', str(h),
                  '--radius-px', str(r), '--time-us', str(tau),
                  '--tag', f'v21_16b_tr1_{label}', '--out-csv', csv]
    subprocess.run(cmd, check=True, timeout=600, env=env, capture_output=True)
    auc16_tr1 = float(pd.read_csv(csv).loc[pd.read_csv(csv)['auc'].idxmax()]['auc'])
    row.append(auc16_tr1); row.append(auc16_tr1 - old_auc)

    print(f"{label:<12} old={old_auc:.4f}  32b={auc32:.4f}({auc32-old_auc:+.4f})  16b_opt={auc16:.4f}({auc16-old_auc:+.4f})  16b_tr1={auc16_tr1:.4f}({auc16_tr1-old_auc:+.4f})")
    results.append(row)

print(f"\n{'='*80}")
print(f"  N149 v2.1 (32-bit) vs v2.1 (16-bit opt Tr) vs Original N149")
print(f"{'='*80}")
print(f"{'Dataset':<12} {'orig':>8} {'v2.1_32b':>10} {'Δ32':>8} {'v2.1_16b':>10} {'Δ16':>8} {'16b_tr=τ':>10} {'Δtr1':>8} {'Tr':>6}")
for r in results:
    label, old, a32, d32, a16, d16, a16t1, d16t1 = r
    # find tr_factor
    tr_factor = [t[8] for t in TASKS if t[0]==label][0]
    print(f"{label:<12} {old:>8.4f} {a32:>10.4f} {d32:>+8.4f} {a16:>10.4f} {d16:>+8.4f} {a16t1:>10.4f} {d16t1:>+8.4f}  {tr_factor}")
print("\nDONE")
pd.DataFrame(results, columns=["dataset","orig","v21_32b","d32","v21_16b","d16","v21_16b_tr1","d16_tr1"]).to_csv("data/_v21_full_cmp.csv", index=False)
