"""FPGA radius constraint comparison: fix r=1 or r=2, sweep tau, compare AUC."""
import subprocess, os
import pandas as pd
from pathlib import Path

PY = r"D:/software/Anaconda_envs/envs/myEVS/python.exe"
THR = "0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,8"
TAUS = [2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000]

# (label, clean, noisy, w, h)
DATASETS = [
    ("Drive_1hz",
     r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_1hz_ed24_withlabel/driving_noise_1hz_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_1hz_ed24_withlabel/driving_noise_1hz_labeled.npy",
     346, 260),
    ("Drive_2hz",
     r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_2hz_ed24_withlabel/driving_noise_2hz_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_2hz_ed24_withlabel/driving_noise_2hz_labeled.npy",
     346, 260),
    ("Drive_5hz",
     r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_5hz_ed24_withlabel/driving_noise_5hz_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_5hz_ed24_withlabel/driving_noise_5hz_labeled.npy",
     346, 260),
    ("Drive_8hz",
     r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_8hz_ed24_withlabel/driving_noise_8hz_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_8hz_ed24_withlabel/driving_noise_8hz_labeled.npy",
     346, 260),
    ("ED24_Ped_light",
     r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_1.8_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_1.8.npy",
     346, 260),
    ("ED24_Ped_heavy",
     r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3.npy",
     346, 260),
    ("DVSCLEAN_444_r100",
     r"D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy/MAH00444/ratio100/MAH00444_ratio100_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy/MAH00444/ratio100/MAH00444_ratio100_labeled.npy",
     1280, 720),
    ("LED_scene100",
     r"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/scene_100/slices_00031_00040_100ms/scene_100_100ms_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/scene_100/slices_00031_00040_100ms/scene_100_100ms_labeled.npy",
     1280, 720),
]

# Algorithms: (method, radius_list)
ALGS = [
    ("baf", [1]),
    ("stcf", [1]),
    ("knoise", [1]),
    ("ebf", [1, 2]),
    ("n149", [1, 2]),
]

# Known optimal AUCs from README2 (from full sweeps)
OPT_AUC = {
    ("Drive_1hz", "baf"): 0.9166,
    ("Drive_1hz", "stcf"): 0.9484,
    ("Drive_1hz", "knoise"): 0.6359,
    ("Drive_1hz", "ebf"): 0.9484,
    ("Drive_1hz", "n149"): 0.9381,
    ("Drive_2hz", "baf"): 0.9029,
    ("Drive_2hz", "stcf"): 0.9445,
    ("Drive_2hz", "knoise"): 0.6265,
    ("Drive_2hz", "ebf"): 0.9472,
    ("Drive_2hz", "n149"): 0.9386,
    ("Drive_5hz", "baf"): 0.8651,
    ("Drive_5hz", "stcf"): 0.9309,
    ("Drive_5hz", "knoise"): 0.6239,
    ("Drive_5hz", "ebf"): 0.9408,
    ("Drive_5hz", "n149"): 0.9416,
    ("Drive_8hz", "baf"): 0.8379,
    ("Drive_8hz", "stcf"): 0.9229,
    ("Drive_8hz", "knoise"): 0.6214,
    ("Drive_8hz", "ebf"): 0.9374,
    ("Drive_8hz", "n149"): 0.9418,
    ("ED24_Ped_light", "baf"): 0.9119,
    ("ED24_Ped_light", "stcf"): 0.9460,
    ("ED24_Ped_light", "knoise"): 0.7130,
    ("ED24_Ped_light", "ebf"): 0.9416,
    ("ED24_Ped_light", "n149"): 0.9565,
    ("ED24_Ped_heavy", "baf"): 0.8161,
    ("ED24_Ped_heavy", "stcf"): 0.8791,
    ("ED24_Ped_heavy", "knoise"): 0.6417,
    ("ED24_Ped_heavy", "ebf"): 0.9099,
    ("ED24_Ped_heavy", "n149"): 0.9406,
    ("DVSCLEAN_444_r100", "baf"): 0.9479,
    ("DVSCLEAN_444_r100", "stcf"): 0.9898,
    ("DVSCLEAN_444_r100", "knoise"): 0.6389,
    ("DVSCLEAN_444_r100", "ebf"): 0.9940,
    ("DVSCLEAN_444_r100", "n149"): 0.9970,
    ("LED_scene100", "baf"): 0.7994,
    ("LED_scene100", "stcf"): 0.8841,
    ("LED_scene100", "knoise"): 0.5323,
    ("LED_scene100", "ebf"): 0.8569,
    ("LED_scene100", "n149"): 0.9133,
}

all_rows = []

for ds_label, clean, noisy, w, h in DATASETS:
    print(f"\n{'='*70}")
    print(f"  {ds_label}  ({w}x{h})")
    print(f"{'='*70}")
    print(f"{'Method':<8} {'r':>2} {'Best tau':>8} {'AUC(r-cons)':>12} {'Opt AUC':>10} {'Δ':>10}  {'Δ%':>8}")
    print("-" * 68)

    for method, r_list in ALGS:
        for r in r_list:
            best_auc = 0.0
            best_tau = 0
            for tau in TAUS:
                tag = f"fpga_{ds_label}_{method}_r{r}_tau{tau}"
                out_csv = f"data/_fpga_{tag}.csv"
                cmd = [PY, '-m', 'myevs.cli', 'roc',
                       '--clean', clean, '--noisy', noisy,
                       '--assume', 'npy', '--width', str(w), '--height', str(h),
                       '--tick-ns', '1000', '--engine', 'cpp',
                       '--method', method, '--radius-px', str(r), '--time-us', str(tau),
                       '--param', 'min-neighbors', '--values', THR,
                       '--match-us', '0', '--match-bin-radius', '0',
                       '--tag', tag, '--out-csv', out_csv, '--append']
                timeout = 600 if w > 640 else 300
                try:
                    subprocess.run(cmd, check=True, timeout=timeout, capture_output=True)
                    df = pd.read_csv(out_csv)
                    auc = float(df.loc[df['auc'].idxmax()]['auc'])
                    if auc > best_auc:
                        best_auc = auc
                        best_tau = tau
                except Exception as e:
                    print(f"  FAIL {method} r={r} tau={tau}: {e}")
                    continue

            opt_auc = OPT_AUC.get((ds_label, method), None)
            if opt_auc:
                delta = best_auc - opt_auc
                delta_pct = (delta / opt_auc) * 100
                flag = " ***" if abs(delta) > 0.01 else ""
                print(f"{method:<8} {r:>2} tau={best_tau:>6} {best_auc:>12.4f} {opt_auc:>10.4f} {delta:>+10.4f}{flag} {delta_pct:>+7.1f}%")
                all_rows.append((ds_label, method, r, best_tau, best_auc, opt_auc, delta))
            else:
                print(f"{method:<8} {r:>2} tau={best_tau:>6} {best_auc:>12.4f} {'N/A':>10}")

# Summary
print(f"\n\n{'='*80}")
print(f"  SUMMARY: N149 at r=1/r=2 vs other algorithms at constrained radius")
print(f"{'='*80}")

for ds_label in set(r[0] for r in all_rows):
    subset = [r for r in all_rows if r[0] == ds_label]
    print(f"\n--- {ds_label} ---")
    print(f"{'Method':<8} {'r':>2} {'AUC(r-c)':>10} {'Opt AUC':>10} {'Δ':>10}")
    subset_sorted = sorted(subset, key=lambda x: x[4], reverse=True)
    for row in subset_sorted:
        flag = " ***" if abs(row[6]) > 0.01 else ""
        print(f"{row[1]:<8} {row[2]:>2} {row[4]:>10.4f} {row[5]:>10.4f} {row[6]:>+10.4f}{flag}")

print("\nDONE")
df = pd.DataFrame(all_rows, columns=["dataset","method","r","best_tau","auc_constrained","auc_optimal","delta"])
df.to_csv("data/_fpga_radius_cmp.csv", index=False)
