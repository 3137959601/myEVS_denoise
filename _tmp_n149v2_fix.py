"""Fix: re-run ED24 Bicycle v2 validation with correct file mappings + DVSCLEAN + LED."""
import subprocess, os
import pandas as pd

PY = r"D:/software/Anaconda_envs/envs/myEVS/python.exe"
THR = "0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,8"

# Corrected tasks: (label, clean, noisy, w, h, r, tau, old_auc)
TASKS = [
    # ED24 Bicycle — correct file mappings
    # light=1.8, light_mid=2.1, mid=2.5, heavy=3.3(no npy)
    ("ED24_Bike_light",
     r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_1.8_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_1.8.npy",
     346, 260, [5], [512000], 0.9845),
    ("ED24_Bike_light_mid",
     r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_2.1_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_2.1.npy",
     346, 260, [5], [512000], 0.9827),
    ("ED24_Bike_mid",
     r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_2.5_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_2.5.npy",
     346, 260, [5], [512000], 0.9787),
    # DVSCLEAN
    ("DVSCLEAN_444_r100",
     r"D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy/MAH00444/ratio100/MAH00444_ratio100_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy/MAH00444/ratio100/MAH00444_ratio100_labeled.npy",
     1280, 720, [5], [128000], 0.9978),
    # LED
    ("LED_scene100",
     r"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/scene_100/slices_00031_00040_100ms/scene_100_100ms_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/scene_100/slices_00031_00040_100ms/scene_100_100ms_labeled.npy",
     1280, 720, [2], [16000], 0.9133),
]

results = []
for label, clean, noisy, w, h, r_list, tau_list, old_auc in TASKS:
    print(f"\n{'='*60}")
    print(f"  {label}  (old AUC={old_auc:.4f})")
    print(f"{'='*60}")
    best_auc = 0.0
    best_tag = ""
    for r in r_list:
        for tau in tau_list:
            tag = f"n149v2_{label}_r{r}_tau{tau}"
            out_csv = f"data/_n149v2_{tag}.csv"
            cmd = [PY, '-m', 'myevs.cli', 'roc',
                   '--clean', clean, '--noisy', noisy,
                   '--assume', 'npy', '--width', str(w), '--height', str(h),
                   '--tick-ns', '1000', '--engine', 'cpp',
                   '--method', 'n149', '--radius-px', str(r), '--time-us', str(tau),
                   '--param', 'min-neighbors', '--values', THR,
                   '--match-us', '0', '--match-bin-radius', '0',
                   '--tag', tag, '--out-csv', out_csv, '--append']
            timeout = 900 if w > 640 else 600
            print(f"  {tag}...", end=" ", flush=True)
            subprocess.run(cmd, check=True, timeout=timeout)
            df = pd.read_csv(out_csv)
            auc = float(df.loc[df['auc'].idxmax()]['auc'])
            print(f"AUC={auc:.4f}")
            if auc > best_auc:
                best_auc = auc
                best_tag = tag
    delta = best_auc - old_auc
    status = "SAME" if abs(delta) < 0.002 else ("BETTER" if delta > 0 else "WORSE")
    print(f"  BEST: {best_tag}  AUC={best_auc:.4f}  Δ={delta:+.4f}  [{status}]")
    results.append((label, old_auc, best_auc, delta, status, best_tag))

# Summary
print(f"\n\n{'='*70}")
print(f"  SUMMARY: N149 v2 vs N149 original")
print(f"{'='*70}")
print(f"{'Dataset':<24} {'Old AUC':>8} {'New AUC':>8} {'ΔAUC':>10}  Status")
print("-" * 68)
for label, old_auc, new_auc, delta, status, tag in results:
    flag = " ***" if abs(delta) > 0.002 else ""
    print(f"{label:<24} {old_auc:>8.4f} {new_auc:>8.4f} {delta:>+10.4f}{flag}  {status}")

print("\nDONE")
pd.DataFrame(results, columns=["dataset","old_auc","new_auc","delta","status","best_tag"]).to_csv("data/_n149v2_fix_summary.csv", index=False)
