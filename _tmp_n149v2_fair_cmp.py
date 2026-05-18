"""N149 v2 vs original: fair comparison at r=2 on Driving datasets.
Same data, same (r,tau), same thresholds. Only difference: beta/sfrac via env var.
"""
import subprocess, os
import pandas as pd

PY = r"D:/software/Anaconda_envs/envs/myEVS/python.exe"
THR = "0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,8"
R = 2
TAUS = [8000, 16000, 32000, 64000, 128000, 256000]

DRIVE = [
    ("Drive_1hz",
     r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_1hz_ed24_withlabel/driving_noise_1hz_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_1hz_ed24_withlabel/driving_noise_1hz_labeled.npy"),
    ("Drive_2hz",
     r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_2hz_ed24_withlabel/driving_noise_2hz_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_2hz_ed24_withlabel/driving_noise_2hz_labeled.npy"),
    ("Drive_5hz",
     r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_5hz_ed24_withlabel/driving_noise_5hz_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_5hz_ed24_withlabel/driving_noise_5hz_labeled.npy"),
    ("Drive_8hz",
     r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_8hz_ed24_withlabel/driving_noise_8hz_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_8hz_ed24_withlabel/driving_noise_8hz_labeled.npy"),
]

BASE_CMD = [PY, '-m', 'myevs.cli', 'roc',
            '--assume', 'npy', '--width', '346', '--height', '260',
            '--tick-ns', '1000', '--engine', 'cpp', '--method', 'n149',
            '--match-us', '0', '--match-bin-radius', '0']

all_rows = []

for label, clean, noisy in DRIVE:
    print(f"\n{'='*60}")
    print(f"  {label}  (r=2)")
    print(f"{'='*60}")
    print(f"{'tau(us)':>10}  {'v2 AUC':>8}  {'orig AUC':>8}  {'Δ(v2-orig)':>12}")
    print("-" * 45)

    for tau in TAUS:
        tag_v2 = f"v2_{label}_r2_tau{tau}"
        tag_orig = f"orig_{label}_r2_tau{tau}"
        csv_v2 = f"data/_cmp_v2_{tag_v2}.csv"
        csv_orig = f"data/_cmp_orig_{tag_orig}.csv"

        common = ['--radius-px', str(R), '--time-us', str(tau),
                  '--param', 'min-neighbors', '--values', THR,
                  '--clean', clean, '--noisy', noisy]

        # v2 (default: beta/sfrac off)
        cmd_v2 = BASE_CMD + common + ['--tag', tag_v2, '--out-csv', csv_v2, '--append']
        subprocess.run(cmd_v2, check=True, timeout=300)
        auc_v2 = float(pd.read_csv(csv_v2).loc[pd.read_csv(csv_v2)['auc'].idxmax()]['auc'])

        # original (beta/sfrac on)
        env_orig = os.environ.copy()
        env_orig["MYEVS_N149_USE_BETA"] = "1"
        env_orig["MYEVS_N149_USE_SFRAC"] = "1"
        cmd_orig = BASE_CMD + common + ['--tag', tag_orig, '--out-csv', csv_orig, '--append']
        subprocess.run(cmd_orig, check=True, timeout=300, env=env_orig)
        auc_orig = float(pd.read_csv(csv_orig).loc[pd.read_csv(csv_orig)['auc'].idxmax()]['auc'])

        delta = auc_v2 - auc_orig
        flag = " ***" if abs(delta) > 0.002 else ""
        print(f"{tau:>10}  {auc_v2:>8.4f}  {auc_orig:>8.4f}  {delta:>+12.4f}{flag}")
        all_rows.append((label, R, tau, auc_v2, auc_orig, delta))

# Summary per level
print(f"\n\n{'='*70}")
print(f"  SUMMARY: v2 vs original at r=2")
print(f"{'='*70}")
print(f"{'Level':<14} {'Best tau v2':>12} {'v2 AUC':>8} {'Best tau orig':>12} {'orig AUC':>8} {'Δ':>10}")
print("-" * 68)
for label, _, _, _, _, _ in [r for r in all_rows if r[2] == TAUS[0]]:
    subset = [r for r in all_rows if r[0] == label]
    best_v2 = max(subset, key=lambda x: x[3])
    best_orig = max(subset, key=lambda x: x[4])
    delta_best = best_v2[3] - best_orig[4]
    print(f"{label:<14} tau={best_v2[2]:>6}  {best_v2[3]:>8.4f}  tau={best_orig[2]:>6}  {best_orig[4]:>8.4f}  {delta_best:>+10.4f}")

print("\nDONE")
df = pd.DataFrame(all_rows, columns=["level","r","tau","auc_v2","auc_orig","delta"])
df.to_csv("data/_n149_v2_vs_orig_r2.csv", index=False)
