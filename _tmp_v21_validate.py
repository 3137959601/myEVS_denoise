"""Validate u_denom (tau/2 vs tau) and B_denom (1+u2 vs 1+u) on 7 datasets."""
import subprocess, os, itertools
import pandas as pd

PY = r"D:/software/Anaconda_envs/envs/myEVS/python.exe"
THR = "0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,8"

TASKS = [
    ("Drive_1hz", r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_1hz_ed24_withlabel/driving_noise_1hz_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_1hz_ed24_withlabel/driving_noise_1hz_labeled.npy",
     346, 260, 2, 32000),
    ("Drive_8hz", r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_8hz_ed24_withlabel/driving_noise_8hz_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_8hz_ed24_withlabel/driving_noise_8hz_labeled.npy",
     346, 260, 2, 32000),
    ("Ped_light", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_1.8_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_1.8.npy",
     346, 260, 5, 256000),
    ("Ped_heavy", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3.npy",
     346, 260, 5, 256000),
    ("Bike_light", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_1.8_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_1.8.npy",
     346, 260, 5, 512000),
    ("DVSCLEAN_444", r"D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy/MAH00444/ratio100/MAH00444_ratio100_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy/MAH00444/ratio100/MAH00444_ratio100_labeled.npy",
     1280, 720, 5, 128000),
    ("LED_100", r"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/scene_100/slices_00031_00040_100ms/scene_100_100ms_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/scene_100/slices_00031_00040_100ms/scene_100_100ms_labeled.npy",
     1280, 720, 2, 16000),
]

U_DEN = [(0.5, "tau/2"), (1.0, "tau")]
B_DEN = [("1+u2", 0), ("1+u", 1)]

all_rows = []

for ds, clean, noisy, w, h, r, tau in TASKS:
    print(f"\n{'='*55}")
    print(f"  {ds}  (r={r}, tau={tau})")
    print(f"{'='*55}")
    for (u_val, u_name), (b_name, b_val) in itertools.product(U_DEN, B_DEN):
        env = os.environ.copy()
        if u_val != 0.5: env["MYEVS_N149_U_DENOM"] = str(u_val)
        if b_val != 1: env["MYEVS_N149_B_DENOM"] = b_name
        tag = f"v21_{ds}_u{u_name}_b{b_name}"
        out_csv = f"data/_v21_{tag}.csv"
        cmd = [PY, '-m', 'myevs.cli', 'roc',
               '--clean', clean, '--noisy', noisy,
               '--assume', 'npy', '--width', str(w), '--height', str(h),
               '--tick-ns', '1000', '--engine', 'cpp', '--method', 'n149',
               '--radius-px', str(r), '--time-us', str(tau),
               '--param', 'min-neighbors', '--values', THR,
               '--match-us', '0', '--match-bin-radius', '0',
               '--tag', tag, '--out-csv', out_csv, '--append']
        subprocess.run(cmd, check=True, timeout=600, env=env)
        auc = float(pd.read_csv(out_csv).loc[pd.read_csv(out_csv)['auc'].idxmax()]['auc'])
        print(f"  D={u_name:<6} B={b_name:<5}  AUC={auc:.4f}")
        all_rows.append((ds, u_name, b_name, auc))

print(f"\n\n{'='*65}")
print(f"  BEST per dataset")
print(f"{'='*65}")
print(f"{'Dataset':<16} {'Best D':>8} {'Best B':>8} {'AUC':>8}  {'Δ vs runner-up':>14}")
for ds in set(r[0] for r in all_rows):
    subset = [(r[1], r[2], r[3]) for r in all_rows if r[0] == ds]
    subset.sort(key=lambda x: x[2], reverse=True)
    best, second = subset[0], subset[1] if len(subset) > 1 else subset[0]
    print(f"{ds:<16} {best[0]:>8} {best[1]:>8} {best[2]:>8.4f}  {best[2]-second[2]:>+14.4f}")

print("\nDONE")
df = pd.DataFrame(all_rows, columns=["dataset","u_den","b_den","auc"])
df.to_csv("data/_n149_v21_validate.csv", index=False)
