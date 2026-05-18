"""Sweep N149 formula hyperparameters: alpha, u_denom, B_denom."""
import subprocess, os, itertools
import pandas as pd

PY = r"D:/software/Anaconda_envs/envs/myEVS/python.exe"
THR = "0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,8"

# Test on 2 representative datasets
TASKS = [
    ("Drive_8hz", r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_8hz_ed24_withlabel/driving_noise_8hz_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_8hz_ed24_withlabel/driving_noise_8hz_labeled.npy",
     346, 260, 2, 32000),
    ("Ped_heavy", r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3.npy",
     346, 260, 5, 256000),
]

ALPHA = [("sq", 0, "(1-m)^2"), ("lin", 1, "1-m"), ("m2", 2, "1-m^2")]
U_DEN = [(0.5, "tau/2"), (1.0, "tau"), (2.0, "2tau")]
B_DEN = [("1+u2", 0, "1+u^2"), ("1+u", 1, "1+u"), ("(1+u)2", 2, "(1+u)^2")]

all_rows = []

for ds, clean, noisy, w, h, r, tau in TASKS:
    print(f"\n{'='*65}")
    print(f"  {ds}  (r={r}, tau={tau})")
    print(f"{'='*65}")
    print(f"{'alpha':>10} {'u_den':>8} {'B_den':>10} {'AUC':>8}")
    print("-" * 42)

    for (a_name, a_val, a_desc), (u_val, u_desc), (b_name, b_val, b_desc) in itertools.product(ALPHA, U_DEN, B_DEN):
        env = os.environ.copy()
        if a_val != 0: env["MYEVS_N149_ALPHA_FORM"] = a_name
        if u_val != 0.5: env["MYEVS_N149_U_DENOM"] = str(u_val)
        if b_val != 0: env["MYEVS_N149_B_DENOM"] = b_name

        tag = f"opt_{ds}_a{a_name}_u{u_desc}_b{b_name}"
        out_csv = f"data/_opt_{tag}.csv"
        cmd = [PY, '-m', 'myevs.cli', 'roc',
               '--clean', clean, '--noisy', noisy,
               '--assume', 'npy', '--width', str(w), '--height', str(h),
               '--tick-ns', '1000', '--engine', 'cpp', '--method', 'n149',
               '--radius-px', str(r), '--time-us', str(tau),
               '--param', 'min-neighbors', '--values', THR,
               '--match-us', '0', '--match-bin-radius', '0',
               '--tag', tag, '--out-csv', out_csv, '--append']
        subprocess.run(cmd, check=True, timeout=300, env=env)
        auc = float(pd.read_csv(out_csv).loc[pd.read_csv(out_csv)['auc'].idxmax()]['auc'])
        print(f"{a_desc:>10} {u_desc:>8} {b_desc:>10} {auc:>8.4f}")
        all_rows.append((ds, a_desc, u_desc, b_desc, auc))

# Summary per dataset
for ds in ["Drive_8hz", "Ped_heavy"]:
    subset = [(r[1], r[2], r[3], r[4]) for r in all_rows if r[0] == ds]
    subset.sort(key=lambda x: x[3], reverse=True)
    print(f"\n\nTop 5 for {ds}:")
    for a, u, b, auc in subset[:5]:
        print(f"  alpha={a}  u_den={u}  B_den={b}  AUC={auc:.4f}")

# Find best combo across both datasets
best_by_ds = {}
for ds in ["Drive_8hz", "Ped_heavy"]:
    subset = [(r[1], r[2], r[3], r[4]) for r in all_rows if r[0] == ds]
    best_by_ds[ds] = max(subset, key=lambda x: x[3])

print(f"\n\nBest per dataset:")
for ds, (a, u, b, auc) in best_by_ds.items():
    print(f"  {ds}: alpha={a}  u_den={u}  B_den={b}  AUC={auc:.4f}")

print("\nDONE")
df = pd.DataFrame(all_rows, columns=["dataset","alpha","u_denom","b_denom","auc"])
df.to_csv("data/_n149_formula_opt.csv", index=False)
