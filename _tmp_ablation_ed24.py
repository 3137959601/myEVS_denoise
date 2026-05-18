"""N149 ablation on ED24 Pedestrian heavy."""
import subprocess, os
import pandas as pd

PY = r"D:/software/Anaconda_envs/envs/myEVS/python.exe"
CLEAN = r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3_signal_only.npy"
NOISY = r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3.npy"
THR = "0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,8"
R, TAU = 5, 256000

ablations = [
    ("Baseline", {}),
    ("No hot_state", {"MYEVS_N149_NO_HOT": "1"}),
    ("No beta", {"MYEVS_N149_NO_BETA": "1"}),
    ("No mix", {"MYEVS_N149_NO_MIX": "1"}),
    ("No opp-polarity", {"MYEVS_N149_NO_OPP": "1"}),
    ("No sfrac", {"MYEVS_N149_NO_SFRAC": "1"}),
    ("No spatial w", {"MYEVS_N149_NO_SPATIAL": "1"}),
    ("hot+beta off", {"MYEVS_N149_NO_HOT": "1", "MYEVS_N149_NO_BETA": "1"}),
    ("hot+beta+mix off", {"MYEVS_N149_NO_HOT": "1", "MYEVS_N149_NO_BETA": "1", "MYEVS_N149_NO_MIX": "1"}),
    ("All temporal off", {"MYEVS_N149_NO_HOT": "1", "MYEVS_N149_NO_BETA": "1", "MYEVS_N149_NO_MIX": "1", "MYEVS_N149_NO_OPP": "1"}),
]

print(f"{'Ablation':<22} {'AUC':>8} {'F1':>8} {'AUC_drop':>10}")
print("-" * 55)
baseline_auc = None
for name, env_vars in ablations:
    env = os.environ.copy()
    env.update(env_vars)
    out = f"data/_abl_ed24_{name.replace(' ','_')}.csv"
    cmd = [PY, '-m', 'myevs.cli', 'roc',
           '--clean', CLEAN, '--noisy', NOISY,
           '--assume', 'npy', '--width', '346', '--height', '260',
           '--tick-ns', '1000', '--engine', 'cpp',
           '--method', 'n149', '--radius-px', str(R), '--time-us', str(TAU),
           '--param', 'min-neighbors', '--values', THR,
           '--match-us', '0', '--match-bin-radius', '0',
           '--tag', name.replace(' ','_'), '--out-csv', out, '--append']
    print(f"Running: {name}...")
    subprocess.run(cmd, check=True, timeout=600, env=env)
    df = pd.read_csv(out)
    best = df.loc[df['auc'].idxmax()]
    auc = float(best['auc']); f1 = float(best['f1'])
    if name == "Baseline": baseline_auc = auc
    drop = auc - baseline_auc if baseline_auc else 0
    flag = " ***" if abs(drop) > 0.002 else ""
    print(f"{name:<22} {auc:>8.4f} {f1:>8.4f} {drop:>+10.4f}{flag}")

print("\nDONE")
