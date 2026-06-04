"""Quick test: tau-sweep 1-200ms vs current method on Driving 1hz."""
import subprocess, os, pandas as pd

PY = r"D:/software/Anaconda_envs/envs/myEVS/python.exe"
TAU = "1000,2000,5000,10000,15000,20000,25000,30000,40000,50000,64000,80000,100000,128000,160000,200000"
clean = r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_1hz_ed24_withlabel/driving_noise_1hz_signal_only.npy"
noisy = r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_1hz_ed24_withlabel/driving_noise_1hz_labeled.npy"
OUT = r"D:/hjx_workspace/scientific_reserach/projects/myEVS/data/_tau_cmp"
W, H = 346, 260
os.makedirs(OUT, exist_ok=True)

tests = [
    ("pfd", ["--radius-px","1","--min-neighbors","1","--refractory-us","1","--pfd-mode","a"]),
    ("stcf_original", ["--radius-px","1","--min-neighbors","2"]),
    ("ts", ["--radius-px","2"]),
    ("ynoise", ["--radius-px","2"]),
    ("knoise", ["--radius-px","1"]),
]

current_auc = {
    "pfd": 0.9123, "stcf_original": 0.9136,
    "ts": 0.9298, "ynoise": 0.9408, "knoise": 0.6359,
}

for method, extra in tests:
    csv = f"{OUT}/{method}_1hz_tau1-200.csv"
    if os.path.exists(csv): os.remove(csv)
    cmd = [PY, "-m", "myevs.cli", "roc",
           "--clean", clean, "--noisy", noisy,
           "--assume", "npy", "--width", str(W), "--height", str(H),
           "--tick-ns", "1000", "--engine", "cpp", "--method", method,
           *extra,
           "--param", "time-us", "--values", TAU,
           "--match-us", "0", "--match-bin-radius", "0",
           "--tag", f"{method}_tau1-200", "--out-csv", csv, "--append"]
    env = os.environ.copy()
    subprocess.run(cmd, check=True, timeout=600, env=env, capture_output=True)
    df = pd.read_csv(csv)
    auc = float(df["auc"].iloc[0])
    delta = auc - current_auc[method]
    print(f"{method:>15}: tau-sweep AUC={auc:.4f}  current={current_auc[method]:.4f}  δ={delta:+.4f}")

print("DONE")
