"""Test unified tau ranges: BAF/STCF_orig [2,200]ms, PFD [1,32]ms across datasets."""
import subprocess, os, pandas as pd

PY = r"D:/software/Anaconda_envs/envs/myEVS/python.exe"
OUT = r"D:/hjx_workspace/scientific_reserach/projects/myEVS/data/_tau_cmp"
os.makedirs(OUT, exist_ok=True)

# Unified tau ranges
TAU_BAF = "2000,5000,10000,15000,20000,25000,30000,40000,50000,64000,80000,100000,128000,160000,200000"
TAU_PFD = "1000,2000,4000,8000,16000,32000"

# Test datasets: one level each
TESTS = [
    ("drive", "1hz",
     r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_1hz_ed24_withlabel/driving_noise_1hz_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_1hz_ed24_withlabel/driving_noise_1hz_labeled.npy",
     346, 260),
    ("bike", "1.8",
     r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_1.8_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_1.8.npy",
     346, 260),
    ("dvsclean", "MAH00444_ratio50",
     r"D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy/MAH00444/ratio50/MAH00444_ratio50_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy/MAH00444/ratio50/MAH00444_ratio50_labeled.npy",
     1280, 720),
    ("led", "scene_100",
     r"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/scene_100/slices_00031_00040_100ms/scene_100_100ms_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/scene_100/slices_00031_00040_100ms/scene_100_100ms_labeled.npy",
     1280, 720),
]

current_baf = {"drive_1hz": 0.9136, "bike_1.8": 0.9489, "dvsclean_MAH00444_ratio50": 0.9828, "led_scene_100": 0.8576}
current_pfd = {"drive_1hz": 0.9123, "bike_1.8": 0.8962, "dvsclean_MAH00444_ratio50": 0.9744, "led_scene_100": 0.8560}

print(f"{'Dataset':>10} {'Algo':>5} {'Unified AUC':>12} {'Current':>10} {'Delta':>8}")
print("-"*50)

for ds, lv, clean, noisy, w, h in TESTS:
    # BAF [2,200]ms
    csv = f"{OUT}/baf_{ds}_{lv}_unified.csv"
    if os.path.exists(csv): os.remove(csv)
    cmd = [PY, "-m", "myevs.cli", "roc", "--clean", clean, "--noisy", noisy,
           "--assume", "npy", "--width", str(w), "--height", str(h),
           "--tick-ns", "1000", "--engine", "cpp", "--method", "baf",
           "--radius-px", "1", "--min-neighbors", "1",
           "--param", "time-us", "--values", TAU_BAF,
           "--match-us", "0", "--match-bin-radius", "0",
           "--tag", "baf_unified", "--out-csv", csv, "--append"]
    env = os.environ.copy()
    subprocess.run(cmd, check=True, timeout=600, env=env, capture_output=True)
    df = pd.read_csv(csv)
    auc = float(df["auc"].iloc[0])
    cur = current_baf[f"{ds}_{lv}"]
    print(f"{ds:>10} {'BAF':>5} {auc:>12.4f} {cur:>10.4f} {auc-cur:>+8.4f}")

    # PFD [1,32]ms
    csv = f"{OUT}/pfd_{ds}_{lv}_unified.csv"
    if os.path.exists(csv): os.remove(csv)
    cmd = [PY, "-m", "myevs.cli", "roc", "--clean", clean, "--noisy", noisy,
           "--assume", "npy", "--width", str(w), "--height", str(h),
           "--tick-ns", "1000", "--engine", "cpp", "--method", "pfd",
           "--radius-px", "1", "--min-neighbors", "1", "--refractory-us", "1", "--pfd-mode", "a",
           "--param", "time-us", "--values", TAU_PFD,
           "--match-us", "0", "--match-bin-radius", "0",
           "--tag", "pfd_unified", "--out-csv", csv, "--append"]
    env = os.environ.copy()
    subprocess.run(cmd, check=True, timeout=600, env=env, capture_output=True)
    df = pd.read_csv(csv)
    auc = float(df["auc"].iloc[0])
    cur = current_pfd[f"{ds}_{lv}"]
    print(f"{ds:>10} {'PFD':>5} {auc:>12.4f} {cur:>10.4f} {auc-cur:>+8.4f}")

print("DONE")
