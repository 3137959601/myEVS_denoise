"""N149 ablation: ED24 Ped light + DVSCLEAN + LED."""
import subprocess, os, sys
import pandas as pd

PY = r"D:/software/Anaconda_envs/envs/myEVS/python.exe"
THR = "0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,8"

DATASETS = [
    {
        "name": "ED24_Ped_light",
        "clean": r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_1.8_signal_only.npy",
        "noisy": r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_1.8.npy",
        "width": 346, "height": 260, "r": 5, "tau": 256000,
    },
    {
        "name": "DVSCLEAN_444_r100",
        "clean": r"D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy/MAH00444/ratio100/MAH00444_ratio100_signal_only.npy",
        "noisy": r"D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy/MAH00444/ratio100/MAH00444_ratio100_labeled.npy",
        "width": 1280, "height": 720, "r": 5, "tau": 128000,
    },
    {
        "name": "LED_scene100",
        "clean": r"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/scene_100/slices_00031_00040_100ms/scene_100_100ms_signal_only.npy",
        "noisy": r"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/scene_100/slices_00031_00040_100ms/scene_100_100ms_labeled.npy",
        "width": 1280, "height": 720, "r": 2, "tau": 16000,
    },
]

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

for ds in DATASETS:
    dname = ds["name"]
    print(f"\n{'='*70}")
    print(f"Dataset: {dname}  (r={ds['r']}, tau={ds['tau']}us, {ds['width']}x{ds['height']})")
    print(f"{'='*70}")
    print(f"{'Ablation':<22} {'AUC':>8} {'F1':>8} {'ΔAUC':>10}  {'Δ%':>8}")
    print("-" * 60)

    baseline_auc = None
    results = []

    for name, env_vars in ablations:
        env = os.environ.copy()
        env.update(env_vars)
        tag = f"abl_{dname}_{name.replace(' ','_')}"
        out_csv = f"data/_abl_{tag}.csv"
        cmd = [PY, '-m', 'myevs.cli', 'roc',
               '--clean', ds["clean"], '--noisy', ds["noisy"],
               '--assume', 'npy', '--width', str(ds["width"]), '--height', str(ds["height"]),
               '--tick-ns', '1000', '--engine', 'cpp',
               '--method', 'n149', '--radius-px', str(ds["r"]), '--time-us', str(ds["tau"]),
               '--param', 'min-neighbors', '--values', THR,
               '--match-us', '0', '--match-bin-radius', '0',
               '--tag', tag, '--out-csv', out_csv, '--append']
        print(f"  [{dname}] {name}...", end=" ", flush=True)
        subprocess.run(cmd, check=True, timeout=900, env=env)
        df = pd.read_csv(out_csv)
        best = df.loc[df['auc'].idxmax()]
        auc = float(best['auc']); f1 = float(best['f1'])
        if name == "Baseline":
            baseline_auc = auc
        drop = auc - baseline_auc if baseline_auc else 0
        drop_pct = (drop / baseline_auc * 100) if baseline_auc else 0
        flag = " ***" if abs(drop) > 0.002 else ""
        results.append((name, auc, f1, drop, drop_pct, flag))
        print(f"AUC={auc:.4f}  Δ={drop:+.4f}{flag}")

    print(f"\n  Summary for {dname} (Baseline AUC={baseline_auc:.4f}):")
    for name, auc, f1, drop, drop_pct, flag in results:
        print(f"  {name:<22} {auc:.4f}  Δ={drop:+.4f}{' ***' if abs(drop) > 0.002 else ''}")

print("\n\nALL DONE")
