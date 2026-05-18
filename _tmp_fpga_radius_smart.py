"""Extract existing r=1/r=2 data + run missing, compare with optimal."""
import subprocess, os, glob
import pandas as pd
from pathlib import Path

PY = r"D:/software/Anaconda_envs/envs/myEVS/python.exe"
THR = "0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,8"
TAUS = [2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000]

# ============================================================
# Part 1: Extract existing AUC from CSV files
# ============================================================
def best_auc_from_csv(csv_path, tag_prefix=None):
    """Find best AUC in a CSV, optionally filtering by tag prefix."""
    try:
        df = pd.read_csv(csv_path)
        if 'auc' not in df.columns:
            return 0, "", ""
        if tag_prefix:
            df = df[df['tag'].str.startswith(tag_prefix)]
        if len(df) == 0:
            return 0, "", ""
        best = df.loc[df['auc'].idxmax()]
        return float(best['auc']), str(best['tag']), str(best.get('value',''))
    except:
        return 0, "", ""

def extract_baf_stcf(dataset_dir, method, level, r_target):
    """Extract best AUC at given radius from BAF/STCF ROC CSV."""
    pattern = f"{dataset_dir}/roc_{method}_{level}*.csv"
    files = glob.glob(pattern)
    if not files:
        return 0, ""
    best_auc, best_tag = 0, ""
    for f in files:
        auc, tag, _ = best_auc_from_csv(f, f"{method}_r{r_target}")
        if auc > best_auc:
            best_auc, best_tag = auc, tag
    return best_auc, best_tag

def extract_ebf_n149(dataset_dir, method, level, r_target):
    """Extract best AUC at given radius from EBF/N149 ROC CSV."""
    # Try different patterns
    patterns = [
        f"{dataset_dir}/roc_{method}_{level}.csv",
        f"{dataset_dir}/roc_{method}_{level}_ed24.csv",
        f"{dataset_dir}/roc_{method}_{level}_ed24_paperfix.csv",
    ]
    best_auc, best_tag = 0, ""
    for pattern in patterns:
        for f in glob.glob(pattern):
            auc, tag, _ = best_auc_from_csv(f, f"{method}_r{r_target}")
            if auc > best_auc:
                best_auc, best_tag = auc, tag
    return best_auc, best_tag

def extract_knoise(dataset_dir, level):
    """KNoise has no radius, just extract best overall."""
    patterns = [
        f"{dataset_dir}/roc_knoise_{level}.csv",
        f"{dataset_dir}/roc_knoise_{level}_ed24.csv",
    ]
    best_auc, best_tag = 0, ""
    for pattern in patterns:
        for f in glob.glob(pattern):
            auc, tag, _ = best_auc_from_csv(f)
            if auc > best_auc:
                best_auc, best_tag = auc, tag
    return best_auc, best_tag

# ============================================================
# Part 2: Run missing (r,tau) combos
# ============================================================
def run_missing(ds_label, clean, noisy, w, h, method, r, tau):
    """Run a single CLI call and return best AUC."""
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
    try:
        subprocess.run(cmd, check=True, timeout=600, capture_output=True)
        df = pd.read_csv(out_csv)
        return float(df.loc[df['auc'].idxmax()]['auc']), tag
    except Exception as e:
        print(f"  FAIL {tag}: {e}")
        return 0, ""

def sweep_missing(ds_label, clean, noisy, w, h, method, r):
    """Sweep taus for a missing (method, r) combo, return best."""
    best_auc, best_tag, best_tau = 0, "", 0
    for tau in TAUS:
        auc, tag = run_missing(ds_label, clean, noisy, w, h, method, r, tau)
        if auc > best_auc:
            best_auc, best_tag, best_tau = auc, tag, tau
    return best_auc, best_tau, best_tag

# ============================================================
# Main: iterate datasets
# ============================================================
DATASETS = [
    ("Drive_1hz", "D:/.../DND21/mydriving_ED24",
     r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_1hz_ed24_withlabel/driving_noise_1hz_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_1hz_ed24_withlabel/driving_noise_1hz_labeled.npy",
     346, 260, "1hz"),
    ("Drive_2hz", "D:/...",
     r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_2hz_ed24_withlabel/driving_noise_2hz_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_2hz_ed24_withlabel/driving_noise_2hz_labeled.npy",
     346, 260, "2hz"),
    ("Drive_5hz", "D:/...",
     r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_5hz_ed24_withlabel/driving_noise_5hz_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_5hz_ed24_withlabel/driving_noise_5hz_labeled.npy",
     346, 260, "5hz"),
    ("Drive_8hz", "D:/...",
     r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_8hz_ed24_withlabel/driving_noise_8hz_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_8hz_ed24_withlabel/driving_noise_8hz_labeled.npy",
     346, 260, "8hz"),
    ("ED24_Ped_light", "D:/.../ED24/myPedestrain_06",
     r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_1.8_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_1.8.npy",
     346, 260, "light"),
    ("ED24_Ped_heavy", "D:/...",
     r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3_signal_only.npy",
     r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3.npy",
     346, 260, "heavy"),
]

DRIVE_DIR = r"D:/hjx_workspace/scientific_reserach/projects/myEVS/data/DND21/mydriving_ED24"
PED_DIR = r"D:/hjx_workspace/scientific_reserach/projects/myEVS/data/ED24/myPedestrain_06"

OPT_AUC = {
    ("Drive_1hz","baf"):0.9166,("Drive_1hz","stcf"):0.9484,("Drive_1hz","knoise"):0.6359,
    ("Drive_1hz","ebf"):0.9484,("Drive_1hz","n149"):0.9381,
    ("Drive_2hz","baf"):0.9029,("Drive_2hz","stcf"):0.9445,("Drive_2hz","knoise"):0.6265,
    ("Drive_2hz","ebf"):0.9472,("Drive_2hz","n149"):0.9386,
    ("Drive_5hz","baf"):0.8651,("Drive_5hz","stcf"):0.9309,("Drive_5hz","knoise"):0.6239,
    ("Drive_5hz","ebf"):0.9408,("Drive_5hz","n149"):0.9416,
    ("Drive_8hz","baf"):0.8379,("Drive_8hz","stcf"):0.9229,("Drive_8hz","knoise"):0.6214,
    ("Drive_8hz","ebf"):0.9374,("Drive_8hz","n149"):0.9418,
    ("ED24_Ped_light","baf"):0.9119,("ED24_Ped_light","stcf"):0.9460,("ED24_Ped_light","knoise"):0.7130,
    ("ED24_Ped_light","ebf"):0.9416,("ED24_Ped_light","n149"):0.9565,
    ("ED24_Ped_heavy","baf"):0.8161,("ED24_Ped_heavy","stcf"):0.8791,("ED24_Ped_heavy","knoise"):0.6417,
    ("ED24_Ped_heavy","ebf"):0.9099,("ED24_Ped_heavy","n149"):0.9406,
}

all_rows = []

for ds_label, data_dir, clean, noisy, w, h, level in DATASETS:
    # Determine which data directory to use
    if "Driving" in ds_label or "Drive" in ds_label:
        ext_dir = DRIVE_DIR
        ext_level = level  # "1hz", "2hz", etc.
    else:
        ext_dir = PED_DIR
        ext_level = level  # "light", "heavy", etc.

    print(f"\n{'='*60}")
    print(f"  {ds_label}")
    print(f"{'='*60}")
    print(f"{'Method':<8} {'r':>2} {'src':>6} {'AUC(r-c)':>10} {'best(tau)':>12} {'Opt AUC':>10} {'Δ':>10}")
    print("-" * 68)

    # BAF r=1: extract from existing
    baf_r1_auc, baf_r1_tag = extract_baf_stcf(ext_dir, "baf", ext_level, 1)
    opt = OPT_AUC.get((ds_label, "baf"), 0)
    delta = baf_r1_auc - opt if opt else 0
    print(f"{'BAF':<8} {'1':>2} {'EXIST':>6} {baf_r1_auc:>10.4f} {baf_r1_tag:>12} {opt:>10.4f} {delta:>+10.4f}")
    all_rows.append((ds_label, "BAF", 1, "existing", baf_r1_auc, opt, delta))

    # STCF r=1: extract from existing
    stcf_r1_auc, stcf_r1_tag = extract_baf_stcf(ext_dir, "stcf", ext_level, 1)
    opt = OPT_AUC.get((ds_label, "stcf"), 0)
    delta = stcf_r1_auc - opt if opt else 0
    print(f"{'STCF':<8} {'1':>2} {'EXIST':>6} {stcf_r1_auc:>10.4f} {stcf_r1_tag:>12} {opt:>10.4f} {delta:>+10.4f}")
    all_rows.append((ds_label, "STCF", 1, "existing", stcf_r1_auc, opt, delta))

    # KNoise: extract from existing
    knoise_auc, knoise_tag = extract_knoise(ext_dir, ext_level)
    opt = OPT_AUC.get((ds_label, "knoise"), 0)
    delta = knoise_auc - opt if opt else 0
    print(f"{'KNoise':<8} {'-':>2} {'EXIST':>6} {knoise_auc:>10.4f} {knoise_tag:>12} {opt:>10.4f} {delta:>+10.4f}")
    all_rows.append((ds_label, "KNoise", 0, "existing", knoise_auc, opt, delta))

    # EBF r=1: NEED TO RUN (missing in all datasets)
    print(f"{'EBF':<8} {'1':>2} {'RUN':>6} ...", end=" ", flush=True)
    ebf_r1_auc, ebf_r1_tau, ebf_r1_tag = sweep_missing(ds_label, clean, noisy, w, h, "ebf", 1)
    opt = OPT_AUC.get((ds_label, "ebf"), 0)
    delta = ebf_r1_auc - opt if opt else 0
    print(f"{ebf_r1_auc:>10.4f} tau={ebf_r1_tau:>6} {opt:>10.4f} {delta:>+10.4f}")
    all_rows.append((ds_label, "EBF", 1, "new-run", ebf_r1_auc, opt, delta))

    # EBF r=2: extract from existing (Driving, LED) or need to run (ED24)
    ebf_r2_auc, ebf_r2_tag = extract_ebf_n149(ext_dir, "ebf", ext_level, 2)
    if ebf_r2_auc == 0:
        print(f"{'EBF':<8} {'2':>2} {'RUN':>6} ...", end=" ", flush=True)
        ebf_r2_auc, ebf_r2_tau, ebf_r2_tag = sweep_missing(ds_label, clean, noisy, w, h, "ebf", 2)
        src = "new-run"
    else:
        src = "existing"
    opt = OPT_AUC.get((ds_label, "ebf"), 0)
    delta = ebf_r2_auc - opt if opt else 0
    print(f"{'EBF':<8} {'2':>2} {src:>6} {ebf_r2_auc:>10.4f} {ebf_r2_tag:>12} {opt:>10.4f} {delta:>+10.4f}")
    all_rows.append((ds_label, "EBF", 2, src, ebf_r2_auc, opt, delta))

    # N149 r=1: NEED TO RUN (missing in all datasets)
    print(f"{'N149':<8} {'1':>2} {'RUN':>6} ...", end=" ", flush=True)
    n149_r1_auc, n149_r1_tau, n149_r1_tag = sweep_missing(ds_label, clean, noisy, w, h, "n149", 1)
    opt = OPT_AUC.get((ds_label, "n149"), 0)
    delta = n149_r1_auc - opt if opt else 0
    print(f"{n149_r1_auc:>10.4f} tau={n149_r1_tau:>6} {opt:>10.4f} {delta:>+10.4f}")
    all_rows.append((ds_label, "N149", 1, "new-run", n149_r1_auc, opt, delta))

    # N149 r=2: extract from existing (Driving) or need to run (ED24)
    n149_r2_auc, n149_r2_tag = extract_ebf_n149(ext_dir, "n149", ext_level, 2)
    if n149_r2_auc == 0:
        print(f"{'N149':<8} {'2':>2} {'RUN':>6} ...", end=" ", flush=True)
        n149_r2_auc, n149_r2_tau, n149_r2_tag = sweep_missing(ds_label, clean, noisy, w, h, "n149", 2)
        src = "new-run"
    else:
        src = "existing"
    opt = OPT_AUC.get((ds_label, "n149"), 0)
    delta = n149_r2_auc - opt if opt else 0
    print(f"{'N149':<8} {'2':>2} {src:>6} {n149_r2_auc:>10.4f} {n149_r2_tag:>12} {opt:>10.4f} {delta:>+10.4f}")
    all_rows.append((ds_label, "N149", 2, src, n149_r2_auc, opt, delta))

# Final summary
print(f"\n\n{'='*80}")
print(f"  SUMMARY: r=1 and r=2 constrained vs optimal")
print(f"{'='*80}")
for ds_label in sorted(set(r[0] for r in all_rows)):
    subset = [r for r in all_rows if r[0] == ds_label]
    print(f"\n--- {ds_label} ---")
    print(f"{'Method/r':<12} {'AUC(c)':>8} {'Opt':>8} {'Δ':>8}")
    subset_sorted = sorted(subset, key=lambda x: x[5], reverse=True)  # sort by constrained AUC
    for row in subset_sorted:
        m = f"{row[1]}_r{row[2]}"
        flag = " ***" if abs(row[6]) > 0.01 else ""
        print(f"{m:<12} {row[4]:>8.4f} {row[5]:>8.4f} {row[6]:>+8.4f}{flag}")

print("\nDONE")
df = pd.DataFrame(all_rows, columns=["dataset","method","r","source","auc_constrained","auc_optimal","delta"])
df.to_csv("data/_fpga_radius_final.csv", index=False)
