"""DVSCLEAN MAH00447 ratio50/100: full comparison + N149 r=3 vs r=5. 12-threaded."""
import subprocess, os, time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

PY = r"D:/software/Anaconda_envs/envs/myEVS/python.exe"
THR_EBF = "0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,8"
TAU_BAF = "2000,5000,10000,15000,20000,25000,30000,40000,50000,64000,80000,100000,128000,160000,200000"
OUT = r"D:/hjx_workspace/scientific_reserach/projects/myEVS/data/missing_alg/dvsclean"

SCENES = {
    "447_50": (r"D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy/MAH00447/ratio50/MAH00447_ratio50_signal_only.npy",
               r"D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy/MAH00447/ratio50/MAH00447_ratio50_labeled.npy"),
    "447_100": (r"D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy/MAH00447/ratio100/MAH00447_ratio100_signal_only.npy",
                r"D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy/MAH00447/ratio100/MAH00447_ratio100_labeled.npy"),
}
W, H = 1280, 720

os.makedirs(f"{OUT}/_mah00447", exist_ok=True)

# ========== Task builders ==========
tasks = []

# 1. BAF: tau sweep [2,200]ms
for lv, (cl, ny) in SCENES.items():
    tasks.append(("baf", lv, cl, ny, {"method": "baf", "radius_px": 1, "min_neighbors": 1,
                  "param": "time-us", "values": TAU_BAF}))

# 2. STCF_orig: per k, tau sweep
for lv, (cl, ny) in SCENES.items():
    for k in [1, 2, 3, 4, 5, 6]:
        tasks.append(("stcf", lv, cl, ny, {"method": "stcf_original", "radius_px": 1, "min_neighbors": k,
                      "param": "time-us", "values": TAU_BAF}))

# 3. EBF: r×tau, threshold sweep
for lv, (cl, ny) in SCENES.items():
    for r in [2, 3, 4]:
        for tau in [16000, 32000, 64000, 128000]:
            tasks.append(("ebf", lv, cl, ny, {"method": "ebf", "radius_px": r, "time_us": tau,
                          "param": "min-neighbors", "values": THR_EBF}))

# 4. TS: r×tau, threshold sweep
for lv, (cl, ny) in SCENES.items():
    for r in [1, 2, 3]:
        for tau in [16000, 32000, 64000, 128000]:
            tasks.append(("ts", lv, cl, ny, {"method": "ts", "radius_px": r, "time_us": tau,
                          "param": "min-neighbors", "values": "0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5"}))

# 5. PFD: r=1, per (m,k), tau sweep
for lv, (cl, ny) in SCENES.items():
    for m in [1, 2]:
        for k in [1, 2]:
            tasks.append(("pfd", lv, cl, ny, {"method": "pfd", "radius_px": 1, "min_neighbors": k,
                          "refractory_us": m, "pfd_mode": "a",
                          "param": "time-us", "values": "1000,2000,4000,8000,16000,32000,64000"}))

# 6. N149: r=3 and r=5, optimal params
for lv, (cl, ny) in SCENES.items():
    for r in [3, 5]:
        tasks.append(("n149", lv, cl, ny, {"method": "n149", "radius_px": r, "time_us": 128000,
                      "param": "min-neighbors", "values": THR_EBF,
                      "env": {"MYEVS_N149_SIGMA": "2.5", "MYEVS_N149_ALPHA_FIXED": "0.25"}}))


def run_one(algo, lv, clean, noisy, cfg):
    env_extra = cfg.pop("env", {})
    tag = f"mah447_{algo}_{lv}_{cfg.get('radius_px','')}_{cfg.get('time_us','')}_{cfg.get('min_neighbors','')}"
    csv = f"{OUT}/_mah00447/{tag}.csv"
    os.makedirs(os.path.dirname(csv), exist_ok=True)
    if os.path.exists(csv): os.remove(csv)

    env = os.environ.copy()
    env["MYEVS_N149_HOT_BITS"] = "16"
    for k, v in env_extra.items(): env[k] = v

    cmd = [PY, '-m', 'myevs.cli', 'roc',
           '--clean', clean, '--noisy', noisy,
           '--assume', 'npy', '--width', str(W), '--height', str(H),
           '--tick-ns', '1000', '--engine', 'cpp',
           '--method', cfg.pop("method"),
           '--param', cfg.pop("param"), '--values', cfg.pop("values"),
           '--match-us', '0', '--match-bin-radius', '0',
           '--tag', tag, '--out-csv', csv, '--append']

    for k, v in cfg.items():
        cmd.extend([f'--{k.replace("_","-")}', str(v)])

    subprocess.run(cmd, check=True, timeout=600, env=env, capture_output=True)
    df = pd.read_csv(csv)
    return algo, lv, cfg.get('radius_px',0), cfg.get('time_us',0), cfg.get('min_neighbors',0), float(df["auc"].iloc[0])


total = len(tasks)
baf_n = sum(1 for t in tasks if t[0]=="baf")
stcf_n = sum(1 for t in tasks if t[0]=="stcf")
ebf_n = sum(1 for t in tasks if t[0]=="ebf")
ts_n = sum(1 for t in tasks if t[0]=="ts")
pfd_n = sum(1 for t in tasks if t[0]=="pfd")
n149_n = sum(1 for t in tasks if t[0]=="n149")
print(f"MAH00447 full: {total} tasks (BAF={baf_n} STCF={stcf_n} EBF={ebf_n} TS={ts_n} PFD={pfd_n} N149={n149_n})")
t0 = time.time()

results = []
with ThreadPoolExecutor(max_workers=12) as ex:
    futures = {ex.submit(run_one, *t): t for t in tasks}
    done = 0
    for f in as_completed(futures):
        algo, lv, r, tau, k, auc = f.result()
        results.append((algo, lv, r, tau, k, auc))
        done += 1
        rate = done/(time.time()-t0) if time.time()>t0 else 0
        eta = (total-done)/rate/60 if rate>0 else 0
        print(f"[{done}/{total} | {rate:.1f}t/s | ETA {eta:.0f}m] {algo} {lv} r={r} tau={tau} AUC={auc:.4f}", flush=True)

df = pd.DataFrame(results, columns=["algo","level","r","tau","k","auc"])
df.to_csv(f"{OUT}/_mah00447/all_results.csv", index=False)

# Summary
print("\n" + "=" * 70)
print("MAH00447 RESULTS")
print("=" * 70)
for lv in ["447_50", "447_100"]:
    print(f"\n{lv}:")
    sub = df[df.level==lv]

    # BAF
    baf = sub[sub.algo=="baf"]
    print(f"  BAF:        AUC={baf.auc.max():.4f}")

    # STCF: best k
    stcf_best = sub[sub.algo=="stcf"].loc[sub[sub.algo=="stcf"].groupby("k")["auc"].transform("max") == sub[sub.algo=="stcf"]["auc"]]
    stcf_top = sub[sub.algo=="stcf"].nlargest(1, "auc")
    print(f"  STCF_orig:  AUC={stcf_top.auc.values[0]:.4f} (k={int(stcf_top.k.values[0])})")

    # EBF
    ebf = sub[sub.algo=="ebf"]
    ebf_top = ebf.nlargest(1, "auc")
    print(f"  EBF:        AUC={ebf_top.auc.values[0]:.4f} (r={int(ebf_top.r.values[0])} tau={int(ebf_top.tau.values[0])})")

    # TS
    ts = sub[sub.algo=="ts"]
    ts_top = ts.nlargest(1, "auc")
    print(f"  TS:         AUC={ts_top.auc.values[0]:.4f} (r={int(ts_top.r.values[0])} tau={int(ts_top.tau.values[0])})")

    # PFD
    pfd = sub[sub.algo=="pfd"]
    pfd_top = pfd.nlargest(1, "auc")
    print(f"  PFD:        AUC={pfd_top.auc.values[0]:.4f} (m={int(pfd_top.k.values[0])} k={int(pfd_top.tau.values[0])})")

    # N149 r=3 vs r=5
    n149 = sub[sub.algo=="n149"]
    for r in [3, 5]:
        nr = n149[n149.r==r]
        if len(nr) > 0:
            print(f"  N149 r={r}:   AUC={nr.auc.max():.4f}")

print("\nDONE")
