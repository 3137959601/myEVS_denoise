"""Recompute Chapter 7 AUC/F1 with the documented sweep methodology.

Default scope excludes MLPF, EvFlow, KNoise and STCF, as requested.  The key
summary rule is:

1. select the tag/curve with the best AUC for each dataset-level-algorithm;
2. report max F1 within that selected AUC-best tag.

This avoids the common mistake of taking F1 from the first row whose repeated
tag-level AUC is maximal.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PY = Path(r"D:/software/Anaconda_envs/envs/myEVS/python.exe")
OUT_ROOT = ROOT / "data" / "chapter7_f1_recompute"

THR_STD = "0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,8"
THR_TS = "0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5"
TAU_UNIFIED = "2000,5000,10000,15000,20000,25000,30000,40000,50000,64000,80000,100000,128000,160000,200000"
TAU_LED = "1000,2000,4000,8000,16000,32000"

DEFAULT_ALGS = ("baf", "stcf_orig", "pfd", "ebf", "ynoise", "ts", "n149_v22")


@dataclass(frozen=True)
class Sample:
    dataset: str
    level: str
    clean: str
    noisy: str
    width: int
    height: int


def drive_samples() -> list[Sample]:
    root = Path(r"D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24")
    out = []
    for lv in ("1hz", "3hz", "5hz", "7hz", "10hz"):
        d = root / f"driving_noise_{lv}_ed24_withlabel"
        clean = d / f"driving_noise_{lv}_signal_only.npy"
        noisy = d / f"driving_noise_{lv}_labeled.npy"
        if clean.exists() and noisy.exists():
            out.append(Sample("drive", lv, str(clean), str(noisy), 346, 260))
    return out


def ped_samples() -> list[Sample]:
    root = Path(r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06")
    return [
        Sample("ped", lv, str(root / f"Pedestrain_06_{lv}_signal_only.npy"), str(root / f"Pedestrain_06_{lv}.npy"), 346, 260)
        for lv in ("1.8", "2.1", "2.5", "3.3")
    ]


def bike_samples() -> list[Sample]:
    root = Path(r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02")
    return [
        Sample("bike", lv, str(root / f"Bicycle_02_{lv}_signal_only.npy"), str(root / f"Bicycle_02_{lv}.npy"), 346, 260)
        for lv in ("1.8", "2.1", "2.5", "3.3")
    ]


def dvsclean_samples() -> list[Sample]:
    root = Path(r"D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy")
    out = []
    for scene in ("MAH00444", "MAH00446", "MAH00447", "MAH00448", "MAH00449"):
        for ratio in ("ratio50", "ratio100"):
            lv = f"{scene}_{ratio}"
            d = root / scene / ratio
            out.append(Sample("dvsclean", lv, str(d / f"{scene}_{ratio}_signal_only.npy"), str(d / f"{scene}_{ratio}_labeled.npy"), 1280, 720))
    return out


def led_samples() -> list[Sample]:
    root = Path(r"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy")
    scenes = ("scene_100", "scene_1004", "scene_1018", "scene_1028", "scene_1032", "scene_1033", "scene_1034", "scene_1043", "scene_1045", "scene_1046")
    out = []
    for scene in scenes:
        d = root / scene / "slices_00031_00040_100ms"
        out.append(Sample("led", scene, str(d / f"{scene}_100ms_signal_only.npy"), str(d / f"{scene}_100ms_labeled.npy"), 1280, 720))
    return out


def all_samples() -> list[Sample]:
    return drive_samples() + ped_samples() + bike_samples() + dvsclean_samples() + led_samples()


def csv_values(values: list[int | float]) -> str:
    return ",".join(str(v) for v in values)


def cases_for(sample: Sample, alg: str) -> list[tuple[str, list[str], dict[str, str]]]:
    ds = sample.dataset
    if alg == "baf":
        taus = TAU_LED if ds == "led" else TAU_UNIFIED
        return [("baf_r1", ["--method", "baf", "--engine", "cpp", "--radius-px", "1", "--min-neighbors", "1", "--param", "time-us", "--values", taus], {})]

    if alg == "stcf_orig":
        taus = TAU_LED if ds == "led" else TAU_UNIFIED
        return [
            (f"stcf_orig_k{k}", ["--method", "stcf_original", "--engine", "cpp", "--radius-px", "1", "--min-neighbors", str(k), "--param", "time-us", "--values", taus], {})
            for k in (1, 2, 3, 4, 5, 6)
        ]

    if alg == "pfd":
        tau_map = {
            "drive": "1000,2000,4000,8000,16000,32000,64000",
            "ped": "1000,2000,4000,8000,16000,32000,64000,128000,256000,512000",
            "bike": "1000,2000,4000,8000,16000,32000,64000,128000,256000,512000",
            "dvsclean": "1000,2000,4000,8000,16000,32000,64000,128000",
            "led": TAU_LED,
        }
        out = []
        for m in (1, 2):
            for k in (1, 2):
                out.append((
                    f"pfd_r1_m{m}_k{k}",
                    ["--method", "pfd", "--engine", "cpp", "--radius-px", "1", "--min-neighbors", str(k), "--refractory-us", str(m), "--pfd-mode", "a", "--param", "time-us", "--values", tau_map[ds]],
                    {},
                ))
        return out

    if alg == "ebf":
        if ds == "drive":
            grid = [(2, 32000)]
        elif ds == "led":
            grid = [(2, 8000)]
        elif ds == "dvsclean":
            grid = [(r, t) for r in (3, 4, 5) for t in (32000, 64000, 128000)]
        else:
            grid = [(r, t) for r in (2, 3, 4, 5) for t in (16000, 32000, 64000, 128000, 256000)]
        return [(f"ebf_r{r}_tau{t}", ["--method", "ebf", "--engine", "cpp", "--radius-px", str(r), "--time-us", str(t), "--param", "min-neighbors", "--values", THR_STD], {}) for r, t in grid]

    if alg == "ynoise":
        if ds == "drive":
            grid = [(2, 16000)]
        elif ds == "led":
            grid = [(2, 8000)]
        elif ds == "dvsclean":
            grid = [(4, 32000)]
        else:
            grid = [(r, t) for r in (2, 3, 4) for t in (16000, 32000, 64000, 128000)]
        return [(f"ynoise_r{r}_tau{t}", ["--method", "ynoise", "--engine", "cpp", "--radius-px", str(r), "--time-us", str(t), "--param", "min-neighbors", "--values", THR_STD], {}) for r, t in grid]

    if alg == "ts":
        if ds == "led":
            grid = [(1, 8000)]
        elif ds == "dvsclean":
            grid = [(r, t) for r in (1, 2) for t in (16000, 32000, 64000, 128000)]
        elif ds == "drive":
            grid = [(2, t) for t in (16000, 32000, 64000)]
        else:
            grid = [(r, t) for r in (1, 2, 3) for t in (16000, 32000, 64000, 128000)]
        return [(f"ts_r{r}_tau{t}", ["--method", "ts", "--engine", "cpp", "--radius-px", str(r), "--time-us", str(t), "--param", "min-neighbors", "--values", THR_TS], {}) for r, t in grid]

    if alg == "n149_v22":
        params = {
            "drive": (2, 32000, "1.75", "0.05"),
            "ped": (5, 256000, "2.75", "0.25"),
            "bike": (5, 256000, "2.75", "0.25"),
            "dvsclean": (5, 128000, "2.5", "0.25"),
            "led": (2, 8000, "2.0", "1.0"),
        }
        r, tau, sigma, alpha = params[ds]
        env = {
            "MYEVS_N149_HOT_BITS": "8",
            "MYEVS_N149_HOT_DECAY_K": "2",
            "MYEVS_N149_SIGMA": sigma,
            "MYEVS_N149_ALPHA_FIXED": alpha,
        }
        return [(f"n149_v22_r{r}_tau{tau}", ["--method", "n149", "--engine", "cpp", "--radius-px", str(r), "--time-us", str(tau), "--param", "min-neighbors", "--values", THR_STD], env)]

    raise ValueError(f"unknown algorithm: {alg}")


def run_case(sample: Sample, alg: str, tag: str, args: list[str], env_extra: dict[str, str], out_csv: Path) -> None:
    env = os.environ.copy()
    env.update(env_extra)
    cmd = [
        str(PY), "-m", "myevs.cli", "roc",
        "--clean", sample.clean,
        "--noisy", sample.noisy,
        "--assume", "npy",
        "--width", str(sample.width),
        "--height", str(sample.height),
        "--tick-ns", "1000",
        "--match-us", "0",
        "--match-bin-radius", "0",
        "--tag", tag,
        "--out-csv", str(out_csv),
        "--append",
    ] + args
    subprocess.run(cmd, cwd=ROOT, env=env, check=True, timeout=1200, capture_output=True, text=True)


def best_auc_tag_max_f1(csv_path: Path) -> dict[str, object]:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"empty csv: {csv_path}")
    for col in ("auc", "f1", "value", "tpr", "fpr", "precision"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    tag_auc = df.groupby("tag", as_index=False)["auc"].max().sort_values("auc", ascending=False)
    best_tag = str(tag_auc.iloc[0]["tag"])
    best_auc = float(tag_auc.iloc[0]["auc"])
    sub = df[df["tag"] == best_tag].copy()
    best_f1_row = sub.sort_values(["f1", "tpr"], ascending=[False, False]).iloc[0]
    best_auc_row = sub.sort_values(["auc", "f1"], ascending=[False, False]).iloc[0]
    return {
        "best_tag": best_tag,
        "auc": best_auc,
        "f1_at_auc_best": float(best_f1_row["f1"]),
        "f1_value": best_f1_row.get("value", ""),
        "auc_best_value": best_auc_row.get("value", ""),
        "tpr_at_f1": best_f1_row.get("tpr", ""),
        "fpr_at_f1": best_f1_row.get("fpr", ""),
        "precision_at_f1": best_f1_row.get("precision", ""),
    }


def run_one(sample: Sample, alg: str, force: bool) -> dict[str, object]:
    out_dir = OUT_ROOT / sample.dataset / alg
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"roc_{alg}_{sample.level}.csv"
    if force and out_csv.exists():
        out_csv.unlink()
    if not out_csv.exists():
        for tag, args, env in cases_for(sample, alg):
            run_case(sample, alg, tag, args, env, out_csv)
    best = best_auc_tag_max_f1(out_csv)
    return {
        "dataset": sample.dataset,
        "level": sample.level,
        "algorithm": alg,
        "roc_csv": str(out_csv),
        **best,
    }


def parse_csv_set(text: str, allowed: tuple[str, ...] | list[str]) -> list[str]:
    allowed_set = set(allowed)
    if text.strip().lower() == "all":
        return list(allowed)
    out = []
    for item in text.split(","):
        v = item.strip().lower()
        if not v:
            continue
        if v not in allowed_set:
            raise ValueError(f"unknown item '{v}', allowed: {', '.join(allowed)}")
        out.append(v)
    return list(dict.fromkeys(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="all", help="all or comma list: drive,ped,bike,dvsclean,led")
    ap.add_argument("--algorithms", default=",".join(DEFAULT_ALGS), help="all or comma list")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="delete and rerun per-sample ROC CSVs")
    args = ap.parse_args()

    datasets = parse_csv_set(args.datasets, ("drive", "ped", "bike", "dvsclean", "led"))
    algorithms = parse_csv_set(args.algorithms, DEFAULT_ALGS)
    samples = [s for s in all_samples() if s.dataset in datasets]
    missing = [s for s in samples if not (Path(s.clean).exists() and Path(s.noisy).exists())]
    if missing:
        raise FileNotFoundError("missing inputs:\n" + "\n".join(f"{s.dataset}/{s.level}: {s.clean} | {s.noisy}" for s in missing))

    jobs = [(s, alg) for s in samples for alg in algorithms]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[chapter7-f1] jobs={len(jobs)} samples={len(samples)} algorithms={','.join(algorithms)} workers={args.workers}")

    rows = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_one, s, alg, args.force): (s, alg) for s, alg in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            s, alg = futures[fut]
            row = fut.result()
            rows.append(row)
            rate = i / max(time.time() - t0, 1e-6)
            print(
                f"[{i}/{len(jobs)} | {rate:.2f} job/s] {s.dataset}/{s.level}/{alg} "
                f"AUC={row['auc']:.6f} F1={row['f1_at_auc_best']:.6f} tag={row['best_tag']}",
                flush=True,
            )

    df = pd.DataFrame(rows).sort_values(["dataset", "level", "algorithm"])
    summary = OUT_ROOT / "chapter7_f1_summary.csv"
    df.to_csv(summary, index=False)

    mean = df.groupby(["dataset", "algorithm"], as_index=False).agg(
        mean_auc=("auc", "mean"),
        mean_f1=("f1_at_auc_best", "mean"),
    )
    mean.to_csv(OUT_ROOT / "chapter7_f1_dataset_mean.csv", index=False)
    print(f"[chapter7-f1] wrote {summary}")
    print(mean.sort_values(["dataset", "mean_auc"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
