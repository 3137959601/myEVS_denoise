"""Sweep monotone hot-discount functions for N149 v2.2 on Ped/Bike 3.3V.

The C++ implementation maps the integer hot state H to q=H/Q, where Q is one
normalized hot-count unit.  Each candidate below is a monotone decreasing
function f(q) in [fmin, 1], except the rational baseline.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PY = Path(r"D:/software/Anaconda_envs/envs/myEVS/python.exe")
OUT_DIR = ROOT / "data" / "Hyperparameter ablation_study" / "hot_func_sweep_20260605"
THR = "0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,8"


@dataclass(frozen=True)
class Sample:
    key: str
    label: str
    clean: Path
    noisy: Path
    width: int = 346
    height: int = 260
    radius: int = 5
    tau_us: int = 256000
    sigma: str = "2.75"
    alpha: str = "0.25"


@dataclass(frozen=True)
class Candidate:
    key: str
    family: str
    params: dict[str, str]
    formula: str


SAMPLES = (
    Sample(
        key="ped_3p3",
        label="Ped 3.3V",
        clean=Path(r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3_signal_only.npy"),
        noisy=Path(r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3.npy"),
    ),
    Sample(
        key="bike_3p3",
        label="Bike 3.3V",
        clean=Path(r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_3.3_signal_only.npy"),
        noisy=Path(r"D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_3.3.npy"),
    ),
)


def fmt_num(v: float) -> str:
    return ("%g" % v).replace(".", "p")


def candidates() -> list[Candidate]:
    out: list[Candidate] = []

    for k in (1, 2, 3, 4):
        out.append(
            Candidate(
                key=f"rational_K{k}",
                family="rational",
                params={"MYEVS_N149_HOT_FUNC": "rational", "MYEVS_N149_HOT_K": str(k)},
                formula=f"(q+1)/({k}q+1)",
            )
        )

    for fmin in (0.4, 0.5, 0.6):
        for c in (0.5, 1.0, 2.0, 4.0):
            out.append(
                Candidate(
                    key=f"exp_f{fmt_num(fmin)}_c{fmt_num(c)}",
                    family="exp",
                    params={
                        "MYEVS_N149_HOT_FUNC": "exp",
                        "MYEVS_N149_HOT_FMIN": str(fmin),
                        "MYEVS_N149_HOT_C": str(c),
                    },
                    formula=f"{fmin}+(1-{fmin})exp(-{c}q)",
                )
            )

    for fmin in (0.4, 0.5, 0.6):
        for q0 in (0.5, 1.0, 2.0):
            for p in (1.0, 2.0):
                out.append(
                    Candidate(
                        key=f"hill_f{fmt_num(fmin)}_q{fmt_num(q0)}_p{fmt_num(p)}",
                        family="hill",
                        params={
                            "MYEVS_N149_HOT_FUNC": "hill",
                            "MYEVS_N149_HOT_FMIN": str(fmin),
                            "MYEVS_N149_HOT_Q0": str(q0),
                            "MYEVS_N149_HOT_P": str(p),
                        },
                        formula=f"{fmin}+(1-{fmin})/(1+(q/{q0})^{p})",
                    )
                )

    for fmin in (0.4, 0.5, 0.6):
        for c in (0.5, 1.0, 2.0):
            for p in (1.0, 2.0):
                out.append(
                    Candidate(
                        key=f"power_f{fmt_num(fmin)}_c{fmt_num(c)}_p{fmt_num(p)}",
                        family="power",
                        params={
                            "MYEVS_N149_HOT_FUNC": "power",
                            "MYEVS_N149_HOT_FMIN": str(fmin),
                            "MYEVS_N149_HOT_C": str(c),
                            "MYEVS_N149_HOT_P": str(p),
                        },
                        formula=f"{fmin}+(1-{fmin})/(1+{c}q)^{p}",
                    )
                )

    for fmin in (0.4, 0.5, 0.6):
        for c in (0.25, 0.5, 1.0):
            out.append(
                Candidate(
                    key=f"linear_f{fmt_num(fmin)}_c{fmt_num(c)}",
                    family="linear",
                    params={
                        "MYEVS_N149_HOT_FUNC": "linear",
                        "MYEVS_N149_HOT_FMIN": str(fmin),
                        "MYEVS_N149_HOT_C": str(c),
                    },
                    formula=f"max({fmin},1-{c}q)",
                )
            )

    return out


def run_one(sample: Sample, cand: Candidate, force: bool) -> dict[str, object]:
    for path in (sample.clean, sample.noisy):
        if not path.exists():
            raise FileNotFoundError(path)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"hotfunc_{sample.key}_{cand.key}"
    out_csv = OUT_DIR / f"{tag}.csv"
    if force and out_csv.exists():
        out_csv.unlink()

    if not out_csv.exists():
        env = os.environ.copy()
        env.update(
            {
                "MYEVS_N149_HOT_BITS": "8",
                "MYEVS_N149_HOT_INT_BITS": "3",
                "MYEVS_N149_HOT_DECAY_K": "2",
                "MYEVS_N149_SIGMA": sample.sigma,
                "MYEVS_N149_ALPHA_FIXED": sample.alpha,
            }
        )
        env.update(cand.params)
        cmd = [
            str(PY),
            "-m",
            "myevs.cli",
            "roc",
            "--clean",
            str(sample.clean),
            "--noisy",
            str(sample.noisy),
            "--assume",
            "npy",
            "--width",
            str(sample.width),
            "--height",
            str(sample.height),
            "--tick-ns",
            "1000",
            "--engine",
            "cpp",
            "--method",
            "n149",
            "--radius-px",
            str(sample.radius),
            "--time-us",
            str(sample.tau_us),
            "--param",
            "min-neighbors",
            "--values",
            THR,
            "--match-us",
            "0",
            "--match-bin-radius",
            "0",
            "--tag",
            tag,
            "--out-csv",
            str(out_csv),
        ]
        subprocess.run(cmd, cwd=ROOT, env=env, check=True, timeout=1200, capture_output=True, text=True)

    df = pd.read_csv(out_csv)
    for col in ("auc", "f1", "value", "tpr", "fpr", "precision"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    auc = float(df["auc"].max())
    best_f1 = df.sort_values(["f1", "tpr"], ascending=[False, False]).iloc[0]
    return {
        "sample": sample.key,
        "sample_label": sample.label,
        "candidate": cand.key,
        "family": cand.family,
        "formula": cand.formula,
        "auc": auc,
        "best_f1": float(best_f1["f1"]),
        "f1_threshold": best_f1.get("value", ""),
        "tpr_at_f1": best_f1.get("tpr", ""),
        "fpr_at_f1": best_f1.get("fpr", ""),
        "precision_at_f1": best_f1.get("precision", ""),
        "roc_csv": str(out_csv),
    }


def write_outputs(rows: list[dict[str, object]]) -> None:
    raw = pd.DataFrame(rows).sort_values(["family", "candidate", "sample"])
    raw.to_csv(OUT_DIR / "hot_func_sweep_raw.csv", index=False, encoding="utf-8-sig")

    agg = (
        raw.groupby(["candidate", "family", "formula"], as_index=False)
        .agg(mean_auc=("auc", "mean"), mean_f1=("best_f1", "mean"), min_auc=("auc", "min"), min_f1=("best_f1", "min"))
        .sort_values(["mean_auc", "mean_f1"], ascending=[False, False])
    )
    agg.to_csv(OUT_DIR / "hot_func_sweep_ranked.csv", index=False, encoding="utf-8-sig")

    rows_out = []
    for _, r in agg.iterrows():
        row = {"candidate": r["candidate"], "family": r["family"], "formula": r["formula"]}
        sub = raw[raw["candidate"] == r["candidate"]]
        for sample in SAMPLES:
            s = sub[sub["sample"] == sample.key].iloc[0]
            row[f"{sample.label} AUC"] = s["auc"]
            row[f"{sample.label} F1"] = s["best_f1"]
            row[f"{sample.label} F1 thr"] = s["f1_threshold"]
        row["mean_auc"] = r["mean_auc"]
        row["mean_f1"] = r["mean_f1"]
        rows_out.append(row)
    pd.DataFrame(rows_out).to_csv(OUT_DIR / "hot_func_sweep_summary.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    global OUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--families", default="", help="Optional comma-separated family filter.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    OUT_DIR = Path(args.out_dir)
    (OUT_DIR.parent / "logs").mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cands = candidates()
    if args.families.strip():
        keep = {x.strip().lower() for x in args.families.split(",") if x.strip()}
        cands = [c for c in cands if c.family.lower() in keep]
    if args.limit > 0:
        cands = cands[: args.limit]
    with (OUT_DIR.parent / "logs" / f"hot_func_sweep_{time.strftime('%Y%m%d_%H%M%S')}.json").open("w", encoding="utf-8") as f:
        json.dump({"script": __file__, "out_dir": str(OUT_DIR), "args": vars(args), "candidates": [c.key for c in cands]}, f, indent=2)

    tasks = [(sample, cand) for cand in cands for sample in SAMPLES]
    print(f"[hot-func] candidates={len(cands)} tasks={len(tasks)} out={OUT_DIR}")
    rows: list[dict[str, object]] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_one, sample, cand, args.force): (sample, cand) for sample, cand in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            rows.append(row)
            rate = i / max(time.time() - t0, 1e-6)
            print(
                f"[{i:03d}/{len(tasks)} | {rate:.2f} t/s] "
                f"{row['sample_label']} {row['candidate']} auc={row['auc']:.6f} f1={row['best_f1']:.6f}",
                flush=True,
            )

    write_outputs(rows)
    ranked = pd.read_csv(OUT_DIR / "hot_func_sweep_ranked.csv")
    print("\n[hot-func] top 15:")
    print(ranked.head(15).to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
