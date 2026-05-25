from __future__ import annotations

import csv
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd


PY = Path(r"D:/software/Anaconda_envs/envs/myEVS/python.exe")
ROOT = Path(r"d:/hjx_workspace/scientific_reserach/projects/myEVS")
NPY_ROOT = Path(r"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy")
OUT_ROOT = ROOT / "data/LED/scene_sweep_full"

SCENES = [
    "scene_100",
    "scene_1004",
    "scene_1018",
    "scene_1028",
    "scene_1032",
    "scene_1033",
    "scene_1034",
    "scene_1043",
    "scene_1045",
    "scene_1046",
]

THR_VALUES = "8,16,24,32,48,64"
TAU_MIN = 2000
TAU_MAX = 512000
R_MIN = 1
R_MAX = 6
EPS = 5e-4
MAX_ITERS = 8


@dataclass
class EvalRow:
    scene: str
    r: int
    tau: int
    auc: float
    f1: float
    tag: str
    csv_path: Path


def _scene_paths(scene: str) -> Tuple[Path, Path]:
    d = NPY_ROOT / scene / "slices_00031_00040_100ms"
    clean = d / f"{scene}_100ms_signal_only.npy"
    noisy = d / f"{scene}_100ms_labeled.npy"
    return clean, noisy


def _run_roc(scene: str, r: int, tau: int, out_csv: Path) -> EvalRow:
    clean, noisy = _scene_paths(scene)
    tag = f"evflow_r{r}_tau{tau}_{scene}"
    cmd = [
        str(PY),
        "-m",
        "myevs.cli",
        "roc",
        "--clean",
        str(clean),
        "--noisy",
        str(noisy),
        "--assume",
        "npy",
        "--width",
        "1280",
        "--height",
        "720",
        "--tick-ns",
        "1000",
        "--engine",
        "cpp",
        "--method",
        "evflow",
        "--radius-px",
        str(r),
        "--time-us",
        str(tau),
        "--param",
        "min-neighbors",
        "--values",
        THR_VALUES,
        "--match-us",
        "0",
        "--match-bin-radius",
        "0",
        "--tag",
        tag,
        "--out-csv",
        str(out_csv),
        "--append",
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)

    df = pd.read_csv(out_csv)
    dft = df[df["tag"] == tag]
    if dft.empty:
        raise RuntimeError(f"no rows for tag {tag}")
    auc = float(dft["auc"].iloc[0])
    i = dft["f1"].astype(float).idxmax()
    f1 = float(dft.loc[i, "f1"])
    return EvalRow(scene=scene, r=r, tau=tau, auc=auc, f1=f1, tag=tag, csv_path=out_csv)


def _candidate_taus(best_tau: int) -> List[int]:
    vals = {best_tau}
    vals.add(max(TAU_MIN, best_tau // 2))
    vals.add(min(TAU_MAX, best_tau * 2))
    vals.add(max(TAU_MIN, int(best_tau * 0.75)))
    vals.add(min(TAU_MAX, int(best_tau * 1.5)))
    return sorted(v for v in vals if TAU_MIN <= v <= TAU_MAX)


def _candidate_rs(best_r: int) -> List[int]:
    return sorted({v for v in [best_r - 1, best_r, best_r + 1] if R_MIN <= v <= R_MAX})


def run_scene(scene: str) -> dict:
    out_dir = OUT_ROOT / scene / "evflow"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "roc_evflow_adaptive.csv"
    if out_csv.exists():
        out_csv.unlink()

    tested = set()
    rows: List[EvalRow] = []

    # seed grid
    seed_rs = [2, 3]
    seed_taus = [8000, 16000, 32000]
    for r in seed_rs:
        for tau in seed_taus:
            k = (r, tau)
            if k in tested:
                continue
            tested.add(k)
            rows.append(_run_roc(scene, r, tau, out_csv))

    best = max(rows, key=lambda x: x.auc)
    prev_best_auc = best.auc

    for _ in range(MAX_ITERS):
        c_rs = _candidate_rs(best.r)
        c_taus = _candidate_taus(best.tau)
        iter_new = []
        for r in c_rs:
            for tau in c_taus:
                k = (r, tau)
                if k in tested:
                    continue
                tested.add(k)
                iter_new.append((r, tau))
        if not iter_new:
            break

        iter_rows: List[EvalRow] = []
        for r, tau in iter_new:
            iter_rows.append(_run_roc(scene, r, tau, out_csv))
        rows.extend(iter_rows)
        new_best = max(rows, key=lambda x: x.auc)

        improved = new_best.auc - prev_best_auc
        hit_boundary = (
            new_best.r in (R_MIN, R_MAX)
            or new_best.tau in (TAU_MIN, TAU_MAX)
            or new_best.tau in (min(c_taus), max(c_taus))
        )
        best = new_best

        if improved < EPS and not hit_boundary:
            break
        prev_best_auc = max(prev_best_auc, new_best.auc)

    # update scene summary row for evflow
    sum_csv = OUT_ROOT / scene / "scene_sweep_summary.csv"
    if sum_csv.exists():
        df_sum = pd.read_csv(sum_csv)
        mcol = "Method" if "Method" in df_sum.columns else "method"
        auc_col = "AUC_best" if "AUC_best" in df_sum.columns else "auc"
        f1_col = "F1" if "F1" in df_sum.columns else "f1"
        tag_col = "BestTagByAUC" if "BestTagByAUC" in df_sum.columns else "tag"
        mask = df_sum[mcol].astype(str).str.lower() == "evflow"
        if mask.any():
            df_sum.loc[mask, auc_col] = best.auc
            if f1_col in df_sum.columns:
                df_sum.loc[mask, f1_col] = best.f1
            if tag_col in df_sum.columns:
                df_sum.loc[mask, tag_col] = best.tag
        else:
            add = {c: "" for c in df_sum.columns}
            add[mcol] = "evflow"
            add[auc_col] = best.auc
            if f1_col in df_sum.columns:
                add[f1_col] = best.f1
            if tag_col in df_sum.columns:
                add[tag_col] = best.tag
            df_sum = pd.concat([df_sum, pd.DataFrame([add])], ignore_index=True)
        df_sum.to_csv(sum_csv, index=False)

    return {
        "scene": scene,
        "best_auc": best.auc,
        "best_f1": best.f1,
        "best_r": best.r,
        "best_tau": best.tau,
        "best_tag": best.tag,
        "trials": len(tested),
    }


def main() -> int:
    out_summary = ROOT / "data/LED/evflow_adaptive10_best_20260522.csv"
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    results = []

    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(run_scene, s): s for s in SCENES}
        for fut in as_completed(futs):
            s = futs[fut]
            res = fut.result()
            print(
                f"[DONE] {s}: auc={res['best_auc']:.6f} "
                f"r={res['best_r']} tau={res['best_tau']} trials={res['trials']}",
                flush=True,
            )
            results.append(res)

    results = sorted(results, key=lambda x: x["scene"])
    with out_summary.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["scene", "best_auc", "best_f1", "best_r", "best_tau", "best_tag", "trials"],
        )
        w.writeheader()
        w.writerows(results)

    print(f"saved: {out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

