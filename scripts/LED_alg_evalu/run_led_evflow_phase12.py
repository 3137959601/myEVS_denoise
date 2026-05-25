from __future__ import annotations

import csv
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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

THR = "8,16,24,32,48,64"
R_BOUNDS = (1, 6)
TAU_BOUNDS = (2000, 512000)
EPS = 5e-4


@dataclass
class Best:
    scene: str
    auc: float
    f1: float
    r: int
    tau: int
    thr: float
    tag: str
    trials: int


def scene_paths(scene: str) -> Tuple[Path, Path]:
    d = NPY_ROOT / scene / "slices_00031_00040_100ms"
    return d / f"{scene}_100ms_signal_only.npy", d / f"{scene}_100ms_labeled.npy"


def run_roc(scene: str, r: int, tau: int, out_csv: Path) -> None:
    clean, noisy = scene_paths(scene)
    tag = f"evflow_r{r}_tau{tau}_{scene}"
    cmd = [
        str(PY), "-m", "myevs.cli", "roc",
        "--clean", str(clean), "--noisy", str(noisy),
        "--assume", "npy", "--width", "1280", "--height", "720",
        "--tick-ns", "1000", "--engine", "cpp", "--method", "evflow",
        "--radius-px", str(r), "--time-us", str(tau),
        "--param", "min-neighbors", "--values", THR,
        "--match-us", "0", "--match-bin-radius", "0",
        "--tag", tag, "--out-csv", str(out_csv), "--append",
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)


def best_from_csv(scene: str, out_csv: Path) -> Best:
    df = pd.read_csv(out_csv)
    # best by AUC
    i = df["auc"].astype(float).idxmax()
    row = df.loc[i]
    tag = str(row["tag"])
    parts = tag.split("_")
    r = int(parts[1][1:])  # rX
    tau = int(parts[2][3:])  # tauY
    thr = float(row["value"])
    auc = float(row["auc"])
    f1 = float(row["f1"])
    return Best(scene=scene, auc=auc, f1=f1, r=r, tau=tau, thr=thr, tag=tag, trials=df["tag"].nunique())


def eval_scene(scene: str) -> Best:
    out_dir = OUT_ROOT / scene / "evflow"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "roc_evflow_phase12.csv"
    if out_csv.exists():
        out_csv.unlink()

    tested = set()

    # phase 1 coarse
    coarse_r = [2, 3]
    coarse_tau = [8000, 16000, 32000]
    for r in coarse_r:
        for tau in coarse_tau:
            tested.add((r, tau))
            run_roc(scene, r, tau, out_csv)
    b1 = best_from_csv(scene, out_csv)

    # phase 2 refine only when on boundary of coarse grid
    phase1_r_min, phase1_r_max = min(coarse_r), max(coarse_r)
    phase1_t_min, phase1_t_max = min(coarse_tau), max(coarse_tau)
    hit = (b1.r in (phase1_r_min, phase1_r_max)) or (b1.tau in (phase1_t_min, phase1_t_max))
    if hit:
        cands_r = sorted({max(R_BOUNDS[0], b1.r - 1), b1.r, min(R_BOUNDS[1], b1.r + 1)})
        cands_tau = sorted({
            max(TAU_BOUNDS[0], b1.tau // 2),
            max(TAU_BOUNDS[0], int(b1.tau * 0.75)),
            b1.tau,
            min(TAU_BOUNDS[1], int(b1.tau * 1.5)),
            min(TAU_BOUNDS[1], b1.tau * 2),
        })
        for r in cands_r:
            for tau in cands_tau:
                if (r, tau) in tested:
                    continue
                tested.add((r, tau))
                run_roc(scene, r, tau, out_csv)
        b2 = best_from_csv(scene, out_csv)
        if b2.auc - b1.auc >= EPS:
            return Best(scene=scene, auc=b2.auc, f1=b2.f1, r=b2.r, tau=b2.tau, thr=b2.thr, tag=b2.tag, trials=len(tested))
    return Best(scene=scene, auc=b1.auc, f1=b1.f1, r=b1.r, tau=b1.tau, thr=b1.thr, tag=b1.tag, trials=len(tested))


def update_scene_summary(best: Best) -> None:
    f = OUT_ROOT / best.scene / "scene_sweep_summary.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    mcol = "Method" if "Method" in df.columns else "method"
    auc_col = "AUC_best" if "AUC_best" in df.columns else "auc"
    f1_col = "F1" if "F1" in df.columns else "f1"
    tag_col = "BestTagByAUC" if "BestTagByAUC" in df.columns else "tag"
    thr_col = "Threshold@Best-DA" if "Threshold@Best-DA" in df.columns else ("value" if "value" in df.columns else None)
    mask = df[mcol].astype(str).str.lower() == "evflow"
    if mask.any():
        df.loc[mask, auc_col] = best.auc
        if f1_col in df.columns:
            df.loc[mask, f1_col] = best.f1
        if tag_col in df.columns:
            df.loc[mask, tag_col] = best.tag
        if thr_col and thr_col in df.columns:
            df.loc[mask, thr_col] = best.thr
    df.to_csv(f, index=False)


def main() -> int:
    out = ROOT / "data/LED/evflow_phase12_best_20260522.csv"
    results: List[Best] = []

    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = {ex.submit(eval_scene, s): s for s in SCENES}
        for fut in as_completed(futs):
            b = fut.result()
            update_scene_summary(b)
            print(f"[DONE] {b.scene} auc={b.auc:.6f} r={b.r} tau={b.tau} thr={b.thr} trials={b.trials}", flush=True)
            results.append(b)

    results = sorted(results, key=lambda x: x.scene)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scene", "best_auc", "best_f1", "best_r", "best_tau", "best_thr", "best_tag", "trials"])
        for b in results:
            w.writerow([b.scene, b.auc, b.f1, b.r, b.tau, b.thr, b.tag, b.trials])
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

