from __future__ import annotations

import csv
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

THR = "8,16,24,32,48,64"
COARSE_RS = [2, 3]
COARSE_TAUS = [8000, 16000, 32000]
MAX_EVENTS_COARSE = 300000
MAX_WORKERS = 4


@dataclass
class BestCfg:
    scene: str
    r: int
    tau: int
    thr: float
    auc_coarse: float
    f1_coarse: float
    auc_full: float
    f1_full: float
    tag_full: str
    trials_coarse: int


def scene_paths(scene: str) -> Tuple[Path, Path]:
    d = NPY_ROOT / scene / "slices_00031_00040_100ms"
    return d / f"{scene}_100ms_signal_only.npy", d / f"{scene}_100ms_labeled.npy"


def _truncate_if_needed(scene: str, src: Path, max_events: int, kind: str) -> Path:
    if max_events <= 0:
        return src
    cache = ROOT / "data/LED/_compact_cache_300k" / scene
    cache.mkdir(parents=True, exist_ok=True)
    out = cache / f"{src.stem}_{kind}_n{max_events}.npy"
    cmd = [
        str(PY),
        "scripts/truncate_npy_events.py",
        "--in",
        str(src),
        "--out",
        str(out),
        "--max-events",
        str(max_events),
        "--overwrite",
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
    return out


def run_roc(scene: str, r: int, tau: int, out_csv: Path, max_events: int, tag_suffix: str) -> None:
    clean, noisy = scene_paths(scene)
    clean = _truncate_if_needed(scene, clean, max_events, "clean")
    noisy = _truncate_if_needed(scene, noisy, max_events, "noisy")
    tag = f"evflow_r{r}_tau{tau}_{scene}_{tag_suffix}"
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


def best_from_csv(out_csv: Path) -> Tuple[float, float, int, int, float, str, int]:
    df = pd.read_csv(out_csv)
    i = df["auc"].astype(float).idxmax()
    row = df.loc[i]
    tag = str(row["tag"])
    # tag: evflow_r{r}_tau{tau}_{scene}_{suffix}
    parts = tag.split("_")
    r = int(parts[1][1:])
    tau = int(parts[2][3:])
    thr = float(row["value"])
    auc = float(row["auc"])
    f1 = float(row["f1"])
    trials = int(df["tag"].nunique())
    return auc, f1, r, tau, thr, tag, trials


def eval_scene(scene: str) -> BestCfg:
    out_dir = OUT_ROOT / scene / "evflow"
    out_dir.mkdir(parents=True, exist_ok=True)
    coarse_csv = out_dir / "roc_evflow_phase1_300k.csv"
    full_csv = out_dir / "roc_evflow_phase2_full_best.csv"
    if coarse_csv.exists():
        coarse_csv.unlink()
    if full_csv.exists():
        full_csv.unlink()

    for r in COARSE_RS:
        for tau in COARSE_TAUS:
            run_roc(scene, r, tau, coarse_csv, MAX_EVENTS_COARSE, "p1")

    auc_c, f1_c, r_best, tau_best, thr_best, _, trials = best_from_csv(coarse_csv)

    # full evaluation only at best (r,tau)
    run_roc(scene, r_best, tau_best, full_csv, 0, "p2full")
    auc_f, f1_f, _, _, _, tag_f, _ = best_from_csv(full_csv)

    # update main scene summary
    summary = OUT_ROOT / scene / "scene_sweep_summary.csv"
    if summary.exists():
        df = pd.read_csv(summary)
        mcol = "Method" if "Method" in df.columns else "method"
        auc_col = "AUC_best" if "AUC_best" in df.columns else "auc"
        f1_col = "F1" if "F1" in df.columns else "f1"
        tag_col = "BestTagByAUC" if "BestTagByAUC" in df.columns else "tag"
        mask = df[mcol].astype(str).str.lower() == "evflow"
        if mask.any():
            df.loc[mask, auc_col] = auc_f
            if f1_col in df.columns:
                df.loc[mask, f1_col] = f1_f
            if tag_col in df.columns:
                df.loc[mask, tag_col] = tag_f
            df.to_csv(summary, index=False)

    return BestCfg(
        scene=scene,
        r=r_best,
        tau=tau_best,
        thr=thr_best,
        auc_coarse=auc_c,
        f1_coarse=f1_c,
        auc_full=auc_f,
        f1_full=f1_f,
        tag_full=tag_f,
        trials_coarse=trials,
    )


def main() -> int:
    out = ROOT / "data/LED/evflow_phase12_300k_best_20260523.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    results: List[BestCfg] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(eval_scene, s): s for s in SCENES}
        for fut in as_completed(futs):
            r = fut.result()
            print(
                f"[DONE] {r.scene} coarse_auc={r.auc_coarse:.6f} "
                f"full_auc={r.auc_full:.6f} best=(r{r.r},tau{r.tau},thr{r.thr})",
                flush=True,
            )
            results.append(r)

    results = sorted(results, key=lambda x: x.scene)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "scene",
                "best_r",
                "best_tau",
                "best_thr_phase1",
                "auc_phase1_300k",
                "f1_phase1_300k",
                "auc_phase2_full",
                "f1_phase2_full",
                "tag_phase2_full",
                "trials_phase1",
            ]
        )
        for r in results:
            w.writerow(
                [
                    r.scene,
                    r.r,
                    r.tau,
                    r.thr,
                    r.auc_coarse,
                    r.f1_coarse,
                    r.auc_full,
                    r.f1_full,
                    r.tag_full,
                    r.trials_coarse,
                ]
            )
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

