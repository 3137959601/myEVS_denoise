from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from myevs.denoise.ops.ebfopt_part2.n149_n145_s52_euclid_compactlut_backbone import score_stream_n149
from myevs.timebase import TimeBase


@dataclass(frozen=True)
class FixedCfg:
    method: str
    radius: int
    tau_us: int
    threshold: float


class _Events:
    def __init__(self, arr: np.ndarray):
        self.t = np.ascontiguousarray(arr["t"].astype(np.uint64, copy=False))
        self.x = np.ascontiguousarray(arr["x"].astype(np.int32, copy=False))
        self.y = np.ascontiguousarray(arr["y"].astype(np.int32, copy=False))
        self.p = np.ascontiguousarray(arr["p"].astype(np.int8, copy=False))
        self.label = np.ascontiguousarray((arr["label"] > 0).astype(np.uint8, copy=False))


def _read_scene_best(ref_csv: Path) -> dict[str, FixedCfg]:
    df = pd.read_csv(ref_csv)
    out: dict[str, FixedCfg] = {}
    for _, r in df.iterrows():
        method = str(r["Method"]).strip().lower()
        if method not in {
            "baf",
            "stcf",
            "ebf",
            "knoise",
            "evflow",
            "ynoise",
            "ts",
            "mlpf",
            "pfd",
            "n149",
        }:
            continue
        out[method] = FixedCfg(
            method=method,
            radius=int(r["BestRadius"]),
            tau_us=int(r["BestTauUs"]),
            threshold=float(r["Threshold@Best-DA"]),
        )
    return out


def _run_cli_point(
    py: str,
    *,
    clean: Path,
    noisy: Path,
    width: int,
    height: int,
    tick_ns: float,
    method: str,
    engine: str,
    radius: int,
    tau_us: int,
    threshold: float,
    tag: str,
    out_csv: Path,
    mlpf_model: str = "",
    mlpf_patch: int = 7,
) -> pd.Series:
    cmd = [
        py,
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
        str(width),
        "--height",
        str(height),
        "--tick-ns",
        str(tick_ns),
        "--engine",
        engine,
        "--method",
        method,
        "--radius-px",
        str(radius),
        "--time-us",
        str(tau_us),
        "--param",
        "min-neighbors",
        "--values",
        f"{float(threshold):.8f}",
        "--match-us",
        "0",
        "--match-bin-radius",
        "0",
        "--tag",
        tag,
        "--out-csv",
        str(out_csv),
    ]
    if method == "mlpf":
        if not mlpf_model or not Path(mlpf_model).exists():
            raise FileNotFoundError(f"MLPF model not found: {mlpf_model}")
        cmd.extend(["--mlpf-model", str(mlpf_model), "--mlpf-patch", str(int(mlpf_patch))])
    subprocess.run(cmd, check=True)
    df = pd.read_csv(out_csv)
    pick = df.iloc[(df["value"] - float(threshold)).abs().argmin()]
    return pick


def _point_from_scores(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    y = labels.astype(np.uint8, copy=False)
    keep = (scores > float(threshold)).astype(np.uint8, copy=False)
    tp = int(np.sum((keep == 1) & (y == 1)))
    fp = int(np.sum((keep == 1) & (y == 0)))
    fn = int(np.sum((keep == 0) & (y == 1)))
    tn = int(np.sum((keep == 0) & (y == 0)))
    tpr = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    fpr = (fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1 = (2.0 * precision * tpr / (precision + tpr)) if (precision + tpr) > 0 else 0.0
    sr = tpr
    nr = 1.0 - fpr
    da = 0.5 * (sr + nr)
    return {"tpr": tpr, "fpr": fpr, "precision": precision, "f1": f1, "sr": sr, "nr": nr, "da": da}


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate one LED scene using fixed best points copied from a reference scene summary.")
    ap.add_argument("--ref-summary", default="data/LED/scene_sweep_full/scene_100/scene_sweep_summary.csv")
    ap.add_argument("--target-scene", default="scene_1004")
    ap.add_argument("--npy-root", default=r"D:\hjx_workspace\scientific_reserach\dataset\LED\converted_npy")
    ap.add_argument("--out-root", default="data/LED/scene_fixed_from_scene100")
    ap.add_argument("--python", default=r"D:/software/Anaconda_envs/envs/myEVS/python.exe")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--tick-ns", type=float, default=1000.0)
    ap.add_argument("--mlpf-model", default="data/LED/models/mlpf_torch_scene_1004.pt")
    ap.add_argument("--mlpf-patch", type=int, default=7)
    args = ap.parse_args()

    ref = Path(args.ref_summary)
    fixed = _read_scene_best(ref)
    required = ["baf", "stcf", "ebf", "knoise", "evflow", "ynoise", "ts", "mlpf", "pfd", "n149"]
    miss = [m for m in required if m not in fixed]
    if miss:
        raise RuntimeError(f"Missing methods in ref summary: {miss}")

    noisy = Path(args.npy_root) / args.target_scene / "slices_00031_00040_100ms" / f"{args.target_scene}_100ms_labeled.npy"
    clean = noisy.with_name(noisy.name.replace("_labeled.npy", "_signal_only.npy"))
    if not noisy.exists() or not clean.exists():
        raise FileNotFoundError(f"Missing scene npy pair: {noisy} / {clean}")

    out_dir = Path(args.out_root) / args.target_scene
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str | int]] = []

    engines = {
        "baf": "python",
        "stcf": "python",
        "ebf": "python",
        "knoise": "python",
        "evflow": "numba",
        "ynoise": "python",
        "ts": "numba",
        "mlpf": "python",
        "pfd": "numba",
    }
    order = ["baf", "stcf", "ebf", "knoise", "evflow", "ynoise", "ts", "mlpf", "pfd"]
    print(f"[fixed-eval] target={args.target_scene} methods={len(required)}")
    for i, m in enumerate(order, start=1):
        cfg = fixed[m]
        print(
            f"[fixed-eval] {i}/{len(required)} {m}: r={cfg.radius} tau={cfg.tau_us} thr={cfg.threshold}",
            flush=True,
        )
        out_csv = out_dir / m / f"roc_{m}_{args.target_scene}_fixed_from_scene100.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        tag = f"{m}_fixed_scene100_to_{args.target_scene}"
        p = _run_cli_point(
            args.python,
            clean=clean,
            noisy=noisy,
            width=int(args.width),
            height=int(args.height),
            tick_ns=float(args.tick_ns),
            method="stc" if m == "stcf" else m,
            engine=engines[m],
            radius=int(cfg.radius),
            tau_us=int(cfg.tau_us),
            threshold=float(cfg.threshold),
            tag=tag,
            out_csv=out_csv,
            mlpf_model=str(args.mlpf_model),
            mlpf_patch=int(args.mlpf_patch),
        )
        sr = float(p["tpr"])
        nr = 1.0 - float(p["fpr"])
        da = 0.5 * (sr + nr)
        rows.append(
            {
                "Method": m,
                "BestRadius_from_scene100": int(cfg.radius),
                "BestTauUs_from_scene100": int(cfg.tau_us),
                "Threshold_from_scene100": float(cfg.threshold),
                "AUC_on_scene100": np.nan,
                "TPR_on_scene1004": float(p["tpr"]),
                "FPR_on_scene1004": float(p["fpr"]),
                "SR_on_scene1004": sr,
                "NR_on_scene1004": nr,
                "DA_on_scene1004": da,
                "F1_on_scene1004": float(p["f1"]),
            }
        )

    # n149 fixed point
    print(
        f"[fixed-eval] {len(required)}/{len(required)} n149: r={fixed['n149'].radius} tau={fixed['n149'].tau_us} thr={fixed['n149'].threshold}",
        flush=True,
    )
    arr = np.load(noisy, mmap_mode="r")
    ev = _Events(arr)
    tb = TimeBase(tick_ns=float(args.tick_ns))
    ncfg = fixed["n149"]
    scores = score_stream_n149(
        ev,
        width=int(args.width),
        height=int(args.height),
        radius_px=int(ncfg.radius),
        tau_us=int(ncfg.tau_us),
        tb=tb,
    )
    pt = _point_from_scores(scores, ev.label, float(ncfg.threshold))
    rows.append(
        {
            "Method": "n149",
            "BestRadius_from_scene100": int(ncfg.radius),
            "BestTauUs_from_scene100": int(ncfg.tau_us),
            "Threshold_from_scene100": float(ncfg.threshold),
            "AUC_on_scene100": np.nan,
            "TPR_on_scene1004": float(pt["tpr"]),
            "FPR_on_scene1004": float(pt["fpr"]),
            "SR_on_scene1004": float(pt["sr"]),
            "NR_on_scene1004": float(pt["nr"]),
            "DA_on_scene1004": float(pt["da"]),
            "F1_on_scene1004": float(pt["f1"]),
        }
    )

    ref_df = pd.read_csv(ref)[["Method", "AUC_best", "DA_best", "F1", "Threshold@Best-DA", "BestRadius", "BestTauUs"]].copy()
    ref_df["Method"] = ref_df["Method"].astype(str).str.lower()
    out = pd.DataFrame(rows)
    out["Method"] = out["Method"].astype(str).str.lower()
    out = out.merge(
        ref_df.rename(
            columns={
                "AUC_best": "AUC_scene100_best",
                "DA_best": "DA_scene100_best",
                "F1": "F1_scene100_best",
                "Threshold@Best-DA": "Threshold_scene100_best",
                "BestRadius": "BestRadius_scene100_best",
                "BestTauUs": "BestTauUs_scene100_best",
            }
        ),
        on="Method",
        how="left",
    )
    out = out.sort_values("DA_on_scene1004", ascending=False)
    out_csv = out_dir / "scene1004_fixed_from_scene100_summary.csv"
    out.to_csv(out_csv, index=False)
    print(f"[fixed-eval] saved: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

