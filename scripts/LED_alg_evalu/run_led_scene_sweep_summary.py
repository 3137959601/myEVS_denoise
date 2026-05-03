from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from myevs.timebase import TimeBase
from myevs.denoise.ops.ebfopt_part2.n149_n145_s52_euclid_compactlut_backbone import score_stream_n149


@dataclass(frozen=True)
class AlgoCfg:
    name: str
    method: str
    engine: str
    radii: list[int]
    taus_us: list[int]
    thresholds: list[float]


class _Events:
    def __init__(self, arr: np.ndarray):
        self.t = np.ascontiguousarray(arr["t"].astype(np.uint64, copy=False))
        self.x = np.ascontiguousarray(arr["x"].astype(np.int32, copy=False))
        self.y = np.ascontiguousarray(arr["y"].astype(np.int32, copy=False))
        self.p = np.ascontiguousarray(arr["p"].astype(np.int8, copy=False))
        self.label = np.ascontiguousarray((arr["label"] > 0).astype(np.uint8, copy=False))


def _csv_values(vals: list[float | int]) -> str:
    out = []
    for v in vals:
        if isinstance(v, float) and not float(v).is_integer():
            out.append(f"{float(v):.6f}".rstrip("0").rstrip("."))
        else:
            out.append(str(int(v)))
    return ",".join(out)


def _prepare_truncated_pair(noisy_path: Path, *, max_events: int, cache_root: Path) -> tuple[Path, Path]:
    clean_path = noisy_path.with_name(noisy_path.name.replace("_labeled.npy", "_signal_only.npy"))
    if max_events <= 0:
        return noisy_path, clean_path
    key = noisy_path.stem.replace("_100ms_labeled", "")
    out_dir = cache_root / key
    out_dir.mkdir(parents=True, exist_ok=True)
    trunc_noisy = out_dir / f"{key}_labeled_max{max_events}.npy"
    trunc_clean = out_dir / f"{key}_signal_only_max{max_events}.npy"
    if trunc_noisy.exists() and trunc_clean.exists():
        return trunc_noisy, trunc_clean
    noisy = np.load(noisy_path, mmap_mode="r")
    n = min(int(max_events), int(noisy.shape[0]))
    noisy_cut = np.array(noisy[:n], copy=True)
    np.save(trunc_noisy, noisy_cut)
    clean = np.load(clean_path, mmap_mode="r")
    if n > 0:
        tmax = int(noisy_cut["t"][-1])
        clean_cut = np.array(clean[clean["t"] <= tmax], copy=True)
    else:
        clean_cut = np.array(clean[:0], copy=True)
    np.save(trunc_clean, clean_cut)
    return trunc_noisy, trunc_clean


def _run_cli_roc(
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
    thresholds: list[float],
    tag: str,
    out_csv: Path,
    mlpf_model: str = "",
    mlpf_patch: int = 7,
) -> pd.DataFrame:
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
        _csv_values(thresholds),
        "--match-us",
        "0",
        "--match-bin-radius",
        "0",
        "--tag",
        tag,
        "--out-csv",
        str(out_csv),
    ]
    if str(method).lower() == "mlpf":
        if not mlpf_model or not Path(mlpf_model).exists():
            raise FileNotFoundError(f"MLPF model not found: {mlpf_model}")
        cmd.extend(["--mlpf-model", str(mlpf_model), "--mlpf-patch", str(int(mlpf_patch))])
    subprocess.run(cmd, check=True)
    return pd.read_csv(out_csv)


def _algo_space(*, evflow_lite: bool = False) -> list[AlgoCfg]:
    # reduced grid for quick threshold diagnosis
    return [
        AlgoCfg("baf", "baf", "python", [1, 2], [2000, 8000], [1.0]),
        AlgoCfg("stcf", "stc", "python", [1, 2], [4000, 8000], [1, 2, 3, 4]),
        AlgoCfg("ebf", "ebf", "python", [2, 3], [16000, 32000], [x * 0.5 for x in range(0, 11)]),
        AlgoCfg("knoise", "knoise", "python", [1], [2000, 8000], [0, 1, 2, 3, 4]),
        AlgoCfg(
            "evflow",
            "evflow",
            "numba",
            [2],
            [8000] if evflow_lite else [8000, 16000],
            [64.0] if evflow_lite else [8, 16, 24, 32, 48, 64],
        ),
        AlgoCfg("ynoise", "ynoise", "python", [2, 3], [8000, 16000], [1, 2, 3, 4, 6, 8]),
        AlgoCfg("ts", "ts", "numba", [2], [8000, 16000], [0.1, 0.2, 0.4, 0.6, 0.8]),
        AlgoCfg("mlpf", "mlpf", "python", [3], [8000, 16000], [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        AlgoCfg("pfd", "pfd", "numba", [2, 3], [8000, 16000], [1, 2, 3, 4, 6]),
    ]


def _roc_from_scores(scores: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    y = labels.astype(np.int8, copy=False)
    s = scores.astype(np.float64, copy=False)
    n = int(y.shape[0])
    pos = int(y.sum())
    neg = int(n - pos)
    order = np.argsort(-s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order]
    tp_cum = np.cumsum(y_sorted, dtype=np.int64)
    fp_cum = np.cumsum(1 - y_sorted, dtype=np.int64)
    change = np.empty((n,), dtype=bool)
    change[:-1] = s_sorted[:-1] != s_sorted[1:]
    change[-1] = True
    idx = np.nonzero(change)[0]
    tp = np.concatenate([np.asarray([0], dtype=np.int64), tp_cum[idx]])
    fp = np.concatenate([np.asarray([0], dtype=np.int64), fp_cum[idx]])
    thr = np.concatenate([np.asarray([np.inf], dtype=np.float64), s_sorted[idx]])
    tpr = tp.astype(np.float64) / float(pos) if pos else np.zeros_like(tp, dtype=np.float64)
    fpr = fp.astype(np.float64) / float(neg) if neg else np.zeros_like(tp, dtype=np.float64)
    auc = float((getattr(np, "trapezoid", None) or np.trapz)(y=tpr, x=fpr))
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=np.float64), where=(tp + fp) > 0)
    f1 = np.divide(2.0 * precision * tpr, precision + tpr, out=np.zeros_like(precision), where=(precision + tpr) > 0)
    acc = np.divide(tp + (neg - fp), n, out=np.zeros_like(precision), where=n > 0)
    return pd.DataFrame(
        {
            "value": thr.astype(float),
            "tpr": tpr.astype(float),
            "fpr": fpr.astype(float),
            "precision": precision.astype(float),
            "f1": f1.astype(float),
            "accuracy": acc.astype(float),
            "auc": auc,
        }
    )


def _select_rows(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    # LED paper definition:
    # SR = TP/GP = TPR
    # NR = TN/GN = 1-FPR
    # DA = 0.5 * (SR + NR)
    d = df.copy()
    d["sr"] = d["tpr"].astype(float)
    d["nr"] = 1.0 - d["fpr"].astype(float)
    d["da"] = 0.5 * (d["sr"] + d["nr"])

    best_auc_tag = df.groupby("tag", as_index=False)["auc"].first().sort_values("auc", ascending=False).iloc[0]["tag"]
    df_auc = d[d["tag"] == best_auc_tag].copy()
    row_auc = df_auc.sort_values(["da", "f1", "sr", "nr"], ascending=[False, False, False, False]).iloc[0]
    row_da = d.sort_values(["da", "f1", "sr", "nr"], ascending=[False, False, False, False]).iloc[0]
    return row_auc, row_da


def main() -> int:
    ap = argparse.ArgumentParser(description="Run reduced sweep for one LED scene and summarize AUC/DA metrics.")
    ap.add_argument("--scene", default="scene_100")
    ap.add_argument("--npy-root", default=r"D:\hjx_workspace\scientific_reserach\dataset\LED\converted_npy")
    ap.add_argument("--out-root", default="data/LED/scene_sweep")
    ap.add_argument("--python", default=r"D:/software/Anaconda_envs/envs/myEVS/python.exe")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--tick-ns", type=float, default=1000.0)
    ap.add_argument("--max-events", type=int, default=300000)
    ap.add_argument("--mlpf-model-pattern", default="data/LED/models/mlpf_torch_{scene}.pt")
    ap.add_argument("--mlpf-patch", type=int, default=3)
    ap.add_argument("--evflow-lite", action="store_true", help="Use minimal EVFLOW sweep to reduce runtime.")
    args = ap.parse_args()

    npy_path = Path(args.npy_root) / args.scene / "slices_00031_00040_100ms" / f"{args.scene}_100ms_labeled.npy"
    if not npy_path.exists():
        raise FileNotFoundError(f"Scene file not found: {npy_path}")
    out_root = Path(args.out_root) / args.scene
    out_root.mkdir(parents=True, exist_ok=True)
    cache_root = out_root / "_cache"

    noisy, clean = _prepare_truncated_pair(npy_path, max_events=int(args.max_events), cache_root=cache_root)
    ev_arr = np.load(noisy, mmap_mode="r")
    ev = _Events(ev_arr)
    tb = TimeBase(tick_ns=float(args.tick_ns))
    summary_rows = []
    algs = _algo_space(evflow_lite=bool(args.evflow_lite))
    print(f"[scene-sweep] scene={args.scene} max_events={int(args.max_events)} algorithms={len(algs)+1}", flush=True)
    for ai, alg in enumerate(algs, start=1):
        all_rows = []
        total_jobs = len(alg.radii) * len(alg.taus_us)
        done = 0
        print(f"[scene-sweep] algorithm {ai}/{len(algs)+1}: {alg.name} jobs={total_jobs}", flush=True)
        for r in alg.radii:
            for tau in alg.taus_us:
                done += 1
                print(f"[scene-sweep][{alg.name}] {done}/{total_jobs} r={r} tau={tau}", flush=True)
                out_csv = out_root / alg.name / f"roc_{alg.name}_r{r}_tau{tau}.csv"
                out_csv.parent.mkdir(parents=True, exist_ok=True)
                tag = f"{alg.name}_r{r}_tau{tau}_{args.scene}"
                if out_csv.exists():
                    df = pd.read_csv(out_csv)
                else:
                    df = _run_cli_roc(
                        args.python,
                        clean=clean,
                        noisy=noisy,
                        width=int(args.width),
                        height=int(args.height),
                        tick_ns=float(args.tick_ns),
                        method=alg.method,
                        engine=alg.engine,
                        radius=int(r),
                        tau_us=int(tau),
                        thresholds=[float(v) for v in alg.thresholds],
                        tag=tag,
                        out_csv=out_csv,
                        mlpf_model=str(args.mlpf_model_pattern).format(scene=args.scene, level="slices_00031_00040_100ms"),
                        mlpf_patch=int(args.mlpf_patch),
                    )
                all_rows.append(df)
        full = pd.concat(all_rows, ignore_index=True)
        row_auc, row_da = _select_rows(full)
        summary_rows.append(
            {
                "Method": alg.name,
                "AUC_best": float(row_auc["auc"]),
                "DA@Best-AUC": float(row_auc["da"]),
                "DA_best": float(row_da["da"]),
                "AUC@Best-DA": float(row_da["auc"]),
                "SR@Best-DA": float(row_da["sr"]),
                "NR@Best-DA": float(row_da["nr"]),
                "F1": float(row_da["f1"]),
                "BestTagByAUC": str(row_auc["tag"]),
                "BestTagByDA": str(row_da["tag"]),
                "Threshold@Best-DA": float(row_da["value"]),
            }
        )

    # N149 score-based sweep
    print(f"[scene-sweep] algorithm {len(algs)+1}/{len(algs)+1}: n149 jobs=4", flush=True)
    n149_rows = []
    n149_done = 0
    for r in [2, 3]:
        for tau in [16000, 32000]:
            n149_done += 1
            print(f"[scene-sweep][n149] {n149_done}/4 r={r} tau={tau}", flush=True)
            tag = f"n149_r{r}_tau{tau}_{args.scene}"
            scores = score_stream_n149(ev, width=int(args.width), height=int(args.height), radius_px=int(r), tau_us=int(tau), tb=tb)
            df = _roc_from_scores(scores, ev.label)
            df["tag"] = tag
            n149_rows.append(df)
    n149_full = pd.concat(n149_rows, ignore_index=True)
    n149_auc, n149_da = _select_rows(n149_full)
    summary_rows.append(
        {
            "Method": "n149",
            "AUC_best": float(n149_auc["auc"]),
            "DA@Best-AUC": float(n149_auc["da"]),
            "DA_best": float(n149_da["da"]),
            "AUC@Best-DA": float(n149_da["auc"]),
            "SR@Best-DA": float(n149_da["sr"]),
            "NR@Best-DA": float(n149_da["nr"]),
            "F1": float(n149_da["f1"]),
            "BestTagByAUC": str(n149_auc["tag"]),
            "BestTagByDA": str(n149_da["tag"]),
            "Threshold@Best-DA": float(n149_da["value"]),
        }
    )

    out_csv = out_root / "scene_sweep_summary.csv"
    pd.DataFrame(summary_rows).sort_values("AUC_best", ascending=False).to_csv(out_csv, index=False)
    print(f"saved: {out_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
