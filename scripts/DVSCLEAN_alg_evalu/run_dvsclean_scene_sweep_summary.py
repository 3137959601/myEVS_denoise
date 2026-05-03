from __future__ import annotations

import argparse
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from myevs.denoise.ops.ebfopt_part2.n149_n145_s52_euclid_compactlut_backbone import score_stream_n149
from myevs.timebase import TimeBase


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
    out: list[str] = []
    for v in vals:
        if isinstance(v, float) and not float(v).is_integer():
            out.append(f"{float(v):.6f}".rstrip("0").rstrip("."))
        else:
            out.append(str(int(v)))
    return ",".join(out)


def _algo_space(*, evflow_lite: bool) -> list[AlgoCfg]:
    return [
        AlgoCfg("baf", "baf", "python", [1, 2, 3], [2000, 8000, 16000, 32000], [1.0]),
        AlgoCfg("stcf", "stc", "python", [1, 2, 3], [1000, 4000, 8000, 16000, 32000], [1, 2, 3, 4, 5, 6]),
        AlgoCfg("ebf", "ebf", "python", [2, 3, 4], [16000, 32000, 64000], [x * 0.5 for x in range(0, 17)]),
        AlgoCfg("knoise", "knoise", "python", [1], [1000, 2000, 4000, 8000, 16000, 32000], [0, 1, 2, 3, 4, 5, 6]),
        AlgoCfg(
            "evflow",
            "evflow",
            "numba",
            [2] if evflow_lite else [2, 3],
            [8000] if evflow_lite else [8000, 16000, 32000],
            [64.0] if evflow_lite else [0, 8, 16, 24, 32, 48, 64],
        ),
        AlgoCfg("ynoise", "ynoise", "python", [2, 3, 4], [8000, 16000, 32000, 64000], [1, 2, 3, 4, 6, 8, 10, 12]),
        AlgoCfg("ts", "ts", "numba", [2, 3, 4], [8000, 16000, 32000, 64000], [x / 10.0 for x in range(1, 10)]),
        AlgoCfg("mlpf", "mlpf", "python", [3], [8000, 16000, 32000, 64000], [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        AlgoCfg("pfd", "pfd", "numba", [2, 3, 4], [8000, 16000, 32000, 64000], [1, 2, 3, 4, 5, 6, 8]),
    ]


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
    if method == "mlpf":
        if not mlpf_model or not Path(mlpf_model).exists():
            raise FileNotFoundError(f"MLPF model not found: {mlpf_model}")
        cmd.extend(["--mlpf-model", str(mlpf_model), "--mlpf-patch", str(int(mlpf_patch))])
    subprocess.run(cmd, check=True)
    return pd.read_csv(out_csv)


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


def _append_extra_metrics(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["sr"] = d["tpr"].astype(float)
    d["nr"] = 1.0 - d["fpr"].astype(float)
    d["da"] = 0.5 * (d["sr"] + d["nr"])
    eps = 1e-12
    snr_linear = d["sr"] / (1.0 - d["nr"] + eps)  # equals TPR/FPR
    d["snr_linear"] = snr_linear
    d["snr_db"] = 10.0 * np.log10(np.maximum(snr_linear, eps))
    return d


def _pick_rows(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    d = _append_extra_metrics(df)
    best_auc_tag = d.groupby("tag", as_index=False)["auc"].first().sort_values("auc", ascending=False).iloc[0]["tag"]
    d_auc = d[d["tag"] == best_auc_tag].copy()
    row_auc = d_auc.sort_values(["da", "f1", "sr", "nr"], ascending=[False, False, False, False]).iloc[0]
    row_da = d.sort_values(["da", "f1", "sr", "nr"], ascending=[False, False, False, False]).iloc[0]
    row_snr = d.sort_values(["snr_db", "da", "f1"], ascending=[False, False, False]).iloc[0]
    return row_auc, row_da, row_snr


def main() -> int:
    ap = argparse.ArgumentParser(description="DVSCLEAN: per-sample full sweep summary with DA and SNR.")
    ap.add_argument("--scene", required=True, help="e.g. MAH00444")
    ap.add_argument("--level", required=True, help="ratio50 or ratio100")
    ap.add_argument("--npy-root", default=r"D:\hjx_workspace\scientific_reserach\dataset\DVSCLEAN\converted_npy")
    ap.add_argument("--out-root", default="data/DVSCLEAN/scene_sweep_full")
    ap.add_argument("--python", default=r"D:/software/Anaconda_envs/envs/myEVS/python.exe")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--tick-ns", type=float, default=1000.0)
    ap.add_argument("--mlpf-model", default="")
    ap.add_argument("--mlpf-patch", type=int, default=7)
    ap.add_argument("--evflow-lite", action="store_true")
    args = ap.parse_args()

    scene = str(args.scene)
    level = str(args.level)
    noisy = Path(args.npy_root) / scene / level / f"{scene}_{level}_labeled.npy"
    clean = noisy.with_name(noisy.name.replace("_labeled.npy", "_signal_only.npy"))
    if not noisy.exists() or not clean.exists():
        raise FileNotFoundError(f"Missing pair: {noisy} / {clean}")
    if not args.mlpf_model:
        raise ValueError("--mlpf-model is required")

    out_dir = Path(args.out_root) / f"{scene}_{level}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ev_arr = np.load(noisy, mmap_mode="r")
    ev = _Events(ev_arr)
    tb = TimeBase(tick_ns=float(args.tick_ns))

    summary_rows: list[dict] = []
    algs = _algo_space(evflow_lite=bool(args.evflow_lite))
    print(f"[DVSCLEAN-sweep] {scene}_{level}: algorithms={len(algs)+1}", flush=True)

    for ai, alg in enumerate(algs, start=1):
        print(f"[DVSCLEAN-sweep] {ai}/{len(algs)+1} {alg.name}", flush=True)
        all_rows: list[pd.DataFrame] = []
        total = len(alg.radii) * len(alg.taus_us)
        done = 0
        for r in alg.radii:
            for tau in alg.taus_us:
                done += 1
                print(f"[DVSCLEAN-sweep][{alg.name}] {done}/{total} r={r} tau={tau}", flush=True)
                csv_path = out_dir / alg.name / f"roc_{alg.name}_r{r}_tau{tau}.csv"
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                tag = f"{alg.name}_r{r}_tau{tau}_{scene}_{level}"
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
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
                        out_csv=csv_path,
                        mlpf_model=str(args.mlpf_model) if alg.name == "mlpf" else "",
                        mlpf_patch=int(args.mlpf_patch),
                    )
                all_rows.append(df)
        full = pd.concat(all_rows, ignore_index=True)
        row_auc, row_da, row_snr = _pick_rows(full)
        summary_rows.append(
            {
                "Method": alg.name,
                "AUC_best": float(row_auc["auc"]),
                "DA@Best-AUC": float(row_auc["da"]),
                "SNRdB@Best-AUC": float(row_auc["snr_db"]),
                "DA_best": float(row_da["da"]),
                "AUC@Best-DA": float(row_da["auc"]),
                "SNRdB@Best-DA": float(row_da["snr_db"]),
                "SR@Best-DA": float(row_da["sr"]),
                "NR@Best-DA": float(row_da["nr"]),
                "F1": float(row_da["f1"]),
                "SNRdB_best": float(row_snr["snr_db"]),
                "DA@Best-SNRdB": float(row_snr["da"]),
                "BestTagByAUC": str(row_auc["tag"]),
                "BestTagByDA": str(row_da["tag"]),
                "BestTagBySNRdB": str(row_snr["tag"]),
                "Threshold@Best-DA": float(row_da["value"]),
            }
        )

    print(f"[DVSCLEAN-sweep] {len(algs)+1}/{len(algs)+1} n149", flush=True)
    n149_rows = []
    for r in [2, 3, 4, 5]:
        for tau in [16000, 32000, 64000, 128000, 256000, 512000]:
            print(f"[DVSCLEAN-sweep][n149] r={r} tau={tau}", flush=True)
            tag = f"n149_r{r}_tau{tau}_{scene}_{level}"
            scores = score_stream_n149(ev, width=int(args.width), height=int(args.height), radius_px=int(r), tau_us=int(tau), tb=tb)
            d = _roc_from_scores(scores, ev.label)
            d["tag"] = tag
            n149_rows.append(d)
    n149_full = pd.concat(n149_rows, ignore_index=True)
    row_auc, row_da, row_snr = _pick_rows(n149_full)
    summary_rows.append(
        {
            "Method": "n149",
            "AUC_best": float(row_auc["auc"]),
            "DA@Best-AUC": float(row_auc["da"]),
            "SNRdB@Best-AUC": float(row_auc["snr_db"]),
            "DA_best": float(row_da["da"]),
            "AUC@Best-DA": float(row_da["auc"]),
            "SNRdB@Best-DA": float(row_da["snr_db"]),
            "SR@Best-DA": float(row_da["sr"]),
            "NR@Best-DA": float(row_da["nr"]),
            "F1": float(row_da["f1"]),
            "SNRdB_best": float(row_snr["snr_db"]),
            "DA@Best-SNRdB": float(row_snr["da"]),
            "BestTagByAUC": str(row_auc["tag"]),
            "BestTagByDA": str(row_da["tag"]),
            "BestTagBySNRdB": str(row_snr["tag"]),
            "Threshold@Best-DA": float(row_da["value"]),
        }
    )

    out_csv = out_dir / "scene_sweep_summary.csv"
    pd.DataFrame(summary_rows).sort_values("AUC_best", ascending=False).to_csv(out_csv, index=False)
    print(f"saved: {out_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

