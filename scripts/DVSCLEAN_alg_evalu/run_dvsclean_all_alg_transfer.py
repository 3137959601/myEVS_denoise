from __future__ import annotations

import argparse
import json
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


def _csv_values(vals: list[float | int]) -> str:
    out = []
    for v in vals:
        if isinstance(v, float) and not float(v).is_integer():
            out.append(f"{float(v):.6f}".rstrip("0").rstrip("."))
        else:
            out.append(str(int(v)))
    return ",".join(out)


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
) -> pd.DataFrame:
    values = _csv_values(thresholds)
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
        values,
        "--match-us",
        "0",
        "--match-bin-radius",
        "0",
        "--tag",
        tag,
        "--out-csv",
        str(out_csv),
        "--progress",
    ]
    if str(method).lower() == "mlpf" and str(mlpf_model).strip():
        cmd.extend(["--mlpf-model", str(mlpf_model)])
    subprocess.run(cmd, check=True)
    return pd.read_csv(out_csv)


def _load_events(path: Path):
    arr = np.load(path, mmap_mode="r")

    class _Ev:
        t = np.ascontiguousarray(arr["t"].astype(np.uint64, copy=False))
        x = np.ascontiguousarray(arr["x"].astype(np.int32, copy=False))
        y = np.ascontiguousarray(arr["y"].astype(np.int32, copy=False))
        p = np.ascontiguousarray(arr["p"].astype(np.int8, copy=False))
        label = np.ascontiguousarray((arr["label"] > 0).astype(np.uint8, copy=False))

    return _Ev


def _score_n149(ev, *, width: int, height: int, radius: int, tau_us: int, tb: TimeBase) -> np.ndarray:
    return score_stream_n149(
        ev,
        width=width,
        height=height,
        radius_px=radius,
        tau_us=tau_us,
        tb=tb,
    ).astype(np.float64, copy=False)


def _roc_from_scores(scores: np.ndarray, labels: np.ndarray):
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
    tpr = tp.astype(np.float64) / float(pos)
    fpr = fp.astype(np.float64) / float(neg)
    auc = float((getattr(np, "trapezoid", None) or np.trapz)(y=tpr, x=fpr))
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=np.float64), where=(tp + fp) > 0)
    f1 = np.divide(2.0 * prec * tpr, prec + tpr, out=np.zeros_like(prec), where=(prec + tpr) > 0)
    best_idx = int(np.argmax(np.stack([f1, tpr, prec, -fpr], axis=1), axis=0)[0])
    return {
        "auc": auc,
        "thr": thr,
        "tp": tp,
        "fp": fp,
        "tpr": tpr,
        "fpr": fpr,
        "prec": prec,
        "f1": f1,
        "best_idx": best_idx,
    }


def _point_at_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float):
    keep = scores > float(threshold)
    y = labels.astype(np.uint8, copy=False)
    pos = int(y.sum())
    neg = int(y.shape[0] - pos)
    tp = int(np.count_nonzero(keep & (y != 0)))
    fp = int(np.count_nonzero(keep & (y == 0)))
    tpr = tp / pos if pos else 0.0
    fpr = fp / neg if neg else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2.0 * precision * tpr / (precision + tpr) if (precision + tpr) else 0.0
    return {"threshold": float(threshold), "tpr": tpr, "fpr": fpr, "precision": precision, "f1": f1}


def _algo_space() -> list[AlgoCfg]:
    return [
        AlgoCfg("baf", "baf", "python", [1, 2, 3], [2000, 8000, 16000, 32000], [1.0]),
        AlgoCfg("stcf", "stc", "python", [1, 2, 3], [1000, 4000, 8000, 16000, 32000], [1, 2, 3, 4, 5, 6]),
        AlgoCfg("ebf", "ebf", "python", [2, 3, 4], [16000, 32000, 64000], [x * 0.5 for x in range(0, 17)]),
        AlgoCfg("knoise", "knoise", "python", [1], [1000, 2000, 4000, 8000, 16000, 32000], [0, 1, 2, 3, 4, 5, 6]),
        AlgoCfg("evflow", "evflow", "numba", [2, 3], [8000, 16000, 32000], [0, 8, 16, 24, 32, 48, 64]),
        AlgoCfg("ynoise", "ynoise", "python", [2, 3, 4], [8000, 16000, 32000, 64000], [1, 2, 3, 4, 6, 8, 10, 12]),
        AlgoCfg("ts", "ts", "numba", [2, 3, 4], [8000, 16000, 32000, 64000], [x / 10.0 for x in range(1, 10)]),
        AlgoCfg("mlpf", "mlpf", "python", [3], [8000, 16000, 32000, 64000], [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        AlgoCfg("pfd", "pfd", "numba", [2, 3, 4], [8000, 16000, 32000, 64000], [1, 2, 3, 4, 5, 6, 8]),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description="DVSCLEAN all-algorithms transfer evaluation.")
    ap.add_argument("--npy-root", default=r"D:\hjx_workspace\scientific_reserach\dataset\DVSCLEAN\converted_npy")
    ap.add_argument("--out-root", default="data/DVSCLEAN/all_alg")
    ap.add_argument("--python", default=r"D:/software/Anaconda_envs/envs/myEVS/python.exe")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--tick-ns", type=float, default=1000.0)
    ap.add_argument("--tune-scene", default="MAH00446")
    ap.add_argument("--tune-level", default="ratio100")
    ap.add_argument("--mlpf-model-pattern", default="data/DVSCLEAN/models/mlpf_torch_{scene}_{level}.pt")
    args = ap.parse_args()

    npy_root = Path(args.npy_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    samples = sorted(npy_root.glob("*/*/*_labeled.npy"))
    if len(samples) < 2:
        raise SystemExit("Need at least two DVSCLEAN samples for transfer evaluation.")

    tune_key = f"{args.tune_scene}_{args.tune_level}"
    keyed = {}
    for p in samples:
        scene = p.parents[1].name
        level = p.parent.name
        keyed[f"{scene}_{level}"] = p
    if tune_key not in keyed:
        raise SystemExit(f"Tuning sample not found: {tune_key}")
    tune_path = keyed[tune_key]
    eval_keys = [k for k in sorted(keyed.keys()) if k != tune_key]

    # matching clean path for cli roc
    tune_clean = tune_path.with_name(tune_path.name.replace("_labeled.npy", "_signal_only.npy"))
    if not tune_clean.exists():
        raise SystemExit(f"Missing clean npy for tuning sample: {tune_clean}")

    tune_results = []
    best_cfg = {}

    def _mlpf_model_for(scene_name: str, level_name: str) -> str:
        model = str(args.mlpf_model_pattern).format(scene=scene_name, level=level_name)
        if not Path(model).exists():
            raise FileNotFoundError(f"MLPF model not found: {model}")
        return model

    print(f"[DVSCLEAN] tuning sample: {tune_key}")
    for alg in _algo_space():
        alg_dir = out_root / "tuning" / alg.name
        alg_dir.mkdir(parents=True, exist_ok=True)
        best_row = None
        for r in alg.radii:
            for tau in alg.taus_us:
                tag = f"{alg.name}_r{r}_tau{tau}_{tune_key}"
                out_csv = alg_dir / f"roc_{alg.name}_r{r}_tau{tau}.csv"
                df = _run_cli_roc(
                    args.python,
                    clean=tune_clean,
                    noisy=tune_path,
                    width=args.width,
                    height=args.height,
                    tick_ns=args.tick_ns,
                    method=alg.method,
                    engine=alg.engine,
                    radius=r,
                    tau_us=tau,
                    thresholds=[float(v) for v in alg.thresholds],
                    tag=tag,
                    out_csv=out_csv,
                    mlpf_model=(
                        _mlpf_model_for(args.tune_scene, args.tune_level)
                        if alg.name == "mlpf"
                        else ""
                    ),
                )
                auc = float(df["auc"].iloc[0])
                best_f1_idx = int(df["f1"].astype(float).idxmax())
                row = df.iloc[best_f1_idx]
                cand = {
                    "algorithm": alg.name,
                    "scene_level": tune_key,
                    "radius": int(r),
                    "tau_us": int(tau),
                    "auc": auc,
                    "best_f1": float(row["f1"]),
                    "best_threshold": float(row["value"]),
                    "best_tpr": float(row["tpr"]),
                    "best_fpr": float(row["fpr"]),
                    "best_precision": float(row["precision"]),
                    "csv": str(out_csv),
                }
                tune_results.append(cand)
                if best_row is None or (cand["auc"], cand["best_f1"]) > (best_row["auc"], best_row["best_f1"]):
                    best_row = cand
        best_cfg[alg.name] = best_row
        print(f"  [best] {alg.name}: r={best_row['radius']} tau={best_row['tau_us']} auc={best_row['auc']:.6f} thr={best_row['best_threshold']:.6f} f1={best_row['best_f1']:.6f}")

    # N149 tuning (score-based)
    n149_r = [2, 3, 4, 5]
    n149_tau = [16000, 32000, 64000, 128000, 256000, 512000]
    tb = TimeBase(tick_ns=float(args.tick_ns))
    ev_tune = _load_events(tune_path)
    best_n149 = None
    for r in n149_r:
        for tau in n149_tau:
            scores = _score_n149(ev_tune, width=args.width, height=args.height, radius=r, tau_us=tau, tb=tb)
            roc = _roc_from_scores(scores, ev_tune.label)
            bi = int(roc["best_idx"])
            cand = {
                "algorithm": "n149",
                "scene_level": tune_key,
                "radius": int(r),
                "tau_us": int(tau),
                "auc": float(roc["auc"]),
                "best_f1": float(roc["f1"][bi]),
                "best_threshold": float(roc["thr"][bi]),
                "best_tpr": float(roc["tpr"][bi]),
                "best_fpr": float(roc["fpr"][bi]),
                "best_precision": float(roc["prec"][bi]),
                "csv": "",
            }
            tune_results.append(cand)
            if best_n149 is None or (cand["auc"], cand["best_f1"]) > (best_n149["auc"], best_n149["best_f1"]):
                best_n149 = cand
    best_cfg["n149"] = best_n149
    print(f"  [best] n149: r={best_n149['radius']} tau={best_n149['tau_us']} auc={best_n149['auc']:.6f} thr={best_n149['best_threshold']:.6f} f1={best_n149['best_f1']:.6f}")

    pd.DataFrame(tune_results).to_csv(out_root / "tuning_all_alg_results.csv", index=False)
    pd.DataFrame(list(best_cfg.values())).to_csv(out_root / "tuning_best_config.csv", index=False)

    transfer_rows = []
    for k in eval_keys:
        noisy = keyed[k]
        clean = noisy.with_name(noisy.name.replace("_labeled.npy", "_signal_only.npy"))
        scene = noisy.parents[1].name
        level = noisy.parent.name
        print(f"[DVSCLEAN] transfer eval: {k}")

        for alg in _algo_space():
            cfg = best_cfg[alg.name]
            eval_dir = out_root / "transfer" / alg.name
            eval_dir.mkdir(parents=True, exist_ok=True)
            out_csv = eval_dir / f"roc_{alg.name}_{k}.csv"
            df = _run_cli_roc(
                args.python,
                clean=clean,
                noisy=noisy,
                width=args.width,
                height=args.height,
                tick_ns=args.tick_ns,
                method=alg.method,
                engine=alg.engine,
                radius=int(cfg["radius"]),
                tau_us=int(cfg["tau_us"]),
                thresholds=[float(v) for v in alg.thresholds],
                tag=f"{alg.name}_r{int(cfg['radius'])}_tau{int(cfg['tau_us'])}_{k}",
                out_csv=out_csv,
                mlpf_model=(
                    _mlpf_model_for(scene, level)
                    if alg.name == "mlpf"
                    else ""
                ),
            )
            auc = float(df["auc"].iloc[0])
            # fixed-threshold transfer point
            thr = float(cfg["best_threshold"])
            idx = int((df["value"].astype(float) - thr).abs().idxmin())
            row = df.iloc[idx]
            transfer_rows.append(
                {
                    "algorithm": alg.name,
                    "scene": scene,
                    "level": level,
                    "scene_level": k,
                    "radius": int(cfg["radius"]),
                    "tau_us": int(cfg["tau_us"]),
                    "tuned_threshold": thr,
                    "auc": auc,
                    "f1_at_tuned_threshold": float(row["f1"]),
                    "tpr_at_tuned_threshold": float(row["tpr"]),
                    "fpr_at_tuned_threshold": float(row["fpr"]),
                    "precision_at_tuned_threshold": float(row["precision"]),
                    "csv": str(out_csv),
                }
            )

        # n149 transfer
        ev = _load_events(noisy)
        cfg = best_cfg["n149"]
        scores = _score_n149(ev, width=args.width, height=args.height, radius=int(cfg["radius"]), tau_us=int(cfg["tau_us"]), tb=tb)
        roc = _roc_from_scores(scores, ev.label)
        pt = _point_at_threshold(scores, ev.label, float(cfg["best_threshold"]))
        transfer_rows.append(
            {
                "algorithm": "n149",
                "scene": scene,
                "level": level,
                "scene_level": k,
                "radius": int(cfg["radius"]),
                "tau_us": int(cfg["tau_us"]),
                "tuned_threshold": float(cfg["best_threshold"]),
                "auc": float(roc["auc"]),
                "f1_at_tuned_threshold": float(pt["f1"]),
                "tpr_at_tuned_threshold": float(pt["tpr"]),
                "fpr_at_tuned_threshold": float(pt["fpr"]),
                "precision_at_tuned_threshold": float(pt["precision"]),
                "csv": "",
            }
        )

    transfer_df = pd.DataFrame(transfer_rows)
    transfer_df.to_csv(out_root / "transfer_eval_all_alg.csv", index=False)

    # summary
    summary = (
        transfer_df.groupby("algorithm", as_index=False)
        .agg(
            mean_auc=("auc", "mean"),
            mean_f1=("f1_at_tuned_threshold", "mean"),
            mean_tpr=("tpr_at_tuned_threshold", "mean"),
            mean_fpr=("fpr_at_tuned_threshold", "mean"),
            mean_precision=("precision_at_tuned_threshold", "mean"),
        )
        .sort_values("mean_auc", ascending=False)
    )
    summary.to_csv(out_root / "transfer_eval_algorithm_summary.csv", index=False)

    meta = {
        "tuning_sample": tune_key,
        "num_transfer_samples": len(eval_keys),
        "transfer_samples": eval_keys,
        "notes": "AUC uses fixed (algorithm,r,tau) tuned on one sample; F1 is evaluated at tuned threshold on transfer samples.",
    }
    (out_root / "transfer_eval_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out_root / 'transfer_eval_all_alg.csv'}")
    print(f"saved: {out_root / 'transfer_eval_algorithm_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
