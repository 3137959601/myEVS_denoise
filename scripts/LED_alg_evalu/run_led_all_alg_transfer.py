from __future__ import annotations

import argparse
import json
import random
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


def _read_meta_for_sample(sample_path: Path) -> dict:
    meta_path = sample_path.with_name("led_100ms_meta.json")
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


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


def main() -> int:
    ap = argparse.ArgumentParser(description="LED all-algorithms transfer evaluation.")
    ap.add_argument("--npy-root", default=r"D:\hjx_workspace\scientific_reserach\dataset\LED\converted_npy")
    ap.add_argument("--out-root", default="data/LED/all_alg")
    ap.add_argument("--python", default=r"D:/software/Anaconda_envs/envs/myEVS/python.exe")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--tick-ns", type=float, default=1000.0)
    ap.add_argument("--num-scenes", type=int, default=10)
    ap.add_argument("--num-tune-scenes", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max-events", type=int, default=300000)
    ap.add_argument("--mlpf-model-pattern", default="data/LED/models/mlpf_torch_{scene}.pt")
    args = ap.parse_args()

    npy_root = Path(args.npy_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    cache_root = out_root / "_cache_truncated"

    def _mlpf_model_for(scene_name: str, level_name: str) -> str:
        model = str(args.mlpf_model_pattern).format(scene=scene_name, level=level_name)
        if not Path(model).exists():
            raise FileNotFoundError(f"MLPF model not found: {model}")
        return model

    samples = sorted(npy_root.glob("*/slices_*/*_100ms_labeled.npy"))
    if len(samples) < max(2, args.num_scenes):
        raise SystemExit(f"Need at least {args.num_scenes} LED samples, found: {len(samples)}")

    selected = samples[: int(args.num_scenes)]
    rng = random.Random(int(args.seed))
    tune_samples = rng.sample(selected, k=int(args.num_tune_scenes))
    tune_set = {str(p) for p in tune_samples}
    eval_samples = [p for p in selected if str(p) not in tune_set]

    scene_stats = []
    for p in selected:
        m = _read_meta_for_sample(p)
        key = p.stem.replace("_100ms_labeled", "")
        scene_stats.append(
            {
                "scene_key": key,
                "scene": p.parents[1].name,
                "level": p.parent.name,
                "npy": str(p),
                "noise_per_signal": m.get("noise_per_signal", np.nan),
                "estimated_noise_hz_per_pixel": m.get("estimated_noise_hz_per_pixel", np.nan),
                "duration_s": m.get("duration_s", np.nan),
                "signal_events": m.get("signal_events", np.nan),
                "noise_events": m.get("noise_events", np.nan),
                "selected_as_tune": str(p) in tune_set,
            }
        )
    pd.DataFrame(scene_stats).to_csv(out_root / "scene_stats.csv", index=False)

    tune_results = []
    best_cfg_rows = []
    tb = TimeBase(tick_ns=float(args.tick_ns))

    print("[LED] tune scenes:")
    for p in tune_samples:
        print(f"  - {p}")
    print(f"[LED] stage=tuning  algorithms={len(_algo_space()) + 1}  tune_scenes={len(tune_samples)}")

    alg_list = _algo_space()
    for ai, alg in enumerate(alg_list, start=1):
        total_jobs = len(tune_samples) * len(alg.radii) * len(alg.taus_us)
        print(f"[LED][tuning] algorithm {ai}/{len(alg_list)+1}: {alg.name}  jobs={total_jobs}")
        all_cands = []
        done = 0
        for tune_path in tune_samples:
            tune_key = tune_path.stem.replace("_100ms_labeled", "")
            tune_noisy, tune_clean = _prepare_truncated_pair(tune_path, max_events=int(args.max_events), cache_root=cache_root)
            for r in alg.radii:
                for tau in alg.taus_us:
                    done += 1
                    print(f"[LED][tuning][{alg.name}] {done}/{total_jobs} scene={tune_key} r={r} tau={tau}")
                    tag = f"{alg.name}_r{r}_tau{tau}_{tune_key}"
                    out_csv = out_root / "tuning" / alg.name / f"roc_{alg.name}_{tune_key}_r{r}_tau{tau}.csv"
                    out_csv.parent.mkdir(parents=True, exist_ok=True)
                    if out_csv.exists():
                        df = pd.read_csv(out_csv)
                    else:
                        df = _run_cli_roc(
                            args.python,
                            clean=tune_clean,
                            noisy=tune_noisy,
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
                                _mlpf_model_for(tune_path.parents[1].name, tune_path.parent.name)
                                if alg.name == "mlpf"
                                else ""
                            ),
                        )
                    auc = float(df["auc"].iloc[0])
                    best_f1_idx = int(df["f1"].astype(float).idxmax())
                    row = df.iloc[best_f1_idx]
                    cand = {
                        "algorithm": alg.name,
                        "scene_key": tune_key,
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
                    all_cands.append(cand)

        g = (
            pd.DataFrame(all_cands)
            .groupby(["radius", "tau_us"], as_index=False)
            .agg(mean_auc=("auc", "mean"), mean_f1=("best_f1", "mean"), mean_thr=("best_threshold", "mean"))
        )
        g = g.sort_values(["mean_auc", "mean_f1"], ascending=[False, False]).reset_index(drop=True)
        top = g.iloc[0]
        best_cfg = {
            "algorithm": alg.name,
            "radius": int(top["radius"]),
            "tau_us": int(top["tau_us"]),
            "tuned_threshold": float(top["mean_thr"]),
            "tune_mean_auc": float(top["mean_auc"]),
            "tune_mean_f1": float(top["mean_f1"]),
            "num_tune_scenes": int(args.num_tune_scenes),
        }
        best_cfg_rows.append(best_cfg)
        pd.DataFrame(tune_results).to_csv(out_root / "tuning_all_alg_results_partial.csv", index=False)
        pd.DataFrame(best_cfg_rows).to_csv(out_root / "tuning_best_config_partial.csv", index=False)
        print(
            f"  [best] {alg.name}: r={best_cfg['radius']} tau={best_cfg['tau_us']} "
            f"thr={best_cfg['tuned_threshold']:.6f} mean_auc={best_cfg['tune_mean_auc']:.6f}"
        )

    # N149 tuning (score-based)
    print(f"[LED][tuning] algorithm {len(alg_list)+1}/{len(alg_list)+1}: n149")
    n149_r = [2, 3, 4, 5]
    n149_tau = [16000, 32000, 64000, 128000, 256000, 512000]
    n149_cands = []
    n149_total = len(tune_samples) * len(n149_r) * len(n149_tau)
    n149_done = 0
    for tune_path in tune_samples:
        tune_key = tune_path.stem.replace("_100ms_labeled", "")
        tune_noisy, _ = _prepare_truncated_pair(tune_path, max_events=int(args.max_events), cache_root=cache_root)
        ev_tune = _load_events(tune_noisy)
        for r in n149_r:
            for tau in n149_tau:
                n149_done += 1
                print(f"[LED][tuning][n149] {n149_done}/{n149_total} scene={tune_key} r={r} tau={tau}")
                scores = _score_n149(ev_tune, width=args.width, height=args.height, radius=r, tau_us=tau, tb=tb)
                roc = _roc_from_scores(scores, ev_tune.label)
                bi = int(roc["best_idx"])
                n149_cands.append(
                    {
                        "algorithm": "n149",
                        "scene_key": tune_key,
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
                )
    tune_results.extend(n149_cands)
    gn = (
        pd.DataFrame(n149_cands)
        .groupby(["radius", "tau_us"], as_index=False)
        .agg(mean_auc=("auc", "mean"), mean_f1=("best_f1", "mean"), mean_thr=("best_threshold", "mean"))
    )
    gn = gn.sort_values(["mean_auc", "mean_f1"], ascending=[False, False]).reset_index(drop=True)
    topn = gn.iloc[0]
    best_cfg_rows.append(
        {
            "algorithm": "n149",
            "radius": int(topn["radius"]),
            "tau_us": int(topn["tau_us"]),
            "tuned_threshold": float(topn["mean_thr"]),
            "tune_mean_auc": float(topn["mean_auc"]),
            "tune_mean_f1": float(topn["mean_f1"]),
            "num_tune_scenes": int(args.num_tune_scenes),
        }
    )

    pd.DataFrame(tune_results).to_csv(out_root / "tuning_all_alg_results.csv", index=False)
    best_cfg_df = pd.DataFrame(best_cfg_rows)
    best_cfg_df.to_csv(out_root / "tuning_best_config.csv", index=False)

    best_cfg_map = {r["algorithm"]: r for r in best_cfg_rows}
    print(f"[LED] stage=transfer  eval_scenes={len(eval_samples)}")
    transfer_rows = []
    for noisy in eval_samples:
        noisy_eval, clean = _prepare_truncated_pair(noisy, max_events=int(args.max_events), cache_root=cache_root)
        key = noisy.stem.replace("_100ms_labeled", "")
        meta = _read_meta_for_sample(noisy)
        print(f"[LED][transfer] scene={key}")
        for ai, alg in enumerate(alg_list, start=1):
            cfg = best_cfg_map[alg.name]
            print(
                f"[LED][transfer] algorithm {ai}/{len(alg_list)+1}: {alg.name} "
                f"r={int(cfg['radius'])} tau={int(cfg['tau_us'])} thr={float(cfg['tuned_threshold']):.6f}"
            )
            out_csv = out_root / "transfer" / alg.name / f"roc_{alg.name}_{key}.csv"
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            if out_csv.exists():
                df = pd.read_csv(out_csv)
            else:
                df = _run_cli_roc(
                    args.python,
                    clean=clean,
                    noisy=noisy_eval,
                    width=args.width,
                    height=args.height,
                    tick_ns=args.tick_ns,
                    method=alg.method,
                    engine=alg.engine,
                    radius=int(cfg["radius"]),
                    tau_us=int(cfg["tau_us"]),
                    thresholds=[float(v) for v in alg.thresholds],
                    tag=f"{alg.name}_r{int(cfg['radius'])}_tau{int(cfg['tau_us'])}_{key}",
                    out_csv=out_csv,
                    mlpf_model=(
                        _mlpf_model_for(noisy.parents[1].name, noisy.parent.name)
                        if alg.name == "mlpf"
                        else ""
                    ),
                )
            auc = float(df["auc"].iloc[0])
            thr = float(cfg["tuned_threshold"])
            idx = int((df["value"].astype(float) - thr).abs().idxmin())
            row = df.iloc[idx]
            transfer_rows.append(
                {
                    "algorithm": alg.name,
                    "scene_key": key,
                    "radius": int(cfg["radius"]),
                    "tau_us": int(cfg["tau_us"]),
                    "tuned_threshold": thr,
                    "auc": auc,
                    "f1_at_tuned_threshold": float(row["f1"]),
                    "tpr_at_tuned_threshold": float(row["tpr"]),
                    "fpr_at_tuned_threshold": float(row["fpr"]),
                    "precision_at_tuned_threshold": float(row["precision"]),
                    "noise_per_signal": meta.get("noise_per_signal", np.nan),
                    "estimated_noise_hz_per_pixel": meta.get("estimated_noise_hz_per_pixel", np.nan),
                    "csv": str(out_csv),
                }
            )

        cfg = best_cfg_map["n149"]
        print(
            f"[LED][transfer] algorithm {len(alg_list)+1}/{len(alg_list)+1}: n149 "
            f"r={int(cfg['radius'])} tau={int(cfg['tau_us'])} thr={float(cfg['tuned_threshold']):.6f}"
        )
        ev = _load_events(noisy_eval)
        scores = _score_n149(ev, width=args.width, height=args.height, radius=int(cfg["radius"]), tau_us=int(cfg["tau_us"]), tb=tb)
        roc = _roc_from_scores(scores, ev.label)
        pt = _point_at_threshold(scores, ev.label, float(cfg["tuned_threshold"]))
        transfer_rows.append(
            {
                "algorithm": "n149",
                "scene_key": key,
                "radius": int(cfg["radius"]),
                "tau_us": int(cfg["tau_us"]),
                "tuned_threshold": float(cfg["tuned_threshold"]),
                "auc": float(roc["auc"]),
                "f1_at_tuned_threshold": float(pt["f1"]),
                "tpr_at_tuned_threshold": float(pt["tpr"]),
                "fpr_at_tuned_threshold": float(pt["fpr"]),
                "precision_at_tuned_threshold": float(pt["precision"]),
                "noise_per_signal": meta.get("noise_per_signal", np.nan),
                "estimated_noise_hz_per_pixel": meta.get("estimated_noise_hz_per_pixel", np.nan),
                "csv": "",
            }
        )
        pd.DataFrame(transfer_rows).to_csv(out_root / "transfer_eval_all_alg_partial.csv", index=False)

    transfer_df = pd.DataFrame(transfer_rows)
    transfer_df.to_csv(out_root / "transfer_eval_all_alg.csv", index=False)
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
        .reset_index(drop=True)
    )
    summary.to_csv(out_root / "transfer_eval_algorithm_summary.csv", index=False)

    meta = {
        "num_total_selected_scenes": int(args.num_scenes),
        "num_tune_scenes": int(args.num_tune_scenes),
        "seed": int(args.seed),
        "tune_samples": [str(p) for p in tune_samples],
        "eval_samples": [str(p) for p in eval_samples],
    }
    (out_root / "transfer_eval_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] saved to: {out_root}")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
