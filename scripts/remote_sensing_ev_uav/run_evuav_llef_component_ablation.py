from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from evuav_common import (
    HEIGHT,
    RESULT_ROOT,
    SELECTED_TEST_SEQUENCES,
    WIDTH,
    ensure_dirs,
    metrics_from_target_keep,
    parse_sequence,
    rank_auc,
    sequence_reference_path,
    write_csv,
    write_json,
)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from myevs.denoise.ops.ebfopt_part2.n149_n145_s52_euclid_compactlut_backbone import score_stream_n149
from myevs.timebase import TimeBase


R_DEF = 4
TAU_DEF = 256000
SIGMA_DEF = 3.0
ALPHA_DEF = 4.0
THRESHOLDS = tuple(float(x) for x in np.arange(0.0, 16.0001, 0.25))

VARIANTS: dict[str, dict[str, str]] = {
    "baseline": {},
    "no_spatial": {"MYEVS_N149_NO_SPATIAL": "1"},
    "no_opp": {"MYEVS_N149_NO_OPP": "1"},
    "no_polarity": {"MYEVS_N149_BLIND": "1"},
    "no_hot": {"MYEVS_N149_NO_HOT": "1"},
    "time_only": {
        "MYEVS_N149_NO_SPATIAL": "1",
        "MYEVS_N149_NO_HOT": "1",
        "MYEVS_N149_BLIND": "1",
    },
    "time_same_only": {
        "MYEVS_N149_NO_SPATIAL": "1",
        "MYEVS_N149_NO_HOT": "1",
        "MYEVS_N149_NO_OPP": "1",
    },
}

VARIANT_LABELS = {
    "baseline": "LLEF",
    "no_spatial": "No spatial",
    "no_opp": "No opp.",
    "no_polarity": "No polarity",
    "no_hot": "No hot",
    "time_only": "Time only",
    "time_same_only": "Time same-only",
}

VARIANT_NOTES = {
    "baseline": "完整 LLEF。",
    "no_spatial": "去掉空间距离衰减，邻域内事件只保留时间权重。",
    "no_opp": "去掉反极性邻域贡献，只使用同极性邻域证据。",
    "no_polarity": "忽略极性，正负极性邻域事件均作为同类支撑。",
    "no_hot": "去掉中心热状态可靠性折扣。",
    "time_only": "只保留时间窗内邻域支撑，去掉空间距离衰减、极性感知区分和热状态折扣。",
    "time_same_only": "只保留同极性时间窗邻域支撑，用于诊断 LLEF time-only 与 EBF 的差异来源。",
}


@dataclass(frozen=True)
class N149Input:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    p: np.ndarray
    label: np.ndarray


def _load_reference(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arr = np.load(path, allow_pickle=False)
    return (
        np.asarray(arr["t"], dtype=np.uint64),
        np.asarray(arr["x"], dtype=np.int32),
        np.asarray(arr["y"], dtype=np.int32),
        np.asarray(arr["p"], dtype=np.int8),
        np.asarray(arr["target"], dtype=np.uint8),
    )


def _auc_from_curve(rows: list[dict]) -> float:
    pts = sorted((1.0 - float(r["bsr"]), float(r["trr"])) for r in rows)
    if len(pts) < 2:
        return float("nan")
    auc = 0.0
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        auc += (x1 - x0) * (y0 + y1) * 0.5
    return float(max(0.0, min(1.0, auc)))


def _score_variant(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    *,
    r: int,
    tau_us: int,
    sigma: float,
    alpha: float,
    env_extra: dict[str, str],
) -> np.ndarray:
    env_keys = {
        "MYEVS_N149_SIGMA",
        "MYEVS_N149_ALPHA_FIXED",
        "MYEVS_N149_NO_SPATIAL",
        "MYEVS_N149_NO_OPP",
        "MYEVS_N149_NO_HOT",
        "MYEVS_N149_BLIND",
    }
    old_env = {k: os.environ.get(k) for k in env_keys}
    try:
        for k in env_keys:
            os.environ.pop(k, None)
        os.environ["MYEVS_N149_SIGMA"] = str(float(sigma))
        os.environ["MYEVS_N149_ALPHA_FIXED"] = str(float(alpha))
        for k, v in env_extra.items():
            os.environ[k] = str(v)
        ev = N149Input(t=t, x=x, y=y, p=p, label=np.zeros((t.shape[0],), dtype=np.int8))
        tb = TimeBase(tick_ns=1000.0)
        return np.asarray(
            score_stream_n149(ev, width=WIDTH, height=HEIGHT, radius_px=int(r), tau_us=int(tau_us), tb=tb),
            dtype=np.float32,
        )
    finally:
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def run_one(raw: str, variant: str, defaults: dict) -> tuple[list[dict], dict]:
    seq = parse_sequence(raw)
    path = sequence_reference_path(seq)
    if not path.exists():
        raise FileNotFoundError(f"missing reference stream {path}; run convert_evuav_to_myevs.py first")
    t, x, y, p, target = _load_reference(path)
    scores = _score_variant(
        t,
        x,
        y,
        p,
        r=int(defaults["r"]),
        tau_us=int(defaults["tau_us"]),
        sigma=float(defaults["sigma"]),
        alpha=float(defaults["alpha"]),
        env_extra=VARIANTS[variant],
    )

    rows: list[dict] = []
    for thr in THRESHOLDS:
        keep = scores >= float(thr)
        rows.append(
            {
                "sequence": seq.stem,
                "variant": variant,
                "variant_label": VARIANT_LABELS[variant],
                "threshold": float(thr),
                **metrics_from_target_keep(keep, target),
            }
        )

    best = max(rows, key=lambda row: float(row["f1_target_bg"]))
    auc_curve = _auc_from_curve(rows)
    auc_score = rank_auc(scores, target)
    edge_hit = int(float(best["threshold"]) in {float(THRESHOLDS[0]), float(THRESHOLDS[-1])})
    best_row = {
        "sequence": seq.stem,
        "variant": variant,
        "variant_label": VARIANT_LABELS[variant],
        "r": int(defaults["r"]),
        "tau_us": int(defaults["tau_us"]),
        "sigma": float(defaults["sigma"]),
        "alpha": float(defaults["alpha"]),
        "best_threshold": float(best["threshold"]),
        "auc_curve": auc_curve,
        "auc_score": auc_score,
        "f1": float(best["f1_target_bg"]),
        "trr": float(best["trr"]),
        "bsr": float(best["bsr"]),
        "precision": float(best["target_precision"]),
        "tbr_gain": float(best["target_background_ratio_gain"]),
        "target_tp": float(best["target_tp"]),
        "background_fp": float(best["background_fp"]),
        "background_tn": float(best["background_tn"]),
        "target_fn": float(best["target_fn"]),
        "edge_hit": edge_hit,
        "note": VARIANT_NOTES[variant],
    }
    for row in rows:
        row["auc_curve"] = auc_curve
        row["auc_score"] = auc_score
        row["best_threshold"] = float(best["threshold"])
        row["best_f1"] = float(best["f1_target_bg"])
        row["edge_hit"] = edge_hit
    return rows, best_row


def _mean_summary(best_rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    baseline_f1 = float(np.mean([float(r["f1"]) for r in best_rows if r["variant"] == "baseline"]))
    baseline_auc = float(np.mean([float(r["auc_score"]) for r in best_rows if r["variant"] == "baseline"]))
    for variant in VARIANTS:
        sub = [r for r in best_rows if r["variant"] == variant]
        if not sub:
            continue
        f1 = float(np.mean([float(r["f1"]) for r in sub]))
        auc = float(np.mean([float(r["auc_score"]) for r in sub]))
        out.append(
            {
                "variant": variant,
                "variant_label": VARIANT_LABELS[variant],
                "auc_score": auc,
                "delta_auc": auc - baseline_auc,
                "f1": f1,
                "delta_f1": f1 - baseline_f1,
                "trr": float(np.mean([float(r["trr"]) for r in sub])),
                "bsr": float(np.mean([float(r["bsr"]) for r in sub])),
                "precision": float(np.mean([float(r["precision"]) for r in sub])),
                "mean_threshold": float(np.mean([float(r["best_threshold"]) for r in sub])),
                "edge_hit_count": int(np.sum([int(r["edge_hit"]) for r in sub])),
                "note": VARIANT_NOTES[variant],
            }
        )
    return out


def _fmt(v: float, n: int = 4) -> str:
    return f"{float(v):.{n}f}"


def write_markdown(path: Path, best_rows: list[dict], mean_rows: list[dict], defaults: dict, sequences: list[str]) -> None:
    by_variant = {r["variant"]: r for r in mean_rows}
    lines: list[str] = []
    lines.append("# EV-UAV LLEF 组件消融实验结果\n")
    lines.append("## 实验设置\n")
    lines.append("本实验固定第 16 章确定的 EV-UAV 推荐参数，只改变 LLEF 的内部组件，并对每个变体重新扫描判决阈值，以 F1 最优点报告结果。\n")
    lines.append(f"- 序列：`{', '.join(sequences)}`")
    lines.append(f"- 固定参数：`r={defaults['r']}, tau={int(defaults['tau_us']) // 1000} ms, sigma={defaults['sigma']}, alpha={defaults['alpha']}`")
    lines.append(f"- 阈值扫描：`{THRESHOLDS[0]:g}` 到 `{THRESHOLDS[-1]:g}`，步长 `0.25`")
    lines.append("- 正类为 EV-UAV target event，负类为 background event；最终排序以 F1 为主，AUC 作为排序能力参考。\n")

    lines.append("## 变体定义\n")
    lines.append("| 变体 | 含义 |")
    lines.append("|---|---|")
    for variant in VARIANTS:
        lines.append(f"| {VARIANT_LABELS[variant]} | {VARIANT_NOTES[variant]} |")

    lines.append("\n## 六个序列均值\n")
    lines.append("| 变体 | AUC | ΔAUC | F1 | ΔF1 | TRR | BSR | Precision | 平均最优阈值 | 边界命中 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for variant in VARIANTS:
        row = by_variant[variant]
        lines.append(
            "| "
            + " | ".join(
                [
                    VARIANT_LABELS[variant],
                    _fmt(row["auc_score"]),
                    f"{float(row['delta_auc']):+.4f}",
                    _fmt(row["f1"]),
                    f"{float(row['delta_f1']):+.4f}",
                    _fmt(row["trr"]),
                    _fmt(row["bsr"]),
                    _fmt(row["precision"]),
                    _fmt(row["mean_threshold"], 2),
                    str(int(row["edge_hit_count"])),
                ]
            )
            + " |"
        )

    lines.append("\n## 逐序列 F1 最优结果\n")
    for seq in sequences:
        lines.append(f"\n### {seq}\n")
        lines.append("| 变体 | AUC | F1 | TRR | BSR | Precision | 最优阈值 | ΔF1 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        base = next(r for r in best_rows if r["sequence"] == seq and r["variant"] == "baseline")
        base_f1 = float(base["f1"])
        for variant in VARIANTS:
            row = next(r for r in best_rows if r["sequence"] == seq and r["variant"] == variant)
            lines.append(
                "| "
                + " | ".join(
                    [
                        VARIANT_LABELS[variant],
                        _fmt(row["auc_score"]),
                        _fmt(row["f1"]),
                        _fmt(row["trr"]),
                        _fmt(row["bsr"]),
                        _fmt(row["precision"]),
                        _fmt(row["best_threshold"], 2),
                        f"{float(row['f1']) - base_f1:+.4f}",
                    ]
                )
                + " |"
            )

    edge_rows = [r for r in best_rows if int(r["edge_hit"]) != 0]
    lines.append("\n## 边界检查\n")
    if edge_rows:
        lines.append("以下最优阈值命中了扫描边界，需要继续扩展阈值范围后再报告：\n")
        lines.append("| 序列 | 变体 | 最优阈值 | F1 |")
        lines.append("|---|---|---:|---:|")
        for row in edge_rows:
            lines.append(f"| {row['sequence']} | {VARIANT_LABELS[row['variant']]} | {_fmt(row['best_threshold'], 2)} | {_fmt(row['f1'])} |")
    else:
        lines.append("所有变体的 F1 最优阈值均未命中扫描边界；当前阈值范围足够覆盖本轮消融。")

    lines.append("\n## 可写入论文的初步结论\n")
    base = by_variant["baseline"]
    losses = [
        (variant, float(by_variant[variant]["delta_f1"]))
        for variant in VARIANTS
        if variant != "baseline"
    ]
    losses_sorted = sorted(losses, key=lambda item: item[1])
    worst_variant, worst_delta = losses_sorted[0]
    lines.append(
        f"在 EV-UAV 遥感小目标事件实验中，完整 LLEF 的平均 F1 为 {_fmt(base['f1'])}。"
        f"组件消融后，{VARIANT_LABELS[worst_variant]} 的平均 F1 下降最大（{worst_delta:+.4f}），"
        "说明该组件对目标事件保留与背景事件抑制的阈值工作点最关键。"
        "其余组件的变化可结合 TRR、BSR 和 Precision 判断：TRR 下降表示目标事件被过度滤除，BSR 下降表示背景抑制变弱，Precision 下降表示保留事件中背景比例升高。"
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="LLEF component ablation on EV-UAV target/background sequences.")
    ap.add_argument("--sequences", nargs="*", default=list(SELECTED_TEST_SEQUENCES))
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS.keys()), choices=tuple(VARIANTS.keys()))
    ap.add_argument("--r", type=int, default=R_DEF)
    ap.add_argument("--tau-us", type=int, default=TAU_DEF)
    ap.add_argument("--sigma", type=float, default=SIGMA_DEF)
    ap.add_argument("--alpha", type=float, default=ALPHA_DEF)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out-dir", default=str(RESULT_ROOT / "component_ablation_llef"))
    args = ap.parse_args()

    ensure_dirs()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    defaults = {"r": int(args.r), "tau_us": int(args.tau_us), "sigma": float(args.sigma), "alpha": float(args.alpha)}
    tasks = [(s, v, defaults) for s in args.sequences for v in args.variants]

    curve_rows: list[dict] = []
    best_rows: list[dict] = []
    if int(args.jobs) <= 1:
        for task in tasks:
            rows, best = run_one(*task)
            print(f"[done] {task[0]} {task[1]} F1={best['f1']:.4f} AUC={best['auc_score']:.4f}", flush=True)
            curve_rows.extend(rows)
            best_rows.append(best)
    else:
        with ProcessPoolExecutor(max_workers=int(args.jobs)) as ex:
            futs = {ex.submit(run_one, *task): task for task in tasks}
            for fut in as_completed(futs):
                raw, variant, _ = futs[fut]
                rows, best = fut.result()
                print(f"[done] {raw} {variant} F1={best['f1']:.4f} AUC={best['auc_score']:.4f}", flush=True)
                curve_rows.extend(rows)
                best_rows.append(best)

    curve_rows = sorted(curve_rows, key=lambda r: (str(r["sequence"]), str(r["variant"]), float(r["threshold"])))
    best_rows = sorted(best_rows, key=lambda r: (str(r["sequence"]), list(VARIANTS.keys()).index(str(r["variant"]))))
    mean_rows = _mean_summary(best_rows)

    curve_csv = out_dir / "evuav_llef_component_ablation_threshold_curves.csv"
    best_csv = out_dir / "evuav_llef_component_ablation_best_f1.csv"
    mean_csv = out_dir / "evuav_llef_component_ablation_mean.csv"
    md_path = out_dir / "evuav_llef_component_ablation_tables.md"

    write_csv(curve_csv, curve_rows)
    write_csv(best_csv, best_rows)
    write_csv(mean_csv, mean_rows)
    write_markdown(md_path, best_rows, mean_rows, defaults, list(args.sequences))
    write_json(
        out_dir / "evuav_llef_component_ablation_run.json",
        {
            "sequences": list(args.sequences),
            "variants": list(args.variants),
            "defaults": defaults,
            "thresholds": list(THRESHOLDS),
            "jobs": int(args.jobs),
            "outputs": {
                "curve_csv": str(curve_csv),
                "best_csv": str(best_csv),
                "mean_csv": str(mean_csv),
                "markdown": str(md_path),
            },
            "variant_notes": VARIANT_NOTES,
        },
    )

    print(f"wrote threshold curves -> {curve_csv}")
    print(f"wrote best-F1 rows -> {best_csv}")
    print(f"wrote mean summary -> {mean_csv}")
    print(f"wrote markdown tables -> {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
