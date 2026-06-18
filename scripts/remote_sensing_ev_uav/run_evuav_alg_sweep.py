from __future__ import annotations

import argparse
import math
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
    SHOT_LEVELS_RATIO,
    WIDTH,
    ensure_dirs,
    metrics_from_keep,
    metrics_from_target_keep,
    noisy_path,
    parse_sequence,
    rank_auc,
    sequence_reference_path,
    write_csv,
    write_json,
)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from myevs.denoise.ops.baf import BafOp
from myevs.denoise.ops.ebf import EbfOp
from myevs.denoise.ops.pfd import PfdOp
from myevs.denoise.ops.ts import TsOp
from myevs.denoise.ops.base import Dims
from myevs.denoise.types import DenoiseConfig
from myevs.timebase import TimeBase
from myevs.denoise.ops.ebfopt_part2.n149_n145_s52_euclid_compactlut_backbone import score_stream_n149


ALGORITHMS = ("baf", "stcf", "pfd", "ts", "ebf", "llef")
TAU_US_GRID = (1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000, 1024000)
CONT_TAU_US_GRID = (8000, 16000, 32000, 64000, 128000, 256000, 512000)
CONT_R_GRID = (1, 2, 3, 4, 5)
STCF_ORIG_K_GRID = (1, 2, 3, 4, 5, 6)
THR_STD = tuple(float(x) for x in np.arange(0.0, 8.0001, 0.25))
THR_TS = tuple(float(x) for x in np.arange(0.0, 1.0001, 0.005))


@dataclass(frozen=True)
class N149Input:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    p: np.ndarray
    label: np.ndarray


def _load_events(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arr = np.load(path, allow_pickle=False)
    return (
        np.asarray(arr["t"], dtype=np.uint64),
        np.asarray(arr["x"], dtype=np.int32),
        np.asarray(arr["y"], dtype=np.int32),
        np.asarray(arr["p"], dtype=np.int8),
        np.asarray(arr["label"], dtype=np.uint8),
        np.asarray(arr["target"], dtype=np.uint8),
    )


def _binary_accept_scores(op_cls, t, x, y, p, *, tau_us: int, r: int, min_neighbors: float, refractory_us: int = 1) -> np.ndarray:
    dims = Dims(width=WIDTH, height=HEIGHT)
    tb = TimeBase(tick_ns=1000.0)
    cfg = DenoiseConfig(
        method="",
        time_window_us=int(tau_us),
        radius_px=int(r),
        min_neighbors=float(min_neighbors),
        refractory_us=int(refractory_us),
    )
    op = op_cls(dims, cfg, tb)
    keep = np.zeros((t.shape[0],), dtype=bool)
    for i in range(t.shape[0]):
        keep[i] = bool(op.accept(int(x[i]), int(y[i]), int(p[i]), int(t[i])))
    return keep


def _stcf_orig_accept_scores(t, x, y, p, *, tau_us: int, k: int) -> np.ndarray:
    """Original STCF paper-style 3x3, polarity-agnostic k-neighbor filter."""
    win_ticks = int(TimeBase(tick_ns=1000.0).us_to_ticks(int(tau_us)))
    need = max(1, int(k))
    last_ts = np.zeros((WIDTH * HEIGHT,), dtype=np.uint64)
    keep = np.zeros((t.shape[0],), dtype=bool)
    if win_ticks <= 0:
        return keep
    for i in range(t.shape[0]):
        xi = int(x[i])
        yi = int(y[i])
        ti = int(t[i])
        cnt = 0
        y0 = max(0, yi - 1)
        y1 = min(HEIGHT - 1, yi + 1)
        x0 = max(0, xi - 1)
        x1 = min(WIDTH - 1, xi + 1)
        for yy in range(y0, y1 + 1):
            base = yy * WIDTH
            for xx in range(x0, x1 + 1):
                if xx == xi and yy == yi:
                    continue
                ts = int(last_ts[base + xx])
                if ts == 0:
                    continue
                dt = ti - ts
                if 0 <= dt <= win_ticks:
                    cnt += 1
                    if cnt >= need:
                        keep[i] = True
                        break
            if keep[i]:
                break
        last_ts[yi * WIDTH + xi] = np.uint64(ti)
    return keep


def _ebf_scores(t, x, y, p, *, tau_us: int, r: int) -> np.ndarray:
    dims = Dims(width=WIDTH, height=HEIGHT)
    tb = TimeBase(tick_ns=1000.0)
    cfg = DenoiseConfig(method="ebf", time_window_us=int(tau_us), radius_px=int(r), min_neighbors=0.0)
    op = EbfOp(dims, cfg, tb)
    out = np.zeros((t.shape[0],), dtype=np.float32)
    for i in range(t.shape[0]):
        out[i] = float(op.score(int(x[i]), int(y[i]), int(p[i]), int(t[i])))
    return out


def _ts_scores(t, x, y, p, *, tau_us: int, r: int) -> np.ndarray:
    dims = Dims(width=WIDTH, height=HEIGHT)
    tb = TimeBase(tick_ns=1000.0)
    cfg = DenoiseConfig(method="ts", time_window_us=int(tau_us), radius_px=int(r), min_neighbors=0.0)
    op = TsOp(dims, cfg, tb)
    decay_ticks = max(1, int(tb.us_to_ticks(int(tau_us))))
    inv_decay = 1.0 / float(decay_ticks)
    out = np.zeros((t.shape[0],), dtype=np.float32)
    for i in range(t.shape[0]):
        xi = int(x[i])
        yi = int(y[i])
        pi = int(p[i])
        ti = int(t[i])
        cell = op.pos_ts if pi > 0 else op.neg_ts
        y0 = max(0, yi - r)
        y1 = min(HEIGHT - 1, yi + r)
        x0 = max(0, xi - r)
        x1 = min(WIDTH - 1, xi + r)
        support = 0
        surf = 0.0
        for yy in range(y0, y1 + 1):
            base = yy * WIDTH
            for xx in range(x0, x1 + 1):
                ts = int(cell[base + xx])
                if ts == 0:
                    continue
                dt = ti - ts
                if dt < 0:
                    dt = -dt
                surf += math.exp(-float(dt) * inv_decay)
                support += 1
        out[i] = float(surf / support) if support > 0 else 0.0
        cell[yi * WIDTH + xi] = np.uint64(ti)
    return out


def _llef_scores(t, x, y, p, *, tau_us: int, r: int, sigma: float = 2.5, alpha: float | None = None) -> np.ndarray:
    old_sigma = os.environ.get("MYEVS_N149_SIGMA")
    old_alpha = os.environ.get("MYEVS_N149_ALPHA_FIXED")
    os.environ["MYEVS_N149_SIGMA"] = str(float(sigma))
    if alpha is None:
        os.environ.pop("MYEVS_N149_ALPHA_FIXED", None)
    else:
        os.environ["MYEVS_N149_ALPHA_FIXED"] = str(float(alpha))
    tb = TimeBase(tick_ns=1000.0)
    try:
        ev = N149Input(t=t, x=x, y=y, p=p, label=np.zeros((t.shape[0],), dtype=np.int8))
        return np.asarray(score_stream_n149(ev, width=WIDTH, height=HEIGHT, radius_px=int(r), tau_us=int(tau_us), tb=tb), dtype=np.float32)
    finally:
        if old_sigma is None:
            os.environ.pop("MYEVS_N149_SIGMA", None)
        else:
            os.environ["MYEVS_N149_SIGMA"] = old_sigma
        if old_alpha is None:
            os.environ.pop("MYEVS_N149_ALPHA_FIXED", None)
        else:
            os.environ["MYEVS_N149_ALPHA_FIXED"] = old_alpha


def _continuous_configs(alg: str) -> list[tuple[int, int, float | None, float | None, str]]:
    configs: list[tuple[int, int, float | None, float | None, str]] = []
    for r in CONT_R_GRID:
        for tau in CONT_TAU_US_GRID:
            configs.append((int(r), int(tau), None, None, f"r{r}_tau{tau}"))
    if alg == "llef":
        configs.append((4, 256000, 3.0, 4.0, "r4_tau256000_sigma3_alpha4"))
    return configs


def _curve_from_scores(scores: np.ndarray, labels: np.ndarray, target: np.ndarray, thresholds: tuple[float, ...], *, higher_is_ref: bool = True) -> list[dict]:
    rows: list[dict] = []
    for thr in thresholds:
        keep = scores >= float(thr) if higher_is_ref else scores <= float(thr)
        m = metrics_from_keep(keep, labels, target)
        rows.append({"param": "score-threshold", "value": float(thr), **m})
    return rows


def _curve_from_scores_target(scores: np.ndarray, target: np.ndarray, thresholds: tuple[float, ...], *, higher_is_target: bool = True) -> list[dict]:
    rows: list[dict] = []
    for thr in thresholds:
        keep = scores >= float(thr) if higher_is_target else scores <= float(thr)
        rows.append({"param": "score-threshold", "value": float(thr), **metrics_from_target_keep(keep, target)})
    return rows


def _auc_from_curve(rows: list[dict]) -> float:
    pts = sorted((1.0 - float(r["inrr"]), float(r["rerr"])) for r in rows)
    if len(pts) < 2:
        return float("nan")
    auc = 0.0
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        auc += (x1 - x0) * (y0 + y1) * 0.5
    return float(max(0.0, min(1.0, auc)))


def _auc_from_target_curve(rows: list[dict]) -> float:
    pts = sorted((1.0 - float(r["bsr"]), float(r["trr"])) for r in rows)
    if len(pts) < 2:
        return float("nan")
    auc = 0.0
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        auc += (x1 - x0) * (y0 + y1) * 0.5
    return float(max(0.0, min(1.0, auc)))


def _add_summary_fields(rows: list[dict], *, sequence: str, hz: float, alg: str, tag: str, edge_values: set[float], task: str = "shot-noise") -> list[dict]:
    if task == "target-background":
        auc = _auc_from_target_curve(rows)
        best_f1 = max(rows, key=lambda r: float(r["f1_target_bg"]))
        best_trr95 = [r for r in rows if float(r["trr"]) >= 0.95]
        best_trr90 = [r for r in rows if float(r["trr"]) >= 0.90]
        best_bsr95 = max((float(r["bsr"]) for r in best_trr95), default=float("nan"))
        best_bsr90 = max((float(r["bsr"]) for r in best_trr90), default=float("nan"))
        out = []
        for r in rows:
            v = float(r["value"])
            edge_hit = 1 if v in edge_values and r is best_f1 else 0
            out.append(
                {
                    "sequence": sequence,
                    "noise_level": hz,
                    "algorithm": alg,
                    "tag": tag,
                    "auc_curve": auc,
                    "best_f1": float(best_f1["f1_target_bg"]),
                    "best_f1_value": float(best_f1["value"]),
                    "best_bsr_at_trr95": best_bsr95,
                    "best_bsr_at_trr90": best_bsr90,
                    "edge_hit": edge_hit,
                    **r,
                }
            )
        return out

    auc = _auc_from_curve(rows)
    best_f1 = max(rows, key=lambda r: float(r["f1_ref_noise"]))
    best_terr95 = [r for r in rows if float(r["terr"]) >= 0.95]
    best_terr90 = [r for r in rows if float(r["terr"]) >= 0.90]
    best_inrr95 = max((float(r["inrr"]) for r in best_terr95), default=float("nan"))
    best_inrr90 = max((float(r["inrr"]) for r in best_terr90), default=float("nan"))
    out = []
    for r in rows:
        v = float(r["value"])
        edge_hit = 1 if v in edge_values and r is best_f1 else 0
        out.append(
            {
                "sequence": sequence,
                "noise_level": hz,
                "algorithm": alg,
                "tag": tag,
                "auc_curve": auc,
                "best_f1": float(best_f1["f1_ref_noise"]),
                "best_f1_value": float(best_f1["value"]),
                "best_inrr_at_terr95": best_inrr95,
                "best_inrr_at_terr90": best_inrr90,
                "edge_hit": edge_hit,
                **r,
            }
        )
    return out


def evaluate_algorithm(alg: str, t, x, y, p, labels, target) -> list[dict]:
    all_rows: list[dict] = []
    if alg == "baf":
        r = 1
        rows = []
        for tau in TAU_US_GRID:
            keep = _binary_accept_scores(BafOp, t, x, y, p, tau_us=tau, r=r, min_neighbors=1.0)
            rows.append({"param": "time-us", "value": float(tau), **metrics_from_keep(keep, labels, target)})
        all_rows.extend(_add_summary_fields(rows, sequence="", hz=0, alg=alg, tag=f"r{r}", edge_values={float(TAU_US_GRID[0]), float(TAU_US_GRID[-1])}))
    elif alg == "stcf":
        for k in STCF_ORIG_K_GRID:
            rows = []
            for tau in TAU_US_GRID:
                keep = _stcf_orig_accept_scores(t, x, y, p, tau_us=tau, k=k)
                rows.append({"param": "time-us", "value": float(tau), **metrics_from_keep(keep, labels, target)})
            all_rows.extend(_add_summary_fields(rows, sequence="", hz=0, alg=alg, tag=f"stcf_orig_k{k}", edge_values={float(TAU_US_GRID[0]), float(TAU_US_GRID[-1])}))
    elif alg == "pfd":
        for r in (1, 2):
            rows = []
            for tau in TAU_US_GRID:
                keep = _binary_accept_scores(PfdOp, t, x, y, p, tau_us=tau, r=r, min_neighbors=1.0, refractory_us=1)
                rows.append({"param": "time-us", "value": float(tau), **metrics_from_keep(keep, labels, target)})
            all_rows.extend(_add_summary_fields(rows, sequence="", hz=0, alg=alg, tag=f"r{r}_m1_k1", edge_values={float(TAU_US_GRID[0]), float(TAU_US_GRID[-1])}))
    elif alg in ("ts", "ebf", "llef"):
        scorer = {"ts": _ts_scores, "ebf": _ebf_scores, "llef": _llef_scores}[alg]
        thresholds = THR_TS if alg == "ts" else THR_STD
        for r, tau, sigma, alpha, tag in _continuous_configs(alg):
            if alg == "llef" and sigma is not None:
                scores = scorer(t, x, y, p, tau_us=tau, r=r, sigma=float(sigma), alpha=alpha)
            else:
                scores = scorer(t, x, y, p, tau_us=tau, r=r)
            rows = _curve_from_scores(scores, labels, target, thresholds)
            # Use rank AUC as an additional score-based AUC for continuous methods.
            score_auc = rank_auc(scores, labels)
            tagged = _add_summary_fields(rows, sequence="", hz=0, alg=alg, tag=tag, edge_values={float(thresholds[0]), float(thresholds[-1])})
            for row in tagged:
                row["auc_score"] = score_auc
            all_rows.extend(tagged)
    else:
        raise ValueError(f"unsupported algorithm: {alg}")
    return all_rows


def evaluate_algorithm_target_background(alg: str, t, x, y, p, target) -> list[dict]:
    all_rows: list[dict] = []
    if alg == "baf":
        r = 1
        rows = []
        for tau in TAU_US_GRID:
            keep = _binary_accept_scores(BafOp, t, x, y, p, tau_us=tau, r=r, min_neighbors=1.0)
            rows.append({"param": "time-us", "value": float(tau), **metrics_from_target_keep(keep, target)})
        all_rows.extend(_add_summary_fields(rows, sequence="", hz=0, alg=alg, tag=f"r{r}", edge_values={float(TAU_US_GRID[0]), float(TAU_US_GRID[-1])}, task="target-background"))
    elif alg == "stcf":
        for k in STCF_ORIG_K_GRID:
            rows = []
            for tau in TAU_US_GRID:
                keep = _stcf_orig_accept_scores(t, x, y, p, tau_us=tau, k=k)
                rows.append({"param": "time-us", "value": float(tau), **metrics_from_target_keep(keep, target)})
            all_rows.extend(_add_summary_fields(rows, sequence="", hz=0, alg=alg, tag=f"stcf_orig_k{k}", edge_values={float(TAU_US_GRID[0]), float(TAU_US_GRID[-1])}, task="target-background"))
    elif alg == "pfd":
        for r in (1, 2):
            rows = []
            for tau in TAU_US_GRID:
                keep = _binary_accept_scores(PfdOp, t, x, y, p, tau_us=tau, r=r, min_neighbors=1.0, refractory_us=1)
                rows.append({"param": "time-us", "value": float(tau), **metrics_from_target_keep(keep, target)})
            all_rows.extend(_add_summary_fields(rows, sequence="", hz=0, alg=alg, tag=f"r{r}_m1_k1", edge_values={float(TAU_US_GRID[0]), float(TAU_US_GRID[-1])}, task="target-background"))
    elif alg in ("ts", "ebf", "llef"):
        scorer = {"ts": _ts_scores, "ebf": _ebf_scores, "llef": _llef_scores}[alg]
        thresholds = THR_TS if alg == "ts" else THR_STD
        for r, tau, sigma, alpha, tag in _continuous_configs(alg):
            if alg == "llef" and sigma is not None:
                scores = scorer(t, x, y, p, tau_us=tau, r=r, sigma=float(sigma), alpha=alpha)
            else:
                scores = scorer(t, x, y, p, tau_us=tau, r=r)
            rows = _curve_from_scores_target(scores, target, thresholds)
            score_auc = rank_auc(scores, target)
            tagged = _add_summary_fields(rows, sequence="", hz=0, alg=alg, tag=tag, edge_values={float(thresholds[0]), float(thresholds[-1])}, task="target-background")
            for row in tagged:
                row["auc_score"] = score_auc
            all_rows.extend(tagged)
    else:
        raise ValueError(f"unsupported algorithm: {alg}")
    return all_rows


def _run_one_task(task: tuple[str, float, str, str, int, str]) -> list[dict]:
    raw, level, mode, alg, limit_events, eval_task = task
    seq = parse_sequence(raw)
    if eval_task == "target-background":
        path = sequence_reference_path(seq)
        if not path.exists():
            raise FileNotFoundError(f"missing reference stream {path}; run convert_evuav_to_myevs.py first")
    else:
        path = noisy_path(seq, level, mode=mode)
        if not path.exists():
            raise FileNotFoundError(f"missing noisy stream {path}; run generate_evuav_shot_noise.py first")
    t, x, y, p, labels, target = _load_events(path)
    if limit_events and int(limit_events) > 0:
        n = min(int(limit_events), t.shape[0])
        t, x, y, p, labels, target = t[:n], x[:n], y[:n], p[:n], labels[:n], target[:n]
    if eval_task == "target-background":
        rows = evaluate_algorithm_target_background(alg, t, x, y, p, target)
    else:
        rows = evaluate_algorithm(alg, t, x, y, p, labels, target)
    for r in rows:
        r["sequence"] = seq.stem
        r["noise_mode"] = mode
        r["noise_level"] = float(level)
        r["task"] = eval_task
        r["algorithm"] = alg
        r["events_evaluated"] = int(t.shape[0])
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep six traditional filters on EV-UAV streams.")
    ap.add_argument("--sequences", nargs="*", default=list(SELECTED_TEST_SEQUENCES))
    ap.add_argument("--task", choices=("shot-noise", "target-background"), default="target-background")
    ap.add_argument("--levels-ratio", nargs="*", type=float, default=list(SHOT_LEVELS_RATIO))
    ap.add_argument("--levels-hz", nargs="*", type=float, default=None, help="Optional legacy Hz/pixel mode.")
    ap.add_argument("--algorithms", default="baf,stcf,pfd,ts,ebf,llef")
    ap.add_argument("--out-csv", default=str(RESULT_ROOT / "metrics" / "evuav_sweep_rows.csv"))
    ap.add_argument("--limit-events", type=int, default=0, help="Debug only: limit events per stream.")
    ap.add_argument("--jobs", type=int, default=1, help="Parallel worker processes over sequence/level/algorithm tasks.")
    args = ap.parse_args()
    ensure_dirs()
    algs = [a.strip().lower() for a in args.algorithms.split(",") if a.strip()]
    mode = "hz" if args.levels_hz is not None else "ratio"
    levels = [0.0] if args.task == "target-background" else (list(args.levels_hz) if mode == "hz" else list(args.levels_ratio))
    all_rows: list[dict] = []
    run_meta = {
        "sequences": args.sequences,
        "task": args.task,
        "noise_mode": mode,
        "levels": levels,
        "algorithms": algs,
        "limit_events": int(args.limit_events),
        "jobs": int(args.jobs),
    }
    tasks = [(raw, float(level), mode, alg, int(args.limit_events), args.task) for raw in args.sequences for level in levels for alg in algs]
    if int(args.jobs) <= 1:
        for task in tasks:
            raw, level, _, alg, _ = task
            print(f"[run] {raw} {mode}={level:g} {alg}")
            all_rows.extend(_run_one_task(task))
    else:
        with ProcessPoolExecutor(max_workers=int(args.jobs)) as ex:
            futs = {ex.submit(_run_one_task, task): task for task in tasks}
            for fut in as_completed(futs):
                raw, level, _, alg, _, _ = futs[fut]
                rows = fut.result()
                print(f"[done] {raw} {mode}={level:g} {alg} rows={len(rows)}")
                all_rows.extend(rows)
    out = Path(args.out_csv)
    write_csv(out, all_rows)
    write_json(RESULT_ROOT / "logs" / "evuav_sweep_last_run.json", run_meta)
    print(f"wrote {len(all_rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
