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
    format_float,
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


R_DEF = 3
TAU_DEF = 128000
SIGMA_DEF = 2.5
ALPHA_DEF: float | None = None

R_GRID = (1, 2, 3, 4, 5)
TAU_GRID = (8000, 16000, 32000, 64000, 128000, 256000, 512000)
SIGMA_GRID = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
ALPHA_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 4.0, 6.0, 8.0)
THRESHOLDS = tuple(float(x) for x in np.arange(0.0, 8.0001, 0.25))


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


def _best_threshold_metrics(scores: np.ndarray, target: np.ndarray) -> dict:
    rows: list[dict] = []
    for thr in THRESHOLDS:
        keep = scores >= float(thr)
        rows.append({"threshold": float(thr), **metrics_from_target_keep(keep, target)})
    best = max(rows, key=lambda r: float(r["f1_target_bg"]))
    auc_curve = _auc_from_curve(rows)
    auc_score = rank_auc(scores, target)
    return {
        "auc": auc_curve,
        "auc_score": auc_score,
        "best_threshold": float(best["threshold"]),
        "f1": float(best["f1_target_bg"]),
        "trr": float(best["trr"]),
        "bsr": float(best["bsr"]),
        "precision": float(best["target_precision"]),
        "tbr_gain": float(best["target_background_ratio_gain"]),
    }


def _score_llef(t, x, y, p, *, r: int, tau_us: int, sigma: float, alpha: float | None) -> np.ndarray:
    old_sigma = os.environ.get("MYEVS_N149_SIGMA")
    old_alpha = os.environ.get("MYEVS_N149_ALPHA_FIXED")
    os.environ["MYEVS_N149_SIGMA"] = str(float(sigma))
    if alpha is None:
        os.environ.pop("MYEVS_N149_ALPHA_FIXED", None)
    else:
        os.environ["MYEVS_N149_ALPHA_FIXED"] = str(float(alpha))
    try:
        ev = N149Input(t=t, x=x, y=y, p=p, label=np.zeros((t.shape[0],), dtype=np.int8))
        tb = TimeBase(tick_ns=1000.0)
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


def _task_values(param: str) -> tuple[object, ...]:
    if param == "r":
        return tuple(R_GRID)
    if param == "tau":
        return tuple(TAU_GRID)
    if param == "sigma":
        return tuple(SIGMA_GRID)
    if param == "alpha":
        return tuple(ALPHA_GRID)
    raise ValueError(f"unsupported param: {param}")


def _resolve_params(param: str, value: object, defaults: dict) -> tuple[int, int, float, float | None]:
    r = int(defaults["r"])
    tau = int(defaults["tau_us"])
    sigma = float(defaults["sigma"])
    alpha = defaults["alpha"]
    if param == "r":
        r = int(value)
    elif param == "tau":
        tau = int(value)
    elif param == "sigma":
        sigma = float(value)
    elif param == "alpha":
        alpha = float(value)
    return r, tau, sigma, alpha


def run_one(raw: str, param: str, value: object, defaults: dict) -> dict:
    seq = parse_sequence(raw)
    path = sequence_reference_path(seq)
    if not path.exists():
        raise FileNotFoundError(f"missing reference stream {path}; run convert_evuav_to_myevs.py first")
    t, x, y, p, target = _load_reference(path)
    r, tau, sigma, alpha = _resolve_params(param, value, defaults)
    scores = _score_llef(t, x, y, p, r=r, tau_us=tau, sigma=sigma, alpha=alpha)
    m = _best_threshold_metrics(scores, target)
    return {
        "sequence": seq.stem,
        "param": param,
        "value": value,
        "r": r,
        "tau_us": tau,
        "sigma": sigma,
        "alpha": "adaptive" if alpha is None else float(alpha),
        **m,
    }


def summarize(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for param in ("r", "tau", "sigma", "alpha"):
        values = sorted({str(r["value"]) for r in rows if r["param"] == param}, key=lambda v: float(v))
        for value in values:
            sub = [r for r in rows if r["param"] == param and str(r["value"]) == value]
            if not sub:
                continue
            out.append(
                {
                    "sequence": "MEAN",
                    "param": param,
                    "value": value,
                    "auc": float(np.mean([float(r["auc"]) for r in sub])),
                    "auc_score": float(np.mean([float(r["auc_score"]) for r in sub])),
                    "f1": float(np.mean([float(r["f1"]) for r in sub])),
                    "trr": float(np.mean([float(r["trr"]) for r in sub])),
                    "bsr": float(np.mean([float(r["bsr"]) for r in sub])),
                    "precision": float(np.mean([float(r["precision"]) for r in sub])),
                    "tbr_gain": float(np.mean([float(r["tbr_gain"]) for r in sub])),
                }
            )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="LLEF hyperparameter sweep on EV-UAV target/background sequences.")
    ap.add_argument("--sequences", nargs="*", default=list(SELECTED_TEST_SEQUENCES))
    ap.add_argument("--params", nargs="*", default=["r", "tau", "sigma"], choices=("r", "tau", "sigma", "alpha"))
    ap.add_argument("--default-r", type=int, default=R_DEF)
    ap.add_argument("--default-tau-us", type=int, default=TAU_DEF)
    ap.add_argument("--default-sigma", type=float, default=SIGMA_DEF)
    ap.add_argument("--default-alpha", type=float, default=None, help="Unset means adaptive alpha.")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out-dir", default=str(RESULT_ROOT / "hyperparam"))
    args = ap.parse_args()

    ensure_dirs()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    defaults = {
        "r": int(args.default_r),
        "tau_us": int(args.default_tau_us),
        "sigma": float(args.default_sigma),
        "alpha": None if args.default_alpha is None else float(args.default_alpha),
    }
    tasks = [(s, p, v, defaults) for s in args.sequences for p in args.params for v in _task_values(p)]
    rows: list[dict] = []
    if int(args.jobs) <= 1:
        for t in tasks:
            print(f"[run] {t[0]} {t[1]}={t[2]}")
            rows.append(run_one(*t))
    else:
        with ProcessPoolExecutor(max_workers=int(args.jobs)) as ex:
            futs = {ex.submit(run_one, *t): t for t in tasks}
            for fut in as_completed(futs):
                s, p, v, _ = futs[fut]
                row = fut.result()
                print(f"[done] {s} {p}={v} AUC={row['auc']:.4f} F1={row['f1']:.4f}", flush=True)
                rows.append(row)

    rows = sorted(rows, key=lambda r: (str(r["sequence"]), str(r["param"]), float(r["value"])))
    summary_rows = summarize(rows)

    raw_csv = out_dir / "evuav_llef_hyperparam_raw.csv"
    summary_csv = out_dir / "evuav_llef_hyperparam_summary.csv"
    write_csv(raw_csv, rows)
    write_csv(summary_csv, summary_rows)
    write_json(
        out_dir / "evuav_llef_hyperparam_run.json",
        {
            "sequences": list(args.sequences),
            "params": list(args.params),
            "defaults": defaults,
            "grids": {"r": list(R_GRID), "tau_us": list(TAU_GRID), "sigma": list(SIGMA_GRID), "alpha": list(ALPHA_GRID), "threshold": list(THRESHOLDS)},
            "jobs": int(args.jobs),
            "outputs": {"raw_csv": str(raw_csv), "summary_csv": str(summary_csv)},
            "note": "Unset default alpha keeps the original adaptive-alpha logic. Setting MYEVS_N149_ALPHA_FIXED or --default-alpha uses a fixed alpha.",
        },
    )
    print(f"wrote {len(rows)} rows -> {raw_csv}")
    print(f"wrote {len(summary_rows)} rows -> {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
