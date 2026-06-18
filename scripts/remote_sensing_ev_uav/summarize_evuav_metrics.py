from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from evuav_common import RESULT_ROOT, write_csv


def _to_float(v, default=0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize EV-UAV sweep rows by best F1 per sequence/noise/algorithm.")
    ap.add_argument("--in-csv", default=str(RESULT_ROOT / "metrics" / "evuav_sweep_rows.csv"))
    ap.add_argument("--out-csv", default=str(RESULT_ROOT / "metrics" / "evuav_summary_best_f1.csv"))
    ap.add_argument("--task", choices=("auto", "shot-noise", "target-background"), default="auto")
    args = ap.parse_args()
    import csv

    with Path(args.in_csv).open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    task = args.task
    if task == "auto":
        task = rows[0].get("task", "shot-noise") if rows else "shot-noise"
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["sequence"], r.get("noise_level", r.get("noise_hz", "")), r["algorithm"])].append(r)

    out_rows: list[dict] = []
    for (seq, hz, alg), rs in sorted(groups.items()):
        if task == "target-background":
            best = max(rs, key=lambda r: _to_float(r.get("f1_target_bg")))
            out_rows.append(
                {
                    "sequence": seq,
                    "noise_level": hz,
                    "noise_mode": best.get("noise_mode", ""),
                    "task": best.get("task", task),
                    "algorithm": alg,
                    "tag": best.get("tag", ""),
                    "param": best.get("param", ""),
                    "value": best.get("value", ""),
                    "auc_curve": best.get("auc_curve", ""),
                    "auc_score": best.get("auc_score", ""),
                    "f1_target_bg": best.get("f1_target_bg", ""),
                    "target_precision": best.get("target_precision", ""),
                    "trr": best.get("trr", ""),
                    "bsr": best.get("bsr", ""),
                    "background_keep_rate": best.get("background_keep_rate", ""),
                    "target_background_ratio_gain": best.get("target_background_ratio_gain", ""),
                    "best_bsr_at_trr95": best.get("best_bsr_at_trr95", ""),
                    "best_bsr_at_trr90": best.get("best_bsr_at_trr90", ""),
                    "edge_hit": best.get("edge_hit", ""),
                }
            )
        else:
            best = max(rs, key=lambda r: _to_float(r.get("f1_ref_noise")))
            out_rows.append(
                {
                    "sequence": seq,
                    "noise_level": hz,
                    "noise_mode": best.get("noise_mode", ""),
                    "task": best.get("task", task),
                    "algorithm": alg,
                    "tag": best.get("tag", ""),
                    "param": best.get("param", ""),
                    "value": best.get("value", ""),
                    "auc_curve": best.get("auc_curve", ""),
                    "auc_score": best.get("auc_score", ""),
                    "f1_ref_noise": best.get("f1_ref_noise", ""),
                    "precision": best.get("precision", ""),
                    "recall_ref": best.get("recall_ref", ""),
                    "terr": best.get("terr", ""),
                    "rerr": best.get("rerr", ""),
                    "inrr": best.get("inrr", ""),
                    "best_inrr_at_terr95": best.get("best_inrr_at_terr95", ""),
                    "best_inrr_at_terr90": best.get("best_inrr_at_terr90", ""),
                    "edge_hit": best.get("edge_hit", ""),
                }
            )

    # Add method-level means.
    mean_rows: list[dict] = []
    by_alg: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in out_rows:
        by_alg[(r["noise_level"], r["algorithm"])].append(r)
    for (hz, alg), rs in sorted(by_alg.items()):
        base = {
            "sequence": "MEAN",
            "noise_level": hz,
            "noise_mode": rs[0].get("noise_mode", ""),
            "task": rs[0].get("task", task),
            "algorithm": alg,
            "tag": "",
            "param": "",
            "value": "",
            "auc_curve": _nanmean([_to_float(r["auc_curve"], np.nan) for r in rs]),
            "auc_score": _nanmean([_to_float(r["auc_score"], np.nan) for r in rs]),
            "edge_hit": int(any(str(r.get("edge_hit", "0")) == "1" for r in rs)),
        }
        if task == "target-background":
            base.update(
                {
                    "f1_target_bg": _nanmean([_to_float(r["f1_target_bg"], np.nan) for r in rs]),
                    "target_precision": _nanmean([_to_float(r["target_precision"], np.nan) for r in rs]),
                    "trr": _nanmean([_to_float(r["trr"], np.nan) for r in rs]),
                    "bsr": _nanmean([_to_float(r["bsr"], np.nan) for r in rs]),
                    "background_keep_rate": _nanmean([_to_float(r["background_keep_rate"], np.nan) for r in rs]),
                    "target_background_ratio_gain": _nanmean([_to_float(r["target_background_ratio_gain"], np.nan) for r in rs]),
                    "best_bsr_at_trr95": _nanmean([_to_float(r["best_bsr_at_trr95"], np.nan) for r in rs]),
                    "best_bsr_at_trr90": _nanmean([_to_float(r["best_bsr_at_trr90"], np.nan) for r in rs]),
                }
            )
        else:
            base.update(
                {
                    "f1_ref_noise": _nanmean([_to_float(r["f1_ref_noise"], np.nan) for r in rs]),
                    "precision": _nanmean([_to_float(r["precision"], np.nan) for r in rs]),
                    "recall_ref": _nanmean([_to_float(r["recall_ref"], np.nan) for r in rs]),
                    "terr": _nanmean([_to_float(r["terr"], np.nan) for r in rs]),
                    "rerr": _nanmean([_to_float(r["rerr"], np.nan) for r in rs]),
                    "inrr": _nanmean([_to_float(r["inrr"], np.nan) for r in rs]),
                    "best_inrr_at_terr95": _nanmean([_to_float(r["best_inrr_at_terr95"], np.nan) for r in rs]),
                    "best_inrr_at_terr90": _nanmean([_to_float(r["best_inrr_at_terr90"], np.nan) for r in rs]),
                }
            )
        mean_rows.append(base)

    write_csv(Path(args.out_csv), out_rows + mean_rows)
    print(f"wrote {len(out_rows) + len(mean_rows)} rows -> {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
