from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

from myevs.timebase import TimeBase

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from sweep_ebf_slim_labelscore_grid import (
    ROC_HEADER,
    _best_f1_index,
    _float_tag,
    _patch_aocc_in_roc_csv,
    _patch_esr_mean_in_roc_csv,
    _roc_points_from_scores,
    _score_stream,
    _write_roc_rows,
    aocc_from_xyt,
    event_structural_ratio_mean_from_xy,
    load_labeled_npy,
)


def _parse_int_list(s: str) -> list[int]:
    items = [x.strip() for x in str(s).split(",") if x.strip()]
    return [int(float(it)) for it in items]


def _parse_float_list(s: str) -> list[float]:
    items = [x.strip() for x in str(s).split(",") if x.strip()]
    return [float(it) for it in items]


def _join_ints(values: list[int], *, sep: str = "_") -> str:
    return sep.join(str(int(v)) for v in values)


def _tau_list_tag(tau_us_list: list[int]) -> str:
    tau_us_list = [int(v) for v in tau_us_list]
    if all((v % 1000) == 0 for v in tau_us_list):
        tau_ms = [v // 1000 for v in tau_us_list]
        return f"tau{_join_ints(tau_ms)}ms"
    return f"tau{_join_ints(tau_us_list)}us"


def main() -> int:
    ap = argparse.ArgumentParser(description="Standalone n139 sweep (migrated from sweep_ebf_slim_labelscore_grid.py)")
    ap.add_argument("--max-events", type=int, default=int(os.environ.get("EBF_MAX_EVENTS", "0")), help="0=all")
    ap.add_argument(
        "--out-dir",
        default="data/ED24/myPedestrain_06/EBF_Part2/_slim_n139_785_prescreen400k",
        help="output directory",
    )

    ap.add_argument("--width", type=int, default=346)
    ap.add_argument("--height", type=int, default=260)
    ap.add_argument("--tick-ns", type=float, default=1000.0)

    ap.add_argument("--s-list", default="3,5,7,9")
    ap.add_argument("--tau-us-list", default="8000,16000,32000,64000,128000,256000,512000,1024000")
    ap.add_argument("--n139-low-list", default="0.1,0.2,0.3,0.4,0.5")
    ap.add_argument("--n139-high-list", default="0.5,0.6,0.7,0.8,0.9")
    ap.add_argument("--roc-max-points", type=int, default=5000)

    ap.add_argument(
        "--esr-mode",
        default=str(os.environ.get("MYEVS_ESR_MODE", "best")),
        choices=["best", "all", "off"],
    )
    ap.add_argument(
        "--aocc-mode",
        default=str(os.environ.get("MYEVS_AOCC_MODE", "best")),
        choices=["best", "all", "off"],
    )

    ap.add_argument(
        "--light",
        default=r"D:\hjx_workspace\scientific_reserach\dataset\ED24\myPedestrain_06\Pedestrain_06_1.8.npy",
    )
    ap.add_argument(
        "--mid",
        default=r"D:\hjx_workspace\scientific_reserach\dataset\ED24\myPedestrain_06\Pedestrain_06_2.5.npy",
    )
    ap.add_argument(
        "--heavy",
        default=r"D:\hjx_workspace\scientific_reserach\dataset\ED24\myPedestrain_06\Pedestrain_06_3.3.npy",
    )

    args = ap.parse_args()

    tb = TimeBase(tick_ns=float(args.tick_ns))
    s_list = _parse_int_list(args.s_list)
    for s in s_list:
        if s < 3 or s % 2 == 0:
            raise SystemExit(f"--s-list expects odd diameters >=3 (got {s})")
    tau_us_list = _parse_int_list(args.tau_us_list)
    n139_low_list = _parse_float_list(args.n139_low_list)
    n139_high_list = _parse_float_list(args.n139_high_list)

    env_inputs = {"light": str(args.light), "mid": str(args.mid), "heavy": str(args.heavy)}

    out_dir = str(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    esr_mode = str(args.esr_mode).strip().lower()
    aocc_mode = str(args.aocc_mode).strip().lower()
    need_best_recipes = (esr_mode == "best") or (aocc_mode == "best")

    s_tag = f"s{_join_ints([int(s) for s in s_list])}"
    tau_tag = _tau_list_tag([int(x) for x in tau_us_list])
    roc_csv = {
        env: os.path.join(out_dir, f"roc_ebf_n139_{env}_labelscore_{s_tag}_{tau_tag}.csv")
        for env in env_inputs
    }

    for env, p in roc_csv.items():
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(ROC_HEADER)

    kernel_cache: dict[str, object] = {}
    best_auc_by_env: dict[str, tuple[str, float]] = {"light": ("", -1.0), "mid": ("", -1.0), "heavy": ("", -1.0)}
    best_f1_by_env: dict[str, tuple[str, float]] = {"light": ("", -1.0), "mid": ("", -1.0), "heavy": ("", -1.0)}
    n139_records: dict[str, list[dict[str, float | int | str]]] = {"light": [], "mid": [], "heavy": []}

    for env, in_path in env_inputs.items():
        ev = load_labeled_npy(in_path, max_events=int(args.max_events))
        n = int(ev.label.shape[0])
        pos = int(np.sum(ev.label))
        neg = int(n - pos)
        print(f"loaded: env={env} n={n} pos={pos} neg={neg} in={in_path}")

        best_auc_recipe: dict[str, object] | None = None
        best_f1_recipe: dict[str, object] | None = None

        with open(roc_csv[env], "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)

            for s in s_list:
                r = int((int(s) - 1) // 2)
                for tau_us in tau_us_list:
                    low_high_pairs: list[tuple[float, float]] = []
                    for lo in n139_low_list:
                        for hi in n139_high_list:
                            if float(lo) >= float(hi):
                                continue
                            low_high_pairs.append((float(lo), float(hi)))
                    if not low_high_pairs:
                        raise SystemExit("n139 requires at least one valid pair where low < high")

                    for n139_lo, n139_hi in low_high_pairs:
                        tag = (
                            f"ebf_n139_labelscore_s{int(s)}_tau{int(tau_us)}"
                            f"_l{_float_tag(n139_lo)}_h{_float_tag(n139_hi)}"
                        )

                        scores = _score_stream(
                            ev,
                            width=int(args.width),
                            height=int(args.height),
                            radius_px=int(r),
                            tau_us=int(tau_us),
                            tb=tb,
                            variant="n139",
                            kernel_cache=kernel_cache,
                            env_name=str(env),
                            n139_low=float(n139_lo),
                            n139_high=float(n139_hi),
                        )

                        auc, thr, tp, fp, _fpr, _tpr = _roc_points_from_scores(
                            ev.label,
                            scores,
                            max_points=int(args.roc_max_points),
                        )

                        best_i = _best_f1_index(thr, tp, fp, pos=pos, neg=neg)
                        best_thr = float(thr[int(best_i)])

                        if auc > best_auc_by_env[env][1]:
                            best_auc_by_env[env] = (tag, float(auc))

                        tp_b = int(tp[int(best_i)])
                        fp_b = int(fp[int(best_i)])
                        tpr_b = (tp_b / pos) if pos > 0 else 0.0
                        prec_den_b = tp_b + fp_b
                        precision_b = (tp_b / prec_den_b) if prec_den_b > 0 else 0.0
                        den_b = precision_b + tpr_b
                        best_f1 = float((2.0 * precision_b * tpr_b / den_b) if den_b > 0 else 0.0)
                        if best_f1 > best_f1_by_env[env][1]:
                            best_f1_by_env[env] = (tag, float(best_f1))

                        n139_records[env].append(
                            {
                                "s": int(s),
                                "tau_us": int(tau_us),
                                "low": float(n139_lo),
                                "high": float(n139_hi),
                                "auc": float(auc),
                                "best_f1": float(best_f1),
                                "tag": str(tag),
                            }
                        )

                        if bool(need_best_recipes):
                            if best_auc_recipe is None or float(auc) > float(best_auc_recipe["auc"]):
                                best_auc_recipe = {
                                    "auc": float(auc),
                                    "tag": str(tag),
                                    "best_i": int(best_i),
                                    "best_thr": float(best_thr),
                                    "s": int(s),
                                    "r": int(r),
                                    "tau_us": int(tau_us),
                                    "n139_low": float(n139_lo),
                                    "n139_high": float(n139_hi),
                                }
                            if best_f1_recipe is None or float(best_f1) > float(best_f1_recipe["f1"]):
                                best_f1_recipe = {
                                    "f1": float(best_f1),
                                    "tag": str(tag),
                                    "best_i": int(best_i),
                                    "best_thr": float(best_thr),
                                    "s": int(s),
                                    "r": int(r),
                                    "tau_us": int(tau_us),
                                    "n139_low": float(n139_lo),
                                    "n139_high": float(n139_hi),
                                }

                        esr_mean: float | None = None
                        if esr_mode == "all":
                            kept = scores >= best_thr
                            esr_mean = float(
                                event_structural_ratio_mean_from_xy(
                                    ev.x[kept],
                                    ev.y[kept],
                                    width=int(args.width),
                                    height=int(args.height),
                                    chunk_size=30000,
                                )
                            )

                        aocc: float | None = None
                        if aocc_mode == "all":
                            kept = scores >= best_thr
                            aocc = float(
                                aocc_from_xyt(
                                    ev.x[kept],
                                    ev.y[kept],
                                    ev.t[kept],
                                    width=int(args.width),
                                    height=int(args.height),
                                )
                            )

                        _write_roc_rows(
                            w,
                            tag=tag,
                            thresholds=thr,
                            tp=tp,
                            fp=fp,
                            pos=pos,
                            neg=neg,
                            auc=float(auc),
                            esr_mean=(None if esr_mean is None else float(esr_mean)),
                            esr_at_index=int(best_i),
                            aocc=(None if aocc is None else float(aocc)),
                            aocc_at_index=int(best_i),
                        )

                    print(
                        f"done: env={env} s={int(s)} tau_us={int(tau_us)} "
                        f"pairs={int(len(low_high_pairs))}"
                    )

        if bool(need_best_recipes) and (best_auc_recipe is not None or best_f1_recipe is not None):
            recipes: list[dict[str, object]] = []
            if best_auc_recipe is not None:
                recipes.append(best_auc_recipe)
            if best_f1_recipe is not None and (
                best_auc_recipe is None or str(best_f1_recipe.get("tag")) != str(best_auc_recipe.get("tag"))
            ):
                recipes.append(best_f1_recipe)

            scores_cache: dict[str, np.ndarray] = {}

            def _scores_and_kept_for_recipe(recipe: dict[str, object]) -> tuple[str, int, np.ndarray]:
                tag = str(recipe["tag"])
                best_i = int(recipe["best_i"])
                best_thr = float(recipe["best_thr"])

                if tag in scores_cache:
                    scores = scores_cache[tag]
                else:
                    scores = _score_stream(
                        ev,
                        width=int(args.width),
                        height=int(args.height),
                        radius_px=int(recipe["r"]),
                        tau_us=int(recipe["tau_us"]),
                        tb=tb,
                        variant="n139",
                        kernel_cache=kernel_cache,
                        env_name=str(env),
                        n139_low=float(recipe["n139_low"]),
                        n139_high=float(recipe["n139_high"]),
                    )
                    scores_cache[tag] = scores

                kept = scores >= best_thr
                return tag, best_i, kept

            if esr_mode == "best":
                esr_targets: dict[str, tuple[int, float]] = {}
                for recipe in recipes:
                    tag, best_i, kept = _scores_and_kept_for_recipe(recipe)
                    if tag in esr_targets:
                        continue
                    esr_targets[tag] = (
                        int(best_i),
                        float(
                            event_structural_ratio_mean_from_xy(
                                ev.x[kept],
                                ev.y[kept],
                                width=int(args.width),
                                height=int(args.height),
                                chunk_size=30000,
                            )
                        ),
                    )
                _patch_esr_mean_in_roc_csv(roc_csv[env], esr_targets=esr_targets)

            if aocc_mode == "best":
                aocc_targets: dict[str, tuple[int, float]] = {}
                for recipe in recipes:
                    tag, best_i, kept = _scores_and_kept_for_recipe(recipe)
                    if tag in aocc_targets:
                        continue
                    aocc_targets[tag] = (
                        int(best_i),
                        float(
                            aocc_from_xyt(
                                ev.x[kept],
                                ev.y[kept],
                                ev.t[kept],
                                width=int(args.width),
                                height=int(args.height),
                            )
                        ),
                    )
                _patch_aocc_in_roc_csv(roc_csv[env], aocc_targets=aocc_targets)

        if n139_records[env]:
            recs = n139_records[env]
            by_low: dict[float, dict[str, float | int | str]] = {}
            by_high: dict[float, dict[str, float | int | str]] = {}

            for rec in recs:
                lo = float(rec["low"])
                hi = float(rec["high"])
                if (lo not in by_low) or float(rec["auc"]) > float(by_low[lo]["auc"]):
                    by_low[lo] = rec
                if (hi not in by_high) or float(rec["auc"]) > float(by_high[hi]["auc"]):
                    by_high[hi] = rec

            auc_low_csv = os.path.join(out_dir, f"auc2_ebf_n139_{env}_by_low.csv")
            with open(auc_low_csv, "w", newline="", encoding="utf-8") as f:
                w2 = csv.writer(f)
                w2.writerow(["env", "low", "best_auc", "best_f1", "best_s", "best_tau_us", "best_high", "best_tag"])
                for lo in sorted(by_low.keys()):
                    rec = by_low[lo]
                    w2.writerow(
                        [
                            env,
                            float(lo),
                            float(rec["auc"]),
                            float(rec["best_f1"]),
                            int(rec["s"]),
                            int(rec["tau_us"]),
                            float(rec["high"]),
                            str(rec["tag"]),
                        ]
                    )

            auc_high_csv = os.path.join(out_dir, f"auc2_ebf_n139_{env}_by_high.csv")
            with open(auc_high_csv, "w", newline="", encoding="utf-8") as f:
                w2 = csv.writer(f)
                w2.writerow(["env", "high", "best_auc", "best_f1", "best_s", "best_tau_us", "best_low", "best_tag"])
                for hi in sorted(by_high.keys()):
                    rec = by_high[hi]
                    w2.writerow(
                        [
                            env,
                            float(hi),
                            float(rec["auc"]),
                            float(rec["best_f1"]),
                            int(rec["s"]),
                            int(rec["tau_us"]),
                            float(rec["low"]),
                            str(rec["tag"]),
                        ]
                    )

            print(f"saved: {auc_low_csv}")
            print(f"saved: {auc_high_csv}")

    print("=== BEST (by env, AUC) ===")
    for env in ("light", "mid", "heavy"):
        tag, auc = best_auc_by_env[env]
        print(f"{env}: {tag} auc={auc:.6f}")
    print("=== BEST (by env, F1) ===")
    for env in ("light", "mid", "heavy"):
        tag, f1 = best_f1_by_env[env]
        print(f"{env}: {tag} f1={f1:.6f}")

    for env in ("light", "mid", "heavy"):
        print(f"saved: {roc_csv[env]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
