from __future__ import annotations

import argparse
import csv
import importlib.util
import time
from pathlib import Path

import numpy as np
from myevs.metrics.aocc import aocc_from_xyt
from myevs.metrics.esr import event_structural_ratio_mean_from_xy


def _load_sweep_module():
    here = Path(__file__).resolve()
    sweep_path = here.parents[1] / "ED24_alg_evalu" / "sweep_ebf_slim_labelscore_grid.py"
    spec = importlib.util.spec_from_file_location("_sweep_ebf_slim_labelscore_grid", sweep_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"failed to load sweep module spec: {sweep_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _parse_int_list(s: str) -> list[int]:
    return [int(float(x.strip())) for x in str(s).split(",") if x.strip()]


def _parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def _roc_points(labels01: np.ndarray, scores: np.ndarray):
    y = labels01.astype(np.int8, copy=False)
    s = scores.astype(np.float64, copy=False)
    n = int(y.shape[0])
    pos = int(np.sum(y))
    neg = int(n - pos)
    if n <= 0 or pos <= 0 or neg <= 0:
        thr = np.asarray([np.inf, -np.inf], dtype=np.float64)
        tp = np.asarray([0, pos], dtype=np.int64)
        fp = np.asarray([0, neg], dtype=np.int64)
        tpr = np.asarray([0.0, 1.0], dtype=np.float64)
        fpr = np.asarray([0.0, 1.0], dtype=np.float64)
        auc = 0.0
        return thr, tp, fp, tpr, fpr, auc

    order = np.argsort(-s, kind="mergesort")
    ys = y[order]
    ss = s[order]
    tp_cum = np.cumsum(ys > 0, dtype=np.int64)
    fp_cum = np.cumsum(ys == 0, dtype=np.int64)
    change = np.empty((n,), dtype=bool)
    change[:-1] = ss[:-1] != ss[1:]
    change[-1] = True
    idx = np.nonzero(change)[0]

    thr = np.concatenate([np.asarray([np.inf], dtype=np.float64), ss[idx].astype(np.float64, copy=False)])
    tp = np.concatenate([np.asarray([0], dtype=np.int64), tp_cum[idx]])
    fp = np.concatenate([np.asarray([0], dtype=np.int64), fp_cum[idx]])
    tpr = tp.astype(np.float64) / float(pos)
    fpr = fp.astype(np.float64) / float(neg)
    auc = float((getattr(np, "trapezoid", None) or np.trapz)(y=tpr, x=fpr))
    return thr, tp, fp, tpr, fpr, auc


def _best_f1_idx(tp: np.ndarray, fp: np.ndarray, pos: int, neg: int) -> int:
    best_i = 0
    best_key = (-1.0, 0.0, 0.0, -1.0)
    for i in range(int(tp.shape[0])):
        tp_i = int(tp[i])
        fp_i = int(fp[i])
        tpr = (tp_i / pos) if pos > 0 else 0.0
        fpr = (fp_i / neg) if neg > 0 else 0.0
        prec_den = tp_i + fp_i
        precision = (tp_i / prec_den) if prec_den > 0 else 0.0
        f1_den = precision + tpr
        f1 = (2.0 * precision * tpr / f1_den) if f1_den > 0 else 0.0
        key = (float(f1), float(tpr), float(precision), -float(fpr))
        if key > best_key:
            best_key = key
            best_i = i
    return int(best_i)


def _safe_auc(v: object) -> float:
    try:
        return float(v)
    except Exception:
        return -1.0


def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep n179 on one labeled pair and export ROC + bestpoint summary.")
    ap.add_argument("--noisy", required=True, help="labeled noisy npy (must include label field)")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--summary-csv", required=True)
    ap.add_argument("--runtime-csv", required=True)
    ap.add_argument("--dataset", default="unknown")
    ap.add_argument("--scene", default="unknown")
    ap.add_argument("--level", default="unknown")
    ap.add_argument("--variant", default="n180", choices=["n179", "n180", "n181", "n182", "n183", "n184", "n185"])
    ap.add_argument("--width", type=int, default=0, help="0 means auto infer from input x max + 1")
    ap.add_argument("--height", type=int, default=0, help="0 means auto infer from input y max + 1")
    ap.add_argument("--tick-ns", type=float, default=1000.0)
    ap.add_argument("--s-list", default="5,7,9")
    ap.add_argument("--tau-us-list", default="16000,32000,64000,128000,256000")
    ap.add_argument("--k-sfrac-list", default="0.4", help="n179 support-relief coefficient list")
    ap.add_argument("--k-mix-list", default="0.0", help="n179 polarity-mix coefficient list")
    ap.add_argument("--beta-init-list", default="", help="n179 beta_init list; empty means backbone default")
    ap.add_argument("--rhythm-pressure-coeff-list", default="0.0")
    ap.add_argument("--rhythm-good-coeff-list", default="0.75")
    ap.add_argument("--support-good-coeff-list", default="0.0")
    ap.add_argument("--pi-bad-coeff-list", default="0.5")
    ap.add_argument("--pi-r-coeff-list", default="0.5")
    ap.add_argument("--pi-good-coeff-list", default="0.75")
    ap.add_argument("--n181-mode", default="conservative", choices=["conservative", "minimal"])
    ap.add_argument("--n182-pi-lambda", type=float, default=0.25)
    ap.add_argument("--n183-pi-source", default="avg", choices=["bad", "r", "avg", "max", "mix"])
    ap.add_argument("--n184-pi-alpha", type=float, default=0.5)
    ap.add_argument("--n184-pi-alpha-list", default="", help="comma list for n184 pi alpha sweep, e.g. 0,0.1,...,1.5")
    ap.add_argument("--max-events", type=int, default=0)
    ap.add_argument("--tag-prefix", default="n180")
    ap.add_argument(
        "--signal-label-value",
        type=int,
        default=1,
        help="label value meaning signal. default=1",
    )
    ap.add_argument("--roc-max-points", type=int, default=5000)
    ap.add_argument("--mesr-mode", choices=["best", "off"], default="best")
    ap.add_argument("--aocc-mode", choices=["best", "off"], default="best")
    ap.add_argument("--aocc-style", choices=["paper", "normalized"], default="paper")
    ap.add_argument("--chunk-size", type=int, default=30000)
    args = ap.parse_args()
    signal_label_value = int(args.signal_label_value)

    sweep = _load_sweep_module()
    ev = sweep.load_labeled_npy(str(args.noisy), max_events=int(args.max_events))
    tb = sweep.TimeBase(tick_ns=float(args.tick_ns))
    s_list = _parse_int_list(args.s_list)
    tau_list = _parse_int_list(args.tau_us_list)
    k_sfrac_list = _parse_float_list(args.k_sfrac_list)
    k_mix_list = _parse_float_list(args.k_mix_list)
    beta_init_list = _parse_float_list(args.beta_init_list) if str(args.beta_init_list).strip() else [None]
    rhythm_pressure_coeff_list = _parse_float_list(args.rhythm_pressure_coeff_list)
    rhythm_good_coeff_list = _parse_float_list(args.rhythm_good_coeff_list)
    support_good_coeff_list = _parse_float_list(args.support_good_coeff_list)
    pi_bad_coeff_list = _parse_float_list(args.pi_bad_coeff_list)
    pi_r_coeff_list = _parse_float_list(args.pi_r_coeff_list)
    pi_good_coeff_list = _parse_float_list(args.pi_good_coeff_list)
    n184_pi_alpha_list = _parse_float_list(args.n184_pi_alpha_list) if str(args.n184_pi_alpha_list).strip() else [float(args.n184_pi_alpha)]

    out_csv = Path(str(args.out_csv))
    summary_csv = Path(str(args.summary_csv))
    runtime_csv = Path(str(args.runtime_csv))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    runtime_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[list[object]] = []
    summary_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []

    n = int(ev.t.shape[0])
    width = int(args.width) if int(args.width) > 0 else (int(np.max(ev.x)) + 1 if n > 0 else 0)
    height = int(args.height) if int(args.height) > 0 else (int(np.max(ev.y)) + 1 if n > 0 else 0)
    t0 = int(ev.t[0]) if n > 0 else 0
    t1 = int(ev.t[-1]) if n > 0 else 0
    dur_s = max(1e-9, (t1 - t0) * float(args.tick_ns) * 1e-9)
    event_rate = float(n) / dur_s

    for s in s_list:
        radius = int((int(s) - 1) // 2)
        for tau_us in tau_list:
            for k_sfrac in k_sfrac_list:
                for k_mix in k_mix_list:
                    for beta_init in beta_init_list:
                        for rhythm_pressure_coeff in rhythm_pressure_coeff_list:
                            for rhythm_good_coeff in rhythm_good_coeff_list:
                                for support_good_coeff in support_good_coeff_list:
                                    for pi_bad_coeff in pi_bad_coeff_list:
                                        for pi_r_coeff in pi_r_coeff_list:
                                            for pi_good_coeff in pi_good_coeff_list:
                                                for n184_pi_alpha in n184_pi_alpha_list:
                                                    ks_tag = int(round(float(k_sfrac) * 1000.0))
                                                    km_tag = int(round(float(k_mix) * 1000.0))
                                                    rp_tag = int(round(float(rhythm_pressure_coeff) * 1000.0))
                                                    rg_tag = int(round(float(rhythm_good_coeff) * 1000.0))
                                                    sg_tag = int(round(float(support_good_coeff) * 1000.0))
                                                    pb_tag = int(round(float(pi_bad_coeff) * 1000.0))
                                                    pr_tag = int(round(float(pi_r_coeff) * 1000.0))
                                                    pg_tag = int(round(float(pi_good_coeff) * 1000.0))
                                                    if beta_init is None:
                                                        b_tag = "bdef"
                                                    else:
                                                        b_tag = f"b{int(round(float(beta_init) * 1000.0))}"
                                                    tag = (
                                                        f"{args.tag_prefix}_s{s}_r{radius}_tau{tau_us}"
                                                        f"_ks{ks_tag}_km{km_tag}_rp{rp_tag}_rg{rg_tag}_sg{sg_tag}"
                                                        f"_pb{pb_tag}_pr{pr_tag}_pg{pg_tag}_{b_tag}"
                                                    )
                                                    if str(args.variant).lower() == "n181":
                                                        tag = (
                                                            f"{args.tag_prefix}_{str(args.n181_mode)}_s{s}_r{radius}_tau{tau_us}"
                                                            f"_ks{ks_tag}_rg{rg_tag}_{b_tag}"
                                                        )
                                                    if str(args.variant).lower() == "n182":
                                                        lp_tag = int(round(float(args.n182_pi_lambda) * 1000.0))
                                                        tag = (
                                                            f"{args.tag_prefix}_{str(args.n181_mode)}_s{s}_r{radius}_tau{tau_us}"
                                                            f"_ks{ks_tag}_rg{rg_tag}_lp{lp_tag}_{b_tag}"
                                                        )
                                                    if str(args.variant).lower() in {"n184", "n185"}:
                                                        a_tag = int(round(float(n184_pi_alpha) * 1000.0))
                                                        tag = (
                                                            f"{args.tag_prefix}_s{s}_r{radius}_tau{tau_us}"
                                                            f"_ks{ks_tag}_km{km_tag}_rp{rp_tag}_rg{rg_tag}_sg{sg_tag}"
                                                            f"_a{a_tag}_{b_tag}"
                                                        )
                                                    t_start = time.perf_counter()
                                                    scores = sweep.score_stream_ebf(
                                                    ev,
                                                    width=width,
                                                    height=height,
                                                    radius_px=int(radius),
                                                    tau_us=int(tau_us),
                                                    tb=tb,
                                                    variant=str(args.variant),
                                                    n179_beta_init=beta_init,
                                                    n179_k_sfrac=float(k_sfrac),
                                                    n179_k_mix=float(k_mix),
                                                    n179_rhythm_pressure_coeff=float(rhythm_pressure_coeff),
                                                    n179_rhythm_good_coeff=float(rhythm_good_coeff),
                                                    n179_support_good_coeff=float(support_good_coeff),
                                                    n179_pi_bad_coeff=float(pi_bad_coeff),
                                                    n179_pi_r_coeff=float(pi_r_coeff),
                                                    n179_pi_good_coeff=float(pi_good_coeff),
                                                    n181_mode=str(args.n181_mode),
                                                    n182_pi_lambda=float(args.n182_pi_lambda),
                                                    n183_pi_source=str(args.n183_pi_source),
                                                    n184_pi_alpha=float(n184_pi_alpha),
                                                )
                                                elapsed = float(time.perf_counter() - t_start)
                                                eps = (float(n) / elapsed) if elapsed > 0 else 0.0
                                                runtime_rows.append(
                                                    {
                                                        "dataset": str(args.dataset),
                                                        "scene": str(args.scene),
                                                        "level": str(args.level),
                                                        "algorithm": str(args.variant),
                                                        "tag": tag,
                                                        "events": n,
                                                        "duration_s": dur_s,
                                                        "event_rate_eps": event_rate,
                                                        "runtime_sec": elapsed,
                                                        "throughput_eps": eps,
                                                        "realtime_ok": int(eps >= event_rate),
                                                    }
                                                )
        
                                                y_signal = (
                                                    np.asarray(ev.label, dtype=np.int64) == int(signal_label_value)
                                                ).astype(np.int8, copy=False)
                                                thr, tp, fp, tpr, fpr, auc = _roc_points(y_signal, scores)
                                                pos = int(np.sum(y_signal > 0))
                                                neg = int(n - pos)
                                                best_i = _best_f1_idx(tp, fp, pos, neg)

                                                keep_idx: np.ndarray
                                                max_points = int(args.roc_max_points)
                                                if max_points > 0 and int(thr.shape[0]) > max_points:
                                                    keep_idx = np.unique(
                                                        np.linspace(0, int(thr.shape[0]) - 1, num=max_points, dtype=np.int64)
                                                    )
                                                else:
                                                    keep_idx = np.arange(int(thr.shape[0]), dtype=np.int64)

                                                best_mesr = None
                                                best_aocc = None
                                                if int(best_i) >= 0:
                                                    thr_best = float(thr[int(best_i)])
                                                    if np.isinf(thr_best):
                                                        keep_best = np.zeros((scores.shape[0],), dtype=bool)
                                                    else:
                                                        keep_best = np.asarray(scores, dtype=np.float64) >= thr_best
                                                    xk = np.asarray(ev.x[keep_best], dtype=np.int32)
                                                    yk = np.asarray(ev.y[keep_best], dtype=np.int32)
                                                    tk = np.asarray(ev.t[keep_best], dtype=np.int64)
                                                    if args.mesr_mode == "best":
                                                        best_mesr = float(
                                                            event_structural_ratio_mean_from_xy(
                                                                xk,
                                                                yk,
                                                                width=int(width),
                                                                height=int(height),
                                                                chunk_size=int(args.chunk_size),
                                                            )
                                                        )
                                                    if args.aocc_mode == "best":
                                                        t_us = np.round(tk.astype(np.float64) * tb.tick_us).astype(
                                                            np.int64, copy=False
                                                        )
                                                        best_aocc = float(
                                                            aocc_from_xyt(
                                                                xk,
                                                                yk,
                                                                t_us,
                                                                width=int(width),
                                                                height=int(height),
                                                                style=str(args.aocc_style),
                                                            )
                                                        )

                                                for i in keep_idx:
                                                    tp_i = int(tp[i])
                                                    fp_i = int(fp[i])
                                                    tn_i = int(neg - fp_i)
                                                    fn_i = int(pos - tp_i)
                                                    tpr_i = float(tpr[i])
                                                    fpr_i = float(fpr[i])
                                                    prec_den = tp_i + fp_i
                                                    precision = (tp_i / prec_den) if prec_den > 0 else 0.0
                                                    f1_den = precision + tpr_i
                                                    f1 = (2.0 * precision * tpr_i / f1_den) if f1_den > 0 else 0.0
                                                    acc = (tp_i + tn_i) / float(max(1, n))
                                                    rows.append(
                                                        [
                                                            tag,
                                                            str(args.variant),
                                                            "min-neighbors",
                                                            f"{float(thr[i]):.9g}",
                                                            "paper",
                                                            "0",
                                                            n,
                                                            pos,
                                                            neg,
                                                            tp_i + fp_i,
                                                            tp_i,
                                                            fp_i,
                                                            tp_i,
                                                            fp_i,
                                                            tn_i,
                                                            fn_i,
                                                            f"{tpr_i:.9f}",
                                                            f"{fpr_i:.9f}",
                                                            f"{precision:.9f}",
                                                            f"{acc:.9f}",
                                                            f"{f1:.9f}",
                                                            f"{auc:.9f}",
                                                            (
                                                                ""
                                                                if int(i) != int(best_i) or best_mesr is None
                                                                else f"{float(best_mesr):.9f}"
                                                            ),
                                                            (
                                                                ""
                                                                if int(i) != int(best_i) or best_aocc is None
                                                                else f"{float(best_aocc):.9f}"
                                                            ),
                                                        ]
                                                    )

                                                b_tp = int(tp[best_i])
                                                b_fp = int(fp[best_i])
                                                b_tpr = float(tpr[best_i])
                                                b_fpr = float(fpr[best_i])
                                                b_precision = (b_tp / (b_tp + b_fp)) if (b_tp + b_fp) > 0 else 0.0
                                                b_f1_den = b_precision + b_tpr
                                                b_f1 = (2.0 * b_precision * b_tpr / b_f1_den) if b_f1_den > 0 else 0.0
                                                summary_rows.append(
                                                    {
                                                        "dataset": str(args.dataset),
                                                        "scene": str(args.scene),
                                                        "level": str(args.level),
                                                        "algorithm": str(args.variant),
                                                        "tag": tag,
                                                        "s": int(s),
                                                        "radius_px": int(radius),
                                                        "tau_us": int(tau_us),
                                                        "k_sfrac": float(k_sfrac),
                                                        "k_mix": float(k_mix),
                                                        "beta_init": (None if beta_init is None else float(beta_init)),
                                                        "rhythm_pressure_coeff": float(rhythm_pressure_coeff),
                                                        "rhythm_good_coeff": float(rhythm_good_coeff),
                                                        "support_good_coeff": float(support_good_coeff),
                                                        "pi_bad_coeff": float(pi_bad_coeff),
                                                        "pi_r_coeff": float(pi_r_coeff),
                                                        "pi_good_coeff": float(pi_good_coeff),
                                                        "auc": float(auc),
                                                        "best_f1": float(b_f1),
                                                        "best_f1_threshold": float(thr[best_i]),
                                                        "best_f1_tpr": float(b_tpr),
                                                        "best_f1_fpr": float(b_fpr),
                                                        "best_f1_precision": float(b_precision),
                                                        "best_f1_mesr": (None if best_mesr is None else float(best_mesr)),
                                                        "best_f1_aocc": (None if best_aocc is None else float(best_aocc)),
                                                    }
                                                )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "tag",
                "method",
                "param",
                "value",
                "roc_convention",
                "match_us",
                "events_total",
                "signal_total",
                "noise_total",
                "events_kept",
                "signal_kept",
                "noise_kept",
                "tp",
                "fp",
                "tn",
                "fn",
                "tpr",
                "fpr",
                "precision",
                "accuracy",
                "f1",
                "auc",
                "esr_mean",
                "aocc",
            ]
        )
        w.writerows(rows)

    if summary_rows:
        summary_rows.sort(key=lambda r: (_safe_auc(r.get("auc")), _safe_auc(r.get("best_f1"))), reverse=True)
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
        if summary_rows:
            w.writeheader()
            w.writerows(summary_rows)

    with runtime_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(runtime_rows[0].keys()) if runtime_rows else [])
        if runtime_rows:
            w.writeheader()
            w.writerows(runtime_rows)

    print(f"saved roc: {out_csv}")
    print(f"saved summary: {summary_csv}")
    print(f"saved runtime: {runtime_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
