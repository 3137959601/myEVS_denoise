from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass

import numpy as np

from myevs.denoise.numba_ebf import ebf_scores_stream_numba, ebf_state_init
from myevs.metrics.aocc import aocc_from_xyt
from myevs.metrics.esr import event_structural_ratio_mean_from_xy
from myevs.timebase import TimeBase


@dataclass(frozen=True)
class LabeledEvents:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    p: np.ndarray
    label: np.ndarray


def _parse_int_list(s: str) -> list[int]:
    items = [x.strip() for x in str(s).split(",") if x.strip()]
    out: list[int] = []
    for it in items:
        out.append(int(float(it)))
    return out


def _parse_float_list(s: str) -> list[float]:
    items = [x.strip() for x in str(s).split(",") if x.strip()]
    out: list[float] = []
    for it in items:
        out.append(float(it))
    return out


def _float_tag(v: float) -> str:
    txt = f"{float(v):.6g}"
    txt = txt.replace("-", "m").replace(".", "p")
    return txt


def _join_ints(values: list[int], *, sep: str = "_") -> str:
    return sep.join(str(int(v)) for v in values)


def _tau_list_tag(tau_us_list: list[int]) -> str:
    tau_us_list = [int(v) for v in tau_us_list]
    if all((v % 1000) == 0 for v in tau_us_list):
        tau_ms = [v // 1000 for v in tau_us_list]
        return f"tau{_join_ints(tau_ms)}ms"
    return f"tau{_join_ints(tau_us_list)}us"


def load_labeled_npy(path: str, *, max_events: int = 0) -> LabeledEvents:
    arr = np.load(path, mmap_mode="r", allow_pickle=True)
    if max_events > 0:
        arr = arr[:max_events]

    if getattr(arr, "dtype", None) is not None and getattr(arr.dtype, "names", None):
        names = set(arr.dtype.names)
        need = {"t", "x", "y", "p", "label"}
        if not need.issubset(names):
            missing = sorted(need - names)
            raise SystemExit(f"input structured npy missing fields: {missing}")
        t = arr["t"].astype(np.uint64, copy=False)
        x = arr["x"].astype(np.int32, copy=False)
        y = arr["y"].astype(np.int32, copy=False)
        p = arr["p"].astype(np.int8, copy=False)
        label = arr["label"].astype(np.int8, copy=False)
    else:
        a2 = np.asarray(arr)
        if a2.ndim != 2 or a2.shape[1] < 5:
            raise SystemExit("input must be structured (t/x/y/p/label) or 2D array with >=5 columns")

        c0 = a2[: min(10000, a2.shape[0]), 0]
        is_bin0 = bool(np.all((c0 == 0) | (c0 == 1)))
        if is_bin0:
            label = a2[:, 0].astype(np.int8, copy=False)
            t = a2[:, 1].astype(np.uint64, copy=False)
            y = a2[:, 2].astype(np.int32, copy=False)
            x = a2[:, 3].astype(np.int32, copy=False)
            p = a2[:, 4].astype(np.int8, copy=False)
        else:
            t = a2[:, 0].astype(np.uint64, copy=False)
            x = a2[:, 1].astype(np.int32, copy=False)
            y = a2[:, 2].astype(np.int32, copy=False)
            p = a2[:, 3].astype(np.int8, copy=False)
            label = a2[:, 4].astype(np.int8, copy=False)

    label = (label > 0).astype(np.int8, copy=False)

    return LabeledEvents(
        t=np.ascontiguousarray(t),
        x=np.ascontiguousarray(x),
        y=np.ascontiguousarray(y),
        p=np.ascontiguousarray(p),
        label=np.ascontiguousarray(label),
    )


ROC_HEADER = [
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


def _roc_points_from_scores(
    y_true01: np.ndarray,
    y_score: np.ndarray,
    *,
    max_points: int,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y_true01).astype(np.int8, copy=False)
    s = np.asarray(y_score).astype(np.float64, copy=False)
    if y.ndim != 1 or s.ndim != 1 or y.shape[0] != s.shape[0]:
        raise ValueError("y_true and y_score must be 1D arrays of the same length")

    n = int(y.shape[0])
    pos = int(np.sum(y))
    neg = int(n - pos)
    if n == 0 or pos == 0 or neg == 0:
        thr = np.asarray([np.inf, -np.inf], dtype=np.float64)
        tp = np.asarray([0, pos], dtype=np.int64)
        fp = np.asarray([0, neg], dtype=np.int64)
        fpr = np.asarray([0.0, 1.0], dtype=np.float64)
        tpr = np.asarray([0.0, 1.0], dtype=np.float64)
        auc = 0.0
        return auc, thr, tp, fp, fpr, tpr

    order = np.argsort(-s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order]

    tp_cum = np.cumsum(y_sorted, dtype=np.int64)
    fp_cum = np.cumsum(1 - y_sorted, dtype=np.int64)

    change = np.empty((n,), dtype=bool)
    change[:-1] = s_sorted[:-1] != s_sorted[1:]
    change[-1] = True
    idx = np.nonzero(change)[0]

    tp_u = tp_cum[idx]
    fp_u = fp_cum[idx]
    thr_u = s_sorted[idx].astype(np.float64, copy=False)

    tp_u = np.concatenate([np.asarray([0], dtype=np.int64), tp_u])
    fp_u = np.concatenate([np.asarray([0], dtype=np.int64), fp_u])
    thr_u = np.concatenate([np.asarray([np.inf], dtype=np.float64), thr_u])

    tpr_u = tp_u.astype(np.float64) / float(pos)
    fpr_u = fp_u.astype(np.float64) / float(neg)

    try:
        auc = float(np.trapezoid(tpr_u, fpr_u))
    except Exception:
        auc = float(np.trapz(tpr_u, fpr_u))

    m = int(max_points)
    if m > 0 and int(thr_u.shape[0]) > m:
        keep = np.unique(
            np.concatenate(
                [
                    np.linspace(0, thr_u.shape[0] - 1, num=m, dtype=np.int64),
                ]
            )
        )
        thr_u = thr_u[keep]
        tp_u = tp_u[keep]
        fp_u = fp_u[keep]
        fpr_u = fpr_u[keep]
        tpr_u = tpr_u[keep]

    return auc, thr_u, tp_u, fp_u, fpr_u, tpr_u


def _best_f1_index(
    thresholds: np.ndarray,
    tp: np.ndarray,
    fp: np.ndarray,
    *,
    pos: int,
    neg: int,
) -> int:
    """Pick threshold index that maximizes F1.

    Tie-breakers: higher TPR, then higher precision, then lower FPR.
    """

    best_i = 0
    best_key = (-1.0, 0.0, 0.0, -1.0)
    p = int(pos)
    n = int(neg)

    for i in range(int(thresholds.shape[0])):
        tp_i = int(tp[i])
        fp_i = int(fp[i])
        tpr = (tp_i / p) if p > 0 else 0.0
        fpr = (fp_i / n) if n > 0 else 0.0
        prec_den = tp_i + fp_i
        precision = (tp_i / prec_den) if prec_den > 0 else 0.0
        f1_den = precision + tpr
        f1 = (2.0 * precision * tpr / f1_den) if f1_den > 0 else 0.0
        key = (float(f1), float(tpr), float(precision), -float(fpr))
        if key > best_key:
            best_key = key
            best_i = int(i)
    return int(best_i)


def _write_roc_rows(
    writer: csv.writer,
    *,
    tag: str,
    thresholds: np.ndarray,
    tp: np.ndarray,
    fp: np.ndarray,
    pos: int,
    neg: int,
    auc: float,
    esr_mean: float | None,
    esr_at_index: int,
    aocc: float | None,
    aocc_at_index: int,
) -> None:
    n = int(pos + neg)
    for i in range(int(thresholds.shape[0])):
        thr = float(thresholds[i])
        tp_i = int(tp[i])
        fp_i = int(fp[i])
        tn_i = int(neg - fp_i)
        fn_i = int(pos - tp_i)

        events_kept = tp_i + fp_i
        signal_kept = tp_i
        noise_kept = fp_i

        tpr = (tp_i / pos) if pos > 0 else 0.0
        fpr = (fp_i / neg) if neg > 0 else 0.0

        prec_den = tp_i + fp_i
        precision = (tp_i / prec_den) if prec_den > 0 else 0.0
        acc = ((tp_i + tn_i) / n) if n > 0 else 0.0
        f1_den = precision + tpr
        f1 = (2.0 * precision * tpr / f1_den) if f1_den > 0 else 0.0

        writer.writerow(
            [
                tag,
                "ebf",
                "min-neighbors",
                thr,
                "paper",
                0,
                n,
                int(pos),
                int(neg),
                int(events_kept),
                int(signal_kept),
                int(noise_kept),
                int(tp_i),
                int(fp_i),
                int(tn_i),
                int(fn_i),
                float(tpr),
                float(fpr),
                float(precision),
                float(acc),
                float(f1),
                float(auc),
                ("" if int(i) != int(esr_at_index) or esr_mean is None else float(esr_mean)),
                ("" if int(i) != int(aocc_at_index) or aocc is None else float(aocc)),
            ]
        )


def _patch_esr_mean_in_roc_csv(
    csv_path: str,
    *,
    esr_targets: dict[str, tuple[int, float]],
) -> None:
    """Patch esr_mean values in an existing ROC CSV.

    esr_targets maps: tag -> (esr_at_index_within_tag_rows, esr_mean_value)
    """

    if not esr_targets:
        return

    tmp_path = csv_path + ".tmp"
    tag_row_i: dict[str, int] = {}

    with open(csv_path, "r", newline="", encoding="utf-8") as fin, open(
        tmp_path, "w", newline="", encoding="utf-8"
    ) as fout:
        r = csv.DictReader(fin)
        if r.fieldnames is None:
            return
        fieldnames = list(r.fieldnames)
        w = csv.DictWriter(fout, fieldnames=fieldnames)
        w.writeheader()

        for row in r:
            tag = (row.get("tag") or "").strip()
            if not tag:
                w.writerow(row)
                continue

            i = int(tag_row_i.get(tag, 0))
            tag_row_i[tag] = i + 1

            tgt = esr_targets.get(tag)
            if tgt is not None:
                esr_i, esr_v = tgt
                if int(i) == int(esr_i):
                    row["esr_mean"] = str(float(esr_v))
            w.writerow(row)

    os.replace(tmp_path, csv_path)


def _patch_aocc_in_roc_csv(
    csv_path: str,
    *,
    aocc_targets: dict[str, tuple[int, float]],
) -> None:
    """Patch aocc values in an existing ROC CSV.

    aocc_targets maps: tag -> (aocc_at_index_within_tag_rows, aocc_value)
    """

    if not aocc_targets:
        return

    tmp_path = csv_path + ".tmp"
    tag_row_i: dict[str, int] = {}

    with open(csv_path, "r", newline="", encoding="utf-8") as fin, open(
        tmp_path, "w", newline="", encoding="utf-8"
    ) as fout:
        r = csv.DictReader(fin)
        if r.fieldnames is None:
            return
        fieldnames = list(r.fieldnames)
        w = csv.DictWriter(fout, fieldnames=fieldnames)
        w.writeheader()

        for row in r:
            tag = (row.get("tag") or "").strip()
            if not tag:
                w.writerow(row)
                continue

            i = int(tag_row_i.get(tag, 0))
            tag_row_i[tag] = i + 1

            tgt = aocc_targets.get(tag)
            if tgt is not None:
                aocc_i, aocc_v = tgt
                if int(i) == int(aocc_i):
                    row["aocc"] = str(float(aocc_v))
            w.writerow(row)

    os.replace(tmp_path, csv_path)


def _score_stream(
    ev: LabeledEvents,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_us: int,
    tb: TimeBase,
    variant: str,
    kernel_cache: dict[str, object],
    env_name: str | None = None,
    n139_low: float | None = None,
    n139_high: float | None = None,
) -> np.ndarray:
    v = str(variant).strip().lower()
    scores = np.empty((ev.t.shape[0],), dtype=np.float32)
    tau_ticks = int(tb.us_to_ticks(int(tau_us)))

    if v in {"ebf", "baseline"}:
        last_ts, last_pol = ebf_state_init(int(width), int(height))
        ebf_scores_stream_numba(
            t=ev.t,
            x=ev.x,
            y=ev.y,
            p=ev.p,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_ticks=int(tau_ticks),
            last_ts=last_ts,
            last_pol=last_pol,
            scores_out=scores,
        )
        return scores

    if v in {"s80", "ebf_s80", "ebfs80"}:
        from myevs.denoise.ops.ebfopt_part2.s80_baseline_aocclite_gate import score_stream_s80

        return score_stream_s80(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_ticks=int(tau_ticks),
            kernel_cache=kernel_cache,
            scores_out=scores,
        )

    if v in {"s81", "ebf_s81", "ebfs81"}:
        from myevs.denoise.ops.ebfopt_part2.s81_block_controller_aocclite import score_stream_s81

        return score_stream_s81(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )

    if v in {"s82", "ebf_s82", "ebfs82"}:
        from myevs.denoise.ops.ebfopt_part2.s82_spatial_block_stability_controller import score_stream_s82

        return score_stream_s82(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )

    if v in {"s52", "ebf_s52", "ebfs52"}:
        from myevs.denoise.ops.ebfopt_part2.s52_ebf_labelscore_selfocc_supportboost_autobeta_mixgateopp_div_u2 import (
            try_build_s52_ebf_labelscore_selfocc_supportboost_autobeta_mixgateopp_div_u2_scores_kernel,
        )

        ker = kernel_cache.get("ker_s52")
        if ker is None:
            ker = try_build_s52_ebf_labelscore_selfocc_supportboost_autobeta_mixgateopp_div_u2_scores_kernel()
            kernel_cache["ker_s52"] = ker
        if ker is None:
            raise SystemExit("s52 requires numba, but kernel build failed")

        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        hot_state = np.zeros((int(width) * int(height),), dtype=np.int32)
        beta_state = np.zeros((1,), dtype=np.float32)
        mix_state = np.zeros((1,), dtype=np.float32)
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            last_ts,
            last_pol,
            hot_state,
            beta_state,
            mix_state,
            scores,
        )
        return scores

    if v in {"s83", "ebf_s83", "ebfs83"}:
        from myevs.denoise.ops.ebfopt_part2.s83_s52_hotless_localmix_proxy import score_stream_s83

        return score_stream_s83(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )

    if v in {"s84", "ebf_s84", "ebfs84"}:
        from myevs.denoise.ops.ebfopt_part2.s84_s52_hotless_ema_proxy import score_stream_s84

        return score_stream_s84(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )

    if v in {"s85", "ebf_s85", "ebfs85"}:
        from myevs.denoise.ops.ebfopt_part2.s85_s52_hotstate_u8_quantized import score_stream_s85

        return score_stream_s85(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )

    if v in {"s86", "ebf_s86", "ebfs86"}:
        from myevs.denoise.ops.ebfopt_part2.s86_s52_sparsehot_u8_cache import score_stream_s86

        return score_stream_s86(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )

    if v in {"s87", "ebf_s87", "ebfs87"}:
        from myevs.denoise.ops.ebfopt_part2.s87_s52_blockwise_mixstate import score_stream_s87

        return score_stream_s87(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )

    if v in {"n1", "ebf_n1", "ebfn1"}:
        from myevs.denoise.ops.ebfopt_part2.n1_essm_block_state_machine import score_stream_n1

        return score_stream_n1(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )

    if v in {"n2", "ebf_n2", "ebfn2"}:
        from myevs.denoise.ops.ebfopt_part2.n2_dynamic_lateral_inhibition import score_stream_n2

        return score_stream_n2(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )

    if v in {"n3", "ebf_n3", "ebfn3"}:
        from myevs.denoise.ops.ebfopt_part2.n3_bayesian_rate_likelihood import score_stream_n3

        return score_stream_n3(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )

    if v in {"n4", "ebf_n4", "ebfn4"}:
        from myevs.denoise.ops.ebfopt_part2.n4_local_momentum_consistency import score_stream_n4

        return score_stream_n4(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )

    if v in {"n5", "ebf_n5", "ebfn5"}:
        from myevs.denoise.ops.ebfopt_part2.n5_dual_timescale_rate_ratio import score_stream_n5

        return score_stream_n5(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )

    if v in {"n6", "ebf_n6", "ebfn6"}:
        from myevs.denoise.ops.ebfopt_part2.n6_inhib_momentum_escape import score_stream_n6

        return score_stream_n6(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )

    if v in {"n7", "ebf_n7", "ebfn7"}:
        from myevs.denoise.ops.ebfopt_part2.n7_dual_field_selfinhib_footprint import score_stream_n7

        return score_stream_n7(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )

    if v in {"n71", "n7.1", "ebf_n71", "ebfn71"}:
        from myevs.denoise.ops.ebfopt_part2.n71_dual_timescale_footprint import score_stream_n71

        return score_stream_n71(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )

    if v in {"n72", "n7.2", "ebf_n72", "ebfn72"}:
        from myevs.denoise.ops.ebfopt_part2.n72_dual_timescale_global_burst import score_stream_n72

        return score_stream_n72(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )

    if v in {"n8", "ebf_n8", "ebfn8"}:
        from myevs.denoise.ops.ebfopt_part2.n8_causal_traj_dualfield import score_stream_n8

        return score_stream_n8(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )

    if v in {"n81", "n8.1", "ebf_n81", "ebfn81"}:
        from myevs.denoise.ops.ebfopt_part2.n81_causal_gate_dualfield import score_stream_n81

        return score_stream_n81(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )

    if v in {"n82", "n8.2", "ebf_n82", "ebfn82"}:
        from myevs.denoise.ops.ebfopt_part2.n82_causal_softgate_dualfield import score_stream_n82

        return score_stream_n82(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )

    if v in {"n83", "n8.3", "ebf_n83", "ebfn83"}:
        from myevs.denoise.ops.ebfopt_part2.n83_causal_adaptive_confidence import score_stream_n83

        return score_stream_n83(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )

    if v in {"n84", "ebf_n84", "ebfn84"}:
        from myevs.denoise.ops.ebfopt_part2.n84_event_chain_state_model import score_stream_n84

        return score_stream_n84(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )

    if v in {"n85", "ebf_n85", "ebfn85"}:
        from myevs.denoise.ops.ebfopt_part2.n85_event_chain_simplified import score_stream_n85

        return score_stream_n85(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )

    if v in {"n86", "ebf_n86", "ebfn86"}:
        from myevs.denoise.ops.ebfopt_part2.n86_event_chain_confirmed import score_stream_n86

        return score_stream_n86(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )

    if v in {"n87", "ebf_n87", "ebfn87"}:
        from myevs.denoise.ops.ebfopt_part2.n87_event_chain_confirmed_soft import score_stream_n87

        return score_stream_n87(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )

    if v in {"n88", "ebf_n88", "ebfn88"}:
        from myevs.denoise.ops.ebfopt_part2.n88_event_chain_minimal import score_stream_n88

        return score_stream_n88(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )

    if v in {"n89", "ebf_n89", "ebfn89"}:
        from myevs.denoise.ops.ebfopt_part2.n89_single_threshold_blockhot import score_stream_n89

        return score_stream_n89(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )

    if v in {"n90", "ebf_n90", "ebfn90"}:
        from myevs.denoise.ops.ebfopt_part2.n90_baseline_nb2_causal import score_stream_n90

        return score_stream_n90(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )

    if v in {"n91", "ebf_n91", "ebfn91"}:
        from myevs.denoise.ops.ebfopt_part2.n91_pixel_rhythm_baseline import score_stream_n91

        return score_stream_n91(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )

    if v in {"n92", "ebf_n92", "ebfn92"}:
        from myevs.denoise.ops.ebfopt_part2.n92_local_temporal_spatial_continuity import score_stream_n92

        return score_stream_n92(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )

    if v in {"n93", "ebf_n93", "ebfn93"}:
        from myevs.denoise.ops.ebfopt_part2.n93_spatiotemporal_weighted_support import score_stream_n93

        return score_stream_n93(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )

    if v in {"n94", "ebf_n94", "ebfn94"}:
        from myevs.denoise.ops.ebfopt_part2.n94_spatiotemporal_linear_support import score_stream_n94

        return score_stream_n94(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )

    if v in {"n95", "ebf_n95", "ebfn95"}:
        from myevs.denoise.ops.ebfopt_part2.n95_spatiotemporal_min_support import score_stream_n95

        return score_stream_n95(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )

    if v in {"s55", "ebf_s55", "ebfs55"}:
        from myevs.denoise.ops.ebfopt_part2.s55_ebf_labelscore_selfocc_supportboost_autobeta_eventmixgateopp_supportlerp_div_u2 import (
            try_build_s55_ebf_labelscore_selfocc_supportboost_autobeta_eventmixgateopp_supportlerp_div_u2_scores_kernel,
        )

        ker = kernel_cache.get("ker_s55")
        if ker is None:
            ker = try_build_s55_ebf_labelscore_selfocc_supportboost_autobeta_eventmixgateopp_supportlerp_div_u2_scores_kernel()
            kernel_cache["ker_s55"] = ker
        if ker is None:
            raise SystemExit("s55 requires numba, but kernel build failed")

        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        hot_state = np.zeros((int(width) * int(height),), dtype=np.int32)
        beta_state = np.zeros((1,), dtype=np.float32)
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            last_ts,
            last_pol,
            hot_state,
            beta_state,
            scores,
        )
        return scores

    if v in {"s60", "ebf_s60", "ebfs60"}:
        from myevs.denoise.ops.ebfopt_part2.s60_ebf_labelscore_dualtau_delta_selfocc_supportboost_autobeta_mixgateopp_div_u2 import (
            try_build_s60_ebf_labelscore_dualtau_delta_selfocc_supportboost_autobeta_mixgateopp_div_u2_scores_kernel,
        )

        ker = kernel_cache.get("ker_s60")
        if ker is None:
            ker = try_build_s60_ebf_labelscore_dualtau_delta_selfocc_supportboost_autobeta_mixgateopp_div_u2_scores_kernel()
            kernel_cache["ker_s60"] = ker
        if ker is None:
            raise SystemExit("s60 requires numba, but kernel build failed")

        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        hot_state = np.zeros((int(width) * int(height),), dtype=np.int32)
        beta_state = np.zeros((1,), dtype=np.float32)
        mix_state = np.zeros((1,), dtype=np.float32)
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            last_ts,
            last_pol,
            hot_state,
            beta_state,
            mix_state,
            scores,
        )
        return scores

    if v in {"s61", "ebf_s61", "ebfs61"}:
        from myevs.denoise.ops.ebfopt_part2.s61_ebf_labelscore_dualtau_condlong_selfocc_mixgateopp_div_u2 import (
            try_build_s61_ebf_labelscore_dualtau_condlong_selfocc_mixgateopp_div_u2_scores_kernel,
        )

        ker = kernel_cache.get("ker_s61")
        if ker is None:
            ker = try_build_s61_ebf_labelscore_dualtau_condlong_selfocc_mixgateopp_div_u2_scores_kernel()
            kernel_cache["ker_s61"] = ker
        if ker is None:
            raise SystemExit("s61 requires numba, but kernel build failed")

        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        hot_state = np.zeros((int(width) * int(height),), dtype=np.int32)
        beta_state = np.zeros((1,), dtype=np.float32)
        mix_state = np.zeros((1,), dtype=np.float32)
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            last_ts,
            last_pol,
            hot_state,
            beta_state,
            mix_state,
            scores,
        )
        return scores

    if v in {"s62", "ebf_s62", "ebfs62"}:
        from myevs.denoise.ops.ebfopt_part2.s62_ebf_labelscore_dualtau_condlong_selfocc_supportgate_mixgateopp_div_u2 import (
            try_build_s62_ebf_labelscore_dualtau_condlong_selfocc_supportgate_mixgateopp_div_u2_scores_kernel,
        )

        ker = kernel_cache.get("ker_s62")
        if ker is None:
            ker = try_build_s62_ebf_labelscore_dualtau_condlong_selfocc_supportgate_mixgateopp_div_u2_scores_kernel()
            kernel_cache["ker_s62"] = ker
        if ker is None:
            raise SystemExit("s62 requires numba, but kernel build failed")

        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        hot_state = np.zeros((int(width) * int(height),), dtype=np.int32)
        beta_state = np.zeros((1,), dtype=np.float32)
        mix_state = np.zeros((1,), dtype=np.float32)
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            last_ts,
            last_pol,
            hot_state,
            beta_state,
            mix_state,
            scores,
        )
        return scores

    if v in {"s63", "ebf_s63", "ebfs63"}:
        from myevs.denoise.ops.ebfopt_part2.s63_ebf_labelscore_dualtau_condlong_selfocc_trajgate_mixgateopp_div_u2 import (
            try_build_s63_ebf_labelscore_dualtau_condlong_selfocc_trajgate_mixgateopp_div_u2_scores_kernel,
        )

        ker = kernel_cache.get("ker_s63")
        if ker is None:
            ker = try_build_s63_ebf_labelscore_dualtau_condlong_selfocc_trajgate_mixgateopp_div_u2_scores_kernel()
            kernel_cache["ker_s63"] = ker
        if ker is None:
            raise SystemExit("s63 requires numba, but kernel build failed")

        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        hot_state = np.zeros((int(width) * int(height),), dtype=np.int32)
        beta_state = np.zeros((1,), dtype=np.float32)
        mix_state = np.zeros((1,), dtype=np.float32)
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            last_ts,
            last_pol,
            hot_state,
            beta_state,
            mix_state,
            scores,
        )
        return scores

    if v in {"s64", "ebf_s64", "ebfs64"}:
        from myevs.denoise.ops.ebfopt_part2.s64_ebf_labelscore_dualtau1p5_condlong_selfocc_trajgate_mixgateopp_div_u2 import (
            try_build_s64_ebf_labelscore_dualtau1p5_condlong_selfocc_trajgate_mixgateopp_div_u2_scores_kernel,
        )

        ker = kernel_cache.get("ker_s64")
        if ker is None:
            ker = try_build_s64_ebf_labelscore_dualtau1p5_condlong_selfocc_trajgate_mixgateopp_div_u2_scores_kernel()
            kernel_cache["ker_s64"] = ker
        if ker is None:
            raise SystemExit("s64 requires numba, but kernel build failed")

        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        hot_state = np.zeros((int(width) * int(height),), dtype=np.int32)
        beta_state = np.zeros((1,), dtype=np.float32)
        mix_state = np.zeros((1,), dtype=np.float32)
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            last_ts,
            last_pol,
            hot_state,
            beta_state,
            mix_state,
            scores,
        )
        return scores

    if v in {"s65", "ebf_s65", "ebfs65"}:
        from myevs.denoise.ops.ebfopt_part2.s65_ebf_labelscore_dualtau_condlong_selfocc_dirgate_mixgateopp_div_u2 import (
            try_build_s65_ebf_labelscore_dualtau_condlong_selfocc_dirgate_mixgateopp_div_u2_scores_kernel,
        )

        ker = kernel_cache.get("ker_s65")
        if ker is None:
            ker = try_build_s65_ebf_labelscore_dualtau_condlong_selfocc_dirgate_mixgateopp_div_u2_scores_kernel()
            kernel_cache["ker_s65"] = ker
        if ker is None:
            raise SystemExit("s65 requires numba, but kernel build failed")

        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        hot_state = np.zeros((int(width) * int(height),), dtype=np.int32)
        beta_state = np.zeros((1,), dtype=np.float32)
        mix_state = np.zeros((1,), dtype=np.float32)
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            last_ts,
            last_pol,
            hot_state,
            beta_state,
            mix_state,
            scores,
        )
        return scores

    if v in {"s66", "ebf_s66", "ebfs66"}:
        from myevs.denoise.ops.ebfopt_part2.s66_ebf_labelscore_dualtau_condlong_selfocc_distgate_mixgateopp_div_u2 import (
            try_build_s66_ebf_labelscore_dualtau_condlong_selfocc_distgate_mixgateopp_div_u2_scores_kernel,
        )

        ker = kernel_cache.get("ker_s66")
        if ker is None:
            ker = try_build_s66_ebf_labelscore_dualtau_condlong_selfocc_distgate_mixgateopp_div_u2_scores_kernel()
            kernel_cache["ker_s66"] = ker
        if ker is None:
            raise SystemExit("s66 requires numba, but kernel build failed")

        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        hot_state = np.zeros((int(width) * int(height),), dtype=np.int32)
        beta_state = np.zeros((1,), dtype=np.float32)
        mix_state = np.zeros((1,), dtype=np.float32)
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            last_ts,
            last_pol,
            hot_state,
            beta_state,
            mix_state,
            scores,
        )
        return scores

    if v in {"s67", "ebf_s67", "ebfs67"}:
        from myevs.denoise.ops.ebfopt_part2.s67_ebf_labelscore_dualtau_condlong_selfocc_deltar2gate_mixgateopp_div_u2 import (
            try_build_s67_ebf_labelscore_dualtau_condlong_selfocc_deltar2gate_mixgateopp_div_u2_scores_kernel,
        )

        ker = kernel_cache.get("ker_s67")
        if ker is None:
            ker = try_build_s67_ebf_labelscore_dualtau_condlong_selfocc_deltar2gate_mixgateopp_div_u2_scores_kernel()
            kernel_cache["ker_s67"] = ker
        if ker is None:
            raise SystemExit("s67 requires numba, but kernel build failed")

        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        hot_state = np.zeros((int(width) * int(height),), dtype=np.int32)
        beta_state = np.zeros((1,), dtype=np.float32)
        mix_state = np.zeros((1,), dtype=np.float32)
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            last_ts,
            last_pol,
            hot_state,
            beta_state,
            mix_state,
            scores,
        )
        return scores

    if v in {"s68", "ebf_s68", "ebfs68"}:
        from myevs.denoise.ops.ebfopt_part2.s68_ebf_labelscore_dualtau_condlong_selfocc_deltacentroidgate_mixgateopp_div_u2 import (
            try_build_s68_ebf_labelscore_dualtau_condlong_selfocc_deltacentroidgate_mixgateopp_div_u2_scores_kernel,
        )

        ker = kernel_cache.get("ker_s68")
        if ker is None:
            ker = try_build_s68_ebf_labelscore_dualtau_condlong_selfocc_deltacentroidgate_mixgateopp_div_u2_scores_kernel()
            kernel_cache["ker_s68"] = ker
        if ker is None:
            raise SystemExit("s68 requires numba, but kernel build failed")

        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        hot_state = np.zeros((int(width) * int(height),), dtype=np.int32)
        beta_state = np.zeros((1,), dtype=np.float32)
        mix_state = np.zeros((1,), dtype=np.float32)
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            last_ts,
            last_pol,
            hot_state,
            beta_state,
            mix_state,
            scores,
        )
        return scores

    if v in {"s69", "ebf_s69", "ebfs69"}:
        from myevs.denoise.ops.ebfopt_part2.s69_ebf_labelscore_dualtau_condlong_selfocc_trajgate_mixgateopp_deltacap2_div_u2 import (
            try_build_s69_ebf_labelscore_dualtau_condlong_selfocc_trajgate_mixgateopp_deltacap2_div_u2_scores_kernel,
        )

        ker = kernel_cache.get("ker_s69")
        if ker is None:
            ker = try_build_s69_ebf_labelscore_dualtau_condlong_selfocc_trajgate_mixgateopp_deltacap2_div_u2_scores_kernel()
            kernel_cache["ker_s69"] = ker
        if ker is None:
            raise SystemExit("s69 requires numba, but kernel build failed")

        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        hot_state = np.zeros((int(width) * int(height),), dtype=np.int32)
        beta_state = np.zeros((1,), dtype=np.float32)
        mix_state = np.zeros((1,), dtype=np.float32)
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            last_ts,
            last_pol,
            hot_state,
            beta_state,
            mix_state,
            scores,
        )
        return scores

    if v in {"s70", "ebf_s70", "ebfs70"}:
        from myevs.denoise.ops.ebfopt_part2.s70_ebf_labelscore_dualtau_condlong_selfocc_trajgate_mixgateopp_deltacap2_puritygate_div_u2 import (
            try_build_s70_ebf_labelscore_dualtau_condlong_selfocc_trajgate_mixgateopp_deltacap2_puritygate_div_u2_scores_kernel,
        )

        ker = kernel_cache.get("ker_s70")
        if ker is None:
            ker = try_build_s70_ebf_labelscore_dualtau_condlong_selfocc_trajgate_mixgateopp_deltacap2_puritygate_div_u2_scores_kernel()
            kernel_cache["ker_s70"] = ker
        if ker is None:
            raise SystemExit("s70 requires numba, but kernel build failed")

        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        hot_state = np.zeros((int(width) * int(height),), dtype=np.int32)
        beta_state = np.zeros((1,), dtype=np.float32)
        mix_state = np.zeros((1,), dtype=np.float32)
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            last_ts,
            last_pol,
            hot_state,
            beta_state,
            mix_state,
            scores,
        )
        return scores

    if v in {"s71", "ebf_s71", "ebfs71"}:
        from myevs.denoise.ops.ebfopt_part2.s71_ebf_labelscore_dualtau_condlong_selfocc_trajgate_mixgateopp_deltacap2_softpuritygate_div_u2 import (
            try_build_s71_ebf_labelscore_dualtau_condlong_selfocc_trajgate_mixgateopp_deltacap2_softpuritygate_div_u2_scores_kernel,
        )

        ker = kernel_cache.get("ker_s71")
        if ker is None:
            ker = try_build_s71_ebf_labelscore_dualtau_condlong_selfocc_trajgate_mixgateopp_deltacap2_softpuritygate_div_u2_scores_kernel()
            kernel_cache["ker_s71"] = ker
        if ker is None:
            raise SystemExit("s71 requires numba, but kernel build failed")

        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        hot_state = np.zeros((int(width) * int(height),), dtype=np.int32)
        beta_state = np.zeros((1,), dtype=np.float32)
        mix_state = np.zeros((1,), dtype=np.float32)
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            last_ts,
            last_pol,
            hot_state,
            beta_state,
            mix_state,
            scores,
        )
        return scores

    if v in {"s72", "ebf_s72", "ebfs72"}:
        from myevs.denoise.ops.ebfopt_part2.s72_ebf_labelscore_dualtau_trajgate_only import (
            try_build_s72_ebf_labelscore_dualtau_trajgate_only_scores_kernel,
        )

        ker = kernel_cache.get("ker_s72")
        if ker is None:
            ker = try_build_s72_ebf_labelscore_dualtau_trajgate_only_scores_kernel()
            kernel_cache["ker_s72"] = ker
        if ker is None:
            raise SystemExit("s72 requires numba, but kernel build failed")

        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            last_ts,
            last_pol,
            scores,
        )
        return scores

    if v in {"s73", "ebf_s73", "ebfs73"}:
        from myevs.denoise.ops.ebfopt_part2.s73_ebf_labelscore_dualtau_trajgate_only_flip import (
            try_build_s73_ebf_labelscore_dualtau_trajgate_only_flip_scores_kernel,
        )

        ker = kernel_cache.get("ker_s73")
        if ker is None:
            ker = try_build_s73_ebf_labelscore_dualtau_trajgate_only_flip_scores_kernel()
            kernel_cache["ker_s73"] = ker
        if ker is None:
            raise SystemExit("s73 requires numba, but kernel build failed")

        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            last_ts,
            last_pol,
            scores,
        )
        return scores

    if v in {"s74", "ebf_s74", "ebfs74"}:
        from myevs.denoise.ops.ebfopt_part2.s74_ebf_labelscore_surprise_adaptive_null_fixed import (
            try_build_s74_ebf_labelscore_surprise_adaptive_null_fixed_scores_kernel,
        )

        ker = kernel_cache.get("ker_s74")
        if ker is None:
            ker = try_build_s74_ebf_labelscore_surprise_adaptive_null_fixed_scores_kernel()
            kernel_cache["ker_s74"] = ker
        if ker is None:
            raise SystemExit("s74 requires numba, but kernel build failed")

        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        hot_state = np.zeros((int(width) * int(height),), dtype=np.int32)
        rate_ema = np.zeros((1,), dtype=np.float64)
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            last_ts,
            last_pol,
            hot_state,
            rate_ema,
            scores,
        )
        return scores

    if v in {"s75", "ebf_s75", "ebfs75"}:
        from myevs.denoise.ops.ebfopt_part2.s75_ebf_labelscore_hotmask_ratio_raw import (
            try_build_s75_ebf_labelscore_hotmask_ratio_raw_scores_kernel,
        )

        ker = kernel_cache.get("ker_s75")
        if ker is None:
            ker = try_build_s75_ebf_labelscore_hotmask_ratio_raw_scores_kernel()
            kernel_cache["ker_s75"] = ker
        if ker is None:
            raise SystemExit("s75 requires numba, but kernel build failed")

        env = (str(env_name).strip().lower() if env_name is not None else "heavy")
        if env not in {"light", "mid", "heavy"}:
            env = "heavy"

        hot_key = f"hotmask_{env}"
        hot_mask = kernel_cache.get(hot_key)
        if hot_mask is None:
            hot_default = {
                "heavy": "data/ED24/myPedestrain_06/EBF_Part2/hotmask_heavy_score_neg_minus_2pos_topk32768_dil1.npy",
                "mid": "data/ED24/myPedestrain_06/EBF_Part2/hotmask_mid_score_neg_minus_2pos_topk32768_dil1.npy",
                "light": "data/ED24/myPedestrain_06/EBF_Part2/hotmask_light_score_neg_minus_2pos_topk32768_dil1.npy",
            }
            hot_path = hot_default[env]
            if not os.path.exists(hot_path):
                raise SystemExit(f"s75 hotmask not found for env={env!r}: {hot_path}")
            hm2 = np.load(hot_path)
            hot_mask = np.ascontiguousarray(hm2.astype(np.uint8, copy=False).reshape(-1))
            kernel_cache[hot_key] = hot_mask

        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            last_ts,
            last_pol,
            hot_mask,
            scores,
        )
        return scores

    if v in {"s76", "ebf_s76", "ebfs76"}:
        from myevs.denoise.ops.ebfopt_part2.s76_aocc_activity_sobel_gradmag import (
            try_build_s76_aocc_activity_sobel_gradmag_scores_kernel,
        )

        ker = kernel_cache.get("ker_s76")
        if ker is None:
            ker = try_build_s76_aocc_activity_sobel_gradmag_scores_kernel()
            kernel_cache["ker_s76"] = ker
        if ker is None:
            raise SystemExit("s76 requires numba, but kernel build failed")

        last_t = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_a = np.zeros((int(width) * int(height),), dtype=np.float32)
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            last_t,
            last_a,
            scores,
        )
        return scores

    if v in {"s77", "ebf_s77", "ebfs77"}:
        from myevs.denoise.ops.ebfopt_part2.s77_aocc_polnorm_multiscale_grad import (
            try_build_s77_aocc_polnorm_multiscale_scores_kernel,
        )

        ker = kernel_cache.get("ker_s77")
        if ker is None:
            ker = try_build_s77_aocc_polnorm_multiscale_scores_kernel()
            kernel_cache["ker_s77"] = ker
        if ker is None:
            raise SystemExit("s77 requires numba, but kernel build failed")

        npx = int(width) * int(height)

        last_ts_pos_s = np.zeros((npx,), dtype=np.uint64)
        last_as_pos_s = np.zeros((npx,), dtype=np.float32)
        last_ts_pos_l = np.zeros((npx,), dtype=np.uint64)
        last_as_pos_l = np.zeros((npx,), dtype=np.float32)

        last_ts_neg_s = np.zeros((npx,), dtype=np.uint64)
        last_as_neg_s = np.zeros((npx,), dtype=np.float32)
        last_ts_neg_l = np.zeros((npx,), dtype=np.uint64)
        last_as_neg_l = np.zeros((npx,), dtype=np.float32)

        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            last_ts_pos_s,
            last_as_pos_s,
            last_ts_pos_l,
            last_as_pos_l,
            last_ts_neg_s,
            last_as_neg_s,
            last_ts_neg_l,
            last_as_neg_l,
            scores,
        )
        return scores

    if v in {"s78", "ebf_s78", "ebfs78"}:
        from myevs.denoise.ops.ebfopt_part2.s78_aocc_polnorm_multiscale_crosspol_penalty import (
            try_build_s78_aocc_polnorm_multiscale_crosspol_penalty_scores_kernel,
        )

        ker = kernel_cache.get("ker_s78")
        if ker is None:
            ker = try_build_s78_aocc_polnorm_multiscale_crosspol_penalty_scores_kernel()
            kernel_cache["ker_s78"] = ker
        if ker is None:
            raise SystemExit("s78 requires numba, but kernel build failed")

        npx = int(width) * int(height)

        last_ts_pos_s = np.zeros((npx,), dtype=np.uint64)
        last_as_pos_s = np.zeros((npx,), dtype=np.float32)
        last_ts_pos_l = np.zeros((npx,), dtype=np.uint64)
        last_as_pos_l = np.zeros((npx,), dtype=np.float32)

        last_ts_neg_s = np.zeros((npx,), dtype=np.uint64)
        last_as_neg_s = np.zeros((npx,), dtype=np.float32)
        last_ts_neg_l = np.zeros((npx,), dtype=np.uint64)
        last_as_neg_l = np.zeros((npx,), dtype=np.float32)

        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            last_ts_pos_s,
            last_as_pos_s,
            last_ts_pos_l,
            last_as_pos_l,
            last_ts_neg_s,
            last_as_neg_s,
            last_ts_neg_l,
            last_as_neg_l,
            scores,
        )
        return scores

    if v in {"s79", "ebf_s79", "ebfs79"}:
        from myevs.denoise.ops.ebfopt_part2.s79_aocc_discrete_windows_continuity import (
            try_build_s79_aocc_discrete_windows_continuity_scores_kernel,
        )

        ker = kernel_cache.get("ker_s79")
        if ker is None:
            ker = try_build_s79_aocc_discrete_windows_continuity_scores_kernel()
            kernel_cache["ker_s79"] = ker
        if ker is None:
            raise SystemExit("s79 requires numba, but kernel build failed")

        npx = int(width) * int(height)

        last_bin_pos = np.full((npx,), -1, dtype=np.int64)
        c0_pos = np.zeros((npx,), dtype=np.float32)
        c1_pos = np.zeros((npx,), dtype=np.float32)
        c2_pos = np.zeros((npx,), dtype=np.float32)

        last_bin_neg = np.full((npx,), -1, dtype=np.int64)
        c0_neg = np.zeros((npx,), dtype=np.float32)
        c1_neg = np.zeros((npx,), dtype=np.float32)
        c2_neg = np.zeros((npx,), dtype=np.float32)

        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            last_bin_pos,
            c0_pos,
            c1_pos,
            c2_pos,
            last_bin_neg,
            c0_neg,
            c1_neg,
            c2_neg,
            scores,
        )
        return scores

    if v in {"n96", "ebf_n96", "ebfn96"}:
        from myevs.denoise.ops.ebfopt_part2.n96_top1_v_backbone import score_stream_n96

        return score_stream_n96(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )

    if v in {"n97", "ebf_n97", "ebfn97"}:
        from myevs.denoise.ops.ebfopt_part2.n97_top1_2d_backbone import score_stream_n97

        return score_stream_n97(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )

    if v in {"n98", "ebf_n98", "ebfn98"}:
        from myevs.denoise.ops.ebfopt_part2.n98_top3_2d_backbone import score_stream_n98

        return score_stream_n98(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )

    if v in {"n99", "ebf_n99", "ebfn99"}:
        from myevs.denoise.ops.ebfopt_part2.n99_top3_2d_max_backbone import score_stream_n99

        return score_stream_n99(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )

    if v in {"n100", "ebf_n100", "ebfn100"}:
        from myevs.denoise.ops.ebfopt_part2.n100_top3_2d_max_support_backbone import score_stream_n100

        return score_stream_n100(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )

    if v in {"n101", "ebf_n101", "ebfn101"}:
        from myevs.denoise.ops.ebfopt_part2.n101_top3_2d_mean_support_backbone import score_stream_n101

        return score_stream_n101(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )

    if v in {"n102", "ebf_n102", "ebfn102"}:
        from myevs.denoise.ops.ebfopt_part2.n102_top3_2d_mean_goodsupport_backbone import score_stream_n102

        return score_stream_n102(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )

    if v in {"n103", "ebf_n103", "ebfn103"}:
        os.environ.setdefault("MYEVS_N103_TOPK", "5")
        from myevs.denoise.ops.ebfopt_part2.n103_topk_2d_mean_support_backbone import score_stream_n103

        return score_stream_n103(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )

    if v in {"n104", "ebf_n104", "ebfn104"}:
        os.environ.setdefault("MYEVS_N103_TOPK", "7")
        from myevs.denoise.ops.ebfopt_part2.n103_topk_2d_mean_support_backbone import score_stream_n103

        return score_stream_n103(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )

    if v in {"n105", "ebf_n105", "ebfn105"}:
        os.environ.setdefault("MYEVS_N103_TOPK", "9")
        from myevs.denoise.ops.ebfopt_part2.n103_topk_2d_mean_support_backbone import score_stream_n103

        return score_stream_n103(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )

    if v in {"n106", "ebf_n106", "ebfn106"}:
        from myevs.denoise.ops.ebfopt_part2.n106_sector_density_backbone import score_stream_n106

        return score_stream_n106(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )

    if v in {"n108", "ebf_n108", "ebfn108"}:
        from myevs.denoise.ops.ebfopt_part2.n108_contrast_enhancement_backbone import score_stream_n108
        return score_stream_n108(
            ev, width=width, height=height, radius_px=radius_px, tau_us=tau_us, tb=tb,
        )
    if v in {"n109", "ebf_n109", "ebfn109"}:
        from myevs.denoise.ops.ebfopt_part2.n109_self_adaptive_anisotropy_backbone import score_stream_n109

        return score_stream_n109(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )
    if v in {"n110", "ebf_n110", "ebfn110"}:
        from myevs.denoise.ops.ebfopt_part2.n110_sector_transition_gate_backbone import score_stream_n110

        return score_stream_n110(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )
    if v in {"n111", "ebf_n111", "ebfn111"}:
        from myevs.denoise.ops.ebfopt_part2.n111_sector_polarity_contrast_backbone import score_stream_n111

        return score_stream_n111(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )
    if v in {"n112", "ebf_n112", "ebfn112"}:
        from myevs.denoise.ops.ebfopt_part2.n112_dual_scale_purity_backbone import score_stream_n112

        return score_stream_n112(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )
    if v in {"n113", "ebf_n113", "ebfn113"}:
        from myevs.denoise.ops.ebfopt_part2.n113_power_law_decay_backbone import score_stream_n113

        return score_stream_n113(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )
    if v in {"n114", "ebf_n114", "ebfn114"}:
        from myevs.denoise.ops.ebfopt_part2.n114_isotropic_subtraction_backbone import score_stream_n114

        return score_stream_n114(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )
    if v in {"n116", "ebf_n116", "ebfn116"}:
        from myevs.denoise.ops.ebfopt_part2.n116_isochronous_burst_gate_backbone import score_stream_n116

        return score_stream_n116(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )
    if v in {"n117", "ebf_n117", "ebfn117"}:
        from myevs.denoise.ops.ebfopt_part2.n117_bipolar_echo_boost_backbone import score_stream_n117

        return score_stream_n117(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )
    if v in {"n118", "ebf_n118", "ebfn118"}:
        from myevs.denoise.ops.ebfopt_part2.n118_polarity_dipole_backbone import score_stream_n118

        return score_stream_n118(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )
    if v in {"n120", "ebf_n120", "ebfn120"}:
        from myevs.denoise.ops.ebfopt_part2.n120_self_inhibition_backbone import score_stream_n120

        return score_stream_n120(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )
    if v in {"n121", "ebf_n121", "ebfn121"}:
        from myevs.denoise.ops.ebfopt_part2.n121_center_surround_backbone import score_stream_n121

        return score_stream_n121(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )
    if v in {"n123", "ebf_n123", "ebfn123"}:
        from myevs.denoise.ops.ebfopt_part2.n123_isotropic_max8_backbone import score_stream_n123

        return score_stream_n123(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )
    if v in {"n124", "ebf_n124", "ebfn124"}:
        from myevs.denoise.ops.ebfopt_part2.n124_synergy_trifilter_backbone import score_stream_n124

        return score_stream_n124(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )
    if v in {"n125", "ebf_n125", "ebfn125"}:
        from myevs.denoise.ops.ebfopt_part2.n125_micro_topo_path_backbone import score_stream_n125

        return score_stream_n125(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )
    if v in {"n126", "ebf_n126", "ebfn126"}:
        from myevs.denoise.ops.ebfopt_part2.n126_top1_bonus_backbone import score_stream_n126

        return score_stream_n126(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )
    if v in {"n127", "ebf_n127", "ebfn127"}:
        from myevs.denoise.ops.ebfopt_part2.n127_top2_bonus_backbone import score_stream_n127

        return score_stream_n127(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )
    if v in {"n128", "ebf_n128", "ebfn128"}:
        from myevs.denoise.ops.ebfopt_part2.n128_spatiotemporal_joint_decay_backbone import score_stream_n128

        return score_stream_n128(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )
    if v in {"n129", "ebf_n129", "ebfn129"}:
        from myevs.denoise.ops.ebfopt_part2.n129_struct_purity_joint_backbone import score_stream_n129

        return score_stream_n129(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )
    if v in {"n131", "ebf_n131", "ebfn131"}:
        from myevs.denoise.ops.ebfopt_part2.n131_pure_struct_bandpass_backbone import score_stream_n131

        return score_stream_n131(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )
    if v in {"n132", "ebf_n132", "ebfn132"}:
        from myevs.denoise.ops.ebfopt_part2.n132_conditional_weak_struct_gate_backbone import score_stream_n132

        return score_stream_n132(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )
    if v in {"n133", "ebf_n133", "ebfn133"}:
        from myevs.denoise.ops.ebfopt_part2.n133_soft_overactivation_bandpass_backbone import score_stream_n133

        return score_stream_n133(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )
    if v in {"n134", "ebf_n134", "ebfn134"}:
        from myevs.denoise.ops.ebfopt_part2.n134_stateful_support_pruning_backbone import score_stream_n134

        return score_stream_n134(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )
    if v in {"n135", "ebf_n135", "ebfn135"}:
        from myevs.denoise.ops.ebfopt_part2.n135_confidence_map_dualtrack_backbone import score_stream_n135

        return score_stream_n135(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )
    if v in {"n137", "ebf_n137", "ebfn137"}:
        from myevs.denoise.ops.ebfopt_part2.n137_pure_axis_emax_filter import score_stream_n137

        return score_stream_n137(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )
    if v in {"n139", "ebf_n139", "ebfn139"}:
        from myevs.denoise.ops.ebfopt_part2.n139_binary_struct_bandpass import score_stream_n139

        return score_stream_n139(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            low_thresh=(None if n139_low is None else float(n139_low)),
            high_thresh=(None if n139_high is None else float(n139_high)),
            scores_out=scores,
        )
    if v in {"n140", "ebf_n140", "ebfn140"}:
        from myevs.denoise.ops.ebfopt_part2.n140_gaussian_spatial_decay_backbone import score_stream_n140

        return score_stream_n140(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )
    if v in {"n141", "ebf_n141", "ebfn141"}:
        from myevs.denoise.ops.ebfopt_part2.n141_gaussian_spatiotemporal_decay_backbone import score_stream_n141

        return score_stream_n141(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )
    if v in {"n142", "ebf_n142", "ebfn142"}:
        from myevs.denoise.ops.ebfopt_part2.n142_quadratic_time_decay_backbone import score_stream_n142

        return score_stream_n142(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )
    if v in {"n143", "ebf_n143", "ebfn143"}:
        from myevs.denoise.ops.ebfopt_part2.n143_bilateral_gaussian_approx_backbone import score_stream_n143

        return score_stream_n143(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )
    if v in {"n144", "ebf_n144", "ebfn144"}:
        from myevs.denoise.ops.ebfopt_part2.n144_bilateral_gaussian_linear_time_backbone import score_stream_n144

        return score_stream_n144(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )
    if v in {"n145", "ebf_n145", "ebfn145"}:
        from myevs.denoise.ops.ebfopt_part2.n145_bilateral_gaussian_sq_linear_time_backbone import score_stream_n145

        return score_stream_n145(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )
    if v in {"n146", "ebf_n146", "ebfn146"}:
        from myevs.denoise.ops.ebfopt_part2.n146_polarity_soft_fusion_backbone import score_stream_n146

        return score_stream_n146(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )
    if v in {"n147", "ebf_n147", "ebfn147"}:
        from myevs.denoise.ops.ebfopt_part2.n147_n145_s52_fusion_backbone import score_stream_n147

        return score_stream_n147(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )
    if v in {"n148", "ebf_n148", "ebfn148"}:
        from myevs.denoise.ops.ebfopt_part2.n148_n145_s52_euclid_octant_backbone import score_stream_n148

        return score_stream_n148(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )
    if v in {"n149", "ebf_n149", "ebfn149"}:
        from myevs.denoise.ops.ebfopt_part2.n149_n145_s52_euclid_compactlut_backbone import score_stream_n149

        return score_stream_n149(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )
    if v in {"n150", "ebf_n150", "ebfn150"}:
        from myevs.denoise.ops.ebfopt_part2.n150_n149_s52lite_nohot_backbone import score_stream_n150

        return score_stream_n150(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )
    if v in {"n151", "ebf_n151", "ebfn151"}:
        from myevs.denoise.ops.ebfopt_part2.n151_n150_recurrence_proxy_backbone import score_stream_n151

        return score_stream_n151(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )
    if v in {"n152", "ebf_n152", "ebfn152"}:
        from myevs.denoise.ops.ebfopt_part2.n152_n150_supportmix_center_proxy_backbone import score_stream_n152

        return score_stream_n152(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )
    if v in {"n153", "ebf_n153", "ebfn153"}:
        from myevs.denoise.ops.ebfopt_part2.n153_n152_fixedproxy_backbone import score_stream_n153

        return score_stream_n153(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )
    if v in {"n160", "ebf_n160", "ebfn160"}:
        from myevs.denoise.ops.ebfopt_part2.n160_polarity_purity_fixed_backbone import score_stream_n160

        return score_stream_n160(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )
    if v in {"n161", "ebf_n161", "ebfn161"}:
        from myevs.denoise.ops.ebfopt_part2.n161_polarity_linear_fixed_backbone import score_stream_n161

        return score_stream_n161(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )
    if v in {"n162", "ebf_n162", "ebfn162"}:
        from myevs.denoise.ops.ebfopt_part2.n162_polarity_residual_fixed_backbone import score_stream_n162

        return score_stream_n162(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )
    if v in {"n170", "ebf_n170", "ebfn170"}:
        from myevs.denoise.ops.ebfopt_part2.n170_polarity_transition_fixed_backbone import score_stream_n170

        return score_stream_n170(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )
    if v in {"n171", "ebf_n171", "ebfn171"}:
        from myevs.denoise.ops.ebfopt_part2.n171_s52lite_rhythm_fixed_backbone import score_stream_n171

        return score_stream_n171(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )
    if v in {"n107", "ebf_n107", "ebfn107"}:
        from myevs.denoise.ops.ebfopt_part2.n107_projected_energy_backbone import score_stream_n107

        return score_stream_n107(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            scores_out=scores,
        )

    raise SystemExit(
        f"unknown variant: {variant!r}. supported: ebf | s52 | s55 | s60 | s61 | s62 | s63 | s64 | s65 | s66 | s67 | s68 | s69 | s70 | s71 | s72 | s73 | s74 | s75 | s76 | s77 | s78 | s79 | s80 | s81 | s82 | s83 | s84 | s85 | s86 | s87 | n1 | n2 | n3 | n4 | n5 | n6 | n7 | n71 | n72 | n8 | n81 | n82 | n83 | n84 | n85 | n86 | n87 | n88 | n89 | n90 | n91 | n92 | n93 | n94 | n95 | n96 | n97 | n98 | n99 | n100 | n101 | n102 | n103 | n104 | n105 | n106 | n107 | n108 | n109 | n110 | n111 | n112 | n113 | n114 | n116 | n117 | n118 | n120 | n121 | n123 | n124 | n125 | n126 | n127 | n128 | n129 | n131 | n132 | n133 | n134 | n135 | n137 | n139 | n140 | n141 | n142 | n143 | n144 | n145 | n146 | n147 | n148 | n149 | n150 | n151 | n152 | n153 | n160 | n161 | n162 | n170 | n171"
    )


def score_stream_ebf(
    ev: LabeledEvents,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_us: int,
    tb: TimeBase,
    _kernel_cache: dict[str, object] | None = None,
    variant: str = "ebf",
) -> np.ndarray:
    """Public scoring entry for compatibility with downstream analysis.

    This intentionally mirrors the call pattern used by scripts under
    scripts/noise_analyze/ (e.g., segment_f1.py) so they can evaluate variants
    through this slim implementation.
    """

    kernel_cache = _kernel_cache if _kernel_cache is not None else {}
    return _score_stream(
        ev,
        width=int(width),
        height=int(height),
        radius_px=int(radius_px),
        tau_us=int(tau_us),
        tb=tb,
        variant=str(variant),
        kernel_cache=kernel_cache,
        env_name=None,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Slim sweep for ED24 labeled .npy (score->ROC, minimal variants).")
    ap.add_argument(
        "--variant",
        default="ebf",
        help="ebf | s52 | s55 | s60 | s61 | s62 | s63 | s64 | s65 | s66 | s67 | s68 | s69 | s70 | s71 | s72 | s73 | s74 | s75 | s76 | s77 | s78 | s79 | s80 | s81 | s82 | s83 | s84 | s85 | s86 | s87 | n1 | n2 | n3 | n4 | n5 | n6 | n7 | n71 | n72 | n8 | n81 | n82 | n83 | n84 | n85 | n86 | n87 | n88 | n89 | n90 | n91 | n92 | n93 | n94 | n95 | n96 | n97 | n98 | n99 | n100 | n101 | n102 | n103 | n104 | n105 | n106 | n107 | n108 | n109 | n110 | n111 | n112 | n113 | n114 | n116 | n117 | n118 | n120 | n121 | n123 | n124 | n125 | n126 | n127 | n128 | n129 | n131 | n132 | n133 | n134 | n135 | n137 | n139 | n140 | n141 | n142 | n143 | n144 | n145 | n146 | n147 | n148 | n149 | n150 | n151 | n152 | n153 | n160 | n161 | n162 | n170 | n171",
    )
    ap.add_argument("--max-events", type=int, default=int(os.environ.get("EBF_MAX_EVENTS", "0")), help="0=all")
    ap.add_argument("--out-dir", default="data/ED24/myPedestrain_06/EBF_Part2/_slim", help="output directory")

    ap.add_argument("--width", type=int, default=346)
    ap.add_argument("--height", type=int, default=260)
    ap.add_argument("--tick-ns", type=float, default=1000.0)

    ap.add_argument("--s-list", default="3,5,7,9")
    ap.add_argument("--tau-us-list", default="8000,16000,32000,64000,128000,256000,512000,1024000")
    ap.add_argument(
        "--n139-low-list",
        default="0.1,0.2,0.3,0.4,0.5",
        help="for n139 only: low-threshold grid",
    )
    ap.add_argument(
        "--n139-high-list",
        default="0.5,0.6,0.7,0.8,0.9",
        help="for n139 only: high-threshold grid",
    )
    ap.add_argument("--roc-max-points", type=int, default=5000)

    ap.add_argument(
        "--esr-mode",
        default=str(os.environ.get("MYEVS_ESR_MODE", "best")),
        choices=["best", "all", "off"],
        help=(
            "MESR/ESR compute mode for esr_mean column: "
            "best=only compute at best-AUC tag and best-F1 tag per env; "
            "all=compute at best-F1 point for every tag; off=skip ESR entirely."
        ),
    )
    ap.add_argument(
        "--aocc-mode",
        default=str(os.environ.get("MYEVS_AOCC_MODE", "best")),
        choices=["best", "all", "off"],
        help=(
            "AOCC compute mode for aocc column: "
            "best=only compute at best-AUC tag and best-F1 tag per env; "
            "all=compute at best-F1 point for every tag; off=skip AOCC entirely."
        ),
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

    v = str(args.variant).strip().lower()
    if v in {"ebf", "baseline"}:
        roc_prefix = "roc_ebf"
        tag_prefix = "ebf"
    elif v in {"s55", "ebf_s55", "ebfs55"}:
        roc_prefix = "roc_ebf_s55"
        tag_prefix = "ebf_s55"
    elif v in {"s60", "ebf_s60", "ebfs60"}:
        roc_prefix = "roc_ebf_s60"
        tag_prefix = "ebf_s60"
    elif v in {"s61", "ebf_s61", "ebfs61"}:
        roc_prefix = "roc_ebf_s61"
        tag_prefix = "ebf_s61"
    elif v in {"s62", "ebf_s62", "ebfs62"}:
        roc_prefix = "roc_ebf_s62"
        tag_prefix = "ebf_s62"
    elif v in {"s63", "ebf_s63", "ebfs63"}:
        roc_prefix = "roc_ebf_s63"
        tag_prefix = "ebf_s63"
    elif v in {"s64", "ebf_s64", "ebfs64"}:
        roc_prefix = "roc_ebf_s64"
        tag_prefix = "ebf_s64"
    elif v in {"s65", "ebf_s65", "ebfs65"}:
        roc_prefix = "roc_ebf_s65"
        tag_prefix = "ebf_s65"
    elif v in {"s66", "ebf_s66", "ebfs66"}:
        roc_prefix = "roc_ebf_s66"
        tag_prefix = "ebf_s66"
    elif v in {"s67", "ebf_s67", "ebfs67"}:
        roc_prefix = "roc_ebf_s67"
        tag_prefix = "ebf_s67"
    elif v in {"s68", "ebf_s68", "ebfs68"}:
        roc_prefix = "roc_ebf_s68"
        tag_prefix = "ebf_s68"
    elif v in {"s69", "ebf_s69", "ebfs69"}:
        roc_prefix = "roc_ebf_s69"
        tag_prefix = "ebf_s69"
    elif v in {"s70", "ebf_s70", "ebfs70"}:
        roc_prefix = "roc_ebf_s70"
        tag_prefix = "ebf_s70"
    elif v in {"s71", "ebf_s71", "ebfs71"}:
        roc_prefix = "roc_ebf_s71"
        tag_prefix = "ebf_s71"
    elif v in {"s72", "ebf_s72", "ebfs72"}:
        roc_prefix = "roc_ebf_s72"
        tag_prefix = "ebf_s72"
    elif v in {"s73", "ebf_s73", "ebfs73"}:
        roc_prefix = "roc_ebf_s73"
        tag_prefix = "ebf_s73"
    elif v in {"s74", "ebf_s74", "ebfs74"}:
        roc_prefix = "roc_ebf_s74"
        tag_prefix = "ebf_s74"
    elif v in {"s75", "ebf_s75", "ebfs75"}:
        roc_prefix = "roc_ebf_s75"
        tag_prefix = "ebf_s75"
    elif v in {"s76", "ebf_s76", "ebfs76"}:
        roc_prefix = "roc_ebf_s76"
        tag_prefix = "ebf_s76"
    elif v in {"s77", "ebf_s77", "ebfs77"}:
        roc_prefix = "roc_ebf_s77"
        tag_prefix = "ebf_s77"
    elif v in {"s78", "ebf_s78", "ebfs78"}:
        roc_prefix = "roc_ebf_s78"
        tag_prefix = "ebf_s78"
    elif v in {"s79", "ebf_s79", "ebfs79"}:
        roc_prefix = "roc_ebf_s79"
        tag_prefix = "ebf_s79"
    elif v in {"s80", "ebf_s80", "ebfs80"}:
        roc_prefix = "roc_ebf_s80"
        tag_prefix = "ebf_s80"
    elif v in {"s81", "ebf_s81", "ebfs81"}:
        roc_prefix = "roc_ebf_s81"
        tag_prefix = "ebf_s81"
    elif v in {"s82", "ebf_s82", "ebfs82"}:
        roc_prefix = "roc_ebf_s82"
        tag_prefix = "ebf_s82"
    elif v in {"s52", "ebf_s52", "ebfs52"}:
        roc_prefix = "roc_ebf_s52"
        tag_prefix = "ebf_s52"
    elif v in {"s83", "ebf_s83", "ebfs83"}:
        roc_prefix = "roc_ebf_s83"
        tag_prefix = "ebf_s83"
    elif v in {"s84", "ebf_s84", "ebfs84"}:
        roc_prefix = "roc_ebf_s84"
        tag_prefix = "ebf_s84"
    elif v in {"s85", "ebf_s85", "ebfs85"}:
        roc_prefix = "roc_ebf_s85"
        tag_prefix = "ebf_s85"
    elif v in {"s86", "ebf_s86", "ebfs86"}:
        roc_prefix = "roc_ebf_s86"
        tag_prefix = "ebf_s86"
    elif v in {"s87", "ebf_s87", "ebfs87"}:
        roc_prefix = "roc_ebf_s87"
        tag_prefix = "ebf_s87"
    elif v in {"n1", "ebf_n1", "ebfn1"}:
        roc_prefix = "roc_ebf_n1"
        tag_prefix = "ebf_n1"
    elif v in {"n2", "ebf_n2", "ebfn2"}:
        roc_prefix = "roc_ebf_n2"
        tag_prefix = "ebf_n2"
    elif v in {"n3", "ebf_n3", "ebfn3"}:
        roc_prefix = "roc_ebf_n3"
        tag_prefix = "ebf_n3"
    elif v in {"n4", "ebf_n4", "ebfn4"}:
        roc_prefix = "roc_ebf_n4"
        tag_prefix = "ebf_n4"
    elif v in {"n5", "ebf_n5", "ebfn5"}:
        roc_prefix = "roc_ebf_n5"
        tag_prefix = "ebf_n5"
    elif v in {"n6", "ebf_n6", "ebfn6"}:
        roc_prefix = "roc_ebf_n6"
        tag_prefix = "ebf_n6"
    elif v in {"n7", "ebf_n7", "ebfn7"}:
        roc_prefix = "roc_ebf_n7"
        tag_prefix = "ebf_n7"
    elif v in {"n71", "n7.1", "ebf_n71", "ebfn71"}:
        roc_prefix = "roc_ebf_n71"
        tag_prefix = "ebf_n71"
    elif v in {"n72", "n7.2", "ebf_n72", "ebfn72"}:
        roc_prefix = "roc_ebf_n72"
        tag_prefix = "ebf_n72"
    elif v in {"n8", "ebf_n8", "ebfn8"}:
        roc_prefix = "roc_ebf_n8"
        tag_prefix = "ebf_n8"
    elif v in {"n81", "n8.1", "ebf_n81", "ebfn81"}:
        roc_prefix = "roc_ebf_n81"
        tag_prefix = "ebf_n81"
    elif v in {"n82", "n8.2", "ebf_n82", "ebfn82"}:
        roc_prefix = "roc_ebf_n82"
        tag_prefix = "ebf_n82"
    elif v in {"n83", "n8.3", "ebf_n83", "ebfn83"}:
        roc_prefix = "roc_ebf_n83"
        tag_prefix = "ebf_n83"
    elif v in {"n84", "ebf_n84", "ebfn84"}:
        roc_prefix = "roc_ebf_n84"
        tag_prefix = "ebf_n84"
    elif v in {"n85", "ebf_n85", "ebfn85"}:
        roc_prefix = "roc_ebf_n85"
        tag_prefix = "ebf_n85"
    elif v in {"n86", "ebf_n86", "ebfn86"}:
        roc_prefix = "roc_ebf_n86"
        tag_prefix = "ebf_n86"
    elif v in {"n87", "ebf_n87", "ebfn87"}:
        roc_prefix = "roc_ebf_n87"
        tag_prefix = "ebf_n87"
    elif v in {"n88", "ebf_n88", "ebfn88"}:
        roc_prefix = "roc_ebf_n88"
        tag_prefix = "ebf_n88"
    elif v in {"n89", "ebf_n89", "ebfn89"}:
        roc_prefix = "roc_ebf_n89"
        tag_prefix = "ebf_n89"
    elif v in {"n90", "ebf_n90", "ebfn90"}:
        roc_prefix = "roc_ebf_n90"
        tag_prefix = "ebf_n90"
    elif v in {"n91", "ebf_n91", "ebfn91"}:
        roc_prefix = "roc_ebf_n91"
        tag_prefix = "ebf_n91"
    elif v in {"n92", "ebf_n92", "ebfn92"}:
        roc_prefix = "roc_ebf_n92"
        tag_prefix = "ebf_n92"
    elif v in {"n93", "ebf_n93", "ebfn93"}:
        roc_prefix = "roc_ebf_n93"
        tag_prefix = "ebf_n93"
    elif v in {"n94", "ebf_n94", "ebfn94"}:
        roc_prefix = "roc_ebf_n94"
        tag_prefix = "ebf_n94"
    elif v in {"n95", "ebf_n95", "ebfn95"}:
        roc_prefix = "roc_ebf_n95"
        tag_prefix = "ebf_n95"
    elif v in {"n96", "ebf_n96", "ebfn96"}:
        roc_prefix = "roc_ebf_n96"
        tag_prefix = "ebf_n96"
    elif v in {"n97", "ebf_n97", "ebfn97"}:
        roc_prefix = "roc_ebf_n97"
        tag_prefix = "ebf_n97"
    elif v in {"n98", "ebf_n98", "ebfn98"}:
        roc_prefix = "roc_ebf_n98"
        tag_prefix = "ebf_n98"
    elif v in {"n99", "ebf_n99", "ebfn99"}:
        roc_prefix = "roc_ebf_n99"
        tag_prefix = "ebf_n99"
    elif v in {"n100", "ebf_n100", "ebfn100"}:
        roc_prefix = "roc_ebf_n100"
        tag_prefix = "ebf_n100"
    elif v in {"n101", "ebf_n101", "ebfn101"}:
        roc_prefix = "roc_ebf_n101"
        tag_prefix = "ebf_n101"
    elif v in {"n102", "ebf_n102", "ebfn102"}:
        roc_prefix = "roc_ebf_n102"
        tag_prefix = "ebf_n102"
    elif v in {"n103", "ebf_n103", "ebfn103"}:
        roc_prefix = "roc_ebf_n103"
        tag_prefix = "ebf_n103"
    elif v in {"n104", "ebf_n104", "ebfn104"}:
        roc_prefix = "roc_ebf_n104"
        tag_prefix = "ebf_n104"
    elif v in {"n105", "ebf_n105", "ebfn105"}:
        roc_prefix = "roc_ebf_n105"
        tag_prefix = "ebf_n105"
    elif v in {"n106", "ebf_n106", "ebfn106"}:
        roc_prefix = "roc_ebf_n106"
        tag_prefix = "ebf_n106"
    elif v in {"n108", "ebf_n108", "ebfn108"}:
        roc_prefix = "roc_ebf_n108"
        tag_prefix = "ebf_n108"
    elif v in {"n109", "ebf_n109", "ebfn109"}:
        roc_prefix = "roc_ebf_n109"
        tag_prefix = "ebf_n109"
    elif v in {"n110", "ebf_n110", "ebfn110"}:
        roc_prefix = "roc_ebf_n110"
        tag_prefix = "ebf_n110"
    elif v in {"n111", "ebf_n111", "ebfn111"}:
        roc_prefix = "roc_ebf_n111"
        tag_prefix = "ebf_n111"
    elif v in {"n112", "ebf_n112", "ebfn112"}:
        roc_prefix = "roc_ebf_n112"
        tag_prefix = "ebf_n112"
    elif v in {"n113", "ebf_n113", "ebfn113"}:
        roc_prefix = "roc_ebf_n113"
        tag_prefix = "ebf_n113"
    elif v in {"n114", "ebf_n114", "ebfn114"}:
        roc_prefix = "roc_ebf_n114"
        tag_prefix = "ebf_n114"
    elif v in {"n116", "ebf_n116", "ebfn116"}:
        roc_prefix = "roc_ebf_n116"
        tag_prefix = "ebf_n116"
    elif v in {"n117", "ebf_n117", "ebfn117"}:
        roc_prefix = "roc_ebf_n117"
        tag_prefix = "ebf_n117"
    elif v in {"n118", "ebf_n118", "ebfn118"}:
        roc_prefix = "roc_ebf_n118"
        tag_prefix = "ebf_n118"
    elif v in {"n120", "ebf_n120", "ebfn120"}:
        roc_prefix = "roc_ebf_n120"
        tag_prefix = "ebf_n120"
    elif v in {"n121", "ebf_n121", "ebfn121"}:
        roc_prefix = "roc_ebf_n121"
        tag_prefix = "ebf_n121"
    elif v in {"n123", "ebf_n123", "ebfn123"}:
        roc_prefix = "roc_ebf_n123"
        tag_prefix = "ebf_n123"
    elif v in {"n124", "ebf_n124", "ebfn124"}:
        roc_prefix = "roc_ebf_n124"
        tag_prefix = "ebf_n124"
    elif v in {"n125", "ebf_n125", "ebfn125"}:
        roc_prefix = "roc_ebf_n125"
        tag_prefix = "ebf_n125"
    elif v in {"n126", "ebf_n126", "ebfn126"}:
        roc_prefix = "roc_ebf_n126"
        tag_prefix = "ebf_n126"
    elif v in {"n127", "ebf_n127", "ebfn127"}:
        roc_prefix = "roc_ebf_n127"
        tag_prefix = "ebf_n127"
    elif v in {"n128", "ebf_n128", "ebfn128"}:
        roc_prefix = "roc_ebf_n128"
        tag_prefix = "ebf_n128"
    elif v in {"n129", "ebf_n129", "ebfn129"}:
        roc_prefix = "roc_ebf_n129"
        tag_prefix = "ebf_n129"
    elif v in {"n131", "ebf_n131", "ebfn131"}:
        roc_prefix = "roc_ebf_n131"
        tag_prefix = "ebf_n131"
    elif v in {"n132", "ebf_n132", "ebfn132"}:
        roc_prefix = "roc_ebf_n132"
        tag_prefix = "ebf_n132"
    elif v in {"n133", "ebf_n133", "ebfn133"}:
        roc_prefix = "roc_ebf_n133"
        tag_prefix = "ebf_n133"
    elif v in {"n134", "ebf_n134", "ebfn134"}:
        roc_prefix = "roc_ebf_n134"
        tag_prefix = "ebf_n134"
    elif v in {"n135", "ebf_n135", "ebfn135"}:
        roc_prefix = "roc_ebf_n135"
        tag_prefix = "ebf_n135"
    elif v in {"n137", "ebf_n137", "ebfn137"}:
        roc_prefix = "roc_ebf_n137"
        tag_prefix = "ebf_n137"
    elif v in {"n139", "ebf_n139", "ebfn139"}:
        roc_prefix = "roc_ebf_n139"
        tag_prefix = "ebf_n139"
    elif v in {"n140", "ebf_n140", "ebfn140"}:
        roc_prefix = "roc_ebf_n140"
        tag_prefix = "ebf_n140"
    elif v in {"n141", "ebf_n141", "ebfn141"}:
        roc_prefix = "roc_ebf_n141"
        tag_prefix = "ebf_n141"
    elif v in {"n142", "ebf_n142", "ebfn142"}:
        roc_prefix = "roc_ebf_n142"
        tag_prefix = "ebf_n142"
    elif v in {"n143", "ebf_n143", "ebfn143"}:
        roc_prefix = "roc_ebf_n143"
        tag_prefix = "ebf_n143"
    elif v in {"n144", "ebf_n144", "ebfn144"}:
        roc_prefix = "roc_ebf_n144"
        tag_prefix = "ebf_n144"
    elif v in {"n145", "ebf_n145", "ebfn145"}:
        roc_prefix = "roc_ebf_n145"
        tag_prefix = "ebf_n145"
    elif v in {"n146", "ebf_n146", "ebfn146"}:
        roc_prefix = "roc_ebf_n146"
        tag_prefix = "ebf_n146"
    elif v in {"n147", "ebf_n147", "ebfn147"}:
        roc_prefix = "roc_ebf_n147"
        tag_prefix = "ebf_n147"
    elif v in {"n148", "ebf_n148", "ebfn148"}:
        roc_prefix = "roc_ebf_n148"
        tag_prefix = "ebf_n148"
    elif v in {"n149", "ebf_n149", "ebfn149"}:
        roc_prefix = "roc_ebf_n149"
        tag_prefix = "ebf_n149"
    elif v in {"n150", "ebf_n150", "ebfn150"}:
        roc_prefix = "roc_ebf_n150"
        tag_prefix = "ebf_n150"
    elif v in {"n151", "ebf_n151", "ebfn151"}:
        roc_prefix = "roc_ebf_n151"
        tag_prefix = "ebf_n151"
    elif v in {"n152", "ebf_n152", "ebfn152"}:
        roc_prefix = "roc_ebf_n152"
        tag_prefix = "ebf_n152"
    elif v in {"n153", "ebf_n153", "ebfn153"}:
        roc_prefix = "roc_ebf_n153"
        tag_prefix = "ebf_n153"
    elif v in {"n160", "ebf_n160", "ebfn160"}:
        roc_prefix = "roc_ebf_n160"
        tag_prefix = "ebf_n160"
    elif v in {"n161", "ebf_n161", "ebfn161"}:
        roc_prefix = "roc_ebf_n161"
        tag_prefix = "ebf_n161"
    elif v in {"n162", "ebf_n162", "ebfn162"}:
        roc_prefix = "roc_ebf_n162"
        tag_prefix = "ebf_n162"
    elif v in {"n170", "ebf_n170", "ebfn170"}:
        roc_prefix = "roc_ebf_n170"
        tag_prefix = "ebf_n170"
    elif v in {"n171", "ebf_n171", "ebfn171"}:
        roc_prefix = "roc_ebf_n171"
        tag_prefix = "ebf_n171"
    elif v in {"n107", "ebf_n107", "ebfn107"}:
        roc_prefix = "roc_ebf_n107"
        tag_prefix = "ebf_n107"
    else:
        raise SystemExit(
            f"unknown --variant: {args.variant!r}. choices: ebf | s52 | s55 | s60 | s61 | s62 | s63 | s64 | s65 | s66 | s67 | s68 | s69 | s70 | s71 | s72 | s73 | s74 | s75 | s76 | s77 | s78 | s79 | s80 | s81 | s82 | s83 | s84 | s85 | s86 | s87 | n1 | n2 | n3 | n4 | n5 | n6 | n7 | n71 | n72 | n8 | n81 | n82 | n83 | n84 | n85 | n86 | n87 | n88 | n89 | n90 | n91 | n92 | n93 | n94 | n95 | n96 | n97 | n98 | n99 | n100 | n101 | n102 | n103 | n104 | n105 | n106 | n107 | n108 | n109 | n110 | n111 | n112 | n113 | n114 | n116 | n117 | n118 | n120 | n121 | n123 | n124 | n125 | n126 | n127 | n128 | n129 | n131 | n132 | n133 | n134 | n135 | n137 | n139 | n140 | n141 | n142 | n143 | n144 | n145 | n146 | n147 | n148 | n149 | n150 | n151 | n152 | n153 | n160 | n161 | n162 | n170 | n171"
        )

    out_dir = str(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    esr_mode = str(args.esr_mode).strip().lower()
    aocc_mode = str(args.aocc_mode).strip().lower()
    need_best_recipes = (esr_mode == "best") or (aocc_mode == "best")

    s_tag = f"s{_join_ints([int(s) for s in s_list])}"
    tau_tag = _tau_list_tag([int(x) for x in tau_us_list])
    roc_csv = {env: os.path.join(out_dir, f"{roc_prefix}_{env}_labelscore_{s_tag}_{tau_tag}.csv") for env in env_inputs}

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
                    if v in {"n139", "ebf_n139", "ebfn139"}:
                        low_high_pairs: list[tuple[float, float]] = []
                        for lo in n139_low_list:
                            for hi in n139_high_list:
                                if float(lo) >= float(hi):
                                    continue
                                low_high_pairs.append((float(lo), float(hi)))
                        if not low_high_pairs:
                            raise SystemExit("n139 requires at least one valid pair where low < high")
                    else:
                        low_high_pairs = [(float("nan"), float("nan"))]

                    for n139_lo, n139_hi in low_high_pairs:
                        if v in {"n139", "ebf_n139", "ebfn139"}:
                            tag = (
                                f"{tag_prefix}_labelscore_s{int(s)}_tau{int(tau_us)}"
                                f"_l{_float_tag(n139_lo)}_h{_float_tag(n139_hi)}"
                            )
                        else:
                            tag = f"{tag_prefix}_labelscore_s{int(s)}_tau{int(tau_us)}"

                        scores = _score_stream(
                            ev,
                            width=int(args.width),
                            height=int(args.height),
                            radius_px=int(r),
                            tau_us=int(tau_us),
                            tb=tb,
                            variant=str(args.variant),
                            kernel_cache=kernel_cache,
                            env_name=str(env),
                            n139_low=(None if np.isnan(n139_lo) else float(n139_lo)),
                            n139_high=(None if np.isnan(n139_hi) else float(n139_hi)),
                        )

                        auc, thr, tp, fp, _fpr, _tpr = _roc_points_from_scores(
                            ev.label,
                            scores,
                            max_points=int(args.roc_max_points),
                        )

                        # Best-F1 operating point (used for best-F1 selection and ESR/AOCC sampling).
                        best_i = _best_f1_index(thr, tp, fp, pos=pos, neg=neg)
                        best_thr = float(thr[int(best_i)])

                        if auc > best_auc_by_env[env][1]:
                            best_auc_by_env[env] = (tag, float(auc))

                        # best-F1 value (for summary)
                        tp_b = int(tp[int(best_i)])
                        fp_b = int(fp[int(best_i)])
                        tpr_b = (tp_b / pos) if pos > 0 else 0.0
                        prec_den_b = tp_b + fp_b
                        precision_b = (tp_b / prec_den_b) if prec_den_b > 0 else 0.0
                        den_b = precision_b + tpr_b
                        best_f1 = float((2.0 * precision_b * tpr_b / den_b) if den_b > 0 else 0.0)
                        if best_f1 > best_f1_by_env[env][1]:
                            best_f1_by_env[env] = (tag, float(best_f1))

                        if v in {"n139", "ebf_n139", "ebfn139"}:
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
                                    "n139_low": (None if np.isnan(n139_lo) else float(n139_lo)),
                                    "n139_high": (None if np.isnan(n139_hi) else float(n139_hi)),
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
                                    "n139_low": (None if np.isnan(n139_lo) else float(n139_lo)),
                                    "n139_high": (None if np.isnan(n139_hi) else float(n139_hi)),
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

                        if v in {"n139", "ebf_n139", "ebfn139"}:
                            print(
                                f"auc={auc:.6f} env={env} s={int(s)} tau_us={int(tau_us)} "
                                f"low={float(n139_lo):.3f} high={float(n139_hi):.3f} "
                                f"points={int(thr.shape[0])} tag={tag}"
                            )
                        else:
                            print(
                                f"auc={auc:.6f} env={env} s={int(s)} tau_us={int(tau_us)} points={int(thr.shape[0])} tag={tag}"
                            )

        # For {esr,aocc}-mode=best: compute metrics only for the best-AUC and best-F1 tags, then patch CSV.
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
                        variant=str(args.variant),
                        kernel_cache=kernel_cache,
                        env_name=str(env),
                        n139_low=(None if recipe.get("n139_low") is None else float(recipe["n139_low"])),
                        n139_high=(None if recipe.get("n139_high") is None else float(recipe["n139_high"])),
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

        if v in {"n139", "ebf_n139", "ebfn139"} and n139_records[env]:
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
