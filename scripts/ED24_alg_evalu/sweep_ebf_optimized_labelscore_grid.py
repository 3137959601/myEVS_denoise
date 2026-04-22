from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
import math
import re

import numpy as np

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
    out: list[int] = []
    for part in str(s).replace(" ", "").split(","):
        if not part:
            continue
        out.append(int(part))
    return out


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
            # [label, t, y, x, p]
            label = a2[:, 0].astype(np.int8, copy=False)
            t = a2[:, 1].astype(np.uint64, copy=False)
            y = a2[:, 2].astype(np.int32, copy=False)
            x = a2[:, 3].astype(np.int32, copy=False)
            p = a2[:, 4].astype(np.int8, copy=False)
        else:
            # [t, x, y, p, label]
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


def _try_build_numba_kernel(*, use_soft_weight: int, use_recent_scale: int):
    try:
        from numba import njit  # type: ignore
    except Exception:
        return None

    use_soft_int = 1 if int(use_soft_weight) != 0 else 0
    use_recent_int = 1 if int(use_recent_scale) != 0 else 0

    @njit(cache=True)
    def ebfopt_scores_stream(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        width: int,
        height: int,
        radius_px: int,
        tau_ticks: int,
        last_ts: np.ndarray,
        last_pol: np.ndarray,
        scores_out: np.ndarray,
    ) -> None:
        """Compute EBF_optimized normalized score for each event.

        - Raw score = original EBF sum (same-pol * linear age weight)
        - Global EMA of inv_dt (events per tick) used as noise proxy
        - Normalize by expected noise score scale
        """

        n = int(t.shape[0])
        w = int(width)
        h = int(height)

        rr = int(radius_px)
        if rr < 0:
            rr = 0
        if rr > 8:
            rr = 8

        tau = int(tau_ticks)

        ema_alpha = 0.01
        last_t_global = -1
        ema_inv_dt = 0.0

        noise_weight_k = 4.0
        noise_weight_min = 0.02

        neigh_px = (2 * rr + 1) * (2 * rr + 1) - 1
        area = w * h
        if area <= 0:
            area = 1
        if neigh_px <= 0:
            neigh_px = 1

        if rr <= 0 or tau <= 0:
            for i in range(n):
                xi = int(x[i])
                yi = int(y[i])
                if xi < 0 or xi >= w or yi < 0 or yi >= h:
                    scores_out[i] = 0.0
                    continue

                ti = int(t[i])
                pi = 1 if int(p[i]) > 0 else -1

                idx0 = yi * w + xi
                last_ts[idx0] = np.uint64(ti)
                last_pol[idx0] = np.int8(pi)

                # pass-through score
                scores_out[i] = np.inf
            return

        inv_tau = 1.0 / float(tau)

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            if xi < 0 or xi >= w or yi < 0 or yi >= h:
                scores_out[i] = 0.0
                continue

            ti = int(t[i])
            pi = 1 if int(p[i]) > 0 else -1

            # Raw EBF score
            idx0 = yi * w + xi
            score = 0.0

            y0 = yi - rr
            if y0 < 0:
                y0 = 0
            y1 = yi + rr
            if y1 >= h:
                y1 = h - 1

            x0 = xi - rr
            if x0 < 0:
                x0 = 0
            x1 = xi + rr
            if x1 >= w:
                x1 = w - 1

            for yy in range(y0, y1 + 1):
                base = yy * w
                for xx in range(x0, x1 + 1):
                    if xx == xi and yy == yi:
                        continue

                    idx = base + xx

                    if int(last_pol[idx]) != pi:
                        continue

                    ts = int(last_ts[idx])
                    if ts == 0:
                        continue

                    dt = (ti - ts) if ti >= ts else (ts - ti)
                    if dt > tau:
                        continue

                    score += (float(tau - dt) * inv_tau)

            last_ts[idx0] = np.uint64(ti)
            last_pol[idx0] = np.int8(pi)

            # Update global EMA noise proxy (softly down-weight high scores)
            if last_t_global >= 0:
                dtg = ti - last_t_global
                if dtg > 0:
                    if use_soft_int == 1:
                        k = noise_weight_k
                        if k <= 1e-12:
                            k = 1e-12
                        ww = 1.0 / (1.0 + (score / k))
                        if ww < noise_weight_min:
                            ww = noise_weight_min
                        if ww > 1.0:
                            ww = 1.0
                    else:
                        ww = 1.0

                    inv_dt = ww / float(dtg)
                    if ema_inv_dt <= 0.0:
                        ema_inv_dt = inv_dt
                    else:
                        ema_inv_dt = (1.0 - ema_alpha) * ema_inv_dt + ema_alpha * inv_dt
            last_t_global = ti

            # Normalize by expected noise score scale
            # Warm-up: ema_inv_dt not ready yet
            if ema_inv_dt <= 0.0:
                scores_out[i] = score
                continue

            per_pixel_rate = ema_inv_dt / float(area)
            m = per_pixel_rate * float(tau)

            # Two scale models for A/B comparison:
            # - linear (old): exp_score = neigh_px * (m * 0.25)
            # - recent (new): exp_score = neigh_px * 0.5 * (1 - (1-exp(-m))/m)
            if use_recent_int == 1:
                if m <= 1e-6:
                    per_neigh = m * 0.25
                else:
                    per_neigh = 0.5 * (1.0 - ((1.0 - math.exp(-m)) / m))
                exp_score = float(neigh_px) * per_neigh
            else:
                exp_score = float(neigh_px) * (m * 0.25)
            if exp_score <= 0.0:
                scores_out[i] = score
                continue

            scores_out[i] = score / exp_score

    return ebfopt_scores_stream


def score_stream_ebf_optimized(
    ev: LabeledEvents,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_us: int,
    tb: TimeBase,
    _kernel_cache: dict[str, object],
    variant: str,
) -> np.ndarray:
    scores = np.empty((ev.t.shape[0],), dtype=np.float32)

    vid = str(variant).strip().lower()
    if vid == "equalw_linear":
        use_soft, use_recent = 0, 0
    elif vid == "softw_linear":
        use_soft, use_recent = 1, 0
    elif vid == "softw_linear_block":
        use_soft, use_recent = 1, 0
    elif vid == "softw_linear_blockmix":
        use_soft, use_recent = 1, 0
    elif vid == "softw_linear_purity":
        use_soft, use_recent = 1, 0
    elif vid == "softw_linear_polrate":
        use_soft, use_recent = 1, 0
    elif vid == "softw_linear_binrate":
        use_soft, use_recent = 1, 0
    elif vid == "softw_linear_same_minus_opp":
        use_soft, use_recent = 1, 0
    else:
        # default / softw_recent
        use_soft, use_recent = 1, 1

    # 注意：numba kernel 目前只覆盖“全局 rate proxy”三种变体；V4(block) 强制走 Python 实现。
    ker = None
    if vid in {"equalw_linear", "softw_linear", "softw_recent"}:
        ker_key = f"ker:{vid}"
        ker = _kernel_cache.get(ker_key)
        if ker is None:
            ker = _try_build_numba_kernel(use_soft_weight=use_soft, use_recent_scale=use_recent)
            _kernel_cache[ker_key] = ker

    if ker is not None:
        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
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

    from myevs.denoise.ops.base import Dims
    from myevs.denoise.ops.ebfopt_variants import create_ebfopt_variant
    from myevs.denoise.types import DenoiseConfig

    cfg = DenoiseConfig(
        method="ebf_optimized",
        pipeline=None,
        time_window_us=int(tau_us),
        radius_px=int(radius_px),
        min_neighbors=0.0,
        refractory_us=0,
        show_on=True,
        show_off=True,
    )
    op = create_ebfopt_variant(str(variant), Dims(width=int(width), height=int(height)), cfg, tb)
    n = int(ev.t.shape[0])
    for i in range(n):
        scores[i] = float(op.score_norm(int(ev.x[i]), int(ev.y[i]), int(ev.p[i]), int(ev.t[i])))
        if (i + 1) % 500000 == 0:
            print(f"scored: {i+1}/{n} (r={radius_px}, tau_us={tau_us})")
    return scores


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
]


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


def _roc_points_from_scores(
    y_true01: np.ndarray,
    y_score: np.ndarray,
    *,
    max_points: int,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (auc, thresholds, tp, fp, fpr, tpr) using standard ROC construction.

    Convention here matches myevs.metrics.roc_score_label:
    - Sort by score descending
    - Predicted positive for threshold thr: score >= thr
    - Adds the conventional starting point (0,0) at threshold=+inf
    """

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

    # add start point (0,0) at +inf
    tp_u = np.concatenate([np.asarray([0], dtype=np.int64), tp_u])
    fp_u = np.concatenate([np.asarray([0], dtype=np.int64), fp_u])
    thr_u = np.concatenate([np.asarray([np.inf], dtype=np.float64), thr_u])

    tpr_u = tp_u.astype(np.float64) / float(pos)
    fpr_u = fp_u.astype(np.float64) / float(neg)

    # exact AUC from full curve
    auc = float((getattr(np, "trapezoid", None) or np.trapz)(y=tpr_u, x=fpr_u))

    # downsample for output if needed
    if max_points is not None and int(max_points) > 0 and fpr_u.shape[0] > int(max_points):
        m = int(max_points)
        keep = np.unique(
            np.concatenate(
                [
                    np.asarray([0, fpr_u.shape[0] - 1], dtype=np.int64),
                    np.linspace(0, fpr_u.shape[0] - 1, num=m, dtype=np.int64),
                ]
            )
        )
        thr_u = thr_u[keep]
        tp_u = tp_u[keep]
        fp_u = fp_u[keep]
        fpr_u = fpr_u[keep]
        tpr_u = tpr_u[keep]

    return auc, thr_u, tp_u, fp_u, fpr_u, tpr_u


def _write_roc_rows(
    writer: csv.writer,
    *,
    tag: str,
    method: str,
    param: str,
    thresholds: np.ndarray,
    tp: np.ndarray,
    fp: np.ndarray,
    pos: int,
    neg: int,
    auc: float,
    esr_mean: float | None,
    esr_at_index: int,
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
                method,
                param,
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
            ]
        )


def _plot_roc_png(*, csv_path: str, png_path: str, title: str) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        print(f"skip plot (matplotlib unavailable): {type(e).__name__}: {e}")
        return

    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    if not rows:
        print(f"skip plot (empty csv): {csv_path}")
        return

    tags = sorted({(row.get("tag") or "") for row in rows if (row.get("tag") or "")})
    if not tags:
        print(f"skip plot (no tags): {csv_path}")
        return

    # Keep only top-3 tau curves per window diameter s, based on per-tag AUC.
    by_tag_auc: dict[str, float] = {}
    for row in rows:
        t = (row.get("tag") or "").strip()
        if not t or t in by_tag_auc:
            continue
        a = (row.get("auc") or "").strip()
        if not a:
            continue
        try:
            by_tag_auc[t] = float(a)
        except Exception:
            continue

    s_groups: dict[str, list[tuple[str, float]]] = {}
    for t, a in by_tag_auc.items():
        m = re.search(r"_s(\d+)_tau(\d+)", t)
        if not m:
            continue
        s_key = m.group(1)
        s_groups.setdefault(s_key, []).append((t, a))

    keep: set[str] = set()
    for _s_key, arr in s_groups.items():
        arr_sorted = sorted(arr, key=lambda x: x[1], reverse=True)
        for t, _a in arr_sorted[:3]:
            keep.add(t)

    if keep:
        tags = [t for t in tags if t in keep]

    def _legend_label(tag: str) -> str:
        # tag example: ebfopt_labelscore_s5_tau16000
        m = re.search(r"_s(\d+)_tau(\d+)", tag)
        s_s = m.group(1) if m else "?"
        tau_s = m.group(2) if m else "?"

        auc_val: float | None = None
        for row in rows:
            if row.get("tag") != tag:
                continue
            a = (row.get("auc") or "").strip()
            if a:
                try:
                    auc_val = float(a)
                except Exception:
                    auc_val = None
                break

        if auc_val is None:
            return f"ebfopt_s{s_s} tau{tau_s}"
        return f"ebfopt_s{s_s} tau{tau_s} (AUC={auc_val:.4f})"

    plt.figure(figsize=(8, 6), dpi=160)
    for tag in tags:
        fpr = [float(row["fpr"]) for row in rows if row.get("tag") == tag]
        tpr = [float(row["tpr"]) for row in rows if row.get("tag") == tag]
        if not fpr:
            continue
        plt.plot(fpr, tpr, linewidth=1.0, label=_legend_label(tag))

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(title)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(False)
    plt.legend(fontsize=7, ncol=1)
    os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Sweep EBF_optimized AUC(score+label) on labeled .npy (normalized score)."
    )
    ap.add_argument("--max-events", type=int, default=int(os.environ.get("EBF_MAX_EVENTS", "0")), help="0=all")
    ap.add_argument("--out-dir", default="data/ED24/myPedestrain_06/EBF_optimized", help="output directory")
    ap.add_argument(
        "--plot-only",
        action="store_true",
        help="Only regenerate PNG from existing ROC CSV (useful when CSV is open/locked on Windows).",
    )
    ap.add_argument("--width", type=int, default=346)
    ap.add_argument("--height", type=int, default=260)
    ap.add_argument("--tick-ns", type=float, default=1000.0)
    ap.add_argument("--s-list", default="3,5,7,9")
    ap.add_argument("--tau-us-list", default="8000,16000,32000,64000,128000,256000,512000,1024000")
    ap.add_argument("--roc-max-points", type=int, default=5000)

    ap.add_argument(
        "--beta-list",
        default=str(os.environ.get("MYEVS_EBFOPT_BLOCKMIX_BETA_LIST", "")),
        help=(
            "Optional beta sweep list for variant=softw_linear_blockmix (EBFV41). "
            "Comma-separated floats in [0,1], e.g. 0,0.05,0.1,0.2. "
            "If set, each beta writes outputs into a subdir: <out-dir>/beta_<beta>."
        ),
    )

    ap.add_argument(
        "--gamma-list",
        default=str(os.environ.get("MYEVS_EBFOPT_OPP_GAMMA_LIST", "")),
        help=(
            "Optional gamma sweep list for variant=softw_linear_same_minus_opp (EBFV5). "
            "Comma-separated floats >=0, e.g. 0,0.25,0.5,1.0. "
            "If set, each gamma writes outputs into a subdir: <out-dir>/gamma_<gamma>."
        ),
    )

    ap.add_argument(
        "--variant",
        default=str(os.environ.get("MYEVS_EBFOPT_VARIANT", "")),
        help=(
            "EBF_optimized variant id for A/B compare. "
            "Choices: equalw_linear(EBFV1) | softw_linear(EBFV2) | softw_recent(EBFV3) | softw_linear_block(EBFV4) | softw_linear_blockmix(EBFV41) | softw_linear_same_minus_opp(EBFV5) | softw_linear_purity(EBFV6) | softw_linear_polrate(EBFV7) | softw_linear_binrate(EBFV8) | softw_linear_timeconst_rateema(EBFV9). "
            "Aliases accepted: EBFV1/EBFV2/EBFV3/EBFV4/EBFV41/EBFV5/EBFV6/EBFV7/EBFV8/EBFV9. "
            "If empty, fall back to MYEVS_EBFOPT_SCALE_MODEL (linear/recent) -> softw_{model}."
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

    variant_raw = str(args.variant).strip()
    variant = variant_raw.lower()
    if not variant:
        scale_model = str(os.environ.get("MYEVS_EBFOPT_SCALE_MODEL", "recent")).strip().lower()
        variant = "softw_linear" if scale_model == "linear" else "softw_recent"

    # Accept short version-like aliases.
    variant = {
        "ebfv1": "equalw_linear",
        "ebfv2": "softw_linear",
        "ebfv3": "softw_recent",
        "ebfv4": "softw_linear_block",
        "ebfv41": "softw_linear_blockmix",
        "ebfv5": "softw_linear_same_minus_opp",
        "ebfv6": "softw_linear_purity",
        "ebfv7": "softw_linear_polrate",
        "ebfv8": "softw_linear_binrate",
        "ebfv9": "softw_linear_timeconst_rateema",
    }.get(variant, variant)

    if variant not in {
        "equalw_linear",
        "softw_linear",
        "softw_recent",
        "softw_linear_block",
        "softw_linear_blockmix",
        "softw_linear_same_minus_opp",
        "softw_linear_purity",
        "softw_linear_polrate",
        "softw_linear_binrate",
        "softw_linear_timeconst_rateema",
    }:
        raise SystemExit(
            f"--variant invalid: {variant_raw!r}. "
            "choices: equalw_linear(EBFV1), softw_linear(EBFV2), softw_recent(EBFV3), softw_linear_block(EBFV4), softw_linear_blockmix(EBFV41), softw_linear_same_minus_opp(EBFV5), softw_linear_purity(EBFV6), softw_linear_polrate(EBFV7), softw_linear_binrate(EBFV8), softw_linear_timeconst_rateema(EBFV9)"
        )

    def _parse_beta_list(raw: str) -> list[float]:
        s = str(raw).strip()
        if not s:
            return []
        out: list[float] = []
        seen: set[str] = set()
        for part in s.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                b = float(part)
            except Exception:
                raise SystemExit(f"--beta-list invalid item: {part!r}")
            if not (b == b):
                continue
            if b < 0.0:
                b = 0.0
            if b > 1.0:
                b = 1.0
            # de-dup by a stable string representation
            b_key = ("{:.6f}".format(b)).rstrip("0").rstrip(".")
            if b_key in seen:
                continue
            seen.add(b_key)
            out.append(b)
        return out

    beta_list = _parse_beta_list(str(args.beta_list))
    if beta_list and variant != "softw_linear_blockmix":
        raise SystemExit("--beta-list is only supported for --variant softw_linear_blockmix(EBFV41)")

    gamma_list = _parse_beta_list(str(args.gamma_list))
    if gamma_list and variant != "softw_linear_same_minus_opp":
        raise SystemExit("--gamma-list is only supported for --variant softw_linear_same_minus_opp(EBFV5)")

    if beta_list and gamma_list:
        raise SystemExit("--beta-list and --gamma-list cannot be used together")

    tb = TimeBase(tick_ns=float(args.tick_ns))
    s_list = _parse_int_list(args.s_list)
    for s in s_list:
        if s < 3 or s % 2 == 0:
            raise SystemExit(f"--s-list expects odd diameters >=3 (got {s})")
    tau_us_list = _parse_int_list(args.tau_us_list)

    def _fmt_s_part(ss: list[int]) -> str:
        return "s" + "_".join(str(int(x)) for x in ss)

    def _fmt_tau_part_ms(tau_list_us: list[int]) -> str:
        ms_parts: list[str] = []
        for tu in tau_list_us:
            if int(tu) % 1000 != 0:
                # fallback: keep microseconds explicitly
                ms_parts.append(f"{int(tu)}us")
            else:
                ms_parts.append(str(int(tu) // 1000))
        return "tau" + "_".join(ms_parts) + "ms"

    s_part = _fmt_s_part(s_list)
    tau_part = _fmt_tau_part_ms(tau_us_list)

    env_inputs = {
        "light": str(args.light),
        "mid": str(args.mid),
        "heavy": str(args.heavy),
    }

    base_out_dir = str(args.out_dir)

    def _run_one(*, out_dir: str, beta_note: str | None = None) -> None:
        os.makedirs(out_dir, exist_ok=True)

        roc_csv = {
            env: os.path.join(out_dir, f"roc_ebf_optimized_{variant}_{env}_labelscore_{s_part}_{tau_part}.csv")
            for env in env_inputs
        }
        roc_png = {
            env: os.path.join(out_dir, f"roc_ebf_optimized_{variant}_{env}_labelscore_{s_part}_{tau_part}.png")
            for env in env_inputs
        }

        title_suffix = f" beta={beta_note}" if beta_note else ""

        if bool(args.plot_only):
            for env in ("light", "mid", "heavy"):
                if not os.path.exists(roc_csv[env]):
                    print(f"skip plot-only (missing csv): env={env} path={roc_csv[env]}")
                    continue
                _plot_roc_png(
                    csv_path=roc_csv[env],
                    png_path=roc_png[env],
                    title=f"EBF_optimized[{variant}] ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us{title_suffix}",
                )
                print(f"saved: {roc_png[env]}")
            return

        kernel_cache: dict[str, object] = {}

        best_global = ("", -1.0)
        best_by_env: dict[str, tuple[str, float]] = {
            "light": ("", -1.0),
            "mid": ("", -1.0),
            "heavy": ("", -1.0),
        }

        write_enabled: dict[str, bool] = {}
        for env, p in roc_csv.items():
            os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
            try:
                with open(p, "w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(ROC_HEADER)
                write_enabled[env] = True
            except PermissionError:
                if os.path.exists(p):
                    print(f"warn: cannot write (locked). env={env} path={p} -> skip recompute, plot only")
                    write_enabled[env] = False
                else:
                    raise

        for env, in_path in env_inputs.items():
            if not write_enabled.get(env, True):
                _plot_roc_png(
                    csv_path=roc_csv[env],
                    png_path=roc_png[env],
                    title=f"EBF_optimized[{variant}] ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us{title_suffix}",
                )
                print(f"saved: {roc_png[env]}")
                continue

            ev = load_labeled_npy(in_path, max_events=int(args.max_events))
            n = int(ev.label.shape[0])
            pos = int(np.sum(ev.label))
            neg = int(n - pos)
            print(f"loaded: env={env} n={n} pos={pos} neg={neg} in={in_path}")

            with open(roc_csv[env], "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)

                for s in s_list:
                    r = (s - 1) // 2
                    for tau_us in tau_us_list:
                        tag = f"ebfopt_{variant}_labelscore_s{s}_tau{tau_us}"

                        scores = score_stream_ebf_optimized(
                            ev,
                            width=int(args.width),
                            height=int(args.height),
                            radius_px=int(r),
                            tau_us=int(tau_us),
                            tb=tb,
                            _kernel_cache=kernel_cache,
                            variant=variant,
                        )

                        auc, thr, tp, fp, _fpr, _tpr = _roc_points_from_scores(
                            ev.label,
                            scores,
                            max_points=int(args.roc_max_points),
                        )

                        # No-reference metric (E-MLB ESR) computed at the best-F1 operating point.
                        best_i = _best_f1_index(thr, tp, fp, pos=pos, neg=neg)
                        best_thr = float(thr[int(best_i)])
                        kept = scores >= best_thr
                        esr_mean = event_structural_ratio_mean_from_xy(
                            ev.x[kept],
                            ev.y[kept],
                            width=int(args.width),
                            height=int(args.height),
                            chunk_size=30000,
                        )

                        if auc > best_global[1]:
                            best_global = (tag, float(auc))
                        if auc > best_by_env[env][1]:
                            best_by_env[env] = (tag, float(auc))

                        _write_roc_rows(
                            w,
                            tag=tag,
                            method="ebf_optimized",
                            param="min-neighbors",
                            thresholds=thr,
                            tp=tp,
                            fp=fp,
                            pos=pos,
                            neg=neg,
                            auc=float(auc),
                            esr_mean=float(esr_mean),
                            esr_at_index=int(best_i),
                        )

                        print(f"auc={auc:.6f} env={env} s={s} tau_us={tau_us} points={int(thr.shape[0])}")

            _plot_roc_png(
                csv_path=roc_csv[env],
                png_path=roc_png[env],
                title=f"EBF_optimized ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us{title_suffix}",
            )

        print("=== BEST (by env) ===")
        for env in ("light", "mid", "heavy"):
            tag, auc = best_by_env[env]
            if auc >= 0:
                print(f"{env}: {tag} auc={auc:.6f}")
        print("=== BEST (global) ===")
        print(f"{best_global[0]} auc={best_global[1]:.6f}")
        for env in ("light", "mid", "heavy"):
            print(f"saved: {roc_csv[env]}")
            print(f"saved: {roc_png[env]}")

    if beta_list:
        prev = os.environ.get("MYEVS_EBFOPT_BLOCKMIX_BETA")
        try:
            for b in beta_list:
                beta_str = ("{:.6f}".format(float(b))).rstrip("0").rstrip(".")
                if not beta_str:
                    beta_str = "0"
                os.environ["MYEVS_EBFOPT_BLOCKMIX_BETA"] = str(float(b))
                out_dir = os.path.join(base_out_dir, f"beta_{beta_str}")
                print(f"\n=== BLOCKMIX BETA={beta_str} out={out_dir} ===")
                _run_one(out_dir=out_dir, beta_note=beta_str)
        finally:
            if prev is None:
                os.environ.pop("MYEVS_EBFOPT_BLOCKMIX_BETA", None)
            else:
                os.environ["MYEVS_EBFOPT_BLOCKMIX_BETA"] = prev
        return 0

    if gamma_list:
        prev = os.environ.get("MYEVS_EBFOPT_OPP_GAMMA")
        try:
            for g in gamma_list:
                gamma_str = ("{:.6f}".format(float(g))).rstrip("0").rstrip(".")
                if not gamma_str:
                    gamma_str = "0"
                os.environ["MYEVS_EBFOPT_OPP_GAMMA"] = str(float(g))
                out_dir = os.path.join(base_out_dir, f"gamma_{gamma_str}")
                print(f"\n=== SAME-MINUS-OPP GAMMA={gamma_str} out={out_dir} ===")
                _run_one(out_dir=out_dir, beta_note=f"gamma={gamma_str}")
        finally:
            if prev is None:
                os.environ.pop("MYEVS_EBFOPT_OPP_GAMMA", None)
            else:
                os.environ["MYEVS_EBFOPT_OPP_GAMMA"] = prev
        return 0

    _run_one(out_dir=base_out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
