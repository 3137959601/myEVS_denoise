from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from myevs.timebase import TimeBase


@dataclass(frozen=True)
class LabeledEvents:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    p: np.ndarray
    label: np.ndarray


@dataclass(frozen=True)
class RocBestPoint:
    tag: str
    thr: float
    f1: float
    auc: float | None


def _load_sweep_module():
    here = Path(__file__).resolve()
    sweep_path = here.parents[1] / "ED24_alg_evalu" / "sweep_ebf_labelscore_grid.py"
    spec = importlib.util.spec_from_file_location("_sweep_ebf_labelscore_grid", sweep_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"failed to load sweep module spec: {sweep_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_labeled_npy(path: str, *, start_events: int = 0, max_events: int = 0) -> LabeledEvents:
    arr = np.load(path, mmap_mode="r", allow_pickle=True)
    s0 = int(start_events)
    if s0 < 0:
        s0 = 0
    if s0 > 0:
        arr = arr[s0:]
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


def _tag_to_float(s: str) -> float:
    s = str(s)
    s = s.replace("p", ".").replace("m", "-")
    return float(s)


def _apply_env_from_tag(variant: str, tag: str) -> dict[str, str | None]:
    v = str(variant).strip().lower()
    old: dict[str, str | None] = {}

    def _set(k: str, val: str) -> None:
        old[k] = os.environ.get(k)
        os.environ[k] = str(val)

    if v in {"s36", "ebf_s36", "ebfs36", "surprise_occupancy", "surprise_stateoccupancy", "s28_stateoccupancy"}:
        m = re.search(r"_tr([0-9mp]+)", tag)
        if m:
            tau_rate_us = int(round(_tag_to_float(m.group(1))))
            if tau_rate_us < 0:
                tau_rate_us = 0
            _set("MYEVS_EBF_S36_TAU_RATE_US", str(tau_rate_us))
        return old

    if v in {
        "s37",
        "ebf_s37",
        "ebfs37",
        "surprise_occupancy_3state",
        "surprise_stateoccupancy_3state",
        "s28_stateoccupancy_3state",
    }:
        m = re.search(r"_tr([0-9mp]+)", tag)
        if m:
            tau_rate_us = int(round(_tag_to_float(m.group(1))))
            if tau_rate_us < 0:
                tau_rate_us = 0
            _set("MYEVS_EBF_S37_TAU_RATE_US", str(tau_rate_us))
        return old

    if v in {
        "s38",
        "ebf_s38",
        "ebfs38",
        "surprise_occupancy_nbocc",
        "surprise_stateocc_nbocc",
        "s28_stateocc_nbocc",
    }:
        m = re.search(r"_tr([0-9mp]+)", tag)
        if m:
            tau_rate_us = int(round(_tag_to_float(m.group(1))))
            if tau_rate_us < 0:
                tau_rate_us = 0
            _set("MYEVS_EBF_S38_TAU_RATE_US", str(tau_rate_us))
        return old

    if v in {
        "s39",
        "ebf_s39",
        "ebfs39",
        "surprise_occupancy_nbocc_mix",
        "surprise_stateocc_nbocc_mix",
        "s28_stateocc_nbocc_mix",
    }:
        m = re.search(r"_kn([0-9mp]+)", tag)
        if m:
            _set("MYEVS_EBF_S39_K_NBMIX", str(_tag_to_float(m.group(1))))
        m = re.search(r"_tr([0-9mp]+)", tag)
        if m:
            tau_rate_us = int(round(_tag_to_float(m.group(1))))
            if tau_rate_us < 0:
                tau_rate_us = 0
            _set("MYEVS_EBF_S39_TAU_RATE_US", str(tau_rate_us))
        return old

    if v in {
        "s40",
        "ebf_s40",
        "ebfs40",
        "surprise_occupancy_nbocc_mix_fuse_geom",
        "surprise_stateocc_nbocc_mix_fuse_geom",
    }:
        m = re.search(r"_kn([0-9mp]+)", tag)
        if m:
            _set("MYEVS_EBF_S40_K_NBMIX", str(_tag_to_float(m.group(1))))
        m = re.search(r"_tr([0-9mp]+)", tag)
        if m:
            tau_rate_us = int(round(_tag_to_float(m.group(1))))
            if tau_rate_us < 0:
                tau_rate_us = 0
            _set("MYEVS_EBF_S40_TAU_RATE_US", str(tau_rate_us))
        return old

    if v in {
        "s41",
        "ebf_s41",
        "ebfs41",
        "surprise_occupancy_nbocc_mix_pow2",
        "surprise_stateocc_nbocc_mix_pow2",
    }:
        m = re.search(r"_kn([0-9mp]+)", tag)
        if m:
            _set("MYEVS_EBF_S41_K_NBMIX", str(_tag_to_float(m.group(1))))
        m = re.search(r"_tr([0-9mp]+)", tag)
        if m:
            tau_rate_us = int(round(_tag_to_float(m.group(1))))
            if tau_rate_us < 0:
                tau_rate_us = 0
            _set("MYEVS_EBF_S41_TAU_RATE_US", str(tau_rate_us))
        return old

    if v in {
        "s42",
        "ebf_s42",
        "ebfs42",
        "surprise_occupancy_nbocc_mix_gated_self2",
        "surprise_stateocc_nbocc_mix_gated_self2",
    }:
        m = re.search(r"_kn([0-9mp]+)", tag)
        if m:
            _set("MYEVS_EBF_S42_K_NBMIX", str(_tag_to_float(m.group(1))))
        m = re.search(r"_tr([0-9mp]+)", tag)
        if m:
            tau_rate_us = int(round(_tag_to_float(m.group(1))))
            if tau_rate_us < 0:
                tau_rate_us = 0
            _set("MYEVS_EBF_S42_TAU_RATE_US", str(tau_rate_us))
        return old

    if v in {
        "s43",
        "ebf_s43",
        "ebfs43",
        "surprise_occupancy_nbocc_mix_u2",
        "surprise_stateocc_nbocc_mix_u2",
    }:
        m = re.search(r"_kn([0-9mp]+)", tag)
        if m:
            _set("MYEVS_EBF_S43_K_NBMIX", str(_tag_to_float(m.group(1))))
        m = re.search(r"_tr([0-9mp]+)", tag)
        if m:
            tau_rate_us = int(round(_tag_to_float(m.group(1))))
            if tau_rate_us < 0:
                tau_rate_us = 0
            _set("MYEVS_EBF_S43_TAU_RATE_US", str(tau_rate_us))
        return old

    if v in {
        "s44",
        "ebf_s44",
        "ebfs44",
        "ebf_labelscore_selfocc_div_u2",
        "ebf_labelscore_selfocc_u2",
    }:
        m = re.search(r"_tr([0-9mp]+)", tag)
        if m:
            tau_rate_us = int(round(_tag_to_float(m.group(1))))
            if tau_rate_us < 0:
                tau_rate_us = 0
            _set("MYEVS_EBF_S44_TAU_RATE_US", str(tau_rate_us))
        return old

    if v in {
        "s50",
        "ebf_s50",
        "ebfs50",
        "ebf_labelscore_selfocc_supportboost_div_u2",
        "ebf_labelscore_selfocc_supportboost_u2",
    }:
        m = re.search(r"_b([0-9mp]+)", tag)
        if m:
            beta = float(_tag_to_float(m.group(1)))
            _set("MYEVS_EBF_S50_BETA", str(beta))
        m = re.search(r"_c([0-9mp]+)", tag)
        if m:
            cnt0 = int(round(_tag_to_float(m.group(1))))
            if cnt0 < 1:
                cnt0 = 1
            _set("MYEVS_EBF_S50_CNT0", str(cnt0))
        m = re.search(r"_tr([0-9mp]+)", tag)
        if m:
            tau_rate_us = int(round(_tag_to_float(m.group(1))))
            if tau_rate_us < 0:
                tau_rate_us = 0
            _set("MYEVS_EBF_S50_TAU_RATE_US", str(tau_rate_us))
        return old

    return old


def _restore_env(old: dict[str, str | None]) -> None:
    for k, prev in old.items():
        if prev is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(prev)


def _variant_tag_contains(variant: str) -> str | None:
    v = str(variant).strip().lower()
    if v in {"ebf", "v0", "baseline"}:
        return None
    if v.startswith("ebf_"):
        v2 = v
    else:
        v2 = f"ebf_{v}"

    m = re.match(r"^(ebf_s\d+)(?:\b|_)", v2)
    if m:
        return m.group(1) + "_"
    return None


def _read_best_point(
    roc_csv: str,
    *,
    s: int,
    tau_us: int,
    tag: str | None,
    tag_contains: str | None,
) -> RocBestPoint:
    p = Path(roc_csv)
    if not p.exists():
        raise SystemExit(f"roc csv not found: {roc_csv}")

    suffix = f"_labelscore_s{s}_tau{tau_us}"

    best_by_tag: dict[str, RocBestPoint] = {}
    with p.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            t = (row.get("tag") or "").strip()
            if not t or suffix not in t:
                continue
            if tag is not None and t != tag:
                continue
            if tag is None and tag_contains is not None and tag_contains not in t:
                continue

            try:
                f1 = float(row["f1"])
            except Exception:
                continue

            try:
                thr = float(row["value"])
            except Exception:
                continue

            a = (row.get("auc") or "").strip()
            if a:
                try:
                    auc: float | None = float(a)
                except Exception:
                    auc = None
            else:
                auc = None

            cur = best_by_tag.get(t)
            if cur is None or f1 > cur.f1:
                best_by_tag[t] = RocBestPoint(tag=t, thr=thr, f1=f1, auc=auc)

    if not best_by_tag:
        raise SystemExit(
            f"no matching rows found in ROC CSV for suffix={suffix!r} and tag={tag!r}\nfile={roc_csv}"
        )

    if tag is not None:
        return best_by_tag[tag]

    return max(best_by_tag.values(), key=lambda x: x.f1)


def _load_hotmask(path: str, *, width: int, height: int) -> np.ndarray:
    arr = np.load(path)
    a = np.asarray(arr)
    if a.ndim == 2:
        if a.shape != (height, width):
            raise SystemExit(f"hotmask shape mismatch: expected {(height, width)}, got {a.shape}")
        m = a.astype(np.uint8, copy=False).reshape(-1)
        return (m != 0).astype(np.uint8)
    if a.ndim == 1:
        if a.size != width * height:
            raise SystemExit(f"hotmask size mismatch: expected {width*height}, got {a.size}")
        m = a.astype(np.uint8, copy=False)
        return (m != 0).astype(np.uint8)
    raise SystemExit(f"unsupported hotmask ndim: {a.ndim}")


def _dilate_mask(mask_hw: np.ndarray, *, r: int) -> np.ndarray:
    r = int(r)
    if r <= 0:
        return mask_hw.astype(np.bool_, copy=False)

    h, w = mask_hw.shape
    m = mask_hw.astype(np.bool_, copy=False)
    out = m.copy()
    for dy in range(-r, r + 1):
        y0 = max(0, -dy)
        y1 = min(h, h - dy)
        ys = slice(y0, y1)
        yt = slice(y0 + dy, y1 + dy)
        for dx in range(-r, r + 1):
            x0 = max(0, -dx)
            x1 = min(w, w - dx)
            xs = slice(x0, x1)
            xt = slice(x0 + dx, x1 + dx)
            out[yt, xt] |= m[ys, xs]
    return out


def _topk_mask_from_counts(counts: np.ndarray, *, topk: int) -> np.ndarray:
    topk = int(topk)
    if topk <= 0:
        return np.zeros_like(counts, dtype=np.bool_)
    if topk >= int(counts.size):
        return np.ones_like(counts, dtype=np.bool_)

    idx = np.argpartition(counts, -topk)[-topk:]
    out = np.zeros_like(counts, dtype=np.bool_)
    out[idx] = True
    return out


def _tau_rate_us_from_tag(tag: str) -> int:
    m = re.search(r"_tr(\d+)_", tag)
    if not m:
        return 0
    v = int(m.group(1))
    if v < 0:
        v = 0
    return v


def _tau_rate_us_for_variant(variant: str, tag: str, tau_us: int) -> int:
    """Return tau_rate_us used for debug u normalization.

    Most variants encode tau_rate via `_tr...` in tag; if missing we default to tau.

    s51 is different: its kernel ties tau_rate to tau internally (tr=tau/2) and does
    not expose it as a sweep hyperparameter, so tags don't include `_tr...`. To keep
    debug `u_self` consistent with the kernel, mirror that rule here.
    """

    v = str(variant).strip().lower()
    if v in {
        "s51",
        "ebf_s51",
        "ebfs51",
        "ebf_labelscore_selfocc_supportboost_autobeta_div_u2",
        "ebf_labelscore_selfocc_supportboost_autobeta_u2",
        "s52",
        "ebf_s52",
        "ebfs52",
        "ebf_labelscore_selfocc_supportboost_autobeta_mixgateopp_div_u2",
        "ebf_labelscore_selfocc_supportboost_autobeta_mixgateopp_u2",
        "s53",
        "ebf_s53",
        "ebfs53",
        "ebf_labelscore_selfocc_supportboost_autobeta_eventmixgateopp_div_u2",
        "ebf_labelscore_selfocc_supportboost_autobeta_eventmixgateopp_u2",
        "s54",
        "ebf_s54",
        "ebfs54",
        "ebf_labelscore_selfocc_supportboost_autobeta_eventmixgateopp_root4_div_u2",
        "ebf_labelscore_selfocc_supportboost_autobeta_eventmixgateopp_root4_u2",
        "s55",
        "ebf_s55",
        "ebfs55",
        "ebf_labelscore_selfocc_supportboost_autobeta_eventmixgateopp_supportlerp_div_u2",
        "ebf_labelscore_selfocc_supportboost_autobeta_eventmixgateopp_supportlerp_u2",
    }:
        tr = int(tau_us) // 2
        if tr <= 0:
            tr = int(tau_us)
        return int(tr)

    tr = _tau_rate_us_from_tag(str(tag))
    if tr > 0:
        return int(tr)
    return int(tau_us)


def _k_nbmix_from_tag(tag: str) -> float | None:
    m = re.search(r"_kn([0-9mp]+)", tag)
    if not m:
        return None
    try:
        return float(_tag_to_float(m.group(1)))
    except Exception:
        return None


def dump_u_events(
    *,
    labeled_npy: str,
    out_csv: str,
    width: int,
    height: int,
    tick_ns: float,
    s: int,
    tau_us: int,
    start_events: int,
    max_events: int,
    samepol_dt_thr: float,
    toggle_dt_thr: float,
    nb_radius: int,
    nb_win_us: int,
    cluster_k: int,
    nearhot_r: int,
    highrate_topk: int,
    hotmask_npy: str,
    variant: str,
    roc_csv: str,
    tag: str | None,
    thr: float | None,
) -> None:
    if s < 3 or (s % 2) != 1:
        raise SystemExit(f"--s must be odd diameter >=3 (got {s})")

    tb = TimeBase(float(tick_ns))
    tau_ticks = int(tb.us_to_ticks(int(tau_us)))
    if tau_ticks <= 0:
        tau_ticks = 1

    ev = load_labeled_npy(labeled_npy, start_events=int(start_events), max_events=int(max_events))
    n = int(ev.label.size)

    hotmask: np.ndarray | None = None
    near_hotmask: np.ndarray | None = None
    if hotmask_npy:
        hotmask = _load_hotmask(hotmask_npy, width=int(width), height=int(height))
        hotmask_hw = hotmask.reshape((int(height), int(width))).astype(np.bool_, copy=False)
        near_hotmask = _dilate_mask(hotmask_hw, r=int(nearhot_r)).reshape(-1)

    # Pick best-F1 point if not provided.
    best_tag = tag or ""
    if thr is None:
        if not roc_csv:
            raise SystemExit("need --roc-csv when --thr is not provided")
        tag_contains = _variant_tag_contains(str(variant))
        best = _read_best_point(roc_csv, s=int(s), tau_us=int(tau_us), tag=tag, tag_contains=tag_contains)
        best_tag = best.tag
        thr = float(best.thr)
    else:
        if not best_tag:
            best_tag = "(manual_thr)"

    # Compute scores and kept mask.
    sweep = _load_sweep_module()
    radius_px = (int(s) - 1) // 2
    old_env = _apply_env_from_tag(variant, best_tag)
    try:
        scores = sweep.score_stream_ebf(
            ev,
            width=int(width),
            height=int(height),
            radius_px=int(radius_px),
            tau_us=int(tau_us),
            tb=tb,
            _kernel_cache={},
            variant=str(variant),
        )
    finally:
        _restore_env(old_env)

    if scores.shape[0] != n:
        raise SystemExit(f"scores length mismatch: scores={scores.shape[0]} events={n}")

    thr_f = float(thr)
    kept = scores >= thr_f

    # Prepare high-rate mask from noise-only counts.
    label = ev.label.astype(np.bool_, copy=False)
    is_noise = ~label

    counts_noise = np.zeros((int(width) * int(height),), dtype=np.int32)
    for i in range(n):
        if not bool(is_noise[i]):
            continue
        x = int(ev.x[i])
        y = int(ev.y[i])
        if x < 0 or x >= int(width) or y < 0 or y >= int(height):
            continue
        idx = y * int(width) + x
        counts_noise[idx] += 1
    highrate_mask = _topk_mask_from_counts(counts_noise, topk=int(highrate_topk))

    # Shared streaming state for categories + occupancy.
    last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
    last_p = np.zeros((int(width) * int(height),), dtype=np.int8)
    hot_state = np.zeros((int(width) * int(height),), dtype=np.int32)

    # Global rate EMA state for debug (events/tick), matching s28/s36/s37/s38 kernels.
    r_ema = 0.0
    prev_t = 0

    nb_win_ticks = int(tb.us_to_ticks(int(nb_win_us)))

    # tau_rate for u normalization (debug-only): variant-aware.
    tau_rate_us = _tau_rate_us_for_variant(str(variant), best_tag, int(tau_us))
    tau_rate_ticks = int(tb.us_to_ticks(int(tau_rate_us)))
    if tau_rate_ticks <= 0:
        tau_rate_ticks = tau_ticks

    # Optional s39 strength from tag (for debug-only columns).
    k_nbmix = _k_nbmix_from_tag(best_tag)
    if k_nbmix is None or not np.isfinite(float(k_nbmix)) or float(k_nbmix) < 0.0:
        k_nbmix = 0.0

    radius_px = (int(s) - 1) // 2
    if radius_px < 0:
        radius_px = 0
    if radius_px > 8:
        radius_px = 8

    inv_tau = 1.0 / float(max(1, int(tau_ticks)))
    n_pix = float(int(width) * int(height))
    eps = 1e-6

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "i",
                "t",
                "x",
                "y",
                "p",
                "label",
                "score",
                "kept",
                "cat",
                "u",
                "u_eff",
                "u_self",
                "u_nb",
                "u_nb_mix",
                "mix",
                "hot_state",
                "dt0_norm",
                "nb_cnt",
                "m",
                "raw",
                "raw_all",
                "raw_opp",
                "r_ema",
                "r_pix",
                "r_eff",
                "mu",
                "var",
                "z_dbg",
            ],
        )
        w.writeheader()

        for i in range(n):
            xi = int(ev.x[i])
            yi = int(ev.y[i])
            ti = int(ev.t[i])
            pi = 1 if int(ev.p[i]) > 0 else -1

            if xi < 0 or xi >= int(width) or yi < 0 or yi >= int(height):
                w.writerow(
                    {
                        "i": i,
                        "t": ti,
                        "x": xi,
                        "y": yi,
                        "p": pi,
                        "label": int(ev.label[i]),
                        "score": float(scores[i]),
                        "kept": int(bool(kept[i])),
                        "cat": "oob",
                        "u": "",
                        "hot_state": "",
                        "dt0_norm": "",
                        "nb_cnt": "",
                    }
                )
                continue

            idx0 = yi * int(width) + xi

            # Update global rate EMA (events/tick), matching kernels.
            if i > 0:
                dtg = ti - prev_t
                if dtg > 0:
                    inst = 1.0 / float(dtg)
                    a_rate = 1.0 - float(np.exp(-float(dtg) / float(max(1, int(tau_rate_ticks)))))
                    r_ema = float(r_ema + a_rate * (inst - r_ema))
            prev_t = ti

            # Occupancy update (matches s36/s37 kernels).
            ts0 = int(last_ts[idx0])
            h0 = int(hot_state[idx0])
            dt0 = tau_ticks if ts0 == 0 else (ti - ts0)
            if dt0 < 0:
                dt0 = -dt0

            if dt0 != 0:
                h0 = h0 - dt0
                if h0 < 0:
                    h0 = 0
            inc = tau_ticks - dt0
            if inc > 0:
                h0 = h0 + inc
                if h0 > 2147483647:
                    h0 = 2147483647

            u_self = float(h0) / float(h0 + tau_rate_ticks)

            # Neighborhood recency masses in the same window as raw support (radius_px).
            raw_w = 0
            raw_all_w = 0
            rr = int(radius_px)
            if rr > 0:
                y0 = yi - rr
                if y0 < 0:
                    y0 = 0
                y1 = yi + rr
                if y1 >= int(height):
                    y1 = int(height) - 1
                x0 = xi - rr
                if x0 < 0:
                    x0 = 0
                x1 = xi + rr
                if x1 >= int(width):
                    x1 = int(width) - 1

                m = (x1 - x0 + 1) * (y1 - y0 + 1) - 1
                if m < 1:
                    m = 1

                for yy in range(y0, y1 + 1):
                    base = yy * int(width)
                    for xx in range(x0, x1 + 1):
                        if xx == xi and yy == yi:
                            continue
                        nidx = base + xx
                        tsn = int(last_ts[nidx])
                        if tsn == 0:
                            continue
                        dt = ti - tsn
                        if dt < 0:
                            dt = -dt
                        if dt > tau_ticks:
                            continue
                        rec = tau_ticks - dt
                        if rec <= 0:
                            continue
                        raw_all_w += int(rec)
                        if int(last_p[nidx]) == pi:
                            raw_w += int(rec)
            else:
                m = 1

            raw = float(raw_w) * inv_tau
            raw_all = float(raw_all_w) * inv_tau
            raw_opp = float(max(0, int(raw_all_w) - int(raw_w))) * inv_tau

            u_nb = raw_all / (raw_all + float(m) + eps)
            mix = 0.0
            if raw_all_w > 0:
                mix = float(max(0, int(raw_all_w) - int(raw_w))) / float(raw_all_w)
            if mix < 0.0:
                mix = 0.0
            if mix > 1.0:
                mix = 1.0

            u_nb_mix = float(k_nbmix) * float(u_nb) * float(mix)
            if u_nb_mix < 0.0:
                u_nb_mix = 0.0
            if u_nb_mix > 1.0:
                u_nb_mix = 1.0

            # Effective occupancy used by the null-model rate modulation.
            v = str(variant).strip().lower()
            if v in {
                "s38",
                "ebf_s38",
                "ebfs38",
                "surprise_occupancy_nbocc",
                "surprise_stateocc_nbocc",
                "s28_stateocc_nbocc",
            }:
                u_eff = 1.0 - (1.0 - float(u_self)) * (1.0 - float(u_nb))
            elif v in {
                "s39",
                "ebf_s39",
                "ebfs39",
                "surprise_occupancy_nbocc_mix",
                "surprise_stateocc_nbocc_mix",
                "s28_stateocc_nbocc_mix",
            }:
                u_eff = 1.0 - (1.0 - float(u_self)) * (1.0 - float(u_nb_mix))
            else:
                u_eff = float(u_self)

            if u_eff < 0.0:
                u_eff = 0.0
            if u_eff > 1.0:
                u_eff = 1.0

            # Debug r_eff + mu/var + z computed from u_eff (mirrors kernels for s36/s37/s38/s39).
            r_pix = float(r_ema) / float(max(1.0, n_pix))
            if r_pix < 0.0:
                r_pix = 0.0

            mult = None
            if v in {
                "s37",
                "ebf_s37",
                "ebfs37",
                "surprise_occupancy_3state",
                "surprise_stateoccupancy_3state",
                "s28_stateoccupancy_3state",
            }:
                u0 = float(u_self)
                if u0 < (1.0 / 3.0):
                    mult = 1.0
                elif u0 < (2.0 / 3.0):
                    mult = 2.0
                else:
                    mult = 4.0
                r_eff = float(r_pix) * float(mult)
            else:
                s1 = 1.0 + float(u_eff)
                r_eff = float(r_pix) * float(s1 * s1)

            a = float(r_eff) * float(max(1, int(tau_ticks)))
            if a < 1e-3:
                ew = 0.5 * a - (a * a) / 6.0
                ew2 = (a / 3.0) - (a * a) / 12.0
            else:
                ea = float(np.exp(-a))
                ew = 1.0 - (1.0 - ea) / a
                ew2 = (a * a - 2.0 * a + 2.0 - 2.0 * ea) / (a * a)

            mu_per = 0.5 * ew
            e2_per = 0.5 * ew2
            var_per = e2_per - mu_per * mu_per
            if var_per < 0.0:
                var_per = 0.0

            mu = float(m) * float(mu_per)
            var = float(m) * float(var_per)
            denom = float(np.sqrt(float(var) + eps))
            z_dbg = (float(raw) - float(mu)) / float(max(eps, denom))

            # Category features.
            is_hot = bool(hotmask[idx0] != 0) if hotmask is not None else False
            is_near_hot = bool(near_hotmask[idx0]) if near_hotmask is not None else False
            is_highrate = bool(highrate_mask[idx0])

            prev_p = int(last_p[idx0])

            dt_norm = 1.0
            samepol = False
            toggle = False
            if ts0 != 0:
                dt = ti - ts0
                if dt < 0:
                    dt = 0
                dt_norm = float(dt) / float(max(1, tau_ticks))
                samepol = prev_p == pi
                toggle = prev_p == -pi

            nb_cnt = 0
            if nb_win_ticks > 0 and int(nb_radius) > 0:
                r = int(nb_radius)
                for dy in range(-r, r + 1):
                    yy = yi + dy
                    if yy < 0 or yy >= int(height):
                        continue
                    for dx in range(-r, r + 1):
                        xx = xi + dx
                        if dx == 0 and dy == 0:
                            continue
                        if xx < 0 or xx >= int(width):
                            continue
                        nidx = yy * int(width) + xx
                        tsn = int(last_ts[nidx])
                        if tsn == 0:
                            continue
                        dt_nb = ti - tsn
                        if dt_nb < 0:
                            dt_nb = 0
                        if dt_nb <= nb_win_ticks:
                            nb_cnt += 1

            # Priority-based exclusive categorization.
            if is_hot:
                cat = "hotmask"
            elif is_near_hot:
                cat = "near_hotmask"
            elif is_highrate:
                cat = "highrate_pixel"
            elif samepol and dt_norm < float(samepol_dt_thr):
                cat = "samepol_shortdt"
            elif toggle and dt_norm < float(toggle_dt_thr):
                cat = "toggle_shortdt"
            elif nb_cnt == 0:
                cat = "isolated_nb0"
            elif nb_cnt >= int(cluster_k):
                cat = "cluster_nb_ge_k"
            else:
                cat = "other"

            # Commit state updates (shared for next events).
            last_ts[idx0] = np.uint64(ti)
            last_p[idx0] = np.int8(pi)
            hot_state[idx0] = np.int32(h0)

            w.writerow(
                {
                    "i": i,
                    "t": ti,
                    "x": xi,
                    "y": yi,
                    "p": pi,
                    "label": int(ev.label[i]),
                    "score": float(scores[i]),
                    "kept": int(bool(kept[i])),
                    "cat": cat,
                    # Backward compatibility: keep "u" as self occupancy.
                    "u": float(u_self),
                    "u_eff": float(u_eff),
                    "u_self": float(u_self),
                    "u_nb": float(u_nb),
                    "u_nb_mix": float(u_nb_mix),
                    "mix": float(mix),
                    "hot_state": int(h0),
                    "dt0_norm": float(dt_norm),
                    "nb_cnt": int(nb_cnt),
                    "m": int(m),
                    "raw": float(raw),
                    "raw_all": float(raw_all),
                    "raw_opp": float(raw_opp),
                    "r_ema": float(r_ema),
                    "r_pix": float(r_pix),
                    "r_eff": float(r_eff),
                    "mu": float(mu),
                    "var": float(var),
                    "z_dbg": float(z_dbg),
                }
            )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Replay the best-F1 ROC operating point and dump per-event u (state occupancy) values to CSV. "
            "This is intended for analyzing s36/s37/s38/s39 behavior on hotmask/noise categories."
        )
    )

    ap.add_argument("--labeled-npy", required=True, help="Input labeled .npy with fields t/x/y/p/label")
    ap.add_argument("--out-csv", required=True, help="Output per-event CSV path")

    ap.add_argument("--width", type=int, default=346)
    ap.add_argument("--height", type=int, default=260)
    ap.add_argument("--tick-ns", type=float, default=1000.0)

    ap.add_argument("--s", type=int, default=9)
    ap.add_argument("--tau-us", type=int, default=128000)
    ap.add_argument(
        "--start-events",
        type=int,
        default=200000,
        help="Skip first N events (default=200k => focus on seg1: [200k,400k))",
    )
    ap.add_argument(
        "--max-events",
        type=int,
        default=200000,
        help="Max events to load after start-events (default=200k => one segment)",
    )

    ap.add_argument("--samepol-dt-thr", type=float, default=0.01)
    ap.add_argument("--toggle-dt-thr", type=float, default=0.01)
    ap.add_argument("--nb-radius", type=int, default=1)
    ap.add_argument("--nb-win-us", type=int, default=2000)
    ap.add_argument("--cluster-k", type=int, default=3)
    ap.add_argument("--nearhot-r", type=int, default=1)
    ap.add_argument("--highrate-topk", type=int, default=32768)

    ap.add_argument("--hotmask-npy", default="")

    ap.add_argument(
        "--variant",
        required=True,
        help="Variant name (expected: s36|s37|s38|s39|s40|s41|s42|s43|s44|s50|s51). Used for reproducing the tag via env vars.",
    )
    ap.add_argument("--roc-csv", default="", help="ROC CSV path (used to auto-pick best-F1 thr/tag)")
    ap.add_argument("--tag", default="", help="Optional exact tag inside ROC CSV")
    ap.add_argument("--thr", type=float, default=float("nan"), help="Manual threshold; overrides ROC selection")

    args = ap.parse_args()

    thr = None if not np.isfinite(float(args.thr)) else float(args.thr)
    tag = (args.tag or "").strip() or None

    dump_u_events(
        labeled_npy=str(args.labeled_npy),
        out_csv=str(args.out_csv),
        width=int(args.width),
        height=int(args.height),
        tick_ns=float(args.tick_ns),
        s=int(args.s),
        tau_us=int(args.tau_us),
        start_events=int(args.start_events),
        max_events=int(args.max_events),
        samepol_dt_thr=float(args.samepol_dt_thr),
        toggle_dt_thr=float(args.toggle_dt_thr),
        nb_radius=int(args.nb_radius),
        nb_win_us=int(args.nb_win_us),
        cluster_k=int(args.cluster_k),
        nearhot_r=int(args.nearhot_r),
        highrate_topk=int(args.highrate_topk),
        hotmask_npy=str(args.hotmask_npy).strip(),
        variant=str(args.variant),
        roc_csv=str(args.roc_csv).strip(),
        tag=tag,
        thr=thr,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
