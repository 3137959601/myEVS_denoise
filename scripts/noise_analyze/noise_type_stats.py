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
    # Reuse the exact scoring implementation used by the sweep script.
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

        # Heuristic: if first column is binary, assume [label, t, y, x, p].
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
    # tag uses p as dot and m as minus (e.g. 0p1, m0p2)
    s = str(s)
    s = s.replace("p", ".").replace("m", "-")
    return float(s)


def _apply_env_from_tag(variant: str, tag: str) -> dict[str, str | None]:
    """Set env vars so the sweep scoring reproduces the given tag.

    Returns old values for restoration.
    """

    v = str(variant).strip().lower()
    old: dict[str, str | None] = {}

    def _set(k: str, val: str) -> None:
        old[k] = os.environ.get(k)
        os.environ[k] = str(val)

    if v in {"s14", "ebf_s14", "ebfs14"}:
        m = re.search(r"_a([0-9mp]+)_raw([0-9mp]+)", tag)
        if not m:
            raise SystemExit(f"cannot parse s14 params from tag: {tag}")
        _set("MYEVS_EBF_S14_ALPHA", str(_tag_to_float(m.group(1))))
        _set("MYEVS_EBF_S14_RAW_THR", str(_tag_to_float(m.group(2))))
        return old

    if v in {"s25", "ebf_s25", "ebfs25"}:
        m = re.search(r"_a([0-9mp]+)_raw([0-9mp]+)_dt([0-9mp]+)_rraw([0-9mp]+)_g([0-9mp]+)", tag)
        if not m:
            raise SystemExit(f"cannot parse s25 params from tag: {tag}")
        _set("MYEVS_EBF_S25_ALPHA", str(_tag_to_float(m.group(1))))
        _set("MYEVS_EBF_S25_RAW_THR", str(_tag_to_float(m.group(2))))
        _set("MYEVS_EBF_S25_DT_THR", str(_tag_to_float(m.group(3))))
        _set("MYEVS_EBF_S25_REF_RAW_THR", str(_tag_to_float(m.group(4))))
        _set("MYEVS_EBF_S25_GAMMA", str(_tag_to_float(m.group(5))))
        return old

    if v in {"s28", "ebf_s28", "ebfs28", "noise_surprise", "surprise_zscore", "surprise_z"}:
        m = re.search(r"_tr([0-9mp]+)", tag)
        if m:
            tau_rate_us = int(round(_tag_to_float(m.group(1))))
            if tau_rate_us < 0:
                tau_rate_us = 0
            _set("MYEVS_EBF_S28_TAU_RATE_US", str(tau_rate_us))
        return old

    if v in {"s35", "ebf_s35", "ebfs35", "surprise_pixelstate", "surprise_pixel_state", "s28_pixelstate"}:
        # tag examples:
        #   ebf_s35_g1_h8_labelscore_s9_tau128000
        #   ebf_s35_tr128000_g1_h8_labelscore_s9_tau128000
        m = re.search(r"_g([0-9mp]+)", tag)
        if m:
            _set("MYEVS_EBF_S35_GAMMA", str(_tag_to_float(m.group(1))))
        m = re.search(r"_h([0-9mp]+)", tag)
        if m:
            _set("MYEVS_EBF_S35_HMAX", str(_tag_to_float(m.group(1))))
        m = re.search(r"_tr([0-9mp]+)", tag)
        if m:
            tau_rate_us = int(round(_tag_to_float(m.group(1))))
            if tau_rate_us < 0:
                tau_rate_us = 0
            _set("MYEVS_EBF_S35_TAU_RATE_US", str(tau_rate_us))
        return old

    if v in {"s36", "ebf_s36", "ebfs36", "surprise_occupancy", "surprise_stateoccupancy", "s28_stateoccupancy"}:
        # tag examples:
        #   ebf_s36_labelscore_s9_tau128000
        #   ebf_s36_tr64000_labelscore_s9_tau128000
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
        # tag examples:
        #   ebf_s37_labelscore_s9_tau128000
        #   ebf_s37_tr64000_labelscore_s9_tau128000
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
        # tag examples:
        #   ebf_s38_labelscore_s9_tau128000
        #   ebf_s38_tr64000_labelscore_s9_tau128000
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
        # tag examples:
        #   ebf_s39_kn1_labelscore_s9_tau128000
        #   ebf_s39_tr64000_kn1_labelscore_s9_tau128000
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
        # tag examples:
        #   ebf_s40_kn1_labelscore_s9_tau128000
        #   ebf_s40_tr64000_kn1_labelscore_s9_tau128000
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
        # tag examples:
        #   ebf_s41_kn1_labelscore_s9_tau128000
        #   ebf_s41_tr64000_kn1_labelscore_s9_tau128000
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
        # Examples:
        #   ebf_s42_kn1_labelscore_s9_tau128000
        #   ebf_s42_tr64000_kn1_labelscore_s9_tau128000
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
        # Examples:
        #   ebf_s43_kn1_labelscore_s9_tau128000
        #   ebf_s43_tr64000_kn1_labelscore_s9_tau128000
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
        # Examples:
        #   ebf_s44_labelscore_s9_tau128000
        #   ebf_s44_tr64000_labelscore_s9_tau128000
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
        # Examples:
        #   ebf_s50_b0_c4_labelscore_s7_tau64000
        #   ebf_s50_tr32000_b1_c8_labelscore_s7_tau64000
        m = re.search(r"_b([0-9mp]+)", tag)
        if m:
            _set("MYEVS_EBF_S50_BETA", str(_tag_to_float(m.group(1))))
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

    if v in {"s33", "ebf_s33", "ebfs33", "surprise_abnhot", "surprise_abnhot_penalty", "s28_abnhot_penalty"}:
        # tag examples:
        #   ebf_s33_b0p75_labelscore_s9_tau128000
        #   ebf_s33_tr128000_b0p75_labelscore_s9_tau128000
        m = re.search(r"_b([0-9mp]+)", tag)
        if m:
            _set("MYEVS_EBF_S33_BETA", str(_tag_to_float(m.group(1))))
        m = re.search(r"_tr([0-9mp]+)", tag)
        if m:
            tau_rate_us = int(round(_tag_to_float(m.group(1))))
            if tau_rate_us < 0:
                tau_rate_us = 0
            _set("MYEVS_EBF_S33_TAU_RATE_US", str(tau_rate_us))
        return old

    if v in {"s34", "ebf_s34", "ebfs34", "surprise_self", "surprise_self_shortdt", "s28_shortdt"}:
        # tag examples:
        #   ebf_s34_k0p2_labelscore_s9_tau128000
        #   ebf_s34_tr128000_k0p2_labelscore_s9_tau128000
        m = re.search(r"_k([0-9mp]+)", tag)
        if m:
            _set("MYEVS_EBF_S34_K_SELF", str(_tag_to_float(m.group(1))))
        m = re.search(r"_tr([0-9mp]+)", tag)
        if m:
            tau_rate_us = int(round(_tag_to_float(m.group(1))))
            if tau_rate_us < 0:
                tau_rate_us = 0
            _set("MYEVS_EBF_S34_TAU_RATE_US", str(tau_rate_us))
        return old

    return old


def _restore_env(old: dict[str, str | None]) -> None:
    for k, prev in old.items():
        if prev is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(prev)


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


def _variant_tag_contains(variant: str) -> str | None:
    v = str(variant).strip().lower()
    if v in {"ebf", "v0", "baseline"}:
        return None
    if v.startswith("ebf_"):
        v2 = v
    else:
        v2 = f"ebf_{v}"

    # Common form in ROC tags: "ebf_s25_..." / "ebf_s14_..."
    m = re.match(r"^(ebf_s\d+)(?:\b|_)", v2)
    if m:
        return m.group(1) + "_"
    return None


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


def _category_name(cat_id: int) -> str:
    return {
        1: "hotmask",
        2: "near_hotmask",
        3: "highrate_pixel",
        4: "samepol_shortdt",
        5: "toggle_shortdt",
        6: "isolated_nb0",
        7: "cluster_nb_ge_k",
        8: "other",
    }.get(int(cat_id), "unknown")


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


def _write_summary_csv(
    out_csv: str,
    *,
    dataset: str,
    variant: str,
    best_tag: str,
    thr: float | None,
    s: int,
    tau_us: int,
    max_events: int,
    samepol_dt_thr: float,
    toggle_dt_thr: float,
    nb_radius: int,
    nb_win_us: int,
    cluster_k: int,
    nearhot_r: int,
    highrate_topk: int,
    rows: list[dict[str, object]],
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "variant",
                "tag",
                "thr",
                "s",
                "tau_us",
                "max_events",
                "samepol_dt_thr",
                "toggle_dt_thr",
                "nb_radius",
                "nb_win_us",
                "cluster_k",
                "nearhot_r",
                "highrate_topk",
                "category",
                "total_events",
                "noise_total",
                "noise_kept",
                "noise_removed",
                "noise_kept_rate",
                "signal_total",
                "signal_kept",
                "signal_removed",
                "signal_kept_rate",
            ],
        )
        w.writeheader()
        for row in rows:
            row2 = {
                "dataset": dataset,
                "variant": variant,
                "tag": best_tag,
                "thr": "" if thr is None else float(thr),
                "s": int(s),
                "tau_us": int(tau_us),
                "max_events": int(max_events),
                "samepol_dt_thr": float(samepol_dt_thr),
                "toggle_dt_thr": float(toggle_dt_thr),
                "nb_radius": int(nb_radius),
                "nb_win_us": int(nb_win_us),
                "cluster_k": int(cluster_k),
                "nearhot_r": int(nearhot_r),
                "highrate_topk": int(highrate_topk),
                **row,
            }
            w.writerow(row2)


def _write_topk_pixels_csv(out_csv: str, *, width: int, height: int, fp_kept_per_pixel: np.ndarray, k: int) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    idxs = np.argsort(fp_kept_per_pixel)[::-1]
    idxs = idxs[: int(k)]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["x", "y", "idx", "fp_kept"]) 
        w.writeheader()
        for idx in idxs:
            c = int(fp_kept_per_pixel[idx])
            if c <= 0:
                break
            y = int(idx) // int(width)
            x = int(idx) - y * int(width)
            w.writerow({"x": x, "y": y, "idx": int(idx), "fp_kept": c})


def noise_type_stats(
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
    topk_pixels_csv: str,
    topk_pixels: int,
) -> None:
    if s < 3 or (s % 2) != 1:
        raise SystemExit(f"--s must be odd diameter >=3 (got {s})")

    tb = TimeBase(float(tick_ns))
    tau_ticks = int(tb.us_to_ticks(int(tau_us)))

    ev = load_labeled_npy(labeled_npy, start_events=int(start_events), max_events=int(max_events))

    hotmask: np.ndarray | None = None
    hotmask_hw: np.ndarray | None = None
    near_hotmask: np.ndarray | None = None
    if hotmask_npy:
        hotmask = _load_hotmask(hotmask_npy, width=int(width), height=int(height))
        hotmask_hw = hotmask.reshape((int(height), int(width))).astype(np.bool_, copy=False)
        near_hotmask = _dilate_mask(hotmask_hw, r=int(nearhot_r)).reshape(-1)

    kept: np.ndarray | None = None
    best_tag = ""

    if thr is None and roc_csv:
        tag_contains = _variant_tag_contains(str(variant))
        best = _read_best_point(roc_csv, s=int(s), tau_us=int(tau_us), tag=tag, tag_contains=tag_contains)
        best_tag = best.tag
        thr = float(best.thr)

        sweep = _load_sweep_module()
        radius_px = (int(s) - 1) // 2
        old_env = _apply_env_from_tag(variant, best.tag)
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
        kept = scores >= float(thr)

    elif thr is not None:
        best_tag = tag or "(manual_thr)"
        sweep = _load_sweep_module()
        radius_px = (int(s) - 1) // 2
        old_env = _apply_env_from_tag(variant, best_tag) if tag else {}
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
        kept = scores >= float(thr)

    else:
        best_tag = tag or "(no_denoise)"

    n = int(ev.label.size)
    label = ev.label.astype(np.bool_, copy=False)
    is_noise = ~label

    # High-rate pixels from noise-only counts (useful even when hotmask is absent).
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

    # Streaming per-pixel state to derive same-pixel short-dt and toggle.
    last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
    last_p = np.zeros((int(width) * int(height),), dtype=np.int8)

    nb_win_ticks = int(tb.us_to_ticks(int(nb_win_us)))

    # Categories as int8:
    # 1 hotmask
    # 2 near_hotmask
    # 3 highrate_pixel
    # 4 samepol_shortdt
    # 5 toggle_shortdt
    # 6 isolated_nb0
    # 7 cluster_nb_ge_k
    # 8 other
    cat = np.full((n,), 8, dtype=np.int8)

    fp_kept_per_pixel = np.zeros((int(width) * int(height),), dtype=np.int32)

    for i in range(n):
        x = int(ev.x[i])
        y = int(ev.y[i])
        if x < 0 or x >= int(width) or y < 0 or y >= int(height):
            continue
        idx = y * int(width) + x

        is_hot = False
        if hotmask is not None:
            is_hot = bool(hotmask[idx] != 0)

        is_near_hot = False
        if near_hotmask is not None:
            is_near_hot = bool(near_hotmask[idx])

        prev_ts = int(last_ts[idx])
        prev_p = int(last_p[idx])

        dt_norm = 1.0
        samepol = False
        toggle = False
        if prev_ts != 0:
            dt = int(ev.t[i]) - prev_ts
            if dt < 0:
                dt = 0
            dt_norm = float(dt) / float(max(1, tau_ticks))
            pi = int(ev.p[i])
            samepol = prev_p == pi
            toggle = prev_p == -pi

        # Neighborhood recent activity count (exclude self pixel).
        nb_cnt = 0
        if nb_win_ticks > 0 and int(nb_radius) > 0:
            t_i = int(ev.t[i])
            r = int(nb_radius)
            for dy in range(-r, r + 1):
                yy = y + dy
                if yy < 0 or yy >= int(height):
                    continue
                for dx in range(-r, r + 1):
                    xx = x + dx
                    if dx == 0 and dy == 0:
                        continue
                    if xx < 0 or xx >= int(width):
                        continue
                    nidx = yy * int(width) + xx
                    tsn = int(last_ts[nidx])
                    if tsn == 0:
                        continue
                    dt_nb = t_i - tsn
                    if dt_nb < 0:
                        dt_nb = 0
                    if dt_nb <= nb_win_ticks:
                        nb_cnt += 1

        # Priority-based exclusive categorization.
        if is_hot:
            cat[i] = 1
        elif is_near_hot:
            cat[i] = 2
        elif bool(highrate_mask[idx]):
            cat[i] = 3
        elif samepol and dt_norm < float(samepol_dt_thr):
            cat[i] = 4
        elif toggle and dt_norm < float(toggle_dt_thr):
            cat[i] = 5
        elif nb_cnt == 0:
            cat[i] = 6
        elif nb_cnt >= int(cluster_k):
            cat[i] = 7
        else:
            cat[i] = 8

        if kept is not None and bool(kept[i]) and bool(is_noise[i]):
            fp_kept_per_pixel[idx] += 1

        last_ts[idx] = np.uint64(ev.t[i])
        last_p[idx] = np.int8(ev.p[i])

    categories = [1, 2, 3, 4]
    rows: list[dict[str, object]] = []

    categories = [1, 2, 3, 4, 5, 6, 7, 8]

    for cid in categories:
        m_cat = cat == cid
        total_events = int(m_cat.sum())

        noise_total = int((m_cat & is_noise).sum())
        signal_total = int((m_cat & label).sum())

        if kept is None:
            noise_kept = 0
            signal_kept = 0
        else:
            noise_kept = int((m_cat & is_noise & kept).sum())
            signal_kept = int((m_cat & label & kept).sum())

        noise_removed = noise_total - noise_kept
        signal_removed = signal_total - signal_kept

        noise_kept_rate = float(noise_kept) / float(noise_total) if noise_total > 0 else 0.0
        signal_kept_rate = float(signal_kept) / float(signal_total) if signal_total > 0 else 0.0

        rows.append(
            {
                "category": _category_name(cid),
                "total_events": total_events,
                "noise_total": noise_total,
                "noise_kept": noise_kept,
                "noise_removed": noise_removed,
                "noise_kept_rate": noise_kept_rate,
                "signal_total": signal_total,
                "signal_kept": signal_kept,
                "signal_removed": signal_removed,
                "signal_kept_rate": signal_kept_rate,
            }
        )

    dataset_name = Path(labeled_npy).name
    _write_summary_csv(
        out_csv,
        dataset=dataset_name,
        variant=str(variant),
        best_tag=str(best_tag),
        thr=thr,
        s=int(s),
        tau_us=int(tau_us),
        max_events=int(max_events),
        samepol_dt_thr=float(samepol_dt_thr),
        toggle_dt_thr=float(toggle_dt_thr),
        nb_radius=int(nb_radius),
        nb_win_us=int(nb_win_us),
        cluster_k=int(cluster_k),
        nearhot_r=int(nearhot_r),
        highrate_topk=int(highrate_topk),
        rows=rows,
    )

    if topk_pixels_csv and int(topk_pixels) > 0 and kept is not None:
        _write_topk_pixels_csv(
            topk_pixels_csv,
            width=int(width),
            height=int(height),
            fp_kept_per_pixel=fp_kept_per_pixel,
            k=int(topk_pixels),
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Classify noise types and compare before/after denoising at a chosen threshold. "
            "Outputs a summary CSV with per-category counts and kept/removed rates."
        )
    )

    ap.add_argument("--labeled-npy", required=True, help="Input labeled .npy with fields t/x/y/p/label")
    ap.add_argument("--out-csv", required=True, help="Output summary CSV path")

    ap.add_argument("--width", type=int, default=346)
    ap.add_argument("--height", type=int, default=260)
    ap.add_argument("--tick-ns", type=float, default=1000.0)

    ap.add_argument("--s", type=int, default=9, help="Odd diameter, e.g. 9")
    ap.add_argument("--tau-us", type=int, default=128000)
    ap.add_argument(
        "--start-events",
        type=int,
        default=200000,
        help=(
            "Skip first N events (for segment-level analysis). "
            "Default=200k to focus on the hardest segment (seg1: [200k,400k))."
        ),
    )
    ap.add_argument(
        "--max-events",
        type=int,
        default=200000,
        help="Max events to load after start-events (default=200k => one segment)",
    )

    ap.add_argument(
        "--samepol-dt-thr",
        type=float,
        default=0.01,
        help="Normalized threshold dt/tau for same-pixel same-pol short-dt category",
    )
    ap.add_argument(
        "--toggle-dt-thr",
        type=float,
        default=0.01,
        help="Normalized threshold dt/tau for same-pixel toggle short-dt category",
    )

    ap.add_argument("--nb-radius", type=int, default=1, help="Neighborhood radius (pixels) for recent-activity counting")
    ap.add_argument(
        "--nb-win-us",
        type=int,
        default=2000,
        help="Neighborhood time window (us) for recent-activity counting",
    )
    ap.add_argument(
        "--cluster-k",
        type=int,
        default=3,
        help="If neighborhood recent-activity count >= k, categorize as cluster",
    )
    ap.add_argument(
        "--nearhot-r",
        type=int,
        default=1,
        help="Dilation radius for near-hotmask category (pixels). Requires --hotmask-npy.",
    )
    ap.add_argument(
        "--highrate-topk",
        type=int,
        default=32768,
        help="Top-K pixels by noise-only event count to mark as high-rate pixels",
    )

    ap.add_argument("--hotmask-npy", default="", help="Optional hotmask .npy (H,W) or (H*W,) nonzero=hot")

    ap.add_argument(
        "--variant",
        default="ebf",
        help="Score variant name, same as sweep_ebf_labelscore_grid.py (e.g. ebf|s14|s25|s9|s23)",
    )
    ap.add_argument(
        "--roc-csv",
        default="",
        help="ROC CSV path. If provided and --thr not set, auto-pick best-F1 threshold at (s,tau).",
    )
    ap.add_argument("--tag", default="", help="Optional exact tag inside ROC CSV")
    ap.add_argument("--thr", type=float, default=float("nan"), help="Manual threshold; overrides ROC selection")

    ap.add_argument("--topk-pixels-csv", default="", help="Optional output CSV of top FP-kept pixels")
    ap.add_argument("--topk-pixels", type=int, default=20)

    args = ap.parse_args()

    thr = None if not np.isfinite(float(args.thr)) else float(args.thr)
    tag = (args.tag or "").strip() or None

    noise_type_stats(
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
        topk_pixels_csv=str(args.topk_pixels_csv).strip(),
        topk_pixels=int(args.topk_pixels),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
