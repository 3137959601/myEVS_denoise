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
class RocBestPoint:
    tag: str
    thr: float
    f1: float
    auc: float | None


def _load_sweep_module():
    # Reuse the exact scoring implementation used by the sweep script,
    # without turning scripts/ into a package.
    here = Path(__file__).resolve()
    sweep_path = here.with_name("sweep_ebf_labelscore_grid.py")
    spec = importlib.util.spec_from_file_location("_sweep_ebf_labelscore_grid", sweep_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"failed to load sweep module spec: {sweep_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tag_to_float(s: str) -> float:
    # tag uses p as dot and m as minus (e.g. 0p1, m0p2)
    s = str(s)
    s = s.replace("p", ".").replace("m", "-")
    return float(s)


def _apply_env_from_tag(variant: str, tag: str) -> dict[str, str | None]:
    """Set env vars so score_stream_ebf reproduces the given tag.

    Returns a dict of previous values for restoration.
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
        alpha = _tag_to_float(m.group(1))
        raw_thr = _tag_to_float(m.group(2))
        _set("MYEVS_EBF_S14_ALPHA", str(alpha))
        _set("MYEVS_EBF_S14_RAW_THR", str(raw_thr))
        return old

    if v in {"s25", "ebf_s25", "ebfs25"}:
        m = re.search(r"_a([0-9mp]+)_raw([0-9mp]+)_dt([0-9mp]+)_rraw([0-9mp]+)_g([0-9mp]+)", tag)
        if not m:
            raise SystemExit(f"cannot parse s25 params from tag: {tag}")
        alpha = _tag_to_float(m.group(1))
        raw_thr = _tag_to_float(m.group(2))
        dt_thr = _tag_to_float(m.group(3))
        ref_raw_thr = _tag_to_float(m.group(4))
        gamma = _tag_to_float(m.group(5))
        _set("MYEVS_EBF_S25_ALPHA", str(alpha))
        _set("MYEVS_EBF_S25_RAW_THR", str(raw_thr))
        _set("MYEVS_EBF_S25_DT_THR", str(dt_thr))
        _set("MYEVS_EBF_S25_REF_RAW_THR", str(ref_raw_thr))
        _set("MYEVS_EBF_S25_GAMMA", str(gamma))
        return old

    if v in {"s26", "ebf_s26", "ebfs26", "s26_actnorm", "actnorm_hotness", "actnorm_hotness_fusion"}:
        m = re.search(r"_a([0-9mp]+)_b([0-9mp]+)_k([0-9mp]+)_e([0-9mp]+)", tag)
        if not m:
            raise SystemExit(f"cannot parse s26 params from tag: {tag}")
        alpha = _tag_to_float(m.group(1))
        beta = _tag_to_float(m.group(2))
        kappa = _tag_to_float(m.group(3))
        eta = _tag_to_float(m.group(4))
        _set("MYEVS_EBF_S26_ALPHA", str(alpha))
        _set("MYEVS_EBF_S26_BETA", str(beta))
        _set("MYEVS_EBF_S26_KAPPA", str(kappa))
        _set("MYEVS_EBF_S26_ETA", str(eta))
        return old

    if v in {"s27", "ebf_s27", "ebfs27", "s27_relabnorm", "relabnorm_hotness", "relative_abnormal_hotness"}:
        m = re.search(r"_a([0-9mp]+)_b([0-9mp]+)_k([0-9mp]+)_l([0-9mp]+)", tag)
        if not m:
            raise SystemExit(f"cannot parse s27 params from tag: {tag}")
        alpha = _tag_to_float(m.group(1))
        beta = _tag_to_float(m.group(2))
        kappa = _tag_to_float(m.group(3))
        lambda_nb = _tag_to_float(m.group(4))
        _set("MYEVS_EBF_S27_ALPHA", str(alpha))
        _set("MYEVS_EBF_S27_BETA", str(beta))
        _set("MYEVS_EBF_S27_KAPPA", str(kappa))
        _set("MYEVS_EBF_S27_LAMBDA_NB", str(lambda_nb))
        return old

    if v in {"s28", "ebf_s28", "ebfs28", "noise_surprise", "surprise_zscore", "surprise_z"}:
        # Optional override: _tr<tau_rate_us>
        m = re.search(r"_tr([0-9mp]+)", tag)
        if m:
            tau_rate_us = int(round(_tag_to_float(m.group(1))))
            if tau_rate_us < 0:
                tau_rate_us = 0
            _set("MYEVS_EBF_S28_TAU_RATE_US", str(tau_rate_us))
        return old

    # baseline and many other variants don’t require env overrides for a single tag.
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

            try:
                f1 = float(row["f1"])
            except Exception:
                continue

            try:
                thr = float(row["value"])
            except Exception:
                continue

            auc: float | None
            a = (row.get("auc") or "").strip()
            if a:
                try:
                    auc = float(a)
                except Exception:
                    auc = None
            else:
                auc = None

            cur = best_by_tag.get(t)
            if cur is None or f1 > cur.f1:
                best_by_tag[t] = RocBestPoint(tag=t, thr=thr, f1=f1, auc=auc)

    if not best_by_tag:
        raise SystemExit(
            f"no matching rows found in ROC CSV for suffix={suffix!r} and tag={tag!r}\n"
            f"file={roc_csv}"
        )

    if tag is not None:
        return best_by_tag[tag]

    # If tag not specified, choose global best-F1 tag at (s,tau).
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


def analyze_fp_noise_types(
    *,
    labeled_npy: str,
    roc_csv: str,
    variant: str,
    width: int,
    height: int,
    s: int,
    tau_us: int,
    tick_ns: float,
    max_events: int,
    tag: str | None,
    hotmask_npy: str,
    topk_pixels: int,
) -> None:
    if s < 3 or (s % 2) != 1:
        raise SystemExit(f"--s must be odd diameter >=3 (got {s})")
    radius_px = (int(s) - 1) // 2

    tb = TimeBase(float(tick_ns))
    tau_ticks = int(tb.us_to_ticks(int(tau_us)))

    sweep = _load_sweep_module()
    ev = sweep.load_labeled_npy(labeled_npy, max_events=int(max_events))

    best = _read_best_point(roc_csv, s=int(s), tau_us=int(tau_us), tag=tag)

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

    thr = float(best.thr)
    pred = scores >= thr
    label = ev.label.astype(np.bool_, copy=False)

    fp = pred & (~label)
    tp = pred & label
    fn = (~pred) & label
    tn = (~pred) & (~label)

    n = int(label.size)
    n_fp = int(fp.sum())
    n_tp = int(tp.sum())
    n_fn = int(fn.sum())
    n_tn = int(tn.sum())

    hotmask: np.ndarray | None = None
    if hotmask_npy:
        hotmask = _load_hotmask(hotmask_npy, width=int(width), height=int(height))

    # O(1) streaming features from per-pixel last event.
    last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
    last_p = np.zeros((int(width) * int(height),), dtype=np.int8)

    fp_hot = 0
    fp_samepol_dt004 = 0
    fp_samepol_dt010 = 0
    fp_toggle_dt010 = 0
    fp_other = 0

    fp_count_per_pixel = np.zeros((int(width) * int(height),), dtype=np.int32)

    for i in range(n):
        x = int(ev.x[i])
        y = int(ev.y[i])
        if x < 0 or x >= int(width) or y < 0 or y >= int(height):
            continue
        idx = y * int(width) + x

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

        if bool(fp[i]):
            fp_count_per_pixel[idx] += 1

            is_hot = False
            if hotmask is not None:
                is_hot = bool(hotmask[idx] != 0)

            if is_hot:
                fp_hot += 1
            if samepol and dt_norm < 0.004:
                fp_samepol_dt004 += 1
            if samepol and dt_norm < 0.010:
                fp_samepol_dt010 += 1
            if toggle and dt_norm < 0.010:
                fp_toggle_dt010 += 1

            if (not is_hot) and (not (samepol and dt_norm < 0.010)) and (not (toggle and dt_norm < 0.010)):
                fp_other += 1

        # update state
        last_ts[idx] = np.uint64(ev.t[i])
        last_p[idx] = np.int8(ev.p[i])

    def _pct(k: int, total: int) -> float:
        if total <= 0:
            return 0.0
        return 100.0 * float(k) / float(total)

    print("\n=== FP Noise-Type Breakdown (labelscore) ===")
    print(f"variant={variant}  tag={best.tag}")
    print(f"best-F1 point: thr(value)={thr:.6g}  F1={best.f1:.6f}  AUC={best.auc}")
    print(f"s={s} (r={radius_px})  tau_us={tau_us}  tick_ns={tick_ns}  labeled_npy={labeled_npy}")

    print("\nConfusion counts:")
    print(f"  TP={n_tp}  FP={n_fp}  TN={n_tn}  FN={n_fn}  total={n}")

    print("\nFP categories (can overlap; 'other' is disjoint):")
    if hotmask is None:
        print("  hotmask: (disabled)")
    else:
        print(f"  hotmask: {fp_hot}  ({_pct(fp_hot, n_fp):.2f}%)")

    print(f"  same-pixel same-pol dt_norm<0.004: {fp_samepol_dt004}  ({_pct(fp_samepol_dt004, n_fp):.2f}%)")
    print(f"  same-pixel same-pol dt_norm<0.010: {fp_samepol_dt010}  ({_pct(fp_samepol_dt010, n_fp):.2f}%)")
    print(f"  same-pixel toggle dt_norm<0.010:   {fp_toggle_dt010}  ({_pct(fp_toggle_dt010, n_fp):.2f}%)")
    print(f"  other (not hotmask & not dt<0.01): {fp_other}  ({_pct(fp_other, n_fp):.2f}%)")

    if topk_pixels > 0 and n_fp > 0:
        k = int(topk_pixels)
        # top-k pixel indices by FP count
        idxs = np.argsort(fp_count_per_pixel)[::-1]
        idxs = idxs[:k]
        print(f"\nTop-{k} FP pixels (idx=y*w+x):")
        shown = 0
        for idx in idxs:
            c = int(fp_count_per_pixel[idx])
            if c <= 0:
                break
            y = int(idx) // int(width)
            x = int(idx) - y * int(width)
            print(f"  ({x:3d},{y:3d})  fp={c}")
            shown += 1
        if shown == 0:
            print("  (none)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze FP noise types for a chosen labelscore ROC tag.")

    ap.add_argument("--variant", default="s14", help="Score variant, e.g. ebf|s14|s25|s9|s23 ...")
    ap.add_argument("--roc-csv", required=True, help="ROC CSV file produced by sweep_ebf_labelscore_grid.py")
    ap.add_argument("--tag", default="", help="Optional exact tag; if empty, auto-pick best-F1 tag at (s,tau).")

    ap.add_argument("--labeled-npy", default="", help="Optional labeled npy path; if empty use --env defaults")
    ap.add_argument("--env", default="heavy", choices=["light", "mid", "heavy"], help="Which default ED24 file to use")

    ap.add_argument("--width", type=int, default=346)
    ap.add_argument("--height", type=int, default=260)
    ap.add_argument("--tick-ns", type=float, default=1000.0)

    ap.add_argument("--s", type=int, default=9)
    ap.add_argument("--tau-us", type=int, default=128000)
    ap.add_argument("--max-events", type=int, default=200000)

    ap.add_argument("--hotmask-npy", default="", help="Optional hotmask .npy (H,W) or (H*W,) nonzero=hot")
    ap.add_argument("--topk-pixels", type=int, default=15)

    args = ap.parse_args()

    default_labeled = {
        "light": r"D:\hjx_workspace\scientific_reserach\dataset\ED24\myPedestrain_06\Pedestrain_06_1.8.npy",
        "mid": r"D:\hjx_workspace\scientific_reserach\dataset\ED24\myPedestrain_06\Pedestrain_06_2.5.npy",
        "heavy": r"D:\hjx_workspace\scientific_reserach\dataset\ED24\myPedestrain_06\Pedestrain_06_3.3.npy",
    }

    labeled_npy = (args.labeled_npy or "").strip() or default_labeled[str(args.env)]
    tag = (args.tag or "").strip() or None

    analyze_fp_noise_types(
        labeled_npy=labeled_npy,
        roc_csv=str(args.roc_csv),
        variant=str(args.variant),
        width=int(args.width),
        height=int(args.height),
        s=int(args.s),
        tau_us=int(args.tau_us),
        tick_ns=float(args.tick_ns),
        max_events=int(args.max_events),
        tag=tag,
        hotmask_npy=str(args.hotmask_npy).strip(),
        topk_pixels=int(args.topk_pixels),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
