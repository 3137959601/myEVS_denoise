from __future__ import annotations

import argparse
import csv
import os
import re
import time
from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np

from myevs.denoise import DenoiseConfig, denoise_stream
from myevs.events import EventBatch, filter_visibility_batches, unwrap_tick_batches
from myevs.io.auto import open_events
from myevs.metrics.aocc import aocc_from_xyt
from myevs.metrics.esr import event_structural_ratio_mean_from_xy
from myevs.timebase import TimeBase

from myevs.denoise.ops.ebfopt_part2.n149_n145_s52_euclid_compactlut_backbone import score_stream_n149

SUPPORTED_ALGS = ("baf", "stcf", "ebf", "n149", "knoise", "evflow", "ynoise", "ts", "mlpf")


@dataclass(frozen=True)
class InputPair:
    clean: str
    noisy: str
    width: int
    height: int


@dataclass(frozen=True)
class OpConfig:
    method: str
    time_us: int
    radius_px: int
    min_neighbors: float
    refractory_us: int
    mlpf_model_path: str = ""
    mlpf_patch: int = 7


@dataclass(frozen=True)
class N149Input:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    p: np.ndarray
    label: np.ndarray


def _to_float(v: str | None, default: float = 0.0) -> float:
    try:
        return float(v) if v is not None and str(v).strip() != "" else float(default)
    except Exception:
        return float(default)


def _parse_algorithms(raw: str) -> list[str]:
    vals = [s.strip().lower() for s in str(raw).split(",") if s.strip()]
    if not vals or "all" in vals:
        return list(SUPPORTED_ALGS)
    out: list[str] = []
    for a in vals:
        if a not in SUPPORTED_ALGS:
            raise ValueError(f"Unknown algorithm: {a}. valid={','.join(SUPPORTED_ALGS)}")
        if a not in out:
            out.append(a)
    return out


def _parse_levels(raw: str) -> list[str]:
    vals = [s.strip().lower() for s in str(raw).split(",") if s.strip()]
    valid = ("light", "mid", "heavy")
    if not vals or "all" in vals:
        return list(valid)
    out: list[str] = []
    for lv in vals:
        if lv not in valid:
            raise ValueError(f"Unknown level: {lv}. valid=light,mid,heavy")
        if lv not in out:
            out.append(lv)
    return out


def _parse_metrics(raw: str) -> tuple[bool, bool]:
    vals = [s.strip().lower() for s in str(raw).split(",") if s.strip()]
    if not vals:
        vals = ["mesr", "aocc"]
    if "none" in vals:
        return False, False
    run_mesr = ("mesr" in vals) or ("all" in vals)
    run_aocc = ("aocc" in vals) or ("all" in vals)
    return run_mesr, run_aocc


def _resolve_ed24(level: str) -> InputPair:
    root = r"D:\hjx_workspace\scientific_reserach\dataset\ED24\myPedestrain_06"
    noisy = os.path.join(root, f"Pedestrain_06_{'1.8' if level == 'light' else ('2.5' if level == 'mid' else '3.3')}.npy")
    clean = os.path.join(root, f"Pedestrain_06_{'1.8' if level == 'light' else ('2.5' if level == 'mid' else '3.3')}_signal_only.npy")
    return InputPair(clean=clean, noisy=noisy, width=346, height=260)


def _resolve_driving(level: str) -> InputPair:
    root = r"D:\hjx_workspace\scientific_reserach\dataset\DND21\mydriving"
    d = os.path.join(root, f"driving_noise_{level}_slomo_shot_withlabel")
    if not os.path.isdir(d):
        raise FileNotFoundError(f"Missing dataset dir: {d}")
    all_npy = [os.path.join(d, n) for n in os.listdir(d) if n.lower().endswith(".npy")]
    if not all_npy:
        raise FileNotFoundError(f"No npy files in {d}")
    clean = next((p for p in all_npy if re.search(r"signal_only|clean", os.path.basename(p), re.IGNORECASE)), None)
    if clean is None:
        raise FileNotFoundError(f"Cannot find clean npy in {d}")
    noisy = next((p for p in all_npy if p != clean and "label" not in os.path.basename(p).lower()), None)
    if noisy is None:
        noisy = next((p for p in all_npy if p != clean), None)
    if noisy is None:
        raise FileNotFoundError(f"Cannot find noisy npy in {d}")
    return InputPair(clean=clean, noisy=noisy, width=346, height=260)


def _roc_csv_path(dataset: str, level: str, alg: str) -> str:
    if dataset == "ed24":
        if alg == "n149":
            return os.path.join("data", "ED24", "myPedestrain_06", "N149", f"roc_n149_{level}.csv")
        return os.path.join("data", "ED24", "myPedestrain_06", alg.upper(), f"roc_{alg}_{level}.csv")
    return os.path.join("data", "DND21", "mydriving", level, alg.upper(), f"roc_{alg}_{level}.csv")


def _load_rows(path: str) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _best_auc_tag(rows: list[dict[str, str]]) -> str:
    best_tag = ""
    best_auc = -1e18
    seen: set[str] = set()
    for r in rows:
        tag = str(r.get("tag", ""))
        if not tag or tag in seen:
            continue
        seen.add(tag)
        auc = _to_float(r.get("auc"), 0.0)
        if auc > best_auc:
            best_auc = auc
            best_tag = tag
    return best_tag


def _best_f1_row(rows: list[dict[str, str]]) -> dict[str, str]:
    return max(rows, key=lambda r: _to_float(r.get("f1"), -1e18))


def _best_f1_in_tag(rows: list[dict[str, str]], tag: str) -> dict[str, str]:
    tagged = [r for r in rows if str(r.get("tag", "")) == tag]
    if not tagged:
        raise RuntimeError(f"No rows for tag={tag}")
    return max(tagged, key=lambda r: _to_float(r.get("f1"), -1e18))


def _infer_base_cfg(alg: str, tag: str) -> tuple[int, int]:
    # Defaults aligned with current sweep scripts.
    radius = {"knoise": 1, "mlpf": 3}.get(alg, 1)
    time_us = 2000

    m_r = re.search(r"(?:^|_)r(\d+)(?:_|$)", tag)
    if m_r:
        radius = int(m_r.group(1))

    m_tau_us = re.search(r"tau(\d+)(?:_|$)", tag)
    if m_tau_us:
        time_us = int(m_tau_us.group(1))

    m_tau_ms = re.search(r"tau(\d+)ms", tag)
    if m_tau_ms:
        time_us = int(m_tau_ms.group(1)) * 1000

    m_decay = re.search(r"decay(\d+)", tag)
    if m_decay:
        time_us = int(m_decay.group(1))

    return radius, time_us


def _build_config(alg: str, row: dict[str, str]) -> OpConfig:
    tag = str(row.get("tag", ""))
    method = "stc" if alg == "stcf" else alg
    radius, time_us = _infer_base_cfg(alg, tag)
    min_neighbors = 2.0
    refractory = 50

    param = str(row.get("param", "")).strip().lower()
    value = _to_float(row.get("value"), 0.0)

    if param == "time-us":
        time_us = int(round(value))
    elif param == "radius-px":
        radius = int(round(value))
    elif param in ("min-neighbors", "score-threshold"):
        min_neighbors = float(value)
    elif param == "refractory-us":
        refractory = int(round(value))

    return OpConfig(
        method=method,
        time_us=int(time_us),
        radius_px=int(radius),
        min_neighbors=float(min_neighbors),
        refractory_us=int(refractory),
    )


def _default_mlpf_model_path(dataset: str, level: str) -> str:
    if dataset == "ed24":
        return os.path.join("data", "ED24", "myPedestrain_06", "MLPF", f"mlpf_torch_{level}.pt")
    return os.path.join("data", "DND21", "mydriving", level, "MLPF", f"mlpf_torch_{level}.pt")


def _collect_arrays(batches: Iterable[EventBatch]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ts: list[np.ndarray] = []
    for b in batches:
        if len(b) == 0:
            continue
        xs.append(np.asarray(b.x, dtype=np.int32))
        ys.append(np.asarray(b.y, dtype=np.int32))
        ts.append(np.asarray(b.t, dtype=np.int64))
    if not xs:
        z = np.empty((0,), dtype=np.int32)
        return z, z, np.empty((0,), dtype=np.int64)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ts)


def _load_noisy_arrays(pair: InputPair, tick_ns: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r = open_events(
        pair.noisy,
        width=pair.width,
        height=pair.height,
        batch_events=1_000_000,
        tick_ns=float(tick_ns),
        assume="npy",
    )
    batches = unwrap_tick_batches(r.batches, bits=None)
    batches = filter_visibility_batches(batches, show_on=True, show_off=True)
    ts: list[np.ndarray] = []
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    for b in batches:
        if len(b) == 0:
            continue
        ts.append(np.asarray(b.t, dtype=np.uint64))
        xs.append(np.asarray(b.x, dtype=np.int32))
        ys.append(np.asarray(b.y, dtype=np.int32))
        ps.append(np.asarray(b.p, dtype=np.int8))
    if not ts:
        zt = np.empty((0,), dtype=np.uint64)
        zi = np.empty((0,), dtype=np.int32)
        zp = np.empty((0,), dtype=np.int8)
        return zt, zi, zi, zp
    return (
        np.concatenate(ts),
        np.concatenate(xs),
        np.concatenate(ys),
        np.concatenate(ps),
    )


def _denoise_and_metrics(
    pair: InputPair,
    cfg: OpConfig,
    tick_ns: float,
    run_mesr: bool,
    run_aocc: bool,
    chunk_size: int,
    aocc_style: str,
) -> tuple[float | None, float | None, int]:
    tb = TimeBase(tick_ns=float(tick_ns))
    r = open_events(
        pair.noisy,
        width=pair.width,
        height=pair.height,
        batch_events=1_000_000,
        tick_ns=float(tick_ns),
        assume="npy",
    )
    batches = unwrap_tick_batches(r.batches, bits=None)
    batches = filter_visibility_batches(batches, show_on=True, show_off=True)
    den = denoise_stream(
        r.meta,
        batches,
        DenoiseConfig(
            method=cfg.method,
            time_window_us=int(cfg.time_us),
            radius_px=int(cfg.radius_px),
            min_neighbors=float(cfg.min_neighbors),
            refractory_us=int(cfg.refractory_us),
            mlpf_model_path=str(cfg.mlpf_model_path),
            mlpf_patch=int(cfg.mlpf_patch),
            show_on=True,
            show_off=True,
        ),
        timebase=tb,
        engine="python",
    )
    x, y, t = _collect_arrays(den)
    n = int(x.shape[0])

    mesr = None
    if run_mesr:
        mesr = float(event_structural_ratio_mean_from_xy(x, y, width=pair.width, height=pair.height, chunk_size=int(chunk_size)))

    aocc = None
    if run_aocc:
        t_us = np.round(t.astype(np.float64) * tb.tick_us).astype(np.int64, copy=False)
        aocc = float(aocc_from_xyt(x, y, t_us, width=pair.width, height=pair.height, style=aocc_style))

    return mesr, aocc, n


def _n149_and_metrics(
    pair: InputPair,
    cfg: OpConfig,
    threshold: float,
    tick_ns: float,
    run_mesr: bool,
    run_aocc: bool,
    chunk_size: int,
    aocc_style: str,
) -> tuple[float | None, float | None, int]:
    tb = TimeBase(tick_ns=float(tick_ns))
    t, x, y, p = _load_noisy_arrays(pair, tick_ns=float(tick_ns))
    ev = N149Input(
        t=t,
        x=x,
        y=y,
        p=p,
        label=np.zeros((t.shape[0],), dtype=np.int8),
    )
    scores = score_stream_n149(
        ev,
        width=int(pair.width),
        height=int(pair.height),
        radius_px=int(cfg.radius_px),
        tau_us=int(cfg.time_us),
        tb=tb,
    )
    thr = float(threshold)
    if np.isinf(thr):
        keep = np.zeros((scores.shape[0],), dtype=bool)
    else:
        keep = np.asarray(scores, dtype=np.float64) >= thr
    xk = np.asarray(x[keep], dtype=np.int32)
    yk = np.asarray(y[keep], dtype=np.int32)
    tk = np.asarray(t[keep], dtype=np.int64)
    n = int(xk.shape[0])

    mesr = None
    if run_mesr:
        mesr = float(event_structural_ratio_mean_from_xy(xk, yk, width=pair.width, height=pair.height, chunk_size=int(chunk_size)))

    aocc = None
    if run_aocc:
        t_us = np.round(tk.astype(np.float64) * tb.tick_us).astype(np.int64, copy=False)
        aocc = float(aocc_from_xyt(xk, yk, t_us, width=pair.width, height=pair.height, style=aocc_style))

    return mesr, aocc, n


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compute MESR/AOCC at best-AUC and best-F1 operating points from existing ROC CSV files."
    )
    ap.add_argument("--dataset", choices=["ed24", "driving"], required=True)
    ap.add_argument("--algorithms", default="all", help="comma list, e.g. baf,stcf,ebf or all")
    ap.add_argument("--levels", default="all", help="comma list: light,mid,heavy or all")
    ap.add_argument("--metrics", default="mesr,aocc", help="comma list: mesr,aocc,all,none")
    ap.add_argument("--points", default="best-auc,best-f1", help="comma list: best-auc,best-f1")
    ap.add_argument("--tick-ns", type=float, default=1000.0)
    ap.add_argument("--chunk-size", type=int, default=30000, help="MESR chunk size")
    ap.add_argument("--aocc-style", choices=["paper", "normalized"], default="paper")
    ap.add_argument("--out-csv", default="", help="output summary csv path")
    ap.add_argument(
        "--mlpf-model-pattern",
        default="",
        help="optional model path pattern for real MLPF, supports {level} and {dataset}",
    )
    args = ap.parse_args()

    algs = _parse_algorithms(args.algorithms)
    levels = _parse_levels(args.levels)
    run_mesr, run_aocc = _parse_metrics(args.metrics)
    points = [s.strip().lower() for s in str(args.points).split(",") if s.strip()]
    if not points:
        points = ["best-auc", "best-f1"]
    points = [p for p in points if p in ("best-auc", "best-f1")]
    if not points:
        raise ValueError("--points must include best-auc and/or best-f1")

    if not args.out_csv:
        if args.dataset == "ed24":
            out_csv = os.path.join("data", "ED24", "myPedestrain_06", "bestpoint_mesr_aocc_summary.csv")
        else:
            out_csv = os.path.join("data", "DND21", "mydriving", "bestpoint_mesr_aocc_summary.csv")
    else:
        out_csv = args.out_csv
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)

    rows_out: list[dict[str, object]] = []
    cache: dict[tuple[str, str, str, str, int, int, float, int], tuple[float | None, float | None, int]] = {}

    for alg in algs:
        for lv in levels:
            if alg == "n149" and args.dataset != "ed24":
                print(f"[skip] n149 currently supports dataset=ed24 only (got {args.dataset})")
                continue
            roc_csv = _roc_csv_path(args.dataset, lv, alg)
            if not os.path.exists(roc_csv):
                print(f"[skip] missing roc csv: {roc_csv}")
                continue

            rows = _load_rows(roc_csv)
            if not rows:
                print(f"[skip] empty roc csv: {roc_csv}")
                continue

            if args.dataset == "ed24":
                pair = _resolve_ed24(lv)
            else:
                pair = _resolve_driving(lv)

            if "best-auc" in points:
                tag_auc = _best_auc_tag(rows)
                op_auc = _best_f1_in_tag(rows, tag_auc)
                cfg_auc = _build_config(alg, op_auc)
                if alg == "mlpf":
                    model_path = (
                        args.mlpf_model_pattern.format(level=lv, dataset=args.dataset)
                        if str(args.mlpf_model_pattern).strip()
                        else _default_mlpf_model_path(args.dataset, lv)
                    )
                    if os.path.exists(model_path):
                        cfg_auc = replace(cfg_auc, mlpf_model_path=model_path, mlpf_patch=2 * cfg_auc.radius_px + 1)
                    else:
                        print(f"[warn] mlpf model missing for {lv}: {model_path}; fallback to proxy mode")
                key_auc = (
                    alg,
                    lv,
                    "best-auc",
                    op_auc.get("tag", ""),
                    cfg_auc.time_us,
                    cfg_auc.radius_px,
                    cfg_auc.min_neighbors,
                    cfg_auc.refractory_us,
                )
                if key_auc not in cache:
                    t0 = time.time()
                    if alg == "n149":
                        cache[key_auc] = _n149_and_metrics(
                            pair=pair,
                            cfg=cfg_auc,
                            threshold=_to_float(op_auc.get("value"), 0.0),
                            tick_ns=float(args.tick_ns),
                            run_mesr=run_mesr,
                            run_aocc=run_aocc,
                            chunk_size=int(args.chunk_size),
                            aocc_style=args.aocc_style,
                        )
                    else:
                        cache[key_auc] = _denoise_and_metrics(
                            pair=pair,
                            cfg=cfg_auc,
                            tick_ns=float(args.tick_ns),
                            run_mesr=run_mesr,
                            run_aocc=run_aocc,
                            chunk_size=int(args.chunk_size),
                            aocc_style=args.aocc_style,
                        )
                    print(f"[done] {alg}/{lv}/best-auc in {time.time()-t0:.2f}s")
                mesr, aocc, kept = cache[key_auc]
                rows_out.append(
                    {
                        "dataset": args.dataset,
                        "level": lv,
                        "algorithm": alg,
                        "point": "best-auc",
                        "roc_csv": roc_csv,
                        "tag": op_auc.get("tag", ""),
                        "method": cfg_auc.method,
                        "param": op_auc.get("param", ""),
                        "value": op_auc.get("value", ""),
                        "auc": op_auc.get("auc", ""),
                        "f1": op_auc.get("f1", ""),
                        "time_us": cfg_auc.time_us,
                        "radius_px": cfg_auc.radius_px,
                        "min_neighbors": cfg_auc.min_neighbors,
                        "refractory_us": cfg_auc.refractory_us,
                        "events_kept": kept,
                        "mesr": "" if mesr is None else float(mesr),
                        "aocc": "" if aocc is None else float(aocc),
                    }
                )

            if "best-f1" in points:
                op_f1 = _best_f1_row(rows)
                cfg_f1 = _build_config(alg, op_f1)
                if alg == "mlpf":
                    model_path = (
                        args.mlpf_model_pattern.format(level=lv, dataset=args.dataset)
                        if str(args.mlpf_model_pattern).strip()
                        else _default_mlpf_model_path(args.dataset, lv)
                    )
                    if os.path.exists(model_path):
                        cfg_f1 = replace(cfg_f1, mlpf_model_path=model_path, mlpf_patch=2 * cfg_f1.radius_px + 1)
                    else:
                        print(f"[warn] mlpf model missing for {lv}: {model_path}; fallback to proxy mode")
                key_f1 = (
                    alg,
                    lv,
                    "best-f1",
                    op_f1.get("tag", ""),
                    cfg_f1.time_us,
                    cfg_f1.radius_px,
                    cfg_f1.min_neighbors,
                    cfg_f1.refractory_us,
                )
                if key_f1 not in cache:
                    t0 = time.time()
                    if alg == "n149":
                        cache[key_f1] = _n149_and_metrics(
                            pair=pair,
                            cfg=cfg_f1,
                            threshold=_to_float(op_f1.get("value"), 0.0),
                            tick_ns=float(args.tick_ns),
                            run_mesr=run_mesr,
                            run_aocc=run_aocc,
                            chunk_size=int(args.chunk_size),
                            aocc_style=args.aocc_style,
                        )
                    else:
                        cache[key_f1] = _denoise_and_metrics(
                            pair=pair,
                            cfg=cfg_f1,
                            tick_ns=float(args.tick_ns),
                            run_mesr=run_mesr,
                            run_aocc=run_aocc,
                            chunk_size=int(args.chunk_size),
                            aocc_style=args.aocc_style,
                        )
                    print(f"[done] {alg}/{lv}/best-f1 in {time.time()-t0:.2f}s")
                mesr, aocc, kept = cache[key_f1]
                rows_out.append(
                    {
                        "dataset": args.dataset,
                        "level": lv,
                        "algorithm": alg,
                        "point": "best-f1",
                        "roc_csv": roc_csv,
                        "tag": op_f1.get("tag", ""),
                        "method": cfg_f1.method,
                        "param": op_f1.get("param", ""),
                        "value": op_f1.get("value", ""),
                        "auc": op_f1.get("auc", ""),
                        "f1": op_f1.get("f1", ""),
                        "time_us": cfg_f1.time_us,
                        "radius_px": cfg_f1.radius_px,
                        "min_neighbors": cfg_f1.min_neighbors,
                        "refractory_us": cfg_f1.refractory_us,
                        "events_kept": kept,
                        "mesr": "" if mesr is None else float(mesr),
                        "aocc": "" if aocc is None else float(aocc),
                    }
                )

    header = [
        "dataset",
        "level",
        "algorithm",
        "point",
        "roc_csv",
        "tag",
        "method",
        "param",
        "value",
        "auc",
        "f1",
        "time_us",
        "radius_px",
        "min_neighbors",
        "refractory_us",
        "events_kept",
        "mesr",
        "aocc",
    ]
    merged_rows: list[dict[str, object]] = []
    if os.path.exists(out_csv):
        with open(out_csv, "r", encoding="utf-8", newline="") as f:
            old_rows = list(csv.DictReader(f))
        # Replace scope: only rows touched by this run.
        # Key granularity: dataset + algorithm + level + point.
        touched = {
            (
                str(r.get("dataset", "")),
                str(r.get("algorithm", "")),
                str(r.get("level", "")),
                str(r.get("point", "")),
            )
            for r in rows_out
        }
        for r in old_rows:
            k = (
                str(r.get("dataset", "")),
                str(r.get("algorithm", "")),
                str(r.get("level", "")),
                str(r.get("point", "")),
            )
            if k not in touched:
                merged_rows.append(r)
    merged_rows.extend(rows_out)

    # Stable ordering for readability.
    merged_rows = sorted(
        merged_rows,
        key=lambda r: (
            str(r.get("dataset", "")),
            str(r.get("algorithm", "")),
            str(r.get("level", "")),
            str(r.get("point", "")),
        ),
    )

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in merged_rows:
            w.writerow(r)

    print(f"[ok] wrote: {out_csv} (new={len(rows_out)}, total={len(merged_rows)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
