"""Hot-state validation experiments for ED24 N149/LLEF.

Outputs are written under data/hot_state_validation by default:
- raw_stats: pixel-rate and inter-event interval diagnostics
- hot_trace: per-event H/q/discount/raw evidence traces
- tables: quantiles, AUC/F1 sweeps, no-hot error breakdown
- figures: CDFs, histograms, sweep plots, error bars
- logs: run parameters and subprocess metadata
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

try:
    import numba
except Exception:  # pragma: no cover
    numba = None

from myevs.events import EventBatch, EventStreamMeta, to_pm1, unwrap_tick_batches
from myevs.metrics.roc_auc import KeyPacker, auc_trapz, signal_mask
from myevs.timebase import TimeBase


DEFAULT_OUT = ROOT / "data" / "hot_state_validation"
DEFAULT_DATA_ROOT = Path(r"D:/hjx_workspace/scientific_reserach/dataset/ED24")
DEFAULT_PY = Path(r"D:/software/Anaconda_envs/envs/myEVS/python.exe")
THR = "0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,8"


@dataclass(frozen=True)
class Sample:
    dataset: str
    level: str
    clean: Path
    noisy: Path
    width: int = 346
    height: int = 260

    @property
    def key(self) -> str:
        return f"{self.dataset}_{self.level.replace('.', 'p')}v"

    @property
    def label(self) -> str:
        return f"{self.dataset} {self.level}V"


def ed24_samples(data_root: Path, levels: Iterable[str]) -> list[Sample]:
    levels = tuple(str(x).replace("v", "").replace("V", "") for x in levels)
    ped_root = data_root / "myPedestrain_06"
    bike_root = data_root / "myBicycle_02"
    out: list[Sample] = []
    for level in levels:
        out.append(
            Sample(
                "ed24_ped06",
                level,
                ped_root / f"Pedestrain_06_{level}_signal_only.npy",
                ped_root / f"Pedestrain_06_{level}.npy",
            )
        )
        out.append(
            Sample(
                "ed24_bike02",
                level,
                bike_root / f"Bicycle_02_{level}_signal_only.npy",
                bike_root / f"Bicycle_02_{level}.npy",
            )
        )
    return out


def ensure_dirs(out_dir: Path) -> dict[str, Path]:
    dirs = {
        "raw_stats": out_dir / "raw_stats",
        "hot_trace": out_dir / "hot_trace",
        "figures": out_dir / "figures",
        "tables": out_dir / "tables",
        "logs": out_dir / "logs",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def load_event_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arr = np.load(path, mmap_mode="r")
    names = getattr(arr.dtype, "names", None)
    if names is not None:
        t = np.asarray(arr["t"], dtype=np.uint64)
        x = np.asarray(arr["x"], dtype=np.int32)
        y = np.asarray(arr["y"], dtype=np.int32)
        p = np.asarray(arr["p"], dtype=np.int8)
    else:
        if arr.ndim != 2 or arr.shape[1] < 4:
            raise ValueError(f"Unsupported npy shape for {path}: {arr.shape}")
        t = np.asarray(arr[:, 0], dtype=np.uint64)
        x = np.asarray(arr[:, 1], dtype=np.int32)
        y = np.asarray(arr[:, 2], dtype=np.int32)
        p = np.asarray(arr[:, 3], dtype=np.int8)
    if p.size and int(p.min()) >= 0 and int(p.max()) <= 1:
        p = to_pm1(p)
    return unwrap_arrays(t, x, y, p)


def unwrap_arrays(t: np.ndarray, x: np.ndarray, y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    batch = EventBatch(t=np.asarray(t, dtype=np.uint64), x=x.astype(np.uint16, copy=False), y=y.astype(np.uint16, copy=False), p=p)
    batches = list(unwrap_tick_batches([batch], bits=None))
    b = batches[0]
    return np.asarray(b.t, dtype=np.uint64), np.asarray(b.x, dtype=np.int32), np.asarray(b.y, dtype=np.int32), np.asarray(b.p, dtype=np.int8)


def build_exact_clean_index(sample: Sample) -> tuple[np.ndarray, KeyPacker]:
    ct, cx, cy, cp = load_event_arrays(sample.clean)
    meta = EventStreamMeta(sample.width, sample.height)
    packer = KeyPacker.for_meta(meta)
    keys = packer.pack(ct, cx, cy, cp)
    keys.sort()
    return np.unique(keys), packer


def labels_for_noisy(sample: Sample, t: np.ndarray, x: np.ndarray, y: np.ndarray, p: np.ndarray) -> np.ndarray:
    clean_keys, packer = build_exact_clean_index(sample)
    return signal_mask(clean_keys=clean_keys, packer=packer, t=t, x=x, y=y, p=p, match_ticks=0, match_bin_radius=0).astype(np.uint8)


def compute_auc_from_scores(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)
    y = labels[order].astype(np.uint8)
    n_pos = int(np.count_nonzero(y))
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.0
    # Kept when score >= threshold.
    tp = np.cumsum(y[::-1], dtype=np.int64)
    fp = np.cumsum((1 - y[::-1]).astype(np.uint8), dtype=np.int64)
    tpr = tp.astype(np.float64) / float(n_pos)
    fpr = fp.astype(np.float64) / float(n_neg)
    return float(auc_trapz(fpr, tpr))


def best_f1_from_scores(labels: np.ndarray, scores: np.ndarray, thresholds: Iterable[float]) -> dict[str, float]:
    labels_bool = labels.astype(bool)
    sig = int(np.count_nonzero(labels_bool))
    best = {"threshold": 0.0, "f1": 0.0, "tpr": 0.0, "fpr": 0.0, "precision": 0.0}
    noise = int(labels_bool.size - sig)
    for thr in thresholds:
        keep = scores >= float(thr)
        tp = int(np.count_nonzero(keep & labels_bool))
        fp = int(np.count_nonzero(keep & ~labels_bool))
        fn = sig - tp
        tn = noise - fp
        prec = float(tp) / float(tp + fp) if (tp + fp) else 0.0
        rec = float(tp) / float(sig) if sig else 0.0
        fpr = float(fp) / float(fp + tn) if (fp + tn) else 0.0
        f1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        if (f1, rec) > (best["f1"], best["tpr"]):
            best = {"threshold": float(thr), "f1": f1, "tpr": rec, "fpr": fpr, "precision": prec}
    return best


def save_plot(path: Path) -> None:
    if plt is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def experiment_a(sample: Sample, dirs: dict[str, Path], tick_ns: float) -> pd.DataFrame:
    t, x, y, p = load_event_arrays(sample.noisy)
    labels = labels_for_noisy(sample, t, x, y, p)
    npx = sample.width * sample.height
    idx = y * sample.width + x
    valid = (x >= 0) & (x < sample.width) & (y >= 0) & (y < sample.height)
    idx = idx[valid]
    labels_v = labels[valid]

    total = np.bincount(idx, minlength=npx)
    sig = np.bincount(idx[labels_v == 1], minlength=npx)
    noise = np.bincount(idx[labels_v == 0], minlength=npx)
    duration_s = max(float(int(t[-1]) - int(t[0])) * tick_ns * 1e-9, 1e-12) if t.size else 1e-12
    rate = total.astype(np.float64) / duration_s
    yy, xx = np.divmod(np.arange(npx), sample.width)
    pix = pd.DataFrame(
        {
            "dataset": sample.dataset,
            "level": sample.level,
            "x": xx,
            "y": yy,
            "total_count": total,
            "signal_count": sig,
            "noise_count": noise,
            "event_rate_hz": rate,
            "noise_fraction": np.divide(noise, total, out=np.zeros_like(rate), where=total > 0),
        }
    )
    pix.to_csv(dirs["raw_stats"] / f"pixel_rate_{sample.key}.csv", index=False, encoding="utf-8-sig")

    last = np.zeros((npx,), dtype=np.uint64)
    iei = np.full((t.shape[0],), np.nan, dtype=np.float64)
    for i in range(t.shape[0]):
        if not valid[i]:
            continue
        k = int(y[i]) * sample.width + int(x[i])
        prev = int(last[k])
        if prev > 0 and int(t[i]) >= prev:
            iei[i] = (float(int(t[i]) - prev) * tick_ns) / 1000.0
        last[k] = t[i]
    iei_df = pd.DataFrame({"dataset": sample.dataset, "level": sample.level, "label": labels, "iei_us": iei})
    iei_df[np.isfinite(iei_df["iei_us"])].to_csv(dirs["raw_stats"] / f"inter_event_interval_{sample.key}.csv", index=False)

    if plt is not None:
        event_rate = rate[idx]
        plt.figure(figsize=(5.2, 3.2))
        bins = np.linspace(0, max(1.0, np.nanpercentile(np.log10(event_rate + 1.0), 99.5)), 80)
        plt.hist(np.log10(event_rate[labels_v == 1] + 1.0), bins=bins, alpha=0.65, density=True, label="signal")
        plt.hist(np.log10(event_rate[labels_v == 0] + 1.0), bins=bins, alpha=0.65, density=True, label="noise")
        plt.xlabel("log10(pixel event rate + 1)")
        plt.ylabel("density")
        plt.title(sample.label)
        plt.legend()
        save_plot(dirs["figures"] / f"pixel_rate_hist_{sample.key}.pdf")

        plt.figure(figsize=(5.2, 3.2))
        for lab, name in ((1, "signal"), (0, "noise")):
            vals = iei[(labels == lab) & np.isfinite(iei)]
            vals = np.sort(vals)
            if vals.size:
                plt.plot(vals, np.arange(1, vals.size + 1) / vals.size, label=name)
        plt.xscale("log")
        plt.xlabel("same-pixel inter-event interval (us)")
        plt.ylabel("CDF")
        plt.title(sample.label)
        plt.legend()
        save_plot(dirs["figures"] / f"iei_cdf_{sample.key}.pdf")

    top_rows = []
    for pct in (1, 5):
        cutoff = np.percentile(total[total > 0], 100 - pct) if np.any(total > 0) else 0
        mask_px = total >= cutoff
        in_top = mask_px[idx]
        top_rows.append(
            {
                "dataset": sample.dataset,
                "level": sample.level,
                "top_percent": pct,
                "pixels": int(np.count_nonzero(mask_px)),
                "events": int(np.count_nonzero(in_top)),
                "signal_events": int(np.count_nonzero(in_top & (labels_v == 1))),
                "noise_events": int(np.count_nonzero(in_top & (labels_v == 0))),
                "noise_fraction": float(np.mean(labels_v[in_top] == 0)) if np.any(in_top) else 0.0,
            }
        )
    return pd.DataFrame(top_rows)


def _trace_kernel_py(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    width: int,
    height: int,
    radius: int,
    tau_ticks: int,
    sigma: float,
    hot_unit: int,
    hot_mask: int,
    hot_decay_k: float,
    hot_k: float,
    alpha_fixed: float,
    no_hot: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = int(t.shape[0])
    npx = int(width) * int(height)
    last_ts = np.zeros(npx, dtype=np.uint64)
    last_pol = np.zeros(npx, dtype=np.int8)
    hot_state = np.zeros(npx, dtype=np.int32)
    h_pre = np.zeros(n, dtype=np.float32)
    h_post = np.zeros(n, dtype=np.float32)
    discount = np.ones(n, dtype=np.float32)
    raw_same_out = np.zeros(n, dtype=np.float32)
    raw_opp_out = np.zeros(n, dtype=np.float32)
    score = np.zeros(n, dtype=np.float32)
    inv_2sig2 = 1.0 / (2.0 * max(sigma, 0.1) * max(sigma, 0.1))
    tau = max(int(tau_ticks), 1)

    for i in range(n):
        xi = int(x[i])
        yi = int(y[i])
        if xi < 0 or xi >= width or yi < 0 or yi >= height:
            continue
        pi = 1 if int(p[i]) > 0 else -1
        ti = int(t[i])
        idx0 = yi * width + xi
        prev_h = int(hot_state[idx0])
        ts0 = int(last_ts[idx0])
        dt0 = tau if ts0 == 0 else abs(ti - ts0)
        decay = int(math.ceil((float(hot_decay_k) * float(dt0) * float(hot_unit)) / float(tau)))
        hp = max(prev_h - decay, 0)
        hn = min(max(hp + int(hot_unit), 0), hot_mask)
        h_pre[i] = hp
        h_post[i] = hn

        x0 = max(0, xi - radius)
        x1 = min(width - 1, xi + radius)
        y0 = max(0, yi - radius)
        y1 = min(height - 1, yi + radius)
        raw_same = 0.0
        raw_opp = 0.0
        for yy in range(y0, y1 + 1):
            for xx in range(x0, x1 + 1):
                if xx == xi and yy == yi:
                    continue
                k = yy * width + xx
                pn = int(last_pol[k])
                if pn != pi and pn != -pi:
                    continue
                ts = int(last_ts[k])
                if ts == 0 or ti <= ts:
                    continue
                dt = ti - ts
                if dt > tau:
                    continue
                base_time = 1.0 - float(dt) / float(tau)
                if base_time <= 0.0:
                    continue
                dsq = (xx - xi) * (xx - xi) + (yy - yi) * (yy - yi)
                wst = base_time * base_time * math.exp(-float(dsq) * inv_2sig2)
                if pn == pi:
                    raw_same += wst
                else:
                    raw_opp += wst

        last_ts[idx0] = np.uint64(ti)
        last_pol[idx0] = np.int8(pi)
        hot_state[idx0] = np.int32(hn)

        disc = 1.0 if no_hot else (float(hn) + float(hot_unit)) / (float(hot_k) * float(hn) + float(hot_unit))
        raw_gated = raw_same + float(alpha_fixed) * raw_opp
        raw_same_out[i] = raw_same
        raw_opp_out[i] = raw_opp
        discount[i] = disc
        score[i] = raw_gated * disc
    return h_pre, h_post, discount, raw_same_out, raw_opp_out, score


if numba is not None:
    _trace_kernel = numba.njit(cache=True)(_trace_kernel_py)
else:
    _trace_kernel = _trace_kernel_py


def hot_unit_from_bits(hot_bits: int, hot_int_bits: int) -> int:
    return max(1, 1 << max(0, int(hot_bits) - max(1, int(hot_int_bits))))


def experiment_b(sample: Sample, dirs: dict[str, Path], args: argparse.Namespace, no_hot: bool = False) -> pd.DataFrame:
    t, x, y, p = load_event_arrays(sample.noisy)
    labels = labels_for_noisy(sample, t, x, y, p)
    tb = TimeBase(tick_ns=float(args.tick_ns))
    tau_ticks = int(tb.us_to_ticks(int(args.tau_us)))
    hot_unit = hot_unit_from_bits(int(args.hot_bits), int(args.hot_int_bits))
    hot_mask = 0x7FFFFFFF if int(args.hot_bits) >= 31 else (1 << int(args.hot_bits)) - 1

    h_pre, h_post, discount, raw_same, raw_opp, score = _trace_kernel(
        np.asarray(t, dtype=np.uint64),
        np.asarray(x, dtype=np.int32),
        np.asarray(y, dtype=np.int32),
        np.asarray(p, dtype=np.int8),
        int(sample.width),
        int(sample.height),
        int(args.radius),
        int(tau_ticks),
        float(args.sigma),
        int(hot_unit),
        int(hot_mask),
        float(args.hot_decay_k),
        float(args.hot_k),
        float(args.alpha),
        bool(no_hot),
    )
    q_pre = h_pre / float(hot_unit)
    q_post = h_post / float(hot_unit)
    tag = f"{sample.key}{'_nohot' if no_hot else ''}"
    np.savez_compressed(
        dirs["hot_trace"] / f"trace_{tag}.npz",
        x=x.astype(np.int32),
        y=y.astype(np.int32),
        t=t.astype(np.uint64),
        p=p.astype(np.int8),
        label=labels.astype(np.uint8),
        score=score.astype(np.float32),
        H_pre=h_pre.astype(np.float32),
        H_post=h_post.astype(np.float32),
        q_pre=q_pre.astype(np.float32),
        q_post=q_post.astype(np.float32),
        discount=discount.astype(np.float32),
        raw_same=raw_same.astype(np.float32),
        raw_opp=raw_opp.astype(np.float32),
    )
    trace_df = pd.DataFrame(
        {
            "x": x,
            "y": y,
            "t": t,
            "p": p,
            "label": labels,
            "score": score,
            "H_pre": h_pre,
            "H_post": h_post,
            "q_pre": q_pre,
            "q_post": q_post,
            "discount": discount,
            "raw_same": raw_same,
            "raw_opp": raw_opp,
        }
    )
    try:
        trace_df.to_parquet(dirs["hot_trace"] / f"trace_{tag}.parquet", index=False)
    except Exception:
        trace_df.to_csv(dirs["hot_trace"] / f"trace_{tag}.csv.gz", index=False)

    rows = []
    for lab, cls in ((1, "signal"), (0, "noise")):
        vals = h_pre[labels == lab]
        rows.append(
            {
                "dataset": sample.dataset,
                "level": sample.level,
                "class": cls,
                "count": int(vals.size),
                "mean": float(np.mean(vals)) if vals.size else 0.0,
                "median": float(np.median(vals)) if vals.size else 0.0,
                "p75": float(np.percentile(vals, 75)) if vals.size else 0.0,
                "p90": float(np.percentile(vals, 90)) if vals.size else 0.0,
                "p95": float(np.percentile(vals, 95)) if vals.size else 0.0,
                "p99": float(np.percentile(vals, 99)) if vals.size else 0.0,
            }
        )

    if plt is not None and not no_hot:
        plt.figure(figsize=(5.2, 3.2))
        for lab, name in ((1, "signal"), (0, "noise")):
            vals = np.sort(h_pre[labels == lab])
            if vals.size:
                plt.plot(vals, np.arange(1, vals.size + 1) / vals.size, label=name)
        plt.xlabel("H_pre")
        plt.ylabel("CDF")
        plt.title(sample.label)
        plt.legend()
        save_plot(dirs["figures"] / f"hot_state_cdf_{sample.key}.pdf")

        plt.figure(figsize=(4.2, 3.2))
        plt.boxplot([h_pre[labels == 1], h_pre[labels == 0]], tick_labels=["signal", "noise"], showfliers=False)
        plt.ylabel("H_pre")
        plt.title(sample.label)
        save_plot(dirs["figures"] / f"hot_state_box_{sample.key}.pdf")
    return pd.DataFrame(rows)


def hot_region_breakdown(sample: Sample, dirs: dict[str, Path], args: argparse.Namespace) -> pd.DataFrame:
    trace_path = dirs["hot_trace"] / f"trace_{sample.key}.npz"
    nohot_path = dirs["hot_trace"] / f"trace_{sample.key}_nohot.npz"
    if not trace_path.exists():
        experiment_b(sample, dirs, args, no_hot=False)
    if not nohot_path.exists():
        experiment_b(sample, dirs, args, no_hot=True)
    z = np.load(trace_path)
    zn = np.load(nohot_path)
    x = z["x"].astype(np.int32)
    y = z["y"].astype(np.int32)
    labels = z["label"].astype(np.uint8)
    scores = z["score"].astype(np.float32)
    scores_nohot = zn["score"].astype(np.float32)
    idx = y * sample.width + x
    valid = (x >= 0) & (x < sample.width) & (y >= 0) & (y < sample.height)
    total = np.bincount(idx[valid], minlength=sample.width * sample.height)
    cutoff = np.percentile(total[total > 0], 99) if np.any(total > 0) else 0
    hot_px = total >= cutoff
    region = hot_px[idx] & valid

    thresholds = [float(v) for v in THR.split(",")]
    base_best = best_f1_from_scores(labels, scores, thresholds)
    nohot_best = best_f1_from_scores(labels, scores_nohot, thresholds)

    rows = []
    for name, sc, best in (("llef", scores, base_best), ("no_hot", scores_nohot, nohot_best)):
        keep = sc >= float(best["threshold"])
        for rname, mask in (("all", np.ones_like(labels, dtype=bool)), ("hot_top1_total_rate", region)):
            m = mask
            lab = labels.astype(bool)
            tp = int(np.count_nonzero(m & keep & lab))
            fp = int(np.count_nonzero(m & keep & ~lab))
            tn = int(np.count_nonzero(m & ~keep & ~lab))
            fn = int(np.count_nonzero(m & ~keep & lab))
            sig = tp + fn
            noise = fp + tn
            rows.append(
                {
                    "dataset": sample.dataset,
                    "level": sample.level,
                    "method": name,
                    "region": rname,
                    "threshold": float(best["threshold"]),
                    "tp": tp,
                    "fp": fp,
                    "tn": tn,
                    "fn": fn,
                    "signal_kept_rate": float(tp) / sig if sig else 0.0,
                    "noise_kept_rate": float(fp) / noise if noise else 0.0,
                    "f1_global_threshold": float(best["f1"]),
                }
            )
    return pd.DataFrame(rows)


def run_cli_roc(sample: Sample, dirs: dict[str, Path], args: argparse.Namespace, env_extra: dict[str, str], tag: str) -> pd.DataFrame:
    py = Path(args.python)
    out_csv = dirs["tables"] / f"roc_{tag}_{sample.key}.csv"
    env = os.environ.copy()
    env.update(
        {
            "MYEVS_N149_HOT_BITS": str(args.hot_bits),
            "MYEVS_N149_HOT_INT_BITS": str(args.hot_int_bits),
            "MYEVS_N149_HOT_DECAY_K": str(args.hot_decay_k),
            "MYEVS_N149_HOT_K": str(args.hot_k),
            "MYEVS_N149_HOT_FUNC": "rational",
            "MYEVS_N149_SIGMA": str(args.sigma),
            "MYEVS_N149_ALPHA_FIXED": str(args.alpha),
        }
    )
    env.update(env_extra)
    cmd = [
        str(py),
        "-m",
        "myevs.cli",
        "roc",
        "--clean",
        str(sample.clean),
        "--noisy",
        str(sample.noisy),
        "--assume",
        "npy",
        "--width",
        str(sample.width),
        "--height",
        str(sample.height),
        "--tick-ns",
        str(args.tick_ns),
        "--engine",
        "cpp",
        "--method",
        "n149",
        "--radius-px",
        str(args.radius),
        "--time-us",
        str(args.tau_us),
        "--param",
        "min-neighbors",
        "--values",
        THR,
        "--match-us",
        "0",
        "--match-bin-radius",
        "0",
        "--tag",
        tag,
        "--out-csv",
        str(out_csv),
    ]
    if args.force or not out_csv.exists():
        subprocess.run(cmd, cwd=ROOT, env=env, check=True, timeout=1200, capture_output=True, text=True)
    df = pd.read_csv(out_csv)
    for col in ("auc", "f1", "value", "tpr", "fpr", "precision"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    best = df.sort_values(["f1", "tpr"], ascending=[False, False]).iloc[0]
    return pd.DataFrame(
        [
            {
                "dataset": sample.dataset,
                "level": sample.level,
                "tag": tag,
                "auc": float(df["auc"].max()),
                "best_f1": float(best["f1"]),
                "f1_threshold": float(best["value"]),
                "tpr_at_f1": float(best["tpr"]),
                "fpr_at_f1": float(best["fpr"]),
                "precision_at_f1": float(best["precision"]),
                "roc_csv": str(out_csv),
            }
        ]
    )


def experiment_c(samples: list[Sample], dirs: dict[str, Path], args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    lambda_rows = []
    for k in args.lambda_list:
        for s in samples:
            lambda_rows.append(run_cli_roc(s, dirs, args, {"MYEVS_N149_HOT_DECAY_K": str(k)}, f"lambda{k}"))
    lam = pd.concat(lambda_rows, ignore_index=True)
    lam.to_csv(dirs["tables"] / "hot_lambda_auc_f1_ed24.csv", index=False, encoding="utf-8-sig")

    funcs = {
        "rational_default": {"MYEVS_N149_HOT_FUNC": "rational", "MYEVS_N149_HOT_K": "2"},
        "exp": {"MYEVS_N149_HOT_FUNC": "exp", "MYEVS_N149_HOT_FMIN": "0.5", "MYEVS_N149_HOT_C": "1.0"},
        "hill": {"MYEVS_N149_HOT_FUNC": "hill", "MYEVS_N149_HOT_FMIN": "0.5", "MYEVS_N149_HOT_Q0": "1.0", "MYEVS_N149_HOT_P": "1.0"},
        "power": {"MYEVS_N149_HOT_FUNC": "power", "MYEVS_N149_HOT_FMIN": "0.5", "MYEVS_N149_HOT_C": "1.0", "MYEVS_N149_HOT_P": "1.0"},
        "linear": {"MYEVS_N149_HOT_FUNC": "linear", "MYEVS_N149_HOT_FMIN": "0.5", "MYEVS_N149_HOT_C": "0.5"},
    }
    func_rows = []
    for name, env in funcs.items():
        for s in samples:
            func_rows.append(run_cli_roc(s, dirs, args, env, f"func_{name}"))
    fun = pd.concat(func_rows, ignore_index=True)
    fun.to_csv(dirs["tables"] / "hot_func_auc_f1_ed24.csv", index=False, encoding="utf-8-sig")

    if plt is not None:
        agg = lam.groupby("tag", as_index=False).agg(mean_auc=("auc", "mean"), mean_f1=("best_f1", "mean"))
        agg["lambda"] = agg["tag"].str.replace("lambda", "", regex=False).astype(float)
        agg = agg.sort_values("lambda")
        plt.figure(figsize=(5.2, 3.2))
        plt.plot(agg["lambda"], agg["mean_auc"], marker="o", label="AUC")
        plt.plot(agg["lambda"], agg["mean_f1"], marker="s", label="F1")
        plt.xlabel("hot decay lambda")
        plt.ylabel("mean score")
        plt.legend()
        save_plot(dirs["figures"] / "hot_lambda_sweep_ed24.pdf")

        fagg = fun.groupby("tag", as_index=False).agg(mean_auc=("auc", "mean"), mean_f1=("best_f1", "mean"))
        plt.figure(figsize=(6.0, 3.4))
        xx = np.arange(fagg.shape[0])
        plt.bar(xx - 0.18, fagg["mean_auc"], width=0.36, label="AUC")
        plt.bar(xx + 0.18, fagg["mean_f1"], width=0.36, label="F1")
        plt.xticks(xx, fagg["tag"].str.replace("func_", "", regex=False), rotation=25, ha="right")
        plt.ylabel("mean score")
        plt.legend()
        save_plot(dirs["figures"] / "hot_func_sweep_ed24.pdf")
    return lam, fun


def write_run_log(dirs: dict[str, Path], args: argparse.Namespace, samples: list[Sample], elapsed_s: float) -> None:
    arg_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": elapsed_s,
        "args": arg_dict,
        "samples": [asdict(s) | {"clean": str(s.clean), "noisy": str(s.noisy)} for s in samples],
        "env": {k: os.environ.get(k) for k in sorted(os.environ) if k.startswith("MYEVS_")},
    }
    with (dirs["logs"] / f"hot_state_validation_{time.strftime('%Y%m%d_%H%M%S')}.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--python", default=str(DEFAULT_PY if DEFAULT_PY.exists() else "python"))
    parser.add_argument("--levels", default="1.8,2.5,3.3")
    parser.add_argument("--experiments", default="A,B,D", help="Comma-separated subset: A,B,C,D")
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--tau-us", type=int, default=256000)
    parser.add_argument("--tick-ns", type=float, default=1000.0)
    parser.add_argument("--sigma", type=float, default=2.75)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--hot-bits", type=int, default=8)
    parser.add_argument("--hot-int-bits", type=int, default=3)
    parser.add_argument("--hot-decay-k", type=float, default=2.0)
    parser.add_argument("--hot-k", type=float, default=2.0)
    parser.add_argument("--lambda-list", default="0,1,2,3,4")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.lambda_list = [float(x) for x in str(args.lambda_list).split(",") if x.strip()]

    t0 = time.time()
    dirs = ensure_dirs(args.out_dir)
    levels = [x.strip() for x in str(args.levels).split(",") if x.strip()]
    samples = ed24_samples(args.data_root, levels)
    missing = [str(p) for s in samples for p in (s.clean, s.noisy) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing dataset files:\n" + "\n".join(missing))

    todo = {x.strip().upper() for x in str(args.experiments).split(",") if x.strip()}
    print(f"[hot-state-validation] samples={len(samples)} experiments={','.join(sorted(todo))} out={args.out_dir}")

    if "A" in todo:
        rows = [experiment_a(s, dirs, float(args.tick_ns)) for s in samples]
        pd.concat(rows, ignore_index=True).to_csv(dirs["tables"] / "pixel_rate_summary_ed24.csv", index=False, encoding="utf-8-sig")
        print("[A] wrote pixel-rate and IEI diagnostics")

    if "B" in todo:
        qrows = [experiment_b(s, dirs, args, no_hot=False) for s in samples]
        qdf = pd.concat(qrows, ignore_index=True)
        qdf.to_csv(dirs["tables"] / "hot_state_quantiles_ed24.csv", index=False, encoding="utf-8-sig")
        print("[B] wrote hot-state traces and quantiles")

    if "C" in todo:
        experiment_c(samples, dirs, args)
        print("[C] wrote lambda/function AUC-F1 sweeps")

    if "D" in todo:
        rows = [hot_region_breakdown(s, dirs, args) for s in samples]
        df = pd.concat(rows, ignore_index=True)
        df.to_csv(dirs["tables"] / "no_hot_error_breakdown_ed24.csv", index=False, encoding="utf-8-sig")
        kept = df[["dataset", "level", "method", "region", "signal_kept_rate", "noise_kept_rate"]]
        kept.to_csv(dirs["tables"] / "hot_region_kept_rate_ed24.csv", index=False, encoding="utf-8-sig")
        if plt is not None:
            hot = df[df["region"] == "hot_top1_total_rate"]
            piv = hot.pivot_table(index=["dataset", "level"], columns="method", values="fp", aggfunc="sum")
            piv = piv.reset_index()
            labels = [f"{r.dataset.replace('ed24_', '')}\n{r.level}V" for r in piv.itertuples()]
            xx = np.arange(len(labels))
            plt.figure(figsize=(7.0, 3.5))
            plt.bar(xx - 0.18, piv.get("llef", pd.Series(np.zeros(len(piv)))), width=0.36, label="LLEF")
            plt.bar(xx + 0.18, piv.get("no_hot", pd.Series(np.zeros(len(piv)))), width=0.36, label="No hot")
            plt.xticks(xx, labels, rotation=0)
            plt.ylabel("noise kept in hot top-1%")
            plt.legend()
            save_plot(dirs["figures"] / "hot_region_error_bar_ed24.pdf")
        print("[D] wrote no-hot error breakdown")

    # Aggregate the per-sample hot CDFs into the required main figure name.
    if "B" in todo and plt is not None:
        plt.figure(figsize=(5.5, 3.4))
        for s in samples:
            z = np.load(dirs["hot_trace"] / f"trace_{s.key}.npz")
            vals = np.sort(z["H_pre"].astype(np.float32))
            if vals.size:
                plt.plot(vals, np.arange(1, vals.size + 1) / vals.size, label=s.label)
        plt.xlabel("H_pre")
        plt.ylabel("CDF")
        plt.legend(fontsize=7)
        save_plot(dirs["figures"] / "hot_state_cdf_ed24.pdf")

    write_run_log(dirs, args, samples, time.time() - t0)
    print(f"[hot-state-validation] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
