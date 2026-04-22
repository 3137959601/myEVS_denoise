from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def _read_csv(path: str) -> list[dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return [dict(x) for x in r]


def _to_f(v: str, default: float = float("nan")) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def plot_777(*, hist_csv: str, nearest_csv: str, out_png: str, title: str) -> None:
    rows = _read_csv(hist_csv)
    near_rows = _read_csv(nearest_csv)

    by_feat: dict[str, dict[str, list[tuple[float, float, float]]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        feat = str(r.get("feature", "")).strip()
        cls = str(r.get("class", "")).strip().lower()
        lo = _to_f(str(r.get("bin_lo", "")))
        hi = _to_f(str(r.get("bin_hi", "")))
        rr = _to_f(str(r.get("ratio", "")), default=0.0)
        if feat and cls in {"noise", "signal"} and np.isfinite(lo) and np.isfinite(hi) and np.isfinite(rr):
            by_feat[feat][cls].append((lo, hi, rr))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax_dt, ax_ratio, ax_ratio_hi, ax_near = axes.ravel()

    # 1) dt_min_us custom bins
    dt_noise = sorted(by_feat.get("dt_min_us", {}).get("noise", []), key=lambda z: z[0])
    dt_signal = sorted(by_feat.get("dt_min_us", {}).get("signal", []), key=lambda z: z[0])
    labels = []
    y_noise = []
    y_signal = []
    n_dt = max(len(dt_noise), len(dt_signal))
    for i in range(n_dt):
        row = dt_noise[i] if i < len(dt_noise) else dt_signal[i]
        lo, hi, _ = row
        if hi > 1e12:
            labels.append(">30ms")
        elif lo < 1000:
            labels.append(f"[{int(lo)},{int(hi)})us")
        else:
            labels.append(f"[{lo/1000.0:.0f},{hi/1000.0:.0f})ms")
        y_noise.append(dt_noise[i][2] if i < len(dt_noise) else 0.0)
        y_signal.append(dt_signal[i][2] if i < len(dt_signal) else 0.0)

    x = np.arange(n_dt, dtype=float)
    ax_dt.bar(x - 0.18, y_noise, width=0.36, color="#d62728", alpha=0.75, label="noise")
    ax_dt.bar(x + 0.18, y_signal, width=0.36, color="#2ca02c", alpha=0.75, label="signal")
    ax_dt.set_xticks(x)
    ax_dt.set_xticklabels(labels, rotation=15)
    ax_dt.set_ylabel("ratio")
    ax_dt.set_title("dt_min (nearest same-pol in 9x9)")
    ax_dt.grid(axis="y", linestyle="--", alpha=0.35)
    ax_dt.legend(fontsize=8)

    # 2) ratio_pure all events
    for ax, feat, ttl in (
        (ax_ratio, "ratio_pure", "ratio_pure (all events)"),
        (ax_ratio_hi, "ratio_pure_highscore", "ratio_pure (S_base > 3.0)"),
    ):
        noise = sorted(by_feat.get(feat, {}).get("noise", []), key=lambda z: z[0])
        signal = sorted(by_feat.get(feat, {}).get("signal", []), key=lambda z: z[0])
        n = max(len(noise), len(signal))
        xx = []
        ww = []
        yn = []
        ys = []
        for i in range(n):
            row = noise[i] if i < len(noise) else signal[i]
            lo, hi, _ = row
            xx.append(0.5 * (lo + hi))
            ww.append(max(hi - lo, 1e-6))
            yn.append(noise[i][2] if i < len(noise) else 0.0)
            ys.append(signal[i][2] if i < len(signal) else 0.0)
        xx = np.asarray(xx, dtype=float)
        ww = np.asarray(ww, dtype=float)
        yn = np.asarray(yn, dtype=float)
        ys = np.asarray(ys, dtype=float)

        ax.bar(xx - 0.2 * ww, yn, width=0.4 * ww, color="#d62728", alpha=0.75, label="noise")
        ax.bar(xx + 0.2 * ww, ys, width=0.4 * ww, color="#2ca02c", alpha=0.75, label="signal")
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("value")
        ax.set_ylabel("ratio")
        ax.set_title(ttl)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.legend(fontsize=8)

    # 4) nearest side inner/outer ratios
    near = {str(r.get("class", "")).strip().lower(): r for r in near_rows}
    classes = ["noise", "signal"]
    inner = [float(near.get(c, {}).get("inner_ratio", "nan")) for c in classes]
    outer = [float(near.get(c, {}).get("outer_ratio", "nan")) for c in classes]
    x2 = np.arange(len(classes), dtype=float)
    ax_near.bar(x2 - 0.18, inner, width=0.36, color="#1f77b4", alpha=0.8, label="nearest in inner (d<=2)")
    ax_near.bar(x2 + 0.18, outer, width=0.36, color="#ff7f0e", alpha=0.8, label="nearest in outer (d>2)")
    ax_near.set_xticks(x2)
    ax_near.set_xticklabels(classes)
    ax_near.set_ylim(0.0, 1.0)
    ax_near.set_ylabel("ratio")
    ax_near.set_title("Nearest neighbor side")
    ax_near.grid(axis="y", linestyle="--", alpha=0.35)
    ax_near.legend(fontsize=8)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot 7.77 GT histograms and nearest-side stats")
    ap.add_argument("--hist-csv", required=True)
    ap.add_argument("--nearest-csv", required=True)
    ap.add_argument("--out-png", required=True)
    ap.add_argument("--title", default="7.77 Baseline Time/Structure GT Stats")
    args = ap.parse_args()

    plot_777(
        hist_csv=str(args.hist_csv),
        nearest_csv=str(args.nearest_csv),
        out_png=str(args.out_png),
        title=str(args.title),
    )
    print(f"Saved: {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
