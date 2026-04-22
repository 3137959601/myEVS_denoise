from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def _read_hist(path: str) -> list[dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = [dict(x) for x in r]
    if not rows:
        raise SystemExit(f"empty hist csv: {path}")
    return rows


def _to_f(v: str, default: float = float("nan")) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def plot_histograms(*, hist_csv: str, out_png: str) -> None:
    rows = _read_hist(hist_csv)

    by_feat: dict[str, dict[str, list[tuple[float, float, float]]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        feat = str(r.get("feature", "")).strip()
        cls = str(r.get("class", "")).strip().lower()
        if not feat or cls not in {"noise", "signal"}:
            continue
        lo = _to_f(str(r.get("bin_lo", "")))
        hi = _to_f(str(r.get("bin_hi", "")))
        rr = _to_f(str(r.get("ratio", "")), default=0.0)
        if np.isfinite(lo) and np.isfinite(hi) and np.isfinite(rr):
            by_feat[feat][cls].append((lo, hi, rr))

    feat_order = ["outer_ratio", "mix_smooth", "anisotropy", "velocity_mean"]
    titles = {
        "outer_ratio": "Outer-Ring Energy Ratio",
        "mix_smooth": "Spatiotemporal Smooth Mix",
        "anisotropy": "Anisotropy Tensor Score",
        "velocity_mean": "Mean Apparent Velocity",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    for ax, feat in zip(axes, feat_order):
        cur = by_feat.get(feat, {})
        noise = sorted(cur.get("noise", []), key=lambda z: z[0])
        signal = sorted(cur.get("signal", []), key=lambda z: z[0])

        if feat != "velocity_mean":
            x = []
            w = []
            y_noise = []
            y_signal = []
            n = max(len(noise), len(signal))
            for i in range(n):
                if i < len(noise):
                    lo, hi, _ = noise[i]
                else:
                    lo, hi, _ = signal[i]
                x.append(0.5 * (lo + hi))
                w.append(max(hi - lo, 1e-6))
                y_noise.append(noise[i][2] if i < len(noise) else 0.0)
                y_signal.append(signal[i][2] if i < len(signal) else 0.0)

            x = np.asarray(x, dtype=float)
            w = np.asarray(w, dtype=float)
            y_noise = np.asarray(y_noise, dtype=float)
            y_signal = np.asarray(y_signal, dtype=float)

            ax.bar(x - 0.2 * w, y_noise, width=0.4 * w, color="#d62728", alpha=0.7, label="noise (GT=0)")
            ax.bar(x + 0.2 * w, y_signal, width=0.4 * w, color="#2ca02c", alpha=0.7, label="signal (GT=1)")
            ax.set_xlim(0.0, 1.0)
            ax.set_xlabel("value")
            ax.set_ylabel("ratio")
            ax.set_title(titles.get(feat, feat))
            ax.grid(axis="y", linestyle="--", alpha=0.35)
            ax.legend(fontsize=8)
        else:
            # velocity bins are custom intervals, including +inf.
            labels = [
                "[0,0.1)",
                "[0.1,0.5)",
                "[0.5,1)",
                "[1,5)",
                ">=5",
            ]
            y_noise = [0.0] * 5
            y_signal = [0.0] * 5
            for i, _lo, rr in [(idx, z[0], z[2]) for idx, z in enumerate(noise[:5])]:
                y_noise[i] = rr
            for i, _lo, rr in [(idx, z[0], z[2]) for idx, z in enumerate(signal[:5])]:
                y_signal[i] = rr

            x = np.arange(5, dtype=float)
            ax.bar(x - 0.18, y_noise, width=0.36, color="#d62728", alpha=0.7, label="noise (GT=0)")
            ax.bar(x + 0.18, y_signal, width=0.36, color="#2ca02c", alpha=0.7, label="signal (GT=1)")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=10)
            ax.set_xlabel("velocity (px/ms)")
            ax.set_ylabel("ratio")
            ax.set_title(titles.get(feat, feat))
            ax.grid(axis="y", linestyle="--", alpha=0.35)
            ax.legend(fontsize=8)

    fig.suptitle("7.75 GT-Based Feature Histogram (Heavy)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot 7.75 GT feature histograms from hist.csv")
    ap.add_argument("--hist-csv", required=True)
    ap.add_argument("--out-png", required=True)
    args = ap.parse_args()

    plot_histograms(hist_csv=str(args.hist_csv), out_png=str(args.out_png))
    print(f"Saved: {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
