from __future__ import annotations

import argparse
import csv
import os

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


def _bin_label(lo: float, hi: float) -> str:
    if hi > 1e17:
        return f">={int(lo)}"
    if lo.is_integer() and hi.is_integer():
        return f"[{int(lo)},{int(hi)})"
    return f"[{lo:.1f},{hi:.1f})"


def plot_779(*, hist_csv: str, out_png: str, title: str) -> None:
    rows = _read_csv(hist_csv)
    rows = [r for r in rows if str(r.get("feature", "")).strip() == "s_base"]

    noise = []
    signal = []
    for r in rows:
        cls = str(r.get("class", "")).strip().lower()
        rec = (
            _to_f(str(r.get("bin_lo", ""))),
            _to_f(str(r.get("bin_hi", ""))),
            _to_f(str(r.get("ratio", "")), default=0.0),
            int(float(str(r.get("count", "0")))),
        )
        if cls == "noise":
            noise.append(rec)
        elif cls == "signal":
            signal.append(rec)

    noise = sorted(noise, key=lambda z: z[0])
    signal = sorted(signal, key=lambda z: z[0])
    n = max(len(noise), len(signal))

    labels = []
    y_noise_ratio = []
    y_signal_ratio = []
    y_noise_count = []
    y_signal_count = []
    for i in range(n):
        src = noise[i] if i < len(noise) else signal[i]
        lo, hi, _, _ = src
        labels.append(_bin_label(lo, hi))
        y_noise_ratio.append(noise[i][2] if i < len(noise) else 0.0)
        y_signal_ratio.append(signal[i][2] if i < len(signal) else 0.0)
        y_noise_count.append(noise[i][3] if i < len(noise) else 0)
        y_signal_count.append(signal[i][3] if i < len(signal) else 0)

    x = np.arange(n, dtype=float)
    fig, (ax_ratio, ax_count) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax_ratio.bar(x - 0.18, y_noise_ratio, width=0.36, color="#d62728", alpha=0.8, label="noise")
    ax_ratio.bar(x + 0.18, y_signal_ratio, width=0.36, color="#2ca02c", alpha=0.8, label="signal")
    ax_ratio.set_ylabel("ratio")
    ax_ratio.set_title("S_base band ratio by class")
    ax_ratio.grid(axis="y", linestyle="--", alpha=0.35)
    ax_ratio.legend(fontsize=9)

    ax_count.bar(x - 0.18, y_noise_count, width=0.36, color="#d62728", alpha=0.8, label="noise")
    ax_count.bar(x + 0.18, y_signal_count, width=0.36, color="#2ca02c", alpha=0.8, label="signal")
    ax_count.set_ylabel("count")
    ax_count.set_title("S_base band count by class")
    ax_count.grid(axis="y", linestyle="--", alpha=0.35)
    ax_count.legend(fontsize=9)

    ax_count.set_xticks(x)
    ax_count.set_xticklabels(labels, rotation=18)
    ax_count.set_xlabel("S_base bins")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot 7.79 S_base bandpass histograms")
    ap.add_argument("--hist-csv", required=True)
    ap.add_argument("--out-png", required=True)
    ap.add_argument("--title", default="7.79 Baseline S_base Bandpass Stats")
    args = ap.parse_args()

    plot_779(hist_csv=str(args.hist_csv), out_png=str(args.out_png), title=str(args.title))
    print(f"Saved: {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
