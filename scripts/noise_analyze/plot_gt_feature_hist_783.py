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


def _collect(rows: list[dict[str, str]], cls: str) -> list[tuple[float, float, float]]:
    out: list[tuple[float, float, float]] = []
    for r in rows:
        if str(r.get("feature", "")).strip() != "a_score":
            continue
        if str(r.get("class", "")).strip().lower() != cls:
            continue
        lo = _to_f(str(r.get("bin_lo", "")))
        hi = _to_f(str(r.get("bin_hi", "")))
        rr = _to_f(str(r.get("ratio", "")), default=0.0)
        if np.isfinite(lo) and np.isfinite(hi) and np.isfinite(rr):
            out.append((lo, hi, rr))
    out.sort(key=lambda z: z[0])
    return out


def plot_783(*, hist_csv: str, out_png: str, title: str) -> None:
    rows = _read_csv(hist_csv)
    noise = _collect(rows, "noise")
    signal = _collect(rows, "signal")

    n = max(len(noise), len(signal))
    x = np.arange(n, dtype=float)
    labels: list[str] = []
    yn: list[float] = []
    ys: list[float] = []
    for i in range(n):
        row = noise[i] if i < len(noise) else signal[i]
        lo, hi, _ = row
        labels.append(f"[{lo:.1f},{hi:.1f})")
        yn.append(noise[i][2] if i < len(noise) else 0.0)
        ys.append(signal[i][2] if i < len(signal) else 0.0)

    fig, ax = plt.subplots(1, 1, figsize=(11.5, 4.8))
    ax.bar(x - 0.18, yn, width=0.36, color="#d62728", alpha=0.8, label="noise")
    ax.bar(x + 0.18, ys, width=0.36, color="#2ca02c", alpha=0.8, label="signal")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25)
    ax.set_xlabel("A_score bins")
    ax.set_ylabel("ratio")
    ax.set_title("A_score distribution for S_base>=3.0")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(fontsize=10)

    ax.text(0.02, 0.95, "Check left [1.0,2.0) and right [2.5,4.0] ranges", transform=ax.transAxes, va="top", fontsize=9)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot 7.83 A_score histogram")
    ap.add_argument("--hist-csv", required=True)
    ap.add_argument("--out-png", required=True)
    ap.add_argument("--title", default="7.83 Heavy High-score: A_score histogram")
    args = ap.parse_args()

    plot_783(hist_csv=str(args.hist_csv), out_png=str(args.out_png), title=str(args.title))
    print(f"Saved: {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
