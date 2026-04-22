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


def _collect(rows: list[dict[str, str]], feature: str, cls: str) -> list[tuple[float, float, float]]:
    out: list[tuple[float, float, float]] = []
    for r in rows:
        if str(r.get("feature", "")).strip() != feature:
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


def plot_782(*, hist_csv: str, out_png: str, title: str) -> None:
    rows = _read_csv(hist_csv)

    t_noise = _collect(rows, "tspan_ms", "noise")
    t_signal = _collect(rows, "tspan_ms", "signal")
    u_noise = _collect(rows, "uself_count", "noise")
    u_signal = _collect(rows, "uself_count", "signal")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax_t, ax_u = axes.ravel()

    # Temporal span histogram (ratio).
    n_t = max(len(t_noise), len(t_signal))
    x = np.arange(n_t, dtype=float)
    labels_t: list[str] = []
    yn = []
    ys = []
    for i in range(n_t):
        row = t_noise[i] if i < len(t_noise) else t_signal[i]
        lo, hi, _ = row
        if np.isinf(hi):
            labels_t.append(f">={lo:.0f}")
        else:
            labels_t.append(f"[{lo:.1f},{hi:.1f})")
        yn.append(t_noise[i][2] if i < len(t_noise) else 0.0)
        ys.append(t_signal[i][2] if i < len(t_signal) else 0.0)

    ax_t.bar(x - 0.18, yn, width=0.36, color="#d62728", alpha=0.8, label="noise")
    ax_t.bar(x + 0.18, ys, width=0.36, color="#2ca02c", alpha=0.8, label="signal")
    ax_t.set_xticks(x)
    ax_t.set_xticklabels(labels_t, rotation=20)
    ax_t.set_title("T_span_ms (S_base>=3)")
    ax_t.set_xlabel("Time-span bins (ms)")
    ax_t.set_ylabel("ratio")
    ax_t.grid(axis="y", linestyle="--", alpha=0.35)
    ax_t.legend(fontsize=9)

    # Mark 2ms boundary emphasis.
    if len(labels_t) > 0:
        ax_t.text(0.02, 0.96, "Focus range: [0,2ms)", transform=ax_t.transAxes, va="top", ha="left", fontsize=9)

    # Self activity histogram (ratio).
    n_u = max(len(u_noise), len(u_signal))
    x2 = np.arange(n_u, dtype=float)
    labels_u: list[str] = []
    un = []
    us = []
    for i in range(n_u):
        row = u_noise[i] if i < len(u_noise) else u_signal[i]
        lo, hi, _ = row
        if np.isinf(hi):
            labels_u.append(">5")
        else:
            c = int(round((lo + hi) * 0.5))
            labels_u.append(str(c))
        un.append(u_noise[i][2] if i < len(u_noise) else 0.0)
        us.append(u_signal[i][2] if i < len(u_signal) else 0.0)

    ax_u.bar(x2 - 0.18, un, width=0.36, color="#d62728", alpha=0.8, label="noise")
    ax_u.bar(x2 + 0.18, us, width=0.36, color="#2ca02c", alpha=0.8, label="signal")
    ax_u.set_xticks(x2)
    ax_u.set_xticklabels(labels_u)
    ax_u.set_title("U_self count in past 30ms (S_base>=3)")
    ax_u.set_xlabel("Center-pixel trigger count bins in past 30ms")
    ax_u.set_ylabel("ratio")
    ax_u.grid(axis="y", linestyle="--", alpha=0.35)
    ax_u.legend(fontsize=9)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot 7.82 histograms")
    ap.add_argument("--hist-csv", required=True)
    ap.add_argument("--out-png", required=True)
    ap.add_argument("--title", default="7.82 Heavy High-score: T_span & U_self")
    args = ap.parse_args()

    plot_782(hist_csv=str(args.hist_csv), out_png=str(args.out_png), title=str(args.title))
    print(f"Saved: {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
