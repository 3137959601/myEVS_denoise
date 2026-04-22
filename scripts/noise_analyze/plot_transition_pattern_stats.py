from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def _read_summary(path: str) -> list[dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = [dict(x) for x in r]
    if not rows:
        raise SystemExit(f"empty summary csv: {path}")
    return rows


def _read_hist(path: str) -> list[dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = [dict(x) for x in r]
    if not rows:
        raise SystemExit(f"empty hist csv: {path}")
    return rows


def _to_f(v: str) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _plot_polarity_ratio(rows: list[dict[str, str]], out_png: str) -> None:
    keys = []
    same_vals = []
    opp_vals = []

    for r in rows:
        cls = r.get("class", "")
        scope = r.get("scope", "")
        step = r.get("step", "")
        win = str(r.get("window", "")).strip()
        if scope == "neighborhood" and win:
            name = f"{cls}-{scope}{win}-{step}"
        else:
            name = f"{cls}-{scope}-{step}"
        keys.append(name)
        same_vals.append(_to_f(r.get("same_pol_rate", "nan")))
        opp_vals.append(_to_f(r.get("opp_pol_rate", "nan")))

    x = np.arange(len(keys))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(11, len(keys) * 0.45), 5.8))
    ax.bar(x - width / 2, same_vals, width=width, label="same polarity", color="#1f77b4")
    ax.bar(x + width / 2, opp_vals, width=width, label="opposite polarity", color="#ff7f0e")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("ratio")
    ax.set_title("Polarity Ratio by Class/Scope/Step")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=65, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _plot_dt_quantiles(rows: list[dict[str, str]], out_png: str) -> None:
    keys = []
    p50 = []
    p90 = []
    p99 = []

    for r in rows:
        cls = r.get("class", "")
        scope = r.get("scope", "")
        step = r.get("step", "")
        win = str(r.get("window", "")).strip()
        if scope == "neighborhood" and win:
            name = f"{cls}-{scope}{win}-{step}"
        else:
            name = f"{cls}-{scope}-{step}"
        keys.append(name)
        p50.append(_to_f(r.get("dt_p50_us", "nan")))
        p90.append(_to_f(r.get("dt_p90_us", "nan")))
        p99.append(_to_f(r.get("dt_p99_us", "nan")))

    x = np.arange(len(keys))

    fig, ax = plt.subplots(figsize=(max(11, len(keys) * 0.45), 5.8))
    ax.plot(x, p50, marker="o", linewidth=1.5, label="p50")
    ax.plot(x, p90, marker="s", linewidth=1.5, label="p90")
    ax.plot(x, p99, marker="^", linewidth=1.5, label="p99")
    ax.set_yscale("log")
    ax.set_ylabel("interval (us, log scale)")
    ax.set_title("Prev-Event Interval Quantiles")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=65, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _plot_hist_overlay(rows: list[dict[str, str]], out_png: str) -> None:
    # Focus on pre1 (fallback to next1 for backward compatibility).
    focus_step = "pre1"
    has_pre1 = False
    for r in rows:
        if r.get("step", "") == "pre1":
            has_pre1 = True
            break
    if not has_pre1:
        focus_step = "next1"

    grouped: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        step = r.get("step", "")
        if step != focus_step:
            continue
        cls = r.get("class", "")
        scope = r.get("scope", "")
        win = str(r.get("window", "")).strip()
        key = scope if scope == "pixel" else f"{scope}{win}"
        lo = _to_f(r.get("bin_lo_us", "nan"))
        hi = _to_f(r.get("bin_hi_us", "nan"))
        rr = _to_f(r.get("ratio", "nan"))
        if np.isfinite(lo) and np.isfinite(hi) and np.isfinite(rr):
            mid = np.sqrt(max(lo, 1e-9) * max(hi, 1e-9))
            grouped[key][cls].append((mid, rr))

    keys = sorted(grouped.keys())
    if not keys:
        return

    fig, axes = plt.subplots(len(keys), 1, figsize=(9.5, 3.0 * len(keys)), sharex=True)
    if len(keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, keys):
        for cls, color in (("noise", "#d62728"), ("signal", "#2ca02c")):
            arr = grouped[key].get(cls, [])
            if not arr:
                continue
            arr = sorted(arr, key=lambda z: z[0])
            xs = [z[0] for z in arr]
            ys = [z[1] for z in arr]
            ax.plot(xs, ys, marker="o", linewidth=1.4, label=cls, color=color)
        ax.set_xscale("log")
        ax.set_ylabel("ratio")
        ax.set_title(f"Interval Histogram Ratio ({key}, {focus_step})")
        ax.grid(axis="both", linestyle="--", alpha=0.35)
        ax.legend()

    axes[-1].set_xlabel("interval (us, log scale)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_transition_stats(*, summary_csv: str, hist_csv: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    rows_summary = _read_summary(summary_csv)
    rows_hist = _read_hist(hist_csv)

    _plot_polarity_ratio(rows_summary, os.path.join(out_dir, "polarity_ratio_bar.png"))
    _plot_dt_quantiles(rows_summary, os.path.join(out_dir, "interval_quantiles_log.png"))
    _plot_hist_overlay(rows_hist, os.path.join(out_dir, "interval_hist_next1_overlay.png"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot charts from transition_pattern_stats CSV outputs.")
    ap.add_argument("--summary-csv", required=True)
    ap.add_argument("--hist-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    plot_transition_stats(
        summary_csv=str(args.summary_csv),
        hist_csv=str(args.hist_csv),
        out_dir=str(args.out_dir),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
