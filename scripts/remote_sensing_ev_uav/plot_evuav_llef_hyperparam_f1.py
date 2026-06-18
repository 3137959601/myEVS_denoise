"""Plot EV-UAV LLEF single-factor hyperparameter sweeps using F1.

Exports a double-column 1x4 figure:
  - fig_evuav_llef_hyperparam_f1_1x4.png
  - fig_evuav_llef_hyperparam_f1_1x4.pdf
  - fig_evuav_llef_hyperparam_f1_1x4.svg
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd


DEFAULT_BASE = Path(
    r"D:\hjx_workspace\scientific_reserach\projects\myEVS"
    r"\data\remote_sensing_ev_uav_noise\hyperparam_final_single_factor"
)

PARAMS = ["r", "tau", "sigma", "alpha"]
PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)"]
X_LABELS = ["$r$", r"$\tau$ (ms)", r"$\sigma$", r"$\alpha$"]
SEQUENCES = ["test_005", "test_009", "test_014", "test_019", "test_020", "test_021"]
SEQ_LABELS = {seq: seq.replace("_", " ") for seq in SEQUENCES}
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
ALPHA_PLOT_VALUES = {0.0, 0.5, 1.0, 1.5, 2.0, 4.0, 6.0, 8.0}


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "axes.spines.right": True,
        "axes.spines.top": True,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,
        "figure.dpi": 300,
    }
)


def value_to_float(value: object) -> float:
    return float(str(value).replace("p", "."))


def style_x_axis(ax: plt.Axes, param: str, values: list[float]) -> None:
    if param == "tau":
        ax.set_xscale("log")
        wanted = [8000, 16000, 32000, 64000, 128000, 256000, 512000]
        ticks = [v for v in wanted if any(abs(v - x) < 1e-6 for x in values)]
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}"))
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    elif param == "r":
        ax.set_xticks(values)
        ax.set_xlim(min(values) - 0.3, max(values) + 0.3)
    elif param == "sigma":
        wanted = [1.0, 2.0, 3.0, 4.0, 5.0]
        ticks = [v for v in wanted if any(abs(v - x) < 1e-6 for x in values)]
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    elif param == "alpha":
        wanted = [0, 1, 2, 4, 6, 8]
        ticks = [v for v in wanted if any(abs(v - x) < 1e-6 for x in values)]
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: "0" if abs(v) < 1e-9 else f"{v:g}"))
        ax.set_xlim(-0.2, 8.2)


def save_all(fig: plt.Figure, path_no_ext: Path) -> None:
    fig.savefig(path_no_ext.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.015)
    fig.savefig(path_no_ext.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.015)
    fig.savefig(path_no_ext.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.015)


def plot(base_dir: Path, out_dir: Path) -> list[Path]:
    raw = pd.read_csv(base_dir / "evuav_llef_hyperparam_raw.csv")

    raw["vn"] = raw["value"].apply(value_to_float)

    fig, axes = plt.subplots(1, 4, figsize=(7.10, 2.25), sharey=False)

    for idx, (ax, param, panel, xlabel) in enumerate(zip(axes, PARAMS, PANEL_LABELS, X_LABELS)):
        sub = raw[raw["param"] == param].copy().sort_values("vn")
        if param == "alpha":
            sub = sub[sub["vn"].apply(lambda x: any(abs(x - v) < 1e-9 for v in ALPHA_PLOT_VALUES))]

        for color, seq in zip(COLORS, SEQUENCES):
            seq_df = sub[sub["sequence"] == seq]
            if seq_df.empty:
                continue
            ax.plot(
                seq_df["vn"],
                seq_df["f1"],
                color=color,
                marker="o",
                markersize=3.0,
                linewidth=1.0,
                label=SEQ_LABELS[seq],
            )

        best_rows = sub.loc[sub.groupby("sequence")["f1"].idxmax()]
        for color, seq in zip(COLORS, SEQUENCES):
            best = best_rows[best_rows["sequence"] == seq]
            if not best.empty:
                ax.plot(best["vn"].iloc[0], best["f1"].iloc[0], "D", color=color, markersize=3.6, zorder=5)

        ax.text(-0.08, 1.02, panel, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")
        ax.set_xlabel(xlabel, labelpad=0.6)
        ax.set_ylabel("F1" if idx == 0 else "", labelpad=0.6)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
        ax.tick_params(axis="both", which="major", pad=1.2)
        style_x_axis(ax, param, sorted(sub["vn"].unique()))

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=6,
        bbox_to_anchor=(0.5, 0.045),
        frameon=True,
        fancybox=False,
        edgecolor="0.35",
        facecolor="white",
        framealpha=1.0,
        markerscale=1.1,
        columnspacing=0.65,
        handlelength=1.2,
        handletextpad=0.3,
    )

    fig.tight_layout(rect=[0, 0.18, 1, 1], pad=0.12, w_pad=0.35)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "fig_evuav_llef_hyperparam_f1_1x4"
    save_all(fig, stem)
    plt.close(fig)
    return [stem.with_suffix(ext) for ext in [".png", ".pdf", ".svg"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or (args.base_dir / "figures")
    paths = plot(args.base_dir, out_dir)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
