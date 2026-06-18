"""Plot Driving hyperparameter sweeps for IEEE paper layout.

Exports:
  - fig_drive_sweep.*: single-column 2x2 figure
  - fig_drive_sweep_1x4.*: double-column 1x4 figure
"""
from __future__ import annotations

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd


BASE = r"D:/hjx_workspace/scientific_reserach/projects/myEVS/data/Hyperparameter ablation_study/drive"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(BASE, "phase1_drive.csv"))
    df2 = pd.read_csv(os.path.join(BASE, "phase2_sigma.csv"))
    df2["param"] = "sigma"
    df2["value"] = df2["sigma"].apply(lambda x: str(x).replace(".", "p"))
    df = df[df["param"] != "sigma"]
    df2["f1"] = 0.0
    return pd.concat([df, df2[["level", "param", "value", "auc", "f1"]]], ignore_index=True)


DF = load_data()
LEVEL_LABELS = {"1hz": "1 Hz", "3hz": "3 Hz", "5hz": "5 Hz", "7hz": "7 Hz", "10hz": "10 Hz"}
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
LEVELS = ["1hz", "3hz", "5hz", "7hz", "10hz"]
PARAMS = ["r", "tau", "sigma", "alpha"]
PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)"]
X_LABELS = ["$r$", r"$\tau$ (ms)", r"$\sigma$", r"$\alpha$"]

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
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,
        "figure.dpi": 300,
    }
)


def value_to_float(value: object) -> float | None:
    if value == "ema":
        return None
    return float(str(value).replace("p", "."))


def style_x_axis(ax: plt.Axes, param: str, values: list[float]) -> None:
    if param == "tau":
        ax.set_xscale("log")
        wanted = [4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000]
        ticks = [v for v in wanted if any(abs(v - x) < 1e-6 for x in values)]
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}"))
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    elif param == "r":
        ax.set_xticks(values)
        ax.set_xlim(min(values) - 0.3, max(values) + 0.3)
    elif param == "alpha":
        wanted = [0, 0.25, 0.5, 0.75, 1.0]
        ticks = [v for v in wanted if any(abs(v - x) < 1e-6 for x in values)]
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: "0" if abs(v) < 0.001 else f"{v:g}"))
        ax.set_xlim(-0.03, 1.05)
    elif param == "sigma":
        wanted = [1.0, 1.5, 2.0, 2.5, 3.0]
        ticks = [v for v in wanted if any(abs(v - x) < 1e-6 for x in values)]
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))


def save_all(fig: plt.Figure, path_no_ext: str) -> None:
    fig.savefig(path_no_ext + ".png", dpi=600, bbox_inches="tight", pad_inches=0.015)
    fig.savefig(path_no_ext + ".pdf", bbox_inches="tight", pad_inches=0.015)
    fig.savefig(path_no_ext + ".svg", bbox_inches="tight", pad_inches=0.015)


def make_plot(figsize: tuple[float, float], shape: tuple[int, int], layout: str) -> plt.Figure:
    fig, axs = plt.subplots(*shape, figsize=figsize)
    axes = list(axs.flat) if shape == (2, 2) else list(axs)
    is_single = layout == "single"
    marker_size = 2.7 if is_single else 3.0
    line_width = 0.9 if is_single else 1.0
    panel_fs = 8.5 if is_single else 9.0
    diamond_size = 3.3 if is_single else 3.6

    for idx, (ax, param, panel, xlabel) in enumerate(zip(axes, PARAMS, PANEL_LABELS, X_LABELS)):
        sub = DF[DF["param"] == param].copy()
        if sub.empty:
            continue
        sub["vn"] = sub["value"].apply(value_to_float)
        sub = sub.dropna(subset=["vn"]).sort_values("vn")

        for color, level in zip(COLORS, LEVELS):
            level_df = sub[sub["level"] == level]
            if level_df.empty:
                continue
            ax.plot(
                level_df["vn"],
                level_df["auc"],
                color=color,
                marker="o",
                label=LEVEL_LABELS[level],
                markersize=marker_size,
                linewidth=line_width,
            )

        ax.text(-0.08, 1.02, panel, transform=ax.transAxes, fontsize=panel_fs, fontweight="bold", va="bottom", ha="left")
        ax.set_xlabel(xlabel, labelpad=0.6)
        ax.set_ylabel("AUC" if is_single or idx == 0 else "", labelpad=0.6)
        values = sorted(sub["vn"].unique())
        style_x_axis(ax, param, values)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

        best_rows = sub.loc[sub.groupby("level")["auc"].idxmax()]
        for color, level in zip(COLORS, LEVELS):
            best = best_rows[best_rows["level"] == level]
            if not best.empty:
                ax.plot(best["vn"].iloc[0], best["auc"].iloc[0], "D", color=color, markersize=diamond_size, zorder=5)

        ax.tick_params(axis="both", which="major", pad=1.2)

    handles, labels = axes[0].get_legend_handles_labels()
    legend_y = 0.105 if is_single else 0.045
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, legend_y),
        frameon=True,
        fancybox=False,
        edgecolor="0.35",
        facecolor="white",
        framealpha=1.0,
        markerscale=1.1,
        columnspacing=0.75,
        handlelength=1.2,
        handletextpad=0.35,
    )
    bottom = 0.18 if is_single else 0.18
    fig.tight_layout(rect=[0, bottom, 1, 1], pad=0.12, w_pad=0.35, h_pad=0.35)
    return fig


fig = make_plot((3.45, 3.05), (2, 2), "single")
save_all(fig, os.path.join(BASE, "fig_drive_sweep"))
print("Saved fig_drive_sweep")
plt.close(fig)

fig = make_plot((7.10, 2.20), (1, 4), "double")
save_all(fig, os.path.join(BASE, "fig_drive_sweep_1x4"))
print("Saved fig_drive_sweep_1x4")
plt.close(fig)
