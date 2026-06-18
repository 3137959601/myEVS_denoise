"""Plot ED24 Pedestrian/Bicycle hyperparameter sweeps for double-column layout."""
from __future__ import annotations

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd


BASE = r"D:/hjx_workspace/scientific_reserach/projects/myEVS/data/Hyperparameter ablation_study"


def load_data() -> pd.DataFrame:
    frames = []
    for dataset in ["ped", "bike"]:
        df = pd.read_csv(os.path.join(BASE, dataset, f"phase1_{dataset}.csv"))
        df["dataset"] = dataset
        sigma_csv = os.path.join(BASE, dataset, "phase2_sigma.csv")
        if os.path.exists(sigma_csv):
            df2 = pd.read_csv(sigma_csv)
            if "sigma" in df2.columns:
                df2["level"] = df2["level"].astype(str).str.replace("p", ".", regex=False)
                df2["param"] = "sigma"
                df2["value"] = df2["sigma"].apply(lambda x: str(x).replace(".", "p"))
                df2["f1"] = 0.0
                df2["dataset"] = dataset
                df = df[df["param"] != "sigma"]
                df = pd.concat([df, df2[["level", "param", "value", "auc", "f1", "dataset"]]], ignore_index=True)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


DF = load_data()
LEVELS = [("ped", "1.8"), ("ped", "2.5"), ("ped", "3.3"), ("bike", "1.8"), ("bike", "2.5"), ("bike", "3.3")]
LABELS = {f"{d}_{lv}": f"{'Ped' if d == 'ped' else 'Bike'} {lv}V" for d, lv in LEVELS}
COLORS = ["#d62728", "#e377c2", "#bcbd22", "#1f77b4", "#17becf", "#9467bd"]
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
        wanted = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
        ticks = [v for v in wanted if any(abs(v - x) < 1e-6 for x in values)]
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))


def save_all(fig: plt.Figure, path_no_ext: str) -> None:
    fig.savefig(path_no_ext + ".png", dpi=600, bbox_inches="tight", pad_inches=0.015)
    fig.savefig(path_no_ext + ".pdf", bbox_inches="tight", pad_inches=0.015)
    fig.savefig(path_no_ext + ".svg", bbox_inches="tight", pad_inches=0.015)


fig, axes = plt.subplots(1, 4, figsize=(7.10, 2.25))
for idx, (ax, param, panel, xlabel) in enumerate(zip(axes, PARAMS, PANEL_LABELS, X_LABELS)):
    sub = DF[DF["param"] == param].copy()
    if sub.empty:
        continue
    sub["key"] = sub["dataset"].astype(str) + "_" + sub["level"].astype(str)
    sub["vn"] = sub["value"].apply(value_to_float)
    sub = sub.dropna(subset=["vn"]).sort_values("vn")

    for color, (dataset, level) in zip(COLORS, LEVELS):
        key = f"{dataset}_{level}"
        level_df = sub[sub["key"] == key]
        if level_df.empty:
            continue
        linestyle = "--" if dataset == "bike" else "-"
        ax.plot(
            level_df["vn"],
            level_df["auc"],
            color=color,
            marker="o",
            label=LABELS[key],
            markersize=3.0,
            linewidth=1.0,
            linestyle=linestyle,
        )

    ax.text(-0.08, 1.02, panel, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")
    ax.set_xlabel(xlabel, labelpad=0.6)
    ax.set_ylabel("AUC" if idx == 0 else "", labelpad=0.6)
    values = sorted(sub["vn"].unique())
    style_x_axis(ax, param, values)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

    best_rows = sub.loc[sub.groupby("key")["auc"].idxmax()]
    for color, (dataset, level) in zip(COLORS, LEVELS):
        key = f"{dataset}_{level}"
        best = best_rows[best_rows["key"] == key]
        if not best.empty:
            ax.plot(best["vn"].iloc[0], best["auc"].iloc[0], "D", color=color, markersize=3.6, zorder=5)

    ax.tick_params(axis="both", which="major", pad=1.2)

handles, labels = axes[0].get_legend_handles_labels()
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
save_all(fig, os.path.join(BASE, "fig_ed24_sweep"))
print("Saved fig_ed24_sweep")
plt.close(fig)
