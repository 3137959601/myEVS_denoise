"""Plot Driving hyperparameter sweeps (4-panel: r, tau, sigma, alpha)."""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

# Read Phase 1 Drive data
base = r"D:/hjx_workspace/scientific_reserach/projects/myEVS/data/Hyperparameter ablation_study/drive"
df = pd.read_csv(os.path.join(base, "phase1_drive.csv"))

# Replace sigma with Phase 2 fine-sweep data
df2 = pd.read_csv(os.path.join(base, "phase2_sigma.csv"))
df2["param"] = "sigma"
df2["value"] = df2["sigma"].apply(lambda x: str(x).replace(".", "p"))
df = df[df["param"] != "sigma"]
# Add missing f1 column (not used in plot, just for concat compatibility)
df2["f1"] = 0.0
df = pd.concat([df, df2[["level", "param", "value", "auc", "f1"]]], ignore_index=True)

# Map noise levels to clean labels and colors
LEVEL_LABELS = {
    "1hz": "1 Hz", "3hz": "3 Hz", "5hz": "5 Hz", "7hz": "7 Hz", "10hz": "10 Hz",
}
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
LEVELS = ["1hz", "3hz", "5hz", "7hz", "10hz"]

# Set style
plt.rcParams.update({
    "font.family": "Arial", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 13,
    "legend.fontsize": 9, "figure.dpi": 150,
})

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
params = ["r", "tau", "sigma", "alpha"]
titles = [r"(a) Radius $r$", r"(b) Time Window $\tau$ (us)", r"(c) Spatial Scale $\sigma$", r"(d) Polarity Weight $\alpha$"]
xlabels = ["$r$", r"$\tau$ (us)", r"$\sigma$", r"$\alpha$"]

for ax, param, title, xlab in zip(axes.flat, params, titles, xlabels):
    sub = df[df["param"] == param].copy()
    if sub.empty: continue

    # Parse values
    def parse_val(v):
        if v == "ema": return None
        return float(str(v).replace("p", "."))
    sub["val_num"] = sub["value"].apply(parse_val)
    sub = sub.dropna(subset=["val_num"])
    sub = sub.sort_values("val_num")

    for i, lv in enumerate(LEVELS):
        sl = sub[sub["level"] == lv]
        if sl.empty: continue
        ax.plot(sl["val_num"], sl["auc"], "o-", color=COLORS[i],
                label=LEVEL_LABELS[lv], markersize=4, linewidth=1.2)

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlab)
    ax.set_ylabel("AUC")
    # Format x-axis: show actual swept values
    swept_vals = sorted(sub["val_num"].unique())
    if param == "tau":
        ax.set_xscale("log")
        ax.set_xticks(swept_vals)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    elif param == "r":
        ax.set_xticks(swept_vals)
        ax.set_xlim(min(swept_vals)-0.3, max(swept_vals)+0.3)
    elif param == "alpha":
        ax.set_xticks([v for v in swept_vals if v <= 1.0])
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: "0" if abs(v)<0.001 else f"{v:.2f}"))
        ax.set_xlim(-0.03, 1.05)
    elif param == "sigma":
        ax.set_xticks(swept_vals)

    # Mark optimal
    best_row = sub.loc[sub.groupby("level")["auc"].idxmax()]
    for i, lv in enumerate(LEVELS):
        br = best_row[best_row["level"] == lv]
        if not br.empty:
            ax.plot(br["val_num"].iloc[0], br["auc"].iloc[0], "D",
                    color=COLORS[i], markersize=6, zorder=5)

# Single legend
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02),
           frameon=True, fancybox=True, shadow=True)

plt.tight_layout(rect=[0, 0.06, 1, 1.0])

out_path = os.path.join(base, "fig_drive_sweep.png")
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved to {out_path}")
plt.close()
