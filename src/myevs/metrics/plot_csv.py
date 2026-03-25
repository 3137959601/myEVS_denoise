from __future__ import annotations

"""Plot metrics tables (CSV) into publication-style figures.

Design goals (research workflow):
- CSV as the interchange format: `myevs sweep --out-csv ...` produces a table.
- Plotting should be generic: you can sweep different params (min-neighbors,
  time-us, refractory-us, ...) and add new metrics columns later.
- Minimal dependencies: csv + numpy + matplotlib.

Typical CSV schema from `myevs sweep`:
  param,value,events_out,kept_ratio,removed_ratio

But this plotter works with any CSV that has:
- one X column (numeric or categorical)
- one or more Y columns (numeric)
- optional GROUP column to plot multiple series
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class PlotConfig:
    in_csv: str
    out_path: str
    x: str = "value"
    y: tuple[str, ...] = ("kept_ratio",)
    group: str | None = None
    kind: str = "auto"  # auto|line|scatter|bar|step
    title: str | None = None
    xlabel: str | None = None
    ylabel: str | None = None
    logx: bool = False
    logy: bool = False
    dpi: int = 220
    figsize: tuple[float, float] = (6.0, 4.0)
    style: str = "paper"  # paper|presentation
    legend: bool = True
    grid: bool = True


def _try_float(s: str) -> float | None:
    try:
        return float(s)
    except Exception:
        return None


def read_csv_table(path: str) -> list[dict[str, str]]:
    # Use utf-8-sig to gracefully handle an optional UTF-8 BOM.
    # This commonly occurs when CSVs are produced by Windows PowerShell Export-Csv.
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        rows: list[dict[str, str]] = []
        for row in r:
            # keep empty lines out
            if not any((v or "").strip() for v in row.values()):
                continue
            rows.append({k: (v if v is not None else "") for k, v in row.items()})
        return rows


def _infer_is_numeric_column(rows: list[Mapping[str, str]], col: str) -> bool:
    ok = 0
    bad = 0
    for row in rows:
        s = (row.get(col, "") or "").strip()
        if s == "":
            continue
        if _try_float(s) is None:
            bad += 1
        else:
            ok += 1
        if bad > 0 and ok == 0:
            return False
    return ok > 0 and bad == 0


def _as_numeric(rows: list[Mapping[str, str]], col: str) -> np.ndarray:
    out = []
    for row in rows:
        s = (row.get(col, "") or "").strip()
        if s == "":
            out.append(np.nan)
        else:
            v = _try_float(s)
            out.append(np.nan if v is None else float(v))
    return np.asarray(out, dtype=np.float64)


def _as_str(rows: list[Mapping[str, str]], col: str) -> list[str]:
    return [((row.get(col, "") or "").strip()) for row in rows]


def _apply_research_style(*, style: str) -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            # Font: try Times/Arial for paper; keep SimHei for Chinese labels.
            "font.family": "sans-serif",
            "font.sans-serif": ["Times New Roman", "Arial", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            # Axes
            "axes.linewidth": 1.0,
            "axes.labelsize": 11 if style == "paper" else 13,
            "axes.titlesize": 12 if style == "paper" else 14,
            # Ticks
            "xtick.labelsize": 10 if style == "paper" else 12,
            "ytick.labelsize": 10 if style == "paper" else 12,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            # Lines
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
            # Figure
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def _auto_kind(*, x_is_numeric: bool, n_points: int) -> str:
    if not x_is_numeric:
        return "bar" if n_points <= 30 else "scatter"
    return "line" if n_points >= 2 else "scatter"


_ROC_LINESTYLES: tuple[str, ...] = ("-", "--", "-.", ":")
_ROC_MARKERS: tuple[str, ...] = ("o", "s", "^", "v", "D", "P", "X", "*", "+", "x")


def _is_roc_xy(*, x: str, y: tuple[str, ...]) -> bool:
    # Convention in this repo: ROC tables use columns named exactly fpr/tpr.
    # Keep this check strict so other plots remain unaffected.
    return x.strip().lower() == "fpr" and len(y) == 1 and y[0].strip().lower() == "tpr"


def _add_roc_endpoints(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Ensure (0,0) and (1,1) are explicitly present so ROC curves look like
    # standard paper plots (and match common plotting scripts).
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    x = np.clip(x, 0.0, 1.0)
    y = np.clip(y, 0.0, 1.0)

    eps = 1e-12

    def _has_point(x0: float, y0: float) -> bool:
        if x.size == 0:
            return False
        return bool(np.any((np.abs(x - x0) <= eps) & (np.abs(y - y0) <= eps)))

    if not _has_point(0.0, 0.0):
        x = np.concatenate((x, np.array([0.0], dtype=np.float64)))
        y = np.concatenate((y, np.array([0.0], dtype=np.float64)))

    if not _has_point(1.0, 1.0):
        x = np.concatenate((x, np.array([1.0], dtype=np.float64)))
        y = np.concatenate((y, np.array([1.0], dtype=np.float64)))

    return x, y


def _auc_trapz_sorted(x: np.ndarray, y: np.ndarray) -> float:
    """Trapezoidal AUC assuming x is sorted ascending."""

    if x.size < 2 or y.size < 2 or x.size != y.size:
        return 0.0

    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is not None:
        return float(trapezoid(y=y, x=x))

    trapz = getattr(np, "trapz", None)
    if trapz is not None:
        return float(trapz(y=y, x=x))

    dx = x[1:] - x[:-1]
    return float(np.sum(dx * (y[1:] + y[:-1]) * 0.5))


def plot_csv(cfg: PlotConfig) -> str:
    """Plot CSV to cfg.out_path, return output path."""

    import matplotlib.pyplot as plt

    rows = read_csv_table(cfg.in_csv)
    if not rows:
        raise ValueError(f"CSV has no rows: {cfg.in_csv}")

    # Validate columns
    cols = set(rows[0].keys())
    if cfg.x not in cols:
        raise ValueError(f"x column not found: {cfg.x}. Available: {sorted(cols)}")
    for y in cfg.y:
        if y not in cols:
            raise ValueError(f"y column not found: {y}. Available: {sorted(cols)}")
    if cfg.group is not None and cfg.group not in cols:
        raise ValueError(f"group column not found: {cfg.group}. Available: {sorted(cols)}")

    _apply_research_style(style=cfg.style)

    x_is_numeric = _infer_is_numeric_column(rows, cfg.x)
    kind = cfg.kind.lower().strip()
    if kind == "auto":
        kind = _auto_kind(x_is_numeric=x_is_numeric, n_points=len(rows))

    fig, ax = plt.subplots(figsize=cfg.figsize)
    is_roc = _is_roc_xy(x=cfg.x, y=cfg.y)

    def draw_series(
        series_rows: list[Mapping[str, str]],
        *,
        label_prefix: str | None = None,
        style_idx: int = 0,
    ) -> None:
        if x_is_numeric:
            # Special-case: ROC curves (tpr vs fpr) should include (0,0) and (1,1)
            # endpoints to match standard paper plotting conventions.
            if is_roc:
                ycol = cfg.y[0]
                x = _as_numeric(series_rows, cfg.x)
                y = _as_numeric(series_rows, ycol)
                x, y = _add_roc_endpoints(x, y)
                order = np.argsort(x)
                x = x[order]
                y = y[order]

                auc = _auc_trapz_sorted(x, y)
                base_label = label_prefix if label_prefix is not None else ycol
                label = f"{base_label}, AUC={auc:.3f}"

                ls = _ROC_LINESTYLES[int(style_idx) % len(_ROC_LINESTYLES)]
                mk = _ROC_MARKERS[int(style_idx) % len(_ROC_MARKERS)]

                if kind == "line":
                    ax.plot(x, y, linestyle=ls, marker=mk, label=label)
                elif kind == "step":
                    ax.step(x, y, where="mid", linestyle=ls, marker=mk, label=label)
                elif kind == "scatter":
                    ax.scatter(x, y, marker=mk, label=label)
                elif kind == "bar":
                    ax.bar(x, y, label=label)
                else:
                    raise ValueError(f"Unknown kind: {kind}")
                return

            x = _as_numeric(series_rows, cfg.x)
            order = np.argsort(x)
            x = x[order]
            for ycol in cfg.y:
                y = _as_numeric(series_rows, ycol)[order]
                label = ycol if label_prefix is None else f"{label_prefix}:{ycol}"
                if kind == "line":
                    ax.plot(x, y, marker="o", label=label)
                elif kind == "step":
                    ax.step(x, y, where="mid", label=label)
                elif kind == "scatter":
                    ax.scatter(x, y, label=label)
                elif kind == "bar":
                    ax.bar(x, y, label=label)
                else:
                    raise ValueError(f"Unknown kind: {kind}")
        else:
            x = _as_str(series_rows, cfg.x)
            for ycol in cfg.y:
                y = _as_numeric(series_rows, ycol)
                label = ycol if label_prefix is None else f"{label_prefix}:{ycol}"
                if kind in ("bar", "auto"):
                    ax.bar(x, y, label=label)
                elif kind in ("line", "step"):
                    ax.plot(x, y, marker="o", label=label)
                elif kind == "scatter":
                    ax.scatter(x, y, label=label)
                else:
                    raise ValueError(f"Unknown kind: {kind}")

    if cfg.group is None:
        draw_series(rows, style_idx=0)
    else:
        # Group by column value
        groups: dict[str, list[Mapping[str, str]]] = {}
        for r in rows:
            k = (r.get(cfg.group, "") or "").strip()
            groups.setdefault(k, []).append(r)
        # stable order
        for i, g in enumerate(sorted(groups.keys())):
            draw_series(groups[g], label_prefix=(g if g else "(empty)"), style_idx=i)

    if cfg.logx:
        ax.set_xscale("log")
    if cfg.logy:
        ax.set_yscale("log")

    ax.set_title(cfg.title or Path(cfg.in_csv).name)
    ax.set_xlabel(cfg.xlabel or cfg.x)
    if cfg.ylabel:
        ax.set_ylabel(cfg.ylabel)
    elif len(cfg.y) == 1:
        ax.set_ylabel(cfg.y[0])
    else:
        ax.set_ylabel("value")

    if cfg.grid:
        ax.grid(True, which="major", linestyle="--", alpha=0.35)

    # nicer spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if cfg.legend and (cfg.group is not None or len(cfg.y) > 1):
        if is_roc:
            ax.legend(frameon=False, loc="lower right")
        else:
            ax.legend(frameon=False)

    out_path = str(cfg.out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(cfg.dpi))
    plt.close(fig)
    return out_path


def plot_csv_quick(
    in_csv: str,
    out_path: str,
    *,
    x: str = "value",
    y: Iterable[str] = ("kept_ratio", "removed_ratio"),
    kind: str = "auto",
    group: str | None = None,
    title: str | None = None,
) -> str:
    cfg = PlotConfig(
        in_csv=in_csv,
        out_path=out_path,
        x=x,
        y=tuple(y),
        kind=kind,
        group=group,
        title=title,
    )
    return plot_csv(cfg)
