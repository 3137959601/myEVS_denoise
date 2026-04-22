from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np


def _outcome(label: int, kept: int) -> str:
    is_sig = int(label) != 0
    is_kept = int(kept) != 0
    if is_sig and is_kept:
        return "tp"
    if (not is_sig) and is_kept:
        return "fp"
    if is_sig and (not is_kept):
        return "fn"
    return "tn"


def u_hist(
    *,
    in_csv: str,
    out_csv: str,
    out_png: str,
    cats: list[str],
    outcomes: list[str],
    bins: int,
    col: str = "u",
) -> None:
    cats_set = {c.strip() for c in cats if c.strip()}
    outcomes_set = {o.strip().lower() for o in outcomes if o.strip()}
    col = str(col).strip() or "u"

    groups: dict[tuple[str, str], list[float]] = {}

    with open(in_csv, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        need = {"cat", col, "label", "kept"}
        if r.fieldnames is None or not need.issubset(set(r.fieldnames)):
            raise SystemExit(f"input CSV missing required columns {sorted(need)}")

        for row in r:
            cat = (row.get("cat") or "").strip()
            if cats_set and cat not in cats_set:
                continue

            u_s = (row.get(col) or "").strip()
            if not u_s:
                continue
            try:
                u = float(u_s)
            except Exception:
                continue
            if not np.isfinite(u):
                continue

            try:
                label = int(float(row.get("label") or 0))
                kept = int(float(row.get("kept") or 0))
            except Exception:
                continue

            oc = _outcome(label, kept)
            if outcomes_set and oc not in outcomes_set:
                continue

            key = (cat, oc)
            groups.setdefault(key, []).append(u)

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)

    bounded01_cols = {"u", "u_eff", "u_self", "u_nb", "u_nb_mix", "mix"}
    use_unit_range = col in bounded01_cols

    if use_unit_range:
        edges = np.linspace(0.0, 1.0, int(bins) + 1)
    else:
        all_vals: list[float] = []
        for vals in groups.values():
            all_vals.extend(vals)
        if not all_vals:
            return
        lo = float(np.min(np.asarray(all_vals, dtype=np.float64)))
        hi = float(np.max(np.asarray(all_vals, dtype=np.float64)))
        if not np.isfinite(lo) or not np.isfinite(hi):
            return
        if hi <= lo:
            hi = lo + 1.0
        edges = np.linspace(lo, hi, int(bins) + 1)

    # Write histogram CSV.
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["cat", "outcome", "bin_left", "bin_right", "count", "frac"],
        )
        w.writeheader()
        for (cat, oc), vals in sorted(groups.items()):
            a = np.asarray(vals, dtype=np.float64)
            cnt, _ = np.histogram(a, bins=edges)
            total = int(cnt.sum())
            if total <= 0:
                continue
            for i in range(int(bins)):
                c = int(cnt[i])
                w.writerow(
                    {
                        "cat": cat,
                        "outcome": oc,
                        "bin_left": float(edges[i]),
                        "bin_right": float(edges[i + 1]),
                        "count": c,
                        "frac": float(c) / float(total),
                    }
                )

    # Optional PNG plot.
    if out_png:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return

        centers = (edges[:-1] + edges[1:]) * 0.5

        plt.figure(figsize=(10, 6))
        for (cat, oc), vals in sorted(groups.items()):
            a = np.asarray(vals, dtype=np.float64)
            cnt, _ = np.histogram(a, bins=edges)
            total = float(max(1, int(cnt.sum())))
            y = cnt.astype(np.float64) / total
            plt.plot(centers, y, label=f"{cat}:{oc} (n={int(cnt.sum())})")

        plt.xlabel(str(col))
        plt.ylabel("fraction")
        plt.title(f"{col} distribution histogram")
        plt.grid(True, alpha=0.3)
        plt.legend()

        os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute histograms from dump_u_events CSV.")
    ap.add_argument("--in-csv", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-png", default="", help="Optional output PNG path")
    ap.add_argument(
        "--col",
        default="u",
        help="Column to summarize (default: u). Examples: u_eff,u_self,u_nb,u_nb_mix,mix,z_dbg",
    )
    ap.add_argument(
        "--cats",
        default="hotmask,near_hotmask,highrate_pixel",
        help="Comma-separated categories to include (empty means all)",
    )
    ap.add_argument(
        "--outcomes",
        default="fp,tp",
        help="Comma-separated outcomes to include (fp,tp,fn,tn; empty means all)",
    )
    ap.add_argument("--bins", type=int, default=64)
    args = ap.parse_args()

    cats = [c.strip() for c in str(args.cats).split(",")] if str(args.cats).strip() else []
    outcomes = [o.strip() for o in str(args.outcomes).split(",")] if str(args.outcomes).strip() else []

    u_hist(
        in_csv=str(args.in_csv),
        out_csv=str(args.out_csv),
        out_png=str(args.out_png).strip(),
        cats=cats,
        outcomes=outcomes,
        bins=int(args.bins),
        col=str(args.col),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
