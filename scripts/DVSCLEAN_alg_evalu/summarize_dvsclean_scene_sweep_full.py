from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize DVSCLEAN full sweep results.")
    ap.add_argument("--root", default="data/DVSCLEAN/scene_sweep_full")
    args = ap.parse_args()

    root = Path(args.root)
    rows: list[pd.DataFrame] = []
    for d in sorted(root.glob("MAH*_ratio*")):
        f = d / "scene_sweep_summary.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        df["Sample"] = d.name
        scene, level = d.name.split("_", 1)
        df["Scene"] = scene
        df["Level"] = level
        rows.append(df)
    if not rows:
        raise SystemExit(f"No results found in: {root}")

    all_df = pd.concat(rows, ignore_index=True)
    out_all = root / "all_samples_full_summary.csv"
    all_df.to_csv(out_all, index=False)

    mean_df = (
        all_df.groupby("Method", as_index=False)[["AUC_best", "DA_best", "SNRdB_best", "F1"]]
        .mean()
        .sort_values("AUC_best", ascending=False)
    )
    out_mean = root / "all_samples_full_mean.csv"
    mean_df.to_csv(out_mean, index=False)
    print(f"saved: {out_all}")
    print(f"saved: {out_mean}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

