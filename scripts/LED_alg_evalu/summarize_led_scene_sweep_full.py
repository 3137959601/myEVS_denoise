from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize LED full-scene sweep results across all scene_* folders.")
    ap.add_argument("--root", default="data/LED/scene_sweep_full")
    args = ap.parse_args()

    root = Path(args.root)
    rows: list[pd.DataFrame] = []
    for d in sorted(root.glob("scene_*")):
        if not d.is_dir():
            continue
        f = d / "scene_sweep_summary.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        df["Scene"] = d.name
        rows.append(df)

    if not rows:
        raise SystemExit(f"No scene_sweep_summary.csv found under: {root}")

    all_df = pd.concat(rows, ignore_index=True)
    out_all = root / "all_scenes_full_summary.csv"
    all_df.to_csv(out_all, index=False)

    mean_df = (
        all_df.groupby("Method", as_index=False)[["AUC_best", "DA_best", "F1"]]
        .mean()
        .sort_values("AUC_best", ascending=False)
    )
    out_mean = root / "all_scenes_full_mean.csv"
    mean_df.to_csv(out_mean, index=False)

    rank_df = all_df[["Scene", "Method", "AUC_best"]].copy()
    rank_df["rank_auc"] = rank_df.groupby("Scene")["AUC_best"].rank(method="dense", ascending=False)
    out_rank = root / "all_scenes_full_rank.csv"
    rank_df.sort_values(["Scene", "rank_auc", "Method"]).to_csv(out_rank, index=False)

    print(f"saved: {out_all}")
    print(f"saved: {out_mean}")
    print(f"saved: {out_rank}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

