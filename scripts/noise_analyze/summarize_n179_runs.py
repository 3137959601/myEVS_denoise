from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _load_best(summary_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)
    if df.empty:
        return pd.DataFrame()
    dfa = df.sort_values(["auc", "best_f1"], ascending=[False, False]).iloc[[0]].copy()
    dfa["point"] = "best_auc"
    dff = df.sort_values(["best_f1", "auc"], ascending=[False, False]).iloc[[0]].copy()
    dff["point"] = "best_f1"
    return pd.concat([dfa, dff], ignore_index=True)


def _collect_pair(summary_csv: Path, runtime_csv: Path) -> pd.DataFrame:
    best = _load_best(summary_csv)
    if best.empty:
        return best
    rt = pd.read_csv(runtime_csv) if runtime_csv.exists() else pd.DataFrame()
    if not rt.empty:
        rt = rt.rename(columns={"level": "runtime_level"})
        best = best.merge(
            rt[["tag", "events", "duration_s", "event_rate_eps", "runtime_sec", "throughput_eps", "realtime_ok"]],
            on="tag",
            how="left",
        )
    return best


def _append_from_glob(rows: list[pd.DataFrame], pattern: str):
    for s in sorted(Path(".").glob(pattern)):
        r = s.with_name(s.name.replace("summary_", "runtime_"))
        try:
            rows.append(_collect_pair(s, r))
        except Exception:
            continue


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize n179 full-run results across datasets.")
    ap.add_argument("--out-csv", default="data/summary/n179_fullrun_summary_20260506.csv")
    args = ap.parse_args()

    rows: list[pd.DataFrame] = []
    _append_from_glob(rows, "data/ED24/*/N179/summary_n179_*.csv")
    _append_from_glob(rows, "data/DND21/mydriving/N179/summary_n179_*.csv")
    _append_from_glob(rows, "data/DVSCLEAN/scene_sweep_full/*/*/N179/summary_n179.csv")
    _append_from_glob(rows, "data/LED/scene_sweep_full/*/N179/summary_n179.csv")

    if not rows:
        raise SystemExit("no n179 summary files found")
    out = pd.concat(rows, ignore_index=True)
    for c in ["auc", "best_f1", "runtime_sec", "throughput_eps", "event_rate_eps"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if "best_f1_tpr" in out.columns and "best_f1_fpr" in out.columns:
        out["best_f1_tpr"] = pd.to_numeric(out["best_f1_tpr"], errors="coerce")
        out["best_f1_fpr"] = pd.to_numeric(out["best_f1_fpr"], errors="coerce")
        out["da_at_bestpoint"] = 0.5 * (out["best_f1_tpr"] + (1.0 - out["best_f1_fpr"]))
        out["snrdb_at_bestpoint"] = 10.0 * np.log10(
            np.maximum(out["best_f1_tpr"].to_numpy(dtype=float), 1e-12)
            / np.maximum(out["best_f1_fpr"].to_numpy(dtype=float), 1e-12)
        )
    out = out.sort_values(["dataset", "scene", "level", "point"]).reset_index(drop=True)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"saved: {out_path}")
    print(out.head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
