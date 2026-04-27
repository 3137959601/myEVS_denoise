from __future__ import annotations

from pathlib import Path
import pandas as pd
from pandas.errors import EmptyDataError


def _get_metric_col(df: pd.DataFrame, cands: list[str]) -> str | None:
    cols = {str(c).lower(): str(c) for c in df.columns}
    for c in cands:
        if c.lower() in cols:
            return cols[c.lower()]
    return None


def main() -> None:
    root = Path("data/ED24/myPedestrain_06")
    alg_dirs = {
        "knoise": root / "KNOISE",
        "evflow": root / "EVFLOW",
        "ynoise": root / "YNOISE",
        "ts": root / "TS",
        "mlpf": root / "MLPF",
        "pfd": root / "PFD",
    }
    levels = ("light", "mid", "heavy")

    rows = []
    skipped: list[str] = []
    for alg, directory in alg_dirs.items():
        for level in levels:
            csv_path = directory / f"roc_{alg}_{level}.csv"
            if not csv_path.exists():
                continue

            try:
                df = pd.read_csv(csv_path)
            except EmptyDataError:
                skipped.append(f"{alg}/{level}: empty csv (no header) -> {csv_path}")
                continue
            except Exception as e:
                skipped.append(f"{alg}/{level}: read failed ({type(e).__name__}) -> {csv_path}")
                continue
            if df.empty:
                skipped.append(f"{alg}/{level}: empty dataframe -> {csv_path}")
                continue

            need_cols = {"tag", "auc", "f1", "value", "precision", "tpr", "fpr"}
            missing = [c for c in need_cols if c not in df.columns]
            if missing:
                skipped.append(f"{alg}/{level}: missing columns {missing} -> {csv_path}")
                continue

            by_tag_auc = df.groupby("tag", as_index=False)["auc"].max()
            best_auc_idx = by_tag_auc["auc"].idxmax()
            best_auc_tag = str(by_tag_auc.loc[best_auc_idx, "tag"])
            best_auc = float(by_tag_auc.loc[best_auc_idx, "auc"])
            tag_rows = df[df["tag"] == best_auc_tag].copy()
            best_auc_op_idx = tag_rows["f1"].idxmax()
            best_auc_op_row = df.loc[best_auc_op_idx]

            best_f1_idx = df["f1"].idxmax()
            best_row = df.loc[best_f1_idx]
            esr_col = _get_metric_col(df, ["esr_mean", "mesr"])
            aocc_col = _get_metric_col(df, ["aocc"])

            rows.append(
                {
                    "algorithm": alg.upper(),
                    "level": level,
                    "best_auc": best_auc,
                    "best_auc_tag": best_auc_tag,
                    "best_auc_threshold": float(best_auc_op_row["value"]),
                    "best_f1": float(best_row["f1"]),
                    "best_f1_tag": str(best_row["tag"]),
                    "best_f1_threshold": float(best_row["value"]),
                    "best_f1_precision": float(best_row["precision"]),
                    "best_f1_tpr": float(best_row["tpr"]),
                    "best_f1_fpr": float(best_row["fpr"]),
                    "best_auc_mesr": (float(best_auc_op_row[esr_col]) if esr_col and str(best_auc_op_row[esr_col]).strip() else None),
                    "best_auc_aocc": (float(best_auc_op_row[aocc_col]) if aocc_col and str(best_auc_op_row[aocc_col]).strip() else None),
                    "best_f1_mesr": (float(best_row[esr_col]) if esr_col and str(best_row[esr_col]).strip() else None),
                    "best_f1_aocc": (float(best_row[aocc_col]) if aocc_col and str(best_row[aocc_col]).strip() else None),
                    "csv_path": str(csv_path).replace("\\", "/"),
                }
            )

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        print("no valid rows found for round2 summary.")
    else:
        out_df = out_df.sort_values(["algorithm", "level"]).reset_index(drop=True)
    out_file = root / "horizontal_round2_new_methods_summary.csv"
    out_df.to_csv(out_file, index=False)
    print(f"saved: {out_file}")
    if not out_df.empty:
        print(out_df.to_string(index=False))
    if skipped:
        print("skipped files:")
        for s in skipped:
            print(f"  - {s}")


if __name__ == "__main__":
    main()
