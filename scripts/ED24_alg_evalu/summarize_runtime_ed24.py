from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

LEVELS = ["light", "mid", "heavy"]


def _to_num(v):
    try:
        if v is None:
            return None
        s = str(v).strip()
        if s == "" or s.lower() == "nan":
            return None
        return float(s)
    except Exception:
        return None


def _pick_latest_per_level(df: pd.DataFrame) -> dict[str, float | None]:
    out: dict[str, float | None] = {lv: None for lv in LEVELS}
    if df.empty:
        return out

    for col in ["algorithm", "method", "param", "tag", "start_time"]:
        if col not in df.columns:
            continue

    if "level" not in df.columns or "elapsed_sec" not in df.columns:
        return out

    work = df.copy()
    work["level"] = work["level"].astype(str).str.strip().str.lower()
    work = work[work["level"].isin(LEVELS)]
    if work.empty:
        return out

    if "end_time" in work.columns:
        # Most runtime csvs contain timestamps; choose the latest record for each level.
        work["_end_time"] = pd.to_datetime(work["end_time"], errors="coerce")
        latest = (
            work.sort_values(["level", "_end_time"], ascending=[True, False])
            .groupby("level", as_index=False)
            .head(1)
        )
    else:
        # Fallback: if timestamps are absent, keep last row in file order per level.
        latest = work.groupby("level", as_index=False).tail(1)

    for _, r in latest.iterrows():
        lv = str(r["level"])
        out[lv] = _to_num(r.get("elapsed_sec"))
    return out


def _safe_round(v: float | None, nd=3):
    if v is None:
        return ""
    return round(float(v), nd)


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize ED24 runtime in a unified rule.")
    ap.add_argument(
        "--out-csv",
        default="data/ED24/myPedestrain_06/runtime_unified_ed24.csv",
        help="output csv",
    )
    args = ap.parse_args()

    root = Path("data/ED24/myPedestrain_06")

    runtime_files = {
        "BAF": root / "BAF" / "runtime_baf.csv",
        "STCF": root / "STCF" / "runtime_stcf.csv",
        "EBF": root / "EBF" / "runtime_ebf.csv",
        "N149": root / "N149_rerun_tmp" / "runtime_n149.csv",
        "KNOISE": root / "KNOISE" / "runtime_knoise.csv",
        "EVFLOW": root / "EVFLOW" / "runtime_evflow.csv",
        "YNOISE": root / "YNOISE" / "runtime_ynoise.csv",
        "TS": root / "TS" / "runtime_ts.csv",
        "MLPF": root / "MLPF" / "runtime_mlpf.csv",
        "PFD": root / "PFD" / "runtime_pfd.csv",
    }

    rows: list[dict[str, object]] = []

    for alg, p in runtime_files.items():
        row: dict[str, object] = {
            "algorithm": alg,
            "light_sec": "",
            "mid_sec": "",
            "heavy_sec": "",
            "avg_sec_per_level": "",
            "sum_sec_3levels": "",
            "n_levels": 0,
            "source_file": str(p).replace("\\", "/"),
            "note": "",
        }
        if not p.exists():
            row["note"] = "missing runtime csv"
            rows.append(row)
            continue

        try:
            df = pd.read_csv(p)
        except Exception as e:
            row["note"] = f"read error: {type(e).__name__}"
            rows.append(row)
            continue

        lv_map = _pick_latest_per_level(df)
        vals = [lv_map.get(lv) for lv in LEVELS]
        valid = [v for v in vals if v is not None]

        row["light_sec"] = _safe_round(lv_map.get("light"))
        row["mid_sec"] = _safe_round(lv_map.get("mid"))
        row["heavy_sec"] = _safe_round(lv_map.get("heavy"))
        row["n_levels"] = len(valid)

        if valid:
            row["avg_sec_per_level"] = _safe_round(sum(valid) / len(valid))
            row["sum_sec_3levels"] = _safe_round(sum(valid))
        else:
            row["note"] = "no valid level runtime"

        rows.append(row)

    out = pd.DataFrame(rows)
    order = [
        "EBF",
        "N149",
        "TS",
        "PFD",
        "BAF",
        "STCF",
        "KNOISE",
        "EVFLOW",
        "YNOISE",
        "MLPF",
    ]
    out["_ord"] = out["algorithm"].map({k: i for i, k in enumerate(order)}).fillna(999)
    out = out.sort_values("_ord").drop(columns=["_ord"]).reset_index(drop=True)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"saved: {out_path}")
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
