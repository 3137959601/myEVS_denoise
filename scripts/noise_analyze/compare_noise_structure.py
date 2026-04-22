from __future__ import annotations

import argparse
import csv
import os
from typing import Any

import numpy as np


def _try_float(x: Any) -> float | None:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        v = float(s)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return float(v)


def _read_rows(path: str) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise SystemExit(f"empty CSV header: {path}")
        rows = [dict(row) for row in r]
    return list(r.fieldnames), rows


def _numeric_columns(fieldnames: list[str], rows: list[dict[str, str]]) -> list[str]:
    skip = {
        "win_start",
        "win_end",
        "seg_start_events",
        "seg_max_events",
    }
    cols: list[str] = []
    for k in fieldnames:
        if k in skip:
            continue
        ok = False
        for row in rows[: min(20, len(rows))]:
            v = _try_float(row.get(k))
            if v is not None:
                ok = True
                break
        if ok:
            cols.append(k)
    return cols


def compare_noise_structure(*, seg0_csv: str, seg1_csv: str, out_csv: str, top_n: int = 12) -> None:
    f0, r0 = _read_rows(seg0_csv)
    f1, r1 = _read_rows(seg1_csv)

    cols0 = set(_numeric_columns(f0, r0))
    cols1 = set(_numeric_columns(f1, r1))
    cols = sorted(cols0 & cols1)
    if not cols:
        raise SystemExit("no common numeric columns")

    def _mean(rows: list[dict[str, str]], col: str) -> float:
        vals: list[float] = []
        for row in rows:
            v = _try_float(row.get(col))
            if v is not None:
                vals.append(v)
        if not vals:
            return float("nan")
        return float(np.mean(np.asarray(vals, dtype=np.float64)))

    items: list[dict[str, object]] = []
    for c in cols:
        m0 = _mean(r0, c)
        m1 = _mean(r1, c)
        if not (np.isfinite(m0) and np.isfinite(m1)):
            continue
        d = float(m1 - m0)
        ratio = float(m1 / m0) if abs(m0) > 1e-12 else float("inf")
        # Drift score: relative change magnitude.
        score = float(abs(d) / (abs(m0) + 1e-12))
        items.append(
            {
                "metric": c,
                "seg0_mean": float(m0),
                "seg1_mean": float(m1),
                "delta": float(d),
                "ratio": float(ratio),
                "rel_change": float(score),
            }
        )

    items.sort(key=lambda x: float(x["rel_change"]) if np.isfinite(float(x["rel_change"])) else -1.0, reverse=True)

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["metric", "seg0_mean", "seg1_mean", "delta", "ratio", "rel_change", "rank"],
        )
        w.writeheader()
        for i, it in enumerate(items, start=1):
            row = dict(it)
            row["rank"] = int(i)
            w.writerow(row)

    # Also print a compact top list to stdout for quick inspection.
    k = int(max(0, top_n))
    if k > 0:
        print("top drift metrics (by relative change):")
        for it in items[:k]:
            print(
                f"- {it['metric']}: seg0={it['seg0_mean']:.6g}, seg1={it['seg1_mean']:.6g}, "
                f"delta={it['delta']:.6g}, ratio={it['ratio']:.6g}"
            )


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare seg0 vs seg1 noise structure stats CSVs (mean over windows).")
    ap.add_argument("--seg0-csv", required=True)
    ap.add_argument("--seg1-csv", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--top-n", type=int, default=12)
    args = ap.parse_args()

    compare_noise_structure(
        seg0_csv=str(args.seg0_csv),
        seg1_csv=str(args.seg1_csv),
        out_csv=str(args.out_csv),
        top_n=int(args.top_n),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
