from __future__ import annotations

import argparse
import csv
import os

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


def u_quantiles(
    *,
    in_csv: str,
    out_csv: str,
    cats: list[str],
    outcomes: list[str],
    ps: list[float],
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
    p_cols = [f"p{int(round(p*100)):02d}" for p in ps]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["cat", "outcome", "n", *p_cols])
        w.writeheader()

        for (cat, oc), vals in sorted(groups.items()):
            a = np.asarray(vals, dtype=np.float64)
            if a.size == 0:
                continue
            qs = np.quantile(a, ps)
            row = {"cat": cat, "outcome": oc, "n": int(a.size)}
            for col, q in zip(p_cols, qs):
                row[col] = float(q)
            w.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute quantiles from dump_u_events CSV.")
    ap.add_argument("--in-csv", required=True)
    ap.add_argument("--out-csv", required=True)
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
    ap.add_argument(
        "--ps",
        default="0.1,0.25,0.5,0.75,0.9,0.95,0.99",
        help="Comma-separated quantile probabilities",
    )
    args = ap.parse_args()

    cats = [c.strip() for c in str(args.cats).split(",")] if str(args.cats).strip() else []
    outcomes = [o.strip() for o in str(args.outcomes).split(",")] if str(args.outcomes).strip() else []
    ps = [float(x) for x in str(args.ps).split(",") if str(x).strip()]

    u_quantiles(
        in_csv=str(args.in_csv),
        out_csv=str(args.out_csv),
        cats=cats,
        outcomes=outcomes,
        ps=ps,
        col=str(args.col),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
