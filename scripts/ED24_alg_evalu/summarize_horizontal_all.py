from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


OUT_COLUMNS = [
    "dataset",
    "scene",
    "env",
    "algorithm",
    "tag",
    "radius_px",
    "tau_us",
    "auc",
    "f1",
    "tpr",
    "fpr",
    "precision",
    "accuracy",
    "mesr",
    "aocc",
    "runtime_sec",
    "threshold",
    "source_csv",
    "round",
]


def _run(py: str, script: Path) -> None:
    cmd = [py, str(script)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.stdout:
        print(r.stdout.rstrip())
    if r.returncode != 0:
        if r.stderr:
            print(r.stderr.rstrip(), file=sys.stderr)
        raise RuntimeError(f"failed: {' '.join(cmd)} (exit={r.returncode})")


def _tag_radius_tau(tag: str) -> tuple[str, str]:
    t = str(tag or "")
    m_r = re.search(r"_r(\d+)", t)
    # tau-like tokens in your tags: _tauNNN, _decayNNN
    m_tau = re.search(r"_(?:tau|decay)(\d+)", t)
    radius = m_r.group(1) if m_r else ""
    tau = m_tau.group(1) if m_tau else ""
    return radius, tau


def _normalize_round1(df1: pd.DataFrame) -> pd.DataFrame:
    if df1.empty:
        return pd.DataFrame(columns=OUT_COLUMNS)
    out = df1.copy()
    out["round"] = "round1"
    for c in OUT_COLUMNS:
        if c not in out.columns:
            out[c] = ""
    return out[OUT_COLUMNS]


def _normalize_round2(df2: pd.DataFrame) -> pd.DataFrame:
    if df2.empty:
        return pd.DataFrame(columns=OUT_COLUMNS)

    rows: list[dict[str, object]] = []
    for _, r in df2.iterrows():
        tag = str(r.get("best_f1_tag", "") or "")
        radius, tau = _tag_radius_tau(tag)
        rows.append(
            {
                "dataset": "ED24",
                "scene": "myPedestrain_06",
                "env": str(r.get("level", "") or ""),
                "algorithm": str(r.get("algorithm", "") or ""),
                "tag": tag,
                "radius_px": radius,
                "tau_us": tau,
                "auc": r.get("best_auc", ""),
                "f1": r.get("best_f1", ""),
                "tpr": r.get("best_f1_tpr", ""),
                "fpr": r.get("best_f1_fpr", ""),
                "precision": r.get("best_f1_precision", ""),
                "accuracy": "",
                "mesr": r.get("best_f1_mesr", ""),
                "aocc": r.get("best_f1_aocc", ""),
                "runtime_sec": "",
                "threshold": r.get("best_f1_threshold", ""),
                "source_csv": str(r.get("csv_path", "") or ""),
                "round": "round2",
            }
        )
    out = pd.DataFrame(rows)
    for c in OUT_COLUMNS:
        if c not in out.columns:
            out[c] = ""
    return out[OUT_COLUMNS]


def _norm_key(v: object) -> str:
    return str(v or "").strip().lower()


def _as_num_or_empty(v: object) -> object:
    s = str(v).strip()
    if s == "" or s.lower() == "nan":
        return ""
    try:
        return float(s)
    except Exception:
        return s


def _merge_mesr_aocc_from_bestpoint(all_df: pd.DataFrame, bestpoint_csv: Path) -> pd.DataFrame:
    if all_df.empty or (not bestpoint_csv.exists()):
        return all_df

    try:
        bp = pd.read_csv(bestpoint_csv)
    except Exception as e:
        print(f"warn: cannot read {bestpoint_csv}: {type(e).__name__}: {e}", file=sys.stderr)
        return all_df
    if bp.empty:
        return all_df

    need = {"dataset", "level", "algorithm", "point", "mesr", "aocc", "tag"}
    if not need.issubset(set(bp.columns)):
        miss = sorted(list(need - set(bp.columns)))
        print(f"warn: {bestpoint_csv} missing columns: {miss}", file=sys.stderr)
        return all_df

    # only use best-f1 for horizontal unified row update
    bp = bp[bp["point"].astype(str).str.lower() == "best-f1"].copy()
    if bp.empty:
        return all_df

    # exact map: dataset + level + algorithm + tag
    exact_map: dict[tuple[str, str, str, str], tuple[object, object]] = {}
    # fallback map: dataset + level + algorithm
    fallback_map: dict[tuple[str, str, str], tuple[object, object]] = {}
    for _, r in bp.iterrows():
        ds = _norm_key(r.get("dataset", ""))
        lv = _norm_key(r.get("level", ""))
        alg = _norm_key(r.get("algorithm", ""))
        tag = str(r.get("tag", "") or "").strip()
        mesr = _as_num_or_empty(r.get("mesr", ""))
        aocc = _as_num_or_empty(r.get("aocc", ""))
        if mesr == "" and aocc == "":
            continue
        if tag:
            exact_map[(ds, lv, alg, tag)] = (mesr, aocc)
        fallback_map[(ds, lv, alg)] = (mesr, aocc)

    if not exact_map and not fallback_map:
        return all_df

    out = all_df.copy()
    for i, r in out.iterrows():
        ds = _norm_key(r.get("dataset", ""))
        lv = _norm_key(r.get("env", ""))
        alg = _norm_key(r.get("algorithm", ""))
        tag = str(r.get("tag", "") or "").strip()
        pick = exact_map.get((ds, lv, alg, tag))
        if pick is None:
            pick = fallback_map.get((ds, lv, alg))
        if pick is None:
            continue
        mesr, aocc = pick
        if mesr != "":
            out.at[i, "mesr"] = mesr
        if aocc != "":
            out.at[i, "aocc"] = aocc
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Run round1+round2 summaries and merge into one aligned csv.")
    ap.add_argument("--python", default=sys.executable, help="python interpreter path")
    ap.add_argument("--out-csv", default="data/ED24/myPedestrain_06/horizontal_summary_all.csv")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    round1_script = root / "summarize_horizontal_round1.py"
    round2_script = root / "summarize_horizontal_round2_new_methods.py"

    _run(args.python, round1_script)
    _run(args.python, round2_script)

    round1_csv = Path("data/ED24/myPedestrain_06/horizontal_round1_summary.csv")
    round2_csv = Path("data/ED24/myPedestrain_06/horizontal_round2_new_methods_summary.csv")
    bestpoint_csv = Path("data/ED24/myPedestrain_06/bestpoint_mesr_aocc_summary.csv")

    n1 = pd.DataFrame(columns=OUT_COLUMNS)
    n2 = pd.DataFrame(columns=OUT_COLUMNS)
    if round1_csv.exists():
        try:
            n1 = _normalize_round1(pd.read_csv(round1_csv))
        except Exception as e:
            print(f"warn: cannot normalize {round1_csv}: {type(e).__name__}: {e}", file=sys.stderr)
    if round2_csv.exists():
        try:
            n2 = _normalize_round2(pd.read_csv(round2_csv))
        except Exception as e:
            print(f"warn: cannot normalize {round2_csv}: {type(e).__name__}: {e}", file=sys.stderr)

    all_df = pd.concat([n1, n2], ignore_index=True, sort=False)
    all_df = _merge_mesr_aocc_from_bestpoint(all_df, bestpoint_csv)
    if not all_df.empty:
        all_df = all_df.sort_values(["env", "algorithm", "round"]).reset_index(drop=True)
        all_df = all_df.fillna("")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        all_df.to_csv(out_path, index=False)
        saved_path = out_path
    except PermissionError:
        fallback = out_path.with_name(f"{out_path.stem}_latest{out_path.suffix}")
        all_df.to_csv(fallback, index=False)
        saved_path = fallback
        print(f"warn: target file is busy, wrote fallback file: {fallback}", file=sys.stderr)
    print(f"saved merged: {saved_path}")
    if not all_df.empty:
        print(all_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
