from __future__ import annotations

import argparse
import csv
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Iterable


_ENV_KEYS = ["light", "mid", "heavy"]
_ENV_LABEL = {"light": "1.8V", "mid": "2.5V", "heavy": "3.3V"}

_RE_S_TAU = re.compile(r"_s(\d+)_tau(\d+)(?:$|_)", re.IGNORECASE)


@dataclass(frozen=True)
class RocRow:
    tag: str
    thr: float
    f1: float | None
    tpr: float | None
    fpr: float | None
    precision: float | None
    accuracy: float | None
    auc: float | None


def _to_float(x: Any) -> float | None:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _parse_s_tau(tag: str) -> tuple[int | None, int | None]:
    m = _RE_S_TAU.search(tag or "")
    if not m:
        return None, None
    try:
        return int(m.group(1)), int(m.group(2))
    except Exception:
        return None, None


def _read_rows(csv_path: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if not row:
                continue
            rows.append({k: (v or "") for k, v in row.items() if k is not None})
    return rows


def _group_by_tag(rows: Iterable[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    g: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        tag = (row.get("tag") or "").strip()
        if not tag:
            continue
        g.setdefault(tag, []).append(row)
    return g


def _rows_for_tag(rows: list[dict[str, str]], tag: str) -> list[RocRow]:
    out: list[RocRow] = []
    for r in rows:
        if (r.get("tag") or "").strip() != tag:
            continue
        thr = _to_float(r.get("value"))
        if thr is None:
            continue
        out.append(
            RocRow(
                tag=tag,
                thr=float(thr),
                f1=_to_float(r.get("f1")),
                tpr=_to_float(r.get("tpr")),
                fpr=_to_float(r.get("fpr")),
                precision=_to_float(r.get("precision")),
                accuracy=_to_float(r.get("accuracy")),
                auc=_to_float(r.get("auc")),
            )
        )
    out.sort(key=lambda rr: rr.thr)
    return out


def _best_row_by_f1(tag_rows: list[RocRow]) -> RocRow:
    def key(r: RocRow) -> tuple[float, float, float, float]:
        f1v = float(r.f1) if r.f1 is not None else -1.0
        tprv = float(r.tpr) if r.tpr is not None else 0.0
        precv = float(r.precision) if r.precision is not None else 0.0
        fprv = float(r.fpr) if r.fpr is not None else 1.0
        return (f1v, tprv, precv, -fprv)

    if not tag_rows:
        raise RuntimeError("empty tag rows")
    return max(tag_rows, key=key)


def _nearest_by_thr(rows: list[RocRow], thr: float) -> RocRow:
    if not rows:
        raise RuntimeError("empty tag rows")
    lo = 0
    hi = len(rows) - 1
    if thr <= rows[0].thr:
        return rows[0]
    if thr >= rows[-1].thr:
        return rows[-1]
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if rows[mid].thr < thr:
            lo = mid
        else:
            hi = mid
    return rows[lo] if abs(rows[lo].thr - thr) <= abs(rows[hi].thr - thr) else rows[hi]


def _thr_stats(vals: list[float]) -> tuple[float, float, float, float, float]:
    if not vals:
        return (math.nan, math.nan, math.nan, math.nan, math.nan)
    mn = min(vals)
    mx = max(vals)
    mean = sum(vals) / float(len(vals))
    var = sum((x - mean) ** 2 for x in vals) / float(len(vals))
    std = math.sqrt(var)
    return (mn, mx, mx - mn, mean, std)


def _best_global_f1_threshold(env_rows: dict[str, list[RocRow]]) -> float:
    cand: set[float] = set()
    for rows in env_rows.values():
        for r in rows:
            cand.add(float(r.thr))
    cands = sorted(cand)
    if not cands:
        raise RuntimeError("no threshold candidates")

    best_thr = cands[0]
    best_score = -1.0
    for thr in cands:
        f1s: list[float] = []
        for env in _ENV_KEYS:
            rr = _nearest_by_thr(env_rows[env], float(thr))
            f1v = float(rr.f1) if rr.f1 is not None else -1.0
            f1s.append(f1v)
        mean_f1 = sum(f1s) / float(len(f1s))
        if mean_f1 > best_score + 1e-15:
            best_score = mean_f1
            best_thr = thr

    return float(best_thr)


def _find_default_env_csv(in_dir: str, env: str) -> str:
    patt = re.compile(
        rf"^roc_ebf_optimized_(?:[a-z0-9_]+_)?{re.escape(env)}_labelscore.*\.csv$",
        re.IGNORECASE,
    )
    if not os.path.isdir(in_dir):
        raise FileNotFoundError(in_dir)
    cand = [
        os.path.join(in_dir, fn)
        for fn in os.listdir(in_dir)
        if os.path.isfile(os.path.join(in_dir, fn)) and patt.match(fn)
    ]
    if not cand:
        raise FileNotFoundError(f"missing roc csv for env={env} under: {in_dir}")
    return max(cand, key=lambda p: os.path.getmtime(p))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Select best (s,tau) tag for EBF_optimized using fixed-threshold mean-F1 and/or threshold stability, "
            "based on existing ROC CSV outputs (light/mid/heavy)."
        )
    )
    ap.add_argument(
        "--in-dir",
        default="data/ED24/myPedestrain_06/EBF_optimized_V8_binrate",
        help="Directory containing roc_ebf_optimized_{env}_labelscore*.csv (one per env)",
    )
    ap.add_argument("--dataset", default="myPedestrain_06")
    ap.add_argument(
        "--select",
        default="fixed-f1",
        choices=["fixed-f1", "stable-then-fixed", "stable-under-auc"],
        help=(
            "How to rank tags. fixed-f1: maximize fixed-threshold mean-F1. "
            "stable-then-fixed: minimize Thr_std then maximize fixed-mean-F1. "
            "stable-under-auc: among tags with mean-AUC >= (best-mean-AUC - eps), minimize Thr_std then maximize fixed-mean-F1."
        ),
    )
    ap.add_argument(
        "--auc-eps",
        type=float,
        default=0.005,
        help="Only used for select=stable-under-auc. Allow mean-AUC drop within eps from best-mean-AUC.",
    )
    ap.add_argument(
        "--min-mean-auc",
        type=float,
        default=float("nan"),
        help="If set, discard tags with mean-AUC < this value.",
    )
    ap.add_argument(
        "--out-csv",
        default="",
        help="Output per-tag summary CSV. Default: <in-dir>/tag_selection_summary.csv",
    )

    args = ap.parse_args()

    in_dir = str(args.in_dir)
    dataset = str(args.dataset)
    select_mode = str(args.select)
    auc_eps = float(args.auc_eps)
    min_mean_auc = float(args.min_mean_auc)

    env_csv = {env: _find_default_env_csv(in_dir, env) for env in _ENV_KEYS}
    env_rows_raw = {env: _read_rows(env_csv[env]) for env in _ENV_KEYS}

    tags_common: set[str] | None = None
    for env in _ENV_KEYS:
        tags_env = set((r.get("tag") or "").strip() for r in env_rows_raw[env] if (r.get("tag") or "").strip())
        tags_common = tags_env if tags_common is None else (tags_common & tags_env)

    tags = sorted(tags_common or set())
    if not tags:
        raise SystemExit("no common tags across env csvs")

    summary_rows: list[dict[str, Any]] = []

    # Precompute best mean-AUC (for stable-under-auc)
    mean_auc_by_tag: dict[str, float] = {}
    for tag in tags:
        aucs: list[float] = []
        ok = True
        for env in _ENV_KEYS:
            # AUC is constant within a tag; take first valid
            auc_val: float | None = None
            for r in env_rows_raw[env]:
                if (r.get("tag") or "").strip() != tag:
                    continue
                auc_val = _to_float(r.get("auc"))
                if auc_val is not None:
                    break
            if auc_val is None:
                ok = False
                break
            aucs.append(float(auc_val))
        if ok:
            mean_auc_by_tag[tag] = sum(aucs) / float(len(aucs))

    if not mean_auc_by_tag:
        raise SystemExit("no tag has valid AUC across envs")

    best_mean_auc = max(mean_auc_by_tag.values())

    # Build per-tag statistics
    for tag in tags:
        mean_auc = mean_auc_by_tag.get(tag)
        if mean_auc is None:
            continue
        if min_mean_auc == min_mean_auc and mean_auc < float(min_mean_auc) - 1e-15:
            continue

        env_rows = {env: _rows_for_tag(env_rows_raw[env], tag) for env in _ENV_KEYS}
        if any(not env_rows[env] for env in _ENV_KEYS):
            continue

        # per-env best by F1 (for stability comparison)
        best_by_env = {env: _best_row_by_f1(env_rows[env]) for env in _ENV_KEYS}
        thrs = [float(best_by_env[env].thr) for env in _ENV_KEYS]
        thr_min, thr_max, thr_range, thr_mean, thr_std = _thr_stats(thrs)

        # fixed-threshold (best-global-f1) for this tag
        thr_fixed = _best_global_f1_threshold(env_rows)
        fixed_by_env = {env: _nearest_by_thr(env_rows[env], thr_fixed) for env in _ENV_KEYS}
        fixed_f1s = [float(fixed_by_env[env].f1) if fixed_by_env[env].f1 is not None else float("nan") for env in _ENV_KEYS]
        fixed_mean_f1 = sum(fixed_f1s) / float(len(fixed_f1s))

        s, tau_us = _parse_s_tau(tag)

        row: dict[str, Any] = {
            "dataset": dataset,
            "tag": tag,
            "s": s if s is not None else "",
            "tau(us)": tau_us if tau_us is not None else "",
            "auc_light": _to_float(best_by_env["light"].auc),
            "auc_mid": _to_float(best_by_env["mid"].auc),
            "auc_heavy": _to_float(best_by_env["heavy"].auc),
            "auc_mean": mean_auc,
            "thr_bestf1_light": best_by_env["light"].thr,
            "thr_bestf1_mid": best_by_env["mid"].thr,
            "thr_bestf1_heavy": best_by_env["heavy"].thr,
            "thr_range": thr_range,
            "thr_std": thr_std,
            "thr_fixed_bestglobalf1": thr_fixed,
            "f1_fixed_light": fixed_by_env["light"].f1,
            "f1_fixed_mid": fixed_by_env["mid"].f1,
            "f1_fixed_heavy": fixed_by_env["heavy"].f1,
            "f1_fixed_mean": fixed_mean_f1,
        }
        summary_rows.append(row)

    if not summary_rows:
        raise SystemExit("no valid tags after filtering")

    # Apply selection
    candidates = list(summary_rows)
    if select_mode == "stable-under-auc":
        thr = best_mean_auc - auc_eps
        candidates = [r for r in candidates if float(r["auc_mean"]) >= thr - 1e-15]
        if not candidates:
            candidates = list(summary_rows)

    def key_fixed_f1(r: dict[str, Any]) -> tuple[float, float, float]:
        return (float(r["f1_fixed_mean"]), -float(r["thr_std"]), float(r["auc_mean"]))

    def key_stable_then_fixed(r: dict[str, Any]) -> tuple[float, float, float]:
        return (-float(r["thr_std"]), float(r["f1_fixed_mean"]), float(r["auc_mean"]))

    if select_mode == "fixed-f1":
        best = max(candidates, key=key_fixed_f1)
    else:
        best = max(candidates, key=key_stable_then_fixed)

    out_csv = str(args.out_csv).strip() or os.path.join(in_dir, "tag_selection_summary.csv")
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)

    header = [
        "dataset",
        "tag",
        "s",
        "tau(us)",
        "auc_light",
        "auc_mid",
        "auc_heavy",
        "auc_mean",
        "thr_bestf1_light",
        "thr_bestf1_mid",
        "thr_bestf1_heavy",
        "thr_range",
        "thr_std",
        "thr_fixed_bestglobalf1",
        "f1_fixed_light",
        "f1_fixed_mid",
        "f1_fixed_heavy",
        "f1_fixed_mean",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in sorted(summary_rows, key=lambda rr: (float(rr["f1_fixed_mean"]), -float(rr["auc_mean"])), reverse=True):
            w.writerow({k: r.get(k, "") for k in header})

    print(f"saved: {out_csv}")
    print("=== SELECTION ===")
    print(f"select={select_mode} best_mean_auc={best_mean_auc:.6g} auc_eps={auc_eps}")
    print(
        "best_tag={tag} s={s} tau_us={tau} auc_mean={auc:.6g} thr_std={std:.6g} fixed_mean_f1={mf1:.6g} thr_fixed={thr:.6g}".format(
            tag=best["tag"],
            s=best["s"],
            tau=best["tau(us)"],
            auc=float(best["auc_mean"]),
            std=float(best["thr_std"]),
            mf1=float(best["f1_fixed_mean"]),
            thr=float(best["thr_fixed_bestglobalf1"]),
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
