from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable


_ENV_KEYS = ["light", "mid", "heavy"]
_ENV_LABEL = {"light": "1.8V", "mid": "2.5V", "heavy": "3.3V"}


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
    raw: dict[str, str]


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


def _pick_best_row_by_f1(rows: list[dict[str, str]]) -> dict[str, str]:
    def key(row: dict[str, str]) -> tuple[float, float, float, float]:
        f1 = _to_float(row.get("f1"))
        tpr = _to_float(row.get("tpr"))
        prec = _to_float(row.get("precision"))
        fpr = _to_float(row.get("fpr"))

        f1v = float(f1) if f1 is not None else -1.0
        tprv = float(tpr) if tpr is not None else 0.0
        precv = float(prec) if prec is not None else 0.0
        fprv = float(fpr) if fpr is not None else 1.0
        return (f1v, tprv, precv, -fprv)

    return max(rows, key=key)


def _rows_for_tag(csv_path: str, tag: str) -> list[RocRow]:
    rows = _read_rows(csv_path)
    tag_rows = [r for r in rows if (r.get("tag") or "").strip() == tag]
    out: list[RocRow] = []
    for r in tag_rows:
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
                raw=r,
            )
        )
    out.sort(key=lambda rr: rr.thr)
    return out


def _nearest_by_thr(rows: list[RocRow], thr: float) -> RocRow:
    if not rows:
        raise RuntimeError("empty tag rows")
    # binary-search like nearest
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
    # rows[lo].thr < thr <= rows[hi].thr
    return rows[lo] if abs(rows[lo].thr - thr) <= abs(rows[hi].thr - thr) else rows[hi]


def _best_global_tag(env_csv: dict[str, str]) -> str:
    # Pick tag that maximizes mean AUC across envs (only tags present in all envs).
    by_env: dict[str, dict[str, float]] = {}
    common: set[str] | None = None

    for env, path in env_csv.items():
        rows = _read_rows(path)
        g = _group_by_tag(rows)
        auc_by_tag: dict[str, float] = {}
        for tag, tag_rows in g.items():
            auc_val: float | None = None
            for r in tag_rows:
                auc_val = _to_float(r.get("auc"))
                if auc_val is not None:
                    break
            if auc_val is None:
                continue
            auc_by_tag[tag] = float(auc_val)
        by_env[env] = auc_by_tag
        tags = set(auc_by_tag.keys())
        common = tags if common is None else (common & tags)

    if not common:
        raise RuntimeError("no common tags across env csvs")

    best_tag = ""
    best_mean_auc = -1.0
    for tag in sorted(common):
        aucs = [by_env[env][tag] for env in _ENV_KEYS]
        mean_auc = sum(aucs) / float(len(aucs))
        if mean_auc > best_mean_auc + 1e-15:
            best_mean_auc = mean_auc
            best_tag = tag
    if not best_tag:
        raise RuntimeError("failed to pick best tag")
    return best_tag


def _best_f1_thr(tag_rows: list[RocRow]) -> float:
    if not tag_rows:
        raise RuntimeError("empty tag rows")
    best = None
    best_key: tuple[float, float, float, float] | None = None
    for r in tag_rows:
        f1v = float(r.f1) if r.f1 is not None else -1.0
        tprv = float(r.tpr) if r.tpr is not None else 0.0
        precv = float(r.precision) if r.precision is not None else 0.0
        fprv = float(r.fpr) if r.fpr is not None else 1.0
        k = (f1v, tprv, precv, -fprv)
        if best_key is None or k > best_key:
            best_key = k
            best = r
    assert best is not None
    return float(best.thr)


def _thr_mode_best_global_f1(env_rows: dict[str, list[RocRow]]) -> float:
    # Candidate thresholds: union of all observed thresholds.
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
            row = _nearest_by_thr(env_rows[env], thr)
            f1v = float(row.f1) if row.f1 is not None else -1.0
            f1s.append(f1v)
        mean_f1 = sum(f1s) / float(len(f1s))
        if mean_f1 > best_score + 1e-15:
            best_score = mean_f1
            best_thr = thr

    return float(best_thr)


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def _write_csv(path: str, header: list[str], rows: list[dict[str, Any]]) -> str:
    _ensure_dir(path)
    try:
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        return path
    except PermissionError:
        base, ext = os.path.splitext(path)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        alt = f"{base}_{ts}{ext or '.csv'}"
        with open(alt, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        return alt


def _find_default_env_csv(in_dir: str, env: str) -> str:
    # Compatible naming:
    # - legacy: roc_ebf_optimized_{env}_labelscore_*.csv
    # - new:    roc_ebf_optimized_{variant}_{env}_labelscore_*.csv
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Evaluate ONE fixed threshold across light/mid/heavy for EBF_optimized ROC CSV outputs. "
            "Useful for checking 'one-threshold generalization' of normalized scores."
        )
    )
    ap.add_argument(
        "--in-dir",
        default="data/ED24/myPedestrain_06/EBF_optimized_V2scale_linear",
        help="Directory containing roc_ebf_optimized_{env}_*.csv",
    )
    ap.add_argument("--dataset", default="myPedestrain_06")
    ap.add_argument(
        "--tag",
        default="",
        help=(
            "(Optional) Force a specific tag, e.g. ebfopt_labelscore_s9_tau128000. "
            "If empty, pick the tag that maximizes mean AUC across envs."
        ),
    )
    ap.add_argument(
        "--thr-mode",
        default="best-global-f1",
        choices=["best-global-f1", "from-env-bestf1", "mean-bestf1", "median-bestf1", "explicit"],
        help=(
            "How to choose the single fixed threshold. "
            "best-global-f1 scans candidate thresholds and picks max mean-F1 across envs (nearest-match)."
        ),
    )
    ap.add_argument(
        "--thr-env",
        default="mid",
        choices=_ENV_KEYS,
        help="Used when thr-mode=from-env-bestf1 (choose which env's best-F1 threshold becomes the fixed threshold).",
    )
    ap.add_argument(
        "--thr",
        type=float,
        default=float("nan"),
        help="Used when thr-mode=explicit",
    )
    ap.add_argument(
        "--out-csv",
        default="",
        help="Output CSV path. Default: <in-dir>/fixed_thr_eval_ebfopt.csv",
    )

    args = ap.parse_args(argv)

    in_dir = str(args.in_dir)
    dataset = str(args.dataset)

    env_csv = {env: _find_default_env_csv(in_dir, env) for env in _ENV_KEYS}

    tag = str(args.tag).strip()
    if not tag:
        tag = _best_global_tag(env_csv)

    env_rows = {env: _rows_for_tag(env_csv[env], tag) for env in _ENV_KEYS}
    for env in _ENV_KEYS:
        if not env_rows[env]:
            raise SystemExit(f"tag not found in env={env}: {tag!r} (csv={env_csv[env]})")

    thr_mode = str(args.thr_mode)
    if thr_mode == "explicit":
        thr_fixed = float(args.thr)
        if thr_fixed != thr_fixed:  # nan
            raise SystemExit("--thr-mode explicit requires --thr")
    elif thr_mode == "from-env-bestf1":
        thr_fixed = _best_f1_thr(env_rows[str(args.thr_env)])
    elif thr_mode in {"mean-bestf1", "median-bestf1"}:
        thrs = [_best_f1_thr(env_rows[env]) for env in _ENV_KEYS]
        if thr_mode == "mean-bestf1":
            thr_fixed = sum(thrs) / float(len(thrs))
        else:
            thrs2 = sorted(thrs)
            thr_fixed = thrs2[1]
    elif thr_mode == "best-global-f1":
        thr_fixed = _thr_mode_best_global_f1(env_rows)
    else:
        raise SystemExit(f"unknown --thr-mode: {thr_mode}")

    header = [
        "dataset",
        "env",
        "env_label",
        "tag",
        "thr_mode",
        "thr_fixed",
        "thr_used",
        "thr_abs_err",
        "auc",
        "f1",
        "tpr",
        "fpr",
        "precision",
        "accuracy",
        "source_csv",
    ]

    out_rows: list[dict[str, Any]] = []
    f1s: list[float] = []
    for env in _ENV_KEYS:
        nearest = _nearest_by_thr(env_rows[env], float(thr_fixed))
        f1v = float(nearest.f1) if nearest.f1 is not None else float("nan")
        if f1v == f1v:
            f1s.append(f1v)
        out_rows.append(
            {
                "dataset": dataset,
                "env": env,
                "env_label": _ENV_LABEL.get(env, env),
                "tag": tag,
                "thr_mode": thr_mode,
                "thr_fixed": float(thr_fixed),
                "thr_used": float(nearest.thr),
                "thr_abs_err": abs(float(nearest.thr) - float(thr_fixed)),
                "auc": nearest.auc if nearest.auc is not None else "",
                "f1": nearest.f1 if nearest.f1 is not None else "",
                "tpr": nearest.tpr if nearest.tpr is not None else "",
                "fpr": nearest.fpr if nearest.fpr is not None else "",
                "precision": nearest.precision if nearest.precision is not None else "",
                "accuracy": nearest.accuracy if nearest.accuracy is not None else "",
                "source_csv": env_csv[env].replace("\\\\", "/"),
            }
        )

    mean_f1 = (sum(f1s) / float(len(f1s))) if f1s else float("nan")
    out_rows.append(
        {
            "dataset": dataset,
            "env": "MEAN",
            "env_label": "MEAN",
            "tag": tag,
            "thr_mode": thr_mode,
            "thr_fixed": float(thr_fixed),
            "thr_used": "",
            "thr_abs_err": "",
            "auc": "",
            "f1": mean_f1,
            "tpr": "",
            "fpr": "",
            "precision": "",
            "accuracy": "",
            "source_csv": "",
        }
    )

    out_csv = str(args.out_csv).strip() or os.path.join(in_dir, "fixed_thr_eval_ebfopt.csv")
    out_path = _write_csv(out_csv, header, out_rows)

    print(f"tag={tag}")
    print(f"thr_mode={thr_mode} thr_fixed={thr_fixed}")
    for r in out_rows:
        if r["env"] == "MEAN":
            continue
        print(
            f"{r['env']}: thr_used={r['thr_used']:.6g} f1={r['f1']} tpr={r['tpr']} fpr={r['fpr']} auc={r['auc']}"
        )
    print(f"MEAN_F1={mean_f1}")
    print(f"saved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
