from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from typing import Any
import pandas as pd


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


@dataclass(frozen=True)
class AlgoCsv:
    algorithm: str
    env: str
    path: str | list[str]


def _best_stats(path: str) -> dict[str, object]:
    rows: list[dict[str, str]] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({k: (v or "") for k, v in r.items() if k is not None})
    if not rows:
        raise RuntimeError(f"empty csv: {path}")

    # 1) pick best tag by AUC
    by_tag: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        tag = (r.get("tag") or "").strip()
        if not tag:
            continue
        by_tag.setdefault(tag, []).append(r)
    if not by_tag:
        raise RuntimeError(f"no tag in csv: {path}")

    best_auc_tag = ""
    best_auc = -1.0
    for tag, lst in by_tag.items():
        auc = _to_float(lst[0].get("auc")) or -1.0
        if auc > best_auc:
            best_auc = auc
            best_auc_tag = tag

    def f1_key(r: dict[str, str]) -> tuple[float, float, float]:
        f1 = _to_float(r.get("f1")) or -1.0
        tpr = _to_float(r.get("tpr")) or 0.0
        fpr = _to_float(r.get("fpr")) or 1.0
        return (float(f1), float(tpr), -float(fpr))

    # 2) within best-AUC tag, pick best threshold by F1
    best_auc_cand = by_tag[best_auc_tag]
    best_auc_op = max(best_auc_cand, key=f1_key)

    # 3) global best-F1 point (across all tags/thresholds)
    best_f1_op = max(rows, key=f1_key)

    return {
        "best_auc": float(best_auc),
        "best_auc_tag": best_auc_tag,
        "best_auc_threshold": best_auc_op.get("value", ""),
        "best_f1_op": best_f1_op,
    }


def _extract(tag: str) -> tuple[str, str]:
    t = str(tag or "")
    m_r = re.search(r"_r(\d+)", t)
    m_tau = re.search(r"_tau(\d+)", t)
    r = m_r.group(1) if m_r else ""
    tau = m_tau.group(1) if m_tau else ""
    return r, tau


def _get_runtime_sec(algorithm: str, env: str) -> float | None:
    alg = str(algorithm).strip().lower()
    base_dir = os.path.join("data", "ED24", "myPedestrain_06")
    if alg in {"baf", "stcf", "ebf"}:
        p = os.path.join(base_dir, alg.upper(), f"runtime_{alg}.csv")
    elif alg == "n149":
        p = os.path.join(base_dir, "N149", "runtime_n149.csv")
    else:
        p = ""
    if p and os.path.exists(p):
        try:
            df = pd.read_csv(p)
            hit = df[df["level"].astype(str).str.lower() == str(env).lower()]
            if not hit.empty and "elapsed_sec" in hit.columns:
                return float(hit.iloc[-1]["elapsed_sec"])
        except Exception:
            pass
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize ED24 horizontal round1 best points.")
    ap.add_argument("--out-csv", default="data/ED24/myPedestrain_06/horizontal_round1_summary.csv")
    args = ap.parse_args()

    items: list[AlgoCsv] = []
    for env in ("light", "mid", "heavy"):
        items.append(AlgoCsv("BAF", env, f"data/ED24/myPedestrain_06/BAF/roc_baf_{env}.csv"))
        items.append(AlgoCsv("STCF", env, f"data/ED24/myPedestrain_06/STCF/roc_stcf_{env}.csv"))
        items.append(
            AlgoCsv(
                "EBF",
                env,
                [
                    f"data/ED24/myPedestrain_06/EBF/roc_ebf_{env}.csv",
                    f"data/ED24/myPedestrain_06/EBF/roc_ebf_{env}_label_exact.csv",
                ],
            )
        )
        items.append(AlgoCsv("N149", env, f"data/ED24/myPedestrain_06/N149/roc_n149_{env}.csv"))

    out_rows: list[dict[str, Any]] = []
    for it in items:
        if isinstance(it.path, str):
            src_csv = it.path
            if not os.path.exists(src_csv):
                raise FileNotFoundError(src_csv)
        else:
            src_csv = ""
            for cand in it.path:
                if os.path.exists(cand):
                    src_csv = cand
                    break
            if not src_csv:
                raise FileNotFoundError(str(it.path))

        stats = _best_stats(src_csv)
        br = stats["best_f1_op"]
        assert isinstance(br, dict)
        tag = br.get("tag", "")
        r, tau = _extract(tag)
        esr = br.get("esr_mean", "") or br.get("mesr", "")
        aocc = br.get("aocc", "")
        runtime_sec = _get_runtime_sec(it.algorithm, it.env)
        out_rows.append(
            {
                "dataset": "ED24",
                "scene": "myPedestrain_06",
                "env": it.env,
                "algorithm": it.algorithm,
                "tag": tag,
                "radius_px": r,
                "tau_us": tau,
                "auc": br.get("auc", ""),
                "f1": br.get("f1", ""),
                "best_auc": stats.get("best_auc", ""),
                "best_auc_tag": stats.get("best_auc_tag", ""),
                "best_auc_threshold": stats.get("best_auc_threshold", ""),
                "tpr": br.get("tpr", ""),
                "fpr": br.get("fpr", ""),
                "precision": br.get("precision", ""),
                "accuracy": br.get("accuracy", ""),
                "mesr": esr,
                "aocc": aocc,
                "runtime_sec": ("" if runtime_sec is None else runtime_sec),
                "threshold": br.get("value", ""),
                "source_csv": src_csv.replace("\\", "/"),
            }
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    header = [
        "dataset",
        "scene",
        "env",
        "algorithm",
        "tag",
        "radius_px",
        "tau_us",
        "auc",
        "f1",
        "best_auc",
        "best_auc_tag",
        "best_auc_threshold",
        "tpr",
        "fpr",
        "precision",
        "accuracy",
        "mesr",
        "aocc",
        "runtime_sec",
        "threshold",
        "source_csv",
    ]
    with open(args.out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for row in out_rows:
            w.writerow(row)

    print(f"saved: {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
