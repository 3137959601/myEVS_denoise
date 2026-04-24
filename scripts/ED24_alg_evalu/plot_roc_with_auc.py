from __future__ import annotations

import argparse
import csv
import os
import re

import numpy as np


def _read_rows(path: str) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _strip_env_suffix(tag: str) -> str:
    return re.sub(r"_(light|mid|heavy)$", "", str(tag), flags=re.IGNORECASE)


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot ROC with cleaner labels and per-curve AUC in legend.")
    ap.add_argument("--in-csv", required=True, help="ROC csv path")
    ap.add_argument("--out-png", required=True, help="output png path")
    ap.add_argument("--title", default="ROC")
    ap.add_argument("--strip-env-suffix", action="store_true", help="remove _light/_mid/_heavy suffix from legend tags")
    ap.add_argument("--dpi", type=int, default=160)
    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        raise SystemExit(f"matplotlib is required: {e}")

    rows = _read_rows(args.in_csv)
    if not rows:
        raise SystemExit(f"empty csv: {args.in_csv}")

    tags = sorted({(r.get("tag") or "") for r in rows if (r.get("tag") or "")})
    if not tags:
        raise SystemExit(f"no tag column values in: {args.in_csv}")

    auc_by_tag: dict[str, float] = {}
    for r in rows:
        t = str(r.get("tag", ""))
        if not t or t in auc_by_tag:
            continue
        try:
            auc_by_tag[t] = float(r.get("auc", "nan"))
        except Exception:
            auc_by_tag[t] = float("nan")

    plt.figure(figsize=(8, 6), dpi=int(args.dpi))
    for tag in tags:
        fpr = [float(r["fpr"]) for r in rows if r.get("tag") == tag]
        tpr = [float(r["tpr"]) for r in rows if r.get("tag") == tag]
        if not fpr:
            continue
        show_tag = _strip_env_suffix(tag) if bool(args.strip_env_suffix) else tag
        auc = auc_by_tag.get(tag, float("nan"))
        if np.isfinite(auc):
            label = f"{show_tag} | AUC={auc:.6f}"
        else:
            label = show_tag
        plt.plot(fpr, tpr, linewidth=1.0, label=label)

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(str(args.title))
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(False)
    plt.legend(fontsize=7, ncol=1)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_png)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out_png)
    plt.close()

    print(f"saved: {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

