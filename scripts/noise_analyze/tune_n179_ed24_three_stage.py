from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

LEVELS = {
    "light": "D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_1.8.npy",
    "light_mid": "D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_2.1.npy",
    "mid": "D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_2.5.npy",
    "heavy": "D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3.npy",
}


def run(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    p = subprocess.run(cmd)
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def read_best_auc(summary_csv: Path) -> dict[str, str]:
    rows = []
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        rows = list(rd)
    if not rows:
        raise SystemExit(f"empty summary: {summary_csv}")
    rows.sort(key=lambda r: float(r.get("auc", "-1")), reverse=True)
    best = rows[0]
    return {
        "s": best["s"],
        "tau_us": best["tau_us"],
        "k_sfrac": best["k_sfrac"],
        "k_mix": best["k_mix"],
        "auc": best["auc"],
        "f1": best["best_f1"],
        "tag": best["tag"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Three-stage N179 tuning on ED24 pedestrian under main pipeline")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--max-events", type=int, default=0)
    ap.add_argument("--out-root", default="data/ED24/myPedestrain_06/N179/tune3")
    ap.add_argument("--s-list", default="5,7,9")
    ap.add_argument("--tau-stage1", default="16000,32000,64000,128000,256000,512000")
    ap.add_argument("--k-sfrac-stage2", default="0.4,0.55,0.67,0.8,0.9")
    ap.add_argument("--k-mix-stage2", default="0.0,0.04,0.08,0.125,0.16,0.3,1.0")
    ap.add_argument("--signal-label-value", type=int, default=1)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    final_rows = []

    for level, noisy in LEVELS.items():
        print(f"\n===== [{level}] stage1: tau sweep with fixed K =====", flush=True)
        ldir = out_root / level
        ldir.mkdir(parents=True, exist_ok=True)

        s1_roc = ldir / "stage1_roc.csv"
        s1_sum = ldir / "stage1_summary.csv"
        s1_run = ldir / "stage1_runtime.csv"
        run([
            args.python,
            "scripts/noise_analyze/sweep_n179_pair.py",
            "--noisy", noisy,
            "--out-csv", str(s1_roc),
            "--summary-csv", str(s1_sum),
            "--runtime-csv", str(s1_run),
            "--dataset", "ED24",
            "--scene", "myPedestrain_06",
            "--level", level,
            "--width", "346",
            "--height", "260",
            "--tick-ns", "1000",
            "--s-list", args.s_list,
            "--tau-us-list", args.tau_stage1,
            "--k-sfrac-list", "0.8",
            "--k-mix-list", "0.125",
            "--max-events", str(args.max_events),
            "--tag-prefix", "n179s1",
            "--signal-label-value", str(args.signal_label_value),
        ])
        b1 = read_best_auc(s1_sum)
        print(f"[{level}] stage1 best: tag={b1['tag']} auc={b1['auc']} f1={b1['f1']}", flush=True)

        print(f"===== [{level}] stage2: K sweep at tau* =====", flush=True)
        s2_roc = ldir / "stage2_roc.csv"
        s2_sum = ldir / "stage2_summary.csv"
        s2_run = ldir / "stage2_runtime.csv"
        run([
            args.python,
            "scripts/noise_analyze/sweep_n179_pair.py",
            "--noisy", noisy,
            "--out-csv", str(s2_roc),
            "--summary-csv", str(s2_sum),
            "--runtime-csv", str(s2_run),
            "--dataset", "ED24",
            "--scene", "myPedestrain_06",
            "--level", level,
            "--width", "346",
            "--height", "260",
            "--tick-ns", "1000",
            "--s-list", b1["s"],
            "--tau-us-list", b1["tau_us"],
            "--k-sfrac-list", args.k_sfrac_stage2,
            "--k-mix-list", args.k_mix_stage2,
            "--max-events", str(args.max_events),
            "--tag-prefix", "n179s2",
            "--signal-label-value", str(args.signal_label_value),
        ])
        b2 = read_best_auc(s2_sum)
        print(f"[{level}] stage2 best: tag={b2['tag']} auc={b2['auc']} f1={b2['f1']}", flush=True)

        print(f"===== [{level}] stage3: final curve on tuned params =====", flush=True)
        s3_roc = ldir / "stage3_roc.csv"
        s3_sum = ldir / "stage3_summary.csv"
        s3_run = ldir / "stage3_runtime.csv"
        run([
            args.python,
            "scripts/noise_analyze/sweep_n179_pair.py",
            "--noisy", noisy,
            "--out-csv", str(s3_roc),
            "--summary-csv", str(s3_sum),
            "--runtime-csv", str(s3_run),
            "--dataset", "ED24",
            "--scene", "myPedestrain_06",
            "--level", level,
            "--width", "346",
            "--height", "260",
            "--tick-ns", "1000",
            "--s-list", b2["s"],
            "--tau-us-list", b2["tau_us"],
            "--k-sfrac-list", b2["k_sfrac"],
            "--k-mix-list", b2["k_mix"],
            "--max-events", str(args.max_events),
            "--tag-prefix", "n179s3",
            "--signal-label-value", str(args.signal_label_value),
        ])
        b3 = read_best_auc(s3_sum)
        final_rows.append({"level": level, **b3})
        print(f"[{level}] FINAL: tag={b3['tag']} auc={b3['auc']} f1={b3['f1']}", flush=True)

    out_final = out_root / "final_best_by_level.csv"
    with out_final.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["level", "s", "tau_us", "k_sfrac", "k_mix", "auc", "f1", "tag"])
        w.writeheader()
        w.writerows(final_rows)
    print(f"saved: {out_final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
