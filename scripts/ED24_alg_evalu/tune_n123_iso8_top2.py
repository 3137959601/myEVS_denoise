from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
import time
from pathlib import Path

ENV_NAMES = ("light", "mid", "heavy")


def parse_float_list(s: str) -> list[float]:
    out: list[float] = []
    for part in str(s).split(","):
        p = part.strip()
        if not p:
            continue
        out.append(float(p))
    if not out:
        raise ValueError("empty float list")
    return out


def fmt_f(v: float) -> str:
    return f"{float(v):.3f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")


def read_env_best(csv_path: Path) -> dict[str, float | str]:
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8", newline="")))
    if not rows:
        raise RuntimeError(f"empty csv: {csv_path}")

    by_tag: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        tag = str(r.get("tag", "")).strip()
        if not tag:
            continue
        by_tag.setdefault(tag, []).append(r)
    if not by_tag:
        raise RuntimeError(f"no tag rows in csv: {csv_path}")

    best_auc_tag = max(by_tag, key=lambda t: float(by_tag[t][0]["auc"]))
    best_auc = float(by_tag[best_auc_tag][0]["auc"])
    best_f1_row = max(rows, key=lambda r: float(r["f1"]))
    return {
        "best_auc_tag": best_auc_tag,
        "best_auc": best_auc,
        "best_f1_tag": str(best_f1_row["tag"]),
        "best_f1": float(best_f1_row["f1"]),
        "best_f1_thr": float(best_f1_row["value"]),
    }


def run_one(
    *,
    python_exe: str,
    project_root: Path,
    out_dir: Path,
    lam: float,
    s_list: str,
    tau_us_list: str,
    max_events: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["MYEVS_N123_LAMBDA"] = str(float(lam))

    cmd = [
        python_exe,
        "scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py",
        "--variant",
        "n123",
        "--esr-mode",
        "off",
        "--aocc-mode",
        "off",
        "--out-dir",
        str(out_dir),
        "--s-list",
        s_list,
        "--tau-us-list",
        tau_us_list,
        "--max-events",
        str(int(max_events)),
    ]

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(project_root), env=env, check=False)
    dt = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"sweep failed for lambda={lam}: return code {proc.returncode}")

    per_env: dict[str, dict[str, float | str]] = {}
    for env_name in ENV_NAMES:
        matches = glob.glob(str(out_dir / f"roc_ebf_n123_{env_name}_labelscore_*.csv"))
        if len(matches) != 1:
            raise RuntimeError(f"expected one ROC csv for {env_name}, got {len(matches)} under {out_dir}")
        per_env[env_name] = read_env_best(Path(matches[0]))

    mean_auc = sum(float(per_env[e]["best_auc"]) for e in ENV_NAMES) / float(len(ENV_NAMES))
    mean_f1 = sum(float(per_env[e]["best_f1"]) for e in ENV_NAMES) / float(len(ENV_NAMES))

    return {
        "lambda": float(lam),
        "mean_auc": float(mean_auc),
        "mean_f1": float(mean_f1),
        "elapsed_sec": float(dt),
        "per_env": per_env,
        "out_dir": str(out_dir),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Tune n123 iso8-top2 params via slim sweep.")
    ap.add_argument("--python-exe", default="")
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--base-out-dir", default="data/ED24/myPedestrain_06/EBF_Part2/_tune_n123_iso8_top2")

    ap.add_argument("--lambda-grid", default="0.5,1.0,1.5,2.0")
    ap.add_argument("--s-list", default="7,9")
    ap.add_argument("--tau-us-list", default="64000,128000")
    ap.add_argument("--max-events", type=int, default=200000)

    ap.add_argument("--full-validate", action="store_true")
    ap.add_argument("--full-s-list", default="3,5,7,9")
    ap.add_argument("--full-tau-us-list", default="8000,16000,32000,64000,128000,256000,512000,1024000")
    ap.add_argument("--full-max-events", type=int, default=0)

    args = ap.parse_args()

    python_exe = args.python_exe.strip() or os.sys.executable
    project_root = Path(args.project_root).resolve()
    base_out_dir = (project_root / args.base_out_dir).resolve()
    base_out_dir.mkdir(parents=True, exist_ok=True)

    lambda_grid = parse_float_list(args.lambda_grid)

    results: list[dict] = []
    for lam in lambda_grid:
        name = f"lam{fmt_f(float(lam))}"
        out_dir = base_out_dir / name
        print(f"[RUN] {name}")
        rec = run_one(
            python_exe=python_exe,
            project_root=project_root,
            out_dir=out_dir,
            lam=float(lam),
            s_list=str(args.s_list),
            tau_us_list=str(args.tau_us_list),
            max_events=int(args.max_events),
        )
        rec["name"] = name
        results.append(rec)
        print(
            f"[OK] {name} mean_auc={rec['mean_auc']:.6f} "
            f"mean_f1={rec['mean_f1']:.6f} elapsed={rec['elapsed_sec']:.1f}s"
        )

    results.sort(key=lambda r: (float(r["mean_f1"]), float(r["mean_auc"])), reverse=True)

    summary_csv = base_out_dir / "summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "rank",
                "name",
                "lambda",
                "mean_f1",
                "mean_auc",
                "light_f1",
                "mid_f1",
                "heavy_f1",
                "light_auc",
                "mid_auc",
                "heavy_auc",
                "out_dir",
            ]
        )
        for i, r in enumerate(results, start=1):
            w.writerow(
                [
                    i,
                    r["name"],
                    r["lambda"],
                    r["mean_f1"],
                    r["mean_auc"],
                    r["per_env"]["light"]["best_f1"],
                    r["per_env"]["mid"]["best_f1"],
                    r["per_env"]["heavy"]["best_f1"],
                    r["per_env"]["light"]["best_auc"],
                    r["per_env"]["mid"]["best_auc"],
                    r["per_env"]["heavy"]["best_auc"],
                    r["out_dir"],
                ]
            )

    summary_json = base_out_dir / "summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    best = results[0]
    print("\n=== TOP1 (prescreen) ===")
    print(json.dumps(best, ensure_ascii=False, indent=2))

    if args.full_validate:
        name = str(best["name"])
        out_dir = base_out_dir / f"full_{name}"
        print(f"\n[RUN] full validate: {name}")
        full = run_one(
            python_exe=python_exe,
            project_root=project_root,
            out_dir=out_dir,
            lam=float(best["lambda"]),
            s_list=str(args.full_s_list),
            tau_us_list=str(args.full_tau_us_list),
            max_events=int(args.full_max_events),
        )
        full["name"] = name
        full_json = base_out_dir / "full_validate_top1.json"
        with full_json.open("w", encoding="utf-8") as f:
            json.dump(full, f, ensure_ascii=False, indent=2)
        print("\n=== TOP1 (full validate) ===")
        print(json.dumps(full, ensure_ascii=False, indent=2))

    print(f"\nSaved summary: {summary_csv}")
    print(f"Saved detail : {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
