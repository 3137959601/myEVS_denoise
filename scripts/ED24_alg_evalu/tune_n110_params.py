from __future__ import annotations

import argparse
import csv
import glob
import itertools
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


ENV_NAMES = ("light", "mid", "heavy")


@dataclass
class EnvBest:
    best_auc_tag: str
    best_auc: float
    best_f1_tag: str
    best_f1: float
    best_f1_thr: float


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


def fmt_param(v: float) -> str:
    txt = f"{float(v):.3f}".rstrip("0").rstrip(".")
    return txt.replace("-", "m").replace(".", "p")


def read_env_best(csv_path: Path) -> EnvBest:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
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
    best_f1_tag = str(best_f1_row["tag"])
    best_f1 = float(best_f1_row["f1"])
    best_f1_thr = float(best_f1_row["value"])

    return EnvBest(
        best_auc_tag=best_auc_tag,
        best_auc=best_auc,
        best_f1_tag=best_f1_tag,
        best_f1=best_f1,
        best_f1_thr=best_f1_thr,
    )


def run_one(
    *,
    python_exe: str,
    project_root: Path,
    sweep_script: Path,
    out_dir: Path,
    alpha: float,
    beta: float,
    gamma: float,
    s_list: str,
    tau_us_list: str,
    max_events: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["MYEVS_N110_ALPHA"] = str(float(alpha))
    env["MYEVS_N110_BETA"] = str(float(beta))
    env["MYEVS_N110_GAMMA"] = str(float(gamma))

    cmd = [
        python_exe,
        str(sweep_script),
        "--variant",
        "n110",
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
        raise RuntimeError(f"sweep failed: return code {proc.returncode}")

    per_env: dict[str, EnvBest] = {}
    for env_name in ENV_NAMES:
        matches = glob.glob(str(out_dir / f"roc_ebf_n110_{env_name}_labelscore_*.csv"))
        if len(matches) != 1:
            raise RuntimeError(f"expected one ROC csv for {env_name}, got {len(matches)} under {out_dir}")
        per_env[env_name] = read_env_best(Path(matches[0]))

    mean_auc = sum(per_env[e].best_auc for e in ENV_NAMES) / float(len(ENV_NAMES))
    mean_f1 = sum(per_env[e].best_f1 for e in ENV_NAMES) / float(len(ENV_NAMES))

    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(gamma),
        "mean_auc": float(mean_auc),
        "mean_f1": float(mean_f1),
        "elapsed_sec": float(dt),
        "per_env": {
            e: {
                "best_auc_tag": per_env[e].best_auc_tag,
                "best_auc": per_env[e].best_auc,
                "best_f1_tag": per_env[e].best_f1_tag,
                "best_f1": per_env[e].best_f1,
                "best_f1_thr": per_env[e].best_f1_thr,
            }
            for e in ENV_NAMES
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Tune n110 alpha/beta/gamma by sweeping and aggregating ED24 metrics.")
    ap.add_argument("--python-exe", default="", help="Python executable path. Default: current interpreter")
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--sweep-script", default="scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py")
    ap.add_argument("--base-out-dir", default="data/ED24/myPedestrain_06/EBF_Part2/_tune_n110_retry")

    ap.add_argument("--alpha-grid", default="0.2,0.4,0.6,0.8")
    ap.add_argument("--beta-grid", default="0.4,0.8,1.2")
    ap.add_argument("--gamma-grid", default="0.2,0.5,0.8")

    ap.add_argument("--s-list", default="7,9")
    ap.add_argument("--tau-us-list", default="64000,128000")
    ap.add_argument("--max-events", type=int, default=200000)

    ap.add_argument("--full-validate", action="store_true", help="Run a final full-grid sweep for the top combo")
    ap.add_argument("--full-s-list", default="3,5,7,9")
    ap.add_argument("--full-tau-us-list", default="8000,16000,32000,64000,128000,256000,512000,1024000")
    ap.add_argument("--full-max-events", type=int, default=0)

    args = ap.parse_args()

    python_exe = args.python_exe.strip() or os.sys.executable
    project_root = Path(args.project_root).resolve()
    sweep_script = (project_root / args.sweep_script).resolve()
    base_out_dir = (project_root / args.base_out_dir).resolve()
    base_out_dir.mkdir(parents=True, exist_ok=True)

    alphas = parse_float_list(args.alpha_grid)
    betas = parse_float_list(args.beta_grid)
    gammas = parse_float_list(args.gamma_grid)

    results: list[dict] = []
    for alpha, beta, gamma in itertools.product(alphas, betas, gammas):
        combo_name = f"a{fmt_param(alpha)}_b{fmt_param(beta)}_g{fmt_param(gamma)}"
        out_dir = base_out_dir / combo_name
        print(f"[RUN] {combo_name}")
        rec = run_one(
            python_exe=python_exe,
            project_root=project_root,
            sweep_script=sweep_script,
            out_dir=out_dir,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            s_list=args.s_list,
            tau_us_list=args.tau_us_list,
            max_events=int(args.max_events),
        )
        rec["combo"] = combo_name
        rec["out_dir"] = str(out_dir)
        results.append(rec)
        print(
            f"[OK] {combo_name} mean_auc={rec['mean_auc']:.6f} mean_f1={rec['mean_f1']:.6f} elapsed={rec['elapsed_sec']:.1f}s"
        )

    results.sort(key=lambda r: (float(r["mean_f1"]), float(r["mean_auc"])), reverse=True)

    summary_json = base_out_dir / "summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    summary_csv = base_out_dir / "summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "rank",
            "combo",
            "alpha",
            "beta",
            "gamma",
            "mean_f1",
            "mean_auc",
            "light_f1",
            "mid_f1",
            "heavy_f1",
            "light_auc",
            "mid_auc",
            "heavy_auc",
            "out_dir",
        ])
        for i, r in enumerate(results, start=1):
            w.writerow([
                i,
                r["combo"],
                r["alpha"],
                r["beta"],
                r["gamma"],
                r["mean_f1"],
                r["mean_auc"],
                r["per_env"]["light"]["best_f1"],
                r["per_env"]["mid"]["best_f1"],
                r["per_env"]["heavy"]["best_f1"],
                r["per_env"]["light"]["best_auc"],
                r["per_env"]["mid"]["best_auc"],
                r["per_env"]["heavy"]["best_auc"],
                r["out_dir"],
            ])

    best = results[0]
    print("\n=== TOP1 (prescreen) ===")
    print(json.dumps(best, ensure_ascii=False, indent=2))

    if args.full_validate:
        top_combo_name = str(best["combo"])
        full_out_dir = base_out_dir / f"full_{top_combo_name}"
        print(f"\n[RUN] full validate: {top_combo_name}")
        full = run_one(
            python_exe=python_exe,
            project_root=project_root,
            sweep_script=sweep_script,
            out_dir=full_out_dir,
            alpha=float(best["alpha"]),
            beta=float(best["beta"]),
            gamma=float(best["gamma"]),
            s_list=str(args.full_s_list),
            tau_us_list=str(args.full_tau_us_list),
            max_events=int(args.full_max_events),
        )
        full["combo"] = top_combo_name
        full["out_dir"] = str(full_out_dir)
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
