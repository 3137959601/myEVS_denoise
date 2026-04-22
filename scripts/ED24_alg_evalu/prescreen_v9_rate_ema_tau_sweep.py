from __future__ import annotations

import argparse
import csv
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


_ENV_KEYS = ["light", "mid", "heavy"]


@dataclass(frozen=True)
class SummaryRow:
    rate_ema_tau_us: int
    auc_mean: float
    thr_std: float
    thr_range: float
    thr_fixed: float
    f1_fixed_mean: float


def _run(cmd: list[str], *, env: dict[str, str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return [dict(row) for row in r]


def _to_float(x: Any, *, default: float = float("nan")) -> float:
    try:
        s = str(x).strip()
        if not s:
            return default
        return float(s)
    except Exception:
        return default


def _mean(vals: list[float]) -> float:
    v = [x for x in vals if x == x]
    return sum(v) / float(len(v)) if v else float("nan")


def _write_csv(path: Path, rows: list[SummaryRow]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "rate_ema_tau_us",
        "auc_mean",
        "thr_std",
        "thr_range",
        "thr_fixed",
        "f1_fixed_mean",
    ]

    try:
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            for r in rows:
                w.writerow(
                    {
                        "rate_ema_tau_us": int(r.rate_ema_tau_us),
                        "auc_mean": r.auc_mean,
                        "thr_std": r.thr_std,
                        "thr_range": r.thr_range,
                        "thr_fixed": r.thr_fixed,
                        "f1_fixed_mean": r.f1_fixed_mean,
                    }
                )
        return path
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        alt = path.with_name(path.stem + f"_{ts}" + path.suffix)
        with alt.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            for r in rows:
                w.writerow(
                    {
                        "rate_ema_tau_us": int(r.rate_ema_tau_us),
                        "auc_mean": r.auc_mean,
                        "thr_std": r.thr_std,
                        "thr_range": r.thr_range,
                        "thr_fixed": r.thr_fixed,
                        "f1_fixed_mean": r.f1_fixed_mean,
                    }
                )
        return alt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Prescreen sweep for V9 time-constant global rate EMA (MYEVS_EBFOPT_RATE_EMA_TAU_US). "
            "Fixed: variant=EBFV9, s=9, tau=128ms, max-events=200k. "
            "Runs sweep -> best_params/thr_stability -> fixed-threshold eval -> summary CSV."
        )
    )
    ap.add_argument(
        "--out-base",
        default="data/ED24/myPedestrain_06/_prescreen_v9_rateema_tau_s9_tau128ms",
        help="Base output directory (each tau writes a subdir)",
    )
    ap.add_argument(
        "--rate-ema-tau-us-list",
        default="20000,50000,100000,200000,500000",
        help="Comma-separated list of MYEVS_EBFOPT_RATE_EMA_TAU_US values (microseconds)",
    )
    ap.add_argument("--max-events", type=int, default=200_000)
    ap.add_argument("--s", type=int, default=9)
    ap.add_argument("--tau-us", type=int, default=128_000)
    ap.add_argument("--roc-max-points", type=int, default=5000)
    ap.add_argument(
        "--python",
        default="",
        help="Optional python executable override. Default: current interpreter.",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="If set, skip recomputing when expected outputs already exist.",
    )

    args = ap.parse_args(argv)

    out_base = Path(str(args.out_base))
    out_base.mkdir(parents=True, exist_ok=True)

    tau_list: list[int] = []
    for part in str(args.rate_ema_tau_us_list).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            tau_list.append(int(float(part)))
        except Exception:
            raise SystemExit(f"invalid --rate-ema-tau-us-list item: {part!r}")

    tau_list = [int(x) for x in tau_list if int(x) > 0]
    if not tau_list:
        raise SystemExit("empty --rate-ema-tau-us-list")

    python = str(args.python).strip() or os.fspath(Path(os.sys.executable))

    # IMPORTANT: tag must match sweep_ebf_optimized_labelscore_grid.py's tag format.
    tag = f"ebfopt_softw_linear_timeconst_rateema_labelscore_s{int(args.s)}_tau{int(args.tau_us)}"

    summary_rows: list[SummaryRow] = []

    for tau_rate_us in tau_list:
        tau_rate_us = int(tau_rate_us)
        tau_dir = out_base / f"rate_tau_{tau_rate_us}us"
        tau_dir.mkdir(parents=True, exist_ok=True)

        env = dict(os.environ)
        env.pop("MYEVS_EBFOPT_SCALE_ALPHA", None)  # keep scale exponent at default=1.0
        env.pop("MYEVS_EBFOPT_RATE_EMA_ALPHA", None)  # avoid confusion: V9 uses TAU_US
        env["MYEVS_EBFOPT_RATE_EMA_TAU_US"] = str(int(tau_rate_us))

        roc_csv = {
            k: tau_dir / f"roc_ebf_optimized_softw_linear_timeconst_rateema_{k}_labelscore_s9_tau128ms.csv" for k in _ENV_KEYS
        }
        best_csv = tau_dir / "best_params_v9_rateema_tau_forced_s9_tau128ms.csv"
        stable_csv = tau_dir / "thr_stability_v9_rateema_tau_forced_s9_tau128ms.csv"
        fixed_csv = tau_dir / "fixed_thr_eval_v9_rateema_tau_s9_tau128ms_bestglobalf1.csv"

        all_exist = (
            all(p.exists() for p in roc_csv.values())
            and best_csv.exists()
            and stable_csv.exists()
            and fixed_csv.exists()
        )
        if bool(args.skip_existing) and all_exist:
            print(f"skip existing: rate_ema_tau_us={tau_rate_us} out={tau_dir}")
        else:
            _run(
                [
                    python,
                    "scripts/ED24_alg_evalu/sweep_ebf_optimized_labelscore_grid.py",
                    "--variant",
                    "EBFV9",
                    "--max-events",
                    str(int(args.max_events)),
                    "--s-list",
                    str(int(args.s)),
                    "--tau-us-list",
                    str(int(args.tau_us)),
                    "--roc-max-points",
                    str(int(args.roc_max_points)),
                    "--out-dir",
                    os.fspath(tau_dir),
                ],
                env=env,
            )

            _run(
                [
                    python,
                    "scripts/ED24_alg_evalu/summarize_best_params_ebf_optimized_ed24.py",
                    "--in-dir",
                    os.fspath(tau_dir),
                    "--dataset",
                    "myPedestrain_06",
                    "--force-tag",
                    tag,
                    "--out-csv",
                    os.fspath(best_csv),
                    "--out-stability-csv",
                    os.fspath(stable_csv),
                ],
                env=env,
            )

            _run(
                [
                    python,
                    "scripts/ED24_alg_evalu/summarize_fixed_threshold_ebf_optimized_ed24.py",
                    "--in-dir",
                    os.fspath(tau_dir),
                    "--dataset",
                    "myPedestrain_06",
                    "--tag",
                    tag,
                    "--thr-mode",
                    "best-global-f1",
                    "--out-csv",
                    os.fspath(fixed_csv),
                ],
                env=env,
            )

        best_rows = _read_csv(best_csv)
        aucs = [_to_float(r.get("auc")) for r in best_rows if (r.get("auc") or "").strip()]
        auc_mean = float(_mean(aucs))

        stable_rows = _read_csv(stable_csv)
        thr_std = _to_float(stable_rows[0].get("Thr_std")) if stable_rows else float("nan")
        thr_range = _to_float(stable_rows[0].get("Thr_range")) if stable_rows else float("nan")

        fixed_rows = _read_csv(fixed_csv)
        mean_row = next((r for r in fixed_rows if (r.get("env") or "").strip().upper() == "MEAN"), None)
        if mean_row is not None:
            thr_fixed = _to_float(mean_row.get("thr_fixed"))
            f1_fixed_mean = _to_float(mean_row.get("f1"))
        else:
            thr_fixed = _to_float(fixed_rows[0].get("thr_fixed")) if fixed_rows else float("nan")
            f1_fixed_mean = float(
                _mean(
                    [
                        _to_float(r.get("f1"))
                        for r in fixed_rows
                        if (r.get("env") or "").strip() in _ENV_KEYS
                    ]
                )
            )

        summary_rows.append(
            SummaryRow(
                rate_ema_tau_us=int(tau_rate_us),
                auc_mean=float(auc_mean),
                thr_std=float(thr_std),
                thr_range=float(thr_range),
                thr_fixed=float(thr_fixed),
                f1_fixed_mean=float(f1_fixed_mean),
            )
        )

        print(
            f"done: rate_ema_tau_us={tau_rate_us} auc_mean={auc_mean:.6f} thr_std={thr_std:.6f} f1_fixed_mean={f1_fixed_mean:.6f} out={tau_dir}"
        )

    summary_rows_sorted = sorted(summary_rows, key=lambda r: r.rate_ema_tau_us)
    out_csv = out_base / "rateema_tau_sweep_summary_v9_s9_tau128ms.csv"
    out_path = _write_csv(out_csv, summary_rows_sorted)
    print(f"\nsaved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
