from __future__ import annotations

import argparse
import csv
import importlib.util
import time
from pathlib import Path

import numpy as np


def _load_sweep_module():
    here = Path(__file__).resolve()
    sweep_path = here.parents[1] / "ED24_alg_evalu" / "sweep_ebf_slim_labelscore_grid.py"
    spec = importlib.util.spec_from_file_location("_sweep_ebf_slim_labelscore_grid", sweep_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"failed to load sweep module spec: {sweep_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _parse_csv_list(s: str) -> list[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark score_stream variant throughput on one labeled npy.")
    ap.add_argument("--labeled-npy", required=True)
    ap.add_argument("--variant-list", default="n175,n176")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--width", type=int, default=346)
    ap.add_argument("--height", type=int, default=260)
    ap.add_argument("--tick-ns", type=float, default=1000.0)
    ap.add_argument("--s", type=int, default=9)
    ap.add_argument("--tau-us", type=int, default=128000)
    ap.add_argument("--max-events", type=int, default=400000)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    sweep = _load_sweep_module()
    ev = sweep.load_labeled_npy(str(args.labeled_npy), max_events=int(args.max_events))
    tb = sweep.TimeBase(tick_ns=float(args.tick_ns))
    r = int((int(args.s) - 1) // 2)

    out_path = Path(str(args.out_csv))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    variants = _parse_csv_list(str(args.variant_list))

    rows: list[list[object]] = []
    for variant in variants:
        kernel_cache: dict[str, object] = {}
        scores = sweep.score_stream_ebf(
            ev,
            width=int(args.width),
            height=int(args.height),
            radius_px=int(r),
            tau_us=int(args.tau_us),
            tb=tb,
            _kernel_cache=kernel_cache,
            variant=str(variant),
        )

        times: list[float] = []
        for _ in range(max(1, int(args.repeats))):
            t0 = time.perf_counter()
            scores = sweep.score_stream_ebf(
                ev,
                width=int(args.width),
                height=int(args.height),
                radius_px=int(r),
                tau_us=int(args.tau_us),
                tb=tb,
                _kernel_cache=kernel_cache,
                variant=str(variant),
            )
            times.append(float(time.perf_counter() - t0))

        best_s = min(times)
        mean_s = sum(times) / float(len(times))
        n = int(ev.t.shape[0])
        rows.append(
            [
                str(variant),
                int(n),
                int(args.s),
                int(args.tau_us),
                int(args.repeats),
                f"{best_s:.9f}",
                f"{mean_s:.9f}",
                f"{float(n) / best_s:.3f}",
                f"{float(n) / mean_s:.3f}",
                f"{float(np.mean(scores)):.9f}",
                f"{float(np.max(scores)):.9f}",
            ]
        )

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "variant",
                "events",
                "s",
                "tau_us",
                "repeats",
                "best_seconds",
                "mean_seconds",
                "best_events_per_s",
                "mean_events_per_s",
                "score_mean",
                "score_max",
            ]
        )
        w.writerows(rows)

    print(f"wrote: {out_path}")
    for row in rows:
        print(
            f"{row[0]}: best={row[5]}s mean={row[6]}s "
            f"best_events_per_s={row[7]} mean_events_per_s={row[8]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
