from __future__ import annotations

import argparse

from evuav_common import RESULT_ROOT, ensure_dirs, iter_npz_files, load_evuav_npz, safe_div, write_csv


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect EV-UAV split statistics for sequence selection.")
    ap.add_argument("--out-csv", default=str(RESULT_ROOT / "metrics" / "evuav_sequence_stats.csv"))
    args = ap.parse_args()
    ensure_dirs()

    rows: list[dict] = []
    for split, path in iter_npz_files():
        ev = load_evuav_npz(path)
        n = int(ev.shape[0])
        t = ev["t"].astype(float)
        duration_ms = float(t.max() - t.min()) if n else 0.0
        target = int((ev["label"] > 0).sum())
        bg = int(n - target)
        rows.append(
            {
                "split": split,
                "sequence": path.stem,
                "events": n,
                "duration_ms": f"{duration_ms:.3f}",
                "target_events": target,
                "background_events": bg,
                "target_fraction": f"{safe_div(target, n):.8f}",
                "event_rate_per_ms": f"{safe_div(n, duration_ms):.8f}",
                "target_rate_per_ms": f"{safe_div(target, duration_ms):.8f}",
            }
        )

    write_csv(
        RESULT_ROOT / "metrics" / "evuav_sequence_stats.csv",
        rows,
        [
            "split",
            "sequence",
            "events",
            "duration_ms",
            "target_events",
            "background_events",
            "target_fraction",
            "event_rate_per_ms",
            "target_rate_per_ms",
        ],
    )
    print(f"wrote {len(rows)} rows -> {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

