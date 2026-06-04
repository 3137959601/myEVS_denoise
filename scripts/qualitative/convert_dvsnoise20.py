from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from myevs.qualitative.aedat4_convert import convert_aedat4_to_npz
from myevs.qualitative.config import DEFAULT_CASES_CONFIG, ensure_default_configs, load_cases_config


def main() -> int:
    p = argparse.ArgumentParser(description="Convert all configured DVSNOISE20 AEDAT4 files to myEVS npz events.")
    p.add_argument("--cases", default=str(DEFAULT_CASES_CONFIG))
    p.add_argument("--case-id", action="append", default=None, help="Case id to convert; repeat for multiple cases.")
    p.add_argument("--max-events", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    ensure_default_configs(args.cases, overwrite=False)
    cfg = load_cases_config(args.cases)
    selected = set(args.case_id or [])
    converted = 0
    skipped = 0

    for case_id, case in cfg.get("cases", {}).items():
        if selected and case_id not in selected:
            continue
        raw = case.get("raw_aedat4")
        if not raw:
            continue
        out = Path(str(case.get("noisy", "")))
        if out.exists() and not bool(args.overwrite):
            print(f"skip existing: {case_id} -> {out}")
            skipped += 1
            continue
        try:
            dst = convert_aedat4_to_npz(in_path=raw, out_path=out, max_events=int(args.max_events))
        except RuntimeError as e:
            print(str(e), file=sys.stderr)
            print(f"failed case: {case_id}", file=sys.stderr)
            return 2
        print(f"converted {case_id}: {dst}")
        converted += 1

    print(f"done: converted={converted}, skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
