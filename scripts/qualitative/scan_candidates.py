from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from myevs.qualitative.config import DEFAULT_CASES_CONFIG, DEFAULT_QUALITATIVE_DIR, ensure_default_configs
from myevs.qualitative.select_windows import scan_candidates


def main() -> int:
    p = argparse.ArgumentParser(description="Scan qualitative figure candidate windows.")
    p.add_argument("--cases", default=str(DEFAULT_CASES_CONFIG))
    p.add_argument("--out-root", default=str(DEFAULT_QUALITATIVE_DIR))
    p.add_argument("--case-id", action="append", default=None, help="Case id to scan; repeat to scan multiple cases.")
    p.add_argument("--max-candidates", type=int, default=20)
    p.add_argument("--stride-ms", type=int, default=None)
    args = p.parse_args()
    ensure_default_configs(args.cases, DEFAULT_QUALITATIVE_DIR / "algorithm_params.yaml")
    out = scan_candidates(
        cases_path=args.cases,
        output_root=args.out_root,
        case_ids=args.case_id,
        max_candidates_per_window=int(args.max_candidates),
        stride_ms=args.stride_ms,
    )
    print(f"candidate manifest: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
