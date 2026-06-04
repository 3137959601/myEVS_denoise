from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from myevs.qualitative.config import DEFAULT_ALGORITHM_PARAMS, DEFAULT_CASES_CONFIG, DEFAULT_QUALITATIVE_DIR, ensure_default_configs
from myevs.qualitative.run_methods import render_case


def main() -> int:
    p = argparse.ArgumentParser(description="Render all qualitative methods for one selected case.")
    p.add_argument("--case-id", required=True)
    p.add_argument("--cases", default=str(DEFAULT_CASES_CONFIG))
    p.add_argument("--params", default=str(DEFAULT_ALGORITHM_PARAMS))
    p.add_argument("--out-root", default=str(DEFAULT_QUALITATIVE_DIR))
    p.add_argument("--start-us", type=int, default=None)
    p.add_argument("--window-ms", type=int, default=None)
    p.add_argument("--no-panel", action="store_true")
    args = p.parse_args()
    ensure_default_configs(args.cases, args.params)
    out = render_case(
        cases_path=args.cases,
        params_path=args.params,
        case_id=args.case_id,
        output_root=args.out_root,
        start_us=args.start_us,
        window_ms=args.window_ms,
        make_panel_output=not bool(args.no_panel),
    )
    print(f"render manifest: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
