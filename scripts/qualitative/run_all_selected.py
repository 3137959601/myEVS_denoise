from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from myevs.qualitative.config import DEFAULT_ALGORITHM_PARAMS, DEFAULT_CASES_CONFIG, DEFAULT_QUALITATIVE_DIR, ensure_default_configs, load_cases_config
from myevs.qualitative.run_methods import render_case


def main() -> int:
    p = argparse.ArgumentParser(description="Render all enabled qualitative cases with selected/default windows.")
    p.add_argument("--cases", default=str(DEFAULT_CASES_CONFIG))
    p.add_argument("--params", default=str(DEFAULT_ALGORITHM_PARAMS))
    p.add_argument("--out-root", default=str(DEFAULT_QUALITATIVE_DIR))
    args = p.parse_args()
    ensure_default_configs(args.cases, args.params)
    cfg = load_cases_config(args.cases)
    for case_id, case in cfg.get("cases", {}).items():
        if not bool(case.get("enabled", True)):
            continue
        if case.get("raw_aedat4") and not Path(str(case.get("noisy", ""))).exists():
            print(f"skip {case_id}: convert AEDAT4 first -> {case.get('noisy')}")
            continue
        out = render_case(
            cases_path=args.cases,
            params_path=args.params,
            case_id=case_id,
            output_root=args.out_root,
            make_panel_output=True,
        )
        print(f"{case_id}: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
