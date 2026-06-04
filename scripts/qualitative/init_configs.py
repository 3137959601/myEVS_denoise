from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from myevs.qualitative.config import DEFAULT_ALGORITHM_PARAMS, DEFAULT_CASES_CONFIG, ensure_default_configs


def main() -> int:
    p = argparse.ArgumentParser(description="Create qualitative figure default config files.")
    p.add_argument("--cases", default=str(DEFAULT_CASES_CONFIG))
    p.add_argument("--params", default=str(DEFAULT_ALGORITHM_PARAMS))
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    ensure_default_configs(args.cases, args.params, overwrite=bool(args.overwrite))
    print(f"cases:  {args.cases}")
    print(f"params: {args.params}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
