from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from myevs.qualitative.config import DEFAULT_QUALITATIVE_DIR
from myevs.qualitative.external import expected_external_manifest, import_external_result


def main() -> int:
    p = argparse.ArgumentParser(description="Import or prepare external EDnCNN/EDFormer qualitative outputs.")
    sub = p.add_subparsers(dest="cmd", required=True)
    p_imp = sub.add_parser("copy")
    p_imp.add_argument("--source", required=True)
    p_imp.add_argument("--case-id", required=True)
    p_imp.add_argument("--method", required=True, choices=["EDnCNN", "EDFormer"])
    p_imp.add_argument("--out-root", default=str(DEFAULT_QUALITATIVE_DIR))
    p_exp = sub.add_parser("expected")
    p_exp.add_argument("--case-id", action="append", required=True)
    p_exp.add_argument("--method", action="append", default=["EDnCNN", "EDFormer"])
    p_exp.add_argument("--out-root", default=str(DEFAULT_QUALITATIVE_DIR))
    args = p.parse_args()
    if args.cmd == "copy":
        out = import_external_result(source=args.source, case_id=args.case_id, method=args.method, output_root=args.out_root)
    else:
        out = expected_external_manifest(output_root=args.out_root, case_ids=args.case_id, methods=args.method)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
