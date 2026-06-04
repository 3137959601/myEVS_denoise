from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from myevs.qualitative.aedat4_convert import convert_aedat4_to_npz


def main() -> int:
    p = argparse.ArgumentParser(description="Convert DVSNOISE20 AEDAT4 files to myEVS npz events.")
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)
    p.add_argument("--max-events", type=int, default=0)
    args = p.parse_args()
    try:
        out = convert_aedat4_to_npz(in_path=args.in_path, out_path=args.out_path, max_events=int(args.max_events))
    except RuntimeError as e:
        raise SystemExit(str(e))
    print(f"converted: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
