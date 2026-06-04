from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from myevs.qualitative.layout import save_panel


def main() -> int:
    p = argparse.ArgumentParser(description="Merge individual images into one publication panel.")
    p.add_argument("--images", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--labels", nargs="*", default=None)
    p.add_argument("--cols", type=int, default=4)
    args = p.parse_args()
    labels = args.labels if args.labels else None
    if labels is not None and len(labels) != len(args.images):
        raise SystemExit("--labels must have the same count as --images")
    out = save_panel(args.images, args.out, labels=labels, cols=int(args.cols))
    print(f"panel: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
