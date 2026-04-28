from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description="Truncate event npy to first N rows.")
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--max-events", type=int, required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--print-path-only", action="store_true")
    args = ap.parse_args()

    nmax = int(args.max_events)
    if nmax <= 0:
        raise SystemExit("--max-events must be > 0")

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {out_path} (use --overwrite)")

    arr = np.load(str(in_path), mmap_mode="r")
    n = int(arr.shape[0])
    k = min(n, nmax)

    # preserve original dtype/structure
    out = np.asarray(arr[:k]).copy()
    np.save(str(out_path), out)
    if args.print_path_only:
        print(str(out_path))
    else:
        print(f"saved: {out_path} rows={k}/{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
