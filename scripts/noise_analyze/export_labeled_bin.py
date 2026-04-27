from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description="Export labeled event npy to a compact binary file for C++ scorers.")
    ap.add_argument("--labeled-npy", required=True)
    ap.add_argument("--out-bin", required=True)
    ap.add_argument("--max-events", type=int, default=400000)
    ap.add_argument("--width", type=int, default=346)
    ap.add_argument("--height", type=int, default=260)
    args = ap.parse_args()

    arr = np.load(str(args.labeled_npy), mmap_mode="r")
    n = int(arr.shape[0])
    if int(args.max_events) > 0:
        n = min(n, int(args.max_events))
    out = Path(str(args.out_bin))
    out.parent.mkdir(parents=True, exist_ok=True)

    t = np.asarray(arr["t"][:n], dtype=np.uint64)
    x = np.asarray(arr["x"][:n], dtype=np.uint16)
    y = np.asarray(arr["y"][:n], dtype=np.uint16)
    p = np.asarray(arr["p"][:n], dtype=np.int8)
    label = np.asarray(arr["label"][:n], dtype=np.uint8)

    with out.open("wb") as f:
        f.write(struct.pack("<8sQII", b"MYEVSBIN", n, int(args.width), int(args.height)))
        t.tofile(f)
        x.tofile(f)
        y.tofile(f)
        p.tofile(f)
        label.tofile(f)

    print(f"wrote: {out} events={n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
