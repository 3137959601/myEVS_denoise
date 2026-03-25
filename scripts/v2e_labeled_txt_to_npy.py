"""Convert v2e labeled events.txt to .npy/.npz for fast processing.

Input format (v2e events.txt):
- Comment lines start with '#'
- Data columns:
    time_s (float), x (int), y (int), polarity01 (0/1), signal_label (1=signal, 0=noise)

Output:
- .npy (default): a structured array with fields: t, x, y, p, label
  where:
    t is uint64 ticks, computed from time_s using --tick-ns
    p is int8 in {+1,-1}
    label is uint8 in {0,1}

- .npz: contains arrays with the same names.

This script is intentionally dependency-free (no pandas).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


DTYPE_STRUCT = np.dtype(
    [
        ("t", np.uint64),
        ("x", np.uint16),
        ("y", np.uint16),
        ("p", np.int8),
        ("label", np.uint8),
    ]
)


def _count_data_lines(txt_path: Path) -> int:
    n = 0
    with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            n += 1
    return n


def _parse_line(line: str) -> tuple[float, int, int, int, int] | None:
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    parts = s.split()
    if len(parts) < 5:
        return None
    try:
        t_s = float(parts[0])
        x = int(parts[1])
        y = int(parts[2])
        p01 = int(float(parts[3]))
        label = int(float(parts[4]))
    except ValueError:
        return None
    return t_s, x, y, p01, label


def convert(
    *,
    in_txt: Path,
    out_path: Path,
    tick_ns: float,
    batch_lines: int,
    overwrite: bool,
    save_npz: bool,
) -> None:
    if out_path.exists() and not overwrite:
        raise SystemExit(f"Output exists: {out_path} (use --overwrite)")

    n = _count_data_lines(in_txt)
    if n <= 0:
        raise SystemExit(f"No data lines found in {in_txt}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    scale = 1e9 / float(tick_ns)  # seconds -> ticks

    # Write structured .npy via memmap to avoid huge RAM usage.
    tmp_npy = out_path
    if tmp_npy.suffix.lower() != ".npy":
        tmp_npy = out_path.with_suffix(out_path.suffix + ".tmp.npy")

    arr = np.lib.format.open_memmap(
        str(tmp_npy), mode="w+", dtype=DTYPE_STRUCT, shape=(n,)
    )

    i = 0
    buf = []
    with in_txt.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            rec = _parse_line(line)
            if rec is None:
                continue
            buf.append(rec)
            if len(buf) >= batch_lines:
                i = _flush(buf, arr, i, scale)
                buf.clear()
        if buf:
            i = _flush(buf, arr, i, scale)
            buf.clear()

    if i != n:
        # In case some lines were unparsable; shrink by rewriting.
        arr.flush()
        # Reload and slice.
        arr2 = np.load(str(tmp_npy), mmap_mode="r")
        sliced = np.asarray(arr2[:i]).copy()
        np.save(str(tmp_npy), sliced)
        n = i

    if save_npz or out_path.suffix.lower() == ".npz":
        npz_path = out_path if out_path.suffix.lower() == ".npz" else out_path.with_suffix(".npz")
        mm = np.load(str(tmp_npy), mmap_mode="r")
        np.savez(
            str(npz_path),
            t=np.asarray(mm["t"], dtype=np.uint64),
            x=np.asarray(mm["x"], dtype=np.uint16),
            y=np.asarray(mm["y"], dtype=np.uint16),
            p=np.asarray(mm["p"], dtype=np.int8),
            label=np.asarray(mm["label"], dtype=np.uint8),
            tick_ns=np.array([float(tick_ns)], dtype=np.float64),
        )

        if out_path.suffix.lower() == ".npz":
            # If user explicitly requested .npz, remove tmp .npy
            try:
                tmp_npy.unlink(missing_ok=True)  # py3.8+; safe on 3.11
            except TypeError:
                # missing_ok not available on very old Pythons.
                if tmp_npy.exists():
                    tmp_npy.unlink()
            return

    # Finalize .npy output
    final_npy = out_path if out_path.suffix.lower() == ".npy" else out_path.with_suffix(".npy")
    if final_npy != tmp_npy:
        # Rename/move tmp to final
        if final_npy.exists() and overwrite:
            final_npy.unlink()
        tmp_npy.replace(final_npy)


def _flush(buf: list[tuple[float, int, int, int, int]], arr: np.ndarray, i0: int, scale: float) -> int:
    # Convert a small buffer to numpy and write.
    # NOTE: rounding is important to reduce floating error.
    t_s = np.fromiter((r[0] for r in buf), dtype=np.float64, count=len(buf))
    x = np.fromiter((r[1] for r in buf), dtype=np.int64, count=len(buf))
    y = np.fromiter((r[2] for r in buf), dtype=np.int64, count=len(buf))
    p01 = np.fromiter((r[3] for r in buf), dtype=np.int64, count=len(buf))
    lab = np.fromiter((r[4] for r in buf), dtype=np.int64, count=len(buf))

    t = np.asarray(np.rint(t_s * scale), dtype=np.uint64)
    x = np.asarray(x, dtype=np.uint16)
    y = np.asarray(y, dtype=np.uint16)
    p = np.where(p01.astype(np.int64) > 0, 1, -1).astype(np.int8)
    label = np.asarray(lab > 0, dtype=np.uint8)

    n = t.shape[0]
    sl = slice(i0, i0 + n)
    arr[sl]["t"] = t
    arr[sl]["x"] = x
    arr[sl]["y"] = y
    arr[sl]["p"] = p
    arr[sl]["label"] = label
    return i0 + n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_txt", required=True, help="input v2e labeled events.txt")
    ap.add_argument("--out", dest="out_path", required=True, help="output path (.npy or .npz)")
    ap.add_argument("--tick-ns", type=float, default=1000.0, help="target tick size in ns (default 1000 for 1us)")
    ap.add_argument("--batch-lines", type=int, default=200_000, help="lines per flush (speed/memory tradeoff)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--also-npz", action="store_true", help="also write .npz next to .npy")
    args = ap.parse_args()

    in_txt = Path(args.in_txt)
    out_path = Path(args.out_path)

    save_npz = args.also_npz or out_path.suffix.lower() == ".npz"

    convert(
        in_txt=in_txt,
        out_path=out_path,
        tick_ns=float(args.tick_ns),
        batch_lines=int(args.batch_lines),
        overwrite=bool(args.overwrite),
        save_npz=bool(save_npz),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
