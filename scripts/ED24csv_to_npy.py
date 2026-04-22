r"""Convert whitespace/comma separated event CSV to labeled .npy compatible with myEVS.

Expected columns (header must contain these names, case-insensitive):
- timestamp, x, y, polarity, label

Input conventions (configurable by args):
- timestamp unit: microseconds by default (set --timestamp-unit)
- polarity: 0/1 by default (0->-1, 1->+1)
- label: dataset label value that indicates SIGNAL is configurable by --signal-label-value
  Output will always follow myEVS convention: label=1 for signal, 0 for noise.

Output format (compatible with scripts/sweep_ebf_labelscore_grid.py, etc.):
- structured .npy array with dtype fields:
    t(uint64), x(uint16), y(uint16), p(int8), label(uint8)

Notes:
- timestamps are rebased to start from 0 (t = (timestamp - first_timestamp) converted to ticks)
- coordinates can be optionally filtered to camera bounds (default: drop OOB)

No pandas dependency.
"""

from __future__ import annotations

import argparse
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


def _split_parts(line: str) -> list[str]:
    return line.strip().replace(",", " ").split()


def _parse_header(header_line: str) -> dict[str, int]:
    cols = _split_parts(header_line)
    if not cols:
        raise SystemExit("CSV header is empty")

    wanted = {"timestamp", "x", "y", "polarity", "label"}
    idx: dict[str, int] = {}
    for i, c in enumerate(cols):
        c2 = c.strip().lower()
        if c2 in wanted and c2 not in idx:
            idx[c2] = i

    missing = sorted(wanted - set(idx.keys()))
    if missing:
        raise SystemExit(f"CSV header missing columns: {missing}; got={cols}")
    return idx


def _count_data_lines(csv_path: Path) -> int:
    n = 0
    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        _ = f.readline()  # header
        for line in f:
            if line.strip():
                n += 1
    return n


def _ts_to_ticks(delta_ts: np.ndarray, *, timestamp_unit: str, tick_ns: float) -> np.ndarray:
    # Convert delta timestamp to output ticks.
    # tick_ns: ns per tick
    unit = str(timestamp_unit).lower()
    if unit == "us":
        # delta_us * 1000ns/us / tick_ns
        return np.asarray(np.rint(delta_ts.astype(np.float64) * (1000.0 / float(tick_ns))), dtype=np.uint64)
    if unit == "ns":
        # delta_ns / tick_ns
        return np.asarray(np.rint(delta_ts.astype(np.float64) / float(tick_ns)), dtype=np.uint64)
    if unit == "ticks":
        return np.asarray(delta_ts, dtype=np.uint64)
    raise SystemExit(f"Unsupported --timestamp-unit: {timestamp_unit!r} (use us/ns/ticks)")


def _flush(
    buf: list[tuple[int, int, int, int, int]],
    out: np.ndarray,
    i0: int,
    *,
    tick_ns: float,
    timestamp_unit: str,
    t0: int,
    width: int,
    height: int,
    signal_label_value: int,
    drop_oob: bool,
) -> int:
    ts = np.fromiter((r[0] for r in buf), dtype=np.int64, count=len(buf))
    x = np.fromiter((r[1] for r in buf), dtype=np.int64, count=len(buf))
    y = np.fromiter((r[2] for r in buf), dtype=np.int64, count=len(buf))
    pol01 = np.fromiter((r[3] for r in buf), dtype=np.int64, count=len(buf))
    lab = np.fromiter((r[4] for r in buf), dtype=np.int64, count=len(buf))

    if drop_oob:
        m = (x >= 0) & (x < int(width)) & (y >= 0) & (y < int(height))
    else:
        m = np.ones((x.shape[0],), dtype=bool)

    if not np.any(m):
        return i0

    ts = ts[m]
    x = x[m]
    y = y[m]
    pol01 = pol01[m]
    lab = lab[m]

    delta = (ts - int(t0)).astype(np.int64)
    t = _ts_to_ticks(delta, timestamp_unit=str(timestamp_unit), tick_ns=float(tick_ns))

    p = np.where(pol01.astype(np.int64) > 0, 1, -1).astype(np.int8)

    # Map to myEVS convention: label=1 for signal, 0 for noise
    label = (lab.astype(np.int64) == int(signal_label_value)).astype(np.uint8)

    n = int(t.shape[0])
    sl = slice(i0, i0 + n)
    out[sl]["t"] = t
    out[sl]["x"] = np.asarray(x, dtype=np.uint16)
    out[sl]["y"] = np.asarray(y, dtype=np.uint16)
    out[sl]["p"] = p
    out[sl]["label"] = label
    return i0 + n


def convert_one(
    *,
    in_csv: Path,
    out_npy: Path,
    tick_ns: float,
    timestamp_unit: str,
    width: int,
    height: int,
    signal_label_value: int,
    batch_lines: int,
    overwrite: bool,
    drop_oob: bool,
) -> Path:
    if out_npy.exists() and not overwrite:
        raise SystemExit(f"Output exists: {out_npy} (use --overwrite)")

    n_lines = _count_data_lines(in_csv)
    if n_lines <= 0:
        raise SystemExit(f"No data rows found in {in_csv}")

    out_npy.parent.mkdir(parents=True, exist_ok=True)

    tmp_npy = out_npy
    if tmp_npy.suffix.lower() != ".npy":
        tmp_npy = out_npy.with_suffix(out_npy.suffix + ".tmp.npy")

    out = np.lib.format.open_memmap(str(tmp_npy), mode="w+", dtype=DTYPE_STRUCT, shape=(n_lines,))

    with in_csv.open("r", encoding="utf-8", errors="ignore") as f:
        idx = _parse_header(f.readline())

        buf: list[tuple[int, int, int, int, int]] = []
        i = 0
        t0: int | None = None

        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = _split_parts(s)
            if len(parts) < 5:
                continue

            try:
                ts = int(float(parts[idx["timestamp"]]))
                xx = int(float(parts[idx["x"]]))
                yy = int(float(parts[idx["y"]]))
                pol = int(float(parts[idx["polarity"]]))
                lab = int(float(parts[idx["label"]]))
            except Exception:
                continue

            if t0 is None:
                t0 = ts

            buf.append((ts, xx, yy, pol, lab))
            if len(buf) >= int(batch_lines):
                i = _flush(
                    buf,
                    out,
                    i,
                    tick_ns=float(tick_ns),
                    timestamp_unit=str(timestamp_unit),
                    t0=int(t0),
                    width=int(width),
                    height=int(height),
                    signal_label_value=int(signal_label_value),
                    drop_oob=bool(drop_oob),
                )
                buf.clear()

        if buf:
            if t0 is None:
                raise SystemExit(f"No parsable rows found in {in_csv}")
            i = _flush(
                buf,
                out,
                i,
                tick_ns=float(tick_ns),
                timestamp_unit=str(timestamp_unit),
                t0=int(t0),
                width=int(width),
                height=int(height),
                signal_label_value=int(signal_label_value),
                drop_oob=bool(drop_oob),
            )
            buf.clear()

    # Shrink if we dropped OOB or skipped unparsable lines
    if i != n_lines:
        out.flush()
        mm = np.load(str(tmp_npy), mmap_mode="r")
        np.save(str(tmp_npy), np.asarray(mm[:i]).copy())

    final_npy = out_npy if out_npy.suffix.lower() == ".npy" else out_npy.with_suffix(".npy")
    if final_npy != tmp_npy:
        if final_npy.exists() and overwrite:
            final_npy.unlink()
        tmp_npy.replace(final_npy)

    return final_npy


def main() -> int:
    ap = argparse.ArgumentParser(description="Whitespace/CSV events -> labeled npy (t/x/y/p/label).")
    ap.add_argument("--in", dest="in_csv", required=True, help="input csv path")
    ap.add_argument(
        "--out",
        required=True,
        help="output .npy path",
    )

    ap.add_argument("--tick-ns", type=float, default=1000.0, help="target tick size (ns), default 1000=1us")
    ap.add_argument("--timestamp-unit", choices=["us", "ns", "ticks"], default="us")
    ap.add_argument("--width", type=int, default=346)
    ap.add_argument("--height", type=int, default=260)
    ap.add_argument(
        "--signal-label-value",
        type=int,
        default=0,
        help="value in input CSV 'label' column that indicates SIGNAL (will be mapped to output label=1)",
    )
    ap.add_argument("--batch-lines", type=int, default=200_000)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--keep-oob", action="store_true", help="keep out-of-bounds events instead of dropping")

    args = ap.parse_args()

    in_p = Path(args.in_csv)
    if not in_p.exists():
        raise SystemExit(f"Missing input: {in_p}")

    out_p = Path(args.out)
    out_npy = convert_one(
        in_csv=in_p,
        out_npy=out_p,
        tick_ns=float(args.tick_ns),
        timestamp_unit=str(args.timestamp_unit),
        width=int(args.width),
        height=int(args.height),
        signal_label_value=int(args.signal_label_value),
        batch_lines=int(args.batch_lines),
        overwrite=bool(args.overwrite),
        drop_oob=not bool(args.keep_oob),
    )

    arr = np.load(str(out_npy), mmap_mode="r")
    pos = int(np.sum(arr["label"]))
    neg = int(arr.shape[0] - pos)
    print(f"converted: {in_p} -> {out_npy} n={arr.shape[0]} signal(label=1)={pos} noise(label=0)={neg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
