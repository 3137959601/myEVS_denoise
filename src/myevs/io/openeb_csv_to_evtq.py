from __future__ import annotations

import argparse
import csv
import os
import struct
from dataclasses import dataclass


_EVTQ_HDR = struct.Struct("<4sHHHHI")  # magic, version, headerBytes, width, height, reserved
_EVTQ_EVT = struct.Struct("<QHHBB")  # t, x, y, p(1=ON,0=OFF), reserved


@dataclass(frozen=True)
class ConvertStats:
    total_rows: int
    written_events: int
    dropped_events: int
    first_ts_in: int | None
    last_ts_in: int | None


def _is_int_like(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    if s[0] in "+-":
        s = s[1:]
    return s.isdigit()


def _row_to_ints(row: list[str]) -> list[int] | None:
    if len(row) < 4:
        return None
    vals = row[:4]
    if not all(_is_int_like(v) for v in vals):
        return None
    return [int(v) for v in vals]


def _parse_layout(layout: str) -> tuple[int, int, int, int]:
    key = layout.strip().lower().replace(" ", "")
    if key in ("xypt", "x,y,p,t"):
        return (0, 1, 2, 3)
    if key in ("txyp", "t,x,y,p"):
        return (1, 2, 3, 0)
    raise ValueError(f"unsupported --layout: {layout}")


def _to_tick(ts: int, *, ts_unit: str, tick_ns: float) -> int:
    if ts_unit == "tick":
        return int(ts)
    # us -> tick
    return int(round(ts * (1000.0 / float(tick_ns))))


def _normalize_polarity(p_raw: int) -> int:
    # Accept 0/1 or -1/+1
    return 1 if p_raw > 0 else 0


def convert_openeb_csv_to_evtq(
    in_csv: str,
    out_evtq: str,
    *,
    width: int,
    height: int,
    layout: str,
    ts_unit: str,
    tick_ns: float,
    zero_start: bool,
) -> ConvertStats:
    """Convert OpenEB CSV (x,y,p,t) to myEVS EVTQ.

    Kept as a reusable library function so it can be used from both CLI/scripts.
    """

    if width <= 0 or height <= 0:
        raise ValueError("width/height must be positive")
    if ts_unit not in ("us", "tick"):
        raise ValueError("--ts-unit must be one of: us, tick")

    os.makedirs(os.path.dirname(os.path.abspath(out_evtq)), exist_ok=True)

    total_rows = 0
    written = 0
    dropped = 0
    first_ts_in: int | None = None
    last_ts_in: int | None = None
    ts0: int | None = None

    x_i, y_i, p_i, t_i = _parse_layout(layout)

    with open(in_csv, "r", newline="", encoding="utf-8") as fi, open(out_evtq, "wb") as fo:
        fo.write(_EVTQ_HDR.pack(b"EVTQ", 1, _EVTQ_HDR.size, int(width), int(height), 0))

        reader = csv.reader(fi)

        # Auto skip header when first row is non-integer labels.
        first_row = next(reader, None)
        if first_row is not None:
            vals = _row_to_ints(first_row)
            if vals is not None:
                row = vals
                x = row[x_i]
                y = row[y_i]
                p = row[p_i]
                t = row[t_i]
                total_rows += 1

                if first_ts_in is None:
                    first_ts_in = t
                last_ts_in = t
                if zero_start and ts0 is None:
                    ts0 = t
                if zero_start and ts0 is not None:
                    t -= ts0

                if 0 <= x < width and 0 <= y < height:
                    t_tick = _to_tick(t, ts_unit=ts_unit, tick_ns=tick_ns)
                    fo.write(_EVTQ_EVT.pack(int(t_tick), int(x), int(y), _normalize_polarity(int(p)), 0))
                    written += 1
                else:
                    dropped += 1

        for raw_row in reader:
            vals = _row_to_ints(raw_row)
            if vals is None:
                continue

            x = vals[x_i]
            y = vals[y_i]
            p = vals[p_i]
            t = vals[t_i]
            total_rows += 1

            if first_ts_in is None:
                first_ts_in = t
            last_ts_in = t
            if zero_start and ts0 is None:
                ts0 = t
            if zero_start and ts0 is not None:
                t -= ts0

            if 0 <= x < width and 0 <= y < height:
                t_tick = _to_tick(t, ts_unit=ts_unit, tick_ns=tick_ns)
                fo.write(_EVTQ_EVT.pack(int(t_tick), int(x), int(y), _normalize_polarity(int(p)), 0))
                written += 1
            else:
                dropped += 1

    return ConvertStats(
        total_rows=total_rows,
        written_events=written,
        dropped_events=dropped,
        first_ts_in=first_ts_in,
        last_ts_in=last_ts_in,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Convert OpenEB CSV into myEVS EVTQ.\n"
            "OpenEB default CSV layout is x,y,p,t with timestamp in microseconds."
        )
    )
    p.add_argument("--in", dest="in_csv", required=True, help="input csv path")
    p.add_argument("--out", dest="out_evtq", required=True, help="output evtq path")
    p.add_argument("--width", type=int, required=True, help="sensor width")
    p.add_argument("--height", type=int, required=True, help="sensor height")
    p.add_argument(
        "--layout",
        default="xypt",
        choices=["xypt", "txyp", "x,y,p,t", "t,x,y,p"],
        help="column order in input csv",
    )
    p.add_argument(
        "--ts-unit",
        default="us",
        choices=["us", "tick"],
        help="timestamp unit in input csv",
    )
    p.add_argument(
        "--tick-ns",
        type=float,
        default=12.5,
        help="target tick size in ns for EVTQ timestamps",
    )
    p.add_argument(
        "--no-zero-start",
        action="store_true",
        help="do not shift the first timestamp to 0 before conversion",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    st = convert_openeb_csv_to_evtq(
        args.in_csv,
        args.out_evtq,
        width=int(args.width),
        height=int(args.height),
        layout=str(args.layout),
        ts_unit=str(args.ts_unit),
        tick_ns=float(args.tick_ns),
        zero_start=(not bool(args.no_zero_start)),
    )

    print(f"in:  {args.in_csv}")
    print(f"out: {args.out_evtq}")
    print(f"rows: {st.total_rows}")
    print(f"written: {st.written_events}")
    print(f"dropped(out-of-range): {st.dropped_events}")
    if st.first_ts_in is not None:
        print(f"timestamp_in: first={st.first_ts_in}, last={st.last_ts_in}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
