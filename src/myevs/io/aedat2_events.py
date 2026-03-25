from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable, Iterator, Tuple

import numpy as np

from ..events import EventBatch, EventStreamMeta


_AEDAT2_MAGIC = b"#!AER-DAT2.0"


@dataclass(frozen=True)
class Aedat2Info:
    meta: EventStreamMeta
    header_lines: tuple[str, ...]
    source_time_unit: str  # usually "us" for v2e


@dataclass(frozen=True)
class _V2eCodec:
    width: int
    height: int
    x_shift: int
    y_shift: int
    p_shift: int
    x_mask: int
    y_mask: int
    aps_mask: int
    # For DAVIS346/240 written by v2e, bit10 is used to mark "special" (noise) events.
    # For DVS640, that bit overlaps the X field in common encodings, so we disable it.
    special_mask: int | None
    flip_x: bool
    flip_y: bool


_V2E_CODECS: dict[tuple[int, int], _V2eCodec] = {
    # v2e AEDat2Output: DAVIS346/DAVIS240 share the same bit layout.
    (346, 260): _V2eCodec(
        width=346,
        height=260,
        x_shift=12,
        y_shift=22,
        p_shift=11,
        x_mask=(1 << 10) - 1,
        y_mask=(1 << 9) - 1,
        aps_mask=0x80000000,
        special_mask=1 << 10,
        flip_x=True,
        flip_y=True,
    ),
    (240, 180): _V2eCodec(
        width=240,
        height=180,
        x_shift=12,
        y_shift=22,
        p_shift=11,
        x_mask=(1 << 10) - 1,
        y_mask=(1 << 9) - 1,
        aps_mask=0x80000000,
        special_mask=1 << 10,
        flip_x=True,
        flip_y=True,
    ),
    # v2e AEDat2Output: DVS640 bit layout differs.
    (640, 480): _V2eCodec(
        width=640,
        height=480,
        x_shift=1,
        y_shift=11,
        p_shift=0,
        x_mask=(1 << 10) - 1,
        y_mask=(1 << 10) - 1,
        aps_mask=0x80000000,
        special_mask=None,
        flip_x=True,
        flip_y=True,
    ),
}


def looks_like_aedat2(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(len(_AEDAT2_MAGIC))
        return head.startswith(_AEDAT2_MAGIC)
    except Exception:
        return False


def _parse_tick_us(header_lines: list[str]) -> float:
    # Example: "# Timestamps tick is 1 us"
    for ln in header_lines:
        m = re.search(r"tick\s+is\s+([0-9]*\.?[0-9]+)\s*us", ln, flags=re.IGNORECASE)
        if m:
            try:
                v = float(m.group(1))
                if v > 0:
                    return v
            except Exception:
                pass
    return 1.0


def _try_read_v2e_args_geometry(aedat_path: str) -> tuple[int, int] | None:
    # DND21 ships v2e-args.txt next to .aedat, containing output_width/output_height.
    args_path = os.path.join(os.path.dirname(os.path.abspath(aedat_path)), "v2e-args.txt")
    if not os.path.exists(args_path):
        return None

    try:
        w = None
        h = None
        with open(args_path, "r", encoding="utf-8", errors="ignore") as f:
            for ln in f:
                s = ln.strip()
                if s.lower().startswith("output_width"):
                    m = re.search(r"(\d+)", s)
                    if m:
                        w = int(m.group(1))
                elif s.lower().startswith("output_height"):
                    m = re.search(r"(\d+)", s)
                    if m:
                        h = int(m.group(1))
                if w is not None and h is not None:
                    break
        if w is None or h is None:
            return None
        if (int(w), int(h)) in _V2E_CODECS:
            return int(w), int(h)
        return None
    except Exception:
        return None


def _skip_header(f) -> list[str]:
    header_lines: list[str] = []
    while True:
        pos = f.tell()
        ln = f.readline()
        if not ln:
            break
        if ln.startswith(b"#"):
            s = ln.decode("utf-8", errors="replace").rstrip("\r\n")
            header_lines.append(s)
            # Many AEDAT2 files terminate the header explicitly. Stop here to avoid
            # accidentally consuming binary payload that happens to start with '#'.
            if s.strip().lower().startswith("# end of header"):
                break
            continue
        # first non-comment line belongs to binary payload
        f.seek(pos)
        break
    return header_lines


def _codec_for_meta(meta: EventStreamMeta) -> _V2eCodec:
    wh = (int(meta.width), int(meta.height))
    if wh not in _V2E_CODECS:
        raise ValueError(
            f"Unsupported geometry for AEDAT2 writing: {wh[0]}x{wh[1]} (supported: {sorted(_V2E_CODECS.keys())})"
        )
    return _V2E_CODECS[wh]


def _encode_aedat2_payload_bytes(
    codec: _V2eCodec,
    meta: EventStreamMeta,
    b: EventBatch,
    *,
    tick_ns: float,
    dst_tick_us: float,
) -> bytes:
    # Ensure signed types for flips/subtraction.
    x = np.asarray(b.x, dtype=np.int64)
    y = np.asarray(b.y, dtype=np.int64)
    p = np.asarray(b.p, dtype=np.int8)
    t = np.asarray(b.t, dtype=np.uint64)

    w = int(meta.width)
    h = int(meta.height)

    in_range = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    if not bool(np.any(in_range)):
        return b""

    x = x[in_range]
    y = y[in_range]
    p = p[in_range]
    t = t[in_range]

    if codec.flip_x:
        x = (w - 1) - x
    if codec.flip_y:
        y = (h - 1) - y

    p01 = (p > 0).astype(np.uint32)

    addr = (
        ((x.astype(np.uint32) & np.uint32(codec.x_mask)) << np.uint32(codec.x_shift))
        | ((y.astype(np.uint32) & np.uint32(codec.y_mask)) << np.uint32(codec.y_shift))
        | (p01 << np.uint32(codec.p_shift))
    ).astype(np.uint32, copy=False)

    # Ensure APS/special bits are clear.
    addr &= np.uint32(~np.uint32(codec.aps_mask))
    if codec.special_mask is not None:
        addr &= np.uint32(~np.uint32(codec.special_mask))

    # Convert myEVS ticks -> dst timestamp ticks (usually 1us).
    # internal_time_us = t * tick_ns / 1000
    # dst_ts = round(internal_time_us / dst_tick_us)
    if dst_tick_us <= 0:
        raise ValueError("dst_tick_us must be > 0")
    scale = (float(tick_ns) / 1000.0) / float(dst_tick_us)
    ts_u64 = np.round(t.astype(np.float64) * scale).astype(np.uint64)
    if ts_u64.size:
        ts_max = int(ts_u64.max())
        if ts_max >= (1 << 32):
            raise ValueError(
                "Timestamp too large for AEDAT2 (uint32). "
                "Use a shorter clip, enable wrapping, or write to EVTQ/HDF5 instead. "
                f"max={ts_max}"
            )
    ts = ts_u64.astype(np.uint32, copy=False)

    # Interleave (addr, ts) as big-endian u32.
    out = np.empty((addr.shape[0] * 2,), dtype=np.dtype(">u4"))
    out[0::2] = addr
    out[1::2] = ts
    return out.tobytes(order="C")


def write_aedat2(
    path: str,
    meta: EventStreamMeta,
    batches: Iterable[EventBatch],
    *,
    tick_ns: float = 12.5,
    dst_tick_us: float = 1.0,
) -> None:
    """Write polarity events to an AEDAT2.0 file (v2e/jAER style).

    The output payload is big-endian uint32 pairs:
    (address, timestamp_tick_us), repeated.
    """

    codec = _codec_for_meta(meta)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        # Header (text lines starting with '#')
        f.write(b"#!AER-DAT2.0\r\n")
        f.write(b"# This is an AEDAT2.0 file generated by myEVS\r\n")
        f.write(f"# Timestamps tick is {dst_tick_us:g} us\r\n".encode("utf-8"))
        f.write(b"# End Of Header\r\n")

        for b in batches:
            if len(b) == 0:
                continue
            payload = _encode_aedat2_payload_bytes(codec, meta, b, tick_ns=float(tick_ns), dst_tick_us=float(dst_tick_us))
            if payload:
                f.write(payload)


def write_aedat2_passthrough(
    path: str,
    meta: EventStreamMeta,
    batches: Iterable[EventBatch],
    *,
    tick_ns: float = 12.5,
    dst_tick_us: float = 1.0,
) -> Iterator[EventBatch]:
    """Write AEDAT2 while yielding the same batches.

    Useful when you want to compute metrics and persist the denoised stream
    without iterating the generator twice.
    """

    codec = _codec_for_meta(meta)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    f = open(path, "wb")
    try:
        f.write(b"#!AER-DAT2.0\r\n")
        f.write(b"# This is an AEDAT2.0 file generated by myEVS\r\n")
        f.write(f"# Timestamps tick is {dst_tick_us:g} us\r\n".encode("utf-8"))
        f.write(b"# End Of Header\r\n")

        for b in batches:
            if len(b) == 0:
                continue
            payload = _encode_aedat2_payload_bytes(codec, meta, b, tick_ns=float(tick_ns), dst_tick_us=float(dst_tick_us))
            if payload:
                f.write(payload)
            yield b
    finally:
        f.close()


def _choose_codec(
    aedat_path: str,
    f,
    header_lines: list[str],
    *,
    width: int | None,
    height: int | None,
) -> _V2eCodec:
    wh = (int(width), int(height)) if (width is not None and height is not None) else None
    if wh is not None and wh in _V2E_CODECS:
        return _V2E_CODECS[wh]

    from_args = _try_read_v2e_args_geometry(aedat_path)
    if from_args is not None:
        return _V2E_CODECS[from_args]

    # Heuristic inference: decode a small prefix and pick a codec that yields in-range coords.
    pos = f.tell()
    probe = f.read(200_000 * 8)  # 200k events max
    f.seek(pos)

    if not probe:
        return _V2E_CODECS[(346, 260)]

    probe = probe[: (len(probe) // 8) * 8]
    if not probe:
        return _V2E_CODECS[(346, 260)]

    raw = np.frombuffer(probe, dtype=np.dtype(">u4")).astype(np.uint32)
    addr = raw[0::2]

    best: _V2eCodec | None = None
    best_score = -1.0

    for codec in _V2E_CODECS.values():
        keep = (addr & np.uint32(codec.aps_mask)) == 0
        if codec.special_mask is not None:
            keep &= (addr & np.uint32(codec.special_mask)) == 0
        if not bool(np.any(keep)):
            continue

        a = addr[keep]
        x = (a >> np.uint32(codec.x_shift)) & np.uint32(codec.x_mask)
        y = (a >> np.uint32(codec.y_shift)) & np.uint32(codec.y_mask)

        # prefer codecs that keep coords within declared geometry
        in_range = (x < np.uint32(codec.width)) & (y < np.uint32(codec.height))
        frac = float(np.mean(in_range.astype(np.float32)))
        if frac <= 0.99:
            continue

        # among valid codecs, prefer the one that uses the larger portion of the address space
        x_max = int(x.max())
        y_max = int(y.max())
        score = frac + 0.5 * ((x_max + 1) / float(codec.width) + (y_max + 1) / float(codec.height))

        if score > best_score:
            best_score = score
            best = codec

    return best if best is not None else _V2E_CODECS[(346, 260)]


def read_aedat2(
    path: str,
    *,
    width: int | None = None,
    height: int | None = None,
    batch_events: int = 1_000_000,
    tick_ns: float = 12.5,
) -> Tuple[Aedat2Info, Iterator[EventBatch]]:
    """Read v2e/jAER-style AEDAT2.0 polarity events.

    Supported (auto-inferred) geometries:
    - 346x260 (DAVIS346)
    - 240x180 (DAVIS240)
    - 640x480 (DVS640)

    File payload is interpreted as big-endian uint32 pairs:
    (address, timestamp_us_tick), repeated.

    Timestamps are converted into myEVS ticks using `tick_ns`.
    """

    f = open(path, "rb")
    try:
        header_lines = _skip_header(f)
        if not header_lines or not header_lines[0].encode("utf-8", errors="ignore").startswith(_AEDAT2_MAGIC):
            raise ValueError("Not an AEDAT2.0 file (missing #!AER-DAT2.0 header)")

        src_tick_us = _parse_tick_us(header_lines)
        codec = _choose_codec(path, f, header_lines, width=width, height=height)

        meta = EventStreamMeta(width=int(codec.width), height=int(codec.height), time_unit="tick")
        info = Aedat2Info(meta=meta, header_lines=tuple(header_lines), source_time_unit="us")

        tick_per_us = (1000.0 / float(tick_ns))
        scale = float(src_tick_us) * tick_per_us

        def gen() -> Iterator[EventBatch]:
            try:
                carry = b""
                chunk_bytes = max(1, int(batch_events)) * 8

                while True:
                    data = f.read(chunk_bytes)
                    if not data:
                        break
                    if carry:
                        data = carry + data
                        carry = b""

                    n = len(data) // 8
                    if n <= 0:
                        carry = data
                        continue

                    body = data[: n * 8]
                    carry = data[n * 8 :]

                    raw = np.frombuffer(body, dtype=np.dtype(">u4")).astype(np.uint32)
                    addr = raw[0::2]
                    ts = raw[1::2]

                    keep = (addr & np.uint32(codec.aps_mask)) == 0
                    if codec.special_mask is not None:
                        keep &= (addr & np.uint32(codec.special_mask)) == 0
                    if not bool(np.any(keep)):
                        continue

                    addr = addr[keep]
                    ts = ts[keep]

                    x = ((addr >> np.uint32(codec.x_shift)) & np.uint32(codec.x_mask)).astype(np.int64)
                    y = ((addr >> np.uint32(codec.y_shift)) & np.uint32(codec.y_mask)).astype(np.int64)
                    p01 = ((addr >> np.uint32(codec.p_shift)) & np.uint32(1)).astype(np.uint8)

                    in_range = (x < np.int64(codec.width)) & (y < np.int64(codec.height))
                    if not bool(np.any(in_range)):
                        continue

                    x = x[in_range]
                    y = y[in_range]
                    p01 = p01[in_range]
                    ts = ts[in_range]

                    if codec.flip_x:
                        x = (int(codec.width) - 1) - x
                    if codec.flip_y:
                        y = (int(codec.height) - 1) - y

                    # Convert to myEVS polarity convention (+1/-1)
                    p = np.where(p01 > 0, 1, -1).astype(np.int8)

                    # Convert timestamp tick(us) -> myEVS tick
                    t = np.round(ts.astype(np.float64) * scale).astype(np.uint64)

                    # Final types
                    yield EventBatch(
                        t=t,
                        x=np.asarray(x, dtype=np.uint16),
                        y=np.asarray(y, dtype=np.uint16),
                        p=p,
                    )
            finally:
                f.close()

        return info, gen()

    except Exception:
        f.close()
        raise
