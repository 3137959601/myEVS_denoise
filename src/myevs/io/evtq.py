from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np

from ..events import EventBatch, EventStreamMeta


_EVTQ_HDR = struct.Struct("<4sHHHHI")  # magic, version, headerBytes, width, height, reserved
_EVTQ_EVT = struct.Struct("<QHHBB")    # t, x, y, p(1=ON,0=OFF), reserved


@dataclass(frozen=True)
class EvtqInfo:
    meta: EventStreamMeta
    version: int


def read_evtq(path: str, batch_events: int = 1_000_000) -> Tuple[EvtqInfo, Iterator[EventBatch]]:
    # NOTE: Don't use `with open(...)` here.
    # We return a generator that lazily reads from the file; using `with` would
    # close the file before the caller starts iterating, causing:
    #   ValueError: read of closed file
    f = open(path, "rb")
    try:
        hdr = f.read(_EVTQ_HDR.size)
        if len(hdr) != _EVTQ_HDR.size:
            raise ValueError("Invalid EVTQ: short header")
        magic, ver, header_bytes, w, h, _ = _EVTQ_HDR.unpack(hdr)
        if magic != b"EVTQ":
            raise ValueError(f"Invalid EVTQ magic: {magic!r}")
        if header_bytes > _EVTQ_HDR.size:
            f.read(header_bytes - _EVTQ_HDR.size)

        meta = EventStreamMeta(width=int(w), height=int(h), time_unit="tick")
        info = EvtqInfo(meta=meta, version=int(ver))

        def gen() -> Iterator[EventBatch]:
            try:
                t_buf = []
                x_buf = []
                y_buf = []
                p_buf = []
                read_size = _EVTQ_EVT.size * 1024
                data = f.read(read_size)
                while data:
                    # trim to whole events
                    n = (len(data) // _EVTQ_EVT.size) * _EVTQ_EVT.size
                    if n <= 0:
                        data = f.read(read_size)
                        continue
                    mv = memoryview(data)[:n]
                    for off in range(0, n, _EVTQ_EVT.size):
                        t, x, y, p01, _r = _EVTQ_EVT.unpack_from(mv, off)
                        t_buf.append(t)
                        x_buf.append(x)
                        y_buf.append(y)
                        p_buf.append(1 if p01 else -1)
                        if len(t_buf) >= batch_events:
                            yield EventBatch(
                                t=np.asarray(t_buf, dtype=np.uint64),
                                x=np.asarray(x_buf, dtype=np.uint16),
                                y=np.asarray(y_buf, dtype=np.uint16),
                                p=np.asarray(p_buf, dtype=np.int8),
                            )
                            t_buf.clear(); x_buf.clear(); y_buf.clear(); p_buf.clear()
                    data = f.read(read_size)
                if t_buf:
                    yield EventBatch(
                        t=np.asarray(t_buf, dtype=np.uint64),
                        x=np.asarray(x_buf, dtype=np.uint16),
                        y=np.asarray(y_buf, dtype=np.uint16),
                        p=np.asarray(p_buf, dtype=np.int8),
                    )
            finally:
                f.close()

        return info, gen()
    except Exception:
        f.close()
        raise


def write_evtq(path: str, meta: EventStreamMeta, batches: Iterator[EventBatch]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        hdr = _EVTQ_HDR.pack(b"EVTQ", 1, _EVTQ_HDR.size, meta.width, meta.height, 0)
        f.write(hdr)
        for b in batches:
            if len(b) == 0:
                continue
            # Ensure types
            t = np.asarray(b.t, dtype=np.uint64)
            x = np.asarray(b.x, dtype=np.uint16)
            y = np.asarray(b.y, dtype=np.uint16)
            p01 = (np.asarray(b.p, dtype=np.int8) > 0).astype(np.uint8)
            for i in range(t.shape[0]):
                f.write(_EVTQ_EVT.pack(int(t[i]), int(x[i]), int(y[i]), int(p01[i]), 0))
