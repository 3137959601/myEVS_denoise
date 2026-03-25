from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np

from ..events import EventBatch, EventStreamMeta, to_pm1


@dataclass(frozen=True)
class NpzInfo:
    meta: EventStreamMeta


def _load_npz_arrays(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z = np.load(path, mmap_mode="r")
    # Required keys.
    t = z["t"]
    x = z["x"]
    y = z["y"]
    p = z["p"]
    return t, x, y, p


def read_npz_events(
    path: str,
    *,
    width: int,
    height: int,
    batch_events: int = 1_000_000,
) -> Tuple[NpzInfo, Iterator[EventBatch]]:
    """Read events from a .npz produced by scripts/v2e_labeled_txt_to_npy.py.

    Expected keys: t, x, y, p (label may exist but is ignored for denoise/view).
    t is in ticks (uint64). p is either 0/1 or -1/+1.
    """

    meta = EventStreamMeta(width=int(width), height=int(height), time_unit="tick")
    info = NpzInfo(meta=meta)

    t, x, y, p = _load_npz_arrays(path)

    def gen() -> Iterator[EventBatch]:
        n = int(t.shape[0])
        for i0 in range(0, n, int(batch_events)):
            i1 = min(n, i0 + int(batch_events))
            tt = np.asarray(t[i0:i1], dtype=np.uint64)
            xx = np.asarray(x[i0:i1], dtype=np.uint16)
            yy = np.asarray(y[i0:i1], dtype=np.uint16)
            pp = np.asarray(p[i0:i1])
            if pp.dtype != np.int8:
                pp = pp.astype(np.int8)
            # Accept either 0/1 or -1/+1
            if pp.size and int(pp.min()) >= 0 and int(pp.max()) <= 1:
                pp = to_pm1(pp)
            yield EventBatch(t=tt, x=xx, y=yy, p=pp)

    return info, gen()


def read_npy_events(
    path: str,
    *,
    width: int,
    height: int,
    batch_events: int = 1_000_000,
) -> Tuple[NpzInfo, Iterator[EventBatch]]:
    """Read events from a .npy.

    Supported layouts:
    1) Structured dtype with fields 't','x','y','p' (and optional 'label').
    2) Numeric array with shape (N,4) or (N,5) in columns [t,x,y,p,(label)].

    t is assumed to be ticks (uint64).
    p can be 0/1 or -1/+1.
    """

    meta = EventStreamMeta(width=int(width), height=int(height), time_unit="tick")
    info = NpzInfo(meta=meta)

    arr = np.load(path, mmap_mode="r")

    has_fields = getattr(arr.dtype, "names", None) is not None

    def _slice_fields(i0: int, i1: int) -> EventBatch:
        tt = np.asarray(arr["t"][i0:i1], dtype=np.uint64)
        xx = np.asarray(arr["x"][i0:i1], dtype=np.uint16)
        yy = np.asarray(arr["y"][i0:i1], dtype=np.uint16)
        pp = np.asarray(arr["p"][i0:i1])
        if pp.dtype != np.int8:
            pp = pp.astype(np.int8)
        if pp.size and int(pp.min()) >= 0 and int(pp.max()) <= 1:
            pp = to_pm1(pp)
        return EventBatch(t=tt, x=xx, y=yy, p=pp)

    def _slice_cols(i0: int, i1: int) -> EventBatch:
        chunk = arr[i0:i1]
        if chunk.ndim != 2 or chunk.shape[1] < 4:
            raise ValueError(f"Unsupported npy array shape: {chunk.shape}")
        tt = np.asarray(chunk[:, 0], dtype=np.uint64)
        xx = np.asarray(chunk[:, 1], dtype=np.uint16)
        yy = np.asarray(chunk[:, 2], dtype=np.uint16)
        pp = np.asarray(chunk[:, 3])
        if pp.dtype != np.int8:
            pp = pp.astype(np.int8)
        if pp.size and int(pp.min()) >= 0 and int(pp.max()) <= 1:
            pp = to_pm1(pp)
        return EventBatch(t=tt, x=xx, y=yy, p=pp)

    def gen() -> Iterator[EventBatch]:
        n = int(arr.shape[0])
        for i0 in range(0, n, int(batch_events)):
            i1 = min(n, i0 + int(batch_events))
            yield _slice_fields(i0, i1) if has_fields else _slice_cols(i0, i1)

    return info, gen()
