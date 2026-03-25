from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator, Tuple

from ..events import EventBatch, EventStreamMeta
from .aedat2_events import looks_like_aedat2, read_aedat2
from .csv_events import read_csv_events
from .npz_events import read_npy_events, read_npz_events
from .evtq import read_evtq
from .hdf5_events import read_hdf5
from .usb_raw_evt3 import read_usb_raw_evt3


@dataclass(frozen=True)
class OpenResult:
    meta: EventStreamMeta
    batches: Iterator[EventBatch]


def open_events(
    path: str,
    *,
    width: int | None = None,
    height: int | None = None,
    batch_events: int = 1_000_000,
    tick_ns: float = 12.5,
    hdf5_plugin_path: str | None = None,
    assume: str | None = None,
) -> OpenResult:
    """Open an event stream.

    - `.evtq`: contains width/height.
    - `.csv`: requires width/height.
    - `.hdf5` / `.h5`: width/height from metadata if available (otherwise pass explicitly).
    - `.aedat` / `.aedat2`: AEDAT-2.0 polarity events (v2e/jAER style); geometry inferred for common DAVIS sizes.
    - `.bin` / others: treated as USB raw EVT3 by default; requires width/height.

    Use `assume` to override: "evtq" | "csv" | "hdf5" | "aedat2" | "usb_raw_evt3".
    """

    ext = os.path.splitext(path)[1].lower()
    kind = assume
    if kind is None:
        if ext == ".evtq":
            kind = "evtq"
        elif ext == ".csv":
            kind = "csv"
        elif ext == ".npz":
            kind = "npz"
        elif ext == ".npy":
            kind = "npy"
        elif ext in (".hdf5", ".h5"):
            kind = "hdf5"
        elif ext in (".aedat", ".aedat2"):
            kind = "aedat2"
        else:
            kind = "aedat2" if looks_like_aedat2(path) else "usb_raw_evt3"

    if kind == "evtq":
        info, batches = read_evtq(path, batch_events=batch_events)
        return OpenResult(meta=info.meta, batches=batches)

    if kind == "hdf5":
        info, batches = read_hdf5(
            path,
            width=width,
            height=height,
            batch_events=batch_events,
            tick_ns=tick_ns,
            hdf5_plugin_path=hdf5_plugin_path,
        )
        return OpenResult(meta=info.meta, batches=batches)

    if kind in ("aedat2", "aedat"):
        info, batches = read_aedat2(
            path,
            width=width,
            height=height,
            batch_events=batch_events,
            tick_ns=tick_ns,
        )
        return OpenResult(meta=info.meta, batches=batches)

    if kind in ("npz", "npy"):
        if width is None or height is None:
            raise ValueError(
                f"width/height required for kind={kind}. "
                f"Example: myevs stats --in {path} --width 346 --height 260"
            )
        if kind == "npz":
            info, batches = read_npz_events(path, width=width, height=height, batch_events=batch_events)
        else:
            info, batches = read_npy_events(path, width=width, height=height, batch_events=batch_events)
        return OpenResult(meta=info.meta, batches=batches)

    if width is None or height is None:
        raise ValueError(
            f"width/height required for kind={kind}. "
            f"Example: myevs view --in {path} --width 640 --height 512 ..."
        )

    if kind == "csv":
        info, batches = read_csv_events(path, width=width, height=height, batch_events=batch_events)
        return OpenResult(meta=info.meta, batches=batches)

    if kind in ("usb_raw_evt3", "usb", "usb_raw"):
        info, batches = read_usb_raw_evt3(path, width=width, height=height, batch_events=batch_events)
        return OpenResult(meta=info.meta, batches=batches)

    raise ValueError(f"Unknown kind: {kind}")
