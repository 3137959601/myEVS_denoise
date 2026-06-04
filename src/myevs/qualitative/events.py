from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from myevs.events import EventBatch
from myevs.io.auto import open_events
from myevs.timebase import TimeBase


def read_window_batches(
    path: str | Path,
    *,
    width: int,
    height: int,
    tick_ns: float,
    assume: str | None,
    start_us: int,
    duration_us: int,
    batch_events: int = 1_000_000,
) -> Iterator[EventBatch]:
    result = open_events(
        str(path),
        width=int(width),
        height=int(height),
        tick_ns=float(tick_ns),
        assume=assume,
        batch_events=int(batch_events),
    )
    tb = TimeBase(tick_ns=float(tick_ns))
    start_tick = int(tb.us_to_ticks(int(start_us)))
    end_tick = int(tb.us_to_ticks(int(start_us + duration_us)))
    for b in result.batches:
        if len(b) == 0:
            continue
        t = np.asarray(b.t, dtype=np.uint64)
        keep = (t >= np.uint64(start_tick)) & (t < np.uint64(end_tick))
        if keep.any():
            yield EventBatch(t=t[keep], x=b.x[keep], y=b.y[keep], p=b.p[keep])
        if t.size and int(t[-1]) >= end_tick:
            break


def concat_event_batches(batches: Iterable[EventBatch]) -> EventBatch:
    ts: list[np.ndarray] = []
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    for b in batches:
        if len(b) == 0:
            continue
        ts.append(np.asarray(b.t, dtype=np.uint64))
        xs.append(np.asarray(b.x, dtype=np.uint16))
        ys.append(np.asarray(b.y, dtype=np.uint16))
        ps.append(np.asarray(b.p, dtype=np.int8))
    if not ts:
        return EventBatch(
            t=np.asarray([], dtype=np.uint64),
            x=np.asarray([], dtype=np.uint16),
            y=np.asarray([], dtype=np.uint16),
            p=np.asarray([], dtype=np.int8),
        )
    return EventBatch(t=np.concatenate(ts), x=np.concatenate(xs), y=np.concatenate(ys), p=np.concatenate(ps))


def save_npz_events(path: str | Path, batch: EventBatch) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        p,
        t=np.asarray(batch.t, dtype=np.uint64),
        x=np.asarray(batch.x, dtype=np.uint16),
        y=np.asarray(batch.y, dtype=np.uint16),
        p=np.asarray(batch.p, dtype=np.int8),
    )


def event_stats(batch: EventBatch, *, width: int, height: int) -> dict[str, float | int]:
    n = len(batch)
    if n == 0:
        return {
            "events": 0,
            "on": 0,
            "off": 0,
            "active_pixels": 0,
            "active_pixel_ratio": 0.0,
            "t_first": "",
            "t_last": "",
        }
    x = np.asarray(batch.x, dtype=np.int64)
    y = np.asarray(batch.y, dtype=np.int64)
    p = np.asarray(batch.p, dtype=np.int8)
    inb = (x >= 0) & (x < int(width)) & (y >= 0) & (y < int(height))
    if inb.any():
        active = np.unique(y[inb] * int(width) + x[inb]).shape[0]
    else:
        active = 0
    return {
        "events": int(n),
        "on": int(np.count_nonzero(p > 0)),
        "off": int(np.count_nonzero(p <= 0)),
        "active_pixels": int(active),
        "active_pixel_ratio": float(active) / float(max(1, int(width) * int(height))),
        "t_first": int(np.asarray(batch.t)[0]),
        "t_last": int(np.asarray(batch.t)[-1]),
    }
