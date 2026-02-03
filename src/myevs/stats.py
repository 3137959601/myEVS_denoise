from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .events import EventBatch, EventStreamMeta


@dataclass(frozen=True)
class StreamStats:
    total: int
    on: int
    off: int
    t_first: int | None
    t_last: int | None

    @property
    def duration_ticks(self) -> int | None:
        if self.t_first is None or self.t_last is None:
            return None
        return int(self.t_last - self.t_first)


def compute_stats(_meta: EventStreamMeta, batches: Iterator[EventBatch]) -> StreamStats:
    total = 0
    on = 0
    off = 0
    t_first = None
    t_last = None

    for b in batches:
        if len(b) == 0:
            continue
        total += len(b)
        on += int(np.count_nonzero(b.p > 0))
        off += int(np.count_nonzero(b.p <= 0))
        t_arr = np.asarray(b.t, dtype=np.uint64)
        tb0 = int(t_arr.min())
        tbl = int(t_arr.max())
        if t_first is None or tb0 < t_first:
            t_first = tb0
        if t_last is None or tbl > t_last:
            t_last = tbl

    return StreamStats(total=total, on=on, off=off, t_first=t_first, t_last=t_last)
