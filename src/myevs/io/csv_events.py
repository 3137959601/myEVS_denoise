from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np

from ..events import EventBatch, EventStreamMeta


@dataclass(frozen=True)
class CsvInfo:
    meta: EventStreamMeta


def read_csv_events(path: str, width: int, height: int, batch_events: int = 1_000_000) -> Tuple[CsvInfo, Iterator[EventBatch]]:
    meta = EventStreamMeta(width=width, height=height, time_unit="tick")
    info = CsvInfo(meta=meta)

    def gen() -> Iterator[EventBatch]:
        t_buf = []
        x_buf = []
        y_buf = []
        p_buf = []
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                t_buf.append(int(row["t"]))
                x_buf.append(int(row["x"]))
                y_buf.append(int(row["y"]))
                p01 = int(row["p"])
                p_buf.append(1 if p01 else -1)
                if len(t_buf) >= batch_events:
                    yield EventBatch(
                        t=np.asarray(t_buf, dtype=np.uint64),
                        x=np.asarray(x_buf, dtype=np.uint16),
                        y=np.asarray(y_buf, dtype=np.uint16),
                        p=np.asarray(p_buf, dtype=np.int8),
                    )
                    t_buf.clear(); x_buf.clear(); y_buf.clear(); p_buf.clear()
        if t_buf:
            yield EventBatch(
                t=np.asarray(t_buf, dtype=np.uint64),
                x=np.asarray(x_buf, dtype=np.uint16),
                y=np.asarray(y_buf, dtype=np.uint16),
                p=np.asarray(p_buf, dtype=np.int8),
            )

    return info, gen()


def write_csv_events(path: str, batches: Iterator[EventBatch]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "x", "y", "p"])
        for b in batches:
            if len(b) == 0:
                continue
            t = np.asarray(b.t, dtype=np.uint64)
            x = np.asarray(b.x, dtype=np.uint16)
            y = np.asarray(b.y, dtype=np.uint16)
            p01 = (np.asarray(b.p, dtype=np.int8) > 0).astype(np.uint8)
            for i in range(t.shape[0]):
                w.writerow([int(t[i]), int(x[i]), int(y[i]), int(p01[i])])
