from __future__ import annotations

"""TS (Time Surface filter).

Reference:
- cuke-emlb/include/denoisors/time_surface.hpp

Core idea:
- For each polarity, keep a time-surface map of last timestamps.
- Compute neighborhood mean exp(-(t-ts)/decay) at event location.
- Keep if surface value >= threshold.
"""

from dataclasses import dataclass
import math

import numpy as np

from ...timebase import TimeBase
from ..types import DenoiseConfig
from .base import Dims


@dataclass
class TsOp:
    name: str = "ts"

    def __init__(self, dims: Dims, cfg: DenoiseConfig, tb: TimeBase):
        self.name = "ts"
        self.dims = dims
        self.cfg = cfg
        self.tb = tb

        n = int(dims.width) * int(dims.height)
        self.pos_ts = np.zeros((n,), dtype=np.uint64)
        self.neg_ts = np.zeros((n,), dtype=np.uint64)

    def accept(self, x: int, y: int, p: int, t: int) -> bool:
        r = max(0, min(int(self.cfg.radius_px), 8))
        decay_ticks = int(self.tb.us_to_ticks(int(self.cfg.time_window_us)))
        thr = float(self.cfg.min_neighbors)

        idx0 = self.dims.idx(x, y)
        cell = self.pos_ts if p > 0 else self.neg_ts

        if decay_ticks <= 0:
            cell[idx0] = np.uint64(t)
            return thr <= 0.0

        support = 0
        surf = 0.0
        inv_decay = 1.0 / float(decay_ticks)

        y0 = max(0, y - r)
        y1 = min(self.dims.height - 1, y + r)
        x0 = max(0, x - r)
        x1 = min(self.dims.width - 1, x + r)

        for yy in range(y0, y1 + 1):
            base = yy * self.dims.width
            for xx in range(x0, x1 + 1):
                ts = int(cell[base + xx])
                if ts == 0:
                    continue
                dt = t - ts
                # Keep numerical behavior stable when timestamps are not monotonic.
                if dt < 0:
                    dt = -dt
                surf += math.exp(-float(dt) * inv_decay)
                support += 1

        score = (surf / float(support)) if support > 0 else 0.0
        cell[idx0] = np.uint64(t)
        return score >= thr

