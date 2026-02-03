from __future__ import annotations

"""BAF (Background Activity Filter) - simplified.

Qt reference:
- EventFrameProcessor::acceptBafEvent()
- method id: 4

Idea:
- Maintain per-pixel last timestamp (ignoring polarity).
- If a pixel fires but its neighborhood had no activity within the time window,
  treat it as background noise and drop it.

Important detail:
- Update current pixel's timestamp ALWAYS (even if we drop).
  This avoids "no seed" areas where nothing can ever pass.
"""

from dataclasses import dataclass

import numpy as np

from ...timebase import TimeBase
from ..types import DenoiseConfig
from .base import Dims


@dataclass
class BafOp:
    name: str = "baf"

    def __init__(self, dims: Dims, cfg: DenoiseConfig, tb: TimeBase):
        self.name = "baf"
        self.dims = dims
        self.cfg = cfg
        self.tb = tb

        n = dims.width * dims.height
        self.last_any = np.zeros((n,), dtype=np.uint64)

    def accept(self, x: int, y: int, p: int, t: int) -> bool:  # p unused
        r = max(0, min(int(self.cfg.radius_px), 8))
        win_ticks = int(self.tb.us_to_ticks(int(self.cfg.time_window_us)))

        idx0 = self.dims.idx(x, y)

        # r==0: Qt treats as pass-through, but still updates timestamp.
        if r <= 0 or win_ticks <= 0:
            self.last_any[idx0] = np.uint64(t)
            return True

        t0 = t - win_ticks if t > win_ticks else 0

        has_neighbor = False
        y0 = max(0, y - r)
        y1 = min(self.dims.height - 1, y + r)
        x0 = max(0, x - r)
        x1 = min(self.dims.width - 1, x + r)

        for yy in range(y0, y1 + 1):
            base = yy * self.dims.width
            for xx in range(x0, x1 + 1):
                if xx == x and yy == y:
                    continue
                ts = int(self.last_any[base + xx])
                if ts != 0 and ts >= t0:
                    has_neighbor = True
                    break
            if has_neighbor:
                break

        # Always update self timestamp
        self.last_any[idx0] = np.uint64(t)
        return has_neighbor
