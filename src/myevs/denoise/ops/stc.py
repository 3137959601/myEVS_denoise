from __future__ import annotations

"""STC (Spatio-Temporal Correlation) filter.

Qt reference:
- EventFrameProcessor::acceptDenoiseEvent()
- method id: 1
- combo id: 5 (STC + Refractory)

Algorithm summary (same-polarity only):
- Maintain per-pixel last timestamp for ON and OFF separately.
- Always update the current pixel's timestamp (IMPORTANT).
- Count how many pixels in the radius-r neighborhood have timestamp >= t - win.
- Keep the event if count >= min_neighbors.

Notes for beginners:
- This filter is relatively expensive because it checks a small neighborhood
  for every single event.
- Start with small radius (r=1) and small batches.
"""

from dataclasses import dataclass

import numpy as np

from ...timebase import TimeBase
from ..types import DenoiseConfig
from .base import Dims


@dataclass
class StcOp:
    name: str = "stc"

    def __init__(self, dims: Dims, cfg: DenoiseConfig, tb: TimeBase):
        self.name = "stc"
        self.dims = dims
        self.cfg = cfg
        self.tb = tb

        n = dims.width * dims.height
        # last timestamp for same polarity
        self.last_on = np.zeros((n,), dtype=np.uint64)
        self.last_off = np.zeros((n,), dtype=np.uint64)

    def accept(self, x: int, y: int, p: int, t: int) -> bool:
        # Parameters (Qt names):
        r = max(0, min(int(self.cfg.radius_px), 8))
        need = max(0, int(self.cfg.min_neighbors))
        win_ticks = int(self.tb.us_to_ticks(int(self.cfg.time_window_us)))

        idx0 = self.dims.idx(x, y)
        last = self.last_on if p > 0 else self.last_off

        # IMPORTANT: always update, even if we drop.
        last[idx0] = np.uint64(t)

        # Fast paths (match Qt)
        if need <= 1:
            return True
        if win_ticks <= 0:
            return False

        t0 = t - win_ticks if t > win_ticks else 0

        # Count neighborhood pixels with recent activity (includes self)
        y0 = max(0, y - r)
        y1 = min(self.dims.height - 1, y + r)
        x0 = max(0, x - r)
        x1 = min(self.dims.width - 1, x + r)

        cnt = 0
        for yy in range(y0, y1 + 1):
            base = yy * self.dims.width
            for xx in range(x0, x1 + 1):
                ts = int(last[base + xx])
                if ts != 0 and ts >= t0:
                    cnt += 1
                    if cnt >= need:
                        return True
        return False
