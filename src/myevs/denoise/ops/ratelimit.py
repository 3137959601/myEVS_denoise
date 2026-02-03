from __future__ import annotations

"""Rate limit (per pixel).

Qt reference:
- EventFrameProcessor::acceptRateLimitEvent()
- method id: 6

Logic:
- For each pixel (and each polarity), count events in a time window.
- If count exceeds limit (min_neighbors), drop until window rolls.
"""

from dataclasses import dataclass

import numpy as np

from ...timebase import TimeBase
from ..types import DenoiseConfig
from .base import Dims


@dataclass
class RateLimitOp:
    name: str = "ratelimit"

    def __init__(self, dims: Dims, cfg: DenoiseConfig, tb: TimeBase):
        self.name = "ratelimit"
        self.dims = dims
        self.cfg = cfg
        self.tb = tb

        n = dims.width * dims.height
        self.win_start_on = np.zeros((n,), dtype=np.uint64)
        self.win_start_off = np.zeros((n,), dtype=np.uint64)
        self.win_count_on = np.zeros((n,), dtype=np.uint16)
        self.win_count_off = np.zeros((n,), dtype=np.uint16)

    def accept(self, x: int, y: int, p: int, t: int) -> bool:
        idx0 = self.dims.idx(x, y)

        win_ticks = int(self.tb.us_to_ticks(int(self.cfg.time_window_us)))
        if win_ticks <= 0:
            return True

        limit = max(1, int(self.cfg.min_neighbors))

        if p > 0:
            ws = self.win_start_on
            cnt = self.win_count_on
        else:
            ws = self.win_start_off
            cnt = self.win_count_off

        t0 = int(ws[idx0])
        c = int(cnt[idx0])

        if t0 == 0 or t < t0 or (t - t0) > win_ticks:
            ws[idx0] = np.uint64(t)
            cnt[idx0] = np.uint16(1)
            return True

        if c < limit:
            if c < 0xFFFF:
                c += 1
                cnt[idx0] = np.uint16(c)
            return True

        return False
