from __future__ import annotations

"""Hot pixel mask.

Qt reference:
- EventFrameProcessor::acceptHotPixelEvent()
- method id: 3

Logic:
- For each pixel, count events in a time window (time_window_us).
- If count >= min_neighbors (threshold), treat as hot pixel and mask (drop)
  for refractory_us duration.

Important details (match Qt):
- If pixel is currently masked, drop immediately and do NOT update window count.
- When triggering masking, reset the window state to avoid immediate re-trigger.
"""

from dataclasses import dataclass

import numpy as np

from ...timebase import TimeBase
from ..types import DenoiseConfig
from .base import Dims


@dataclass
class HotPixelOp:
    name: str = "hotpixel"

    def __init__(self, dims: Dims, cfg: DenoiseConfig, tb: TimeBase):
        self.name = "hotpixel"
        self.dims = dims
        self.cfg = cfg
        self.tb = tb

        n = dims.width * dims.height
        self.win_start = np.zeros((n,), dtype=np.uint64)
        self.masked_until = np.zeros((n,), dtype=np.uint64)
        self.win_count = np.zeros((n,), dtype=np.uint16)

    def accept(self, x: int, y: int, p: int, t: int) -> bool:  # p unused, kept for uniform signature
        idx0 = self.dims.idx(x, y)

        win_ticks = int(self.tb.us_to_ticks(int(self.cfg.time_window_us)))
        mask_ticks = int(self.tb.us_to_ticks(int(self.cfg.refractory_us)))
        thr = max(1, int(self.cfg.min_neighbors))

        # If still masked: drop and do NOT update window.
        mu = int(self.masked_until[idx0])
        if mu != 0 and t < mu:
            return False

        if win_ticks <= 0:
            return True

        ws = int(self.win_start[idx0])
        cnt = int(self.win_count[idx0])

        # Start new window if needed
        if ws == 0 or t < ws or (t - ws) > win_ticks:
            self.win_start[idx0] = np.uint64(t)
            self.win_count[idx0] = np.uint16(1)
            return True

        # Same window: increment
        if cnt < 0xFFFF:
            cnt += 1
            self.win_count[idx0] = np.uint16(cnt)

        if cnt >= thr:
            # Trigger: mask a while, then reset window to avoid near-permanent masking.
            until = (t + mask_ticks) if mask_ticks > 0 else (ws + win_ticks)
            self.masked_until[idx0] = np.uint64(until)
            self.win_start[idx0] = np.uint64(0)
            self.win_count[idx0] = np.uint16(0)
            return False

        return True
