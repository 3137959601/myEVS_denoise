from __future__ import annotations

"""Ynoise (Yang noise filter).

Reference:
- cuke-emlb/include/denoisors/yang_noise.hpp

Core idea:
- Maintain per-pixel timestamp + polarity maps.
- Count neighborhood events within a temporal window and same polarity.
- Keep if density >= threshold.
"""

from dataclasses import dataclass

import numpy as np

from ...timebase import TimeBase
from ..types import DenoiseConfig
from .base import Dims


@dataclass
class YnoiseOp:
    name: str = "ynoise"

    def __init__(self, dims: Dims, cfg: DenoiseConfig, tb: TimeBase):
        self.name = "ynoise"
        self.dims = dims
        self.cfg = cfg
        self.tb = tb

        n = int(dims.width) * int(dims.height)
        self.last_ts = np.zeros((n,), dtype=np.uint64)
        self.last_pol = np.zeros((n,), dtype=np.int8)  # {-1, 0, +1}

    def accept(self, x: int, y: int, p: int, t: int) -> bool:
        r = max(0, min(int(self.cfg.radius_px), 8))
        win_ticks = int(self.tb.us_to_ticks(int(self.cfg.time_window_us)))
        thr = max(0, int(self.cfg.min_neighbors))
        pp = 1 if p > 0 else -1

        idx0 = self.dims.idx(x, y)

        if win_ticks <= 0:
            self.last_ts[idx0] = np.uint64(t)
            self.last_pol[idx0] = np.int8(pp)
            return thr <= 0

        density = 0
        y0 = max(0, y - r)
        y1 = min(self.dims.height - 1, y + r)
        x0 = max(0, x - r)
        x1 = min(self.dims.width - 1, x + r)

        for yy in range(y0, y1 + 1):
            base = yy * self.dims.width
            for xx in range(x0, x1 + 1):
                idx = base + xx
                ts = int(self.last_ts[idx])
                if ts == 0:
                    continue
                dt = (t - ts) if t >= ts else (ts - t)
                if dt > win_ticks:
                    continue
                if int(self.last_pol[idx]) != pp:
                    continue
                density += 1

        self.last_ts[idx0] = np.uint64(t)
        self.last_pol[idx0] = np.int8(pp)
        return density >= thr

