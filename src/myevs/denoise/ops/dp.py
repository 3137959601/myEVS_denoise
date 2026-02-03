from __future__ import annotations

"""DP (Differing Polarity) filter with same-polarity timeout.

Qt reference:
- EventFrameProcessor::acceptDpEvent()
- method id: 8

Rules:
- If current polarity differs from last polarity at this pixel => keep
- If same polarity but time since last event >= time_window_us => keep
- Otherwise drop

Important detail:
- Update last polarity and last timestamp *always*, even when dropping.
"""

from dataclasses import dataclass

import numpy as np

from ...timebase import TimeBase
from ..types import DenoiseConfig
from .base import Dims


@dataclass
class DpOp:
    name: str = "dp"

    def __init__(self, dims: Dims, cfg: DenoiseConfig, tb: TimeBase):
        self.name = "dp"
        self.dims = dims
        self.cfg = cfg
        self.tb = tb

        n = dims.width * dims.height
        self.last_pol = np.zeros((n,), dtype=np.int8)  # 0 unknown, +1 ON, -1 OFF
        self.last_ts_any = np.zeros((n,), dtype=np.uint64)

    def accept(self, x: int, y: int, p: int, t: int) -> bool:
        idx0 = self.dims.idx(x, y)

        thr = int(self.tb.us_to_ticks(int(self.cfg.time_window_us)))

        cur_pol = 1 if p > 0 else -1
        prev_pol = int(self.last_pol[idx0])
        prev_ts = int(self.last_ts_any[idx0])

        # Always update state
        self.last_pol[idx0] = np.int8(cur_pol)
        self.last_ts_any[idx0] = np.uint64(t)

        if prev_pol == 0 or prev_ts == 0:
            return True
        if prev_pol != cur_pol:
            return True

        # same polarity
        if thr <= 0:
            return False
        if t <= prev_ts:
            return False
        return (t - prev_ts) >= thr
