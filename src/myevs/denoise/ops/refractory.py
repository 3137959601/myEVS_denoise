from __future__ import annotations

"""Refractory filter.

Qt reference:
- EventFrameProcessor::acceptRefractoryEvent()
- method id: 2

Idea:
- For each pixel, keep the last *accepted* timestamp (per polarity).
- If a new event arrives too soon (< refractory_us), drop it.

Important detail:
- When dropping, do NOT update last accepted timestamp.
  (Otherwise the reject window could keep extending.)
"""

from dataclasses import dataclass

import numpy as np

from ...timebase import TimeBase
from ..types import DenoiseConfig
from .base import Dims


@dataclass
class RefractoryOp:
    name: str = "refractory"

    def __init__(self, dims: Dims, cfg: DenoiseConfig, tb: TimeBase):
        self.name = "refractory"
        self.dims = dims
        self.cfg = cfg
        self.tb = tb

        n = dims.width * dims.height
        self.last_acc_on = np.zeros((n,), dtype=np.uint64)
        self.last_acc_off = np.zeros((n,), dtype=np.uint64)

    def accept(self, x: int, y: int, p: int, t: int) -> bool:
        thr = int(self.tb.us_to_ticks(int(self.cfg.refractory_us)))
        if thr <= 0:
            return True

        idx0 = self.dims.idx(x, y)
        last_acc = self.last_acc_on if p > 0 else self.last_acc_off
        prev = int(last_acc[idx0])

        if prev != 0 and t > prev and (t - prev) < thr:
            return False

        last_acc[idx0] = np.uint64(t)
        return True
