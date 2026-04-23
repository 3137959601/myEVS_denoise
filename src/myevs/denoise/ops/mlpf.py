from __future__ import annotations

"""MLPF (multi-layer-perceptron-inspired filter, lightweight).

Reference model in cuke-emlb uses TorchScript + batched inference. To keep this
project dependency-light, we implement a deterministic proxy that mirrors its
feature construction:
- 7x7 neighborhood around current event
- recency channel: 1 - dt / duration (clipped to [0, 1])
- polarity channel: same-polarity preference
"""

from dataclasses import dataclass

import numpy as np

from ...timebase import TimeBase
from ..types import DenoiseConfig
from .base import Dims


@dataclass
class MlpfOp:
    name: str = "mlpf"

    def __init__(self, dims: Dims, cfg: DenoiseConfig, tb: TimeBase):
        self.name = "mlpf"
        self.dims = dims
        self.cfg = cfg
        self.tb = tb

        n = int(dims.width) * int(dims.height)
        self.last_ts = np.zeros((n,), dtype=np.uint64)
        self.last_pol = np.zeros((n,), dtype=np.int8)

    def accept(self, x: int, y: int, p: int, t: int) -> bool:
        # Keep 7x7 window fixed as in reference MLP input.
        r = 3
        win_ticks = int(self.tb.us_to_ticks(int(self.cfg.time_window_us)))
        thr = float(self.cfg.min_neighbors)
        pp = 1 if p > 0 else -1

        idx0 = self.dims.idx(x, y)
        if win_ticks <= 0:
            self.last_ts[idx0] = np.uint64(t)
            self.last_pol[idx0] = np.int8(pp)
            return thr <= 0.0

        y0 = max(0, y - r)
        y1 = min(self.dims.height - 1, y + r)
        x0 = max(0, x - r)
        x1 = min(self.dims.width - 1, x + r)

        score = 0.0
        inv_win = 1.0 / float(win_ticks)
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
                recency = 1.0 - float(dt) * inv_win
                # same-polarity gate, matching the spirit of the second channel.
                if int(self.last_pol[idx]) == pp:
                    score += recency

        self.last_ts[idx0] = np.uint64(t)
        self.last_pol[idx0] = np.int8(pp)
        return score >= thr

