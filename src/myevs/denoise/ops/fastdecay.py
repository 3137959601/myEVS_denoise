from __future__ import annotations

"""Fast decay noise filter (dv-processing port).

Reference concept:
- dv::noise::FastDecayNoiseFilter (dv-processing)

Core idea
---------
We build a *low-resolution* grid by subdividing the sensor plane into square cells.
Each incoming event is mapped to one cell:

  cx = x // subdivision_factor
  cy = y // subdivision_factor

For each cell we maintain two state variables:
- last timestamp (ticks)
- an activity value (float)

When an event arrives at time `t`, we apply an exponential decay to the activity
value based on the time since the cell was last hit, then add 1:

  decay_mult = 2 ** (-(t - last_t) / half_life)
  value = value * decay_mult + 1

`half_life` is expressed in ticks and has the usual meaning: after
`dt == half_life`, activity is halved (since 2**(-1) == 0.5).

Finally we *retain* the event if the updated value is above a threshold:

  keep = (value > noise_threshold)

Important detail (matches dv-processing)
--------------------------------------
State is updated regardless of keep/drop. This prevents "never-seed" deadlocks
and makes the filter stable in long runs.

Parameter mapping (Qt-aligned config)
-------------------------------------
This project intentionally keeps a small shared parameter set (like the Qt UI).
To avoid introducing new CLI flags, we reuse those shared fields:
- cfg.time_window_us  -> half-life (microseconds)
- cfg.radius_px       -> subdivision_factor
- cfg.min_neighbors   -> noise_threshold
"""

from dataclasses import dataclass

import numpy as np

from ...timebase import TimeBase
from ..types import DenoiseConfig
from .base import Dims


@dataclass
class FastDecayOp:
    name: str = "fastdecay"

    def __init__(self, dims: Dims, cfg: DenoiseConfig, tb: TimeBase):
        self.name = "fastdecay"
        self.dims = dims
        self.cfg = cfg
        self.tb = tb

        # Reuse Qt-shared params (see module docstring).
        #radius_px 在 FastDecayOp 中被复用为 subdivision_factor，不是传统意义上的 neighborhood radius
        # 网格划分因子：控制空间聚合粒度
        self.subdivision_factor = max(1, int(cfg.radius_px))
        # 半衰期（指数衰减的时间常数）
        self.half_life_ticks = int(tb.us_to_ticks(int(cfg.time_window_us)))
        if self.half_life_ticks < 1:
            raise ValueError("FastDecayOp requires time_window_us >= 1us (half-life must be > 0).")
        # 噪声阈值：只有计数值超过此值才被认为是信号
        self.noise_threshold = float(cfg.min_neighbors) # reused as decay-map activity threshold

        # +1 matches dv-processing: integer division can produce index == (w//f)
        # for pixels near the boundary when w is not divisible by f.
        low_w = (int(dims.width) // self.subdivision_factor) + 1
        low_h = (int(dims.height) // self.subdivision_factor) + 1

        self.low_w = int(low_w)
        self.low_h = int(low_h)

        n = self.low_w * self.low_h
        # 存储每个网格单元的状态
        self.last_ts = np.zeros((n,), dtype=np.uint64)
        self.value = np.zeros((n,), dtype=np.float32)

    def accept(self, x: int, y: int, p: int, t: int) -> bool:  # p unused
        # 1. 计算网格索引
        cx = int(x) // self.subdivision_factor
        cy = int(y) // self.subdivision_factor
        idx0 = (cy * self.low_w) + cx

        last = int(self.last_ts[idx0])
        dt = int(t) - last

        if dt <= 0:
            decay_mult = 1.0
        else:
            # 指数衰减公式：衰减系数 = 2^(-Δt/半衰期)
            decay_mult = 2.0 ** (-float(dt) / float(self.half_life_ticks))

        # Update state unconditionally (dv-processing behaviour)
        self.last_ts[idx0] = np.uint64(t)

        new_val = (float(self.value[idx0]) * float(decay_mult)) + 1.0
        self.value[idx0] = np.float32(new_val)

        return new_val > self.noise_threshold
