from __future__ import annotations

"""EvFlow (EventFlow filter).

Reference:
- cuke-emlb/include/denoisors/event_flow.hpp

Core idea:
- Keep a short temporal deque of recent events.
- In local radius, fit a plane t(x, y) = ax + by + c.
- Convert (a, b) to flow magnitude and keep events with small flow.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

from ...timebase import TimeBase
from ..types import DenoiseConfig
from .base import Dims


@dataclass
class EvFlowOp:
    name: str = "evflow"

    def __init__(self, dims: Dims, cfg: DenoiseConfig, tb: TimeBase):
        self.name = "evflow"
        self.dims = dims
        self.cfg = cfg
        self.tb = tb
        self.events: deque[tuple[int, int, int]] = deque()  # (x, y, t)

    def _fit_flow(self, x: int, y: int, t: int) -> float:
        r = max(1, min(int(self.cfg.radius_px), 8))
        # C++ reference needs more than 3 local events.
        rows: list[tuple[float, float, float]] = []
        b: list[float] = []
        for xx, yy, tt in self.events:
            if abs(x - xx) <= r and abs(y - yy) <= r:
                rows.append((float(xx), float(yy), 1.0))
                b.append(float(tt - t) * 1.0e-3)

        if len(rows) <= 3:
            return float("inf")

        a_mat = np.asarray(rows, dtype=np.float64)
        b_vec = np.asarray(b, dtype=np.float64)
        try:
            sol, _, _, _ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
        except np.linalg.LinAlgError:
            return float("inf")

        ax = float(sol[0])
        ay = float(sol[1])
        if abs(ax) < 1e-12 or abs(ay) < 1e-12:
            return float("inf")

        invx = -1.0 / ax
        invy = -1.0 / ay
        return float((invx * invx + invy * invy) ** 0.5)

    def accept(self, x: int, y: int, p: int, t: int) -> bool:  # p unused in reference
        win_ticks = int(self.tb.us_to_ticks(int(self.cfg.time_window_us)))
        thr = float(self.cfg.min_neighbors)

        flow = self._fit_flow(x, y, t)
        keep = flow <= thr

        if win_ticks > 0:
            while self.events:
                _, _, t0 = self.events[0]
                dt = (t - t0) if t >= t0 else (t0 - t)
                if dt >= win_ticks:
                    self.events.popleft()
                else:
                    break
        else:
            self.events.clear()

        self.events.append((x, y, t))
        return keep

