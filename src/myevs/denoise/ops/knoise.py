from __future__ import annotations

"""Knoise (Khodamoradi noise filter).

Reference:
- cuke-emlb/include/denoisors/khodamoradi_noise.hpp

Core idea:
- Keep the latest event on each x-column and each y-row.
- For a new event, count short-term same-polarity support from neighboring
  columns/rows with local adjacency constraints.
- Keep if support >= threshold.
"""

from dataclasses import dataclass

import numpy as np

from ...timebase import TimeBase
from ..types import DenoiseConfig
from .base import Dims


@dataclass
class KnoiseOp:
    name: str = "knoise"

    def __init__(self, dims: Dims, cfg: DenoiseConfig, tb: TimeBase):
        self.name = "knoise"
        self.dims = dims
        self.cfg = cfg
        self.tb = tb

        w = int(dims.width)
        h = int(dims.height)

        # Per-column latest event (timestamp, y, polarity)
        self.x_ts = np.zeros((w,), dtype=np.uint64)
        self.x_y = np.zeros((w,), dtype=np.int32)
        self.x_p = np.zeros((w,), dtype=np.int8)

        # Per-row latest event (timestamp, x, polarity)
        self.y_ts = np.zeros((h,), dtype=np.uint64)
        self.y_x = np.zeros((h,), dtype=np.int32)
        self.y_p = np.zeros((h,), dtype=np.int8)

    def _close_in_time_and_pol(self, ts: int, pol: int, t: int, p: int, win: int) -> bool:
        if ts == 0:
            return False
        if pol != p:
            return False
        # Keep behavior stable under wrapped/unwrapped timestamps.
        dt = (t - ts) if t >= ts else (ts - t)
        return dt <= win

    def accept(self, x: int, y: int, p: int, t: int) -> bool:
        need = max(0, int(self.cfg.min_neighbors))
        win_ticks = int(self.tb.us_to_ticks(int(self.cfg.time_window_us)))

        if win_ticks <= 0:
            # Still update state to avoid deadlocks.
            self.x_ts[x] = np.uint64(t)
            self.x_y[x] = np.int32(y)
            self.x_p[x] = np.int8(1 if p > 0 else -1)
            self.y_ts[y] = np.uint64(t)
            self.y_x[y] = np.int32(x)
            self.y_p[y] = np.int8(1 if p > 0 else -1)
            return need <= 0

        pp = 1 if p > 0 else -1
        w = int(self.dims.width)
        h = int(self.dims.height)
        support = 0

        x_minus = x > 0
        x_plus = x < (w - 1)
        y_minus = y > 0
        y_plus = y < (h - 1)

        # x-1 column
        if x_minus:
            xx = x - 1
            if self._close_in_time_and_pol(int(self.x_ts[xx]), int(self.x_p[xx]), t, pp, win_ticks):
                yy = int(self.x_y[xx])
                if (y_minus and yy == y - 1) or (yy == y) or (y_plus and yy == y + 1):
                    support += 1

        # x column
        if self._close_in_time_and_pol(int(self.x_ts[x]), int(self.x_p[x]), t, pp, win_ticks):
            yy = int(self.x_y[x])
            if (y_minus and yy == y - 1) or (y_plus and yy == y + 1):
                support += 1

        # x+1 column
        if x_plus:
            xx = x + 1
            if self._close_in_time_and_pol(int(self.x_ts[xx]), int(self.x_p[xx]), t, pp, win_ticks):
                yy = int(self.x_y[xx])
                if (y_minus and yy == y - 1) or (yy == y) or (y_plus and yy == y + 1):
                    support += 1

        # y-1 row
        if y_minus:
            yy = y - 1
            if self._close_in_time_and_pol(int(self.y_ts[yy]), int(self.y_p[yy]), t, pp, win_ticks):
                xx = int(self.y_x[yy])
                if (x_minus and xx == x - 1) or (xx == x) or (x_plus and xx == x + 1):
                    support += 1

        # y row
        if self._close_in_time_and_pol(int(self.y_ts[y]), int(self.y_p[y]), t, pp, win_ticks):
            xx = int(self.y_x[y])
            if (x_minus and xx == x - 1) or (x_plus and xx == x + 1):
                support += 1

        # y+1 row
        if y_plus:
            yy = y + 1
            if self._close_in_time_and_pol(int(self.y_ts[yy]), int(self.y_p[yy]), t, pp, win_ticks):
                xx = int(self.y_x[yy])
                if (x_minus and xx == x - 1) or (xx == x) or (x_plus and xx == x + 1):
                    support += 1

        # Update state after evaluation.
        self.x_ts[x] = np.uint64(t)
        self.x_y[x] = np.int32(y)
        self.x_p[x] = np.int8(pp)
        self.y_ts[y] = np.uint64(t)
        self.y_x[y] = np.int32(x)
        self.y_p[y] = np.int8(pp)

        return support >= need

