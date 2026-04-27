from __future__ import annotations

"""PFD (Polarity-Focused Denoising).

Reference implementation adapted from:
- D:/hjx_workspace/scientific_reserach/PFD/PFDs.cpp

This op follows the event-by-event version (PFDs) and keeps the two-stage logic:
1) Stage-1: same-polarity temporal support in local neighborhood.
2) Stage-2: polarity-flip consistency score in local neighborhood.

Parameter mapping in this project:
- time_window_us  -> delta_t0 (stage-1) and delta_t (stage-2), same value.
- radius_px       -> local neighborhood radius r (PFD repo uses fixed 1).
- min_neighbors   -> stage-2 neighbor activity threshold (keep if neighbors_count > min_neighbors).
- refractory_us   -> stage-1 minimum support count var (keep stage-1 if support >= var).

Notes:
- We implement PFD-A style score: |cur_flip - neigh_flip_mean| <= 1.
- Polarity is normalized to {-1, +1}.
- pfd_mode="b" enables PFD-B style score: |neigh_flip_mean| <= 1.
"""

from dataclasses import dataclass

import numpy as np

from ...timebase import TimeBase
from ..types import DenoiseConfig
from .base import Dims


@dataclass
class PfdOp:
    name: str = "pfd"

    def __init__(self, dims: Dims, cfg: DenoiseConfig, tb: TimeBase):
        self.name = "pfd"
        self.dims = dims
        self.cfg = cfg
        self.tb = tb

        n = int(dims.width) * int(dims.height)

        # Stage-1 polarity-specific last timestamp maps.
        self.last_on = np.zeros((n,), dtype=np.uint64)
        self.last_off = np.zeros((n,), dtype=np.uint64)

        # Latest polarity and latest event timestamp per pixel.
        self.last_pol = np.zeros((n,), dtype=np.int8)
        self.last_evt = np.zeros((n,), dtype=np.uint64)

        # Per-pixel FIFO of polarity-flip timestamps.
        # Fixed size matches upstream default (fifoSize=5).
        self._fifo_size = 5
        self.flip_buf = np.zeros((n, self._fifo_size), dtype=np.uint64)
        self.flip_head = np.zeros((n,), dtype=np.int32)
        self.flip_count = np.zeros((n,), dtype=np.int32)

    def _push_flip(self, idx: int, t: int) -> None:
        c = int(self.flip_count[idx])
        h = int(self.flip_head[idx])
        if c < self._fifo_size:
            pos = (h + c) % self._fifo_size
            self.flip_buf[idx, pos] = np.uint64(t)
            self.flip_count[idx] = np.int32(c + 1)
            return
        # overwrite oldest, then move head
        self.flip_buf[idx, h] = np.uint64(t)
        self.flip_head[idx] = np.int32((h + 1) % self._fifo_size)

    def _count_recent_flips(self, idx: int, t: int, win_ticks: int) -> int:
        c = int(self.flip_count[idx])
        if c <= 0:
            return 0
        h = int(self.flip_head[idx])
        cnt = 0
        for k in range(c):
            pos = (h + k) % self._fifo_size
            ft = int(self.flip_buf[idx, pos])
            if ft == 0:
                continue
            dt = t - ft
            if dt < 0:
                dt = -dt
            if dt <= win_ticks:
                cnt += 1
        return cnt

    def accept(self, x: int, y: int, p: int, t: int) -> bool:
        w = int(self.dims.width)
        h = int(self.dims.height)
        idx0 = int(y) * w + int(x)

        pol = 1 if int(p) > 0 else -1
        lastp = int(self.last_pol[idx0])
        if lastp != 0 and lastp != pol:
            self._push_flip(idx0, t)
        self.last_pol[idx0] = np.int8(pol)
        self.last_evt[idx0] = np.uint64(t)

        r = max(1, min(int(self.cfg.radius_px), 8))
        win_ticks = int(self.tb.us_to_ticks(int(self.cfg.time_window_us)))
        if win_ticks <= 0:
            win_ticks = 1

        # stage-1 threshold (var in upstream code). Keep practical bounded range.
        stage1_var = int(self.cfg.refractory_us)
        if stage1_var <= 0:
            stage1_var = 1
        if stage1_var > (2 * r + 1) * (2 * r + 1) - 1:
            stage1_var = (2 * r + 1) * (2 * r + 1) - 1

        # stage-2 neighbor threshold (neibor in upstream code).
        neigh_thr = float(self.cfg.min_neighbors)

        # Stage-1: same polarity temporal support.
        # Update self timestamp first to match upstream ordering.
        if pol > 0:
            self.last_on[idx0] = np.uint64(t)
            same = self.last_on
        else:
            self.last_off[idx0] = np.uint64(t)
            same = self.last_off

        support = 0
        y0 = max(0, y - r)
        y1 = min(h - 1, y + r)
        x0 = max(0, x - r)
        x1 = min(w - 1, x + r)
        for yy in range(y0, y1 + 1):
            base = yy * w
            for xx in range(x0, x1 + 1):
                if xx == x and yy == y:
                    continue
                ts = int(same[base + xx])
                if ts == 0:
                    continue
                dt = t - ts
                if dt < 0:
                    dt = -dt
                if dt < win_ticks:
                    support += 1
        first_ok = support >= stage1_var
        if not first_ok:
            return False

        # Stage-2: polarity-flip consistency.
        cur_flip = self._count_recent_flips(idx0, t, win_ticks)
        neigh_active = 0
        neigh_flip_sum = 0

        for yy in range(y0, y1 + 1):
            base = yy * w
            for xx in range(x0, x1 + 1):
                if xx == x and yy == y:
                    continue
                idx = base + xx
                tev = int(self.last_evt[idx])
                if tev != 0:
                    dt = t - tev
                    if dt < 0:
                        dt = -dt
                    if dt <= win_ticks:
                        neigh_active += 1
                neigh_flip_sum += self._count_recent_flips(idx, t, win_ticks)

        if neigh_active <= neigh_thr:
            return False

        neigh_flip_mean = float(neigh_flip_sum) / float(neigh_active)
        mode = str(getattr(self.cfg, "pfd_mode", "a") or "a").strip().lower()
        if mode == "b":
            score = abs(neigh_flip_mean)
        else:
            score = abs(float(cur_flip) - neigh_flip_mean)
        return score <= 1.0
