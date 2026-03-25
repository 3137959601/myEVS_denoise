from __future__ import annotations

"""EBF (Guo 2025) — Event-Based Filter (author code port).

This is a lightweight, deterministic EBF implementation based on the reference
script you pointed to:
    D:/hjx_workspace/scientific_reserach/EBF_Guo_2025/EBF/py/ebf1231retest.py

What the reference code does (essentials)
----------------------------------------
The script maintains two full-resolution maps:
- `timestampImage`: last event timestamp at each pixel
- `lastPolMap`: polarity of that last event

For each incoming event (x, y, p, t), it extracts a spatial neighbourhood patch
around (x, y), and for each neighbour pixel it computes:
- a *recency* weight (linear decay):

    w_age = clip((tau - |t - ts_nei|) / tau, 0, 1)

- a *polarity match* gate:

    w_pol = 1 if pol_nei == p else 0

Then EBF output is essentially the sum over the neighbourhood:

    score = sum(w_age * w_pol)

The current pixel is excluded from the neighbourhood sum.

We use the score for denoising:
- keep the event if `score > threshold`

State update rule
-----------------
Like the reference script, we update the time/polarity maps *regardless* of
keep/drop. This avoids "no seed" deadlocks and matches other ops' conventions.

Parameter mapping (Qt-aligned config)
-------------------------------------
To keep the CLI stable, we reuse the shared fields in `DenoiseConfig`:
- cfg.time_window_us  -> tau (linear decay window)
- cfg.radius_px       -> neighbourhood radius r (patch size = (2r+1)^2)
- cfg.min_neighbors   -> threshold on `score` (float)

Notes
-----
- The author script uses a 5x5 central patch in its evaluation ("TI25").
  In this implementation that corresponds to `radius_px=2`.
"""

from dataclasses import dataclass

import numpy as np

from ...timebase import TimeBase
from ..types import DenoiseConfig
from .base import Dims


@dataclass
class EbfOp:
    name: str = "ebf"

    def __init__(self, dims: Dims, cfg: DenoiseConfig, tb: TimeBase):
        self.name = "ebf"
        self.dims = dims
        self.cfg = cfg
        self.tb = tb

        n = int(dims.width) * int(dims.height)
        self.last_ts = np.zeros((n,), dtype=np.uint64)
        self.last_pol = np.zeros((n,), dtype=np.int8)  # {-1, 0, +1}

    def score(self, x: int, y: int, p: int, t: int) -> float:
        """Compute EBF score for an event and update internal state.

        This matches the reference implementation's behavior:
        - score is computed from neighborhood recency + same-polarity gating
        - state (last timestamp/polarity maps) is updated regardless of keep/drop
        """

        r = max(0, min(int(self.cfg.radius_px), 8))
        tau_ticks = int(self.tb.us_to_ticks(int(self.cfg.time_window_us)))

        p01 = 1 if p > 0 else -1

        idx0 = self.dims.idx(x, y)

        # Pass-through, but still update state.
        if r <= 0 or tau_ticks <= 0:
            self.last_ts[idx0] = np.uint64(t)
            self.last_pol[idx0] = np.int8(p01)
            return float("inf")

        inv_tau = 1.0 / float(tau_ticks)

        score = 0.0
        y0 = max(0, y - r)
        y1 = min(self.dims.height - 1, y + r)
        x0 = max(0, x - r)
        x1 = min(self.dims.width - 1, x + r)

        for yy in range(y0, y1 + 1):
            base = yy * self.dims.width
            for xx in range(x0, x1 + 1):
                if xx == x and yy == y:
                    continue

                idx = base + xx

                # Same-polarity gate (match reference's polarityweight clipping)
                if int(self.last_pol[idx]) != p01:
                    continue

                ts = int(self.last_ts[idx])
                if ts == 0:
                    continue

                dt = (t - ts) if t >= ts else (ts - t)
                if dt > tau_ticks:
                    continue
                # 分数贡献 = (τ - Δt) / τ
                score += (float(tau_ticks - dt) * inv_tau)

        # Always update self state (match reference)
        self.last_ts[idx0] = np.uint64(t)
        self.last_pol[idx0] = np.int8(p01)

        return float(score)

    def accept(self, x: int, y: int, p: int, t: int) -> bool:
        thr = float(self.cfg.min_neighbors)
        return self.score(x, y, p, t) > thr
