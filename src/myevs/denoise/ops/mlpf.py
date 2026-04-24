from __future__ import annotations

"""MLPF operation.

Two modes:
1) Real TorchScript inference (if cfg.mlpf_model_path is provided).
2) Lightweight proxy fallback (no model path).

Reference-aligned feature layout (single event):
- channel 0 (recency): 1 - (t_now - t_last(x,y)) / duration
- channel 1 (polarity): constant (+1/-1) over the full patch
flattened to shape [2 * patch * patch].
"""

from dataclasses import dataclass
import math
from pathlib import Path

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

        # Keep patch configurable but bounded.
        p = int(getattr(cfg, "mlpf_patch", 7) or 7)
        if p < 3:
            p = 3
        if p % 2 == 0:
            p += 1
        self.patch = int(min(p, 21))
        self.radius = self.patch // 2
        self.area = self.patch * self.patch
        self.in_dim = 2 * self.area

        # Optional TorchScript model.
        self._torch = None
        self._model = None
        model_path = str(getattr(cfg, "mlpf_model_path", "") or "").strip()
        if model_path:
            pth = Path(model_path)
            if not pth.exists():
                raise FileNotFoundError(f"MLPF model file not found: {pth}")
            try:
                import torch  # type: ignore
            except Exception as e:  # pragma: no cover
                raise RuntimeError(f"mlpf_model_path is set but torch is unavailable: {type(e).__name__}: {e}")
            self._torch = torch
            self._model = torch.jit.load(str(pth), map_location="cpu")
            self._model.eval()

    def _idx(self, x: int, y: int) -> int:
        return y * int(self.dims.width) + x

    def _build_feature(self, x: int, y: int, p: int, t: int, win_ticks: int) -> np.ndarray:
        feat = np.zeros((self.in_dim,), dtype=np.float32)
        pp = 1.0 if p > 0 else -1.0
        inv_win = 1.0 / float(max(1, win_ticks))

        k = 0
        for dy in range(-self.radius, self.radius + 1):
            yy = y + dy
            for dx in range(-self.radius, self.radius + 1):
                xx = x + dx
                if 0 <= xx < int(self.dims.width) and 0 <= yy < int(self.dims.height):
                    ts = int(self.last_ts[self._idx(xx, yy)])
                    # Follow cuke-emlb style: no clipping here.
                    recency = 1.0 - float(t - ts) * inv_win
                    feat[k] = float(recency)
                    feat[k + self.area] = float(pp)
                k += 1
        return feat

    def accept(self, x: int, y: int, p: int, t: int) -> bool:
        win_ticks = int(self.tb.us_to_ticks(int(self.cfg.time_window_us)))
        thr = float(self.cfg.min_neighbors)

        idx0 = self._idx(x, y)
        if win_ticks <= 0:
            self.last_ts[idx0] = np.uint64(t)
            return thr <= 0.0

        feat = self._build_feature(x, y, p, t, win_ticks)

        # Always update time-surface after feature extraction.
        self.last_ts[idx0] = np.uint64(t)

        # Real model path: threshold applies on probability.
        if self._model is not None and self._torch is not None:
            torch = self._torch
            with torch.no_grad():
                inp = torch.from_numpy(feat).view(1, -1)
                out = self._model(inp)
                v = float(out.reshape(-1)[0].item())
                prob = v if (0.0 <= v <= 1.0) else (1.0 / (1.0 + math.exp(-v)))
            return prob >= thr

        # Proxy fallback: deterministic score from recency channel.
        # Keep behavior compatible with old threshold scale (roughly 0..30).
        score = 0.0
        rec = feat[: self.area]
        for v in rec:
            if v > 0.0:
                score += float(v)
        return score >= thr
