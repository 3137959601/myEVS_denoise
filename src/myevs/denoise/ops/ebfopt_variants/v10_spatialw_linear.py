from __future__ import annotations

"""EBF 变体 V10：加入空间距离权重（线性衰减）。

该文件用于把 V10 从评测脚本中解耦出来，便于复用与维护。

V10 定义：
- 仍使用 baseline EBF 的时间线性衰减 w_age = clip((tau-|dt|)/tau, 0, 1)
- 仍使用同极性门控 w_pol
- 新增空间权重 w_sp = max(0, 1 - sqrt(dx^2+dy^2)/r)
- score = sum(w_age * w_pol * w_sp)

注意：不做归一化，因此分数尺度会改变，阈值会漂移（ROC sweep 会覆盖）。
"""

from dataclasses import dataclass

import numpy as np

from ....timebase import TimeBase
from ...types import DenoiseConfig
from ..base import Dims


def build_spatial_lut(radius_px: int) -> np.ndarray:
    """Build a (2r+1, 2r+1) spatial weight LUT for V10.

    w(dx,dy) = max(0, 1 - sqrt(dx^2+dy^2)/r)

    Center weight is 0 by convention (center pixel is excluded anyway).
    """

    r = int(radius_px)
    if r <= 0:
        return np.zeros((1, 1), dtype=np.float32)

    size = 2 * r + 1
    lut = np.zeros((size, size), dtype=np.float32)
    for iy in range(size):
        dy = iy - r
        for ix in range(size):
            dx = ix - r
            if dx == 0 and dy == 0:
                lut[iy, ix] = 0.0
                continue
            d = float((dx * dx + dy * dy) ** 0.5)
            w = 1.0 - (d / float(r))
            if w < 0.0:
                w = 0.0
            lut[iy, ix] = float(w)
    return lut


def try_build_v10_spatialw_linear_scores_kernel():
    """Build and return Numba kernel for V10 score streaming.

    Returns None if numba is unavailable.

    Kernel signature:
        (t,x,y,p,width,height,radius_px,tau_ticks,spatial_lut,last_ts,last_pol,scores_out) -> None
    """

    try:
        from numba import njit  # type: ignore
    except Exception:
        return None

    @njit(cache=True)
    def ebf_v10_scores_stream(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        width: int,
        height: int,
        radius_px: int,
        tau_ticks: int,
        spatial_lut: np.ndarray,
        last_ts: np.ndarray,
        last_pol: np.ndarray,
        scores_out: np.ndarray,
    ) -> None:
        n = int(t.shape[0])
        w = int(width)
        h = int(height)

        rr = int(radius_px)
        if rr < 0:
            rr = 0
        if rr > 8:
            rr = 8

        tau = int(tau_ticks)

        if rr <= 0 or tau <= 0:
            for i in range(n):
                xi = int(x[i])
                yi = int(y[i])
                if xi < 0 or xi >= w or yi < 0 or yi >= h:
                    scores_out[i] = 0.0
                    continue

                ti = int(t[i])
                pi = 1 if int(p[i]) > 0 else -1

                idx0 = yi * w + xi
                last_ts[idx0] = np.uint64(ti)
                last_pol[idx0] = np.int8(pi)
                scores_out[i] = np.inf
            return

        inv_tau = 1.0 / float(tau)

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            if xi < 0 or xi >= w or yi < 0 or yi >= h:
                scores_out[i] = 0.0
                continue

            ti = int(t[i])
            pi = 1 if int(p[i]) > 0 else -1

            idx0 = yi * w + xi
            score = 0.0

            y0 = yi - rr
            if y0 < 0:
                y0 = 0
            y1 = yi + rr
            if y1 >= h:
                y1 = h - 1

            x0 = xi - rr
            if x0 < 0:
                x0 = 0
            x1 = xi + rr
            if x1 >= w:
                x1 = w - 1

            for yy in range(y0, y1 + 1):
                base = yy * w
                oy = yy - yi + rr
                for xx in range(x0, x1 + 1):
                    if xx == xi and yy == yi:
                        continue

                    idx = base + xx

                    if int(last_pol[idx]) != pi:
                        continue

                    ts = int(last_ts[idx])
                    if ts == 0:
                        continue

                    dt = (ti - ts) if ti >= ts else (ts - ti)
                    if dt > tau:
                        continue

                    ox = xx - xi + rr
                    w_sp = float(spatial_lut[oy, ox])
                    if w_sp <= 0.0:
                        continue

                    score += (float(tau - dt) * inv_tau) * w_sp

            last_ts[idx0] = np.uint64(ti)
            last_pol[idx0] = np.int8(pi)
            scores_out[i] = score

    return ebf_v10_scores_stream


@dataclass
class EbfV10SpatialWLinearOp:
    name: str = "ebf_v10"

    def __init__(self, dims: Dims, cfg: DenoiseConfig, tb: TimeBase):
        self.name = "ebf_v10"
        self.dims = dims
        self.cfg = cfg
        self.tb = tb

        n = int(dims.width) * int(dims.height)
        self.last_ts = np.zeros((n,), dtype=np.uint64)
        self.last_pol = np.zeros((n,), dtype=np.int8)  # {-1, 0, +1}

        r = max(0, min(int(self.cfg.radius_px), 8))
        self._r = int(r)
        self._spatial_lut = build_spatial_lut(int(r))

    def score(self, x: int, y: int, p: int, t: int) -> float:
        r = int(self._r)
        tau_ticks = int(self.tb.us_to_ticks(int(self.cfg.time_window_us)))

        p01 = 1 if p > 0 else -1
        idx0 = self.dims.idx(x, y)

        # Pass-through, but still update state.
        if r <= 0 or tau_ticks <= 0:
            self.last_ts[idx0] = np.uint64(t)
            self.last_pol[idx0] = np.int8(p01)
            return float("inf")

        inv_tau = 1.0 / float(tau_ticks)
        lut = self._spatial_lut

        score = 0.0
        y0 = max(0, y - r)
        y1 = min(self.dims.height - 1, y + r)
        x0 = max(0, x - r)
        x1 = min(self.dims.width - 1, x + r)

        for yy in range(y0, y1 + 1):
            base = yy * self.dims.width
            oy = yy - y + r
            for xx in range(x0, x1 + 1):
                if xx == x and yy == y:
                    continue

                idx = base + xx

                if int(self.last_pol[idx]) != p01:
                    continue

                ts = int(self.last_ts[idx])
                if ts == 0:
                    continue

                dt = (t - ts) if t >= ts else (ts - t)
                if dt > tau_ticks:
                    continue

                ox = xx - x + r
                w_sp = float(lut[oy, ox])
                if w_sp <= 0.0:
                    continue

                w_age = float(tau_ticks - dt) * inv_tau
                score += w_age * w_sp

        self.last_ts[idx0] = np.uint64(t)
        self.last_pol[idx0] = np.int8(p01)

        return float(score)

    def accept(self, x: int, y: int, p: int, t: int) -> bool:
        thr = float(self.cfg.min_neighbors)
        return self.score(x, y, p, t) > thr
