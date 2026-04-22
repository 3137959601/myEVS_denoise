from __future__ import annotations

"""EBF_optimized 变体 V4：soft-weight 噪声率 + 线性 scale + block 级噪声代理。

目标（面向硬件部署）
--------------------
你希望：
- 结构尽可能简单（易于硬件实现/资源可控）
- 阈值更稳定（不同噪声条件/空间非均匀噪声下可比性更好）
- 精度基本不掉（尽量不改变 raw EBF 的判别特性）

因此 V4 的改动坚持“最小增量”：

- 保持不变：
  - raw score：与原始 EBF 相同
  - noise weight：与 V2 相同（w = 1/(1+s/k)，截断到 [w_min,1]）
  - scale：仍用线性近似（与 V2 相同）

- 唯一核心改动：
  - 把噪声率 proxy 从“全局一条 EMA”改成“block/tile 一条 EMA”。

直觉
----
全局 rate 在真实数据里很容易被空间非均匀噪声（热点/局部高频区域）或局部运动影响。
block 级 rate 仍然非常简单：
- 每个 block 只维护两个量：last_t_block、ema_inv_dt_block
- scale 只把 area 换成 block_area
但它能显著减轻“局部噪声强 → 全局 scale 被抬高 → 阈值漂移”的问题。

实现细节（当前默认）
--------------------
- block 大小固定为 32x32（硬件友好，且对 346x260 这种分辨率仍然合理）
- 边缘 block 面积按实际尺寸计算（不是简单用 32x32）

如果后续需要，我们再把 block 大小做成可配置项（但现在先保持最简单）。
"""

import numpy as np

from ._base import scale_linear
from .v2_softw_linear import EbfOptV2SoftWLinear


class EbfOptV4SoftWLinearBlock(EbfOptV2SoftWLinear):
    """V4：soft-weight + 线性 scale，但噪声率 proxy 采用 block 级 EMA。"""

    variant_id: str = "softw_linear_block"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 固定 block 尺寸（硬件友好：2 的幂；也避免引入新配置项）
        self._block_w: int = 32
        self._block_h: int = 32

        w = int(self.dims.width)
        h = int(self.dims.height)
        bw = int(self._block_w)
        bh = int(self._block_h)

        self._nbx: int = max(1, (w + bw - 1) // bw)
        self._nby: int = max(1, (h + bh - 1) // bh)
        nblocks = int(self._nbx * self._nby)

        self._last_t_block = np.full((nblocks,), -1, dtype=np.int64)
        self._ema_inv_dt_block = np.zeros((nblocks,), dtype=np.float64)
        self._block_area = np.zeros((nblocks,), dtype=np.int32)

        # 预计算每个 block 的实际面积（边缘块更小）
        for by in range(self._nby):
            y0 = by * bh
            y1 = min(h, (by + 1) * bh)
            hh = max(1, y1 - y0)
            for bx in range(self._nbx):
                x0 = bx * bw
                x1 = min(w, (bx + 1) * bw)
                ww = max(1, x1 - x0)
                bid = by * self._nbx + bx
                self._block_area[bid] = int(ww * hh)

    def _block_id(self, x: int, y: int) -> int:
        bx = int(x) // int(self._block_w)
        by = int(y) // int(self._block_h)
        if bx < 0:
            bx = 0
        if by < 0:
            by = 0
        if bx >= self._nbx:
            bx = self._nbx - 1
        if by >= self._nby:
            by = self._nby - 1
        return int(by * self._nbx + bx)

    def _update_block_rate(self, bid: int, t: int, *, weight: float) -> None:
        last = int(self._last_t_block[bid])
        self._last_t_block[bid] = int(t)
        if last < 0:
            return

        dt = int(t) - int(last)
        if dt <= 0:
            return

        w = float(weight)
        if not np.isfinite(w) or w <= 0.0:
            return

        inv_dt = w / float(dt)
        ema = float(self._ema_inv_dt_block[bid])
        if ema <= 0.0:
            self._ema_inv_dt_block[bid] = float(inv_dt)
        else:
            a = float(self._ema_alpha)
            self._ema_inv_dt_block[bid] = (1.0 - a) * ema + a * float(inv_dt)

    def score_norm(self, x: int, y: int, p: int, t: int) -> float:
        # 复用 base 的 raw EBF（包含 last_ts/last_pol 更新）
        r = max(0, min(int(self.cfg.radius_px), 8))
        tau_ticks = int(self.tb.us_to_ticks(int(self.cfg.time_window_us)))

        s = float(self.score_raw(int(x), int(y), int(p), int(t)))
        if not np.isfinite(s):
            return s

        # 仍使用 V2 的 soft-weight
        w = float(self._noise_weight(s))

        bid = self._block_id(int(x), int(y))
        self._update_block_rate(int(bid), int(t), weight=w)

        # warm-up：block EMA 尚未形成时，退化为 raw-score
        ema_inv_dt = float(self._ema_inv_dt_block[bid])
        if ema_inv_dt <= 0.0:
            return s

        if tau_ticks <= 0 or r <= 0:
            return s

        neigh_px = int((2 * r + 1) ** 2 - 1)
        if neigh_px <= 0:
            return s

        area = int(self._block_area[bid])
        scale = float(scale_linear(ema_inv_dt=ema_inv_dt, area=area, tau_ticks=int(tau_ticks), neigh_px=int(neigh_px)))
        if scale <= 0.0 or (not np.isfinite(scale)):
            return s

        return s / scale
