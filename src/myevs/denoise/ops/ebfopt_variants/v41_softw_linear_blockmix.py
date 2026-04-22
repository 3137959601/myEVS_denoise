from __future__ import annotations

"""EBF_optimized 变体 V4.1：block/global 混合噪声代理（hybrid proxy）。

为什么要有 V4.1
--------------
V4（pure block-rate）虽然让阈值“看起来更稳定”，但实际 AUC / 固定阈值可迁移性变差。
主要原因是：block 内事件到达率不仅包含噪声，也包含目标/运动造成的真实事件密度。
如果 scale 只用 block-rate，会把“局部活动（包含信号）”误当成“局部噪声强”，从而把
score_norm 压扁，影响可分性。

V4.1 的思路
-----------
保持硬件友好 + 改动最小：
- 保持：V2 的 soft-weight（按 raw-score 降权更新噪声率）
- 保持：线性 scale（最简单、最易硬件实现）
- 改动：同时维护 global EMA 与 block EMA，并把 per-pixel rate 做线性混合：

    lambda_pix_eff = (1-β) * lambda_pix_global + β * lambda_pix_block

其中：
- lambda_pix_global = ema_inv_dt_global / (W*H)
- lambda_pix_block  = ema_inv_dt_block  / block_area

β 取一个固定小常数（默认 0.1），避免引入新配置项；后续如果需要再做参数化 sweep。

目标：
- 尽量保留 V2 的判别能力（global 为主）
- 同时让 block 级 proxy 作为“局部噪声修正项”（小比例介入）
"""

import numpy as np

import os

from .v2_softw_linear import EbfOptV2SoftWLinear


class EbfOptV41SoftWLinearBlockMix(EbfOptV2SoftWLinear):
    """V4.1：soft-weight + 线性 scale + block/global 混合噪声率。"""

    variant_id: str = "softw_linear_blockmix"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 固定 block 尺寸（硬件友好：2 的幂；避免引入新配置项）
        self._block_w: int = 32
        self._block_h: int = 32

        # 混合系数 beta：默认 0.1。
        # 为了不引入新配置项，这里只提供环境变量覆盖（便于 sweep 扫频）：
        #   MYEVS_EBFOPT_BLOCKMIX_BETA=0.05
        self._beta: float = 0.10
        beta_env = str(os.environ.get("MYEVS_EBFOPT_BLOCKMIX_BETA", "")).strip()
        if beta_env:
            try:
                b = float(beta_env)
                if np.isfinite(b):
                    self._beta = float(max(0.0, min(1.0, b)))
            except Exception:
                pass

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
        # raw EBF（包含 last_ts/last_pol 更新）
        r = max(0, min(int(self.cfg.radius_px), 8))
        tau_ticks = int(self.tb.us_to_ticks(int(self.cfg.time_window_us)))

        s = float(self.score_raw(int(x), int(y), int(p), int(t)))
        if not np.isfinite(s):
            return s

        # soft-weight（与 V2 一致）
        w = float(self._noise_weight(s))

        # 同时更新 global 与 block 的到达率 EMA
        self._update_global_rate(int(t), weight=w)
        bid = self._block_id(int(x), int(y))
        self._update_block_rate(int(bid), int(t), weight=w)

        # warm-up：global EMA 还没有形成时，退化为 raw-score
        ema_g = float(self._ema_inv_dt)
        if ema_g <= 0.0:
            return s

        if tau_ticks <= 0 or r <= 0:
            return s

        neigh_px = int((2 * r + 1) ** 2 - 1)
        if neigh_px <= 0:
            return s

        width = int(self.dims.width)
        height = int(self.dims.height)
        area_g = max(1, width * height)
        lambda_g = ema_g / float(area_g)

        ema_b = float(self._ema_inv_dt_block[bid])
        if ema_b > 0.0:
            area_b = max(1, int(self._block_area[bid]))
            lambda_b = ema_b / float(area_b)
            beta = float(self._beta)
            beta = 0.0 if (not np.isfinite(beta)) else max(0.0, min(1.0, beta))
            lambda_eff = (1.0 - beta) * lambda_g + beta * lambda_b
        else:
            lambda_eff = lambda_g

        m = float(lambda_eff) * float(tau_ticks)
        exp_score = float(neigh_px) * (m * 0.25)
        if exp_score <= 0.0 or (not np.isfinite(exp_score)):
            return s

        # 防止极端情况下除零
        exp_score = max(1e-6, float(exp_score))
        return s / exp_score
