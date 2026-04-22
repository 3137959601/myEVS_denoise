from __future__ import annotations

"""EBF_optimized 变体 V3：软权重全局噪声率 + 最近事件饱和 scale。

为什么需要这版
--------------
V2 已经解决了“噪声率估计被 signal 污染”的主要问题，但它的 scale 仍是线性近似：

  scale ≈ 0.25 * (lambda_pix * tau) * N_neigh

这个近似在 heavy/高到达率场景下会变得偏粗糙，因为我们的 EBF 实现本质是：
- 每个像素只保留最近一次事件（last_ts/last_pol）
- 并不是统计窗口内所有事件的累加

因此，单邻居的期望贡献应当具有“饱和”特性：到达率越大，越接近 0.5（同极性概率）
而不是无限线性增长。

scale（最近事件饱和模型）
-------------------------
设 m = lambda_pix * tau。若把每像素到达视作 Poisson：
- 最后一次事件年龄 A 的分布：f(a)=lambda*exp(-lambda*a)
- 线性时间权重贡献：E[(1-A/tau) * 1(A<=tau)] = 1 - (1-exp(-m))/m
- 再乘同极性概率 0.5

得到单邻居期望贡献：
  per_neigh = 0.5 * (1 - (1-exp(-m))/m)

整体 scale：
  scale = N_neigh * per_neigh

该式在 m→0 时一阶近似为 m/4（与 V2 线性式一致），m 大时自然饱和。

相较于 V2 的区别
-----------------
- 保持：噪声率 soft-weight 估计不变。
- 改动：scale 从线性近似换为最近事件饱和模型（更贴合实现）。
"""

import numpy as np

from ._base import EbfOptVariantBase, scale_recent_event


class EbfOptV3SoftWRecent(EbfOptVariantBase):
    """软权重噪声率 + 最近事件饱和 scale。"""

    variant_id: str = "softw_recent"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._noise_weight_k: float = 4.0
        self._noise_weight_min: float = 0.02

    def _noise_weight(self, score_raw: float) -> float:
        s = float(score_raw)
        k = max(1e-6, float(self._noise_weight_k))
        w = 1.0 / (1.0 + (s / k))
        w = max(float(self._noise_weight_min), min(1.0, float(w)))
        return float(w)

    def _expected_noise_score_scale(self, *, tau_ticks: int, r: int) -> float:
        if tau_ticks <= 0 or r <= 0:
            return 1.0
        if self._ema_inv_dt <= 0.0:
            return 1.0

        w = int(self.dims.width)
        h = int(self.dims.height)
        area = max(1, w * h)

        neigh_px = int((2 * r + 1) ** 2 - 1)
        if neigh_px <= 0:
            return 1.0

        return float(
            scale_recent_event(
                ema_inv_dt=float(self._ema_inv_dt),
                area=area,
                tau_ticks=int(tau_ticks),
                neigh_px=int(neigh_px),
            )
        )
