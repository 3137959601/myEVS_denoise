from __future__ import annotations

"""EBF_optimized 变体 V2：软权重全局噪声率 + 线性 scale。

核心改动
--------
在 V1（等权）基础上，引入“按 raw-score 软降权”的噪声率估计：

- raw-score 越大，越像信号；它对全局噪声率 EMA 的贡献就越小。
- raw-score 越小，越像噪声；它更充分地参与噪声率估计。

权重函数（当前实现）
--------------------
w = 1 / (1 + s / k)
并截断到 [w_min, 1]，避免在极端情况下完全不更新。

保持不变
--------
- scale 仍使用 V1 的旧线性近似：0.25 * (lambda_pix*tau) * N_neigh

相较于 V1 的区别
-----------------
- 新增：噪声率估计的“信号降权”，减轻 light 场景 signal 污染。
- 未改变：scale 公式（仍是线性近似）。
"""

import numpy as np

from ._base import EbfOptVariantBase, scale_linear


class EbfOptV2SoftWLinear(EbfOptVariantBase):
    """软权重噪声率 + 线性 scale。"""

    variant_id: str = "softw_linear"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 这两个参数目前与 sweep 脚本保持一致，便于对照。
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

        return float(scale_linear(ema_inv_dt=float(self._ema_inv_dt), area=area, tau_ticks=int(tau_ticks), neigh_px=int(neigh_px)))
