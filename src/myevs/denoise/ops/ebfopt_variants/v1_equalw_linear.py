from __future__ import annotations

"""EBF_optimized 变体 V1：等权全局事件率 + 线性 scale（baseline）。

功能
----
- 保持原始 EBF 的局部相关性打分不变。
- 全局噪声率 proxy：所有事件等权参与 EMA（不做信号/噪声区分）。
- scale：使用旧线性近似：

  scale = 0.25 * (lambda_pix * tau) * N_neigh

适用目的
--------
该版本主要用于对照：它展示了“只做全局归一化但不抑制 signal 污染”的效果上限/问题。

相较于上一版（原始 EBF）
--------------------------
- 新增：全局噪声率估计 + 归一化。
- 未新增：任何噪声筛选/降权机制（所以 light 场景容易把信号当噪声抬高 scale）。
"""

import numpy as np

from ._base import EbfOptVariantBase, scale_linear


class EbfOptV1EqualWLinear(EbfOptVariantBase):
    """等权噪声率 + 线性 scale。"""

    variant_id: str = "equalw_linear"

    def _noise_weight(self, score_raw: float) -> float:
        return 1.0

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
