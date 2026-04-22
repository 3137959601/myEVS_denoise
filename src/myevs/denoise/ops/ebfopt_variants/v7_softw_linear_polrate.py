from __future__ import annotations

"""EBF_optimized 变体 V7：按极性拆分全局噪声率 EMA（ON/OFF 两条）。

目标
----
优先提升阈值稳定性，同时尽量不损伤 AUC。

动机
----
raw-score 只统计“同极性邻居”的贡献，但 V2/V3 的全局噪声率 proxy 使用的是
所有事件混在一起的到达率（soft-weight 仅按 score 降权）。当不同噪声强度/电压下
ON/OFF 事件比例发生偏移时，用“混合 rate”去归一化“同极性 raw-score”会产生
系统偏差，导致 score_norm 的尺度与阈值在环境间漂移。

做法
----
- 保持 raw-score 与 V2 完全一致（不改排序信息）
- 噪声率 proxy 改为两条：
  - ema_inv_dt_on：仅由 ON 事件更新
  - ema_inv_dt_off：仅由 OFF 事件更新
- 归一化 scale 时，使用当前事件极性对应的 ema（ON 用 on，OFF 用 off）。

保持不变
--------
- soft-weight 噪声率更新权重仍使用 V2 的 w(score_raw)
- scale 模型仍使用 V2 的线性 scale（硬件友好）

备注
----
numba kernel 目前不覆盖该变体（会走 Python 实现），但逻辑简单，后续如有需要可再补。
"""

import numpy as np

from ._base import EbfOptVariantBase, scale_linear


class EbfOptV7SoftWLinearPolRate(EbfOptVariantBase):
    """V2 的 soft-weight + 线性 scale，但 rate proxy 按极性拆分。"""

    variant_id: str = "softw_linear_polrate"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 与 V2 保持一致
        self._noise_weight_k: float = 4.0
        self._noise_weight_min: float = 0.02

        self._last_t_on: int | None = None
        self._last_t_off: int | None = None
        self._ema_inv_dt_on: float = 0.0
        self._ema_inv_dt_off: float = 0.0

    def _noise_weight(self, score_raw: float) -> float:
        s = float(score_raw)
        k = max(1e-6, float(self._noise_weight_k))
        w = 1.0 / (1.0 + (s / k))
        w = max(float(self._noise_weight_min), min(1.0, float(w)))
        return float(w)

    def _update_rate_pol(self, *, t: int, p01: int, weight: float) -> None:
        if p01 > 0:
            last = self._last_t_on
            self._last_t_on = int(t)
        else:
            last = self._last_t_off
            self._last_t_off = int(t)

        if last is None:
            return

        dt = int(t) - int(last)
        if dt <= 0:
            return

        w = float(weight)
        if not np.isfinite(w) or w <= 0.0:
            return

        inv_dt = w / float(dt)
        a = float(self._ema_alpha)

        if p01 > 0:
            if self._ema_inv_dt_on <= 0.0:
                self._ema_inv_dt_on = inv_dt
            else:
                self._ema_inv_dt_on = (1.0 - a) * self._ema_inv_dt_on + a * inv_dt
        else:
            if self._ema_inv_dt_off <= 0.0:
                self._ema_inv_dt_off = inv_dt
            else:
                self._ema_inv_dt_off = (1.0 - a) * self._ema_inv_dt_off + a * inv_dt

    def _expected_noise_score_scale(self, *, tau_ticks: int, r: int) -> float:
        # 未使用：本变体在 score_norm 内部按极性选择 rate 并直接算 scale。
        return 1.0

    def score_norm(self, x: int, y: int, p: int, t: int) -> float:
        r = max(0, min(int(self.cfg.radius_px), 8))
        tau_ticks = int(self.tb.us_to_ticks(int(self.cfg.time_window_us)))

        s = float(self.score_raw(int(x), int(y), int(p), int(t)))
        if not np.isfinite(s):
            return s

        p01 = 1 if int(p) > 0 else -1
        w = float(self._noise_weight(s))
        self._update_rate_pol(t=int(t), p01=p01, weight=w)

        ema = float(self._ema_inv_dt_on) if p01 > 0 else float(self._ema_inv_dt_off)
        if ema <= 0.0 or tau_ticks <= 0 or r <= 0:
            return float(s)

        area = max(1, int(self.dims.width) * int(self.dims.height))
        neigh_px = int((2 * r + 1) ** 2 - 1)
        if neigh_px <= 0:
            return float(s)

        scale = float(scale_linear(ema_inv_dt=ema, area=area, tau_ticks=int(tau_ticks), neigh_px=int(neigh_px)))
        return float(self._normalize(float(s), float(scale)))
