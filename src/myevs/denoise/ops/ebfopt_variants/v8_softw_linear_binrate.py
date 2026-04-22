from __future__ import annotations

"""EBF_optimized 变体 V8：定窗计数的全局 rate proxy（bin-count） + V2 soft-weight + 线性 scale。

目标
----
阈值稳定性优先，同时尽量不损伤 AUC。

动机
----
V2/V3 使用“相邻事件 dt 的 1/dt”做 EMA（再乘 soft-weight）。该估计对 dt 抖动很敏感，
尤其在 heavy / bursty 场景下方差较大；rate 直接进入 scale，会导致 score_norm 尺度抖动，
阈值在环境间更难对齐。

做法
----
- 维护一个固定最小时间窗 bin（默认 1ms）
- 在 bin 内累计加权事件数 sum_w（w 来自 V2 的 soft-weight）
- 当当前时间 t 与 bin_start 的间隔 >= bin_ticks 时：
  - rate_est = sum_w / dt  (events per tick)
  - 用 rate_est 更新 EMA（alpha 与基类一致 0.01）
  - 重置 bin

归一化 scale
-----------
仍然使用 V2 的线性 scale：scale_linear(ema_inv_dt, area, tau_ticks, neigh_px)。

保持不变
--------
- raw-score 与原始 EBF 一致
- soft-weight 函数与 V2 一致

备注
----
该变体同样会走 Python 实现（numba kernel 未覆盖）。
"""

import os

import numpy as np

from ._base import EbfOptVariantBase, scale_linear


class EbfOptV8SoftWLinearBinRate(EbfOptVariantBase):
    """定窗计数 rate proxy + V2 soft-weight + 线性 scale。"""

    variant_id: str = "softw_linear_binrate"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._noise_weight_k: float = 4.0
        self._noise_weight_min: float = 0.02

        # bin size: default 1000us, overridable for quick sweep/debug
        bin_us = int(os.environ.get("MYEVS_EBFOPT_BINRATE_US", "1000") or "1000")
        if bin_us <= 0:
            bin_us = 1000
        self._bin_ticks: int = int(self.tb.us_to_ticks(int(bin_us)))
        if self._bin_ticks <= 0:
            self._bin_ticks = 1

        self._bin_t0: int | None = None
        self._bin_sum_w: float = 0.0

    def _noise_weight(self, score_raw: float) -> float:
        s = float(score_raw)
        k = max(1e-6, float(self._noise_weight_k))
        w = 1.0 / (1.0 + (s / k))
        w = max(float(self._noise_weight_min), min(1.0, float(w)))
        return float(w)

    def _expected_noise_score_scale(self, *, tau_ticks: int, r: int) -> float:
        # 未使用：本变体在 score_norm 内直接算 scale。
        return 1.0

    def _update_bin_rate(self, *, t: int, w: float) -> None:
        ti = int(t)
        if self._bin_t0 is None:
            self._bin_t0 = ti
            ww0 = float(w)
            self._bin_sum_w = ww0 if (np.isfinite(ww0) and ww0 > 0.0) else 0.0
            return

        # accumulate weight
        ww = float(w)
        if np.isfinite(ww) and ww > 0.0:
            self._bin_sum_w += ww

        dt = ti - int(self._bin_t0)
        if dt < int(self._bin_ticks):
            return

        # Estimate rate over [t0, t]
        if dt <= 0:
            self._bin_t0 = ti
            self._bin_sum_w = ww if (np.isfinite(ww) and ww > 0.0) else 0.0
            return

        rate_est = float(self._bin_sum_w) / float(dt)

        if self._ema_inv_dt <= 0.0:
            self._ema_inv_dt = rate_est
        else:
            a = float(self._ema_alpha)
            self._ema_inv_dt = (1.0 - a) * self._ema_inv_dt + a * rate_est

        # reset bin
        self._bin_t0 = ti
        self._bin_sum_w = 0.0

    def score_norm(self, x: int, y: int, p: int, t: int) -> float:
        r = max(0, min(int(self.cfg.radius_px), 8))
        tau_ticks = int(self.tb.us_to_ticks(int(self.cfg.time_window_us)))

        s = float(self.score_raw(int(x), int(y), int(p), int(t)))
        if not np.isfinite(s):
            return s

        w = float(self._noise_weight(s))
        self._update_bin_rate(t=int(t), w=w)

        if self._ema_inv_dt <= 0.0 or tau_ticks <= 0 or r <= 0:
            return float(s)

        area = max(1, int(self.dims.width) * int(self.dims.height))
        neigh_px = int((2 * r + 1) ** 2 - 1)
        if neigh_px <= 0:
            return float(s)

        scale = float(scale_linear(ema_inv_dt=float(self._ema_inv_dt), area=area, tau_ticks=int(tau_ticks), neigh_px=int(neigh_px)))
        return float(self._normalize(float(s), float(scale)))
