from __future__ import annotations

"""EBF_optimized 变体 V6：purity 归一化（同极性占比）+ V2 的 soft-weight + 线性 scale。

动机
----
V4/V4.1 与 V5 的预筛结果表明：
- 把 local(block) 活动率用于 scale（归一化尺度）会误伤真实运动/边缘（AUC 单调负收益）。
- 把异极性当作统一负证据直接扣进 raw score（same-gamma*opp）同样会系统性伤害排序。

因此 V6 尝试一种更“温和”的用法：
- 不改 scale（仍用 V2 的全局噪声率 + 线性 scale）。
- raw score 仍以同极性累加为主（same_sum）。
- 同时统计异极性邻居贡献（opp_sum），但不作为负证据扣分。
- 用 purity 因子对归一化分数做置信度调制：

    score_norm = (same_sum / scale_global) * purity
    purity = same_sum / (same_sum + opp_sum + eps)

其中 eps 是一个很小的常数，避免除零。

实现/硬件友好性
---------------
- 仍然是固定半径邻域双重循环。
- 相比 V2，只多维护一条 opp_sum（加法 + 一次除法）。

注意
----
- purity 只会把“同极性不够一致”的事件分数压低，不会像 same-opp 那样把边缘结构整体拉低。
- 噪声率估计的 soft-weight 仍使用 same_sum（保持与 V2 的一致性）。
"""

import numpy as np

from .v2_softw_linear import EbfOptV2SoftWLinear


class EbfOptV6SoftWLinearPurity(EbfOptV2SoftWLinear):
    """V6：purity 归一化（same/(same+opp)）调制 score_norm。"""

    variant_id: str = "softw_linear_purity"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._purity_eps: float = 1e-6

    def score_norm(self, x: int, y: int, p: int, t: int) -> float:
        r = max(0, min(int(self.cfg.radius_px), 8))
        tau_ticks = int(self.tb.us_to_ticks(int(self.cfg.time_window_us)))

        p01 = 1 if p > 0 else -1
        idx0 = self.dims.idx(int(x), int(y))

        # pass-through（但依旧要更新状态）
        if r <= 0 or tau_ticks <= 0:
            self.last_ts[idx0] = np.uint64(t)
            self.last_pol[idx0] = np.int8(p01)
            return float("inf")

        inv_tau = 1.0 / float(tau_ticks)

        same_sum = 0.0
        opp_sum = 0.0

        y0 = max(0, int(y) - r)
        y1 = min(self.dims.height - 1, int(y) + r)
        x0 = max(0, int(x) - r)
        x1 = min(self.dims.width - 1, int(x) + r)

        for yy in range(y0, y1 + 1):
            base = yy * self.dims.width
            for xx in range(x0, x1 + 1):
                if xx == x and yy == y:
                    continue

                idx = base + xx
                pol = int(self.last_pol[idx])
                if pol == 0:
                    continue

                ts = int(self.last_ts[idx])
                if ts == 0:
                    continue

                dt = (t - ts) if t >= ts else (ts - t)
                if dt > tau_ticks:
                    continue

                aw = float(tau_ticks - dt) * inv_tau

                if pol == p01:
                    same_sum += aw
                elif pol == -p01:
                    opp_sum += aw

        self.last_ts[idx0] = np.uint64(t)
        self.last_pol[idx0] = np.int8(p01)

        if not np.isfinite(same_sum):
            return float(same_sum)

        # global noise proxy update uses same_sum (keep V2 behavior)
        w = float(self._noise_weight(float(same_sum)))
        self._update_global_rate(int(t), weight=w)

        scale = float(self._expected_noise_score_scale(tau_ticks=tau_ticks, r=r))
        base = float(same_sum) / float(scale)

        denom = float(same_sum + opp_sum + float(self._purity_eps))
        purity = float(same_sum) / denom if denom > 0.0 else 0.0
        if purity < 0.0:
            purity = 0.0
        if purity > 1.0:
            purity = 1.0

        return float(base * purity)
