from __future__ import annotations

"""EBF_optimized 变体 V5：raw score 增强（same - gamma*opp）+ V2 的 soft-weight + 线性 scale。

动机
----
V4/V4.1 的实验已经比较清楚地表明：
- 把 local(block) 到达率用于 scale（归一化尺度）会把“真实运动/边缘活动”也当成噪声压扁。
- 这类方法往往在 AUC 上出现单调负收益。

因此 V5 换一条路径：
- 不改 scale（仍用 V2 的全局噪声率 + 线性 scale）。
- 只改 raw score 的判别力：

    S_raw = sum_same(Aw) - gamma * sum_opp(Aw)

其中：
- sum_same：同极性邻居的线性时间权重累加（与原始 EBF 相同）
- sum_opp：异极性邻居的线性时间权重累加
- gamma：异极性惩罚系数（>=0），gamma=0 退化为 V2

实现/硬件友好性
---------------
- 仍然是固定半径邻域的双重循环，只是把极性判断从“!= 就跳过”改成“opp 也累加到另一条和”。
- 额外代价：一个乘法与一次减法。

参数
----
- gamma 默认 0.5
- 可用环境变量覆盖：MYEVS_EBFOPT_OPP_GAMMA

注意
----
raw score 允许为负。
- 负分数在 accept 判定中自然更难通过。
- 但用于噪声率估计的 soft-weight 不应被负值破坏：这里对 weight 使用 max(0, score_raw)。
"""

import os

import numpy as np

from .v2_softw_linear import EbfOptV2SoftWLinear


class EbfOptV5SoftWLinearSameMinusOpp(EbfOptV2SoftWLinear):
    """V5：same - gamma*opp raw score + V2 的噪声率/scale。"""

    variant_id: str = "softw_linear_same_minus_opp"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._opp_gamma: float = 0.5
        raw = str(os.environ.get("MYEVS_EBFOPT_OPP_GAMMA", "")).strip()
        if raw:
            try:
                g = float(raw)
                if np.isfinite(g):
                    if g < 0.0:
                        g = 0.0
                    self._opp_gamma = float(g)
            except Exception:
                # ignore invalid env
                pass

    def _noise_weight(self, score_raw: float) -> float:
        # raw 允许为负；soft-weight 只对“高正分”做降权
        s = float(score_raw)
        if not np.isfinite(s):
            return 1.0
        if s < 0.0:
            s = 0.0
        return float(super()._noise_weight(float(s)))

    def score_raw(self, x: int, y: int, p: int, t: int) -> float:
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
        gamma = float(self._opp_gamma)

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

        return float(same_sum - gamma * opp_sum)
