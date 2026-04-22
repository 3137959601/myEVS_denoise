from __future__ import annotations

"""EBF 改进算法（EBF_optimized）实验变体基类。

设计目标
--------
为了便于你对比不同改进思路，这里把“EBF 原始局部打分”与“全局噪声代理/归一化 scale”拆开：

- 局部打分 `score_raw()`：保持与原始 EBF 一致（同极性 + 线性时间衰减）。
- 归一化 `score_norm()`：由两个可替换部件决定：
  1) 噪声代理（全局事件率 EMA）：用 `_noise_weight(score_raw)` 决定某事件对全局噪声率估计的贡献。
  2) scale 模型：用 `_expected_noise_score_scale(...)` 估计“纯噪声下 raw-score 的期望尺度”。

这样每个实验变体只需要在独立文件中覆写 1~2 个函数即可，便于代码对照。

注意
----
- 这里的“全局噪声率”只是代理（proxy），不保证等于真实噪声率。
- 该模块主要服务 sweep/离线实验；pipeline/CLI 仍通过 `myevs.denoise.ops.ebf_optimized.EbfOptimizedOp` 兼容入口访问。
"""

import math
import os

import numpy as np

from ....timebase import TimeBase
from ...types import DenoiseConfig
from ..base import Dims


class EbfOptVariantBase:
    """EBF_optimized 变体基类。

    子类只需覆写：
    - `_noise_weight(score_raw: float) -> float`
    - `_expected_noise_score_scale(... ) -> float`

    其余（raw score、状态更新、accept 判定）保持一致。
    """

    name: str = "ebf_optimized"

    def __init__(self, dims: Dims, cfg: DenoiseConfig, tb: TimeBase):
        self.name = "ebf_optimized"
        self.dims = dims
        self.cfg = cfg
        self.tb = tb

        n = int(dims.width) * int(dims.height)
        self.last_ts = np.zeros((n,), dtype=np.uint64)
        self.last_pol = np.zeros((n,), dtype=np.int8)  # {-1, 0, +1}

        # 全局事件率（events per tick）EMA：用于噪声强度 proxy。
        self._last_t_global: int | None = None
        self._ema_inv_dt: float = 0.0
        self._ema_alpha: float = self._read_rate_ema_alpha()

        # Optional: adjust normalization strength via exponent alpha.
        # score_norm = score_raw / (scale ** alpha)
        # Default alpha=1.0 keeps legacy behavior.
        self._scale_alpha: float = self._read_scale_alpha()

    @staticmethod
    def _read_rate_ema_alpha() -> float:
        """Read global-rate EMA smoothing factor.

        Notes
        -----
        - This is NOT the scale exponent alpha.
        - Default keeps historical behavior: 0.01.

        Env
        ---
        - MYEVS_EBFOPT_RATE_EMA_ALPHA: float in (0, 1].
        """

        s = (os.environ.get("MYEVS_EBFOPT_RATE_EMA_ALPHA", "") or "").strip()
        if not s:
            return 0.01
        try:
            a = float(s)
        except Exception:
            return 0.01
        if not np.isfinite(a):
            return 0.01
        # clamp to a sane range
        a = max(1e-6, min(1.0, float(a)))
        return float(a)

    @staticmethod
    def _read_scale_alpha() -> float:
        s = (os.environ.get("MYEVS_EBFOPT_SCALE_ALPHA", "") or "").strip()
        if not s:
            return 1.0
        try:
            a = float(s)
        except Exception:
            return 1.0
        if not np.isfinite(a):
            return 1.0
        # allow 0 (means no normalization), but clamp to keep numeric sane
        a = max(0.0, min(4.0, float(a)))
        return float(a)

    def _normalize(self, score_raw: float, scale: float) -> float:
        """Apply normalization with optional exponent alpha."""

        s = float(score_raw)
        sc = float(scale)
        if not np.isfinite(s):
            return s
        if not np.isfinite(sc) or sc <= 0.0:
            return s
        a = float(self._scale_alpha)
        if abs(a - 1.0) <= 1e-12:
            return s / sc
        # handle a==0 -> scale^0 == 1
        return s / float(sc**a)

    # ---------------------- 全局噪声率估计 ----------------------
    def _update_global_rate(self, t: int, *, weight: float) -> None:
        """更新全局噪声率 EMA。

        解释：
        - 相邻两事件的全局时间间隔为 dtg（tick）。
        - 用 `weight/dtg` 作为“加权到达率”估计，weight 越小表示越不像噪声（更像信号）。

        这是我们用来减轻 signal 污染噪声率估计的关键接口。
        """

        last = self._last_t_global
        self._last_t_global = int(t)
        if last is None:
            return
        dt = int(t) - int(last)
        if dt <= 0:
            return

        w = float(weight)
        if not np.isfinite(w) or w <= 0.0:
            return

        inv_dt = w / float(dt)
        if self._ema_inv_dt <= 0.0:
            self._ema_inv_dt = inv_dt
        else:
            a = float(self._ema_alpha)
            self._ema_inv_dt = (1.0 - a) * self._ema_inv_dt + a * inv_dt

    def _noise_weight(self, score_raw: float) -> float:
        """返回该事件对噪声率估计的权重（默认：等权）。"""

        return 1.0

    # ---------------------- scale 估计（由子类决定） ----------------------
    def _expected_noise_score_scale(self, *, tau_ticks: int, r: int) -> float:
        raise NotImplementedError

    # ---------------------- 原始 EBF 打分（保持不变） ----------------------
    def score_raw(self, x: int, y: int, p: int, t: int) -> float:
        """原始 EBF 分数（与 EbfOp 逻辑一致）并更新状态。"""

        r = max(0, min(int(self.cfg.radius_px), 8))
        tau_ticks = int(self.tb.us_to_ticks(int(self.cfg.time_window_us)))

        p01 = 1 if p > 0 else -1
        idx0 = self.dims.idx(x, y)

        # pass-through（但依旧要更新状态）
        if r <= 0 or tau_ticks <= 0:
            self.last_ts[idx0] = np.uint64(t)
            self.last_pol[idx0] = np.int8(p01)
            return float("inf")

        inv_tau = 1.0 / float(tau_ticks)

        score = 0.0
        y0 = max(0, y - r)
        y1 = min(self.dims.height - 1, y + r)
        x0 = max(0, x - r)
        x1 = min(self.dims.width - 1, x + r)

        for yy in range(y0, y1 + 1):
            base = yy * self.dims.width
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
                score += (float(tau_ticks - dt) * inv_tau)

        self.last_ts[idx0] = np.uint64(t)
        self.last_pol[idx0] = np.int8(p01)

        return float(score)

    # ---------------------- 归一化分数与 accept ----------------------
    def score_norm(self, x: int, y: int, p: int, t: int) -> float:
        r = max(0, min(int(self.cfg.radius_px), 8))
        tau_ticks = int(self.tb.us_to_ticks(int(self.cfg.time_window_us)))

        s = float(self.score_raw(x, y, p, t))
        if not np.isfinite(s):
            return s

        w = float(self._noise_weight(s))
        self._update_global_rate(int(t), weight=w)

        scale = self._expected_noise_score_scale(tau_ticks=tau_ticks, r=r)
        return self._normalize(s, float(scale))

    def accept(self, x: int, y: int, p: int, t: int) -> bool:
        thr = float(self.cfg.min_neighbors)
        return self.score_norm(x, y, p, t) > thr


def scale_linear(*, ema_inv_dt: float, area: int, tau_ticks: int, neigh_px: int) -> float:
    """旧 scale：0.25 * (lambda_pix*tau) * neigh_px。"""

    if ema_inv_dt <= 0.0:
        return 1.0
    if tau_ticks <= 0 or neigh_px <= 0:
        return 1.0

    area2 = max(1, int(area))
    per_pixel_rate = float(ema_inv_dt) / float(area2)
    m = per_pixel_rate * float(tau_ticks)

    exp_score = float(neigh_px) * (m * 0.25)
    return max(1e-6, float(exp_score))


def scale_recent_event(*, ema_inv_dt: float, area: int, tau_ticks: int, neigh_px: int) -> float:
    """新 scale（最近事件饱和模型）。

    m = lambda_pix * tau
    per_neigh = 0.5 * (1 - (1-exp(-m))/m)

    小 m 时 per_neigh ~ m/4（与旧式一致）；m 大时自然饱和。
    """

    if ema_inv_dt <= 0.0:
        return 1.0
    if tau_ticks <= 0 or neigh_px <= 0:
        return 1.0

    area2 = max(1, int(area))
    per_pixel_rate = float(ema_inv_dt) / float(area2)
    m = per_pixel_rate * float(tau_ticks)

    if m <= 1e-6:
        per_neigh = m * 0.25
    else:
        per_neigh = 0.5 * (1.0 - ((1.0 - math.exp(-m)) / m))

    exp_score = float(neigh_px) * float(per_neigh)
    return max(1e-6, float(exp_score))
