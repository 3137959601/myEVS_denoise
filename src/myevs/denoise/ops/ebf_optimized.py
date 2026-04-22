from __future__ import annotations

"""EBF_optimized — EBF with global adaptive noise normalization.

Goal
----
Keep the original EBF scoring (Guo 2025) but reduce threshold drift across
noise conditions by normalizing the raw score using a *global* online estimate
of event rate (noise proxy).

Design (minimal + modular)
--------------------------
- Raw score S(x,y): identical to :class:`~myevs.denoise.ops.ebf.EbfOp`.
- Global event-rate estimator: EMA of 1/Δt over the incoming stream.
- Expected-noise score scale: derived from global per-pixel rate, tau window,
  and neighborhood size.
- Decision: keep if S_norm > threshold (reuses cfg.min_neighbors).

Notes
-----
This is intentionally conservative:
- No block-level statistics yet (that's the next step after validating global).
- No extra config fields yet; it reuses existing DenoiseConfig knobs:
  - cfg.time_window_us -> tau
  - cfg.radius_px      -> r
  - cfg.min_neighbors  -> threshold on normalized score (dimensionless)

If you later want more control, we can extend DenoiseConfig or parse method
variants (e.g. 'ebf_optimized:alpha=0.01').
"""

from dataclasses import dataclass
import os

from ...timebase import TimeBase
from ..types import DenoiseConfig
from .base import Dims
from .ebfopt_variants import create_ebfopt_variant, list_variants


@dataclass
class EbfOptimizedOp:
    """兼容入口：内部委托给某个 EBF_optimized 变体实现。

    说明
    ----
    你希望把不同改进算法拆到不同文件里以便对比，但工程中已有 pipeline/CLI
    通过 `EbfOptimizedOp` 导入/实例化。

    因此这里保留 `EbfOptimizedOp` 作为稳定入口：
    - 默认使用最新主线变体（当前为 `softw_recent`）
    - 可用环境变量快速切换：
      - `MYEVS_EBFOPT_VARIANT=equalw_linear|softw_linear|softw_recent`
      - 兼容旧开关：`MYEVS_EBFOPT_SCALE_MODEL=linear|recent`
        （仅在未指定 MYEVS_EBFOPT_VARIANT 时生效）
    """

    name: str = "ebf_optimized"

    def __init__(self, dims: Dims, cfg: DenoiseConfig, tb: TimeBase):
        self.name = "ebf_optimized"
        self.dims = dims
        self.cfg = cfg
        self.tb = tb

        variant = str(os.environ.get("MYEVS_EBFOPT_VARIANT", "")).strip().lower()
        if not variant:
            # 兼容旧逻辑：只区分 scale_model（噪声率仍用 soft-weight）
            scale_model = str(os.environ.get("MYEVS_EBFOPT_SCALE_MODEL", "recent")).strip().lower()
            variant = "softw_linear" if scale_model == "linear" else "softw_recent"

        try:
            self._impl = create_ebfopt_variant(variant, dims, cfg, tb)
        except KeyError as e:
            choices = ", ".join(list_variants())
            raise KeyError(f"unknown MYEVS_EBFOPT_VARIANT={variant!r}. choices=[{choices}]") from e

        # 让外部仍可访问状态数组（有些实验脚本可能会用到）。
        self.last_ts = self._impl.last_ts
        self.last_pol = self._impl.last_pol

    def score_raw(self, x: int, y: int, p: int, t: int) -> float:
        return float(self._impl.score_raw(x, y, p, t))

    def score_norm(self, x: int, y: int, p: int, t: int) -> float:
        return float(self._impl.score_norm(x, y, p, t))

    def accept(self, x: int, y: int, p: int, t: int) -> bool:
        return bool(self._impl.accept(x, y, p, t))
