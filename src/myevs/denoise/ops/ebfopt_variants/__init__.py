from __future__ import annotations

"""EBF_optimized 改进算法变体集合。

用法
----
- sweep/离线实验：通过 `create_ebfopt_variant(variant_id, ...)` 选择不同算法实现。
- pipeline/CLI：如果你仍在用 denoise pipeline，则会通过 `myevs.denoise.ops.ebf_optimized.EbfOptimizedOp` 入口（内部选择默认变体）。

为什么要这样拆
--------------
你要频繁对比不同改进思路时：
- 每个变体独立成一个文件，diff 清晰
- sweep 脚本统一入口，直接 `--variant xxx` 切换
- 每个变体内部用中文详注写明“它做了什么/相对上一版改了什么”
"""

from .v1_equalw_linear import EbfOptV1EqualWLinear
from .v2_softw_linear import EbfOptV2SoftWLinear
from .v3_softw_recent import EbfOptV3SoftWRecent
from .v4_softw_linear_block import EbfOptV4SoftWLinearBlock
from .v41_softw_linear_blockmix import EbfOptV41SoftWLinearBlockMix
from .v5_softw_linear_same_minus_opp import EbfOptV5SoftWLinearSameMinusOpp
from .v6_softw_linear_purity import EbfOptV6SoftWLinearPurity
from .v7_softw_linear_polrate import EbfOptV7SoftWLinearPolRate
from .v8_softw_linear_binrate import EbfOptV8SoftWLinearBinRate
from .v9_softw_linear_timeconst_rateema import EbfOptV9SoftWLinearTimeConstRateEma


# 说明：
# - “canonical id” 维持原先的可读命名（equalw_linear / softw_linear / softw_recent）。
# - 同时提供更像“版本号”的别名（EBFV1/2/3），便于实验记录与对照。

_CANONICAL = {
    EbfOptV1EqualWLinear.variant_id: EbfOptV1EqualWLinear,
    EbfOptV2SoftWLinear.variant_id: EbfOptV2SoftWLinear,
    EbfOptV3SoftWRecent.variant_id: EbfOptV3SoftWRecent,
    EbfOptV4SoftWLinearBlock.variant_id: EbfOptV4SoftWLinearBlock,
    EbfOptV41SoftWLinearBlockMix.variant_id: EbfOptV41SoftWLinearBlockMix,
    EbfOptV5SoftWLinearSameMinusOpp.variant_id: EbfOptV5SoftWLinearSameMinusOpp,
    EbfOptV6SoftWLinearPurity.variant_id: EbfOptV6SoftWLinearPurity,
    EbfOptV7SoftWLinearPolRate.variant_id: EbfOptV7SoftWLinearPolRate,
    EbfOptV8SoftWLinearBinRate.variant_id: EbfOptV8SoftWLinearBinRate,
    EbfOptV9SoftWLinearTimeConstRateEma.variant_id: EbfOptV9SoftWLinearTimeConstRateEma,
}

_ALIASES: dict[str, str] = {
    "ebfv1": EbfOptV1EqualWLinear.variant_id,
    "ebfv2": EbfOptV2SoftWLinear.variant_id,
    "ebfv3": EbfOptV3SoftWRecent.variant_id,
    "ebfv4": EbfOptV4SoftWLinearBlock.variant_id,
    "ebfv41": EbfOptV41SoftWLinearBlockMix.variant_id,
    "ebfv5": EbfOptV5SoftWLinearSameMinusOpp.variant_id,
    "ebfv6": EbfOptV6SoftWLinearPurity.variant_id,
    "ebfv7": EbfOptV7SoftWLinearPolRate.variant_id,
    "ebfv8": EbfOptV8SoftWLinearBinRate.variant_id,
    "ebfv9": EbfOptV9SoftWLinearTimeConstRateEma.variant_id,
}


VARIANTS = dict(_CANONICAL)
for _alias, _target in _ALIASES.items():
    VARIANTS[_alias] = _CANONICAL[_target]


def list_variants() -> list[str]:
    return sorted(VARIANTS.keys())


def create_ebfopt_variant(variant_id: str, *args, **kwargs):
    vid = str(variant_id).strip().lower()
    if vid not in VARIANTS:
        raise KeyError(f"unknown EBF_optimized variant: {variant_id!r}. choices={sorted(VARIANTS.keys())}")
    return VARIANTS[vid](*args, **kwargs)
