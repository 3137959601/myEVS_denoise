"""EBF Part2 (precision-first) experimental ops.

This package contains low-latency, streaming-friendly variants (s1, s2, ...)
that aim to improve separability/precision before adding normalization/
adaptive thresholding.
"""

from .s1_dircoh import S1DirCohParams, s1_dircoh_params_from_env, try_build_s1_dircoh_scores_kernel
from .s2_cohgate import S2CohGateParams, s2_cohgate_params_from_env, try_build_s2_cohgate_scores_kernel
from .s3_softgate import S3SoftGateParams, s3_softgate_params_from_env, try_build_s3_softgate_scores_kernel
from .s4_residual_gate import (
    S4ResidualGateParams,
    s4_residual_gate_params_from_env,
    try_build_s4_residual_gate_scores_kernel,
)
from .s5_elliptic_spatialw import (
    S5EllipticSpatialWParams,
    s5_elliptic_spatialw_params_from_env,
    try_build_s5_elliptic_spatialw_scores_kernel,
)
from .s6_timecoh_gate import (
    S6TimeCohGateParams,
    s6_timecoh_gate_params_from_env,
    try_build_s6_timecoh_gate_scores_kernel,
)
from .s7_plane_gate import S7PlaneGateParams, s7_plane_gate_params_from_env, try_build_s7_plane_gate_scores_kernel
from .s8_plane_r2_gate import (
    S8PlaneR2GateParams,
    s8_plane_r2_gate_params_from_env,
    try_build_s8_plane_r2_gate_scores_kernel,
)
from .s9_refractory_gate import (
    S9RefractoryGateParams,
    s9_refractory_gate_params_from_env,
    try_build_s9_refractory_gate_scores_kernel,
)
from .s10_hotpixel_rate_gate import (
    S10HotPixelRateGateParams,
    s10_hotpixel_rate_gate_params_from_env,
    try_build_s10_hotpixel_rate_gate_scores_kernel,
)
from .s11_relative_hotness_gate import (
    S11RelativeHotnessGateParams,
    s11_relative_hotness_gate_params_from_env,
    try_build_s11_relative_hotness_gate_scores_kernel,
)
from .s12_hotness_zscore_gate import (
    S12HotnessZScoreGateParams,
    s12_hotness_zscore_gate_params_from_env,
    try_build_s12_hotness_zscore_gate_scores_kernel,
)
from .s13_crosspol_support_gate import (
    S13CrossPolSupportGateParams,
    s13_crosspol_support_gate_params_from_env,
    try_build_s13_crosspol_support_gate_scores_kernel,
)
from .s14_crosspol_boost import (
    S14CrossPolBoostParams,
    s14_crosspol_boost_params_from_env,
    try_build_s14_crosspol_boost_scores_kernel,
)
from .s15_flip_flicker_gate import (
    S15FlipFlickerGateParams,
    s15_flip_flicker_gate_params_from_env,
    try_build_s15_flip_flicker_gate_scores_kernel,
)
from .s16_s14_hotness_clamp import (
    S16S14HotnessClampParams,
    s16_s14_hotness_clamp_params_from_env,
    try_build_s16_s14_hotness_clamp_scores_kernel,
)
from .s17_crosspol_spread_boost import (
    S17CrossPolSpreadBoostParams,
    s17_crosspol_spread_boost_params_from_env,
    try_build_s17_crosspol_spread_boost_scores_kernel,
)
from .s18_no_polarity_ebf import try_build_s18_no_polarity_ebf_scores_kernel
from .s19_evidence_fusion_q8 import try_build_s19_evidence_fusion_q8_scores_kernel
from .s20_polhot_evidence_fusion_q8 import try_build_s20_polhot_evidence_fusion_q8_scores_kernel
from .s21_bipolhot_evidence_fusion_q8 import try_build_s21_bipolhot_evidence_fusion_q8_scores_kernel

__all__ = [
    "S1DirCohParams",
    "s1_dircoh_params_from_env",
    "try_build_s1_dircoh_scores_kernel",
    "S2CohGateParams",
    "s2_cohgate_params_from_env",
    "try_build_s2_cohgate_scores_kernel",
    "S3SoftGateParams",
    "s3_softgate_params_from_env",
    "try_build_s3_softgate_scores_kernel",
    "S4ResidualGateParams",
    "s4_residual_gate_params_from_env",
    "try_build_s4_residual_gate_scores_kernel",
    "S5EllipticSpatialWParams",
    "s5_elliptic_spatialw_params_from_env",
    "try_build_s5_elliptic_spatialw_scores_kernel",
    "S6TimeCohGateParams",
    "s6_timecoh_gate_params_from_env",
    "try_build_s6_timecoh_gate_scores_kernel",
    "S7PlaneGateParams",
    "s7_plane_gate_params_from_env",
    "try_build_s7_plane_gate_scores_kernel",
    "S8PlaneR2GateParams",
    "s8_plane_r2_gate_params_from_env",
    "try_build_s8_plane_r2_gate_scores_kernel",
    "S9RefractoryGateParams",
    "s9_refractory_gate_params_from_env",
    "try_build_s9_refractory_gate_scores_kernel",
    "S10HotPixelRateGateParams",
    "s10_hotpixel_rate_gate_params_from_env",
    "try_build_s10_hotpixel_rate_gate_scores_kernel",
    "S11RelativeHotnessGateParams",
    "s11_relative_hotness_gate_params_from_env",
    "try_build_s11_relative_hotness_gate_scores_kernel",
    "S12HotnessZScoreGateParams",
    "s12_hotness_zscore_gate_params_from_env",
    "try_build_s12_hotness_zscore_gate_scores_kernel",
    "S13CrossPolSupportGateParams",
    "s13_crosspol_support_gate_params_from_env",
    "try_build_s13_crosspol_support_gate_scores_kernel",
    "S14CrossPolBoostParams",
    "s14_crosspol_boost_params_from_env",
    "try_build_s14_crosspol_boost_scores_kernel",
    "S15FlipFlickerGateParams",
    "s15_flip_flicker_gate_params_from_env",
    "try_build_s15_flip_flicker_gate_scores_kernel",
    "S16S14HotnessClampParams",
    "s16_s14_hotness_clamp_params_from_env",
    "try_build_s16_s14_hotness_clamp_scores_kernel",
    "S17CrossPolSpreadBoostParams",
    "s17_crosspol_spread_boost_params_from_env",
    "try_build_s17_crosspol_spread_boost_scores_kernel",
    "try_build_s18_no_polarity_ebf_scores_kernel",
    "try_build_s19_evidence_fusion_q8_scores_kernel",
    "try_build_s20_polhot_evidence_fusion_q8_scores_kernel",
    "try_build_s21_bipolhot_evidence_fusion_q8_scores_kernel",
]
