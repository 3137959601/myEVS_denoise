from __future__ import annotations

"""Compatibility wrapper.

Implementation moved to `myevs.denoise.ops.ebfopt_variants.v10_spatialw_linear`.
Keep this module to avoid breaking older imports.
"""

from .ebfopt_variants.v10_spatialw_linear import (  # noqa: F401
    EbfV10SpatialWLinearOp,
    build_spatial_lut,
    try_build_v10_spatialw_linear_scores_kernel,
)

__all__ = [
    "EbfV10SpatialWLinearOp",
    "build_spatial_lut",
    "try_build_v10_spatialw_linear_scores_kernel",
]
