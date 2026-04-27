"""Denoise operations (Qt-aligned).

Each operation is a small, stateful filter that decides whether an event should be kept.

Design goals:
- Easy to add new algorithms.
- Easy to *compose* multiple algorithms into a pipeline for A/B testing.
- Keep the logic as close as possible to the existing Qt implementation.
"""

from .base import DenoiseOp
from .stc import StcOp
from .refractory import RefractoryOp
from .hotpixel import HotPixelOp
from .baf import BafOp
from .ratelimit import RateLimitOp
from .dp import DpOp
from .globalgate import GlobalGateOp
from .fastdecay import FastDecayOp
from .ebf import EbfOp
from .knoise import KnoiseOp
from .evflow import EvFlowOp
from .ynoise import YnoiseOp
from .ts import TsOp
from .mlpf import MlpfOp
from .pfd import PfdOp

__all__ = [
    "DenoiseOp",
    "StcOp",
    "RefractoryOp",
    "HotPixelOp",
    "BafOp",
    "RateLimitOp",
    "DpOp",
    "GlobalGateOp",
    "FastDecayOp",
    "EbfOp",
    "KnoiseOp",
    "EvFlowOp",
    "YnoiseOp",
    "TsOp",
    "MlpfOp",
    "PfdOp",
]
