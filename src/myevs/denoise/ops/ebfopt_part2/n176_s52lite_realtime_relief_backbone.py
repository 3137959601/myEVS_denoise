from __future__ import annotations

import numpy as np

from ....timebase import TimeBase
from .n175_s52lite_rational_gate_backbone import (
    N175_BETA_INIT,
    _score_stream_n175_core,
)


# N176 keeps n175's rational rhythm gate, but shifts the fixed support/mix
# balance toward realtime-friendly, broad-support recovery.
N176_BETA_INIT = N175_BETA_INIT
N176_K_SFRAC = 4.0 / 5.0
N176_K_MIX = 1.0 / 8.0


def score_stream_n176(
    ev,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_us: int,
    tb: TimeBase,
    scores_out: np.ndarray | None = None,
) -> np.ndarray:
    """N176: n175 with stronger support relief and weaker mix penalty.

    Relative to n175:
    - support relief uses k_s=4/5 instead of 2/3;
    - local polarity-mix gain uses k_m=1/8 instead of 1/5;
    - state layout, LUTs and kernel body are otherwise unchanged.
    """

    return _score_stream_n175_core(
        ev,
        width=int(width),
        height=int(height),
        radius_px=int(radius_px),
        tau_us=int(tau_us),
        tb=tb,
        beta_init=float(N176_BETA_INIT),
        k_sfrac=float(N176_K_SFRAC),
        k_mix=float(N176_K_MIX),
        scores_out=scores_out,
    )
