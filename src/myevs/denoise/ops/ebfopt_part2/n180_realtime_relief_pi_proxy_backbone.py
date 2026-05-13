from __future__ import annotations

import numpy as np

from ....timebase import TimeBase
from .n180_s52lite_pi_proxy_gate_backbone import (
    N175_BETA_INIT,
    N175_PI_BAD_COEFF,
    N175_PI_GOOD_COEFF,
    N175_PI_R_COEFF,
    N175_RHYTHM_GOOD_COEFF,
    N175_RHYTHM_PRESSURE_COEFF,
    N175_SUPPORT_GOOD_COEFF,
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
    beta_init: float | None = None,
    k_sfrac: float | None = None,
    k_mix: float | None = None,
    rhythm_pressure_coeff: float | None = None,
    rhythm_good_coeff: float | None = None,
    support_good_coeff: float | None = None,
    pi_bad_coeff: float | None = None,
    pi_r_coeff: float | None = None,
    pi_good_coeff: float | None = None,
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
        beta_init=float(N176_BETA_INIT if beta_init is None else beta_init),
        k_sfrac=float(N176_K_SFRAC if k_sfrac is None else k_sfrac),
        k_mix=float(N176_K_MIX if k_mix is None else k_mix),
        rhythm_pressure_coeff=float(
            N175_RHYTHM_PRESSURE_COEFF if rhythm_pressure_coeff is None else rhythm_pressure_coeff
        ),
        rhythm_good_coeff=float(N175_RHYTHM_GOOD_COEFF if rhythm_good_coeff is None else rhythm_good_coeff),
        support_good_coeff=float(N175_SUPPORT_GOOD_COEFF if support_good_coeff is None else support_good_coeff),
        pi_bad_coeff=float(N175_PI_BAD_COEFF if pi_bad_coeff is None else pi_bad_coeff),
        pi_r_coeff=float(N175_PI_R_COEFF if pi_r_coeff is None else pi_r_coeff),
        pi_good_coeff=float(N175_PI_GOOD_COEFF if pi_good_coeff is None else pi_good_coeff),
        scores_out=scores_out,
    )
