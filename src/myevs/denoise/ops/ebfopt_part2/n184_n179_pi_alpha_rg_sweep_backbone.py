from __future__ import annotations

import numpy as np

from ....timebase import TimeBase
from .n176_s52lite_realtime_relief_backbone import score_stream_n176


N184_K_SFRAC_DEFAULT = 0.40
N184_K_MIX_DEFAULT = 0.0
N184_RHYTHM_PRESSURE_COEFF_DEFAULT = 0.0
N184_RHYTHM_GOOD_COEFF_DEFAULT = 0.75
N184_SUPPORT_GOOD_COEFF_DEFAULT = 0.0
N184_PI_ALPHA_DEFAULT = 0.5


def score_stream_n184(
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
    pi_alpha: float | None = None,
    scores_out: np.ndarray | None = None,
) -> np.ndarray:
    """N184: n179-form with tunable pi coefficient in u' scale.

    u' = u * (1-k_s s) * (1 + a*pi - r_g*good)
    where a is exposed as pi_alpha for joint sweep with r_g.
    """

    return score_stream_n176(
        ev,
        width=int(width),
        height=int(height),
        radius_px=int(radius_px),
        tau_us=int(tau_us),
        tb=tb,
        beta_init=beta_init,
        k_sfrac=float(N184_K_SFRAC_DEFAULT if k_sfrac is None else k_sfrac),
        k_mix=float(N184_K_MIX_DEFAULT if k_mix is None else k_mix),
        rhythm_pressure_coeff=float(
            N184_RHYTHM_PRESSURE_COEFF_DEFAULT if rhythm_pressure_coeff is None else rhythm_pressure_coeff
        ),
        rhythm_good_coeff=float(N184_RHYTHM_GOOD_COEFF_DEFAULT if rhythm_good_coeff is None else rhythm_good_coeff),
        support_good_coeff=float(N184_SUPPORT_GOOD_COEFF_DEFAULT if support_good_coeff is None else support_good_coeff),
        rhythm_pi_coeff=float(N184_PI_ALPHA_DEFAULT if pi_alpha is None else pi_alpha),
        scores_out=scores_out,
    )

