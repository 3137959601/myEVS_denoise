from __future__ import annotations

import numpy as np

from ....timebase import TimeBase
from .n181_simplified_n179_backbone import score_stream_n181


N185_K_SFRAC_DEFAULT = 0.40
N185_RHYTHM_GOOD_COEFF_DEFAULT = 0.75
N185_ALPHA_DEFAULT = 0.5


def score_stream_n185(
    ev,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_us: int,
    tb: TimeBase,
    beta_init: float | None = None,
    k_sfrac: float | None = None,
    rhythm_good_coeff: float | None = None,
    alpha: float | None = None,
    scores_out: np.ndarray | None = None,
) -> np.ndarray:
    """N185: N181-conservative structure + tunable alpha on rhythm_pressure.

    u' = u * (1 - k_s * s) * (1 + alpha * pi - r_g * good)
    with N181 simplified structure as backbone.
    """

    return score_stream_n181(
        ev,
        width=int(width),
        height=int(height),
        radius_px=int(radius_px),
        tau_us=int(tau_us),
        tb=tb,
        beta_init=beta_init,
        k_sfrac=float(N185_K_SFRAC_DEFAULT if k_sfrac is None else k_sfrac),
        rhythm_good_coeff=float(
            N185_RHYTHM_GOOD_COEFF_DEFAULT if rhythm_good_coeff is None else rhythm_good_coeff
        ),
        rhythm_pi_coeff=float(N185_ALPHA_DEFAULT if alpha is None else alpha),
        mode="conservative",
        scores_out=scores_out,
    )

