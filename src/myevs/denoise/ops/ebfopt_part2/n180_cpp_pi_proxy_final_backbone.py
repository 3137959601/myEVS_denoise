from __future__ import annotations

import numpy as np

from ....timebase import TimeBase
from .n180_realtime_relief_pi_proxy_backbone import score_stream_n176


N179_K_SFRAC_DEFAULT = 0.40
N179_K_MIX_DEFAULT = 0.0


def score_stream_n179(
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
    """N179 final entrypoint.

    N179 freezes the n176 formula as the final algorithm family entry.
    Parameters remain open for sweep (radius_px, tau_us, beta_init, k_sfrac, k_mix).
    If K is not provided, use the cross-dataset default selected in README 27.8.
    """

    return score_stream_n176(
        ev,
        width=int(width),
        height=int(height),
        radius_px=int(radius_px),
        tau_us=int(tau_us),
        tb=tb,
        beta_init=beta_init,
        k_sfrac=float(N179_K_SFRAC_DEFAULT if k_sfrac is None else k_sfrac),
        k_mix=float(N179_K_MIX_DEFAULT if k_mix is None else k_mix),
        rhythm_pressure_coeff=rhythm_pressure_coeff,
        rhythm_good_coeff=rhythm_good_coeff,
        support_good_coeff=support_good_coeff,
        pi_bad_coeff=pi_bad_coeff,
        pi_r_coeff=pi_r_coeff,
        pi_good_coeff=pi_good_coeff,
        scores_out=scores_out,
    )
