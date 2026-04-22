from __future__ import annotations

import numpy as np

from ...numba_ebf import ebf_scores_stream_numba, ebf_state_init


def score_stream_s80(
    ev,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_ticks: int,
    kernel_cache: dict[str, object],
    scores_out: np.ndarray | None = None,
) -> np.ndarray:
    """Part2 s80: baseline EBF score × AOCC-lite (per-event) mild gate.

    Notes
    - Baseline: standard EBF neighborhood evidence (same polarity, linear time decay).
    - Controller: reuse s76 AOCC-lite kernel output as a per-event structure proxy.
    - Fusion: multiplicative factor in [0.75, 1.25] to keep controller gentle.

    This function is intentionally called by slim sweep scripts; it is not a Numba kernel.
    """

    n = int(ev.t.shape[0])
    if scores_out is None:
        scores_out = np.empty((n,), dtype=np.float32)

    # 1) baseline EBF scores
    last_ts, last_pol = ebf_state_init(int(width), int(height))
    ebf_scores_stream_numba(
        t=ev.t,
        x=ev.x,
        y=ev.y,
        p=ev.p,
        width=int(width),
        height=int(height),
        radius_px=int(radius_px),
        tau_ticks=int(tau_ticks),
        last_ts=last_ts,
        last_pol=last_pol,
        scores_out=scores_out,
    )

    # 2) AOCC-lite control (reuse s76 kernel)
    from .s76_aocc_activity_sobel_gradmag import (
        try_build_s76_aocc_activity_sobel_gradmag_scores_kernel,
    )

    ker = kernel_cache.get("ker_s76")
    if ker is None:
        ker = try_build_s76_aocc_activity_sobel_gradmag_scores_kernel()
        kernel_cache["ker_s76"] = ker
    if ker is None:
        raise SystemExit("s80 requires numba (via s76 kernel), but kernel build failed")

    aocc_lite = np.empty((n,), dtype=np.float32)
    last_t = np.zeros((int(width) * int(height),), dtype=np.uint64)
    last_a = np.zeros((int(width) * int(height),), dtype=np.float32)
    ker(
        ev.t,
        ev.x,
        ev.y,
        ev.p,
        int(width),
        int(height),
        int(radius_px),
        int(tau_ticks),
        last_t,
        last_a,
        aocc_lite,
    )

    # 3) Fuse: factor in [0.75, 1.25] based on a bounded gate01.
    # gate01 = v/(v+c) maps (0..inf)->(0..1). We assume aocc_lite >= 0.
    c = np.float32(1.0)
    vpos = np.maximum(aocc_lite.astype(np.float32, copy=False), 0.0)
    gate01 = vpos / (vpos + c)
    factor = np.float32(0.75) + np.float32(0.5) * gate01
    scores_out *= factor
    return scores_out
