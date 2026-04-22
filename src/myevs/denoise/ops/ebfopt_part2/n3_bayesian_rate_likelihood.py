from __future__ import annotations

import os

import numpy as np

from ....timebase import TimeBase

try:
    import numba
except Exception:  # pragma: no cover
    numba = None


def _read_int_env(name: str, default: int) -> int:
    try:
        v = int(float(os.environ.get(name, str(default))))
    except Exception:
        return int(default)
    return int(v)


def _read_float_env(name: str, default: float) -> float:
    try:
        v = float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)
    if not bool(np.isfinite(v)):
        return float(default)
    return float(v)


def _require_numba() -> None:
    if numba is None:
        raise RuntimeError("n3 requires numba, but import failed")


def _try_build_n3_kernel():
    _require_numba()

    @numba.njit(cache=True)
    def _kernel(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        width: int,
        height: int,
        radius_px: int,
        support_dt_ticks: int,
        tau_noise_ticks: float,
        prior_alpha: float,
        prior_beta: float,
    ) -> np.ndarray:
        n = int(t.shape[0])
        out = np.zeros((n,), dtype=np.float32)

        npx = width * height
        alpha = np.full((npx,), np.float32(prior_alpha), dtype=np.float32)
        beta = np.full((npx,), np.float32(prior_beta), dtype=np.float32)
        last_update = np.zeros((npx,), dtype=np.uint64)
        last_pos = np.zeros((npx,), dtype=np.uint64)
        last_neg = np.zeros((npx,), dtype=np.uint64)

        r = int(radius_px)
        r2 = int(r * r)

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            if xi < 0 or xi >= width or yi < 0 or yi >= height:
                continue

            ti = np.uint64(t[i])
            idx = yi * width + xi

            # Exponential forgetting for background posterior stats.
            t_prev = last_update[idx]
            if t_prev > 0:
                dtf = float(np.uint64(ti - t_prev))
                if tau_noise_ticks > 1.0:
                    decay = np.exp(-dtf / tau_noise_ticks)
                    alpha[idx] = np.float32(prior_alpha + (alpha[idx] - prior_alpha) * decay)
                    beta[idx] = np.float32(prior_beta + (beta[idx] - prior_beta) * decay)

            arr = last_pos if int(p[i]) > 0 else last_neg

            # Structure evidence: same-polarity recent support in local neighborhood.
            support = 0.0
            for dy in range(-r, r + 1):
                yy = yi + dy
                if yy < 0 or yy >= height:
                    continue
                for dx in range(-r, r + 1):
                    d2 = dx * dx + dy * dy
                    if d2 > r2:
                        continue
                    xx = xi + dx
                    if xx < 0 or xx >= width:
                        continue
                    jdx = yy * width + xx
                    tj = arr[jdx]
                    if tj == 0:
                        continue
                    dtj = np.uint64(ti - tj)
                    if dtj <= np.uint64(support_dt_ticks):
                        support += 1.0 / (1.0 + float(d2))

            # Posterior mean of background rate.
            noise_rate = float(alpha[idx]) / float(beta[idx] + 1e-6)
            # Likelihood-ratio style score.
            score = support / (noise_rate + 1e-3)
            out[i] = np.float32(score)

            # Online posterior update: this event contributes to local background process.
            alpha[idx] = np.float32(alpha[idx] + 1.0)
            beta[idx] = np.float32(beta[idx] + 1.0)
            last_update[idx] = ti
            arr[idx] = ti

        return out

    return _kernel


def score_stream_n3(
    ev,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_us: int,
    tb: TimeBase,
    scores_out: np.ndarray | None = None,
) -> np.ndarray:
    """N3: online Bayesian background-rate likelihood ratio.

    Real-time, one-pass, O(r^2) per event.

    For each pixel we maintain Gamma posterior (alpha, beta) of background rate.
    For each event i, local same-polarity support S_i is measured in short window,
    and score is computed as:

    score_i = S_i / (E[lambda_n(x_i, y_i)] + eps),
      E[lambda_n] = alpha / beta.

    Env vars:
    - MYEVS_N3_SUPPORT_DT_US (default min(tau_us, 32000))
    - MYEVS_N3_TAU_NOISE_US (default 128000)
    - MYEVS_N3_PRIOR_ALPHA (default 2.0)
    - MYEVS_N3_PRIOR_BETA (default 20.0)
    """

    n = int(ev.t.shape[0])
    if scores_out is None:
        scores_out = np.empty((n,), dtype=np.float32)
    if n <= 0:
        return scores_out

    support_dt_us = int(_read_int_env("MYEVS_N3_SUPPORT_DT_US", min(int(tau_us), 32000)))
    tau_noise_us = int(_read_int_env("MYEVS_N3_TAU_NOISE_US", 128000))

    support_dt_ticks = int(tb.us_to_ticks(max(1, support_dt_us)))
    tau_noise_ticks = float(tb.us_to_ticks(max(1, tau_noise_us)))

    prior_alpha = float(_read_float_env("MYEVS_N3_PRIOR_ALPHA", 2.0))
    prior_beta = float(_read_float_env("MYEVS_N3_PRIOR_BETA", 20.0))
    ker = _try_build_n3_kernel()
    scores = ker(
        ev.t.astype(np.uint64, copy=False),
        ev.x.astype(np.int32, copy=False),
        ev.y.astype(np.int32, copy=False),
        ev.p,
        int(width),
        int(height),
        int(radius_px),
        int(max(1, support_dt_ticks)),
        float(max(1.0, tau_noise_ticks)),
        float(max(1e-3, prior_alpha)),
        float(max(1e-3, prior_beta)),
    )
    scores_out[:] = scores
    return scores_out
