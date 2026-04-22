from __future__ import annotations

import os

import numpy as np

from ....timebase import TimeBase

try:
    import numba
except Exception:  # pragma: no cover
    numba = None


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
        raise RuntimeError("n109 requires numba, but import failed")


def _try_build_n109_kernel():
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
        tau_ticks: int,
        sigma_d_px: float,
        alpha: float,
        lambda_mix: float,
        out: np.ndarray,
    ) -> None:
        n = int(t.shape[0])
        npx = int(width) * int(height)

        pos_ts = np.zeros((npx,), dtype=np.uint64)
        neg_ts = np.zeros((npx,), dtype=np.uint64)

        rr = int(radius_px)
        if rr < 0:
            rr = 0
        if rr > 8:
            rr = 8

        tau = int(tau_ticks)
        if tau <= 0:
            tau = 1

        sd = float(sigma_d_px)
        a = float(alpha)
        lm = float(lambda_mix)
        if sd <= 0.0:
            sd = 1e-6
        if a <= 0.0:
            a = 1e-6
        if lm < 0.0:
            lm = 0.0
        elif lm > 1.0:
            lm = 1.0

        inv_2sd2 = 1.0 / (2.0 * sd * sd)
        inv_tau = 1.0 / float(tau)

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            if xi < 0 or xi >= width or yi < 0 or yi >= height:
                out[i] = np.float32(0.0)
                continue

            ti = np.uint64(t[i])
            pi = 1 if int(p[i]) > 0 else -1

            x0 = xi - rr
            if x0 < 0:
                x0 = 0
            x1 = xi + rr
            if x1 >= width:
                x1 = width - 1
            y0 = yi - rr
            if y0 < 0:
                y0 = 0
            y1 = yi + rr
            if y1 >= height:
                y1 = height - 1

            s_total = 0.0
            s_horiz = 0.0
            s_diag1 = 0.0

            for yy in range(y0, y1 + 1):
                base = yy * width
                dy = yy - yi
                for xx in range(x0, x1 + 1):
                    if xx == xi and yy == yi:
                        continue

                    idx_nb = base + xx
                    ts = np.uint64(0)
                    if pi > 0:
                        ts = pos_ts[idx_nb]
                    else:
                        ts = neg_ts[idx_nb]
                    if ts == 0:
                        continue
                    if ti <= ts:
                        continue

                    dt_ticks = int(ti - ts)
                    if dt_ticks <= 0 or dt_ticks > tau:
                        continue

                    dx = xx - xi

                    wt = 1.0 - float(dt_ticks) * inv_tau
                    if wt <= 0.0:
                        continue
                    wd = np.exp(-float(dx * dx + dy * dy) * inv_2sd2)
                    w = wt * wd

                    d_sq = float(dx * dx + dy * dy)

                    s_total += w
                    s_horiz += w * (float(dx * dx) / d_sq)
                    s_diag1 += w * (float((dx + dy) * (dx + dy)) / (2.0 * d_sq))

            s_vert = s_total - s_horiz
            s_diag2 = s_total - s_diag1

            a_hv = abs(s_horiz - s_vert)
            a_diag = abs(s_diag1 - s_diag2)
            a_max = a_hv if a_hv > a_diag else a_diag

            if s_total <= 0.0:
                score = 0.0
            else:
                aniso_ratio = a_max / s_total
                if aniso_ratio < 0.0:
                    aniso_ratio = 0.0
                elif aniso_ratio > 1.0:
                    aniso_ratio = 1.0
                mix_term = (1.0 - lm) + lm * (aniso_ratio ** a)
                score = s_total * mix_term

            if score < 0.0:
                score = 0.0

            out[i] = np.float32(score)

            idx0 = yi * width + xi
            if pi > 0:
                pos_ts[idx0] = ti
            else:
                neg_ts[idx0] = ti

    return _kernel


def score_stream_n109(
    ev,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_us: int,
    tb: TimeBase,
    scores_out: np.ndarray | None = None,
) -> np.ndarray:
    """N109 (7.56-B): mixed floor-preserving anisotropy enhancement.

    Score = S_total * ((1-lambda_mix) + lambda_mix * (A_max / S_total)**alpha)
    """

    n = int(ev.t.shape[0])
    if scores_out is None:
        scores_out = np.empty((n,), dtype=np.float32)
    if n <= 0:
        return scores_out

    tau_ticks = int(tb.us_to_ticks(int(tau_us)))

    sigma_d_px = float(_read_float_env("MYEVS_N109_SIGMA_D_PX", 1.8))
    alpha = float(_read_float_env("MYEVS_N109_ALPHA", 1.5))
    lambda_mix = float(_read_float_env("MYEVS_N109_LAMBDA_MIX", 0.7))

    ker = _try_build_n109_kernel()
    ker(
        ev.t.astype(np.uint64, copy=False),
        ev.x.astype(np.int32, copy=False),
        ev.y.astype(np.int32, copy=False),
        ev.p,
        int(width),
        int(height),
        int(radius_px),
        int(tau_ticks),
        float(sigma_d_px),
        float(alpha),
        float(lambda_mix),
        scores_out,
    )
    return scores_out
