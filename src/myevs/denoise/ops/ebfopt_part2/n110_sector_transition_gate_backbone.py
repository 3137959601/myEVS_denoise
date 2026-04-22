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
        raise RuntimeError("n110 requires numba, but import failed")


def _try_build_n110_kernel():
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
        alpha: float,
        beta: float,
        gamma_opp: float,
        out: np.ndarray,
    ) -> None:
        n = int(t.shape[0])
        npx = int(width) * int(height)

        last_ts = np.zeros((npx,), dtype=np.uint64)
        last_pol = np.zeros((npx,), dtype=np.int8)

        rr = int(radius_px)
        if rr < 0:
            rr = 0
        if rr > 8:
            rr = 8

        tau = int(tau_ticks)
        if tau <= 0:
            tau = 1

        a = float(alpha)
        b = float(beta)
        g = float(gamma_opp)
        if a < 0.0:
            a = 0.0
        if b < 0.0:
            b = 0.0
        if g < 0.0:
            g = 0.0

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

            sector_dense = np.zeros((4,), dtype=np.float64)
            top1_t = np.full((4,), -1, dtype=np.int64)
            top2_t = np.full((4,), -1, dtype=np.int64)
            top1_p = np.zeros((4,), dtype=np.int8)
            top2_p = np.zeros((4,), dtype=np.int8)

            for yy in range(y0, y1 + 1):
                base = yy * width
                for xx in range(x0, x1 + 1):
                    if xx == xi and yy == yi:
                        continue

                    idx_nb = base + xx
                    ts = last_ts[idx_nb]
                    if ts == 0:
                        continue
                    if ti <= ts:
                        continue

                    dt_ticks = int(ti - ts)
                    if dt_ticks <= 0 or dt_ticks > tau:
                        continue

                    pj = int(last_pol[idx_nb])
                    w_age = 1.0 - float(dt_ticks) * inv_tau
                    if w_age <= 0.0:
                        continue

                    k = 0
                    if yy > yi:
                        k += 2
                    if xx > xi:
                        k += 1

                    if pj == pi:
                        sector_dense[k] += w_age

                    tsv = int(ts)
                    if tsv > top1_t[k]:
                        top2_t[k] = top1_t[k]
                        top2_p[k] = top1_p[k]
                        top1_t[k] = tsv
                        top1_p[k] = np.int8(pj)
                    elif tsv > top2_t[k]:
                        top2_t[k] = tsv
                        top2_p[k] = np.int8(pj)

            final_score = 0.0
            for k in range(4):
                dense = sector_dense[k]
                mult = 1.0
                if top2_t[k] != -1:
                    p1 = int(top1_p[k])
                    p2 = int(top2_p[k])
                    if p1 == pi and p2 == pi:
                        mult = 1.0 + a
                    elif p1 != p2:
                        mult = 1.0 - b
                    elif p1 != pi and p2 != pi:
                        mult = g
                    else:
                        mult = 1.0
                if mult < 0.0:
                    mult = 0.0

                contrib = dense * mult
                if contrib > 0.0:
                    final_score += contrib

            out[i] = np.float32(final_score)

            idx0 = yi * width + xi
            last_ts[idx0] = ti
            last_pol[idx0] = np.int8(pi)

    return _kernel


def score_stream_n110(
    ev,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_us: int,
    tb: TimeBase,
    scores_out: np.ndarray | None = None,
) -> np.ndarray:
    """N110 (7.57): sector-wise polarity transition gate.

    Score = sum_k max(0, E_k * M_k)
    where E_k is same-polarity density and M_k is transition multiplier from
    top1/top2 polarity state in each sector.
    """

    n = int(ev.t.shape[0])
    if scores_out is None:
        scores_out = np.empty((n,), dtype=np.float32)
    if n <= 0:
        return scores_out

    tau_ticks = int(tb.us_to_ticks(int(tau_us)))

    alpha = float(_read_float_env("MYEVS_N110_ALPHA", 0.2))
    beta = float(_read_float_env("MYEVS_N110_BETA", 0.4))
    gamma_opp = float(_read_float_env("MYEVS_N110_GAMMA", 0.8))

    ker = _try_build_n110_kernel()
    ker(
        ev.t.astype(np.uint64, copy=False),
        ev.x.astype(np.int32, copy=False),
        ev.y.astype(np.int32, copy=False),
        ev.p,
        int(width),
        int(height),
        int(radius_px),
        int(tau_ticks),
        float(alpha),
        float(beta),
        float(gamma_opp),
        scores_out,
    )
    return scores_out
