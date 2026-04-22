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
        raise RuntimeError("n106 requires numba, but import failed")


def _try_build_n106_kernel():
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
        gamma_iso: float,
        support_n0: float,
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
        g = float(gamma_iso)
        n0 = float(support_n0)
        if sd <= 0.0:
            sd = 1e-6
        if n0 <= 0.0:
            n0 = 1.0
        if g < 0.0:
            g = 0.0

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

            s0 = 0.0  # horizontal axis
            s1 = 0.0  # vertical axis
            s2 = 0.0  # main diagonal
            s3 = 0.0  # anti diagonal
            nsupport = 0

            for yy in range(y0, y1 + 1):
                base = yy * width
                dy = yy - yi
                ady = dy if dy >= 0 else -dy
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

                    nsupport += 1

                    dx = xx - xi
                    adx = dx if dx >= 0 else -dx

                    wt = 1.0 - float(dt_ticks) * inv_tau
                    if wt <= 0.0:
                        continue
                    wd = np.exp(-float(dx * dx + dy * dy) * inv_2sd2)
                    w = wt * wd

                    if adx > ady:
                        s0 += w
                    elif ady > adx:
                        s1 += w
                    else:
                        if dx * dy > 0:
                            s2 += w
                        else:
                            s3 += w

            smax = s0
            if s1 > smax:
                smax = s1
            if s2 > smax:
                smax = s2
            if s3 > smax:
                smax = s3

            smin = s0
            if s1 < smin:
                smin = s1
            if s2 < smin:
                smin = s2
            if s3 < smin:
                smin = s3

            aniso = smax - g * smin
            if aniso < 0.0:
                aniso = 0.0

            nfac = float(nsupport) / n0
            if nfac > 1.0:
                nfac = 1.0
            if nfac < 0.0:
                nfac = 0.0

            out[i] = np.float32(aniso * nfac)

            idx0 = yi * width + xi
            if pi > 0:
                pos_ts[idx0] = ti
            else:
                neg_ts[idx0] = ti

    return _kernel


def score_stream_n106(
    ev,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_us: int,
    tb: TimeBase,
    scores_out: np.ndarray | None = None,
) -> np.ndarray:
    """N106 (7.53): sector-density anisotropy with temporal support gating."""

    n = int(ev.t.shape[0])
    if scores_out is None:
        scores_out = np.empty((n,), dtype=np.float32)
    if n <= 0:
        return scores_out

    tau_ticks = int(tb.us_to_ticks(int(tau_us)))

    sigma_d_px = float(_read_float_env("MYEVS_N106_SIGMA_D_PX", 1.8))
    gamma_iso = float(_read_float_env("MYEVS_N106_GAMMA_ISO", 1.0))
    support_n0 = float(_read_float_env("MYEVS_N106_SUPPORT_N0", 10.0))

    ker = _try_build_n106_kernel()
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
        float(gamma_iso),
        float(support_n0),
        scores_out,
    )
    return scores_out
