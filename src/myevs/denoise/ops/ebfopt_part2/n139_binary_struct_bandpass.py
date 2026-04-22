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
        raise RuntimeError("n139 requires numba, but import failed")


def _try_build_n139_kernel():
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
        tau_base_ticks: int,
        l_thresh: float,
        h_thresh: float,
        eps: float,
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

        tau_base = int(tau_base_ticks)
        if tau_base <= 0:
            tau_base = 1
        inv_tau_base = 1.0 / float(tau_base)

        lo = float(l_thresh)
        hi = float(h_thresh)
        if lo < 0.0:
            lo = 0.0
        if hi > 1.0:
            hi = 1.0
        if hi <= lo:
            lo = 0.1
            hi = 0.9

        epsv = float(eps)
        if epsv <= 0.0:
            epsv = 1e-3

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

            e0 = 0.0
            e90 = 0.0
            e45 = 0.0
            e135 = 0.0
            s_base = 0.0

            for yy in range(y0, y1 + 1):
                base = yy * width
                dy = yy - yi
                for xx in range(x0, x1 + 1):
                    if xx == xi and yy == yi:
                        continue

                    idx_nb = base + xx
                    ts = last_ts[idx_nb]
                    if ts == 0 or ti <= ts:
                        continue
                    if int(last_pol[idx_nb]) != pi:
                        continue

                    dt_ticks = int(ti - ts)
                    if dt_ticks > tau_base:
                        continue

                    w_time = 1.0 - float(dt_ticks) * inv_tau_base
                    if w_time <= 0.0:
                        continue

                    # S_base should be the total temporal support energy.
                    s_base += w_time

                    dx = xx - xi
                    if dy == 0:
                        e0 += w_time
                    if dx == 0:
                        e90 += w_time
                    if dx == dy:
                        e45 += w_time
                    if dx == -dy:
                        e135 += w_time

            emax = e0
            if e90 > emax:
                emax = e90
            if e45 > emax:
                emax = e45
            if e135 > emax:
                emax = e135

            if s_base <= 0.0:
                out[i] = np.float32(0.0)
            else:
                a_true = emax / (s_base + epsv)
                out[i] = np.float32(s_base if (a_true >= lo and a_true <= hi) else 0.0)

            idx0 = yi * width + xi
            last_ts[idx0] = ti
            last_pol[idx0] = np.int8(pi)

    return _kernel


def score_stream_n139(
    ev,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_us: int,
    tb: TimeBase,
    low_thresh: float | None = None,
    high_thresh: float | None = None,
    scores_out: np.ndarray | None = None,
) -> np.ndarray:
    """N139 (7.85): A_true band-pass gate with S_base score output.

    A_true = Emax / (S_base + eps), where S_base is total temporal support.
    score = S_base if low_thresh <= A_true <= high_thresh else 0.
    """

    n = int(ev.t.shape[0])
    if scores_out is None:
        scores_out = np.empty((n,), dtype=np.float32)
    if n <= 0:
        return scores_out

    tau_base_ticks = int(tb.us_to_ticks(int(tau_us)))
    lo = float(_read_float_env("MYEVS_N139_L_THRESH", 0.1)) if low_thresh is None else float(low_thresh)
    hi = float(_read_float_env("MYEVS_N139_H_THRESH", 0.9)) if high_thresh is None else float(high_thresh)
    eps = float(_read_float_env("MYEVS_N139_EPS", 1e-3))

    ker = _try_build_n139_kernel()
    ker(
        ev.t.astype(np.uint64, copy=False),
        ev.x.astype(np.int32, copy=False),
        ev.y.astype(np.int32, copy=False),
        ev.p,
        int(width),
        int(height),
        int(radius_px),
        int(tau_base_ticks),
        float(lo),
        float(hi),
        float(eps),
        scores_out,
    )
    return scores_out
