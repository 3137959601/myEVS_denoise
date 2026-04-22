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
        raise RuntimeError("n94 requires numba, but import failed")


def _try_build_n94_kernel():
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
        spatial_range_px: float,
        out: np.ndarray,
    ) -> None:
        n = int(t.shape[0])
        npx = int(width) * int(height)

        # Per-pixel latest timestamp for each polarity.
        pos_ts = np.zeros((npx,), dtype=np.uint64)
        neg_ts = np.zeros((npx,), dtype=np.uint64)

        rr = int(radius_px)
        if rr < 0:
            rr = 0
        if rr > 8:
            rr = 8

        tau = int(tau_ticks)
        if tau <= 0:
            for i in range(n):
                out[i] = np.float32(0.0)
                xi = int(x[i])
                yi = int(y[i])
                if xi < 0 or xi >= width or yi < 0 or yi >= height:
                    continue
                idx = yi * width + xi
                ti = np.uint64(t[i])
                if int(p[i]) > 0:
                    pos_ts[idx] = ti
                else:
                    neg_ts[idx] = ti
            return

        dmax = float(spatial_range_px)
        if dmax <= 0.0:
            if rr > 0:
                dmax = np.sqrt(2.0 * float(rr) * float(rr))
            else:
                dmax = 1.0
        inv_dmax = 1.0 / dmax
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

            score = 0.0
            for yy in range(y0, y1 + 1):
                base = yy * width
                dyf = float(yi - yy)
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

                    dt = int(ti - ts) if ti >= ts else int(ts - ti)
                    if dt <= 0 or dt > tau:
                        continue

                    # Time + spatial coupled linear weights.
                    wt = float(tau - dt) * inv_tau
                    dxf = float(xi - xx)
                    dist = np.sqrt(dxf * dxf + dyf * dyf)
                    ws = 1.0 - dist * inv_dmax
                    if ws <= 0.0:
                        continue
                    score += wt * ws

            out[i] = np.float32(score)

            idx0 = yi * width + xi
            if pi > 0:
                pos_ts[idx0] = ti
            else:
                neg_ts[idx0] = ti

    return _kernel


def score_stream_n94(
    ev,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_us: int,
    tb: TimeBase,
    scores_out: np.ndarray | None = None,
) -> np.ndarray:
    """N94: spatiotemporal linear support score.

    Score definition:
      Score_i = sum_j [ ((tau - dt_ij) / tau) * max(0, 1 - d_ij / dmax) ]

    where j iterates same-polarity latest supports in local window.
    """

    n = int(ev.t.shape[0])
    if scores_out is None:
        scores_out = np.empty((n,), dtype=np.float32)
    if n <= 0:
        return scores_out

    tau_ticks = int(tb.us_to_ticks(int(tau_us)))

    # If <=0, kernel falls back to dmax=sqrt(2)*radius.
    spatial_range_px = float(_read_float_env("MYEVS_N94_SPACE_RANGE_PX", 0.0))

    ker = _try_build_n94_kernel()
    ker(
        ev.t.astype(np.uint64, copy=False),
        ev.x.astype(np.int32, copy=False),
        ev.y.astype(np.int32, copy=False),
        ev.p,
        int(width),
        int(height),
        int(radius_px),
        int(tau_ticks),
        float(spatial_range_px),
        scores_out,
    )
    return scores_out
