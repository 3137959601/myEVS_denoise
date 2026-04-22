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
        raise RuntimeError("n134 requires numba, but import failed")


def _try_build_n134_kernel():
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
        s_keep: float,
        bootstrap_first: int,
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

        keep_v = float(s_keep)
        if keep_v < 0.0:
            keep_v = 0.0
        boot = 1 if int(bootstrap_first) != 0 else 0

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

            s_base = 0.0
            for yy in range(y0, y1 + 1):
                base = yy * width
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
                    if w_time > 0.0:
                        s_base += w_time

            out[i] = np.float32(s_base)

            # Stateful support pruning (7.80):
            # only likely events are allowed to enter future support memory.
            idx0 = yi * width + xi
            if s_base >= keep_v or (boot == 1 and last_ts[idx0] == 0):
                last_ts[idx0] = ti
                last_pol[idx0] = np.int8(pi)

    return _kernel


def score_stream_n134(
    ev,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_us: int,
    tb: TimeBase,
    scores_out: np.ndarray | None = None,
) -> np.ndarray:
    """N134 (7.80): baseline score with conditional memory write.

    Score output stays baseline S_base. The only behavior change is whether
    current event is written into the time surface memory, controlled by S_keep.
    """

    n = int(ev.t.shape[0])
    if scores_out is None:
        scores_out = np.empty((n,), dtype=np.float32)
    if n <= 0:
        return scores_out

    tau_base_ticks = int(tb.us_to_ticks(int(tau_us)))
    s_keep = float(_read_float_env("MYEVS_N134_S_KEEP", 1.0))
    bootstrap_first = int(_read_float_env("MYEVS_N134_BOOTSTRAP_FIRST", 1.0))

    ker = _try_build_n134_kernel()
    ker(
        ev.t.astype(np.uint64, copy=False),
        ev.x.astype(np.int32, copy=False),
        ev.y.astype(np.int32, copy=False),
        ev.p,
        int(width),
        int(height),
        int(radius_px),
        int(tau_base_ticks),
        float(s_keep),
        int(bootstrap_first),
        scores_out,
    )
    return scores_out
