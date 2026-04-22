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
        raise RuntimeError("n5 requires numba, but import failed")


def _try_build_n5_kernel():
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
        tau_fast_ticks: float,
        tau_slow_ticks: float,
        support_gain: float,
        fast_gain: float,
        rate_mode_unit: int,
    ) -> np.ndarray:
        n = int(t.shape[0])
        out = np.zeros((n,), dtype=np.float32)

        npx = width * height
        rate_fast = np.zeros((npx,), dtype=np.float32)
        rate_slow = np.zeros((npx,), dtype=np.float32)
        t_fast = np.zeros((npx,), dtype=np.uint64)
        t_slow = np.zeros((npx,), dtype=np.uint64)
        last_pos = np.zeros((npx,), dtype=np.uint64)
        last_neg = np.zeros((npx,), dtype=np.uint64)

        r = int(radius_px)
        r2 = int(r * r)

        eps = 1e-6
        inv_support_dt = 1.0 / float(max(1, support_dt_ticks))

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            if xi < 0 or xi >= width or yi < 0 or yi >= height:
                continue

            ti = np.uint64(t[i])
            idx = yi * width + xi

            # Dual-timescale pixel rate update.
            # rate_mode_unit=0: legacy count-like EMA (better empirical baseline for n5 so far)
            # rate_mode_unit=1: unit-consistent EMA in events/tick (experimental)
            tf = t_fast[idx]
            if tf > 0:
                dtf = float(np.uint64(ti - tf))
                af = 1.0 - np.exp(-dtf / tau_fast_ticks)
                if rate_mode_unit != 0:
                    inst_fast = 1.0 / max(1.0, dtf)
                    rate_fast[idx] = np.float32((1.0 - af) * float(rate_fast[idx]) + af * inst_fast)
                else:
                    rate_fast[idx] = np.float32((1.0 - af) * float(rate_fast[idx]) + af * 1.0)
            else:
                if rate_mode_unit != 0:
                    rate_fast[idx] = np.float32(0.0)
                else:
                    rate_fast[idx] = np.float32(1.0)
            t_fast[idx] = ti

            ts = t_slow[idx]
            if ts > 0:
                dts = float(np.uint64(ti - ts))
                aslow = 1.0 - np.exp(-dts / tau_slow_ticks)
                if rate_mode_unit != 0:
                    inst_slow = 1.0 / max(1.0, dts)
                    rate_slow[idx] = np.float32((1.0 - aslow) * float(rate_slow[idx]) + aslow * inst_slow)
                else:
                    rate_slow[idx] = np.float32((1.0 - aslow) * float(rate_slow[idx]) + aslow * 1.0)
            else:
                if rate_mode_unit != 0:
                    rate_slow[idx] = np.float32(0.0)
                else:
                    rate_slow[idx] = np.float32(1.0)
            t_slow[idx] = ti

            arr = last_pos if int(p[i]) > 0 else last_neg

            # Same-polarity local support in short window.
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
                    if dtj > np.uint64(support_dt_ticks):
                        continue

                    w_t = 1.0 - float(dtj) * inv_support_dt
                    if w_t <= 0.0:
                        continue
                    w_sp = 1.0 / (1.0 + float(d2))
                    support += w_t * w_sp

            if rate_mode_unit != 0:
                support_term = support * inv_support_dt
            else:
                support_term = support
            num = support_gain * support_term + fast_gain * float(rate_fast[idx])
            den = float(rate_slow[idx]) + eps
            score = np.log((num + eps) / den)
            out[i] = np.float32(score)

            arr[idx] = ti

        return out

    return _kernel


def score_stream_n5(
    ev,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_us: int,
    tb: TimeBase,
    scores_out: np.ndarray | None = None,
) -> np.ndarray:
    """N5: dual-timescale background rate ratio with local support.

    Real-time, one-pass, O(r^2) per event.

    Keep a fast/slow per-pixel rate state and combine with short-window
    same-polarity neighborhood support:

    score_i = log((g_s * (S_i / dt_support) + g_f * lambda_fast_i + eps) / (lambda_slow_i + eps))

    Env vars:
    - MYEVS_N5_SUPPORT_DT_US (default min(tau_us, 32000))
    - MYEVS_N5_TAU_FAST_US (default 16000)
    - MYEVS_N5_TAU_SLOW_US (default 128000)
    - MYEVS_N5_SUPPORT_GAIN (default 1.00)
    - MYEVS_N5_FAST_GAIN (default 0.60)
    - MYEVS_N5_RATE_MODE (`count` default, optional `unit`)

    Notes:
    - `count` mode keeps previous n5 behavior (empirically stronger baseline in this repo).
    - `unit` mode uses unit-consistent events/tick rates and support-rate normalization.
    """

    n = int(ev.t.shape[0])
    if scores_out is None:
        scores_out = np.empty((n,), dtype=np.float32)
    if n <= 0:
        return scores_out

    support_dt_us = int(_read_int_env("MYEVS_N5_SUPPORT_DT_US", min(int(tau_us), 32000)))
    tau_fast_us = int(_read_int_env("MYEVS_N5_TAU_FAST_US", 16000))
    tau_slow_us = int(_read_int_env("MYEVS_N5_TAU_SLOW_US", 128000))

    support_dt_ticks = int(tb.us_to_ticks(max(1, support_dt_us)))
    tau_fast_ticks = float(tb.us_to_ticks(max(1, tau_fast_us)))
    tau_slow_ticks = float(tb.us_to_ticks(max(1, tau_slow_us)))

    support_gain = float(_read_float_env("MYEVS_N5_SUPPORT_GAIN", 1.00))
    fast_gain = float(_read_float_env("MYEVS_N5_FAST_GAIN", 0.60))
    rate_mode_s = str(os.environ.get("MYEVS_N5_RATE_MODE", "count")).strip().lower()
    rate_mode_unit = 1 if rate_mode_s in {"unit", "units", "rate", "unitrate"} else 0

    ker = _try_build_n5_kernel()
    scores = ker(
        ev.t.astype(np.uint64, copy=False),
        ev.x.astype(np.int32, copy=False),
        ev.y.astype(np.int32, copy=False),
        ev.p,
        int(width),
        int(height),
        int(radius_px),
        int(max(1, support_dt_ticks)),
        float(max(1.0, tau_fast_ticks)),
        float(max(1.0, tau_slow_ticks)),
        float(max(0.0, support_gain)),
        float(max(0.0, fast_gain)),
        int(rate_mode_unit),
    )
    scores_out[:] = scores
    return scores_out
