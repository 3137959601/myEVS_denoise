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
        raise RuntimeError("n92 requires numba, but import failed")


def _try_build_n92_kernel():
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
        k_use: int,
        k_min: int,
        sigma_d: float,
        out: np.ndarray,
    ) -> None:
        n = int(t.shape[0])
        npx = int(width) * int(height)
        kmax = 8

        # 7.45: keep only one latest timestamp per pixel/polarity.
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

        k = int(k_use)
        if k < 1:
            k = 1
        if k > kmax:
            k = kmax

        km = int(k_min)
        if km < 1:
            km = 1
        if km > k:
            km = k

        sigma = float(sigma_d)
        if sigma <= 0.0:
            sigma = 1.0

        best_ts = np.zeros((kmax,), dtype=np.uint64)
        best_x = np.zeros((kmax,), dtype=np.int32)
        best_y = np.zeros((kmax,), dtype=np.int32)

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            if xi < 0 or xi >= width or yi < 0 or yi >= height:
                out[i] = np.float32(0.0)
                continue

            ti = np.uint64(t[i])
            pi = 1 if int(p[i]) > 0 else -1

            # reset top-k buffers
            for s in range(k):
                best_ts[s] = np.uint64(0)
                best_x[s] = 0
                best_y[s] = 0
            nsel = 0

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

            # Collect top-k recent same-polarity candidates from window.
            # 7.45: each neighbor pixel contributes at most one timestamp.
            for yy in range(y0, y1 + 1):
                base = yy * width
                for xx in range(x0, x1 + 1):
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

                    if nsel < k:
                        best_ts[nsel] = ts
                        best_x[nsel] = xx
                        best_y[nsel] = yy
                        nsel += 1

                        m = nsel - 1
                        while m > 0 and best_ts[m] > best_ts[m - 1]:
                            ttmp = best_ts[m]
                            best_ts[m] = best_ts[m - 1]
                            best_ts[m - 1] = ttmp

                            xtmp = best_x[m]
                            best_x[m] = best_x[m - 1]
                            best_x[m - 1] = xtmp

                            ytmp = best_y[m]
                            best_y[m] = best_y[m - 1]
                            best_y[m - 1] = ytmp
                            m -= 1
                    else:
                        # Keep descending top-k by timestamp.
                        if ts <= best_ts[k - 1]:
                            continue
                        best_ts[k - 1] = ts
                        best_x[k - 1] = xx
                        best_y[k - 1] = yy

                        m = k - 1
                        while m > 0 and best_ts[m] > best_ts[m - 1]:
                            ttmp = best_ts[m]
                            best_ts[m] = best_ts[m - 1]
                            best_ts[m - 1] = ttmp

                            xtmp = best_x[m]
                            best_x[m] = best_x[m - 1]
                            best_x[m - 1] = xtmp

                            ytmp = best_y[m]
                            best_y[m] = best_y[m - 1]
                            best_y[m - 1] = ytmp
                            m -= 1

            score = 0.0
            if nsel >= km:
                sum_dist = 0.0
                # best_* is newest->oldest; iterate old->new along time chain.
                for m in range(nsel - 1, 0, -1):
                    dx = float(best_x[m - 1] - best_x[m])
                    dy = float(best_y[m - 1] - best_y[m])
                    sum_dist += np.sqrt(dx * dx + dy * dy)

                # last chain segment: newest support -> current event
                dx_last = float(xi - best_x[0])
                dy_last = float(yi - best_y[0])
                sum_dist += np.sqrt(dx_last * dx_last + dy_last * dy_last)

                d_bar = sum_dist / float(nsel)
                score = np.exp(-d_bar / sigma)

            out[i] = np.float32(score)

            # Update current event into per-pixel same-polarity latest timestamp.
            idx0 = yi * width + xi
            if pi > 0:
                pos_ts[idx0] = ti
            else:
                neg_ts[idx0] = ti

    return _kernel


def score_stream_n92(
    ev,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_us: int,
    tb: TimeBase,
    scores_out: np.ndarray | None = None,
) -> np.ndarray:
    """N92: local temporal-spatial continuity backbone.

    Independent score axis (without baseline sum): among recent same-polarity
    supports in the local window, the time-ordered chain is scored by spatial
    continuity (smaller mean neighbor distance -> higher score).
    """

    n = int(ev.t.shape[0])
    if scores_out is None:
        scores_out = np.empty((n,), dtype=np.float32)
    if n <= 0:
        return scores_out

    tau_ticks = int(tb.us_to_ticks(int(tau_us)))
    k_use = int(_read_int_env("MYEVS_N92_K", 4))
    k_min = int(_read_int_env("MYEVS_N92_KMIN", 2))
    sigma_d = float(_read_float_env("MYEVS_N92_SIGMA_D", 2.0))

    ker = _try_build_n92_kernel()
    ker(
        ev.t.astype(np.uint64, copy=False),
        ev.x.astype(np.int32, copy=False),
        ev.y.astype(np.int32, copy=False),
        ev.p,
        int(width),
        int(height),
        int(radius_px),
        int(tau_ticks),
        int(k_use),
        int(k_min),
        float(sigma_d),
        scores_out,
    )
    return scores_out
