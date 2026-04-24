from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def is_numba_available() -> bool:
    try:
        import numba  # noqa: F401

        return True
    except Exception:
        return False


def _require_numba():
    try:
        from numba import njit  # type: ignore

        return njit
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Numba is not available. Install it (conda-forge: numba) or use --engine python. "
            f"({type(e).__name__}: {e})"
        )


def _build_ts_kernel():
    njit = _require_numba()

    @njit(cache=True)
    def ts_keep_mask(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        width: int,
        height: int,
        show_on: bool,
        show_off: bool,
        radius_px: int,
        decay_ticks: int,
        threshold: float,
        pos_ts: np.ndarray,
        neg_ts: np.ndarray,
    ) -> np.ndarray:
        n = int(t.shape[0])
        keep = np.zeros((n,), dtype=np.uint8)

        w = int(width)
        h = int(height)

        rr = int(radius_px)
        if rr < 0:
            rr = 0
        if rr > 8:
            rr = 8

        dtau = int(decay_ticks)
        thr = float(threshold)

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            if xi < 0 or xi >= w or yi < 0 or yi >= h:
                continue

            pi = 1 if int(p[i]) > 0 else -1
            if pi > 0:
                if not show_on:
                    continue
                cell = pos_ts
            else:
                if not show_off:
                    continue
                cell = neg_ts

            ti = int(t[i])
            idx0 = yi * w + xi

            if dtau <= 0:
                cell[idx0] = np.uint64(ti)
                if thr <= 0.0:
                    keep[i] = 1
                continue

            y0 = yi - rr
            if y0 < 0:
                y0 = 0
            y1 = yi + rr
            if y1 >= h:
                y1 = h - 1
            x0 = xi - rr
            if x0 < 0:
                x0 = 0
            x1 = xi + rr
            if x1 >= w:
                x1 = w - 1

            inv_decay = 1.0 / float(dtau)
            surf = 0.0
            support = 0
            for yy in range(y0, y1 + 1):
                base = yy * w
                for xx in range(x0, x1 + 1):
                    ts = int(cell[base + xx])
                    if ts == 0:
                        continue
                    dt = ti - ts
                    if dt < 0:
                        dt = -dt
                    surf += math.exp(-float(dt) * inv_decay)
                    support += 1

            score = (surf / float(support)) if support > 0 else 0.0
            cell[idx0] = np.uint64(ti)
            if score >= thr:
                keep[i] = 1

        return keep

    return ts_keep_mask


_TS_KEEP_MASK = None


def ts_keep_mask_numba(
    *,
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    width: int,
    height: int,
    show_on: bool,
    show_off: bool,
    radius_px: int,
    decay_ticks: int,
    threshold: float,
    pos_ts: np.ndarray,
    neg_ts: np.ndarray,
) -> np.ndarray:
    global _TS_KEEP_MASK
    if _TS_KEEP_MASK is None:
        _TS_KEEP_MASK = _build_ts_kernel()

    return _TS_KEEP_MASK(
        t,
        x,
        y,
        p,
        int(width),
        int(height),
        bool(show_on),
        bool(show_off),
        int(radius_px),
        int(decay_ticks),
        float(threshold),
        pos_ts,
        neg_ts,
    )


def ts_state_init(width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    n = int(width) * int(height)
    return np.zeros((n,), dtype=np.uint64), np.zeros((n,), dtype=np.uint64)

