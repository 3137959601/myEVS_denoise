from __future__ import annotations

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


def _build_evflow_kernel():
    njit = _require_numba()

    @njit(cache=True)
    def evflow_keep_mask_and_tail(
        t_all: np.ndarray,
        x_all: np.ndarray,
        y_all: np.ndarray,
        start_idx: int,
        radius_px: int,
        win_ticks: int,
        threshold: float,
    ) -> tuple[np.ndarray, int]:
        n = int(t_all.shape[0])
        s = int(start_idx)
        if s < 0:
            s = 0
        if s > n:
            s = n

        m = n - s
        keep = np.zeros((m,), dtype=np.uint8)

        rr = int(radius_px)
        if rr < 1:
            rr = 1
        if rr > 8:
            rr = 8
        win = int(win_ticks)
        thr = float(threshold)

        # Match EvFlowOp semantics:
        # 1) fit flow from current deque
        # 2) prune deque by current t
        # 3) append current event
        # Here [j0, i) is the deque visible to event i before pruning by i.
        j0 = 0
        for i in range(s, n):
            ti = int(t_all[i])
            xi = int(x_all[i])
            yi = int(y_all[i])

            count = 0
            for j in range(j0, i):
                xj = int(x_all[j])
                yj = int(y_all[j])
                if abs(xi - xj) > rr or abs(yi - yj) > rr:
                    continue
                count += 1

            flow = np.inf
            if count > 3:
                a_mat = np.empty((count, 3), dtype=np.float64)
                b_vec = np.empty((count,), dtype=np.float64)
                k = 0
                for j in range(j0, i):
                    xj = int(x_all[j])
                    yj = int(y_all[j])
                    if abs(xi - xj) > rr or abs(yi - yj) > rr:
                        continue
                    tj = int(t_all[j])
                    a_mat[k, 0] = float(xj)
                    a_mat[k, 1] = float(yj)
                    a_mat[k, 2] = 1.0
                    b_vec[k] = float(tj - ti) * 1.0e-3
                    k += 1

                # Align with numpy lstsq(..., rcond=None) used in python EvFlowOp.
                # numpy uses eps * max(M, N) when rcond=None.
                rcond = np.finfo(np.float64).eps * float(count if count > 3 else 3)
                sol, _, _, _ = np.linalg.lstsq(a_mat, b_vec, rcond=rcond)
                ax = float(sol[0])
                ay = float(sol[1])
                if abs(ax) > 1e-12 and abs(ay) > 1e-12:
                    invx = -1.0 / ax
                    invy = -1.0 / ay
                    flow = float((invx * invx + invy * invy) ** 0.5)

            if flow <= thr:
                keep[i - s] = 1

            # Prune after fitting, then current event is implicitly appended
            # by advancing i in the next iteration.
            if win > 0:
                while j0 < i:
                    dt0 = ti - int(t_all[j0])
                    if dt0 < 0:
                        dt0 = -dt0
                    if dt0 >= win:
                        j0 += 1
                    else:
                        break
            else:
                # clear deque, then append current => next deque starts at i
                j0 = i

        if n <= 0:
            return keep, 0
        if win <= 0:
            # queue keeps only the final event after processing
            return keep, n - 1

        # j0 now points at final deque head after the loop.
        return keep, j0

    return evflow_keep_mask_and_tail


_EVFLOW_KEEP_AND_TAIL = None


def evflow_keep_mask_numba(
    *,
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    radius_px: int,
    win_ticks: int,
    threshold: float,
    prev_t: np.ndarray,
    prev_x: np.ndarray,
    prev_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    global _EVFLOW_KEEP_AND_TAIL
    if _EVFLOW_KEEP_AND_TAIL is None:
        _EVFLOW_KEEP_AND_TAIL = _build_evflow_kernel()

    t_cur = np.asarray(t, dtype=np.uint64)
    x_cur = np.asarray(x, dtype=np.int32)
    y_cur = np.asarray(y, dtype=np.int32)
    t_prev = np.asarray(prev_t, dtype=np.uint64)
    x_prev = np.asarray(prev_x, dtype=np.int32)
    y_prev = np.asarray(prev_y, dtype=np.int32)

    if t_prev.size > 0:
        t_all = np.concatenate([t_prev, t_cur], axis=0)
        x_all = np.concatenate([x_prev, x_cur], axis=0)
        y_all = np.concatenate([y_prev, y_cur], axis=0)
        start_idx = int(t_prev.shape[0])
    else:
        t_all = t_cur
        x_all = x_cur
        y_all = y_cur
        start_idx = 0

    keep_u8, tail_start = _EVFLOW_KEEP_AND_TAIL(
        t_all=t_all,
        x_all=x_all,
        y_all=y_all,
        start_idx=int(start_idx),
        radius_px=int(radius_px),
        win_ticks=int(win_ticks),
        threshold=float(threshold),
    )

    if tail_start < 0:
        tail_start = 0
    if tail_start > int(t_all.shape[0]):
        tail_start = int(t_all.shape[0])

    next_t = t_all[tail_start:].copy()
    next_x = x_all[tail_start:].copy()
    next_y = y_all[tail_start:].copy()

    return keep_u8, next_t, next_x, next_y


def evflow_state_init() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.empty((0,), dtype=np.uint64),
        np.empty((0,), dtype=np.int32),
        np.empty((0,), dtype=np.int32),
    )
