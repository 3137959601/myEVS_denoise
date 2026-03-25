from __future__ import annotations

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
            "Numba is not available. Install it (conda-forge: numba) or use the pure-Python backend. "
            f"({type(e).__name__}: {e})"
        )


def _build_ebf_kernel():
    njit = _require_numba()

    @njit(cache=True)
    def ebf_update_prefix_diffs(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        signal_u8: np.ndarray,
        width: int,
        height: int,
        radius_px: int,
        tau_ticks: int,
        thresholds: np.ndarray,
        last_ts: np.ndarray,
        last_pol: np.ndarray,
        diff_total: np.ndarray,
        diff_signal: np.ndarray,
    ) -> None:
        """Update prefix-diff counters for an EBF score-threshold sweep.

        This processes events sequentially, updates EBF state (last_ts/last_pol)
        regardless of keep/drop, and updates diff_total/diff_signal such that:
            kept_total[j]  = cumsum(diff_total)[j]
            kept_signal[j] = cumsum(diff_signal)[j]
        where threshold j corresponds to thresholds[j] and the keep rule is:
            keep := (score > threshold)
        """

        n = int(t.shape[0])
        w = int(width)
        h = int(height)

        rr = int(radius_px)
        if rr < 0:
            rr = 0
        if rr > 8:
            rr = 8

        tau = int(tau_ticks)
        k = int(thresholds.shape[0])

        # Pass-through (still updates state): score is effectively +inf, so idx=k.
        if rr <= 0 or tau <= 0:
            for i in range(n):
                xi = int(x[i])
                yi = int(y[i])
                if xi < 0 or xi >= w or yi < 0 or yi >= h:
                    continue

                ti = int(t[i])
                pi = 1 if int(p[i]) > 0 else -1

                idx0 = yi * w + xi
                last_ts[idx0] = np.uint64(ti)
                last_pol[idx0] = np.int8(pi)

                # idx=k => kept for all thresholds
                diff_total[0] += 1
                diff_total[k] -= 1
                if signal_u8[i] != 0:
                    diff_signal[0] += 1
                    diff_signal[k] -= 1
            return

        inv_tau = 1.0 / float(tau)

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            if xi < 0 or xi >= w or yi < 0 or yi >= h:
                continue

            ti = int(t[i])
            pi = 1 if int(p[i]) > 0 else -1

            idx0 = yi * w + xi

            score = 0.0

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

            for yy in range(y0, y1 + 1):
                base = yy * w
                for xx in range(x0, x1 + 1):
                    if xx == xi and yy == yi:
                        continue

                    idx = base + xx

                    if int(last_pol[idx]) != pi:
                        continue

                    ts = int(last_ts[idx])
                    if ts == 0:
                        continue

                    dt = (ti - ts) if ti >= ts else (ts - ti)
                    if dt > tau:
                        continue

                    score += (float(tau - dt) * inv_tau)

            # Always update self state
            last_ts[idx0] = np.uint64(ti)
            last_pol[idx0] = np.int8(pi)

            # idx = number of thresholds < score (bisect_left)
            lo = 0
            hi = k
            while lo < hi:
                mid = (lo + hi) // 2
                if thresholds[mid] < score:
                    lo = mid + 1
                else:
                    hi = mid
            idx = lo

            if idx <= 0:
                continue

            diff_total[0] += 1
            diff_total[idx] -= 1
            if signal_u8[i] != 0:
                diff_signal[0] += 1
                diff_signal[idx] -= 1

    return ebf_update_prefix_diffs


_EBF_UPDATE_PREFIX_DIFFS = None


def _build_ebf_scores_kernel():
    njit = _require_numba()

    @njit(cache=True)
    def ebf_scores_stream(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        width: int,
        height: int,
        radius_px: int,
        tau_ticks: int,
        last_ts: np.ndarray,
        last_pol: np.ndarray,
        scores_out: np.ndarray,
    ) -> None:
        """Compute EBF score for each event in a stream (Numba).

        - Processes events sequentially.
        - Updates last_ts/last_pol regardless of keep/drop (matches reference).
        - Writes the continuous score to scores_out.
        """

        n = int(t.shape[0])
        w = int(width)
        h = int(height)

        rr = int(radius_px)
        if rr < 0:
            rr = 0
        if rr > 8:
            rr = 8

        tau = int(tau_ticks)

        # Pass-through (still updates state): score is effectively +inf.
        if rr <= 0 or tau <= 0:
            for i in range(n):
                xi = int(x[i])
                yi = int(y[i])
                if xi < 0 or xi >= w or yi < 0 or yi >= h:
                    scores_out[i] = 0.0
                    continue

                ti = int(t[i])
                pi = 1 if int(p[i]) > 0 else -1

                idx0 = yi * w + xi
                last_ts[idx0] = np.uint64(ti)
                last_pol[idx0] = np.int8(pi)

                scores_out[i] = np.inf
            return

        inv_tau = 1.0 / float(tau)

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            if xi < 0 or xi >= w or yi < 0 or yi >= h:
                scores_out[i] = 0.0
                continue

            ti = int(t[i])
            pi = 1 if int(p[i]) > 0 else -1

            idx0 = yi * w + xi

            score = 0.0

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

            for yy in range(y0, y1 + 1):
                base = yy * w
                for xx in range(x0, x1 + 1):
                    if xx == xi and yy == yi:
                        continue

                    idx = base + xx

                    if int(last_pol[idx]) != pi:
                        continue

                    ts = int(last_ts[idx])
                    if ts == 0:
                        continue

                    dt = (ti - ts) if ti >= ts else (ts - ti)
                    if dt > tau:
                        continue

                    score += (float(tau - dt) * inv_tau)

            # Always update self state
            last_ts[idx0] = np.uint64(ti)
            last_pol[idx0] = np.int8(pi)

            scores_out[i] = score

    return ebf_scores_stream


_EBF_SCORES_STREAM = None


def ebf_update_prefix_diffs_numba(
    *,
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    signal_u8: np.ndarray,
    width: int,
    height: int,
    radius_px: int,
    tau_ticks: int,
    thresholds: np.ndarray,
    last_ts: np.ndarray,
    last_pol: np.ndarray,
    diff_total: np.ndarray,
    diff_signal: np.ndarray,
) -> None:
    global _EBF_UPDATE_PREFIX_DIFFS
    if _EBF_UPDATE_PREFIX_DIFFS is None:
        _EBF_UPDATE_PREFIX_DIFFS = _build_ebf_kernel()

    _EBF_UPDATE_PREFIX_DIFFS(
        t,
        x,
        y,
        p,
        signal_u8,
        int(width),
        int(height),
        int(radius_px),
        int(tau_ticks),
        thresholds,
        last_ts,
        last_pol,
        diff_total,
        diff_signal,
    )


def ebf_scores_stream_numba(
    *,
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    width: int,
    height: int,
    radius_px: int,
    tau_ticks: int,
    last_ts: np.ndarray,
    last_pol: np.ndarray,
    scores_out: np.ndarray,
) -> None:
    global _EBF_SCORES_STREAM
    if _EBF_SCORES_STREAM is None:
        _EBF_SCORES_STREAM = _build_ebf_scores_kernel()

    _EBF_SCORES_STREAM(
        t,
        x,
        y,
        p,
        int(width),
        int(height),
        int(radius_px),
        int(tau_ticks),
        last_ts,
        last_pol,
        scores_out,
    )


def ebf_state_init(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    n = int(width) * int(height)
    return np.zeros((n,), dtype=np.uint64), np.zeros((n,), dtype=np.int8)
