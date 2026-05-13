from __future__ import annotations

import numpy as np

from ....timebase import TimeBase

try:
    import numba
except Exception:  # pragma: no cover
    numba = None


N177_SIGMA = 2.8
N177_BETA_INIT = 0.65
N177_K_SFRAC = 4.0 / 5.0
N177_K_MIX = 1.0 / 8.0
N177_RSTATE_INIT = 0.10

_N177_KERNEL = None
_N177_OFFSET_CACHE: dict[tuple[int, float], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _require_numba() -> None:
    if numba is None:
        raise RuntimeError("n177 requires numba, but import failed")


def _build_sparse_star_offsets(radius_px: int, sigma_space: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rr = max(0, min(8, int(radius_px)))
    sig = max(1e-6, float(sigma_space))
    inv_2sig2 = 1.0 / (2.0 * sig * sig)

    offsets: list[tuple[int, int]] = []
    if rr <= 0:
        return (
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.float32),
        )

    # Near field: keep all 8 immediate neighbors.
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx != 0 or dy != 0:
                offsets.append((dx, dy))

    # Far field: sample axis rays at mid/end radius and one diagonal ring.
    radii = sorted({max(2, rr // 2), rr})
    for u in radii:
        if u <= rr:
            offsets.extend([(u, 0), (-u, 0), (0, u), (0, -u)])

    d = min(2, rr)
    if d >= 2:
        offsets.extend([(d, d), (d, -d), (-d, d), (-d, -d)])

    # Remove duplicates while preserving deterministic order.
    seen: set[tuple[int, int]] = set()
    dxs: list[int] = []
    dys: list[int] = []
    ws: list[float] = []
    for dx, dy in offsets:
        if (dx, dy) in seen:
            continue
        seen.add((dx, dy))
        d2 = dx * dx + dy * dy
        dxs.append(dx)
        dys.append(dy)
        ws.append(float(np.exp(-float(d2) * inv_2sig2)))

    return (
        np.asarray(dxs, dtype=np.int32),
        np.asarray(dys, dtype=np.int32),
        np.asarray(ws, dtype=np.float32),
    )


def _try_build_n177_kernel():
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
        off_dx: np.ndarray,
        off_dy: np.ndarray,
        off_w: np.ndarray,
        out: np.ndarray,
    ) -> None:
        n = int(t.shape[0])
        npx = int(width) * int(height)
        last_ts = np.zeros((npx,), dtype=np.uint64)
        last_pol = np.zeros((npx,), dtype=np.int8)

        tau_base = int(tau_base_ticks)
        if tau_base <= 0:
            tau_base = 1
        inv_tau = 1.0 / float(tau_base)
        sample_count = int(off_dx.shape[0])
        eps = 1e-6

        b = float(N177_BETA_INIT)
        mstate = 0.0
        rstate = float(N177_RSTATE_INIT)

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            if xi < 0 or xi >= width or yi < 0 or yi >= height:
                out[i] = np.float32(0.0)
                continue

            ti = int(t[i])
            pi = 1 if int(p[i]) > 0 else -1
            idx0 = yi * width + xi

            raw_same = 0.0
            raw_opp = 0.0
            cnt_support = 0

            for k in range(sample_count):
                nx = xi + int(off_dx[k])
                ny = yi + int(off_dy[k])
                if nx < 0 or nx >= width or ny < 0 or ny >= height:
                    continue
                idx = ny * width + nx
                pol_nb = int(last_pol[idx])
                if pol_nb != pi and pol_nb != -pi:
                    continue
                ts = int(last_ts[idx])
                if ts == 0 or ti <= ts:
                    continue
                dt = ti - ts
                if dt > tau_base:
                    continue
                base_time = 1.0 - float(dt) * inv_tau
                if base_time <= 0.0:
                    continue
                wst = base_time * base_time * float(off_w[k])
                if pol_nb == pi:
                    raw_same += wst
                    cnt_support += 1
                else:
                    raw_opp += wst

            prev_pol0 = int(last_pol[idx0])
            ts0 = int(last_ts[idx0])
            dt0 = tau_base
            if ts0 != 0:
                dt0 = ti - ts0
                if dt0 < 0:
                    dt0 = -dt0
                if dt0 > tau_base:
                    dt0 = tau_base

            mix = 0.0
            denom_mix = raw_same + raw_opp
            if denom_mix > 0.0:
                mix = raw_opp / (denom_mix + eps)
                if mix < 0.0:
                    mix = 0.0
                if mix > 1.0:
                    mix = 1.0

            mstate = mstate + (mix - mstate) / 4096.0
            if mstate < 0.0:
                mstate = 0.0
            if mstate > 1.0:
                mstate = 1.0
            alpha_eff = 1.0 - mstate
            if alpha_eff < 0.0:
                alpha_eff = 0.0
            alpha_eff = alpha_eff * alpha_eff

            u_lite = 1.0 - float(dt0) * inv_tau
            if u_lite < 0.0:
                u_lite = 0.0
            if u_lite > 1.0:
                u_lite = 1.0

            rhythm_bad = 0.0
            rhythm_good = 0.0
            if prev_pol0 == -pi:
                rhythm_bad = u_lite
            elif prev_pol0 == pi:
                rhythm_good = u_lite

            rstate = rstate + (rhythm_bad - rstate) / 4096.0
            if rstate < 0.0:
                rstate = 0.0
            if rstate > 1.0:
                rstate = 1.0
            rhythm_pressure = 0.5 * (rhythm_bad + rstate)
            if rhythm_pressure < 0.0:
                rhythm_pressure = 0.0
            if rhythm_pressure > 1.0:
                rhythm_pressure = 1.0

            sfrac = 0.0
            if sample_count > 0:
                sfrac = float(cnt_support) / float(sample_count)
                if sfrac < 0.0:
                    sfrac = 0.0
                if sfrac > 1.0:
                    sfrac = 1.0

            alpha_eff = alpha_eff * (1.0 - rhythm_pressure / 3.0)
            if alpha_eff < 0.0:
                alpha_eff = 0.0

            relief = 1.0 - N177_K_SFRAC * sfrac
            if relief < 0.25:
                relief = 0.25
            if relief > 1.0:
                relief = 1.0
            mix_gain = 1.0 + N177_K_MIX * mix
            if mix_gain < 0.5:
                mix_gain = 0.5
            if mix_gain > 2.0:
                mix_gain = 2.0
            rhythm_scale = 1.0 + 0.5 * rhythm_pressure - 0.25 * rhythm_good
            if rhythm_scale < 0.5:
                rhythm_scale = 0.5
            if rhythm_scale > 1.75:
                rhythm_scale = 1.75
            u_eff = u_lite * relief * mix_gain * rhythm_scale
            if u_eff < 0.0:
                u_eff = 0.0
            if u_eff > 1.0:
                u_eff = 1.0

            b = b + (u_eff - b) / 4096.0
            if b < 0.0:
                b = 0.0
            if b > 1.0:
                b = 1.0

            raw_gated = raw_same + alpha_eff * raw_opp
            base_score = raw_gated / (1.0 + u_eff * u_eff)
            support_scale = 1.0 + b * sfrac * (1.0 + 0.25 * rhythm_good)
            if support_scale < 1.0:
                support_scale = 1.0
            if support_scale > 2.0:
                support_scale = 2.0
            out[i] = np.float32(base_score * support_scale)

            last_ts[idx0] = np.uint64(ti)
            last_pol[idx0] = np.int8(pi)

    return _kernel


def _get_n177_kernel():
    global _N177_KERNEL
    if _N177_KERNEL is None:
        _N177_KERNEL = _try_build_n177_kernel()
    return _N177_KERNEL


def _get_offsets_cached(radius_px: int, sigma_space: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rr = max(0, min(8, int(radius_px)))
    sig = max(1e-6, float(sigma_space))
    key = (rr, sig)
    cached = _N177_OFFSET_CACHE.get(key)
    if cached is not None:
        return cached
    cached = _build_sparse_star_offsets(rr, sig)
    _N177_OFFSET_CACHE[key] = cached
    return cached


def score_stream_n177(
    ev,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_us: int,
    tb: TimeBase,
    scores_out: np.ndarray | None = None,
) -> np.ndarray:
    """N177: sparse-star n176 for realtime-oriented scoring.

    It preserves n176's polarity/rhythm/support gate but replaces the dense
    square neighborhood with a deterministic 16-20 point star stencil.
    """

    n = int(ev.t.shape[0])
    if scores_out is None:
        scores_out = np.empty((n,), dtype=np.float32)
    if n <= 0:
        return scores_out

    tau_base_ticks = int(tb.us_to_ticks(int(tau_us)))
    off_dx, off_dy, off_w = _get_offsets_cached(int(radius_px), float(N177_SIGMA))
    ker = _get_n177_kernel()
    ker(
        ev.t.astype(np.uint64, copy=False),
        ev.x.astype(np.int32, copy=False),
        ev.y.astype(np.int32, copy=False),
        ev.p,
        int(width),
        int(height),
        int(radius_px),
        int(tau_base_ticks),
        off_dx,
        off_dy,
        off_w,
        scores_out,
    )
    return scores_out
