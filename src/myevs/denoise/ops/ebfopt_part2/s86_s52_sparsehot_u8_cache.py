from __future__ import annotations

import os

import numpy as np

from ....timebase import TimeBase

try:
    import numba
except Exception:  # pragma: no cover
    numba = None


def _read_int_env(name: str, default: int, min_value: int) -> int:
    s = (os.environ.get(name, "") or "").strip()
    if not s:
        return int(default)
    try:
        v = int(float(s))
    except Exception:
        return int(default)
    if v < min_value:
        return int(min_value)
    return int(v)


def _require_numba() -> None:
    if numba is None:
        raise RuntimeError("s86 requires numba, but import failed")


def _next_pow2(x: int) -> int:
    p = 1
    while p < int(x):
        p <<= 1
    return int(p)


def _try_build_s86_kernel():
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
        cache_cap_pow2: int,
        max_probe: int,
        out: np.ndarray,
    ) -> None:
        n = int(t.shape[0])
        npx = int(width) * int(height)

        last_ts = np.zeros((npx,), dtype=np.uint64)
        last_pol = np.zeros((npx,), dtype=np.int8)

        # Sparse cache for per-pixel hot occupancy: key(pixel id) -> uint8 hot value.
        cap = int(cache_cap_pow2)
        if cap < 1024:
            cap = 1024
        key = np.full((cap,), np.int32(-1), dtype=np.int32)
        val = np.zeros((cap,), dtype=np.uint8)
        mask = cap - 1

        rr = int(radius_px)
        if rr < 0:
            rr = 0
        if rr > 8:
            rr = 8

        tau = int(tau_ticks)
        if tau <= 0:
            tau = 1

        inv_tau = 1.0 / float(tau)
        inv_qtau = 64.0 / float(tau)
        eps = 1e-6

        N = 4096.0
        b = 0.0
        mstate = 0.0

        probe_limit = int(max_probe)
        if probe_limit < 1:
            probe_limit = 1
        if probe_limit > 64:
            probe_limit = 64

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            if xi < 0 or xi >= width or yi < 0 or yi >= height:
                out[i] = np.float32(0.0)
                continue

            ti = int(t[i])
            pi = 1 if int(p[i]) > 0 else -1
            idx0 = yi * width + xi

            ts0 = int(last_ts[idx0])
            dt0 = tau if ts0 == 0 else (ti - ts0)
            if dt0 < 0:
                dt0 = -dt0
            if dt0 > tau:
                dt0 = tau

            # Sparse lookup (open addressing with tombstones).
            home = int((np.uint64(idx0) * np.uint64(2654435761)) & np.uint64(mask))
            slot_found = -1
            slot_empty_or_tomb = -1
            for step in range(probe_limit):
                slot = (home + step) & mask
                k = int(key[slot])
                if k == idx0:
                    slot_found = slot
                    break
                if k == -2:
                    if slot_empty_or_tomb < 0:
                        slot_empty_or_tomb = slot
                    continue
                if k == -1:
                    if slot_empty_or_tomb < 0:
                        slot_empty_or_tomb = slot
                    break

            h0 = 0
            if slot_found >= 0:
                h0 = int(val[slot_found])

            dq = int(float(dt0) * inv_qtau + 0.5)
            if dq < 0:
                dq = 0
            if dq > 64:
                dq = 64

            h0 = h0 - dq
            if h0 < 0:
                h0 = 0

            inc = 64 - dq
            if inc > 0:
                h0 = h0 + inc
                if h0 > 255:
                    h0 = 255

            # Sparse write-back: keep only non-zero states.
            if slot_found >= 0:
                if h0 <= 0:
                    key[slot_found] = np.int32(-2)
                    val[slot_found] = np.uint8(0)
                else:
                    val[slot_found] = np.uint8(h0)
            else:
                if h0 > 0:
                    if slot_empty_or_tomb >= 0:
                        key[slot_empty_or_tomb] = np.int32(idx0)
                        val[slot_empty_or_tomb] = np.uint8(h0)
                    else:
                        # If probe budget fails, replace home slot directly.
                        key[home] = np.int32(idx0)
                        val[home] = np.uint8(h0)

            raw_same = 0
            raw_opp = 0
            cnt_support = 0

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

            for yy in range(y0, y1 + 1):
                base = yy * width
                for xx in range(x0, x1 + 1):
                    if xx == xi and yy == yi:
                        continue

                    idx = base + xx
                    pol_nb = int(last_pol[idx])
                    if pol_nb != pi and pol_nb != -pi:
                        continue

                    ts = int(last_ts[idx])
                    if ts == 0 or ti <= ts:
                        continue

                    dt = ti - ts
                    if dt > tau:
                        continue

                    w_age = tau - dt
                    if w_age <= 0:
                        continue

                    if pol_nb == pi:
                        raw_same += w_age
                        cnt_support += 1
                    else:
                        raw_opp += w_age

            last_ts[idx0] = np.uint64(ti)
            last_pol[idx0] = np.int8(pi)

            u_self = float(h0) / 255.0
            if u_self < 0.0:
                u_self = 0.0
            if u_self > 1.0:
                u_self = 1.0

            b = b + (u_self - b) / N
            if b < 0.0:
                b = 0.0
            if b > 1.0:
                b = 1.0

            denom = float(raw_same + raw_opp)
            mix = 0.0
            if denom > 0.0:
                mix = float(raw_opp) / (denom + eps)
                if mix < 0.0:
                    mix = 0.0
                if mix > 1.0:
                    mix = 1.0

            mstate = mstate + (mix - mstate) / N
            if mstate < 0.0:
                mstate = 0.0
            if mstate > 1.0:
                mstate = 1.0

            alpha_eff = 1.0 - mstate
            if alpha_eff < 0.0:
                alpha_eff = 0.0
            alpha_eff = alpha_eff * alpha_eff

            raw_gated = (float(raw_same) + alpha_eff * float(raw_opp)) * inv_tau
            base_score = raw_gated / (1.0 + u_self * u_self)

            cnt_possible = (x1 - x0 + 1) * (y1 - y0 + 1) - 1
            if cnt_possible <= 0:
                sfrac = 0.0
            else:
                sfrac = float(cnt_support) / float(cnt_possible)
                if sfrac < 0.0:
                    sfrac = 0.0
                if sfrac > 1.0:
                    sfrac = 1.0

            out[i] = np.float32(base_score * (1.0 + b * sfrac))

    return _kernel


def score_stream_s86(
    ev,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_us: int,
    tb: TimeBase,
    scores_out: np.ndarray | None = None,
) -> np.ndarray:
    """S86: S52 with sparse uint8 hot cache (open addressing).

    Keeps S52 fusion logic and replaces dense hot_state table with a sparse cache.
    """

    n = int(ev.t.shape[0])
    if scores_out is None:
        scores_out = np.empty((n,), dtype=np.float32)
    if n <= 0:
        return scores_out

    tau_ticks = int(tb.us_to_ticks(int(tau_us)))

    # Default capacity targets sparse active pixels and avoids dense full-frame storage.
    cap = _read_int_env("MYEVS_S86_CACHE_CAP", 32768, 1024)
    cap_pow2 = _next_pow2(cap)
    probe = _read_int_env("MYEVS_S86_MAX_PROBE", 12, 1)

    ker = _try_build_s86_kernel()
    ker(
        ev.t.astype(np.uint64, copy=False),
        ev.x.astype(np.int32, copy=False),
        ev.y.astype(np.int32, copy=False),
        ev.p,
        int(width),
        int(height),
        int(radius_px),
        int(tau_ticks),
        int(cap_pow2),
        int(probe),
        scores_out,
    )
    return scores_out
