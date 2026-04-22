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
        raise RuntimeError("s87 requires numba, but import failed")


def _try_build_s87_kernel():
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
        block_size_px: int,
        out: np.ndarray,
    ) -> None:
        n = int(t.shape[0])
        npx = int(width) * int(height)

        last_ts = np.zeros((npx,), dtype=np.uint64)
        last_pol = np.zeros((npx,), dtype=np.int8)
        hot_state = np.zeros((npx,), dtype=np.int32)

        rr = int(radius_px)
        if rr < 0:
            rr = 0
        if rr > 8:
            rr = 8

        tau = int(tau_ticks)
        if tau <= 0:
            tau = 1
        inv_tau = 1.0 / float(tau)
        eps = 1e-6
        acc_max = 2147483647

        # Keep beta_state global as s52; only mix_state is block-wise.
        beta_state = 0.0
        N = 4096.0

        tr = tau // 2
        if tr <= 0:
            tr = 1

        bs = int(block_size_px)
        if bs < 4:
            bs = 4
        bx_n = (int(width) + bs - 1) // bs
        by_n = (int(height) + bs - 1) // bs
        mix_blocks = np.zeros((bx_n * by_n,), dtype=np.float32)

        if rr <= 0:
            for i in range(n):
                xi = int(x[i])
                yi = int(y[i])
                ti = int(t[i])
                if xi < 0 or xi >= width or yi < 0 or yi >= height:
                    out[i] = np.float32(0.0)
                    continue
                pi = 1 if int(p[i]) > 0 else -1
                idx0 = yi * width + xi
                last_ts[idx0] = np.uint64(ti)
                last_pol[idx0] = np.int8(pi)
                out[i] = np.float32(np.inf)
            return

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            ti = int(t[i])

            if xi < 0 or xi >= width or yi < 0 or yi >= height:
                out[i] = np.float32(0.0)
                continue

            pi = 1 if int(p[i]) > 0 else -1
            idx0 = yi * width + xi

            ts0 = int(last_ts[idx0])
            h0 = int(hot_state[idx0])

            dt0 = tau if ts0 == 0 else (ti - ts0)
            if dt0 < 0:
                dt0 = -dt0

            if dt0 != 0:
                h0 = h0 - dt0
                if h0 < 0:
                    h0 = 0

            inc = tau - dt0
            if inc > 0:
                h0 = h0 + inc
                if h0 > acc_max:
                    h0 = acc_max

            raw_w = 0
            opp_w = 0
            cnt_support = 0

            y0 = yi - rr
            if y0 < 0:
                y0 = 0
            y1 = yi + rr
            if y1 >= height:
                y1 = height - 1

            x0 = xi - rr
            if x0 < 0:
                x0 = 0
            x1 = xi + rr
            if x1 >= width:
                x1 = width - 1

            for yy in range(y0, y1 + 1):
                base = yy * width
                for xx in range(x0, x1 + 1):
                    if xx == xi and yy == yi:
                        continue

                    idx = base + xx
                    pol = int(last_pol[idx])
                    if pol != pi and pol != -pi:
                        continue

                    ts = int(last_ts[idx])
                    if ts == 0:
                        continue

                    dt = ti - ts
                    if dt < 0:
                        dt = -dt
                    if dt > tau:
                        continue

                    w_age = tau - dt
                    if w_age <= 0:
                        continue

                    if pol == pi:
                        raw_w += w_age
                        cnt_support += 1
                    else:
                        opp_w += w_age

            last_ts[idx0] = np.uint64(ti)
            last_pol[idx0] = np.int8(pi)
            hot_state[idx0] = np.int32(h0)

            hf = float(h0)
            u_self = hf / (hf + float(tr) + eps)
            if u_self < 0.0:
                u_self = 0.0
            if u_self > 1.0:
                u_self = 1.0

            beta_state = beta_state + (u_self - beta_state) / N
            if beta_state < 0.0:
                beta_state = 0.0
            if beta_state > 1.0:
                beta_state = 1.0

            denom = float(raw_w + opp_w)
            mix = 0.0
            if denom > 0.0:
                mix = float(opp_w) / denom
                if mix < 0.0:
                    mix = 0.0
                if mix > 1.0:
                    mix = 1.0

            bx = xi // bs
            by = yi // bs
            bidx = by * bx_n + bx
            mstate = float(mix_blocks[bidx])
            mstate = mstate + (mix - mstate) / N
            if mstate < 0.0:
                mstate = 0.0
            if mstate > 1.0:
                mstate = 1.0
            mix_blocks[bidx] = np.float32(mstate)

            alpha_eff = 1.0 - mstate
            if alpha_eff < 0.0:
                alpha_eff = 0.0
            alpha_eff = alpha_eff * alpha_eff

            raw_gated = (float(raw_w) + float(alpha_eff) * float(opp_w)) * inv_tau
            base_score = float(raw_gated / (1.0 + (u_self * u_self)))

            cnt_possible = (x1 - x0 + 1) * (y1 - y0 + 1) - 1
            if cnt_possible <= 0:
                sfrac = 0.0
            else:
                sfrac = float(cnt_support) / float(cnt_possible)
                if sfrac < 0.0:
                    sfrac = 0.0
                if sfrac > 1.0:
                    sfrac = 1.0

            out[i] = np.float32(base_score * (1.0 + beta_state * sfrac))

    return _kernel


def score_stream_s87(
    ev,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_us: int,
    tb: TimeBase,
    scores_out: np.ndarray,
) -> np.ndarray:
    """s87: S52 with block-wise mix_state (only this path is changed)."""

    _require_numba()
    tau_ticks = int(tb.us_to_ticks(int(tau_us)))
    if tau_ticks <= 0:
        tau_ticks = 1

    block_size_px = _read_int_env("MYEVS_S87_BLOCK_SIZE", 32, 4)

    ker = _try_build_s87_kernel()
    ker(
        ev.t,
        ev.x,
        ev.y,
        ev.p,
        int(width),
        int(height),
        int(radius_px),
        int(tau_ticks),
        int(block_size_px),
        scores_out,
    )
    return scores_out


__all__ = ["score_stream_s87"]
