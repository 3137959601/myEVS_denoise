from __future__ import annotations

import numpy as np

from ....timebase import TimeBase

try:
    import numba
except Exception:  # pragma: no cover
    numba = None


# Geometric-mean gate constants (fixed, no user knobs).
N173_SIGMA = 2.8
N173_RSTATE_INIT = 0.10


def _require_numba() -> None:
    if numba is None:
        raise RuntimeError("n173 requires numba, but import failed")


def _build_compact_kernel_tables(radius_px: int, sigma_space: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rr = int(radius_px)
    if rr < 0:
        rr = 0
    if rr > 8:
        rr = 8

    sig = float(sigma_space)
    if sig <= 1e-6:
        sig = 1e-6
    inv_2sig2 = 1.0 / (2.0 * sig * sig)

    axis_u = np.arange(1, rr + 1, dtype=np.int32)
    diag_u = np.arange(1, rr + 1, dtype=np.int32)

    lut_axis = np.zeros((rr,), dtype=np.float32)
    lut_diag = np.zeros((rr,), dtype=np.float32)
    for k in range(rr):
        u = int(k + 1)
        lut_axis[k] = np.float32(np.exp(-float(u * u) * inv_2sig2))
        lut_diag[k] = np.float32(np.exp(-float(2 * u * u) * inv_2sig2))

    int_u_list: list[int] = []
    int_v_list: list[int] = []
    lut_int_list: list[float] = []
    for u in range(2, rr + 1):
        for v in range(1, u):
            int_u_list.append(int(u))
            int_v_list.append(int(v))
            lut_int_list.append(float(np.exp(-float(u * u + v * v) * inv_2sig2)))

    int_u = np.asarray(int_u_list, dtype=np.int32)
    int_v = np.asarray(int_v_list, dtype=np.int32)
    lut_int = np.asarray(lut_int_list, dtype=np.float32)

    return axis_u, diag_u, int_u, int_v, lut_int if rr >= 2 else np.zeros((0,), dtype=np.float32)


def _try_build_n173_kernel():
    _require_numba()

    @numba.njit(inline="always", cache=True)
    def _acc_neighbor(
        idx_nb: int,
        ti: int,
        pi: int,
        tau_base: int,
        inv_tau_base: float,
        w_space: float,
        last_ts: np.ndarray,
        last_pol: np.ndarray,
    ) -> tuple[float, float, int]:
        pol_nb = int(last_pol[idx_nb])
        if pol_nb != pi and pol_nb != -pi:
            return 0.0, 0.0, 0

        ts = int(last_ts[idx_nb])
        if ts == 0 or ti <= ts:
            return 0.0, 0.0, 0

        dt_ticks = ti - ts
        if dt_ticks > tau_base:
            return 0.0, 0.0, 0

        ratio = float(dt_ticks) * inv_tau_base
        base_time = 1.0 - ratio
        if base_time <= 0.0:
            return 0.0, 0.0, 0

        w_time = base_time * base_time
        wst = w_time * w_space
        if pol_nb == pi:
            return wst, 0.0, 1
        return 0.0, wst, 0

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
        axis_u: np.ndarray,
        diag_u: np.ndarray,
        int_u: np.ndarray,
        int_v: np.ndarray,
        lut_axis: np.ndarray,
        lut_diag: np.ndarray,
        lut_int: np.ndarray,
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

        N = 4096.0
        eps = 1e-6

        mstate = 0.0
        rstate = float(N173_RSTATE_INIT)
        if rstate < 0.0:
            rstate = 0.0
        if rstate > 1.0:
            rstate = 1.0

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            if xi < 0 or xi >= width or yi < 0 or yi >= height:
                out[i] = np.float32(0.0)
                continue

            ti = int(t[i])
            pi = 1 if int(p[i]) > 0 else -1
            idx0 = yi * width + xi

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

            raw_same = 0.0
            raw_opp = 0.0
            cnt_support = 0

            n_axis = int(axis_u.shape[0])
            for k in range(n_axis):
                u = int(axis_u[k])
                w_axis = float(lut_axis[k])
                if w_axis <= 0.0:
                    continue

                nx = xi + u
                if nx <= x1:
                    rs, ro, cs = _acc_neighbor(yi * width + nx, ti, pi, tau_base, inv_tau_base, w_axis, last_ts, last_pol)
                    raw_same += rs
                    raw_opp += ro
                    cnt_support += cs
                nx = xi - u
                if nx >= x0:
                    rs, ro, cs = _acc_neighbor(yi * width + nx, ti, pi, tau_base, inv_tau_base, w_axis, last_ts, last_pol)
                    raw_same += rs
                    raw_opp += ro
                    cnt_support += cs
                ny = yi + u
                if ny <= y1:
                    rs, ro, cs = _acc_neighbor(ny * width + xi, ti, pi, tau_base, inv_tau_base, w_axis, last_ts, last_pol)
                    raw_same += rs
                    raw_opp += ro
                    cnt_support += cs
                ny = yi - u
                if ny >= y0:
                    rs, ro, cs = _acc_neighbor(ny * width + xi, ti, pi, tau_base, inv_tau_base, w_axis, last_ts, last_pol)
                    raw_same += rs
                    raw_opp += ro
                    cnt_support += cs

            n_diag = int(diag_u.shape[0])
            for k in range(n_diag):
                u = int(diag_u[k])
                w_diag = float(lut_diag[k])
                if w_diag <= 0.0:
                    continue

                nx = xi + u
                ny = yi + u
                if nx <= x1 and ny <= y1:
                    rs, ro, cs = _acc_neighbor(ny * width + nx, ti, pi, tau_base, inv_tau_base, w_diag, last_ts, last_pol)
                    raw_same += rs
                    raw_opp += ro
                    cnt_support += cs
                nx = xi + u
                ny = yi - u
                if nx <= x1 and ny >= y0:
                    rs, ro, cs = _acc_neighbor(ny * width + nx, ti, pi, tau_base, inv_tau_base, w_diag, last_ts, last_pol)
                    raw_same += rs
                    raw_opp += ro
                    cnt_support += cs
                nx = xi - u
                ny = yi + u
                if nx >= x0 and ny <= y1:
                    rs, ro, cs = _acc_neighbor(ny * width + nx, ti, pi, tau_base, inv_tau_base, w_diag, last_ts, last_pol)
                    raw_same += rs
                    raw_opp += ro
                    cnt_support += cs
                nx = xi - u
                ny = yi - u
                if nx >= x0 and ny >= y0:
                    rs, ro, cs = _acc_neighbor(ny * width + nx, ti, pi, tau_base, inv_tau_base, w_diag, last_ts, last_pol)
                    raw_same += rs
                    raw_opp += ro
                    cnt_support += cs

            n_int = int(int_u.shape[0])
            for k in range(n_int):
                u = int(int_u[k])
                v = int(int_v[k])
                w_uv = float(lut_int[k])
                if w_uv <= 0.0:
                    continue

                nx = xi + u
                ny = yi + v
                if nx <= x1 and ny <= y1:
                    rs, ro, cs = _acc_neighbor(ny * width + nx, ti, pi, tau_base, inv_tau_base, w_uv, last_ts, last_pol)
                    raw_same += rs
                    raw_opp += ro
                    cnt_support += cs
                nx = xi + u
                ny = yi - v
                if nx <= x1 and ny >= y0:
                    rs, ro, cs = _acc_neighbor(ny * width + nx, ti, pi, tau_base, inv_tau_base, w_uv, last_ts, last_pol)
                    raw_same += rs
                    raw_opp += ro
                    cnt_support += cs
                nx = xi - u
                ny = yi + v
                if nx >= x0 and ny <= y1:
                    rs, ro, cs = _acc_neighbor(ny * width + nx, ti, pi, tau_base, inv_tau_base, w_uv, last_ts, last_pol)
                    raw_same += rs
                    raw_opp += ro
                    cnt_support += cs
                nx = xi - u
                ny = yi - v
                if nx >= x0 and ny >= y0:
                    rs, ro, cs = _acc_neighbor(ny * width + nx, ti, pi, tau_base, inv_tau_base, w_uv, last_ts, last_pol)
                    raw_same += rs
                    raw_opp += ro
                    cnt_support += cs

                nx = xi + v
                ny = yi + u
                if nx <= x1 and ny <= y1:
                    rs, ro, cs = _acc_neighbor(ny * width + nx, ti, pi, tau_base, inv_tau_base, w_uv, last_ts, last_pol)
                    raw_same += rs
                    raw_opp += ro
                    cnt_support += cs
                nx = xi + v
                ny = yi - u
                if nx <= x1 and ny >= y0:
                    rs, ro, cs = _acc_neighbor(ny * width + nx, ti, pi, tau_base, inv_tau_base, w_uv, last_ts, last_pol)
                    raw_same += rs
                    raw_opp += ro
                    cnt_support += cs
                nx = xi - v
                ny = yi + u
                if nx >= x0 and ny <= y1:
                    rs, ro, cs = _acc_neighbor(ny * width + nx, ti, pi, tau_base, inv_tau_base, w_uv, last_ts, last_pol)
                    raw_same += rs
                    raw_opp += ro
                    cnt_support += cs
                nx = xi - v
                ny = yi - u
                if nx >= x0 and ny >= y0:
                    rs, ro, cs = _acc_neighbor(ny * width + nx, ti, pi, tau_base, inv_tau_base, w_uv, last_ts, last_pol)
                    raw_same += rs
                    raw_opp += ro
                    cnt_support += cs

            prev_pol0 = int(last_pol[idx0])
            ts0 = int(last_ts[idx0])
            dt0 = tau_base if ts0 == 0 else (ti - ts0)
            if dt0 < 0:
                dt0 = -dt0
            if dt0 > tau_base:
                dt0 = tau_base

            denom_mix = raw_same + raw_opp
            mix = 0.0
            if denom_mix > 0.0:
                mix = raw_opp / (denom_mix + eps)
                if mix < 0.0:
                    mix = 0.0
                if mix > 1.0:
                    mix = 1.0

            mstate = mstate + (mix - mstate) / N
            if mstate < 0.0:
                mstate = 0.0
            if mstate > 1.0:
                mstate = 1.0

            # Lightweight center suppression proxy from center inter-event interval.
            u_lite = 1.0 - (float(dt0) * inv_tau_base)
            if u_lite < 0.0:
                u_lite = 0.0
            if u_lite > 1.0:
                u_lite = 1.0

            rhythm_bad = 0.0
            if prev_pol0 == -pi:
                rhythm_bad = u_lite
            rstate = rstate + (rhythm_bad - rstate) / N
            if rstate < 0.0:
                rstate = 0.0
            if rstate > 1.0:
                rstate = 1.0

            cnt_possible = (x1 - x0 + 1) * (y1 - y0 + 1) - 1
            if cnt_possible <= 0:
                sfrac = 0.0
            else:
                sfrac = float(cnt_support) / float(cnt_possible)
                if sfrac < 0.0:
                    sfrac = 0.0
                if sfrac > 1.0:
                    sfrac = 1.0

            purity = 1.0 - mix
            if purity < 0.0:
                purity = 0.0
            if purity > 1.0:
                purity = 1.0

            opp_rel_base = purity * (1.0 - mstate) * (1.0 - rstate)
            if opp_rel_base < 0.0:
                opp_rel_base = 0.0
            if opp_rel_base > 1.0:
                opp_rel_base = 1.0
            opp_rel = np.sqrt(opp_rel_base)

            denom_support = 1.0 + sfrac
            u_eff = u_lite * (1.0 + mstate + rstate) / denom_support
            if u_eff < 0.0:
                u_eff = 0.0
            if u_eff > 2.0:
                u_eff = 2.0

            score_core = raw_same + opp_rel * raw_opp
            support_gain = 1.0 + sfrac
            out[i] = np.float32((score_core * support_gain) / (1.0 + (u_eff * u_eff)))

            last_ts[idx0] = np.uint64(ti)
            last_pol[idx0] = np.int8(pi)

    return _kernel


def score_stream_n173(
    ev,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_us: int,
    tb: TimeBase,
    scores_out: np.ndarray | None = None,
) -> np.ndarray:
    """N173: geometric-mean rhythm gate.

    Key points:
    - Same structure as n172, but uses sqrt product gate on opposite evidence.
    - Keeps coefficient-free multiplicative form with bounded algebraic terms.
    - Keeps per-pixel state identical to n153/n171 (last_ts + last_pol).
    """

    n = int(ev.t.shape[0])
    if scores_out is None:
        scores_out = np.empty((n,), dtype=np.float32)
    if n <= 0:
        return scores_out

    tau_base_ticks = int(tb.us_to_ticks(int(tau_us)))
    sigma_space = float(N173_SIGMA)
    if sigma_space <= 1e-6:
        sigma_space = 1e-6

    axis_u, diag_u, int_u, int_v, lut_int = _build_compact_kernel_tables(int(radius_px), float(sigma_space))
    lut_axis = np.zeros((axis_u.shape[0],), dtype=np.float32)
    lut_diag = np.zeros((diag_u.shape[0],), dtype=np.float32)
    inv_2sig2 = 1.0 / (2.0 * sigma_space * sigma_space)
    for k in range(axis_u.shape[0]):
        u = int(axis_u[k])
        lut_axis[k] = np.float32(np.exp(-float(u * u) * inv_2sig2))
        lut_diag[k] = np.float32(np.exp(-float(2 * u * u) * inv_2sig2))

    ker = _try_build_n173_kernel()
    ker(
        ev.t.astype(np.uint64, copy=False),
        ev.x.astype(np.int32, copy=False),
        ev.y.astype(np.int32, copy=False),
        ev.p,
        int(width),
        int(height),
        int(radius_px),
        int(tau_base_ticks),
        axis_u,
        diag_u,
        int_u,
        int_v,
        lut_axis,
        lut_diag,
        lut_int,
        scores_out,
    )
    return scores_out
