from __future__ import annotations

import numpy as np

from ....timebase import TimeBase

try:
    import numba
except Exception:  # pragma: no cover
    numba = None


# Rationalized constants (fraction-friendly form).
N183_SIGMA = 2.8
N183_BETA_INIT = 0.65
N183_K_SFRAC = 2.0 / 3.0
N183_K_MIX = 1.0 / 5.0
N183_RSTATE_INIT = 0.10
N183_RHYTHM_GOOD_COEFF = 0.75
N183_PI_LAMBDA = 0.50
N183_PI_ALPHA = 0.50
N183_MODE_CONSERVATIVE = 1
N183_MODE_MINIMAL = 0
N183_PI_MODE_BAD = 0
N183_PI_MODE_R = 1
N183_PI_MODE_AVG = 2
N183_PI_MODE_MAX = 3
N183_PI_MODE_MIX = 4

_N183_KERNEL = None
_N183_TABLE_CACHE: dict[tuple[int, float], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}


def _require_numba() -> None:
    if numba is None:
        raise RuntimeError("N183 requires numba, but import failed")


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


def _try_build_n183_kernel():
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
        beta_init: float,
        k_sfrac: float,
        k_mix: float,
        rhythm_good_coeff: float,
        pi_lambda: float,
        pi_alpha: float,
        pi_mode: int,
        simplify_mode: int,
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

        b = float(beta_init)
        if b < 0.0:
            b = 0.0
        if b > 1.0:
            b = 1.0
        ks = float(k_sfrac)
        if ks < 0.0:
            ks = 0.0
        if ks > 1.5:
            ks = 1.5
        km = float(k_mix)
        if km < 0.0:
            km = 0.0
        if km > 2.0:
            km = 2.0
        rgood_coeff = float(rhythm_good_coeff)
        if rgood_coeff < 0.0:
            rgood_coeff = 0.0
        if rgood_coeff > 1.0:
            rgood_coeff = 1.0
        lpi = float(pi_lambda)
        if lpi < 0.0:
            lpi = 0.0
        if lpi > 1.0:
            lpi = 1.0
        pa = float(pi_alpha)
        if pa < 0.0:
            pa = 0.0
        if pa > 1.0:
            pa = 1.0
        mode = int(simplify_mode)
        psrc_mode = int(pi_mode)

        mstate = 0.0
        rstate = float(N183_RSTATE_INIT)
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

            alpha_eff = 1.0 - mstate
            if alpha_eff < 0.0:
                alpha_eff = 0.0
            alpha_eff = alpha_eff * alpha_eff

            # Lightweight center suppression proxy from center inter-event interval.
            u_lite = 1.0 - (float(dt0) * inv_tau_base)
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
            rstate = rstate + (rhythm_bad - rstate) / N
            if rstate < 0.0:
                rstate = 0.0
            if rstate > 1.0:
                rstate = 1.0
            if psrc_mode == N183_PI_MODE_BAD:
                pi_src = rhythm_bad
            elif psrc_mode == N183_PI_MODE_R:
                pi_src = rstate
            elif psrc_mode == N183_PI_MODE_MAX:
                pi_src = rhythm_bad if rhythm_bad >= rstate else rstate
            elif psrc_mode == N183_PI_MODE_MIX:
                pi_src = pa * rhythm_bad + (1.0 - pa) * rstate
            else:
                pi_src = 0.5 * (rhythm_bad + rstate)
            if pi_src < 0.0:
                pi_src = 0.0
            if pi_src > 1.0:
                pi_src = 1.0

            cnt_possible = (x1 - x0 + 1) * (y1 - y0 + 1) - 1
            if cnt_possible <= 0:
                sfrac = 0.0
            else:
                sfrac = float(cnt_support) / float(cnt_possible)
                if sfrac < 0.0:
                    sfrac = 0.0
                if sfrac > 1.0:
                    sfrac = 1.0

            relief = 1.0 - ks * sfrac
            if relief < 0.25:
                relief = 0.25
            if relief > 1.0:
                relief = 1.0
            rhythm_scale = 1.0
            if mode == N183_MODE_CONSERVATIVE:
                rhythm_scale = 1.0 - rgood_coeff * rhythm_good
            if rhythm_scale < 0.5:
                rhythm_scale = 0.5
            if rhythm_scale > 1.75:
                rhythm_scale = 1.75
            # N179-style: place +pi and -good in one competitive linear term.
            pi_good_scale = 1.0 + lpi * pi_src - rgood_coeff * rhythm_good
            if pi_good_scale < 0.5:
                pi_good_scale = 0.5
            if pi_good_scale > 1.75:
                pi_good_scale = 1.75
            u_eff = u_lite * relief * pi_good_scale
            if u_eff < 0.0:
                u_eff = 0.0
            if u_eff > 1.0:
                u_eff = 1.0

            b = b + (u_eff - b) / N
            if b < 0.0:
                b = 0.0
            if b > 1.0:
                b = 1.0

            raw_gated = raw_same + alpha_eff * raw_opp
            base_score = raw_gated / (1.0 + (u_eff * u_eff))
            support_scale = 1.0 + b * sfrac
            if support_scale < 1.0:
                support_scale = 1.0
            if support_scale > 2.0:
                support_scale = 2.0
            out[i] = np.float32(base_score * support_scale)

            last_ts[idx0] = np.uint64(ti)
            last_pol[idx0] = np.int8(pi)

    return _kernel


def _get_n183_kernel():
    global _N183_KERNEL
    if _N183_KERNEL is None:
        _N183_KERNEL = _try_build_n183_kernel()
    return _N183_KERNEL


def _get_compact_tables_cached(
    radius_px: int,
    sigma_space: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rr = int(radius_px)
    if rr < 0:
        rr = 0
    if rr > 8:
        rr = 8
    sig = float(sigma_space)
    if sig <= 1e-6:
        sig = 1e-6
    key = (rr, sig)
    cached = _N183_TABLE_CACHE.get(key)
    if cached is not None:
        return cached

    axis_u, diag_u, int_u, int_v, lut_int = _build_compact_kernel_tables(rr, sig)
    lut_axis = np.zeros((axis_u.shape[0],), dtype=np.float32)
    lut_diag = np.zeros((diag_u.shape[0],), dtype=np.float32)
    inv_2sig2 = 1.0 / (2.0 * sig * sig)
    for k in range(axis_u.shape[0]):
        u = int(axis_u[k])
        lut_axis[k] = np.float32(np.exp(-float(u * u) * inv_2sig2))
        lut_diag[k] = np.float32(np.exp(-float(2 * u * u) * inv_2sig2))

    cached = (axis_u, diag_u, int_u, int_v, lut_axis, lut_diag, lut_int)
    _N183_TABLE_CACHE[key] = cached
    return cached


def _score_stream_n183_core(
    ev,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_us: int,
    tb: TimeBase,
    beta_init: float,
    k_sfrac: float,
    k_mix: float = 0.0,
    rhythm_good_coeff: float = N183_RHYTHM_GOOD_COEFF,
    pi_lambda: float = N183_PI_LAMBDA,
    pi_alpha: float = N183_PI_ALPHA,
    pi_mode: int = N183_PI_MODE_AVG,
    simplify_mode: int = N183_MODE_CONSERVATIVE,
    scores_out: np.ndarray | None = None,
) -> np.ndarray:
    n = int(ev.t.shape[0])
    if scores_out is None:
        scores_out = np.empty((n,), dtype=np.float32)
    if n <= 0:
        return scores_out

    tau_base_ticks = int(tb.us_to_ticks(int(tau_us)))
    sigma_space = float(N183_SIGMA)
    if sigma_space <= 1e-6:
        sigma_space = 1e-6

    axis_u, diag_u, int_u, int_v, lut_axis, lut_diag, lut_int = _get_compact_tables_cached(
        int(radius_px),
        float(sigma_space),
    )

    ker = _get_n183_kernel()
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
        float(beta_init),
        float(k_sfrac),
        float(k_mix),
        float(rhythm_good_coeff),
        float(pi_lambda),
        float(pi_alpha),
        int(pi_mode),
        int(simplify_mode),
        scores_out,
    )
    return scores_out


def score_stream_n183(
    ev,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_us: int,
    tb: TimeBase,
    beta_init: float | None = None,
    k_sfrac: float | None = None,
    rhythm_good_coeff: float | None = None,
    pi_lambda: float | None = None,
    pi_alpha: float | None = None,
    pi_source: str = "avg",
    mode: str = "conservative",
    scores_out: np.ndarray | None = None,
) -> np.ndarray:
    """N183: N179-style pi-source ablation.

    pi_source choices: bad | r | avg | max | mix
    """

    m = str(mode).strip().lower()
    simplify_mode = N183_MODE_MINIMAL if m in {"minimal", "min"} else N183_MODE_CONSERVATIVE
    psrc = str(pi_source).strip().lower()
    if psrc == "bad":
        pi_mode = N183_PI_MODE_BAD
    elif psrc in {"r", "ema"}:
        pi_mode = N183_PI_MODE_R
    elif psrc == "max":
        pi_mode = N183_PI_MODE_MAX
    elif psrc == "mix":
        pi_mode = N183_PI_MODE_MIX
    else:
        pi_mode = N183_PI_MODE_AVG

    return _score_stream_n183_core(
        ev,
        width=int(width),
        height=int(height),
        radius_px=int(radius_px),
        tau_us=int(tau_us),
        tb=tb,
        beta_init=float(N183_BETA_INIT if beta_init is None else beta_init),
        k_sfrac=float(N183_K_SFRAC if k_sfrac is None else k_sfrac),
        k_mix=0.0,
        rhythm_good_coeff=float(N183_RHYTHM_GOOD_COEFF if rhythm_good_coeff is None else rhythm_good_coeff),
        pi_lambda=float(N183_PI_LAMBDA if pi_lambda is None else pi_lambda),
        pi_alpha=float(N183_PI_ALPHA if pi_alpha is None else pi_alpha),
        pi_mode=int(pi_mode),
        simplify_mode=int(simplify_mode),
        scores_out=scores_out,
    )
