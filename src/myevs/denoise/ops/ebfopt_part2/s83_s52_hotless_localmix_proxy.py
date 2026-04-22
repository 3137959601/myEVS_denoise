from __future__ import annotations

import numpy as np

from ....timebase import TimeBase

try:
    import numba
except Exception:  # pragma: no cover
    numba = None


def _require_numba() -> None:
    if numba is None:
        raise RuntimeError("s83 requires numba, but import failed")


def _try_build_s83_kernel():
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

        tau = int(tau_ticks)
        if tau <= 0:
            tau = 1

        inv_tau = 1.0 / float(tau)
        eps = 1e-6

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
            pol0 = int(last_pol[idx0])

            if ts0 == 0 or ti <= ts0:
                dt0 = tau
            else:
                dt0 = ti - ts0
                if dt0 > tau:
                    dt0 = tau

            rec = 1.0 - float(dt0) * inv_tau
            if rec < 0.0:
                rec = 0.0

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

            raw_same = 0
            raw_opp = 0
            cnt_support = 0

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

            cnt_possible = (x1 - x0 + 1) * (y1 - y0 + 1) - 1
            if cnt_possible <= 0:
                sfrac = 0.0
            else:
                sfrac = float(cnt_support) / float(cnt_possible)
                if sfrac < 0.0:
                    sfrac = 0.0
                if sfrac > 1.0:
                    sfrac = 1.0

            denom = float(raw_same + raw_opp)
            mix = 0.0
            if denom > 0.0:
                mix = float(raw_opp) / (denom + eps)
                if mix < 0.0:
                    mix = 0.0
                if mix > 1.0:
                    mix = 1.0

            purity = 1.0 - mix
            if purity < 0.0:
                purity = 0.0

            gate = purity * purity
            gate_eff = gate + (1.0 - gate) * sfrac
            alpha_eff = gate_eff * (0.5 + 0.5 * sfrac)

            if pol0 == pi:
                u_self = rec * (0.85 + 0.15 * mix)
            elif pol0 == -pi:
                u_self = rec * (0.55 + 0.45 * mix)
            else:
                u_self = rec * (0.70 + 0.30 * mix)

            if u_self < 0.0:
                u_self = 0.0
            if u_self > 1.0:
                u_self = 1.0

            raw_gated = (float(raw_same) + alpha_eff * float(raw_opp)) * inv_tau
            base_score = raw_gated / (1.0 + u_self * u_self)

            beta_eff = 1.0 - 0.5 * u_self
            if beta_eff < 0.0:
                beta_eff = 0.0
            if beta_eff > 1.0:
                beta_eff = 1.0

            out[i] = np.float32(base_score * (1.0 + beta_eff * sfrac))

    return _kernel


def score_stream_s83(
    ev,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_us: int,
    tb: TimeBase,
    scores_out: np.ndarray | None = None,
) -> np.ndarray:
    """S83: hot_state-free S52-style local mix fusion.

    Design goals:
    - No per-pixel hot_state table (memory back to baseline state size).
    - No global adaptive scalar states (beta/mix); all modulation is event-local.
    - Keep soft gating style from S52/S53/S55 (avoid hard conditions).
    """

    n = int(ev.t.shape[0])
    if scores_out is None:
        scores_out = np.empty((n,), dtype=np.float32)
    if n <= 0:
        return scores_out

    tau_ticks = int(tb.us_to_ticks(int(tau_us)))

    ker = _try_build_s83_kernel()
    ker(
        ev.t.astype(np.uint64, copy=False),
        ev.x.astype(np.int32, copy=False),
        ev.y.astype(np.int32, copy=False),
        ev.p,
        int(width),
        int(height),
        int(radius_px),
        int(tau_ticks),
        scores_out,
    )
    return scores_out
