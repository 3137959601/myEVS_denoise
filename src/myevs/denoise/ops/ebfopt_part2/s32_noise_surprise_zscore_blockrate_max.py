from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class S32NoiseSurpriseZScoreBlockRateMaxParams:
    """Parameters for Part2 s32 (s28 with block-level local-rate max correction).

    Motivation (from s29/s30 lessons):
    - s28 uses a *global* event-rate EMA as the noise proxy. This can under-model
      locally hot regions (hotmask/near-hot), which still dominate heavy FP.
    - s30 tried to estimate a local rate from the neighborhood itself and got
      worse (too reactive / too correlated with the raw support, hurting ranking).

    Core idea:
    - Keep s28's score definition: z = (raw - mu(r)) / sqrt(var(r) + eps), where
      raw is same-polarity neighborhood support (baseline EBF raw).
    - Maintain a coarse *block-level* EMA of event rate in addition to the global
      EMA, and use the more conservative (larger) per-pixel rate when raw is
      already high:
        r_pix_eff = max(r_pix_global, r_pix_block)
      but only apply the max-correction when raw >= raw_thr, to keep the ranking
      perturbation small in light/mid.

    Properties:
    - Streaming, single-pass, O(r^2), Numba-only.
    - Extra state: per-block last timestamp + per-block rate EMA.

    Hyperparameters:
    - tau_rate_us: EMA time constant (microseconds) for both global and block
      rate estimates. 0 means "auto" (use tau_us of the current sweep point).

    Notes:
    - Block size and raw_thr are intentionally fixed constants to avoid adding
      more sweep dimensions at this stage.
    """

    tau_rate_us: int = 0


def _env_int(env: dict[str, str], name: str, default: int) -> int:
    s = (env.get(name, "") or "").strip()
    if not s:
        return int(default)
    try:
        v = int(float(s))
    except Exception:
        return int(default)
    if v < 0:
        v = 0
    return int(v)


def s32_noise_surprise_zscore_blockrate_max_params_from_env(
    env: dict[str, str] | None = None,
) -> S32NoiseSurpriseZScoreBlockRateMaxParams:
    """Read s32 parameters from environment.

    - MYEVS_EBF_S32_TAU_RATE_US (default 0; 0 means auto)
    """

    if env is None:
        env = os.environ  # type: ignore[assignment]

    tau_rate_us = _env_int(env, "MYEVS_EBF_S32_TAU_RATE_US", 0)
    return S32NoiseSurpriseZScoreBlockRateMaxParams(tau_rate_us=int(tau_rate_us))


def try_build_s32_noise_surprise_zscore_blockrate_max_scores_kernel():
    """Build and return Numba kernel for s32 score streaming.

    Returns None if numba is unavailable.

    Kernel signature:
      (t,x,y,p,width,height,radius_px,tau_ticks,tau_rate_ticks,
       last_ts,last_pol,rate_ema,block_last_t,block_rate_ema,scores_out) -> None

    Arrays:
    - last_ts: uint64 per pixel
    - last_pol: int8 per pixel
    - rate_ema: float64 array of shape (1,) storing global event rate (events/tick)
    - block_last_t: int64 per block storing last timestamp (ticks); init -1
    - block_rate_ema: float64 per block storing block event rate (events/tick)

    Output:
    - scores_out: float32 z-score per event
    """

    try:
        from numba import njit  # type: ignore
    except Exception:
        return None

    @njit(cache=True)
    def ebf_s32_scores_stream(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        width: int,
        height: int,
        radius_px: int,
        tau_ticks: int,
        tau_rate_ticks: int,
        last_ts: np.ndarray,
        last_pol: np.ndarray,
        rate_ema: np.ndarray,
        block_last_t: np.ndarray,
        block_rate_ema: np.ndarray,
        scores_out: np.ndarray,
    ) -> None:
        n = int(t.shape[0])
        w = int(width)
        h = int(height)

        rr = int(radius_px)
        if rr < 0:
            rr = 0
        if rr > 8:
            rr = 8

        tau = int(tau_ticks)
        if tau <= 0:
            tau = 1

        tr = int(tau_rate_ticks)
        if tr <= 0:
            tr = tau

        # Global rate EMA in events/tick.
        r_ema = float(rate_ema[0])
        if not np.isfinite(r_ema) or r_ema < 0.0:
            r_ema = 0.0

        inv_tau = 1.0 / float(tau)
        n_pix = float(w * h)
        eps = 1e-6

        # Fixed block size for local background model.
        bw = 32
        bh = 32
        nbx = (w + bw - 1) // bw
        nby = (h + bh - 1) // bh

        # Sanity: if state size mismatches, fall back to pure global (still updates pixel state).
        use_blocks = True
        if int(block_last_t.shape[0]) != int(nbx * nby):
            use_blocks = False
        if int(block_rate_ema.shape[0]) != int(nbx * nby):
            use_blocks = False

        # Pass-through behavior (still updates state): score is +inf.
        if rr <= 0:
            prev_t = 0
            for i in range(n):
                xi = int(x[i])
                yi = int(y[i])
                ti = int(t[i])

                if i > 0:
                    dtg = ti - prev_t
                    if dtg > 0:
                        inst = 1.0 / float(dtg)
                        a_rate = 1.0 - np.exp(-float(dtg) / float(tr))
                        r_ema = r_ema + a_rate * (inst - r_ema)
                prev_t = ti

                if xi < 0 or xi >= w or yi < 0 or yi >= h:
                    scores_out[i] = 0.0
                    continue

                pi = 1 if int(p[i]) > 0 else -1
                idx0 = yi * w + xi
                last_ts[idx0] = np.uint64(ti)
                last_pol[idx0] = np.int8(pi)

                if use_blocks:
                    bx = xi // bw
                    by = yi // bh
                    bid = by * nbx + bx
                    last_tb = int(block_last_t[bid])
                    if last_tb >= 0:
                        dtb = ti - last_tb
                        if dtb > 0:
                            instb = 1.0 / float(dtb)
                            a_rate_b = 1.0 - np.exp(-float(dtb) / float(tr))
                            rb = float(block_rate_ema[bid])
                            if not np.isfinite(rb) or rb < 0.0:
                                rb = 0.0
                            rb = rb + a_rate_b * (instb - rb)
                            block_rate_ema[bid] = rb
                    block_last_t[bid] = np.int64(ti)

                scores_out[i] = np.inf

            rate_ema[0] = r_ema
            return

        raw_thr = 3.0

        prev_t = 0
        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            ti = int(t[i])

            # Update global rate EMA (events/tick) using inter-event dt.
            if i > 0:
                dtg = ti - prev_t
                if dtg > 0:
                    inst = 1.0 / float(dtg)
                    a_rate = 1.0 - np.exp(-float(dtg) / float(tr))
                    r_ema = r_ema + a_rate * (inst - r_ema)
            prev_t = ti

            if xi < 0 or xi >= w or yi < 0 or yi >= h:
                scores_out[i] = 0.0
                continue

            pi = 1 if int(p[i]) > 0 else -1
            idx0 = yi * w + xi

            prev_ts0 = int(last_ts[idx0])

            rb = 0.0
            bx = 0
            by = 0
            if use_blocks:
                bx = xi // bw
                by = yi // bh
                bid = by * nbx + bx

                last_tb = int(block_last_t[bid])
                rb = float(block_rate_ema[bid])
                if not np.isfinite(rb) or rb < 0.0:
                    rb = 0.0

                if last_tb >= 0:
                    dtb = ti - last_tb
                    if dtb > 0:
                        instb = 1.0 / float(dtb)
                        a_rate_b = 1.0 - np.exp(-float(dtb) / float(tr))
                        rb = rb + a_rate_b * (instb - rb)
                block_rate_ema[bid] = rb
                block_last_t[bid] = np.int64(ti)

            # Baseline raw support (same polarity) as in EBF.
            raw_w = 0

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

            # Geometric neighbor count.
            m = (x1 - x0 + 1) * (y1 - y0 + 1) - 1
            if m < 1:
                m = 1

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

                    dt = ti - ts
                    if dt < 0:
                        dt = -dt
                    if dt > tau:
                        continue

                    raw_w += (tau - dt)

            # Update per-pixel state.
            last_ts[idx0] = np.uint64(ti)
            last_pol[idx0] = np.int8(pi)

            raw = float(raw_w) * inv_tau

            # Global noise proxy (per pixel).
            r_pix_g = r_ema / n_pix
            if r_pix_g < 0.0:
                r_pix_g = 0.0

            r_pix = r_pix_g

            # Block-level local background: only allow to increase effective noise,
            # and only when raw is already high.
            # Additional gate: only trigger for bursty/self-hot pixels (small self dt)
            # to avoid suppressing genuine moving signals in generally active blocks.
            dt0_ok = False
            if prev_ts0 != 0:
                dt0 = ti - prev_ts0
                if dt0 < 0:
                    dt0 = -dt0
                dt_self_thr = tau // 8
                if dt_self_thr < 1:
                    dt_self_thr = 1
                if dt0 <= dt_self_thr:
                    dt0_ok = True

            if use_blocks and raw >= raw_thr and dt0_ok:
                # Convert block rate to per-pixel rate.
                x_end = (bx + 1) * bw
                if x_end > w:
                    x_end = w
                y_end = (by + 1) * bh
                if y_end > h:
                    y_end = h
                area = float((x_end - bx * bw) * (y_end - by * bh))
                if area < 1.0:
                    area = 1.0

                r_pix_b = rb / area
                if r_pix_b > r_pix:
                    r_pix = r_pix_b

            a = r_pix * float(tau)

            if a < 1e-3:
                ew = 0.5 * a - (a * a) / 6.0
                ew2 = (a / 3.0) - (a * a) / 12.0
            else:
                ea = np.exp(-a)
                ew = 1.0 - (1.0 - ea) / a
                ew2 = (a * a - 2.0 * a + 2.0 - 2.0 * ea) / (a * a)

            # Mark (polarity) match probability is ~0.5 under noise.
            mu_per = 0.5 * ew
            e2_per = 0.5 * ew2
            var_per = e2_per - mu_per * mu_per
            if var_per < 0.0:
                var_per = 0.0

            mu = float(m) * mu_per
            var = float(m) * var_per

            denom = np.sqrt(var + eps)
            scores_out[i] = float((raw - mu) / denom)

        rate_ema[0] = r_ema

    return ebf_s32_scores_stream


__all__ = [
    "S32NoiseSurpriseZScoreBlockRateMaxParams",
    "s32_noise_surprise_zscore_blockrate_max_params_from_env",
    "try_build_s32_noise_surprise_zscore_blockrate_max_scores_kernel",
]
