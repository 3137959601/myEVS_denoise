from __future__ import annotations

"""Global burst gate (anti-flicker).

Qt reference:
- EventFrameProcessor::updateGlobalGate()
- method id: 7

Qt computes event rate from wall-clock time (QElapsedTimer).
Offline Python cannot rely on wall-clock, so we use event timestamps:
- dt = t_last - t_first (ticks)
- rate = visible_events / dt_seconds

Behaviour aligned with Qt:
- Maintain EMA of visible event rate.
- If EMA > threshold => gate becomes active (with hysteresis and hold time).
- When active, downsample events deterministically by a factor `gateFactor`.

Parameters reuse (same as Qt):
- min_neighbors: threshold in K/s
- time_window_us: EMA time constant (tau)
- refractory_us: minimal active duration (hold)

Deterministic sampling matches Qt `keepByGateFactor()`.
"""

from dataclasses import dataclass

import math

from ...timebase import TimeBase
from ..types import DenoiseConfig


def _mix32(v: int) -> int:
    # Same integer hash as Qt
    v &= 0xFFFFFFFF
    v ^= (v >> 16)
    v = (v * 0x7FEB352D) & 0xFFFFFFFF
    v ^= (v >> 15)
    v = (v * 0x846CA68B) & 0xFFFFFFFF
    v ^= (v >> 16)
    return v & 0xFFFFFFFF


def keep_by_gate_factor(gate_factor: int, x: int, y: int, t: int) -> bool:
    if gate_factor <= 1:
        return True
    h = _mix32((x * 73856093) ^ (y * 19349663) ^ (t & 0xFFFFFFFF))
    return (h % int(gate_factor)) == 0


@dataclass
class GlobalGateOp:
    """State for global gate."""

    name: str = "globalgate"

    def __init__(self, cfg: DenoiseConfig, tb: TimeBase):
        self.name = "globalgate"
        self.cfg = cfg
        self.tb = tb

        # ===== State variables (important signals) =====
        self.rate_ema: float = 0.0
        self.gate_active: bool = False
        self.hold_until_t: int = 0  # event timestamp (ticks)

    def compute_gate_factor(self) -> int:
        # In Qt:
        #   thr = minNeighbors * 1000 (K/s -> /s)
        #   factor = ceil(rate/thr)
        thr = float(max(1, int(self.cfg.min_neighbors))) * 1000.0
        rate = float(self.rate_ema)

        factor = 1
        if thr > 0.0 and rate > thr:
            factor = int(math.ceil(rate / thr))
        if self.gate_active:
            factor = max(factor, 2)
        return int(min(max(factor, 1), 64))

    def update_after_batch(self, *, visible_events: int, t_first: int, t_last: int) -> None:
        """Update EMA and gate state using *event time* (ticks)."""

        if visible_events <= 0:
            return
        if t_last <= t_first:
            return

        dt_ticks = int(t_last - t_first)
        dt_us = float(dt_ticks) * self.tb.tick_us
        if dt_us <= 0.0:
            return

        dt_s = dt_us / 1_000_000.0
        inst_rate = float(visible_events) / dt_s

        tau_us = float(max(1, int(self.cfg.time_window_us)))
        alpha = 1.0 - math.exp(-dt_us / tau_us)

        if self.rate_ema <= 0.0:
            self.rate_ema = inst_rate
        else:
            self.rate_ema += alpha * (inst_rate - self.rate_ema)

        thr = float(max(1, int(self.cfg.min_neighbors))) * 1000.0
        thr_off = thr * 0.80
        hold_ticks = int(self.tb.us_to_ticks(int(self.cfg.refractory_us)))

        now_t = int(t_last)

        if not self.gate_active:
            if self.rate_ema > thr:
                self.gate_active = True
                self.hold_until_t = now_t + max(0, hold_ticks)
        else:
            # Hold time: do not exit before hold_until_t
            if hold_ticks > 0 and now_t < self.hold_until_t:
                return
            if self.rate_ema < thr_off:
                self.gate_active = False
