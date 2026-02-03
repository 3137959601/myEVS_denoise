from __future__ import annotations

"""Timebase helpers.

Important:
- Your EVS timestamps are in *camera ticks*.
- You confirmed: 1 tick = 12.5 ns.

We expose convenient conversions:
- tick -> microseconds
- microseconds -> ticks

Why we need this:
- All denoise parameters in the Qt UI are in microseconds (us).
- But event timestamps in files (.evtq/.csv/usb_raw) are in ticks.

So every denoise algorithm must convert its threshold from us to ticks.
"""

from dataclasses import dataclass


# ===== Fixed timebase (from your hardware) =====
TICK_NS: float = 12.5
TICK_US: float = TICK_NS / 1000.0  # 0.0125 us


@dataclass(frozen=True)
class TimeBase:
    """Timestamp tick timebase.

    You usually don't need to change this.
    Keep it configurable so you can reuse the tool for other sensors.
    """

    tick_ns: float = TICK_NS

    @property
    def tick_us(self) -> float:
        return float(self.tick_ns) / 1000.0

    @property
    def ticks_per_us(self) -> float:
        # 1us / 0.0125us = 80 ticks/us
        return 1.0 / self.tick_us

    def us_to_ticks(self, us: int) -> int:
        """Convert microseconds threshold to integer ticks (rounded)."""
        return int(round(float(us) * self.ticks_per_us))

    def ticks_to_us(self, ticks: int) -> float:
        return float(ticks) * self.tick_us
