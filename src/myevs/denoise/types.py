from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DenoiseConfig:
    """Denoise parameters (intentionally aligned to Qt UI).

    Qt reuses a small set of parameters across multiple algorithms. We keep the
    same pattern here so you can directly copy the same numbers.

    Timestamps are in ticks, but these parameters are in microseconds (us).
    Conversion is handled by `TimeBase`.
    """

    # Single Qt method (0..8) or a name like "stc" / "hotpixel" / "dp".
    # If `pipeline` is provided, `method` is ignored.
    method: str = "none"

    # Pipeline for composition testing.
    # Example:
    #   pipeline=["globalgate", "stc", "refractory"]
    pipeline: list[str] | None = None

    # === Shared parameters (Qt naming) ===
    time_window_us: int = 2000
    radius_px: int = 1
    min_neighbors: float = 2
    refractory_us: int = 50

    # === Polarity visibility (match Qt showOn/showOff) ===
    show_on: bool = True
    show_off: bool = True
