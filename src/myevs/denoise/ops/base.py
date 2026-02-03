from __future__ import annotations

"""Base types for denoise ops.

We keep the interface very small:
- `accept(x, y, p, t)` returns True (keep) or False (drop)
- operations may update internal state *even when dropping*

Important note (matches Qt behaviour):
- Some filters must update their state even when they drop an event.
  Example: STC (method 1) updates last timestamp *always*, otherwise you get
  a "no seed -> never accept" deadlock.
"""

from dataclasses import dataclass
from typing import Protocol


class DenoiseOp(Protocol):
    """Stateful filter operation."""

    name: str

    def accept(self, x: int, y: int, p: int, t: int) -> bool:
        ...


@dataclass(frozen=True)
class Dims:
    """Sensor geometry."""

    width: int
    height: int

    def idx(self, x: int, y: int) -> int:
        return y * self.width + x
