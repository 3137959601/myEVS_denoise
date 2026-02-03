from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

import numpy as np


@dataclass(frozen=True)
class EventStreamMeta:
    width: int
    height: int
    time_unit: str = "tick"  # Qt uses camera ticks; keep as integer ticks.


@dataclass(frozen=True)
class EventBatch:
    """A batch of events in SoA form for speed."""

    t: np.ndarray  # uint64
    x: np.ndarray  # uint16
    y: np.ndarray  # uint16
    p: np.ndarray  # int8, +1 for ON, -1 for OFF

    def __len__(self) -> int:
        return int(self.t.shape[0])


def iter_batches(events: Iterable[EventBatch]) -> Iterator[EventBatch]:
    for b in events:
        if len(b) == 0:
            continue
        yield b


def concat_batches(batches: Iterable[EventBatch], max_events: int) -> Iterator[EventBatch]:
    """Rebatch stream into batches of ~max_events."""
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    ts: list[np.ndarray] = []
    n = 0
    for b in batches:
        if len(b) == 0:
            continue
        xs.append(b.x)
        ys.append(b.y)
        ps.append(b.p)
        ts.append(b.t)
        n += len(b)
        if n >= max_events:
            yield EventBatch(
                t=np.concatenate(ts),
                x=np.concatenate(xs),
                y=np.concatenate(ys),
                p=np.concatenate(ps),
            )
            xs, ys, ps, ts = [], [], [], []
            n = 0
    if n:
        yield EventBatch(t=np.concatenate(ts), x=np.concatenate(xs), y=np.concatenate(ys), p=np.concatenate(ps))


def to_p01(p_pm1: np.ndarray) -> np.ndarray:
    """Convert +1/-1 to 1/0."""
    return (p_pm1 > 0).astype(np.uint8)


def to_pm1(p_01: np.ndarray) -> np.ndarray:
    """Convert 1/0 to +1/-1."""
    return np.where(p_01.astype(np.uint8) > 0, 1, -1).astype(np.int8)


def filter_visibility_batches(
    batches: Iterable[EventBatch], *, show_on: bool = True, show_off: bool = True
) -> Iterator[EventBatch]:
    """Filter a stream by polarity visibility.

    Qt side computes its "denoise kept ratio" on *visible* events only
    (after showOn/showOff), so for apples-to-apples comparisons we need
    the same filtering when computing stats/ratios.
    """

    if show_on and show_off:
        yield from iter_batches(batches)
        return

    for b in iter_batches(batches):
        p = np.asarray(b.p, dtype=np.int8)
        keep = np.ones((p.shape[0],), dtype=bool)
        if not show_on:
            keep &= p <= 0
        if not show_off:
            keep &= p > 0
        if keep.any():
            yield EventBatch(t=b.t[keep], x=b.x[keep], y=b.y[keep], p=b.p[keep])


def unwrap_tick_batches(
    batches: Iterable[EventBatch], *, bits: int | None = None
) -> Iterator[EventBatch]:
    """Unwrap wrapped device tick timestamps into a monotonic 64-bit stream.

    Problem we saw in your logs:
    - t_first > t_last => negative duration_ticks
    - Refractory/STC compare dt in ticks; if t wraps, dt becomes wrong.

    This wrapper detects large backward jumps and adds an epoch offset.
    - bits=None: auto-infer 30-bit vs 32-bit wrap by observed max tick.
    - bits=30/32: force a specific modulus.
    """

    modulus: int | None = None
    threshold: int | None = None
    epoch = 0
    prev_last_raw: int | None = None

    def _ensure_modulus(max_raw: int) -> None:
        nonlocal modulus, threshold
        if modulus is not None:
            return
        if bits in (30, 32):
            modulus = 1 << int(bits)
            threshold = 1 << (int(bits) - 1)
            return
        # auto
        if max_raw <= 0x3FFFFFFF:
            modulus = 1 << 30
            threshold = 1 << 29
        elif max_raw <= 0xFFFFFFFF:
            modulus = 1 << 32
            threshold = 1 << 31
        else:
            modulus = None
            threshold = None

    for b in iter_batches(batches):
        t_raw = np.asarray(b.t, dtype=np.uint64)
        if t_raw.size == 0:
            continue

        _ensure_modulus(int(t_raw.max()))
        if modulus is None or threshold is None:
            yield b
            continue

        # Detect wrap across batch boundary
        first_raw = int(t_raw[0])
        if prev_last_raw is not None and first_raw < prev_last_raw and (prev_last_raw - first_raw) > threshold:
            epoch += modulus

        # Detect wrap(s) inside this batch
        if t_raw.size >= 2:
            dec = t_raw[1:] < t_raw[:-1]
            if bool(np.any(dec)):
                back = (t_raw[:-1] - t_raw[1:]).astype(np.uint64)
                is_wrap = dec & (back > np.uint64(threshold))
                if bool(np.any(is_wrap)):
                    wrap_u8 = np.zeros((t_raw.shape[0],), dtype=np.uint8)
                    wrap_u8[1:] = is_wrap.astype(np.uint8)
                    offsets = np.cumsum(wrap_u8, dtype=np.uint64) * np.uint64(modulus)
                    t_unwrapped = t_raw + offsets + np.uint64(epoch)
                    epoch += int(offsets[-1])
                    prev_last_raw = int(t_raw[-1])
                    yield EventBatch(t=t_unwrapped, x=b.x, y=b.y, p=b.p)
                    continue

        prev_last_raw = int(t_raw[-1])
        if epoch == 0:
            yield b
        else:
            yield EventBatch(t=t_raw + np.uint64(epoch), x=b.x, y=b.y, p=b.p)
