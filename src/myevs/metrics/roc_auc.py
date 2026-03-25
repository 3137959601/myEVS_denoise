from __future__ import annotations

"""ROC/AUC evaluation for denoise algorithms.

Goal
----
Given:
- a *clean reference* event stream (ideally signal-only)
- a *noisy* event stream (signal + noise)
- a denoise method and a sweep of one parameter

We evaluate denoising as a binary classifier on **events**.

Important: the meaning of TP/FP/TN/FN depends on the ROC convention
(see :func:`confusion_from_totals`). In this project we support two conventions:

1) ``roc_convention="paper"`` (default; matches most denoising papers)
   - Ground truth positive: signal (event belongs to clean)
   - Predicted positive: kept (event survives the denoiser)
   - TP: signal kept
   - FP: noise kept
   - TN: noise dropped
   - FN: signal dropped
   - TPR = signal_kept / signal_total
   - FPR = noise_kept / noise_total

2) ``roc_convention="noise-drop"`` (legacy)
   - Ground truth positive: noise (event is NOT in clean)
   - Predicted positive: dropped (event removed by the denoiser)
   - TP: noise dropped
   - FP: signal dropped
   - TN: signal kept
   - FN: noise kept
   - TPR = noise_dropped / noise_total    (noise rejection)
   - FPR = signal_dropped / signal_total  (signal loss)

Matching rule / limitation
--------------------------
We support two labeling modes:
- Exact match: signal iff (t, x, y, p) matches exactly.
- Time-tolerant match: signal iff there exists a clean event with the same (x, y, p)
    whose timestamp is within about ±match_us.

The time-tolerant mode is useful for v2e runs where enabling noise slightly shifts
signal event timestamps (or causes small jitter) so exact matching underestimates
signal membership.

Implementation note
-------------------
The tolerant mode is implemented as a fast *approximation*:

- Quantize time: t_bin := t // match_ticks
- Check membership in the same bin (t_bin)
- Optionally also check neighbor bins (t_bin±1, ±2, ...) controlled by match_bin_radius.

Why check neighbor bins at all?
It reduces *boundary misses* caused by quantization. For example, with match_us=200,
two events that are only 1us apart can still fall into adjacent bins if one is near
the boundary (199us vs 200us).

Trade-off / pitfall:
Checking neighbor bins *expands the effective match window* and increases the chance
of accidental matches at high event density (e.g. heavy noise). In that case, too-large
match_us and/or match_bin_radius can mislabel noise as signal and inflate ROC/AUC.

If you see suspiciously good results on heavy noise, try a smaller match_us and/or
set match_bin_radius=0.
"""

from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np

from ..events import EventBatch, EventStreamMeta, filter_visibility_batches, unwrap_tick_batches


def _bits_needed(n: int) -> int:
    # bits to represent values in [0, n-1]
    if n <= 1:
        return 1
    v = int(n - 1)
    b = 0
    while v > 0:
        b += 1
        v >>= 1
    return max(1, b)


@dataclass(frozen=True)
class KeyPacker:
    """Pack (t,x,y,p) into uint64 keys for fast membership queries."""

    width: int
    height: int

    x_bits: int
    y_bits: int
    shift_t: int
    shift_y: int

    @staticmethod
    def for_meta(meta: EventStreamMeta) -> "KeyPacker":
        w = int(meta.width)
        h = int(meta.height)
        xb = _bits_needed(w)
        yb = _bits_needed(h)
        shift_y = xb + 1  # +1 for polarity bit
        shift_t = yb + shift_y
        if shift_t >= 63:
            raise ValueError(f"Cannot pack keys: width={w} height={h} uses too many bits")
        return KeyPacker(width=w, height=h, x_bits=xb, y_bits=yb, shift_t=shift_t, shift_y=shift_y)

    def pack(self, t: np.ndarray, x: np.ndarray, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        t64 = np.asarray(t, dtype=np.uint64)
        x64 = np.asarray(x, dtype=np.uint64)
        y64 = np.asarray(y, dtype=np.uint64)
        p64 = (np.asarray(p, dtype=np.int8) > 0).astype(np.uint64)

        # Optional bound check (kept separate so callers can slice first).
        t_max = int(t64.max()) if t64.size else 0
        if t_max >= (1 << (64 - self.shift_t)):
            raise ValueError(
                "Timestamp too large to pack into uint64 key. "
                "Use a smaller timebase / shorter clip, or implement a different matching scheme."
            )

        return (t64 << np.uint64(self.shift_t)) | (y64 << np.uint64(self.shift_y)) | (x64 << np.uint64(1)) | p64


def _in_bounds_mask(meta: EventStreamMeta, b: EventBatch) -> np.ndarray:
    w = int(meta.width)
    h = int(meta.height)
    x = np.asarray(b.x)
    y = np.asarray(b.y)
    return (x < w) & (y < h)


def _membership_mask(sorted_keys: np.ndarray, query_keys: np.ndarray) -> np.ndarray:
    if sorted_keys.size == 0 or query_keys.size == 0:
        return np.zeros((query_keys.shape[0],), dtype=bool)
    idx = np.searchsorted(sorted_keys, query_keys)
    ok = idx < sorted_keys.shape[0]
    out = np.zeros((query_keys.shape[0],), dtype=bool)
    if bool(np.any(ok)):
        ii = idx[ok]
        out[ok] = sorted_keys[ii] == query_keys[ok]
    return out


def build_clean_index(
    meta: EventStreamMeta,
    batches: Iterable[EventBatch],
    *,
    show_on: bool,
    show_off: bool,
    unwrap_ts: bool,
    ts_bits: int | None,
    match_ticks: int = 0,
    match_bin_radius: int = 1,
) -> tuple[np.ndarray, KeyPacker]:
    """Build a sorted unique key index for the clean stream."""

    packer = KeyPacker.for_meta(meta)

    b_it: Iterable[EventBatch] = batches
    if unwrap_ts:
        b_it = unwrap_tick_batches(b_it, bits=ts_bits)
    b_it = filter_visibility_batches(b_it, show_on=show_on, show_off=show_off)

    chunks: list[np.ndarray] = []
    for b in b_it:
        inb = _in_bounds_mask(meta, b)
        if not bool(np.any(inb)):
            continue
        t = b.t[inb]
        if match_ticks > 0:
            t = (np.asarray(t, dtype=np.uint64) // np.uint64(match_ticks)).astype(np.uint64)
        keys = packer.pack(t, b.x[inb], b.y[inb], b.p[inb])
        if keys.size:
            chunks.append(keys)

    if not chunks:
        return np.empty((0,), dtype=np.uint64), packer

    all_keys = np.concatenate(chunks).astype(np.uint64, copy=False)
    all_keys.sort()
    uniq = np.unique(all_keys)
    return uniq, packer


def signal_mask(
    *,
    clean_keys: np.ndarray,
    packer: KeyPacker,
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    match_ticks: int,
    match_bin_radius: int,
) -> np.ndarray:
    """Return boolean mask for whether each event is considered signal.

    - match_ticks<=0: exact match on raw t
    - match_ticks>0: quantize time (t // match_ticks) and check neighbor bins (±radius)
    """

    if match_ticks <= 0:
        keys = packer.pack(t, x, y, p)
        return _membership_mask(clean_keys, keys)

    if match_bin_radius < 0:
        match_bin_radius = 0

    t0 = (np.asarray(t, dtype=np.uint64) // np.uint64(match_ticks)).astype(np.uint64)
    r = int(match_bin_radius)

    def _shift_bins(base: np.ndarray, offset: int) -> np.ndarray:
        if offset == 0:
            return base
        if offset > 0:
            return base + np.uint64(offset)
        dd = np.uint64(-offset)
        return np.where(base > dd, base - dd, np.uint64(0))

    # Check bins in a symmetric window to reduce boundary misses.
    # Note: this is still an approximation (see module docstring).
    m = np.zeros((t0.shape[0],), dtype=bool)
    for off in range(-r, r + 1):
        tb = _shift_bins(t0, off)
        m |= _membership_mask(clean_keys, packer.pack(tb, x, y, p))

    return m


@dataclass(frozen=True)
class Totals:
    total: int
    signal: int
    noise: int


@dataclass(frozen=True)
class Kept:
    total: int
    signal: int
    noise: int


@dataclass(frozen=True)
class Confusion:
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def tpr(self) -> float:
        denom = self.tp + self.fn
        return (float(self.tp) / float(denom)) if denom > 0 else 0.0

    @property
    def fpr(self) -> float:
        denom = self.fp + self.tn
        return (float(self.fp) / float(denom)) if denom > 0 else 0.0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return (float(self.tp) / float(denom)) if denom > 0 else 0.0

    @property
    def accuracy(self) -> float:
        denom = self.tp + self.fp + self.tn + self.fn
        return (float(self.tp + self.tn) / float(denom)) if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.tpr
        denom = p + r
        return (2.0 * p * r / denom) if denom > 0 else 0.0


def compute_totals_for_noisy(
    noisy_meta: EventStreamMeta,
    noisy_batches: Iterable[EventBatch],
    *,
    clean_keys: np.ndarray,
    packer: KeyPacker,
    show_on: bool,
    show_off: bool,
    unwrap_ts: bool,
    ts_bits: int | None,
    match_ticks: int = 0,
    match_bin_radius: int = 1,
) -> Totals:
    """Count signal/noise totals in the noisy stream using clean membership."""

    b_it: Iterable[EventBatch] = noisy_batches
    if unwrap_ts:
        b_it = unwrap_tick_batches(b_it, bits=ts_bits)
    b_it = filter_visibility_batches(b_it, show_on=show_on, show_off=show_off)

    total = 0
    signal = 0

    for b in b_it:
        inb = _in_bounds_mask(noisy_meta, b)
        if not bool(np.any(inb)):
            continue
        t = b.t[inb]
        x = b.x[inb]
        y = b.y[inb]
        p = b.p[inb]
        m = signal_mask(
            clean_keys=clean_keys,
            packer=packer,
            t=t,
            x=x,
            y=y,
            p=p,
            match_ticks=int(match_ticks),
            match_bin_radius=int(match_bin_radius),
        )
        sig_n = int(np.count_nonzero(m))
        n = int(t.shape[0])
        total += n
        signal += sig_n

    noise = total - signal
    return Totals(total=total, signal=signal, noise=noise)


def compute_kept_for_denoised(
    denoised_meta: EventStreamMeta,
    denoised_batches: Iterable[EventBatch],
    *,
    clean_keys: np.ndarray,
    packer: KeyPacker,
    match_ticks: int = 0,
    match_bin_radius: int = 1,
) -> Kept:
    """Count kept signal/noise events in denoised output."""

    total = 0
    signal = 0

    for b in denoised_batches:
        if len(b) == 0:
            continue
        # denoise_stream already filters by bounds/visibility, but keep this safe.
        inb = _in_bounds_mask(denoised_meta, b)
        if not bool(np.any(inb)):
            continue
        t = b.t[inb]
        x = b.x[inb]
        y = b.y[inb]
        p = b.p[inb]
        m = signal_mask(
            clean_keys=clean_keys,
            packer=packer,
            t=t,
            x=x,
            y=y,
            p=p,
            match_ticks=int(match_ticks),
            match_bin_radius=int(match_bin_radius),
        )
        sig_n = int(np.count_nonzero(m))
        n = int(t.shape[0])
        total += n
        signal += sig_n

    noise = total - signal
    return Kept(total=total, signal=signal, noise=noise)


def confusion_from_totals(tot: Totals, kept: Kept, *, roc_convention: str = "paper") -> Confusion:
    """Build TP/FP/TN/FN.

    roc_convention:
    - "paper": positive=signal, predicted positive=kept
      TP=signal_kept, FP=noise_kept, TN=noise_dropped, FN=signal_dropped
      => TPR=signal_kept/signal_total, FPR=noise_kept/noise_total

    - "noise-drop": (legacy) positive=noise, predicted positive=drop
      TP=noise_dropped, FP=signal_dropped, TN=signal_kept, FN=noise_kept
      => TPR=noise_rejection, FPR=signal_loss
    """

    conv = str(roc_convention).strip().lower()
    if conv == "paper":
        tp = int(kept.signal)
        fp = int(kept.noise)
        tn = int(tot.noise - kept.noise)
        fn = int(tot.signal - kept.signal)
    elif conv == "noise-drop":
        tn = int(kept.signal)
        fn = int(kept.noise)
        fp = int(tot.signal - kept.signal)
        tp = int(tot.noise - kept.noise)
    else:
        raise ValueError(f"Unknown roc_convention: {roc_convention!r}")

    # Clamp small negatives due to potential mismatches.
    if tp < 0:
        tp = 0
    if fp < 0:
        fp = 0
    if tn < 0:
        tn = 0
    if fn < 0:
        fn = 0

    return Confusion(tp=tp, fp=fp, tn=tn, fn=fn)


def auc_trapz(fpr: np.ndarray, tpr: np.ndarray, *, add_endpoints: bool = True) -> float:
    """Compute ROC AUC via trapezoidal rule.

    By default, adds the conventional ROC endpoints (0,0) and (1,1) if missing,
    so the result is comparable to standard ROC-AUC in papers.
    """

    if fpr.size == 0 or tpr.size == 0 or fpr.size != tpr.size:
        return 0.0

    order = np.argsort(fpr)
    x = np.asarray(fpr, dtype=np.float64)[order]
    y = np.asarray(tpr, dtype=np.float64)[order]

    # Ensure finite
    x = np.clip(x, 0.0, 1.0)
    y = np.clip(y, 0.0, 1.0)

    if add_endpoints and x.size > 0:
        if x[0] > 0.0:
            x = np.concatenate((np.array([0.0], dtype=np.float64), x))
            y = np.concatenate((np.array([0.0], dtype=np.float64), y))
        if x[-1] < 1.0:
            x = np.concatenate((x, np.array([1.0], dtype=np.float64)))
            y = np.concatenate((y, np.array([1.0], dtype=np.float64)))

    if x.size < 2:
        return 0.0

    # NumPy 2.0+ removed np.trapz in favor of np.trapezoid.
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is not None:
        return float(trapezoid(y=y, x=x))

    trapz = getattr(np, "trapz", None)
    if trapz is not None:
        return float(trapz(y=y, x=x))

    # Last-resort fallback (should rarely happen): manual trapezoidal integration.
    dx = x[1:] - x[:-1]
    return float(np.sum(dx * (y[1:] + y[:-1]) * 0.5))
