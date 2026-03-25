from __future__ import annotations

from bisect import bisect_left
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np

from ..denoise import DenoiseConfig
from ..events import EventBatch, filter_visibility_batches, unwrap_tick_batches
from ..io.auto import open_events
from ..timebase import TimeBase
from .roc_auc import Kept, Totals, confusion_from_totals, signal_mask


def _wrap_progress(batches: Iterator[EventBatch], *, enabled: bool, desc: str) -> Iterator[EventBatch]:
    if not enabled:
        return batches

    try:
        from tqdm import tqdm  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"--progress requires tqdm. Install it or use the conda env. ({type(e).__name__}: {e})")

    pbar = tqdm(total=None, unit="ev", desc=desc, dynamic_ncols=True)

    def _gen() -> Iterator[EventBatch]:
        try:
            for b in batches:
                yield b
                pbar.update(len(b))
        finally:
            pbar.close()

    return _gen()


def compute_roc_rows_ebf_score_threshold_sweep(
    *,
    noisy_path: str,
    width: int | None,
    height: int | None,
    batch_events: int,
    tick_ns: float,
    hdf5_plugin_path: str | None,
    assume: str | None,
    progress: bool,
    tb: TimeBase,
    unwrap_ts: bool,
    ts_bits: int | None,
    show_on: bool,
    show_off: bool,
    clean_keys,
    packer,
    match_ticks: int,
    match_bin_radius: int,
    tot: Totals,
    tag: str,
    method: str,
    roc_convention: str,
    match_us: int,
    values: Sequence[float],
    time_us: int,
    radius_px: int,
    refractory_us: int,
    print_fn: Callable[[str], None] = print,
) -> list[dict[str, object]]:
    """Fast path for EBF ROC when sweeping the score threshold.

    Rationale: EBF updates its internal state regardless of keep/drop, so the
    score stream is independent of the chosen threshold. We can compute scores
    once, then derive kept counts for all thresholds.

    This is intentionally an internal helper used by CLI.
    """

    thr_unique = sorted({float(v) for v in values})
    if not thr_unique:
        raise SystemExit("--values is empty")

    thresholds = np.asarray(thr_unique, dtype=np.float64)

    use_numba = False
    try:
        from ..denoise.numba_ebf import ebf_state_init, ebf_update_prefix_diffs_numba, is_numba_available

        use_numba = bool(is_numba_available())
    except Exception:
        use_numba = False

    if use_numba:
        print_fn("ebf sweep: using numba backend")
    else:
        print_fn("ebf sweep: using python backend (slow)")

    # Re-open noisy once for scoring (streams are forward-only).
    r_score = open_events(
        noisy_path,
        width=width,
        height=height,
        batch_events=batch_events,
        tick_ns=float(tick_ns),
        hdf5_plugin_path=hdf5_plugin_path,
        assume=assume,
    )

    w = int(r_score.meta.width)
    h = int(r_score.meta.height)

    score_batches = _wrap_progress(
        r_score.batches,
        enabled=bool(progress),
        desc=f"roc ebf score: {noisy_path.split('/')[-1].split('\\\\')[-1]}",
    )
    if unwrap_ts:
        score_batches = unwrap_tick_batches(score_batches, bits=ts_bits)
    score_batches = filter_visibility_batches(score_batches, show_on=show_on, show_off=show_off)

    k = len(thr_unique)
    diff_total = np.zeros((k + 1,), dtype=np.int64)
    diff_signal = np.zeros((k + 1,), dtype=np.int64)

    if use_numba:
        last_ts, last_pol = ebf_state_init(w, h)
        tau_ticks = int(tb.us_to_ticks(int(time_us)))
        r_px = int(radius_px)

        for b in score_batches:
            x0 = np.asarray(b.x)
            y0 = np.asarray(b.y)
            inb = (x0 < w) & (y0 < h)
            if not bool(np.any(inb)):
                continue

            t = np.asarray(b.t[inb], dtype=np.uint64)
            x = np.asarray(b.x[inb], dtype=np.int32)
            y = np.asarray(b.y[inb], dtype=np.int32)
            p_arr = np.asarray(b.p[inb], dtype=np.int8)

            sig = signal_mask(
                clean_keys=clean_keys,
                packer=packer,
                t=t,
                x=x,
                y=y,
                p=p_arr,
                match_ticks=int(match_ticks),
                match_bin_radius=int(match_bin_radius),
            )

            ebf_update_prefix_diffs_numba(
                t=t,
                x=x,
                y=y,
                p=p_arr,
                signal_u8=sig.astype(np.uint8),
                width=w,
                height=h,
                radius_px=r_px,
                tau_ticks=tau_ticks,
                thresholds=thresholds,
                last_ts=last_ts,
                last_pol=last_pol,
                diff_total=diff_total,
                diff_signal=diff_signal,
            )

    else:
        from ..denoise.ops.base import Dims
        from ..denoise.ops.ebf import EbfOp

        # EBF scoring op (stateful). Threshold is applied outside.
        cfg_score = DenoiseConfig(
            method=str(method),
            pipeline=None,
            time_window_us=int(time_us),
            radius_px=int(radius_px),
            min_neighbors=0,
            refractory_us=int(refractory_us),
            show_on=show_on,
            show_off=show_off,
        )
        op = EbfOp(Dims(width=w, height=h), cfg_score, tb)

        for b in score_batches:
            x0 = np.asarray(b.x)
            y0 = np.asarray(b.y)
            inb = (x0 < w) & (y0 < h)
            if not bool(np.any(inb)):
                continue

            t = np.asarray(b.t[inb], dtype=np.uint64)
            x = np.asarray(b.x[inb], dtype=np.int32)
            y = np.asarray(b.y[inb], dtype=np.int32)
            p_arr = np.asarray(b.p[inb], dtype=np.int8)

            sig = signal_mask(
                clean_keys=clean_keys,
                packer=packer,
                t=t,
                x=x,
                y=y,
                p=p_arr,
                match_ticks=int(match_ticks),
                match_bin_radius=int(match_bin_radius),
            )

            for i in range(int(t.shape[0])):
                s = float(op.score(int(x[i]), int(y[i]), int(p_arr[i]), int(t[i])))
                idx = int(bisect_left(thr_unique, s))  # thresholds < score
                if idx <= 0:
                    continue
                diff_total[0] += 1
                diff_total[idx] -= 1
                if bool(sig[i]):
                    diff_signal[0] += 1
                    diff_signal[idx] -= 1

    kept_total_u = np.cumsum(diff_total[:-1], dtype=np.int64)
    kept_signal_u = np.cumsum(diff_signal[:-1], dtype=np.int64)

    thr_to_idx = {thr: i for i, thr in enumerate(thr_unique)}

    rows: list[dict[str, object]] = []
    for v in values:
        vv = float(v)
        ii = int(thr_to_idx[vv])
        kt = int(kept_total_u[ii])
        ks = int(kept_signal_u[ii])
        kept = Kept(total=kt, signal=ks, noise=int(kt - ks))
        conf = confusion_from_totals(tot, kept, roc_convention=str(roc_convention))

        rows.append(
            {
                "tag": tag,
                "method": str(method),
                "param": "min-neighbors",
                "value": vv,
                "roc_convention": str(roc_convention),
                "match_us": int(match_us),
                "events_total": int(tot.total),
                "signal_total": int(tot.signal),
                "noise_total": int(tot.noise),
                "events_kept": int(kept.total),
                "signal_kept": int(kept.signal),
                "noise_kept": int(kept.noise),
                "tp": int(conf.tp),
                "fp": int(conf.fp),
                "tn": int(conf.tn),
                "fn": int(conf.fn),
                "tpr": float(conf.tpr),
                "fpr": float(conf.fpr),
                "precision": float(conf.precision),
                "accuracy": float(conf.accuracy),
                "f1": float(conf.f1),
            }
        )

        print_fn(
            f"r={int(radius_px):>2} min-neighbors={vv:>8}  kept={kept.total:<10} "
            f"tpr={conf.tpr:.6f} fpr={conf.fpr:.6f} "
            f"tp={conf.tp} fp={conf.fp} tn={conf.tn} fn={conf.fn}"
        )

    return rows
