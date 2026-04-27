from __future__ import annotations

import argparse
import os
import csv
from dataclasses import replace
from typing import Iterator

import numpy as np

from .denoise import DenoiseConfig, denoise_stream
from .events import EventBatch, filter_visibility_batches, unwrap_tick_batches
from .io.auto import open_events
from .io.csv_events import write_csv_events
from .io.aedat2_events import write_aedat2, write_aedat2_passthrough
from .io.evtq import write_evtq
from .io.hdf5_events import write_hdf5
from .stats import compute_stats
from .timebase import TimeBase


def _add_common_in(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--in",
        dest="in_path",
        required=True,
        help="input path (.evtq/.csv/.hdf5/.h5/.aedat/.bin/.npy/.npz)",
    )
    parser.add_argument(
        "--assume",
        default=None,
        choices=["evtq", "csv", "hdf5", "aedat2", "usb_raw_evt3", "npy", "npz"],
        help="override input kind (omit to auto-detect)",
    )
    parser.add_argument("--width", type=int, default=640, help="required for csv/usb_raw, optional for hdf5")
    parser.add_argument("--height", type=int, default=512, help="required for csv/usb_raw, optional for hdf5")
    parser.add_argument("--batch-events", type=int, default=1_000_000)
    parser.add_argument(
        "--hdf5-plugin-path",
        default=None,
        help="optional HDF5 plugin dir (for OpenEB compressed HDF5, e.g. .../build/lib/hdf5/plugin)",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="show a progress bar (tqdm). Adds small overhead; best used for long runs",
    )


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


def _open(args) -> tuple:
    r = open_events(
        args.in_path,
        width=args.width,
        height=args.height,
        batch_events=args.batch_events,
        tick_ns=float(getattr(args, "tick_ns", 12.5)),
        hdf5_plugin_path=getattr(args, "hdf5_plugin_path", None),
        assume=args.assume,
    )
    desc = f"{getattr(args, 'cmd', 'run')}: {os.path.basename(str(args.in_path))}"
    return r.meta, _wrap_progress(r.batches, enabled=bool(getattr(args, "progress", False)), desc=desc)


def _open_path(path: str, args, *, desc: str) -> tuple:
    r = open_events(
        path,
        width=args.width,
        height=args.height,
        batch_events=args.batch_events,
        tick_ns=float(getattr(args, "tick_ns", 12.5)),
        hdf5_plugin_path=getattr(args, "hdf5_plugin_path", None),
        assume=getattr(args, "assume", None),
    )
    return r.meta, _wrap_progress(r.batches, enabled=bool(getattr(args, "progress", False)), desc=desc)


def cmd_convert(args) -> int:
    meta, batches = _open(args)
    out_path = args.out_path
    ext = os.path.splitext(out_path)[1].lower()

    if ext == ".evtq":
        write_evtq(out_path, meta, batches)
        return 0
    if ext == ".csv":
        write_csv_events(out_path, batches)
        return 0
    if ext in (".hdf5", ".h5"):
        write_hdf5(out_path, meta, batches, tick_ns=float(getattr(args, "tick_ns", 12.5)))
        return 0

    if ext in (".aedat", ".aedat2"):
        write_aedat2(out_path, meta, batches, tick_ns=float(getattr(args, "tick_ns", 12.5)), dst_tick_us=1.0)
        return 0

    raise SystemExit("--out must end with .evtq, .csv, .hdf5/.h5 or .aedat/.aedat2")


def cmd_denoise(args) -> int:
    meta, batches = _open(args)
    # Timebase: you confirmed 1 tick = 12.5ns.
    # Keep it configurable in case you reuse this tool on other sensors.
    tb = TimeBase(tick_ns=float(args.tick_ns))

    if bool(getattr(args, "unwrap_ts", True)):
        bits = str(getattr(args, "ts_bits", "auto"))
        bits_i = int(bits) if bits.isdigit() else None
        batches = unwrap_tick_batches(batches, bits=bits_i)

    pipeline = None
    if args.pipeline:
        # Comma-separated, e.g. "globalgate,stc,refractory" or "7,1,2"
        pipeline = [x.strip() for x in str(args.pipeline).split(",") if x.strip()]

    cfg = DenoiseConfig(
        method=args.method,
        pipeline=pipeline,
        time_window_us=args.time_us,
        radius_px=args.radius_px,
        min_neighbors=args.min_neighbors,
        refractory_us=args.refractory_us,
        show_on=(not args.hide_on),
        show_off=(not args.hide_off),
        mlpf_model_path=str(getattr(args, "mlpf_model", "") or ""),
        mlpf_patch=int(getattr(args, "mlpf_patch", 7) or 7),
        pfd_mode=str(getattr(args, "pfd_mode", "a") or "a"),
    )

    out_path = args.out_path
    ext = os.path.splitext(out_path)[1].lower()

    den = denoise_stream(meta, batches, cfg, timebase=tb, engine=str(getattr(args, "engine", "python")))

    if ext == ".evtq":
        write_evtq(out_path, meta, den)
        return 0
    if ext == ".csv":
        write_csv_events(out_path, den)
        return 0
    if ext in (".hdf5", ".h5"):
        write_hdf5(out_path, meta, den, tick_ns=float(tb.tick_ns))
        return 0

    if ext in (".aedat", ".aedat2"):
        write_aedat2(out_path, meta, den, tick_ns=float(tb.tick_ns), dst_tick_us=1.0)
        return 0

    raise SystemExit("--out must end with .evtq, .csv, .hdf5/.h5 or .aedat/.aedat2")


def cmd_view(args) -> int:
    meta, batches = _open(args)
    # Lazy import so users can run convert/denoise/stats without OpenCV installed.
    from .viz import view_stream

    style = str(getattr(args, "style", "myevs")).strip().lower()

    mode = str(args.mode)
    events_per_frame = int(args.events_per_frame)
    raw_step = int(args.raw_step)
    deadzone = int(args.deadzone)
    binary = bool(args.binary)
    hold = bool(args.hold)
    scheme = int(args.scheme)
    unwrap_ts = bool(getattr(args, "unwrap_ts", True))

    # One-click preset to mimic Prophesee-like event visualization.
    # Only override fields that are still at parser defaults, so user flags win.
    if style == "prophesee":
        if raw_step == 10:
            raw_step = 20
        if deadzone == 3:
            deadzone = 0
        if not binary:
            binary = True
        if hold:
            hold = False
        if unwrap_ts:
            unwrap_ts = False

    tb = TimeBase(tick_ns=float(args.tick_ns))
    flip_x = bool(getattr(args, "flip_x", False) or getattr(args, "rotate_180", False))
    flip_y = bool(getattr(args, "flip_y", False) or getattr(args, "rotate_180", False))

    if unwrap_ts:
        bits = str(getattr(args, "ts_bits", "auto"))
        bits_i = int(bits) if bits.isdigit() else None
        batches = unwrap_tick_batches(batches, bits=bits_i)

    view_stream(
        meta,
        batches,
        mode=mode,
        fps=args.fps,
        events_per_frame=events_per_frame,
        tick_us=tb.tick_us,
        color=args.color,
        scheme_id=scheme,
        window_name=args.window,
        raw_step=raw_step,
        deadzone=deadzone,
        binary=binary,
        hold=hold,
        show_on=(not args.hide_on),
        show_off=(not args.hide_off),
        realtime=args.realtime,
        out_video=args.out_video,
        video_fps=args.video_fps,
        no_gui=args.no_gui,
        flip_x=flip_x,
        flip_y=flip_y,
    )
    return 0


def cmd_stats(args) -> int:
    meta, batches = _open(args)
    try:
        size = os.path.getsize(args.in_path)
        print(f"file: {args.in_path} ({size} bytes)")
    except OSError:
        pass
    print(f"meta: {int(meta.width)}x{int(meta.height)} time_unit={meta.time_unit}")

    show_on = not bool(getattr(args, "hide_on", False))
    show_off = not bool(getattr(args, "hide_off", False))

    if bool(getattr(args, "unwrap_ts", True)):
        bits = str(getattr(args, "ts_bits", "auto"))
        bits_i = int(bits) if bits.isdigit() else None
        batches = unwrap_tick_batches(batches, bits=bits_i)

    batches = filter_visibility_batches(batches, show_on=show_on, show_off=show_off)
    st = compute_stats(meta, batches)
    print(f"events: {st.total} (on={st.on}, off={st.off})")
    if st.t_first is not None:
        print(f"t_first: {st.t_first}")
        print(f"t_last:  {st.t_last}")
        print(f"duration_ticks: {st.duration_ticks}")
    if st.total == 0:
        print("hint: events=0 usually means the input file has no events (only header) or you pointed to a wrong/empty file.")
    return 0


def cmd_compare_stats(args) -> int:
    a_meta, a_batches = _open_path(args.in_a, args, desc=f"compare A: {os.path.basename(str(args.in_a))}")
    b_meta, b_batches = _open_path(args.in_b, args, desc=f"compare B: {os.path.basename(str(args.in_b))}")

    if int(a_meta.width) != int(b_meta.width) or int(a_meta.height) != int(b_meta.height):
        raise SystemExit(f"meta mismatch: A={a_meta.width}x{a_meta.height}, B={b_meta.width}x{b_meta.height}")

    show_on = not bool(getattr(args, "hide_on", False))
    show_off = not bool(getattr(args, "hide_off", False))

    if bool(getattr(args, "unwrap_ts", True)):
        bits = str(getattr(args, "ts_bits", "auto"))
        bits_i = int(bits) if bits.isdigit() else None
        a_batches = unwrap_tick_batches(a_batches, bits=bits_i)
        b_batches = unwrap_tick_batches(b_batches, bits=bits_i)

    a_batches = filter_visibility_batches(a_batches, show_on=show_on, show_off=show_off)
    b_batches = filter_visibility_batches(b_batches, show_on=show_on, show_off=show_off)

    st_a = compute_stats(a_meta, a_batches)
    st_b = compute_stats(b_meta, b_batches)

    total_a = float(st_a.total)
    total_b = float(st_b.total)
    kept_ratio = (total_b / total_a) if total_a > 0 else 0.0
    removed_ratio = 1.0 - kept_ratio if total_a > 0 else 0.0

    print(f"A: {args.in_a}")
    print(f"  events: {st_a.total} (on={st_a.on}, off={st_a.off})")
    print(f"B: {args.in_b}")
    print(f"  events: {st_b.total} (on={st_b.on}, off={st_b.off})")
    print(f"kept_ratio: {kept_ratio:.6f}  removed_ratio: {removed_ratio:.6f}")

    if args.out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["name", "path", "events", "on", "off"])
            w.writerow(["A", args.in_a, st_a.total, st_a.on, st_a.off])
            w.writerow(["B", args.in_b, st_b.total, st_b.on, st_b.off])
            w.writerow([])
            w.writerow(["kept_ratio", kept_ratio])
            w.writerow(["removed_ratio", removed_ratio])

    return 0


def cmd_sweep(args) -> int:
    tb = TimeBase(tick_ns=float(args.tick_ns))

    show_on = not bool(getattr(args, "hide_on", False))
    show_off = not bool(getattr(args, "hide_off", False))
    unwrap_ts = bool(getattr(args, "unwrap_ts", True))
    bits = str(getattr(args, "ts_bits", "auto"))
    bits_i = int(bits) if bits.isdigit() else None

    # Parse values list (comma-separated)
    raw_vals = [x.strip() for x in str(args.values).split(",") if x.strip()]
    if str(args.param) == "min-neighbors":
        values = [float(x) for x in raw_vals]
    else:
        values = [int(float(x)) for x in raw_vals]
    if not values:
        raise SystemExit("--values must be a comma-separated list")

    # Compute input total once (one pass)
    r0 = open_events(
        args.in_path,
        width=args.width,
        height=args.height,
        batch_events=args.batch_events,
        tick_ns=float(args.tick_ns),
        hdf5_plugin_path=getattr(args, "hdf5_plugin_path", None),
        assume=args.assume,
    )
    in_batches = _wrap_progress(
        r0.batches,
        enabled=bool(getattr(args, "progress", False)),
        desc=f"sweep input: {os.path.basename(str(args.in_path))}",
    )
    if unwrap_ts:
        in_batches = unwrap_tick_batches(in_batches, bits=bits_i)
    in_batches = filter_visibility_batches(in_batches, show_on=show_on, show_off=show_off)
    st_in = compute_stats(r0.meta, in_batches)
    total_in = int(st_in.total)
    on_in = int(st_in.on)
    off_in = int(st_in.off)
    print(f"input events: {total_in}  (on={st_in.on}, off={st_in.off})")

    rows: list[tuple[float, int, int, int, float, float]] = []
    for v in values:
        # Re-open input each sweep point (stream is forward-only)
        r = open_events(
            args.in_path,
            width=args.width,
            height=args.height,
            batch_events=args.batch_events,
            tick_ns=float(args.tick_ns),
            hdf5_plugin_path=getattr(args, "hdf5_plugin_path", None),
            assume=args.assume,
        )

        cfg = DenoiseConfig(
            method=args.method,
            pipeline=None,
            time_window_us=args.time_us,
            radius_px=args.radius_px,
            min_neighbors=args.min_neighbors,
            refractory_us=args.refractory_us,
            show_on=(not args.hide_on),
            show_off=(not args.hide_off),
            mlpf_model_path=str(getattr(args, "mlpf_model", "") or ""),
            mlpf_patch=int(getattr(args, "mlpf_patch", 7) or 7),
            pfd_mode=str(getattr(args, "pfd_mode", "a") or "a"),
        )

        # Override one parameter
        p = str(args.param)
        if p == "time-us":
            cfg = replace(cfg, time_window_us=int(v))
        elif p == "radius-px":
            cfg = replace(cfg, radius_px=int(v))
        elif p == "min-neighbors":
            cfg = replace(cfg, min_neighbors=float(v))
        elif p == "refractory-us":
            cfg = replace(cfg, refractory_us=int(v))
        else:
            raise SystemExit(f"Unknown param: {p}")

        batches = _wrap_progress(
            r.batches,
            enabled=bool(getattr(args, "progress", False)),
            desc=f"sweep {args.param}={v}",
        )
        if unwrap_ts:
            batches = unwrap_tick_batches(batches, bits=bits_i)

        den = denoise_stream(
            r.meta,
            batches,
            cfg,
            timebase=tb,
            engine=str(getattr(args, "engine", "python")),
        )
        den = filter_visibility_batches(den, show_on=show_on, show_off=show_off)
        st_out = compute_stats(r.meta, den)
        total_out = int(st_out.total)
        on_out = int(st_out.on)
        off_out = int(st_out.off)
        kept = (float(total_out) / float(total_in)) if total_in > 0 else 0.0
        removed = 1.0 - kept if total_in > 0 else 0.0
        rows.append((float(v), total_out, on_out, off_out, kept, removed))
        print(f"{p}={v:>8}  out={total_out:<10} kept={kept:.6f} removed={removed:.6f}")

    if args.out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "param",
                    "value",
                    "events_in",
                    "on_in",
                    "off_in",
                    "events_out",
                    "on_out",
                    "off_out",
                    "kept_ratio",
                    "removed_ratio",
                ]
            )
            for v, out_n, out_on, out_off, kept, removed in rows:
                w.writerow([args.param, v, total_in, on_in, off_in, out_n, out_on, out_off, kept, removed])

    return 0


def cmd_roc(args) -> int:
    """Compute ROC points and AUC against a clean reference.

    ROC convention is controlled by --roc-convention.
    - paper: positive=signal, predicted positive=kept (matches most papers)
    - noise-drop: positive=noise, predicted positive=drop (legacy)
    """

    from .metrics.roc_auc import (
        Kept,
        auc_trapz,
        build_clean_index,
        compute_kept_for_denoised,
        compute_totals_for_noisy,
        confusion_from_totals,
        signal_mask,
    )

    tb = TimeBase(tick_ns=float(args.tick_ns))

    roc_conv = str(getattr(args, "roc_convention", "paper") or "paper").strip().lower()
    if roc_conv not in ("paper", "noise-drop"):
        raise SystemExit(f"--roc-convention must be 'paper' or 'noise-drop' (got {roc_conv!r})")
    if roc_conv == "paper":
        print("roc convention: paper (positive=signal, predicted positive=kept)")
    else:
        print("roc convention: noise-drop (positive=noise, predicted positive=drop)")

    match_us = int(getattr(args, "match_us", 0) or 0)
    match_ticks = tb.us_to_ticks(match_us) if match_us > 0 else 0
    match_bin_radius = int(getattr(args, "match_bin_radius", 1) or 0)

    if match_ticks > 0:
        print(f"label match: match_us={match_us}  match_ticks={match_ticks}  bin_radius={match_bin_radius}")
    else:
        print("label match: exact (t,x,y,p)")

    show_on = not bool(getattr(args, "hide_on", False))
    show_off = not bool(getattr(args, "hide_off", False))
    unwrap_ts = bool(getattr(args, "unwrap_ts", True))
    bits = str(getattr(args, "ts_bits", "auto"))
    bits_i = int(bits) if bits.isdigit() else None

    # Parse values list (comma-separated).
    raw_vals = [x.strip() for x in str(args.values).split(",") if x.strip()]
    if not raw_vals:
        raise SystemExit("--values must be a comma-separated list")

    p = str(args.param)
    if p == "min-neighbors":
        values: list[float] = [float(x) for x in raw_vals]
    else:
        values = [float(int(x)) for x in raw_vals]

    # ---- Build clean index once ----
    r_clean = open_events(
        args.clean_path,
        width=args.width,
        height=args.height,
        batch_events=args.batch_events,
        tick_ns=float(args.tick_ns),
        hdf5_plugin_path=getattr(args, "hdf5_plugin_path", None),
        assume=args.assume,
    )
    clean_batches = _wrap_progress(
        r_clean.batches,
        enabled=bool(getattr(args, "progress", False)),
        desc=f"roc clean: {os.path.basename(str(args.clean_path))}",
    )
    clean_keys, packer = build_clean_index(
        r_clean.meta,
        clean_batches,
        show_on=show_on,
        show_off=show_off,
        unwrap_ts=unwrap_ts,
        ts_bits=bits_i,
        match_ticks=int(match_ticks),
        match_bin_radius=int(match_bin_radius),
    )

    # ---- Compute noisy totals once (same for all sweep points) ----
    r_noisy0 = open_events(
        args.noisy_path,
        width=args.width,
        height=args.height,
        batch_events=args.batch_events,
        tick_ns=float(args.tick_ns),
        hdf5_plugin_path=getattr(args, "hdf5_plugin_path", None),
        assume=args.assume,
    )
    if int(r_noisy0.meta.width) != int(r_clean.meta.width) or int(r_noisy0.meta.height) != int(r_clean.meta.height):
        raise SystemExit(
            f"meta mismatch: clean={r_clean.meta.width}x{r_clean.meta.height}, noisy={r_noisy0.meta.width}x{r_noisy0.meta.height}"
        )
    noisy_batches0 = _wrap_progress(
        r_noisy0.batches,
        enabled=bool(getattr(args, "progress", False)),
        desc=f"roc totals: {os.path.basename(str(args.noisy_path))}",
    )
    tot = compute_totals_for_noisy(
        r_noisy0.meta,
        noisy_batches0,
        clean_keys=clean_keys,
        packer=packer,
        show_on=show_on,
        show_off=show_off,
        unwrap_ts=unwrap_ts,
        ts_bits=bits_i,
        match_ticks=int(match_ticks),
        match_bin_radius=int(match_bin_radius),
    )
    print(f"noisy totals: events={tot.total}  signal={tot.signal}  noise={tot.noise}")

    tag = str(getattr(args, "tag", "") or "").strip()
    if not tag:
        tag = os.path.splitext(os.path.basename(str(args.noisy_path)))[0]

    rows: list[dict[str, object]] = []

    save_dir = str(getattr(args, "save_denoised_dir", "") or "").strip()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # ---- Fast path for EBF score-threshold sweep (author-style) ----
    # When param is the score threshold (min-neighbors), re-running the full denoise
    # for each threshold is unnecessarily expensive. Because EBF updates its state
    # regardless of keep/drop, the score sequence is independent of the threshold.
    # So we can compute scores once, then derive kept counts for all thresholds.
    method_token = str(getattr(args, "method", "") or "").strip().lower()
    method_token = {
        "10": "ebf",
        "eventbasedfilter": "ebf",
    }.get(method_token, method_token)
    engine = str(getattr(args, "engine", "python") or "python").strip().lower()

    if method_token == "ebf" and p == "min-neighbors" and engine == "python":
        if save_dir:
            raise SystemExit(
                "EBF score-threshold sweep does not support --save-denoised-dir (would require writing one stream per threshold). "
                "Omit --save-denoised-dir, or run per-threshold (slower) by sweeping a different parameter."
            )

        from .metrics.roc_ebf_threshold_sweep import compute_roc_rows_ebf_score_threshold_sweep

        rows = compute_roc_rows_ebf_score_threshold_sweep(
            noisy_path=str(args.noisy_path),
            width=args.width,
            height=args.height,
            batch_events=int(args.batch_events),
            tick_ns=float(args.tick_ns),
            hdf5_plugin_path=getattr(args, "hdf5_plugin_path", None),
            assume=args.assume,
            progress=bool(getattr(args, "progress", False)),
            tb=tb,
            unwrap_ts=unwrap_ts,
            ts_bits=bits_i,
            show_on=show_on,
            show_off=show_off,
            clean_keys=clean_keys,
            packer=packer,
            match_ticks=int(match_ticks),
            match_bin_radius=int(match_bin_radius),
            tot=tot,
            tag=tag,
            method=str(args.method),
            roc_convention=roc_conv,
            match_us=int(match_us),
            values=values,
            time_us=int(args.time_us),
            radius_px=int(args.radius_px),
            refractory_us=int(args.refractory_us),
            print_fn=print,
        )

    else:
        for v in values:
            # Re-open noisy each sweep point (stream is forward-only)
            r = open_events(
                args.noisy_path,
                width=args.width,
                height=args.height,
                batch_events=args.batch_events,
                tick_ns=float(args.tick_ns),
                hdf5_plugin_path=getattr(args, "hdf5_plugin_path", None),
                assume=args.assume,
            )

            cfg = DenoiseConfig(
                method=args.method,
                pipeline=None,
                time_window_us=int(args.time_us),
                radius_px=int(args.radius_px),
                min_neighbors=float(args.min_neighbors),
                refractory_us=int(args.refractory_us),
                show_on=show_on,
                show_off=show_off,
                mlpf_model_path=str(getattr(args, "mlpf_model", "") or ""),
                mlpf_patch=int(getattr(args, "mlpf_patch", 7) or 7),
                pfd_mode=str(getattr(args, "pfd_mode", "a") or "a"),
            )

            # Override one parameter.
            if p == "time-us":
                cfg = replace(cfg, time_window_us=int(v))
            elif p == "radius-px":
                cfg = replace(cfg, radius_px=int(v))
            elif p == "min-neighbors":
                cfg = replace(cfg, min_neighbors=float(v))
            elif p == "refractory-us":
                cfg = replace(cfg, refractory_us=int(v))
            else:
                raise SystemExit(f"Unknown param: {p}")

            batches = _wrap_progress(
                r.batches,
                enabled=bool(getattr(args, "progress", False)),
                desc=f"roc r={cfg.radius_px} {args.param}={v}",
            )
            if unwrap_ts:
                batches = unwrap_tick_batches(batches, bits=bits_i)

            den = denoise_stream(
                r.meta,
                batches,
                cfg,
                timebase=tb,
                engine=str(getattr(args, "engine", "python")),
            )

            if save_dir:
                # Write denoised output while it is being consumed for metrics.
                v_str = str(v)
                safe_v = "".join((c if (c.isalnum() or c in "._-") else "_") for c in v_str)
                safe_tag = "".join((c if (c.isalnum() or c in "._-") else "_") for c in tag)
                safe_method = "".join((c if (c.isalnum() or c in "._-") else "_") for c in str(args.method))
                fname = f"{safe_tag}_{safe_method}_{p}_{safe_v}.aedat"
                out_den = os.path.join(save_dir, fname)
                den = write_aedat2_passthrough(out_den, r.meta, den, tick_ns=float(tb.tick_ns), dst_tick_us=1.0)

            kept = compute_kept_for_denoised(
                r.meta,
                den,
                clean_keys=clean_keys,
                packer=packer,
                match_ticks=int(match_ticks),
                match_bin_radius=int(match_bin_radius),
            )
            conf = confusion_from_totals(tot, kept, roc_convention=roc_conv)

            rows.append(
                {
                    "tag": tag,
                    "method": str(args.method),
                    "param": p,
                    "value": v,
                    "roc_convention": roc_conv,
                    "match_us": match_us,
                    "events_total": tot.total,
                    "signal_total": tot.signal,
                    "noise_total": tot.noise,
                    "events_kept": kept.total,
                    "signal_kept": kept.signal,
                    "noise_kept": kept.noise,
                    "tp": conf.tp,
                    "fp": conf.fp,
                    "tn": conf.tn,
                    "fn": conf.fn,
                    "tpr": conf.tpr,
                    "fpr": conf.fpr,
                    "precision": conf.precision,
                    "accuracy": conf.accuracy,
                    "f1": conf.f1,
                }
            )

            print(
                f"r={cfg.radius_px:>2} {p}={v:>8}  kept={kept.total:<10} "
                f"tpr={conf.tpr:.6f} fpr={conf.fpr:.6f} "
                f"tp={conf.tp} fp={conf.fp} tn={conf.tn} fn={conf.fn}"
            )

    fpr_arr = np.asarray([float(r["fpr"]) for r in rows], dtype=np.float64)
    tpr_arr = np.asarray([float(r["tpr"]) for r in rows], dtype=np.float64)
    auc = auc_trapz(fpr_arr, tpr_arr)
    print(f"auc: {auc:.6f}")

    if args.out_csv:
        out_csv = str(args.out_csv)
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)

        append = bool(getattr(args, "append", False))
        header = [
                "tag",
                "method",
                "param",
                "value",
                "roc_convention",
                "match_us",
                "events_total",
                "signal_total",
                "noise_total",
                "events_kept",
                "signal_kept",
                "noise_kept",
                "tp",
                "fp",
                "tn",
                "fn",
                "tpr",
                "fpr",
                "precision",
                "accuracy",
                "f1",
                "auc",
            ]

        if append and os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
            with open(out_csv, "r", newline="", encoding="utf-8") as f0:
                existing = next(csv.reader(f0), None)
            if existing and existing != header:
                raise SystemExit(
                    "--append used but the existing CSV header differs from the current format. "
                    "Delete the old file or use a new --out-csv path."
                )

        need_header = not (append and os.path.exists(out_csv) and os.path.getsize(out_csv) > 0)

        mode = "a" if append else "w"
        with open(out_csv, mode, newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if need_header:
                w.writerow(header)
            for r in rows:
                w.writerow([r.get(k, "") for k in header[:-1]] + [auc])

    return 0


def cmd_plot_csv(args) -> int:
    from .metrics.plot_csv import PlotConfig, plot_csv

    cfg = PlotConfig(
        in_csv=args.in_csv,
        out_path=args.out,
        x=args.x,
        y=tuple(args.y),
        group=args.group,
        kind=args.kind,
        title=args.title,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        logx=args.logx,
        logy=args.logy,
        dpi=int(args.dpi),
        style=args.style,
        legend=(not args.no_legend),
        grid=(not args.no_grid),
    )

    out_path = plot_csv(cfg)
    print(f"saved: {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="myevs", description="Offline EVS tools")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_conv = sub.add_parser("convert", help="convert between evtq/csv/hdf5 (input auto-detect)")
    _add_common_in(p_conv)
    p_conv.add_argument("--tick-ns", type=float, default=12.5, help="timestamp tick length in ns (for hdf5 time conversion)")
    p_conv.add_argument("--out", dest="out_path", required=True, help="output .evtq/.csv/.hdf5/.aedat")
    p_conv.set_defaults(func=cmd_convert)

    p_usb = sub.add_parser("convert-usb-raw", help="usb raw (.bin) -> evtq/csv/hdf5")
    _add_common_in(p_usb)
    p_usb.set_defaults(assume="usb_raw_evt3")
    p_usb.add_argument("--tick-ns", type=float, default=12.5, help="timestamp tick length in ns")
    p_usb.add_argument("--out", dest="out_path", required=True, help="output .evtq/.csv/.hdf5/.aedat")
    p_usb.set_defaults(func=cmd_convert)

    p_den = sub.add_parser("denoise", help="denoise and save")
    _add_common_in(p_den)
    p_den.add_argument("--out", dest="out_path", required=True, help="output .evtq/.csv/.hdf5/.aedat")
    p_den.add_argument(
        "--method",
        default="none",
        help=(
            "Qt method id/name: 0 none, 1 stc, 2 refractory, 3 hotpixel, 4 baf, "
            "5 combo(stc+refractory), 6 ratelimit, 7 globalgate, 8 dp, "
            "9 fastdecay (dv-processing FastDecayNoiseFilter; --time-us=half-life, --radius-px=subdivision, --min-neighbors=threshold), "
            "10 ebf (Guo 2025; --time-us=tau, --radius-px=radius (TI25 uses 2), --min-neighbors=score-threshold), "
            "11 ebf_optimized (research; global adaptive noise normalization; --min-neighbors=normalized-threshold), "
            "12 knoise, 13 evflow, 14 ynoise, 15 ts, 16 mlpf, 17 pfd"
        ),
    )
    p_den.add_argument(
        "--pipeline",
        default=None,
        help=(
            "Comma-separated pipeline for composition testing, e.g. 'globalgate,stc,refractory' or '7,1,2'. "
            "If provided, it overrides --method."
        ),
    )
    p_den.add_argument(
        "--engine",
        choices=["python", "numba"],
        default="python",
        help="execution engine (numba currently accelerates stc / ts / evflow / pfd for single-method runs)",
    )
    p_den.add_argument(
        "--tick-ns",
        type=float,
        default=12.5,
        help="timestamp tick length in ns (default 12.5ns per your sensor)",
    )
    p_den.add_argument("--time-us", type=int, default=2000, help="time window (us), shared by many methods")
    p_den.add_argument("--radius-px", type=int, default=1, help="spatial radius (px), for stc/baf")
    p_den.add_argument("--min-neighbors", type=float, default=2, help="threshold/limit (meaning depends on method)")
    p_den.add_argument("--refractory-us", type=int, default=50, help="refractory/mask/hold time (us)")
    p_den.add_argument("--mlpf-model", default="", help="optional TorchScript model path for method=mlpf")
    p_den.add_argument("--mlpf-patch", type=int, default=7, help="mlpf patch size (odd, e.g. 7/9/11)")
    p_den.add_argument("--pfd-mode", choices=["a", "b", "A", "B"], default="a", help="pfd mode: a (default) or b")
    p_den.add_argument("--hide-on", action="store_true", help="hide ON events")
    p_den.add_argument("--hide-off", action="store_true", help="hide OFF events")
    p_den.add_argument(
        "--no-unwrap-ts",
        dest="unwrap_ts",
        action="store_false",
        help="disable timestamp unwrapping (wrap/epoch expansion)",
    )
    p_den.add_argument(
        "--ts-bits",
        default="auto",
        choices=["auto", "30", "32"],
        help="wrapped timestamp bit width for unwrapping",
    )
    p_den.set_defaults(func=cmd_denoise)

    p_view = sub.add_parser("view", help="preview events as frames")
    _add_common_in(p_view)
    p_view.add_argument("--mode", choices=["fps", "events"], default="fps")
    p_view.add_argument("--fps", type=float, default=60.0)
    p_view.add_argument("--events-per-frame", type=int, default=200_000)
    p_view.add_argument(
        "--style",
        default="myevs",
        choices=["myevs", "prophesee"],
        help="view style preset: 'prophesee' gives sharper/noisier event look without changing temporal sampling",
    )
    p_view.add_argument("--tick-ns", type=float, default=12.5, help="timestamp tick length in ns")
    p_view.add_argument("--color", choices=["onoff", "gray", "onoff_rb"], default="onoff")
    p_view.add_argument("--scheme", type=int, default=0, help="Qt scheme id: 0 (white bg, red/blue), 1 (dark bg, custom colors)")
    p_view.add_argument("--raw-step", type=int, default=10, help="raw-gray accumulation step per event (Qt rawGrayStep)")
    p_view.add_argument("--deadzone", type=int, default=3, help="deadzone around 127 for binary color mode (Qt rawEnhanceDeadzone)")
    p_view.add_argument("--binary", action="store_true", help="binary polarity color (Qt rawPolarityBinaryEnabled)")
    p_view.add_argument(
        "--realtime",
        action="store_true",
        help="throttle playback in wall-clock time to match --fps (best-effort)",
    )
    p_view.add_argument("--out-video", default=None, help="write video to .mp4/.avi (optional)")
    p_view.add_argument(
        "--video-fps",
        type=float,
        default=None,
        help="output video fps (defaults to --fps)",
    )
    p_view.add_argument(
        "--no-gui",
        action="store_true",
        help="do not open a window (useful with --out-video)",
    )
    p_view.add_argument(
        "--flip-x",
        action="store_true",
        help="flip horizontally (mirror left-right) before rendering/export",
    )
    p_view.add_argument(
        "--flip-y",
        action="store_true",
        help="flip vertically (mirror up-down) before rendering/export",
    )
    p_view.add_argument(
        "--rotate-180",
        action="store_true",
        help="rotate by 180° (equivalent to --flip-x --flip-y)",
    )
    p_view.add_argument(
        "--no-hold",
        dest="hold",
        action="store_false",
        default=True,
        help="do NOT hold gray buffer between frames (Qt rawGrayHold=false)",
    )
    p_view.add_argument("--hide-on", action="store_true", help="hide ON events")
    p_view.add_argument("--hide-off", action="store_true", help="hide OFF events")
    p_view.add_argument(
        "--no-unwrap-ts",
        dest="unwrap_ts",
        action="store_false",
        help="disable timestamp unwrapping (wrap/epoch expansion)",
    )
    p_view.add_argument(
        "--ts-bits",
        default="auto",
        choices=["auto", "30", "32"],
        help="wrapped timestamp bit width for unwrapping",
    )
    p_view.add_argument("--window", default="myEVS")
    p_view.set_defaults(func=cmd_view)

    p_st = sub.add_parser("stats", help="basic stats")
    _add_common_in(p_st)
    p_st.add_argument("--tick-ns", type=float, default=12.5, help="timestamp tick length in ns (for hdf5 time conversion)")
    p_st.add_argument("--hide-on", action="store_true", help="hide ON events")
    p_st.add_argument("--hide-off", action="store_true", help="hide OFF events")
    p_st.add_argument(
        "--no-unwrap-ts",
        dest="unwrap_ts",
        action="store_false",
        help="disable timestamp unwrapping (wrap/epoch expansion)",
    )
    p_st.add_argument(
        "--ts-bits",
        default="auto",
        choices=["auto", "30", "32"],
        help="wrapped timestamp bit width for unwrapping",
    )
    p_st.set_defaults(func=cmd_stats)

    p_cmp = sub.add_parser("compare-stats", help="compare two streams by event counts (kept/removed ratio)")
    p_cmp.add_argument("--in-a", required=True, help="input A (.evtq/.csv/.hdf5/.raw)")
    p_cmp.add_argument("--in-b", required=True, help="input B (.evtq/.csv/.hdf5/.raw)")
    p_cmp.add_argument(
        "--assume",
        default=None,
        choices=["evtq", "csv", "hdf5", "aedat2", "usb_raw_evt3", "npy", "npz"],
        help="override input kind for BOTH A/B (omit to auto-detect by extension)",
    )
    p_cmp.add_argument("--width", type=int, default=None, help="required for csv/usb_raw, optional for hdf5")
    p_cmp.add_argument("--height", type=int, default=None, help="required for csv/usb_raw, optional for hdf5")
    p_cmp.add_argument("--tick-ns", type=float, default=12.5, help="timestamp tick length in ns (for hdf5 time conversion)")
    p_cmp.add_argument(
        "--hdf5-plugin-path",
        default=None,
        help="optional HDF5 plugin dir (for OpenEB compressed HDF5)",
    )
    p_cmp.add_argument("--batch-events", type=int, default=1_000_000)
    p_cmp.add_argument("--progress", action="store_true", help="show a progress bar (tqdm)")
    p_cmp.add_argument("--hide-on", action="store_true", help="hide ON events")
    p_cmp.add_argument("--hide-off", action="store_true", help="hide OFF events")
    p_cmp.add_argument(
        "--no-unwrap-ts",
        dest="unwrap_ts",
        action="store_false",
        help="disable timestamp unwrapping (wrap/epoch expansion)",
    )
    p_cmp.add_argument(
        "--ts-bits",
        default="auto",
        choices=["auto", "30", "32"],
        help="wrapped timestamp bit width for unwrapping",
    )
    p_cmp.add_argument("--out-csv", default=None, help="optional csv report path")
    p_cmp.set_defaults(func=cmd_compare_stats)

    p_sw = sub.add_parser("sweep", help="sweep a denoise parameter and report retention (no output files)")
    _add_common_in(p_sw)
    p_sw.add_argument(
        "--method",
        default="hotpixel",
        help=(
            "Qt method id/name: 0 none, 1 stc, 2 refractory, 3 hotpixel, 4 baf, "
            "5 combo(stc+refractory), 6 ratelimit, 7 globalgate, 8 dp, "
            "9 fastdecay, 10 ebf, 11 ebf_optimized, 12 knoise, 13 evflow, 14 ynoise, 15 ts, 16 mlpf, 17 pfd"
        ),
    )
    p_sw.add_argument(
        "--param",
        required=True,
        choices=["time-us", "radius-px", "min-neighbors", "refractory-us"],
        help="which parameter to sweep",
    )
    p_sw.add_argument(
        "--engine",
        choices=["python", "numba"],
        default="python",
        help="execution engine (numba currently accelerates stc / ts / evflow / pfd for single-method runs)",
    )
    p_sw.add_argument("--values", required=True, help="comma-separated values, e.g. 5,10,20,50")
    p_sw.add_argument("--tick-ns", type=float, default=12.5)
    p_sw.add_argument("--time-us", type=int, default=2000)
    p_sw.add_argument("--radius-px", type=int, default=1)
    p_sw.add_argument("--min-neighbors", type=float, default=2)
    p_sw.add_argument("--refractory-us", type=int, default=50)
    p_sw.add_argument("--mlpf-model", default="", help="optional TorchScript model path for method=mlpf")
    p_sw.add_argument("--mlpf-patch", type=int, default=7, help="mlpf patch size (odd, e.g. 7/9/11)")
    p_sw.add_argument("--pfd-mode", choices=["a", "b", "A", "B"], default="a", help="pfd mode: a (default) or b")
    p_sw.add_argument("--hide-on", action="store_true", help="hide ON events")
    p_sw.add_argument("--hide-off", action="store_true", help="hide OFF events")
    p_sw.add_argument(
        "--no-unwrap-ts",
        dest="unwrap_ts",
        action="store_false",
        help="disable timestamp unwrapping (wrap/epoch expansion)",
    )
    p_sw.add_argument(
        "--ts-bits",
        default="auto",
        choices=["auto", "30", "32"],
        help="wrapped timestamp bit width for unwrapping",
    )
    p_sw.add_argument("--out-csv", default=None, help="optional csv table path")
    p_sw.set_defaults(func=cmd_sweep)

    p_roc = sub.add_parser("roc", help="sweep a denoise parameter and compute ROC/AUC vs a clean reference")
    p_roc.add_argument("--clean", dest="clean_path", required=True, help="clean reference stream (.evtq/.csv/.hdf5/.aedat)")
    p_roc.add_argument("--noisy", dest="noisy_path", required=True, help="noisy stream (.evtq/.csv/.hdf5/.aedat)")
    p_roc.add_argument(
        "--assume",
        default=None,
        choices=["evtq", "csv", "hdf5", "aedat2", "usb_raw_evt3", "npy", "npz"],
        help="override input kind for BOTH clean/noisy (omit to auto-detect)",
    )
    p_roc.add_argument("--width", type=int, default=None, help="required for csv/usb_raw, optional for hdf5")
    p_roc.add_argument("--height", type=int, default=None, help="required for csv/usb_raw, optional for hdf5")
    p_roc.add_argument("--batch-events", type=int, default=1_000_000)
    p_roc.add_argument(
        "--hdf5-plugin-path",
        default=None,
        help="optional HDF5 plugin dir (for OpenEB compressed HDF5)",
    )
    p_roc.add_argument("--progress", action="store_true", help="show a progress bar (tqdm)")
    p_roc.add_argument(
        "--method",
        default="stc",
        help=(
            "Denoise method id/name: 0 none, 1 stc, 2 refractory, 3 hotpixel, 4 baf, "
            "5 combo(stc+refractory), 6 ratelimit, 7 globalgate, 8 dp, 9 fastdecay, 10 ebf, 11 ebf_optimized, "
            "12 knoise, 13 evflow, 14 ynoise, 15 ts, 16 mlpf, 17 pfd"
        ),
    )
    p_roc.add_argument(
        "--param",
        required=True,
        choices=["time-us", "radius-px", "min-neighbors", "refractory-us"],
        help="which parameter to sweep",
    )
    p_roc.add_argument(
        "--values",
        required=True,
        help="comma-separated sweep values (min-neighbors accepts floats like 0.1,0.2,...)",
    )
    p_roc.add_argument(
        "--engine",
        choices=["python", "numba"],
        default="python",
        help="execution engine (numba currently accelerates stc / ts / evflow / pfd for single-method runs)",
    )
    p_roc.add_argument("--tick-ns", type=float, default=12.5)
    p_roc.add_argument("--time-us", type=int, default=2000)
    p_roc.add_argument("--radius-px", type=int, default=1)
    p_roc.add_argument("--min-neighbors", type=float, default=2)
    p_roc.add_argument("--refractory-us", type=int, default=50)
    p_roc.add_argument("--mlpf-model", default="", help="optional TorchScript model path for method=mlpf")
    p_roc.add_argument("--mlpf-patch", type=int, default=7, help="mlpf patch size (odd, e.g. 7/9/11)")
    p_roc.add_argument("--pfd-mode", choices=["a", "b", "A", "B"], default="a", help="pfd mode: a (default) or b")
    p_roc.add_argument("--hide-on", action="store_true", help="hide ON events")
    p_roc.add_argument("--hide-off", action="store_true", help="hide OFF events")
    p_roc.add_argument(
        "--no-unwrap-ts",
        dest="unwrap_ts",
        action="store_false",
        help="disable timestamp unwrapping (wrap/epoch expansion)",
    )
    p_roc.add_argument(
        "--ts-bits",
        default="auto",
        choices=["auto", "30", "32"],
        help="wrapped timestamp bit width for unwrapping",
    )
    p_roc.add_argument("--tag", default=None, help="optional curve label (for grouping in plots)")
    p_roc.add_argument(
        "--roc-convention",
        default="paper",
        choices=["paper", "noise-drop"],
        help=(
            "ROC/TPR/FPR convention: paper=positive(signal), predicted positive(kept); "
            "noise-drop=positive(noise), predicted positive(drop) (legacy)."
        ),
    )
    p_roc.add_argument(
        "--match-us",
        type=int,
        default=1000,
        help="time tolerance (us) for labeling signal by matching against clean (0 = exact match)",
    )
    p_roc.add_argument(
        "--match-bin-radius",
        type=int,
        default=1,
        help=(
            "tolerant-match bin neighborhood radius (0=only same time bin; 1=also check ±1 bins). "
            "Only used when --match-us>0."
        ),
    )
    p_roc.add_argument(
        "--save-denoised-dir",
        default=None,
        help="optional dir to save denoised output (.aedat) for EACH sweep value",
    )
    p_roc.add_argument("--out-csv", default=None, help="output CSV path (recommended)")
    p_roc.add_argument("--append", action="store_true", help="append rows to --out-csv if it exists")
    p_roc.set_defaults(func=cmd_roc)

    p_pc = sub.add_parser("plot-csv", help="plot a CSV table into publication-style figure")
    p_pc.add_argument("--in", dest="in_csv", required=True, help="input csv path")
    p_pc.add_argument("--out", required=True, help="output image path (.png/.pdf/.svg)")
    p_pc.add_argument("--x", default="value", help="x column name")
    p_pc.add_argument(
        "--y",
        nargs="+",
        default=["kept_ratio", "removed_ratio"],
        help="one or more y column names",
    )
    p_pc.add_argument("--group", default=None, help="optional column for grouping into multiple series")
    p_pc.add_argument("--kind", default="auto", choices=["auto", "line", "scatter", "bar", "step"], help="plot type")
    p_pc.add_argument("--title", default=None)
    p_pc.add_argument("--xlabel", default=None)
    p_pc.add_argument("--ylabel", default=None)
    p_pc.add_argument("--logx", action="store_true")
    p_pc.add_argument("--logy", action="store_true")
    p_pc.add_argument("--dpi", type=int, default=220)
    p_pc.add_argument("--style", choices=["paper", "presentation"], default="paper")
    p_pc.add_argument("--no-legend", action="store_true")
    p_pc.add_argument("--no-grid", action="store_true")
    p_pc.set_defaults(func=cmd_plot_csv)

    return p


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
