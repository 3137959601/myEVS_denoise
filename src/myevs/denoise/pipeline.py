from __future__ import annotations

"""Composable denoise pipeline (Qt-aligned).

You asked for:
1) 1 tick = 12.5ns timebase (so all "microseconds" thresholds must be converted)
2) Implement *all* existing Qt denoise methods
3) Make algorithms modular and composable (pipeline)
4) Add lots of comments for a Python beginner

This module is the executor:
- It builds a list of small "ops" (filters) from config.
- It loops over events and keeps/drops them.
- It keeps the subtle state-update rules identical to Qt.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence

import numpy as np

from ..events import EventBatch, EventStreamMeta
from ..timebase import TimeBase
from .types import DenoiseConfig
from .ops.base import Dims, DenoiseOp
from .ops.baf import BafOp
from .ops.dp import DpOp
from .ops.globalgate import GlobalGateOp, keep_by_gate_factor
from .ops.ebf import EbfOp
from .ops.ebf_optimized import EbfOptimizedOp
from .ops.knoise import KnoiseOp
from .ops.evflow import EvFlowOp
from .ops.ynoise import YnoiseOp
from .ops.ts import TsOp
from .ops.mlpf import MlpfOp
from .ops.pfd import PfdOp
from .ops.hotpixel import HotPixelOp
from .ops.fastdecay import FastDecayOp
from .ops.ratelimit import RateLimitOp
from .ops.refractory import RefractoryOp
from .ops.stc import StcOp


# ===== Method mapping (Qt) =====
# Qt side uses integer method IDs. We support both IDs and names in Python.
_METHOD_ID_TO_NAME = {
    0: "none",
    1: "stc",
    2: "refractory",
    3: "hotpixel",
    4: "baf",
    5: "combo",  # stc + refractory
    6: "ratelimit",
    7: "globalgate",
    8: "dp",
    9: "fastdecay",  # dv-processing FastDecayNoiseFilter (non-Qt)
    10: "ebf",  # Guo 2025 EBF (non-Qt)
    11: "ebf_optimized",  # EBF w/ global adaptive noise normalization (research)
    12: "knoise",  # KhodamoradiNoise (cuke-emlb)
    13: "evflow",  # EventFlow (cuke-emlb)
    14: "ynoise",  # YangNoise (cuke-emlb)
    15: "ts",  # TimeSurface (cuke-emlb)
    16: "mlpf",  # MLP-inspired lightweight proxy (cuke-emlb aligned features)
    17: "pfd",  # Polarity-Focused Denoising (PFD/PFDs)
    18: "n149",  # N149 score core (research baseline)
    19: "stcf_original",  # Original STCF (paper version, polarity-agnostic)
}

_NAME_TO_METHOD_ID = {v: k for k, v in _METHOD_ID_TO_NAME.items()}


def _normalize_method_token(token: str) -> str:
    """Normalize user-provided method tokens.

    Accepts:
    - numbers: "5"
    - names: "refractory", "hotpixel"...
    - common aliases
    """

    t = (token or "").strip().lower()
    if not t:
        return "none"

    aliases = {
        "0": "none",
        "1": "stc",
        "2": "refractory",
        "3": "hotpixel",
        "4": "baf",
        "5": "combo",
        "6": "ratelimit",
        "7": "globalgate",
        "8": "dp",
        "9": "fastdecay",
        "10": "ebf",
        "11": "ebf_optimized",
        "12": "knoise",
        "13": "evflow",
        "14": "ynoise",
        "15": "ts",
        "16": "mlpf",
        "17": "pfd",
        "18": "n149",
        "19": "stcf_original",
        "rate": "ratelimit",
        "global": "globalgate",
        "dg": "dp",
        "fast_decay": "fastdecay",
        "fast-decay": "fastdecay",
        "fastdecaynoisefilter": "fastdecay",
        "eventbasedfilter": "ebf",
        "ebfopt": "ebf_optimized",
        "ebf_optim": "ebf_optimized",
        "stcf": "stc",
        "knoisefilter": "knoise",
        "eventflow": "evflow",
        "yangnoise": "ynoise",
        "timesurface": "ts",
        "polarityfocuseddenoising": "pfd",
        "pfds": "pfd",
    }
    if t in aliases:
        return aliases[t]
    return t


def _resolve_mlpf_native_model(model_path: str, cfg: DenoiseConfig, tb: TimeBase) -> tuple[Path, int, int, bool]:
    """Resolve the C++ MLPF weight file and metadata.

    The native backend does not link libtorch. It consumes exported NumPy
    weights plus the same JSON metadata produced by the training script.
    Passing a `.pt` path is allowed when a same-stem `.npz` export exists.
    """

    raw = (model_path or "").strip()
    if not raw:
        raise ValueError("engine='cpp' method=mlpf requires --mlpf-model pointing to exported .npz weights or a .pt with same-stem .npz")

    p = Path(raw)
    if p.suffix.lower() == ".npz":
        weights_path = p
    else:
        weights_path = p.with_suffix(".npz")

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Missing native MLPF weights: {weights_path}. "
            f"Export once with: python scripts/export_mlpf_weights.py --model {p}"
        )

    meta_path = p.with_suffix(".json") if p.suffix.lower() != ".npz" else weights_path.with_suffix(".json")
    meta = {}
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            meta = loaded

    patch = int(meta.get("patch", int(getattr(cfg, "mlpf_patch", 7) or 7)))
    if "duration_ticks" in meta:
        duration_ticks = int(meta["duration_ticks"])
    else:
        duration_us = int(meta.get("duration_us", int(getattr(cfg, "time_window_us", 100000) or 100000)))
        duration_ticks = int(tb.us_to_ticks(duration_us))

    output_type = str(meta.get("output_type", "logit") or "logit").strip().lower()
    if output_type not in ("logit", "prob", "probability", "sigmoid"):
        raise ValueError(f"Unsupported MLPF output_type in metadata: {output_type!r}")
    output_is_prob = output_type in ("prob", "probability", "sigmoid")
    return weights_path, patch, duration_ticks, output_is_prob


def _build_ops(meta: EventStreamMeta, cfg: DenoiseConfig, tb: TimeBase) -> tuple[list[DenoiseOp], GlobalGateOp | None]:
    """Create op instances.

    Returns:
    - ops: list of local filters (STC/Refractory/HotPixel/BAF/RateLimit/DP)
    - global_gate: special global gate op (needs per-batch update)
    """

    dims = Dims(width=int(meta.width), height=int(meta.height))

    # Determine pipeline order.
    # If user provides pipeline, use it.
    # Else map from Qt method ID.
    pipeline_tokens: list[str]
    if cfg.pipeline:
        pipeline_tokens = [_normalize_method_token(x) for x in cfg.pipeline]
    else:
        pipeline_tokens = [_normalize_method_token(str(cfg.method))]

    # Expand combo (Qt method 5)
    expanded: list[str] = []
    for t in pipeline_tokens:
        if t == "combo":
            expanded.extend(["stc", "refractory"])  # Qt requires BOTH
        else:
            expanded.append(t)

    ops: list[DenoiseOp] = []
    global_gate: GlobalGateOp | None = None

    for t in expanded:
        if t in ("none", "off"):
            continue
        if t == "globalgate":
            # Global gate should run FIRST (Qt does that), but we still respect user order.
            global_gate = GlobalGateOp(cfg, tb)
            continue
        if t == "stc":
            ops.append(StcOp(dims, cfg, tb))
        elif t == "refractory":
            ops.append(RefractoryOp(dims, cfg, tb))
        elif t == "hotpixel":
            ops.append(HotPixelOp(dims, cfg, tb))
        elif t == "baf":
            ops.append(BafOp(dims, cfg, tb))
        elif t == "ratelimit":
            ops.append(RateLimitOp(dims, cfg, tb))
        elif t == "dp":
            ops.append(DpOp(dims, cfg, tb))
        elif t == "fastdecay":
            ops.append(FastDecayOp(dims, cfg, tb))
        elif t == "ebf":
            ops.append(EbfOp(dims, cfg, tb))
        elif t == "ebf_optimized":
            ops.append(EbfOptimizedOp(dims, cfg, tb))
        elif t == "knoise":
            ops.append(KnoiseOp(dims, cfg, tb))
        elif t == "evflow":
            ops.append(EvFlowOp(dims, cfg, tb))
        elif t == "ynoise":
            ops.append(YnoiseOp(dims, cfg, tb))
        elif t == "ts":
            ops.append(TsOp(dims, cfg, tb))
        elif t == "mlpf":
            ops.append(MlpfOp(dims, cfg, tb))
        elif t == "pfd":
            ops.append(PfdOp(dims, cfg, tb))
        else:
            raise ValueError(f"Unknown denoise method token: {t!r}")

    return ops, global_gate


def denoise_stream(
    meta: EventStreamMeta,
    batches: Iterator[EventBatch],
    cfg: DenoiseConfig,
    *,
    timebase: TimeBase | None = None,
    engine: str = "python",
) -> Iterator[EventBatch]:
    """Filter events.

    Important variables:
    - `t` is in *ticks* (NOT microseconds)
    - cfg parameters are in microseconds (same as Qt UI)
    - timebase converts microseconds -> ticks
    """

    if meta.width <= 0 or meta.height <= 0:
        raise ValueError("Invalid meta width/height")

    tb = timebase or TimeBase()

    eng = (engine or "python").strip().lower()
    if eng not in ("python", "numba", "cpp"):
        raise ValueError(f"Unknown engine: {engine!r} (expected 'python', 'numba', or 'cpp')")

    if eng == "cpp":
        tokens: list[str]
        if cfg.pipeline:
            tokens = [_normalize_method_token(x) for x in cfg.pipeline]
        else:
            tokens = [_normalize_method_token(str(cfg.method))]

        if len(tokens) != 1:
            raise ValueError("engine='cpp' supports only a single method token without pipeline composition")

        token = tokens[0]
        try:
            from myevs import _native_emlb
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Failed to import myevs._native_emlb. Reinstall with native build support: "
                "python -m pip install -e ."
            ) from e

        w = int(meta.width)
        h = int(meta.height)
        win_ticks = int(tb.us_to_ticks(int(cfg.time_window_us)))
        show_on = bool(cfg.show_on)
        show_off = bool(cfg.show_off)

        if token in ("stc", "stcf"):
            native = _native_emlb.StcNative(
                w, h, int(win_ticks), max(0, int(cfg.radius_px)), max(0, int(cfg.min_neighbors)), show_on, show_off
            )
        elif token in ("stcf_original", "stc_original"):
            native = _native_emlb.StcfOriginalNative(
                w, h, int(win_ticks), max(1, int(cfg.min_neighbors)), show_on, show_off
            )
        elif token == "baf":
            native = _native_emlb.BafNative(
                w, h, int(win_ticks), max(0, int(cfg.radius_px)), show_on, show_off
            )
        elif token == "ebf":
            native = _native_emlb.EbfNative(
                w, h, int(win_ticks), max(0, int(cfg.radius_px)), float(cfg.min_neighbors), show_on, show_off
            )
        elif token == "n149":
            native = _native_emlb.N149Native(
                w, h, int(win_ticks), max(0, int(cfg.radius_px)), float(cfg.min_neighbors), show_on, show_off
            )
        elif token == "knoise":
            native = _native_emlb.KNoiseNative(
                w, h, int(win_ticks), max(0, int(cfg.min_neighbors)), show_on, show_off
            )
        elif token == "ynoise":
            native = _native_emlb.YNoiseNative(
                w,
                h,
                int(win_ticks),
                max(0, int(cfg.radius_px)),
                max(0, int(cfg.min_neighbors)),
                show_on,
                show_off,
            )
        elif token == "ts":
            native = _native_emlb.TimeSurfaceNative(
                w,
                h,
                int(win_ticks),
                max(0, int(cfg.radius_px)),
                float(cfg.min_neighbors),
                show_on,
                show_off,
            )
        elif token == "evflow":
            native = _native_emlb.EventFlowNative(
                w,
                h,
                int(win_ticks),
                max(1, int(cfg.radius_px)),
                float(cfg.min_neighbors),
                show_on,
                show_off,
            )
        elif token == "mlpf":
            weights_path, patch, duration_ticks, output_is_prob = _resolve_mlpf_native_model(
                str(getattr(cfg, "mlpf_model_path", "") or ""), cfg, tb
            )
            weights = np.load(weights_path)
            required = ("fc1_weight", "fc1_bias", "fc2_weight", "fc2_bias")
            missing = [name for name in required if name not in weights]
            if missing:
                raise ValueError(f"Native MLPF weight file is missing arrays: {missing}")
            native = _native_emlb.MlpfNative(
                w,
                h,
                int(duration_ticks),
                float(cfg.min_neighbors),
                int(patch),
                np.ascontiguousarray(weights["fc1_weight"], dtype=np.float32),
                np.ascontiguousarray(weights["fc1_bias"], dtype=np.float32),
                np.ascontiguousarray(weights["fc2_weight"], dtype=np.float32),
                np.ascontiguousarray(weights["fc2_bias"], dtype=np.float32),
                bool(output_is_prob),
                show_on,
                show_off,
            )
        elif token == "pfd":
            pfd_mode_str = str(getattr(cfg, "pfd_mode", "a") or "a").strip().lower()
            native = _native_emlb.PfdNative(
                w,
                h,
                int(win_ticks),
                max(1, int(cfg.radius_px)),
                float(cfg.min_neighbors),
                max(1, int(cfg.refractory_us)),
                pfd_mode_str == "b",
                show_on,
                show_off,
            )
        else:
            raise ValueError("engine='cpp' currently supports stc / baf / ebf / n149 / knoise / ynoise / ts / evflow / mlpf / pfd without pipeline")

        for b in batches:
            if len(b) == 0:
                continue

            t_arr = np.asarray(b.t, dtype=np.uint64)
            x_arr = np.asarray(b.x, dtype=np.int32)
            y_arr = np.asarray(b.y, dtype=np.int32)
            p_arr = np.asarray(b.p, dtype=np.int8)

            keep_u8 = native.accept_batch(t_arr, x_arr, y_arr, p_arr)
            if keep_u8.any():
                keep = np.asarray(keep_u8, dtype=bool)
                yield EventBatch(t=t_arr[keep], x=b.x[keep], y=b.y[keep], p=p_arr[keep])
        return

    # Fast path: numba backend for selected single-op methods.
    if eng == "numba":
        tokens: list[str]
        if cfg.pipeline:
            tokens = [_normalize_method_token(x) for x in cfg.pipeline]
        else:
            tokens = [_normalize_method_token(str(cfg.method))]

        # Keep numba path explicit and predictable: single method only.
        if len(tokens) != 1:
            raise ValueError("engine='numba' supports only a single method token without pipeline composition")

        if tokens[0] == "stc":
            try:
                from .numba_stc import is_numba_available, stc_keep_mask_numba, stc_state_init
            except Exception as e:  # pragma: no cover
                raise RuntimeError(f"Failed to import numba backend: {type(e).__name__}: {e}")

            if not is_numba_available():
                raise RuntimeError("Numba is not available. Install it (conda-forge: numba) or use --engine python.")

            w = int(meta.width)
            h = int(meta.height)
            last_on, last_off = stc_state_init(w, h)

            r = max(0, min(int(cfg.radius_px), 8))
            need = max(0, int(cfg.min_neighbors))
            win_ticks = int(tb.us_to_ticks(int(cfg.time_window_us)))

            for b in batches:
                if len(b) == 0:
                    continue

                t_arr = np.asarray(b.t, dtype=np.uint64)
                x_arr = np.asarray(b.x, dtype=np.int32)
                y_arr = np.asarray(b.y, dtype=np.int32)
                p_arr = np.asarray(b.p, dtype=np.int8)

                keep_u8 = stc_keep_mask_numba(
                    t=t_arr,
                    x=x_arr,
                    y=y_arr,
                    p=p_arr,
                    width=w,
                    height=h,
                    show_on=bool(cfg.show_on),
                    show_off=bool(cfg.show_off),
                    radius_px=r,
                    min_neighbors=need,
                    win_ticks=win_ticks,
                    last_on=last_on,
                    last_off=last_off,
                )

                if keep_u8.any():
                    keep = keep_u8.astype(bool)
                    yield EventBatch(t=t_arr[keep], x=b.x[keep], y=b.y[keep], p=p_arr[keep])
            return

        if tokens[0] == "ts":
            try:
                from .numba_ts import is_numba_available, ts_keep_mask_numba, ts_state_init
            except Exception as e:  # pragma: no cover
                raise RuntimeError(f"Failed to import numba backend: {type(e).__name__}: {e}")

            if not is_numba_available():
                raise RuntimeError("Numba is not available. Install it (conda-forge: numba) or use --engine python.")

            w = int(meta.width)
            h = int(meta.height)
            pos_ts, neg_ts = ts_state_init(w, h)

            r = max(0, min(int(cfg.radius_px), 8))
            decay_ticks = int(tb.us_to_ticks(int(cfg.time_window_us)))
            thr = float(cfg.min_neighbors)

            for b in batches:
                if len(b) == 0:
                    continue

                t_arr = np.asarray(b.t, dtype=np.uint64)
                x_arr = np.asarray(b.x, dtype=np.int32)
                y_arr = np.asarray(b.y, dtype=np.int32)
                p_arr = np.asarray(b.p, dtype=np.int8)

                keep_u8 = ts_keep_mask_numba(
                    t=t_arr,
                    x=x_arr,
                    y=y_arr,
                    p=p_arr,
                    width=w,
                    height=h,
                    show_on=bool(cfg.show_on),
                    show_off=bool(cfg.show_off),
                    radius_px=r,
                    decay_ticks=decay_ticks,
                    threshold=thr,
                    pos_ts=pos_ts,
                    neg_ts=neg_ts,
                )

                if keep_u8.any():
                    keep = keep_u8.astype(bool)
                    yield EventBatch(t=t_arr[keep], x=b.x[keep], y=b.y[keep], p=p_arr[keep])
            return

        if tokens[0] == "evflow":
            try:
                from .numba_evflow import evflow_keep_mask_numba, evflow_state_init, is_numba_available
            except Exception as e:  # pragma: no cover
                raise RuntimeError(f"Failed to import numba backend: {type(e).__name__}: {e}")

            if not is_numba_available():
                raise RuntimeError("Numba is not available. Install it (conda-forge: numba) or use --engine python.")

            w = int(meta.width)
            h = int(meta.height)
            r = max(1, min(int(cfg.radius_px), 8))
            win_ticks = int(tb.us_to_ticks(int(cfg.time_window_us)))
            thr = float(cfg.min_neighbors)

            prev_t, prev_x, prev_y = evflow_state_init()

            for b in batches:
                if len(b) == 0:
                    continue

                t_arr = np.asarray(b.t, dtype=np.uint64)
                x_arr = np.asarray(b.x, dtype=np.int32)
                y_arr = np.asarray(b.y, dtype=np.int32)
                p_arr = np.asarray(b.p, dtype=np.int8)

                # Match python path semantics: invisible/out-of-bounds events are ignored
                # and do not update evflow's temporal queue state.
                valid = (x_arr >= 0) & (x_arr < w) & (y_arr >= 0) & (y_arr < h)
                if not cfg.show_on:
                    valid &= (p_arr <= 0)
                if not cfg.show_off:
                    valid &= (p_arr >= 0)

                idx = np.nonzero(valid)[0]
                if idx.size <= 0:
                    continue

                t_v = np.asarray(t_arr[idx], dtype=np.uint64)
                x_v = np.asarray(x_arr[idx], dtype=np.int32)
                y_v = np.asarray(y_arr[idx], dtype=np.int32)

                keep_v_u8, prev_t, prev_x, prev_y = evflow_keep_mask_numba(
                    t=t_v,
                    x=x_v,
                    y=y_v,
                    radius_px=r,
                    win_ticks=win_ticks,
                    threshold=thr,
                    prev_t=prev_t,
                    prev_x=prev_x,
                    prev_y=prev_y,
                )

                if keep_v_u8.any():
                    keep = np.zeros((t_arr.shape[0],), dtype=bool)
                    keep_idx = idx[keep_v_u8.astype(bool)]
                    keep[keep_idx] = True
                    yield EventBatch(t=t_arr[keep], x=b.x[keep], y=b.y[keep], p=p_arr[keep])
            return

        if tokens[0] == "pfd":
            try:
                from .numba_pfd import is_numba_available, pfd_keep_mask_numba, pfd_state_init
            except Exception as e:  # pragma: no cover
                raise RuntimeError(f"Failed to import numba backend: {type(e).__name__}: {e}")

            if not is_numba_available():
                raise RuntimeError("Numba is not available. Install it (conda-forge: numba) or use --engine python.")

            w = int(meta.width)
            h = int(meta.height)
            last_on, last_off, last_pol, last_evt, flip_buf, flip_head, flip_count = pfd_state_init(w, h, fifo_size=5)

            r = max(1, min(int(cfg.radius_px), 8))
            win_ticks = int(tb.us_to_ticks(int(cfg.time_window_us)))
            neigh_thr = float(cfg.min_neighbors)
            stage1_var = int(cfg.refractory_us)
            mode = str(getattr(cfg, "pfd_mode", "a") or "a").strip().lower()
            mode_b = mode == "b"

            for b in batches:
                if len(b) == 0:
                    continue

                t_arr = np.asarray(b.t, dtype=np.uint64)
                x_arr = np.asarray(b.x, dtype=np.int32)
                y_arr = np.asarray(b.y, dtype=np.int32)
                p_arr = np.asarray(b.p, dtype=np.int8)

                keep_u8 = pfd_keep_mask_numba(
                    t=t_arr,
                    x=x_arr,
                    y=y_arr,
                    p=p_arr,
                    width=w,
                    height=h,
                    show_on=bool(cfg.show_on),
                    show_off=bool(cfg.show_off),
                    radius_px=r,
                    win_ticks=win_ticks,
                    min_neighbors=neigh_thr,
                    stage1_var=stage1_var,
                    mode_b=mode_b,
                    last_on=last_on,
                    last_off=last_off,
                    last_pol=last_pol,
                    last_evt=last_evt,
                    flip_buf=flip_buf,
                    flip_head=flip_head,
                    flip_count=flip_count,
                )

                if keep_u8.any():
                    keep = keep_u8.astype(bool)
                    yield EventBatch(t=t_arr[keep], x=b.x[keep], y=b.y[keep], p=p_arr[keep])
            return

        raise ValueError("engine='numba' currently supports stc / ts / evflow / pfd without pipeline")

    # Build ops once (they keep state across batches)
    ops, global_gate = _build_ops(meta, cfg, tb)

    # Fast path: no filters
    if not ops and global_gate is None:
        yield from batches
        return

    w = int(meta.width)
    h = int(meta.height)

    for b in batches:
        if len(b) == 0:
            continue

        # Convert once per batch (avoid repeated dtype conversions inside loops)
        t_arr = np.asarray(b.t, dtype=np.uint64)
        x_arr = np.asarray(b.x, dtype=np.int32)
        y_arr = np.asarray(b.y, dtype=np.int32)
        p_arr = np.asarray(b.p, dtype=np.int8)  # +1/-1

        keep = np.zeros((t_arr.shape[0],), dtype=bool)

        # ===== Global gate factor (Qt uses previous EMA) =====
        gate_factor = global_gate.compute_gate_factor() if global_gate is not None else 1

        # ===== Per-batch signals for updating global gate (EMA) =====
        # visible_events: after show-on/off AND inside image bounds
        visible_events = 0
        t_first_visible = None
        t_last_visible = None

        for i in range(t_arr.shape[0]):
            x = int(x_arr[i])
            y = int(y_arr[i])
            t = int(t_arr[i])
            p = 1 if int(p_arr[i]) > 0 else -1

            # Bounds check (Qt does per-event)
            if x < 0 or x >= w or y < 0 or y >= h:
                continue

            # Polarity visibility (Qt: showOn/showOff is display-side but also affects denoise stats)
            if p > 0 and not cfg.show_on:
                continue
            if p < 0 and not cfg.show_off:
                continue

            # Count for global-rate EMA (Qt counts BEFORE local denoise and BEFORE gate sampling)
            visible_events += 1
            if t_first_visible is None:
                t_first_visible = t
            t_last_visible = t

            # Global gate sampling (Qt applies it first)
            if global_gate is not None and gate_factor > 1:
                if not keep_by_gate_factor(gate_factor, x, y, t):
                    continue

            # Local ops: keep only if ALL accept
            ok = True
            for op in ops:
                if not op.accept(x, y, p, t):
                    ok = False
                    break
            if ok:
                keep[i] = True

        # Update global gate EMA/state after processing this batch
        if global_gate is not None and visible_events > 0 and t_first_visible is not None and t_last_visible is not None:
            global_gate.update_after_batch(
                visible_events=visible_events,
                t_first=int(t_first_visible),
                t_last=int(t_last_visible),
            )

        if keep.any():
            yield EventBatch(t=t_arr[keep], x=b.x[keep], y=b.y[keep], p=p_arr[keep])
