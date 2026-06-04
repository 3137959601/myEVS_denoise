from __future__ import annotations

import csv
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from myevs.denoise import DenoiseConfig, denoise_stream
from myevs.events import EventStreamMeta
from myevs.timebase import TimeBase

from .config import DEFAULT_QUALITATIVE_DIR, load_algorithm_params, load_cases_config
from .events import concat_event_batches, event_stats, read_window_batches, save_npz_events
from .layout import make_panel
from .render import render_events_to_image, write_pdf_from_image, write_png


@contextmanager
def temporary_env(values: dict[str, Any] | None) -> Iterator[None]:
    if not values:
        yield
        return
    old: dict[str, str | None] = {}
    for key, value in values.items():
        old[str(key)] = os.environ.get(str(key))
        os.environ[str(key)] = str(value)
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _selected_window(case: dict[str, Any], *, start_us: int | None, window_ms: int | None) -> tuple[int, int]:
    selected = case.get("selected_window") if isinstance(case.get("selected_window"), dict) else {}
    s = int(start_us if start_us is not None else selected.get("start_us", 0))
    w_ms = int(window_ms if window_ms is not None else selected.get("window_ms", case.get("preferred_window_ms", 50)))
    return s, w_ms * 1000


def _methods_for_case(params: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    group = str(case.get("group", "driving"))
    groups = params.get("groups", {})
    if group not in groups:
        raise KeyError(f"No algorithm parameter group for case group {group!r}")
    methods = groups[group].get("methods", {})
    if not isinstance(methods, dict):
        raise ValueError(f"Invalid methods config for group {group!r}")
    order = [m for m in params.get("method_order", []) if m in methods]
    return {name: methods[name] for name in order}


def render_case(
    *,
    cases_path: str | Path,
    params_path: str | Path,
    case_id: str,
    output_root: str | Path = DEFAULT_QUALITATIVE_DIR,
    start_us: int | None = None,
    window_ms: int | None = None,
    make_panel_output: bool = True,
) -> Path:
    cases_cfg = load_cases_config(cases_path)
    params_cfg = load_algorithm_params(params_path)
    case = cases_cfg["cases"][case_id]
    render_cfg = dict(cases_cfg.get("render", {}))
    out_root = Path(output_root)
    noisy_path = Path(str(case.get("noisy", "")))
    if not noisy_path.exists():
        raw = case.get("raw_aedat4")
        if raw:
            raise FileNotFoundError(
                f"Converted input is missing for {case_id}: {noisy_path}. "
                f"Convert first with: python scripts/qualitative/convert_aedat4.py --in {raw} --out {noisy_path}"
            )
        raise FileNotFoundError(f"Noisy input is missing for {case_id}: {noisy_path}")
    case_dir = out_root / "rendered" / case_id
    event_dir = out_root / "events" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    event_dir.mkdir(parents=True, exist_ok=True)

    start_us_i, duration_us = _selected_window(case, start_us=start_us, window_ms=window_ms)
    width = int(case["width"])
    height = int(case["height"])
    tick_ns = float(case.get("tick_ns", 1000))
    assume = case.get("assume")
    meta = EventStreamMeta(width=width, height=height)
    tb = TimeBase(tick_ns=tick_ns)

    source_batch = concat_event_batches(
        read_window_batches(
            case["noisy"],
            width=width,
            height=height,
            tick_ns=tick_ns,
            assume=assume,
            start_us=start_us_i,
            duration_us=duration_us,
        )
    )

    rows: list[dict[str, Any]] = []
    image_paths: list[Path] = []
    methods = _methods_for_case(params_cfg, case)
    for display_name, method_cfg in methods.items():
        kind = str(method_cfg.get("kind", "method"))
        if kind == "input":
            batch = source_batch
        else:
            cfg = DenoiseConfig(
                method=str(method_cfg["method"]),
                time_window_us=int(method_cfg.get("time_us", 2000)),
                radius_px=int(method_cfg.get("radius_px", 1)),
                min_neighbors=float(method_cfg.get("min_neighbors", 1)),
                refractory_us=int(method_cfg.get("refractory_us", 1)),
                pfd_mode=str(method_cfg.get("pfd_mode", "a")),
            )
            with temporary_env(method_cfg.get("env")):
                batch = concat_event_batches(
                    denoise_stream(meta, iter([source_batch]), cfg, timebase=tb, engine=str(method_cfg.get("engine", "cpp")))
                )

        event_path = event_dir / f"{display_name}.npz"
        save_npz_events(event_path, batch)
        img = render_events_to_image(
            batch,
            width=width,
            height=height,
            raw_step=int(render_cfg.get("raw_step", 10)),
            deadzone=int(render_cfg.get("deadzone", 3)),
            binary=bool(render_cfg.get("binary", False)),
            scheme=int(render_cfg.get("scheme", 0)),
            show_on=bool(render_cfg.get("show_on", True)),
            show_off=bool(render_cfg.get("show_off", True)),
        )
        img_path = case_dir / f"{display_name}.png"
        write_png(img_path, img)
        image_paths.append(img_path)
        st = event_stats(batch, width=width, height=height)
        rows.append(
            {
                "case_id": case_id,
                "method": display_name,
                "start_us": start_us_i,
                "window_us": duration_us,
                "events_path": str(event_path).replace("\\", "/"),
                "image_path": str(img_path).replace("\\", "/"),
                **st,
            }
        )

    summary_path = case_dir / "render_manifest.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "case_id",
            "method",
            "start_us",
            "window_us",
            "events_path",
            "image_path",
            "events",
            "on",
            "off",
            "active_pixels",
            "active_pixel_ratio",
            "t_first",
            "t_last",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    if make_panel_output and image_paths:
        panel_dir = out_root / "panels"
        panel_png = panel_dir / f"{case_id}_panel.png"
        panel_pdf = panel_dir / f"{case_id}_panel.pdf"
        labels = [p.stem for p in image_paths]
        panel = make_panel(image_paths, labels=labels, cols=min(4, len(image_paths)))
        write_png(panel_png, panel)
        write_pdf_from_image(panel_pdf, panel, dpi=600)
    return summary_path
