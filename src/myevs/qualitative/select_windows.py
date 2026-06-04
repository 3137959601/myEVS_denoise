from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from .config import DEFAULT_QUALITATIVE_DIR, load_cases_config
from .events import concat_event_batches, event_stats, read_window_batches
from .render import render_events_to_image, write_png


def _missing_input_message(case: dict[str, Any], noisy_path: Path) -> str:
    raw = case.get("raw_aedat4")
    if raw:
        return (
            f"missing converted input: {noisy_path}. Convert first: "
            f"D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/qualitative/convert_aedat4.py "
            f"--in {raw} --out {noisy_path}"
        )
    return f"missing noisy input: {noisy_path}"


def _duration_us_for_case(case: dict[str, Any]) -> tuple[int, int]:
    path = Path(str(case["noisy"]))
    assume = case.get("assume")
    from myevs.io.auto import open_events

    result = open_events(
        str(path),
        width=int(case["width"]),
        height=int(case["height"]),
        tick_ns=float(case.get("tick_ns", 1000)),
        assume=str(assume) if assume else None,
        batch_events=1_000_000,
    )
    first: int | None = None
    last: int | None = None
    for b in result.batches:
        if len(b) == 0:
            continue
        t = np.asarray(b.t, dtype=np.uint64)
        if first is None:
            first = int(t[0])
        last = int(t[-1])
    if first is None or last is None:
        return 0, 0
    tick_us = float(case.get("tick_ns", 1000)) / 1000.0
    return int(round(first * tick_us)), int(round(last * tick_us))


def scan_candidates(
    *,
    cases_path: str | Path,
    output_root: str | Path = DEFAULT_QUALITATIVE_DIR,
    case_ids: list[str] | None = None,
    max_candidates_per_window: int = 20,
    stride_ms: int | None = None,
) -> Path:
    cfg = load_cases_config(cases_path)
    out_root = Path(output_root)
    thumbs_root = out_root / "candidates"
    manifest_path = out_root / "candidate_manifest.csv"
    render_cfg = dict(cfg.get("render", {}))

    rows: list[dict[str, Any]] = []
    cases = cfg.get("cases", {})
    for case_id, case in cases.items():
        if case_ids and case_id not in case_ids:
            continue
        if not bool(case.get("enabled", True)):
            continue
        noisy_path = Path(str(case.get("noisy", "")))
        if not noisy_path.exists():
            rows.append(
                {
                    "case_id": case_id,
                    "group": case.get("group", ""),
                    "label": case.get("label", ""),
                    "status": "missing_input",
                    "message": _missing_input_message(case, noisy_path),
                }
            )
            continue

        t_first_us, t_last_us = _duration_us_for_case(case)
        if t_last_us <= t_first_us:
            rows.append({"case_id": case_id, "status": "empty", "message": "no events found"})
            continue

        for window_ms in case.get("scan_windows_ms", [int(case.get("preferred_window_ms", 50))]):
            window_us = int(window_ms) * 1000
            stride_us = int(stride_ms * 1000) if stride_ms else max(10_000, window_us // 2)
            starts = list(range(t_first_us, max(t_first_us + 1, t_last_us - window_us + 1), stride_us))
            scored: list[tuple[float, int, dict[str, Any], Any]] = []
            for start_us in starts:
                batch = concat_event_batches(
                    read_window_batches(
                        noisy_path,
                        width=int(case["width"]),
                        height=int(case["height"]),
                        tick_ns=float(case.get("tick_ns", 1000)),
                        assume=case.get("assume"),
                        start_us=int(start_us),
                        duration_us=int(window_us),
                    )
                )
                st = event_stats(batch, width=int(case["width"]), height=int(case["height"]))
                score = float(st["active_pixel_ratio"]) * 100000.0 + float(st["events"])
                scored.append((score, start_us, st, batch))

            scored.sort(key=lambda x: x[0], reverse=True)
            for rank, (score, start_us, st, batch) in enumerate(scored[: int(max_candidates_per_window)], start=1):
                rel = Path(case_id) / f"{case_id}_w{int(window_ms)}ms_rank{rank:02d}_t{int(start_us)}us.png"
                img_path = thumbs_root / rel
                img = render_events_to_image(
                    batch,
                    width=int(case["width"]),
                    height=int(case["height"]),
                    raw_step=int(render_cfg.get("raw_step", 10)),
                    deadzone=int(render_cfg.get("deadzone", 3)),
                    binary=bool(render_cfg.get("binary", False)),
                    scheme=int(render_cfg.get("scheme", 0)),
                    show_on=bool(render_cfg.get("show_on", True)),
                    show_off=bool(render_cfg.get("show_off", True)),
                )
                write_png(img_path, img)
                rows.append(
                    {
                        "case_id": case_id,
                        "group": case.get("group", ""),
                        "label": case.get("label", ""),
                        "status": "candidate",
                        "rank": rank,
                        "score": f"{score:.6f}",
                        "start_us": int(start_us),
                        "window_ms": int(window_ms),
                        "window_us": int(window_us),
                        "events": st["events"],
                        "on": st["on"],
                        "off": st["off"],
                        "active_pixels": st["active_pixels"],
                        "active_pixel_ratio": f"{float(st['active_pixel_ratio']):.8f}",
                        "thumbnail": str(img_path).replace("\\", "/"),
                        "selected": "no",
                        "message": "",
                    }
                )

    out_root.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "group",
        "label",
        "status",
        "rank",
        "score",
        "start_us",
        "window_ms",
        "window_us",
        "events",
        "on",
        "off",
        "active_pixels",
        "active_pixel_ratio",
        "thumbnail",
        "selected",
        "message",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return manifest_path
