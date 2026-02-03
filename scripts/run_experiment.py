from __future__ import annotations

"""Experiment runner (convert -> denoise -> compare -> optional video/plots).

Why a script?
- Research workflow: you often want to batch-run multiple denoise settings and
  compare results without manually typing many CLI commands.
- This script is config-driven (TOML) so you can:
  - enable/disable steps
  - test multiple methods/pipelines
  - compare "stacked pipeline" vs "sequential denoise" easily

Run:
  python scripts/run_experiment.py --config experiments/example_experiment.toml

Notes:
- Prefer running inside your conda env (myevs) where numpy/opencv/matplotlib are installed.
- This script imports the myevs package. Ensure editable install:
    pip install -e .
"""

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

try:
    import tomllib  # py3.11+
except Exception:  # pragma: no cover
    tomllib = None  # type: ignore

from myevs.denoise import DenoiseConfig, denoise_stream
from myevs.io.auto import open_events
from myevs.io.evtq import write_evtq
from myevs.stats import compute_stats
from myevs.timebase import TimeBase


@dataclass(frozen=True)
class Ctx:
    root: Path
    out_dir: Path
    tb: TimeBase


def _p(root: Path, s: str) -> Path:
    # allow both \ and /
    s2 = str(s).strip().replace("/", os.sep).replace("\\", os.sep)
    p = Path(s2)
    return p if p.is_absolute() else (root / p)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_toml(path: Path) -> dict[str, Any]:
    if tomllib is None:
        raise RuntimeError("Python < 3.11: tomllib not available. Use Python 3.11+ or install tomli.")
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Invalid TOML")
    return data


def _bool(d: dict[str, Any], key: str, default: bool) -> bool:
    v = d.get(key, default)
    return bool(v)


def _maybe_plot_csv(in_csv: Path, out_img: Path, *, x: str, y: Iterable[str], kind: str, title: str | None) -> None:
    try:
        from myevs.metrics.plot_csv import plot_csv_quick

        _ensure_parent(out_img)
        plot_csv_quick(str(in_csv), str(out_img), x=x, y=list(y), kind=kind, title=title)
        print(f"plot: {out_img}")
    except Exception as e:
        print(f"plot skipped ({type(e).__name__}: {e})")


def _write_stats_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    _ensure_parent(path)
    if not rows:
        return
    keys: list[str] = []
    # stable union of keys
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _wrap_progress(batches: Iterator, *, enabled: bool, desc: str):
    if not enabled:
        return batches
    try:
        from tqdm import tqdm  # type: ignore
    except Exception as e:
        raise RuntimeError(f"progress requires tqdm ({type(e).__name__}: {e})")

    pbar = tqdm(total=None, unit="ev", desc=desc, dynamic_ncols=True)

    def _gen():
        try:
            for b in batches:
                yield b
                try:
                    pbar.update(len(b))
                except Exception:
                    pbar.update(1)
        finally:
            pbar.close()

    return _gen()


def _as_list(v: Any) -> list[Any]:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    return [v]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="TOML config path")
    ap.add_argument("--force", action="store_true", help="overwrite outputs if they exist")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    cfg_path = _p(root, args.config)
    cfg = _read_toml(cfg_path)

    out_dir = _p(root, cfg.get("output", {}).get("dir", "runs/run1"))
    out_dir.mkdir(parents=True, exist_ok=True)

    tb = TimeBase(tick_ns=float(cfg.get("timebase", {}).get("tick_ns", 12.5)))
    ctx = Ctx(root=root, out_dir=out_dir, tb=tb)

    input_cfg = cfg.get("input", {})
    in_path = _p(root, input_cfg.get("path"))
    assume = input_cfg.get("assume", None)
    width = input_cfg.get("width", None)
    height = input_cfg.get("height", None)
    batch_events = int(input_cfg.get("batch_events", 1_000_000))

    progress_enabled = bool(cfg.get("progress", {}).get("enabled", False))

    # Step 1: optional convert (raw/csv -> evtq)
    convert_cfg = cfg.get("convert", {})
    convert_enabled = _bool(convert_cfg, "enabled", True)

    base_evtq = None
    if convert_enabled:
        out_evtq = _p(out_dir, convert_cfg.get("out_evtq", "input.evtq"))
        if out_evtq.exists() and not args.force:
            print(f"convert: skip (exists) {out_evtq}")
            base_evtq = out_evtq
        else:
            print(f"convert: {in_path} -> {out_evtq}")
            r = open_events(
                str(in_path),
                width=width,
                height=height,
                batch_events=batch_events,
                assume=assume,
            )
            write_evtq(
                str(out_evtq),
                r.meta,
                _wrap_progress(r.batches, enabled=progress_enabled, desc=f"convert: {out_evtq.name}"),
            )
            base_evtq = out_evtq
    else:
        base_evtq = in_path

    assert base_evtq is not None

    # Build list of named streams (path)
    streams: dict[str, Path] = {"input": Path(base_evtq)}

    # Step 2: denoise variants
    denoise_list = cfg.get("denoise", [])
    if denoise_list and not isinstance(denoise_list, list):
        raise ValueError("[denoise] must be an array of tables ([[denoise]])")

    # Convenience: generate variants to run all methods without writing many [[denoise]] blocks.
    # This does NOT remove your explicit [[denoise]] entries; it appends generated ones.
    denoise_all = cfg.get("denoise_all", {})
    if isinstance(denoise_all, dict) and bool(denoise_all.get("enabled", False)):
        methods = [str(x) for x in _as_list(denoise_all.get("methods", [])) if str(x).strip()]
        overrides = denoise_all.get("overrides", {})
        if overrides is None:
            overrides = {}
        if overrides and not isinstance(overrides, dict):
            raise ValueError("[denoise_all.overrides] must be a table")

        name_prefix = str(denoise_all.get("name_prefix", "m")).strip() or "m"
        out_prefix = str(denoise_all.get("out_prefix", ""))

        # Keep generated configs minimal: only include keys you explicitly set.
        # Algorithms ignore unused fields anyway, but keeping TOML small makes it
        # clearer which thresholds matter for each method.
        base_params: dict[str, Any] = {}
        if "show_on" in denoise_all:
            base_params["show_on"] = bool(denoise_all.get("show_on"))
        if "show_off" in denoise_all:
            base_params["show_off"] = bool(denoise_all.get("show_off"))

        existing_names = {str(d.get("name")) for d in denoise_list if isinstance(d, dict) and d.get("name")}
        for m in methods:
            if m in ("0", "none"):
                continue
            per = dict(base_params)
            ov = overrides.get(str(m), {}) if isinstance(overrides, dict) else {}
            if ov is not None:
                if not isinstance(ov, dict):
                    raise ValueError(f"denoise_all.overrides[{m!r}] must be a table")
                per.update(ov)

            name = f"{name_prefix}{m}"
            if name in existing_names:
                # avoid collisions
                i = 2
                while f"{name}_{i}" in existing_names:
                    i += 1
                name = f"{name}_{i}"
            existing_names.add(name)

            out_evtq = f"{out_prefix}{name}.evtq"
            denoise_list.append(
                {
                    "name": name,
                    "method": str(m),
                    "out_evtq": out_evtq,
                    **per,
                }
            )

    for d in denoise_list:
        if not isinstance(d, dict):
            continue
        name = str(d.get("name", "unnamed")).strip() or "unnamed"
        enabled = bool(d.get("enabled", True))
        if not enabled:
            continue

        in_ref = d.get("in", None)
        in_path2 = streams.get(str(in_ref), None) if in_ref else Path(base_evtq)
        if in_path2 is None:
            raise ValueError(f"denoise[{name}]: unknown in ref: {in_ref!r}")

        out_evtq = _p(out_dir, d.get("out_evtq", f"{name}.evtq"))
        if out_evtq.exists() and not args.force:
            print(f"denoise: skip (exists) {name} -> {out_evtq}")
            streams[name] = out_evtq
            continue

        method = d.get("method", "none")
        pipeline = d.get("pipeline", None)
        if pipeline is not None and not isinstance(pipeline, list):
            raise ValueError(f"denoise[{name}]: pipeline must be a list")

        cfg_dn = DenoiseConfig(
            method=str(method),
            pipeline=[str(x) for x in pipeline] if pipeline is not None else None,
            time_window_us=int(d.get("time_us", 2000)),
            radius_px=int(d.get("radius_px", 1)),
            min_neighbors=int(d.get("min_neighbors", 2)),
            refractory_us=int(d.get("refractory_us", 50)),
            show_on=bool(d.get("show_on", True)),
            show_off=bool(d.get("show_off", True)),
        )

        engine = str(d.get("engine", "python"))

        print(f"denoise: {name} ({in_path2.name}) -> {out_evtq.name}  method={cfg_dn.method} pipeline={cfg_dn.pipeline}")
        r = open_events(
            str(in_path2),
            width=width,
            height=height,
            batch_events=batch_events,
            assume=assume,
        )
        den = denoise_stream(
            r.meta,
            _wrap_progress(r.batches, enabled=progress_enabled, desc=f"denoise in: {name}"),
            cfg_dn,
            timebase=ctx.tb,
            engine=engine,
        )
        write_evtq(str(out_evtq), r.meta, _wrap_progress(den, enabled=progress_enabled, desc=f"denoise out: {name}"))
        streams[name] = out_evtq

    # Step 3: stats + comparisons
    stats_rows: list[dict[str, Any]] = []

    def stats_one(tag: str, p: Path) -> dict[str, Any]:
        r = open_events(
            str(p),
            width=width,
            height=height,
            batch_events=batch_events,
            assume=assume,
        )
        st = compute_stats(r.meta, _wrap_progress(r.batches, enabled=progress_enabled, desc=f"stats: {tag}"))
        return {
            "tag": tag,
            "path": str(p),
            "width": int(r.meta.width),
            "height": int(r.meta.height),
            "events": int(st.total),
            "on": int(st.on),
            "off": int(st.off),
            "t_first": st.t_first,
            "t_last": st.t_last,
            "duration_ticks": st.duration_ticks,
        }

    for tag, p in streams.items():
        try:
            row = stats_one(tag, p)
            stats_rows.append(row)
            print(f"stats: {tag:<16} events={row['events']}")
        except Exception as e:
            print(f"stats: {tag} failed ({type(e).__name__}: {e})")

    stats_csv = _p(out_dir, cfg.get("stats", {}).get("out_csv", "stats_all.csv"))
    _write_stats_csv(stats_csv, stats_rows)
    print(f"stats csv: {stats_csv}")

    # Pairwise comparisons requested
    comp_cfg = cfg.get("compare", [])
    if comp_cfg and not isinstance(comp_cfg, list):
        raise ValueError("[compare] must be an array of tables ([[compare]])")

    comp_rows: list[dict[str, Any]] = []

    def _find_events(tag: str) -> int | None:
        for r in stats_rows:
            if r.get("tag") == tag:
                return int(r.get("events") or 0)
        return None

    for c in comp_cfg:
        if not isinstance(c, dict):
            continue
        a = str(c.get("a", "input"))
        b = str(c.get("b"))
        if b not in streams:
            raise ValueError(f"compare: unknown b ref: {b}")
        if a not in streams:
            raise ValueError(f"compare: unknown a ref: {a}")

        ea = _find_events(a)
        eb = _find_events(b)
        if ea is None or eb is None or ea <= 0:
            kept = 0.0
            removed = 0.0
        else:
            kept = float(eb) / float(ea)
            removed = 1.0 - kept

        comp_rows.append(
            {
                "a": a,
                "b": b,
                "events_a": ea,
                "events_b": eb,
                "kept_ratio": kept,
                "removed_ratio": removed,
            }
        )
        print(f"compare: {a} -> {b} kept={kept:.6f} removed={removed:.6f}")

    comp_csv = _p(out_dir, cfg.get("compare_out", {}).get("out_csv", "compare.csv"))
    _write_stats_csv(comp_csv, comp_rows)
    if comp_rows:
        print(f"compare csv: {comp_csv}")

    # Optional: plot tables
    plot_cfg = cfg.get("plots", {})
    if isinstance(plot_cfg, dict) and _bool(plot_cfg, "enabled", False):
        # plot compare.csv (bar by default)
        _maybe_plot_csv(
            comp_csv,
            _p(out_dir, plot_cfg.get("compare_png", "compare.png")),
            x=str(plot_cfg.get("x", "b")),
            y=["kept_ratio", "removed_ratio"],
            kind=str(plot_cfg.get("kind", "bar")),
            title=str(plot_cfg.get("title", "Compare")),
        )

    # Optional: video exports (via myevs view implementation)
    videos = cfg.get("video", [])
    if videos and not isinstance(videos, list):
        raise ValueError("[video] must be an array of tables ([[video]])")

    if videos:
        from myevs.viz import view_stream

        for v in videos:
            if not isinstance(v, dict) or not bool(v.get("enabled", True)):
                continue
            src = str(v.get("in", "input"))
            if src not in streams:
                raise ValueError(f"video: unknown in ref: {src}")
            out_video = _p(out_dir, v.get("out", f"{src}.mp4"))
            if out_video.exists() and not args.force:
                print(f"video: skip (exists) {out_video}")
                continue

            r = open_events(
                str(streams[src]),
                width=width,
                height=height,
                batch_events=batch_events,
                assume=assume,
            )
            print(f"video: {src} -> {out_video}")
            view_stream(
                r.meta,
                _wrap_progress(r.batches, enabled=progress_enabled, desc=f"video: {src}"),
                mode=str(v.get("mode", "fps")),
                fps=float(v.get("fps", 60.0)),
                events_per_frame=int(v.get("events_per_frame", 20000)),
                tick_us=ctx.tb.tick_us,
                color=str(v.get("color", "onoff")),
                scheme_id=int(v.get("scheme", 1)),
                raw_step=int(v.get("raw_step", 5)),
                deadzone=int(v.get("deadzone", 3)),
                binary=bool(v.get("binary", False)),
                hold=(not bool(v.get("no_hold", True))),
                show_on=bool(v.get("show_on", True)),
                show_off=bool(v.get("show_off", True)),
                realtime=False,
                out_video=str(out_video),
                video_fps=float(v.get("video_fps", v.get("fps", 60.0))),
                no_gui=True,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
