from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from evuav_common import HEIGHT, RESULT_ROOT, SELECTED_TEST_SEQUENCES, WIDTH, ensure_dirs, parse_sequence, sequence_reference_path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from myevs.events import EventBatch, EventStreamMeta
from myevs.qualitative.events import concat_event_batches, event_stats, save_npz_events
from myevs.qualitative.layout import make_dataset_method_panel, make_panel, save_panel_image
from myevs.qualitative.render import render_events_to_image, write_pdf_from_image, write_png

from run_evuav_alg_sweep import (
    _binary_accept_scores,
    _ebf_scores,
    _llef_scores,
    _stcf_orig_accept_scores,
    _ts_scores,
    BafOp,
    PfdOp,
)


METHOD_ORDER = ("Event frame", "Target ref.", "LLEF", "BAF", "STCF", "EBF", "TS", "PFD")
CASE_PANEL_METHODS = METHOD_ORDER
SUMMARY_PANEL_METHODS = CASE_PANEL_METHODS

RENDER_CFG = {
    "scheme": 0,
    "raw_step": 127,
    "deadzone": 0,
    "binary": True,
    "show_on": True,
    "show_off": True,
}


@dataclass(frozen=True)
class MethodParams:
    method: str
    engine: str
    radius_px: int
    time_us: int
    threshold: float
    refractory_us: int = 1
    pfd_mode: str = "a"
    env: dict[str, str] | None = None


@contextmanager
def temporary_env(values: dict[str, str] | None) -> Iterator[None]:
    if not values:
        yield
        return
    old = {k: os.environ.get(k) for k in values}
    try:
        for k, v in values.items():
            os.environ[k] = str(v)
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _load_reference(seq_raw: str) -> np.ndarray:
    seq = parse_sequence(seq_raw)
    path = sequence_reference_path(seq)
    if not path.exists():
        raise FileNotFoundError(f"missing converted EV-UAV reference stream: {path}")
    return np.load(path, allow_pickle=False)


def _batch_from_struct(arr: np.ndarray) -> EventBatch:
    return EventBatch(
        t=np.asarray(arr["t"], dtype=np.uint64),
        x=np.asarray(arr["x"], dtype=np.uint16),
        y=np.asarray(arr["y"], dtype=np.uint16),
        p=np.asarray(arr["p"], dtype=np.int8),
    )


def _window(arr: np.ndarray, start_us: int, window_us: int) -> np.ndarray:
    t = np.asarray(arr["t"], dtype=np.uint64)
    keep = (t >= np.uint64(start_us)) & (t < np.uint64(start_us + window_us))
    return arr[keep]


def _select_dense_target_window(arr: np.ndarray, *, window_us: int, stride_us: int) -> tuple[int, dict[str, int]]:
    if arr.size == 0:
        return 0, {"target_events": 0, "total_events": 0, "background_events": 0}
    t0 = int(arr["t"][0])
    t1 = int(arr["t"][-1])
    if t1 <= t0 + window_us:
        win = arr
        return t0, {
            "target_events": int(np.count_nonzero(win["target"] == 1)),
            "total_events": int(win.size),
            "background_events": int(np.count_nonzero(win["target"] == 0)),
        }
    best_start = t0
    best_key = (-1, -1)
    for s in range(t0, max(t0 + 1, t1 - window_us + 1), max(1, int(stride_us))):
        win = _window(arr, s, window_us)
        target_n = int(np.count_nonzero(win["target"] == 1))
        total_n = int(win.size)
        key = (target_n, total_n)
        if key > best_key:
            best_key = key
            best_start = int(s)
    win = _window(arr, best_start, window_us)
    return best_start, {
        "target_events": int(np.count_nonzero(win["target"] == 1)),
        "total_events": int(win.size),
        "background_events": int(np.count_nonzero(win["target"] == 0)),
    }


def _load_best_params(summary_csv: Path) -> dict[tuple[str, str], MethodParams]:
    rows: list[dict[str, str]]
    with summary_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    params: dict[tuple[str, str], MethodParams] = {}
    for r in rows:
        seq = str(r.get("sequence", ""))
        if not seq or seq == "MEAN":
            continue
        alg = str(r.get("algorithm", "")).lower()
        tag = str(r.get("tag", ""))
        value = float(r.get("value", 0.0) or 0.0)
        if alg == "baf":
            params[(seq, "BAF")] = MethodParams("baf", "cpp", 1, int(value), 1.0)
        elif alg == "stcf":
            k = 1
            if "k" in tag:
                try:
                    k = int(tag.rsplit("k", 1)[1])
                except Exception:
                    k = 1
            params[(seq, "STCF")] = MethodParams("stcf_original", "cpp", 1, int(value), float(k))
        elif alg == "pfd":
            radius = _parse_tag_int(tag, "r", default=2)
            params[(seq, "PFD")] = MethodParams("pfd", "cpp", radius, int(value), 1.0, refractory_us=1, pfd_mode="a")
        elif alg == "ebf":
            radius = _parse_tag_int(tag, "r", default=4)
            tau = _parse_tag_int(tag, "tau", default=128000)
            params[(seq, "EBF")] = MethodParams("ebf", "cpp", radius, tau, value)
        elif alg == "ts":
            radius = _parse_tag_int(tag, "r", default=2)
            tau = _parse_tag_int(tag, "tau", default=64000)
            params[(seq, "TS")] = MethodParams("ts", "cpp", radius, tau, value)
        elif alg == "llef":
            params[(seq, "LLEF")] = MethodParams(
                "n149",
                "cpp",
                4,
                256000,
                value,
                env={
                    "MYEVS_N149_SIGMA": "3.0",
                    "MYEVS_N149_ALPHA_FIXED": "4.0",
                    "MYEVS_N149_HOT_BITS": "16",
                },
            )
    return params


def _parse_tag_int(tag: str, key: str, *, default: int) -> int:
    parts = str(tag).split("_")
    for part in parts:
        if part.startswith(key):
            raw = part[len(key) :]
            try:
                return int(raw)
            except Exception:
                continue
    return int(default)


def _run_method(batch: EventBatch, params: MethodParams) -> EventBatch:
    t = np.asarray(batch.t, dtype=np.uint64)
    x = np.asarray(batch.x, dtype=np.int32)
    y = np.asarray(batch.y, dtype=np.int32)
    p = np.asarray(batch.p, dtype=np.int8)

    if params.method == "baf":
        keep = _binary_accept_scores(BafOp, t, x, y, p, tau_us=params.time_us, r=params.radius_px, min_neighbors=1.0)
    elif params.method == "stcf_original":
        keep = _stcf_orig_accept_scores(t, x, y, p, tau_us=params.time_us, k=int(round(params.threshold)))
    elif params.method == "pfd":
        keep = _binary_accept_scores(
            PfdOp,
            t,
            x,
            y,
            p,
            tau_us=params.time_us,
            r=params.radius_px,
            min_neighbors=1.0,
            refractory_us=params.refractory_us,
        )
    elif params.method == "ebf":
        scores = _ebf_scores(t, x, y, p, tau_us=params.time_us, r=params.radius_px)
        keep = scores >= float(params.threshold)
    elif params.method == "ts":
        scores = _ts_scores(t, x, y, p, tau_us=params.time_us, r=params.radius_px)
        keep = scores >= float(params.threshold)
    elif params.method == "n149":
        sigma = float((params.env or {}).get("MYEVS_N149_SIGMA", 3.0))
        alpha = float((params.env or {}).get("MYEVS_N149_ALPHA_FIXED", 4.0))
        scores = _llef_scores(t, x, y, p, tau_us=params.time_us, r=params.radius_px, sigma=sigma, alpha=alpha)
        keep = scores >= float(params.threshold)
    else:
        raise ValueError(f"unsupported qualitative method: {params.method}")

    keep = np.asarray(keep, dtype=bool)
    return EventBatch(t=t[keep], x=batch.x[keep], y=batch.y[keep], p=p[keep])


def _render_and_save(name: str, batch: EventBatch, out_dir: Path) -> tuple[Path, dict[str, float | int | str]]:
    img = render_events_to_image(batch, width=WIDTH, height=HEIGHT, **RENDER_CFG)
    path = out_dir / f"{name}.png"
    write_png(path, img)
    stats = event_stats(batch, width=WIDTH, height=HEIGHT)
    return path, {"method": name, "image_path": str(path).replace("\\", "/"), **stats}


def render_sequence(seq: str, params: dict[tuple[str, str], MethodParams], *, out_root: Path, window_ms: int, stride_ms: int) -> dict:
    arr = _load_reference(seq)
    start_us, win_stats = _select_dense_target_window(arr, window_us=window_ms * 1000, stride_us=stride_ms * 1000)
    win = _window(arr, start_us, window_ms * 1000)
    source_batch = _batch_from_struct(win)
    target_batch = _batch_from_struct(win[win["target"] == 1])

    case_id = f"{seq}_w{window_ms}ms_t{start_us}us"
    rendered_dir = out_root / "rendered" / case_id
    events_dir = out_root / "events" / case_id
    rendered_dir.mkdir(parents=True, exist_ok=True)
    events_dir.mkdir(parents=True, exist_ok=True)

    image_paths: list[Path] = []
    rows: list[dict] = []

    p, st = _render_and_save("Event frame", source_batch, rendered_dir)
    image_paths.append(p)
    save_npz_events(events_dir / "Event frame.npz", source_batch)
    rows.append({"case_id": case_id, "sequence": seq, "start_us": start_us, "window_ms": window_ms, **st})

    p, st = _render_and_save("Target ref.", target_batch, rendered_dir)
    image_paths.append(p)
    save_npz_events(events_dir / "Target ref..npz", target_batch)
    rows.append({"case_id": case_id, "sequence": seq, "start_us": start_us, "window_ms": window_ms, **st})

    for method in ("LLEF", "BAF", "STCF", "EBF", "TS", "PFD"):
        if (seq, method) not in params:
            continue
        kept = _run_method(source_batch, params[(seq, method)])
        p, st = _render_and_save(method, kept, rendered_dir)
        image_paths.append(p)
        save_npz_events(events_dir / f"{method}.npz", kept)
        mp = params[(seq, method)]
        rows.append(
            {
                "case_id": case_id,
                "sequence": seq,
                "start_us": start_us,
                "window_ms": window_ms,
                "radius_px": mp.radius_px,
                "time_us": mp.time_us,
                "threshold": mp.threshold,
                **st,
            }
        )

    manifest_path = rendered_dir / "render_manifest.csv"
    _write_rows(manifest_path, rows)

    panel = make_panel(image_paths, labels=[p.stem for p in image_paths], cols=4)
    panel_dir = out_root / "panels"
    panel_png = panel_dir / f"{case_id}_panel.png"
    panel_pdf = panel_dir / f"{case_id}_panel.pdf"
    write_png(panel_png, panel)
    write_pdf_from_image(panel_pdf, panel, dpi=600)

    return {
        "case_id": case_id,
        "sequence": seq,
        "start_us": start_us,
        "window_ms": window_ms,
        "panel_png": str(panel_png).replace("\\", "/"),
        "panel_pdf": str(panel_pdf).replace("\\", "/"),
        "render_manifest": str(manifest_path).replace("\\", "/"),
        **win_stats,
    }


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _append_readme(readme_path: Path, selected: list[dict], out_root: Path, seed: int) -> None:
    lines = [
        "",
        "## 18 EV-UAV 定性对比图",
        "",
        f"本节记录随机种子 `{seed}` 下选择的 3 条 EV-UAV test 序列定性图。脚本自动在每条序列中选择 `target event` 数最多的 100 ms 窗口，并使用第 15 章/第 16 章记录的 best-F1 工作点渲染各算法。",
        "",
        "脚本：",
        "",
        "```text",
        r"D:\hjx_workspace\scientific_reserach\projects\myEVS\scripts\remote_sensing_ev_uav\make_evuav_denoise_qualitative.py",
        "```",
        "",
        "输出目录：",
        "",
        "```text",
        str(out_root).replace("/", "\\"),
        "```",
        "",
        "| 序列 | case_id | start_us | window_ms | total events | target events | background events | panel |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in selected:
        rel_panel = Path(item["panel_png"]).as_posix()
        lines.append(
            f"| {item['sequence']} | `{item['case_id']}` | {item['start_us']} | {item['window_ms']} | "
            f"{item['total_events']} | {item['target_events']} | {item['background_events']} | `{rel_panel}` |"
        )
    lines.extend(
        [
            "",
            "说明：`Event frame` 为原始 EV-UAV reference 事件帧，`Target ref.` 仅渲染目标事件；其余列为各算法在同一窗口、同一 best-F1 参数下的输出事件。该定性图用于观察目标结构是否被保留，以及背景事件是否被抑制。",
            "",
        ]
    )
    text = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
    marker = "## 18 EV-UAV 定性对比图"
    if marker in text:
        text = text[: text.index(marker)].rstrip()
    readme_path.write_text(text.rstrip() + "\n" + "\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Render EV-UAV denoising qualitative panels with existing qualitative style.")
    ap.add_argument("--sequences", nargs="*", default=None, help="Default: sample 3 from selected EV-UAV sequences.")
    ap.add_argument("--seed", type=int, default=20260616)
    ap.add_argument("--count", type=int, default=3)
    ap.add_argument("--window-ms", type=int, default=100)
    ap.add_argument("--stride-ms", type=int, default=20)
    ap.add_argument("--summary-csv", default=str(RESULT_ROOT / "metrics" / "evuav_target_background_summary_best_f1_lowrate6.csv"))
    ap.add_argument("--out-root", default=str(RESULT_ROOT / "qualitative"))
    ap.add_argument("--update-readme", action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    if args.sequences:
        sequences = [parse_sequence(s).stem for s in args.sequences]
    else:
        rng = random.Random(int(args.seed))
        pool = list(SELECTED_TEST_SEQUENCES)
        sequences = sorted(rng.sample(pool, min(int(args.count), len(pool))))

    params = _load_best_params(Path(args.summary_csv))
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    selected: list[dict] = []
    for seq in sequences:
        print(f"[render] {seq}", flush=True)
        selected.append(render_sequence(seq, params, out_root=out_root, window_ms=int(args.window_ms), stride_ms=int(args.stride_ms)))

    manifest = out_root / "evuav_qualitative_manifest.csv"
    _write_rows(manifest, selected)

    rendered_root = out_root / "rendered"
    panel_dir = out_root / "panels"
    summary_png = panel_dir / "evuav_three_sequence_method_panel.png"
    summary_pdf = panel_dir / "evuav_three_sequence_method_panel.pdf"
    make_dataset_method_panel(
        rendered_root=rendered_root,
        case_ids=[x["case_id"] for x in selected],
        method_order=SUMMARY_PANEL_METHODS,
        case_labels=[x["sequence"] for x in selected],
        out_path=summary_png,
        max_tile_w_px=WIDTH,
        max_tile_h_px=HEIGHT,
    )
    make_dataset_method_panel(
        rendered_root=rendered_root,
        case_ids=[x["case_id"] for x in selected],
        method_order=SUMMARY_PANEL_METHODS,
        case_labels=[x["sequence"] for x in selected],
        out_path=summary_pdf,
        max_tile_w_px=WIDTH,
        max_tile_h_px=HEIGHT,
    )

    if args.update_readme:
        _append_readme(RESULT_ROOT / "README.md", selected, out_root, int(args.seed))

    print(f"wrote manifest: {manifest}")
    print(f"wrote summary panel: {summary_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
