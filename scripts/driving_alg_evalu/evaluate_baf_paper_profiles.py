from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np


WIDTH = 346
HEIGHT = 260


@dataclass(frozen=True)
class BafProfile:
    name: str
    description: str
    tau_ms: tuple[int, ...]


PROFILES: tuple[BafProfile, ...] = (
    BafProfile(
        name="baf_dnd21_original",
        description="DND21-style BAF: radius=1, polarity ignored, dense tau 1..100 ms",
        tau_ms=tuple(range(1, 101)),
    ),
    BafProfile(
        name="baf_edformer",
        description="EDformer Table 2 BAF: radius=1, polarity ignored, tau 2..200 ms",
        tau_ms=tuple(range(2, 201)),
    ),
    BafProfile(
        name="baf_ebf_source",
        description="EBF source/paper BAF: radius=1, polarity ignored, tau grid 1,5,10,15,20,25,30,40,50 ms",
        tau_ms=(1, 5, 10, 15, 20, 25, 30, 40, 50),
    ),
)


PAPER_TARGETS: dict[str, dict[float, tuple[str, float]]] = {
    "baf_dnd21_original": {
        5.0: ("DND21 Table 2 driving BAF", 0.79),
    },
    "baf_edformer": {
        1.0: ("EDformer Table 2 driving BAF", 0.8479),
        3.0: ("EDformer Table 2 driving BAF", 0.8155),
        5.0: ("EDformer Table 2 driving BAF", 0.7930),
        7.0: ("EDformer Table 2 driving BAF", 0.7732),
        10.0: ("EDformer Table 2 driving BAF", 0.7479),
    },
    "baf_ebf_source": {
        1.0: ("EBF Table II driving BAF", 0.848),
        3.0: ("EBF Table II driving BAF", 0.816),
        5.0: ("EBF Table II driving BAF", 0.793),
    },
}


@dataclass(frozen=True)
class EventSource:
    dataset_id: str
    dataset_label: str
    level: str
    nominal_hz: float
    path: Path
    kind: str
    label_convention: str
    metadata_path: Path | None = None
    v2e_args_path: Path | None = None


@dataclass(frozen=True)
class BafHistResult:
    events_total: int
    signal_total: int
    noise_total: int
    first_t_us: int
    last_t_us: int
    out_of_bounds: int
    nonmonotonic_t: int
    signal_hist: np.ndarray
    noise_hist: np.ndarray

    @property
    def duration_s(self) -> float:
        if self.last_t_us <= self.first_t_us:
            return 0.0
        return float(self.last_t_us - self.first_t_us) / 1_000_000.0

    @property
    def actual_noise_hz_per_pixel(self) -> float:
        dur = self.duration_s
        if dur <= 0.0:
            return 0.0
        return float(self.noise_total) / float(WIDTH * HEIGHT) / dur


def _auc_trapezoid(fpr: np.ndarray, tpr: np.ndarray) -> float:
    if fpr.size == 0 or tpr.size == 0 or fpr.size != tpr.size:
        return 0.0
    order = np.argsort(fpr)
    x = np.asarray(fpr, dtype=np.float64)[order]
    y = np.asarray(tpr, dtype=np.float64)[order]
    x = np.clip(x, 0.0, 1.0)
    y = np.clip(y, 0.0, 1.0)
    if x[0] > 0.0:
        x = np.concatenate((np.array([0.0]), x))
        y = np.concatenate((np.array([0.0]), y))
    if x[-1] < 1.0:
        x = np.concatenate((x, np.array([1.0])))
        y = np.concatenate((y, np.array([1.0])))
    return float(np.trapezoid(y=y, x=x))


def _parse_v2e_args(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or ":" not in s:
                continue
            k, v = s.split(":", 1)
            key = k.strip()
            value = v.strip()
            if key and key not in out:
                out[key] = value
    return out


def _read_json(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _discover_ed24(root: Path) -> tuple[list[EventSource], list[dict[str, str]]]:
    ds_root = root / "mydriving_ED24"
    wanted = (1, 3, 5, 7, 10)
    found: list[EventSource] = []
    missing: list[dict[str, str]] = []
    for hz in wanted:
        level = f"{hz}hz"
        d = ds_root / f"driving_noise_{level}_ed24_withlabel"
        p = d / f"driving_noise_{level}_labeled.npy"
        meta = d / "ed24_driving_meta.json"
        if p.exists():
            found.append(
                EventSource(
                    dataset_id="mydriving_ED24",
                    dataset_label="EDformer release Driving",
                    level=level,
                    nominal_hz=float(hz),
                    path=p,
                    kind="npy_us_label1_signal",
                    label_convention="myEVS npy: 1=signal, 0=noise",
                    metadata_path=meta if meta.exists() else None,
                    v2e_args_path=ds_root / level / "v2e-args.txt",
                )
            )
        else:
            missing.append(
                {
                    "dataset_id": "mydriving_ED24",
                    "level": level,
                    "reason": f"missing converted npy: {p}",
                }
            )
    return found, missing


def _discover_cov05(root: Path) -> tuple[list[EventSource], list[dict[str, str]]]:
    ds_root = root / "mydriving_cov05"
    wanted = (1, 3, 5, 7, 10)
    found: list[EventSource] = []
    missing: list[dict[str, str]] = []
    for hz in wanted:
        level = f"{hz}hz"
        d = ds_root / level
        p = d / f"driving_cov05_{level}_labeled.txt"
        if p.exists():
            found.append(
                EventSource(
                    dataset_id="mydriving_cov05",
                    dataset_label="v2e Driving COV=0.5 labeled",
                    level=level,
                    nominal_hz=float(hz),
                    path=p,
                    kind="v2e_txt_seconds_label1_signal",
                    label_convention="v2e txt: 1=signal, 0=noise",
                    v2e_args_path=d / "v2e-args.txt",
                )
            )
        else:
            missing.append(
                {
                    "dataset_id": "mydriving_cov05",
                    "level": level,
                    "reason": f"missing labeled txt: {p}",
                }
            )
    return found, missing


def _discover_paper(root: Path) -> tuple[list[EventSource], list[dict[str, str]]]:
    ds_root = root / "mydriving_paper"
    levels = (
        ("light", 1.0),
        ("light_mid", 3.0),
        ("mid", 5.0),
    )
    found: list[EventSource] = []
    missing: list[dict[str, str]] = []
    for name, hz in levels:
        d = ds_root / f"driving_noise_{name}_paper_withlabel"
        p = d / f"driving_noise_{name}_labeled.npy"
        meta = d / "paper_noise_meta.json"
        if p.exists():
            found.append(
                EventSource(
                    dataset_id="mydriving_paper",
                    dataset_label="DND21 clean + Python FPN shot noise",
                    level=name,
                    nominal_hz=hz,
                    path=p,
                    kind="npy_us_label1_signal",
                    label_convention="myEVS npy: 1=signal, 0=noise",
                    metadata_path=meta if meta.exists() else None,
                )
            )
        else:
            missing.append(
                {
                    "dataset_id": "mydriving_paper",
                    "level": name,
                    "reason": f"missing labeled npy: {p}",
                }
            )
    return found, missing


def _discover_jaer(root: Path) -> tuple[list[EventSource], list[dict[str, str]]]:
    ds_root = root / "mydriving_jaer"
    levels = (
        ("light", 1.0),
        ("light_mid", 3.0),
        ("mid", 5.0),
    )
    found: list[EventSource] = []
    missing: list[dict[str, str]] = []
    for name, hz in levels:
        d = ds_root / f"driving_noise_{name}_jaer_withlabel"
        p = d / f"driving_noise_{name}_labeled.npy"
        meta = d / "jaer_noise_meta.json"
        if p.exists():
            found.append(
                EventSource(
                    dataset_id="mydriving_jaer",
                    dataset_label="DND21 clean + jAER shot noise",
                    level=name,
                    nominal_hz=hz,
                    path=p,
                    kind="npy_us_label1_signal",
                    label_convention="myEVS npy: 1=signal, 0=noise",
                    metadata_path=meta if meta.exists() else None,
                )
            )
        else:
            missing.append(
                {
                    "dataset_id": "mydriving_jaer",
                    "level": name,
                    "reason": f"missing labeled npy: {p}",
                }
            )
    return found, missing


def discover_sources(root: Path, dataset_ids: set[str]) -> tuple[list[EventSource], list[dict[str, str]]]:
    sources: list[EventSource] = []
    missing: list[dict[str, str]] = []
    discoverers = {
        "mydriving_ED24": _discover_ed24,
        "mydriving_cov05": _discover_cov05,
        "mydriving_paper": _discover_paper,
        "mydriving_jaer": _discover_jaer,
    }
    for dataset_id, fn in discoverers.items():
        if dataset_id not in dataset_ids:
            continue
        found, miss = fn(root)
        sources.extend(found)
        missing.extend(miss)
    sources.sort(key=lambda s: (s.dataset_id, s.nominal_hz, s.level))
    return sources, missing


def _source_tick_ns(source: EventSource) -> float:
    if not source.kind.startswith("npy_"):
        return 1000.0
    meta = _read_json(source.metadata_path)
    if "tick_ns" in meta:
        try:
            return float(meta["tick_ns"])
        except Exception:
            return 1000.0
    if str(meta.get("timestamp_unit", "")).strip().lower() in {"microsecond", "microseconds", "us"}:
        return 1000.0
    return 1000.0


def _iter_npy_events(path: Path, *, tick_ns: float = 1000.0) -> Iterable[tuple[int, int, int, int]]:
    arr = np.load(path, mmap_mode="r")
    names = arr.dtype.names or ()
    required = {"t", "x", "y", "label"}
    missing = required.difference(names)
    if missing:
        raise ValueError(f"{path} is missing required fields: {sorted(missing)}")
    tick_to_us = float(tick_ns) / 1000.0
    t = arr["t"]
    x = arr["x"]
    y = arr["y"]
    label = arr["label"]
    for i in range(arr.shape[0]):
        yield int(round(float(t[i]) * tick_to_us)), int(x[i]), int(y[i]), int(label[i])


def _iter_v2e_txt_events(path: Path) -> Iterable[tuple[int, int, int, int]]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line or line[0] == "#":
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            t_us = int(round(float(parts[0]) * 1_000_000.0))
            yield t_us, int(parts[1]), int(parts[2]), int(parts[4])


def _iter_source_events(source: EventSource) -> Iterable[tuple[int, int, int, int]]:
    if source.kind.startswith("npy_"):
        return _iter_npy_events(source.path, tick_ns=_source_tick_ns(source))
    if source.kind == "v2e_txt_seconds_label1_signal":
        return _iter_v2e_txt_events(source.path)
    raise ValueError(f"unsupported source kind: {source.kind}")


def compute_baf_hist(source: EventSource, max_tau_us: int) -> BafHistResult:
    last = np.zeros((HEIGHT, WIDTH), dtype=np.int64)
    signal_hist = np.zeros((max_tau_us + 1,), dtype=np.int64)
    noise_hist = np.zeros((max_tau_us + 1,), dtype=np.int64)

    events_total = 0
    signal_total = 0
    noise_total = 0
    out_of_bounds = 0
    nonmonotonic_t = 0
    first_t_us: int | None = None
    last_t_us: int | None = None

    for t_us, x, y, label in _iter_source_events(source):
        if first_t_us is None:
            first_t_us = t_us
        if last_t_us is not None and t_us < last_t_us:
            nonmonotonic_t += 1
        last_t_us = t_us

        if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
            out_of_bounds += 1
            continue

        events_total += 1
        is_signal = int(label) == 1
        if is_signal:
            signal_total += 1
        else:
            noise_total += 1

        max_neighbor_ts = 0
        y0 = max(0, y - 1)
        y1 = min(HEIGHT - 1, y + 1)
        x0 = max(0, x - 1)
        x1 = min(WIDTH - 1, x + 1)
        for yy in range(y0, y1 + 1):
            row = last[yy]
            for xx in range(x0, x1 + 1):
                if xx == x and yy == y:
                    continue
                ts = int(row[xx])
                if ts > max_neighbor_ts:
                    max_neighbor_ts = ts

        if max_neighbor_ts > 0:
            dt = int(t_us - max_neighbor_ts)
            if 0 <= dt <= max_tau_us:
                if is_signal:
                    signal_hist[dt] += 1
                else:
                    noise_hist[dt] += 1

        last[y, x] = int(t_us)

    return BafHistResult(
        events_total=events_total,
        signal_total=signal_total,
        noise_total=noise_total,
        first_t_us=int(first_t_us or 0),
        last_t_us=int(last_t_us or 0),
        out_of_bounds=out_of_bounds,
        nonmonotonic_t=nonmonotonic_t,
        signal_hist=signal_hist,
        noise_hist=noise_hist,
    )


def profile_points(result: BafHistResult, profile: BafProfile) -> tuple[list[dict[str, object]], float]:
    sig_cum = np.cumsum(result.signal_hist)
    noise_cum = np.cumsum(result.noise_hist)
    rows: list[dict[str, object]] = []
    fpr_values: list[float] = []
    tpr_values: list[float] = []

    for tau_ms in profile.tau_ms:
        tau_us = int(tau_ms) * 1000
        signal_kept = int(sig_cum[min(tau_us, sig_cum.shape[0] - 1)])
        noise_kept = int(noise_cum[min(tau_us, noise_cum.shape[0] - 1)])
        tp = signal_kept
        fp = noise_kept
        tn = int(result.noise_total - noise_kept)
        fn = int(result.signal_total - signal_kept)
        tpr = float(tp) / float(result.signal_total) if result.signal_total else 0.0
        fpr = float(fp) / float(result.noise_total) if result.noise_total else 0.0
        fpr_values.append(fpr)
        tpr_values.append(tpr)
        rows.append(
            {
                "tau_ms": tau_ms,
                "tau_us": tau_us,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "tpr": tpr,
                "fpr": fpr,
            }
        )

    auc = _auc_trapezoid(np.asarray(fpr_values), np.asarray(tpr_values))
    return rows, auc


def _target_for(profile_name: str, nominal_hz: float) -> tuple[str, float | None, float | None]:
    targets = PAPER_TARGETS.get(profile_name, {})
    if nominal_hz in targets:
        label, value = targets[nominal_hz]
        return label, value, None
    return "", None, None


def _fmt_float(v: object, digits: int = 6) -> str:
    if v is None or v == "":
        return ""
    return f"{float(v):.{digits}f}"


def _v2e_cov_source_note(v2e_root: Path | None) -> str:
    if v2e_root is None:
        return "v2e source not inspected"
    args_py = v2e_root / "v2ecore" / "v2e_args.py"
    emu_py = v2e_root / "v2ecore" / "emulator.py"
    notes: list[str] = []
    if args_py.exists():
        txt = args_py.read_text(encoding="utf-8", errors="ignore")
        if "currently only in leak events" in txt:
            notes.append("v2e_args.py says noise_rate_cov_decades is currently only in leak events")
    if emu_py.exists():
        txt = emu_py.read_text(encoding="utf-8", errors="ignore")
        if "if self.leak_rate_hz > 0" in txt and "self.noise_rate_array = torch.exp" in txt:
            notes.append("emulator.py initializes log-normal noise_rate_array inside the leak_rate_hz > 0 branch")
        if "generate_shot_noise(" in txt and "noise_rate_cov_decades" not in txt[txt.find("generate_shot_noise(") : txt.find("generate_shot_noise(") + 2000]:
            notes.append("default generate_shot_noise path does not directly use noise_rate_cov_decades")
    return "; ".join(notes) if notes else "no v2e COV caveat detected from local source"


def write_csv(path: Path, rows: list[dict[str, object]], header: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    *,
    manifest_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    missing_rows: list[dict[str, object]],
    cov_note: str,
) -> None:
    lines: list[str] = []
    lines.append("# Driving BAF Paper Profiles")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Fixed BAF Semantics")
    lines.append("")
    lines.append("- radius_px: 1 (3x3 nearest-neighbor window)")
    lines.append("- self_excluded: true")
    lines.append("- polarity: ignored")
    lines.append("- label convention: positive=signal, predicted positive=kept")
    lines.append("- same-polarity BAF is diagnostic only and is not included in paper comparison tables")
    lines.append("")
    lines.append("## v2e COV Source Note")
    lines.append("")
    lines.append(cov_note)
    lines.append("")
    lines.append("## Dataset Manifest")
    lines.append("")
    lines.append("| dataset | level | nominal Hz | actual noise Hz/pixel | events | signal | noise | duration s | source tick ns | COV declared |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in manifest_rows:
        lines.append(
            "| {dataset_id} | {level} | {nominal_hz:.3f} | {actual_noise_hz_per_pixel:.6f} | {events_total} | {signal_total} | {noise_total} | {duration_s:.6f} | {source_tick_ns} | {cov_declared} |".format(
                **r
            )
        )
    if missing_rows:
        lines.append("")
        lines.append("Missing inputs:")
        for r in missing_rows:
            lines.append(f"- {r['dataset_id']} {r['level']}: {r['reason']}")
    lines.append("")
    lines.append("## AUC Summary")
    lines.append("")
    lines.append("| profile | dataset | level | nominal Hz | AUC | target | delta |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for r in summary_rows:
        target = _fmt_float(r.get("target_auc"), 4)
        delta = _fmt_float(r.get("target_delta"), 4)
        lines.append(
            f"| {r['profile']} | {r['dataset_id']} | {r['level']} | {float(r['nominal_hz']):.3f} | {float(r['auc']):.6f} | {target} | {delta} |"
        )
    lines.append("")
    lines.append("## Closest Rows To Paper Targets")
    lines.append("")
    target_rows = [
        r
        for r in summary_rows
        if r.get("target_auc") not in ("", None) and r.get("target_delta") not in ("", None)
    ]
    target_rows.sort(key=lambda r: (str(r["profile"]), float(r["nominal_hz"]), abs(float(r["target_delta"]))))
    lines.append("| profile | nominal Hz | dataset | level | AUC | target | abs delta | target table |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---|")
    for r in target_rows:
        lines.append(
            f"| {r['profile']} | {float(r['nominal_hz']):.3f} | {r['dataset_id']} | {r['level']} | {float(r['auc']):.6f} | {float(r['target_auc']):.4f} | {abs(float(r['target_delta'])):.4f} | {r['target_table']} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Evaluate Driving BAF under fixed paper profiles across local DND21 Driving variants."
    )
    ap.add_argument(
        "--dataset-root",
        default=r"D:\hjx_workspace\scientific_reserach\dataset\DND21",
        help="Root containing mydriving_ED24, mydriving_cov05, and mydriving_paper.",
    )
    ap.add_argument(
        "--datasets",
        default="mydriving_ED24,mydriving_cov05,mydriving_paper,mydriving_jaer",
        help="Comma-separated dataset ids to evaluate.",
    )
    ap.add_argument(
        "--out-dir",
        default="data/DND21/baf_paper_profiles",
        help="Output directory for manifest, ROC points, summary, and report.",
    )
    ap.add_argument(
        "--v2e-root",
        default=r"D:\hjx_workspace\scientific_reserach\v2e",
        help="Optional local v2e source root for COV source note.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)
    dataset_ids = {x.strip() for x in str(args.datasets).split(",") if x.strip()}
    profiles = list(PROFILES)
    max_tau_us = max(max(p.tau_ms) for p in profiles) * 1000
    cov_note = _v2e_cov_source_note(Path(args.v2e_root) if args.v2e_root else None)

    sources, missing_rows = discover_sources(dataset_root, dataset_ids)
    if not sources:
        raise SystemExit(f"No input event sources found under {dataset_root}")

    manifest_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    roc_rows: list[dict[str, object]] = []

    for idx, source in enumerate(sources, start=1):
        print(f"[{idx}/{len(sources)}] {source.dataset_id} {source.level}: {source.path}")
        result = compute_baf_hist(source, max_tau_us=max_tau_us)
        meta = _read_json(source.metadata_path)
        v2e_args = _parse_v2e_args(source.v2e_args_path)
        cov_declared = meta.get("sigma_decades", v2e_args.get("noise_rate_cov_decades", ""))
        source_tick_ns = _source_tick_ns(source) if source.kind.startswith("npy_") else 1000.0
        requested_hz = (
            meta.get("target_shot_rate_hz_per_pixel")
            or meta.get("target_hz_per_pixel")
            or v2e_args.get("shot_noise_rate_hz", "")
            or source.nominal_hz
        )

        manifest_row = {
            "dataset_id": source.dataset_id,
            "dataset_label": source.dataset_label,
            "level": source.level,
            "nominal_hz": source.nominal_hz,
            "requested_hz": requested_hz,
            "actual_noise_hz_per_pixel": result.actual_noise_hz_per_pixel,
            "events_total": result.events_total,
            "signal_total": result.signal_total,
            "noise_total": result.noise_total,
            "duration_s": result.duration_s,
            "first_t_us": result.first_t_us,
            "last_t_us": result.last_t_us,
            "out_of_bounds": result.out_of_bounds,
            "nonmonotonic_t": result.nonmonotonic_t,
            "cov_declared": cov_declared,
            "source_tick_ns": source_tick_ns,
            "source_path": str(source.path),
            "source_kind": source.kind,
            "label_convention": source.label_convention,
            "v2e_args_path": str(source.v2e_args_path or ""),
            "metadata_path": str(source.metadata_path or ""),
        }
        manifest_rows.append(manifest_row)

        for profile in profiles:
            points, auc = profile_points(result, profile)
            target_table, target_auc, _ = _target_for(profile.name, source.nominal_hz)
            target_delta = (auc - float(target_auc)) if target_auc is not None else ""
            summary_row = {
                **manifest_row,
                "profile": profile.name,
                "profile_description": profile.description,
                "tau_grid_ms": " ".join(str(v) for v in profile.tau_ms),
                "radius_px": 1,
                "polarity": "ignored",
                "self_excluded": "true",
                "auc": auc,
                "target_table": target_table,
                "target_auc": target_auc if target_auc is not None else "",
                "target_delta": target_delta,
            }
            summary_rows.append(summary_row)
            for p in points:
                roc_rows.append(
                    {
                        **summary_row,
                        "tau_ms": p["tau_ms"],
                        "tau_us": p["tau_us"],
                        "tp": p["tp"],
                        "fp": p["fp"],
                        "tn": p["tn"],
                        "fn": p["fn"],
                        "tpr": p["tpr"],
                        "fpr": p["fpr"],
                    }
                )
            print(f"  {profile.name}: auc={auc:.6f}")

    manifest_header = [
        "dataset_id",
        "dataset_label",
        "level",
        "nominal_hz",
        "requested_hz",
        "actual_noise_hz_per_pixel",
        "events_total",
        "signal_total",
        "noise_total",
        "duration_s",
        "first_t_us",
        "last_t_us",
        "out_of_bounds",
        "nonmonotonic_t",
        "cov_declared",
        "source_tick_ns",
        "source_path",
        "source_kind",
        "label_convention",
        "v2e_args_path",
        "metadata_path",
    ]
    summary_header = manifest_header + [
        "profile",
        "profile_description",
        "tau_grid_ms",
        "radius_px",
        "polarity",
        "self_excluded",
        "auc",
        "target_table",
        "target_auc",
        "target_delta",
    ]
    roc_header = summary_header + ["tau_ms", "tau_us", "tp", "fp", "tn", "fn", "tpr", "fpr"]
    missing_header = ["dataset_id", "level", "reason"]

    write_csv(out_dir / "dataset_manifest.csv", manifest_rows, manifest_header)
    write_csv(out_dir / "baf_profile_summary.csv", summary_rows, summary_header)
    write_csv(out_dir / "baf_profile_roc_points.csv", roc_rows, roc_header)
    write_csv(out_dir / "missing_inputs.csv", missing_rows, missing_header)
    write_report(
        out_dir / "REPORT.md",
        manifest_rows=manifest_rows,
        summary_rows=summary_rows,
        missing_rows=missing_rows,
        cov_note=cov_note,
    )

    print(f"[done] wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
