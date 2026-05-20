from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.metrics import auc, roc_curve


WIDTH = 346
HEIGHT = 260


@dataclass(frozen=True)
class BafProfile:
    name: str
    tau_ms: tuple[int, ...]


PROFILES: tuple[BafProfile, ...] = (
    BafProfile("baf_edformer_tau_2_200ms", tuple(range(2, 201))),
    BafProfile("baf_ebf_source_tau_grid", (1, 5, 10, 15, 20, 25, 30, 40, 50)),
)


EDFORMER_BAF_TARGETS = {
    1: 0.8479,
    3: 0.8155,
    5: 0.7930,
    7: 0.7732,
    10: 0.7479,
}


@dataclass
class BafHist:
    hz: int
    path: Path
    events_total: int
    signal_total: int
    noise_total: int
    first_t_us: int
    last_t_us: int
    nonmonotonic_t: int
    out_of_bounds: int
    signal_hist: np.ndarray
    noise_hist: np.ndarray
    signal_overflow: int
    noise_overflow: int

    @property
    def duration_s(self) -> float:
        if self.last_t_us <= self.first_t_us:
            return 0.0
        return float(self.last_t_us - self.first_t_us) / 1_000_000.0

    @property
    def actual_noise_hz_per_pixel(self) -> float:
        if self.duration_s <= 0.0:
            return 0.0
        return float(self.noise_total) / float(WIDTH * HEIGHT) / self.duration_s


def parse_hz_list(text: str) -> list[int]:
    out: list[int] = []
    for part in str(text).split(","):
        s = part.strip().lower().replace("hz", "")
        if s:
            out.append(int(s))
    return out


def iter_edformer_txt(path: Path) -> Iterable[tuple[int, int, int, int, int]]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line or line[0] == "#" or line.startswith("timestamp"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            yield int(float(parts[0])), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])


def compute_baf_hist(path: Path, *, hz: int, max_tau_us: int, include_self: bool) -> BafHist:
    last = np.zeros((HEIGHT, WIDTH), dtype=np.int64)
    signal_hist = np.zeros((max_tau_us + 1,), dtype=np.int64)
    noise_hist = np.zeros((max_tau_us + 1,), dtype=np.int64)

    events_total = 0
    signal_total = 0
    noise_total = 0
    signal_overflow = 0
    noise_overflow = 0
    nonmonotonic_t = 0
    out_of_bounds = 0
    first_t_us: int | None = None
    last_t_us: int | None = None

    for t_us, x, y, _p, raw_label in iter_edformer_txt(path):
        if first_t_us is None:
            first_t_us = t_us
        if last_t_us is not None and t_us < last_t_us:
            nonmonotonic_t += 1
        last_t_us = t_us

        if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
            out_of_bounds += 1
            continue

        events_total += 1
        # EDformer Driving txt convention: 0=signal, 1=noise.
        is_noise = int(raw_label) == 1
        is_signal = not is_noise
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
                if not include_self and xx == x and yy == y:
                    continue
                ts = int(row[xx])
                if ts > max_neighbor_ts:
                    max_neighbor_ts = ts

        if max_neighbor_ts > 0:
            dt = int(t_us - max_neighbor_ts)
        else:
            dt = max_tau_us + 1

        if 0 <= dt <= max_tau_us:
            if is_signal:
                signal_hist[dt] += 1
            else:
                noise_hist[dt] += 1
        else:
            if is_signal:
                signal_overflow += 1
            else:
                noise_overflow += 1

        last[y, x] = int(t_us)

    return BafHist(
        hz=hz,
        path=path,
        events_total=events_total,
        signal_total=signal_total,
        noise_total=noise_total,
        first_t_us=int(first_t_us or 0),
        last_t_us=int(last_t_us or 0),
        nonmonotonic_t=nonmonotonic_t,
        out_of_bounds=out_of_bounds,
        signal_hist=signal_hist,
        noise_hist=noise_hist,
        signal_overflow=signal_overflow,
        noise_overflow=noise_overflow,
    )


def auc_trapz(fpr: np.ndarray, tpr: np.ndarray) -> float:
    order = np.argsort(fpr)
    x = np.clip(np.asarray(fpr, dtype=np.float64)[order], 0.0, 1.0)
    y = np.clip(np.asarray(tpr, dtype=np.float64)[order], 0.0, 1.0)
    if x.size == 0:
        return 0.0
    if x[0] > 0.0:
        x = np.concatenate((np.array([0.0]), x))
        y = np.concatenate((np.array([0.0]), y))
    if x[-1] < 1.0:
        x = np.concatenate((x, np.array([1.0])))
        y = np.concatenate((y, np.array([1.0])))
    return float(np.trapezoid(y=y, x=x))


def profile_auc_signal_kept(
    hist: BafHist, profile: BafProfile, *, tau_scale_us: int
) -> tuple[float, list[dict[str, object]]]:
    sig_cum = np.cumsum(hist.signal_hist)
    noise_cum = np.cumsum(hist.noise_hist)
    rows: list[dict[str, object]] = []
    fpr_values: list[float] = []
    tpr_values: list[float] = []
    for tau_ms in profile.tau_ms:
        tau_us = int(tau_ms) * int(tau_scale_us)
        idx = min(tau_us, sig_cum.shape[0] - 1)
        signal_kept = int(sig_cum[idx])
        noise_kept = int(noise_cum[idx])
        tpr = float(signal_kept) / float(hist.signal_total) if hist.signal_total else 0.0
        fpr = float(noise_kept) / float(hist.noise_total) if hist.noise_total else 0.0
        fpr_values.append(fpr)
        tpr_values.append(tpr)
        rows.append(
            {
                "hz": hist.hz,
                "profile": profile.name,
                "tau_ms": tau_ms,
                "tau_us": tau_us,
                "signal_kept": signal_kept,
                "noise_kept": noise_kept,
                "tpr_signal_kept": tpr,
                "fpr_noise_kept": fpr,
            }
        )
    return auc_trapz(np.asarray(fpr_values), np.asarray(tpr_values)), rows


def sklearn_hist_auc_noise_positive(hist: BafHist, *, max_tau_us: int) -> float:
    # Equivalent to sklearn roc_curve(raw_label, baf_noise_score), where raw_label uses
    # EDformer txt semantics 1=noise. dt is the BAF noise score; overflow means no
    # neighbor within max_tau_us and is assigned the largest score.
    signal_counts = np.concatenate((hist.signal_hist, np.array([hist.signal_overflow], dtype=np.int64)))
    noise_counts = np.concatenate((hist.noise_hist, np.array([hist.noise_overflow], dtype=np.int64)))
    scores = np.arange(max_tau_us + 2, dtype=np.float64)
    y_true = np.concatenate(
        (
            np.zeros(int(signal_counts.sum()), dtype=np.uint8),
            np.ones(int(noise_counts.sum()), dtype=np.uint8),
        )
    )
    y_score = np.concatenate(
        (
            np.repeat(scores, signal_counts.astype(np.int64)),
            np.repeat(scores, noise_counts.astype(np.int64)),
        )
    )
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(auc(fpr, tpr))


def write_csv(path: Path, rows: list[dict[str, object]], header: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Evaluate myEVS BAF on EDformer Driving driving_mix_result.txt using label-direct AUC."
    )
    ap.add_argument(
        "--base-path",
        default=r"D:\hjx_workspace\scientific_reserach\dataset\DND21\mydriving_ED24",
        help="Directory containing <hz>hz/driving_mix_result.txt.",
    )
    ap.add_argument("--hz", default="1,3,5,7,10", help="Comma-separated Hz list.")
    ap.add_argument("--filename", default="driving_mix_result.txt")
    ap.add_argument("--skip-missing", action="store_true")
    ap.add_argument(
        "--self-mode",
        choices=["exclude", "include"],
        default="exclude",
        help="exclude matches the paper profile; include is diagnostic only.",
    )
    ap.add_argument(
        "--tau-scale-us",
        type=int,
        default=1000,
        help="Microseconds per tau-grid unit. Default 1000 means tau list is in ms; 1 is a unit diagnostic only.",
    )
    ap.add_argument(
        "--out-dir",
        default="data/DND21/edformer_official_auc/baf_on_driving_mix",
        help="Output directory.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    base = Path(args.base_path)
    out_dir = Path(args.out_dir)
    max_tau_us = max(max(p.tau_ms) for p in PROFILES) * 1000

    summary_rows: list[dict[str, object]] = []
    roc_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []

    hz_values = parse_hz_list(args.hz)
    for idx, hz in enumerate(hz_values, start=1):
        path = base / f"{hz}hz" / str(args.filename)
        if not path.exists():
            msg = f"missing {hz}Hz input: {path}"
            if args.skip_missing:
                print(f"[skip] {msg}")
                continue
            raise FileNotFoundError(msg)

        print(f"[{idx}/{len(hz_values)}] {hz}Hz {path}")
        include_self = str(args.self_mode) == "include"
        hist = compute_baf_hist(path, hz=hz, max_tau_us=max_tau_us, include_self=include_self)
        sklearn_auc = sklearn_hist_auc_noise_positive(hist, max_tau_us=max_tau_us)
        manifest = {
            "hz": hz,
            "path": str(path),
            "events_total": hist.events_total,
            "signal_total_raw_label0": hist.signal_total,
            "noise_total_raw_label1": hist.noise_total,
            "duration_s_if_us": hist.duration_s,
            "actual_noise_hz_per_pixel": hist.actual_noise_hz_per_pixel,
            "first_t_us": hist.first_t_us,
            "last_t_us": hist.last_t_us,
            "nonmonotonic_t": hist.nonmonotonic_t,
            "out_of_bounds": hist.out_of_bounds,
            "label_convention": "EDformer Driving txt: 0=signal, 1=noise",
            "baf_semantics": f"radius=1, 3x3, self_excluded={str(not include_self).lower()}, polarity=ignored",
            "sklearn_auc_raw_label1_noise_dt_score": sklearn_auc,
            "self_mode": args.self_mode,
        }
        manifest_rows.append(manifest)

        for profile in PROFILES:
            profile_auc, points = profile_auc_signal_kept(hist, profile, tau_scale_us=int(args.tau_scale_us))
            target = EDFORMER_BAF_TARGETS.get(hz, "")
            row = {
                **manifest,
                "profile": profile.name,
                "tau_grid_ms": " ".join(str(x) for x in profile.tau_ms),
                "auc_signal_kept": profile_auc,
                "edformer_table2_baf_target": target,
                "target_delta": (profile_auc - float(target)) if target != "" and profile.name == "baf_edformer_tau_2_200ms" else "",
                "tau_scale_us": int(args.tau_scale_us),
            }
            summary_rows.append(row)
            for p in points:
                roc_rows.append({**row, **p})
            print(f"  {profile.name}: auc_signal_kept={profile_auc:.6f}")
        print(f"  sklearn raw-label noise-score diagnostic={sklearn_auc:.6f}")

    manifest_header = [
        "hz",
        "path",
        "events_total",
        "signal_total_raw_label0",
        "noise_total_raw_label1",
        "duration_s_if_us",
        "actual_noise_hz_per_pixel",
        "first_t_us",
        "last_t_us",
        "nonmonotonic_t",
        "out_of_bounds",
        "label_convention",
        "baf_semantics",
        "sklearn_auc_raw_label1_noise_dt_score",
        "self_mode",
    ]
    summary_header = manifest_header + [
        "profile",
        "tau_grid_ms",
        "auc_signal_kept",
        "edformer_table2_baf_target",
        "target_delta",
        "tau_scale_us",
    ]
    roc_header = summary_header + [
        "tau_ms",
        "tau_us",
        "signal_kept",
        "noise_kept",
        "tpr_signal_kept",
        "fpr_noise_kept",
    ]

    write_csv(out_dir / "manifest.csv", manifest_rows, manifest_header)
    write_csv(out_dir / "summary.csv", summary_rows, summary_header)
    write_csv(out_dir / "roc_points.csv", roc_rows, roc_header)
    print(f"[done] wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
