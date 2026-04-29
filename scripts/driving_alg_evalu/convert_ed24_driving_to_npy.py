from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def discover_levels(src_root: Path) -> list[tuple[str, str, str, float]]:
    levels: list[tuple[str, str, str, float]] = []
    for d in src_root.iterdir():
        if not d.is_dir():
            continue
        m = re.fullmatch(r"(\d+(?:\.\d+)?)hz", d.name.lower())
        if not m:
            continue
        target_hz = float(m.group(1))
        levels.append((f"{m.group(1).replace('.', 'p')}hz", d.name, "driving_mix_result.txt", target_hz))
    return sorted(levels, key=lambda item: item[3])


DTYPE = np.dtype(
    [
        ("t", "<u8"),
        ("x", "<u2"),
        ("y", "<u2"),
        ("p", "i1"),
        ("label", "u1"),
    ]
)


def load_driving_mix_txt(path: Path) -> np.ndarray:
    rows: list[tuple[int, int, int, int, int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("timestamp"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            t = int(float(parts[0]))
            x = int(parts[1])
            y = int(parts[2])
            p_raw = int(parts[3])
            src_label = int(parts[4])

            # ED24 mixed files use label=0 for signal and label=1 for noise.
            # myEVS ROC code uses label=1 as signal and label=0 as noise.
            label = 1 if src_label == 0 else 0
            p = 1 if p_raw > 0 else -1
            rows.append((t, x, y, p, label))

    arr = np.asarray(rows, dtype=DTYPE)
    if arr.size:
        order = np.argsort(arr["t"], kind="stable")
        arr = arr[order]
    return arr


def save_signal_only(noisy: np.ndarray, out_path: Path) -> np.ndarray:
    clean = noisy[noisy["label"] != 0].copy()
    clean["label"] = 1
    np.save(out_path, clean)
    return clean


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert externally generated ED24-style Driving mixed txt files to labeled myEVS NPY files."
    )
    ap.add_argument(
        "--src-root",
        default=r"D:\hjx_workspace\scientific_reserach\dataset\DND21\mydriving_ED24",
    )
    ap.add_argument(
        "--out-root",
        default=r"D:\hjx_workspace\scientific_reserach\dataset\DND21\mydriving_ED24",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src_root = Path(args.src_root)
    out_root = Path(args.out_root)
    summary = []

    levels = discover_levels(src_root)
    if not levels:
        raise SystemExit(f"No '*hz' source directories found under: {src_root}")

    for level, in_dir, in_name, target_hz in levels:
        src_path = src_root / in_dir / in_name
        if not src_path.exists():
            raise SystemExit(f"Missing input file: {src_path}")

        out_dir = out_root / f"driving_noise_{level}_ed24_withlabel"
        out_dir.mkdir(parents=True, exist_ok=True)
        noisy_path = out_dir / f"driving_noise_{level}_labeled.npy"
        clean_path = out_dir / f"driving_noise_{level}_signal_only.npy"
        meta_path = out_dir / "ed24_driving_meta.json"

        if (noisy_path.exists() or clean_path.exists()) and not args.overwrite:
            raise SystemExit(f"Output exists: {out_dir}. Use --overwrite to replace.")

        noisy = load_driving_mix_txt(src_path)
        np.save(noisy_path, noisy)
        clean = save_signal_only(noisy, clean_path)

        events = int(noisy.shape[0])
        signal = int(np.count_nonzero(noisy["label"]))
        noise = int(events - signal)
        t_min = int(noisy["t"].min()) if events else 0
        t_max = int(noisy["t"].max()) if events else 0
        duration_s = float(t_max - t_min) / 1_000_000.0 if t_max > t_min else 0.0
        hz_per_pixel = float(noise) / (346.0 * 260.0 * duration_s) if duration_s > 0 else 0.0

        meta = {
            "level": level,
            "source": str(src_path),
            "target_hz_per_pixel": target_hz,
            "timestamp_unit": "microsecond",
            "source_label_convention": "0=signal, 1=noise",
            "myevs_label_convention": "1=signal, 0=noise",
            "events": events,
            "signal": signal,
            "noise": noise,
            "t_min_us": t_min,
            "t_max_us": t_max,
            "duration_s": duration_s,
            "estimated_noise_hz_per_pixel": hz_per_pixel,
            "clean_npy": str(clean_path),
            "noisy_npy": str(noisy_path),
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        summary.append(meta)

        print(
            f"{level}: events={events} signal={signal} noise={noise} "
            f"duration_s={duration_s:.6f} est_hz={hz_per_pixel:.3f}"
        )
        print(f"  clean: {clean_path}")
        print(f"  noisy: {noisy_path}")

    (out_root / "ed24_driving_conversion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
