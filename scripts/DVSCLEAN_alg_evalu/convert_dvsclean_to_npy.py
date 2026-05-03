from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np


DTYPE = np.dtype(
    [
        ("t", "<u8"),
        ("x", "<u2"),
        ("y", "<u2"),
        ("p", "i1"),
        ("label", "u1"),
    ]
)


def convert_one(path: Path, out_root: Path, *, overwrite: bool) -> dict:
    m = re.search(r"_(50|100)\.hdf5$", path.name, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot infer noise ratio from filename: {path}")
    ratio = m.group(1)
    scene = path.parent.name
    level = f"ratio{ratio}"

    out_dir = out_root / scene / level
    out_dir.mkdir(parents=True, exist_ok=True)
    noisy_path = out_dir / f"{scene}_{level}_labeled.npy"
    clean_path = out_dir / f"{scene}_{level}_signal_only.npy"
    meta_path = out_dir / f"{scene}_{level}_meta.json"
    if (noisy_path.exists() or clean_path.exists()) and not overwrite:
        raise SystemExit(f"Output exists: {out_dir}. Use --overwrite.")

    with h5py.File(path, "r") as f:
        g = f["events"]
        ts_s = np.asarray(g["timestamp"][:], dtype=np.float64)
        x = np.asarray(g["x"][:], dtype=np.int64)
        y = np.asarray(g["y"][:], dtype=np.int64)
        pol = np.asarray(g["polarity"][:], dtype=np.float32)
        src_label = np.asarray(g["label"][:], dtype=np.int8)

    arr = np.empty((ts_s.shape[0],), dtype=DTYPE)
    # Store timestamps as integer microseconds; downstream uses --tick-ns 1000.
    arr["t"] = np.round(ts_s * 1_000_000.0).astype(np.uint64)
    arr["x"] = x.astype(np.uint16)
    arr["y"] = y.astype(np.uint16)
    # DVSCLEAN polarity is a float in [0,1]; binarize to myEVS {-1,+1}.
    arr["p"] = np.where(pol > 0.5, 1, -1).astype(np.int8)
    # DVSCLEAN label convention: 0=signal, 1=noise.
    # myEVS label convention: 1=signal, 0=noise.
    arr["label"] = np.where(src_label == 0, 1, 0).astype(np.uint8)

    if arr.shape[0] > 0:
        order = np.argsort(arr["t"], kind="stable")
        arr = arr[order]

    clean = arr[arr["label"] != 0].copy()
    clean["label"] = 1
    np.save(noisy_path, arr)
    np.save(clean_path, clean)

    events = int(arr.shape[0])
    signal = int(np.count_nonzero(arr["label"]))
    noise = int(events - signal)
    t_min = int(arr["t"].min()) if events else 0
    t_max = int(arr["t"].max()) if events else 0
    duration_s = float(t_max - t_min) / 1_000_000.0 if t_max > t_min else 0.0
    width = int(x.max()) + 1 if x.size else 0
    height = int(y.max()) + 1 if y.size else 0
    noise_per_signal = float(noise) / float(signal) if signal > 0 else 0.0
    noise_hz_per_pixel = float(noise) / (float(width * height) * duration_s) if width and height and duration_s > 0 else 0.0

    meta = {
        "source": str(path),
        "scene": scene,
        "level": level,
        "filename_suffix_ratio": ratio,
        "source_label_convention": "0=signal, 1=noise",
        "myevs_label_convention": "1=signal, 0=noise",
        "timestamp_unit_out": "microsecond",
        "polarity_rule": "p=+1 if polarity>0.5 else -1",
        "width": width,
        "height": height,
        "events": events,
        "signal": signal,
        "noise": noise,
        "noise_per_signal": noise_per_signal,
        "duration_s": duration_s,
        "estimated_noise_hz_per_pixel": noise_hz_per_pixel,
        "clean_npy": str(clean_path),
        "noisy_npy": str(noisy_path),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert DVSCLEAN HDF5 files to myEVS labeled NPY.")
    ap.add_argument("--src-root", default=r"D:\hjx_workspace\scientific_reserach\dataset\DVSCLEAN")
    ap.add_argument("--out-root", default=r"D:\hjx_workspace\scientific_reserach\dataset\DVSCLEAN\converted_npy")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src_root = Path(args.src_root)
    out_root = Path(args.out_root)
    files = sorted(src_root.rglob("*.hdf5"))
    if not files:
        raise SystemExit(f"No .hdf5 files found under: {src_root}")

    summary = [convert_one(p, out_root, overwrite=bool(args.overwrite)) for p in files]
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "dvsclean_conversion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for row in summary:
        print(
            f"{row['scene']} {row['level']}: events={row['events']} signal={row['signal']} "
            f"noise={row['noise']} noise/signal={row['noise_per_signal']:.3f} "
            f"hz/pixel={row['estimated_noise_hz_per_pixel']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
