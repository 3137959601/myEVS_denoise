from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def _load_led_slice(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # LED npy layout is (4, N): [x, y, polarity(0/1), timestamp(us)]
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[0] != 4:
        raise SystemExit(f"Unexpected LED slice shape: {path} -> {arr.shape}")
    x = arr[0].astype(np.int64, copy=False)
    y = arr[1].astype(np.int64, copy=False)
    p = np.where(arr[2] > 0, 1, -1).astype(np.int8, copy=False)
    t = arr[3].astype(np.uint64, copy=False)
    return x, y, p, t


def _concat_events(parts: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], *, label_signal: int) -> np.ndarray:
    if not parts:
        return np.empty((0,), dtype=DTYPE)
    x = np.concatenate([p[0] for p in parts], axis=0)
    y = np.concatenate([p[1] for p in parts], axis=0)
    pol = np.concatenate([p[2] for p in parts], axis=0)
    t = np.concatenate([p[3] for p in parts], axis=0)
    out = np.empty((t.shape[0],), dtype=DTYPE)
    out["t"] = t
    out["x"] = x.astype(np.uint16)
    out["y"] = y.astype(np.uint16)
    out["p"] = pol
    out["label"] = np.uint8(label_signal)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Stitch LED 10ms slices to 100ms and convert to myEVS labeled npy.")
    ap.add_argument("--src-root", default=r"D:\hjx_workspace\scientific_reserach\dataset\LED")
    ap.add_argument("--out-root", default=r"D:\hjx_workspace\scientific_reserach\dataset\LED\converted_npy")
    ap.add_argument("--slice-start", type=int, default=31)
    ap.add_argument("--slice-end", type=int, default=40)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src_root = Path(args.src_root)
    out_root = Path(args.out_root)
    raw_root = src_root / "raw_events"
    den_root = src_root / "denoise_events"
    noise_root = src_root / "noise_events"

    scenes = sorted([p.name for p in raw_root.iterdir() if p.is_dir()])
    if not scenes:
        raise SystemExit(f"No scenes found under: {raw_root}")

    summary: list[dict] = []
    idxs = list(range(int(args.slice_start), int(args.slice_end) + 1))
    for scene in scenes:
        den_parts = []
        noise_parts = []
        raw_parts = []
        den_gaps = []
        noise_gaps = []
        raw_gaps = []
        prev_den_tmax = None
        prev_noise_tmax = None
        prev_raw_tmax = None

        for idx in idxs:
            name = f"{idx:05d}.npy"
            den_p = den_root / scene / name
            noise_p = noise_root / scene / name
            raw_p = raw_root / scene / name
            if not (den_p.exists() and noise_p.exists() and raw_p.exists()):
                raise SystemExit(f"Missing slice for {scene} {name}")

            den_evt = _load_led_slice(den_p)
            noise_evt = _load_led_slice(noise_p)
            raw_evt = _load_led_slice(raw_p)
            den_parts.append(den_evt)
            noise_parts.append(noise_evt)
            raw_parts.append(raw_evt)

            den_tmin, den_tmax = int(den_evt[3].min()), int(den_evt[3].max())
            noise_tmin, noise_tmax = int(noise_evt[3].min()), int(noise_evt[3].max())
            raw_tmin, raw_tmax = int(raw_evt[3].min()), int(raw_evt[3].max())
            if prev_den_tmax is not None:
                den_gaps.append(den_tmin - prev_den_tmax)
            if prev_noise_tmax is not None:
                noise_gaps.append(noise_tmin - prev_noise_tmax)
            if prev_raw_tmax is not None:
                raw_gaps.append(raw_tmin - prev_raw_tmax)
            prev_den_tmax = den_tmax
            prev_noise_tmax = noise_tmax
            prev_raw_tmax = raw_tmax

        signal = _concat_events(den_parts, label_signal=1)
        noise = _concat_events(noise_parts, label_signal=0)
        raw = _concat_events(raw_parts, label_signal=0)

        noisy = np.concatenate([signal, noise], axis=0)
        if noisy.shape[0] > 0:
            order = np.argsort(noisy["t"], kind="stable")
            noisy = noisy[order]

        out_dir = out_root / scene / f"slices_{args.slice_start:05d}_{args.slice_end:05d}_100ms"
        out_dir.mkdir(parents=True, exist_ok=True)
        noisy_path = out_dir / f"{scene}_100ms_labeled.npy"
        clean_path = out_dir / f"{scene}_100ms_signal_only.npy"
        meta_path = out_dir / "led_100ms_meta.json"
        if (noisy_path.exists() or clean_path.exists()) and not args.overwrite:
            raise SystemExit(f"Output exists: {out_dir} (use --overwrite)")

        np.save(noisy_path, noisy)
        np.save(clean_path, signal)

        x_max = int(max(signal["x"].max(initial=0), noise["x"].max(initial=0), raw["x"].max(initial=0)))
        y_max = int(max(signal["y"].max(initial=0), noise["y"].max(initial=0), raw["y"].max(initial=0)))
        width = x_max + 1
        height = y_max + 1
        t_min = int(noisy["t"].min()) if noisy.shape[0] else 0
        t_max = int(noisy["t"].max()) if noisy.shape[0] else 0
        duration_s = float(t_max - t_min) / 1_000_000.0 if t_max > t_min else 0.0
        noise_per_signal = float(noise.shape[0]) / float(signal.shape[0]) if signal.shape[0] else 0.0
        noise_hz_per_pixel = float(noise.shape[0]) / (float(width * height) * duration_s) if duration_s > 0 else 0.0
        raw_minus_sum = int(raw.shape[0] - signal.shape[0] - noise.shape[0])

        meta = {
            "scene": scene,
            "slice_start": int(args.slice_start),
            "slice_end": int(args.slice_end),
            "num_slices": len(idxs),
            "timestamp_unit": "us",
            "continuity_gap_raw_us": raw_gaps,
            "continuity_gap_denoise_us": den_gaps,
            "continuity_gap_noise_us": noise_gaps,
            "signal_events": int(signal.shape[0]),
            "noise_events": int(noise.shape[0]),
            "raw_events": int(raw.shape[0]),
            "raw_minus_signal_minus_noise": raw_minus_sum,
            "noise_per_signal": noise_per_signal,
            "estimated_noise_hz_per_pixel": noise_hz_per_pixel,
            "width": width,
            "height": height,
            "t_min_us": t_min,
            "t_max_us": t_max,
            "duration_s": duration_s,
            "clean_npy": str(clean_path),
            "noisy_npy": str(noisy_path),
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        summary.append(meta)
        print(
            f"{scene}: signal={signal.shape[0]} noise={noise.shape[0]} raw={raw.shape[0]} "
            f"ratio={noise_per_signal:.4f} hz/pixel={noise_hz_per_pixel:.6f} raw_minus_sum={raw_minus_sum}"
        )

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "led_100ms_conversion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
