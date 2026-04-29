from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from myevs.io.aedat2_events import read_aedat2


DTYPE_SIG = np.dtype([("t", "<u8"), ("x", "<u2"), ("y", "<u2"), ("p", "i1")])
DTYPE_LABELED = np.dtype([("t", "<u8"), ("x", "<u2"), ("y", "<u2"), ("p", "i1"), ("label", "u1")])


LEVELS = (
    ("light", "driving_noise_light", "driving_jaer_shot_1hz.aedat", 1.0),
    ("light_mid", "driving_noise_light_mid", "driving_jaer_shot_3hz.aedat", 3.0),
    ("mid", "driving_noise_mid", "driving_jaer_shot_5hz.aedat", 5.0),
)


def _load_aedat(path: Path, *, width: int, height: int, tick_ns: float) -> np.ndarray:
    _, batches = read_aedat2(str(path), width=width, height=height, batch_events=1_000_000, tick_ns=tick_ns)
    chunks: list[np.ndarray] = []
    for b in batches:
        if len(b) == 0:
            continue
        arr = np.empty((len(b),), dtype=DTYPE_SIG)
        arr["t"] = np.asarray(b.t, dtype=np.uint64)
        arr["x"] = np.asarray(b.x, dtype=np.uint16)
        arr["y"] = np.asarray(b.y, dtype=np.uint16)
        arr["p"] = np.asarray(b.p, dtype=np.int8)
        chunks.append(arr)
    if not chunks:
        return np.empty((0,), dtype=DTYPE_SIG)
    out = np.concatenate(chunks)
    order = np.argsort(out["t"], kind="stable")
    return out[order]


def _key_iter(arr: np.ndarray):
    return zip(arr["t"].tolist(), arr["x"].tolist(), arr["y"].tolist(), arr["p"].tolist())


def _label_by_clean_multiset(noisy: np.ndarray, clean_counts: Counter) -> tuple[np.ndarray, int]:
    labels = np.zeros((noisy.shape[0],), dtype=np.uint8)
    matched = 0
    counts = clean_counts.copy()
    for i, key in enumerate(_key_iter(noisy)):
        n = counts.get(key, 0)
        if n > 0:
            labels[i] = 1
            counts[key] = n - 1
            matched += 1
    return labels, matched


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert JAER-filtered Driving AEDAT2 files to labeled myEVS NPY files.")
    ap.add_argument("--clean-aedat", required=True)
    ap.add_argument("--jaer-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--width", type=int, default=346)
    ap.add_argument("--height", type=int, default=260)
    ap.add_argument("--tick-ns", type=float, default=12.5)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    clean_path = Path(args.clean_aedat)
    jaer_root = Path(args.jaer_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    clean = _load_aedat(clean_path, width=args.width, height=args.height, tick_ns=args.tick_ns)
    if clean.size == 0:
        raise SystemExit(f"No clean events decoded from {clean_path}")
    clean_counts = Counter(_key_iter(clean))

    summary = []
    for level, in_dir, in_name, hz in LEVELS:
        noisy_path = jaer_root / in_dir / in_name
        if not noisy_path.exists():
            raise SystemExit(f"Missing JAER AEDAT: {noisy_path}")

        out_dir = out_root / f"driving_noise_{level}_jaer_withlabel"
        out_dir.mkdir(parents=True, exist_ok=True)
        sig_path = out_dir / f"driving_noise_{level}_signal_only.npy"
        lab_path = out_dir / f"driving_noise_{level}_labeled.npy"
        meta_path = out_dir / "jaer_noise_meta.json"

        if lab_path.exists() and not args.overwrite:
            raise SystemExit(f"Output exists: {lab_path} (use --overwrite)")

        noisy = _load_aedat(noisy_path, width=args.width, height=args.height, tick_ns=args.tick_ns)
        labels, matched = _label_by_clean_multiset(noisy, clean_counts)

        labeled = np.empty((noisy.shape[0],), dtype=DTYPE_LABELED)
        labeled["t"] = noisy["t"]
        labeled["x"] = noisy["x"]
        labeled["y"] = noisy["y"]
        labeled["p"] = noisy["p"]
        labeled["label"] = labels

        np.save(sig_path, clean, allow_pickle=False)
        np.save(lab_path, labeled, allow_pickle=False)

        duration_s = (int(clean["t"][-1]) - int(clean["t"][0])) * float(args.tick_ns) * 1e-9
        meta = {
            "source_clean_aedat": str(clean_path),
            "source_noisy_aedat": str(noisy_path),
            "level": level,
            "target_shot_rate_hz_per_pixel": hz,
            "width": int(args.width),
            "height": int(args.height),
            "tick_ns": float(args.tick_ns),
            "clean_events": int(clean.shape[0]),
            "noisy_events": int(noisy.shape[0]),
            "matched_signal_events": int(matched),
            "unmatched_clean_events": int(clean.shape[0] - matched),
            "noise_events": int(noisy.shape[0] - matched),
            "duration_s": duration_s,
            "estimated_noise_rate_hz_per_pixel": float((noisy.shape[0] - matched) / max(duration_s * args.width * args.height, 1e-12)),
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        summary.append(meta)
        print(
            f"[ok] {level}: clean={clean.shape[0]} noisy={noisy.shape[0]} "
            f"matched={matched} noise={noisy.shape[0] - matched} "
            f"rate={meta['estimated_noise_rate_hz_per_pixel']:.3f}Hz/pix"
        )

    (out_root / "jaer_conversion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
