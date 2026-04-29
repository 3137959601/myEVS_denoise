from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DTYPE_SIG = np.dtype([("t", "<u8"), ("x", "<u2"), ("y", "<u2"), ("p", "i1")])
DTYPE_LABELED = np.dtype([("t", "<u8"), ("x", "<u2"), ("y", "<u2"), ("p", "i1"), ("label", "u1")])


def _load_clean_txt(path: Path, t_unit: str) -> np.ndarray:
    rows: list[tuple[int, int, int, int]] = []
    scale = 1_000_000.0 if t_unit == "s" else 1_000.0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 4:
                continue
            try:
                t_raw = float(parts[0])
                x = int(parts[1])
                y = int(parts[2])
                p01 = int(parts[3])
            except Exception:
                continue
            t_us = int(round(t_raw * scale))
            p = 1 if p01 > 0 else -1
            rows.append((t_us, x, y, p))
    if not rows:
        raise RuntimeError(f"no valid events in {path}")
    arr = np.zeros((len(rows),), dtype=DTYPE_SIG)
    arr["t"] = np.array([r[0] for r in rows], dtype=np.uint64)
    arr["x"] = np.array([r[1] for r in rows], dtype=np.uint16)
    arr["y"] = np.array([r[2] for r in rows], dtype=np.uint16)
    arr["p"] = np.array([r[3] for r in rows], dtype=np.int8)
    # Ensure monotonic by time (stable keeps original tie order)
    idx = np.argsort(arr["t"], kind="stable")
    return arr[idx]


def _sample_lognormal_rates(
    width: int,
    height: int,
    target_hz: float,
    sigma_decades: float,
    rng: np.random.Generator,
    per_pixel_cap_hz: float,
) -> np.ndarray:
    n = width * height
    z = rng.normal(loc=0.0, scale=float(sigma_decades), size=n)
    rates = target_hz * np.power(10.0, z)
    # Re-normalize global mean back to target then apply cap.
    rates *= target_hz / float(np.mean(rates))
    rates = np.clip(rates, 0.0, float(per_pixel_cap_hz))
    # Re-normalize again after clipping.
    rates *= target_hz / max(float(np.mean(rates)), 1e-12)
    return rates.reshape((height, width))


def _generate_shot_noise(
    clean: np.ndarray,
    width: int,
    height: int,
    rate_hz: float,
    sigma_decades: float,
    seed: int,
    per_pixel_cap_hz: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t0 = int(clean["t"][0])
    t1 = int(clean["t"][-1])
    dur_s = max((t1 - t0) / 1_000_000.0, 1e-9)

    rates = _sample_lognormal_rates(
        width=width,
        height=height,
        target_hz=float(rate_hz),
        sigma_decades=float(sigma_decades),
        rng=rng,
        per_pixel_cap_hz=float(per_pixel_cap_hz),
    )
    lam = rates * dur_s
    counts = rng.poisson(lam=lam).astype(np.int64, copy=False)
    total = int(np.sum(counts))
    if total <= 0:
        return np.zeros((0,), dtype=DTYPE_SIG)

    ys, xs = np.nonzero(counts > 0)
    rep = counts[ys, xs]
    x_all = np.repeat(xs.astype(np.uint16), rep)
    y_all = np.repeat(ys.astype(np.uint16), rep)
    # Uniform in [t0, t1), like synthetic shot process over recording interval.
    t_all = rng.integers(low=t0, high=max(t0 + 1, t1), size=total, endpoint=False, dtype=np.int64).astype(np.uint64)
    p_all = np.where(rng.random(total) < 0.5, -1, 1).astype(np.int8)

    noise = np.zeros((total,), dtype=DTYPE_SIG)
    noise["t"] = t_all
    noise["x"] = x_all
    noise["y"] = y_all
    noise["p"] = p_all
    idx = np.argsort(noise["t"], kind="stable")
    return noise[idx]


def _merge_with_labels(clean: np.ndarray, noise: np.ndarray) -> np.ndarray:
    out = np.zeros((clean.shape[0] + noise.shape[0],), dtype=DTYPE_LABELED)
    out[: clean.shape[0]]["t"] = clean["t"]
    out[: clean.shape[0]]["x"] = clean["x"]
    out[: clean.shape[0]]["y"] = clean["y"]
    out[: clean.shape[0]]["p"] = clean["p"]
    out[: clean.shape[0]]["label"] = 1

    off = clean.shape[0]
    out[off:]["t"] = noise["t"]
    out[off:]["x"] = noise["x"]
    out[off:]["y"] = noise["y"]
    out[off:]["p"] = noise["p"]
    out[off:]["label"] = 0

    idx = np.argsort(out["t"], kind="stable")
    return out[idx]


def _save_level(
    out_root: Path,
    level_name: str,
    clean: np.ndarray,
    noisy_labeled: np.ndarray,
    meta: dict,
) -> None:
    d = out_root / f"driving_noise_{level_name}_paper_withlabel"
    d.mkdir(parents=True, exist_ok=True)

    p_sig = d / f"driving_noise_{level_name}_signal_only.npy"
    p_lab = d / f"driving_noise_{level_name}_labeled.npy"
    p_meta = d / "paper_noise_meta.json"

    np.save(p_sig, clean, allow_pickle=False)
    np.save(p_lab, noisy_labeled, allow_pickle=False)
    p_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate DND21 driving paper-style shot-noise labeled npy (1/3/5 Hz).")
    ap.add_argument("--clean-txt", required=True, help="path to paper clean txt, e.g. v2e-dvs-events.txt")
    ap.add_argument("--out-root", required=True, help="output dataset root, e.g. .../dataset/DND21/mydriving_paper")
    ap.add_argument("--width", type=int, default=346)
    ap.add_argument("--height", type=int, default=260)
    ap.add_argument("--sigma-decades", type=float, default=0.5, help="log10-rate sigma across pixels")
    ap.add_argument("--per-pixel-cap-hz", type=float, default=25.0)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--t-unit", choices=["s", "ms"], default="s", help="timestamp unit in txt")
    args = ap.parse_args()

    clean_txt = Path(args.clean_txt)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    clean = _load_clean_txt(clean_txt, t_unit=args.t_unit)
    rates = [("light", 1.0), ("light_mid", 3.0), ("mid", 5.0)]

    for i, (name, hz) in enumerate(rates):
        noise = _generate_shot_noise(
            clean=clean,
            width=int(args.width),
            height=int(args.height),
            rate_hz=float(hz),
            sigma_decades=float(args.sigma_decades),
            seed=int(args.seed + 100 * i),
            per_pixel_cap_hz=float(args.per_pixel_cap_hz),
        )
        labeled = _merge_with_labels(clean=clean, noise=noise)
        meta = {
            "source_clean_txt": str(clean_txt),
            "level": name,
            "target_shot_rate_hz_per_pixel": hz,
            "sigma_decades": args.sigma_decades,
            "per_pixel_cap_hz": args.per_pixel_cap_hz,
            "seed": int(args.seed + 100 * i),
            "width": int(args.width),
            "height": int(args.height),
            "clean_events": int(clean.shape[0]),
            "noise_events": int(noise.shape[0]),
            "total_events": int(labeled.shape[0]),
            "duration_us": int(clean["t"][-1] - clean["t"][0]),
        }
        _save_level(out_root=out_root, level_name=name, clean=clean, noisy_labeled=labeled, meta=meta)
        print(
            f"[ok] {name}: clean={clean.shape[0]} noise={noise.shape[0]} total={labeled.shape[0]} "
            f"target={hz}Hz/pix"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

