from __future__ import annotations

import argparse

import numpy as np

from evuav_common import (
    DTYPE_REF,
    HEIGHT,
    SELECTED_TEST_SEQUENCES,
    SHOT_LEVELS_RATIO,
    WIDTH,
    ensure_dirs,
    format_float,
    load_reference,
    noisy_meta_path,
    noisy_path,
    parse_sequence,
    sequence_reference_path,
    write_json,
)


def _sample_lognormal_rates(
    width: int,
    height: int,
    target_hz: float,
    sigma_decades: float,
    rng: np.random.Generator,
    per_pixel_cap_hz: float,
) -> np.ndarray:
    n = int(width) * int(height)
    z = rng.normal(loc=0.0, scale=float(sigma_decades), size=n)
    rates = target_hz * np.power(10.0, z)
    rates *= target_hz / max(float(np.mean(rates)), 1e-12)
    rates = np.clip(rates, 0.0, float(per_pixel_cap_hz))
    rates *= target_hz / max(float(np.mean(rates)), 1e-12)
    return rates.reshape((height, width))


def generate_shot_noise(
    reference: np.ndarray,
    *,
    width: int,
    height: int,
    rate_hz: float,
    sigma_decades: float,
    seed: int,
    per_pixel_cap_hz: float,
) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(int(seed))
    t0 = int(reference["t"][0])
    t1 = int(reference["t"][-1])
    dur_s = max((t1 - t0) / 1_000_000.0, 1e-9)
    rates = _sample_lognormal_rates(width, height, rate_hz, sigma_decades, rng, per_pixel_cap_hz)
    counts = rng.poisson(lam=rates * dur_s).astype(np.int64, copy=False)
    total = int(np.sum(counts))
    noise = np.zeros((total,), dtype=DTYPE_REF)
    if total:
        ys, xs = np.nonzero(counts > 0)
        rep = counts[ys, xs]
        noise["x"] = np.repeat(xs.astype(np.uint16), rep)
        noise["y"] = np.repeat(ys.astype(np.uint16), rep)
        noise["t"] = rng.integers(low=t0, high=max(t0 + 1, t1), size=total, endpoint=False, dtype=np.int64).astype(np.uint64)
        noise["p"] = np.where(rng.random(total) < 0.5, -1, 1).astype(np.int8)
        noise["label"] = 0
        noise["target"] = 0
        noise["source"] = 3
        noise = noise[np.argsort(noise["t"], kind="stable")]
    stats = {
        "duration_s": dur_s,
        "target_shot_rate_hz_per_pixel": float(rate_hz),
        "actual_noise_rate_hz_per_pixel": float(total / max(dur_s * width * height, 1e-12)),
        "sigma_decades": float(sigma_decades),
        "per_pixel_cap_hz": float(per_pixel_cap_hz),
        "rate_mean_hz": float(np.mean(rates)),
        "rate_p95_hz": float(np.quantile(rates, 0.95)),
        "rate_max_hz": float(np.max(rates)),
    }
    return noise, stats


def generate_shot_noise_by_ratio(
    reference: np.ndarray,
    *,
    width: int,
    height: int,
    ratio: float,
    sigma_decades: float,
    seed: int,
    per_pixel_cap_weight: float,
) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(int(seed))
    t0 = int(reference["t"][0])
    t1 = int(reference["t"][-1])
    dur_s = max((t1 - t0) / 1_000_000.0, 1e-9)
    n_noise = int(round(reference.shape[0] * float(ratio)))
    noise = np.zeros((n_noise,), dtype=DTYPE_REF)
    if n_noise <= 0:
        return noise, {
            "duration_s": dur_s,
            "noise_ratio_to_reference": float(ratio),
            "actual_noise_ratio_to_reference": 0.0,
            "sigma_decades": float(sigma_decades),
            "per_pixel_cap_weight": float(per_pixel_cap_weight),
            "equivalent_noise_rate_hz_per_pixel": 0.0,
        }

    weights = _sample_lognormal_rates(
        width=width,
        height=height,
        target_hz=1.0,
        sigma_decades=float(sigma_decades),
        rng=rng,
        per_pixel_cap_hz=float(per_pixel_cap_weight),
    ).reshape(-1)
    weights = weights / max(float(np.sum(weights)), 1e-12)
    flat = rng.choice(width * height, size=n_noise, replace=True, p=weights)
    ys = (flat // width).astype(np.uint16)
    xs = (flat % width).astype(np.uint16)
    noise["x"] = xs
    noise["y"] = ys
    noise["t"] = rng.integers(low=t0, high=max(t0 + 1, t1), size=n_noise, endpoint=False, dtype=np.int64).astype(np.uint64)
    noise["p"] = np.where(rng.random(n_noise) < 0.5, -1, 1).astype(np.int8)
    noise["label"] = 0
    noise["target"] = 0
    noise["source"] = 3
    noise = noise[np.argsort(noise["t"], kind="stable")]
    return noise, {
        "duration_s": dur_s,
        "noise_ratio_to_reference": float(ratio),
        "actual_noise_ratio_to_reference": float(n_noise / max(reference.shape[0], 1)),
        "sigma_decades": float(sigma_decades),
        "per_pixel_cap_weight": float(per_pixel_cap_weight),
        "weight_p95": float(np.quantile(weights, 0.95)),
        "weight_max": float(np.max(weights)),
        "equivalent_noise_rate_hz_per_pixel": float(n_noise / max(dur_s * width * height, 1e-12)),
    }


def merge_reference_noise(reference: np.ndarray, noise: np.ndarray) -> np.ndarray:
    out = np.empty((reference.shape[0] + noise.shape[0],), dtype=DTYPE_REF)
    out[: reference.shape[0]] = reference
    out[reference.shape[0] :] = noise
    return out[np.argsort(out["t"], kind="stable")]


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate jAER-style shot-noise EV-UAV streams.")
    ap.add_argument("--sequences", nargs="*", default=list(SELECTED_TEST_SEQUENCES))
    ap.add_argument("--levels-ratio", nargs="*", type=float, default=list(SHOT_LEVELS_RATIO))
    ap.add_argument("--levels-hz", nargs="*", type=float, default=None, help="Optional legacy Hz/pixel mode; not the default paper setting.")
    ap.add_argument("--sigma-decades", type=float, default=0.5)
    ap.add_argument("--per-pixel-cap-hz", type=float, default=25.0)
    ap.add_argument("--per-pixel-cap-weight", type=float, default=25.0)
    ap.add_argument("--seed", type=int, default=20260615)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    ensure_dirs()
    mode = "hz" if args.levels_hz is not None else "ratio"
    levels = list(args.levels_hz) if mode == "hz" else list(args.levels_ratio)
    for si, raw in enumerate(args.sequences):
        seq = parse_sequence(raw)
        ref_path = sequence_reference_path(seq)
        if not ref_path.exists():
            raise SystemExit(f"missing reference {ref_path}; run convert_evuav_to_myevs.py first")
        reference = load_reference(ref_path)
        for li, level in enumerate(levels):
            out_path = noisy_path(seq, level, mode=mode)
            if out_path.exists() and not args.overwrite:
                print(f"[skip] exists {out_path}")
                continue
            if mode == "hz":
                noise, stats = generate_shot_noise(
                    reference,
                    width=WIDTH,
                    height=HEIGHT,
                    rate_hz=float(level),
                    sigma_decades=float(args.sigma_decades),
                    seed=int(args.seed + si * 1000 + li * 100),
                    per_pixel_cap_hz=float(args.per_pixel_cap_hz),
                )
                level_name = f"shot_{format_float(level)}hz"
            else:
                noise, stats = generate_shot_noise_by_ratio(
                    reference,
                    width=WIDTH,
                    height=HEIGHT,
                    ratio=float(level),
                    sigma_decades=float(args.sigma_decades),
                    seed=int(args.seed + si * 1000 + li * 100),
                    per_pixel_cap_weight=float(args.per_pixel_cap_weight),
                )
                level_name = f"shot_r{format_float(level)}"
            noisy = merge_reference_noise(reference, noise)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, noisy, allow_pickle=False)
            meta = {
                "sequence": seq.stem,
                "split": seq.split,
                "source_reference": str(ref_path),
                "output_noisy": str(out_path),
                "noise_model": "jAER-style Python shot noise; log-normal per-pixel spatial dispersion",
                "noise_mode": mode,
                "level": level_name,
                "seed": int(args.seed + si * 1000 + li * 100),
                "width": WIDTH,
                "height": HEIGHT,
                "reference_events": int(reference.shape[0]),
                "target_reference_events": int(np.sum(reference["target"] == 1)),
                "background_reference_events": int(np.sum((reference["label"] == 1) & (reference["target"] == 0))),
                "injected_noise_events": int(noise.shape[0]),
                "total_events": int(noisy.shape[0]),
                **stats,
            }
            write_json(noisy_meta_path(seq, level, mode=mode), meta)
            print(
                f"[ok] {seq.stem} {level:g}{'Hz' if mode == 'hz' else 'x'}: ref={reference.shape[0]} noise={noise.shape[0]} "
                f"eq={stats.get('equivalent_noise_rate_hz_per_pixel', stats.get('actual_noise_rate_hz_per_pixel', 0.0)):.4f}Hz/pix"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
