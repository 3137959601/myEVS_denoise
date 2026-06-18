from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evuav_common import HEIGHT, RESULT_ROOT, WIDTH, ensure_dirs, noisy_path, parse_sequence, sequence_reference_path


def _load(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def _accumulate(arr: np.ndarray, *, target_only: bool = False, noise_only: bool = False) -> np.ndarray:
    if target_only:
        arr = arr[arr["target"] == 1]
    if noise_only:
        arr = arr[arr["label"] == 0]
    img = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    if arr.size:
        np.add.at(img, (arr["y"].astype(np.int32), arr["x"].astype(np.int32)), 1.0)
    if img.max() > 0:
        img = np.log1p(img) / np.log1p(img.max())
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description="Render quick qualitative panels for EV-UAV shot-noise streams.")
    ap.add_argument("--sequence", default="test_019")
    ap.add_argument("--noise-ratio", type=float, default=1.0)
    ap.add_argument("--noise-hz", type=float, default=None, help="Optional legacy Hz/pixel stream.")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    ensure_dirs()
    seq = parse_sequence(args.sequence)
    mode = "hz" if args.noise_hz is not None else "ratio"
    level = float(args.noise_hz) if mode == "hz" else float(args.noise_ratio)
    ref = _load(sequence_reference_path(seq))
    noisy = _load(noisy_path(seq, level, mode=mode))
    panels = [
        ("reference", _accumulate(ref)),
        ("target reference", _accumulate(ref, target_only=True)),
        ("noisy", _accumulate(noisy)),
        ("injected noise", _accumulate(noisy, noise_only=True)),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.0), constrained_layout=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=9)
        ax.set_axis_off()
    suffix = f"{level:g}hz" if mode == "hz" else f"r{level:g}"
    out = Path(args.out) if args.out else RESULT_ROOT / "figures" / f"{seq.stem}_shot_{suffix}_quick_panel.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
