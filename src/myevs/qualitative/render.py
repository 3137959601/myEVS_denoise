from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from myevs.events import EventBatch


def render_events_to_image(
    batch: EventBatch,
    *,
    width: int,
    height: int,
    raw_step: int = 10,
    deadzone: int = 3,
    binary: bool = False,
    scheme: int = 0,
    show_on: bool = True,
    show_off: bool = True,
    flip_x: bool = False,
    flip_y: bool = False,
) -> np.ndarray:
    from myevs.viz.viewer import _SCHEMES, _make_polarity_lut  # local private reuse by design for style parity

    gray = np.full((int(height), int(width)), 127, dtype=np.int16)
    if len(batch) > 0:
        x = np.asarray(batch.x, dtype=np.int32)
        y = np.asarray(batch.y, dtype=np.int32)
        p = np.asarray(batch.p, dtype=np.int8)
        vis = np.ones((p.shape[0],), dtype=bool)
        if not show_on:
            vis &= p <= 0
        if not show_off:
            vis &= p > 0
        x = x[vis]
        y = y[vis]
        p = p[vis]
        if flip_x:
            x = (int(width) - 1) - x
        if flip_y:
            y = (int(height) - 1) - y
        inb = (x >= 0) & (x < int(width)) & (y >= 0) & (y < int(height))
        if inb.any():
            idx = (y[inb] * int(width) + x[inb]).astype(np.int64, copy=False)
            dv = np.where(p[inb] > 0, int(raw_step), -int(raw_step)).astype(np.int16, copy=False)
            flat = gray.reshape(-1)
            np.add.at(flat, idx, dv)
            np.clip(gray, 0, 255, out=gray)

    color_scheme = _SCHEMES.get(int(scheme), _SCHEMES[0])
    lut = _make_polarity_lut(
        color_scheme,
        binary=bool(binary),
        deadzone=int(deadzone),
        show_on=bool(show_on),
        show_off=bool(show_off),
    )
    return lut[gray.astype(np.uint8, copy=False)]


def write_png(path: str | Path, image_bgr: np.ndarray) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(p), image_bgr):
        raise RuntimeError(f"Failed to write image: {p}")


def write_pdf_from_image(path: str | Path, image_bgr: np.ndarray, *, dpi: int = 600) -> None:
    from matplotlib import pyplot as plt

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    fig = plt.figure(figsize=(w / float(dpi), h / float(dpi)), dpi=int(dpi))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(rgb, interpolation="nearest")
    ax.axis("off")
    fig.savefig(p, dpi=int(dpi), bbox_inches="tight", pad_inches=0)
    plt.close(fig)
