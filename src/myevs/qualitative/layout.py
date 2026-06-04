from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


def _read_image(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def make_panel(
    images: Sequence[str | Path],
    *,
    labels: Sequence[str] | None = None,
    cols: int = 4,
    pad_px: int = 10,
    label_height_px: int = 44,
    bg_bgr: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    if not images:
        raise ValueError("At least one image is required")
    imgs = [_read_image(p) for p in images]
    h = max(img.shape[0] for img in imgs)
    w = max(img.shape[1] for img in imgs)
    normalized: list[np.ndarray] = []
    for img in imgs:
        canvas = np.full((h, w, 3), bg_bgr, dtype=np.uint8)
        y0 = (h - img.shape[0]) // 2
        x0 = (w - img.shape[1]) // 2
        canvas[y0 : y0 + img.shape[0], x0 : x0 + img.shape[1]] = img
        normalized.append(canvas)

    labels_l = list(labels or [Path(str(p)).stem for p in images])
    cols = max(1, int(cols))
    rows = int(np.ceil(len(normalized) / float(cols)))
    tile_h = h + int(label_height_px)
    tile_w = w
    out_h = rows * tile_h + (rows + 1) * int(pad_px)
    out_w = cols * tile_w + (cols + 1) * int(pad_px)
    out = np.full((out_h, out_w, 3), bg_bgr, dtype=np.uint8)

    for idx, img in enumerate(normalized):
        r = idx // cols
        c = idx % cols
        x0 = int(pad_px) + c * (tile_w + int(pad_px))
        y0 = int(pad_px) + r * (tile_h + int(pad_px))
        label = labels_l[idx]
        out[y0 : y0 + h, x0 : x0 + w] = img
        _draw_centered_text(
            out,
            str(label),
            x0=x0,
            y0=y0 + h,
            w=tile_w,
            h=int(label_height_px),
            font_size=24,
        )
    return out


def save_panel(
    images: Sequence[str | Path],
    out_path: str | Path,
    *,
    labels: Sequence[str] | None = None,
    cols: int = 4,
) -> Path:
    from .render import write_pdf_from_image, write_png

    out = Path(out_path)
    panel = make_panel(images, labels=labels, cols=cols)
    if out.suffix.lower() == ".pdf":
        write_pdf_from_image(out, panel, dpi=600)
    else:
        write_png(out, panel)
    return out


def make_dataset_method_panel(
    *,
    rendered_root: str | Path,
    case_ids: Sequence[str],
    method_order: Sequence[str],
    out_path: str | Path,
    case_labels: Sequence[str] | None = None,
    tile_pad_px: int = 8,
    caption_h_px: int = 60,
    row_label_w_px: int = 60,
    max_tile_w_px: int = 346,
    max_tile_h_px: int = 260,
    missing_bg_bgr: tuple[int, int, int] = (245, 245, 245),
) -> Path:
    """Create a paper panel with datasets as rows and methods as columns."""

    root = Path(rendered_root)
    cases = list(case_ids)
    methods = list(method_order)
    labels = list(case_labels or cases)
    if len(labels) != len(cases):
        raise ValueError("case_labels must match case_ids length")

    found: list[np.ndarray] = []
    for case_id in cases:
        for method in methods:
            p = root / case_id / f"{method}.png"
            if p.exists():
                found.append(_read_image(p))
    if not found:
        raise FileNotFoundError(f"No rendered images found under {root}")

    tile_h = min(max(img.shape[0] for img in found), int(max_tile_h_px))
    tile_w = min(max(img.shape[1] for img in found), int(max_tile_w_px))
    rows = len(cases)
    cols = len(methods)
    out_h = rows * tile_h + (rows + 1) * tile_pad_px + caption_h_px
    out_w = row_label_w_px + cols * tile_w + (cols + 1) * tile_pad_px
    out = np.full((out_h, out_w, 3), (255, 255, 255), dtype=np.uint8)

    for r, case_id in enumerate(cases):
        y0 = tile_pad_px + r * (tile_h + tile_pad_px)
        _draw_vertical_label(out, labels[r], x_center=row_label_w_px // 2, y0=y0, y1=y0 + tile_h)
        for c, method in enumerate(methods):
            x0 = row_label_w_px + tile_pad_px + c * (tile_w + tile_pad_px)
            p = root / case_id / f"{method}.png"
            if p.exists():
                canvas = np.full((tile_h, tile_w, 3), (255, 255, 255), dtype=np.uint8)
                img = _read_image(p)
                img = _fit_image(img, tile_w=tile_w, tile_h=tile_h)
                yy = (tile_h - img.shape[0]) // 2
                xx = (tile_w - img.shape[1]) // 2
                canvas[yy : yy + img.shape[0], xx : xx + img.shape[1]] = img
            else:
                canvas = np.full((tile_h, tile_w, 3), missing_bg_bgr, dtype=np.uint8)
                cv2.putText(
                    canvas,
                    "pending",
                    (max(5, tile_w // 2 - 42), max(24, tile_h // 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (100, 100, 100),
                    1,
                    cv2.LINE_AA,
                )
            out[y0 : y0 + tile_h, x0 : x0 + tile_w] = canvas

    caption_y0 = tile_pad_px + rows * (tile_h + tile_pad_px)
    caption_labels = _panel_caption_labels(methods)
    for c, caption in enumerate(caption_labels):
        x0 = row_label_w_px + tile_pad_px + c * (tile_w + tile_pad_px)
        _draw_centered_text(out, caption, x0=x0, y0=caption_y0, w=tile_w, h=caption_h_px, font_size=28)

    return save_panel_image(out, out_path)


def _fit_image(img: np.ndarray, *, tile_w: int, tile_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h <= tile_h and w <= tile_w:
        return img
    scale = min(float(tile_w) / float(w), float(tile_h) / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)


def save_panel_image(image_bgr: np.ndarray, out_path: str | Path) -> Path:
    from .render import write_pdf_from_image, write_png

    out = Path(out_path)
    if out.suffix.lower() == ".pdf":
        write_pdf_from_image(out, image_bgr, dpi=600)
    else:
        write_png(out, image_bgr)
    return out


def _panel_caption_labels(methods: Sequence[str]) -> list[str]:
    pretty = {
        "Noisy": "Event frame",
        "Ours": "Ours",
        "BAF": "BAF",
        "STCF": "STCF",
        "EBF": "EBF",
        "TS": "TS",
        "PFD": "PFD",
        "EDnCNN": "EDnCNN",
        "EDFormer": "EDFormer",
    }
    labels: list[str] = []
    for i, method in enumerate(methods):
        letter = chr(ord("a") + i)
        labels.append(f"({letter}) {pretty.get(method, method)}")
    return labels


def _pil_font(size: int):
    try:
        from PIL import ImageFont
    except Exception:
        return None
    for name in ("times.ttf", "Times New Roman.ttf", "Times New Roman"):
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            continue
    try:
        return ImageFont.truetype("DejaVuSerif.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def _draw_centered_text(img_bgr: np.ndarray, text: str, *, x0: int, y0: int, w: int, h: int, font_size: int) -> None:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
        cv2.putText(
            img_bgr,
            text,
            (x0 + max(0, (w - tw) // 2), y0 + max(th + 2, (h + th) // 2 - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        return

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    font = _pil_font(font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text((x0 + (w - tw) / 2, y0 + (h - th) / 2 - 1), text, fill=(20, 20, 20), font=font)
    img_bgr[:, :] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def _draw_vertical_label(img_bgr: np.ndarray, text: str, *, x_center: int, y0: int, y1: int) -> None:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        # Fallback: compact stacked letters.
        y = y0 + 18
        for ch in text:
            cv2.putText(img_bgr, ch, (max(0, x_center - 7), y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
            y += 14
        return

    font = _pil_font(24)
    tmp = Image.new("RGB", (260, 36), (255, 255, 255))
    draw = ImageDraw.Draw(tmp)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tmp = Image.new("RGB", (tw + 8, th + 8), (255, 255, 255))
    draw = ImageDraw.Draw(tmp)
    draw.text((4, 3), text, fill=(20, 20, 20), font=font)
    rot = tmp.rotate(90, expand=True)

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    x = int(x_center - rot.size[0] / 2)
    y = int((y0 + y1) / 2 - rot.size[1] / 2)
    pil.paste(rot, (max(0, x), max(0, y)))
    img_bgr[:, :] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
