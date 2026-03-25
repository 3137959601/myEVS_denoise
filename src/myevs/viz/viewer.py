from __future__ import annotations

"""Simple offline viewer (Qt-like raw-gray + polarity color mapping).

Why not draw events directly as colors?
- Your Qt app accumulates into an 8-bit gray buffer (base=127) and then maps it
  to colors. That makes the offline viewer look consistent with the Qt UI.

We mimic the important parts:
- Raw gray accumulation: ON += raw_step, OFF -= raw_step (clipped 0..255)
- Optional hold: keep gray buffer between frames or reset to 127 each frame
- Polarity color mapping:
  - scheme 0: white background, ON red, OFF blue
  - scheme 1: custom colors (from your Qt code)
  - binary mode + deadzone matches Qt rules
"""

from dataclasses import dataclass
from typing import Iterator, Literal, Tuple

import cv2
import numpy as np

from ..events import EventBatch, EventStreamMeta


ColorMode = Literal["gray", "onoff", "onoff_rb"]


@dataclass(frozen=True)
class ColorScheme:
    # OpenCV uses BGR order.
    bg_bgr: Tuple[int, int, int]
    on_bgr: Tuple[int, int, int]
    off_bgr: Tuple[int, int, int]


# These values are taken from Qt `flushFrame()`:
# - scheme 0 (Qt): bg RGB(255,255,255), ON RGB(255,0,0), OFF RGB(0,0,255)
# - scheme 1 (Qt):
#     bg  RGB(30,37,52)   -> BGR(52,37,30)
#     on  RGB(216,223,236)-> BGR(236,223,216)
#     off RGB(64,126,201) -> BGR(201,126,64)
_SCHEMES: dict[int, ColorScheme] = {
    0: ColorScheme(bg_bgr=(255, 255, 255), on_bgr=(0, 0, 255), off_bgr=(255, 0, 0)),
    1: ColorScheme(bg_bgr=(52, 37, 30), on_bgr=(236, 223, 216), off_bgr=(201, 126, 64)),
}


def _make_polarity_lut(
    scheme: ColorScheme,
    *,
    binary: bool,
    deadzone: int,
    show_on: bool,
    show_off: bool,
) -> np.ndarray:
    """Create a 256-entry LUT mapping gray value -> BGR (Qt-like)."""

    base = 127
    dead = int(np.clip(deadzone, 0, 64)) if binary else 0
    denom = float(max(1, 128 - dead))

    bg = np.asarray(scheme.bg_bgr, dtype=np.float32)
    on = np.asarray(scheme.on_bgr, dtype=np.float32)
    off = np.asarray(scheme.off_bgr, dtype=np.float32)

    lut = np.empty((256, 3), dtype=np.uint8)
    for g in range(256):
        dv = int(g) - base
        ad = abs(dv)
        if ad <= dead:
            lut[g] = bg.astype(np.uint8)
            continue

        if binary:
            if dv > 0 and show_on:
                lut[g] = on.astype(np.uint8)
            elif dv < 0 and show_off:
                lut[g] = off.astype(np.uint8)
            else:
                lut[g] = bg.astype(np.uint8)
            continue

        a = float(max(0, ad - dead)) / denom
        if a < 0.0:
            a = 0.0
        elif a > 1.0:
            a = 1.0

        if dv > 0 and show_on:
            c = bg + a * (on - bg)
        elif dv < 0 and show_off:
            c = bg + a * (off - bg)
        else:
            c = bg
        lut[g] = np.clip(c, 0, 255).astype(np.uint8)

    return lut


def _accumulate_raw_gray(
    gray: np.ndarray,
    b: EventBatch,
    *,
    raw_step: int,
    show_on: bool,
    show_off: bool,
    flip_x: bool,
    flip_y: bool,
) -> None:
    """In-place raw-gray accumulation (Qt-like)."""

    x = b.x.astype(np.int32)
    y = b.y.astype(np.int32)
    p = b.p.astype(np.int8)

    # Filter polarity (Qt does this before accumulation)
    vis = np.ones((p.shape[0],), dtype=bool)
    if not show_on:
        vis &= (p <= 0)
    if not show_off:
        vis &= (p > 0)

    if not vis.any():
        return

    x = x[vis]
    y = y[vis]
    p = p[vis]

    # Correct accumulation even with repeated indices.
    # `np.add.at` is C-optimized and handles duplicates properly.
    h, w = gray.shape
    if flip_x:
        x = (w - 1) - x
    if flip_y:
        y = (h - 1) - y

    in_range = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    if not bool(np.any(in_range)):
        return
    x = x[in_range]
    y = y[in_range]
    p = p[in_range]

    idx = (y * w + x).astype(np.int64, copy=False)
    dv = np.where(p > 0, int(raw_step), -int(raw_step)).astype(np.int16, copy=False)
    flat = gray.reshape(-1)
    np.add.at(flat, idx, dv)
    np.clip(gray, 0, 255, out=gray)


def _render_polarity_color(
    gray: np.ndarray,
    scheme: ColorScheme,
    *,
    binary: bool,
    deadzone: int,
    show_on: bool,
    show_off: bool,
) -> np.ndarray:
    """Map raw-gray buffer to BGR image using Qt rules."""

    base = 127

    # Qt: deadzone is only effective in binary mode.
    dead = int(np.clip(deadzone, 0, 64)) if binary else 0
    denom = float(max(1, 128 - dead))

    dv = gray.astype(np.int16) - base
    ad = np.abs(dv).astype(np.int16)

    # Background mask
    bg_mask = ad <= dead

    # Polarity masks (based on dv sign)
    on_mask = (dv > 0) & (~bg_mask)
    off_mask = (dv < 0) & (~bg_mask)

    # Apply showOn/showOff
    if not show_on:
        on_mask[:, :] = False
    if not show_off:
        off_mask[:, :] = False

    img = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
    img[:, :, :] = np.asarray(scheme.bg_bgr, dtype=np.uint8)

    if binary:
        if on_mask.any():
            img[on_mask] = np.asarray(scheme.on_bgr, dtype=np.uint8)
        if off_mask.any():
            img[off_mask] = np.asarray(scheme.off_bgr, dtype=np.uint8)
        return img

    # Non-binary: interpolate from background toward target color using |dv|.
    # a = clamp((ad-dead)/denom, 0..1)
    a = np.clip((ad.astype(np.float32) - float(dead)) / denom, 0.0, 1.0)

    bg = np.asarray(scheme.bg_bgr, dtype=np.float32)
    on = np.asarray(scheme.on_bgr, dtype=np.float32)
    off = np.asarray(scheme.off_bgr, dtype=np.float32)

    # Start from bg everywhere
    out = np.tile(bg[None, None, :], (gray.shape[0], gray.shape[1], 1))

    if on_mask.any():
        out[on_mask] = bg + a[on_mask, None] * (on - bg)
    if off_mask.any():
        out[off_mask] = bg + a[off_mask, None] * (off - bg)

    return np.clip(out, 0, 255).astype(np.uint8)


def view_stream(
    meta: EventStreamMeta,
    batches: Iterator[EventBatch],
    *,
    mode: Literal["fps", "events"] = "fps",
    fps: float = 60.0,
    events_per_frame: int = 200_000,
    tick_us: float = 0.0125,
    color: ColorMode = "onoff",
    scheme_id: int = 0,
    window_name: str = "myEVS",
    key_delay_ms: int = 1,
    raw_step: int = 10,
    deadzone: int = 3,
    binary: bool = False,
    hold: bool = True,
    show_on: bool = True,
    show_off: bool = True,
    realtime: bool = False,
    out_video: str | None = None,
    video_fps: float | None = None,
    no_gui: bool = False,
    flip_x: bool = False,
    flip_y: bool = False,
) -> None:
    if mode == "fps" and fps <= 0:
        raise ValueError("fps must be > 0")
    if mode == "events" and events_per_frame <= 0:
        raise ValueError("events_per_frame must be > 0")
    if tick_us <= 0:
        raise ValueError("tick_us must be > 0")

    # Backward-compatible alias used in older README.
    if color == "onoff_rb":
        color = "onoff"

    h, w = int(meta.height), int(meta.width)
    scheme = _SCHEMES.get(int(scheme_id), _SCHEMES[0])

    lut = _make_polarity_lut(
        scheme,
        binary=binary,
        deadzone=deadzone,
        show_on=show_on,
        show_off=show_off,
    )

    # Raw gray buffer (Qt base=127). Use int16 for fast +/- accumulation.
    gray = np.full((h, w), 127, dtype=np.int16)

    if not no_gui:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        try:
            cv2.resizeWindow(window_name, w, h)
        except Exception:
            pass

    vw = None
    wrote_frames = 0
    if out_video:
        import os

        os.makedirs(os.path.dirname(os.path.abspath(out_video)), exist_ok=True)
        out_fps = float(video_fps if video_fps is not None else fps)
        if out_fps <= 0:
            raise ValueError("video_fps must be > 0")

        ext = os.path.splitext(out_video)[1].lower()
        fourcc = cv2.VideoWriter_fourcc(*("mp4v" if ext in (".mp4", ".m4v") else "XVID"))
        vw = cv2.VideoWriter(out_video, fourcc, out_fps, (w, h), True)
        if not vw.isOpened():
            raise RuntimeError(f"failed to open video writer: {out_video}")

    frame_dt_ticks = int(round((1_000_000.0 / float(fps)) / float(tick_us))) if mode == "fps" else None
    frame_end = None
    acc_n = 0
    frame_count = 0
    t0_wall = None

    def flush() -> None:
        nonlocal frame_count, t0_wall, wrote_frames

        gray_u8 = gray.astype(np.uint8, copy=False)
        # Render display image
        if color == "gray":
            disp = gray_u8
        else:
            disp = lut[gray_u8]

        # Always write as BGR for video
        if disp.ndim == 2:
            disp_bgr = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
        else:
            disp_bgr = disp

        if vw is not None:
            vw.write(disp_bgr)
            wrote_frames += 1

        if not no_gui:
            cv2.imshow(window_name, disp_bgr)
            k = cv2.waitKey(key_delay_ms) & 0xFF
            # Allow ESC / q to exit.
            if k in (27, ord("q")):  # ESC / q
                raise KeyboardInterrupt()

            # If user clicks the window close button, OpenCV may recreate the window
            # on the next `imshow()`. Detect and exit gracefully.
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    raise KeyboardInterrupt()
            except Exception:
                # Some OpenCV builds/platforms can raise here; treat as exit.
                raise KeyboardInterrupt()

        # Reset frame if not holding
        if not hold:
            gray[:, :] = 127

        # Optional: try to play at the requested FPS in wall-clock time.
        # This does NOT guarantee 60fps if rendering/accumulation is too slow.
        frame_count += 1
        if realtime and mode == "fps":
            import time

            if t0_wall is None:
                t0_wall = time.perf_counter()
            target = t0_wall + (frame_count / float(fps))
            now = time.perf_counter()
            dt = target - now
            if dt > 0:
                time.sleep(dt)

    try:
        for b in batches:
            if len(b) == 0:
                continue

            if mode == "events":
                # IMPORTANT: batches can be huge (e.g. 1,000,000 events). To make
                # `--events-per-frame` effective, we must slice inside the batch.
                start = 0
                n = len(b)
                while start < n:
                    remain = int(events_per_frame) - int(acc_n)
                    if remain <= 0:
                        flush()
                        acc_n = 0
                        remain = int(events_per_frame)

                    end = int(min(n, start + remain))
                    sub = EventBatch(t=b.t[start:end], x=b.x[start:end], y=b.y[start:end], p=b.p[start:end])
                    _accumulate_raw_gray(
                        gray,
                        sub,
                        raw_step=raw_step,
                        show_on=show_on,
                        show_off=show_off,
                        flip_x=flip_x,
                        flip_y=flip_y,
                    )
                    acc_n += (end - start)
                    start = end

                    if acc_n >= events_per_frame:
                        flush()
                        acc_n = 0
                continue

            # fps mode: slice by timestamp
            t = np.asarray(b.t, dtype=np.uint64)
            if frame_end is None:
                frame_end = int(t[0]) + int(frame_dt_ticks)

            # Fast path for monotonic timestamps (typical for EVTQ/HDF5):
            # avoid Python per-event scanning by using vectorized boundary search.
            if t.shape[0] <= 1 or bool(np.all(t[1:] >= t[:-1])):
                start = 0
                n_t = int(t.shape[0])
                dt = int(frame_dt_ticks)
                while start < n_t:
                    # If there is a big timestamp gap, fast-forward to the next
                    # frame that actually contains events, instead of rendering
                    # tons of empty frames (which looks like a frozen image).
                    if int(t[start]) >= int(frame_end):
                        if dt <= 0:
                            break
                        skips = (int(t[start]) - int(frame_end)) // dt
                        frame_end += (skips + 1) * dt
                        continue

                    end = int(start + np.searchsorted(t[start:], np.uint64(frame_end), side="left"))

                    if end > start:
                        sub = EventBatch(t=t[start:end], x=b.x[start:end], y=b.y[start:end], p=b.p[start:end])
                        _accumulate_raw_gray(
                            gray,
                            sub,
                            raw_step=raw_step,
                            show_on=show_on,
                            show_off=show_off,
                            flip_x=flip_x,
                            flip_y=flip_y,
                        )

                    if end < n_t:
                        flush()
                        frame_end += dt
                        start = end
                    else:
                        start = end
            else:
                # Fallback for rare out-of-order batches.
                start = 0
                while start < t.shape[0]:
                    if int(t[start]) >= int(frame_end):
                        dt = int(frame_dt_ticks)
                        if dt <= 0:
                            break
                        skips = (int(t[start]) - int(frame_end)) // dt
                        frame_end += (skips + 1) * dt
                        continue

                    end = start
                    while end < t.shape[0] and int(t[end]) < int(frame_end):
                        end += 1

                    if end > start:
                        sub = EventBatch(t=t[start:end], x=b.x[start:end], y=b.y[start:end], p=b.p[start:end])
                        _accumulate_raw_gray(
                            gray,
                            sub,
                            raw_step=raw_step,
                            show_on=show_on,
                            show_off=show_off,
                            flip_x=flip_x,
                            flip_y=flip_y,
                        )

                    if end < t.shape[0]:
                        flush()
                        frame_end += int(frame_dt_ticks)
                        start = end
                    else:
                        start = end

        flush()
    except KeyboardInterrupt:
        pass
    finally:
        import os

        if vw is not None:
            vw.release()
            # Avoid leaving a misleading empty/placeholder file when export failed
            # before any frame could be written (e.g. HDF5 decode/plugin issues).
            if out_video and wrote_frames == 0 and os.path.exists(out_video):
                try:
                    os.remove(out_video)
                except OSError:
                    pass
        if not no_gui:
            cv2.destroyWindow(window_name)
