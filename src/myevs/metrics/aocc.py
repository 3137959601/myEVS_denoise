from __future__ import annotations

import os

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


def _configure_cv2_for_aocc() -> None:
    """Configure OpenCV to be less disruptive on desktop workloads.

    AOCC computation calls GaussianBlur+Sobel many times; OpenCV may spawn many
    threads and temporarily saturate CPU, making the machine feel "stuck".
    We cap threads to keep the system responsive.
    """

    if cv2 is None:
        return

    try:
        n = int(os.environ.get("MYEVS_AOCC_CV2_THREADS", "4"))
    except Exception:
        n = 1
    if n <= 0:
        n = 1

    try:
        cv2.setNumThreads(int(n))
    except Exception:
        pass

    try:
        if hasattr(cv2, "ocl") and hasattr(cv2.ocl, "setUseOpenCL"):
            cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass


_configure_cv2_for_aocc()


# AOCC output unit.
# We report AOCC in a common "paper-style" display unit of 1e7.
# That is: aocc_reported = aocc_raw / AOCC_UNIT.
#
# This keeps values around ~0.1..2.0 for typical ED24 runs and makes tables easier to scan.
AOCC_UNIT = float(os.environ.get("MYEVS_AOCC_UNIT", "1e7"))


def _scale_aocc(aocc_raw: float) -> float:
    unit = float(AOCC_UNIT)
    if not np.isfinite(unit) or unit <= 0:
        unit = 1e7
    return float(aocc_raw) / unit


def _gaussian_blur5_sigma2(image_u8: np.ndarray) -> np.ndarray:
    """Small, dependency-free Gaussian blur (ksize=5, sigma=2).

    The official AOCC reference implementation applies `cv2.GaussianBlur(frame, (5,5), 2)`
    before Sobel. We implement an equivalent Gaussian smoothing via separable 1D kernels.

    Notes
    - Input is treated as float32 internally.
    - Edge handling: constant 0 padding (matches the overall "empty background" assumption).
    """

    img = np.asarray(image_u8)
    if img.ndim != 2:
        raise ValueError("image_u8 must be 2D")

    sigma = 2.0
    ksize = 5
    r = ksize // 2

    x = np.arange(-r, r + 1, dtype=np.float32)
    k1 = np.exp(-(x * x) / (2.0 * sigma * sigma))
    k1 = (k1 / np.sum(k1)).astype(np.float32, copy=False)

    f = img.astype(np.float32, copy=False)

    # Convolve rows
    p = np.pad(f, ((0, 0), (r, r)), mode="constant", constant_values=0.0)
    tmp = np.empty_like(f, dtype=np.float32)
    for j in range(f.shape[1]):
        # sum_{k=-r..r} k1[k+r] * p[:, j+k+r]
        s = 0.0
        for kk in range(ksize):
            s = s + k1[kk] * p[:, j + kk]
        tmp[:, j] = s

    # Convolve cols
    p2 = np.pad(tmp, ((r, r), (0, 0)), mode="constant", constant_values=0.0)
    out = np.empty_like(tmp, dtype=np.float32)
    for i in range(f.shape[0]):
        s = 0.0
        for kk in range(ksize):
            s = s + k1[kk] * p2[i + kk, :]
        out[i, :] = s

    return out


def _contrast_official_cv2(image_u8: np.ndarray) -> float:
    """Official-style contrast: GaussianBlur(5, sigma=2) + Sobel mag std.

    Mirrors AOCC official implementation in D:/hjx_workspace/scientific_reserach/AOCC/AOCC.py:
    - frame is uint8 with values {0,255}
    - blur then Sobel with ksize=3, CV_64F
    - contrast = std( sqrt(Gx^2 + Gy^2) )
    """

    if cv2 is None:
        # fallback: blur (numpy) + sobel (numpy)
        blurred = _gaussian_blur5_sigma2(image_u8)
        return _sobel_contrast_std(blurred)

    gray = np.asarray(image_u8)
    if gray.ndim != 2:
        raise ValueError("image_u8 must be 2D")

    blurred = cv2.GaussianBlur(gray, (5, 5), 2)
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    return float(np.std(magnitude))


def _sobel_contrast_std(binary_img: np.ndarray) -> float:
    """Return Sobel-gradient-magnitude standard deviation for a binary image.

    This matches the AOCC paper's "contrast" definition in the common simplified form:
    - Build a binary event frame within a time window.
    - Compute Sobel gradients.
    - Contrast C is std( sqrt(Gx^2 + Gy^2) ).

    The implementation is dependency-free (numpy only) and uses a 1-pixel zero padding.
    """

    img = np.asarray(binary_img)
    if img.ndim != 2:
        raise ValueError("binary_img must be a 2D array")

    # Convert to float32 for math.
    f = img.astype(np.float32, copy=False)
    p = np.pad(f, pad_width=((1, 1), (1, 1)), mode="constant", constant_values=0.0)

    # Sobel kernels:
    # Gx = [[ 1, 0,-1],[ 2, 0,-2],[ 1, 0,-1]]
    # Gy = [[ 1, 2, 1],[ 0, 0, 0],[-1,-2,-1]]
    gx = (
        p[0:-2, 2:] + 2.0 * p[1:-1, 2:] + p[2:, 2:]
        - (p[0:-2, 0:-2] + 2.0 * p[1:-1, 0:-2] + p[2:, 0:-2])
    )
    gy = (
        p[0:-2, 0:-2] + 2.0 * p[0:-2, 1:-1] + p[0:-2, 2:]
        - (p[2:, 0:-2] + 2.0 * p[2:, 1:-1] + p[2:, 2:])
    )

    mag = np.hypot(gx, gy)
    return float(np.std(mag, dtype=np.float64))


def aocc_from_xyt(
    x: np.ndarray,
    y: np.ndarray,
    t_us: np.ndarray,
    *,
    width: int,
    height: int,
    dt_us_list: list[int] | None = None,
    max_windows_per_dt: int = 30,
    style: str = "paper",
) -> float:
    """Compute AOCC (Area of Continuous Contrast Curve) from an event stream.

    Parameters
    - x,y,t_us: 1D arrays of the kept/denoised events (polarity is ignored).
      t_us is expected to be in microseconds.
    - width,height: sensor resolution.
        - dt_us_list: time window lengths (microseconds) used to sample the CCC.
            If None, defaults are chosen by `style`.
        - max_windows_per_dt: cap the number of temporal windows sampled per dt.
      This keeps runtime bounded for long sequences (AOCC is typically used as a
      reference metric, not to score every sweep point).
        - style:
                - "paper": follow the official AOCC repo defaults more closely:
                    * interval axis is in microseconds (so AOCC is in contrast*us)
                    * binary frames use value 255 (0/255), and apply Gaussian blur (5, sigma=2)
                    * skip empty windows when averaging contrast (like `np.any(frame)` gating)
                    * default dt list: 2000..200000 step 2000 (us)
                - "normalized": a cheaper, more unit-stable variant:
                    * interval axis is in seconds (so AOCC is in contrast*s)
                    * binary frames use value 1 (0/1), no blur
                    * empty windows contribute 0 contrast
                    * default dt list: [2,4,8,...,400] ms (log-ish)

    Definition (discrete approximation)
    - For each dt: split the event stream into consecutive windows of length dt,
      build a binary event frame per window, compute contrast C(dt) as the mean
      Sobel-gradient-magnitude std over windows.
    - AOCC is the area under the C(dt) curve: integrate C w.r.t. dt.

    Returns
        - AOCC scalar in display units of 1e7 by default.
            (i.e. returns aocc_raw / MYEVS_AOCC_UNIT, default unit=1e7)
    """

    w = int(width)
    h = int(height)
    if w <= 0 or h <= 0:
        raise ValueError("width/height must be positive")

    xs = np.asarray(x).astype(np.int32, copy=False)
    ys = np.asarray(y).astype(np.int32, copy=False)
    ts = np.asarray(t_us).astype(np.int64, copy=False)

    if xs.ndim != 1 or ys.ndim != 1 or ts.ndim != 1:
        raise ValueError("x, y, t_us must be 1D arrays")
    if xs.shape[0] != ys.shape[0] or xs.shape[0] != ts.shape[0]:
        raise ValueError("x, y, t_us must have the same length")

    n = int(xs.shape[0])
    if n <= 0:
        return 0.0

    # Filter out-of-bounds (important if upstream transforms exist)
    ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    if not bool(np.all(ok)):
        xs = xs[ok]
        ys = ys[ok]
        ts = ts[ok]
        n = int(xs.shape[0])
        if n <= 0:
            return 0.0

    # Ensure time is non-decreasing for window slicing.
    if n >= 2 and bool(np.any(ts[1:] < ts[:-1])):
        order = np.argsort(ts, kind="mergesort")
        xs = xs[order]
        ys = ys[order]
        ts = ts[order]

    style_key = str(style or "").strip().lower()
    if style_key not in {"paper", "normalized"}:
        raise ValueError("style must be 'paper' or 'normalized'")

    if dt_us_list is None:
        if style_key == "paper":
            # Official code: arange(2000, 200001, 2000)
            dt_us_list = list(range(2000, 200001, 2000))
        else:
            dt_us_list = [2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 400000]

    dts = [int(v) for v in dt_us_list if int(v) > 0]
    if not dts:
        return 0.0
    dts = sorted(set(dts))

    t0 = int(ts[0])
    t1 = int(ts[-1])
    if t1 <= t0:
        return 0.0

    c_vals: list[float] = []
    x_axis: list[float] = []

    for dt_us in dts:
        dt = int(dt_us)
        n_windows = int((t1 - t0) // dt) + 1
        if n_windows <= 0:
            c_vals.append(0.0)
            if style_key == "paper":
                x_axis.append(float(dt))
            else:
                x_axis.append(float(dt) * 1e-6)
            continue

        # Map events to window ids.
        win_id = ((ts - t0) // dt).astype(np.int64, copy=False)

        # Select which windows to sample.
        m = int(max_windows_per_dt)
        # Allow disabling the cap for exact-but-slower matching.
        # - m <= 0: sample all windows
        if m <= 0:
            m = n_windows

        if n_windows <= m:
            sel = np.arange(n_windows, dtype=np.int64)
        else:
            sel = np.unique(np.linspace(0, n_windows - 1, num=m, dtype=np.int64))

        # For each selected window, build binary frame and compute contrast.
        # Use searchsorted to find the event range for a window id.
        # (win_id is non-decreasing because ts is sorted)
        contrasts: list[float] = []
        img = np.zeros((h, w), dtype=np.uint8)
        for wid in sel.tolist():
            i0 = int(np.searchsorted(win_id, wid, side="left"))
            i1 = int(np.searchsorted(win_id, wid, side="right"))
            if i1 <= i0:
                if style_key == "normalized":
                    contrasts.append(0.0)
                continue

            img.fill(0)
            if style_key == "paper":
                img[ys[i0:i1], xs[i0:i1]] = 255
                contrasts.append(_contrast_official_cv2(img))
            else:
                img[ys[i0:i1], xs[i0:i1]] = 1
                contrasts.append(_sobel_contrast_std(img))

        if style_key == "paper":
            # Official code skips empty windows (only appends frames if np.any(frame)).
            # Here: if a selected window had no events, we skipped it (no 0 padding).
            c_dt = float(np.mean(np.asarray(contrasts, dtype=np.float64))) if contrasts else 0.0
            c_vals.append(c_dt)
            x_axis.append(float(dt))
        else:
            c_dt = float(np.mean(np.asarray(contrasts, dtype=np.float64))) if contrasts else 0.0
            c_vals.append(c_dt)
            x_axis.append(float(dt) * 1e-6)

    if len(c_vals) < 2:
        return float(c_vals[0]) * float(x_axis[0])

    aocc_raw = float((getattr(np, "trapezoid", None) or np.trapz)(y=np.asarray(c_vals), x=np.asarray(x_axis)))
    if not np.isfinite(aocc_raw):
        return 0.0
    return _scale_aocc(aocc_raw)
