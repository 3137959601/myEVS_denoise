from __future__ import annotations

from pathlib import Path

import numpy as np


def convert_aedat4_to_npz(
    *,
    in_path: str | Path,
    out_path: str | Path,
    max_events: int = 0,
) -> Path:
    """Convert DVSNOISE20 AEDAT4 to npz using optional dv_processing.

    This function intentionally keeps dv_processing optional because the local
    project environment may not have the EDFormer/DV stack installed.
    """

    try:
        import dv_processing as dv  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Reading .aedat4 requires dv_processing. Install it in this environment, "
            "or convert the file on a server with dv_processing and copy the .npz back. "
            "Examples: python scripts/qualitative/convert_aedat4.py --in <file.aedat4> --out <file.npz>; "
            "or python scripts/qualitative/convert_dvsnoise20.py"
        ) from e

    inp = Path(in_path)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    reader = dv.io.MonoCameraRecording(str(inp))
    chunks = []
    total = 0
    while reader.isRunning():
        events = reader.getNextEventBatch()
        if events is None:
            continue
        arr = events.numpy()
        if arr is None or len(arr) == 0:
            continue
        chunks.append(arr)
        total += int(len(arr))
        if max_events > 0 and total >= int(max_events):
            break

    if not chunks:
        raise RuntimeError(f"No events decoded from {inp}")

    data = np.concatenate(chunks)
    if max_events > 0:
        data = data[: int(max_events)]

    names = data.dtype.names or ()
    t_name = "timestamp" if "timestamp" in names else "t"
    x_name = "x"
    y_name = "y"
    p_name = "polarity" if "polarity" in names else "p"
    t = np.asarray(data[t_name], dtype=np.uint64)
    x = np.asarray(data[x_name], dtype=np.uint16)
    y = np.asarray(data[y_name], dtype=np.uint16)
    p_raw = np.asarray(data[p_name])
    p = np.where(p_raw.astype(np.int8) > 0, 1, -1).astype(np.int8)
    if t.size:
        t = t - np.uint64(int(t[0]))
    np.savez_compressed(out, t=t, x=x, y=y, p=p)
    return out
