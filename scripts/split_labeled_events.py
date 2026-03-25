"""Split labeled event arrays into signal/noise event files.

Works with:
- .npy produced by scripts/v2e_labeled_txt_to_npy.py (structured dtype with fields t,x,y,p,label)
- .npz produced by scripts/v2e_labeled_txt_to_npy.py (keys t,x,y,p,label)

Outputs:
- signal-only .npy with fields t,x,y,p
- optional noise-only .npy

These outputs can be fed into myevs via --assume npy.

为什么需要分离出干净的事件：

这是在做 membership 查询：对输出里的每个事件，判断它是否属于 signal 集合（clean），
从而把 kept 进一步分成 signal_kept 和 noise_kept，再得到 TPR/FPR。

理论上也可以不生成 clean 文件，
直接在评估时用 label 统计 TP/FP——但 myevs 目前的 ROC 实现是基于 clean/noisy 两流的 membership，
所以我用“split clean”是为了复用现有 myevs roc，并且彻底绕开 match-us。

"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


DTYPE_4 = np.dtype(
    [
        ("t", np.uint64),
        ("x", np.uint16),
        ("y", np.uint16),
        ("p", np.int8),
    ]
)


def _load(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    suf = path.suffix.lower()
    if suf == ".npz":
        z = np.load(str(path), mmap_mode="r")
        return z["t"], z["x"], z["y"], z["p"], z["label"]

    if suf == ".npy":
        arr = np.load(str(path), mmap_mode="r")
        names = getattr(arr.dtype, "names", None)
        if names is not None:
            for k in ("t", "x", "y", "p", "label"):
                if k not in names:
                    raise SystemExit(f"Missing field {k!r} in {path}")
            return arr["t"], arr["x"], arr["y"], arr["p"], arr["label"]

        if arr.ndim == 2 and arr.shape[1] >= 5:
            return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]

        raise SystemExit(f"Unsupported npy layout: shape={getattr(arr,'shape',None)} dtype={arr.dtype}")

    raise SystemExit(f"Unsupported input suffix: {path.suffix}")


def _write_npy(path: Path, t, x, y, p, mask: np.ndarray) -> int:
    n = int(np.count_nonzero(mask))
    path.parent.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(str(path), mode="w+", dtype=DTYPE_4, shape=(n,))
    if n:
        out["t"] = np.asarray(t[mask], dtype=np.uint64)
        out["x"] = np.asarray(x[mask], dtype=np.uint16)
        out["y"] = np.asarray(y[mask], dtype=np.uint16)
        out["p"] = np.asarray(p[mask], dtype=np.int8)
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="input labeled .npy/.npz")
    ap.add_argument("--out-signal", required=True, help="output signal-only .npy")
    ap.add_argument("--out-noise", default=None, help="optional output noise-only .npy")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_sig = Path(args.out_signal)
    out_noise = Path(args.out_noise) if args.out_noise else None

    for p in [out_sig] + ([out_noise] if out_noise else []):
        if p is None:
            continue
        if p.exists() and not args.overwrite:
            raise SystemExit(f"Output exists: {p} (use --overwrite)")

    t, x, y, p, label = _load(in_path)
    label01 = (np.asarray(label) > 0)

    n_sig = _write_npy(out_sig, t, x, y, p, label01)
    print(f"signal events: {n_sig} -> {out_sig}")

    if out_noise is not None:
        n_noise = _write_npy(out_noise, t, x, y, p, ~label01)
        print(f"noise events: {n_noise} -> {out_noise}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
