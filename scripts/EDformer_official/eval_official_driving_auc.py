from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve


TIMESTAMP_COLUMN = 0
X_COLUMN = 1
Y_COLUMN = 2
POLARITY_COLUMN = 3
LABEL_COLUMN = 4


@dataclass(frozen=True)
class DatasetStats:
    hz: int
    path: Path
    events_total: int
    events_used: int
    chunks: int
    labels_zero: int
    labels_one: int
    t_min: float
    t_max: float
    duration_s_if_us: float


def _import_edformer(edformer_root: Path):
    sys.path.insert(0, str(edformer_root))
    try:
        import torch  # type: ignore
        from model import EDformer  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Cannot import official EDformer model. Install the EDformer dependencies "
            "(torch, sparseconvnet, pytorch3d) in the Python environment used to run this script."
        ) from exc
    return torch, EDformer


def check_environment(edformer_root: Path) -> dict[str, object]:
    out: dict[str, object] = {
        "python": sys.executable,
        "edformer_root": str(edformer_root),
        "imports": {},
    }
    mods = ["torch", "sklearn", "pandas", "numpy", "sparseconvnet", "pytorch3d", "dv_processing"]
    imports: dict[str, dict[str, str]] = {}
    for mod_name in mods:
        try:
            mod = __import__(mod_name)
            imports[mod_name] = {"status": "OK", "version": str(getattr(mod, "__version__", ""))}
        except Exception as exc:
            imports[mod_name] = {
                "status": "FAIL",
                "error": f"{type(exc).__name__}: {str(exc)}",
            }
    out["imports"] = imports
    try:
        _import_edformer(edformer_root)
        out["edformer_model_import"] = "OK"
    except Exception as exc:
        out["edformer_model_import"] = f"FAIL: {type(exc).__name__}: {exc}"
    return out


def parse_hz_list(text: str) -> list[int]:
    values: list[int] = []
    for part in str(text).split(","):
        part = part.strip().lower().replace("hz", "")
        if not part:
            continue
        values.append(int(part))
    return values


def read_official_txt(path: Path, *, max_events: int = 0) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(str(path))
    names = ["timestamp", "x", "y", "polarity", "label"]
    kwargs = {
        "skiprows": 1,
        "sep": r"\s+",
        "header": None,
        "names": names,
        "engine": "c",
        "dtype": {
            "timestamp": np.float64,
            "x": np.float32,
            "y": np.float32,
            "polarity": np.float32,
            "label": np.float32,
        },
    }
    if max_events and max_events > 0:
        kwargs["nrows"] = int(max_events)
    df = pd.read_csv(path, **kwargs)
    return df.to_numpy(dtype=np.float32, copy=False)


def normalize_timestamp_in_place(events: np.ndarray) -> tuple[float, float]:
    t = events[:, TIMESTAMP_COLUMN].astype(np.float64, copy=False)
    t_min = float(np.min(t))
    t_max = float(np.max(t))
    denom = t_max - t_min
    if denom <= 0.0:
        events[:, TIMESTAMP_COLUMN] = 0.0
    else:
        events[:, TIMESTAMP_COLUMN] = ((t - t_min) / denom).astype(np.float32)
    return t_min, t_max


def apply_xy_mode_in_place(events: np.ndarray, mode: str, *, width: int, height: int) -> None:
    if mode == "official":
        return
    if mode == "unit":
        events[:, X_COLUMN] = events[:, X_COLUMN] / float(width)
        events[:, Y_COLUMN] = events[:, Y_COLUMN] / float(height)
        return
    raise ValueError(f"unknown xy mode: {mode}")


def iter_chunks(events: np.ndarray, *, seq_len: int) -> Iterable[np.ndarray]:
    chunks = events.shape[0] // int(seq_len)
    usable = chunks * int(seq_len)
    for start in range(0, usable, int(seq_len)):
        yield events[start : start + int(seq_len), :4]


def _load_state_dict(torch, model_path: Path, device):
    try:
        return torch.load(str(model_path), map_location=device, weights_only=True)
    except TypeError:
        return torch.load(str(model_path), map_location=device)
    except Exception:
        return torch.load(str(model_path), map_location=device)


def evaluate_one(
    *,
    torch,
    model,
    path: Path,
    hz: int,
    seq_len: int,
    device,
    max_events: int,
    xy_mode: str,
    width: int,
    height: int,
) -> tuple[DatasetStats, dict[str, float]]:
    raw = read_official_txt(path, max_events=max_events)
    events_total = int(raw.shape[0])
    chunks = events_total // int(seq_len)
    used = chunks * int(seq_len)
    if chunks <= 0:
        raise ValueError(f"{path} has fewer than seq_len={seq_len} events")

    labels = raw[:used, LABEL_COLUMN].astype(np.int32, copy=True)
    labels_zero = int(np.count_nonzero(labels == 0))
    labels_one = int(np.count_nonzero(labels == 1))

    events = raw[:used, :4].copy()
    t_min, t_max = normalize_timestamp_in_place(events)
    apply_xy_mode_in_place(events, xy_mode, width=width, height=height)

    scores: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for chunk in iter_chunks(events, seq_len=seq_len):
            x = torch.as_tensor(chunk, dtype=torch.float32, device=device).reshape(1, seq_len, 4)
            logits = model(x)
            prob = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
            scores.append(prob)
    score = np.concatenate(scores, axis=0)

    fpr, tpr, _ = roc_curve(labels, score)
    auc_official = float(auc(fpr, tpr))

    # Diagnostic only: if a local conversion flipped label semantics, this value becomes the one near the paper.
    fpr_inv, tpr_inv, _ = roc_curve(1 - labels, score)
    auc_label_inverted = float(auc(fpr_inv, tpr_inv))
    fpr_sig, tpr_sig, _ = roc_curve(1 - labels, 1.0 - score)
    auc_signal_kept_equivalent = float(auc(fpr_sig, tpr_sig))

    stats = DatasetStats(
        hz=hz,
        path=path,
        events_total=events_total,
        events_used=used,
        chunks=chunks,
        labels_zero=labels_zero,
        labels_one=labels_one,
        t_min=t_min,
        t_max=t_max,
        duration_s_if_us=(t_max - t_min) / 1_000_000.0 if t_max > t_min else 0.0,
    )
    metrics = {
        "auc_official_label1_positive": auc_official,
        "auc_label_inverted_diagnostic": auc_label_inverted,
        "auc_signal_kept_equivalent": auc_signal_kept_equivalent,
        "score_mean": float(np.mean(score)),
        "score_min": float(np.min(score)),
        "score_max": float(np.max(score)),
    }
    return stats, metrics


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "hz",
        "path",
        "events_total",
        "events_used",
        "chunks",
        "labels_zero",
        "labels_one",
        "duration_s_if_us",
        "auc_official_label1_positive",
        "auc_label_inverted_diagnostic",
        "auc_signal_kept_equivalent",
        "score_mean",
        "score_min",
        "score_max",
        "xy_mode",
        "seq_len",
        "device",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run the official EDformer model on DND21 Driving driving_mix_result.txt and compute AUC."
    )
    ap.add_argument(
        "--edformer-root",
        default=r"D:\hjx_workspace\scientific_reserach\EDformer",
        help="Path to the official EDformer source directory containing model.py and pretrained_model.pth.",
    )
    ap.add_argument(
        "--base-path",
        default=r"D:\hjx_workspace\scientific_reserach\dataset\DND21\mydriving_ED24",
        help="Directory containing <hz>hz/driving_mix_result.txt.",
    )
    ap.add_argument("--model-path", default="", help="Defaults to <edformer-root>/pretrained_model.pth.")
    ap.add_argument("--hz", default="1,3,5,7,10", help="Comma-separated Hz list.")
    ap.add_argument("--filename", default="driving_mix_result.txt", help="Driving file name under each Hz directory.")
    ap.add_argument("--seq-len", type=int, default=4096, help="Official EDformer sequence length.")
    ap.add_argument("--max-events", type=int, default=0, help="Optional smoke-test cap per file.")
    ap.add_argument("--width", type=int, default=346)
    ap.add_argument("--height", type=int, default=260)
    ap.add_argument(
        "--xy-mode",
        choices=["official", "unit"],
        default="official",
        help="official: pass x/y exactly as EDformer release code does; unit: divide x/y by width/height as a diagnostic.",
    )
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, etc.")
    ap.add_argument("--skip-missing", action="store_true", help="Skip missing Hz files instead of failing.")
    ap.add_argument("--check-env", action="store_true", help="Only print dependency/import status.")
    ap.add_argument(
        "--out-csv",
        default="data/DND21/edformer_official_auc/driving_auc.csv",
        help="Output CSV path.",
    )
    ap.add_argument(
        "--out-json",
        default="data/DND21/edformer_official_auc/driving_auc_env.json",
        help="Environment/status JSON path.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    edformer_root = Path(args.edformer_root)
    env = check_environment(edformer_root)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(env, indent=2), encoding="utf-8")
    print(json.dumps(env, indent=2))
    if args.check_env:
        return 0

    torch, EDformer = _import_edformer(edformer_root)
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model_path = Path(args.model_path) if args.model_path else edformer_root / "pretrained_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(str(model_path))

    model = EDformer().to(device)
    state = _load_state_dict(torch, model_path, device)
    model.load_state_dict(state)
    model.eval()

    base = Path(args.base_path)
    rows: list[dict[str, object]] = []
    for hz in parse_hz_list(args.hz):
        path = base / f"{hz}hz" / str(args.filename)
        if not path.exists():
            msg = f"missing {hz}Hz input: {path}"
            if args.skip_missing:
                print(f"[skip] {msg}")
                continue
            raise FileNotFoundError(msg)
        print(f"[eval] {hz}Hz {path}")
        stats, metrics = evaluate_one(
            torch=torch,
            model=model,
            path=path,
            hz=hz,
            seq_len=int(args.seq_len),
            device=device,
            max_events=int(args.max_events),
            xy_mode=str(args.xy_mode),
            width=int(args.width),
            height=int(args.height),
        )
        row = {
            "hz": stats.hz,
            "path": str(stats.path),
            "events_total": stats.events_total,
            "events_used": stats.events_used,
            "chunks": stats.chunks,
            "labels_zero": stats.labels_zero,
            "labels_one": stats.labels_one,
            "duration_s_if_us": stats.duration_s_if_us,
            **metrics,
            "xy_mode": args.xy_mode,
            "seq_len": args.seq_len,
            "device": str(device),
        }
        rows.append(row)
        print(
            f"  auc={metrics['auc_official_label1_positive']:.6f} "
            f"auc_inv={metrics['auc_label_inverted_diagnostic']:.6f} "
            f"used={stats.events_used} labels(0/1)={stats.labels_zero}/{stats.labels_one}"
        )

    write_csv(Path(args.out_csv), rows)
    print(f"[done] wrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
