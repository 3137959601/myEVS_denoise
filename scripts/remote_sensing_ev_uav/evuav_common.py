from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


WIDTH = 346
HEIGHT = 260
TIME_UNIT = "us"

EVUAV_ROOT = Path(r"D:\hjx_workspace\scientific_reserach\dataset\EV-UAV-dataset")
DATASET_OUT_ROOT = EVUAV_ROOT / "myevs_noise_eval"
RESULT_ROOT = ROOT / "data" / "remote_sensing_ev_uav_noise"

SELECTED_TEST_SEQUENCES = (
    "test_005",
    "test_009",
    "test_014",
    "test_019",
    "test_020",
    "test_021",
)

SHOT_LEVELS_RATIO = (0.25, 0.5, 1.0, 2.0)

DTYPE_REF = np.dtype(
    [
        ("t", "<u8"),
        ("x", "<u2"),
        ("y", "<u2"),
        ("p", "i1"),
        ("label", "u1"),  # 1=original reference, 0=injected noise
        ("target", "u1"),  # 1=EV-UAV target event, 0=background/noise
        ("source", "u1"),  # 1=target_ref, 2=background_ref, 3=shot_noise
    ]
)


@dataclass(frozen=True)
class SequenceId:
    split: str
    stem: str

    @property
    def path(self) -> Path:
        return EVUAV_ROOT / self.split / f"{self.stem}.npz"


def ensure_dirs() -> None:
    for p in (
        RESULT_ROOT,
        RESULT_ROOT / "metrics",
        RESULT_ROOT / "figures",
        RESULT_ROOT / "logs",
        DATASET_OUT_ROOT / "converted",
        DATASET_OUT_ROOT / "noisy",
        DATASET_OUT_ROOT / "meta",
        DATASET_OUT_ROOT / "preview",
    ):
        p.mkdir(parents=True, exist_ok=True)


def parse_sequence(raw: str) -> SequenceId:
    s = str(raw).strip()
    if not s:
        raise ValueError("empty sequence id")
    split = s.split("_", 1)[0]
    if split not in {"train", "val", "test"}:
        raise ValueError(f"sequence id must start with train_, val_, or test_: {raw}")
    return SequenceId(split=split, stem=s)


def iter_npz_files(root: Path = EVUAV_ROOT) -> Iterable[tuple[str, Path]]:
    for split in ("train", "val", "test"):
        for p in sorted((root / split).glob("*.npz")):
            yield split, p


def load_evuav_npz(path: Path) -> np.ndarray:
    z = np.load(path, allow_pickle=True)
    if "ev" not in z.files:
        raise ValueError(f"missing key 'ev' in {path}")
    ev = z["ev"]
    required = {"x", "y", "t", "p", "label"}
    names = set(ev.dtype.names or ())
    missing = required - names
    if missing:
        raise ValueError(f"{path} missing EV-UAV fields: {sorted(missing)}")
    return ev


def evuav_to_reference(ev: np.ndarray) -> np.ndarray:
    out = np.zeros((ev.shape[0],), dtype=DTYPE_REF)
    # EV-UAV timestamps are stored in ms in the downloaded NPZ files.
    out["t"] = np.round(np.asarray(ev["t"], dtype=np.float64) * 1000.0).astype(np.uint64)
    out["x"] = np.asarray(ev["x"], dtype=np.uint16)
    out["y"] = np.asarray(ev["y"], dtype=np.uint16)
    p01 = np.asarray(ev["p"], dtype=np.int8)
    out["p"] = np.where(p01 > 0, 1, -1).astype(np.int8)
    target = (np.asarray(ev["label"], dtype=np.uint8) > 0).astype(np.uint8)
    out["label"] = 1
    out["target"] = target
    out["source"] = np.where(target > 0, 1, 2).astype(np.uint8)
    order = np.argsort(out["t"], kind="stable")
    return out[order]


def load_reference(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def sequence_reference_path(seq: SequenceId) -> Path:
    return DATASET_OUT_ROOT / "converted" / f"{seq.stem}_reference.npy"


def noisy_path(seq: SequenceId, level: float, *, mode: str = "ratio") -> Path:
    if mode == "hz":
        return DATASET_OUT_ROOT / "noisy" / f"{seq.stem}_shot_{format_float(level)}hz_labeled.npy"
    return DATASET_OUT_ROOT / "noisy" / f"{seq.stem}_shot_r{format_float(level)}_labeled.npy"


def noisy_meta_path(seq: SequenceId, level: float, *, mode: str = "ratio") -> Path:
    if mode == "hz":
        return DATASET_OUT_ROOT / "meta" / f"{seq.stem}_shot_{format_float(level)}hz_meta.json"
    return DATASET_OUT_ROOT / "meta" / f"{seq.stem}_shot_r{format_float(level)}_meta.json"


def format_float(v: float) -> str:
    return ("%g" % float(v)).replace(".", "p")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for r in rows:
            for k in r.keys():
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def rank_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.uint8)
    scores = np.asarray(scores, dtype=np.float64)
    pos = labels == 1
    n_pos = int(pos.sum())
    n_neg = int(labels.shape[0] - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.shape[0] + 1, dtype=np.float64)
    # Average ranks for ties.
    vals = scores[order]
    i = 0
    while i < vals.shape[0]:
        j = i + 1
        while j < vals.shape[0] and vals[j] == vals[i]:
            j += 1
        if j - i > 1:
            avg = 0.5 * (i + 1 + j)
            ranks[order[i:j]] = avg
        i = j
    rank_sum_pos = float(ranks[pos].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)


def metrics_from_keep(keep: np.ndarray, labels: np.ndarray, target: np.ndarray) -> dict[str, float]:
    keep = np.asarray(keep, dtype=bool)
    labels = np.asarray(labels, dtype=np.uint8)
    target = np.asarray(target, dtype=np.uint8)
    ref = labels == 1
    noise = labels == 0
    tgt = target == 1

    tp = int(np.sum(keep & ref))
    fp = int(np.sum(keep & noise))
    fn = int(np.sum((~keep) & ref))
    tn = int(np.sum((~keep) & noise))
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    terr = safe_div(int(np.sum(keep & tgt)), int(np.sum(tgt)))
    rerr = safe_div(tp, int(np.sum(ref)))
    inrr = safe_div(tn, int(np.sum(noise)))
    return {
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "precision": precision,
        "recall_ref": recall,
        "f1_ref_noise": f1,
        "terr": terr,
        "rerr": rerr,
        "inrr": inrr,
    }


def metrics_from_target_keep(keep: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Metrics for EV-UAV target/background event selection.

    Positive events are EV-UAV target events. Negative events are original
    background events, treated as task-dependent background interference.
    """
    keep = np.asarray(keep, dtype=bool)
    target = np.asarray(target, dtype=np.uint8)
    tgt = target == 1
    bg = ~tgt

    tp = int(np.sum(keep & tgt))
    fp = int(np.sum(keep & bg))
    fn = int(np.sum((~keep) & tgt))
    tn = int(np.sum((~keep) & bg))
    target_total = int(np.sum(tgt))
    bg_total = int(np.sum(bg))
    precision = safe_div(tp, tp + fp)
    trr = safe_div(tp, target_total)
    bsr = safe_div(tn, bg_total)
    bkr = safe_div(fp, bg_total)
    f1 = safe_div(2.0 * precision * trr, precision + trr)
    before_ratio = safe_div(target_total, bg_total)
    after_ratio = safe_div(tp, fp)
    tbr_gain = safe_div(after_ratio, before_ratio) if before_ratio > 0 else 0.0
    return {
        "target_tp": float(tp),
        "background_fp": float(fp),
        "background_tn": float(tn),
        "target_fn": float(fn),
        "target_precision": precision,
        "trr": trr,
        "bsr": bsr,
        "background_keep_rate": bkr,
        "f1_target_bg": f1,
        "target_background_ratio_gain": tbr_gain,
    }
