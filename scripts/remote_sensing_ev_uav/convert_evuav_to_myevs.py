from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from evuav_common import (
    DATASET_OUT_ROOT,
    SELECTED_TEST_SEQUENCES,
    ensure_dirs,
    evuav_to_reference,
    load_evuav_npz,
    parse_sequence,
    sequence_reference_path,
    write_json,
)


def convert_one(raw: str, overwrite: bool) -> Path:
    seq = parse_sequence(raw)
    out = sequence_reference_path(seq)
    meta_path = DATASET_OUT_ROOT / "meta" / f"{seq.stem}_reference_meta.json"
    if out.exists() and not overwrite:
        return out
    ev = load_evuav_npz(seq.path)
    ref = evuav_to_reference(ev)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, ref, allow_pickle=False)
    meta = {
        "source_npz": str(seq.path),
        "output_npy": str(out),
        "sequence": seq.stem,
        "split": seq.split,
        "width": 346,
        "height": 260,
        "time_unit": "us",
        "events": int(ref.shape[0]),
        "target_events": int(np.sum(ref["target"] == 1)),
        "background_events": int(np.sum(ref["target"] == 0)),
        "label_convention": "label=1 original reference, label=0 injected noise",
        "target_convention": "target=1 EV-UAV target event, target=0 background or injected noise",
        "source_convention": "1=target_ref, 2=background_ref, 3=shot_noise",
    }
    write_json(meta_path, meta)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert EV-UAV NPZ files to myEVS reference structured NPY.")
    ap.add_argument("--sequences", nargs="*", default=list(SELECTED_TEST_SEQUENCES))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    ensure_dirs()
    for s in args.sequences:
        out = convert_one(s, overwrite=bool(args.overwrite))
        print(f"[ok] {s} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

