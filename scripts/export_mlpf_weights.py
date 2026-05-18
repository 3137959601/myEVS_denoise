from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_state_dict(model_path: Path) -> dict:
    try:
        import torch
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyTorch is required to export MLPF weights from TorchScript") from e

    model = torch.jit.load(str(model_path), map_location="cpu")
    state = model.state_dict()
    return {k: v.detach().cpu().numpy() for k, v in state.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description="Export myEVS TorchScript MLPF weights to native C++ .npz format.")
    ap.add_argument("--model", required=True, help="TorchScript .pt model path produced by train_mlpf_torch.py")
    ap.add_argument("--out", default="", help="Output .npz path. Defaults to same stem as --model.")
    ap.add_argument("--meta", default="", help="Input metadata JSON. Defaults to same stem as --model.")
    ap.add_argument("--out-meta", default="", help="Output metadata JSON. Defaults to same stem as output .npz.")
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    out_path = Path(args.out) if args.out else model_path.with_suffix(".npz")
    meta_path = Path(args.meta) if args.meta else model_path.with_suffix(".json")
    out_meta_path = Path(args.out_meta) if args.out_meta else out_path.with_suffix(".json")

    state = _load_state_dict(model_path)
    required = ("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias")
    missing = [name for name in required if name not in state]
    if missing:
        raise ValueError(f"Unsupported MLPF model. Missing state_dict keys: {missing}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        fc1_weight=np.asarray(state["fc1.weight"], dtype=np.float32),
        fc1_bias=np.asarray(state["fc1.bias"], dtype=np.float32),
        fc2_weight=np.asarray(state["fc2.weight"], dtype=np.float32),
        fc2_bias=np.asarray(state["fc2.bias"], dtype=np.float32),
    )

    meta = {}
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            meta = loaded
    meta.update(
        {
            "weight_format": "myevs-mlpf-npz-v1",
            "source_model": str(model_path),
            "npz_path": str(out_path),
            "output_type": meta.get("output_type", "logit"),
        }
    )
    out_meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Exported native MLPF weights: {out_path}")
    print(f"Updated metadata: {out_meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
