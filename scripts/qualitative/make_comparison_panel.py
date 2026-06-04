from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from myevs.qualitative.config import DEFAULT_ALGORITHM_PARAMS, DEFAULT_CASES_CONFIG, DEFAULT_QUALITATIVE_DIR, load_algorithm_params, load_cases_config
from myevs.qualitative.layout import make_dataset_method_panel


def main() -> int:
    p = argparse.ArgumentParser(description="Create a dataset-row/method-column qualitative comparison panel.")
    p.add_argument("--case-id", action="append", required=True, help="Case id row; repeat in desired row order.")
    p.add_argument("--case-label", action="append", default=None, help="Optional row label; repeat to match --case-id.")
    p.add_argument("--methods", nargs="*", default=None)
    p.add_argument("--cases", default=str(DEFAULT_CASES_CONFIG))
    p.add_argument("--params", default=str(DEFAULT_ALGORITHM_PARAMS))
    p.add_argument("--rendered-root", default=str(DEFAULT_QUALITATIVE_DIR / "rendered"))
    p.add_argument("--out", default=str(DEFAULT_QUALITATIVE_DIR / "panels" / "qualitative_comparison.png"))
    args = p.parse_args()

    params = load_algorithm_params(args.params)
    methods = args.methods or params.get("method_order", ["Noisy", "Ours", "BAF", "STCF", "EBF", "TS", "PFD", "EDnCNN", "EDFormer"])
    labels = args.case_label
    if labels is None:
        cfg = load_cases_config(args.cases)
        labels = [str(cfg.get("cases", {}).get(case_id, {}).get("label", case_id)) for case_id in args.case_id]
    out = make_dataset_method_panel(
        rendered_root=args.rendered_root,
        case_ids=args.case_id,
        method_order=methods,
        case_labels=labels,
        out_path=args.out,
    )
    print(f"comparison panel: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
