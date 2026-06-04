from __future__ import annotations

import shutil
from pathlib import Path


def import_external_result(
    *,
    source: str | Path,
    case_id: str,
    method: str,
    output_root: str | Path,
) -> Path:
    src = Path(source)
    if not src.exists():
        raise FileNotFoundError(src)
    dst_dir = Path(output_root) / "external" / case_id
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{method}{src.suffix.lower()}"
    shutil.copy2(src, dst)
    return dst


def expected_external_manifest(*, output_root: str | Path, case_ids: list[str], methods: list[str]) -> Path:
    import csv

    out = Path(output_root) / "external_expected_manifest.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "method", "expected_path", "status"])
        for case_id in case_ids:
            for method in methods:
                expected = Path(output_root) / "external" / case_id / f"{method}.npz"
                w.writerow([case_id, method, str(expected).replace("\\", "/"), "missing"])
    return out
