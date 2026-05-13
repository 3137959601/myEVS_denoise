from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import pandas as pd


def best_st_from_summary(path: Path) -> tuple[int, int]:
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f"empty summary: {path}")
    b = df.sort_values(["auc", "best_f1"], ascending=False).iloc[0]
    return int(b["s"]), int(b["tau_us"])


def run_one(py: str, noisy: str, out_dir: Path, dataset: str, scene: str, level: str, width: int, height: int, signal_label_value: int, s: int, tau_us: int, max_events: int, k_sfrac_list: str, k_mix_list: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        py,
        "scripts/noise_analyze/sweep_n179_pair.py",
        "--noisy", noisy,
        "--out-csv", str(out_dir / "roc_kretune.csv"),
        "--summary-csv", str(out_dir / "summary_kretune.csv"),
        "--runtime-csv", str(out_dir / "runtime_kretune.csv"),
        "--dataset", dataset,
        "--scene", scene,
        "--level", level,
        "--width", str(width),
        "--height", str(height),
        "--tick-ns", "1000",
        "--s-list", str(s),
        "--tau-us-list", str(tau_us),
        "--k-sfrac-list", k_sfrac_list,
        "--k-mix-list", k_mix_list,
        "--signal-label-value", str(signal_label_value),
        "--max-events", str(max_events),
        "--tag-prefix", "n179kfull",
    ]
    print("[RUN]", dataset, scene, level, f"s={s}", f"tau={tau_us}", flush=True)
    p = subprocess.run(cmd)
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def main() -> int:
    ap = argparse.ArgumentParser(description="Fullrun-consistent K retune for N179")
    ap.add_argument("--python", required=True)
    ap.add_argument("--dataset", choices=["DVSCLEAN", "LED"], required=True)
    ap.add_argument("--max-events", type=int, default=0)
    ap.add_argument("--k-sfrac-list", default="0.4,0.55,0.67,0.8,0.9")
    ap.add_argument("--k-mix-list", default="0.0,0.04,0.08,0.125,0.16,0.3,1.0")
    args = ap.parse_args()

    py = args.python

    if args.dataset == "DVSCLEAN":
        scenes = ["MAH00444", "MAH00446", "MAH00447", "MAH00448", "MAH00449"]
        levels = ["ratio50", "ratio100"]
        total = len(scenes) * len(levels)
        idx = 0
        for sc in scenes:
            for lv in levels:
                idx += 1
                noisy = f"D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy/{sc}/{lv}/{sc}_{lv}_labeled.npy"
                summary = Path(f"data/DVSCLEAN/scene_sweep_full/{sc}/{lv}/N179/summary_n179.csv")
                s, tau = best_st_from_summary(summary)
                out_dir = Path(f"data/DVSCLEAN/scene_sweep_full/{sc}/{lv}/N179/kretune_fullrun_20260506")
                print(f"[{idx}/{total}] DVSCLEAN {sc} {lv}", flush=True)
                # NOTE: current converted DVSCLEAN npy uses label=1 for signal.
                run_one(py, noisy, out_dir, "DVSCLEAN", sc, lv, 1280, 720, 1, s, tau, args.max_events, args.k_sfrac_list, args.k_mix_list)

    elif args.dataset == "LED":
        scenes = ["scene_100", "scene_1004", "scene_1018", "scene_1028", "scene_1032", "scene_1033", "scene_1034", "scene_1043", "scene_1045", "scene_1046"]
        total = len(scenes)
        for idx, sc in enumerate(scenes, start=1):
            noisy = f"D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/{sc}/slices_00031_00040_100ms/{sc}_100ms_labeled.npy"
            summary = Path(f"data/LED/scene_sweep_full/{sc}/N179/summary_n179.csv")
            s, tau = best_st_from_summary(summary)
            out_dir = Path(f"data/LED/scene_sweep_full/{sc}/N179/kretune_fullrun_20260506")
            print(f"[{idx}/{total}] LED {sc}", flush=True)
            run_one(py, noisy, out_dir, "LED", sc, "100ms", 1280, 720, 1, s, tau, args.max_events, args.k_sfrac_list, args.k_mix_list)

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
