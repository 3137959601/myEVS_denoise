from __future__ import annotations

import argparse
import csv
import re
import subprocess
from pathlib import Path

import pandas as pd


def _build_jobs() -> list[dict[str, str]]:
    jobs: list[dict[str, str]] = []

    # ED24 pedestrian
    ped_root = Path("D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06")
    for lv, suf in [("light", "1.8"), ("light_mid", "2.1"), ("mid", "2.5"), ("heavy", "3.3")]:
        jobs.append(
            {
                "dataset": "ED24",
                "scene": "myPedestrain_06",
                "level": lv,
                "noisy": str(ped_root / f"Pedestrain_06_{suf}.npy"),
                "out_dir": str(Path("data/ED24/myPedestrain_06/N179")),
            }
        )

    # ED24 bicycle
    bic_root = Path("D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02")
    for lv, suf in [("light", "1.8"), ("light_mid", "2.1"), ("mid", "2.5")]:
        jobs.append(
            {
                "dataset": "ED24",
                "scene": "myBicycle_02",
                "level": lv,
                "noisy": str(bic_root / f"Bicycle_02_{suf}.npy"),
                "out_dir": str(Path("data/ED24/myBicycle_02/N179")),
            }
        )

    # Driving
    drv_root = Path("D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving")
    jobs.extend(
        [
            {
                "dataset": "DND21",
                "scene": "mydriving",
                "level": "light",
                "noisy": str(drv_root / "driving_noise_light_slomo_shot_withlabel/driving_noise_light_labeled.npy"),
                "out_dir": str(Path("data/DND21/mydriving/N179")),
            },
            {
                "dataset": "DND21",
                "scene": "mydriving",
                "level": "light_mid",
                "noisy": str(drv_root / "driving_noise_light_mid_slomo_shot_withlabel/driving_noise_light_mid_labeled.npy"),
                "out_dir": str(Path("data/DND21/mydriving/N179")),
            },
            {
                "dataset": "DND21",
                "scene": "mydriving",
                "level": "mid",
                "noisy": str(drv_root / "driving_noise_mid_slomo_shot_withlabel/driving_noise_mid_labeled.npy"),
                "out_dir": str(Path("data/DND21/mydriving/N179")),
            },
        ]
    )

    # DVSCLEAN 10 subsets
    dv_root = Path("D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy")
    for scene in ["MAH00444", "MAH00446", "MAH00447", "MAH00448", "MAH00449"]:
        for lv in ["ratio50", "ratio100"]:
            jobs.append(
                {
                    "dataset": "DVSCLEAN",
                    "scene": scene,
                    "level": lv,
                    "noisy": str(dv_root / scene / lv / f"{scene}_{lv}_labeled.npy"),
                    "out_dir": str(Path(f"data/DVSCLEAN/scene_sweep_full/{scene}_{lv}/N179")),
                }
            )

    # LED 10 scenes
    led_root = Path("D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy")
    for scene in [
        "scene_100",
        "scene_1004",
        "scene_1018",
        "scene_1028",
        "scene_1032",
        "scene_1033",
        "scene_1034",
        "scene_1043",
        "scene_1045",
        "scene_1046",
    ]:
        jobs.append(
            {
                "dataset": "LED",
                "scene": scene,
                "level": "full100ms",
                "noisy": str(led_root / scene / "slices_00031_00040_100ms" / f"{scene}_100ms_labeled.npy"),
                "out_dir": str(Path(f"data/LED/scene_sweep_full/{scene}/N179")),
            }
        )
    return jobs


def _run_one(
    python_exe: str,
    job: dict[str, str],
    s_list: str,
    tau_list: str,
    k_sfrac_list: str,
    k_mix_list: str,
    beta_list: str,
    max_events: int,
) -> Path:
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    signal_label_value = 1
    if str(job.get("dataset", "")).upper() == "DVSCLEAN":
        signal_label_value = 0
    stem = f"{job['level']}_kgrid_official"
    run_out_dir = out_dir / stem
    run_out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_exe,
        "scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py",
        "--variant",
        "n179",
        "--light",
        job["noisy"],
        "--mid",
        job["noisy"],
        "--heavy",
        job["noisy"],
        "--out-dir",
        str(run_out_dir),
        "--s-list",
        s_list,
        "--tau-us-list",
        tau_list,
        "--n179-k-sfrac-list",
        k_sfrac_list,
        "--n179-k-mix-list",
        k_mix_list,
        "--max-events",
        str(max_events),
        "--esr-mode",
        "off",
        "--aocc-mode",
        "off",
        "--signal-label-value",
        str(signal_label_value),
    ]
    if beta_list.strip():
        cmd += ["--n179-beta-init-list", beta_list]
    subprocess.run(cmd, check=True)
    light_csvs = sorted(run_out_dir.glob("roc_ebf_n179_light_labelscore_*.csv"))
    if not light_csvs:
        raise RuntimeError(f"no light ROC CSV produced in {run_out_dir}")
    return light_csvs[-1]


_TAG_RE = re.compile(
    r"_s(?P<s>\d+)_tau(?P<tau>\d+)_ks(?P<ks>[A-Za-z0-9pm]+)_km(?P<km>[A-Za-z0-9pm]+)(?:_b(?P<b>[A-Za-z0-9pm]+))?$"
)


def _tag_num(token: str | None) -> float | None:
    if token is None or token == "":
        return None
    txt = token.replace("m", "-").replace("p", ".")
    return float(txt)


def _best_from_roc_csv(roc_csv: Path) -> dict[str, object]:
    df = pd.read_csv(roc_csv)
    if df.empty:
        raise RuntimeError(f"empty ROC csv: {roc_csv}")

    rows: list[dict[str, object]] = []
    for tag, g in df.groupby("tag", sort=False):
        auc = float(pd.to_numeric(g["auc"], errors="coerce").dropna().iloc[0])
        g2 = g.copy()
        g2["f1_num"] = pd.to_numeric(g2["f1"], errors="coerce").fillna(-1.0)
        best = g2.sort_values(["f1_num", "tpr", "precision", "fpr"], ascending=[False, False, False, True]).iloc[0]
        m = _TAG_RE.search(str(tag))
        if not m:
            continue
        rows.append(
            {
                "tag": str(tag),
                "s": int(m.group("s")),
                "radius_px": int((int(m.group("s")) - 1) // 2),
                "tau_us": int(m.group("tau")),
                "k_sfrac": float(_tag_num(m.group("ks"))),
                "k_mix": float(_tag_num(m.group("km"))),
                "beta_init": _tag_num(m.group("b")),
                "auc": float(auc),
                "best_f1": float(best["f1_num"]),
                "best_f1_threshold": float(best["value"]),
                "best_f1_tpr": float(pd.to_numeric(best["tpr"], errors="coerce")),
                "best_f1_fpr": float(pd.to_numeric(best["fpr"], errors="coerce")),
                "best_f1_precision": float(pd.to_numeric(best["precision"], errors="coerce")),
                "best_f1_mesr": "",
                "best_f1_aocc": "",
            }
        )
    if not rows:
        raise RuntimeError(f"no parsable tags in ROC csv: {roc_csv}")
    dfb = pd.DataFrame(rows).sort_values(["auc", "best_f1"], ascending=[False, False])
    return dfb.iloc[0].to_dict()


def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep N179 K values on every subset of all datasets.")
    ap.add_argument("--python-exe", default="D:/software/Anaconda_envs/envs/myEVS/python.exe")
    ap.add_argument("--s-list", default="9")
    ap.add_argument("--tau-us-list", default="128000")
    ap.add_argument("--k-sfrac-list", default="0.67,0.75,0.80,0.90")
    ap.add_argument("--k-mix-list", default="0.08,0.10,0.125,0.16")
    ap.add_argument("--beta-init-list", default="")
    ap.add_argument("--max-events", type=int, default=0)
    ap.add_argument("--out-csv", default="data/summary/n179_k_sweep_all_subsets.csv")
    args = ap.parse_args()

    jobs = _build_jobs()
    best_rows: list[dict[str, object]] = []
    total = len(jobs)
    for i, job in enumerate(jobs, 1):
        print(f"[{i}/{total}] {job['dataset']} {job['scene']} {job['level']}")
        roc_csv = _run_one(
            python_exe=str(args.python_exe),
            job=job,
            s_list=str(args.s_list),
            tau_list=str(args.tau_us_list),
            k_sfrac_list=str(args.k_sfrac_list),
            k_mix_list=str(args.k_mix_list),
            beta_list=str(args.beta_init_list),
            max_events=int(args.max_events),
        )
        row = _best_from_roc_csv(roc_csv)
        row["source_summary_csv"] = str(roc_csv)
        row["dataset"] = job["dataset"]
        row["scene"] = job["scene"]
        row["level"] = job["level"]
        best_rows.append(row)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if best_rows:
        cols = [
            "dataset",
            "scene",
            "level",
            "tag",
            "s",
            "radius_px",
            "tau_us",
            "k_sfrac",
            "k_mix",
            "beta_init",
            "auc",
            "best_f1",
            "best_f1_threshold",
            "best_f1_tpr",
            "best_f1_fpr",
            "best_f1_precision",
            "best_f1_mesr",
            "best_f1_aocc",
            "source_summary_csv",
        ]
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in best_rows:
                w.writerow({k: r.get(k, "") for k in cols})
    print(f"saved: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
    signal_label_value = 1
    if str(job.get("dataset", "")).upper() == "DVSCLEAN":
        signal_label_value = 0
