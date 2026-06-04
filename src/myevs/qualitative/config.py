from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_QUALITATIVE_DIR = PROJECT_ROOT / "data" / "qualitative"
DEFAULT_CASES_CONFIG = DEFAULT_QUALITATIVE_DIR / "qualitative_cases.yaml"
DEFAULT_ALGORITHM_PARAMS = DEFAULT_QUALITATIVE_DIR / "algorithm_params.yaml"


def load_mapping(path: str | Path) -> dict[str, Any]:
    """Load a small config file.

    The default files are JSON-formatted YAML subsets so the project does not
    need a PyYAML dependency. If PyYAML is installed, normal YAML also works.
    """

    p = Path(path)
    text = p.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except Exception:
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)

    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {p}")
    return data


def save_json_yaml(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_cases_config(path: str | Path = DEFAULT_CASES_CONFIG) -> dict[str, Any]:
    return load_mapping(path)


def load_algorithm_params(path: str | Path = DEFAULT_ALGORITHM_PARAMS) -> dict[str, Any]:
    return load_mapping(path)


def resolve_project_path(value: str | Path) -> Path:
    p = Path(str(value))
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def ensure_default_configs(
    cases_path: str | Path = DEFAULT_CASES_CONFIG,
    params_path: str | Path = DEFAULT_ALGORITHM_PARAMS,
    *,
    overwrite: bool = False,
) -> None:
    cases_p = Path(cases_path)
    params_p = Path(params_path)
    if overwrite or not cases_p.exists():
        save_json_yaml(cases_p, default_cases_config())
    if overwrite or not params_p.exists():
        save_json_yaml(params_p, default_algorithm_params())


def default_cases_config() -> dict[str, Any]:
    dataset_root = "D:/hjx_workspace/scientific_reserach/dataset"
    return {
        "version": 1,
        "output_root": str(DEFAULT_QUALITATIVE_DIR).replace("\\", "/"),
        "render": {
            "scheme": 0,
            "raw_step": 127,
            "deadzone": 0,
            "binary": True,
            "show_on": True,
            "show_off": True,
            "background": "white",
        },
        "cases": {
            "driving_5hz": {
                "enabled": True,
                "group": "driving",
                "label": "Driving 5Hz",
                "qualitative_only": False,
                "width": 346,
                "height": 260,
                "tick_ns": 1000,
                "assume": "npy",
                "noisy": f"{dataset_root}/DND21/mydriving_ED24/driving_noise_5hz_ed24_withlabel/driving_noise_5hz_labeled.npy",
                "clean": f"{dataset_root}/DND21/mydriving_ED24/driving_noise_5hz_ed24_withlabel/driving_noise_5hz_signal_only.npy",
                "scan_windows_ms": [30, 50, 80, 100],
                "selected_window": {"start_us": 1015292, "window_ms": 50},
                "preferred_window_ms": 50,
                "exclude_levels_note": "2hz and 8hz are intentionally excluded.",
            },
            "driving_3hz": {
                "enabled": False,
                "group": "driving",
                "label": "Driving 3Hz backup",
                "qualitative_only": False,
                "width": 346,
                "height": 260,
                "tick_ns": 1000,
                "assume": "npy",
                "noisy": f"{dataset_root}/DND21/mydriving_ED24/driving_noise_3hz_ed24_withlabel/driving_noise_3hz_labeled.npy",
                "clean": f"{dataset_root}/DND21/mydriving_ED24/driving_noise_3hz_ed24_withlabel/driving_noise_3hz_signal_only.npy",
                "scan_windows_ms": [30, 50, 80, 100],
                "preferred_window_ms": 50,
            },
            "driving_7hz": {
                "enabled": False,
                "group": "driving",
                "label": "Driving 7Hz backup",
                "qualitative_only": False,
                "width": 346,
                "height": 260,
                "tick_ns": 1000,
                "assume": "npy",
                "noisy": f"{dataset_root}/DND21/mydriving_ED24/driving_noise_7hz_ed24_withlabel/driving_noise_7hz_labeled.npy",
                "clean": f"{dataset_root}/DND21/mydriving_ED24/driving_noise_7hz_ed24_withlabel/driving_noise_7hz_signal_only.npy",
                "scan_windows_ms": [30, 50, 80, 100],
                "preferred_window_ms": 50,
            },
            "driving_10hz": {
                "enabled": False,
                "group": "driving",
                "label": "Driving 10Hz backup",
                "qualitative_only": False,
                "width": 346,
                "height": 260,
                "tick_ns": 1000,
                "assume": "npy",
                "noisy": f"{dataset_root}/DND21/mydriving_ED24/driving_noise_10hz_ed24_withlabel/driving_noise_10hz_labeled.npy",
                "clean": f"{dataset_root}/DND21/mydriving_ED24/driving_noise_10hz_ed24_withlabel/driving_noise_10hz_signal_only.npy",
                "scan_windows_ms": [30, 50, 80, 100],
                "preferred_window_ms": 50,
            },
            "ped_3p3": {
                "enabled": True,
                "group": "ped",
                "label": "Pedestrian 3.3",
                "qualitative_only": False,
                "width": 346,
                "height": 260,
                "tick_ns": 1000,
                "assume": "npy",
                "noisy": f"{dataset_root}/ED24/myPedestrain_06/Pedestrain_06_3.3.npy",
                "clean": f"{dataset_root}/ED24/myPedestrain_06/Pedestrain_06_3.3_signal_only.npy",
                "scan_windows_ms": [50, 100, 150],
                "selected_window": {"start_us": 950000, "window_ms": 150},
                "preferred_window_ms": 150,
            },
            "bike_2p5": {
                "enabled": False,
                "group": "bike",
                "label": "Bicycle 2.5V",
                "qualitative_only": False,
                "width": 346,
                "height": 260,
                "tick_ns": 1000,
                "assume": "npy",
                "noisy": f"{dataset_root}/ED24/myBicycle_02/Bicycle_02_2.5.npy",
                "clean": f"{dataset_root}/ED24/myBicycle_02/Bicycle_02_2.5_signal_only.npy",
                "scan_windows_ms": [50, 100, 150],
                "selected_window": {"start_us": 97688, "window_ms": 100},
                "preferred_window_ms": 100,
            },
            "dvsclean_444_ratio100": {
                "enabled": True,
                "group": "dvsclean",
                "label": "DVSCLEAN 444",
                "qualitative_only": False,
                "width": 1280,
                "height": 720,
                "tick_ns": 1000,
                "assume": "npy",
                "noisy": f"{dataset_root}/DVSCLEAN/converted_npy/MAH00444/ratio100/MAH00444_ratio100_labeled.npy",
                "clean": f"{dataset_root}/DVSCLEAN/converted_npy/MAH00444/ratio100/MAH00444_ratio100_signal_only.npy",
                "scan_windows_ms": [50, 100, 150, 200],
                "selected_window": {"start_us": 0, "window_ms": 100},
                "preferred_window_ms": 100,
            },
            "stairs_125854": {
                "enabled": True,
                "group": "dvsnoise20",
                "label": "DVSNOISE20 stairs 12:58:54",
                "qualitative_only": True,
                "width": 346,
                "height": 260,
                "tick_ns": 1000,
                "assume": "npz",
                "raw_aedat4": f"{dataset_root}/DVSNOISE20/stairs-2019_10_10_12_58_54.aedat4",
                "noisy": str(DEFAULT_QUALITATIVE_DIR / "converted" / "dvsnoise20" / "stairs-2019_10_10_12_58_54.npz").replace("\\", "/"),
                "scan_windows_ms": [30, 50, 100],
                "preferred_window_ms": 50,
            },
            "stairs_130316": {
                "enabled": True,
                "group": "dvsnoise20",
                "label": "DVSNOISE20 stairs 13:03:16",
                "qualitative_only": True,
                "width": 346,
                "height": 260,
                "tick_ns": 1000,
                "assume": "npz",
                "raw_aedat4": f"{dataset_root}/DVSNOISE20/stairs-2019_10_10_13_03_16.aedat4",
                "noisy": str(DEFAULT_QUALITATIVE_DIR / "converted" / "dvsnoise20" / "stairs-2019_10_10_13_03_16.npz").replace("\\", "/"),
                "scan_windows_ms": [30, 50, 100],
                "preferred_window_ms": 50,
            },
            "stairs_130353": {
                "enabled": True,
                "group": "dvsnoise20",
                "label": "DVSNOISE20 stairs 13:03:53",
                "qualitative_only": True,
                "width": 346,
                "height": 260,
                "tick_ns": 1000,
                "assume": "npz",
                "raw_aedat4": f"{dataset_root}/DVSNOISE20/stairs-2019_10_10_13_03_53.aedat4",
                "noisy": str(DEFAULT_QUALITATIVE_DIR / "converted" / "dvsnoise20" / "stairs-2019_10_10_13_03_53.npz").replace("\\", "/"),
                "scan_windows_ms": [30, 50, 100],
                "preferred_window_ms": 50,
            },
            "labfast_115638": {
                "enabled": True,
                "group": "dvsnoise20",
                "label": "DVSNOISE20 labFast",
                "qualitative_only": True,
                "width": 346,
                "height": 260,
                "tick_ns": 1000,
                "assume": "npz",
                "raw_aedat4": f"{dataset_root}/DVSNOISE20/labFast-2019_10_23_11_56_38.aedat4",
                "noisy": str(DEFAULT_QUALITATIVE_DIR / "converted" / "dvsnoise20" / "labFast-2019_10_23_11_56_38.npz").replace("\\", "/"),
                "scan_windows_ms": [30, 50, 100],
                "preferred_window_ms": 50,
            },
            "labslow_124009": {
                "enabled": True,
                "group": "dvsnoise20",
                "label": "DVSNOISE20 labSlow",
                "qualitative_only": True,
                "width": 346,
                "height": 260,
                "tick_ns": 1000,
                "assume": "npz",
                "raw_aedat4": f"{dataset_root}/DVSNOISE20/labSlow-2019_10_10_12_40_09.aedat4",
                "noisy": str(DEFAULT_QUALITATIVE_DIR / "converted" / "dvsnoise20" / "labSlow-2019_10_10_12_40_09.npz").replace("\\", "/"),
                "scan_windows_ms": [30, 50, 100],
                "preferred_window_ms": 50,
            },
        },
    }


def default_algorithm_params() -> dict[str, Any]:
    base_methods = {
        "Noisy": {"kind": "input", "source": "noisy", "label": "Noisy"},
        "BAF": {"method": "baf", "engine": "cpp", "radius_px": 1, "time_us": 50000, "min_neighbors": 1},
        "STCF": {"method": "stcf_original", "engine": "cpp", "radius_px": 1, "time_us": 32000, "min_neighbors": 2},
        "EBF": {"method": "ebf", "engine": "cpp", "radius_px": 2, "time_us": 32000, "min_neighbors": 1.8},
        "TS": {"method": "ts", "engine": "cpp", "radius_px": 2, "time_us": 32000, "min_neighbors": 0.5},
        "PFD": {
            "method": "pfd",
            "engine": "cpp",
            "radius_px": 1,
            "time_us": 32000,
            "min_neighbors": 0,
            "refractory_us": 1,
            "pfd_mode": "a",
        },
        "Ours": {
            "method": "n149",
            "engine": "cpp",
            "radius_px": 2,
            "time_us": 32000,
            "min_neighbors": 0.8,
            "env": {"MYEVS_N149_SIGMA": "1.75", "MYEVS_N149_ALPHA_FIXED": "0.05", "MYEVS_N149_HOT_BITS": "16"},
        },
    }
    ped_bike_ours = {
        "method": "n149",
        "engine": "cpp",
        "radius_px": 5,
        "time_us": 256000,
        "min_neighbors": 1.0,
        "env": {"MYEVS_N149_SIGMA": "2.75", "MYEVS_N149_ALPHA_FIXED": "0.25", "MYEVS_N149_HOT_BITS": "16"},
    }
    return {
        "version": 1,
        "notes": [
            "Rules are centralized for qualitative figures; tune values here after visual review.",
            "Some thresholds are default single operating points because README2 records sweep/AUC results rather than one mandatory visualization threshold.",
        ],
        "groups": {
            "driving": {"methods": base_methods},
            "stairs": {"methods": base_methods},
            "dvsnoise20": {"methods": base_methods},
            "ped": {
                "methods": {
                    **base_methods,
                    "BAF": {"method": "baf", "engine": "cpp", "radius_px": 1, "time_us": 16000, "min_neighbors": 1},
                    "STCF": {"method": "stcf_original", "engine": "cpp", "radius_px": 1, "time_us": 32000, "min_neighbors": 2},
                    "EBF": {"method": "ebf", "engine": "cpp", "radius_px": 4, "time_us": 64000, "min_neighbors": 4.0},
                    "TS": {"method": "ts", "engine": "cpp", "radius_px": 1, "time_us": 16000, "min_neighbors": 0.3},
                    "PFD": {
                        "method": "pfd",
                        "engine": "cpp",
                        "radius_px": 1,
                        "time_us": 16000,
                        "min_neighbors": 0,
                        "refractory_us": 1,
                        "pfd_mode": "a",
                    },
                    "Ours": {**ped_bike_ours, "min_neighbors": 3.5},
                }
            },
            "bike": {
                "methods": {
                    **base_methods,
                    "BAF": {"method": "baf", "engine": "cpp", "radius_px": 1, "time_us": 100000, "min_neighbors": 1},
                    "STCF": {"method": "stcf_original", "engine": "cpp", "radius_px": 1, "time_us": 32000, "min_neighbors": 2},
                    "EBF": {"method": "ebf", "engine": "cpp", "radius_px": 4, "time_us": 128000, "min_neighbors": 4.0},
                    "TS": {"method": "ts", "engine": "cpp", "radius_px": 2, "time_us": 32000, "min_neighbors": 0.5},
                    "PFD": {
                        "method": "pfd",
                        "engine": "cpp",
                        "radius_px": 1,
                        "time_us": 32000,
                        "min_neighbors": 0,
                        "refractory_us": 1,
                        "pfd_mode": "a",
                    },
                    "Ours": {**ped_bike_ours, "min_neighbors": 2.0},
                }
            },
            "dvsclean": {
                "methods": {
                    **base_methods,
                    "BAF": {"method": "baf", "engine": "cpp", "radius_px": 1, "time_us": 16000, "min_neighbors": 1},
                    "STCF": {"method": "stcf_original", "engine": "cpp", "radius_px": 1, "time_us": 16000, "min_neighbors": 2},
                    "EBF": {"method": "ebf", "engine": "cpp", "radius_px": 4, "time_us": 64000, "min_neighbors": 4.0},
                    "TS": {"method": "ts", "engine": "cpp", "radius_px": 2, "time_us": 64000, "min_neighbors": 0.5},
                    "PFD": {
                        "method": "pfd",
                        "engine": "cpp",
                        "radius_px": 3,
                        "time_us": 16000,
                        "min_neighbors": 1,
                        "refractory_us": 1,
                        "pfd_mode": "a",
                    },
                    "Ours": {
                        "method": "n149",
                        "engine": "cpp",
                        "radius_px": 5,
                        "time_us": 128000,
                        "min_neighbors": 4.0,
                        "env": {"MYEVS_N149_SIGMA": "2.5", "MYEVS_N149_ALPHA_FIXED": "0.25", "MYEVS_N149_HOT_BITS": "16"},
                    },
                }
            },
        },
        "external_methods": {
            "EDnCNN": {"label": "EDnCNN", "optional": True},
            "EDFormer": {"label": "EDFormer", "optional": True},
        },
        "method_order": ["Noisy", "Ours", "BAF", "STCF", "EBF", "TS", "PFD", "EDnCNN", "EDFormer"],
    }
