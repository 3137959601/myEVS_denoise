from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
import re

import numpy as np

from myevs.metrics.aocc import aocc_from_xyt
from myevs.metrics.esr import event_structural_ratio_mean_from_xy
from myevs.timebase import TimeBase


@dataclass(frozen=True)
class LabeledEvents:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    p: np.ndarray
    label: np.ndarray


def _parse_int_list(s: str) -> list[int]:
    out: list[int] = []
    for part in str(s).replace(" ", "").split(","):
        if not part:
            continue
        out.append(int(part))
    return out


def _parse_float_list(s: str) -> list[float]:
    out: list[float] = []
    for part in str(s).replace(" ", "").split(","):
        if not part:
            continue
        out.append(float(part))
    return out


def _fmt_tag_float(x: float) -> str:
    if not np.isfinite(x):
        return "nan"
    xi = int(round(x))
    if abs(x - xi) < 1e-9:
        return str(xi)
    s = f"{x:.6g}"
    s = s.replace("-", "m").replace(".", "p")
    return s


def load_labeled_npy(path: str, *, max_events: int = 0) -> LabeledEvents:
    arr = np.load(path, allow_pickle=False)
    if int(max_events) > 0:
        arr = arr[: int(max_events)]

    if getattr(arr, "dtype", None) is not None and getattr(arr.dtype, "names", None):
        names = set(arr.dtype.names)
        need = {"t", "x", "y", "p", "label"}
        if not need.issubset(names):
            missing = sorted(need - names)
            raise SystemExit(f"input structured npy missing fields: {missing}")
        t = arr["t"].astype(np.uint64, copy=False)
        x = arr["x"].astype(np.int32, copy=False)
        y = arr["y"].astype(np.int32, copy=False)
        p = arr["p"].astype(np.int8, copy=False)
        label = arr["label"].astype(np.int8, copy=False)
    else:
        a2 = np.asarray(arr)
        if a2.ndim != 2 or a2.shape[1] < 5:
            raise SystemExit("input must be structured (t/x/y/p/label) or 2D array with >=5 columns")

        c0 = a2[: min(10000, a2.shape[0]), 0]
        is_bin0 = bool(np.all((c0 == 0) | (c0 == 1)))
        if is_bin0:
            # [label, t, y, x, p]
            label = a2[:, 0].astype(np.int8, copy=False)
            t = a2[:, 1].astype(np.uint64, copy=False)
            y = a2[:, 2].astype(np.int32, copy=False)
            x = a2[:, 3].astype(np.int32, copy=False)
            p = a2[:, 4].astype(np.int8, copy=False)
        else:
            # [t, x, y, p, label]
            t = a2[:, 0].astype(np.uint64, copy=False)
            x = a2[:, 1].astype(np.int32, copy=False)
            y = a2[:, 2].astype(np.int32, copy=False)
            p = a2[:, 3].astype(np.int8, copy=False)
            label = a2[:, 4].astype(np.int8, copy=False)

    label = (label > 0).astype(np.int8, copy=False)

    return LabeledEvents(
        t=np.ascontiguousarray(t),
        x=np.ascontiguousarray(x),
        y=np.ascontiguousarray(y),
        p=np.ascontiguousarray(p),
        label=np.ascontiguousarray(label),
    )


def score_stream_ebf(
    ev: LabeledEvents,
    *,
    width: int,
    height: int,
    radius_px: int,
    tau_us: int,
    tb: TimeBase,
    _kernel_cache: dict[str, object],
    variant: str = "ebf",
) -> np.ndarray:
    scores = np.empty((ev.t.shape[0],), dtype=np.float32)

    v = str(variant).strip().lower()
    if v in {"ebf", "v0", "baseline"}:
        from myevs.denoise.numba_ebf import ebf_scores_stream_numba, ebf_state_init, is_numba_available

        if is_numba_available():
            last_ts, last_pol = ebf_state_init(int(width), int(height))
            tau_ticks = int(tb.us_to_ticks(int(tau_us)))
            ebf_scores_stream_numba(
                t=ev.t,
                x=ev.x,
                y=ev.y,
                p=ev.p,
                width=int(width),
                height=int(height),
                radius_px=int(radius_px),
                tau_ticks=int(tau_ticks),
                last_ts=last_ts,
                last_pol=last_pol,
                scores_out=scores,
            )
            return scores

    if v in {"s1", "ebf_s1", "ebfs1", "dircoh", "directional_coherence"}:
        from myevs.denoise.ops.ebfopt_part2.s1_dircoh import s1_dircoh_params_from_env, try_build_s1_dircoh_scores_kernel

        ker = _kernel_cache.get("ker_ebf_s1")
        if ker is None:
            ker = try_build_s1_dircoh_scores_kernel()
            _kernel_cache["ker_ebf_s1"] = ker

        # IMPORTANT: For Part2 experiments, silently falling back would produce invalid results.
        if ker is None:
            raise SystemExit(
                "EBF Part2 s1 requires numba kernel, but numba is unavailable. "
                "On Windows this is often caused by user-site NumPy shadowing the conda env (NumPy 2.4 + numba incompat). "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        params = s1_dircoh_params_from_env()
        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            float(params.eta),
            last_ts,
            last_pol,
            scores,
        )
        return scores

    if v in {"s2", "ebf_s2", "ebfs2", "coh_gate", "coherence_gate"}:
        from myevs.denoise.ops.ebfopt_part2.s2_cohgate import s2_cohgate_params_from_env, try_build_s2_cohgate_scores_kernel

        ker = _kernel_cache.get("ker_ebf_s2")
        if ker is None:
            ker = try_build_s2_cohgate_scores_kernel()
            _kernel_cache["ker_ebf_s2"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s2 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        params = s2_cohgate_params_from_env()
        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            float(params.coh_thr),
            float(params.raw_thr),
            float(params.gamma),
            last_ts,
            last_pol,
            scores,
        )
        return scores

    if v in {"s3", "ebf_s3", "ebfs3", "softgate", "coh_softgate"}:
        from myevs.denoise.ops.ebfopt_part2.s3_softgate import s3_softgate_params_from_env, try_build_s3_softgate_scores_kernel

        ker = _kernel_cache.get("ker_ebf_s3")
        if ker is None:
            ker = try_build_s3_softgate_scores_kernel()
            _kernel_cache["ker_ebf_s3"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s3 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        params = s3_softgate_params_from_env()
        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            float(params.coh_thr),
            float(params.raw_thr),
            float(params.gamma),
            float(params.alpha),
            float(params.k_raw),
            float(params.k_coh),
            last_ts,
            last_pol,
            scores,
        )
        return scores

    if v in {"s4", "ebf_s4", "ebfs4", "residual_gate", "resultant_gate"}:
        from myevs.denoise.ops.ebfopt_part2.s4_residual_gate import (
            s4_residual_gate_params_from_env,
            try_build_s4_residual_gate_scores_kernel,
        )

        ker = _kernel_cache.get("ker_ebf_s4")
        if ker is None:
            ker = try_build_s4_residual_gate_scores_kernel()
            _kernel_cache["ker_ebf_s4"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s4 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        params = s4_residual_gate_params_from_env()
        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            float(params.align_thr),
            float(params.raw_thr),
            float(params.gamma),
            last_ts,
            last_pol,
            scores,
        )
        return scores

    if v in {"s5", "ebf_s5", "ebfs5", "elliptic_spatialw", "elliptic"}:
        from myevs.denoise.ops.ebfopt_part2.s5_elliptic_spatialw import (
            s5_elliptic_spatialw_params_from_env,
            try_build_s5_elliptic_spatialw_scores_kernel,
        )

        ker = _kernel_cache.get("ker_ebf_s5")
        if ker is None:
            ker = try_build_s5_elliptic_spatialw_scores_kernel()
            _kernel_cache["ker_ebf_s5"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s5 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        params = s5_elliptic_spatialw_params_from_env()
        theta = float(params.theta_deg) * (np.pi / 180.0)
        cos_t = float(np.cos(theta))
        sin_t = float(np.sin(theta))

        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            float(params.ax),
            float(params.ay),
            float(cos_t),
            float(sin_t),
            last_ts,
            last_pol,
            scores,
        )
        return scores

    if v in {"s6", "ebf_s6", "ebfs6", "timecoh_gate", "time_gate"}:
        from myevs.denoise.ops.ebfopt_part2.s6_timecoh_gate import (
            s6_timecoh_gate_params_from_env,
            try_build_s6_timecoh_gate_scores_kernel,
        )

        ker = _kernel_cache.get("ker_ebf_s6")
        if ker is None:
            ker = try_build_s6_timecoh_gate_scores_kernel()
            _kernel_cache["ker_ebf_s6"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s6 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        params = s6_timecoh_gate_params_from_env()
        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            float(params.timecoh_thr),
            float(params.raw_thr),
            float(params.gamma),
            last_ts,
            last_pol,
            scores,
        )
        return scores

    if v in {"s7", "ebf_s7", "ebfs7", "plane_gate", "plane_residual_gate"}:
        from myevs.denoise.ops.ebfopt_part2.s7_plane_gate import (
            s7_plane_gate_params_from_env,
            try_build_s7_plane_gate_scores_kernel,
        )

        ker = _kernel_cache.get("ker_ebf_s7")
        if ker is None:
            ker = try_build_s7_plane_gate_scores_kernel()
            _kernel_cache["ker_ebf_s7"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s7 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        params = s7_plane_gate_params_from_env()
        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            float(params.sigma_thr),
            float(params.raw_thr),
            float(params.gamma),
            int(params.min_pts),
            last_ts,
            last_pol,
            scores,
        )
        return scores

    if v in {"s8", "ebf_s8", "ebfs8", "plane_r2_gate", "plane_explained_gate", "plane_r2"}:
        from myevs.denoise.ops.ebfopt_part2.s8_plane_r2_gate import (
            s8_plane_r2_gate_params_from_env,
            try_build_s8_plane_r2_gate_scores_kernel,
        )

        ker = _kernel_cache.get("ker_ebf_s8")
        if ker is None:
            ker = try_build_s8_plane_r2_gate_scores_kernel()
            _kernel_cache["ker_ebf_s8"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s8 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        params = s8_plane_r2_gate_params_from_env()
        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            float(params.r2_thr),
            float(params.raw_thr),
            float(params.gamma),
            int(params.min_pts),
            last_ts,
            last_pol,
            scores,
        )
        return scores

    if v in {"s9", "ebf_s9", "ebfs9", "refractory_gate", "burst_gate", "self_refractory"}:
        from myevs.denoise.ops.ebfopt_part2.s9_refractory_gate import (
            s9_refractory_gate_params_from_env,
            try_build_s9_refractory_gate_scores_kernel,
        )

        ker = _kernel_cache.get("ker_ebf_s9")
        if ker is None:
            ker = try_build_s9_refractory_gate_scores_kernel()
            _kernel_cache["ker_ebf_s9"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s9 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        params = s9_refractory_gate_params_from_env()
        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            float(params.dt_thr),
            float(params.raw_thr),
            float(params.gamma),
            last_ts,
            last_pol,
            scores,
        )
        return scores

    if v in {"s22", "ebf_s22", "ebfs22", "anypol_burst_gate", "anypol_burst", "burst_anypol"}:
        from myevs.denoise.ops.ebfopt_part2.s22_anypol_burst_gate import (
            s22_anypol_burst_gate_params_from_env,
            try_build_s22_anypol_burst_gate_scores_kernel,
        )

        ker = _kernel_cache.get("ker_ebf_s22")
        if ker is None:
            ker = try_build_s22_anypol_burst_gate_scores_kernel()
            _kernel_cache["ker_ebf_s22"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s22 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        params = s22_anypol_burst_gate_params_from_env()
        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        dt_thr_ticks = int(tb.us_to_ticks(int(params.dt_thr_us)))
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            int(dt_thr_ticks),
            float(params.raw_thr),
            float(params.gamma),
            last_ts,
            last_pol,
            scores,
        )
        return scores

    if v in {"s23", "ebf_s23", "ebfs23", "featlogit", "feature_logit", "feature_fusion"}:
        from myevs.denoise.ops.ebfopt_part2.s23_featlogit import (
            s23_featlogit_params_from_env,
            try_build_s23_featlogit_scores_kernel,
        )

        params = s23_featlogit_params_from_env()
        use_selfacc = float(params.w_selfacc) != 0.0

        hotmask_path = (os.environ.get("MYEVS_EBF_S23_HOTMASK_NPY", "") or "").strip()
        use_hotmask = (
            (float(getattr(params, "w_hot", 0.0)) != 0.0) or (float(getattr(params, "w_hotnbr", 0.0)) != 0.0)
        ) and bool(hotmask_path)

        hotmask_u8 = None
        if use_hotmask:
            if not os.path.exists(hotmask_path):
                raise SystemExit(f"MYEVS_EBF_S23_HOTMASK_NPY not found: {hotmask_path!r}")
            m = np.load(hotmask_path, allow_pickle=False)
            m = np.asarray(m)
            if m.ndim == 2:
                if int(m.shape[0]) != int(height) or int(m.shape[1]) != int(width):
                    raise SystemExit(
                        f"hotmask shape mismatch: got {tuple(m.shape)}, expected (height,width)=({int(height)},{int(width)})"
                    )
                m = m.reshape((-1,))
            elif m.ndim == 1:
                if int(m.shape[0]) != int(width) * int(height):
                    raise SystemExit(
                        f"hotmask length mismatch: got {int(m.shape[0])}, expected {int(width)*int(height)}"
                    )
            else:
                raise SystemExit(f"hotmask must be 1D (W*H) or 2D (H,W); got shape {tuple(m.shape)}")
            hotmask_u8 = np.ascontiguousarray((m != 0).astype(np.uint8, copy=False))

        ker_key = f"ker_ebf_s23_sa{int(use_selfacc)}_hm{int(use_hotmask)}"
        ker = _kernel_cache.get(ker_key)
        if ker is None:
            ker = try_build_s23_featlogit_scores_kernel(with_selfacc=use_selfacc, with_hotmask=use_hotmask)
            _kernel_cache[ker_key] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s23 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        self_acc_q8 = None
        if use_selfacc:
            self_acc_q8 = np.zeros((int(width) * int(height),), dtype=np.uint16)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        dt_thr_ticks = int(tb.us_to_ticks(int(params.dt_thr_us)))
        if use_selfacc and use_hotmask:
            ker(
                ev.t,
                ev.x,
                ev.y,
                ev.p,
                int(width),
                int(height),
                int(radius_px),
                int(tau_ticks),
                int(dt_thr_ticks),
                float(params.bias),
                float(params.w_same),
                float(params.w_opp),
                float(params.w_oppr),
                float(params.w_toggle),
                float(params.w_dtsmall),
                float(params.w_sameburst),
                float(params.w_selfacc),
                float(getattr(params, "w_hot", 0.0)),
                float(getattr(params, "w_hotnbr", 0.0)),
                last_ts,
                last_pol,
                self_acc_q8,
                hotmask_u8,
                scores,
            )
        elif use_selfacc:
            ker(
                ev.t,
                ev.x,
                ev.y,
                ev.p,
                int(width),
                int(height),
                int(radius_px),
                int(tau_ticks),
                int(dt_thr_ticks),
                float(params.bias),
                float(params.w_same),
                float(params.w_opp),
                float(params.w_oppr),
                float(params.w_toggle),
                float(params.w_dtsmall),
                float(params.w_sameburst),
                float(params.w_selfacc),
                last_ts,
                last_pol,
                self_acc_q8,
                scores,
            )
        elif use_hotmask:
            ker(
                ev.t,
                ev.x,
                ev.y,
                ev.p,
                int(width),
                int(height),
                int(radius_px),
                int(tau_ticks),
                int(dt_thr_ticks),
                float(params.bias),
                float(params.w_same),
                float(params.w_opp),
                float(params.w_oppr),
                float(params.w_toggle),
                float(params.w_dtsmall),
                float(params.w_sameburst),
                float(getattr(params, "w_hot", 0.0)),
                float(getattr(params, "w_hotnbr", 0.0)),
                last_ts,
                last_pol,
                hotmask_u8,
                scores,
            )
        else:
            ker(
                ev.t,
                ev.x,
                ev.y,
                ev.p,
                int(width),
                int(height),
                int(radius_px),
                int(tau_ticks),
                int(dt_thr_ticks),
                float(params.bias),
                float(params.w_same),
                float(params.w_opp),
                float(params.w_oppr),
                float(params.w_toggle),
                float(params.w_dtsmall),
                float(params.w_sameburst),
                last_ts,
                last_pol,
                scores,
            )
        return scores

    if v in {"s10", "ebf_s10", "ebfs10", "hotpixel_rate_gate", "rate_gate", "hotpixel_gate"}:
        from myevs.denoise.ops.ebfopt_part2.s10_hotpixel_rate_gate import (
            s10_hotpixel_rate_gate_params_from_env,
            try_build_s10_hotpixel_rate_gate_scores_kernel,
        )

        ker = _kernel_cache.get("ker_ebf_s10")
        if ker is None:
            ker = try_build_s10_hotpixel_rate_gate_scores_kernel()
            _kernel_cache["ker_ebf_s10"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s10 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        params = s10_hotpixel_rate_gate_params_from_env()
        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        self_acc = np.zeros((int(width) * int(height),), dtype=np.float32)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            float(params.acc_thr),
            float(params.raw_thr),
            float(params.gamma),
            last_ts,
            last_pol,
            self_acc,
            scores,
        )
        return scores

    if v in {"s11", "ebf_s11", "ebfs11", "relative_hotness_gate", "relative_hotness", "relhot"}:
        from myevs.denoise.ops.ebfopt_part2.s11_relative_hotness_gate import (
            s11_relative_hotness_gate_params_from_env,
            try_build_s11_relative_hotness_gate_scores_kernel,
        )

        ker = _kernel_cache.get("ker_ebf_s11")
        if ker is None:
            ker = try_build_s11_relative_hotness_gate_scores_kernel()
            _kernel_cache["ker_ebf_s11"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s11 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        params = s11_relative_hotness_gate_params_from_env()
        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        self_acc = np.zeros((int(width) * int(height),), dtype=np.float32)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            float(params.acc_thr),
            float(params.ratio_thr),
            float(params.raw_thr),
            float(params.gamma),
            last_ts,
            last_pol,
            self_acc,
            scores,
        )
        return scores

    if v in {"s12", "ebf_s12", "ebfs12", "hotness_zscore_gate", "zscore_hotness", "zgate"}:
        from myevs.denoise.ops.ebfopt_part2.s12_hotness_zscore_gate import (
            s12_hotness_zscore_gate_params_from_env,
            try_build_s12_hotness_zscore_gate_scores_kernel,
        )

        ker = _kernel_cache.get("ker_ebf_s12")
        if ker is None:
            ker = try_build_s12_hotness_zscore_gate_scores_kernel()
            _kernel_cache["ker_ebf_s12"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s12 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        params = s12_hotness_zscore_gate_params_from_env()
        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        self_acc = np.zeros((int(width) * int(height),), dtype=np.float32)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            float(params.acc_thr),
            float(params.z_thr),
            float(params.raw_thr),
            float(params.gamma),
            last_ts,
            last_pol,
            self_acc,
            scores,
        )
        return scores

    if v in {"s13", "ebf_s13", "ebfs13", "crosspol_support_gate", "crosspol_gate", "crosspol_support"}:
        from myevs.denoise.ops.ebfopt_part2.s13_crosspol_support_gate import (
            s13_crosspol_support_gate_params_from_env,
            try_build_s13_crosspol_support_gate_scores_kernel,
        )

        ker = _kernel_cache.get("ker_ebf_s13")
        if ker is None:
            ker = try_build_s13_crosspol_support_gate_scores_kernel()
            _kernel_cache["ker_ebf_s13"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s13 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        params = s13_crosspol_support_gate_params_from_env()
        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            float(params.bal_thr),
            float(params.raw_thr),
            float(params.gamma),
            last_ts,
            last_pol,
            scores,
        )
        return scores

    if v in {"s14", "ebf_s14", "ebfs14", "crosspol_boost", "crosspol_boost_score", "crosspol_add"}:
        from myevs.denoise.ops.ebfopt_part2.s14_crosspol_boost import (
            s14_crosspol_boost_params_from_env,
            try_build_s14_crosspol_boost_scores_kernel,
        )

        ker = _kernel_cache.get("ker_ebf_s14")
        if ker is None:
            ker = try_build_s14_crosspol_boost_scores_kernel()
            _kernel_cache["ker_ebf_s14"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s14 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        params = s14_crosspol_boost_params_from_env()
        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            float(params.alpha),
            float(params.raw_thr),
            last_ts,
            last_pol,
            scores,
        )
        return scores

    if v in {"s24", "ebf_s24", "ebfs24", "s14_burstiness_gate", "burstiness_gate"}:
        from myevs.denoise.ops.ebfopt_part2.s24_s14_burstiness_gate import (
            s24_s14_burstiness_gate_params_from_env,
            try_build_s24_s14_burstiness_gate_scores_kernel,
        )

        ker = _kernel_cache.get("ker_ebf_s24")
        if ker is None:
            ker = try_build_s24_s14_burstiness_gate_scores_kernel()
            _kernel_cache["ker_ebf_s24"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s24 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        params = s24_s14_burstiness_gate_params_from_env()
        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        bdt_ticks = int(tb.us_to_ticks(int(params.burst_dt_us)))
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            float(params.alpha),
            float(params.raw_thr),
            int(bdt_ticks),
            float(params.b_thr),
            float(params.gamma),
            last_ts,
            last_pol,
            scores,
        )
        return scores

    if v in {"s25", "ebf_s25", "ebfs25", "s14_refractory_gate", "s14_refractory", "refractory_s14"}:
        from myevs.denoise.ops.ebfopt_part2.s25_s14_refractory_gate import (
            s25_s14_refractory_gate_params_from_env,
            try_build_s25_s14_refractory_gate_scores_kernel,
        )

        ker = _kernel_cache.get("ker_ebf_s25")
        if ker is None:
            ker = try_build_s25_s14_refractory_gate_scores_kernel()
            _kernel_cache["ker_ebf_s25"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s25 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        params = s25_s14_refractory_gate_params_from_env()
        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            float(params.alpha),
            float(params.s14_raw_thr),
            float(params.dt_thr),
            float(params.ref_raw_thr),
            float(params.gamma),
            last_ts,
            last_pol,
            scores,
        )
        return scores

    if v in {"s15", "ebf_s15", "ebfs15", "flip_flicker_gate", "polarity_flip_gate", "flicker"}:
        from myevs.denoise.ops.ebfopt_part2.s15_flip_flicker_gate import (
            s15_flip_flicker_gate_params_from_env,
            try_build_s15_flip_flicker_gate_scores_kernel,
        )

        ker = _kernel_cache.get("ker_ebf_s15")
        if ker is None:
            ker = try_build_s15_flip_flicker_gate_scores_kernel()
            _kernel_cache["ker_ebf_s15"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s15 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        params = s15_flip_flicker_gate_params_from_env()
        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        flip_dt_ticks = int(tb.us_to_ticks(int(params.flip_dt_us)))
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            float(params.alpha),
            float(params.raw_thr),
            int(flip_dt_ticks),
            float(params.beta),
            last_ts,
            last_pol,
            scores,
        )
        return scores

    if v in {"s16", "ebf_s16", "ebfs16", "s14_hotness_clamp", "hotness_clamp"}:
        from myevs.denoise.ops.ebfopt_part2.s16_s14_hotness_clamp import (
            s16_s14_hotness_clamp_params_from_env,
            try_build_s16_s14_hotness_clamp_scores_kernel,
        )

        ker = _kernel_cache.get("ker_ebf_s16")
        if ker is None:
            ker = try_build_s16_s14_hotness_clamp_scores_kernel()
            _kernel_cache["ker_ebf_s16"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s16 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        params = s16_s14_hotness_clamp_params_from_env()
        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        self_acc = np.zeros((int(width) * int(height),), dtype=np.float32)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            float(params.alpha),
            float(params.raw_thr),
            float(params.acc_thr),
            float(params.ratio_thr),
            float(params.gamma),
            last_ts,
            last_pol,
            self_acc,
            scores,
        )
        return scores

    if v in {"s17", "ebf_s17", "ebfs17", "crosspol_spread_boost", "spread_boost"}:
        from myevs.denoise.ops.ebfopt_part2.s17_crosspol_spread_boost import (
            s17_crosspol_spread_boost_params_from_env,
            try_build_s17_crosspol_spread_boost_scores_kernel,
        )

        ker = _kernel_cache.get("ker_ebf_s17")
        if ker is None:
            ker = try_build_s17_crosspol_spread_boost_scores_kernel()
            _kernel_cache["ker_ebf_s17"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s17 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        params = s17_crosspol_spread_boost_params_from_env()
        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            float(params.alpha),
            float(params.raw_thr),
            float(params.var_thr),
            float(params.beta),
            float(params.gamma),
            last_ts,
            last_pol,
            scores,
        )
        return scores

    if v in {"s18", "ebf_s18", "ebfs18", "nopol", "no_polarity", "polarity_free"}:
        from myevs.denoise.ops.ebfopt_part2.s18_no_polarity_ebf import try_build_s18_no_polarity_ebf_scores_kernel

        ker = _kernel_cache.get("ker_ebf_s18")
        if ker is None:
            ker = try_build_s18_no_polarity_ebf_scores_kernel()
            _kernel_cache["ker_ebf_s18"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s18 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            last_ts,
            scores,
        )
        return scores

    if v in {"s19", "ebf_s19", "ebfs19", "evidence_fusion", "fusion", "fusion_q8"}:
        from myevs.denoise.ops.ebfopt_part2.s19_evidence_fusion_q8 import (
            _to_q8,
            s19_evidence_fusion_q8_params_from_env,
            try_build_s19_evidence_fusion_q8_scores_kernel,
        )

        ker = _kernel_cache.get("ker_ebf_s19")
        if ker is None:
            ker = try_build_s19_evidence_fusion_q8_scores_kernel()
            _kernel_cache["ker_ebf_s19"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s19 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        params = s19_evidence_fusion_q8_params_from_env()
        alpha_q8 = int(_to_q8(float(params.alpha)))
        beta_q8 = int(_to_q8(float(params.beta)))

        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        self_acc_w = np.zeros((int(width) * int(height),), dtype=np.int32)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            int(alpha_q8),
            int(beta_q8),
            last_ts,
            last_pol,
            self_acc_w,
            scores,
        )
        return scores

    if v in {"s20", "ebf_s20", "ebfs20", "fusion_polhot", "polhot_fusion", "pol_hotness"}:
        from myevs.denoise.ops.ebfopt_part2.s20_polhot_evidence_fusion_q8 import (
            _to_q8,
            s20_polhot_evidence_fusion_q8_params_from_env,
            try_build_s20_polhot_evidence_fusion_q8_scores_kernel,
        )

        ker = _kernel_cache.get("ker_ebf_s20")
        if ker is None:
            ker = try_build_s20_polhot_evidence_fusion_q8_scores_kernel()
            _kernel_cache["ker_ebf_s20"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s20 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        params = s20_polhot_evidence_fusion_q8_params_from_env()
        alpha_q8 = int(_to_q8(float(params.alpha)))
        beta_q8 = int(_to_q8(float(params.beta)))

        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        acc_neg = np.zeros((int(width) * int(height),), dtype=np.int32)
        acc_pos = np.zeros((int(width) * int(height),), dtype=np.int32)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            int(alpha_q8),
            int(beta_q8),
            last_ts,
            last_pol,
            acc_neg,
            acc_pos,
            scores,
        )
        return scores

    if v in {"s21", "ebf_s21", "ebfs21", "bipolhot_fusion", "bipol_hotness", "polhot_mix"}:
        from myevs.denoise.ops.ebfopt_part2.s21_bipolhot_evidence_fusion_q8 import (
            _to_q8,
            s21_bipolhot_evidence_fusion_q8_params_from_env,
            try_build_s21_bipolhot_evidence_fusion_q8_scores_kernel,
        )

        ker = _kernel_cache.get("ker_ebf_s21")
        if ker is None:
            ker = try_build_s21_bipolhot_evidence_fusion_q8_scores_kernel()
            _kernel_cache["ker_ebf_s21"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s21 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        params = s21_bipolhot_evidence_fusion_q8_params_from_env()
        alpha_q8 = int(_to_q8(float(params.alpha)))
        beta_q8 = int(_to_q8(float(params.beta)))
        kappa_q8 = int(_to_q8(float(params.kappa)))

        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        acc_neg = np.zeros((int(width) * int(height),), dtype=np.int32)
        acc_pos = np.zeros((int(width) * int(height),), dtype=np.int32)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            int(alpha_q8),
            int(beta_q8),
            int(kappa_q8),
            last_ts,
            last_pol,
            acc_neg,
            acc_pos,
            scores,
        )
        return scores

    if v in {"s26", "ebf_s26", "s26_actnorm", "actnorm_hotness"}:
        from myevs.denoise.ops.ebfopt_part2.s26_actnorm_hotness_fusion_q8 import (
            _to_q8,
            s26_actnorm_hotness_fusion_q8_params_from_env,
            try_build_s26_actnorm_hotness_fusion_q8_scores_kernel,
        )

        ker = _kernel_cache.get("ker_ebf_s26")
        if ker is None:
            ker = try_build_s26_actnorm_hotness_fusion_q8_scores_kernel()
            _kernel_cache["ker_ebf_s26"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s26 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        params = s26_actnorm_hotness_fusion_q8_params_from_env()
        alpha_q8 = int(_to_q8(float(params.alpha)))
        beta_q8 = int(_to_q8(float(params.beta)))
        kappa_q8 = int(_to_q8(float(params.kappa)))
        eta_q8 = int(_to_q8(float(params.eta)))

        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        acc_neg = np.zeros((int(width) * int(height),), dtype=np.int32)
        acc_pos = np.zeros((int(width) * int(height),), dtype=np.int32)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            int(alpha_q8),
            int(beta_q8),
            int(kappa_q8),
            int(eta_q8),
            last_ts,
            last_pol,
            acc_neg,
            acc_pos,
            scores,
        )
        return scores

    if v in {"s27", "ebf_s27", "ebfs27", "s27_relabnorm", "relabnorm_hotness", "relative_abnormal_hotness"}:
        from myevs.denoise.ops.ebfopt_part2.s27_relabnorm_hotness_fusion_q8 import (
            _to_q8,
            s27_relabnorm_hotness_fusion_q8_params_from_env,
            try_build_s27_relabnorm_hotness_fusion_q8_scores_kernel,
        )

        ker = _kernel_cache.get("ker_ebf_s27")
        if ker is None:
            ker = try_build_s27_relabnorm_hotness_fusion_q8_scores_kernel()
            _kernel_cache["ker_ebf_s27"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s27 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        params = s27_relabnorm_hotness_fusion_q8_params_from_env()
        alpha_q8 = int(_to_q8(float(params.alpha)))
        beta_q8 = int(_to_q8(float(params.beta)))
        kappa_q8 = int(_to_q8(float(params.kappa)))
        lambda_q8 = int(_to_q8(float(params.lambda_nb)))

        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        acc_neg = np.zeros((int(width) * int(height),), dtype=np.int32)
        acc_pos = np.zeros((int(width) * int(height),), dtype=np.int32)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            int(alpha_q8),
            int(beta_q8),
            int(kappa_q8),
            int(lambda_q8),
            last_ts,
            last_pol,
            acc_neg,
            acc_pos,
            scores,
        )
        return scores

    if v in {"s28", "ebf_s28", "ebfs28", "noise_surprise", "surprise_zscore", "surprise_z"}:
        from myevs.denoise.ops.ebfopt_part2.s28_noise_surprise_zscore import (
            s28_noise_surprise_zscore_params_from_env,
            try_build_s28_noise_surprise_zscore_scores_kernel,
        )

        ker = _kernel_cache.get("ker_ebf_s28")
        if ker is None:
            ker = try_build_s28_noise_surprise_zscore_scores_kernel()
            _kernel_cache["ker_ebf_s28"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s28 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        params = s28_noise_surprise_zscore_params_from_env()
        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        rate_ema = np.zeros((1,), dtype=np.float64)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        if int(params.tau_rate_us) > 0:
            tau_rate_ticks = int(tb.us_to_ticks(int(params.tau_rate_us)))
        else:
            tau_rate_ticks = int(tau_ticks)
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            int(tau_rate_ticks),
            last_ts,
            last_pol,
            rate_ema,
            scores,
        )
        return scores

    if v in {"s35", "ebf_s35", "ebfs35", "surprise_pixelstate", "surprise_pixel_state", "s28_pixelstate"}:
        from myevs.denoise.ops.ebfopt_part2.s35_noise_surprise_zscore_pixelstate import (
            s35_noise_surprise_zscore_pixelstate_params_from_env,
            try_build_s35_noise_surprise_zscore_pixelstate_scores_kernel,
        )

        ker = _kernel_cache.get("ker_ebf_s35")
        if ker is None:
            ker = try_build_s35_noise_surprise_zscore_pixelstate_scores_kernel()
            _kernel_cache["ker_ebf_s35"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s35 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        params = s35_noise_surprise_zscore_pixelstate_params_from_env()
        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        hot_state = np.zeros((int(width) * int(height),), dtype=np.int32)
        rate_ema = np.zeros((1,), dtype=np.float64)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        if int(params.tau_rate_us) > 0:
            tau_rate_ticks = int(tb.us_to_ticks(int(params.tau_rate_us)))
        else:
            tau_rate_ticks = int(tau_ticks)
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            int(tau_rate_ticks),
            float(params.gamma),
            float(params.hmax),
            last_ts,
            last_pol,
            hot_state,
            rate_ema,
            scores,
        )
        return scores

    if v in {"s36", "ebf_s36", "ebfs36", "surprise_occupancy", "surprise_stateoccupancy", "s28_stateoccupancy"}:
        from myevs.denoise.ops.ebfopt_part2.s36_noise_surprise_zscore_stateoccupancy import (
            s36_noise_surprise_zscore_stateoccupancy_params_from_env,
            try_build_s36_noise_surprise_zscore_stateoccupancy_scores_kernel,
        )

        ker = _kernel_cache.get("ker_ebf_s36")
        if ker is None:
            ker = try_build_s36_noise_surprise_zscore_stateoccupancy_scores_kernel()
            _kernel_cache["ker_ebf_s36"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s36 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        params = s36_noise_surprise_zscore_stateoccupancy_params_from_env()
        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        hot_state = np.zeros((int(width) * int(height),), dtype=np.int32)
        rate_ema = np.zeros((1,), dtype=np.float64)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        if int(params.tau_rate_us) > 0:
            tau_rate_ticks = int(tb.us_to_ticks(int(params.tau_rate_us)))
        else:
            tau_rate_ticks = int(tau_ticks)
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            int(tau_rate_ticks),
            last_ts,
            last_pol,
            hot_state,
            rate_ema,
            scores,
        )
        return scores

    if v in {
        "s37",
        "ebf_s37",
        "ebfs37",
        "surprise_occupancy_3state",
        "surprise_stateoccupancy_3state",
        "s28_stateoccupancy_3state",
    }:
        from myevs.denoise.ops.ebfopt_part2.s37_noise_surprise_zscore_stateoccupancy_3state import (
            s37_noise_surprise_zscore_stateoccupancy_3state_params_from_env,
            try_build_s37_noise_surprise_zscore_stateoccupancy_3state_scores_kernel,
        )

        ker = _kernel_cache.get("ker_ebf_s37")
        if ker is None:
            ker = try_build_s37_noise_surprise_zscore_stateoccupancy_3state_scores_kernel()
            _kernel_cache["ker_ebf_s37"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s37 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        params = s37_noise_surprise_zscore_stateoccupancy_3state_params_from_env()
        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        hot_state = np.zeros((int(width) * int(height),), dtype=np.int32)
        rate_ema = np.zeros((1,), dtype=np.float64)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        if int(params.tau_rate_us) > 0:
            tau_rate_ticks = int(tb.us_to_ticks(int(params.tau_rate_us)))
        else:
            tau_rate_ticks = int(tau_ticks)
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            int(tau_rate_ticks),
            last_ts,
            last_pol,
            hot_state,
            rate_ema,
            scores,
        )
        return scores

    if v in {
        "s38",
        "ebf_s38",
        "ebfs38",
        "surprise_occupancy_nbocc",
        "surprise_stateocc_nbocc",
        "s28_stateocc_nbocc",
    }:
        from myevs.denoise.ops.ebfopt_part2.s38_noise_surprise_zscore_stateoccupancy_nbocc_fusion import (
            s38_noise_surprise_zscore_stateocc_nbocc_fusion_params_from_env,
            try_build_s38_noise_surprise_zscore_stateocc_nbocc_fusion_scores_kernel,
        )

        ker = _kernel_cache.get("ker_ebf_s38")
        if ker is None:
            ker = try_build_s38_noise_surprise_zscore_stateocc_nbocc_fusion_scores_kernel()
            _kernel_cache["ker_ebf_s38"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s38 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        params = s38_noise_surprise_zscore_stateocc_nbocc_fusion_params_from_env()
        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        hot_state = np.zeros((int(width) * int(height),), dtype=np.int32)
        rate_ema = np.zeros((1,), dtype=np.float64)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        if int(params.tau_rate_us) > 0:
            tau_rate_ticks = int(tb.us_to_ticks(int(params.tau_rate_us)))
        else:
            tau_rate_ticks = int(tau_ticks)
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            int(tau_rate_ticks),
            last_ts,
            last_pol,
            hot_state,
            rate_ema,
            scores,
        )
        return scores

    if v in {
        "s39",
        "ebf_s39",
        "ebfs39",
        "surprise_occupancy_nbocc_mix",
        "surprise_stateocc_nbocc_mix",
        "s28_stateocc_nbocc_mix",
    }:
        from myevs.denoise.ops.ebfopt_part2.s39_noise_surprise_zscore_stateoccupancy_nbocc_mix import (
            s39_noise_surprise_zscore_stateocc_nbocc_mix_params_from_env,
            try_build_s39_noise_surprise_zscore_stateocc_nbocc_mix_scores_kernel,
        )

        ker = _kernel_cache.get("ker_ebf_s39")
        if ker is None:
            ker = try_build_s39_noise_surprise_zscore_stateocc_nbocc_mix_scores_kernel()
            _kernel_cache["ker_ebf_s39"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s39 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        params = s39_noise_surprise_zscore_stateocc_nbocc_mix_params_from_env()
        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        hot_state = np.zeros((int(width) * int(height),), dtype=np.int32)
        rate_ema = np.zeros((1,), dtype=np.float64)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        if int(params.tau_rate_us) > 0:
            tau_rate_ticks = int(tb.us_to_ticks(int(params.tau_rate_us)))
        else:
            tau_rate_ticks = int(tau_ticks)
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            int(tau_rate_ticks),
            float(params.k_nbmix),
            last_ts,
            last_pol,
            hot_state,
            rate_ema,
            scores,
        )
        return scores

    if v in {
        "s40",
        "ebf_s40",
        "ebfs40",
        "surprise_occupancy_nbocc_mix_fuse_geom",
        "surprise_stateocc_nbocc_mix_fuse_geom",
    }:
        from myevs.denoise.ops.ebfopt_part2.s40_noise_surprise_zscore_stateocc_nbocc_mix_fuse_geom import (
            s40_noise_surprise_zscore_stateocc_nbocc_mix_fuse_geom_params_from_env,
            try_build_s40_noise_surprise_zscore_stateocc_nbocc_mix_fuse_geom_scores_kernel,
        )

        ker = _kernel_cache.get("ker_ebf_s40")
        if ker is None:
            ker = try_build_s40_noise_surprise_zscore_stateocc_nbocc_mix_fuse_geom_scores_kernel()
            _kernel_cache["ker_ebf_s40"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s40 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        params = s40_noise_surprise_zscore_stateocc_nbocc_mix_fuse_geom_params_from_env()
        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        hot_state = np.zeros((int(width) * int(height),), dtype=np.int32)
        rate_ema = np.zeros((1,), dtype=np.float64)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        if int(params.tau_rate_us) > 0:
            tau_rate_ticks = int(tb.us_to_ticks(int(params.tau_rate_us)))
        else:
            tau_rate_ticks = int(tau_ticks)
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            int(tau_rate_ticks),
            float(params.k_nbmix),
            last_ts,
            last_pol,
            hot_state,
            rate_ema,
            scores,
        )
        return scores

    if v in {
        "s41",
        "ebf_s41",
        "ebfs41",
        "surprise_occupancy_nbocc_mix_pow2",
        "surprise_stateocc_nbocc_mix_pow2",
    }:
        from myevs.denoise.ops.ebfopt_part2.s41_noise_surprise_zscore_stateocc_nbocc_mix_pow2 import (
            s41_noise_surprise_zscore_stateocc_nbocc_mix_pow2_params_from_env,
            try_build_s41_noise_surprise_zscore_stateocc_nbocc_mix_pow2_scores_kernel,
        )

        ker = _kernel_cache.get("ker_ebf_s41")
        if ker is None:
            ker = try_build_s41_noise_surprise_zscore_stateocc_nbocc_mix_pow2_scores_kernel()
            _kernel_cache["ker_ebf_s41"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s41 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        params = s41_noise_surprise_zscore_stateocc_nbocc_mix_pow2_params_from_env()
        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        hot_state = np.zeros((int(width) * int(height),), dtype=np.int32)
        rate_ema = np.zeros((1,), dtype=np.float64)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        if int(params.tau_rate_us) > 0:
            tau_rate_ticks = int(tb.us_to_ticks(int(params.tau_rate_us)))
        else:
            tau_rate_ticks = int(tau_ticks)
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            int(tau_rate_ticks),
            float(params.k_nbmix),
            last_ts,
            last_pol,
            hot_state,
            rate_ema,
            scores,
        )
        return scores

    if v in {
        "s42",
        "ebf_s42",
        "ebfs42",
        "surprise_occupancy_nbocc_mix_gated_self2",
        "surprise_stateocc_nbocc_mix_gated_self2",
    }:
        from myevs.denoise.ops.ebfopt_part2.s42_noise_surprise_zscore_stateocc_nbocc_mix_gated_self2 import (
            s42_noise_surprise_zscore_stateocc_nbocc_mix_gated_self2_params_from_env,
            try_build_s42_noise_surprise_zscore_stateocc_nbocc_mix_gated_self2_scores_kernel,
        )

        ker = _kernel_cache.get("ker_ebf_s42")
        if ker is None:
            ker = try_build_s42_noise_surprise_zscore_stateocc_nbocc_mix_gated_self2_scores_kernel()
            _kernel_cache["ker_ebf_s42"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s42 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        params = s42_noise_surprise_zscore_stateocc_nbocc_mix_gated_self2_params_from_env()
        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        hot_state = np.zeros((int(width) * int(height),), dtype=np.int32)
        rate_ema = np.zeros((1,), dtype=np.float64)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        if int(params.tau_rate_us) > 0:
            tau_rate_ticks = int(tb.us_to_ticks(int(params.tau_rate_us)))
        else:
            tau_rate_ticks = int(tau_ticks)
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            int(tau_rate_ticks),
            float(params.k_nbmix),
            last_ts,
            last_pol,
            hot_state,
            rate_ema,
            scores,
        )
        return scores

    if v in {
        "s43",
        "ebf_s43",
        "ebfs43",
        "surprise_occupancy_nbocc_mix_u2",
        "surprise_stateocc_nbocc_mix_u2",
    }:
        from myevs.denoise.ops.ebfopt_part2.s43_noise_surprise_zscore_stateocc_nbocc_mix_u2 import (
            s43_noise_surprise_zscore_stateocc_nbocc_mix_u2_params_from_env,
            try_build_s43_noise_surprise_zscore_stateocc_nbocc_mix_u2_scores_kernel,
        )

        ker = _kernel_cache.get("ker_ebf_s43")
        if ker is None:
            ker = try_build_s43_noise_surprise_zscore_stateocc_nbocc_mix_u2_scores_kernel()
            _kernel_cache["ker_ebf_s43"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s43 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        params = s43_noise_surprise_zscore_stateocc_nbocc_mix_u2_params_from_env()
        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        hot_state = np.zeros((int(width) * int(height),), dtype=np.int32)
        rate_ema = np.zeros((1,), dtype=np.float64)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        if int(params.tau_rate_us) > 0:
            tau_rate_ticks = int(tb.us_to_ticks(int(params.tau_rate_us)))
        else:
            tau_rate_ticks = int(tau_ticks)
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            int(tau_rate_ticks),
            float(params.k_nbmix),
            last_ts,
            last_pol,
            hot_state,
            rate_ema,
            scores,
        )
        return scores

    if v in {
        "s44",
        "ebf_s44",
        "ebfs44",
        "ebf_labelscore_selfocc_div_u2",
        "ebf_labelscore_selfocc_u2",
    }:
        from myevs.denoise.ops.ebfopt_part2.s44_ebf_labelscore_selfocc_div_u2 import (
            s44_ebf_labelscore_selfocc_div_u2_params_from_env,
            try_build_s44_ebf_labelscore_selfocc_div_u2_scores_kernel,
        )

        ker = _kernel_cache.get("ker_ebf_s44")
        if ker is None:
            ker = try_build_s44_ebf_labelscore_selfocc_div_u2_scores_kernel()
            _kernel_cache["ker_ebf_s44"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s44 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        params = s44_ebf_labelscore_selfocc_div_u2_params_from_env()
        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        hot_state = np.zeros((int(width) * int(height),), dtype=np.int32)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        if int(params.tau_rate_us) > 0:
            tau_rate_ticks = int(tb.us_to_ticks(int(params.tau_rate_us)))
        else:
            tau_rate_ticks = int(tau_ticks)
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            int(tau_rate_ticks),
            last_ts,
            last_pol,
            hot_state,
            scores,
        )
        return scores

    if v in {
        "s45",
        "ebf_s45",
        "ebfs45",
        "ebf_labelscore_selfocc_gate_div_u2",
        "ebf_labelscore_selfocc_gate_u2",
    }:
        from myevs.denoise.ops.ebfopt_part2.s45_ebf_labelscore_selfocc_gate_div_u2 import (
            s45_ebf_labelscore_selfocc_gate_div_u2_params_from_env,
            try_build_s45_ebf_labelscore_selfocc_gate_div_u2_scores_kernel,
        )

        ker = _kernel_cache.get("ker_ebf_s45")
        if ker is None:
            ker = try_build_s45_ebf_labelscore_selfocc_gate_div_u2_scores_kernel()
            _kernel_cache["ker_ebf_s45"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s45 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        params = s45_ebf_labelscore_selfocc_gate_div_u2_params_from_env()
        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        hot_state = np.zeros((int(width) * int(height),), dtype=np.int32)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        if int(params.tau_rate_us) > 0:
            tau_rate_ticks = int(tb.us_to_ticks(int(params.tau_rate_us)))
        else:
            tau_rate_ticks = int(tau_ticks)
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            int(tau_rate_ticks),
            float(params.u0),
            last_ts,
            last_pol,
            hot_state,
            scores,
        )
        return scores

    if v in {
        "s46",
        "ebf_s46",
        "ebfs46",
        "ebf_labelscore_selfocc_odds_div_v2",
        "ebf_labelscore_selfocc_odds",
    }:
        from myevs.denoise.ops.ebfopt_part2.s46_ebf_labelscore_selfocc_odds_div_v2 import (
            s46_ebf_labelscore_selfocc_odds_div_v2_params_from_env,
            try_build_s46_ebf_labelscore_selfocc_odds_div_v2_scores_kernel,
        )

        ker = _kernel_cache.get("ker_ebf_s46")
        if ker is None:
            ker = try_build_s46_ebf_labelscore_selfocc_odds_div_v2_scores_kernel()
            _kernel_cache["ker_ebf_s46"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s46 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        params = s46_ebf_labelscore_selfocc_odds_div_v2_params_from_env()
        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        hot_state = np.zeros((int(width) * int(height),), dtype=np.int32)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        if int(params.tau_rate_us) > 0:
            tau_rate_ticks = int(tb.us_to_ticks(int(params.tau_rate_us)))
        else:
            tau_rate_ticks = int(tau_ticks)
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            int(tau_rate_ticks),
            last_ts,
            last_pol,
            hot_state,
            scores,
        )
        return scores

    if v in {
        "s47",
        "ebf_s47",
        "ebfs47",
        "ebf_labelscore_selfocc_abn_div_u2",
        "ebf_labelscore_selfocc_abn_u2",
    }:
        from myevs.denoise.ops.ebfopt_part2.s47_ebf_labelscore_selfocc_abn_div_u2 import (
            s47_ebf_labelscore_selfocc_abn_div_u2_params_from_env,
            try_build_s47_ebf_labelscore_selfocc_abn_div_u2_scores_kernel,
        )

        ker = _kernel_cache.get("ker_ebf_s47")
        if ker is None:
            ker = try_build_s47_ebf_labelscore_selfocc_abn_div_u2_scores_kernel()
            _kernel_cache["ker_ebf_s47"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s47 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        params = s47_ebf_labelscore_selfocc_abn_div_u2_params_from_env()
        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        hot_state = np.zeros((int(width) * int(height),), dtype=np.int32)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        if int(params.tau_rate_us) > 0:
            tau_rate_ticks = int(tb.us_to_ticks(int(params.tau_rate_us)))
        else:
            tau_rate_ticks = int(tau_ticks)
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            int(tau_rate_ticks),
            last_ts,
            last_pol,
            hot_state,
            scores,
        )
        return scores

    if v in {
        "s48",
        "ebf_s48",
        "ebfs48",
        "ebf_labelscore_selfocc_polpersist_div_u2",
        "ebf_labelscore_selfocc_polpersist_u2",
    }:
        from myevs.denoise.ops.ebfopt_part2.s48_ebf_labelscore_selfocc_polpersist_div_u2 import (
            s48_ebf_labelscore_selfocc_polpersist_div_u2_params_from_env,
            try_build_s48_ebf_labelscore_selfocc_polpersist_div_u2_scores_kernel,
        )

        ker = _kernel_cache.get("ker_ebf_s48")
        if ker is None:
            ker = try_build_s48_ebf_labelscore_selfocc_polpersist_div_u2_scores_kernel()
            _kernel_cache["ker_ebf_s48"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s48 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        params = s48_ebf_labelscore_selfocc_polpersist_div_u2_params_from_env()
        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        hot_state = np.zeros((int(width) * int(height),), dtype=np.int32)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        if int(params.tau_rate_us) > 0:
            tau_rate_ticks = int(tb.us_to_ticks(int(params.tau_rate_us)))
        else:
            tau_rate_ticks = int(tau_ticks)
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            int(tau_rate_ticks),
            float(params.eta_toggle),
            last_ts,
            last_pol,
            hot_state,
            scores,
        )
        return scores

    if v in {
        "s49",
        "ebf_s49",
        "ebfs49",
        "ebf_labelscore_selfocc_bipolmax_div_u2",
        "ebf_labelscore_selfocc_bipolmax_u2",
    }:
        from myevs.denoise.ops.ebfopt_part2.s49_ebf_labelscore_selfocc_bipolmax_div_u2 import (
            s49_ebf_labelscore_selfocc_bipolmax_div_u2_params_from_env,
            try_build_s49_ebf_labelscore_selfocc_bipolmax_div_u2_scores_kernel,
        )

        ker = _kernel_cache.get("ker_ebf_s49")
        if ker is None:
            ker = try_build_s49_ebf_labelscore_selfocc_bipolmax_div_u2_scores_kernel()
            _kernel_cache["ker_ebf_s49"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s49 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        params = s49_ebf_labelscore_selfocc_bipolmax_div_u2_params_from_env()
        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        hot_pos = np.zeros((int(width) * int(height),), dtype=np.int32)
        hot_neg = np.zeros((int(width) * int(height),), dtype=np.int32)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        if int(params.tau_rate_us) > 0:
            tau_rate_ticks = int(tb.us_to_ticks(int(params.tau_rate_us)))
        else:
            tau_rate_ticks = int(tau_ticks)
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            int(tau_rate_ticks),
            last_ts,
            last_pol,
            hot_pos,
            hot_neg,
            scores,
        )
        return scores

    if v in {
        "s50",
        "ebf_s50",
        "ebfs50",
        "ebf_labelscore_selfocc_supportboost_div_u2",
        "ebf_labelscore_selfocc_supportboost_u2",
    }:
        from myevs.denoise.ops.ebfopt_part2.s50_ebf_labelscore_selfocc_supportboost_div_u2 import (
            s50_ebf_labelscore_selfocc_supportboost_div_u2_params_from_env,
            try_build_s50_ebf_labelscore_selfocc_supportboost_div_u2_scores_kernel,
        )

        ker = _kernel_cache.get("ker_ebf_s50")
        if ker is None:
            ker = try_build_s50_ebf_labelscore_selfocc_supportboost_div_u2_scores_kernel()
            _kernel_cache["ker_ebf_s50"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s50 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        params = s50_ebf_labelscore_selfocc_supportboost_div_u2_params_from_env()
        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        hot_state = np.zeros((int(width) * int(height),), dtype=np.int32)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        if int(params.tau_rate_us) > 0:
            tau_rate_ticks = int(tb.us_to_ticks(int(params.tau_rate_us)))
        else:
            tau_rate_ticks = int(tau_ticks)
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            int(tau_rate_ticks),
            float(params.beta),
            int(params.cnt0),
            last_ts,
            last_pol,
            hot_state,
            scores,
        )
        return scores

    if v in {
        "s51",
        "ebf_s51",
        "ebfs51",
        "ebf_labelscore_selfocc_supportboost_autobeta_div_u2",
        "ebf_labelscore_selfocc_supportboost_autobeta_u2",
    }:
        from myevs.denoise.ops.ebfopt_part2.s51_ebf_labelscore_selfocc_supportboost_autobeta_div_u2 import (
            try_build_s51_ebf_labelscore_selfocc_supportboost_autobeta_div_u2_scores_kernel,
        )

        ker = _kernel_cache.get("ker_ebf_s51")
        if ker is None:
            ker = try_build_s51_ebf_labelscore_selfocc_supportboost_autobeta_div_u2_scores_kernel()
            _kernel_cache["ker_ebf_s51"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s51 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        hot_state = np.zeros((int(width) * int(height),), dtype=np.int32)
        beta_state = np.zeros((1,), dtype=np.float32)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            last_ts,
            last_pol,
            hot_state,
            beta_state,
            scores,
        )
        return scores

    if v in {
        "s52",
        "ebf_s52",
        "ebfs52",
        "ebf_labelscore_selfocc_supportboost_autobeta_mixgateopp_div_u2",
        "ebf_labelscore_selfocc_supportboost_autobeta_mixgateopp_u2",
    }:
        from myevs.denoise.ops.ebfopt_part2.s52_ebf_labelscore_selfocc_supportboost_autobeta_mixgateopp_div_u2 import (
            try_build_s52_ebf_labelscore_selfocc_supportboost_autobeta_mixgateopp_div_u2_scores_kernel,
        )

        ker = _kernel_cache.get("ker_ebf_s52")
        if ker is None:
            ker = try_build_s52_ebf_labelscore_selfocc_supportboost_autobeta_mixgateopp_div_u2_scores_kernel()
            _kernel_cache["ker_ebf_s52"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s52 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        hot_state = np.zeros((int(width) * int(height),), dtype=np.int32)
        beta_state = np.zeros((1,), dtype=np.float32)
        mix_state = np.zeros((1,), dtype=np.float32)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            last_ts,
            last_pol,
            hot_state,
            beta_state,
            mix_state,
            scores,
        )
        return scores

    if v in {
        "s60",
        "ebf_s60",
        "ebfs60",
        "ebf_labelscore_dualtau_delta_selfocc_supportboost_autobeta_mixgateopp_div_u2",
        "ebf_labelscore_dualtau_delta_selfocc_supportboost_autobeta_mixgateopp_u2",
    }:
        from myevs.denoise.ops.ebfopt_part2.s60_ebf_labelscore_dualtau_delta_selfocc_supportboost_autobeta_mixgateopp_div_u2 import (
            try_build_s60_ebf_labelscore_dualtau_delta_selfocc_supportboost_autobeta_mixgateopp_div_u2_scores_kernel,
        )

        ker = _kernel_cache.get("ker_ebf_s60")
        if ker is None:
            ker = try_build_s60_ebf_labelscore_dualtau_delta_selfocc_supportboost_autobeta_mixgateopp_div_u2_scores_kernel()
            _kernel_cache["ker_ebf_s60"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s60 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        hot_state = np.zeros((int(width) * int(height),), dtype=np.int32)
        beta_state = np.zeros((1,), dtype=np.float32)
        mix_state = np.zeros((1,), dtype=np.float32)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            last_ts,
            last_pol,
            hot_state,
            beta_state,
            mix_state,
            scores,
        )
        return scores

    if v in {
        "s53",
        "ebf_s53",
        "ebfs53",
        "ebf_labelscore_selfocc_supportboost_autobeta_eventmixgateopp_div_u2",
        "ebf_labelscore_selfocc_supportboost_autobeta_eventmixgateopp_u2",
    }:
        from myevs.denoise.ops.ebfopt_part2.s53_ebf_labelscore_selfocc_supportboost_autobeta_eventmixgateopp_div_u2 import (
            try_build_s53_ebf_labelscore_selfocc_supportboost_autobeta_eventmixgateopp_div_u2_scores_kernel,
        )

        ker = _kernel_cache.get("ker_ebf_s53")
        if ker is None:
            ker = try_build_s53_ebf_labelscore_selfocc_supportboost_autobeta_eventmixgateopp_div_u2_scores_kernel()
            _kernel_cache["ker_ebf_s53"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s53 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        hot_state = np.zeros((int(width) * int(height),), dtype=np.int32)
        beta_state = np.zeros((1,), dtype=np.float32)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            last_ts,
            last_pol,
            hot_state,
            beta_state,
            scores,
        )
        return scores

    if v in {
        "s54",
        "ebf_s54",
        "ebfs54",
        "ebf_labelscore_selfocc_supportboost_autobeta_eventmixgateopp_root4_div_u2",
        "ebf_labelscore_selfocc_supportboost_autobeta_eventmixgateopp_root4_u2",
    }:
        from myevs.denoise.ops.ebfopt_part2.s54_ebf_labelscore_selfocc_supportboost_autobeta_eventmixgateopp_root4_div_u2 import (
            try_build_s54_ebf_labelscore_selfocc_supportboost_autobeta_eventmixgateopp_root4_div_u2_scores_kernel,
        )

        ker = _kernel_cache.get("ker_ebf_s54")
        if ker is None:
            ker = try_build_s54_ebf_labelscore_selfocc_supportboost_autobeta_eventmixgateopp_root4_div_u2_scores_kernel()
            _kernel_cache["ker_ebf_s54"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s54 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        hot_state = np.zeros((int(width) * int(height),), dtype=np.int32)
        beta_state = np.zeros((1,), dtype=np.float32)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            last_ts,
            last_pol,
            hot_state,
            beta_state,
            scores,
        )
        return scores

    if v in {
        "s55",
        "ebf_s55",
        "ebfs55",
        "ebf_labelscore_selfocc_supportboost_autobeta_eventmixgateopp_supportlerp_div_u2",
        "ebf_labelscore_selfocc_supportboost_autobeta_eventmixgateopp_supportlerp_u2",
    }:
        from myevs.denoise.ops.ebfopt_part2.s55_ebf_labelscore_selfocc_supportboost_autobeta_eventmixgateopp_supportlerp_div_u2 import (
            try_build_s55_ebf_labelscore_selfocc_supportboost_autobeta_eventmixgateopp_supportlerp_div_u2_scores_kernel,
        )

        ker = _kernel_cache.get("ker_ebf_s55")
        if ker is None:
            ker = try_build_s55_ebf_labelscore_selfocc_supportboost_autobeta_eventmixgateopp_supportlerp_div_u2_scores_kernel()
            _kernel_cache["ker_ebf_s55"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s55 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        hot_state = np.zeros((int(width) * int(height),), dtype=np.int32)
        beta_state = np.zeros((1,), dtype=np.float32)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            last_ts,
            last_pol,
            hot_state,
            beta_state,
            scores,
        )
        return scores

    if v in {"s30", "ebf_s30", "ebfs30", "surprise_localrate", "surprise_localrate_max", "s28_localrate"}:
        from myevs.denoise.ops.ebfopt_part2.s30_surprise_zscore_localrate_max import (
            s30_surprise_zscore_localrate_max_params_from_env,
            try_build_s30_surprise_zscore_localrate_max_scores_kernel,
        )

        ker = _kernel_cache.get("ker_ebf_s30")
        if ker is None:
            ker = try_build_s30_surprise_zscore_localrate_max_scores_kernel()
            _kernel_cache["ker_ebf_s30"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s30 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        params = s30_surprise_zscore_localrate_max_params_from_env()
        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        rate_ema = np.zeros((1,), dtype=np.float64)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        if int(params.tau_rate_us) > 0:
            tau_rate_ticks = int(tb.us_to_ticks(int(params.tau_rate_us)))
        else:
            tau_rate_ticks = int(tau_ticks)
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            int(tau_rate_ticks),
            last_ts,
            last_pol,
            rate_ema,
            scores,
        )
        return scores

    if v in {"s31", "ebf_s31", "ebfs31", "surprise_polbias", "surprise_z_polbias", "s28_polbias"}:
        from myevs.denoise.ops.ebfopt_part2.s31_noise_surprise_zscore_polbias import (
            s31_noise_surprise_zscore_polbias_params_from_env,
            try_build_s31_noise_surprise_zscore_polbias_scores_kernel,
        )

        ker = _kernel_cache.get("ker_ebf_s31")
        if ker is None:
            ker = try_build_s31_noise_surprise_zscore_polbias_scores_kernel()
            _kernel_cache["ker_ebf_s31"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s31 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        params = s31_noise_surprise_zscore_polbias_params_from_env()
        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        rate_pol_ema = np.zeros((2,), dtype=np.float64)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        if int(params.tau_rate_us) > 0:
            tau_rate_ticks = int(tb.us_to_ticks(int(params.tau_rate_us)))
        else:
            tau_rate_ticks = int(tau_ticks)
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            int(tau_rate_ticks),
            last_ts,
            last_pol,
            rate_pol_ema,
            scores,
        )
        return scores

    if v in {"s32", "ebf_s32", "ebfs32", "surprise_blockrate", "surprise_blockrate_max", "s28_blockrate"}:
        from myevs.denoise.ops.ebfopt_part2.s32_noise_surprise_zscore_blockrate_max import (
            s32_noise_surprise_zscore_blockrate_max_params_from_env,
            try_build_s32_noise_surprise_zscore_blockrate_max_scores_kernel,
        )

        ker = _kernel_cache.get("ker_ebf_s32")
        if ker is None:
            ker = try_build_s32_noise_surprise_zscore_blockrate_max_scores_kernel()
            _kernel_cache["ker_ebf_s32"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s32 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        params = s32_noise_surprise_zscore_blockrate_max_params_from_env()
        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        rate_ema = np.zeros((1,), dtype=np.float64)

        # Fixed block size must match s32 kernel.
        bw = 32
        bh = 32
        nbx = (int(width) + bw - 1) // bw
        nby = (int(height) + bh - 1) // bh
        nblocks = int(nbx * nby)
        block_last_t = np.full((nblocks,), -1, dtype=np.int64)
        block_rate_ema = np.zeros((nblocks,), dtype=np.float64)

        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        if int(params.tau_rate_us) > 0:
            tau_rate_ticks = int(tb.us_to_ticks(int(params.tau_rate_us)))
        else:
            tau_rate_ticks = int(tau_ticks)
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            int(tau_rate_ticks),
            last_ts,
            last_pol,
            rate_ema,
            block_last_t,
            block_rate_ema,
            scores,
        )
        return scores

    if v in {"s33", "ebf_s33", "ebfs33", "surprise_abnhot", "surprise_abnhot_penalty", "s28_abnhot_penalty"}:
        from myevs.denoise.ops.ebfopt_part2.s33_noise_surprise_zscore_abnhot_penalty import (
            s33_noise_surprise_zscore_abnhot_penalty_params_from_env,
            try_build_s33_noise_surprise_zscore_abnhot_penalty_scores_kernel,
        )

        ker = _kernel_cache.get("ker_ebf_s33")
        if ker is None:
            ker = try_build_s33_noise_surprise_zscore_abnhot_penalty_scores_kernel()
            _kernel_cache["ker_ebf_s33"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s33 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        params = s33_noise_surprise_zscore_abnhot_penalty_params_from_env()
        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        rate_ema = np.zeros((1,), dtype=np.float64)
        acc_neg = np.zeros((int(width) * int(height),), dtype=np.int32)
        acc_pos = np.zeros((int(width) * int(height),), dtype=np.int32)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        if int(params.tau_rate_us) > 0:
            tau_rate_ticks = int(tb.us_to_ticks(int(params.tau_rate_us)))
        else:
            tau_rate_ticks = int(tau_ticks)
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            int(tau_rate_ticks),
            float(params.beta),
            last_ts,
            last_pol,
            rate_ema,
            acc_neg,
            acc_pos,
            scores,
        )
        return scores

    if v in {"s34", "ebf_s34", "ebfs34", "surprise_selfrate", "surprise_selfrate_max", "s28_selfrate"}:
        from myevs.denoise.ops.ebfopt_part2.s34_noise_surprise_zscore_selfrate_max import (
            s34_noise_surprise_zscore_selfrate_max_params_from_env,
            try_build_s34_noise_surprise_zscore_selfrate_max_scores_kernel,
        )

        ker = _kernel_cache.get("ker_ebf_s34")
        if ker is None:
            ker = try_build_s34_noise_surprise_zscore_selfrate_max_scores_kernel()
            _kernel_cache["ker_ebf_s34"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s34 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        params = s34_noise_surprise_zscore_selfrate_max_params_from_env()
        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        rate_ema = np.zeros((1,), dtype=np.float64)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        if int(params.tau_rate_us) > 0:
            tau_rate_ticks = int(tb.us_to_ticks(int(params.tau_rate_us)))
        else:
            tau_rate_ticks = int(tau_ticks)
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            int(tau_rate_ticks),
            float(params.k_self),
            last_ts,
            last_pol,
            rate_ema,
            scores,
        )
        return scores

    if v in {"s29", "ebf_s29", "ebfs29", "polarity_surprise", "pol_surprise", "surprise_pol_z"}:
        from myevs.denoise.ops.ebfopt_part2.s29_polarity_surprise_zscore import (
            try_build_s29_polarity_surprise_zscore_scores_kernel,
        )

        ker = _kernel_cache.get("ker_ebf_s29")
        if ker is None:
            ker = try_build_s29_polarity_surprise_zscore_scores_kernel()
            _kernel_cache["ker_ebf_s29"] = ker

        if ker is None:
            raise SystemExit(
                "EBF Part2 s29 requires numba kernel, but numba is unavailable. "
                "Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
            )

        last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
        last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
        tau_ticks = int(tb.us_to_ticks(int(tau_us)))
        ker(
            ev.t,
            ev.x,
            ev.y,
            ev.p,
            int(width),
            int(height),
            int(radius_px),
            int(tau_ticks),
            last_ts,
            last_pol,
            scores,
        )
        return scores

    if v in {"ebfv10", "v10", "spatialw_linear", "ebf_v10"}:
        ker = _kernel_cache.get("ker_ebf_v10")
        if ker is None:
            from myevs.denoise.ops.ebf_v10_spatialw_linear import try_build_v10_spatialw_linear_scores_kernel

            ker = try_build_v10_spatialw_linear_scores_kernel()
            _kernel_cache["ker_ebf_v10"] = ker

        lut_key = f"spatial_lut_r{int(radius_px)}"
        lut = _kernel_cache.get(lut_key)
        if lut is None:
            from myevs.denoise.ops.ebf_v10_spatialw_linear import build_spatial_lut

            lut = build_spatial_lut(int(radius_px))
            _kernel_cache[lut_key] = lut

        if ker is not None:
            last_ts = np.zeros((int(width) * int(height),), dtype=np.uint64)
            last_pol = np.zeros((int(width) * int(height),), dtype=np.int8)
            tau_ticks = int(tb.us_to_ticks(int(tau_us)))
            ker(
                ev.t,
                ev.x,
                ev.y,
                ev.p,
                int(width),
                int(height),
                int(radius_px),
                int(tau_ticks),
                lut,
                last_ts,
                last_pol,
                scores,
            )
            return scores

    if v not in {
        "ebf",
        "v0",
        "baseline",
        "ebfv10",
        "v10",
        "spatialw_linear",
        "ebf_v10",
        "s1",
        "ebf_s1",
        "ebfs1",
        "dircoh",
        "directional_coherence",
        "s2",
        "ebf_s2",
        "ebfs2",
        "coh_gate",
        "coherence_gate",
        "s3",
        "ebf_s3",
        "ebfs3",
        "softgate",
        "coh_softgate",
        "s4",
        "ebf_s4",
        "ebfs4",
        "residual_gate",
        "resultant_gate",
        "s5",
        "ebf_s5",
        "ebfs5",
        "elliptic_spatialw",
        "elliptic",
        "s6",
        "ebf_s6",
        "ebfs6",
        "timecoh_gate",
        "time_gate",
        "s7",
        "ebf_s7",
        "ebfs7",
        "plane_gate",
        "plane_residual_gate",
        "s8",
        "ebf_s8",
        "ebfs8",
        "plane_r2_gate",
        "plane_explained_gate",
        "plane_r2",
        "s9",
        "ebf_s9",
        "ebfs9",
        "refractory_gate",
        "burst_gate",
        "self_refractory",
        "s10",
        "ebf_s10",
        "ebfs10",
        "hotpixel_rate_gate",
        "rate_gate",
        "hotpixel_gate",
        "s11",
        "ebf_s11",
        "ebfs11",
        "relative_hotness_gate",
        "relative_hotness",
        "relhot",
        "s12",
        "ebf_s12",
        "ebfs12",
        "hotness_zscore_gate",
        "zscore_hotness",
        "zgate",
        "s13",
        "ebf_s13",
        "ebfs13",
        "crosspol_support_gate",
        "crosspol_gate",
        "crosspol_support",
        "s14",
        "ebf_s14",
        "ebfs14",
        "crosspol_boost",
        "crosspol_boost_score",
        "crosspol_add",
        "s15",
        "ebf_s15",
        "ebfs15",
        "flip_flicker_gate",
        "polarity_flip_gate",
        "flicker",
        "s16",
        "ebf_s16",
        "ebfs16",
        "s14_hotness_clamp",
        "hotness_clamp",
        "s17",
        "ebf_s17",
        "ebfs17",
        "crosspol_spread_boost",
        "spread_boost",
    }:
        raise SystemExit(
            f"unknown variant: {variant!r}. choices: ebf | EBFV10 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | s10 | s11 | s12 | s13 | s14 | s15 | s16 | s17 | s22 | s23 | s24"
        )

    from myevs.denoise.ops.base import Dims
    from myevs.denoise.ops.ebf import EbfOp
    from myevs.denoise.ops.ebf_v10_spatialw_linear import EbfV10SpatialWLinearOp
    from myevs.denoise.types import DenoiseConfig

    is_v10 = v in {"ebfv10", "v10", "spatialw_linear", "ebf_v10"}
    cfg = DenoiseConfig(
        method=("ebf_v10" if is_v10 else "ebf"),
        pipeline=None,
        time_window_us=int(tau_us),
        radius_px=int(radius_px),
        min_neighbors=0.0,
        refractory_us=0,
        show_on=True,
        show_off=True,
    )
    op = (
        EbfV10SpatialWLinearOp(Dims(width=int(width), height=int(height)), cfg, tb)
        if is_v10
        else EbfOp(Dims(width=int(width), height=int(height)), cfg, tb)
    )
    n = int(ev.t.shape[0])
    for i in range(n):
        scores[i] = float(op.score(int(ev.x[i]), int(ev.y[i]), int(ev.p[i]), int(ev.t[i])))
        if (i + 1) % 500000 == 0:
            print(f"scored: {i+1}/{n} (r={radius_px}, tau_us={tau_us})")
    return scores


def _read_existing_tags(out_path: str) -> set[str]:
    if not os.path.exists(out_path):
        return set()
    try:
        with open(out_path, "r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            if r.fieldnames is None:
                return set()
            tags: set[str] = set()
            for row in r:
                t = (row.get("tag") or "").strip()
                auc = (row.get("auc") or "").strip()
                if t and auc:
                    tags.add(t)
            return tags
    except Exception:
        return set()


ROC_HEADER = [
    "tag",
    "method",
    "param",
    "value",
    "roc_convention",
    "match_us",
    "events_total",
    "signal_total",
    "noise_total",
    "events_kept",
    "signal_kept",
    "noise_kept",
    "tp",
    "fp",
    "tn",
    "fn",
    "tpr",
    "fpr",
    "precision",
    "accuracy",
    "f1",
    "auc",
    "esr_mean",
    "aocc",
]


def _scale_aocc(v: float | None) -> float | None:
    # Since 2026-04-09: aocc_from_xyt() already returns values scaled to MYEVS_AOCC_UNIT (default 1e7).
    # Keep this hook for backward compatibility.
    if v is None:
        return None
    return float(v)


def _best_f1_index(
    thresholds: np.ndarray,
    tp: np.ndarray,
    fp: np.ndarray,
    *,
    pos: int,
    neg: int,
) -> int:
    """Pick threshold index that maximizes F1.

    Tie-breakers: higher TPR, then higher precision, then lower FPR.
    """

    best_i = 0
    best_key = (-1.0, 0.0, 0.0, -1.0)
    p = int(pos)
    n = int(neg)

    for i in range(int(thresholds.shape[0])):
        tp_i = int(tp[i])
        fp_i = int(fp[i])
        tpr = (tp_i / p) if p > 0 else 0.0
        fpr = (fp_i / n) if n > 0 else 0.0
        prec_den = tp_i + fp_i
        precision = (tp_i / prec_den) if prec_den > 0 else 0.0
        f1_den = precision + tpr
        f1 = (2.0 * precision * tpr / f1_den) if f1_den > 0 else 0.0
        key = (float(f1), float(tpr), float(precision), -float(fpr))
        if key > best_key:
            best_key = key
            best_i = int(i)
    return int(best_i)


def _roc_points_from_scores(
    y_true01: np.ndarray,
    y_score: np.ndarray,
    *,
    max_points: int,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (auc, thresholds, tp, fp, fpr, tpr) using standard ROC construction.

    Convention here matches myevs.metrics.roc_score_label:
    - Sort by score descending
    - Predicted positive for threshold thr: score >= thr
    - Adds the conventional starting point (0,0) at threshold=+inf
    """

    y = np.asarray(y_true01).astype(np.int8, copy=False)
    s = np.asarray(y_score).astype(np.float64, copy=False)
    if y.ndim != 1 or s.ndim != 1 or y.shape[0] != s.shape[0]:
        raise ValueError("y_true and y_score must be 1D arrays of the same length")

    n = int(y.shape[0])
    pos = int(np.sum(y))
    neg = int(n - pos)
    if n == 0 or pos == 0 or neg == 0:
        thr = np.asarray([np.inf, -np.inf], dtype=np.float64)
        tp = np.asarray([0, pos], dtype=np.int64)
        fp = np.asarray([0, neg], dtype=np.int64)
        fpr = np.asarray([0.0, 1.0], dtype=np.float64)
        tpr = np.asarray([0.0, 1.0], dtype=np.float64)
        auc = 0.0
        return auc, thr, tp, fp, fpr, tpr

    order = np.argsort(-s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order]

    tp_cum = np.cumsum(y_sorted, dtype=np.int64)
    fp_cum = np.cumsum(1 - y_sorted, dtype=np.int64)

    change = np.empty((n,), dtype=bool)
    change[:-1] = s_sorted[:-1] != s_sorted[1:]
    change[-1] = True
    idx = np.nonzero(change)[0]

    tp_u = tp_cum[idx]
    fp_u = fp_cum[idx]
    thr_u = s_sorted[idx].astype(np.float64, copy=False)

    # add start point (0,0) at +inf
    tp_u = np.concatenate([np.asarray([0], dtype=np.int64), tp_u])
    fp_u = np.concatenate([np.asarray([0], dtype=np.int64), fp_u])
    thr_u = np.concatenate([np.asarray([np.inf], dtype=np.float64), thr_u])

    tpr_u = tp_u.astype(np.float64) / float(pos)
    fpr_u = fp_u.astype(np.float64) / float(neg)

    # exact AUC from full curve
    auc = float((getattr(np, "trapezoid", None) or np.trapz)(y=tpr_u, x=fpr_u))

    # downsample for output if needed
    if max_points is not None and int(max_points) > 0 and fpr_u.shape[0] > int(max_points):
        m = int(max_points)
        keep = np.unique(
            np.concatenate(
                [
                    np.asarray([0, fpr_u.shape[0] - 1], dtype=np.int64),
                    np.linspace(0, fpr_u.shape[0] - 1, num=m, dtype=np.int64),
                ]
            )
        )
        thr_u = thr_u[keep]
        tp_u = tp_u[keep]
        fp_u = fp_u[keep]
        fpr_u = fpr_u[keep]
        tpr_u = tpr_u[keep]

    return auc, thr_u, tp_u, fp_u, fpr_u, tpr_u


def _write_roc_rows(
    writer: csv.writer,
    *,
    tag: str,
    method: str,
    param: str,
    thresholds: np.ndarray,
    tp: np.ndarray,
    fp: np.ndarray,
    pos: int,
    neg: int,
    auc: float,
    esr_mean: float | None,
    esr_at_index: int,
    aocc: float | None,
    aocc_at_index: int,
) -> None:
    n = int(pos + neg)
    aocc_scaled = _scale_aocc(aocc)
    for i in range(int(thresholds.shape[0])):
        thr = float(thresholds[i])
        tp_i = int(tp[i])
        fp_i = int(fp[i])
        tn_i = int(neg - fp_i)
        fn_i = int(pos - tp_i)

        events_kept = tp_i + fp_i
        signal_kept = tp_i
        noise_kept = fp_i

        tpr = (tp_i / pos) if pos > 0 else 0.0
        fpr = (fp_i / neg) if neg > 0 else 0.0

        prec_den = tp_i + fp_i
        precision = (tp_i / prec_den) if prec_den > 0 else 0.0
        acc = ((tp_i + tn_i) / n) if n > 0 else 0.0
        f1_den = precision + tpr
        f1 = (2.0 * precision * tpr / f1_den) if f1_den > 0 else 0.0

        writer.writerow(
            [
                tag,
                method,
                param,
                thr,
                "paper",
                0,
                n,
                int(pos),
                int(neg),
                int(events_kept),
                int(signal_kept),
                int(noise_kept),
                int(tp_i),
                int(fp_i),
                int(tn_i),
                int(fn_i),
                float(tpr),
                float(fpr),
                float(precision),
                float(acc),
                float(f1),
                float(auc),
                ("" if int(i) != int(esr_at_index) or esr_mean is None else float(esr_mean)),
                ("" if int(i) != int(aocc_at_index) or aocc_scaled is None else float(aocc_scaled)),
            ]
        )


def _f1_at_index(
    thresholds: np.ndarray,
    tp: np.ndarray,
    fp: np.ndarray,
    *,
    pos: int,
    neg: int,
    i: int,
) -> float:
    _ = float(thresholds[int(i)])
    tp_i = int(tp[int(i)])
    fp_i = int(fp[int(i)])
    tpr = (tp_i / pos) if int(pos) > 0 else 0.0
    prec_den = tp_i + fp_i
    precision = (tp_i / prec_den) if prec_den > 0 else 0.0
    f1_den = precision + tpr
    return float((2.0 * precision * tpr / f1_den) if f1_den > 0 else 0.0)


def _patch_esr_mean_in_roc_csv(
    csv_path: str,
    *,
    esr_targets: dict[str, tuple[int, float]],
) -> None:
    """Patch esr_mean values in an existing ROC CSV.

    esr_targets maps: tag -> (esr_at_index_within_tag_rows, esr_mean_value)
    """

    if not esr_targets:
        return

    tmp_path = csv_path + ".tmp"
    tag_row_i: dict[str, int] = {}

    with open(csv_path, "r", newline="", encoding="utf-8") as fin, open(
        tmp_path, "w", newline="", encoding="utf-8"
    ) as fout:
        r = csv.DictReader(fin)
        if r.fieldnames is None:
            return
        fieldnames = list(r.fieldnames)
        w = csv.DictWriter(fout, fieldnames=fieldnames)
        w.writeheader()

        for row in r:
            tag = (row.get("tag") or "").strip()
            if not tag:
                w.writerow(row)
                continue

            i = int(tag_row_i.get(tag, 0))
            tag_row_i[tag] = i + 1

            tgt = esr_targets.get(tag)
            if tgt is not None:
                esr_i, esr_v = tgt
                if int(i) == int(esr_i):
                    row["esr_mean"] = str(float(esr_v))
            w.writerow(row)

    os.replace(tmp_path, csv_path)


def _patch_aocc_in_roc_csv(
    csv_path: str,
    *,
    aocc_targets: dict[str, tuple[int, float]],
) -> None:
    """Patch aocc values in an existing ROC CSV.

    aocc_targets maps: tag -> (aocc_at_index_within_tag_rows, aocc_value)
    """

    if not aocc_targets:
        return

    tmp_path = csv_path + ".tmp"
    tag_row_i: dict[str, int] = {}

    with open(csv_path, "r", newline="", encoding="utf-8") as fin, open(
        tmp_path, "w", newline="", encoding="utf-8"
    ) as fout:
        r = csv.DictReader(fin)
        if r.fieldnames is None:
            return
        fieldnames = list(r.fieldnames)
        w = csv.DictWriter(fout, fieldnames=fieldnames)
        w.writeheader()

        for row in r:
            tag = (row.get("tag") or "").strip()
            if not tag:
                w.writerow(row)
                continue

            i = int(tag_row_i.get(tag, 0))
            tag_row_i[tag] = i + 1

            tgt = aocc_targets.get(tag)
            if tgt is not None:
                aocc_i, aocc_v = tgt
                if int(i) == int(aocc_i):
                    scaled = _scale_aocc(float(aocc_v))
                    if scaled is not None:
                        row["aocc"] = str(float(scaled))
            w.writerow(row)

    os.replace(tmp_path, csv_path)


def _plot_roc_png(*, csv_path: str, png_path: str, title: str) -> None:
    try:
        import matplotlib  # type: ignore

        matplotlib.use("Agg", force=True)  # avoid Qt backend crashes on Windows
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        print(f"skip plot (matplotlib unavailable): {type(e).__name__}: {e}")
        return

    # Load ROC CSV and plot by tag
    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    if not rows:
        print(f"skip plot (empty csv): {csv_path}")
        return

    tags = sorted({(row.get("tag") or "") for row in rows if (row.get("tag") or "")})
    if not tags:
        print(f"skip plot (no tags): {csv_path}")
        return

    # Keep only top-3 tau curves per window diameter s, based on per-tag AUC.
    by_tag_auc: dict[str, float] = {}
    for row in rows:
        t = (row.get("tag") or "").strip()
        if not t or t in by_tag_auc:
            continue
        a = (row.get("auc") or "").strip()
        if not a:
            continue
        try:
            by_tag_auc[t] = float(a)
        except Exception:
            continue

    s_groups: dict[str, list[tuple[str, float]]] = {}
    for t, a in by_tag_auc.items():
        m = re.search(r"_s(\d+)_tau(\d+)", t)
        if not m:
            continue
        s_key = m.group(1)
        s_groups.setdefault(s_key, []).append((t, a))

    keep: set[str] = set()
    for s_key, arr in s_groups.items():
        arr_sorted = sorted(arr, key=lambda x: x[1], reverse=True)
        for t, _a in arr_sorted[:3]:
            keep.add(t)

    if keep:
        tags = [t for t in tags if t in keep]

    def _legend_label(tag: str) -> str:
        # tag example: ebf_labelscore_s5_tau16000
        if tag.startswith("ebf_v10"):
            prefix = "ebf_v10"
        # NOTE: Order matters. Put longer/multi-digit prefixes first.
        elif tag.startswith("ebf_s34"):
            prefix = "ebf_s34"
        elif tag.startswith("ebf_s33"):
            prefix = "ebf_s33"
        elif tag.startswith("ebf_s32"):
            prefix = "ebf_s32"
        elif tag.startswith("ebf_s31"):
            prefix = "ebf_s31"
        elif tag.startswith("ebf_s30"):
            prefix = "ebf_s30"
        elif tag.startswith("ebf_s29"):
            prefix = "ebf_s29"
        elif tag.startswith("ebf_s28"):
            prefix = "ebf_s28"
        elif tag.startswith("ebf_s27"):
            prefix = "ebf_s27"
        elif tag.startswith("ebf_s26"):
            prefix = "ebf_s26"
        elif tag.startswith("ebf_s25"):
            prefix = "ebf_s25"
        elif tag.startswith("ebf_s24"):
            prefix = "ebf_s24"
        elif tag.startswith("ebf_s23"):
            prefix = "ebf_s23"
        elif tag.startswith("ebf_s22"):
            prefix = "ebf_s22"
        elif tag.startswith("ebf_s21"):
            prefix = "ebf_s21"
        elif tag.startswith("ebf_s20"):
            prefix = "ebf_s20"
        elif tag.startswith("ebf_s19"):
            prefix = "ebf_s19"
        elif tag.startswith("ebf_s18"):
            prefix = "ebf_s18"
        elif tag.startswith("ebf_s17"):
            prefix = "ebf_s17"
        elif tag.startswith("ebf_s16"):
            prefix = "ebf_s16"
        elif tag.startswith("ebf_s15"):
            prefix = "ebf_s15"
        elif tag.startswith("ebf_s14"):
            prefix = "ebf_s14"
        elif tag.startswith("ebf_s13"):
            prefix = "ebf_s13"
        elif tag.startswith("ebf_s12"):
            prefix = "ebf_s12"
        elif tag.startswith("ebf_s11"):
            prefix = "ebf_s11"
        elif tag.startswith("ebf_s10"):
            prefix = "ebf_s10"
        elif tag.startswith("ebf_s9"):
            prefix = "ebf_s9"
        elif tag.startswith("ebf_s8"):
            prefix = "ebf_s8"
        elif tag.startswith("ebf_s7"):
            prefix = "ebf_s7"
        elif tag.startswith("ebf_s6"):
            prefix = "ebf_s6"
        elif tag.startswith("ebf_s5"):
            prefix = "ebf_s5"
        elif tag.startswith("ebf_s4"):
            prefix = "ebf_s4"
        elif tag.startswith("ebf_s3"):
            prefix = "ebf_s3"
        elif tag.startswith("ebf_s2"):
            prefix = "ebf_s2"
        elif tag.startswith("ebf_s1"):
            prefix = "ebf_s1"
        else:
            prefix = "ebf"
        m = re.search(r"_s(\d+)_tau(\d+)", tag)
        s_s = m.group(1) if m else "?"
        tau_s = m.group(2) if m else "?"

        auc_val: float | None = None
        for row in rows:
            if row.get("tag") != tag:
                continue
            a = (row.get("auc") or "").strip()
            if a:
                try:
                    auc_val = float(a)
                except Exception:
                    auc_val = None
                break

        if auc_val is None:
            return f"{prefix}_s{s_s} tau{tau_s}"
        return f"{prefix}_s{s_s} tau{tau_s} (AUC={auc_val:.4f})"

    plt.figure(figsize=(8, 6), dpi=160)
    for tag in tags:
        fpr = [float(row["fpr"]) for row in rows if row.get("tag") == tag]
        tpr = [float(row["tpr"]) for row in rows if row.get("tag") == tag]
        if not fpr:
            continue
        plt.plot(fpr, tpr, linewidth=1.0, label=_legend_label(tag))

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(title)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(False)
    plt.legend(fontsize=7, ncol=1)
    os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep EBF AUC(score+label) on labeled .npy (no CLI modifications).")
    ap.add_argument("--max-events", type=int, default=int(os.environ.get("EBF_MAX_EVENTS", "0")), help="0=all")
    ap.add_argument("--out-dir", default="data/ED24/myPedestrain_06/EBF_Part2", help="output directory")
    ap.add_argument(
        "--variant",
        default=str(os.environ.get("MYEVS_EBF_VARIANT", "ebf")),
        help=(
            "Score variant: ebf (baseline) | EBFV10 (spatial distance linear weight) | "
            "s1 (directional coherence) | s2 (coherence-gated penalty) | "
            "s3 (smooth coh gate) | s4 (resultant gate) | s5 (elliptic spatialw) | s6 (time-coh gate) | "
            "s7 (plane residual gate) | s8 (plane R2/explained-variance gate) | s9 (same-pixel refractory/burst gate) | s10 (hotpixel leaky-rate gate) | s11 (relative-hotness anomaly gate) | s12 (hotness z-score anomaly gate) | s13 (cross-polarity support gate) | s14 (cross-polarity boost) | s24 (s14 + burstiness gate) | s18 (no polarity check) | s19 (evidence fusion q8)"
            " | s20 (pol-hotness fusion q8) | s21 (bi-pol hotness mix fusion q8)"
            " | s22 (same-pixel any-polarity burst gate, no extra state)"
                " | s27 (relative abnormal hotness fusion q8)"
                " | s28 (noise-model surprise z-score)"
                " | s35 (s28 + pixel-state-conditioned adaptive null)"
                " | s36 (s28 + state-occupancy-conditioned adaptive null; fewer hyperparams)"
                " | s37 (s36 + 3-state occupancy->rate mapping)"
                " | s38 (s36 + neighborhood-occupancy fusion)"
                " | s29 (local polarity-surprise z-score)"
                " | s30 (s28 + local-rate correction)"
                " | s31 (s28 + polarity-bias correction)"
                " | s32 (s28 + block-rate max correction)"
                " | s33 (s28 + abnormal-hotness penalty)"
                " | s34 (s28 + pixel self-rate max correction)"
            " | s23 (feature+logit fusion; learnable linear score)"
            " | s25 (s14 + same-pixel refractory gate)"
            " | s26 (activity-normalized hotness fusion q8)"
        ),
    )
    ap.add_argument(
        "--plot-only",
        action="store_true",
        help="Only regenerate PNG from existing ROC CSV (useful when CSV is open/locked on Windows).",
    )
    ap.add_argument("--width", type=int, default=346)
    ap.add_argument("--height", type=int, default=260)
    ap.add_argument("--tick-ns", type=float, default=1000.0)
    ap.add_argument("--s-list", default="3,5,7,9")
    ap.add_argument("--tau-us-list", default="8000,16000,32000,64000,128000,256000,512000,1024000")
    ap.add_argument("--roc-max-points", type=int, default=5000)

    ap.add_argument(
        "--esr-mode",
        default=str(os.environ.get("MYEVS_ESR_MODE", "best")),
        choices=["best", "all", "off"],
        help=(
            "MESR/ESR compute mode for esr_mean column: "
            "best=only compute at best-AUC tag and best-F1 tag per env; "
            "all=compute at best-F1 point for every tag; off=skip ESR entirely."
        ),
    )
    ap.add_argument(
        "--aocc-mode",
        default=str(os.environ.get("MYEVS_AOCC_MODE", "best")),
        choices=["best", "all", "off"],
        help=(
            "AOCC compute mode for aocc column: "
            "best=only compute at best-AUC tag and best-F1 tag per env; "
            "all=compute at best-F1 point for every tag; off=skip AOCC entirely."
        ),
    )

    # Part2 hyperparam sweeps (optional; only used when --variant matches)
    ap.add_argument("--s2-coh-thr-list", default="", help="Optional, e.g. '0.3,0.4,0.5'")
    ap.add_argument("--s2-raw-thr-list", default="", help="Optional, e.g. '2,3,4'")
    ap.add_argument("--s2-gamma-list", default="", help="Optional, e.g. '0.5,1,2'")

    ap.add_argument("--s3-coh-thr-list", default="")
    ap.add_argument("--s3-raw-thr-list", default="")
    ap.add_argument("--s3-gamma-list", default="")
    ap.add_argument("--s3-alpha-list", default="")
    ap.add_argument("--s3-k-raw-list", default="")
    ap.add_argument("--s3-k-coh-list", default="")

    ap.add_argument("--s4-align-thr-list", default="")
    ap.add_argument("--s4-raw-thr-list", default="")
    ap.add_argument("--s4-gamma-list", default="")

    ap.add_argument("--s5-ax-list", default="")
    ap.add_argument("--s5-ay-list", default="")
    ap.add_argument("--s5-theta-deg-list", default="")

    ap.add_argument("--s6-timecoh-thr-list", default="")
    ap.add_argument("--s6-raw-thr-list", default="")
    ap.add_argument("--s6-gamma-list", default="")

    ap.add_argument("--s7-sigma-thr-list", default="")
    ap.add_argument("--s7-raw-thr-list", default="")
    ap.add_argument("--s7-gamma-list", default="")
    ap.add_argument("--s7-min-pts-list", default="")

    ap.add_argument("--s8-r2-thr-list", default="")
    ap.add_argument("--s8-raw-thr-list", default="")
    ap.add_argument("--s8-gamma-list", default="")
    ap.add_argument("--s8-min-pts-list", default="")

    ap.add_argument("--s9-dt-thr-list", default="")
    ap.add_argument("--s9-raw-thr-list", default="")
    ap.add_argument("--s9-gamma-list", default="")

    ap.add_argument("--s22-dt-thr-us-list", default="")
    ap.add_argument("--s22-raw-thr-list", default="")
    ap.add_argument("--s22-gamma-list", default="")

    ap.add_argument("--s10-acc-thr-list", default="")
    ap.add_argument("--s10-raw-thr-list", default="")
    ap.add_argument("--s10-gamma-list", default="")

    ap.add_argument("--s11-acc-thr-list", default="")
    ap.add_argument("--s11-ratio-thr-list", default="")
    ap.add_argument("--s11-raw-thr-list", default="")
    ap.add_argument("--s11-gamma-list", default="")

    ap.add_argument("--s12-acc-thr-list", default="")
    ap.add_argument("--s12-z-thr-list", default="")
    ap.add_argument("--s12-raw-thr-list", default="")
    ap.add_argument("--s12-gamma-list", default="")

    ap.add_argument("--s13-bal-thr-list", default="")
    ap.add_argument("--s13-raw-thr-list", default="")
    ap.add_argument("--s13-gamma-list", default="")

    ap.add_argument("--s14-alpha-list", default="")
    ap.add_argument("--s14-raw-thr-list", default="")

    ap.add_argument("--s24-alpha-list", default="")
    ap.add_argument("--s24-raw-thr-list", default="")
    ap.add_argument("--s24-burst-dt-us-list", default="")
    ap.add_argument("--s24-b-thr-list", default="")
    ap.add_argument("--s24-gamma-list", default="")

    ap.add_argument("--s25-alpha-list", default="")
    ap.add_argument("--s25-raw-thr-list", default="")
    ap.add_argument("--s25-dt-thr-list", default="")
    ap.add_argument("--s25-ref-raw-thr-list", default="")
    ap.add_argument("--s25-gamma-list", default="")

    ap.add_argument("--s15-alpha-list", default="")
    ap.add_argument("--s15-raw-thr-list", default="")
    ap.add_argument("--s15-flip-dt-us-list", default="")
    ap.add_argument("--s15-beta-list", default="")

    ap.add_argument("--s16-alpha-list", default="")
    ap.add_argument("--s16-raw-thr-list", default="")
    ap.add_argument("--s16-acc-thr-list", default="")
    ap.add_argument("--s16-ratio-thr-list", default="")
    ap.add_argument("--s16-gamma-list", default="")

    ap.add_argument("--s17-alpha-list", default="")
    ap.add_argument("--s17-raw-thr-list", default="")
    ap.add_argument("--s17-var-thr-list", default="")
    ap.add_argument("--s17-beta-list", default="")
    ap.add_argument("--s17-gamma-list", default="")

    ap.add_argument("--s19-alpha-list", default="", help="Optional; if omitted defaults to 0.2")
    ap.add_argument("--s19-beta-list", default="", help="Optional, e.g. '0.05,0.1,0.2'")
    ap.add_argument("--s20-alpha-list", default="", help="Optional; if omitted defaults to 0.2")
    ap.add_argument("--s20-beta-list", default="", help="Optional, e.g. '0.3,0.5,0.8'")
    ap.add_argument("--s21-alpha-list", default="", help="Optional; if omitted defaults to 0.2")
    ap.add_argument("--s21-beta-list", default="", help="Optional, e.g. '0.3,0.5,0.8'")
    ap.add_argument(
        "--s21-kappa-list",
        default="",
        help="Optional mix weight in [0,1]. 0=s20 behavior; 1=penalize total hotness.",
    )

    ap.add_argument("--s26-alpha-list", default="", help="Optional; if omitted defaults to 0.2")
    ap.add_argument("--s26-beta-list", default="", help="Optional, e.g. '0.6,0.8,1.0'")
    ap.add_argument("--s26-kappa-list", default="", help="Optional mix weight in [0,1]; default 1.0")
    ap.add_argument(
        "--s26-eta-list",
        default="",
        help=(
            "Optional; activity-normalization strength eta in [0.25,4]. "
            "w=(eta*tau)/(raw_tot+eta*tau); acc_pen=acc_mix*w. Default 1.0."
        ),
    )

    ap.add_argument("--s27-alpha-list", default="", help="Optional; if omitted defaults to 0.2")
    ap.add_argument("--s27-beta-list", default="", help="Optional, e.g. '0.6,0.8,1.2'")
    ap.add_argument("--s27-kappa-list", default="", help="Optional mix weight in [0,1]; default 1.0")
    ap.add_argument(
        "--s27-lambda-nb-list",
        default="",
        help=(
            "Optional; neighborhood baseline weight lambda_nb in [0,2]. "
            "acc_pen=max(0,acc_mix-lambda_nb*mean_nb(decayed_acc_tot)). Default 1.0."
        ),
    )

    ap.add_argument(
        "--s28-tau-rate-us-list",
        default="",
        help=(
            "Optional; EMA time constant for global rate estimate (us). "
            "0 means auto (use current tau_us). If omitted defaults to 0."
        ),
    )

    ap.add_argument(
        "--s35-tau-rate-us-list",
        default="",
        help=(
            "Optional; EMA time constant for global rate estimate (us) for s35. "
            "0 means auto (use current tau_us). If omitted defaults to 0."
        ),
    )

    ap.add_argument(
        "--s36-tau-rate-us-list",
        default="",
        help=(
            "Optional; EMA time constant for global rate estimate (us) for s36. "
            "0 means auto (use current tau_us). If omitted defaults to 0."
        ),
    )

    ap.add_argument(
        "--s37-tau-rate-us-list",
        default="",
        help=(
            "Optional; EMA time constant for global rate estimate (us) for s37. "
            "0 means auto (use current tau_us). If omitted defaults to 0."
        ),
    )

    ap.add_argument(
        "--s38-tau-rate-us-list",
        default="",
        help=(
            "Optional; EMA time constant for global rate estimate (us) for s38. "
            "0 means auto (use current tau_us). If omitted defaults to 0."
        ),
    )

    ap.add_argument(
        "--s39-tau-rate-us-list",
        default="",
        help=(
            "Optional; EMA time constant for global rate estimate (us) for s39. "
            "0 means auto (use current tau_us). If omitted defaults to 0."
        ),
    )
    ap.add_argument(
        "--s39-k-nbmix-list",
        default="",
        help=(
            "Optional; neighborhood-mix occupancy strength k for s39 (>=0). "
            "If omitted defaults to 1.0."
        ),
    )

    ap.add_argument(
        "--s40-tau-rate-us-list",
        default="",
        help=(
            "Optional; EMA time constant for global rate estimate (us) for s40. "
            "0 means auto (use current tau_us). If omitted defaults to 0."
        ),
    )
    ap.add_argument(
        "--s40-k-nbmix-list",
        default="",
        help=(
            "Optional; neighborhood-mix occupancy strength k for s40 (>=0). "
            "If omitted defaults to 1.0."
        ),
    )

    ap.add_argument(
        "--s41-tau-rate-us-list",
        default="",
        help=(
            "Optional; EMA time constant for global rate estimate (us) for s41. "
            "0 means auto (use current tau_us). If omitted defaults to 0."
        ),
    )
    ap.add_argument(
        "--s41-k-nbmix-list",
        default="",
        help=(
            "Optional; neighborhood-mix occupancy strength k for s41 (>=0). "
            "If omitted defaults to 1.0."
        ),
    )

    ap.add_argument(
        "--s42-tau-rate-us-list",
        default="",
        help=(
            "Optional; EMA time constant for global rate estimate (us) for s42. "
            "0 means auto (use current tau_us). If omitted defaults to 0."
        ),
    )
    ap.add_argument(
        "--s42-k-nbmix-list",
        default="",
        help=(
            "Optional; neighborhood-mix occupancy strength k for s42 (>=0). "
            "If omitted defaults to 1.0."
        ),
    )

    ap.add_argument(
        "--s43-tau-rate-us-list",
        default="",
        help=(
            "Optional; EMA time constant for global rate estimate (us) for s43. "
            "0 means auto (use current tau_us). If omitted defaults to 0."
        ),
    )
    ap.add_argument(
        "--s43-k-nbmix-list",
        default="",
        help=(
            "Optional; neighborhood-mix occupancy strength k for s43 (>=0). "
            "If omitted defaults to 1.0."
        ),
    )

    ap.add_argument(
        "--s44-tau-rate-us-list",
        default="",
        help=(
            "Optional; time constant (us) for self-occupancy normalization in s44. "
            "0 means auto (use current tau_us). If omitted defaults to 0."
        ),
    )

    ap.add_argument(
        "--s45-tau-rate-us-list",
        default="",
        help=(
            "Optional; time constant (us) for self-occupancy normalization in s45. "
            "0 means auto (use current tau_us). If omitted defaults to 0."
        ),
    )
    ap.add_argument(
        "--s45-u0-list",
        default="",
        help=(
            "Optional; occupancy gate u0 in [0,1) for s45. Below u0, score==raw (no penalty). "
            "If omitted defaults to 0.0 (equivalent to s44)."
        ),
    )

    ap.add_argument(
        "--s46-tau-rate-us-list",
        default="",
        help=(
            "Optional; time constant (us) for self-occupancy normalization in s46. "
            "0 means auto (use current tau_us). If omitted defaults to 0."
        ),
    )

    ap.add_argument(
        "--s47-tau-rate-us-list",
        default="",
        help=(
            "Optional; time constant (us) for self-occupancy normalization in s47. "
            "0 means auto (use current tau_us). If omitted defaults to 0."
        ),
    )

    ap.add_argument(
        "--s48-tau-rate-us-list",
        default="",
        help=(
            "Optional; time constant (us) for self-occupancy normalization in s48. "
            "0 means auto (use current tau_us). If omitted defaults to 0."
        ),
    )
    ap.add_argument(
        "--s48-eta-toggle-list",
        default="",
        help=(
            "Optional; toggle downweight eta in [0,1] for s48. If omitted defaults to 0.25."
        ),
    )

    ap.add_argument(
        "--s49-tau-rate-us-list",
        default="",
        help=(
            "Optional; time constant (us) for self-occupancy normalization in s49. "
            "0 means auto (use current tau_us). If omitted defaults to 0."
        ),
    )

    ap.add_argument(
        "--s50-tau-rate-us-list",
        default="",
        help=(
            "Optional; time constant (us) for self-occupancy normalization in s50. "
            "0 means auto (use current tau_us). If omitted defaults to 0."
        ),
    )
    ap.add_argument(
        "--s50-beta-list",
        default="",
        help=(
            "Optional; support-breadth boost strength beta (>=0) for s50. "
            "0 disables boost (reduces to s44). If omitted defaults to 0.0."
        ),
    )
    ap.add_argument(
        "--s50-cnt0-list",
        default="",
        help=(
            "Optional; saturation support count cnt0 (>0) for s50. "
            "If omitted defaults to 8."
        ),
    )
    ap.add_argument(
        "--s35-gamma-list",
        default="",
        help=(
            "Optional; pixel-state modulation strength gamma for s35 (>=0). "
            "If omitted defaults to 1.0."
        ),
    )
    ap.add_argument(
        "--s35-hmax-list",
        default="",
        help=(
            "Optional; clamp for H/tau (hmax) for s35 (>=0). "
            "If omitted defaults to 8.0."
        ),
    )

    ap.add_argument(
        "--s30-tau-rate-us-list",
        default="",
        help=(
            "Optional; EMA time constant for global rate estimate (us) for s30. "
            "0 means auto (use current tau_us). If omitted defaults to 0."
        ),
    )

    ap.add_argument(
        "--s31-tau-rate-us-list",
        default="",
        help=(
            "Optional; EMA time constant for global rate/polarity estimate (us) for s31. "
            "0 means auto (use current tau_us). If omitted defaults to 0."
        ),
    )

    ap.add_argument(
        "--s32-tau-rate-us-list",
        default="",
        help=(
            "Optional; EMA time constant for global/block rate estimate (us) for s32. "
            "0 means auto (use current tau_us). If omitted defaults to 0."
        ),
    )

    ap.add_argument(
        "--s33-tau-rate-us-list",
        default="",
        help=(
            "Optional; EMA time constant for global rate estimate (us) for s33. "
            "0 means auto (use current tau_us). If omitted defaults to 0."
        ),
    )
    ap.add_argument(
        "--s33-beta-list",
        default="",
        help=(
            "Optional; penalty strength beta for s33 (>=0). "
            "If omitted defaults to 0.5."
        ),
    )

    ap.add_argument(
        "--s34-tau-rate-us-list",
        default="",
        help=(
            "Optional; EMA time constant for global rate estimate (us) for s34. "
            "0 means auto (use current tau_us). If omitted defaults to 0."
        ),
    )
    ap.add_argument(
        "--s34-k-self-list",
        default="",
        help=(
            "Optional; self-rate scale k_self for s34 (>=0). "
            "If omitted defaults to 0.25."
        ),
    )

    ap.add_argument(
        "--light",
        default=r"D:\hjx_workspace\scientific_reserach\dataset\ED24\myPedestrain_06\Pedestrain_06_1.8.npy",
    )
    ap.add_argument(
        "--mid",
        default=r"D:\hjx_workspace\scientific_reserach\dataset\ED24\myPedestrain_06\Pedestrain_06_2.5.npy",
    )
    ap.add_argument(
        "--heavy",
        default=r"D:\hjx_workspace\scientific_reserach\dataset\ED24\myPedestrain_06\Pedestrain_06_3.3.npy",
    )

    args = ap.parse_args()

    esr_mode = str(args.esr_mode).strip().lower()
    aocc_mode = str(args.aocc_mode).strip().lower()
    need_best_recipes = (esr_mode == "best") or (aocc_mode == "best")

    def _env_override(overrides: dict[str, str]) -> dict[str, str | None]:
        old: dict[str, str | None] = {}
        for k, v in overrides.items():
            old[k] = os.environ.get(k)
            os.environ[k] = v
        return old

    def _env_restore(old: dict[str, str | None], overrides: dict[str, str]) -> None:
        for k in overrides:
            if old.get(k) is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(old[k])

    variant_raw = str(args.variant).strip()
    variant = variant_raw.lower()
    variant = {
        "ebf": "ebf",
        "baseline": "ebf",
        "v0": "ebf",
        "ebfv10": "ebfv10",
        "v10": "ebfv10",
        "spatialw_linear": "ebfv10",
        "ebf_v10": "ebfv10",
        "s1": "s1",
        "ebf_s1": "s1",
        "ebfs1": "s1",
        "dircoh": "s1",
        "directional_coherence": "s1",
        "s2": "s2",
        "ebf_s2": "s2",
        "ebfs2": "s2",
        "coh_gate": "s2",
        "coherence_gate": "s2",
        "s3": "s3",
        "ebf_s3": "s3",
        "ebfs3": "s3",
        "softgate": "s3",
        "coh_softgate": "s3",
        "s4": "s4",
        "ebf_s4": "s4",
        "ebfs4": "s4",
        "residual_gate": "s4",
        "resultant_gate": "s4",
        "s5": "s5",
        "ebf_s5": "s5",
        "ebfs5": "s5",
        "elliptic_spatialw": "s5",
        "elliptic": "s5",
        "s6": "s6",
        "ebf_s6": "s6",
        "ebfs6": "s6",
        "timecoh_gate": "s6",
        "time_gate": "s6",
        "s7": "s7",
        "ebf_s7": "s7",
        "ebfs7": "s7",
        "plane_gate": "s7",
        "plane_residual_gate": "s7",
        "s8": "s8",
        "ebf_s8": "s8",
        "ebfs8": "s8",
        "plane_r2_gate": "s8",
        "plane_explained_gate": "s8",
        "plane_r2": "s8",
        "s9": "s9",
        "ebf_s9": "s9",
        "ebfs9": "s9",
        "refractory_gate": "s9",
        "burst_gate": "s9",
        "self_refractory": "s9",
        "s10": "s10",
        "ebf_s10": "s10",
        "ebfs10": "s10",
        "hotpixel_rate_gate": "s10",
        "rate_gate": "s10",
        "hotpixel_gate": "s10",
        "s11": "s11",
        "ebf_s11": "s11",
        "ebfs11": "s11",
        "relative_hotness_gate": "s11",
        "relative_hotness": "s11",
        "relhot": "s11",
        "s12": "s12",
        "ebf_s12": "s12",
        "ebfs12": "s12",
        "hotness_zscore_gate": "s12",
        "zscore_hotness": "s12",
        "zgate": "s12",
        "s13": "s13",
        "ebf_s13": "s13",
        "ebfs13": "s13",
        "crosspol_support_gate": "s13",
        "crosspol_gate": "s13",
        "crosspol_support": "s13",
        "s14": "s14",
        "ebf_s14": "s14",
        "ebfs14": "s14",
        "crosspol_boost": "s14",
        "crosspol_boost_score": "s14",
        "crosspol_add": "s14",
        "s15": "s15",
        "ebf_s15": "s15",
        "ebfs15": "s15",
        "flip_flicker_gate": "s15",
        "polarity_flip_gate": "s15",
        "flicker": "s15",
        "s16": "s16",
        "ebf_s16": "s16",
        "ebfs16": "s16",
        "s14_hotness_clamp": "s16",
        "hotness_clamp": "s16",
        "s17": "s17",
        "ebf_s17": "s17",
        "ebfs17": "s17",
        "crosspol_spread_boost": "s17",
        "spread_boost": "s17",
        "s18": "s18",
        "ebf_s18": "s18",
        "ebfs18": "s18",
        "nopol": "s18",
        "no_polarity": "s18",
        "polarity_free": "s18",
        "s19": "s19",
        "ebf_s19": "s19",
        "ebfs19": "s19",
        "evidence_fusion": "s19",
        "fusion": "s19",
        "fusion_q8": "s19",
        "s20": "s20",
        "ebf_s20": "s20",
        "ebfs20": "s20",
        "fusion_polhot": "s20",
        "polhot_fusion": "s20",
        "pol_hotness": "s20",
        "s21": "s21",
        "ebf_s21": "s21",
        "ebfs21": "s21",
        "bipolhot_fusion": "s21",
        "bipol_hotness": "s21",
        "polhot_mix": "s21",
        "s22": "s22",
        "ebf_s22": "s22",
        "ebfs22": "s22",
        "anypol_burst_gate": "s22",
        "anypol_burst": "s22",
        "burst_anypol": "s22",
        "s23": "s23",
        "ebf_s23": "s23",
        "ebfs23": "s23",
        "featlogit": "s23",
        "feature_logit": "s23",
        "feature_fusion": "s23",
        "s24": "s24",
        "ebf_s24": "s24",
        "ebfs24": "s24",
        "s14_burstiness_gate": "s24",
        "burstiness_gate": "s24",

        "s25": "s25",
        "ebf_s25": "s25",
        "ebfs25": "s25",
        "s14_refractory_gate": "s25",
        "s14_refractory": "s25",
        "refractory_s14": "s25",

        "s26": "s26",
        "ebf_s26": "s26",
        "ebfs26": "s26",
        "s26_actnorm": "s26",
        "actnorm_hotness": "s26",
        "actnorm_hotness_fusion": "s26",

        "s27": "s27",
        "ebf_s27": "s27",
        "ebfs27": "s27",
        "s27_relabnorm": "s27",
        "relabnorm_hotness": "s27",
        "relative_abnormal_hotness": "s27",

        "s28": "s28",
        "ebf_s28": "s28",
        "ebfs28": "s28",
        "noise_surprise": "s28",
        "surprise_zscore": "s28",
        "surprise_z": "s28",

        "s35": "s35",
        "ebf_s35": "s35",
        "ebfs35": "s35",
        "surprise_pixelstate": "s35",
        "surprise_pixel_state": "s35",
        "s28_pixelstate": "s35",

        "s36": "s36",
        "ebf_s36": "s36",
        "ebfs36": "s36",
        "surprise_occupancy": "s36",
        "surprise_stateoccupancy": "s36",
        "s28_stateoccupancy": "s36",

        "s37": "s37",
        "ebf_s37": "s37",
        "ebfs37": "s37",
        "surprise_occupancy_3state": "s37",
        "surprise_stateoccupancy_3state": "s37",
        "s28_stateoccupancy_3state": "s37",

        "s38": "s38",
        "ebf_s38": "s38",
        "ebfs38": "s38",
        "surprise_occupancy_nbocc": "s38",
        "surprise_stateocc_nbocc": "s38",
        "s28_stateocc_nbocc": "s38",

        "s39": "s39",
        "ebf_s39": "s39",
        "ebfs39": "s39",
        "surprise_occupancy_nbocc_mix": "s39",
        "surprise_stateocc_nbocc_mix": "s39",
        "s28_stateocc_nbocc_mix": "s39",

        "s40": "s40",
        "ebf_s40": "s40",
        "ebfs40": "s40",
        "surprise_occupancy_nbocc_mix_fuse_geom": "s40",
        "surprise_stateocc_nbocc_mix_fuse_geom": "s40",

        "s41": "s41",
        "ebf_s41": "s41",
        "ebfs41": "s41",
        "surprise_occupancy_nbocc_mix_pow2": "s41",
        "surprise_stateocc_nbocc_mix_pow2": "s41",

        "s42": "s42",
        "ebf_s42": "s42",
        "ebfs42": "s42",
        "surprise_occupancy_nbocc_mix_gated_self2": "s42",
        "surprise_stateocc_nbocc_mix_gated_self2": "s42",

        "s43": "s43",
        "ebf_s43": "s43",
        "ebfs43": "s43",
        "surprise_occupancy_nbocc_mix_u2": "s43",
        "surprise_stateocc_nbocc_mix_u2": "s43",

        "s44": "s44",
        "ebf_s44": "s44",
        "ebfs44": "s44",
        "ebf_labelscore_selfocc_div_u2": "s44",
        "ebf_labelscore_selfocc_u2": "s44",

        "s45": "s45",
        "ebf_s45": "s45",
        "ebfs45": "s45",
        "ebf_labelscore_selfocc_gate_div_u2": "s45",
        "ebf_labelscore_selfocc_gate_u2": "s45",

        "s46": "s46",
        "ebf_s46": "s46",
        "ebfs46": "s46",
        "ebf_labelscore_selfocc_odds_div_v2": "s46",
        "ebf_labelscore_selfocc_odds": "s46",

        "s47": "s47",
        "ebf_s47": "s47",
        "ebfs47": "s47",
        "ebf_labelscore_selfocc_abn_div_u2": "s47",
        "ebf_labelscore_selfocc_abn_u2": "s47",

        "s48": "s48",
        "ebf_s48": "s48",
        "ebfs48": "s48",
        "ebf_labelscore_selfocc_polpersist_div_u2": "s48",
        "ebf_labelscore_selfocc_polpersist_u2": "s48",

        "s49": "s49",
        "ebf_s49": "s49",
        "ebfs49": "s49",
        "ebf_labelscore_selfocc_bipolmax_div_u2": "s49",
        "ebf_labelscore_selfocc_bipolmax_u2": "s49",

        "s50": "s50",
        "ebf_s50": "s50",
        "ebfs50": "s50",
        "ebf_labelscore_selfocc_supportboost_div_u2": "s50",
        "ebf_labelscore_selfocc_supportboost_u2": "s50",

        "s51": "s51",
        "ebf_s51": "s51",
        "ebfs51": "s51",
        "ebf_labelscore_selfocc_supportboost_autobeta_div_u2": "s51",
        "ebf_labelscore_selfocc_supportboost_autobeta_u2": "s51",

        "s52": "s52",
        "ebf_s52": "s52",
        "ebfs52": "s52",
        "ebf_labelscore_selfocc_supportboost_autobeta_mixgateopp_div_u2": "s52",
        "ebf_labelscore_selfocc_supportboost_autobeta_mixgateopp_u2": "s52",

        "s53": "s53",
        "ebf_s53": "s53",
        "ebfs53": "s53",
        "ebf_labelscore_selfocc_supportboost_autobeta_eventmixgateopp_div_u2": "s53",
        "ebf_labelscore_selfocc_supportboost_autobeta_eventmixgateopp_u2": "s53",

        "s54": "s54",
        "ebf_s54": "s54",
        "ebfs54": "s54",
        "ebf_labelscore_selfocc_supportboost_autobeta_eventmixgateopp_root4_div_u2": "s54",
        "ebf_labelscore_selfocc_supportboost_autobeta_eventmixgateopp_root4_u2": "s54",

        "s55": "s55",
        "ebf_s55": "s55",
        "ebfs55": "s55",
        "ebf_labelscore_selfocc_supportboost_autobeta_eventmixgateopp_supportlerp_div_u2": "s55",
        "ebf_labelscore_selfocc_supportboost_autobeta_eventmixgateopp_supportlerp_u2": "s55",

        "s60": "s60",
        "ebf_s60": "s60",
        "ebfs60": "s60",
        "ebf_labelscore_dualtau_delta_selfocc_supportboost_autobeta_mixgateopp_div_u2": "s60",
        "ebf_labelscore_dualtau_delta_selfocc_supportboost_autobeta_mixgateopp_u2": "s60",

        "s30": "s30",
        "ebf_s30": "s30",
        "ebfs30": "s30",
        "surprise_localrate": "s30",
        "surprise_localrate_max": "s30",
        "s28_localrate": "s30",

        "s31": "s31",
        "ebf_s31": "s31",
        "ebfs31": "s31",
        "surprise_polbias": "s31",
        "surprise_z_polbias": "s31",
        "s28_polbias": "s31",

        "s32": "s32",
        "ebf_s32": "s32",
        "ebfs32": "s32",
        "surprise_blockrate": "s32",
        "surprise_blockrate_max": "s32",
        "s28_blockrate": "s32",

        "s33": "s33",
        "ebf_s33": "s33",
        "ebfs33": "s33",
        "surprise_abnhot": "s33",
        "surprise_abnhot_penalty": "s33",
        "s28_abnhot_penalty": "s33",

        "s34": "s34",
        "ebf_s34": "s34",
        "ebfs34": "s34",
        "surprise_selfrate": "s34",
        "surprise_selfrate_max": "s34",
        "s28_selfrate": "s34",

        "s29": "s29",
        "ebf_s29": "s29",
        "ebfs29": "s29",
        "polarity_surprise": "s29",
        "pol_surprise": "s29",
        "surprise_pol_z": "s29",
    }.get(variant, variant)
    if variant not in {"ebf", "ebfv10", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23", "s24", "s25", "s26", "s27", "s28", "s29", "s30", "s31", "s32", "s33", "s34", "s35", "s36", "s37", "s38", "s39", "s40", "s41", "s42", "s43", "s44", "s45", "s46", "s47", "s48", "s49", "s50", "s51", "s52", "s53", "s54", "s55", "s60"}:
        raise SystemExit(
            f"unknown --variant: {variant_raw!r}. choices: ebf | EBFV10 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | s10 | s11 | s12 | s13 | s14 | s15 | s16 | s17 | s18 | s19 | s20 | s21 | s22 | s23 | s24 | s25 | s26 | s27 | s28 | s35 | s36 | s37 | s38 | s39 | s29 | s30 | s31 | s32 | s33 | s34"
        )

    tb = TimeBase(tick_ns=float(args.tick_ns))
    s_list = _parse_int_list(args.s_list)
    for s in s_list:
        if s < 3 or s % 2 == 0:
            raise SystemExit(f"--s-list expects odd diameters >=3 (got {s})")
    tau_us_list = _parse_int_list(args.tau_us_list)

    env_inputs = {
        "light": str(args.light),
        "mid": str(args.mid),
        "heavy": str(args.heavy),
    }

    out_dir = str(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Per-env ROC outputs (same header/format as data/mydriving/*/roc_*.csv)
    roc_prefix = {
        "ebf": "roc_ebf",
        "ebfv10": "roc_ebf_v10",
        "s1": "roc_ebf_s1",
        "s2": "roc_ebf_s2",
        "s3": "roc_ebf_s3",
        "s4": "roc_ebf_s4",
        "s5": "roc_ebf_s5",
        "s6": "roc_ebf_s6",
        "s7": "roc_ebf_s7",
        "s8": "roc_ebf_s8",
        "s9": "roc_ebf_s9",
        "s22": "roc_ebf_s22",
        "s10": "roc_ebf_s10",
        "s11": "roc_ebf_s11",
        "s12": "roc_ebf_s12",
        "s13": "roc_ebf_s13",
        "s14": "roc_ebf_s14",
        "s24": "roc_ebf_s24",
        "s25": "roc_ebf_s25",
        "s26": "roc_ebf_s26",
        "s27": "roc_ebf_s27",
        "s28": "roc_ebf_s28",
        "s35": "roc_ebf_s35",
        "s36": "roc_ebf_s36",
        "s37": "roc_ebf_s37",
        "s38": "roc_ebf_s38",
        "s39": "roc_ebf_s39",
        "s40": "roc_ebf_s40",
        "s41": "roc_ebf_s41",
        "s42": "roc_ebf_s42",
        "s43": "roc_ebf_s43",
        "s44": "roc_ebf_s44",
        "s45": "roc_ebf_s45",
        "s46": "roc_ebf_s46",
        "s47": "roc_ebf_s47",
        "s48": "roc_ebf_s48",
        "s49": "roc_ebf_s49",
        "s50": "roc_ebf_s50",
        "s51": "roc_ebf_s51",
        "s52": "roc_ebf_s52",
        "s53": "roc_ebf_s53",
        "s54": "roc_ebf_s54",
        "s55": "roc_ebf_s55",
        "s60": "roc_ebf_s60",
        "s29": "roc_ebf_s29",
        "s30": "roc_ebf_s30",
        "s31": "roc_ebf_s31",
        "s32": "roc_ebf_s32",
        "s33": "roc_ebf_s33",
        "s34": "roc_ebf_s34",
        "s15": "roc_ebf_s15",
        "s16": "roc_ebf_s16",
        "s17": "roc_ebf_s17",
        "s18": "roc_ebf_s18",
        "s19": "roc_ebf_s19",
        "s20": "roc_ebf_s20",
        "s21": "roc_ebf_s21",
        "s23": "roc_ebf_s23",
    }[variant]
    roc_csv = {
        env: os.path.join(out_dir, f"{roc_prefix}_{env}_labelscore_s3_5_7_9_tau8_16_32_64_128_256_512_1024ms.csv")
        for env in env_inputs
    }
    roc_png = {
        env: os.path.join(out_dir, f"{roc_prefix}_{env}_labelscore_s3_5_7_9_tau8_16_32_64_128_256_512_1024ms.png")
        for env in env_inputs
    }

    if bool(args.plot_only):
        for env in ("light", "mid", "heavy"):
            if not os.path.exists(roc_csv[env]):
                print(f"skip plot-only (missing csv): env={env} path={roc_csv[env]}")
                continue
            _plot_roc_png(
                csv_path=roc_csv[env],
                png_path=roc_png[env],
                title={
                    "ebf": f"EBF ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "ebfv10": f"EBF V10 (spatialw_linear) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s1": f"EBF Part2 s1 (directional coherence) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s2": f"EBF Part2 s2 (coherence-gated penalty) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s3": f"EBF Part2 s3 (smooth coh gate) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s4": f"EBF Part2 s4 (resultant gate) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s5": f"EBF Part2 s5 (elliptic spatialw) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s6": f"EBF Part2 s6 (time-coh gate) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s7": f"EBF Part2 s7 (plane residual gate) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s8": f"EBF Part2 s8 (plane R2 gate) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s9": f"EBF Part2 s9 (refractory/burst gate) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s22": f"EBF Part2 s22 (any-pol burst gate) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s23": f"EBF Part2 s23 (feature+logit fusion) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s10": f"EBF Part2 s10 (hotpixel leaky-rate gate) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s11": f"EBF Part2 s11 (relative-hotness anomaly gate) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s12": f"EBF Part2 s12 (hotness z-score anomaly gate) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s13": f"EBF Part2 s13 (cross-polarity support gate) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s14": f"EBF Part2 s14 (cross-polarity boost) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s24": f"EBF Part2 s24 (s14 + burstiness gate) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s25": f"EBF Part2 s25 (s14 + refractory gate) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s26": f"EBF Part2 s26 (act-norm hotness fusion q8) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s27": f"EBF Part2 s27 (rel abnormal hotness fusion q8) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s28": f"EBF Part2 s28 (noise surprise z-score) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s35": f"EBF Part2 s35 (s28 + pixel-state-conditioned null) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s36": f"EBF Part2 s36 (s28 + state-occupancy-conditioned null) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s37": f"EBF Part2 s37 (s36 + 3-state occupancy->rate) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s38": f"EBF Part2 s38 (s36 + nb-occupancy fusion) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s39": f"EBF Part2 s39 (s38 + polarity-mix-weighted nb-occupancy) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s40": f"EBF Part2 s40 (s39 + conservative geom fusion) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s41": f"EBF Part2 s41 (s39 + conservative mix^2 shaping) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s42": f"EBF Part2 s42 (s39 + u_self^2-gated nb term) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s43": f"EBF Part2 s43 (s39 + u_eff^2 compression) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s44": f"EBF Part2 s44 (EBF labelscore / (1+u_self^2)) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s45": f"EBF Part2 s45 (s44 + u0-gated penalty) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s46": f"EBF Part2 s46 (EBF labelscore / (1+odds(u_self)^2)) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s47": f"EBF Part2 s47 (abnormal self-state, score=raw/(1+u_self^2)) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s48": f"EBF Part2 s48 (polarity-persistent self-state, score=raw/(1+u_self^2)) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s49": f"EBF Part2 s49 (bipolar max self-state, score=raw/(1+u_self^2)) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s50": f"EBF Part2 s50 (s44 + support-breadth boost) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s51": f"EBF Part2 s51 (s50 auto-beta, no env hyperparams) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s52": f"EBF Part2 s52 (s51 + auto-gated opp evidence by mix) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s53": f"EBF Part2 s53 (s51 + per-event mix-gated opp evidence) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s54": f"EBF Part2 s54 (s53 shape tweak: opp weight uses sfrac^(1/4)) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s55": f"EBF Part2 s55 (s53 shape tweak: support-adaptive mix gate for opp) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s60": f"EBF Part2 s60 (dualtau delta + selfocc + auto beta + auto mix-gated opp) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s29": f"EBF Part2 s29 (local polarity-surprise z-score) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s30": f"EBF Part2 s30 (s28 + local-rate correction) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s31": f"EBF Part2 s31 (s28 + polarity-bias correction) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s32": f"EBF Part2 s32 (s28 + block-rate max correction) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s33": f"EBF Part2 s33 (s28 - abnormal-hotness penalty) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s34": f"EBF Part2 s34 (s28 + pixel self-rate max) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s15": f"EBF Part2 s15 (flip flicker gate) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s16": f"EBF Part2 s16 (s14 + hotness clamp) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s17": f"EBF Part2 s17 (cross-pol spread trust) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s18": f"EBF Part2 s18 (no polarity check) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s19": f"EBF Part2 s19 (evidence fusion q8) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s20": f"EBF Part2 s20 (pol-hotness fusion q8) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                    "s21": f"EBF Part2 s21 (bi-pol hotness mix fusion q8) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
                }[variant],
            )
            print(f"saved: {roc_png[env]}")
        return 0

    kernel_cache: dict[str, object] = {}

    best_global = ("", -1.0)
    best_by_env: dict[str, tuple[str, float]] = {"light": ("", -1.0), "mid": ("", -1.0), "heavy": ("", -1.0)}
    best_f1_by_env: dict[str, tuple[str, float]] = {"light": ("", -1.0), "mid": ("", -1.0), "heavy": ("", -1.0)}

    # Start fresh ROC csv (these files are large; resuming is less useful)
    write_enabled: dict[str, bool] = {}
    for env, p in roc_csv.items():
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
        try:
            with open(p, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(ROC_HEADER)
            write_enabled[env] = True
        except PermissionError:
            if os.path.exists(p):
                print(f"warn: cannot write (locked). env={env} path={p} -> skip recompute, plot only")
                write_enabled[env] = False
            else:
                raise

    for env, in_path in env_inputs.items():
        if not write_enabled.get(env, True):
            _plot_roc_png(
                csv_path=roc_csv[env],
                png_path=roc_png[env],
                title=f"EBF ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us",
            )
            print(f"saved: {roc_png[env]}")
            continue

        ev = load_labeled_npy(in_path, max_events=int(args.max_events))
        n = int(ev.label.shape[0])
        pos = int(np.sum(ev.label))
        neg = int(n - pos)
        print(f"loaded: env={env} n={n} pos={pos} neg={neg} in={in_path}")

        with open(roc_csv[env], "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)

            # For {esr,aocc}-mode=best: record the recipe for the best-AUC tag and best-F1 tag.
            best_auc_recipe: dict[str, object] | None = None
            best_f1_recipe: dict[str, object] | None = None


            base_tag = {
                "ebf": "ebf",
                "ebfv10": "ebf_v10",
                "s1": "ebf_s1",
                "s2": "ebf_s2",
                "s3": "ebf_s3",
                "s4": "ebf_s4",
                "s5": "ebf_s5",
                "s6": "ebf_s6",
                "s7": "ebf_s7",
                "s8": "ebf_s8",
                "s9": "ebf_s9",
                "s22": "ebf_s22",
                "s23": "ebf_s23",
                "s10": "ebf_s10",
                "s11": "ebf_s11",
                "s12": "ebf_s12",
                "s13": "ebf_s13",
                "s14": "ebf_s14",
                "s24": "ebf_s24",
                "s25": "ebf_s25",
                "s26": "ebf_s26",
                "s27": "ebf_s27",
                "s28": "ebf_s28",
                "s35": "ebf_s35",
                "s36": "ebf_s36",
                "s37": "ebf_s37",
                "s38": "ebf_s38",
                "s39": "ebf_s39",
                "s40": "ebf_s40",
                "s41": "ebf_s41",
                "s42": "ebf_s42",
                "s43": "ebf_s43",
                "s44": "ebf_s44",
                "s45": "ebf_s45",
                "s46": "ebf_s46",
                "s47": "ebf_s47",
                "s48": "ebf_s48",
                "s49": "ebf_s49",
                "s50": "ebf_s50",
                "s51": "ebf_s51",
                "s52": "ebf_s52",
                "s53": "ebf_s53",
                "s54": "ebf_s54",
                "s55": "ebf_s55",
                "s60": "ebf_s60",
                "s29": "ebf_s29",
                "s30": "ebf_s30",
                "s31": "ebf_s31",
                "s32": "ebf_s32",
                "s33": "ebf_s33",
                "s34": "ebf_s34",
                "s15": "ebf_s15",
                "s16": "ebf_s16",
                "s17": "ebf_s17",
                "s18": "ebf_s18",
                "s19": "ebf_s19",
                "s20": "ebf_s20",
                "s21": "ebf_s21",
            }[variant]

            # Optional Part2 hyperparam sweeps (used when --variant matches)
            sweep: list[tuple[str, dict[str, str]]] = [("", {})]
            if variant == "s2":
                coh_list = _parse_float_list(args.s2_coh_thr_list) if str(args.s2_coh_thr_list).strip() else [None]
                raw_list = _parse_float_list(args.s2_raw_thr_list) if str(args.s2_raw_thr_list).strip() else [None]
                gam_list = _parse_float_list(args.s2_gamma_list) if str(args.s2_gamma_list).strip() else [None]
                sweep = []
                for coh in coh_list:
                    for raw in raw_list:
                        for gam in gam_list:
                            overrides: dict[str, str] = {}
                            suffix = ""
                            if coh is not None:
                                overrides["MYEVS_EBF_S2_COH_THR"] = str(float(coh))
                                suffix += f"_coh{_fmt_tag_float(float(coh))}"
                            if raw is not None:
                                overrides["MYEVS_EBF_S2_RAW_THR"] = str(float(raw))
                                suffix += f"_raw{_fmt_tag_float(float(raw))}"
                            if gam is not None:
                                overrides["MYEVS_EBF_S2_GAMMA"] = str(float(gam))
                                suffix += f"_g{_fmt_tag_float(float(gam))}"
                            sweep.append((suffix, overrides))

            elif variant == "s3":
                coh_list = _parse_float_list(args.s3_coh_thr_list) if str(args.s3_coh_thr_list).strip() else [None]
                raw_list = _parse_float_list(args.s3_raw_thr_list) if str(args.s3_raw_thr_list).strip() else [None]
                gam_list = _parse_float_list(args.s3_gamma_list) if str(args.s3_gamma_list).strip() else [None]
                a_list = _parse_float_list(args.s3_alpha_list) if str(args.s3_alpha_list).strip() else [None]
                kr_list = _parse_float_list(args.s3_k_raw_list) if str(args.s3_k_raw_list).strip() else [None]
                kc_list = _parse_float_list(args.s3_k_coh_list) if str(args.s3_k_coh_list).strip() else [None]
                sweep = []
                for coh in coh_list:
                    for raw in raw_list:
                        for gam in gam_list:
                            for a in a_list:
                                for kr in kr_list:
                                    for kc in kc_list:
                                        overrides: dict[str, str] = {}
                                        suffix = ""
                                        if coh is not None:
                                            overrides["MYEVS_EBF_S3_COH_THR"] = str(float(coh))
                                            suffix += f"_coh{_fmt_tag_float(float(coh))}"
                                        if raw is not None:
                                            overrides["MYEVS_EBF_S3_RAW_THR"] = str(float(raw))
                                            suffix += f"_raw{_fmt_tag_float(float(raw))}"
                                        if gam is not None:
                                            overrides["MYEVS_EBF_S3_GAMMA"] = str(float(gam))
                                            suffix += f"_g{_fmt_tag_float(float(gam))}"
                                        if a is not None:
                                            overrides["MYEVS_EBF_S3_ALPHA"] = str(float(a))
                                            suffix += f"_a{_fmt_tag_float(float(a))}"
                                        if kr is not None:
                                            overrides["MYEVS_EBF_S3_K_RAW"] = str(float(kr))
                                            suffix += f"_kr{_fmt_tag_float(float(kr))}"
                                        if kc is not None:
                                            overrides["MYEVS_EBF_S3_K_COH"] = str(float(kc))
                                            suffix += f"_kc{_fmt_tag_float(float(kc))}"
                                        sweep.append((suffix, overrides))

            elif variant == "s4":
                a_list = _parse_float_list(args.s4_align_thr_list) if str(args.s4_align_thr_list).strip() else [None]
                raw_list = _parse_float_list(args.s4_raw_thr_list) if str(args.s4_raw_thr_list).strip() else [None]
                gam_list = _parse_float_list(args.s4_gamma_list) if str(args.s4_gamma_list).strip() else [None]
                sweep = []
                for a in a_list:
                    for raw in raw_list:
                        for gam in gam_list:
                            overrides: dict[str, str] = {}
                            suffix = ""
                            if a is not None:
                                overrides["MYEVS_EBF_S4_ALIGN_THR"] = str(float(a))
                                suffix += f"_a{_fmt_tag_float(float(a))}"
                            if raw is not None:
                                overrides["MYEVS_EBF_S4_RAW_THR"] = str(float(raw))
                                suffix += f"_raw{_fmt_tag_float(float(raw))}"
                            if gam is not None:
                                overrides["MYEVS_EBF_S4_GAMMA"] = str(float(gam))
                                suffix += f"_g{_fmt_tag_float(float(gam))}"
                            sweep.append((suffix, overrides))

            elif variant == "s5":
                ax_list = _parse_float_list(args.s5_ax_list) if str(args.s5_ax_list).strip() else [None]
                ay_list = _parse_float_list(args.s5_ay_list) if str(args.s5_ay_list).strip() else [None]
                th_list = _parse_float_list(args.s5_theta_deg_list) if str(args.s5_theta_deg_list).strip() else [None]
                sweep = []
                for ax in ax_list:
                    for ay in ay_list:
                        for th in th_list:
                            overrides: dict[str, str] = {}
                            suffix = ""
                            if ax is not None:
                                overrides["MYEVS_EBF_S5_AX"] = str(float(ax))
                                suffix += f"_ax{_fmt_tag_float(float(ax))}"
                            if ay is not None:
                                overrides["MYEVS_EBF_S5_AY"] = str(float(ay))
                                suffix += f"_ay{_fmt_tag_float(float(ay))}"
                            if th is not None:
                                overrides["MYEVS_EBF_S5_THETA_DEG"] = str(float(th))
                                suffix += f"_th{_fmt_tag_float(float(th))}"
                            sweep.append((suffix, overrides))

            elif variant == "s6":
                tc_list = _parse_float_list(args.s6_timecoh_thr_list) if str(args.s6_timecoh_thr_list).strip() else [None]
                raw_list = _parse_float_list(args.s6_raw_thr_list) if str(args.s6_raw_thr_list).strip() else [None]
                gam_list = _parse_float_list(args.s6_gamma_list) if str(args.s6_gamma_list).strip() else [None]
                sweep = []
                for tc in tc_list:
                    for raw in raw_list:
                        for gam in gam_list:
                            overrides: dict[str, str] = {}
                            suffix = ""
                            if tc is not None:
                                overrides["MYEVS_EBF_S6_TIMECOH_THR"] = str(float(tc))
                                suffix += f"_tc{_fmt_tag_float(float(tc))}"
                            if raw is not None:
                                overrides["MYEVS_EBF_S6_RAW_THR"] = str(float(raw))
                                suffix += f"_raw{_fmt_tag_float(float(raw))}"
                            if gam is not None:
                                overrides["MYEVS_EBF_S6_GAMMA"] = str(float(gam))
                                suffix += f"_g{_fmt_tag_float(float(gam))}"
                            sweep.append((suffix, overrides))

            elif variant == "s7":
                sig_list = _parse_float_list(args.s7_sigma_thr_list) if str(args.s7_sigma_thr_list).strip() else [None]
                raw_list = _parse_float_list(args.s7_raw_thr_list) if str(args.s7_raw_thr_list).strip() else [None]
                gam_list = _parse_float_list(args.s7_gamma_list) if str(args.s7_gamma_list).strip() else [None]
                mp_list = _parse_int_list(args.s7_min_pts_list) if str(args.s7_min_pts_list).strip() else [None]
                sweep = []
                for sig in sig_list:
                    for raw in raw_list:
                        for gam in gam_list:
                            for mp in mp_list:
                                overrides: dict[str, str] = {}
                                suffix = ""
                                if sig is not None:
                                    overrides["MYEVS_EBF_S7_SIGMA_THR"] = str(float(sig))
                                    suffix += f"_sig{_fmt_tag_float(float(sig))}"
                                if raw is not None:
                                    overrides["MYEVS_EBF_S7_RAW_THR"] = str(float(raw))
                                    suffix += f"_raw{_fmt_tag_float(float(raw))}"
                                if gam is not None:
                                    overrides["MYEVS_EBF_S7_GAMMA"] = str(float(gam))
                                    suffix += f"_g{_fmt_tag_float(float(gam))}"
                                if mp is not None:
                                    overrides["MYEVS_EBF_S7_MIN_PTS"] = str(int(mp))
                                    suffix += f"_mp{int(mp)}"
                                sweep.append((suffix, overrides))

            elif variant == "s8":
                r2_list = _parse_float_list(args.s8_r2_thr_list) if str(args.s8_r2_thr_list).strip() else [None]
                raw_list = _parse_float_list(args.s8_raw_thr_list) if str(args.s8_raw_thr_list).strip() else [None]
                gam_list = _parse_float_list(args.s8_gamma_list) if str(args.s8_gamma_list).strip() else [None]
                mp_list = _parse_int_list(args.s8_min_pts_list) if str(args.s8_min_pts_list).strip() else [None]
                sweep = []
                for r2 in r2_list:
                    for raw in raw_list:
                        for gam in gam_list:
                            for mp in mp_list:
                                overrides: dict[str, str] = {}
                                suffix = ""
                                if r2 is not None:
                                    overrides["MYEVS_EBF_S8_R2_THR"] = str(float(r2))
                                    suffix += f"_r2{_fmt_tag_float(float(r2))}"
                                if raw is not None:
                                    overrides["MYEVS_EBF_S8_RAW_THR"] = str(float(raw))
                                    suffix += f"_raw{_fmt_tag_float(float(raw))}"
                                if gam is not None:
                                    overrides["MYEVS_EBF_S8_GAMMA"] = str(float(gam))
                                    suffix += f"_g{_fmt_tag_float(float(gam))}"
                                if mp is not None:
                                    overrides["MYEVS_EBF_S8_MIN_PTS"] = str(int(mp))
                                    suffix += f"_mp{int(mp)}"
                                sweep.append((suffix, overrides))

            elif variant == "s9":
                dt_list = _parse_float_list(args.s9_dt_thr_list) if str(args.s9_dt_thr_list).strip() else [None]
                raw_list = _parse_float_list(args.s9_raw_thr_list) if str(args.s9_raw_thr_list).strip() else [None]
                gam_list = _parse_float_list(args.s9_gamma_list) if str(args.s9_gamma_list).strip() else [None]
                sweep = []
                for dt in dt_list:
                    for raw in raw_list:
                        for gam in gam_list:
                            overrides = {}
                            suffix = ""
                            if dt is not None:
                                overrides["MYEVS_EBF_S9_DT_THR"] = str(float(dt))
                                suffix += f"_dt{_fmt_tag_float(float(dt))}"
                            if raw is not None:
                                overrides["MYEVS_EBF_S9_RAW_THR"] = str(float(raw))
                                suffix += f"_raw{_fmt_tag_float(float(raw))}"
                            if gam is not None:
                                overrides["MYEVS_EBF_S9_GAMMA"] = str(float(gam))
                                suffix += f"_g{_fmt_tag_float(float(gam))}"
                            sweep.append((suffix, overrides))

            elif variant == "s22":
                dt_list = (
                    _parse_float_list(args.s22_dt_thr_us_list)
                    if str(args.s22_dt_thr_us_list).strip()
                    else [None]
                )
                raw_list = _parse_float_list(args.s22_raw_thr_list) if str(args.s22_raw_thr_list).strip() else [None]
                gam_list = _parse_float_list(args.s22_gamma_list) if str(args.s22_gamma_list).strip() else [None]
                sweep = []
                for dt_us in dt_list:
                    for raw in raw_list:
                        for gam in gam_list:
                            overrides = {}
                            suffix = ""
                            if dt_us is not None:
                                overrides["MYEVS_EBF_S22_DT_THR_US"] = str(float(dt_us))
                                suffix += f"_dtus{_fmt_tag_float(float(dt_us))}"
                            if raw is not None:
                                overrides["MYEVS_EBF_S22_RAW_THR"] = str(float(raw))
                                suffix += f"_raw{_fmt_tag_float(float(raw))}"
                            if gam is not None:
                                overrides["MYEVS_EBF_S22_GAMMA"] = str(float(gam))
                                suffix += f"_g{_fmt_tag_float(float(gam))}"
                            sweep.append((suffix, overrides))

            elif variant == "s10":
                acc_list = _parse_float_list(args.s10_acc_thr_list) if str(args.s10_acc_thr_list).strip() else [None]
                raw_list = _parse_float_list(args.s10_raw_thr_list) if str(args.s10_raw_thr_list).strip() else [None]
                gam_list = _parse_float_list(args.s10_gamma_list) if str(args.s10_gamma_list).strip() else [None]
                sweep = []
                for acc in acc_list:
                    for raw in raw_list:
                        for gam in gam_list:
                            overrides = {}
                            suffix = ""
                            if acc is not None:
                                overrides["MYEVS_EBF_S10_ACC_THR"] = str(float(acc))
                                suffix += f"_acc{_fmt_tag_float(float(acc))}"
                            if raw is not None:
                                overrides["MYEVS_EBF_S10_RAW_THR"] = str(float(raw))
                                suffix += f"_raw{_fmt_tag_float(float(raw))}"
                            if gam is not None:
                                overrides["MYEVS_EBF_S10_GAMMA"] = str(float(gam))
                                suffix += f"_g{_fmt_tag_float(float(gam))}"
                            sweep.append((suffix, overrides))

            elif variant == "s11":
                acc_list = _parse_float_list(args.s11_acc_thr_list) if str(args.s11_acc_thr_list).strip() else [None]
                ratio_list = (
                    _parse_float_list(args.s11_ratio_thr_list) if str(args.s11_ratio_thr_list).strip() else [None]
                )
                raw_list = _parse_float_list(args.s11_raw_thr_list) if str(args.s11_raw_thr_list).strip() else [None]
                gam_list = _parse_float_list(args.s11_gamma_list) if str(args.s11_gamma_list).strip() else [None]
                sweep = []
                for acc in acc_list:
                    for ratio in ratio_list:
                        for raw in raw_list:
                            for gam in gam_list:
                                overrides = {}
                                suffix = ""
                                if acc is not None:
                                    overrides["MYEVS_EBF_S11_ACC_THR"] = str(float(acc))
                                    suffix += f"_acc{_fmt_tag_float(float(acc))}"
                                if ratio is not None:
                                    overrides["MYEVS_EBF_S11_RATIO_THR"] = str(float(ratio))
                                    suffix += f"_ratio{_fmt_tag_float(float(ratio))}"
                                if raw is not None:
                                    overrides["MYEVS_EBF_S11_RAW_THR"] = str(float(raw))
                                    suffix += f"_raw{_fmt_tag_float(float(raw))}"
                                if gam is not None:
                                    overrides["MYEVS_EBF_S11_GAMMA"] = str(float(gam))
                                    suffix += f"_g{_fmt_tag_float(float(gam))}"
                                sweep.append((suffix, overrides))

            elif variant == "s12":
                acc_list = _parse_float_list(args.s12_acc_thr_list) if str(args.s12_acc_thr_list).strip() else [None]
                z_list = _parse_float_list(args.s12_z_thr_list) if str(args.s12_z_thr_list).strip() else [None]
                raw_list = _parse_float_list(args.s12_raw_thr_list) if str(args.s12_raw_thr_list).strip() else [None]
                gam_list = _parse_float_list(args.s12_gamma_list) if str(args.s12_gamma_list).strip() else [None]
                sweep = []
                for acc in acc_list:
                    for zthr in z_list:
                        for raw in raw_list:
                            for gam in gam_list:
                                overrides = {}
                                suffix = ""
                                if acc is not None:
                                    overrides["MYEVS_EBF_S12_ACC_THR"] = str(float(acc))
                                    suffix += f"_acc{_fmt_tag_float(float(acc))}"
                                if zthr is not None:
                                    overrides["MYEVS_EBF_S12_Z_THR"] = str(float(zthr))
                                    suffix += f"_z{_fmt_tag_float(float(zthr))}"
                                if raw is not None:
                                    overrides["MYEVS_EBF_S12_RAW_THR"] = str(float(raw))
                                    suffix += f"_raw{_fmt_tag_float(float(raw))}"
                                if gam is not None:
                                    overrides["MYEVS_EBF_S12_GAMMA"] = str(float(gam))
                                    suffix += f"_g{_fmt_tag_float(float(gam))}"
                                sweep.append((suffix, overrides))

            elif variant == "s13":
                bal_list = _parse_float_list(args.s13_bal_thr_list) if str(args.s13_bal_thr_list).strip() else [None]
                raw_list = _parse_float_list(args.s13_raw_thr_list) if str(args.s13_raw_thr_list).strip() else [None]
                gam_list = _parse_float_list(args.s13_gamma_list) if str(args.s13_gamma_list).strip() else [None]
                sweep = []
                for bal in bal_list:
                    for raw in raw_list:
                        for gam in gam_list:
                            overrides = {}
                            suffix = ""
                            if bal is not None:
                                overrides["MYEVS_EBF_S13_BAL_THR"] = str(float(bal))
                                suffix += f"_bal{_fmt_tag_float(float(bal))}"
                            if raw is not None:
                                overrides["MYEVS_EBF_S13_RAW_THR"] = str(float(raw))
                                suffix += f"_raw{_fmt_tag_float(float(raw))}"
                            if gam is not None:
                                overrides["MYEVS_EBF_S13_GAMMA"] = str(float(gam))
                                suffix += f"_g{_fmt_tag_float(float(gam))}"
                            sweep.append((suffix, overrides))

            elif variant == "s14":
                a_list = _parse_float_list(args.s14_alpha_list) if str(args.s14_alpha_list).strip() else [None]
                raw_list = _parse_float_list(args.s14_raw_thr_list) if str(args.s14_raw_thr_list).strip() else [None]
                sweep = []
                for a in a_list:
                    for raw in raw_list:
                        overrides = {}
                        suffix = ""
                        if a is not None:
                            overrides["MYEVS_EBF_S14_ALPHA"] = str(float(a))
                            suffix += f"_a{_fmt_tag_float(float(a))}"
                        if raw is not None:
                            overrides["MYEVS_EBF_S14_RAW_THR"] = str(float(raw))
                            suffix += f"_raw{_fmt_tag_float(float(raw))}"
                        sweep.append((suffix, overrides))

            elif variant == "s24":
                a_list = _parse_float_list(args.s24_alpha_list) if str(args.s24_alpha_list).strip() else [None]
                raw_list = _parse_float_list(args.s24_raw_thr_list) if str(args.s24_raw_thr_list).strip() else [None]
                bdt_list = (
                    _parse_float_list(args.s24_burst_dt_us_list) if str(args.s24_burst_dt_us_list).strip() else [None]
                )
                bthr_list = _parse_float_list(args.s24_b_thr_list) if str(args.s24_b_thr_list).strip() else [None]
                gam_list = _parse_float_list(args.s24_gamma_list) if str(args.s24_gamma_list).strip() else [None]
                sweep = []
                for a in a_list:
                    for raw in raw_list:
                        for bdt_us in bdt_list:
                            for bthr in bthr_list:
                                for gam in gam_list:
                                    overrides = {}
                                    suffix = ""
                                    if a is not None:
                                        overrides["MYEVS_EBF_S24_ALPHA"] = str(float(a))
                                        suffix += f"_a{_fmt_tag_float(float(a))}"
                                    if raw is not None:
                                        overrides["MYEVS_EBF_S24_RAW_THR"] = str(float(raw))
                                        suffix += f"_raw{_fmt_tag_float(float(raw))}"
                                    if bdt_us is not None:
                                        overrides["MYEVS_EBF_S24_BURST_DT_US"] = str(float(bdt_us))
                                        suffix += f"_bdt{_fmt_tag_float(float(bdt_us))}us"
                                    if bthr is not None:
                                        overrides["MYEVS_EBF_S24_B_THR"] = str(float(bthr))
                                        suffix += f"_b{_fmt_tag_float(float(bthr))}"
                                    if gam is not None:
                                        overrides["MYEVS_EBF_S24_GAMMA"] = str(float(gam))
                                        suffix += f"_g{_fmt_tag_float(float(gam))}"
                                    sweep.append((suffix, overrides))

            elif variant == "s25":
                a_list = _parse_float_list(args.s25_alpha_list) if str(args.s25_alpha_list).strip() else [None]
                raw_list = _parse_float_list(args.s25_raw_thr_list) if str(args.s25_raw_thr_list).strip() else [None]
                dt_list = _parse_float_list(args.s25_dt_thr_list) if str(args.s25_dt_thr_list).strip() else [None]
                ref_raw_list = (
                    _parse_float_list(args.s25_ref_raw_thr_list) if str(args.s25_ref_raw_thr_list).strip() else [None]
                )
                gam_list = _parse_float_list(args.s25_gamma_list) if str(args.s25_gamma_list).strip() else [None]
                sweep = []
                for a in a_list:
                    for raw in raw_list:
                        for dt in dt_list:
                            for ref_raw in ref_raw_list:
                                for gam in gam_list:
                                    overrides = {}
                                    suffix = ""
                                    if a is not None:
                                        overrides["MYEVS_EBF_S25_ALPHA"] = str(float(a))
                                        suffix += f"_a{_fmt_tag_float(float(a))}"
                                    if raw is not None:
                                        overrides["MYEVS_EBF_S25_RAW_THR"] = str(float(raw))
                                        suffix += f"_raw{_fmt_tag_float(float(raw))}"
                                    if dt is not None:
                                        overrides["MYEVS_EBF_S25_DT_THR"] = str(float(dt))
                                        suffix += f"_dt{_fmt_tag_float(float(dt))}"
                                    if ref_raw is not None:
                                        overrides["MYEVS_EBF_S25_REF_RAW_THR"] = str(float(ref_raw))
                                        suffix += f"_rraw{_fmt_tag_float(float(ref_raw))}"
                                    if gam is not None:
                                        overrides["MYEVS_EBF_S25_GAMMA"] = str(float(gam))
                                        suffix += f"_g{_fmt_tag_float(float(gam))}"
                                    sweep.append((suffix, overrides))

            elif variant == "s15":
                a_list = _parse_float_list(args.s15_alpha_list) if str(args.s15_alpha_list).strip() else [None]
                raw_list = _parse_float_list(args.s15_raw_thr_list) if str(args.s15_raw_thr_list).strip() else [None]
                dt_list = (
                    _parse_float_list(args.s15_flip_dt_us_list) if str(args.s15_flip_dt_us_list).strip() else [None]
                )
                beta_list = _parse_float_list(args.s15_beta_list) if str(args.s15_beta_list).strip() else [None]
                sweep = []
                for a in a_list:
                    for raw in raw_list:
                        for dt_us in dt_list:
                            for beta in beta_list:
                                overrides = {}
                                suffix = ""
                                if a is not None:
                                    overrides["MYEVS_EBF_S15_ALPHA"] = str(float(a))
                                    suffix += f"_a{_fmt_tag_float(float(a))}"
                                if raw is not None:
                                    overrides["MYEVS_EBF_S15_RAW_THR"] = str(float(raw))
                                    suffix += f"_raw{_fmt_tag_float(float(raw))}"
                                if dt_us is not None:
                                    overrides["MYEVS_EBF_S15_FLIP_DT_US"] = str(float(dt_us))
                                    suffix += f"_fdt{_fmt_tag_float(float(dt_us))}us"
                                if beta is not None:
                                    overrides["MYEVS_EBF_S15_BETA"] = str(float(beta))
                                    suffix += f"_b{_fmt_tag_float(float(beta))}"
                                sweep.append((suffix, overrides))

            elif variant == "s16":
                a_list = _parse_float_list(args.s16_alpha_list) if str(args.s16_alpha_list).strip() else [None]
                raw_list = _parse_float_list(args.s16_raw_thr_list) if str(args.s16_raw_thr_list).strip() else [None]
                acc_list = _parse_float_list(args.s16_acc_thr_list) if str(args.s16_acc_thr_list).strip() else [None]
                ratio_list = (
                    _parse_float_list(args.s16_ratio_thr_list) if str(args.s16_ratio_thr_list).strip() else [None]
                )
                gam_list = _parse_float_list(args.s16_gamma_list) if str(args.s16_gamma_list).strip() else [None]
                sweep = []
                for a in a_list:
                    for raw in raw_list:
                        for acc in acc_list:
                            for ratio in ratio_list:
                                for gam in gam_list:
                                    overrides = {}
                                    suffix = ""
                                    if a is not None:
                                        overrides["MYEVS_EBF_S16_ALPHA"] = str(float(a))
                                        suffix += f"_a{_fmt_tag_float(float(a))}"
                                    if raw is not None:
                                        overrides["MYEVS_EBF_S16_RAW_THR"] = str(float(raw))
                                        suffix += f"_raw{_fmt_tag_float(float(raw))}"
                                    if acc is not None:
                                        overrides["MYEVS_EBF_S16_ACC_THR"] = str(float(acc))
                                        suffix += f"_acc{_fmt_tag_float(float(acc))}"
                                    if ratio is not None:
                                        overrides["MYEVS_EBF_S16_RATIO_THR"] = str(float(ratio))
                                        suffix += f"_ratio{_fmt_tag_float(float(ratio))}"
                                    if gam is not None:
                                        overrides["MYEVS_EBF_S16_GAMMA"] = str(float(gam))
                                        suffix += f"_g{_fmt_tag_float(float(gam))}"
                                    sweep.append((suffix, overrides))

            elif variant == "s17":
                a_list = _parse_float_list(args.s17_alpha_list) if str(args.s17_alpha_list).strip() else [None]
                raw_list = _parse_float_list(args.s17_raw_thr_list) if str(args.s17_raw_thr_list).strip() else [None]
                var_list = _parse_float_list(args.s17_var_thr_list) if str(args.s17_var_thr_list).strip() else [None]
                beta_list = _parse_float_list(args.s17_beta_list) if str(args.s17_beta_list).strip() else [None]
                gam_list = _parse_float_list(args.s17_gamma_list) if str(args.s17_gamma_list).strip() else [None]
                sweep = []
                for a in a_list:
                    for raw in raw_list:
                        for var_thr in var_list:
                            for beta in beta_list:
                                for gam in gam_list:
                                    overrides = {}
                                    suffix = ""
                                    if a is not None:
                                        overrides["MYEVS_EBF_S17_ALPHA"] = str(float(a))
                                        suffix += f"_a{_fmt_tag_float(float(a))}"
                                    if raw is not None:
                                        overrides["MYEVS_EBF_S17_RAW_THR"] = str(float(raw))
                                        suffix += f"_raw{_fmt_tag_float(float(raw))}"
                                    if var_thr is not None:
                                        overrides["MYEVS_EBF_S17_VAR_THR"] = str(float(var_thr))
                                        suffix += f"_var{_fmt_tag_float(float(var_thr))}"
                                    if beta is not None:
                                        overrides["MYEVS_EBF_S17_BETA"] = str(float(beta))
                                        suffix += f"_b{_fmt_tag_float(float(beta))}"
                                    if gam is not None:
                                        overrides["MYEVS_EBF_S17_GAMMA"] = str(float(gam))
                                        suffix += f"_g{_fmt_tag_float(float(gam))}"
                                    sweep.append((suffix, overrides))

            elif variant == "s19":
                # Hardware-friendly evidence fusion. Default alpha fixed to 0.2 (user requested), sweep beta.
                a_list = _parse_float_list(args.s19_alpha_list) if str(args.s19_alpha_list).strip() else [0.2]
                b_list = _parse_float_list(args.s19_beta_list) if str(args.s19_beta_list).strip() else [0.1]
                sweep = []
                for a in a_list:
                    for b in b_list:
                        overrides = {
                            "MYEVS_EBF_S19_ALPHA": str(float(a)),
                            "MYEVS_EBF_S19_BETA": str(float(b)),
                        }
                        suffix = f"_a{_fmt_tag_float(float(a))}_b{_fmt_tag_float(float(b))}"
                        sweep.append((suffix, overrides))

            elif variant == "s20":
                # Polarity-aware hotness evidence fusion. Default alpha fixed to 0.2, sweep beta.
                a_list = _parse_float_list(args.s20_alpha_list) if str(args.s20_alpha_list).strip() else [0.2]
                b_list = _parse_float_list(args.s20_beta_list) if str(args.s20_beta_list).strip() else [0.5]
                sweep = []
                for a in a_list:
                    for b in b_list:
                        overrides = {
                            "MYEVS_EBF_S20_ALPHA": str(float(a)),
                            "MYEVS_EBF_S20_BETA": str(float(b)),
                        }
                        suffix = f"_a{_fmt_tag_float(float(a))}_b{_fmt_tag_float(float(b))}"
                        sweep.append((suffix, overrides))

            elif variant == "s21":
                # Bi-polar hotness mix fusion. Default alpha fixed to 0.2, sweep beta/kappa.
                a_list = _parse_float_list(args.s21_alpha_list) if str(args.s21_alpha_list).strip() else [0.2]
                b_list = _parse_float_list(args.s21_beta_list) if str(args.s21_beta_list).strip() else [0.5]
                k_list = _parse_float_list(args.s21_kappa_list) if str(args.s21_kappa_list).strip() else [0.5]
                sweep = []
                for a in a_list:
                    for b in b_list:
                        for k in k_list:
                            overrides = {
                                "MYEVS_EBF_S21_ALPHA": str(float(a)),
                                "MYEVS_EBF_S21_BETA": str(float(b)),
                                "MYEVS_EBF_S21_KAPPA": str(float(k)),
                            }
                            suffix = f"_a{_fmt_tag_float(float(a))}_b{_fmt_tag_float(float(b))}_k{_fmt_tag_float(float(k))}"
                            sweep.append((suffix, overrides))

            elif variant == "s26":
                # Activity-normalized hotness fusion. Default alpha fixed to 0.2.
                a_list = _parse_float_list(args.s26_alpha_list) if str(args.s26_alpha_list).strip() else [0.2]
                b_list = _parse_float_list(args.s26_beta_list) if str(args.s26_beta_list).strip() else [0.8]
                k_list = _parse_float_list(args.s26_kappa_list) if str(args.s26_kappa_list).strip() else [1.0]
                e_list = _parse_float_list(args.s26_eta_list) if str(args.s26_eta_list).strip() else [1.0]
                sweep = []
                for a in a_list:
                    for b in b_list:
                        for k in k_list:
                            for e in e_list:
                                overrides = {
                                    "MYEVS_EBF_S26_ALPHA": str(float(a)),
                                    "MYEVS_EBF_S26_BETA": str(float(b)),
                                    "MYEVS_EBF_S26_KAPPA": str(float(k)),
                                    "MYEVS_EBF_S26_ETA": str(float(e)),
                                }
                                suffix = (
                                    f"_a{_fmt_tag_float(float(a))}"
                                    f"_b{_fmt_tag_float(float(b))}"
                                    f"_k{_fmt_tag_float(float(k))}"
                                    f"_e{_fmt_tag_float(float(e))}"
                                )
                                sweep.append((suffix, overrides))

            elif variant == "s27":
                # Relative abnormal hotness fusion. Default alpha fixed to 0.2.
                a_list = _parse_float_list(args.s27_alpha_list) if str(args.s27_alpha_list).strip() else [0.2]
                b_list = _parse_float_list(args.s27_beta_list) if str(args.s27_beta_list).strip() else [0.8]
                k_list = _parse_float_list(args.s27_kappa_list) if str(args.s27_kappa_list).strip() else [1.0]
                l_list = (
                    _parse_float_list(args.s27_lambda_nb_list)
                    if str(args.s27_lambda_nb_list).strip()
                    else [1.0]
                )
                sweep = []
                for a in a_list:
                    for b in b_list:
                        for k in k_list:
                            for l in l_list:
                                overrides = {
                                    "MYEVS_EBF_S27_ALPHA": str(float(a)),
                                    "MYEVS_EBF_S27_BETA": str(float(b)),
                                    "MYEVS_EBF_S27_KAPPA": str(float(k)),
                                    "MYEVS_EBF_S27_LAMBDA_NB": str(float(l)),
                                }
                                suffix = (
                                    f"_a{_fmt_tag_float(float(a))}"
                                    f"_b{_fmt_tag_float(float(b))}"
                                    f"_k{_fmt_tag_float(float(k))}"
                                    f"_l{_fmt_tag_float(float(l))}"
                                )
                                sweep.append((suffix, overrides))

            elif variant == "s28":
                # Noise-model surprise z-score. Default tau_rate_us=0 (auto).
                tr_list = (
                    _parse_float_list(args.s28_tau_rate_us_list)
                    if str(args.s28_tau_rate_us_list).strip()
                    else [0.0]
                )
                sweep = []
                for tr_us in tr_list:
                    overrides: dict[str, str] = {}
                    suffix = ""
                    if tr_us is not None and float(tr_us) > 0.0:
                        overrides["MYEVS_EBF_S28_TAU_RATE_US"] = str(int(float(tr_us)))
                        suffix = f"_tr{_fmt_tag_float(float(tr_us))}"
                    sweep.append((suffix, overrides))

            elif variant == "s35":
                # s28 surprise z-score + pixel-state-conditioned adaptive null.
                # Defaults: tau_rate_us=0(auto), gamma=1.0, hmax=8.0.
                tr_list = (
                    _parse_float_list(args.s35_tau_rate_us_list)
                    if str(args.s35_tau_rate_us_list).strip()
                    else [0.0]
                )
                g_list = (
                    _parse_float_list(args.s35_gamma_list)
                    if str(args.s35_gamma_list).strip()
                    else [1.0]
                )
                h_list = (
                    _parse_float_list(args.s35_hmax_list)
                    if str(args.s35_hmax_list).strip()
                    else [8.0]
                )
                sweep = []
                for tr_us in tr_list:
                    for g in g_list:
                        for hm in h_list:
                            overrides: dict[str, str] = {
                                "MYEVS_EBF_S35_GAMMA": str(float(g)),
                                "MYEVS_EBF_S35_HMAX": str(float(hm)),
                            }
                            suffix = f"_g{_fmt_tag_float(float(g))}_h{_fmt_tag_float(float(hm))}"
                            if tr_us is not None and float(tr_us) > 0.0:
                                overrides["MYEVS_EBF_S35_TAU_RATE_US"] = str(int(float(tr_us)))
                                suffix = f"_tr{_fmt_tag_float(float(tr_us))}" + suffix
                            sweep.append((suffix, overrides))

            elif variant == "s36":
                # s28 surprise z-score + state-occupancy-conditioned adaptive null.
                # Default tau_rate_us=0 (auto).
                tr_list = (
                    _parse_float_list(args.s36_tau_rate_us_list)
                    if str(args.s36_tau_rate_us_list).strip()
                    else [0.0]
                )
                sweep = []
                for tr_us in tr_list:
                    overrides: dict[str, str] = {}
                    suffix = ""
                    if tr_us is not None and float(tr_us) > 0.0:
                        overrides["MYEVS_EBF_S36_TAU_RATE_US"] = str(int(float(tr_us)))
                        suffix = f"_tr{_fmt_tag_float(float(tr_us))}"
                    sweep.append((suffix, overrides))

            elif variant == "s37":
                # s36 + 3-state occupancy->rate mapping. Default tau_rate_us=0 (auto).
                tr_list = (
                    _parse_float_list(args.s37_tau_rate_us_list)
                    if str(args.s37_tau_rate_us_list).strip()
                    else [0.0]
                )
                sweep = []
                for tr_us in tr_list:
                    overrides: dict[str, str] = {}
                    suffix = ""
                    if tr_us is not None and float(tr_us) > 0.0:
                        overrides["MYEVS_EBF_S37_TAU_RATE_US"] = str(int(float(tr_us)))
                        suffix = f"_tr{_fmt_tag_float(float(tr_us))}"
                    sweep.append((suffix, overrides))

            elif variant == "s38":
                # s36 + neighborhood-occupancy fusion. Default tau_rate_us=0 (auto).
                tr_list = (
                    _parse_float_list(args.s38_tau_rate_us_list)
                    if str(args.s38_tau_rate_us_list).strip()
                    else [0.0]
                )
                sweep = []
                for tr_us in tr_list:
                    overrides: dict[str, str] = {}
                    suffix = ""
                    if tr_us is not None and float(tr_us) > 0.0:
                        overrides["MYEVS_EBF_S38_TAU_RATE_US"] = str(int(float(tr_us)))
                        suffix = f"_tr{_fmt_tag_float(float(tr_us))}"
                    sweep.append((suffix, overrides))

            elif variant == "s39":
                # s38 + polarity-mix-weighted neighborhood occupancy.
                # Defaults: tau_rate_us=0 (auto), k_nbmix=1.0.
                tr_list = (
                    _parse_float_list(args.s39_tau_rate_us_list)
                    if str(args.s39_tau_rate_us_list).strip()
                    else [0.0]
                )
                k_list = (
                    _parse_float_list(args.s39_k_nbmix_list)
                    if str(args.s39_k_nbmix_list).strip()
                    else [1.0]
                )

                sweep = []
                for tr_us in tr_list:
                    for k in k_list:
                        overrides: dict[str, str] = {"MYEVS_EBF_S39_K_NBMIX": str(float(k))}
                        suffix = f"_kn{_fmt_tag_float(float(k))}"
                        if tr_us is not None and float(tr_us) > 0.0:
                            overrides["MYEVS_EBF_S39_TAU_RATE_US"] = str(int(float(tr_us)))
                            suffix = f"_tr{_fmt_tag_float(float(tr_us))}" + suffix
                        sweep.append((suffix, overrides))

            elif variant == "s40":
                # s39 + conservative geometric-mean fusion.
                # Defaults: tau_rate_us=0 (auto), k_nbmix=1.0.
                tr_list = (
                    _parse_float_list(args.s40_tau_rate_us_list)
                    if str(args.s40_tau_rate_us_list).strip()
                    else [0.0]
                )
                k_list = (
                    _parse_float_list(args.s40_k_nbmix_list)
                    if str(args.s40_k_nbmix_list).strip()
                    else [1.0]
                )

                sweep = []
                for tr_us in tr_list:
                    for k in k_list:
                        overrides: dict[str, str] = {"MYEVS_EBF_S40_K_NBMIX": str(float(k))}
                        suffix = f"_kn{_fmt_tag_float(float(k))}"
                        if tr_us is not None and float(tr_us) > 0.0:
                            overrides["MYEVS_EBF_S40_TAU_RATE_US"] = str(int(float(tr_us)))
                            suffix = f"_tr{_fmt_tag_float(float(tr_us))}" + suffix
                        sweep.append((suffix, overrides))

            elif variant == "s41":
                # s39 + conservative mix^2 shaping.
                # Defaults: tau_rate_us=0 (auto), k_nbmix=1.0.
                tr_list = (
                    _parse_float_list(args.s41_tau_rate_us_list)
                    if str(args.s41_tau_rate_us_list).strip()
                    else [0.0]
                )
                k_list = (
                    _parse_float_list(args.s41_k_nbmix_list)
                    if str(args.s41_k_nbmix_list).strip()
                    else [1.0]
                )

                sweep = []
                for tr_us in tr_list:
                    for k in k_list:
                        overrides: dict[str, str] = {"MYEVS_EBF_S41_K_NBMIX": str(float(k))}
                        suffix = f"_kn{_fmt_tag_float(float(k))}"
                        if tr_us is not None and float(tr_us) > 0.0:
                            overrides["MYEVS_EBF_S41_TAU_RATE_US"] = str(int(float(tr_us)))
                            suffix = f"_tr{_fmt_tag_float(float(tr_us))}" + suffix
                        sweep.append((suffix, overrides))

            elif variant == "s42":
                # s39 + u_self^2-gated neighborhood term.
                # Defaults: tau_rate_us=0 (auto), k_nbmix=1.0.
                tr_list = (
                    _parse_float_list(args.s42_tau_rate_us_list)
                    if str(args.s42_tau_rate_us_list).strip()
                    else [0.0]
                )
                k_list = (
                    _parse_float_list(args.s42_k_nbmix_list)
                    if str(args.s42_k_nbmix_list).strip()
                    else [1.0]
                )

                sweep = []
                for tr_us in tr_list:
                    for k in k_list:
                        overrides: dict[str, str] = {"MYEVS_EBF_S42_K_NBMIX": str(float(k))}
                        suffix = f"_kn{_fmt_tag_float(float(k))}"
                        if tr_us is not None and float(tr_us) > 0.0:
                            overrides["MYEVS_EBF_S42_TAU_RATE_US"] = str(int(float(tr_us)))
                            suffix = f"_tr{_fmt_tag_float(float(tr_us))}" + suffix
                        sweep.append((suffix, overrides))

            elif variant == "s43":
                # s39 + u_eff^2 compression (more conservative rate modulation).
                # Defaults: tau_rate_us=0 (auto), k_nbmix=1.0.
                tr_list = (
                    _parse_float_list(args.s43_tau_rate_us_list)
                    if str(args.s43_tau_rate_us_list).strip()
                    else [0.0]
                )
                k_list = (
                    _parse_float_list(args.s43_k_nbmix_list)
                    if str(args.s43_k_nbmix_list).strip()
                    else [1.0]
                )

                sweep = []
                for tr_us in tr_list:
                    for k in k_list:
                        overrides: dict[str, str] = {"MYEVS_EBF_S43_K_NBMIX": str(float(k))}
                        suffix = f"_kn{_fmt_tag_float(float(k))}"
                        if tr_us is not None and float(tr_us) > 0.0:
                            overrides["MYEVS_EBF_S43_TAU_RATE_US"] = str(int(float(tr_us)))
                            suffix = f"_tr{_fmt_tag_float(float(tr_us))}" + suffix
                        sweep.append((suffix, overrides))

            elif variant == "s44":
                # Baseline EBF labelscore with self-occupancy penalty.
                # Default: tau_rate_us=0 (auto).
                tr_list = (
                    _parse_float_list(args.s44_tau_rate_us_list)
                    if str(args.s44_tau_rate_us_list).strip()
                    else [0.0]
                )
                sweep = []
                for tr_us in tr_list:
                    overrides: dict[str, str] = {}
                    suffix = ""
                    if tr_us is not None and float(tr_us) > 0.0:
                        overrides["MYEVS_EBF_S44_TAU_RATE_US"] = str(int(float(tr_us)))
                        suffix = f"_tr{_fmt_tag_float(float(tr_us))}"
                    sweep.append((suffix, overrides))

            elif variant == "s45":
                # s44 + gated penalty (u0). Defaults: tau_rate_us=0 (auto), u0=0.0 (==s44).
                tr_list = (
                    _parse_float_list(args.s45_tau_rate_us_list)
                    if str(args.s45_tau_rate_us_list).strip()
                    else [0.0]
                )
                u0_list = (
                    _parse_float_list(args.s45_u0_list)
                    if str(args.s45_u0_list).strip()
                    else [0.0]
                )
                sweep = []
                for tr_us in tr_list:
                    for u0 in u0_list:
                        overrides: dict[str, str] = {"MYEVS_EBF_S45_U0": str(float(u0))}
                        suffix = f"_u0{_fmt_tag_float(float(u0))}"
                        if tr_us is not None and float(tr_us) > 0.0:
                            overrides["MYEVS_EBF_S45_TAU_RATE_US"] = str(int(float(tr_us)))
                            suffix = f"_tr{_fmt_tag_float(float(tr_us))}" + suffix
                        sweep.append((suffix, overrides))

            elif variant == "s46":
                # s44 + odds(u_self)^2 penalty (stronger near u->1). Default: tau_rate_us=0 (auto).
                tr_list = (
                    _parse_float_list(args.s46_tau_rate_us_list)
                    if str(args.s46_tau_rate_us_list).strip()
                    else [0.0]
                )
                sweep = []
                for tr_us in tr_list:
                    overrides: dict[str, str] = {}
                    suffix = ""
                    if tr_us is not None and float(tr_us) > 0.0:
                        overrides["MYEVS_EBF_S46_TAU_RATE_US"] = str(int(float(tr_us)))
                        suffix = f"_tr{_fmt_tag_float(float(tr_us))}"
                    sweep.append((suffix, overrides))

            elif variant == "s47":
                # s44 with abnormal-activity scaled hot_state. Default: tau_rate_us=0 (auto).
                tr_list = (
                    _parse_float_list(args.s47_tau_rate_us_list)
                    if str(args.s47_tau_rate_us_list).strip()
                    else [0.0]
                )
                sweep = []
                for tr_us in tr_list:
                    overrides: dict[str, str] = {}
                    suffix = ""
                    if tr_us is not None and float(tr_us) > 0.0:
                        overrides["MYEVS_EBF_S47_TAU_RATE_US"] = str(int(float(tr_us)))
                        suffix = f"_tr{_fmt_tag_float(float(tr_us))}"
                    sweep.append((suffix, overrides))

            elif variant == "s48":
                # s44 with polarity-persistence scaled hot_state. Defaults: tau_rate_us=0 (auto), eta_toggle=0.25.
                tr_list = (
                    _parse_float_list(args.s48_tau_rate_us_list)
                    if str(args.s48_tau_rate_us_list).strip()
                    else [0.0]
                )
                eta_list = (
                    _parse_float_list(args.s48_eta_toggle_list)
                    if str(args.s48_eta_toggle_list).strip()
                    else [0.25]
                )
                sweep = []
                for tr_us in tr_list:
                    for eta in eta_list:
                        overrides: dict[str, str] = {"MYEVS_EBF_S48_ETA_TOGGLE": str(float(eta))}
                        suffix = f"_eta{_fmt_tag_float(float(eta))}"
                        if tr_us is not None and float(tr_us) > 0.0:
                            overrides["MYEVS_EBF_S48_TAU_RATE_US"] = str(int(float(tr_us)))
                            suffix = f"_tr{_fmt_tag_float(float(tr_us))}" + suffix
                        sweep.append((suffix, overrides))

            elif variant == "s49":
                # s44 with bipolar (pos/neg) self-state, u_self=max(Hpos,Hneg). Default: tau_rate_us=0 (auto).
                tr_list = (
                    _parse_float_list(args.s49_tau_rate_us_list)
                    if str(args.s49_tau_rate_us_list).strip()
                    else [0.0]
                )
                sweep = []
                for tr_us in tr_list:
                    overrides: dict[str, str] = {}
                    suffix = ""
                    if tr_us is not None and float(tr_us) > 0.0:
                        overrides["MYEVS_EBF_S49_TAU_RATE_US"] = str(int(float(tr_us)))
                        suffix = f"_tr{_fmt_tag_float(float(tr_us))}"
                    sweep.append((suffix, overrides))

            elif variant == "s50":
                # s44 + gentle support-breadth boost. Defaults: tau_rate_us=0 (auto), beta=0.0, cnt0=8.
                tr_list = (
                    _parse_float_list(args.s50_tau_rate_us_list)
                    if str(args.s50_tau_rate_us_list).strip()
                    else [0.0]
                )
                beta_list = (
                    _parse_float_list(args.s50_beta_list)
                    if str(args.s50_beta_list).strip()
                    else [0.0]
                )
                cnt0_list = (
                    _parse_float_list(args.s50_cnt0_list)
                    if str(args.s50_cnt0_list).strip()
                    else [8.0]
                )

                sweep = []
                for tr_us in tr_list:
                    for beta in beta_list:
                        for cnt0 in cnt0_list:
                            overrides: dict[str, str] = {
                                "MYEVS_EBF_S50_BETA": str(float(beta)),
                                "MYEVS_EBF_S50_CNT0": str(int(float(cnt0))),
                            }
                            suffix = f"_b{_fmt_tag_float(float(beta))}_c{int(float(cnt0))}"
                            if tr_us is not None and float(tr_us) > 0.0:
                                overrides["MYEVS_EBF_S50_TAU_RATE_US"] = str(int(float(tr_us)))
                                suffix = f"_tr{_fmt_tag_float(float(tr_us))}" + suffix
                            sweep.append((suffix, overrides))

            elif variant == "s51":
                # s50 without env-sensitive hyperparams: auto beta + normalized support fraction.
                # No overrides.
                sweep = [("", {})]

            elif variant == "s52":
                # s51 + auto-gated opposite-polarity neighborhood evidence by global polarity mix.
                # No overrides.
                sweep = [("", {})]

            elif variant == "s53":
                # s51 + opposite-polarity evidence gated per-event by local mix and spatial support.
                # No overrides.
                sweep = [("", {})]

            elif variant == "s54":
                # s53 shape tweak: opp weight uses sfrac^(1/4) instead of sqrt(sfrac).
                # No overrides.
                sweep = [("", {})]

            elif variant == "s30":
                # s28 surprise z-score + local-rate correction. Default tau_rate_us=0 (auto).
                tr_list = (
                    _parse_float_list(args.s30_tau_rate_us_list)
                    if str(args.s30_tau_rate_us_list).strip()
                    else [0.0]
                )
                sweep = []
                for tr_us in tr_list:
                    overrides: dict[str, str] = {}
                    suffix = ""
                    if tr_us is not None and float(tr_us) > 0.0:
                        overrides["MYEVS_EBF_S30_TAU_RATE_US"] = str(int(float(tr_us)))
                        suffix = f"_tr{_fmt_tag_float(float(tr_us))}"
                    sweep.append((suffix, overrides))

            elif variant == "s31":
                # s28 surprise z-score + polarity-bias correction. Default tau_rate_us=0 (auto).
                tr_list = (
                    _parse_float_list(args.s31_tau_rate_us_list)
                    if str(args.s31_tau_rate_us_list).strip()
                    else [0.0]
                )
                sweep = []
                for tr_us in tr_list:
                    overrides: dict[str, str] = {}
                    suffix = ""
                    if tr_us is not None and float(tr_us) > 0.0:
                        overrides["MYEVS_EBF_S31_TAU_RATE_US"] = str(int(float(tr_us)))
                        suffix = f"_tr{_fmt_tag_float(float(tr_us))}"
                    sweep.append((suffix, overrides))

            elif variant == "s32":
                # s28 surprise z-score + block-rate max correction. Default tau_rate_us=0 (auto).
                tr_list = (
                    _parse_float_list(args.s32_tau_rate_us_list)
                    if str(args.s32_tau_rate_us_list).strip()
                    else [0.0]
                )
                sweep = []
                for tr_us in tr_list:
                    overrides: dict[str, str] = {}
                    suffix = ""
                    if tr_us is not None and float(tr_us) > 0.0:
                        overrides["MYEVS_EBF_S32_TAU_RATE_US"] = str(int(float(tr_us)))
                        suffix = f"_tr{_fmt_tag_float(float(tr_us))}"
                    sweep.append((suffix, overrides))

            elif variant == "s33":
                # s28 surprise z-score + abnormal-hotness penalty. Default tau_rate_us=0 (auto), beta=0.5.
                tr_list = (
                    _parse_float_list(args.s33_tau_rate_us_list)
                    if str(args.s33_tau_rate_us_list).strip()
                    else [0.0]
                )
                b_list = (
                    _parse_float_list(args.s33_beta_list)
                    if str(args.s33_beta_list).strip()
                    else [0.5]
                )
                sweep = []
                for tr_us in tr_list:
                    for b in b_list:
                        overrides: dict[str, str] = {
                            "MYEVS_EBF_S33_BETA": str(float(b)),
                        }
                        suffix = f"_b{_fmt_tag_float(float(b))}"
                        if tr_us is not None and float(tr_us) > 0.0:
                            overrides["MYEVS_EBF_S33_TAU_RATE_US"] = str(int(float(tr_us)))
                            suffix = f"_tr{_fmt_tag_float(float(tr_us))}" + suffix
                        sweep.append((suffix, overrides))

            elif variant == "s34":
                # s28 surprise z-score + pixel self-rate max correction. Default tau_rate_us=0 (auto), k_self=0.25.
                tr_list = (
                    _parse_float_list(args.s34_tau_rate_us_list)
                    if str(args.s34_tau_rate_us_list).strip()
                    else [0.0]
                )
                k_list = (
                    _parse_float_list(args.s34_k_self_list)
                    if str(args.s34_k_self_list).strip()
                    else [0.25]
                )
                sweep = []
                for tr_us in tr_list:
                    for k in k_list:
                        overrides: dict[str, str] = {
                            "MYEVS_EBF_S34_K_SELF": str(float(k)),
                        }
                        suffix = f"_k{_fmt_tag_float(float(k))}"
                        if tr_us is not None and float(tr_us) > 0.0:
                            overrides["MYEVS_EBF_S34_TAU_RATE_US"] = str(int(float(tr_us)))
                            suffix = f"_tr{_fmt_tag_float(float(tr_us))}" + suffix
                        sweep.append((suffix, overrides))

            for s in s_list:
                r = (s - 1) // 2
                for tau_us in tau_us_list:
                    for suffix, overrides in sweep:
                        tag = f"{base_tag}{suffix}_labelscore_s{s}_tau{tau_us}"
                        old = _env_override(overrides) if overrides else {}
                        try:
                            scores = score_stream_ebf(
                                ev,
                                width=int(args.width),
                                height=int(args.height),
                                radius_px=int(r),
                                tau_us=int(tau_us),
                                tb=tb,
                                _kernel_cache=kernel_cache,
                                variant=variant,
                            )
                        finally:
                            if overrides:
                                _env_restore(old, overrides)

                        auc, thr, tp, fp, _fpr, _tpr = _roc_points_from_scores(
                            ev.label,
                            scores,
                            max_points=int(args.roc_max_points),
                        )

                        # Best-F1 operating point (used for best-F1 selection and ESR sampling).
                        best_i = _best_f1_index(thr, tp, fp, pos=pos, neg=neg)
                        best_thr = float(thr[int(best_i)])
                        best_f1_val = _f1_at_index(thr, tp, fp, pos=pos, neg=neg, i=int(best_i))

                        if auc > best_global[1]:
                            best_global = (tag, float(auc))
                        if auc > best_by_env[env][1]:
                            best_by_env[env] = (tag, float(auc))
                        if best_f1_val > best_f1_by_env[env][1]:
                            best_f1_by_env[env] = (tag, float(best_f1_val))

                        if bool(need_best_recipes):
                            if best_auc_recipe is None or float(auc) > float(best_auc_recipe["auc"]):
                                best_auc_recipe = {
                                    "auc": float(auc),
                                    "tag": str(tag),
                                    "best_i": int(best_i),
                                    "best_thr": float(best_thr),
                                    "s": int(s),
                                    "r": int(r),
                                    "tau_us": int(tau_us),
                                    "overrides": dict(overrides),
                                }
                            if best_f1_recipe is None or float(best_f1_val) > float(best_f1_recipe["f1"]):
                                best_f1_recipe = {
                                    "f1": float(best_f1_val),
                                    "tag": str(tag),
                                    "best_i": int(best_i),
                                    "best_thr": float(best_thr),
                                    "s": int(s),
                                    "r": int(r),
                                    "tau_us": int(tau_us),
                                    "overrides": dict(overrides),
                                }

                        esr_mean: float | None = None
                        if esr_mode == "all":
                            kept = scores >= best_thr
                            esr_mean = float(
                                event_structural_ratio_mean_from_xy(
                                    ev.x[kept],
                                    ev.y[kept],
                                    width=int(args.width),
                                    height=int(args.height),
                                    chunk_size=30000,
                                )
                            )

                        aocc: float | None = None
                        if aocc_mode == "all":
                            kept = scores >= best_thr
                            aocc = float(
                                aocc_from_xyt(
                                    ev.x[kept],
                                    ev.y[kept],
                                    ev.t[kept],
                                    width=int(args.width),
                                    height=int(args.height),
                                )
                            )

                        _write_roc_rows(
                            w,
                            tag=tag,
                            method=base_tag,
                            param="min-neighbors",
                            thresholds=thr,
                            tp=tp,
                            fp=fp,
                            pos=pos,
                            neg=neg,
                            auc=float(auc),
                            esr_mean=(None if esr_mean is None else float(esr_mean)),
                            esr_at_index=int(best_i),
                            aocc=(None if aocc is None else float(aocc)),
                            aocc_at_index=int(best_i),
                        )

                        print(
                            f"auc={auc:.6f} env={env} s={s} tau_us={tau_us} points={int(thr.shape[0])} tag={tag}"
                        )

        # For {esr,aocc}-mode=best: compute metrics only for the best-AUC and best-F1 tags, then patch CSV.
        if bool(need_best_recipes) and (best_auc_recipe is not None or best_f1_recipe is not None):
            recipes: list[dict[str, object]] = []
            if best_auc_recipe is not None:
                recipes.append(best_auc_recipe)
            if best_f1_recipe is not None and (
                best_auc_recipe is None or str(best_f1_recipe.get("tag")) != str(best_auc_recipe.get("tag"))
            ):
                recipes.append(best_f1_recipe)

            # Cache scores by tag to avoid duplicate scoring when both ESR and AOCC are enabled.
            scores_cache: dict[str, np.ndarray] = {}

            def _scores_and_kept_for_recipe(recipe: dict[str, object]) -> tuple[str, int, np.ndarray]:
                tag = str(recipe["tag"])
                best_i = int(recipe["best_i"])
                best_thr = float(recipe["best_thr"])

                if tag in scores_cache:
                    scores = scores_cache[tag]
                else:
                    overrides = recipe.get("overrides")
                    ov: dict[str, str] = dict(overrides) if isinstance(overrides, dict) else {}
                    old = _env_override(ov) if ov else {}
                    try:
                        scores = score_stream_ebf(
                            ev,
                            width=int(args.width),
                            height=int(args.height),
                            radius_px=int(recipe["r"]),
                            tau_us=int(recipe["tau_us"]),
                            tb=tb,
                            _kernel_cache=kernel_cache,
                            variant=variant,
                        )
                    finally:
                        if ov:
                            _env_restore(old, ov)
                    scores_cache[tag] = scores

                kept = scores >= best_thr
                return tag, best_i, kept

            if esr_mode == "best":
                esr_targets: dict[str, tuple[int, float]] = {}
                for recipe in recipes:
                    tag, best_i, kept = _scores_and_kept_for_recipe(recipe)
                    if tag in esr_targets:
                        continue
                    esr_targets[tag] = (
                        int(best_i),
                        float(
                            event_structural_ratio_mean_from_xy(
                                ev.x[kept],
                                ev.y[kept],
                                width=int(args.width),
                                height=int(args.height),
                                chunk_size=30000,
                            )
                        ),
                    )
                _patch_esr_mean_in_roc_csv(roc_csv[env], esr_targets=esr_targets)

            if aocc_mode == "best":
                aocc_targets: dict[str, tuple[int, float]] = {}
                for recipe in recipes:
                    tag, best_i, kept = _scores_and_kept_for_recipe(recipe)
                    if tag in aocc_targets:
                        continue
                    aocc_targets[tag] = (
                        int(best_i),
                        float(
                            aocc_from_xyt(
                                ev.x[kept],
                                ev.y[kept],
                                ev.t[kept],
                                width=int(args.width),
                                height=int(args.height),
                            )
                        ),
                    )
                _patch_aocc_in_roc_csv(roc_csv[env], aocc_targets=aocc_targets)
        # plot per-env png
        _plot_roc_png(
            csv_path=roc_csv[env],
            png_path=roc_png[env],
            title=(
                (
                    f"EBF ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us"
                    if variant == "ebf"
                    else (
                        f"EBF V10 (spatialw_linear) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us"
                        if variant == "ebfv10"
                        else (
                            f"EBF Part2 s1 (directional coherence) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us"
                            if variant == "s1"
                            else (
                                f"EBF Part2 s2 (coherence-gated penalty) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us"
                                if variant == "s2"
                                else (
                                    f"EBF Part2 s3 (smooth coh gate) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us"
                                    if variant == "s3"
                                    else (
                                        f"EBF Part2 s4 (resultant gate) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us"
                                        if variant == "s4"
                                        else (
                                            f"EBF Part2 s5 (elliptic spatialw) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us"
                                            if variant == "s5"
                                            else (
                                                f"EBF Part2 s6 (time-coh gate) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us"
                                                if variant == "s6"
                                                else f"EBF Part2 s7 (plane residual gate) ROC ({env}) labelscore: s in {s_list}, tau in {tau_us_list}us"
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            ),
        )


    print("=== BEST (by env) ===")
    for env in ("light", "mid", "heavy"):
        tag, auc = best_by_env[env]
        if auc >= 0:
            print(f"{env}: {tag} auc={auc:.6f}")
    print("=== BEST F1 (by env) ===")
    for env in ("light", "mid", "heavy"):
        tag, f1 = best_f1_by_env[env]
        if f1 >= 0:
            print(f"{env}: {tag} f1={f1:.6f}")

    print("=== BEST (global) ===")
    tag, auc = best_global
    if auc >= 0:
        print(f"global: {tag} auc={auc:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
