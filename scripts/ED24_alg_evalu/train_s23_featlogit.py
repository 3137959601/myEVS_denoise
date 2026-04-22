from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np

from myevs.timebase import TimeBase


@dataclass(frozen=True)
class LabeledEvents:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    p: np.ndarray
    label: np.ndarray


def load_labeled_npy(path: str, *, max_events: int = 0) -> LabeledEvents:
    # Keep consistent with scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py
    arr = np.load(path, mmap_mode="r", allow_pickle=True)
    if max_events > 0:
        arr = arr[:max_events]

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


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def _auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUC via rank statistics (no sklearn)."""
    y = y_true.astype(np.int8, copy=False)
    s = y_score.astype(np.float64, copy=False)

    n_pos = int(np.sum(y))
    n = int(y.shape[0])
    n_neg = n - n_pos
    if n_pos <= 0 or n_neg <= 0:
        return float("nan")

    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)

    # handle ties: average ranks for equal scores
    sorted_s = s[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_s[j] == sorted_s[i]:
            j += 1
        if j - i > 1:
            avg = 0.5 * (ranks[order[i]] + ranks[order[j - 1]])
            for k in range(i, j):
                ranks[order[k]] = avg
        i = j

    sum_r_pos = float(np.sum(ranks[y.astype(bool)]))
    auc = (sum_r_pos - n_pos * (n_pos + 1) / 2.0) / (float(n_pos) * float(n_neg))
    return float(auc)


def _best_f1(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Return (best_f1, thr) where predict=score>=thr."""
    y = y_true.astype(np.int8, copy=False)
    s = y_score.astype(np.float64, copy=False)
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    s_sorted = s[order]

    tp = 0
    fp = 0
    pos = int(np.sum(y))
    best_f1 = -1.0
    best_thr = float("inf")
    for i in range(int(y.shape[0])):
        if int(y_sorted[i]) == 1:
            tp += 1
        else:
            fp += 1
        fn = pos - tp
        denom = 2 * tp + fp + fn
        if denom <= 0:
            continue
        f1 = (2.0 * tp) / float(denom)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_thr = float(s_sorted[i])
    return float(best_f1), float(best_thr)


def _standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.mean(x, axis=0)
    sig = np.std(x, axis=0)
    sig = np.where(sig <= 1e-6, 1.0, sig)
    return mu.astype(np.float64), sig.astype(np.float64)


def _standardize_apply(x: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    return (x - mu) / sig


def _parse_csv_names(s: str | None) -> set[str]:
    if s is None:
        return set()
    s = str(s).strip()
    if not s:
        return set()
    return {p.strip() for p in s.split(",") if p.strip()}


def _feature_indices(names: list[str], want: set[str]) -> list[int]:
    idx: list[int] = []
    for n in sorted(want):
        if n not in names:
            raise SystemExit(f"unknown feature name '{n}'. Valid: {names}")
        idx.append(int(names.index(n)))
    return idx


def _train_logreg_sgd(
    x: np.ndarray,
    y: np.ndarray,
    *,
    lr: float = 0.05,
    epochs: int = 20,
    l2: float = 1e-3,
    batch: int = 8192,
    seed: int = 0,
    fixed_offset: np.ndarray | None = None,
    clip_abs: float = float("nan"),
    nonpos_idx: np.ndarray | None = None,
    nonneg_idx: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Train logistic regression with SGD. Returns (w, bias)."""
    rng = np.random.default_rng(int(seed))
    n, d = x.shape
    w = np.zeros((d,), dtype=np.float64)
    b = 0.0
    idx = np.arange(n)

    # class balance weight
    pos = float(np.sum(y))
    neg = float(n - pos)
    w_pos = 0.5 / max(pos, 1.0)
    w_neg = 0.5 / max(neg, 1.0)

    for _ in range(int(epochs)):
        rng.shuffle(idx)
        for i0 in range(0, n, int(batch)):
            ii = idx[i0 : i0 + int(batch)]
            xb = x[ii]
            yb = y[ii].astype(np.float64)

            z = xb @ w + b
            if fixed_offset is not None:
                z = z + fixed_offset[ii]
            p = _sigmoid(z)

            # weighted gradient
            sample_w = np.where(yb > 0.0, w_pos, w_neg)
            diff = (p - yb) * sample_w

            gw = (xb.T @ diff) + l2 * w
            gb = float(np.sum(diff))

            w -= lr * gw
            b -= lr * gb

            if np.isfinite(clip_abs) and float(clip_abs) > 0.0:
                np.clip(w, -float(clip_abs), float(clip_abs), out=w)
            if nonpos_idx is not None and nonpos_idx.size > 0:
                w[nonpos_idx] = np.minimum(w[nonpos_idx], 0.0)
            if nonneg_idx is not None and nonneg_idx.size > 0:
                w[nonneg_idx] = np.maximum(w[nonneg_idx], 0.0)

    return w.astype(np.float64), float(b)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train Part2 s23 (feature+logit fusion) weights on labeled ED24 npy.")
    ap.add_argument("--light", default=r"D:\hjx_workspace\scientific_reserach\dataset\ED24\myPedestrain_06\Pedestrain_06_1.8.npy")
    ap.add_argument("--mid", default=r"D:\hjx_workspace\scientific_reserach\dataset\ED24\myPedestrain_06\Pedestrain_06_2.5.npy")
    ap.add_argument("--heavy", default=r"D:\hjx_workspace\scientific_reserach\dataset\ED24\myPedestrain_06\Pedestrain_06_3.3.npy")
    ap.add_argument("--width", type=int, default=346)
    ap.add_argument("--height", type=int, default=260)
    ap.add_argument("--tick-ns", type=float, default=1000.0)
    ap.add_argument("--s", type=int, default=9, help="diameter (odd), matches sweep defaults")
    ap.add_argument("--tau-us", type=int, default=128000)
    ap.add_argument("--dt-thr-us", type=float, default=4096.0)
    ap.add_argument(
        "--use-selfacc",
        action="store_true",
        help=(
            "if set, include an extra minimal-state feature selfacc "
            "(requires 2 bytes/pixel state during feature streaming)"
        ),
    )
    ap.add_argument(
        "--hotmask-npy",
        type=str,
        default="",
        help="optional: hotpixel mask .npy as (H,W) or (H*W,) array; nonzero treated as hotpixel",
    )
    ap.add_argument("--max-events", type=int, default=50000, help="per env (for training speed)")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument(
        "--clip-abs",
        type=float,
        default=float("nan"),
        help="optional: clip all LEARNED weights to [-clip_abs, +clip_abs] after each SGD step",
    )
    ap.add_argument(
        "--nonpos",
        type=str,
        default="",
        help="optional: comma-separated feature names whose LEARNED weights are constrained <=0 (e.g. 'toggle,dtsmall')",
    )
    ap.add_argument(
        "--nonneg",
        type=str,
        default="",
        help="optional: comma-separated feature names whose LEARNED weights are constrained >=0",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--train-envs",
        type=str,
        default="light,mid,heavy",
        help="comma-separated envs to include in training/eval: light,mid,heavy",
    )
    ap.add_argument(
        "--fix-w-same",
        type=float,
        default=float("nan"),
        help="if set (finite), keep w_same fixed to this value and only learn other weights",
    )

    args = ap.parse_args(argv)

    from myevs.denoise.ops.ebfopt_part2.s23_featlogit import try_build_s23_featlogit_features_kernel

    hotmask_u8: np.ndarray | None = None
    hotmask_path = (str(getattr(args, "hotmask_npy", "")) or "").strip()
    if hotmask_path:
        if not os.path.exists(hotmask_path):
            raise SystemExit(f"--hotmask-npy not found: {hotmask_path!r}")
        m = np.load(hotmask_path, allow_pickle=False)
        m = np.asarray(m)
        if m.ndim == 2:
            if int(m.shape[0]) != int(args.height) or int(m.shape[1]) != int(args.width):
                raise SystemExit(
                    f"hotmask shape mismatch: got {tuple(m.shape)}, expected (height,width)=({int(args.height)},{int(args.width)})"
                )
            m = m.reshape((-1,))
        elif m.ndim == 1:
            if int(m.shape[0]) != int(args.width) * int(args.height):
                raise SystemExit(
                    f"hotmask length mismatch: got {int(m.shape[0])}, expected {int(args.width)*int(args.height)}"
                )
        else:
            raise SystemExit(f"hotmask must be 1D (W*H) or 2D (H,W); got shape {tuple(m.shape)}")
        hotmask_u8 = np.ascontiguousarray((m != 0).astype(np.uint8, copy=False))

    use_selfacc = bool(getattr(args, "use_selfacc", False))
    use_hotmask = hotmask_u8 is not None

    ker = try_build_s23_featlogit_features_kernel(with_selfacc=use_selfacc, with_hotmask=use_hotmask)
    if ker is None:
        raise SystemExit(
            "s23 training requires numba. Fix: set PYTHONNOUSERSITE=1 (PowerShell: $env:PYTHONNOUSERSITE='1') and rerun."
        )

    tb = TimeBase(tick_ns=float(args.tick_ns))
    radius_px = int((int(args.s) - 1) // 2)
    tau_ticks = int(tb.us_to_ticks(int(args.tau_us)))
    dt_thr_ticks = int(tb.us_to_ticks(int(float(args.dt_thr_us))))

    env_paths = {"light": str(args.light), "mid": str(args.mid), "heavy": str(args.heavy)}
    train_envs = _parse_csv_names(getattr(args, "train_envs", "light,mid,heavy"))
    if not train_envs:
        raise SystemExit("--train-envs is empty")
    bad = sorted(train_envs - set(env_paths.keys()))
    if bad:
        raise SystemExit(f"unknown env(s) in --train-envs: {bad}. valid: {sorted(env_paths.keys())}")

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    env_slices: dict[str, slice] = {}
    n0 = 0

    for env, path in env_paths.items():
        if env not in train_envs:
            continue
        ev = load_labeled_npy(path, max_events=int(args.max_events))
        n = int(ev.label.shape[0])
        pos = int(np.sum(ev.label))
        print(f"loaded {env}: n={n} pos={pos} neg={n-pos} path={path}")

        raw_same = np.empty((n,), dtype=np.float32)
        raw_opp = np.empty((n,), dtype=np.float32)
        oppr = np.empty((n,), dtype=np.float32)
        toggle = np.empty((n,), dtype=np.float32)
        dtsmall = np.empty((n,), dtype=np.float32)
        sameburst = np.empty((n,), dtype=np.float32)

        selfacc = None
        if use_selfacc:
            selfacc = np.empty((n,), dtype=np.float32)

        ishot = None
        hotnbr = None
        if use_hotmask:
            ishot = np.empty((n,), dtype=np.float32)
            hotnbr = np.empty((n,), dtype=np.float32)

        last_ts = np.zeros((int(args.width) * int(args.height),), dtype=np.uint64)
        last_pol = np.zeros((int(args.width) * int(args.height),), dtype=np.int8)

        if use_selfacc:
            self_acc_q8 = np.zeros((int(args.width) * int(args.height),), dtype=np.uint16)
            if use_hotmask:
                assert hotmask_u8 is not None
                assert hotnbr is not None
                ker(
                    ev.t,
                    ev.x,
                    ev.y,
                    ev.p,
                    int(args.width),
                    int(args.height),
                    int(radius_px),
                    int(tau_ticks),
                    int(dt_thr_ticks),
                    last_ts,
                    last_pol,
                    self_acc_q8,
                    hotmask_u8,
                    raw_same,
                    raw_opp,
                    oppr,
                    toggle,
                    dtsmall,
                    sameburst,
                    selfacc,
                    hotnbr,
                )
            else:
                ker(
                    ev.t,
                    ev.x,
                    ev.y,
                    ev.p,
                    int(args.width),
                    int(args.height),
                    int(radius_px),
                    int(tau_ticks),
                    int(dt_thr_ticks),
                    last_ts,
                    last_pol,
                    self_acc_q8,
                    raw_same,
                    raw_opp,
                    oppr,
                    toggle,
                    dtsmall,
                    sameburst,
                    selfacc,
                )
            x_env0 = np.stack([raw_same, raw_opp, oppr, toggle, dtsmall, sameburst, selfacc], axis=1)
        else:
            if use_hotmask:
                assert hotmask_u8 is not None
                assert hotnbr is not None
                ker(
                    ev.t,
                    ev.x,
                    ev.y,
                    ev.p,
                    int(args.width),
                    int(args.height),
                    int(radius_px),
                    int(tau_ticks),
                    int(dt_thr_ticks),
                    last_ts,
                    last_pol,
                    hotmask_u8,
                    raw_same,
                    raw_opp,
                    oppr,
                    toggle,
                    dtsmall,
                    sameburst,
                    hotnbr,
                )
            else:
                ker(
                    ev.t,
                    ev.x,
                    ev.y,
                    ev.p,
                    int(args.width),
                    int(args.height),
                    int(radius_px),
                    int(tau_ticks),
                    int(dt_thr_ticks),
                    last_ts,
                    last_pol,
                    raw_same,
                    raw_opp,
                    oppr,
                    toggle,
                    dtsmall,
                    sameburst,
                )
            x_env0 = np.stack([raw_same, raw_opp, oppr, toggle, dtsmall, sameburst], axis=1)

        if use_hotmask:
            assert hotmask_u8 is not None
            assert ishot is not None
            assert hotnbr is not None
            ishot[:] = 0.0
            xv = ev.x.astype(np.int64, copy=False)
            yv = ev.y.astype(np.int64, copy=False)
            valid = (xv >= 0) & (xv < int(args.width)) & (yv >= 0) & (yv < int(args.height))
            pix = yv[valid] * int(args.width) + xv[valid]
            ishot[valid] = hotmask_u8[pix].astype(np.float32, copy=False)
            x_env0 = np.concatenate([x_env0, ishot.reshape((-1, 1)), hotnbr.reshape((-1, 1))], axis=1)

        x_env = x_env0.astype(np.float64, copy=False)
        y_env = ev.label.astype(np.int8, copy=False)
        xs.append(x_env)
        ys.append(y_env)
        env_slices[env] = slice(n0, n0 + n)
        n0 += n

    x_all = np.concatenate(xs, axis=0)
    y_all = np.concatenate(ys, axis=0)

    fix_w_same = float(getattr(args, "fix_w_same", float("nan")))
    use_fixed_same = bool(np.isfinite(fix_w_same))

    feat_full = ["raw_same", "raw_opp", "oppr", "toggle", "dtsmall", "sameburst"]
    if use_selfacc:
        feat_full.append("selfacc")
    if use_hotmask:
        feat_full.append("ishot")
        feat_full.append("hotnbr")

    feat_train = feat_full[1:] if use_fixed_same else feat_full

    nonpos = _parse_csv_names(getattr(args, "nonpos", ""))
    nonneg = _parse_csv_names(getattr(args, "nonneg", ""))
    if nonpos & nonneg:
        both = sorted(nonpos & nonneg)
        raise SystemExit(f"features cannot be both nonpos and nonneg: {both}")

    nonpos_idx = np.asarray(_feature_indices(feat_train, nonpos), dtype=np.int64) if nonpos else None
    nonneg_idx = np.asarray(_feature_indices(feat_train, nonneg), dtype=np.int64) if nonneg else None

    if np.isfinite(float(args.clip_abs)) or nonpos or nonneg:
        print(
            "constraints:",
            f"clip_abs={getattr(args, 'clip_abs', float('nan'))}",
            f"nonpos={sorted(nonpos)}",
            f"nonneg={sorted(nonneg)}",
            f"train_features={feat_train}",
        )

    rng = np.random.default_rng(int(args.seed))
    n = int(y_all.shape[0])
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(float(args.train_frac) * n)
    tr = idx[:n_train]
    va = idx[n_train:]

    mu, sig = _standardize_fit(x_all[tr])
    xtr_full = _standardize_apply(x_all[tr], mu, sig)
    xva_full = _standardize_apply(x_all[va], mu, sig)

    fixed_offset_tr = None
    fixed_offset_va = None
    if use_fixed_same:
        fixed_offset_tr = xtr_full[:, 0] * fix_w_same
        fixed_offset_va = xva_full[:, 0] * fix_w_same
        xtr = xtr_full[:, 1:]
        xva = xva_full[:, 1:]
    else:
        xtr = xtr_full
        xva = xva_full

    w, b = _train_logreg_sgd(
        xtr,
        y_all[tr],
        lr=float(args.lr),
        epochs=int(args.epochs),
        l2=float(args.l2),
        seed=int(args.seed),
        fixed_offset=fixed_offset_tr,
        clip_abs=float(getattr(args, "clip_abs", float("nan"))),
        nonpos_idx=nonpos_idx,
        nonneg_idx=nonneg_idx,
    )

    zva = xva @ w + b
    if fixed_offset_va is not None:
        zva = zva + fixed_offset_va
    auc_all = _auc_score(y_all[va], zva)
    f1_all, thr_all = _best_f1(y_all[va], zva)
    print(f"val(all): auc={auc_all:.6f} f1={f1_all:.6f} thr={thr_all:.6f}")

    for env, sl in env_slices.items():
        mask = (va >= sl.start) & (va < sl.stop)
        ii = va[mask]
        if ii.size < 10:
            continue
        xenv_full = _standardize_apply(x_all[ii], mu, sig)
        if use_fixed_same:
            z = (xenv_full[:, 1:] @ w + b) + (xenv_full[:, 0] * fix_w_same)
        else:
            z = xenv_full @ w + b
        auc = _auc_score(y_all[ii], z)
        f1, thr = _best_f1(y_all[ii], z)
        print(f"val({env}): auc={auc:.6f} f1={f1:.6f} thr={thr:.6f} n={int(ii.size)}")

    print("\n# Recommended to reproduce in sweep (s23 uses RAW features; apply standardization inside weights):")
    print(f"# feature order: {feat_full}")
    print(f"# standardize: (x - mu) / sig")

    if use_fixed_same:
        w_full = np.empty((len(feat_full),), dtype=np.float64)
        w_full[0] = fix_w_same
        w_full[1:] = w
        w_std = w_full
    else:
        w_std = w

    w_raw = w_std / sig
    b_raw = b - float(np.sum((w_std * mu) / sig))

    print(f"$env:MYEVS_EBF_S23_DT_THR_US='{float(args.dt_thr_us)}'")
    print(f"$env:MYEVS_EBF_S23_BIAS='{b_raw}'")
    print(f"$env:MYEVS_EBF_S23_W_SAME='{w_raw[0]}'")
    print(f"$env:MYEVS_EBF_S23_W_OPP='{w_raw[1]}'")
    print(f"$env:MYEVS_EBF_S23_W_OPPR='{w_raw[2]}'")
    print(f"$env:MYEVS_EBF_S23_W_TOGGLE='{w_raw[3]}'")
    print(f"$env:MYEVS_EBF_S23_W_DTSMALL='{w_raw[4]}'")
    print(f"$env:MYEVS_EBF_S23_W_SAMEBURST='{w_raw[5]}'")

    idxw = 6
    if use_selfacc:
        print(f"$env:MYEVS_EBF_S23_W_SELFACC='{w_raw[idxw]}'")
        idxw += 1
    else:
        print(f"$env:MYEVS_EBF_S23_W_SELFACC='0.0'")

    if use_hotmask:
        print(f"$env:MYEVS_EBF_S23_W_HOT='{w_raw[idxw]}'")
        idxw += 1
        print(f"$env:MYEVS_EBF_S23_W_HOTNBR='{w_raw[idxw]}'")
        idxw += 1
    else:
        print(f"$env:MYEVS_EBF_S23_W_HOT='0.0'")
        print(f"$env:MYEVS_EBF_S23_W_HOTNBR='0.0'")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
