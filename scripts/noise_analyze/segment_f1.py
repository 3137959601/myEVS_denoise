from __future__ import annotations

import argparse
import csv
import importlib.util
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class LabeledEvents:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    p: np.ndarray
    label: np.ndarray


@dataclass(frozen=True)
class RocBestPoint:
    tag: str
    thr: float
    f1: float
    auc: float | None


def _load_sweep_module():
    here = Path(__file__).resolve()
    sweep_path = here.parents[1] / "ED24_alg_evalu" / "sweep_ebf_slim_labelscore_grid.py"
    spec = importlib.util.spec_from_file_location("_sweep_ebf_labelscore_grid", sweep_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"failed to load sweep module spec: {sweep_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_labeled_npy(path: str, *, max_events: int = 0) -> LabeledEvents:
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
            label = a2[:, 0].astype(np.int8, copy=False)
            t = a2[:, 1].astype(np.uint64, copy=False)
            y = a2[:, 2].astype(np.int32, copy=False)
            x = a2[:, 3].astype(np.int32, copy=False)
            p = a2[:, 4].astype(np.int8, copy=False)
        else:
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


def _read_best_point_from_roc(
    roc_csv: str,
    *,
    s: int,
    tau_us: int,
    tag_contains: str | None,
    tag: str | None = None,
) -> RocBestPoint:
    p = Path(roc_csv)
    if not p.exists():
        raise SystemExit(f"roc csv not found: {roc_csv}")

    suffix = f"_labelscore_s{s}_tau{tau_us}"

    best_by_tag: dict[str, RocBestPoint] = {}
    with p.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            t = (row.get("tag") or "").strip()
            if not t or suffix not in t:
                continue
            if tag is not None and t != tag:
                continue
            if tag is None and tag_contains is not None and tag_contains not in t:
                continue

            try:
                f1 = float(row["f1"])
                thr = float(row["value"])
            except Exception:
                continue

            a = (row.get("auc") or "").strip()
            auc: float | None
            if a:
                try:
                    auc = float(a)
                except Exception:
                    auc = None
            else:
                auc = None

            cur = best_by_tag.get(t)
            if cur is None or f1 > cur.f1:
                best_by_tag[t] = RocBestPoint(tag=t, thr=thr, f1=f1, auc=auc)

    if not best_by_tag:
        raise SystemExit(
            f"no matching rows found in ROC CSV for suffix={suffix!r} and tag_contains={tag_contains!r} tag={tag!r}\nfile={roc_csv}"
        )

    if tag is not None:
        return best_by_tag[tag]

    return max(best_by_tag.values(), key=lambda x: x.f1)


def _variant_tag_contains(variant: str) -> str | None:
    v = str(variant).strip().lower()
    if v in {"ebf", "v0", "baseline"}:
        return None
    if v.startswith("ebf_"):
        v2 = v
    else:
        v2 = f"ebf_{v}"
    if v2.startswith("ebf_s"):
        # common form: ebf_s52_
        head = v2.split("_", 2)[0]  # ebf
        # but safer: take ebf_sXX
        parts = v2.split("_")
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}_"
    if v2.startswith("ebf_n"):
        parts = v2.split("_")
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}_"
    return None


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if float(b) != 0.0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute per-segment precision/recall/F1 for a chosen (s,tau) and threshold.")
    ap.add_argument("--labeled-npy", required=True)
    ap.add_argument("--out-csv", required=True)

    ap.add_argument("--width", type=int, default=346)
    ap.add_argument("--height", type=int, default=260)
    ap.add_argument("--tick-ns", type=float, default=1000.0)

    ap.add_argument("--variant", default="ebf")
    ap.add_argument("--s", type=int, default=9)
    ap.add_argument("--tau-us", type=int, default=128000)
    ap.add_argument("--max-events", type=int, default=1000000)

    ap.add_argument("--roc-csv", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--thr", type=float, default=float("nan"))

    ap.add_argument("--segment-events", type=int, default=200000)

    args = ap.parse_args()

    labeled = load_labeled_npy(str(args.labeled_npy), max_events=int(args.max_events))
    n = int(labeled.t.shape[0])
    if n <= 0:
        raise SystemExit("empty labeled input")

    thr = None if not np.isfinite(float(args.thr)) else float(args.thr)
    chosen_tag: str | None = (str(args.tag).strip() or None)

    if thr is None:
        roc_csv = str(args.roc_csv).strip()
        if not roc_csv:
            raise SystemExit("need --thr or --roc-csv")
        tag_contains = _variant_tag_contains(str(args.variant))
        best = _read_best_point_from_roc(
            roc_csv,
            s=int(args.s),
            tau_us=int(args.tau_us),
            tag_contains=tag_contains,
            tag=chosen_tag,
        )
        thr = float(best.thr)
        chosen_tag = best.tag

    # Score stream using the exact sweep implementation.
    sweep = _load_sweep_module()
    tb = sweep.TimeBase(tick_ns=float(args.tick_ns))
    kernel_cache: dict[str, object] = {}

    ev = sweep.LabeledEvents(t=labeled.t, x=labeled.x, y=labeled.y, p=labeled.p, label=labeled.label)
    scores = sweep.score_stream_ebf(
        ev,
        width=int(args.width),
        height=int(args.height),
        radius_px=int((int(args.s) - 1) // 2),
        tau_us=int(args.tau_us),
        tb=tb,
        _kernel_cache=kernel_cache,
        variant=str(args.variant),
    )

    kept = scores >= float(thr)
    lab = labeled.label.astype(bool, copy=False)

    seg = int(args.segment_events)
    if seg <= 0:
        seg = n

    out_path = Path(str(args.out_csv))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "variant",
                "tag",
                "thr",
                "s",
                "tau_us",
                "segment_idx",
                "i0",
                "i1",
                "t0_us",
                "t1_us",
                "n",
                "pos",
                "neg",
                "tp",
                "fp",
                "fn",
                "precision",
                "recall",
                "f1",
                "signal_kept_rate",
                "noise_kept_rate",
            ]
        )

        seg_idx = 0
        for i0 in range(0, n, seg):
            i1 = min(n, i0 + seg)
            k = kept[i0:i1]
            y = lab[i0:i1]
            tp = int(np.sum(k & y))
            fp = int(np.sum(k & (~y)))
            fn = int(np.sum((~k) & y))
            pos = int(np.sum(y))
            neg = int((i1 - i0) - pos)

            precision = _safe_div(tp, tp + fp)
            recall = _safe_div(tp, tp + fn)
            f1 = _safe_div(2 * tp, 2 * tp + fp + fn)

            w.writerow(
                [
                    str(args.variant),
                    str(chosen_tag or ""),
                    float(thr),
                    int(args.s),
                    int(args.tau_us),
                    seg_idx,
                    i0,
                    i1,
                    int(labeled.t[i0]),
                    int(labeled.t[i1 - 1]),
                    int(i1 - i0),
                    pos,
                    neg,
                    tp,
                    fp,
                    fn,
                    precision,
                    recall,
                    f1,
                    _safe_div(tp, pos),
                    _safe_div(fp, neg),
                ]
            )
            seg_idx += 1

    print(f"wrote: {out_path} (n={n}, seg={seg})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
