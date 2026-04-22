from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass

import numpy as np

try:
    import numba
except Exception:  # pragma: no cover
    numba = None


@dataclass(frozen=True)
class LabeledEvents:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    p: np.ndarray
    label: np.ndarray


def _require_numba() -> None:
    if numba is None:
        raise SystemExit("n92_continuity_stats requires numba")


def _parse_int_list(s: str) -> list[int]:
    out: list[int] = []
    for it in str(s).split(","):
        it = it.strip()
        if not it:
            continue
        out.append(int(float(it)))
    if not out:
        raise SystemExit("empty integer list")
    return out


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


@numba.njit(cache=True)
def _compute_chain_metrics(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    width: int,
    height: int,
    radius_px: int,
    tau_ticks: int,
    k_use: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(t.shape[0])
    npx = int(width) * int(height)
    kmax = 8

    # 7.45 alignment: keep only one latest timestamp per pixel/polarity.
    pos_ts = np.zeros((npx,), dtype=np.uint64)
    neg_ts = np.zeros((npx,), dtype=np.uint64)

    nsel = np.zeros((n,), dtype=np.int16)
    sum_dist = np.full((n,), np.nan, dtype=np.float32)
    mean_dist = np.full((n,), np.nan, dtype=np.float32)

    rr = int(radius_px)
    if rr < 0:
        rr = 0
    if rr > 8:
        rr = 8

    tau = int(tau_ticks)
    if tau <= 0:
        tau = 1

    k = int(k_use)
    if k < 1:
        k = 1
    if k > kmax:
        k = kmax

    best_ts = np.zeros((kmax,), dtype=np.uint64)
    best_x = np.zeros((kmax,), dtype=np.int32)
    best_y = np.zeros((kmax,), dtype=np.int32)

    for i in range(n):
        xi = int(x[i])
        yi = int(y[i])
        if xi < 0 or xi >= width or yi < 0 or yi >= height:
            nsel[i] = np.int16(0)
            continue

        ti = np.uint64(t[i])
        pi = 1 if int(p[i]) > 0 else -1

        for s in range(k):
            best_ts[s] = np.uint64(0)
            best_x[s] = 0
            best_y[s] = 0
        ncur = 0

        x0 = xi - rr
        if x0 < 0:
            x0 = 0
        x1 = xi + rr
        if x1 >= width:
            x1 = width - 1
        y0 = yi - rr
        if y0 < 0:
            y0 = 0
        y1 = yi + rr
        if y1 >= height:
            y1 = height - 1

        for yy in range(y0, y1 + 1):
            base = yy * width
            for xx in range(x0, x1 + 1):
                idx_nb = base + xx
                ts = np.uint64(0)
                if pi > 0:
                    ts = pos_ts[idx_nb]
                else:
                    ts = neg_ts[idx_nb]
                if ts == 0:
                    continue

                dt = int(ti - ts) if ti >= ts else int(ts - ti)
                if dt <= 0 or dt > tau:
                    continue

                if ncur < k:
                    best_ts[ncur] = ts
                    best_x[ncur] = xx
                    best_y[ncur] = yy
                    ncur += 1

                    m = ncur - 1
                    while m > 0 and best_ts[m] > best_ts[m - 1]:
                        ttmp = best_ts[m]
                        best_ts[m] = best_ts[m - 1]
                        best_ts[m - 1] = ttmp

                        xtmp = best_x[m]
                        best_x[m] = best_x[m - 1]
                        best_x[m - 1] = xtmp

                        ytmp = best_y[m]
                        best_y[m] = best_y[m - 1]
                        best_y[m - 1] = ytmp
                        m -= 1
                else:
                    if ts <= best_ts[k - 1]:
                        continue
                    best_ts[k - 1] = ts
                    best_x[k - 1] = xx
                    best_y[k - 1] = yy

                    m = k - 1
                    while m > 0 and best_ts[m] > best_ts[m - 1]:
                        ttmp = best_ts[m]
                        best_ts[m] = best_ts[m - 1]
                        best_ts[m - 1] = ttmp

                        xtmp = best_x[m]
                        best_x[m] = best_x[m - 1]
                        best_x[m - 1] = xtmp

                        ytmp = best_y[m]
                        best_y[m] = best_y[m - 1]
                        best_y[m - 1] = ytmp
                        m -= 1

        nsel[i] = np.int16(ncur)
        if ncur > 0:
            sd = 0.0
            for m in range(ncur - 1, 0, -1):
                dx = float(best_x[m - 1] - best_x[m])
                dy = float(best_y[m - 1] - best_y[m])
                sd += np.sqrt(dx * dx + dy * dy)

            dx_last = float(xi - best_x[0])
            dy_last = float(yi - best_y[0])
            sd += np.sqrt(dx_last * dx_last + dy_last * dy_last)

            sum_dist[i] = np.float32(sd)
            mean_dist[i] = np.float32(sd / float(ncur))

        idx0 = yi * width + xi
        if pi > 0:
            pos_ts[idx0] = ti
        else:
            neg_ts[idx0] = ti

    return nsel, sum_dist, mean_dist


def _quant(v: np.ndarray, q: float) -> float:
    if v.size == 0:
        return float("nan")
    return float(np.quantile(v, q))


def _write_summary_csv(out_csv: str, rows: list[dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    fields = [
        "env",
        "window",
        "tau_us",
        "k",
        "k_min",
        "class",
        "events",
        "valid_events",
        "valid_rate",
        "nsel_mean",
        "sum_dist_mean",
        "sum_dist_median",
        "sum_dist_p90",
        "sum_dist_p99",
        "mean_dist_mean",
        "mean_dist_median",
        "mean_dist_p90",
        "mean_dist_p99",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def continuity_stats(
    *,
    env_name: str,
    labeled_npy: str,
    out_csv: str,
    width: int,
    height: int,
    tick_ns: float,
    max_events: int,
    windows: list[int],
    tau_us_list: list[int],
    k_use: int,
    k_min: int,
) -> None:
    _require_numba()

    ev = load_labeled_npy(str(labeled_npy), max_events=int(max_events))
    rows: list[dict[str, object]] = []

    tick_us = float(tick_ns) / 1000.0

    for win in windows:
        r = int((int(win) - 1) // 2)
        for tau_us in tau_us_list:
            tau_ticks = int(round(float(tau_us) / max(tick_us, 1e-9)))
            nsel, sum_dist, mean_dist = _compute_chain_metrics(
                ev.t,
                ev.x,
                ev.y,
                ev.p,
                int(width),
                int(height),
                int(r),
                int(tau_ticks),
                int(k_use),
            )

            for cls_v, cls_name in ((0, "noise"), (1, "signal")):
                m = ev.label == cls_v
                total = int(np.sum(m))
                m_valid = np.logical_and(m, nsel >= int(k_min))
                valid = int(np.sum(m_valid))

                nsel_v = nsel[m_valid].astype(np.float64, copy=False)
                sd_v = sum_dist[m_valid].astype(np.float64, copy=False)
                md_v = mean_dist[m_valid].astype(np.float64, copy=False)

                rows.append(
                    {
                        "env": str(env_name),
                        "window": int(win),
                        "tau_us": int(tau_us),
                        "k": int(k_use),
                        "k_min": int(k_min),
                        "class": cls_name,
                        "events": total,
                        "valid_events": valid,
                        "valid_rate": (float(valid) / float(total)) if total > 0 else float("nan"),
                        "nsel_mean": float(np.mean(nsel_v)) if valid > 0 else float("nan"),
                        "sum_dist_mean": float(np.mean(sd_v)) if valid > 0 else float("nan"),
                        "sum_dist_median": float(np.median(sd_v)) if valid > 0 else float("nan"),
                        "sum_dist_p90": _quant(sd_v, 0.9),
                        "sum_dist_p99": _quant(sd_v, 0.99),
                        "mean_dist_mean": float(np.mean(md_v)) if valid > 0 else float("nan"),
                        "mean_dist_median": float(np.median(md_v)) if valid > 0 else float("nan"),
                        "mean_dist_p90": _quant(md_v, 0.9),
                        "mean_dist_p99": _quant(md_v, 0.99),
                    }
                )

    _write_summary_csv(str(out_csv), rows)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Statistics for N92 continuity rule: sort local same-polarity history by time, "
            "sum adjacent spatial distances, compare signal vs noise."
        )
    )
    ap.add_argument("--env-name", default="unknown")
    ap.add_argument("--labeled-npy", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--width", type=int, default=346)
    ap.add_argument("--height", type=int, default=260)
    ap.add_argument("--tick-ns", type=float, default=1000.0)
    ap.add_argument("--max-events", type=int, default=400000)
    ap.add_argument("--windows", default="7,9")
    ap.add_argument("--tau-us-list", default="32000,64000,128000")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--k-min", type=int, default=2)
    args = ap.parse_args()

    continuity_stats(
        env_name=str(args.env_name),
        labeled_npy=str(args.labeled_npy),
        out_csv=str(args.out_csv),
        width=int(args.width),
        height=int(args.height),
        tick_ns=float(args.tick_ns),
        max_events=int(args.max_events),
        windows=_parse_int_list(str(args.windows)),
        tau_us_list=_parse_int_list(str(args.tau_us_list)),
        k_use=int(args.k),
        k_min=int(args.k_min),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
