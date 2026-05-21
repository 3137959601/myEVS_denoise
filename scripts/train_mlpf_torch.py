from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from myevs.events import filter_visibility_batches, unwrap_tick_batches
from myevs.io.auto import open_events
from myevs.metrics.roc_auc import build_clean_index, signal_mask
from myevs.timebase import TimeBase


def collect_events(path: str, *, width: int, height: int, tick_ns: float, batch_events: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r = open_events(path, width=width, height=height, batch_events=batch_events, tick_ns=tick_ns, assume="npy")
    batches = unwrap_tick_batches(r.batches, bits=None)
    batches = filter_visibility_batches(batches, show_on=True, show_off=True)
    t_list: list[np.ndarray] = []
    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    p_list: list[np.ndarray] = []
    for b in batches:
        if len(b) == 0:
            continue
        t_list.append(np.asarray(b.t, dtype=np.uint64))
        x_list.append(np.asarray(b.x, dtype=np.int32))
        y_list.append(np.asarray(b.y, dtype=np.int32))
        p_list.append(np.asarray(b.p, dtype=np.int8))
    if not t_list:
        zt = np.empty((0,), dtype=np.uint64)
        zi = np.empty((0,), dtype=np.int32)
        zp = np.empty((0,), dtype=np.int8)
        return zt, zi, zi, zp
    return np.concatenate(t_list), np.concatenate(x_list), np.concatenate(y_list), np.concatenate(p_list)


def build_features(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    *,
    width: int,
    height: int,
    duration_ticks: int,
    patch: int,
) -> np.ndarray:
    r = patch // 2
    offsets: list[tuple[int, int]] = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            offsets.append((dx, dy))

    n = int(t.shape[0])
    feat_dim = 2 * patch * patch
    feats = np.zeros((n, feat_dim), dtype=np.float32)
    ts_map = np.zeros((height, width), dtype=np.uint64)

    inv_dur = 1.0 / float(max(1, duration_ticks))
    for i in range(n):
        xi = int(x[i])
        yi = int(y[i])
        ti = int(t[i])
        pol = 1.0 if int(p[i]) > 0 else -1.0
        k = 0
        for dx, dy in offsets:
            xx = xi + dx
            yy = yi + dy
            if 0 <= xx < width and 0 <= yy < height:
                prev = int(ts_map[yy, xx])
                recency = 1.0 - float(ti - prev) * inv_dur
                feats[i, k] = recency
                feats[i, k + patch * patch] = pol
            k += 1
        ts_map[yi, xi] = np.uint64(ti)
    return feats


class MLPFNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 20):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)


def binary_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    s = np.asarray(score, dtype=np.float64).reshape(-1)
    pos = y > 0.5
    n_pos = int(pos.sum())
    n_neg = int(y.shape[0] - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, y.shape[0] + 1, dtype=np.float64)
    # Average tied ranks.
    sorted_s = s[order]
    start = 0
    while start < sorted_s.shape[0]:
        end = start + 1
        while end < sorted_s.shape[0] and sorted_s[end] == sorted_s[start]:
            end += 1
        if end - start > 1:
            avg = 0.5 * (start + 1 + end)
            ranks[order[start:end]] = avg
        start = end
    rank_sum_pos = float(ranks[pos].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)


def best_f1(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y_true, dtype=np.float64).reshape(-1) > 0.5
    p = np.asarray(prob, dtype=np.float64).reshape(-1)
    best = 0.0
    best_thr = 0.5
    for thr in np.linspace(0.01, 0.99, 99):
        pred = p >= thr
        tp = float(np.logical_and(pred, y).sum())
        fp = float(np.logical_and(pred, ~y).sum())
        fn = float(np.logical_and(~pred, y).sum())
        denom = 2.0 * tp + fp + fn
        f1 = 0.0 if denom <= 0.0 else (2.0 * tp / denom)
        if f1 > best:
            best = f1
            best_thr = float(thr)
    return best, best_thr


def make_split_indices(labels: np.ndarray, val_ratio: float, seed: int, balance_train: bool) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    y = np.asarray(labels).reshape(-1) > 0.5
    pos = np.flatnonzero(y)
    neg = np.flatnonzero(~y)
    rng.shuffle(pos)
    rng.shuffle(neg)

    n_pos_val = max(1, int(pos.shape[0] * float(val_ratio))) if pos.shape[0] > 0 else 0
    n_neg_val = max(1, int(neg.shape[0] * float(val_ratio))) if neg.shape[0] > 0 else 0
    val_idx = np.concatenate([pos[:n_pos_val], neg[:n_neg_val]])
    tr_pos = pos[n_pos_val:]
    tr_neg = neg[n_neg_val:]
    if balance_train and tr_pos.size > 0 and tr_neg.size > 0:
        m = min(tr_pos.size, tr_neg.size)
        train_idx = np.concatenate([tr_pos[:m], tr_neg[:m]])
    else:
        train_idx = np.concatenate([tr_pos, tr_neg])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx.astype(np.int64), val_idx.astype(np.int64)


def main() -> int:
    ap = argparse.ArgumentParser(description="Train MLPF (torch) from clean/noisy pair and export TorchScript.")
    ap.add_argument("--clean", required=True)
    ap.add_argument("--noisy", required=True)
    ap.add_argument("--width", type=int, default=346)
    ap.add_argument("--height", type=int, default=260)
    ap.add_argument("--tick-ns", type=float, default=1000.0)
    ap.add_argument("--duration-us", type=int, default=100000)
    ap.add_argument("--patch", type=int, default=7, choices=[3, 5, 7, 9, 11])
    ap.add_argument("--max-events", type=int, default=120000)
    ap.add_argument(
        "--sample-mode",
        choices=["first", "linspace"],
        default="first",
        help="how to reduce noisy stream before feature construction; use first for stateful MLPF features",
    )
    ap.add_argument("--batch-events", type=int, default=1_000_000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=20)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-balance", action="store_true", help="disable 1:1 signal/noise balancing for training")
    ap.add_argument("--out-ts", required=True, help="TorchScript output .pt")
    ap.add_argument("--out-meta", required=True, help="metadata json")
    args = ap.parse_args()

    tb = TimeBase(tick_ns=float(args.tick_ns))
    duration_ticks = int(tb.us_to_ticks(int(args.duration_us)))

    print("[1/5] build clean index...")
    r_clean = open_events(
        args.clean,
        width=args.width,
        height=args.height,
        batch_events=int(args.batch_events),
        tick_ns=float(args.tick_ns),
        assume="npy",
    )
    clean_batches = unwrap_tick_batches(r_clean.batches, bits=None)
    clean_keys, packer = build_clean_index(
        r_clean.meta,
        clean_batches,
        show_on=True,
        show_off=True,
        unwrap_ts=False,
        ts_bits=None,
        match_ticks=0,
        match_bin_radius=0,
    )

    print("[2/5] load noisy events...")
    t, x, y, p = collect_events(
        args.noisy,
        width=args.width,
        height=args.height,
        tick_ns=float(args.tick_ns),
        batch_events=int(args.batch_events),
    )
    if t.size == 0:
        raise SystemExit("No noisy events loaded.")

    if int(args.max_events) > 0 and t.shape[0] > int(args.max_events):
        if str(args.sample_mode).lower() == "linspace":
            idx = np.linspace(0, t.shape[0] - 1, int(args.max_events), dtype=np.int64)
            t, x, y, p = t[idx], x[idx], y[idx], p[idx]
        else:
            n_keep = int(args.max_events)
            t, x, y, p = t[:n_keep], x[:n_keep], y[:n_keep], p[:n_keep]

    print("[3/5] build labels...")
    y_sig = signal_mask(
        clean_keys=clean_keys,
        packer=packer,
        t=t,
        x=x,
        y=y,
        p=p,
        match_ticks=0,
        match_bin_radius=0,
    ).astype(np.float32)

    print("[4/5] build features...")
    X = build_features(
        t,
        x,
        y,
        p,
        width=int(args.width),
        height=int(args.height),
        duration_ticks=duration_ticks,
        patch=int(args.patch),
    )

    train_idx, val_idx = make_split_indices(
        y_sig,
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
        balance_train=not bool(args.no_balance),
    )
    n_train = int(train_idx.shape[0])
    n_val = int(val_idx.shape[0])
    if n_train <= 0 or n_val <= 0:
        raise SystemExit("Empty train/validation split. Check labels and val-ratio.")
    X_train = torch.from_numpy(X[train_idx])
    y_train = torch.from_numpy(y_sig[train_idx]).unsqueeze(1)
    X_val = torch.from_numpy(X[val_idx])
    y_val = torch.from_numpy(y_sig[val_idx]).unsqueeze(1)

    pos_total = int((y_sig > 0.5).sum())
    neg_total = int(y_sig.shape[0] - pos_total)
    print(
        f"label stats: total={y_sig.shape[0]} signal={pos_total} noise={neg_total} "
        f"signal_ratio={pos_total / max(1, y_sig.shape[0]):.4f}"
    )
    print(
        f"split stats: train={n_train} val={n_val} "
        f"train_signal={int((y_sig[train_idx] > 0.5).sum())} train_noise={int((y_sig[train_idx] <= 0.5).sum())}"
    )

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=int(args.batch_size), shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=int(args.batch_size), shuffle=False)

    device = torch.device("cpu")
    model = MLPFNet(in_dim=X.shape[1], hidden=int(args.hidden)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    crit = nn.BCEWithLogitsLoss()

    print("[5/5] training...")
    for ep in range(int(args.epochs)):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            tr_loss += float(loss.item()) * xb.shape[0]
        tr_loss /= max(1, n_train)

        model.eval()
        va_loss = 0.0
        va_acc = 0.0
        val_prob_chunks: list[np.ndarray] = []
        val_y_chunks: list[np.ndarray] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = crit(logits, yb)
                va_loss += float(loss.item()) * xb.shape[0]
                prob = torch.sigmoid(logits)
                pred = (prob >= 0.5).float()
                va_acc += float((pred == yb).float().mean().item()) * xb.shape[0]
                val_prob_chunks.append(prob.cpu().numpy().reshape(-1))
                val_y_chunks.append(yb.cpu().numpy().reshape(-1))
        va_loss /= max(1, n_val)
        va_acc /= max(1, n_val)
        val_prob = np.concatenate(val_prob_chunks) if val_prob_chunks else np.empty((0,), dtype=np.float32)
        val_y_np = np.concatenate(val_y_chunks) if val_y_chunks else np.empty((0,), dtype=np.float32)
        va_auc = binary_auc(val_y_np, val_prob)
        va_best_f1, va_best_thr = best_f1(val_y_np, val_prob)
        always_keep_f1 = best_f1(val_y_np, np.ones_like(val_prob))[0]
        print(
            f"epoch {ep+1}/{args.epochs} train_loss={tr_loss:.6f} val_loss={va_loss:.6f} "
            f"val_acc={va_acc:.4f} val_auc={va_auc:.4f} val_best_f1={va_best_f1:.4f} "
            f"best_thr={va_best_thr:.2f} always_keep_f1={always_keep_f1:.4f}"
        )

    out_ts = Path(args.out_ts)
    out_ts.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    example = torch.randn(1, X.shape[1], dtype=torch.float32)
    scripted = torch.jit.trace(model.cpu(), example)
    scripted.save(str(out_ts))

    meta = {
        "patch": int(args.patch),
        "duration_us": int(args.duration_us),
        "duration_ticks": int(duration_ticks),
        "hidden": int(args.hidden),
        "input_dim": int(X.shape[1]),
        "tick_ns": float(args.tick_ns),
        "width": int(args.width),
        "height": int(args.height),
        "train_samples": int(n_train),
        "val_samples": int(n_val),
        "balanced_train": not bool(args.no_balance),
        "sample_mode": str(args.sample_mode),
        "seed": int(args.seed),
        "signal_total": int(pos_total),
        "noise_total": int(neg_total),
        "output_type": "logit",
    }
    out_meta = Path(args.out_meta)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved TorchScript: {out_ts}")
    print(f"saved meta: {out_meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
