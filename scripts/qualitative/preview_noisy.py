"""Preview Noisy images at candidate windows. Only raw input, no algorithms."""
import sys, os, numpy as np
sys.path.insert(0, r'D:/hjx_workspace/scientific_reserach/projects/myEVS/src')
from pathlib import Path
from myevs.events import EventBatch
from myevs.qualitative.render import render_events_to_image, write_png

d = np.load(r'D:/hjx_workspace/scientific_reserach/projects/myEVS/data/qualitative/converted/dvsnoise20/stairs-2019_10_10_12_58_54.npz')
t, x, y, p = d['t'], d['x'], d['y'], d['p']
ws_us = 50000
W, H = 346, 260

out_dir = Path(r'D:/hjx_workspace/scientific_reserach/projects/myEVS/data/qualitative/candidates/stairs_125854')
out_dir.mkdir(parents=True, exist_ok=True)

starts = [6000000, 9000000, 12000000, 15000000, 18000000, 21000000]

for start_us in starts:
    mask = (t >= start_us) & (t < start_us + ws_us)
    cnt = int(mask.sum())
    if cnt == 0:
        continue

    batch = EventBatch(
        t=t[mask][:200000].astype(np.uint64),
        x=x[mask][:200000].astype(np.uint16),
        y=y[mask][:200000].astype(np.uint16),
        p=p[mask][:200000].astype(np.int8)
    )

    img = render_events_to_image(batch, width=W, height=H, binary=True, deadzone=0, raw_step=127, scheme=0)
    out_png = out_dir / f'stairs_noisy_t{start_us//1000000}s_{cnt}ev.png'
    write_png(str(out_png), img)
    print(f't={start_us/1e6:.0f}s: {cnt:>6} ev -> {out_png.name}')
print('DONE')
