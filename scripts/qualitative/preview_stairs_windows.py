"""Preview Noisy images at multiple candidate windows for stairs_125854."""
import sys, os, numpy as np
sys.path.insert(0, r'D:/hjx_workspace/scientific_reserach/projects/myEVS/src')
from pathlib import Path
from myevs.qualitative.render import render_events_to_image

d = np.load(r'D:/hjx_workspace/scientific_reserach/projects/myEVS/data/qualitative/converted/dvsnoise20/stairs-2019_10_10_12_58_54.npz')
t, x, y, p = d['t'], d['x'], d['y'], d['p']
ws = 50000  # 50ms

# Multiple time windows spanning the recording
starts = [6000000, 9000000, 12000000, 15000000, 18000000, 21000000]

out_dir = Path(r'D:/hjx_workspace/scientific_reserach/projects/myEVS/data/qualitative/candidates/stairs_125854')
out_dir.mkdir(parents=True, exist_ok=True)

for start_us in starts:
    mask = (t >= start_us) & (t < start_us + ws)
    cnt = int(mask.sum())
    if cnt == 0:
        print(f't={start_us/1e6:.0f}s: 0 events, skip')
        continue

    xt = x[mask][:200000]; yt = y[mask][:200000]
    pt = p[mask][:200000]; tt = t[mask][:200000]

    out_png = out_dir / f'stairs_noisy_t{start_us/1e6:.0f}s_{cnt}ev.png'
    render_events_to_image(
        t=tt, x=xt, y=yt, p=pt,
        width=346, height=260,
        out_path=str(out_png),
        window_us=ws,
        scheme=0, binary=True, deadzone=0, raw_step=127
    )
    print(f't={start_us/1e6:.0f}s: {cnt:>6} events -> {out_png.name}')
print('DONE')
