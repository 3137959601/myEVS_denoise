"""Preview Noisy at t=7.4s with different window lengths."""
import sys, numpy as np
sys.path.insert(0, r'D:/hjx_workspace/scientific_reserach/projects/myEVS/src')
from pathlib import Path
from myevs.events import EventBatch
from myevs.qualitative.render import render_events_to_image, write_png

d = np.load(r'D:/hjx_workspace/scientific_reserach/projects/myEVS/data/qualitative/converted/dvsnoise20/stairs-2019_10_10_12_58_54.npz')
t, x, y, p = d['t'], d['x'], d['y'], d['p']
start_us = 15000000
W, H = 346, 260

out_dir = Path(r'D:/hjx_workspace/scientific_reserach/projects/myEVS/data/qualitative/candidates/stairs_125854')
out_dir.mkdir(parents=True, exist_ok=True)

for ws_ms in [30, 50, 80, 100, 150, 200]:
    ws_us = ws_ms * 1000
    mask = (t >= start_us) & (t < start_us + ws_us)
    cnt = int(mask.sum())
    batch = EventBatch(
        t=t[mask][:300000].astype(np.uint64),
        x=x[mask][:300000].astype(np.uint16),
        y=y[mask][:300000].astype(np.uint16),
        p=p[mask][:300000].astype(np.int8)
    )
    img = render_events_to_image(batch, width=W, height=H, binary=True, deadzone=0, raw_step=127, scheme=0)
    out_png = out_dir / f'stairs_noisy_t15s_w{ws_ms}ms_{cnt}ev.png'
    write_png(str(out_png), img)
    print(f'w={ws_ms:>3}ms: {cnt:>6} ev -> {out_png.name}')
print('DONE')
