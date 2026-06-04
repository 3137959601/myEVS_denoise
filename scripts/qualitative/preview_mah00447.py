"""Preview Noisy for MAH00447 ratio100 at candidate windows."""
import sys, numpy as np
sys.path.insert(0, r'D:/hjx_workspace/scientific_reserach/projects/myEVS/src')
from pathlib import Path
from myevs.events import EventBatch
from myevs.qualitative.render import render_events_to_image, write_png

d = np.load(r'D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy/MAH00447/ratio100/MAH00447_ratio100_labeled.npy')
t, x, y, p = d['t'], d['x'], d['y'], d['p']
W, H = 1280, 720

total_s = (t[-1]-t[0])/1e6
print(f'Events: {len(t)}, duration: {total_s:.1f}s, {x.max()+1}x{y.max()+1}')

out_dir = Path(r'D:/hjx_workspace/scientific_reserach/projects/myEVS/data/qualitative/candidates/mah00447_ratio100')
out_dir.mkdir(parents=True, exist_ok=True)

# Vary window length from start
for ws_ms in [30, 50, 80, 100, 150, 200, 300]:
    ws_us = ws_ms * 1000
    mask = (t >= int(t[0])) & (t < int(t[0]) + ws_us)
    cnt = int(mask.sum())
    batch = EventBatch(
        t=t[mask][:300000].astype(np.uint64),
        x=x[mask][:300000].astype(np.uint16),
        y=y[mask][:300000].astype(np.uint16),
        p=p[mask][:300000].astype(np.int8)
    )
    img = render_events_to_image(batch, width=W, height=H, binary=True, deadzone=0, raw_step=127, scheme=0)
    out_png = out_dir / f'mah00447_noisy_w{ws_ms}ms_{cnt}ev.png'
    write_png(str(out_png), img)
    print(f'w={ws_ms:>3}ms: {cnt:>6} ev -> {out_png.name}')
print('DONE')
