"""Render Ours (N149) r×tau sweep at selected window using Python API."""
import sys, os, numpy as np
sys.path.insert(0, r'D:/hjx_workspace/scientific_reserach/projects/myEVS/src')
from pathlib import Path
from myevs.events import EventBatch
from myevs.qualitative.render import render_events_to_image, write_png
from myevs.denoise.pipeline import DenoiseConfig, denoise_stream
from myevs.timebase import TimeBase

start_us = 12000000
ws_us = 50000
W, H = 346, 260
sigma, alpha = 1.75, 0.05

d = np.load(r'D:/hjx_workspace/scientific_reserach/projects/myEVS/data/qualitative/converted/dvsnoise20/stairs-2019_10_10_12_58_54.npz')
t_all, x_all, y_all, p_all = d['t'], d['x'], d['y'], d['p']

mask = (t_all >= start_us) & (t_all < start_us + ws_us)
t_w = t_all[mask].astype(np.uint64)
x_w = x_all[mask].astype(np.uint16)
y_w = y_all[mask].astype(np.uint16)
p_w = p_all[mask].astype(np.int8)

out_dir = Path(r'D:/hjx_workspace/scientific_reserach/projects/myEVS/data/qualitative/candidates/stairs_125854/ours_sweep')
out_dir.mkdir(parents=True, exist_ok=True)

# Noisy reference
batch = EventBatch(t=t_w, x=x_w, y=y_w, p=p_w)
img = render_events_to_image(batch, width=W, height=H, binary=True, deadzone=0, raw_step=127, scheme=0)
write_png(str(out_dir / '0_Noisy_ref.png'), img)
print(f'Reference: {len(t_w)} events')

# Setup TimeBase (tick_ns=1000 means 1us per tick)
tb = TimeBase(tick_ns=1000.0)
meta = type('Meta', (), {'width': W, 'height': H})()

R_LIST = [1, 2, 3, 5]
TAU_LIST = [8000, 16000, 32000, 64000]

for r in R_LIST:
    for tau in TAU_LIST:
        tag = f'ours_r{r}_tau{tau//1000}ms'
        os.environ["MYEVS_N149_HOT_BITS"] = "16"
        os.environ["MYEVS_N149_SIGMA"] = str(sigma)
        os.environ["MYEVS_N149_ALPHA_FIXED"] = str(alpha)

        cfg = DenoiseConfig(
            method="18", time_window_us=tau, radius_px=r, min_neighbors=0.2,
            show_on=True, show_off=True
        )

        # Create single batch and denoise
        den = list(denoise_stream(meta, [batch], cfg, timebase=tb, engine="cpp"))
        if not den or len(den[0]) == 0:
            print(f'  r={r} tau={tau//1000}ms: 0 events (all filtered)')
            continue

        d2 = den[0]  # Get the denoised batch
        kept = len(d2)
        batch2 = EventBatch(
            t=d2.t.astype(np.uint64),
            x=d2.x.astype(np.uint16),
            y=d2.y.astype(np.uint16),
            p=d2.p.astype(np.int8)
        )
        img2 = render_events_to_image(batch2, width=W, height=H, binary=True, deadzone=0, raw_step=127, scheme=0)
        out_png = str(out_dir / f'{tag}_{kept}ev.png')
        write_png(out_png, img2)
        print(f'  r={r} tau={tau//1000:>3}ms: {kept:>6} ev -> {Path(out_png).name}')
print('DONE')
