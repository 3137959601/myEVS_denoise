"""Render all algorithms on MAH00447 ratio100 w=50ms, then make panel."""
import sys, os, numpy as np
sys.path.insert(0, r'D:/hjx_workspace/scientific_reserach/projects/myEVS/src')
from pathlib import Path
from myevs.events import EventBatch
from myevs.qualitative.render import render_events_to_image, write_png
from myevs.denoise.pipeline import DenoiseConfig, denoise_stream
from myevs.timebase import TimeBase

clean = np.load(r'D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy/MAH00447/ratio100/MAH00447_ratio100_signal_only.npy')
noisy = np.load(r'D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy/MAH00447/ratio100/MAH00447_ratio100_labeled.npy')

W, H = 1280, 720
start_us = int(noisy['t'][0])
ws_us = 50000

mask = (noisy['t'] >= start_us) & (noisy['t'] < start_us + ws_us)
t_w = noisy['t'][mask].astype(np.uint64)
x_w = noisy['x'][mask].astype(np.uint16)
y_w = noisy['y'][mask].astype(np.uint16)
p_w = noisy['p'][mask].astype(np.int8)
print(f'Window: t={start_us} +{ws_us/1000}ms, {len(t_w)} events')

out_dir = Path(r'D:/hjx_workspace/scientific_reserach/projects/myEVS/data/qualitative/rendered/mah00447_ratio100')
out_dir.mkdir(parents=True, exist_ok=True)

tb = TimeBase(tick_ns=1000.0)
meta = type('Meta', (), {'width': W, 'height': H})()
batch_input = EventBatch(t=t_w, x=x_w, y=y_w, p=p_w)

# Noisy first
img = render_events_to_image(batch_input, width=W, height=H, binary=True, deadzone=0, raw_step=127, scheme=0)
write_png(str(out_dir / 'Noisy.png'), img)
print(f'  Noisy: {len(t_w)} events')

# Algorithm configs for DVSCLEAN
configs = [
    ("Ours", {"method": "18", "radius_px": 3, "time_window_us": 128000, "min_neighbors": 2.0,
              "env": {"MYEVS_N149_SIGMA": "2.5", "MYEVS_N149_ALPHA_FIXED": "0.25"}}),
    ("BAF", {"method": "4", "radius_px": 1, "time_window_us": 50000, "min_neighbors": 1}),
    ("STCF", {"method": "19", "radius_px": 1, "time_window_us": 50000, "min_neighbors": 2}),
    ("EBF", {"method": "10", "radius_px": 3, "time_window_us": 64000, "min_neighbors": 2.0}),
    ("TS", {"method": "15", "radius_px": 1, "time_window_us": 32000, "min_neighbors": 0.2}),
    ("PFD", {"method": "17", "radius_px": 1, "time_window_us": 64000, "min_neighbors": 1,
             "refractory_us": 1, "pfd_mode": "a"}),
]

for name, cfg_dict in configs:
    env = cfg_dict.pop("env", {})
    for k, v in env.items(): os.environ[k] = v

    cfg = DenoiseConfig(**cfg_dict, show_on=True, show_off=True)
    den = list(denoise_stream(meta, [batch_input], cfg, timebase=tb, engine="cpp"))
    if not den or len(den[0]) == 0:
        print(f'  {name}: 0 events')
        continue
    d2 = den[0]
    kept = len(d2)
    batch2 = EventBatch(t=d2.t.astype(np.uint64), x=d2.x.astype(np.uint16),
                        y=d2.y.astype(np.uint16), p=d2.p.astype(np.int8))
    img2 = render_events_to_image(batch2, width=W, height=H, binary=True, deadzone=0, raw_step=127, scheme=0)
    write_png(str(out_dir / f'{name}.png'), img2)
    print(f'  {name}: {kept} events kept')

# Panel
from myevs.qualitative.layout import save_panel
images = [str(out_dir / f'{n}.png') for n in ['Noisy','Ours','BAF','STCF','EBF','TS','PFD']]
labels = ['Noisy','Ours','BAF','STCF','EBF','TS','PFD']
save_panel(images, str(out_dir / 'panel.png'), labels=labels, cols=4)
print(f'Panel: {out_dir / "panel.png"}')
print('DONE')
