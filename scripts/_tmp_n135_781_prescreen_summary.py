from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path('data/ED24/myPedestrain_06/EBF_Part2')


def read_rows(path: Path):
    with path.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def best_auc_f1(path: Path):
    rows = read_rows(path)
    ba = max(rows, key=lambda r: float(r['auc']))
    bf = max(rows, key=lambda r: float(r['f1']))
    return float(ba['auc']), float(bf['f1'])


rows = []
for d in sorted(ROOT.glob('_tune_n135_781_sact*_strust*_h0p9_200k_s9tau128')):
    name = d.name
    sact = float(name.split('_sact', 1)[1].split('_strust', 1)[0].replace('p', '.'))
    strust = float(name.split('_strust', 1)[1].split('_h0p9', 1)[0].replace('p', '.'))

    env = {}
    for e in ('light','mid','heavy'):
        p = d / f'roc_ebf_n135_{e}_labelscore_s9_tau128ms.csv'
        env[e] = best_auc_f1(p)

    ma = sum(env[e][0] for e in ('light','mid','heavy')) / 3.0
    mf = sum(env[e][1] for e in ('light','mid','heavy')) / 3.0
    rows.append((sact, strust, ma, mf, env))

rows.sort(key=lambda x: (x[2], x[3]), reverse=True)
print('s_act,s_trust,mean_auc,mean_f1,light_auc,light_f1,mid_auc,mid_f1,heavy_auc,heavy_f1')
for sact, st, ma, mf, env in rows:
    la, lf = env['light']
    ma2, mf2 = env['mid']
    ha, hf = env['heavy']
    print(f'{sact:.1f},{st:.1f},{ma:.6f},{mf:.6f},{la:.6f},{lf:.6f},{ma2:.6f},{mf2:.6f},{ha:.6f},{hf:.6f}')
