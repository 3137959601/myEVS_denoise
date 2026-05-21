"""Compile phase1 summary CSVs from individual ROC files."""
import pandas as pd, os, glob, re

base = r'D:\hjx_workspace\scientific_reserach\projects\myEVS\data\Hyperparameter ablation_study'

for ds in ['ped', 'bike', 'drive', 'dvsclean', 'led']:
    d = os.path.join(base, ds)
    files = glob.glob(os.path.join(d, f'p1_{ds}_*.csv'))
    if len(files) < 10:
        print(f'{ds}: only {len(files)} files, skipping')
        continue
    rows = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if len(df) == 0: continue
            best = df.loc[df.auc.idxmax()]
            # Parse: p1_ds_LEVEL_PARAM_VALUE.csv
            fname = os.path.basename(f).replace('.csv', '')
            parts = fname.split('_')
            # parts[0]=p1, parts[1]=ds, then level (1-2 tokens), param, value
            if parts[3] in ('r', 'tau', 'sigma', 'alpha'):
                level = parts[2]
                param = parts[3]
                value = '_'.join(parts[4:])
            else:
                level = f'{parts[2]}_{parts[3]}'
                param = parts[4]
                value = '_'.join(parts[5:])
            # Clean up value: p1_drive_1hz_r_1 -> value=1
            rows.append((level, param, value, float(best.auc), float(best.f1)))
        except Exception as e:
            pass

    if not rows:
        print(f'{ds}: no valid rows')
        continue

    df = pd.DataFrame(rows, columns=['level', 'param', 'value', 'auc', 'f1'])
    df.to_csv(os.path.join(d, f'phase1_{ds}.csv'), index=False)
    print(f'{ds}: compiled {len(df)} rows')

    for p in ['r', 'tau', 'sigma', 'alpha']:
        sub = df[df.param == p]
        if len(sub) == 0: continue
        best = sub.loc[sub.auc.idxmax()]
        # Show per-level best
        vals_per_level = {}
        for lv in df.level.unique():
            sl = df[(df.level == lv) & (df.param == p)]
            if len(sl) > 0:
                b = sl.loc[sl.auc.idxmax()]
                vals_per_level[lv] = str(b.value)
        print(f'  Best {p}: mean={best.value} AUC={best.auc:.4f}  per-level: {vals_per_level}')
    print()
