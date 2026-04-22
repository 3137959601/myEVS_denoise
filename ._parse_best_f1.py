import glob, os, sys
import pandas as pd

out_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
paths = sorted(glob.glob(os.path.join(out_dir, 'roc_*.csv')))
if not paths:
    raise SystemExit(f'No roc_*.csv found in {out_dir!r}')
cols = ['f1','precision','tpr','tp','fp','fn','tn','auc','aocc','esr_mean']

# Prefer printing in order: light, mid, heavy when present
order = {'light': 0, 'mid': 1, 'heavy': 2}

def env_key(path):
    b = os.path.basename(path)
    for k in order:
        if f'_{k}_' in b:
            return order[k]
    return 99

for p in sorted(paths, key=env_key):
    df = pd.read_csv(p)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"{os.path.basename(p)} missing columns: {missing}")

    # best F1 (tie-break: higher precision, then higher tpr)
    best = df.sort_values(['f1','precision','tpr'], ascending=[False,False,False]).iloc[0]
    print('\n' + os.path.basename(p) + ' best-F1 row:')
    # keep consistent formatting
    for c in cols:
        v = best[c]
        if isinstance(v, float):
            print(f'{c}: {v:.6f}')
        else:
            print(f'{c}: {v}')
