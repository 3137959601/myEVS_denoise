import os, glob
import pandas as pd

out_dir = r"""data/ED24/myPedestrain_06/EBF_Part2/s23_featlogit_selfacc_burst_prescreen_s9_tau128ms_200k_wsamefix1_train50k_ep15_seed0_clip5_nonpos_toggle_dtsmall_selfacc_esrall_aoccall"""

cols_want = [
    'tag','f1','precision','tpr','tp','fp','fn','tn','auc','aocc','esr_mean'
]

def best_f1_for_env(env: str):
    paths = sorted(glob.glob(os.path.join(out_dir, f"roc_*_{env}_*.csv")))
    if not paths:
        return None, []
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        df['__src__'] = os.path.basename(p)
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)

    # ensure numeric where expected
    for c in ['f1','precision','tpr','tp','fp','fn','tn','auc','aocc','esr_mean']:
        if c in all_df.columns:
            all_df[c] = pd.to_numeric(all_df[c], errors='coerce')

    if 'f1' not in all_df.columns:
        return None, paths

    best = all_df.loc[all_df['f1'].fillna(float('-inf')).idxmax()]
    return best, paths

for env in ['light','mid','heavy']:
    best, paths = best_f1_for_env(env)
    print(f"\n=== BEST F1: {env} ===")
    if best is None:
        print("No roc CSVs found.")
        continue
    row = {c: (best[c] if c in best.index else None) for c in cols_want}
    row['__src__'] = best.get('__src__', None)

    # pretty print (stable order)
    for k in ['tag','f1','precision','tpr','tp','fp','fn','tn','auc','aocc','esr_mean','__src__']:
        v = row.get(k)
        if isinstance(v, float):
            print(f"{k}: {v:.6g}")
        else:
            print(f"{k}: {v}")
