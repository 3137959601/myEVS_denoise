import argparse, json
from pathlib import Path
import pandas as pd
import subprocess

PY = r"D:/software/Anaconda_envs/envs/myEVS/python.exe"

THR_DV_LED = "0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"
THR_ED24 = "0.2,0.4,0.6,0.8"


def run(cmd):
    print("[CMD]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def ensure_reset_csv(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def best_from_csv(p: Path):
    df = pd.read_csv(p)
    g = df[["tag", "auc"]].drop_duplicates("tag")
    b = g.sort_values("auc", ascending=False).iloc[0]
    tag = str(b["tag"])
    auc = float(b["auc"])
    sub = df[df["tag"] == tag]
    f1 = float(sub["f1"].max()) if "f1" in sub.columns else float("nan")
    return {"tag": tag, "auc": auc, "f1": f1}


def train(clean,noisy,width,height,duration,patch,epochs,model,meta):
    model.parent.mkdir(parents=True, exist_ok=True)
    run([
        PY, "scripts/train_mlpf_torch.py",
        "--clean", str(clean), "--noisy", str(noisy),
        "--width", str(width), "--height", str(height),
        "--tick-ns", "1000", "--duration-us", str(duration),
        "--patch", str(patch), "--epochs", str(epochs),
        "--batch-size", "512", "--max-events", "0",
        "--out-ts", str(model), "--out-meta", str(meta)
    ])


def eval_mlpf(clean,noisy,width,height,taus,thr,model,patch,out_csv,tag_prefix,radius):
    ensure_reset_csv(out_csv)
    for tau in taus:
        run([
            PY, "-m", "myevs.cli", "roc",
            "--clean", str(clean), "--noisy", str(noisy),
            "--assume", "npy", "--width", str(width), "--height", str(height),
            "--tick-ns", "1000", "--method", "mlpf", "--engine", "python",
            "--radius-px", str(radius), "--time-us", str(tau),
            "--param", "min-neighbors", "--values", thr,
            "--match-us", "0", "--match-bin-radius", "0",
            "--tag", f"{tag_prefix}_tau{tau}",
            "--out-csv", str(out_csv), "--append",
            "--mlpf-model", str(model), "--mlpf-patch", str(patch)
        ])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--target', required=True, choices=['ed24_ped','ed24_bike_mid','dv10','led10'])
    args=ap.parse_args()

    results=[]

    if args.target=='ed24_ped':
        base=Path(r'D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06')
        out=Path('data/ED24/myPedestrain_06/MLPF')
        model_dir=out/'models_retrain_20260522'
        levels=[('light','1.8'),('light_mid','2.1'),('mid','2.5'),('heavy','3.3')]
        for lv,suf in levels:
            clean=base/f'Pedestrain_06_{suf}_signal_only.npy'; noisy=base/f'Pedestrain_06_{suf}.npy'
            model=model_dir/f'mlpf_torch_{lv}.pt'; meta=model_dir/f'mlpf_torch_{lv}.json'
            train(clean,noisy,346,260,100000,7,4,model,meta)
            out_csv=out/f'roc_mlpf_{lv}.csv'
            eval_mlpf(clean,noisy,346,260,[32000,64000,128000,256000,512000],THR_ED24,model,7,out_csv,'mlpf_retrain',3)
            b=best_from_csv(out_csv); b['dataset']=f'ED24_Ped_{lv}'; b['model']=str(model)
            results.append(b)

    elif args.target=='ed24_bike_mid':
        base=Path(r'D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02')
        out=Path('data/ED24/myBicycle_02/MLPF')
        model_dir=out/'models_retrain_20260522'
        clean=base/'Bicycle_02_2.5_signal_only.npy'; noisy=base/'Bicycle_02_2.5.npy'
        model=model_dir/'mlpf_torch_mid.pt'; meta=model_dir/'mlpf_torch_mid.json'
        train(clean,noisy,346,260,100000,7,4,model,meta)
        out_csv=out/'roc_mlpf_mid.csv'
        eval_mlpf(clean,noisy,346,260,[32000,64000,128000,256000,512000],THR_ED24,model,7,out_csv,'mlpf_retrain',3)
        b=best_from_csv(out_csv); b['dataset']='ED24_Bike_mid'; b['model']=str(model)
        results.append(b)

    elif args.target=='dv10':
        root=Path(r'D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy')
        out_root=Path('data/DVSCLEAN/scene_sweep_full')
        model_dir=Path('data/DVSCLEAN/models_retrain_20260522')
        scenes=['MAH00444','MAH00446','MAH00447','MAH00448','MAH00449']
        for scene in scenes:
            for lv in ['ratio50','ratio100']:
                clean=root/scene/lv/f'{scene}_{lv}_signal_only.npy'
                noisy=root/scene/lv/f'{scene}_{lv}_labeled.npy'
                model=model_dir/f'mlpf_torch_{scene}_{lv}.pt'; meta=model_dir/f'mlpf_torch_{scene}_{lv}.json'
                train(clean,noisy,1280,720,128000,7,6,model,meta)
                out_csv=out_root/f'{scene}_{lv}'/'mlpf'/'roc_mlpf_r3_tau64000.csv'
                eval_mlpf(clean,noisy,1280,720,[8000,16000,32000,64000],THR_DV_LED,model,7,out_csv,f'mlpf_{scene}_{lv}',3)
                b=best_from_csv(out_csv); b['dataset']=f'DV_{scene}_{lv}'; b['model']=str(model)
                results.append(b)

    elif args.target=='led10':
        root=Path(r'D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy')
        out_root=Path('data/LED/scene_sweep_full')
        model_dir=Path('data/LED/models_retrain_20260522')
        scenes=['scene_100','scene_1004','scene_1018','scene_1028','scene_1032','scene_1033','scene_1034','scene_1043','scene_1045','scene_1046']
        for scene in scenes:
            d=root/scene/'slices_00031_00040_100ms'
            clean=d/f'{scene}_100ms_signal_only.npy'; noisy=d/f'{scene}_100ms_labeled.npy'
            model=model_dir/f'mlpf_torch_{scene}.pt'; meta=model_dir/f'mlpf_torch_{scene}.json'
            train(clean,noisy,1280,720,100000,7,4,model,meta)
            out_csv=out_root/scene/'mlpf'/'roc_mlpf_r3_tau16000.csv'
            eval_mlpf(clean,noisy,1280,720,[8000,16000],THR_DV_LED,model,7,out_csv,f'mlpf_{scene}',3)
            b=best_from_csv(out_csv); b['dataset']=f'LED_{scene}'; b['model']=str(model)
            results.append(b)

    res_path=Path('data/MLPF_retrain_20260522_summary.csv')
    pd.DataFrame(results).to_csv(res_path,index=False)
    print(f'[DONE] saved {res_path}')

if __name__=='__main__':
    main()
