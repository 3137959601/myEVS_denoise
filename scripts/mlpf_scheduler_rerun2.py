import subprocess, time
from pathlib import Path
ROOT=Path(r"D:/hjx_workspace/scientific_reserach/projects/myEVS")
PY=r"D:/software/Anaconda_envs/envs/myEVS/python.exe"
LOG=ROOT/"data/MLPF_retrain_logs_20260522_rerun2"
LOG.mkdir(parents=True,exist_ok=True)

dv=[(s,l) for s in ["MAH00444","MAH00446","MAH00447","MAH00448","MAH00449"] for l in ["ratio50","ratio100"]]
led=["scene_100","scene_1004","scene_1018","scene_1028","scene_1032","scene_1033","scene_1034","scene_1043","scene_1045","scene_1046"]

dv_ps=str(ROOT/"scripts/DVSCLEAN_alg_evalu/run_dvsclean_one.ps1")
led_ps=str(ROOT/"scripts/LED_alg_evalu/run_led_scene_full.ps1")

rd=[]; rl=[]; i=0; j=0
while True:
    rd=[p for p in rd if p.poll() is None]
    rl=[p for p in rl if p.poll() is None]
    while len(rd)<5 and i<len(dv):
        s,l=dv[i]; i+=1
        lf=open(LOG/f"dv_{s}_{l}.log","a",encoding="utf-8")
        p=subprocess.Popen(["powershell","-NoProfile","-ExecutionPolicy","Bypass","-File",dv_ps,"-Scene",s,"-Level",l,"-PythonExe",PY],cwd=ROOT,stdout=lf,stderr=subprocess.STDOUT)
        p._lf=lf; rd.append(p)
    while len(rl)<5 and j<len(led):
        s=led[j]; j+=1
        lf=open(LOG/f"led_{s}.log","a",encoding="utf-8")
        p=subprocess.Popen(["powershell","-NoProfile","-ExecutionPolicy","Bypass","-File",led_ps,"-Scene",s,"-PythonExe",PY],cwd=ROOT,stdout=lf,stderr=subprocess.STDOUT)
        p._lf=lf; rl.append(p)
    (LOG/"progress.txt").write_text(f"dv_running={len(rd)} dv_done={i-len(rd)}/10 led_running={len(rl)} led_done={j-len(rl)}/10",encoding="utf-8")
    if i==10 and j==10 and not rd and not rl:
        (LOG/"progress.txt").write_text("done",encoding="utf-8")
        break
    time.sleep(3)
