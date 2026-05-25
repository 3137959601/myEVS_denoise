import subprocess,time
from pathlib import Path
ROOT=Path(r"D:/hjx_workspace/scientific_reserach/projects/myEVS")
PY=r"D:/software/Anaconda_envs/envs/myEVS/python.exe"
LOG=ROOT/"data/MLPF_retrain_logs_20260522_parallel"
LOG.mkdir(parents=True,exist_ok=True)

dv=[(s,l) for s in ["MAH00444","MAH00446","MAH00447","MAH00448","MAH00449"] for l in ["ratio50","ratio100"]]
led=["scene_100","scene_1004","scene_1018","scene_1028","scene_1032","scene_1033","scene_1034","scene_1043","scene_1045","scene_1046"]
rd=[]; rl=[]; i=0; j=0
while True:
    rd=[p for p in rd if p.poll() is None]
    rl=[p for p in rl if p.poll() is None]
    while len(rd)<5 and i<len(dv):
        s,l=dv[i]; i+=1
        f=open(LOG/f"dv_{s}_{l}.log","a",encoding="utf-8")
        cmd=["powershell","-NoProfile","-ExecutionPolicy","Bypass","-File",str(ROOT/"scripts/DVSCLEAN_alg_evalu/run_dvsclean_one.ps1"),"-Scene",s,"-Level",l,"-PythonExe",PY]
        p=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        p._f=f; rd.append(p)
    while len(rl)<5 and j<len(led):
        s=led[j]; j+=1
        f=open(LOG/f"led_{s}.log","a",encoding="utf-8")
        cmd=["powershell","-NoProfile","-ExecutionPolicy","Bypass","-File",str(ROOT/"scripts/LED_alg_evalu/run_led_scene_full.ps1"),"-Scene",s,"-PythonExe",PY]
        p=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        p._f=f; rl.append(p)
    if i==len(dv) and j==len(led) and not rd and not rl:
        break
    time.sleep(2)
