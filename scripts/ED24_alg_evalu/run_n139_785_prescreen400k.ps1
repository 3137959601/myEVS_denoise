# n139 (7.85) prescreen400k full-grid sweep
# 说明：联合 sweep s/tau 与 n139 的 low/high 阈值。

$env:PYTHONNOUSERSITE = "1"

$CONDA_HOOK = "D:/software/anaconda3/shell/condabin/conda-hook.ps1"
if (Test-Path $CONDA_HOOK) {
	& $CONDA_HOOK
	conda activate myEVS
}

$PY = "D:/software/Anaconda_envs/envs/myEVS/python.exe"


& $PY scripts/ED24_alg_evalu/sweep_n139_785_prescreen400k.py `
	--max-events 400000 `
	--s-list 3,5,7,9 `
	--tau-us-list 8000,16000,32000,64000,128000,256000,512000,1024000 `
	--n139-low-list 0.1,0.2,0.3,0.4,0.5 `
	--n139-high-list 0.5,0.6,0.7,0.8,0.9 `
	--esr-mode off `
	--aocc-mode off `
	--out-dir data/ED24/myPedestrain_06/EBF_Part2/_slim_n139_785_prescreen400k
