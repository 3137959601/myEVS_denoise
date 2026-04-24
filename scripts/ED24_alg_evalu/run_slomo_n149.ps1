$env:PYTHONNOUSERSITE = "1"

$CONDA_HOOK = "D:/software/anaconda3/shell/condabin/conda-hook.ps1"
if (Test-Path $CONDA_HOOK) {
  & $CONDA_HOOK
  conda activate myEVS
}

$PY = "D:/software/Anaconda_envs/envs/myEVS/python.exe"

# =========================== TUNE_HERE: N149 sweep ===========================
# ED24 round1 best in current grid is around r=4, tau=4000.
& $PY "scripts/ED24_alg_evalu/run_n149_labelscore_grid.py" `
  --radius-list "3,4,5" `
  --tau-us-list "16000,32000,64000,128000,256000,512000" `
  --out-dir "data/ED24/myPedestrain_06/N149"
