$env:PYTHONNOUSERSITE = "1"

$CONDA_HOOK = "D:/software/anaconda3/shell/condabin/conda-hook.ps1"
if (Test-Path $CONDA_HOOK) {
  & $CONDA_HOOK
  conda activate myEVS
}

$PY = "D:/software/Anaconda_envs/envs/myEVS/python.exe"

# Driving N149 compact sweep (light/light_mid/mid)
& $PY "scripts/ED24_alg_evalu/run_n149_labelscore_grid.py" `
  --max-events 200000 `
  --radius-list "2,3,4,5" `
  --tau-us-list "16000,32000,64000,128000,256000,512000" `
  --out-dir "data/DND21/mydriving/N149" `
  --light "D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving/driving_noise_light_slomo_shot_withlabel/driving_noise_light_labeled.npy" `
  --light-mid "D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving/driving_noise_light_mid_slomo_shot_withlabel/driving_noise_light_mid_labeled.npy" `
  --mid "D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving/driving_noise_mid_slomo_shot_withlabel/driving_noise_mid_labeled.npy" `
  --heavy " "
