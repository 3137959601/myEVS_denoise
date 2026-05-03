param(
  [Parameter(Mandatory = $true)]
  [string]$Scene,
  [Parameter(Mandatory = $true)]
  [ValidateSet("ratio50", "ratio100")]
  [string]$Level,
  [string]$PythonExe = "D:/software/Anaconda_envs/envs/myEVS/python.exe",
  [string]$NpyRoot = "D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy",
  [switch]$EvflowLite = $true
)

$ErrorActionPreference = "Stop"

$base = Join-Path $NpyRoot "$Scene/$Level"
$clean = Join-Path $base "${Scene}_${Level}_signal_only.npy"
$noisy = Join-Path $base "${Scene}_${Level}_labeled.npy"
$modelDir = "data/DVSCLEAN/models"
$model = Join-Path $modelDir "mlpf_torch_${Scene}_${Level}.pt"
$meta = Join-Path $modelDir "mlpf_torch_${Scene}_${Level}.json"

if (!(Test-Path $clean)) { throw "Missing clean file: $clean" }
if (!(Test-Path $noisy)) { throw "Missing noisy file: $noisy" }

Write-Host "[DVSCLEAN][$Scene/$Level] stage 1/2: train MLPF patch=7"
& $PythonExe scripts/train_mlpf_torch.py `
  --clean $clean `
  --noisy $noisy `
  --width 1280 `
  --height 720 `
  --tick-ns 1000 `
  --duration-us 128000 `
  --patch 7 `
  --epochs 6 `
  --batch-size 512 `
  --max-events 0 `
  --out-ts $model `
  --out-meta $meta
if ($LASTEXITCODE -ne 0) { throw "MLPF train failed for $Scene/$Level" }

Write-Host "[DVSCLEAN][$Scene/$Level] stage 2/2: full sweep"
$cmd = @(
  "scripts/DVSCLEAN_alg_evalu/run_dvsclean_scene_sweep_summary.py",
  "--scene", $Scene,
  "--level", $Level,
  "--npy-root", $NpyRoot,
  "--out-root", "data/DVSCLEAN/scene_sweep_full",
  "--mlpf-model", $model,
  "--mlpf-patch", "7"
)
if ($EvflowLite) { $cmd += "--evflow-lite" }
& $PythonExe @cmd
if ($LASTEXITCODE -ne 0) { throw "Sweep failed for $Scene/$Level" }

Write-Host "[DVSCLEAN][$Scene/$Level] done"
