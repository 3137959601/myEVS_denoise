param(
  [Parameter(Mandatory = $true)]
  [string]$Scene,
  [string]$PythonExe = "D:/software/Anaconda_envs/envs/myEVS/python.exe",
  [string]$NpyRoot = "D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy",
  [string]$OutRoot = "data/LED/scene_sweep_full",
  [switch]$EvflowLite = $true
)

$ErrorActionPreference = "Stop"

$sceneDir = Join-Path $NpyRoot "$Scene/slices_00031_00040_100ms"
$clean = Join-Path $sceneDir "${Scene}_100ms_signal_only.npy"
$noisy = Join-Path $sceneDir "${Scene}_100ms_labeled.npy"
$model = "data/LED/models/mlpf_torch_${Scene}.pt"
$meta = "data/LED/models/mlpf_torch_${Scene}.json"

if (!(Test-Path $clean)) { throw "Missing clean file: $clean" }
if (!(Test-Path $noisy)) { throw "Missing noisy file: $noisy" }

Write-Host "[LED][$Scene] stage 1/2: train MLPF patch=7 full events"
& $PythonExe scripts/train_mlpf_torch.py `
  --clean $clean `
  --noisy $noisy `
  --width 1280 `
  --height 720 `
  --tick-ns 1000 `
  --duration-us 100000 `
  --patch 7 `
  --epochs 4 `
  --batch-size 512 `
  --max-events 0 `
  --out-ts $model `
  --out-meta $meta
if ($LASTEXITCODE -ne 0) { throw "MLPF training failed for $Scene" }

Write-Host "[LED][$Scene] stage 2/2: full scene sweep"
$cmd = @(
  "scripts/LED_alg_evalu/run_led_scene_sweep_summary.py",
  "--scene", $Scene,
  "--npy-root", $NpyRoot,
  "--out-root", $OutRoot,
  "--max-events", "0",
  "--mlpf-model-pattern", "data/LED/models/mlpf_torch_{scene}.pt",
  "--mlpf-patch", "7"
)
if ($EvflowLite) { $cmd += "--evflow-lite" }
& $PythonExe @cmd
if ($LASTEXITCODE -ne 0) { throw "Sweep failed for $Scene" }

Write-Host "[LED][$Scene] done"
