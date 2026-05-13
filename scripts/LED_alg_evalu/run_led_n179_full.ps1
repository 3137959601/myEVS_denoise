param(
  [string]$PythonExe = "D:/software/Anaconda_envs/envs/myEVS/python.exe",
  [int]$MaxEvents = 0,
  [string]$NpyRoot = "D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy"
)

$ErrorActionPreference = "Stop"

$scenes = @("scene_100","scene_1004","scene_1018","scene_1028","scene_1032","scene_1033","scene_1034","scene_1043","scene_1045","scene_1046")

for ($i = 0; $i -lt $scenes.Count; $i++) {
  $scene = $scenes[$i]
  $noisy = Join-Path $NpyRoot ("{0}/slices_00031_00040_100ms/{0}_100ms_labeled.npy" -f $scene)
  if (!(Test-Path $noisy)) { throw "Missing file: $noisy" }
  $outDir = "data/LED/scene_sweep_full/{0}/N179" -f $scene
  New-Item -ItemType Directory -Force -Path $outDir | Out-Null
  $rocCsv = Join-Path $outDir "roc_n179.csv"
  $sumCsv = Join-Path $outDir "summary_n179.csv"
  $runCsv = Join-Path $outDir "runtime_n179.csv"

  Write-Host ("[LED][{0}/{1}] {2}" -f ($i + 1), $scenes.Count, $scene)
  & $PythonExe scripts/noise_analyze/sweep_n179_pair.py `
    --noisy $noisy `
    --out-csv $rocCsv `
    --summary-csv $sumCsv `
    --runtime-csv $runCsv `
    --dataset "LED" `
    --scene $scene `
    --level "100ms" `
    --width 1280 `
    --height 720 `
    --tick-ns 1000 `
    --s-list "5,7,9" `
    --tau-us-list "16000,32000,64000,128000,256000,512000" `
    --max-events $MaxEvents `
    --tag-prefix "n179"
  if ($LASTEXITCODE -ne 0) { throw "Failed at LED $scene" }
}

Write-Host "=== DONE: LED n179 sweep ==="

