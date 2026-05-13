param(
  [string]$PythonExe = "D:/software/Anaconda_envs/envs/myEVS/python.exe",
  [int]$MaxEvents = 0,
  [string]$NpyRoot = "D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy"
)

$ErrorActionPreference = "Stop"

$scenes = @("MAH00444","MAH00446","MAH00447","MAH00448","MAH00449")
$levels = @("ratio50","ratio100")
$jobs = @()
foreach ($s in $scenes) {
  foreach ($l in $levels) {
    $jobs += @{ Scene = $s; Level = $l; Noisy = (Join-Path $NpyRoot ("{0}/{1}/{0}_{1}_labeled.npy" -f $s, $l)) }
  }
}

for ($i = 0; $i -lt $jobs.Count; $i++) {
  $j = $jobs[$i]
  if (!(Test-Path $j.Noisy)) { throw "Missing file: $($j.Noisy)" }
  $outDir = "data/DVSCLEAN/scene_sweep_full/{0}/{1}/N179" -f $j.Scene, $j.Level
  New-Item -ItemType Directory -Force -Path $outDir | Out-Null
  $rocCsv = Join-Path $outDir "roc_n179.csv"
  $sumCsv = Join-Path $outDir "summary_n179.csv"
  $runCsv = Join-Path $outDir "runtime_n179.csv"

  Write-Host ("[DVSCLEAN][{0}/{1}] {2} {3}" -f ($i + 1), $jobs.Count, $j.Scene, $j.Level)
  & $PythonExe scripts/noise_analyze/sweep_n179_pair.py `
    --noisy $j.Noisy `
    --out-csv $rocCsv `
    --summary-csv $sumCsv `
    --runtime-csv $runCsv `
    --dataset "DVSCLEAN" `
    --scene $j.Scene `
    --level $j.Level `
    --width 1280 `
    --height 720 `
    --tick-ns 1000 `
    --s-list "5,7,9" `
    --tau-us-list "16000,32000,64000,128000,256000,512000" `
    --max-events $MaxEvents `
    --tag-prefix "n179"
  if ($LASTEXITCODE -ne 0) { throw "Failed at DVSCLEAN $($j.Scene) $($j.Level)" }
}

Write-Host "=== DONE: DVSCLEAN n179 sweep ==="

