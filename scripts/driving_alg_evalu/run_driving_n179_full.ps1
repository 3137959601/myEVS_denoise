param(
  [string]$PythonExe = "D:/software/Anaconda_envs/envs/myEVS/python.exe",
  [int]$MaxEvents = 0,
  [string]$DatasetRoot = "D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving"
)

$ErrorActionPreference = "Stop"

$jobs = @(
  @{ Level = "light"; Noisy = (Join-Path $DatasetRoot "driving_noise_light_slomo_shot_withlabel/driving_noise_light_labeled.npy") },
  @{ Level = "light_mid"; Noisy = (Join-Path $DatasetRoot "driving_noise_light_mid_slomo_shot_withlabel/driving_noise_light_mid_labeled.npy") },
  @{ Level = "mid"; Noisy = (Join-Path $DatasetRoot "driving_noise_mid_slomo_shot_withlabel/driving_noise_mid_labeled.npy") }
)

for ($i = 0; $i -lt $jobs.Count; $i++) {
  $j = $jobs[$i]
  if (!(Test-Path $j.Noisy)) { throw "Missing file: $($j.Noisy)" }
  $outDir = "data/DND21/mydriving/N179"
  New-Item -ItemType Directory -Force -Path $outDir | Out-Null
  $rocCsv = Join-Path $outDir ("roc_n179_{0}.csv" -f $j.Level)
  $sumCsv = Join-Path $outDir ("summary_n179_{0}.csv" -f $j.Level)
  $runCsv = Join-Path $outDir ("runtime_n179_{0}.csv" -f $j.Level)

  Write-Host ("[Driving][{0}/{1}] {2}" -f ($i + 1), $jobs.Count, $j.Level)
  & $PythonExe scripts/noise_analyze/sweep_n179_pair.py `
    --noisy $j.Noisy `
    --out-csv $rocCsv `
    --summary-csv $sumCsv `
    --runtime-csv $runCsv `
    --dataset "DND21" `
    --scene "mydriving" `
    --level $j.Level `
    --width 346 `
    --height 260 `
    --tick-ns 1000 `
    --s-list "5,7,9" `
    --tau-us-list "16000,32000,64000,128000,256000,512000" `
    --max-events $MaxEvents `
    --tag-prefix "n179"
  if ($LASTEXITCODE -ne 0) { throw "Failed at driving $($j.Level)" }
}

Write-Host "=== DONE: driving n179 sweep ==="

Write-Host "=== POST: compute N179 MESR/AOCC bestpoints (Driving) ==="
& $PythonExe scripts/eval_bestpoint_mesr_aocc.py `
  --dataset driving `
  --algorithms n179 `
  --levels light,light_mid,mid `
  --metrics mesr,aocc `
  --points best-auc,best-f1 `
  --out-csv data/DND21/mydriving/bestpoint_mesr_aocc_summary.csv
if ($LASTEXITCODE -ne 0) { throw "Failed MESR/AOCC post step (Driving)" }
