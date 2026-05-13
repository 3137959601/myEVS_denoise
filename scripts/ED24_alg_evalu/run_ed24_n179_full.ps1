param(
  [string]$PythonExe = "D:/software/Anaconda_envs/envs/myEVS/python.exe",
  [int]$MaxEvents = 0
)

$ErrorActionPreference = "Stop"

$jobs = @(
  @{ Scene = "myPedestrain_06"; Level = "light"; Noisy = "D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_1.8.npy" },
  @{ Scene = "myPedestrain_06"; Level = "light_mid"; Noisy = "D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_2.1.npy" },
  @{ Scene = "myPedestrain_06"; Level = "mid"; Noisy = "D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_2.5.npy" },
  @{ Scene = "myPedestrain_06"; Level = "heavy"; Noisy = "D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3.npy" },
  @{ Scene = "myBicycle_02"; Level = "light"; Noisy = "D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_1.8.npy" },
  @{ Scene = "myBicycle_02"; Level = "light_mid"; Noisy = "D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_2.1.npy" },
  @{ Scene = "myBicycle_02"; Level = "mid"; Noisy = "D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_2.5.npy" }
)

for ($i = 0; $i -lt $jobs.Count; $i++) {
  $j = $jobs[$i]
  $noisy = $j.Noisy
  if (!(Test-Path $noisy)) { throw "Missing file: $noisy" }
  $outDir = "data/ED24/{0}/N179" -f $j.Scene
  New-Item -ItemType Directory -Force -Path $outDir | Out-Null
  $rocCsv = Join-Path $outDir ("roc_n179_{0}.csv" -f $j.Level)
  $sumCsv = Join-Path $outDir ("summary_n179_{0}.csv" -f $j.Level)
  $runCsv = Join-Path $outDir ("runtime_n179_{0}.csv" -f $j.Level)

  Write-Host ("[ED24][{0}/{1}] {2} {3}" -f ($i + 1), $jobs.Count, $j.Scene, $j.Level)
  & $PythonExe scripts/noise_analyze/sweep_n179_pair.py `
    --noisy $noisy `
    --out-csv $rocCsv `
    --summary-csv $sumCsv `
    --runtime-csv $runCsv `
    --dataset "ED24" `
    --scene $j.Scene `
    --level $j.Level `
    --width 346 `
    --height 260 `
    --tick-ns 1000 `
    --s-list "5,7,9" `
    --tau-us-list "16000,32000,64000,128000,256000,512000" `
    --max-events $MaxEvents `
    --tag-prefix "n179"
  if ($LASTEXITCODE -ne 0) { throw "Failed at $($j.Scene) $($j.Level)" }
}

Write-Host "=== DONE: ED24 n179 sweep ==="

Write-Host "=== POST: compute N179 MESR/AOCC bestpoints (ED24 pedestrian) ==="
& $PythonExe scripts/eval_bestpoint_mesr_aocc.py `
  --dataset ed24_ped `
  --scene pedestrian `
  --algorithms n179 `
  --levels light,light_mid,mid,heavy `
  --metrics mesr,aocc `
  --points best-auc,best-f1 `
  --out-csv data/ED24/myPedestrain_06/bestpoint_mesr_aocc_summary.csv
if ($LASTEXITCODE -ne 0) { throw "Failed MESR/AOCC post step (ED24 pedestrian)" }

Write-Host "=== POST: compute N179 MESR/AOCC bestpoints (ED24 bicycle) ==="
& $PythonExe scripts/eval_bestpoint_mesr_aocc.py `
  --dataset ed24_bicycle `
  --scene bicycle `
  --algorithms n179 `
  --levels light,light_mid,mid `
  --metrics mesr,aocc `
  --points best-auc,best-f1 `
  --out-csv data/ED24/myBicycle_02/bestpoint_mesr_aocc_summary.csv
if ($LASTEXITCODE -ne 0) { throw "Failed MESR/AOCC post step (ED24 bicycle)" }
