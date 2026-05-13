param(
  [switch]$SkipTrainMlpf = $false,
  [switch]$EvflowLite = $true
)

$ErrorActionPreference = "Stop"
$env:PYTHONNOUSERSITE = "1"

$CONDA_HOOK = "D:/software/anaconda3/shell/condabin/conda-hook.ps1"
if (-not [string]::IsNullOrWhiteSpace($CONDA_HOOK) -and (Test-Path $CONDA_HOOK)) {
  & $CONDA_HOOK
  conda activate myEVS
}

$PY = "D:/software/Anaconda_envs/envs/myEVS/python.exe"
$ED24_DIR = "D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02"
$OUT_ROOT = "data/ED24/myBicycle_02"
$MODEL_ROOT = "data/ED24/myBicycle_02/MLPF/models"
$TICK_NS = 1000
$MATCH_US = 0
$MATCH_BIN_RADIUS = 0

$SPLITS = @(
  @{ Name = "light";     Noisy = "Bicycle_02_1.8.npy"; Clean = "Bicycle_02_1.8_signal_only.npy" },
  @{ Name = "light_mid"; Noisy = "Bicycle_02_2.1.npy"; Clean = "Bicycle_02_2.1_signal_only.npy" },
  @{ Name = "mid";       Noisy = "Bicycle_02_2.5.npy"; Clean = "Bicycle_02_2.5_signal_only.npy" }
)

$ALG_ORDER = @("baf","stcf","ebf","n149","knoise","evflow","ynoise","ts","mlpf","pfd")
$TOTAL_TASKS = $SPLITS.Count * $ALG_ORDER.Count
$script:STEP = 0

foreach ($sp in $SPLITS) {
  foreach ($n in @($sp.Noisy, $sp.Clean)) {
    $p = Join-Path $ED24_DIR $n
    if (!(Test-Path $p)) { throw "Missing required file: $p" }
  }
}

function Write-Stage([string]$Level, [string]$Alg, [string]$Phase = "run") {
  $script:STEP += 1
  $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
  Write-Host ("[{0}] [{1}/{2}] [dataset=myBicycle_02 level={3}] [alg={4}] [phase={5}]" -f $ts, $script:STEP, $TOTAL_TASKS, $Level, $Alg, $Phase)
}

function Reset-Csv([string]$Path) {
  $dir = Split-Path -Parent $Path
  if ($dir) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
  [System.IO.File]::WriteAllText($Path, "")
}

function Plot-Csv([string]$InCsv, [string]$OutPng, [string]$Title) {
  & $PY -m myevs.cli plot-csv --in $InCsv --out $OutPng --x fpr --y tpr --group tag --kind line --xlabel FPR --ylabel TPR --title $Title
  if ($LASTEXITCODE -ne 0) { throw "plot-csv failed: $InCsv" }
}

function Add-Runtime([string]$Alg, [string]$Level, [datetime]$T0, [datetime]$T1) {
  $runtimeCsv = Join-Path (Join-Path $OUT_ROOT $Alg.ToUpper()) ("runtime_{0}.csv" -f $Alg)
  if (!(Test-Path $runtimeCsv)) {
    "algorithm,level,start_time,end_time,elapsed_sec" | Out-File -FilePath $runtimeCsv -Encoding utf8
  }
  $elapsed = [Math]::Round((New-TimeSpan -Start $T0 -End $T1).TotalSeconds, 3)
  ('{0},{1},{2},{3},{4}' -f $Alg, $Level, $T0.ToString("s"), $T1.ToString("s"), $elapsed) | Add-Content -Path $runtimeCsv -Encoding utf8
}

function Export-FilteredCsv([string]$InCsv, [string[]]$Tags, [string]$OutCsv) {
  if (!$Tags -or $Tags.Count -eq 0) {
    Copy-Item -LiteralPath $InCsv -Destination $OutCsv -Force
    return
  }
  $set = @{}
  foreach ($t in $Tags) { $set[$t] = $true }
  Import-Csv $InCsv | Where-Object { $set.ContainsKey($_.tag) } | Export-Csv -NoTypeInformation -Encoding UTF8 $OutCsv
}

function Get-TopTagsByRadius([string]$CsvPath, [int]$TopNPerRadius = 3) {
  $rows = Import-Csv $CsvPath | Group-Object tag | ForEach-Object { $_.Group[0] | Select-Object tag, auc }
  $rSet = New-Object System.Collections.Generic.HashSet[int]
  foreach ($r in $rows) { if ([string]$r.tag -match "_r(\d+)_") { [void]$rSet.Add([int]$matches[1]) } }
  $picked = New-Object System.Collections.Generic.List[string]
  foreach ($rv in @($rSet | Sort-Object)) {
    $prefix = "_r{0}_" -f [int]$rv
    $top = $rows | Where-Object { ([string]$_.tag) -like "*$prefix*" } | Sort-Object { [double]$_.auc } -Descending | Select-Object -First $TopNPerRadius
    foreach ($t in $top) { if ($t.tag) { $picked.Add([string]$t.tag) } }
  }
  return @($picked | Select-Object -Unique)
}

function Get-TopTagsGlobal([string]$CsvPath, [int]$TopN = 4) {
  return @(
    Import-Csv $CsvPath |
      Group-Object tag |
      ForEach-Object { $_.Group[0] | Select-Object tag, auc } |
      Sort-Object { [double]$_.auc } -Descending |
      Select-Object -First $TopN |
      ForEach-Object { [string]$_.tag }
  )
}

function Run-Roc([string]$Clean,[string]$Noisy,[string]$OutCsv,[string]$Tag,[string]$Method,[int]$Radius,[int]$TimeUs,[string]$SweepValues,[string]$Engine="python",[string[]]$ExtraArgs=@()) {
  if ([string]::IsNullOrWhiteSpace($SweepValues)) { throw "Run-Roc empty values: $Tag" }
  $args = @(
    "-m","myevs.cli","roc",
    "--clean",$Clean,"--noisy",$Noisy,
    "--assume","npy","--width","346","--height","260",
    "--tick-ns","$TICK_NS","--engine",$Engine,
    "--method",$Method,"--radius-px","$Radius","--time-us","$TimeUs",
    "--param","min-neighbors","--values","$SweepValues",
    "--match-us","$MATCH_US","--match-bin-radius","$MATCH_BIN_RADIUS",
    "--tag",$Tag,"--out-csv",$OutCsv,"--append","--progress"
  )
  if ($ExtraArgs.Count -gt 0) { $args += $ExtraArgs }
  & $PY @args
  if ($LASTEXITCODE -ne 0) { throw "myevs roc failed: $Method $Tag" }
}

function Train-Mlpf([string]$Level,[string]$Clean,[string]$Noisy,[string]$ModelPath,[string]$MetaPath) {
  Write-Host ("[MLPF][{0}] training model -> {1}" -f $Level, $ModelPath)
  & $PY scripts/train_mlpf_torch.py `
    --clean $Clean `
    --noisy $Noisy `
    --width 346 --height 260 --tick-ns $TICK_NS `
    --duration-us 100000 `
    --patch 5 `
    --epochs 4 `
    --batch-size 512 `
    --max-events 0 `
    --out-ts $ModelPath `
    --out-meta $MetaPath
  if ($LASTEXITCODE -ne 0) { throw "MLPF training failed for $Level" }
}

Write-Host "=== myBicycle_02 ED24 full sweep: START ==="
Write-Host ("EvflowLite={0}, SkipTrainMlpf={1}" -f $EvflowLite, $SkipTrainMlpf)

foreach ($sp in $SPLITS) {
  $level = $sp.Name
  $noisy = Join-Path $ED24_DIR $sp.Noisy
  $clean = Join-Path $ED24_DIR $sp.Clean
  Write-Host ("=== DATASET myBicycle_02 / LEVEL {0}: START ===" -f $level)

  $mlpfModel = Join-Path $MODEL_ROOT ("mlpf_torch_{0}.pt" -f $level)
  $mlpfMeta  = Join-Path $MODEL_ROOT ("mlpf_torch_{0}.json" -f $level)
  if (-not $SkipTrainMlpf) {
    Write-Stage -Level $level -Alg "mlpf" -Phase "train"
    Train-Mlpf -Level $level -Clean $clean -Noisy $noisy -ModelPath $mlpfModel -MetaPath $mlpfMeta
  } elseif (!(Test-Path $mlpfModel)) {
    throw "SkipTrainMlpf is set but model missing: $mlpfModel"
  }

  # BAF
  Write-Stage -Level $level -Alg "baf"
  $t0 = Get-Date
  $out = Join-Path (Join-Path $OUT_ROOT "BAF") ("roc_baf_{0}.csv" -f $level)
  Reset-Csv $out
  foreach ($r in @(1,2,3,4)) {
    & $PY -m myevs.cli roc --clean $clean --noisy $noisy --assume npy --width 346 --height 260 --tick-ns $TICK_NS --method baf --radius-px $r --param time-us --values "20,100,200,500,1000,2000,4000,8000,16000,32000,64000,128000" --match-us $MATCH_US --match-bin-radius $MATCH_BIN_RADIUS --tag ("baf_r{0}" -f $r) --out-csv $out --append --progress
  }
  Plot-Csv $out (Join-Path (Join-Path $OUT_ROOT "BAF") ("roc_baf_{0}.png" -f $level)) ("BAF ROC ({0})" -f $level)
  Add-Runtime "baf" $level $t0 (Get-Date)

  # STCF
  Write-Stage -Level $level -Alg "stcf"
  $t0 = Get-Date
  $out = Join-Path (Join-Path $OUT_ROOT "STCF") ("roc_stcf_{0}.csv" -f $level)
  Reset-Csv $out
  foreach ($r in @(1,2,3,4)) {
    & $PY -m myevs.cli roc --clean $clean --noisy $noisy --assume npy --width 346 --height 260 --tick-ns $TICK_NS --method stc --radius-px $r --param time-us --values "100,200,500,1000,2000,4000,8000,16000,32000,64000,128000,256000,512000" --match-us $MATCH_US --match-bin-radius $MATCH_BIN_RADIUS --tag ("stcf_r{0}" -f $r) --out-csv $out --append --progress
  }
  Plot-Csv $out (Join-Path (Join-Path $OUT_ROOT "STCF") ("roc_stcf_{0}.png" -f $level)) ("STCF ROC ({0})" -f $level)
  Add-Runtime "stcf" $level $t0 (Get-Date)

  # EBF
  Write-Stage -Level $level -Alg "ebf"
  $t0 = Get-Date
  $out = Join-Path (Join-Path $OUT_ROOT "EBF") ("roc_ebf_{0}.csv" -f $level)
  Reset-Csv $out
  foreach ($r in @(3,4,5)) {
    foreach ($tau in @(32000,64000,128000,256000,512000)) {
      Run-Roc -Clean $clean -Noisy $noisy -OutCsv $out -Tag ("ebf_r{0}_tau{1}" -f $r,$tau) -Method "ebf" -Radius $r -TimeUs $tau -SweepValues "0,2,4,6,8,10,12,14,16,18,20,22,24" -Engine "python"
    }
  }
  $top = Get-TopTagsByRadius -CsvPath $out -TopNPerRadius 3
  $plotCsv = Join-Path (Join-Path $OUT_ROOT "EBF") ("roc_ebf_{0}_top3_per_r.csv" -f $level)
  Export-FilteredCsv -InCsv $out -Tags $top -OutCsv $plotCsv
  Plot-Csv $plotCsv (Join-Path (Join-Path $OUT_ROOT "EBF") ("roc_ebf_{0}.png" -f $level)) ("EBF ROC ({0})" -f $level)
  Add-Runtime "ebf" $level $t0 (Get-Date)

  # N149
  Write-Stage -Level $level -Alg "n149"
  $t0 = Get-Date
  $n149Out = Join-Path $OUT_ROOT "N149"
  if ($level -eq "light") {
    & $PY scripts/ED24_alg_evalu/run_n149_labelscore_grid.py --radius-list "3,4,5" --tau-us-list "16000,32000,64000,128000,256000,512000" --out-dir $n149Out --light $noisy --mid " " --heavy " "
  } elseif ($level -eq "light_mid") {
    & $PY scripts/ED24_alg_evalu/run_n149_labelscore_grid.py --radius-list "3,4,5" --tau-us-list "16000,32000,64000,128000,256000,512000" --out-dir $n149Out --light " " --light-mid $noisy --mid " " --heavy " "
  } else {
    & $PY scripts/ED24_alg_evalu/run_n149_labelscore_grid.py --radius-list "3,4,5" --tau-us-list "16000,32000,64000,128000,256000,512000" --out-dir $n149Out --light " " --mid $noisy --heavy " "
  }
  if ($LASTEXITCODE -ne 0) { throw "N149 failed for $level" }
  Add-Runtime "n149" $level $t0 (Get-Date)

  # KNOISE
  Write-Stage -Level $level -Alg "knoise"
  $t0 = Get-Date
  $out = Join-Path (Join-Path $OUT_ROOT "KNOISE") ("roc_knoise_{0}.csv" -f $level)
  Reset-Csv $out
  foreach ($tau in @(16000,32000,64000,128000,256000)) {
    Run-Roc -Clean $clean -Noisy $noisy -OutCsv $out -Tag ("knoise_tau{0}" -f $tau) -Method "knoise" -Radius 1 -TimeUs $tau -SweepValues "1,2,3"
  }
  Plot-Csv $out (Join-Path (Join-Path $OUT_ROOT "KNOISE") ("roc_knoise_{0}.png" -f $level)) ("KNOISE ROC ({0})" -f $level)
  Add-Runtime "knoise" $level $t0 (Get-Date)

  # EVFLOW
  Write-Stage -Level $level -Alg "evflow"
  $t0 = Get-Date
  $out = Join-Path (Join-Path $OUT_ROOT "EVFLOW") ("roc_evflow_{0}.csv" -f $level)
  Reset-Csv $out
  if ($EvflowLite) {
    $thr = "0,10,20,30,40,50,60,70,80"
    foreach ($r in @(3,4)) {
      foreach ($tau in @(16000,32000)) {
        Run-Roc -Clean $clean -Noisy $noisy -OutCsv $out -Tag ("evflow_r{0}_tau{1}" -f $r,$tau) -Method "evflow" -Radius $r -TimeUs $tau -SweepValues $thr -Engine "numba"
      }
    }
  } else {
    $thr = (0..40 | ForEach-Object { $_ * 2 }) -join ","
    foreach ($r in @(2,3,4,5)) {
      foreach ($tau in @(8000,16000,32000,64000)) {
        Run-Roc -Clean $clean -Noisy $noisy -OutCsv $out -Tag ("evflow_r{0}_tau{1}" -f $r,$tau) -Method "evflow" -Radius $r -TimeUs $tau -SweepValues $thr -Engine "numba"
      }
    }
  }
  $top = Get-TopTagsByRadius -CsvPath $out -TopNPerRadius 3
  $plotCsv = Join-Path (Join-Path $OUT_ROOT "EVFLOW") ("roc_evflow_{0}_top3_per_r.csv" -f $level)
  Export-FilteredCsv -InCsv $out -Tags $top -OutCsv $plotCsv
  Plot-Csv $plotCsv (Join-Path (Join-Path $OUT_ROOT "EVFLOW") ("roc_evflow_{0}.png" -f $level)) ("EVFLOW ROC ({0})" -f $level)
  Add-Runtime "evflow" $level $t0 (Get-Date)

  # YNOISE
  Write-Stage -Level $level -Alg "ynoise"
  $t0 = Get-Date
  $out = Join-Path (Join-Path $OUT_ROOT "YNOISE") ("roc_ynoise_{0}.csv" -f $level)
  Reset-Csv $out
  foreach ($r in @(2,3,4,5)) {
    foreach ($tau in @(16000,32000,64000,128000,256000)) {
      Run-Roc -Clean $clean -Noisy $noisy -OutCsv $out -Tag ("ynoise_r{0}_tau{1}" -f $r,$tau) -Method "ynoise" -Radius $r -TimeUs $tau -SweepValues "1,2,3,4,6,8"
    }
  }
  $top = Get-TopTagsByRadius -CsvPath $out -TopNPerRadius 3
  $plotCsv = Join-Path (Join-Path $OUT_ROOT "YNOISE") ("roc_ynoise_{0}_top3_per_r.csv" -f $level)
  Export-FilteredCsv -InCsv $out -Tags $top -OutCsv $plotCsv
  Plot-Csv $plotCsv (Join-Path (Join-Path $OUT_ROOT "YNOISE") ("roc_ynoise_{0}.png" -f $level)) ("YNOISE ROC ({0})" -f $level)
  Add-Runtime "ynoise" $level $t0 (Get-Date)

  # TS
  Write-Stage -Level $level -Alg "ts"
  $t0 = Get-Date
  $out = Join-Path (Join-Path $OUT_ROOT "TS") ("roc_ts_{0}.csv" -f $level)
  Reset-Csv $out
  foreach ($r in @(1,2,3,4)) {
    foreach ($tau in @(16000,32000,64000,128000)) {
      Run-Roc -Clean $clean -Noisy $noisy -OutCsv $out -Tag ("ts_r{0}_decay{1}" -f $r,$tau) -Method "ts" -Radius $r -TimeUs $tau -SweepValues "0.05,0.1,0.2,0.3,0.5" -Engine "numba"
    }
  }
  $top = Get-TopTagsByRadius -CsvPath $out -TopNPerRadius 3
  $plotCsv = Join-Path (Join-Path $OUT_ROOT "TS") ("roc_ts_{0}_top3_per_r.csv" -f $level)
  Export-FilteredCsv -InCsv $out -Tags $top -OutCsv $plotCsv
  Plot-Csv $plotCsv (Join-Path (Join-Path $OUT_ROOT "TS") ("roc_ts_{0}.png" -f $level)) ("TS ROC ({0})" -f $level)
  Add-Runtime "ts" $level $t0 (Get-Date)

  # MLPF
  Write-Stage -Level $level -Alg "mlpf"
  $t0 = Get-Date
  $out = Join-Path (Join-Path $OUT_ROOT "MLPF") ("roc_mlpf_{0}.csv" -f $level)
  Reset-Csv $out
  foreach ($tau in @(32000,64000,128000,256000,512000)) {
    Run-Roc -Clean $clean -Noisy $noisy -OutCsv $out -Tag ("mlpf_tau{0}" -f $tau) -Method "mlpf" -Radius 2 -TimeUs $tau -SweepValues "0.2,0.4,0.6,0.8" -ExtraArgs @("--mlpf-model", $mlpfModel, "--mlpf-patch", "5")
  }
  $top = Get-TopTagsGlobal -CsvPath $out -TopN 4
  $plotCsv = Join-Path (Join-Path $OUT_ROOT "MLPF") ("roc_mlpf_{0}_top4_tau.csv" -f $level)
  Export-FilteredCsv -InCsv $out -Tags $top -OutCsv $plotCsv
  Plot-Csv $plotCsv (Join-Path (Join-Path $OUT_ROOT "MLPF") ("roc_mlpf_{0}.png" -f $level)) ("MLPF ROC ({0})" -f $level)
  Add-Runtime "mlpf" $level $t0 (Get-Date)

  # PFD
  Write-Stage -Level $level -Alg "pfd"
  $t0 = Get-Date
  $out = Join-Path (Join-Path $OUT_ROOT "PFD") ("roc_pfd_{0}.csv" -f $level)
  Reset-Csv $out
  foreach ($m in @(1,2,3)) {
    foreach ($tau in @(8000,16000,32000,64000,128000,256000)) {
      Run-Roc -Clean $clean -Noisy $noisy -OutCsv $out -Tag ("pfd_r3_tau{0}_m{1}" -f $tau,$m) -Method "pfd" -Radius 3 -TimeUs $tau -SweepValues "1,2,3,4,5,6,7,8" -Engine "numba" -ExtraArgs @("--refractory-us","$m","--pfd-mode","a")
    }
  }
  $top = Get-TopTagsByRadius -CsvPath $out -TopNPerRadius 3
  $plotCsv = Join-Path (Join-Path $OUT_ROOT "PFD") ("roc_pfd_{0}_top3_per_r.csv" -f $level)
  Export-FilteredCsv -InCsv $out -Tags $top -OutCsv $plotCsv
  Plot-Csv $plotCsv (Join-Path (Join-Path $OUT_ROOT "PFD") ("roc_pfd_{0}.png" -f $level)) ("PFD ROC ({0})" -f $level)
  Add-Runtime "pfd" $level $t0 (Get-Date)

  Write-Host ("=== DATASET myBicycle_02 / LEVEL {0}: DONE ===" -f $level)
}

Write-Host "=== myBicycle_02 ED24 full sweep: DONE ==="
