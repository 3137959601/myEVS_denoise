$ErrorActionPreference = "Stop"
$env:PYTHONNOUSERSITE = "1"

$CONDA_HOOK = "D:/software/anaconda3/shell/condabin/conda-hook.ps1"
if (Test-Path $CONDA_HOOK) {
  & $CONDA_HOOK
  conda activate myEVS
}

$PY = "D:/software/Anaconda_envs/envs/myEVS/python.exe"
$ED24_DIR = "D:\hjx_workspace\scientific_reserach\dataset\ED24\myPedestrain_06"
$NOISY = Join-Path $ED24_DIR "Pedestrain_06_2.1.npy"
$CLEAN = Join-Path $ED24_DIR "Pedestrain_06_2.1_signal_only.npy"
$LEVEL = "light_mid"
$TICK_NS = 1000
$MATCH_US = 0
$MATCH_BIN_RADIUS = 0

foreach ($p in @($PY, $NOISY, $CLEAN)) {
  if (!(Test-Path $p)) { throw "Missing required path: $p" }
}

function Reset-Csv([string]$Path) {
  $dir = Split-Path -Parent $Path
  if ($dir) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
  [System.IO.File]::WriteAllText($Path, "")
}

function Plot-Csv([string]$InCsv, [string]$OutPng, [string]$Title) {
  & $PY -m myevs.cli plot-csv `
    --in $InCsv `
    --out $OutPng `
    --x fpr --y tpr --group tag --kind line `
    --xlabel FPR --ylabel TPR `
    --title $Title
  if ($LASTEXITCODE -ne 0) { throw "plot-csv failed: $InCsv" }
}

function Add-Runtime([string]$Alg, [datetime]$T0, [datetime]$T1) {
  $runtimeCsv = "data/ED24/myPedestrain_06/{0}/runtime_{1}.csv" -f $Alg.ToUpper(), $Alg
  if (!(Test-Path $runtimeCsv)) {
    "algorithm,level,start_time,end_time,elapsed_sec" | Out-File -FilePath $runtimeCsv -Encoding utf8
  }
  $elapsed = [Math]::Round((New-TimeSpan -Start $T0 -End $T1).TotalSeconds, 3)
  ('{0},{1},{2},{3},{4}' -f $Alg, $LEVEL, $T0.ToString("s"), $T1.ToString("s"), $elapsed) | Add-Content -Path $runtimeCsv -Encoding utf8
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
  foreach ($r in $rows) {
    if ([string]$r.tag -match "_r(\d+)_") { [void]$rSet.Add([int]$matches[1]) }
  }
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

function Run-Roc([string]$Alg, [string]$Tag, [int]$Radius, [int]$Tau, [string]$Values, [string]$OutCsv, [string]$Engine = "python", [string[]]$ExtraArgs = @()) {
  $args = @(
    "-m", "myevs.cli", "roc",
    "--clean", $CLEAN,
    "--noisy", $NOISY,
    "--assume", "npy",
    "--width", "346",
    "--height", "260",
    "--tick-ns", "$TICK_NS",
    "--engine", $Engine,
    "--method", $Alg,
    "--radius-px", "$Radius",
    "--time-us", "$Tau",
    "--param", "min-neighbors",
    "--values", "$Values",
    "--match-us", "$MATCH_US",
    "--match-bin-radius", "$MATCH_BIN_RADIUS",
    "--tag", $Tag,
    "--out-csv", $OutCsv,
    "--append",
    "--progress"
  )
  if ($ExtraArgs.Count -gt 0) { $args += $ExtraArgs }
  & $PY @args
  if ($LASTEXITCODE -ne 0) { throw "myevs roc failed: $Alg $Tag" }
}

Write-Host "=== ED24 2.1v all algorithms sweep: start ==="

# BAF
$t0 = Get-Date
$out = "data/ED24/myPedestrain_06/BAF/roc_baf_${LEVEL}.csv"
Reset-Csv $out
foreach ($r in @(1,2,3,4)) {
  & $PY -m myevs.cli roc --clean $CLEAN --noisy $NOISY --assume npy --width 346 --height 260 --tick-ns $TICK_NS --method baf --radius-px $r --param time-us --values "20,100,200,500,1000,2000,4000,8000,16000,32000,64000,128000" --match-us $MATCH_US --match-bin-radius $MATCH_BIN_RADIUS --tag ("baf_r{0}" -f $r) --out-csv $out --append --progress
}
Plot-Csv $out "data/ED24/myPedestrain_06/BAF/roc_baf_${LEVEL}.png" "BAF ROC (${LEVEL})"
Add-Runtime "baf" $t0 (Get-Date)

# STCF
$t0 = Get-Date
$out = "data/ED24/myPedestrain_06/STCF/roc_stcf_${LEVEL}.csv"
Reset-Csv $out
foreach ($r in @(1,2,3,4)) {
  & $PY -m myevs.cli roc --clean $CLEAN --noisy $NOISY --assume npy --width 346 --height 260 --tick-ns $TICK_NS --method stc --radius-px $r --param time-us --values "100,200,500,1000,2000,4000,8000,16000,32000,64000,128000,256000,512000" --match-us $MATCH_US --match-bin-radius $MATCH_BIN_RADIUS --tag ("stcf_r{0}" -f $r) --out-csv $out --append --progress
}
Plot-Csv $out "data/ED24/myPedestrain_06/STCF/roc_stcf_${LEVEL}.png" "STCF ROC (${LEVEL})"
Add-Runtime "stcf" $t0 (Get-Date)

# EBF
$t0 = Get-Date
$out = "data/ED24/myPedestrain_06/EBF/roc_ebf_${LEVEL}.csv"
Reset-Csv $out
foreach ($r in @(3,4,5)) {
  foreach ($tau in @(32000,64000,128000,256000,512000)) {
    Run-Roc -Alg "ebf" -Tag ("ebf_r{0}_tau{1}" -f $r,$tau) -Radius $r -Tau $tau -Values "0,2,4,6,8,10,12,14,16,18,20,22,24" -OutCsv $out -Engine "numba"
  }
}
$ebfTop = Get-TopTagsByRadius -CsvPath $out -TopNPerRadius 3
$ebfPlotCsv = "data/ED24/myPedestrain_06/EBF/roc_ebf_${LEVEL}_top3_per_r.csv"
Export-FilteredCsv -InCsv $out -Tags $ebfTop -OutCsv $ebfPlotCsv
Plot-Csv $ebfPlotCsv "data/ED24/myPedestrain_06/EBF/roc_ebf_${LEVEL}.png" "EBF ROC (${LEVEL})"
Add-Runtime "ebf" $t0 (Get-Date)

# N149
$t0 = Get-Date
& $PY "scripts/ED24_alg_evalu/run_n149_labelscore_grid.py" `
  --radius-list "3,4,5" `
  --tau-us-list "16000,32000,64000,128000,256000,512000" `
  --out-dir "data/ED24/myPedestrain_06/N149" `
  --light "" --mid "" --heavy "" `
  --light-mid $NOISY
if ($LASTEXITCODE -ne 0) { throw "N149 script failed" }
Add-Runtime "n149" $t0 (Get-Date)

# KNOISE
$t0 = Get-Date
$out = "data/ED24/myPedestrain_06/KNOISE/roc_knoise_${LEVEL}.csv"
Reset-Csv $out
foreach ($tau in @(16000,32000,64000,128000,256000)) {
  Run-Roc -Alg "knoise" -Tag ("knoise_tau{0}" -f $tau) -Radius 1 -Tau $tau -Values "1,2,3" -OutCsv $out
}
Plot-Csv $out "data/ED24/myPedestrain_06/KNOISE/roc_knoise_${LEVEL}.png" "KNOISE ROC (${LEVEL})"
Add-Runtime "knoise" $t0 (Get-Date)

# EVFLOW (dense)
$t0 = Get-Date
$out = "data/ED24/myPedestrain_06/EVFLOW/roc_evflow_${LEVEL}.csv"
Reset-Csv $out
$thr = (0..40 | ForEach-Object { $_ * 2 }) -join ","
foreach ($r in @(2,3,4,5)) {
  foreach ($tau in @(8000,16000,32000,64000)) {
    Run-Roc -Alg "evflow" -Tag ("evflow_r{0}_tau{1}" -f $r,$tau) -Radius $r -Tau $tau -Values $thr -OutCsv $out -Engine "numba"
  }
}
$top = Get-TopTagsByRadius -CsvPath $out -TopNPerRadius 3
$plotCsv = "data/ED24/myPedestrain_06/EVFLOW/roc_evflow_${LEVEL}_top3_per_r.csv"
Export-FilteredCsv -InCsv $out -Tags $top -OutCsv $plotCsv
Plot-Csv $plotCsv "data/ED24/myPedestrain_06/EVFLOW/roc_evflow_${LEVEL}.png" "EVFLOW ROC (${LEVEL})"
Add-Runtime "evflow" $t0 (Get-Date)

# YNOISE
$t0 = Get-Date
$out = "data/ED24/myPedestrain_06/YNOISE/roc_ynoise_${LEVEL}.csv"
Reset-Csv $out
foreach ($r in @(2,3,4,5)) {
  foreach ($tau in @(16000,32000,64000,128000,256000)) {
    Run-Roc -Alg "ynoise" -Tag ("ynoise_r{0}_tau{1}" -f $r,$tau) -Radius $r -Tau $tau -Values "1,2,3,4,6,8" -OutCsv $out
  }
}
$top = Get-TopTagsByRadius -CsvPath $out -TopNPerRadius 3
$plotCsv = "data/ED24/myPedestrain_06/YNOISE/roc_ynoise_${LEVEL}_top3_per_r.csv"
Export-FilteredCsv -InCsv $out -Tags $top -OutCsv $plotCsv
Plot-Csv $plotCsv "data/ED24/myPedestrain_06/YNOISE/roc_ynoise_${LEVEL}.png" "YNOISE ROC (${LEVEL})"
Add-Runtime "ynoise" $t0 (Get-Date)

# TS
$t0 = Get-Date
$out = "data/ED24/myPedestrain_06/TS/roc_ts_${LEVEL}.csv"
Reset-Csv $out
foreach ($r in @(1,2,3,4)) {
  foreach ($tau in @(16000,32000,64000,128000)) {
    Run-Roc -Alg "ts" -Tag ("ts_r{0}_decay{1}" -f $r,$tau) -Radius $r -Tau $tau -Values "0.05,0.1,0.2,0.3,0.5" -OutCsv $out -Engine "numba"
  }
}
$top = Get-TopTagsByRadius -CsvPath $out -TopNPerRadius 3
$plotCsv = "data/ED24/myPedestrain_06/TS/roc_ts_${LEVEL}_top3_per_r.csv"
Export-FilteredCsv -InCsv $out -Tags $top -OutCsv $plotCsv
Plot-Csv $plotCsv "data/ED24/myPedestrain_06/TS/roc_ts_${LEVEL}.png" "TS ROC (${LEVEL})"
Add-Runtime "ts" $t0 (Get-Date)

# MLPF (proxy mode unless model provided elsewhere)
$t0 = Get-Date
$out = "data/ED24/myPedestrain_06/MLPF/roc_mlpf_${LEVEL}.csv"
Reset-Csv $out
foreach ($tau in @(32000,64000,128000,256000,512000)) {
  Run-Roc -Alg "mlpf" -Tag ("mlpf_tau{0}" -f $tau) -Radius 3 -Tau $tau -Values "2,4,6,8,10,12,16,18,20,22,24" -OutCsv $out
}
$top = Get-TopTagsGlobal -CsvPath $out -TopN 4
$plotCsv = "data/ED24/myPedestrain_06/MLPF/roc_mlpf_${LEVEL}_top4_tau.csv"
Export-FilteredCsv -InCsv $out -Tags $top -OutCsv $plotCsv
Plot-Csv $plotCsv "data/ED24/myPedestrain_06/MLPF/roc_mlpf_${LEVEL}.png" "MLPF ROC (${LEVEL})"
Add-Runtime "mlpf" $t0 (Get-Date)

# PFD
$t0 = Get-Date
$out = "data/ED24/myPedestrain_06/PFD/roc_pfd_${LEVEL}.csv"
Reset-Csv $out
foreach ($m in @(1,2,3)) {
  foreach ($tau in @(8000,16000,32000,64000,128000,256000)) {
    Run-Roc -Alg "pfd" -Tag ("pfd_r3_tau{0}_m{1}" -f $tau,$m) -Radius 3 -Tau $tau -Values "1,2,3,4,5,6,7,8" -OutCsv $out -Engine "numba" -ExtraArgs @("--refractory-us", "$m", "--pfd-mode", "a")
  }
}
$top = Get-TopTagsByRadius -CsvPath $out -TopNPerRadius 3
$plotCsv = "data/ED24/myPedestrain_06/PFD/roc_pfd_${LEVEL}_top3_per_r.csv"
Export-FilteredCsv -InCsv $out -Tags $top -OutCsv $plotCsv
Plot-Csv $plotCsv "data/ED24/myPedestrain_06/PFD/roc_pfd_${LEVEL}.png" "PFD ROC (${LEVEL})"
Add-Runtime "pfd" $t0 (Get-Date)

Write-Host "=== ED24 2.1v all algorithms sweep: DONE ==="
