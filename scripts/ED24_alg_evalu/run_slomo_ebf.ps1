# slomo 实验（label 精确评估版）：仅 EBF（score + threshold）
# 目标：先找 EBF 在 r/tau 下的最佳组合，再用于横向对比。
# 输入：*_signal_only.npy(clean) + *.npy(noisy)
# 输出：data/ED24/myPedestrain_06/EBF/roc_ebf_{light,mid,heavy}.csv/.png

$ErrorActionPreference = "Stop"
$env:PYTHONNOUSERSITE = "1"

$CONDA_HOOK = "D:/software/anaconda3/shell/condabin/conda-hook.ps1"
if (Test-Path $CONDA_HOOK) {
  & $CONDA_HOOK
  conda activate myEVS
}

$PY = "D:/software/Anaconda_envs/envs/myEVS/python.exe"
if (-not (Test-Path $PY)) {
  throw "Python not found: $PY"
}

$ED24_DIR = "D:\hjx_workspace\scientific_reserach\dataset\ED24\myPedestrain_06"
if (-not (Test-Path $ED24_DIR)) {
  throw "Dataset directory not found: $ED24_DIR"
}

$LIGHT_NOISY = Join-Path $ED24_DIR "Pedestrain_06_1.8.npy"
$MID_NOISY   = Join-Path $ED24_DIR "Pedestrain_06_2.5.npy"
$HEAVY_NOISY = Join-Path $ED24_DIR "Pedestrain_06_3.3.npy"

$LIGHT_CLEAN = Join-Path $ED24_DIR "Pedestrain_06_1.8_signal_only.npy"
$MID_CLEAN   = Join-Path $ED24_DIR "Pedestrain_06_2.5_signal_only.npy"
$HEAVY_CLEAN = Join-Path $ED24_DIR "Pedestrain_06_3.3_signal_only.npy"

foreach ($p in @($LIGHT_NOISY,$MID_NOISY,$HEAVY_NOISY,$LIGHT_CLEAN,$MID_CLEAN,$HEAVY_CLEAN)) {
  if (-not (Test-Path $p)) {
    throw "Missing required file: $p"
  }
}

# 1 tick = 1us（与现有 ED24 脚本口径一致）
$TICK_NS = 1000
$MATCH_US = 0
$MATCH_BIN_RADIUS = 0

# =========================== TUNE_HERE: EBF sweep ===========================
$EBF_RADIUS_LIST = 3,4,5
$EBF_TAU_LIST = 32000,64000,128000,256000,512000
$EBF_THR_LIST = "0,2,4,6,8,10,12,14,16,18,20,22,24"

$OUT_DIR = "data/ED24/myPedestrain_06/EBF"
New-Item -ItemType Directory -Force -Path $OUT_DIR | Out-Null

$RUNTIME_CSV = Join-Path $OUT_DIR "runtime_ebf.csv"
"algorithm,level,start_time,end_time,elapsed_sec" | Out-File -FilePath $RUNTIME_CSV -Encoding utf8

function Show-TopAuc {
  param(
    [Parameter(Mandatory = $true)][string]$CsvPath,
    [int]$TopN = 10
  )

  Write-Host "--- Top AUC: $CsvPath ---"
  Import-Csv $CsvPath |
    Group-Object tag |
    ForEach-Object { $_.Group[0] | Select-Object tag, auc } |
    Sort-Object { [double]$_.auc } -Descending |
    Select-Object -First $TopN |
    Format-Table -AutoSize
}

function Get-TopTagsByRadius {
  param(
    [Parameter(Mandatory = $true)][string]$CsvPath,
    [Parameter(Mandatory = $true)][int[]]$RadiusList,
    [int]$TopNPerRadius = 3
  )

  $rows = Import-Csv $CsvPath |
    Group-Object tag |
    ForEach-Object { $_.Group[0] | Select-Object tag, auc }

  $picked = New-Object System.Collections.Generic.List[string]
  foreach ($r in $RadiusList) {
    $prefix = ("ebf_r{0}_" -f [int]$r)
    $top = $rows |
      Where-Object { $_.tag -like "$prefix*" } |
      Sort-Object { [double]$_.auc } -Descending |
      Select-Object -First $TopNPerRadius

    foreach ($t in $top) {
      if ($t.tag) {
        $picked.Add([string]$t.tag)
      }
    }
  }

  return @($picked | Select-Object -Unique)
}

function Export-FilteredCsv {
  param(
    [Parameter(Mandatory = $true)][string]$InCsv,
    [Parameter(Mandatory = $true)][string[]]$Tags,
    [Parameter(Mandatory = $true)][string]$OutCsv
  )

  $set = @{}
  foreach ($t in $Tags) { $set[$t] = $true }

  Import-Csv $InCsv |
    Where-Object { $set.ContainsKey($_.tag) } |
    Export-Csv -NoTypeInformation -Encoding UTF8 $OutCsv
}

function Run-Ebf-For-Noise {
  param(
    [Parameter(Mandatory = $true)][string]$NoiseName,
    [Parameter(Mandatory = $true)][string]$CleanPath,
    [Parameter(Mandatory = $true)][string]$NoisyPath
  )

  Write-Host ("=== EBF slomo: {0} (sweep r/tau) ===" -f $NoiseName)
  $T0 = Get-Date

  $outCsv = Join-Path $OUT_DIR ("roc_ebf_{0}.csv" -f $NoiseName)
  [System.IO.File]::WriteAllText($outCsv, "")

  foreach ($r in $EBF_RADIUS_LIST) {
    foreach ($tau in $EBF_TAU_LIST) {
      & $PY -m myevs.cli roc `
        --clean $CleanPath --noisy $NoisyPath `
        --assume npy --width 346 --height 260 --tick-ns $TICK_NS `
        --method ebf --radius-px $r --time-us $tau `
        --param min-neighbors --values $EBF_THR_LIST `
        --match-us $MATCH_US --match-bin-radius $MATCH_BIN_RADIUS `
        --tag ("ebf_r{0}_tau{1}" -f $r, $tau) `
        --out-csv $outCsv --append --progress
    }
  }

  # 仅保留每个 r 下 AUC 最好的 3 条曲线用于绘图。
  $topTags = Get-TopTagsByRadius -CsvPath $outCsv -RadiusList $EBF_RADIUS_LIST -TopNPerRadius 3
  $outTopCsv = Join-Path $OUT_DIR ("roc_ebf_{0}_top3_per_r.csv" -f $NoiseName)
  Export-FilteredCsv -InCsv $outCsv -Tags $topTags -OutCsv $outTopCsv

  & $PY -m myevs.cli plot-csv `
    --in $outTopCsv `
    --out (Join-Path $OUT_DIR ("roc_ebf_{0}.png" -f $NoiseName)) `
    --x fpr --y tpr --group tag --kind line `
    --xlabel FPR --ylabel TPR `
    --title ("EBF ROC ({0})" -f $NoiseName)

  Show-TopAuc -CsvPath $outCsv -TopN 10

  $T1 = Get-Date
  ('{0},{1},{2},{3},{4}' -f "ebf", $NoiseName, $T0.ToString("s"), $T1.ToString("s"), [Math]::Round((New-TimeSpan -Start $T0 -End $T1).TotalSeconds, 3)) |
    Add-Content -Path $RUNTIME_CSV -Encoding utf8
}

Run-Ebf-For-Noise -NoiseName "light" -CleanPath $LIGHT_CLEAN -NoisyPath $LIGHT_NOISY
Run-Ebf-For-Noise -NoiseName "mid"   -CleanPath $MID_CLEAN   -NoisyPath $MID_NOISY
Run-Ebf-For-Noise -NoiseName "heavy" -CleanPath $HEAVY_CLEAN -NoisyPath $HEAVY_NOISY

Write-Host "=== DONE: EBF slomo sweep r/tau ==="
