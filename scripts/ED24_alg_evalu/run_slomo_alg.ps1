param(
  [Parameter(Mandatory = $true)]
  [ValidateSet("knoise", "evflow", "ynoise", "ts", "mlpf")]
  [string]$Algorithm
)

$ErrorActionPreference = "Stop"
$env:PYTHONNOUSERSITE = "1"

$CONDA_HOOK = "D:/software/anaconda3/shell/condabin/conda-hook.ps1"
if (Test-Path $CONDA_HOOK) {
  & $CONDA_HOOK
  conda activate myEVS
}

$PY = "D:/software/Anaconda_envs/envs/myEVS/python.exe"
$ED24_DIR = "D:\hjx_workspace\scientific_reserach\dataset\ED24\myPedestrain_06"
$TICK_NS = 1000
$MATCH_US = 0
$MATCH_BIN_RADIUS = 0

$SPLITS = @(
  @{ Name = "light"; Noisy = "Pedestrain_06_1.8.npy"; Clean = "Pedestrain_06_1.8_signal_only.npy" },
  @{ Name = "mid"; Noisy = "Pedestrain_06_2.5.npy"; Clean = "Pedestrain_06_2.5_signal_only.npy" },
  @{ Name = "heavy"; Noisy = "Pedestrain_06_3.3.npy"; Clean = "Pedestrain_06_3.3_signal_only.npy" }
)

foreach ($sp in $SPLITS) {
  foreach ($n in @($sp.Noisy, $sp.Clean)) {
    $p = Join-Path $ED24_DIR $n
    if (!(Test-Path $p)) {
      throw "Missing required file: $p"
    }
  }
}

function Run-Roc([string]$Clean, [string]$Noisy, [string]$OutCsv, [string]$Tag, [string]$Method, [int]$Radius, [int]$TimeUs, [string]$Values) {
  & $PY -m myevs.cli roc `
    --clean $Clean --noisy $Noisy `
    --assume npy --width 346 --height 260 `
    --tick-ns $TICK_NS `
    --method $Method --radius-px $Radius --time-us $TimeUs `
    --param min-neighbors --values $Values `
    --match-us $MATCH_US --match-bin-radius $MATCH_BIN_RADIUS `
    --tag $Tag --out-csv $OutCsv --append --progress
}

foreach ($sp in $SPLITS) {
  $clean = Join-Path $ED24_DIR $sp.Clean
  $noisy = Join-Path $ED24_DIR $sp.Noisy
  $outDir = "data/ED24/myPedestrain_06/{0}" -f $Algorithm.ToUpper()
  New-Item -ItemType Directory -Force -Path $outDir | Out-Null
  $outCsv = Join-Path $outDir ("roc_{0}_{1}.csv" -f $Algorithm, $sp.Name)
  [System.IO.File]::WriteAllText($outCsv, "")

  Write-Host ("=== {0} ED24 {1} ===" -f $Algorithm, $sp.Name)

  switch ($Algorithm) {
    "knoise" {
      $thr = "0,1,2,3,4"
      foreach ($tau in @(500, 1000, 2000, 4000, 8000)) {
        Run-Roc -Clean $clean -Noisy $noisy -OutCsv $outCsv -Tag ("knoise_tau{0}" -f $tau) -Method "knoise" -Radius 1 -TimeUs $tau -Values $thr
      }
    }
    "evflow" {
      $thr = "1,1.5,2,3,4,6,8,12"
      foreach ($r in @(1, 2)) {
        foreach ($tau in @(1000, 3000, 6000)) {
          Run-Roc -Clean $clean -Noisy $noisy -OutCsv $outCsv -Tag ("evflow_r{0}_tau{1}" -f $r, $tau) -Method "evflow" -Radius $r -TimeUs $tau -Values $thr
        }
      }
    }
    "ynoise" {
      $thr = "1,2,3,4,5,6,8,10"
      foreach ($r in @(1, 2, 3)) {
        foreach ($tau in @(1000, 2000, 4000, 8000, 16000)) {
          Run-Roc -Clean $clean -Noisy $noisy -OutCsv $outCsv -Tag ("ynoise_r{0}_tau{1}" -f $r, $tau) -Method "ynoise" -Radius $r -TimeUs $tau -Values $thr
        }
      }
    }
    "ts" {
      $thr = "0.05,0.1,0.2,0.3,0.4,0.5,0.7,0.9"
      foreach ($r in @(1, 2, 3)) {
        foreach ($tau in @(5000, 10000, 20000, 30000, 60000)) {
          Run-Roc -Clean $clean -Noisy $noisy -OutCsv $outCsv -Tag ("ts_r{0}_decay{1}" -f $r, $tau) -Method "ts" -Radius $r -TimeUs $tau -Values $thr
        }
      }
    }
    "mlpf" {
      $thr = "2,4,6,8,10,12,14,16,20"
      foreach ($tau in @(20000, 50000, 100000, 200000)) {
        Run-Roc -Clean $clean -Noisy $noisy -OutCsv $outCsv -Tag ("mlpf_tau{0}" -f $tau) -Method "mlpf" -Radius 3 -TimeUs $tau -Values $thr
      }
    }
  }

  & $PY -m myevs.cli plot-csv `
    --in $outCsv `
    --out (Join-Path $outDir ("roc_{0}_{1}.png" -f $Algorithm, $sp.Name)) `
    --x fpr --y tpr --group tag --kind line `
    --xlabel FPR --ylabel TPR `
    --title ("{0} ROC ({1})" -f $Algorithm.ToUpper(), $sp.Name)
}

Write-Host ("=== DONE: ED24 {0} ===" -f $Algorithm)

