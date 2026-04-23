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
$TICK_NS = 1000
$MATCH_US = 0
$MATCH_BIN_RADIUS = 0
$WIDTH = 346
$HEIGHT = 260

$ROOT = "D:\hjx_workspace\scientific_reserach\dataset\DND21\mydriving"
$LEVELS = @(
  @{ Name = "light"; Dir = Join-Path $ROOT "driving_noise_light_slomo_shot_withlabel" },
  @{ Name = "mid";   Dir = Join-Path $ROOT "driving_noise_mid_slomo_shot_withlabel" },
  @{ Name = "heavy"; Dir = Join-Path $ROOT "driving_noise_heavy_slomo_shot_withlabel" }
)

function Resolve-Pair([string]$Dir) {
  if (!(Test-Path $Dir)) {
    throw "Missing dataset dir: $Dir"
  }

  $all = Get-ChildItem -Path $Dir -Filter *.npy | Select-Object -ExpandProperty FullName
  if (!$all -or $all.Count -eq 0) {
    throw "No .npy found in: $Dir"
  }

  $clean = $all | Where-Object { $_ -match "signal_only|clean" } | Select-Object -First 1
  if (!$clean) {
    throw "Cannot find clean npy (signal_only/clean) in: $Dir"
  }

  $noisy = $all | Where-Object { $_ -ne $clean } | Where-Object { $_ -notmatch "label" } | Select-Object -First 1
  if (!$noisy) {
    $noisy = $all | Where-Object { $_ -ne $clean } | Select-Object -First 1
  }
  if (!$noisy) {
    throw "Cannot find noisy npy in: $Dir"
  }

  return @{ Clean = $clean; Noisy = $noisy }
}

function Run-Roc([string]$Clean, [string]$Noisy, [string]$OutCsv, [string]$Tag, [string]$Method, [int]$Radius, [int]$TimeUs, [string]$Values) {
  & $PY -m myevs.cli roc `
    --clean $Clean --noisy $Noisy `
    --assume npy --width $WIDTH --height $HEIGHT `
    --tick-ns $TICK_NS `
    --method $Method --radius-px $Radius --time-us $TimeUs `
    --param min-neighbors --values $Values `
    --match-us $MATCH_US --match-bin-radius $MATCH_BIN_RADIUS `
    --tag $Tag --out-csv $OutCsv --append --progress
}

foreach ($lv in $LEVELS) {
  $pair = Resolve-Pair -Dir $lv.Dir
  $clean = $pair.Clean
  $noisy = $pair.Noisy

  $outDir = "data/DND21/mydriving/{0}/{1}" -f $lv.Name, $Algorithm.ToUpper()
  New-Item -ItemType Directory -Force -Path $outDir | Out-Null
  $outCsv = Join-Path $outDir ("roc_{0}_{1}.csv" -f $Algorithm, $lv.Name)
  [System.IO.File]::WriteAllText($outCsv, "")

  Write-Host ("=== {0} driving {1} ===" -f $Algorithm, $lv.Name)
  Write-Host ("clean={0}" -f $clean)
  Write-Host ("noisy={0}" -f $noisy)

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
    --out (Join-Path $outDir ("roc_{0}_{1}.png" -f $Algorithm, $lv.Name)) `
    --x fpr --y tpr --group tag --kind line `
    --xlabel FPR --ylabel TPR `
    --title ("{0} ROC (driving-{1})" -f $Algorithm.ToUpper(), $lv.Name)
}

Write-Host ("=== DONE: driving {0} ===" -f $Algorithm)

