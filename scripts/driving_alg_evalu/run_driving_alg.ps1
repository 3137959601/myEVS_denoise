param(
  [Parameter(Mandatory = $true)]
  [ValidateSet("knoise", "evflow", "ynoise", "ts", "mlpf")]
  [string]$Algorithm,
  [ValidateSet("coarse", "dense")]
  [string]$SweepProfile = "coarse"
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
$IS_DENSE = ($SweepProfile.ToLower() -eq "dense")

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

function Run-Roc([string]$Clean, [string]$Noisy, [string]$OutCsv, [string]$Tag, [string]$Method, [int]$Radius, [int]$TimeUs, [string]$Values, [string]$Engine = "python") {
  & $PY -m myevs.cli roc `
    --clean $Clean --noisy $Noisy `
    --assume npy --width $WIDTH --height $HEIGHT `
    --tick-ns $TICK_NS `
    --engine $Engine `
    --method $Method --radius-px $Radius --time-us $TimeUs `
    --param min-neighbors --values $Values `
    --match-us $MATCH_US --match-bin-radius $MATCH_BIN_RADIUS `
    --tag $Tag --out-csv $OutCsv --append --progress
}

function Get-TagAucRows {
  param([Parameter(Mandatory = $true)][string]$CsvPath)
  if (!(Test-Path $CsvPath)) { return @() }
  $rows = Import-Csv $CsvPath
  if (!$rows -or $rows.Count -eq 0) { return @() }
  return (
    $rows |
      Group-Object tag |
      ForEach-Object { $_.Group[0] | Select-Object tag, auc }
  )
}

function Get-TopTagsByRadius {
  param(
    [Parameter(Mandatory = $true)][string]$CsvPath,
    [int]$TopNPerRadius = 3
  )

  $rows = Get-TagAucRows -CsvPath $CsvPath
  $rSet = New-Object System.Collections.Generic.HashSet[int]
  foreach ($r in $rows) {
    $tag = [string]$r.tag
    if ($tag -match "_r(\d+)_") {
      [void]$rSet.Add([int]$matches[1])
    }
  }
  $rList = @($rSet | Sort-Object)
  $picked = New-Object System.Collections.Generic.List[string]
  foreach ($rv in $rList) {
    $prefix = "_r{0}_" -f [int]$rv
    $top = $rows |
      Where-Object { ([string]$_.tag) -like "*$prefix*" } |
      Sort-Object { [double]$_.auc } -Descending |
      Select-Object -First $TopNPerRadius
    foreach ($t in $top) {
      if ($t.tag) { $picked.Add([string]$t.tag) }
    }
  }
  return @($picked | Select-Object -Unique)
}

function Get-TopTagsGlobal {
  param(
    [Parameter(Mandatory = $true)][string]$CsvPath,
    [int]$TopN = 4
  )
  return @(
    Get-TagAucRows -CsvPath $CsvPath |
      Sort-Object { [double]$_.auc } -Descending |
      Select-Object -First $TopN |
      ForEach-Object { [string]$_.tag }
  )
}

function Export-FilteredCsv {
  param(
    [Parameter(Mandatory = $true)][string]$InCsv,
    [string[]]$Tags,
    [Parameter(Mandatory = $true)][string]$OutCsv
  )

  if (!(Test-Path $InCsv)) {
    throw "Input CSV not found: $InCsv"
  }
  if (!$Tags -or $Tags.Count -eq 0) {
    Copy-Item -LiteralPath $InCsv -Destination $OutCsv -Force
    return
  }

  $set = @{}
  foreach ($t in $Tags) { $set[$t] = $true }

  $filtered = Import-Csv $InCsv | Where-Object { $set.ContainsKey($_.tag) }
  if (!$filtered -or $filtered.Count -eq 0) {
    Copy-Item -LiteralPath $InCsv -Destination $OutCsv -Force
    return
  }

  $filtered | Export-Csv -NoTypeInformation -Encoding UTF8 $OutCsv
}

function New-IntRangeCsv {
  param([int]$Start, [int]$End, [int]$Step = 1)
  $vals = New-Object System.Collections.Generic.List[string]
  for ($v = $Start; $v -le $End; $v += $Step) {
    $vals.Add([string]$v)
  }
  return ($vals -join ",")
}

function New-FloatRangeCsv {
  param([double]$Start, [double]$End, [double]$Step, [string]$Fmt = "0.00")
  $vals = New-Object System.Collections.Generic.List[string]
  $v = $Start
  while ($v -le ($End + 1e-12)) {
    $s = [string]::Format([System.Globalization.CultureInfo]::InvariantCulture, ("{0:" + $Fmt + "}"), $v)
    $vals.Add($s)
    $v += $Step
  }
  return ($vals -join ",")
}

Write-Host ("Sweep profile: {0}" -f $SweepProfile)

foreach ($lv in $LEVELS) {
  $pair = Resolve-Pair -Dir $lv.Dir
  $clean = $pair.Clean
  $noisy = $pair.Noisy

  $outDir = "data/DND21/mydriving/{0}/{1}" -f $lv.Name, $Algorithm.ToUpper()
  New-Item -ItemType Directory -Force -Path $outDir | Out-Null
  $outCsv = Join-Path $outDir ("roc_{0}_{1}.csv" -f $Algorithm, $lv.Name)
  [System.IO.File]::WriteAllText($outCsv, "")
  $runtimeCsv = Join-Path $outDir ("runtime_{0}.csv" -f $Algorithm)
  if (!(Test-Path $runtimeCsv)) {
    "algorithm,level,start_time,end_time,elapsed_sec" | Out-File -FilePath $runtimeCsv -Encoding utf8
  }

  Write-Host ("=== {0} driving {1} ===" -f $Algorithm, $lv.Name)
  Write-Host ("clean={0}" -f $clean)
  Write-Host ("noisy={0}" -f $noisy)
  $t0 = Get-Date

  switch ($Algorithm) {
    "knoise" {
      # ========================= TUNE_HERE: KNOISE sweep =========================
      $thr = "0,1,2,3,4,5,6"
      foreach ($tau in @(1000,2000,4000,8000,16000,32000)) {
        Run-Roc -Clean $clean -Noisy $noisy -OutCsv $outCsv -Tag ("knoise_tau{0}" -f $tau) -Method "knoise" -Radius 1 -TimeUs $tau -Values $thr
      }
    }
    "evflow" {
      # ========================= TUNE_HERE: EVFLOW sweep =========================
      $thr = if ($IS_DENSE) { New-IntRangeCsv -Start 0 -End 80 -Step 2 } else { "0,4,8,12,16,24,32,48,64,80" }
      $rList = if ($IS_DENSE) { @(1, 2, 3, 4, 5) } else { @(2, 3, 4, 5) }
      $tauList = if ($IS_DENSE) { @(2000, 4000, 8000, 16000, 32000, 64000, 128000) } else { @(4000, 8000, 16000, 32000, 64000) }
      foreach ($r in $rList) {
        foreach ($tau in $tauList) {
          Run-Roc -Clean $clean -Noisy $noisy -OutCsv $outCsv -Tag ("evflow_r{0}_tau{1}" -f $r, $tau) -Method "evflow" -Radius $r -TimeUs $tau -Values $thr -Engine "numba"
        }
      }
    }
    "ynoise" {
      # ========================= TUNE_HERE: YNOISE sweep =========================
      $thr = if ($IS_DENSE) { New-IntRangeCsv -Start 0 -End 12 -Step 1 } else { "1,2,3,4,6,8" }
      foreach ($r in @(2, 3, 4)) {
        foreach ($tau in @(8000, 16000, 32000, 64000,128000,256000)) {
          Run-Roc -Clean $clean -Noisy $noisy -OutCsv $outCsv -Tag ("ynoise_r{0}_tau{1}" -f $r, $tau) -Method "ynoise" -Radius $r -TimeUs $tau -Values $thr
        }
      }
    }
    "ts" {
      # =========================== TUNE_HERE: TS sweep ===========================
      $thr = if ($IS_DENSE) { New-FloatRangeCsv -Start 0.01 -End 0.80 -Step 0.01 -Fmt "0.00" } else { "0.05,0.1,0.2,0.3,0.5" }
      foreach ($r in @(2, 3)) {
        foreach ($tau in @(32000, 64000, 128000, 256000)) {
          Run-Roc -Clean $clean -Noisy $noisy -OutCsv $outCsv -Tag ("ts_r{0}_decay{1}" -f $r, $tau) -Method "ts" -Radius $r -TimeUs $tau -Values $thr -Engine "numba"
        }
      }
    }
    "mlpf" {
      # ========================== TUNE_HERE: MLPF sweep ==========================
      $thr = if ($IS_DENSE) { New-IntRangeCsv -Start 0 -End 30 -Step 1 } else { "2,4,6,8,10,12,16,18,20,22,24,26" }
      foreach ($tau in @(32000, 64000, 128000, 256000,512000)) {
        Run-Roc -Clean $clean -Noisy $noisy -OutCsv $outCsv -Tag ("mlpf_tau{0}" -f $tau) -Method "mlpf" -Radius 3 -TimeUs $tau -Values $thr
      }
    }
  }

  $plotCsv = $outCsv
  if ($Algorithm -in @("evflow", "ynoise", "ts")) {
    $topTags = Get-TopTagsByRadius -CsvPath $outCsv -TopNPerRadius 3
    $plotCsv = Join-Path $outDir ("roc_{0}_{1}_top3_per_r.csv" -f $Algorithm, $lv.Name)
    Export-FilteredCsv -InCsv $outCsv -Tags $topTags -OutCsv $plotCsv
  } elseif ($Algorithm -eq "mlpf") {
    $topTags = Get-TopTagsGlobal -CsvPath $outCsv -TopN 4
    $plotCsv = Join-Path $outDir ("roc_{0}_{1}_top4_tau.csv" -f $Algorithm, $lv.Name)
    Export-FilteredCsv -InCsv $outCsv -Tags $topTags -OutCsv $plotCsv
  }

  & $PY -m myevs.cli plot-csv `
    --in $plotCsv `
    --out (Join-Path $outDir ("roc_{0}_{1}.png" -f $Algorithm, $lv.Name)) `
    --x fpr --y tpr --group tag --kind line `
    --xlabel FPR --ylabel TPR `
    --title ("{0} ROC (driving-{1})" -f $Algorithm.ToUpper(), $lv.Name)

  $t1 = Get-Date
  $elapsed = [Math]::Round((New-TimeSpan -Start $t0 -End $t1).TotalSeconds, 3)
  ('{0},{1},{2},{3},{4}' -f $Algorithm, $lv.Name, $t0.ToString("s"), $t1.ToString("s"), $elapsed) | Add-Content -Path $runtimeCsv -Encoding utf8
}

Write-Host ("=== DONE: driving {0} ===" -f $Algorithm)
