param(
  [string]$Algorithm = "",
  [string[]]$Algorithms = @(),
  [string]$MlpfModelPattern = "",
  [int]$MaxEvents = 200000,
  [ValidateSet("coarse", "dense")]
  [string]$SweepProfile = "coarse"
)

$ErrorActionPreference = "Stop"
$env:PYTHONNOUSITE = "1"
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
  @{ Name = "light_mid"; Dir = Join-Path $ROOT "driving_noise_light_mid_slomo_shot_withlabel" },
  @{ Name = "mid"; Dir = Join-Path $ROOT "driving_noise_mid_slomo_shot_withlabel" }
)
$ALL_ALGS = @("baf", "stcf", "ebf", "knoise", "evflow", "ynoise", "ts", "mlpf", "pfd")

function Resolve-Algorithms {
  param(
    [string]$Algorithm,
    [string[]]$Algorithms
  )

  $picked = New-Object System.Collections.Generic.List[string]
  if ($Algorithm -and $Algorithm.Trim().Length -gt 0) {
    $picked.Add($Algorithm.Trim().ToLower())
  }
  foreach ($a in $Algorithms) {
    if ($a -and $a.Trim().Length -gt 0) {
      $picked.Add($a.Trim().ToLower())
    }
  }

  if ($picked.Count -eq 0) {
    return ,$ALL_ALGS
  }
  if ($picked -contains "all") {
    return ,$ALL_ALGS
  }

  $uniq = @()
  foreach ($a in $picked) {
    if ($a -notin $ALL_ALGS) {
      throw "Unknown algorithm '$a'. Valid: all, $($ALL_ALGS -join ', ')"
    }
    if ($a -notin $uniq) {
      $uniq += $a
    }
  }
  if ($uniq.Count -eq 0) {
    return ,$ALL_ALGS
  }
  return ,$uniq
}

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

function Run-Roc([string]$Clean, [string]$Noisy, [string]$OutCsv, [string]$Tag, [string]$Method, [int]$Radius, [int]$TimeUs, [string]$Values, [string]$Engine = "python", [string[]]$ExtraArgs = @()) {
  & $PY -m myevs.cli roc `
    --clean $Clean --noisy $Noisy `
    --assume npy --width $WIDTH --height $HEIGHT `
    --tick-ns $TICK_NS `
    --engine $Engine `
    --method $Method --radius-px $Radius --time-us $TimeUs `
    --param min-neighbors --values $Values `
    --match-us $MATCH_US --match-bin-radius $MATCH_BIN_RADIUS `
    --tag $Tag --out-csv $OutCsv --append --progress `
    @ExtraArgs
  if ($LASTEXITCODE -ne 0) {
    throw "myevs roc failed (exit=$LASTEXITCODE). tag=$Tag method=$Method values=$Values"
  }
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
    if ($tag -match "_r(\d+)(?:_|$)") {
      [void]$rSet.Add([int]$matches[1])
    }
  }
  $rList = @($rSet | Sort-Object)
  $picked = New-Object System.Collections.Generic.List[string]
  foreach ($rv in $rList) {
    $top = $rows |
      Where-Object { ([string]$_.tag) -match ("_r{0}(?:_|$)" -f [int]$rv) } |
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

function Prepare-CompactNpy {
  param(
    [Parameter(Mandatory = $true)][string]$InPath,
    [Parameter(Mandatory = $true)][string]$Level,
    [Parameter(Mandatory = $true)][int]$MaxEvents
  )
  if ($MaxEvents -le 0) {
    return $InPath
  }
  $baseName = [System.IO.Path]::GetFileNameWithoutExtension($InPath)
  $cacheDir = "data/DND21/mydriving/_compact_cache/{0}" -f $Level
  New-Item -ItemType Directory -Force -Path $cacheDir | Out-Null
  $outPath = Join-Path $cacheDir ("{0}_n{1}.npy" -f $baseName, $MaxEvents)
  $out = & $PY scripts/truncate_npy_events.py --in $InPath --out $outPath --max-events $MaxEvents --overwrite --print-path-only
  if ($LASTEXITCODE -ne 0) {
    throw "truncate_npy_events failed for: $InPath"
  }
  $resolved = (($out | Select-Object -Last 1) -as [string]).Trim()
  if ([string]::IsNullOrWhiteSpace($resolved)) {
    $resolved = $outPath
  }
  return $resolved
}

function Reset-CsvFile {
  param(
    [Parameter(Mandatory = $true)][string]$Path,
    [int]$Retries = 8,
    [int]$SleepMs = 400
  )
  for ($i = 0; $i -lt $Retries; $i++) {
    try {
      [System.IO.File]::WriteAllText($Path, "")
      return
    } catch {
      if ($i -ge ($Retries - 1)) {
        throw "Cannot reset CSV (file is locked): $Path`nPlease close viewers/editors using this file."
      }
      Start-Sleep -Milliseconds $SleepMs
    }
  }
}

Write-Host ("Sweep profile: {0}" -f $SweepProfile)
if ($MaxEvents -gt 0) {
  Write-Host ("Max events per stream: {0} (compact mode)" -f $MaxEvents)
}
$SELECTED_ALGS = Resolve-Algorithms -Algorithm $Algorithm -Algorithms $Algorithms
Write-Host ("Selected algorithms: {0}" -f ($SELECTED_ALGS -join ", "))

foreach ($alg in $SELECTED_ALGS) {
foreach ($lv in $LEVELS) {
  $pair = Resolve-Pair -Dir $lv.Dir
  $clean = $pair.Clean
  $noisy = $pair.Noisy
  if ($MaxEvents -gt 0) {
    $clean = Prepare-CompactNpy -InPath $clean -Level $lv.Name -MaxEvents $MaxEvents
    $noisy = Prepare-CompactNpy -InPath $noisy -Level $lv.Name -MaxEvents $MaxEvents
  }

  # Driving output layout aligned to ED24 style:
  # data/DND21/mydriving/{ALG}/roc_{alg}_{level}.csv
  $outDir = "data/DND21/mydriving/{0}" -f $alg.ToUpper()
  New-Item -ItemType Directory -Force -Path $outDir | Out-Null
  $outCsvBase = Join-Path $outDir ("roc_{0}_{1}.csv" -f $alg, $lv.Name)
  $outCsv = $outCsvBase
  try {
    Reset-CsvFile -Path $outCsv
  } catch {
    $stamp = (Get-Date).ToString("yyyyMMdd_HHmmss")
    $outCsv = Join-Path $outDir ("roc_{0}_{1}_{2}.csv" -f $alg, $lv.Name, $stamp)
    Reset-CsvFile -Path $outCsv
    Write-Host ("WARN: base csv is locked, using fallback csv: {0}" -f $outCsv)
  }
  $runtimeCsv = Join-Path $outDir ("runtime_{0}.csv" -f $alg)
  if (!(Test-Path $runtimeCsv)) {
    "algorithm,level,start_time,end_time,elapsed_sec" | Out-File -FilePath $runtimeCsv -Encoding utf8
  }

  Write-Host ("=== {0} driving {1} ===" -f $alg, $lv.Name)
  Write-Host ("clean={0}" -f $clean)
  Write-Host ("noisy={0}" -f $noisy)
  $t0 = Get-Date

  switch ($alg) {
    "baf" {
      # Align with ED24 style: one curve per r by sweeping tau.
      $tauList = if ($IS_DENSE) {
        "1000,2000,4000,8000,12000,16000,24000,32000,48000,64000,96000,128000"
      } else {
        "1000,2000,4000,8000,16000,32000,64000,128000"
      }
      foreach ($r in @(1,2,3,4)) {
        & $PY -m myevs.cli roc `
          --clean $clean --noisy $noisy `
          --assume npy --width $WIDTH --height $HEIGHT `
          --tick-ns $TICK_NS `
          --method baf --radius-px $r --min-neighbors 1 `
          --param time-us --values $tauList `
          --match-us $MATCH_US --match-bin-radius $MATCH_BIN_RADIUS `
          --tag ("baf_r{0}" -f $r) --out-csv $outCsv --append --progress
        if ($LASTEXITCODE -ne 0) {
          throw "myevs roc failed for baf r=$r"
        }
      }
    }
    "stcf" {
      # STCF is the slowest grid in this script; coarse mode uses a reduced tau list.
      $tauList = if ($IS_DENSE) {
        "100,200,500,1000,2000,4000,8000,16000,32000,64000,128000,256000,512000"
      } else {
        "500,1000,2000,4000,8000,16000,32000,64000,128000"
      }
      foreach ($r in @(1,2,3,4)) {
        & $PY -m myevs.cli roc `
          --clean $clean --noisy $noisy `
          --assume npy --width $WIDTH --height $HEIGHT `
          --tick-ns $TICK_NS `
          --method stc --radius-px $r `
          --param time-us --values $tauList `
          --match-us $MATCH_US --match-bin-radius $MATCH_BIN_RADIUS `
          --tag ("stcf_r{0}" -f $r) --out-csv $outCsv --append --progress
        if ($LASTEXITCODE -ne 0) {
          throw "myevs roc failed for stcf r=$r"
        }
      }
    }
    "ebf" {
      $thr = if ($IS_DENSE) { New-FloatRangeCsv -Start 0 -End 8 -Step 0.25 -Fmt "0.00" } else { "0,0.5,1,1.5,2,2.5,3,4,5,6,8" }
      foreach ($r in @(2,3,4)) {
        foreach ($tau in @(16000,32000,64000,128000,256000)) {
          Run-Roc -Clean $clean -Noisy $noisy -OutCsv $outCsv -Tag ("ebf_r{0}_tau{1}" -f $r, $tau) -Method "ebf" -Radius $r -TimeUs $tau -Values $thr
        }
      }
    }
    "knoise" {
      $thr = "0,1,2,3,4,5,6"
      foreach ($tau in @(1000,2000,4000,8000,16000,32000)) {
        Run-Roc -Clean $clean -Noisy $noisy -OutCsv $outCsv -Tag ("knoise_tau{0}" -f $tau) -Method "knoise" -Radius 1 -TimeUs $tau -Values $thr
      }
    }
    "evflow" {
      $thr = if ($IS_DENSE) { New-IntRangeCsv -Start 0 -End 64 -Step 2 } else { "0,8,16,24,32,48,64" }
      $rList = if ($IS_DENSE) { @(2,3,4,5) } else { @(2,3,4) }
      $tauList = if ($IS_DENSE) { @(8000,16000,24000,32000,48000,64000,96000) } else { @(8000,16000,32000) }
      foreach ($r in $rList) {
        foreach ($tau in $tauList) {
          Run-Roc -Clean $clean -Noisy $noisy -OutCsv $outCsv -Tag ("evflow_r{0}_tau{1}" -f $r, $tau) -Method "evflow" -Radius $r -TimeUs $tau -Values $thr -Engine "numba"
        }
      }
    }
    "ynoise" {
      $thr = if ($IS_DENSE) { New-IntRangeCsv -Start 0 -End 12 -Step 1 } else { "1,2,3,4,6,8" }
      foreach ($r in @(2, 3, 4)) {
        foreach ($tau in @(8000, 16000, 32000, 64000,128000,256000)) {
          Run-Roc -Clean $clean -Noisy $noisy -OutCsv $outCsv -Tag ("ynoise_r{0}_tau{1}" -f $r, $tau) -Method "ynoise" -Radius $r -TimeUs $tau -Values $thr
        }
      }
    }
    "ts" {
      $thr = if ($IS_DENSE) { New-FloatRangeCsv -Start 0.01 -End 0.80 -Step 0.01 -Fmt "0.00" } else { "0.05,0.1,0.2,0.3,0.5" }
      foreach ($r in @(2, 3)) {
        foreach ($tau in @(32000, 64000, 128000, 256000)) {
          Run-Roc -Clean $clean -Noisy $noisy -OutCsv $outCsv -Tag ("ts_r{0}_decay{1}" -f $r, $tau) -Method "ts" -Radius $r -TimeUs $tau -Values $thr -Engine "numba"
        }
      }
    }
    "mlpf" {
      $mlpfArgs = @()
      $hasMlpfModel = $false
      if ($MlpfModelPattern -and $MlpfModelPattern.Trim().Length -gt 0) {
        $resolved = $MlpfModelPattern.Replace("{level}", $lv.Name)
        if (!(Test-Path $resolved)) {
          throw "MLPF model not found for level=$($lv.Name): $resolved"
        }
        $mlpfArgs = @("--mlpf-model", $resolved, "--mlpf-patch", "7")
        $hasMlpfModel = $true
      }
      $thr = if ($hasMlpfModel) {
        if ($IS_DENSE) { New-FloatRangeCsv -Start 0.05 -End 0.95 -Step 0.05 -Fmt "0.00" } else { "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9" }
      } else {
        if ($IS_DENSE) { New-IntRangeCsv -Start 0 -End 30 -Step 1 } else { "2,4,6,8,10,12,16,18,20,22,24,26" }
      }
      foreach ($tau in @(32000, 64000, 128000, 256000,512000)) {
        Run-Roc -Clean $clean -Noisy $noisy -OutCsv $outCsv -Tag ("mlpf_tau{0}" -f $tau) -Method "mlpf" -Radius 3 -TimeUs $tau -Values $thr -ExtraArgs $mlpfArgs
      }
    }
    "pfd" {
      $thr = if ($IS_DENSE) { New-IntRangeCsv -Start 0 -End 10 -Step 1 } else { "1,2,3,4,5,6,7,8" }
      $r = 3
      $tauList = if ($IS_DENSE) { @(8000,12000,16000,24000,32000,48000,64000,96000,128000,192000,256000) } else { @(8000,16000,32000,64000,128000,256000) }
      $mList = if ($IS_DENSE) { @(1,2,3,4) } else { @(1,2,3) }
      foreach ($m in $mList) {
        foreach ($tau in $tauList) {
          & $PY -m myevs.cli roc `
            --clean $clean --noisy $noisy `
            --assume npy --width $WIDTH --height $HEIGHT `
            --tick-ns $TICK_NS `
            --engine numba `
            --method pfd --radius-px $r --time-us $tau --refractory-us $m --pfd-mode a `
            --param min-neighbors --values $thr `
            --match-us $MATCH_US --match-bin-radius $MATCH_BIN_RADIUS `
            --tag ("pfd_r{0}_tau{1}_m{2}" -f $r, $tau, $m) --out-csv $outCsv --append --progress
          if ($LASTEXITCODE -ne 0) {
            throw "myevs roc failed for pfd r=$r tau=$tau m=$m"
          }
        }
      }
    }
  }

  $plotCsv = $outCsv
  if ($alg -in @("baf", "stcf", "ebf", "evflow", "ynoise", "ts", "pfd")) {
    $topTags = Get-TopTagsByRadius -CsvPath $outCsv -TopNPerRadius 3
    $plotCsv = Join-Path $outDir ("roc_{0}_{1}_top3_per_r.csv" -f $alg, $lv.Name)
    Export-FilteredCsv -InCsv $outCsv -Tags $topTags -OutCsv $plotCsv
  } elseif ($alg -eq "mlpf") {
    $topTags = Get-TopTagsGlobal -CsvPath $outCsv -TopN 4
    $plotCsv = Join-Path $outDir ("roc_{0}_{1}_top4_tau.csv" -f $alg, $lv.Name)
    Export-FilteredCsv -InCsv $outCsv -Tags $topTags -OutCsv $plotCsv
  }

  & $PY -m myevs.cli plot-csv `
    --in $plotCsv `
    --out (Join-Path $outDir ("roc_{0}_{1}.png" -f $alg, $lv.Name)) `
    --x fpr --y tpr --group tag --kind line `
    --xlabel FPR --ylabel TPR `
    --title ("{0} ROC (driving-{1})" -f $alg.ToUpper(), $lv.Name)
  if ($LASTEXITCODE -ne 0) {
    throw "plot-csv failed (exit=$LASTEXITCODE): $plotCsv"
  }

  $t1 = Get-Date
  $elapsed = [Math]::Round((New-TimeSpan -Start $t0 -End $t1).TotalSeconds, 3)
  ('{0},{1},{2},{3},{4}' -f $alg, $lv.Name, $t0.ToString("s"), $t1.ToString("s"), $elapsed) | Add-Content -Path $runtimeCsv -Encoding utf8
}
Write-Host ("=== DONE: driving {0} ===" -f $alg)
}
