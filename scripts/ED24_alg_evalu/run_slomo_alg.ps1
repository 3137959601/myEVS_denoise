param(
  # Backward-compatible single algorithm selection.
  [string]$Algorithm = "",
  # New multi-algorithm selection. Examples:
  #   -Algorithms knoise,ts
  #   -Algorithms all
  [string[]]$Algorithms = @(),
  # Sweep profile:
  #   auto  : default policy (EVFLOW=dense, others=coarse)
  #   coarse: all algorithms use sparse grid (faster)
  #   dense : all algorithms use denser/continuous grid
  [ValidateSet("auto", "coarse", "dense")]
  [string]$SweepProfile = "auto",
  # Optional TorchScript model pattern for real MLPF inference.
  # Example: data/ED24/myPedestrain_06/MLPF/mlpf_torch_{level}.pt
  [string]$MlpfModelPattern = ""
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
  @{ Name = "light_mid"; Noisy = "Pedestrain_06_2.1.npy"; Clean = "Pedestrain_06_2.1_signal_only.npy" },
  @{ Name = "mid"; Noisy = "Pedestrain_06_2.5.npy"; Clean = "Pedestrain_06_2.5_signal_only.npy" },
  @{ Name = "heavy"; Noisy = "Pedestrain_06_3.3.npy"; Clean = "Pedestrain_06_3.3_signal_only.npy" }
)

$ALL_ALGS = @("knoise", "evflow", "ynoise", "ts", "mlpf", "pfd")

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

function Use-DenseForAlgorithm {
  param([string]$Alg)
  $profile = "auto"
  if ($SweepProfile -and $SweepProfile.Trim().Length -gt 0) {
    $profile = $SweepProfile.ToLower()
  }
  if ($profile -eq "dense") { return $true }
  if ($profile -eq "coarse") { return $false }
  # auto mode
  return ($Alg -eq "evflow")
}

foreach ($sp in $SPLITS) {
  foreach ($n in @($sp.Noisy, $sp.Clean)) {
    $p = Join-Path $ED24_DIR $n
    if (!(Test-Path $p)) {
      throw "Missing required file: $p"
    }
  }
}

function Run-Roc(
  [string]$Clean,
  [string]$Noisy,
  [string]$OutCsv,
  [string]$Tag,
  [string]$Method,
  [int]$Radius,
  [int]$TimeUs,
  [string]$SweepValues,
  [string]$Engine = "python",
  [string[]]$ExtraArgs = @()
) {
  if ([string]::IsNullOrWhiteSpace($SweepValues)) {
    throw "Run-Roc received empty sweep values. tag=$Tag method=$Method"
  }

  $cliArgs = @(
    "-m", "myevs.cli", "roc",
    "--clean", $Clean,
    "--noisy", $Noisy,
    "--assume", "npy",
    "--width", "346",
    "--height", "260",
    "--tick-ns", "$TICK_NS",
    "--engine", $Engine,
    "--method", $Method,
    "--radius-px", "$Radius",
    "--time-us", "$TimeUs",
    "--param", "min-neighbors",
    "--values", "$SweepValues",
    "--match-us", "$MATCH_US",
    "--match-bin-radius", "$MATCH_BIN_RADIUS",
    "--tag", $Tag,
    "--out-csv", $OutCsv,
    "--append",
    "--progress"
  )
  if ($ExtraArgs -and $ExtraArgs.Count -gt 0) {
    $cliArgs += $ExtraArgs
  }

  & $PY @cliArgs
  if ($LASTEXITCODE -ne 0) {
    throw "myevs roc failed (exit=$LASTEXITCODE). tag=$Tag method=$Method values=$SweepValues"
  }
}
function Run-Roc-TimeSweep(
  [string]$Clean,
  [string]$Noisy,
  [string]$OutCsv,
  [string]$Tag,
  [string]$Method,
  [int]$Radius,
  [double]$FixedThr,
  [string]$TimeValues,
  [string]$Engine = "python"
) {
  if ([string]::IsNullOrWhiteSpace($TimeValues)) {
    throw "Run-Roc-TimeSweep received empty time sweep values. tag=$Tag method=$Method"
  }

  $cliArgs = @(
    "-m", "myevs.cli", "roc",
    "--clean", $Clean,
    "--noisy", $Noisy,
    "--assume", "npy",
    "--width", "346",
    "--height", "260",
    "--tick-ns", "$TICK_NS",
    "--engine", $Engine,
    "--method", $Method,
    "--radius-px", "$Radius",
    "--min-neighbors", "$FixedThr",
    "--param", "time-us",
    "--values", "$TimeValues",
    "--match-us", "$MATCH_US",
    "--match-bin-radius", "$MATCH_BIN_RADIUS",
    "--tag", $Tag,
    "--out-csv", $OutCsv,
    "--append",
    "--progress"
  )

  & $PY @cliArgs
  if ($LASTEXITCODE -ne 0) {
    throw "myevs roc time-sweep failed (exit=$LASTEXITCODE). tag=$Tag method=$Method values=$TimeValues"
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

$SELECTED_ALGS = Resolve-Algorithms -Algorithm $Algorithm -Algorithms $Algorithms
Write-Host ("Selected algorithms: {0}" -f ($SELECTED_ALGS -join ", "))
Write-Host ("Sweep profile: {0}" -f $SweepProfile)

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

function Reset-CsvFile {
  param(
    [Parameter(Mandatory = $true)][string]$Path,
    [int]$Retries = 5,
    [int]$SleepMs = 500
  )
  for ($i = 0; $i -lt $Retries; $i++) {
    try {
      [System.IO.File]::WriteAllText($Path, "")
      return
    } catch {
      if ($i -ge ($Retries - 1)) {
        throw "Cannot reset CSV (file is locked): $Path`nPlease close editors/viewers or stop other running sweep processes."
      }
      Start-Sleep -Milliseconds $SleepMs
    }
  }
}

foreach ($alg in $SELECTED_ALGS) {
  $IS_DENSE = Use-DenseForAlgorithm -Alg $alg
  Write-Host ("Algorithm sweep mode: {0} -> {1}" -f $alg, ($(if ($IS_DENSE) { "dense" } else { "coarse" })))
  foreach ($sp in $SPLITS) {
    $clean = Join-Path $ED24_DIR $sp.Clean
    $noisy = Join-Path $ED24_DIR $sp.Noisy
    $outDir = "data/ED24/myPedestrain_06/{0}" -f $alg.ToUpper()
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    $outCsv = Join-Path $outDir ("roc_{0}_{1}.csv" -f $alg, $sp.Name)
    Reset-CsvFile -Path $outCsv
    $runtimeCsv = Join-Path $outDir ("runtime_{0}.csv" -f $alg)
    if (!(Test-Path $runtimeCsv)) {
      "algorithm,level,start_time,end_time,elapsed_sec" | Out-File -FilePath $runtimeCsv -Encoding utf8
    }

    Write-Host ("=== {0} ED24 {1} ===" -f $alg, $sp.Name)
    $t0 = Get-Date

    switch ($alg) {      "knoise" {
        # ========================= TUNE_HERE: KNOISE sweep =========================
        # Standard ROC: fixed tau, sweep threshold.
        $knoiseTauList = @(16000, 32000, 64000, 128000, 256000)
        $thr = "1,2,3"
        foreach ($tau in $knoiseTauList) {
          Run-Roc -Clean $clean -Noisy $noisy -OutCsv $outCsv -Tag ("knoise_tau{0}" -f $tau) -Method "knoise" -Radius 1 -TimeUs $tau -SweepValues $thr
        }

        # Extra profile (for interpretability): fixed threshold, sweep tau.
        $profileCsv = Join-Path $outDir ("profile_knoise_{0}.csv" -f $sp.Name)
        Reset-CsvFile -Path $profileCsv
        # Exponential tau sweep: tau_k = 2000 * 2^k, k=0..15 (16 points).
        # This keeps low-tau resolution and progressively widens high-tau spacing.
        $tauSweepList = @()
        $tau = 2000
        for ($k = 0; $k -lt 16; $k++) {
          $tauSweepList += [int]$tau
          $tau *= 2
        }
        $tauSweepValues = ($tauSweepList -join ",")
        foreach ($thrFix in @(1, 2, 3 )) {
          Run-Roc-TimeSweep -Clean $clean -Noisy $noisy -OutCsv $profileCsv -Tag ("knoise_thr{0}" -f $thrFix) -Method "knoise" -Radius 1 -FixedThr $thrFix -TimeValues $tauSweepValues
        }

        & $PY -m myevs.cli plot-csv `
          --in $profileCsv `
          --out (Join-Path $outDir ("profile_knoise_{0}.png" -f $sp.Name)) `
          --x fpr --y tpr --group tag --kind line `
          --xlabel FPR --ylabel TPR `
          --title ("KNOISE Profile ROC ({0}, fixed-thr sweep-tau)" -f $sp.Name)
        if ($LASTEXITCODE -ne 0) {
          throw "plot-csv failed (exit=$LASTEXITCODE): $profileCsv"
        }
      }
      "evflow" {
        # ========================= TUNE_HERE: EVFLOW sweep =========================
        # Extend threshold dynamic range so curve can approach high-TPR/high-FPR end.
        # Keep point count controlled: dense uses step=2 up to 80 (41 points/tag).
        $thr = if ($IS_DENSE) { New-IntRangeCsv -Start 0 -End 80 -Step 2 } else { "0,4,8,12,16,24,32,48,64,80" }
        # Include smaller radius/window to avoid over-smoothed clustered operating points.
        $rList = if ($IS_DENSE) { @(2, 3, 4, 5) } else { @(2, 3, 4, 5) }
        $tauList = if ($IS_DENSE) { @( 8000, 16000, 32000, 64000) } else { @(8000, 16000, 32000, 64000) }
        foreach ($r in $rList) {
          foreach ($tau in $tauList) {
            Run-Roc -Clean $clean -Noisy $noisy -OutCsv $outCsv -Tag ("evflow_r{0}_tau{1}" -f $r, $tau) -Method "evflow" -Radius $r -TimeUs $tau -SweepValues $thr -Engine "numba"
          }
        }
      }
      "ynoise" {
        # ========================= TUNE_HERE: YNOISE sweep =========================
        # Dense mode mainly refines threshold axis.
        $thr = if ($IS_DENSE) { New-IntRangeCsv -Start 0 -End 12 -Step 1 } else { "1,2,3,4,6,8" }
        foreach ($r in @(2, 3, 4, 5)) {
          foreach ($tau in @(16000,32000, 64000,128000,256000)) {
            Run-Roc -Clean $clean -Noisy $noisy -OutCsv $outCsv -Tag ("ynoise_r{0}_tau{1}" -f $r, $tau) -Method "ynoise" -Radius $r -TimeUs $tau -SweepValues $thr
          }
        }
      }
      "ts" {
        # =========================== TUNE_HERE: TS sweep ===========================
        # Current best AUC in ED24 is around r=2/3, decay=30000~60000.
        # Dense mode uses near-continuous float thresholds.
        $thr = if ($IS_DENSE) { New-FloatRangeCsv -Start 0.01 -End 0.80 -Step 0.01 -Fmt "0.00" } else { "0.05,0.1,0.2,0.3,0.5" }
        foreach ($r in @(1, 2, 3, 4)) {
          foreach ($tau in @(16000, 32000, 64000, 128000)) {
            Run-Roc -Clean $clean -Noisy $noisy -OutCsv $outCsv -Tag ("ts_r{0}_decay{1}" -f $r, $tau) -Method "ts" -Radius $r -TimeUs $tau -SweepValues $thr -Engine "numba"
          }
        }
      }
      "mlpf" {
        # ========================== TUNE_HERE: MLPF sweep ==========================
        $mlpfArgs = @()
        $hasMlpfModel = $false
        if ($MlpfModelPattern -and $MlpfModelPattern.Trim().Length -gt 0) {
          $resolved = $MlpfModelPattern.Replace("{level}", $sp.Name)
          if (!(Test-Path $resolved)) {
            throw "MLPF model not found for level=$($sp.Name): $resolved"
          }
          $mlpfArgs = @("--mlpf-model", $resolved, "--mlpf-patch", "7")
          $hasMlpfModel = $true
        }

        # Model mode: threshold is probability; proxy mode keeps historical integer sweep.
        $thr = if ($hasMlpfModel) {
          if ($IS_DENSE) { New-FloatRangeCsv -Start 0.05 -End 0.95 -Step 0.05 -Fmt "0.00" } else { "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9" }
        } else {
          if ($IS_DENSE) { New-IntRangeCsv -Start 0 -End 30 -Step 1 } else { "2,4,6,8,10,12,16,18,20,22,24" }
        }
        foreach ($tau in @(32000,64000, 128000, 256000, 512000)) {
          Run-Roc -Clean $clean -Noisy $noisy -OutCsv $outCsv -Tag ("mlpf_tau{0}" -f $tau) -Method "mlpf" -Radius 3 -TimeUs $tau -SweepValues $thr -ExtraArgs $mlpfArgs
        }
      }
      "pfd" {
        # =========================== TUNE_HERE: PFD sweep ===========================
        # NOTE:
        # - Keep radius fixed to r=3 (empirical best on current ED24 setting).
        # - Sweep delta_t(time-us), lambda(min-neighbors), m(refractory-us).
        # - Use mode A for ED24 baseline; mode B is implemented but not run here.
        $thr = if ($IS_DENSE) { New-IntRangeCsv -Start 0 -End 10 -Step 1 } else { "1,2,3,4,5,6,7,8" }
        $r = 3
        $tauList = if ($IS_DENSE) { @(8000,12000,16000,24000,32000,48000,64000,96000,128000,192000,256000) } else { @(8000, 16000, 32000, 64000, 128000, 256000) }
        $mList = if ($IS_DENSE) { @(1,2,3,4) } else { @(1,2,3) }
        foreach ($m in $mList) {
          foreach ($tau in $tauList) {
            Run-Roc -Clean $clean -Noisy $noisy -OutCsv $outCsv -Tag ("pfd_r{0}_tau{1}_m{2}" -f $r, $tau, $m) -Method "pfd" -Radius $r -TimeUs $tau -SweepValues $thr -Engine "numba" -ExtraArgs @("--refractory-us", "$m", "--pfd-mode", "a")
          }
        }
      }
    }

    $plotCsv = $outCsv
    if ($alg -in @("evflow", "ynoise", "ts", "pfd")) {
      $topTags = Get-TopTagsByRadius -CsvPath $outCsv -TopNPerRadius 3
      $plotCsv = Join-Path $outDir ("roc_{0}_{1}_top3_per_r.csv" -f $alg, $sp.Name)
      Export-FilteredCsv -InCsv $outCsv -Tags $topTags -OutCsv $plotCsv
    } elseif ($alg -eq "mlpf") {
      $topTags = Get-TopTagsGlobal -CsvPath $outCsv -TopN 4
      $plotCsv = Join-Path $outDir ("roc_{0}_{1}_top4_tau.csv" -f $alg, $sp.Name)
      Export-FilteredCsv -InCsv $outCsv -Tags $topTags -OutCsv $plotCsv
    }

    & $PY -m myevs.cli plot-csv `
      --in $plotCsv `
      --out (Join-Path $outDir ("roc_{0}_{1}.png" -f $alg, $sp.Name)) `
      --x fpr --y tpr --group tag --kind line `
      --xlabel FPR --ylabel TPR `
      --title ("{0} ROC ({1})" -f $alg.ToUpper(), $sp.Name)
    if ($LASTEXITCODE -ne 0) {
      throw "plot-csv failed (exit=$LASTEXITCODE): $plotCsv"
    }

    $t1 = Get-Date
    $elapsed = [Math]::Round((New-TimeSpan -Start $t0 -End $t1).TotalSeconds, 3)
    ('{0},{1},{2},{3},{4}' -f $alg, $sp.Name, $t0.ToString("s"), $t1.ToString("s"), $elapsed) | Add-Content -Path $runtimeCsv -Encoding utf8
  }

  Write-Host ("=== DONE: ED24 {0} ===" -f $alg)
}
