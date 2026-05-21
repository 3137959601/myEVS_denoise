param(
  [string]$Algorithm = "",
  [string[]]$Algorithms = @(),
  [string]$MlpfModelPattern = "",
  [int]$MaxEvents = 0,
  [string]$DatasetRoot = "D:\hjx_workspace\scientific_reserach\dataset\DND21\mydriving_ED24",
  [string]$DatasetSuffix = "ed24_withlabel",
  [ValidateSet("auto", "python", "numba", "cpp")]
  [string]$Engine = "auto",
  [ValidateSet("coarse", "dense")]
  [string]$SweepProfile = "coarse",
  [switch]$PaperAligned
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

$LEVELS = @(
  @{ Name = "1hz"; Dir = Join-Path $DatasetRoot ("driving_noise_1hz_{0}" -f $DatasetSuffix) },
  @{ Name = "2hz"; Dir = Join-Path $DatasetRoot ("driving_noise_2hz_{0}" -f $DatasetSuffix) },
  @{ Name = "3hz"; Dir = Join-Path $DatasetRoot ("driving_noise_3hz_{0}" -f $DatasetSuffix) },
  @{ Name = "5hz"; Dir = Join-Path $DatasetRoot ("driving_noise_5hz_{0}" -f $DatasetSuffix) },
  @{ Name = "7hz"; Dir = Join-Path $DatasetRoot ("driving_noise_7hz_{0}" -f $DatasetSuffix) },
  @{ Name = "8hz"; Dir = Join-Path $DatasetRoot ("driving_noise_8hz_{0}" -f $DatasetSuffix) },
  @{ Name = "10hz"; Dir = Join-Path $DatasetRoot ("driving_noise_10hz_{0}" -f $DatasetSuffix) }
)
$LEVELS = @($LEVELS | Where-Object {
  if (Test-Path $_.Dir) { return $true }
  Write-Host ("WARN: dataset level missing, skip: {0}" -f $_.Dir)
  return $false
})
$ALL_ALGS = @("baf", "stcf", "ebf", "n149", "knoise", "evflow", "ynoise", "ts", "mlpf", "pfd")
$CPP_ALGS = @("baf", "stcf", "ebf", "n149", "knoise", "evflow", "ynoise", "ts", "mlpf", "pfd")
$AUTO_CPP_ALGS = @("baf", "stcf", "ebf", "n149", "knoise", "evflow", "ynoise", "ts", "pfd")

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

function Resolve-EngineForAlg {
  param([string]$Alg, [string]$RequestedEngine = "auto")
  $req = "auto"
  if ($RequestedEngine -and $RequestedEngine.Trim().Length -gt 0) {
    $req = $RequestedEngine.Trim().ToLower()
  }
  if ($req -eq "cpp") {
    if ($Alg -notin $CPP_ALGS) {
      throw "engine=cpp currently supports only: $($CPP_ALGS -join ', '). Requested alg=$Alg"
    }
    return "cpp"
  }
  if ($req -eq "numba") {
    if ($Alg -notin @("stcf", "ts", "evflow", "pfd")) {
      throw "engine=numba is not wired for alg=$Alg in this script. Use -Engine auto or python."
    }
    return "numba"
  }
  if ($req -eq "python") { return "python" }
  if ($Alg -in $AUTO_CPP_ALGS) { return "cpp" }
  return "python"
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
  $pathBytes = [System.Text.Encoding]::UTF8.GetBytes([string]$InPath)
  $sha1 = [System.Security.Cryptography.SHA1]::Create()
  try {
    $hash = $sha1.ComputeHash($pathBytes)
  } finally {
    $sha1.Dispose()
  }
  $hashHex = ([System.BitConverter]::ToString($hash)).Replace("-", "").Substring(0, 10).ToLower()
  $cacheDir = "data/DND21/mydriving_ED24/_compact_cache/{0}" -f $Level
  New-Item -ItemType Directory -Force -Path $cacheDir | Out-Null
  $outPath = Join-Path $cacheDir ("{0}_{1}_n{2}.npy" -f $baseName, $hashHex, $MaxEvents)
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
Write-Host ("Engine policy: {0} (auto => baf/stcf/ebf/n149/knoise/evflow/ynoise/ts/pfd use cpp; mlpf uses python unless -Engine cpp is requested)" -f $Engine)
if ($PaperAligned) {
  Write-Host "Paper-aligned mode: ON (BAF/YNoise/TS/KNoise fixed defaults + sweep t=2..200ms)"
}

foreach ($alg in $SELECTED_ALGS) {
$algEngine = Resolve-EngineForAlg -Alg $alg -RequestedEngine $Engine
Write-Host ("Algorithm engine: {0} -> {1}" -f $alg, $algEngine)
foreach ($lv in $LEVELS) {
  $pair = Resolve-Pair -Dir $lv.Dir
  $clean = $pair.Clean
  $noisy = $pair.Noisy
  if ($MaxEvents -gt 0) {
    $clean = Prepare-CompactNpy -InPath $clean -Level $lv.Name -MaxEvents $MaxEvents
    $noisy = Prepare-CompactNpy -InPath $noisy -Level $lv.Name -MaxEvents $MaxEvents
  }

  # Driving-ED24 output layout aligned to ED24 style:
  # data/DND21/mydriving_ED24/{ALG}/roc_{alg}_{level}.csv
  # Keep STCF original in STCFO folder to distinguish from STCF variant.
  $outDirName = if ($alg -eq "stcf") { "STCFO" } else { $alg.ToUpper() }
  $outDir = "data/DND21/mydriving_ED24/{0}" -f $outDirName
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
    "algorithm,level,engine,start_time,end_time,elapsed_sec" | Out-File -FilePath $runtimeCsv -Encoding utf8
  }

  Write-Host ("=== {0} driving_ED24 {1} engine={2} ===" -f $alg, $lv.Name, $algEngine)
  Write-Host ("clean={0}" -f $clean)
  Write-Host ("noisy={0}" -f $noisy)
  $t0 = Get-Date

  # Common dense threshold for EBF and N149 (aligned for fair comparison)
  # Coarse: 17 points dense in 0-2 range; Dense: step=0.1 in 0-3, step=0.5 up to 8
  $EBF_N149_THR_COARSE = "0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,8"

  switch ($alg) {
    "baf" {
      # Source-aligned BAF: fixed radius=1 (3x3 neighborhood, polarity ignored).
      $tauList = if ($PaperAligned) {
        "2000,5000,10000,15000,20000,25000,30000,40000,50000,64000,80000,100000,128000,160000,200000"
      } elseif ($IS_DENSE) {
        "1000,2000,4000,8000,12000,16000,24000,32000"
      } else {
        "1000,2000,4000,8000,16000,32000"
      }
      & $PY -m myevs.cli roc `
        --clean $clean --noisy $noisy `
        --assume npy --width $WIDTH --height $HEIGHT `
        --tick-ns $TICK_NS `
        --engine $algEngine `
        --method baf --radius-px 1 --min-neighbors 1 `
        --param time-us --values $tauList `
        --match-us $MATCH_US --match-bin-radius $MATCH_BIN_RADIUS `
        --tag "baf_r1" --out-csv $outCsv --append --progress
      if ($LASTEXITCODE -ne 0) { throw "myevs roc failed for baf r=1" }
    }
    "stcf" {
      # Original STCF: fixed 3x3 neighborhood, each K is one curve (sweep tau).
      # Keep tau sweep aligned with BAF to avoid unfair under-sweep.
      $tauList = if ($PaperAligned) {
        "2000,5000,10000,15000,20000,25000,30000,40000,50000,64000,80000,100000,128000,160000,200000"
      } elseif ($IS_DENSE) {
        "1000,2000,4000,8000,12000,16000,24000,32000"
      } else {
        "1000,2000,4000,8000,16000,32000"
      }
      $kList = if ($IS_DENSE) { "1,2,3,4,5,6,7,8" } else { "1,2,3,4,5,6" }
      foreach ($k in ($kList -split ",")) {
        & $PY -m myevs.cli roc `
          --clean $clean --noisy $noisy `
          --assume npy --width $WIDTH --height $HEIGHT `
          --tick-ns $TICK_NS `
          --engine $algEngine `
          --method stcf_original --radius-px 1 --min-neighbors $k `
          --param time-us --values $tauList `
          --match-us $MATCH_US --match-bin-radius $MATCH_BIN_RADIUS `
          --tag ("stcf_orig_k{0}" -f $k) --out-csv $outCsv --append --progress
        if ($LASTEXITCODE -ne 0) { throw "myevs roc failed for stcf_original k=$k" }
      }
    }
    "ebf" {
      # EBF/N149 aligned threshold; narrowed r={2,3}, tau={8K-64K}
      $thr = if ($IS_DENSE) { (New-FloatRangeCsv -Start 0 -End 3 -Step 0.1 -Fmt "0.0") + "," + (New-FloatRangeCsv -Start 3.5 -End 8 -Step 0.5 -Fmt "0.0") } else { $EBF_N149_THR_COARSE }
      foreach ($r in @(2,3)) {
        foreach ($tau in @(8000,16000,32000,64000)) {
          Run-Roc -Clean $clean -Noisy $noisy -OutCsv $outCsv -Tag ("ebf_r{0}_tau{1}" -f $r, $tau) -Method "ebf" -Radius $r -TimeUs $tau -Values $thr -Engine $algEngine
        }
      }
    }
    "n149" {
      # SAME threshold as EBF for fair comparison; narrowed r={2,3}, tau={16K-64K}
      $thr = if ($IS_DENSE) { (New-FloatRangeCsv -Start 0 -End 3 -Step 0.1 -Fmt "0.0") + "," + (New-FloatRangeCsv -Start 3.5 -End 8 -Step 0.5 -Fmt "0.0") } else { $EBF_N149_THR_COARSE }
      foreach ($r in @(2,3)) {
        foreach ($tau in @(16000,32000,64000)) {
          Run-Roc -Clean $clean -Noisy $noisy -OutCsv $outCsv -Tag ("n149_r{0}_tau{1}" -f $r, $tau) -Method "n149" -Radius $r -TimeUs $tau -Values $thr -Engine $algEngine
        }
      }
    }
    "knoise" {
      if ($PaperAligned) {
        $tauVals = @(2000,5000,10000,15000,20000,25000,30000,40000,50000,64000,80000,100000,128000,160000,200000)
        foreach ($tau in $tauVals) {
          Run-Roc -Clean $clean -Noisy $noisy -OutCsv $outCsv -Tag ("knoise_tau{0}" -f $tau) -Method "knoise" -Radius 1 -TimeUs $tau -Values "1" -Engine $algEngine
        }
      } else {
        $thr = "0,1,2,3,4,5,6"
        foreach ($tau in @(1000,2000,4000,8000,16000,32000)) {
          Run-Roc -Clean $clean -Noisy $noisy -OutCsv $outCsv -Tag ("knoise_tau{0}" -f $tau) -Method "knoise" -Radius 1 -TimeUs $tau -Values $thr -Engine $algEngine
        }
      }
    }
    "evflow" {
      # Keep EVFLOW compact on driving due runtime cost.
      $thr = if ($IS_DENSE) { New-IntRangeCsv -Start 0 -End 64 -Step 4 } else { "8,16,24,32,48,64" }
      foreach ($r in @(2)) {
        foreach ($tau in @(8000, 16000)) {
          Run-Roc -Clean $clean -Noisy $noisy -OutCsv $outCsv -Tag ("evflow_r{0}_tau{1}" -f $r, $tau) -Method "evflow" -Radius $r -TimeUs $tau -Values $thr -Engine $algEngine
        }
      }
    }
    "ynoise" {
      $thr = if ($IS_DENSE) { New-IntRangeCsv -Start 0 -End 12 -Step 1 } else { "1,2,3,4,6,8" }
      foreach ($r in @(1, 2, 3)) {
        foreach ($tau in @(8000, 16000, 32000, 64000, 128000, 200000)) {
          Run-Roc -Clean $clean -Noisy $noisy -OutCsv $outCsv -Tag ("ynoise_r{0}_tau{1}" -f $r, $tau) -Method "ynoise" -Radius $r -TimeUs $tau -Values $thr -Engine $algEngine
        }
      }
    }
    "ts" {
      $thr = if ($IS_DENSE) { New-FloatRangeCsv -Start 0.01 -End 0.80 -Step 0.01 -Fmt "0.00" } else { "0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5" }
      foreach ($r in @(1, 2, 3)) {
        foreach ($tau in @(8000, 16000, 32000, 64000, 128000, 200000)) {
          Run-Roc -Clean $clean -Noisy $noisy -OutCsv $outCsv -Tag ("ts_r{0}_decay{1}" -f $r, $tau) -Method "ts" -Radius $r -TimeUs $tau -Values $thr -Engine $algEngine
        }
      }
    }
    "mlpf" {
      $mlpfArgs = @()
      $hasMlpfModel = $false
      $mlpfPatch = 7
      $mlpfDurationUs = 100000
      if ($MlpfModelPattern -and $MlpfModelPattern.Trim().Length -gt 0) {
        $resolved = $MlpfModelPattern.Replace("{level}", $lv.Name)
        if (!(Test-Path $resolved)) {
          throw "MLPF model not found for level=$($lv.Name): $resolved"
        }
        $metaPath = [System.IO.Path]::ChangeExtension($resolved, ".json")
        if (Test-Path $metaPath) {
          $meta = Get-Content -Raw $metaPath | ConvertFrom-Json
          if ($meta.patch) { $mlpfPatch = [int]$meta.patch }
          if ($meta.duration_us) { $mlpfDurationUs = [int]$meta.duration_us }
        } else {
          Write-Host ("WARN: MLPF metadata json not found, using defaults patch={0}, duration_us={1}: {2}" -f $mlpfPatch, $mlpfDurationUs, $metaPath)
        }
        $mlpfArgs = @("--mlpf-model", $resolved, "--mlpf-patch", ([string]$mlpfPatch))
        $hasMlpfModel = $true
      }

      if ($hasMlpfModel) {
        # Real TorchScript MLPF: duration/patch must match training metadata.
        # Sweep only probability threshold; do not sweep tau.
        $thr = New-FloatRangeCsv -Start 0.01 -End 0.99 -Step 0.01 -Fmt "0.00"
        Run-Roc -Clean $clean -Noisy $noisy -OutCsv $outCsv `
          -Tag ("mlpf_model_patch{0}_dur{1}" -f $mlpfPatch, $mlpfDurationUs) `
          -Method "mlpf" -Radius ([int]($mlpfPatch / 2)) -TimeUs $mlpfDurationUs -Values $thr -Engine $algEngine -ExtraArgs $mlpfArgs
      } else {
        if ($algEngine -eq "cpp") {
          throw "MLPF engine=cpp requires -MlpfModelPattern and exported same-stem .npz weights. Run scripts/export_mlpf_weights.py first."
        }
        # Proxy fallback keeps the old score scale; use only for debugging.
        $thr = if ($IS_DENSE) { New-IntRangeCsv -Start 0 -End 30 -Step 1 } else { "2,4,6,8,10,12,16,18,20,22,24,26" }
        foreach ($tau in @(32000, 64000, 128000, 256000, 512000)) {
          Run-Roc -Clean $clean -Noisy $noisy -OutCsv $outCsv -Tag ("mlpf_proxy_tau{0}" -f $tau) -Method "mlpf" -Radius 3 -TimeUs $tau -Values $thr -Engine $algEngine
        }
      }
    }
    "pfd" {
      # Source-aligned PFD uses 3x3 neighborhood (radius=1).
      $thr = if ($IS_DENSE) { New-IntRangeCsv -Start 0 -End 10 -Step 1 } else { "1,2,3,4,5,6,7,8" }
      $r = 1
      $tauList = if ($IS_DENSE) { @(8000,12000,16000,24000,32000) } else { @(8000,16000,32000) }
      $mList = if ($IS_DENSE) { @(1,2,3) } else { @(1,2) }
      foreach ($m in $mList) {
        foreach ($tau in $tauList) {
          & $PY -m myevs.cli roc `
            --clean $clean --noisy $noisy `
            --assume npy --width $WIDTH --height $HEIGHT `
            --tick-ns $TICK_NS `
            --engine $algEngine `
            --method pfd --radius-px $r --time-us $tau --refractory-us $m --pfd-mode a `
            --param min-neighbors --values $thr `
            --match-us $MATCH_US --match-bin-radius $MATCH_BIN_RADIUS `
            --tag ("pfd_r{0}_tau{1}_m{2}" -f $r, $tau, $m) --out-csv $outCsv --append --progress
          if ($LASTEXITCODE -ne 0) { throw "myevs roc failed for pfd r=$r tau=$tau m=$m" }
        }
      }
    }
  }

  $plotCsv = $outCsv
  if ($alg -in @("baf", "ebf", "n149", "evflow", "ynoise", "ts", "pfd")) {
    $topTags = Get-TopTagsByRadius -CsvPath $outCsv -TopNPerRadius 3
    $plotCsv = Join-Path $outDir ("roc_{0}_{1}_top3_per_r.csv" -f $alg, $lv.Name)
    Export-FilteredCsv -InCsv $outCsv -Tags $topTags -OutCsv $plotCsv
  } elseif ($alg -eq "stcf") {
    # STCF_orig: each K is one curve, keep best 3 K-curves by AUC.
    $topTags = Get-TopTagsGlobal -CsvPath $outCsv -TopN 3
    $plotCsv = Join-Path $outDir ("roc_{0}_{1}_top3_k.csv" -f $alg, $lv.Name)
    Export-FilteredCsv -InCsv $outCsv -Tags $topTags -OutCsv $plotCsv
  } elseif ($alg -eq "mlpf") {
    $topTags = Get-TopTagsGlobal -CsvPath $outCsv -TopN 4
    $plotCsv = Join-Path $outDir ("roc_{0}_{1}_top4_tau.csv" -f $alg, $lv.Name)
    Export-FilteredCsv -InCsv $outCsv -Tags $topTags -OutCsv $plotCsv
  }

  $plotTitle = if ($alg -eq "stcf") {
    ("STCF_ORIG ROC (driving-{0})" -f $lv.Name)
  } else {
    ("{0} ROC (driving-{1})" -f $alg.ToUpper(), $lv.Name)
  }

  & $PY -m myevs.cli plot-csv `
    --in $plotCsv `
    --out (Join-Path $outDir ("roc_{0}_{1}.png" -f $alg, $lv.Name)) `
    --x fpr --y tpr --group tag --kind line `
    --xlabel FPR --ylabel TPR `
    --title $plotTitle
  if ($LASTEXITCODE -ne 0) {
    throw "plot-csv failed (exit=$LASTEXITCODE): $plotCsv"
  }

  $t1 = Get-Date
  $elapsed = [Math]::Round((New-TimeSpan -Start $t0 -End $t1).TotalSeconds, 3)
  ('{0},{1},{2},{3},{4},{5}' -f $alg, $lv.Name, $algEngine, $t0.ToString("s"), $t1.ToString("s"), $elapsed) | Add-Content -Path $runtimeCsv -Encoding utf8
}
Write-Host ("=== DONE: driving {0} ===" -f $alg)
}

