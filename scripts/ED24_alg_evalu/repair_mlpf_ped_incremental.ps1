param(
  [string]$PythonExe = "D:/software/Anaconda_envs/envs/myEVS/python.exe",
  [string]$ModelPattern = "data/ED24/myPedestrain_06/MLPF/models_retrain_20260522/mlpf_torch_{level}.pt"
)

$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot "../.."))

$ed24 = "D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06"
$outRoot = "data/ED24/myPedestrain_06/MLPF"

$thr = "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"

function Run-OneTau(
  [string]$Level,
  [string]$Clean,
  [string]$Noisy,
  [string]$Csv,
  [int]$Tau,
  [string]$Model
) {
  & $PythonExe -m myevs.cli roc `
    --clean $Clean --noisy $Noisy --assume npy --width 346 --height 260 `
    --tick-ns 1000 --engine python --method mlpf --radius-px 3 --time-us $Tau `
    --param min-neighbors --values $thr --match-us 0 --match-bin-radius 0 `
    --tag ("mlpf_tau{0}" -f $Tau) --mlpf-model $Model --mlpf-patch 7 `
    --out-csv $Csv --append --progress
  if ($LASTEXITCODE -ne 0) { throw "roc failed: level=$Level tau=$Tau" }
}

function Ensure-Base(
  [string]$Csv
) {
  if (!(Test-Path $Csv)) { return @() }
  try {
    $rows = Import-Csv $Csv
    if ($rows -eq $null) { return @() }
    return $rows
  } catch {
    return @()
  }
}

$jobs = @(
  @{ level="light"; clean="Pedestrain_06_1.8_signal_only.npy"; noisy="Pedestrain_06_1.8.npy"; all_taus=@(32000,64000,128000,256000,512000); rebuild=$true  },
  @{ level="mid"; clean="Pedestrain_06_2.5_signal_only.npy"; noisy="Pedestrain_06_2.5.npy"; all_taus=@(32000,64000,128000,256000,512000); rebuild=$false },
  @{ level="heavy"; clean="Pedestrain_06_3.3_signal_only.npy"; noisy="Pedestrain_06_3.3.npy"; all_taus=@(32000,64000,128000,256000,512000); rebuild=$false }
)

foreach ($j in $jobs) {
  $level = $j.level
  $clean = Join-Path $ed24 $j.clean
  $noisy = Join-Path $ed24 $j.noisy
  $csv = Join-Path $outRoot ("roc_mlpf_{0}.csv" -f $level)
  $model = $ModelPattern.Replace("{level}", $level)
  if (!(Test-Path $model)) { throw "missing model: $model" }
  if (!(Test-Path $clean)) { throw "missing clean: $clean" }
  if (!(Test-Path $noisy)) { throw "missing noisy: $noisy" }

  Write-Host ("=== incremental MLPF {0} ===" -f $level)
  if ($j.rebuild -and (Test-Path $csv)) {
    Remove-Item $csv -Force
    Write-Host "rebuild CSV for light"
  }

  $existing = Ensure-Base -Csv $csv
  $doneTau = @{}
  foreach ($r in $existing) {
    if ($r.tag -match "mlpf_tau(\d+)") { $doneTau[$Matches[1]] = $true }
  }

  foreach ($tau in $j.all_taus) {
    $k = [string]$tau
    if (!$j.rebuild -and $doneTau.ContainsKey($k)) {
      Write-Host ("skip existing tau={0}" -f $tau)
      continue
    }
    Run-OneTau -Level $level -Clean $clean -Noisy $noisy -Csv $csv -Tau $tau -Model $model
  }
}

Write-Host "=== incremental repair done ==="
