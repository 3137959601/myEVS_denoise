param(
  [string[]]$Dataset = @("ed24", "driving", "led"),
  [string]$PythonExe = "D:/software/Anaconda_envs/envs/myEVS/python.exe",
  [string]$OutDir = "data/summary/n179_rhythm_ablation_small",
  [string]$RhythmPressureCoeffList = "0,0.1666666667,0.3333333333,0.5",
  [string]$RhythmGoodCoeffList = "0,0.125,0.25,0.375,0.5",
  [string]$SupportGoodCoeffList = "0.25",
  [string]$KSfracList = "0.4",
  [string]$KMixList = "0.0",
  [int]$MaxEvents = 0
)

$ErrorActionPreference = "Stop"

function Invoke-N179RhythmAblation {
  param(
    [string]$Name,
    [string]$Noisy,
    [string]$DatasetName,
    [string]$Scene,
    [string]$Level,
    [int]$Width,
    [int]$Height,
    [string]$SList,
    [string]$TauList
  )

  Write-Host ""
  Write-Host "=== N179 rhythm ablation: $Name ==="
  Write-Host "data: $Noisy"
  Write-Host "s-list=$SList tau-us-list=$TauList k_sfrac=$KSfracList k_mix=$KMixList"
  Write-Host "rhythm_pressure_coeff=$RhythmPressureCoeffList"
  Write-Host "rhythm_good_coeff=$RhythmGoodCoeffList"
  Write-Host "support_good_coeff=$SupportGoodCoeffList"

  if (!(Test-Path -LiteralPath $Noisy)) {
    throw "Input file not found: $Noisy"
  }

  New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

  $rocCsv = Join-Path $OutDir "${Name}_roc.csv"
  $summaryCsv = Join-Path $OutDir "${Name}_summary.csv"
  $runtimeCsv = Join-Path $OutDir "${Name}_runtime.csv"

  & $PythonExe "scripts/noise_analyze/sweep_n179_pair.py" `
    --noisy $Noisy `
    --out-csv $rocCsv `
    --summary-csv $summaryCsv `
    --runtime-csv $runtimeCsv `
    --dataset $DatasetName `
    --scene $Scene `
    --level $Level `
    --width $Width `
    --height $Height `
    --tick-ns 1000 `
    --s-list $SList `
    --tau-us-list $TauList `
    --k-sfrac-list $KSfracList `
    --k-mix-list $KMixList `
    --rhythm-pressure-coeff-list $RhythmPressureCoeffList `
    --rhythm-good-coeff-list $RhythmGoodCoeffList `
    --support-good-coeff-list $SupportGoodCoeffList `
    --signal-label-value 1 `
    --max-events $MaxEvents `
    --tag-prefix "n179rh"

  Write-Host "saved: $summaryCsv"
}

$selected = @{}
foreach ($item in $Dataset) {
  $selected[$item.ToLowerInvariant()] = $true
}

if ($selected.ContainsKey("ed24")) {
  Invoke-N179RhythmAblation `
    -Name "ed24_ped_heavy" `
    -Noisy "D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3.npy" `
    -DatasetName "ED24" `
    -Scene "myPedestrain_06" `
    -Level "heavy" `
    -Width 346 `
    -Height 260 `
    -SList "9" `
    -TauList "256000"
}

if ($selected.ContainsKey("driving")) {
  Invoke-N179RhythmAblation `
    -Name "driving_mid" `
    -Noisy "D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving/driving_noise_mid_slomo_shot_withlabel/driving_noise_mid_labeled.npy" `
    -DatasetName "Driving" `
    -Scene "mydriving" `
    -Level "mid" `
    -Width 346 `
    -Height 260 `
    -SList "5" `
    -TauList "32000"
}

if ($selected.ContainsKey("led")) {
  Invoke-N179RhythmAblation `
    -Name "led_scene1004" `
    -Noisy "D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/scene_1004/slices_00031_00040_100ms/scene_1004_100ms_labeled.npy" `
    -DatasetName "LED" `
    -Scene "scene_1004" `
    -Level "100ms" `
    -Width 1280 `
    -Height 720 `
    -SList "7" `
    -TauList "16000"
}

Write-Host ""
Write-Host "=== DONE: N179 rhythm ablation small ==="
Write-Host "Output dir: $OutDir"
