param(
  [string]$PythonExe = "D:/software/Anaconda_envs/envs/myEVS/python.exe",
  [string]$ModelPattern = "data/ED24/myPedestrain_06/MLPF/models_retrain_20260522/mlpf_torch_{level}.pt",
  [string]$OutRoot = "data/ED24/myPedestrain_06/MLPF",
  [switch]$Dense
)

$ErrorActionPreference = "Stop"
$repo = Resolve-Path (Join-Path $PSScriptRoot "../..")
Set-Location $repo

$ed24 = "D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06"
$splits = @(
  @{ level="light";     noisy="Pedestrain_06_1.8.npy"; clean="Pedestrain_06_1.8_signal_only.npy" },
  @{ level="light_mid"; noisy="Pedestrain_06_2.1.npy"; clean="Pedestrain_06_2.1_signal_only.npy" },
  @{ level="mid";       noisy="Pedestrain_06_2.5.npy"; clean="Pedestrain_06_2.5_signal_only.npy" },
  @{ level="heavy";     noisy="Pedestrain_06_3.3.npy"; clean="Pedestrain_06_3.3_signal_only.npy" }
)

$tauList = @(32000,64000,128000,256000,512000)
$thr = if ($Dense) { "0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95" } else { "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9" }

$work = @()
foreach($sp in $splits){
  $level = $sp.level
  $model = $ModelPattern.Replace("{level}", $level)
  if(!(Test-Path $model)){ throw "Missing model: $model" }
  $clean = Join-Path $ed24 $sp.clean
  $noisy = Join-Path $ed24 $sp.noisy
  if(!(Test-Path $clean)){ throw "Missing clean: $clean" }
  if(!(Test-Path $noisy)){ throw "Missing noisy: $noisy" }

  $csv = Join-Path $OutRoot ("roc_mlpf_{0}.csv" -f $level)
  $png = Join-Path $OutRoot ("roc_mlpf_{0}.png" -f $level)
  $top = Join-Path $OutRoot ("roc_mlpf_{0}_top4_tau.csv" -f $level)
  $log = Join-Path $OutRoot ("run_mlpf_{0}_parallel.log" -f $level)
  $err = Join-Path $OutRoot ("run_mlpf_{0}_parallel.err.log" -f $level)

  if(Test-Path $csv){ Remove-Item $csv -Force }

  $tauCsv = ($tauList -join ",")
  $cmd = @"
`$ErrorActionPreference='Stop'
`$taus=@($tauCsv)
foreach(`$tau in `$taus){
  & '$PythonExe' -m myevs.cli roc --clean '$clean' --noisy '$noisy' --assume npy --width 346 --height 260 --tick-ns 1000 --engine python --method mlpf --radius-px 3 --time-us `$tau --param min-neighbors --values '$thr' --match-us 0 --match-bin-radius 0 --tag ('mlpf_tau'+`$tau) --mlpf-model '$model' --mlpf-patch 7 --out-csv '$csv' --append --progress
  if(`$LASTEXITCODE -ne 0){ throw \"roc failed level=$level tau=`$tau\" }
}
& '$PythonExe' -m myevs.cli best-tag --in '$csv' --topn 4 --by auc --out '$top'
if(`$LASTEXITCODE -ne 0){ throw 'best-tag failed' }
& '$PythonExe' -m myevs.cli plot-csv --in '$top' --out '$png' --x fpr --y tpr --group tag --kind line --xlabel FPR --ylabel TPR --title 'MLPF ROC ($level)'
if(`$LASTEXITCODE -ne 0){ throw 'plot failed' }
"@
  $ps = Start-Process -FilePath "powershell" -ArgumentList @("-NoProfile","-ExecutionPolicy","Bypass","-Command",$cmd) -RedirectStandardOutput $log -RedirectStandardError $err -PassThru
  $work += [pscustomobject]@{ Level=$level; Pid=$ps.Id; Log=$log; Err=$err; Csv=$csv }
}

Write-Host "Started 4 parallel MLPF workers:"
$work | Format-Table -AutoSize
Write-Host "Use: Get-Content $OutRoot/run_mlpf_<level>_parallel.log -Tail 50"
