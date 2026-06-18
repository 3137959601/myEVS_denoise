param(
  [string]$Datasets = "all",
  [string]$Algorithms = "baf,stcf_orig,pfd,ebf,ynoise,ts,n149_v22",
  [int]$Workers = 8,
  [switch]$Force
)

$ErrorActionPreference = "Stop"
$env:PYTHONNOUSITE = "1"
$env:PYTHONNOUSERSITE = "1"

$PY = "D:/software/Anaconda_envs/envs/myEVS/python.exe"
$ROOT = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$SCRIPT = Join-Path $ROOT "scripts/chapter7_recompute_f1.py"

if (!(Test-Path $SCRIPT)) {
  throw "Missing script: $SCRIPT"
}

Write-Host "Chapter 7 F1 recompute"
Write-Host "  datasets   = $Datasets"
Write-Host "  algorithms = $Algorithms"
Write-Host "  workers    = $Workers"
Write-Host "  output     = data/chapter7_f1_recompute/"
Write-Host ""
Write-Host "Methodology:"
Write-Host "  - Driving uses only 1hz, 3hz, 5hz, 7hz, 10hz."
Write-Host "  - BAF, STCF_orig and PFD are fixed to r=1."
Write-Host "  - MLPF, EvFlow, KNoise and STCF are excluded unless the Python script is extended."
Write-Host "  - Summary selects the best-AUC tag first, then reports max F1 within that tag."
Write-Host ""

$argsList = @(
  $SCRIPT,
  "--datasets", $Datasets,
  "--algorithms", $Algorithms,
  "--workers", "$Workers"
)

if ($Force) {
  $argsList += "--force"
}

& $PY @argsList
if ($LASTEXITCODE -ne 0) {
  throw "chapter7_recompute_f1.py failed with exit code $LASTEXITCODE"
}
