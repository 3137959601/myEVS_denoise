param(
  [string]$Python = "python",
  [string]$EdformerRoot = "D:\hjx_workspace\scientific_reserach\EDformer",
  [string]$BasePath = "D:\hjx_workspace\scientific_reserach\dataset\DND21\mydriving_ED24",
  [string]$Hz = "1,3,5,7,10",
  [string]$Device = "auto",
  [string]$XyMode = "official",
  [int]$MaxEvents = 0,
  [switch]$CheckEnv
)

$ErrorActionPreference = "Stop"
$Script = Join-Path $PSScriptRoot "eval_official_driving_auc.py"
$OutDir = "data/DND21/edformer_official_auc"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$argsList = @(
  $Script,
  "--edformer-root", $EdformerRoot,
  "--base-path", $BasePath,
  "--hz", $Hz,
  "--filename", "driving_mix_result.txt",
  "--device", $Device,
  "--xy-mode", $XyMode,
  "--skip-missing",
  "--out-csv", (Join-Path $OutDir ("driving_auc_{0}.csv" -f $XyMode)),
  "--out-json", (Join-Path $OutDir ("driving_auc_{0}_env.json" -f $XyMode))
)

if ($MaxEvents -gt 0) {
  $argsList += @("--max-events", [string]$MaxEvents)
}
if ($CheckEnv) {
  $argsList += "--check-env"
}

& $Python @argsList
if ($LASTEXITCODE -ne 0) {
  throw "EDformer official Driving AUC failed with exit code $LASTEXITCODE"
}
