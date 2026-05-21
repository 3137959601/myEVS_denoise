param(
  [string]$Algorithm = "",
  [string[]]$Algorithms = @(),
  [string]$MlpfModelPattern = "",
  [int]$MaxEvents = 0,
  [ValidateSet("auto", "python", "numba", "cpp")]
  [string]$Engine = "auto",
  [ValidateSet("coarse", "dense")]
  [string]$SweepProfile = "coarse"
)

# Historical filename kept for compatibility. In driving_ED24_alg_evalu this wrapper
# targets the ED24-provided DND21 driving dataset and follows the same caliber as
# run_driving_alg.ps1 (currently 1hz/2hz/3hz/5hz/8hz; 7hz/10hz reserved).
$MainScript = Join-Path $PSScriptRoot "run_driving_alg.ps1"
& $MainScript `
  -Algorithm $Algorithm `
  -Algorithms $Algorithms `
  -MlpfModelPattern $MlpfModelPattern `
  -MaxEvents $MaxEvents `
  -Engine $Engine `
  -SweepProfile $SweepProfile `
  -DatasetRoot "D:\hjx_workspace\scientific_reserach\dataset\DND21\mydriving_ED24" `
  -DatasetSuffix "ed24_withlabel"
