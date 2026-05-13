param(
  [ValidateSet("auto", "python", "numba", "cpp")]
  [string]$Engine = "auto",
  [int]$MaxEvents = 0,
  [ValidateSet("coarse", "dense")]
  [string]$SweepProfile = "coarse"
)

$MainScript = Join-Path $PSScriptRoot "run_driving_alg.ps1"
& $MainScript -Algorithm "pfd" -Engine $Engine -MaxEvents $MaxEvents -SweepProfile $SweepProfile
