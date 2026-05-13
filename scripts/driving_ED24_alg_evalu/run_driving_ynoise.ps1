param(
  [ValidateSet("auto", "python", "numba", "cpp")]
  [string]$Engine = "auto",
  [int]$MaxEvents = 0,
  [ValidateSet("coarse", "dense")]
  [string]$SweepProfile = "coarse"
)

& "$PSScriptRoot\run_driving_alg.ps1" -Algorithm "ynoise" -Engine $Engine -MaxEvents $MaxEvents -SweepProfile $SweepProfile
