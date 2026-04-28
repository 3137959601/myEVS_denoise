param(
  [string]$MlpfModelPattern = "data/DND21/mydriving/MLPF/mlpf_torch_{level}.pt",
  [ValidateSet("coarse", "dense")]
  [string]$SweepProfile = "coarse",
  [int]$MaxEvents = 200000
)

& "$PSScriptRoot\run_driving_alg.ps1" `
  -Algorithm "mlpf" `
  -MlpfModelPattern $MlpfModelPattern `
  -SweepProfile $SweepProfile `
  -MaxEvents $MaxEvents
