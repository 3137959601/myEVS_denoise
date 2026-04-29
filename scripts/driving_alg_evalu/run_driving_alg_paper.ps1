param(
  [string]$Algorithm = "",
  [string[]]$Algorithms = @(),
  [int]$MaxEvents = 200000,
  [ValidateSet("coarse", "dense")]
  [string]$SweepProfile = "coarse"
)

& "$PSScriptRoot\run_driving_alg.ps1" `
  -Algorithm $Algorithm `
  -Algorithms $Algorithms `
  -MaxEvents $MaxEvents `
  -SweepProfile $SweepProfile `
  -DatasetRoot "D:\hjx_workspace\scientific_reserach\dataset\DND21\mydriving_paper" `
  -DatasetSuffix "paper_withlabel"

