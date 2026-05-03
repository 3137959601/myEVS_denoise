param(
  [string]$PythonExe = "D:/software/Anaconda_envs/envs/myEVS/python.exe",
  [switch]$EvflowLite = $true
)

$ErrorActionPreference = "Stop"

$pairs = @(
  @{ Scene = "MAH00444"; Level = "ratio50" },
  @{ Scene = "MAH00444"; Level = "ratio100" },
  @{ Scene = "MAH00446"; Level = "ratio50" },
  @{ Scene = "MAH00446"; Level = "ratio100" },
  @{ Scene = "MAH00447"; Level = "ratio50" },
  @{ Scene = "MAH00447"; Level = "ratio100" },
  @{ Scene = "MAH00448"; Level = "ratio50" },
  @{ Scene = "MAH00448"; Level = "ratio100" },
  @{ Scene = "MAH00449"; Level = "ratio50" },
  @{ Scene = "MAH00449"; Level = "ratio100" }
)

for ($i = 0; $i -lt $pairs.Count; $i++) {
  $s = $pairs[$i].Scene
  $l = $pairs[$i].Level
  Write-Host "=== [$($i+1)/$($pairs.Count)] DVSCLEAN $s $l ==="
  $args = @(
    "-ExecutionPolicy", "Bypass",
    "-File", "$PSScriptRoot/run_dvsclean_one.ps1",
    "-Scene", $s,
    "-Level", $l,
    "-PythonExe", $PythonExe
  )
  if ($EvflowLite) { $args += "-EvflowLite" }
  powershell @args
  if ($LASTEXITCODE -ne 0) { throw "Failed at $s $l" }
}

Write-Host "=== DVSCLEAN all full sweep done ==="
