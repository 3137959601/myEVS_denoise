# slomo 实验（label 精确评估版）：仅 STCF（STC）
# 输入：*_labeled.npy（noisy） + *_signal_only.npy（clean, label==1）
# 输出：data/ED24/myPedestrain_06/STCF/roc_stcf_{light,mid,heavy}.csv/.png

$env:PYTHONNOUSERSITE = "1"

$CONDA_HOOK = "D:/software/anaconda3/shell/condabin/conda-hook.ps1"
if (Test-Path $CONDA_HOOK) {
  & $CONDA_HOOK
  conda activate myEVS
}

$PY = "D:/software/Anaconda_envs/envs/myEVS/python.exe"

$ED24_DIR = "D:\hjx_workspace\scientific_reserach\dataset\ED24\myPedestrain_06"

$LIGHT_NOISY = Join-Path $ED24_DIR "Pedestrain_06_1.8.npy"
$MID_NOISY   = Join-Path $ED24_DIR "Pedestrain_06_2.5.npy"
$HEAVY_NOISY = Join-Path $ED24_DIR "Pedestrain_06_3.3.npy"

$LIGHT_CLEAN = Join-Path $ED24_DIR "Pedestrain_06_1.8_signal_only.npy"
$MID_CLEAN   = Join-Path $ED24_DIR "Pedestrain_06_2.5_signal_only.npy"
$HEAVY_CLEAN = Join-Path $ED24_DIR "Pedestrain_06_3.3_signal_only.npy"

foreach ($p in @($LIGHT_NOISY,$MID_NOISY,$HEAVY_NOISY,$LIGHT_CLEAN,$MID_CLEAN,$HEAVY_CLEAN)) {
  if (!(Test-Path $p)) {
    throw "Missing required file: $p (generate via v2e_labeled_txt_to_npy.py + split_labeled_events.py)"
  }
}

$TICK_NS = 1000
$MATCH_US = 0
$MATCH_BIN_RADIUS = 0

# ========================== TUNE_HERE: STCF sweep ==========================
# ED24 round1 best:
# - light : r=4, tau≈256000
# - mid   : r=3, tau≈16000
# - heavy : r=2, tau≈16000
$RADIUS_LIST = 1,2,3,4
$TAU_LIST = "100,200,500,1000,2000,4000,8000,16000,32000,64000,128000,256000,512000"
$RUNTIME_CSV = "data/ED24/myPedestrain_06/STCF/runtime_stcf.csv"
"algorithm,level,start_time,end_time,elapsed_sec" | Out-File -FilePath $RUNTIME_CSV -Encoding utf8

Write-Host "=== STCF slomo: light ==="
$T0 = Get-Date
$OUT = "data/ED24/myPedestrain_06/STCF/roc_stcf_light.csv"; [System.IO.File]::WriteAllText($OUT, "")
foreach ($r in $RADIUS_LIST) {
  & $PY -m myevs.cli roc --clean $LIGHT_CLEAN --noisy $LIGHT_NOISY --assume npy --width 346 --height 260 --tick-ns $TICK_NS --method stc --radius-px $r --param time-us --values $TAU_LIST --match-us $MATCH_US --match-bin-radius $MATCH_BIN_RADIUS --tag ("stcf_r{0}" -f $r) --out-csv $OUT --append --progress
}
& $PY -m myevs.cli plot-csv --in $OUT --out "data/ED24/myPedestrain_06/STCF/roc_stcf_light.png" --x fpr --y tpr --group tag --kind line --xlabel FPR --ylabel TPR --title "STCF ROC (light)"
$T1 = Get-Date
('{0},{1},{2},{3},{4}' -f "stcf", "light", $T0.ToString("s"), $T1.ToString("s"), [Math]::Round((New-TimeSpan -Start $T0 -End $T1).TotalSeconds, 3)) | Add-Content -Path $RUNTIME_CSV -Encoding utf8

Write-Host "=== STCF slomo: mid ==="
$T0 = Get-Date
$OUT = "data/ED24/myPedestrain_06/STCF/roc_stcf_mid.csv"; [System.IO.File]::WriteAllText($OUT, "")
foreach ($r in $RADIUS_LIST) {
  & $PY -m myevs.cli roc --clean $MID_CLEAN --noisy $MID_NOISY --assume npy --width 346 --height 260 --tick-ns $TICK_NS --method stc --radius-px $r --param time-us --values $TAU_LIST --match-us $MATCH_US --match-bin-radius $MATCH_BIN_RADIUS --tag ("stcf_r{0}" -f $r) --out-csv $OUT --append --progress
}
& $PY -m myevs.cli plot-csv --in $OUT --out "data/ED24/myPedestrain_06/STCF/roc_stcf_mid.png" --x fpr --y tpr --group tag --kind line --xlabel FPR --ylabel TPR --title "STCF ROC (mid)"
$T1 = Get-Date
('{0},{1},{2},{3},{4}' -f "stcf", "mid", $T0.ToString("s"), $T1.ToString("s"), [Math]::Round((New-TimeSpan -Start $T0 -End $T1).TotalSeconds, 3)) | Add-Content -Path $RUNTIME_CSV -Encoding utf8

Write-Host "=== STCF slomo: heavy ==="
$T0 = Get-Date
$OUT = "data/ED24/myPedestrain_06/STCF/roc_stcf_heavy.csv"; [System.IO.File]::WriteAllText($OUT, "")
foreach ($r in $RADIUS_LIST) {
  & $PY -m myevs.cli roc --clean $HEAVY_CLEAN --noisy $HEAVY_NOISY --assume npy --width 346 --height 260 --tick-ns $TICK_NS --method stc --radius-px $r --param time-us --values $TAU_LIST --match-us $MATCH_US --match-bin-radius $MATCH_BIN_RADIUS --tag ("stcf_r{0}" -f $r) --out-csv $OUT --append --progress
}
& $PY -m myevs.cli plot-csv --in $OUT --out "data/ED24/myPedestrain_06/STCF/roc_stcf_heavy.png" --x fpr --y tpr --group tag --kind line --xlabel FPR --ylabel TPR --title "STCF ROC (heavy)"
$T1 = Get-Date
('{0},{1},{2},{3},{4}' -f "stcf", "heavy", $T0.ToString("s"), $T1.ToString("s"), [Math]::Round((New-TimeSpan -Start $T0 -End $T1).TotalSeconds, 3)) | Add-Content -Path $RUNTIME_CSV -Encoding utf8

Write-Host "=== DONE: STCF slomo ==="
