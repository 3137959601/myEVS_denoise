# slomo 实验（label 精确评估版）：仅 BAF（Background Activity Filter）
# 输入：*_labeled.npy(noisy) + *_signal_only.npy(clean)
# 输出：data/ED24/myPedestrain_06/BAF/roc_baf_{light,mid,heavy}.csv/.png

# 禁止加载用户级 site-packages，避免用户目录 numpy 覆盖 conda 环境
$env:PYTHONNOUSERSITE = "1"

# conda hook 路径（若存在则激活环境）
$CONDA_HOOK = "D:/software/anaconda3/shell/condabin/conda-hook.ps1"
if (-not [string]::IsNullOrWhiteSpace($CONDA_HOOK) -and (Test-Path $CONDA_HOOK)) {
  & $CONDA_HOOK
  conda activate myEVS
}

# Python 可执行路径
$PY = "D:/software/Anaconda_envs/envs/myEVS/python.exe"
if (-not (Test-Path $PY)) {
  throw "Python not found: $PY"
}

# ED24 myPedestrain_06 数据路径（1.8/2.5/3.3 对应 light/mid/heavy）
$ED24_DIR = "D:\hjx_workspace\scientific_reserach\dataset\ED24\myPedestrain_06"
if (-not (Test-Path $ED24_DIR)) {
  throw "Dataset directory not found: $ED24_DIR"
}

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

# labeled.npy：t 已按 --tick-ns 转为 tick；这里设定 1 tick = 1us
$TICK_NS = 1000

# label 精确评估：禁用 tolerant match，严格 (t,x,y,p) 精确匹配
$MATCH_US = 0
$MATCH_BIN_RADIUS = 0

# =========================== TUNE_HERE: BAF sweep ===========================
# ED24 round1 best:
# - light : r=3, tau≈64000
# - mid   : r=1, tau≈16000
# - heavy : r=1, tau≈8000
$RADIUS_LIST = 1,2,3,4
$TAU_LIST = "20,100,200,500,1000,2000,4000,8000,16000,32000,64000,128000"
$RUNTIME_CSV = "data/ED24/myPedestrain_06/BAF/runtime_baf.csv"
"algorithm,level,start_time,end_time,elapsed_sec" | Out-File -FilePath $RUNTIME_CSV -Encoding utf8

Write-Host "=== BAF slomo: light ==="
$T0 = Get-Date
$OUT = "data/ED24/myPedestrain_06/BAF/roc_baf_light.csv"; [System.IO.File]::WriteAllText($OUT, "")
foreach ($r in $RADIUS_LIST) {
  & $PY -m myevs.cli roc --clean $LIGHT_CLEAN --noisy $LIGHT_NOISY --assume npy --width 346 --height 260 --tick-ns $TICK_NS --method baf --radius-px $r --param time-us --values $TAU_LIST --match-us $MATCH_US --match-bin-radius $MATCH_BIN_RADIUS --tag ("baf_r{0}" -f $r) --out-csv $OUT --append --progress
}
& $PY -m myevs.cli plot-csv --in $OUT --out "data/ED24/myPedestrain_06/BAF/roc_baf_light.png" --x fpr --y tpr --group tag --kind line --xlabel FPR --ylabel TPR --title "BAF ROC (light)"
$T1 = Get-Date
('{0},{1},{2},{3},{4}' -f "baf", "light", $T0.ToString("s"), $T1.ToString("s"), [Math]::Round((New-TimeSpan -Start $T0 -End $T1).TotalSeconds, 3)) | Add-Content -Path $RUNTIME_CSV -Encoding utf8

Write-Host "=== BAF slomo: mid ==="
$T0 = Get-Date
$OUT = "data/ED24/myPedestrain_06/BAF/roc_baf_mid.csv"; [System.IO.File]::WriteAllText($OUT, "")
foreach ($r in $RADIUS_LIST) {
  & $PY -m myevs.cli roc --clean $MID_CLEAN --noisy $MID_NOISY --assume npy --width 346 --height 260 --tick-ns $TICK_NS --method baf --radius-px $r --param time-us --values $TAU_LIST --match-us $MATCH_US --match-bin-radius $MATCH_BIN_RADIUS --tag ("baf_r{0}" -f $r) --out-csv $OUT --append --progress
}
& $PY -m myevs.cli plot-csv --in $OUT --out "data/ED24/myPedestrain_06/BAF/roc_baf_mid.png" --x fpr --y tpr --group tag --kind line --xlabel FPR --ylabel TPR --title "BAF ROC (mid)"
$T1 = Get-Date
('{0},{1},{2},{3},{4}' -f "baf", "mid", $T0.ToString("s"), $T1.ToString("s"), [Math]::Round((New-TimeSpan -Start $T0 -End $T1).TotalSeconds, 3)) | Add-Content -Path $RUNTIME_CSV -Encoding utf8

Write-Host "=== BAF slomo: heavy ==="
$T0 = Get-Date
$OUT = "data/ED24/myPedestrain_06/BAF/roc_baf_heavy.csv"; [System.IO.File]::WriteAllText($OUT, "")
foreach ($r in $RADIUS_LIST) {
  & $PY -m myevs.cli roc --clean $HEAVY_CLEAN --noisy $HEAVY_NOISY --assume npy --width 346 --height 260 --tick-ns $TICK_NS --method baf --radius-px $r --param time-us --values $TAU_LIST --match-us $MATCH_US --match-bin-radius $MATCH_BIN_RADIUS --tag ("baf_r{0}" -f $r) --out-csv $OUT --append --progress
}
& $PY -m myevs.cli plot-csv --in $OUT --out "data/ED24/myPedestrain_06/BAF/roc_baf_heavy.png" --x fpr --y tpr --group tag --kind line --xlabel FPR --ylabel TPR --title "BAF ROC (heavy)"
$T1 = Get-Date
('{0},{1},{2},{3},{4}' -f "baf", "heavy", $T0.ToString("s"), $T1.ToString("s"), [Math]::Round((New-TimeSpan -Start $T0 -End $T1).TotalSeconds, 3)) | Add-Content -Path $RUNTIME_CSV -Encoding utf8

Write-Host "=== DONE: BAF slomo ==="
