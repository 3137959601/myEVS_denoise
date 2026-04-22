# slomo 实验（label 精确评估版）：仅 BAF（Background Activity Filter）
# 输入：v2e labeled.txt -> labeled.npy + signal_only.npy（label==1）
# 输出：data/ED24/myPedestrain_06/BAF/roc_baf_{light,mid,heavy}.csv/.png

# 禁止加载用户级 site-packages（避免用户目录 numpy 覆盖 conda env）
$env:PYTHONNOUSERSITE = "1"

# 激活 conda 环境（确保 DLL/依赖路径正确）
$CONDA_HOOK = "D:/software/anaconda3/shell/condabin/conda-hook.ps1"
if (Test-Path $CONDA_HOOK) {
  & $CONDA_HOOK
  conda activate myEVS
}

$PY = "D:/software/Anaconda_envs/envs/myEVS/python.exe"

# ED24 myPedestrain_06 数据路径（3.3/2.5/1.8 分别映射到 light/mid/heavy）
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

# labeled.npy：t 已按 --tick-ns 转为 tick；这里沿用 DND21 设定：1 tick = 1us
$TICK_NS = 1000

# label 精确评估：禁用 tolerant match，严格 (t,x,y,p) 精确匹配
$MATCH_US = 0
$MATCH_BIN_RADIUS = 0

$S_LIST = 3,5,7,9
$TAU_LIST = "20,100,200,500,1000,2000,4000,8000,16000,32000,64000"

Write-Host "=== BAF slomo: light ==="
$OUT = "data/ED24/myPedestrain_06/BAF/roc_baf_light.csv"; [System.IO.File]::WriteAllText($OUT, "")
foreach ($s in $S_LIST) {
  $r = [int](($s - 1) / 2)
  & $PY -m myevs.cli roc --clean $LIGHT_CLEAN --noisy $LIGHT_NOISY --assume npy --width 346 --height 260 --tick-ns $TICK_NS --method baf --radius-px $r --param time-us --values $TAU_LIST --match-us $MATCH_US --match-bin-radius $MATCH_BIN_RADIUS --tag ("baf_s{0}" -f $s) --out-csv $OUT --append --progress
}
& $PY -m myevs.cli plot-csv --in $OUT --out "data/ED24/myPedestrain_06/BAF/roc_baf_light.png" --x fpr --y tpr --group tag --kind line --xlabel FPR --ylabel TPR --title "BAF ROC (light)"

Write-Host "=== BAF slomo: mid ==="
$OUT = "data/ED24/myPedestrain_06/BAF/roc_baf_mid.csv"; [System.IO.File]::WriteAllText($OUT, "")
foreach ($s in $S_LIST) {
  $r = [int](($s - 1) / 2)
  & $PY -m myevs.cli roc --clean $MID_CLEAN --noisy $MID_NOISY --assume npy --width 346 --height 260 --tick-ns $TICK_NS --method baf --radius-px $r --param time-us --values $TAU_LIST --match-us $MATCH_US --match-bin-radius $MATCH_BIN_RADIUS --tag ("baf_s{0}" -f $s) --out-csv $OUT --append --progress
}
& $PY -m myevs.cli plot-csv --in $OUT --out "data/ED24/myPedestrain_06/BAF/roc_baf_mid.png" --x fpr --y tpr --group tag --kind line --xlabel FPR --ylabel TPR --title "BAF ROC (mid)"

Write-Host "=== BAF slomo: heavy ==="
$OUT = "data/ED24/myPedestrain_06/BAF/roc_baf_heavy.csv"; [System.IO.File]::WriteAllText($OUT, "")
foreach ($s in $S_LIST) {
  $r = [int](($s - 1) / 2)
  & $PY -m myevs.cli roc --clean $HEAVY_CLEAN --noisy $HEAVY_NOISY --assume npy --width 346 --height 260 --tick-ns $TICK_NS --method baf --radius-px $r --param time-us --values $TAU_LIST --match-us $MATCH_US --match-bin-radius $MATCH_BIN_RADIUS --tag ("baf_s{0}" -f $s) --out-csv $OUT --append --progress
}
& $PY -m myevs.cli plot-csv --in $OUT --out "data/ED24/myPedestrain_06/BAF/roc_baf_heavy.png" --x fpr --y tpr --group tag --kind line --xlabel FPR --ylabel TPR --title "BAF ROC (heavy)"

Write-Host "=== DONE: BAF slomo ==="
