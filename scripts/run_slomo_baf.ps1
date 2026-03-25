# slomo 实验（label 精确评估版）：仅 BAF（Background Activity Filter）
# 输入：v2e labeled.txt -> labeled.npy + signal_only.npy（label==1）
# 输出：data/mydriving/BAF/roc_baf_{light,mid,heavy}_label_exact.csv/.png

# 禁止加载用户级 site-packages（避免用户目录 numpy 覆盖 conda env）
$env:PYTHONNOUSERSITE = "1"

# 激活 conda 环境（确保 DLL/依赖路径正确）
$CONDA_HOOK = "D:/software/anaconda3/shell/condabin/conda-hook.ps1"
if (Test-Path $CONDA_HOOK) {
  & $CONDA_HOOK
  conda activate myEVS
}

$PY = "D:/software/Anaconda_envs/envs/myEVS/python.exe"

# DND21 slomo(label) 数据路径（你已用 split_labeled_events.py 生成 signal-only）
$LIGHT_DIR = "D:\hjx_workspace\scientific_reserach\dataset\DND21\mydriving\driving_noise_light_slomo_shot_withlabel"
$MID_DIR   = "D:\hjx_workspace\scientific_reserach\dataset\DND21\mydriving\driving_noise_mid_slomo_shot_withlabel"
$HEAVY_DIR = "D:\hjx_workspace\scientific_reserach\dataset\DND21\mydriving\driving_noise_heavy_slomo_shot_withlabel"

$LIGHT_NOISY = Join-Path $LIGHT_DIR "driving_noise_light_labeled.npy"
$MID_NOISY   = Join-Path $MID_DIR   "driving_noise_mid_labeled.npy"
$HEAVY_NOISY = Join-Path $HEAVY_DIR "driving_noise_heavy_labeled.npy"

$LIGHT_CLEAN = Join-Path $LIGHT_DIR "driving_noise_light_signal_only.npy"
$MID_CLEAN   = Join-Path $MID_DIR   "driving_noise_mid_signal_only.npy"
$HEAVY_CLEAN = Join-Path $HEAVY_DIR "driving_noise_heavy_signal_only.npy"

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

$RADIUS_LIST = 1,2,3,4,6
$TAU_LIST = "20,100,200,500,1000,2000,4000,8000"

Write-Host "=== BAF slomo: light ==="
$OUT = "data/mydriving/BAF/roc_baf_light_label_exact.csv"; [System.IO.File]::WriteAllText($OUT, "")
foreach ($r in $RADIUS_LIST) {
  & $PY -m myevs.cli roc --clean $LIGHT_CLEAN --noisy $LIGHT_NOISY --assume npy --width 346 --height 260 --tick-ns $TICK_NS --method baf --radius-px $r --param time-us --values $TAU_LIST --match-us $MATCH_US --match-bin-radius $MATCH_BIN_RADIUS --tag ("baf_r{0}_light" -f $r) --out-csv $OUT --append --progress
}
& $PY -m myevs.cli plot-csv --in $OUT --out "data/mydriving/BAF/roc_baf_light_label_exact.png" --x fpr --y tpr --group tag --kind line --xlabel FPR --ylabel TPR --title "BAF ROC (light, label exact)"

Write-Host "=== BAF slomo: mid ==="
$OUT = "data/mydriving/BAF/roc_baf_mid_label_exact.csv"; [System.IO.File]::WriteAllText($OUT, "")
foreach ($r in $RADIUS_LIST) {
  & $PY -m myevs.cli roc --clean $MID_CLEAN --noisy $MID_NOISY --assume npy --width 346 --height 260 --tick-ns $TICK_NS --method baf --radius-px $r --param time-us --values $TAU_LIST --match-us $MATCH_US --match-bin-radius $MATCH_BIN_RADIUS --tag ("baf_r{0}_mid" -f $r) --out-csv $OUT --append --progress
}
& $PY -m myevs.cli plot-csv --in $OUT --out "data/mydriving/BAF/roc_baf_mid_label_exact.png" --x fpr --y tpr --group tag --kind line --xlabel FPR --ylabel TPR --title "BAF ROC (mid, label exact)"

Write-Host "=== BAF slomo: heavy ==="
$OUT = "data/mydriving/BAF/roc_baf_heavy_label_exact.csv"; [System.IO.File]::WriteAllText($OUT, "")
foreach ($r in $RADIUS_LIST) {
  & $PY -m myevs.cli roc --clean $HEAVY_CLEAN --noisy $HEAVY_NOISY --assume npy --width 346 --height 260 --tick-ns $TICK_NS --method baf --radius-px $r --param time-us --values $TAU_LIST --match-us $MATCH_US --match-bin-radius $MATCH_BIN_RADIUS --tag ("baf_r{0}_heavy" -f $r) --out-csv $OUT --append --progress
}
& $PY -m myevs.cli plot-csv --in $OUT --out "data/mydriving/BAF/roc_baf_heavy_label_exact.png" --x fpr --y tpr --group tag --kind line --xlabel FPR --ylabel TPR --title "BAF ROC (heavy, label exact)"

Write-Host "=== DONE: BAF slomo ==="
