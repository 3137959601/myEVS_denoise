# slomo 实验（label 精确评估版）：仅 EBF (score + threshold) 调参
# 目标：先选最优 (radius,tau)，再和其它算法对比
# 输入：*_labeled.npy（noisy） + *_signal_only.npy（clean, label==1）
# 输出：data/mydriving/EBF/roc_ebf_{light,mid,heavy}_label_exact.csv/.png

# 重要：禁止加载用户级 site-packages（你的用户目录里有 numpy 2.4.x，会覆盖 conda 环境的 numpy 2.3.x，导致 numba 无法导入）
$env:PYTHONNOUSERSITE = "1"

# 确保 conda 环境 DLL 路径正确（避免直接调用 python.exe 时 numpy 等扩展因缺 DLL 崩溃）
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

# aedat2 时间单位：1 tick = 1us
$TICK_NS = 1000

# label 精确评估：禁用 tolerant match，严格 (t,x,y,p) 精确匹配
$MATCH_US = 0
$MATCH_BIN_RADIUS = 0

# 注意：EBF sweep 会在可用时自动使用 numba backend；为了控制耗时，这里仍默认用一个较小的网格。
$EBF_RADIUS_LIST = 1,2,3,4
$EBF_TAU_LIST = 1000,2000,4000
$EBF_THR_LIST = "0,2,4,6,8,10,12,14,16,18,20,22,24"

function Show-TopAuc {
	param(
		[Parameter(Mandatory = $true)][string]$CsvPath,
		[int]$TopN = 10
	)

	Write-Host "--- Top AUC: $CsvPath ---"
	Import-Csv $CsvPath |
		Group-Object tag |
		ForEach-Object { $_.Group[0] | Select-Object tag, auc } |
		Sort-Object { [double]$_.auc } -Descending |
		Select-Object -First $TopN |
		Format-Table -AutoSize
}

Write-Host "=== EBF (ebf) slomo: light (sweep r/tau) ==="
$OUT = "data/mydriving/EBF/roc_ebf_light_label_exact.csv"; [System.IO.File]::WriteAllText($OUT, "")
foreach ($r in $EBF_RADIUS_LIST) {
	foreach ($tau in $EBF_TAU_LIST) {
		& $PY -m myevs.cli roc --clean $LIGHT_CLEAN --noisy $LIGHT_NOISY --assume npy --width 346 --height 260 --tick-ns $TICK_NS --method ebf --radius-px $r --time-us $tau --param min-neighbors --values $EBF_THR_LIST --match-us $MATCH_US --match-bin-radius $MATCH_BIN_RADIUS --tag ("ebf_r{0}_tau{1}_light" -f $r, $tau) --out-csv $OUT --append --progress
	}
}
& $PY -m myevs.cli plot-csv --in $OUT --out "data/mydriving/EBF/roc_ebf_light_label_exact.png" --x fpr --y tpr --group tag --kind line --xlabel FPR --ylabel TPR --title "EBF ROC (light, label exact)"
Show-TopAuc -CsvPath $OUT -TopN 10

Write-Host "=== EBF (ebf) slomo: mid (sweep r/tau) ==="
$OUT = "data/mydriving/EBF/roc_ebf_mid_label_exact.csv"; [System.IO.File]::WriteAllText($OUT, "")
foreach ($r in $EBF_RADIUS_LIST) {
	foreach ($tau in $EBF_TAU_LIST) {
		& $PY -m myevs.cli roc --clean $MID_CLEAN --noisy $MID_NOISY --assume npy --width 346 --height 260 --tick-ns $TICK_NS --method ebf --radius-px $r --time-us $tau --param min-neighbors --values $EBF_THR_LIST --match-us $MATCH_US --match-bin-radius $MATCH_BIN_RADIUS --tag ("ebf_r{0}_tau{1}_mid" -f $r, $tau) --out-csv $OUT --append --progress
	}
}
& $PY -m myevs.cli plot-csv --in $OUT --out "data/mydriving/EBF/roc_ebf_mid_label_exact.png" --x fpr --y tpr --group tag --kind line --xlabel FPR --ylabel TPR --title "EBF ROC (mid, label exact)"
Show-TopAuc -CsvPath $OUT -TopN 10

Write-Host "=== EBF (ebf) slomo: heavy (sweep r/tau) ==="
$OUT = "data/mydriving/EBF/roc_ebf_heavy_label_exact.csv"; [System.IO.File]::WriteAllText($OUT, "")
foreach ($r in $EBF_RADIUS_LIST) {
	foreach ($tau in $EBF_TAU_LIST) {
		& $PY -m myevs.cli roc --clean $HEAVY_CLEAN --noisy $HEAVY_NOISY --assume npy --width 346 --height 260 --tick-ns $TICK_NS --method ebf --radius-px $r --time-us $tau --param min-neighbors --values $EBF_THR_LIST --match-us $MATCH_US --match-bin-radius $MATCH_BIN_RADIUS --tag ("ebf_r{0}_tau{1}_heavy" -f $r, $tau) --out-csv $OUT --append --progress
	}
}
& $PY -m myevs.cli plot-csv --in $OUT --out "data/mydriving/EBF/roc_ebf_heavy_label_exact.png" --x fpr --y tpr --group tag --kind line --xlabel FPR --ylabel TPR --title "EBF ROC (heavy, label exact)"
Show-TopAuc -CsvPath $OUT -TopN 10

Write-Host "=== DONE: EBF slomo sweep r/tau ==="
