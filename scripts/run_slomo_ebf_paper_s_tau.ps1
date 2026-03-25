# slomo 实验（label 精确评估版）：EBF 论文参数复现（s 与 τ 对 AUC 的影响）
# - Step A: 固定 τ=64ms，扫 s
# - Step B: 固定 s=5，扫 τ=16/32/64/128ms
# 阈值：min-neighbors 加密取样以获得更平滑的 ROC
# 输出：data/mydriving/EBF/roc_ebf_*_paper_*.csv + 仅 TopN 曲线的 ROC 图

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

# 时间单位：1 tick = 1us
$TICK_NS = 1000

# label 精确评估：禁用 tolerant match，严格 (t,x,y,p) 精确匹配
$MATCH_US = 0
$MATCH_BIN_RADIUS = 0

# 阈值网格（更密一些，让 ROC 更平滑）
# 注：min-neighbors 是整数语义；这里按 0..50 全覆盖，基本能覆盖 (0,0) 到 (1,1)
$EBF_THR_LIST = (0..50) -join ','

# 论文：s sweep 固定 τ=64ms；τ sweep 固定 s=5
$TAU_FIXED_US = 64000
$S_LIST = 1..8
$S_FIXED = 5
$TAU_LIST_US = @(16000, 32000, 64000, 128000)

function Get-TopTags {
	param(
		[Parameter(Mandatory = $true)][string]$CsvPath,
		[int]$TopN = 10
	)

	$rows = Import-Csv $CsvPath
	$top = $rows |
		Group-Object tag |
		ForEach-Object { $_.Group[0] | Select-Object tag, auc } |
		Sort-Object { [double]$_.auc } -Descending |
		Select-Object -First $TopN

	return @($top | ForEach-Object { $_.tag })
}

function Export-FilteredCsv {
	param(
		[Parameter(Mandatory = $true)][string]$InCsv,
		[Parameter(Mandatory = $true)][string[]]$Tags,
		[Parameter(Mandatory = $true)][string]$OutCsv
	)

	$set = @{}
	foreach ($t in $Tags) { $set[$t] = $true }

	Import-Csv $InCsv |
		Where-Object { $set.ContainsKey($_.tag) } |
		Export-Csv -NoTypeInformation -Encoding UTF8 $OutCsv
}

function Run-Ebf-For-Noise {
	param(
		[Parameter(Mandatory = $true)][string]$NoiseName,
		[Parameter(Mandatory = $true)][string]$CleanPath,
		[Parameter(Mandatory = $true)][string]$NoisyPath,
		[int]$TopN = 10
	)

	$OUT_DIR = "data/mydriving/EBF"
	New-Item -ItemType Directory -Force -Path $OUT_DIR | Out-Null

	Write-Host "=== EBF paper: $NoiseName | Step A: sweep s with tau=64ms ==="
	$OUT_A = Join-Path $OUT_DIR ("roc_ebf_{0}_paper_sweep_s_tau64ms_label_exact.csv" -f $NoiseName)
	[System.IO.File]::WriteAllText($OUT_A, "")

	foreach ($s in $S_LIST) {
		& $PY -m myevs.cli roc --clean $CleanPath --noisy $NoisyPath --assume npy --width 346 --height 260 --tick-ns $TICK_NS --method ebf --radius-px $s --time-us $TAU_FIXED_US --param min-neighbors --values $EBF_THR_LIST --match-us $MATCH_US --match-bin-radius $MATCH_BIN_RADIUS --tag ("ebf_s{0}_tau64ms_{1}" -f $s, $NoiseName) --out-csv $OUT_A --append --progress
	}

	$TOP_TAGS_A = Get-TopTags -CsvPath $OUT_A -TopN $TopN
	$OUT_A_TOP = Join-Path $OUT_DIR ("roc_ebf_{0}_paper_sweep_s_tau64ms_top{1}.csv" -f $NoiseName, $TopN)
	Export-FilteredCsv -InCsv $OUT_A -Tags $TOP_TAGS_A -OutCsv $OUT_A_TOP
	& $PY -m myevs.cli plot-csv --in $OUT_A_TOP --out (Join-Path $OUT_DIR ("roc_ebf_{0}_paper_sweep_s_tau64ms_top{1}.png" -f $NoiseName, $TopN)) --x fpr --y tpr --group tag --kind line --xlabel FPR --ylabel TPR --title ("EBF ROC ({0}) paper: sweep s @ tau=64ms (top{1})" -f $NoiseName, $TopN)

	Write-Host "=== EBF paper: $NoiseName | Step B: sweep tau with s=5 ==="
	$OUT_B = Join-Path $OUT_DIR ("roc_ebf_{0}_paper_s5_sweep_tau_label_exact.csv" -f $NoiseName)
	[System.IO.File]::WriteAllText($OUT_B, "")

	foreach ($tau in $TAU_LIST_US) {
		& $PY -m myevs.cli roc --clean $CleanPath --noisy $NoisyPath --assume npy --width 346 --height 260 --tick-ns $TICK_NS --method ebf --radius-px $S_FIXED --time-us $tau --param min-neighbors --values $EBF_THR_LIST --match-us $MATCH_US --match-bin-radius $MATCH_BIN_RADIUS --tag ("ebf_s5_tau{0}us_{1}" -f $tau, $NoiseName) --out-csv $OUT_B --append --progress
	}

	$TOP_TAGS_B = Get-TopTags -CsvPath $OUT_B -TopN ([Math]::Min($TopN, $TAU_LIST_US.Count))
	$OUT_B_TOP = Join-Path $OUT_DIR ("roc_ebf_{0}_paper_s5_sweep_tau_top{1}.csv" -f $NoiseName, [Math]::Min($TopN, $TAU_LIST_US.Count))
	Export-FilteredCsv -InCsv $OUT_B -Tags $TOP_TAGS_B -OutCsv $OUT_B_TOP
	& $PY -m myevs.cli plot-csv --in $OUT_B_TOP --out (Join-Path $OUT_DIR ("roc_ebf_{0}_paper_s5_sweep_tau_top{1}.png" -f $NoiseName, [Math]::Min($TopN, $TAU_LIST_US.Count))) --x fpr --y tpr --group tag --kind line --xlabel FPR --ylabel TPR --title ("EBF ROC ({0}) paper: sweep tau @ s=5 (top{1})" -f $NoiseName, [Math]::Min($TopN, $TAU_LIST_US.Count))
}

Run-Ebf-For-Noise -NoiseName "light" -CleanPath $LIGHT_CLEAN -NoisyPath $LIGHT_NOISY -TopN 8
Run-Ebf-For-Noise -NoiseName "mid"   -CleanPath $MID_CLEAN   -NoisyPath $MID_NOISY   -TopN 8
Run-Ebf-For-Noise -NoiseName "heavy" -CleanPath $HEAVY_CLEAN -NoisyPath $HEAVY_NOISY -TopN 8

Write-Host "=== DONE: EBF paper s/tau sweeps (label exact) ==="
