# slomo 实验（label 精确评估版）：EBF 2D 小范围联合扫参（s x τ）
# 目的：在 paper-style 的两步 sweep 基础上，做一个小的 2D grid，找“全局最优”的 (s, τ)
# - s ∈ {1,2,3}
# - τ ∈ {16,32,64} ms
# - threshold: min-neighbors = 0..50
# 输出：data/mydriving/EBF/roc_ebf_*_paper_2d_local_label_exact.csv + .png

$ErrorActionPreference = 'Stop'

# 重要：禁止加载用户级 site-packages（用户目录里 numpy 2.4.x 会覆盖 conda 环境 numpy 2.3.x，导致 numba 无法导入）
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

# 阈值网格：min-neighbors 更密一些，让 ROC 更平滑
$EBF_THR_LIST = (0..50) -join ','

# 2D grid（小范围）
$S_LIST = @(1, 2, 3)
$TAU_LIST_US = @(16000, 32000, 64000)

function Run-Ebf-2DLocal {
	param(
		[Parameter(Mandatory = $true)][string]$NoiseName,
		[Parameter(Mandatory = $true)][string]$CleanPath,
		[Parameter(Mandatory = $true)][string]$NoisyPath
	)

	$OUT_DIR = "data/mydriving/EBF"
	New-Item -ItemType Directory -Force -Path $OUT_DIR | Out-Null

	Write-Host "=== EBF paper 2D-local: $NoiseName | sweep s x tau ==="
	$OUT = Join-Path $OUT_DIR ("roc_ebf_{0}_paper_2d_local_label_exact.csv" -f $NoiseName)
	$needSweep = $true
	if (Test-Path $OUT) {
		try {
			if ((Get-Item $OUT).Length -gt 0) {
				$needSweep = $false
				Write-Host "resume: CSV exists and non-empty, skip sweep: $OUT"
			}
		} catch {}
	}
	if ($needSweep) {
		[System.IO.File]::WriteAllText($OUT, "")
	}

	if ($needSweep) {
		foreach ($s in $S_LIST) {
			foreach ($tau in $TAU_LIST_US) {
				& $PY -m myevs.cli roc --clean $CleanPath --noisy $NoisyPath --assume npy --width 346 --height 260 --tick-ns $TICK_NS --method ebf --radius-px $s --time-us $tau --param min-neighbors --values $EBF_THR_LIST --match-us $MATCH_US --match-bin-radius $MATCH_BIN_RADIUS --tag ("ebf_s{0}_tau{1}us_{2}" -f $s, $tau, $NoiseName) --out-csv $OUT --append --progress
			}
		}
	}

	$PNG = Join-Path $OUT_DIR ("roc_ebf_{0}_paper_2d_local.png" -f $NoiseName)
	$TITLE = "EBF ROC ($NoiseName) paper: 2D-local sweep s in {1,2,3}, tau in {16,32,64}ms"
	& $PY -m myevs.cli plot-csv --in $OUT --out $PNG --x fpr --y tpr --group tag --kind line --xlabel FPR --ylabel TPR --title $TITLE

	# 汇总 best tag（按 AUC 最大；同一 tag 每行 auc 相同，所以取首行即可）
	$best = Import-Csv $OUT | Group-Object tag | ForEach-Object { $_.Group[0] | Select-Object tag, auc } | Sort-Object { [double]$_.auc } -Descending | Select-Object -First 1
	Write-Host ("best: {0} | AUC={1}" -f $best.tag, $best.auc)
}

Run-Ebf-2DLocal -NoiseName "light" -CleanPath $LIGHT_CLEAN -NoisyPath $LIGHT_NOISY
Run-Ebf-2DLocal -NoiseName "mid"   -CleanPath $MID_CLEAN   -NoisyPath $MID_NOISY
Run-Ebf-2DLocal -NoiseName "heavy" -CleanPath $HEAVY_CLEAN -NoisyPath $HEAVY_NOISY

Write-Host "=== DONE: EBF paper 2D-local sweeps (label exact) ==="
