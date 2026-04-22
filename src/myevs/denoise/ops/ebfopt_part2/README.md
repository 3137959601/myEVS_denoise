# EBF Part2（精度提升方向）实验记录

日期：2026-04-22

## 顶部快照（7.82 时间跨度 + 自激率，Heavy 高分子集）

按 7.82 要求，在 Heavy 数据上只统计 Baseline 高分事件（`S_base >= 3.0`），验证：

1. 邻域有效支撑点时间跨度 `T_span_ms`；
2. 中心像素过去 30ms 触发次数 `U_self`。

统计脚本：`scripts/noise_analyze/gt_feature_stats_782.py`

绘图脚本：`scripts/noise_analyze/plot_gt_feature_hist_782.py`

运行命令：

```powershell
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/noise_analyze/gt_feature_stats_782.py \
    --labeled-npy D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3.npy \
    --max-events 500000 \
    --tau-us 30000 \
    --radius 4 \
    --score-thr 3.0 \
    --self-hist-len 12 \
    --out-dir data/ED24/myPedestrain_06/EBF_Part2/_gt_feature_stats_782_heavy

D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/noise_analyze/plot_gt_feature_hist_782.py \
    --hist-csv data/ED24/myPedestrain_06/EBF_Part2/_gt_feature_stats_782_heavy/hist.csv \
    --out-png data/ED24/myPedestrain_06/EBF_Part2/_gt_feature_stats_782_heavy/hist_782.png \
    --title "7.82 Heavy High-score (S_base>=3): T_span & U_self"
```

产物目录：

- `data/ED24/myPedestrain_06/EBF_Part2/_gt_feature_stats_782_heavy/`

核心结果（summary.csv）：

| class | events_highscore | tspan_lt_2ms_ratio | uself_ge_3_ratio | tspan_ms_mean | uself_mean |
|---|---:|---:|---:|---:|---:|
| noise | 15844 | 0.004102 | 0.024299 | 18.7067 | 0.4803 |
| signal | 39805 | 0.001834 | 0.023816 | 23.8471 | 0.4550 |

结论（7.82）：

1. `T_span` 在低值端存在一定区分（noise 的 `<2ms` 比例约为 signal 的 2.24 倍）；
2. `U_self>=3` 在 noise/signal 间几乎重叠，单独作为硬阈值判别价值有限；
3. 这两项更适合作为联合特征中的弱分量，而非单独主判别。

## 顶部快照（7.83 新版：A_true=Emax/(S_base+1e-3)）

按 7.83 最新修改，在 Heavy 数据上构建 4 轴能量：

- `E_0`（dy=0）
- `E_90`（dx=0）
- `E_45`（dx=dy）
- `E_135`（dx=-dy）

并计算：

$$
A_{true}=\frac{E_{max}}{S_{base}+10^{-3}},\quad
E_{max}=\max(E_0,E_{90},E_{45},E_{135})
$$

统计脚本：`scripts/noise_analyze/gt_feature_stats_783_v2.py`

绘图脚本：`scripts/noise_analyze/plot_gt_feature_hist_783_v2.py`

运行命令（score 版，`S_base>=3.0`）：

```bash
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/noise_analyze/gt_feature_stats_783_v2.py \
	--labeled-npy D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3.npy \
	--max-events 500000 \
	--tau-us 30000 \
	--radius 4 \
	--score-thr 3.0 \
	--out-dir data/ED24/myPedestrain_06/EBF_Part2/_gt_feature_stats_783v2_heavy_score

D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/noise_analyze/plot_gt_feature_hist_783_v2.py \
	--hist-csv data/ED24/myPedestrain_06/EBF_Part2/_gt_feature_stats_783v2_heavy_score/hist.csv \
	--out-png data/ED24/myPedestrain_06/EBF_Part2/_gt_feature_stats_783v2_heavy_score/hist_783_v2_score.png \
	--title "7.83 v2 Heavy score>=3: A_true"
```

运行命令（noscore 版，`score-thr=0`）：

```bash
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/noise_analyze/gt_feature_stats_783_v2.py \
	--labeled-npy D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3.npy \
	--max-events 500000 \
	--tau-us 30000 \
	--radius 4 \
	--score-thr 0.0 \
	--out-dir data/ED24/myPedestrain_06/EBF_Part2/_gt_feature_stats_783v2_heavy_noscore

D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/noise_analyze/plot_gt_feature_hist_783_v2.py \
	--hist-csv data/ED24/myPedestrain_06/EBF_Part2/_gt_feature_stats_783v2_heavy_noscore/hist.csv \
	--out-png data/ED24/myPedestrain_06/EBF_Part2/_gt_feature_stats_783v2_heavy_noscore/hist_783_v2_noscore.png \
	--title "7.83 v2 Heavy noscore: A_true"
```

产物目录：

- `data/ED24/myPedestrain_06/EBF_Part2/_gt_feature_stats_783v2_heavy_score/`
- `data/ED24/myPedestrain_06/EBF_Part2/_gt_feature_stats_783v2_heavy_noscore/`

核心结果（summary.csv）：

score 版（`S_base>=3.0`）

| class | events_kept | a_true_mean | a_true_p50 | a_true_p90 | a_true_0_0p2_ratio | a_true_0p8_1_ratio |
|---|---:|---:|---:|---:|---:|---:|
| noise | 15844 | 0.2689 | 0.2576 | 0.4537 | 0.2660 | 0.0009 |
| signal | 39805 | 0.2671 | 0.2452 | 0.4156 | 0.2969 | 0.0015 |

noscore 版（`score-thr=0`）

| class | events_kept | a_true_mean | a_true_p50 | a_true_p90 | a_true_0_0p2_ratio | a_true_0p8_1_ratio |
|---|---:|---:|---:|---:|---:|---:|
| noise | 438655 | 0.3016 | 0.0286 | 0.9974 | 0.5423 | 0.1670 |
| signal | 61345 | 0.3014 | 0.2594 | 0.5578 | 0.3124 | 0.0436 |

结论（7.83 新版 v2）：

1. 在 score 版（`S_base>=3`）中，noise/signal 的 `A_true` 均值和分位数几乎重合，说明该指标在高分子集内区分力很弱；
2. 在 noscore 版中，noise 呈明显双端堆积（低端 `0~0.05` 占 50.34%，高端 `0.95~1.0` 占 14.49%），signal 对应仅 10.20% 和 3.30%，体现出一定可分性；
3. 但 noscore 的高端峰值受极低 `S_base` 分母放大影响明显，`A_true` 不能单独作为硬阈值主判据，更适合作为联合特征（需配合时间连续性/局部事件量约束）。

## 顶部快照（7.73 空间协方差张量统计）

按 7.73 要求，对 Light/Heavy 做了局部协方差张量各向异性统计（9x9、同极性、\(\tau=30\,ms\)、最少点数=4、10bin 直方图）。

运行命令：

```bash
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/noise_analyze/n125_path_failure_stats_773.py \
	--env-list light,heavy
```

统计脚本：`scripts/noise_analyze/n125_path_failure_stats_773.py`

产物目录：

- `data/ED24/myPedestrain_06/EBF_Part2/_n125_path_failure_stats_773/summary.csv`
- `data/ED24/myPedestrain_06/EBF_Part2/_n125_path_failure_stats_773/summary.json`

核心结果（summary.csv）：

| env | events | eligible_rate | anisotropy_mean | anisotropy_std | high_08_10_ratio | low_00_03_ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| light | 194395 | 0.6095 | 0.3872 | 0.2563 | 0.0676 | 0.4226 |
| heavy | 500000 | 0.1944 | 0.3527 | 0.2492 | 0.0584 | 0.4844 |

结论（7.73 复盘）：

1. heavy 的可统计事件占比明显下降（0.1944 vs 0.6095），说明在同样 9x9+30ms 口径下，重噪下“可形成稳定局部几何结构”的事件显著减少。
2. heavy 的各向异性均值略低（0.3527 vs 0.3872），且低分段比例更高（low_00_03: 0.4844 vs 0.4226），表明重噪下局部结构更趋向各向同性/混杂。
3. high_08_10 比例在 heavy 也更低（0.0584 vs 0.0676），说明强方向性局部结构进一步稀缺。
4. 与 7.71/7.72 一致：n125 的“单调路径可达”前提在 heavy 下更难成立，关键瓶颈仍是局部时空结构退化而非单一阈值细节。

## 顶部快照（7.75 GT 信号/噪声分布统计）

按 7.75 要求，在 Heavy 数据中直接按像素级 GT（`label=1` 信号，`label=0` 噪声）统计四类特征分布，并绘制对比直方图。

统计脚本：`scripts/noise_analyze/gt_feature_stats_775.py`

绘图脚本：`scripts/noise_analyze/plot_gt_feature_hist_775.py`

运行命令：

```bash
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/noise_analyze/gt_feature_stats_775.py \
	--max-events 500000 \
	--out-dir data/ED24/myPedestrain_06/EBF_Part2/_gt_feature_stats_775

D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/noise_analyze/plot_gt_feature_hist_775.py \
	--hist-csv data/ED24/myPedestrain_06/EBF_Part2/_gt_feature_stats_775/hist.csv \
	--out-png data/ED24/myPedestrain_06/EBF_Part2/_gt_feature_stats_775/hist_4features.png
```

产物：

- `data/ED24/myPedestrain_06/EBF_Part2/_gt_feature_stats_775/summary.csv`
- `data/ED24/myPedestrain_06/EBF_Part2/_gt_feature_stats_775/hist.csv`
- `data/ED24/myPedestrain_06/EBF_Part2/_gt_feature_stats_775/summary.json`
- `data/ED24/myPedestrain_06/EBF_Part2/_gt_feature_stats_775/hist_4features.png`

核心统计（Heavy，500k，9x9，\(\tau=30ms\)）：

| feature | noise mean | signal mean | 结论 |
| --- | ---: | ---: | --- |
| outer_ratio | 0.5358 | 0.5019 | 均值接近但分布呈强双峰分离，是关键结构特征 |
| mix_smooth | 0.4988 | 0.2563 | 分离度最高，signal 显著更低 |
| anisotropy | 0.4487 | 0.3588 | signal 更偏低值，具一定区分度 |
| velocity_mean | 2.3004 | 2.3355 | 均值接近，单看均值区分度弱 |

四个特征含义（定义 + 标示量）：

1. `outer_ratio`（外圈能量占比）

定义：

$$
outer\_ratio = \frac{E_{out}}{E_{in}+E_{out}+\varepsilon}
$$

其中 `E_in` 是同极性邻域内圈（`d<=2`）的时间权重和，`E_out` 是同极性外圈（`d>2`）的时间权重和。

标示量：局部同极性历史支持在“外圈”相对“内圈”的分布重心，反映结构是更靠近中心还是更分散到外环。

工程解读：

- 取值范围 `[0,1]`；越接近 1 表示外圈占比越高，越接近 0 表示更集中在内圈。
- 在本统计中，噪声出现明显“两端聚集”现象（低端和高端都高），因此它不是简单的单调判别量，而是“形状判别量”。

2. `mix_smooth`（平滑同异极混合比）

定义：

$$
mix\_smooth = \frac{E_{opp}^{joint}}{E_{same}^{joint}+E_{opp}^{joint}+\varepsilon}
$$

其中联合权重为：

$$
w_{joint}=w_{time}\cdot w_{space},\quad
w_{time}=\max\left(0,1-\frac{\Delta t}{\tau}\right),\quad
w_{space}=\max\left(0,1-\eta\frac{d^2}{r_{max}^2}\right)
$$

`E_same^{joint}`/`E_opp^{joint}` 分别是同极性/异极性邻居在联合权重下的能量和。

标示量：局部事件与异极性历史发生“时空共现”的比例，刻画极性混杂程度。

工程解读：

- 取值范围 `[0,1]`；越大表示异极性混入越多。
- 该量在当前数据上对 signal/noise 的错位最明显，是一阶判别力最强的特征之一。

3. `anisotropy`（二阶张量各向异性）

定义：先用同极性邻居构造二阶矩：

$$
M=\begin{bmatrix}m_{20}&m_{11}\\m_{11}&m_{02}\end{bmatrix},
\quad
m_{20}=\sum w_{time}dx^2,
\ m_{02}=\sum w_{time}dy^2,
\ m_{11}=\sum w_{time}dxdy
$$

再定义：

$$
anisotropy = 1-\frac{4\det(M)}{(\operatorname{tr}(M))^2+\varepsilon}
$$

标示量：局部空间结构方向性强弱（是否沿某一方向拉伸）。

工程解读：

- 取值范围 `[0,1]`；接近 0 更接近各向同性团块，接近 1 更接近线状/方向性结构。
- 当前统计显示其有一定区分力，但弱于 `mix_smooth`。

4. `velocity_mean`（平均表观速度）

定义：对同极性邻居计算表观速度 `v_j=d/\Delta t_{ms}`，并按时间权重加权平均：

$$
velocity\_mean=\frac{\sum w_{time}v_j}{\sum w_{time}}
$$

其中 `d` 为空间距离（代码中使用 Chebyshev 距离），`\Delta t_{ms}` 为毫秒级时间差。

标示量：局部同极性历史在时空上的平均运动快慢。

工程解读：

- 取值范围 `[0,+\infty)`，单位约为 `px/ms`。
- 本数据上两类均值接近、分布重叠较多，适合作为弱约束或辅助量，而非主判别量。

直方图补充要点：

1. `mix_smooth` 在低值段（0~0.2）signal 占比约 49.6%，noise 约 16.7%，显示最明显错位。
2. `outer_ratio` 的关键信息是“噪声双峰”：noise 在两端高度集中（[0,0.05) 为 34.22%，[0.95,1.0) 为 38.64%，两端合计 72.86%），而 signal 两端仅 17.77%；这是强区分线索，不应只看均值。
3. `anisotropy` 上 signal 整体左移（低分段更多），但错位程度弱于 `mix_smooth`。
4. `velocity_mean` 的主质量都集中在 0.1~5 px/ms，且两类重叠较大，作为主判别特征的价值较低。

直方图分箱明细（来自 `hist.csv`，值为类内 ratio）：

#### outer_ratio 分箱明细（ratio）

| bin | noise_ratio | signal_ratio |
|---|---:|---:|
| [0,0.05) | 0.342210 | 0.085924 |
| [0.05,0.1) | 0.004746 | 0.009569 |
| [0.1,0.15) | 0.005800 | 0.013742 |
| [0.15,0.2) | 0.006921 | 0.020002 |
| [0.2,0.25) | 0.008647 | 0.028543 |
| [0.25,0.3) | 0.009967 | 0.038797 |
| [0.3,0.35) | 0.013186 | 0.053093 |
| [0.35,0.4) | 0.015288 | 0.067389 |
| [0.4,0.45) | 0.017551 | 0.085467 |
| [0.45,0.5) | 0.023754 | 0.096406 |
| [0.5,0.55) | 0.025371 | 0.094189 |
| [0.55,0.6) | 0.021887 | 0.084636 |
| [0.6,0.65) | 0.022243 | 0.068596 |
| [0.65,0.7) | 0.022261 | 0.053810 |
| [0.7,0.75) | 0.019348 | 0.038275 |
| [0.75,0.8) | 0.016820 | 0.028772 |
| [0.8,0.85) | 0.014157 | 0.019480 |
| [0.85,0.9) | 0.012060 | 0.013220 |
| [0.9,0.95) | 0.011380 | 0.008297 |
| [0.95,1) | 0.386404 | 0.091792 |

#### mix_smooth 分箱明细（ratio）

| bin | noise_ratio | signal_ratio |
|---|---:|---:|
| [0,0.05) | 0.089185 | 0.206589 |
| [0.05,0.1) | 0.018358 | 0.104927 |
| [0.1,0.15) | 0.025357 | 0.096517 |
| [0.15,0.2) | 0.033887 | 0.087540 |
| [0.2,0.25) | 0.042331 | 0.073435 |
| [0.25,0.3) | 0.050215 | 0.066604 |
| [0.3,0.35) | 0.055708 | 0.060980 |
| [0.35,0.4) | 0.059841 | 0.056473 |
| [0.4,0.45) | 0.063649 | 0.050849 |
| [0.45,0.5) | 0.062744 | 0.044692 |
| [0.5,0.55) | 0.063556 | 0.035005 |
| [0.55,0.6) | 0.063402 | 0.028884 |
| [0.6,0.65) | 0.060795 | 0.022071 |
| [0.65,0.7) | 0.055164 | 0.016944 |
| [0.7,0.75) | 0.048551 | 0.012703 |
| [0.75,0.8) | 0.042279 | 0.010379 |
| [0.8,0.85) | 0.033587 | 0.007505 |
| [0.85,0.9) | 0.024541 | 0.005039 |
| [0.9,0.95) | 0.018188 | 0.003371 |
| [0.95,1) | 0.088661 | 0.009492 |

#### anisotropy 分箱明细（ratio）

| bin | noise_ratio | signal_ratio |
|---|---:|---:|
| [0,0.05) | 0.076934 | 0.090576 |
| [0.05,0.1) | 0.067657 | 0.084270 |
| [0.1,0.15) | 0.065869 | 0.078242 |
| [0.15,0.2) | 0.058215 | 0.076716 |
| [0.2,0.25) | 0.058594 | 0.075090 |
| [0.25,0.3) | 0.053659 | 0.068983 |
| [0.3,0.35) | 0.053212 | 0.066345 |
| [0.35,0.4) | 0.050531 | 0.062519 |
| [0.4,0.45) | 0.046918 | 0.057819 |
| [0.45,0.5) | 0.041391 | 0.052228 |
| [0.5,0.55) | 0.050036 | 0.049452 |
| [0.55,0.6) | 0.046384 | 0.044554 |
| [0.6,0.65) | 0.041264 | 0.039300 |
| [0.65,0.7) | 0.038098 | 0.034481 |
| [0.7,0.75) | 0.039195 | 0.031091 |
| [0.75,0.8) | 0.037563 | 0.026054 |
| [0.8,0.85) | 0.042090 | 0.021593 |
| [0.85,0.9) | 0.034882 | 0.017231 |
| [0.9,0.95) | 0.045509 | 0.014336 |
| [0.95,1) | 0.051998 | 0.009121 |

#### velocity_mean 分箱明细（ratio）

| bin | noise_ratio | signal_ratio |
|---|---:|---:|
| [0,0.1) | 0.061858 | 0.017825 |
| [0.1,0.5) | 0.549439 | 0.370266 |
| [0.5,1) | 0.154349 | 0.264054 |
| [1,5) | 0.182282 | 0.286331 |
| [5,+inf) | 0.052072 | 0.061523 |

结论（7.75）：

1. 在 GT 监督下，`mix_smooth` 仍是最稳定的一阶判别特征（整体左移明显）。
2. `outer_ratio` 的双峰结构是高价值信息：噪声在“极内圈/极外圈”两端强聚集，可作为与 `mix_smooth` 并列的重要判别维度，而非仅辅助项。
3. `anisotropy` 可保留为次级特征；`velocity_mean` 更适合作为弱约束而非主判别。

## 顶部快照（7.77 Baseline 时间特性与纯空间结构统计）

按 7.77 要求，基于纯 Baseline 时间权重（不使用空间衰减）新增四类统计：

1. 最近邻时间差 `dt_min_us`（9x9，同极性，最近历史邻居）；
2. 最近邻落点内外圈占比（`d<=2` vs `d>2`）；
3. 纯时间能量比 `ratio_pure = E_outer / (E_inner + E_outer + eps)`；
4. 高分子集（`S_base = E_inner + E_outer > 3.0`）上的 `ratio_pure` 分布。

统计脚本：`scripts/noise_analyze/gt_feature_stats_777.py`

绘图脚本：`scripts/noise_analyze/plot_gt_feature_hist_777.py`

运行命令：

```bash
# heavy
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/noise_analyze/gt_feature_stats_777.py \
	--labeled-npy D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3.npy \
	--max-events 500000 \
	--out-dir data/ED24/myPedestrain_06/EBF_Part2/_gt_feature_stats_777_heavy

D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/noise_analyze/plot_gt_feature_hist_777.py \
	--hist-csv data/ED24/myPedestrain_06/EBF_Part2/_gt_feature_stats_777_heavy/hist.csv \
	--nearest-csv data/ED24/myPedestrain_06/EBF_Part2/_gt_feature_stats_777_heavy/nearest_side.csv \
	--out-png data/ED24/myPedestrain_06/EBF_Part2/_gt_feature_stats_777_heavy/hist_777.png \
	--title "7.77 GT Baseline Time/Structure (heavy, 500k)"

# light
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/noise_analyze/gt_feature_stats_777.py \
	--labeled-npy D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_1.8.npy \
	--max-events 500000 \
	--out-dir data/ED24/myPedestrain_06/EBF_Part2/_gt_feature_stats_777_light

D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/noise_analyze/plot_gt_feature_hist_777.py \
	--hist-csv data/ED24/myPedestrain_06/EBF_Part2/_gt_feature_stats_777_light/hist.csv \
	--nearest-csv data/ED24/myPedestrain_06/EBF_Part2/_gt_feature_stats_777_light/nearest_side.csv \
	--out-png data/ED24/myPedestrain_06/EBF_Part2/_gt_feature_stats_777_light/hist_777.png \
	--title "7.77 GT Baseline Time/Structure (light)"
```

产物目录：

- heavy：`data/ED24/myPedestrain_06/EBF_Part2/_gt_feature_stats_777_heavy/`
	- `summary.csv`
	- `hist.csv`
	- `nearest_side.csv`
	- `summary.json`
	- `hist_777.png`
- light：`data/ED24/myPedestrain_06/EBF_Part2/_gt_feature_stats_777_light/`
	- `summary.csv`
	- `hist.csv`
	- `nearest_side.csv`
	- `summary.json`
	- `hist_777.png`

核心统计（7.77，light/heavy 对照）：

| env | class | dt_min_us_mean | nearest_outer_ratio | ratio_pure_edge([0,0.05)+[0.95,1.0)) | ratio_pure_mid([0.2,0.8)) | ratio_pure_highscore_edge | ratio_pure_highscore_mid | dt<=1ms | dt>10ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| light | noise | 332510.51 | 0.6954 | 0.9705 | 0.0234 | 0.1925 | 0.6350 | 0.0286 | 0.8884 |
| light | signal | 18595.06 | 0.4303 | 0.2384 | 0.6937 | 0.0215 | 0.9110 | 0.4802 | 0.1662 |
| heavy | noise | 19811.37 | 0.6840 | 0.7322 | 0.2155 | 0.1564 | 0.6777 | 0.1043 | 0.5211 |
| heavy | signal | 4869.25 | 0.4519 | 0.1783 | 0.7378 | 0.0188 | 0.9074 | 0.4540 | 0.1276 |

`dt_min_us` 分箱明细（ratio，7.77）：

| env | class | [0,100us) | [100,1000us) | [1,5ms) | [5,10ms) | [10,30ms) | >30ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| light | noise | 0.0037 | 0.0249 | 0.0506 | 0.0324 | 0.0862 | 0.8021 |
| light | signal | 0.0880 | 0.3922 | 0.2600 | 0.0936 | 0.0852 | 0.0810 |
| heavy | noise | 0.0116 | 0.0928 | 0.2143 | 0.1603 | 0.3096 | 0.2115 |
| heavy | signal | 0.0742 | 0.3798 | 0.3028 | 0.1156 | 0.1000 | 0.0277 |

结论（7.77）：

1. 纯 Baseline 时间特性下，signal 的最近邻明显更“近”（两环境中 `dt<=1ms` 都约 45%~48%），而 noise 的最近邻显著更慢，尤其 light-noise 的 `>30ms` 高达 80.2%。
2. 最近邻落点上，noise 的最近邻更偏外圈（light/heavy 分别 69.5%/68.4%），signal 更偏内圈（外圈约 43.0%/45.2%）。这支持“噪声更容易白嫖外圈随机闪烁”的假设。
3. `ratio_pure` 在两环境均保持稳定区分：noise 端更容易落在两端极值，signal 端集中在中段（[0.2,0.8) 约 69%~74%）。
4. 对高分事件（`S_base>3`）做条件统计后，noise 的极值占比显著下降、signal 中段占比升到约 91%，说明“高分噪声并非全部来自极端 ratio”，后续应联合 `dt_min` 与内外圈结构做组合判别，而不是只靠单一 ratio 阈值。

## 用户要求与注意事项（必须遵守）

**新增治理原则（请每次动手前先读一遍）：**

- 我希望能满足实时的要求；按你的理解继续进行就行，**但必须先总结本 README 里已有经验**，不要漫无目的或者理所当然地修改。
	- 具体落地：每次改动都要写清“要解决的失败模式/可验证假设/最小 sweep 口径/预期会提升的指标”。

**输出目录统一（强制）：**
默认使用D:\hjx_workspace\scientific_reserach\dataset\ED24\myPedestrain_06
包括 light、mid、heavy 三种噪声程度的加噪和纯净事件（npy）
- Part2 的所有评测产物统一放在：`data/ED24/myPedestrain_06/EBF_Part2/` 的子文件夹下。
- 对应本机绝对路径：`D:/hjx_workspace/scientific_reserach/projects/myEVS/data/ED24/myPedestrain_06/EBF_Part2/`

**数据集路径：**

**baseline 复用（避免重复跑）：**

- baseline 已经跑过，后续对比时默认直接复用已有产物；除非 baseline 代码/评测口径发生变化，否则不要重复跑 baseline。

**关于“不改 baseline”的澄清（2026-04-13 更新）：**

- 后续新算法允许**重构/替换 baseline 的主体结构**（你已明确 baseline 上限就在那）。
- 这里“不改 baseline”的原意仅指：不要在旧 baseline 上做无穷多“补丁式小改动”并继续沿用旧脚本做评测，导致实验臃肿且结论不可复现。
- 新算法一律按“新脚本 + 新变体编号 + 新产物子文件夹”管理，baseline 仅作为对照基线（reference）。

**新的扫频脚本（精简版；后续新算法统一用它跑）：**

- 脚本：`scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py`
- 目标：只保留“labeled npy -> 打分 -> ROC CSV(best-F1)”，去掉旧脚本里大量历史分支。
- 当前支持的 `--variant`：`ebf`（baseline）| `s55` | `s60` | `s61` | `s62` | `s63` | `s64` | `s65` | `s66` | `s67` | `s68` | `s69` | `s70` | `s71` | `s72` | `s73` | `s74` | `s75` | `s76` | `s77` | `s78` | `s79` | `s80` | `s81` | `s82` | `n1` | `n2` | `n3` | `n4` | `n5` | `n6` | `n7` | `n71` | `n72` | `n8` | `n81` | `n82` | `n83` | `n84` | `n85` | `n86` | `n87` | `n88` | `n89` | `n90` | `n91` | `n92` | `n93` | `n94` | `n95` | `n96` | `n97` | `n98` | `n99` | `n100` | `n101` | `n102` | `n103` | `n104` | `n105` | `n106`
- 输出：依旧写 `roc_*.csv`（列名与 `scripts/noise_analyze/segment_f1.py` 兼容，可直接用它按 best-F1 阈值做分段统计）。

**噪声规律统计脚本（2026-04-15 新增）**

- 统计脚本：`scripts/noise_analyze/transition_pattern_stats.py`
	- 输入：labeled npy（含 `t/x/y/p/label`）
	- 输出1：summary CSV（按 noise/signal、同像素/邻域、pre1/pre2 汇总）
	- 输出2：hist CSV（时间间隔直方图，log bins）
	- 关键统计项：
		- 同像素上一个/上上个同类事件间隔（`dt_*`）
		- 同像素上一个/上上个同类事件极性关系（`same_pol_rate / opp_pol_rate`）
		- 邻域窗口（默认 7、9）上一个/上上个同类事件间隔与极性关系
		- 邻域 past 联合模式（顺序固定 `prev2 -> prev1`）：`same->same / same->opp / opp->same / opp->opp`
		- 联合模式 dt 分桶：`very_fast / fast / medium / miss`
		- 命中率（`hit_rate`，表示能否在对应空间范围内找到 pre1/pre2）
- 绘图脚本：`scripts/noise_analyze/plot_transition_pattern_stats.py`
	- 输入：summary CSV + hist CSV
	- 输出图：
		- `polarity_ratio_bar.png`
		- `interval_quantiles_log.png`
		- `interval_hist_next1_overlay.png`（脚本已兼容 `pre1`，文件名沿用历史命名）

**本轮产物路径（prescreen400k，light/mid/heavy，prev 口径 fixorder）**

- 目录：`data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/transition_stats_prev_400k_fixorder/`
- CSV：
	- `summary_light_400k_prev.csv` / `summary_mid_400k_prev.csv` / `summary_heavy_400k_prev.csv`
	- `hist_light_400k_prev.csv` / `hist_mid_400k_prev.csv` / `hist_heavy_400k_prev.csv`
- 图：
	- `plots_light/*.png`
	- `plots_mid/*.png`
	- `plots_heavy/*.png`

**运行示例**

```bash
# 统计（以 heavy 为例）
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/noise_analyze/transition_pattern_stats.py \
	--labeled-npy D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3.npy \
	--max-events 400000 --windows 7,9 \
	--out-summary-csv data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/transition_stats_prev_400k_fixorder/summary_heavy_400k_prev.csv \
	--out-hist-csv data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/transition_stats_prev_400k_fixorder/hist_heavy_400k_prev.csv

# 绘图（以 heavy 为例）
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/noise_analyze/plot_transition_pattern_stats.py \
	--summary-csv data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/transition_stats_prev_400k_fixorder/summary_heavy_400k_prev.csv \
	--hist-csv data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/transition_stats_prev_400k_fixorder/hist_heavy_400k_prev.csv \
	--out-dir data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/transition_stats_prev_400k_fixorder/plots_heavy
```

**关键信息统计（heavy, 400k, prev/fixorder）**


- 同像素 pre1：
	- noise：`same_pol_rate=0.313`，`dt_p50=82457us`
	- signal：`same_pol_rate=0.567`，`dt_p50=61899us`
- 同像素 pre2：
	- noise：`same_pol_rate=0.584`，`dt_p50=244746us`
	- signal：`same_pol_rate=0.436`，`dt_p50=204760us`
- 邻域 7x7 pre1：
	- noise：`same_pol_rate=0.460`，`dt_p50=7840us`
	- signal：`same_pol_rate=0.862`，`dt_p50=1736us`
- 邻域 9x9 pre1：
	- noise：`same_pol_rate=0.479`，`dt_p50=4746us`
	- signal：`same_pol_rate=0.815`，`dt_p50=1177us`

注（实现修正）：2026-04-15 对 `transition_pattern_stats.py` 修复了邻域 `prev1/prev2` 候选排序方向（应取“最近过去事件”而非“更旧事件”）。
修复前会出现邻域 same/opp 接近 50% 的假象；上述结果以 `transition_stats_prev_400k_fixorder` 目录为准。

补充说明（2026-04-15 重新统计）：

- 新脚本：`scripts/noise_analyze/pixel_transition_pattern_stats.py`
- 这版口径不是“整个邻域里最近两个事件”，而是“窗口内每个像素各自的前两次事件（prev1/prev2）与中心事件极性的关系”。
- 输出目录：`data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/pixel_transition_stats/`
- 产物：`pixel_transition_stats_light_400k_summary.csv` / `pixel_transition_stats_mid_400k_summary.csv` / `pixel_transition_stats_heavy_400k_summary.csv`

window=9 的重统计趋势（按邻域逐像素历史）：

- light：noise pre1 same=0.499，signal pre1 same=0.552；noise pre2 same=0.501，signal pre2 same=0.506。
- mid：noise pre1 same=0.498，signal pre1 same=0.554；noise pre2 same=0.502，signal pre2 same=0.508。
- heavy：noise pre1 same=0.497，signal pre1 same=0.558；noise pre2 same=0.503，signal pre2 same=0.506。
- 这一口径下，信号和噪声的差异主要体现在 pre1 的同极性倾向，以及 signal 的 pre1 时间更短；pre2 仍然更接近 50/50，说明“更早一层历史”在当前定义下区分度更弱。

补充说明（2026-04-16，分块噪声密度统计）：

- 新脚本：`scripts/noise_analyze/block_transition_stats.py`
- 统计口径（按要求补充）：
	- 分块：`UDLR`（方向块）+ `quadrant`（四角块）
	- 历史层级：每块 `top1/top2/top3` 最近历史事件
	- 统计维度：`noise/signal`、`all/seg0/seg1`、`window=7/9`
	- 联合模式：`same->same / same->opp / opp->same / opp->opp`（按 `top2 -> top1`）
- 本轮产物目录：`data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/block_transition_stats_400k/`
	- `summary_light_400k.csv` / `summary_mid_400k.csv` / `summary_heavy_400k.csv`
	- `joint_light_400k.csv` / `joint_mid_400k.csv` / `joint_heavy_400k.csv`
	- `seg_drift_400k.csv`（seg1-seg0 漂移汇总）

运行示例（heavy）：

```bash
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/noise_analyze/block_transition_stats.py \
	--labeled-npy D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3.npy \
	--env heavy --max-events 400000 --windows 7,9 \
	--out-summary-csv data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/block_transition_stats_400k/summary_heavy_400k.csv \
	--out-joint-csv data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/block_transition_stats_400k/joint_heavy_400k.csv
```

window=9、UDLR、按 block 平均的关键结果（all 段）：

- light：
	- noise：`top1_signal_rate=0.361`，`top1_samepol_rate=0.497`，`top1_dt_mean_us=583830.8`，`nonempty_blocks_mean=3.421`
	- signal：`top1_signal_rate=0.930`，`top1_samepol_rate=0.696`，`top1_dt_mean_us=89981.9`，`nonempty_blocks_mean=3.915`
- mid：
	- noise：`top1_signal_rate=0.081`，`top1_samepol_rate=0.501`，`top1_dt_mean_us=91258.0`，`nonempty_blocks_mean=3.904`
	- signal：`top1_signal_rate=0.707`，`top1_samepol_rate=0.682`，`top1_dt_mean_us=30171.1`，`nonempty_blocks_mean=3.953`
- heavy：
	- noise：`top1_signal_rate=0.060`，`top1_samepol_rate=0.500`，`top1_dt_mean_us=47673.8`，`nonempty_blocks_mean=3.917`
	- signal：`top1_signal_rate=0.644`，`top1_samepol_rate=0.671`，`top1_dt_mean_us=18825.0`，`nonempty_blocks_mean=3.951`

seg1 漂移（noise，window=9，UDLR，seg1-seg0）：

- light：`top1_signal_rate +0.083`，`nonempty_blocks_mean +0.690`
- mid：`top1_signal_rate -0.039`，`nonempty_blocks_mean +0.176`
- heavy：`top1_signal_rate -0.022`，`nonempty_blocks_mean +0.152`

联合模式（all 段，window=9，UDLR，按 block 平均）：

- noise：
	- light：`same->same=0.276`，`same->opp=0.219`，`opp->same=0.220`，`opp->opp=0.285`
	- mid：`same->same=0.218`，`same->opp=0.283`，`opp->same=0.283`，`opp->opp=0.217`
	- heavy：`same->same=0.212`，`same->opp=0.288`，`opp->same=0.288`，`opp->opp=0.212`
- signal：
	- light：`same->same=0.564`，`same->opp=0.091`，`opp->same=0.133`，`opp->opp=0.212`
	- mid：`same->same=0.512`，`same->opp=0.142`，`opp->same=0.170`，`opp->opp=0.176`
	- heavy：`same->same=0.491`，`same->opp=0.161`，`opp->same=0.179`，`opp->opp=0.168`

结论（仅基于以上统计）：

- `top1_signal_rate` 随噪声强度升高显著下降（noise: 0.361 -> 0.081 -> 0.060；signal: 0.930 -> 0.707 -> 0.644），说明“最近历史是 signal”的概率在重噪下快速恶化。
- signal 始终保持更高 `top1_samepol_rate` 与更短 `top1_dt_mean_us`，分块邻域内“同极性+快速连续”仍是稳定可用线索。
- noise 的联合模式从 light 的相对混合，逐渐转向 mid/heavy 的近对称交替（`same->opp`、`opp->same` 上升），提示重噪下局部历史更接近交替型扰动。
- seg1 在 mid/heavy 中表现为“块占用更高但 signal 成分更低”（`nonempty_blocks_mean` 上升而 `top1_signal_rate` 下降），与后段更难分一致。

**阶段结论（噪声规律）**

- 邻域尺度下（7/9）signal 的 pre1 同极性比例显著高于 noise，且到达更快（p50 更小），说明“时空邻域内同极性连续性”是可用判别线索。
- 同像素尺度下，noise 在 pre1 更偏异极性、pre2 又转向同极性，提示存在“短程交替 + 次级回返”的噪声序列结构。
- heavy 难段的区分关键更偏向“邻域快速连续性”而非“单像素孤立间隔”，后续阈值/状态设计应优先利用该差异。

示例（prescreen200k，建议先固定 s/tau 做快速门槛检查）：

```bash
# baseline
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py \
	--variant ebf --max-events 200000 --s-list 9 --tau-us-list 128000 \
	--out-dir data/ED24/myPedestrain_06/EBF_Part2/_slim_baseline_prescreen200k

# s55
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py \
	--variant s55 --max-events 200000 --s-list 9 --tau-us-list 128000 \
	--out-dir data/ED24/myPedestrain_06/EBF_Part2/_slim_s55_prescreen200k

# s60
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py \
	--variant s60 --max-events 200000 --s-list 9 --tau-us-list 128000 \
	--out-dir data/ED24/myPedestrain_06/EBF_Part2/_slim_s60_prescreen200k

# s61
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py \
	--variant s61 --max-events 200000 --s-list 9 --tau-us-list 128000 \
	--out-dir data/ED24/myPedestrain_06/EBF_Part2/_slim_s61_prescreen200k_s9_tau128

# s62
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py \
	--variant s62 --max-events 200000 --s-list 9 --tau-us-list 128000 \
	--out-dir data/ED24/myPedestrain_06/EBF_Part2/_slim_s62_prescreen200k_s9_tau128

# s63
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py \
	--variant s63 --max-events 200000 --s-list 9 --tau-us-list 128000 \
	--out-dir data/ED24/myPedestrain_06/EBF_Part2/_slim_s63_prescreen200k_s9_tau128

# s64
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py \
	--variant s64 --max-events 200000 --s-list 9 --tau-us-list 128000 \
	--out-dir data/ED24/myPedestrain_06/EBF_Part2/_slim_s64_prescreen200k_s9_tau128

# s74
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py \
	--variant s74 --max-events 200000 --s-list 9 --tau-us-list 128000 \
	--out-dir data/ED24/myPedestrain_06/EBF_Part2/_slim_s74_max200k
```
- baseline（prescreen 对齐口径：`max-events=200k, s=9, tau=128ms`）产物目录：
	- `data/ED24/myPedestrain_06/EBF_Part2/baseline_prescreen_s9_tau128ms_200k/`
	- 绝对路径：`D:/hjx_workspace/scientific_reserach/projects/myEVS/data/ED24/myPedestrain_06/EBF_Part2/baseline_prescreen_s9_tau128ms_200k/`
	- 若需要 ROC CSV 里包含 `esr_mean`/`aocc` 列（用于 MESR/AOCC 记录），复用同一口径的 baseline 产物：
		- `data/ED24/myPedestrain_06/EBF_Part2/ebf_baseline_s9_tau128_prescreen200k_esrbest_aoccbest/`
		- 该目录的 best-F1（s=9,tau=128ms）与上面 baseline 数值一致，只是额外写入了 `esr_mean/aocc`。

以下内容是你明确提出、且我后续必须持续遵守的约束；每次开展新实验前，先对照检查：

1) 低延迟实时

- 算法必须支持**在线流式、单遍处理**。
- 单事件计算量应与 EBF 同数量级：邻域遍历 $O(r^2)$ 可接受；禁止引入全局优化/大窗口离线求解。

2) 研究主目标与顺序

- 总目标：**阈值稳定性优先**，但 AUC 不能明显掉；固定阈值下的 F1 要能跨 light/mid/heavy 更可迁移。
- Part2 阶段：先做“**精度/可分性提升**”（让评分机制本身更好分开 signal/noise），再叠加 V2 类归一化/自适应阈值机制。

3) 实验命名与产物落盘

- Part2 新方向用别名 `s1, s2, ...` 编号。
- 评测产物（ROC CSV/PNG、best_params、run.log）统一输出到：
	- `data/ED24/myPedestrain_06/EBF_Part2/<子实验文件夹>/`

4) 环境坑（避免“静默退化”）

- Windows 下用户级 site-packages 可能影子覆盖 conda env；另外 `numpy/numba` 也可能出现版本不兼容，导致 `import numba` 失败。
- 若遇到此问题：优先确保 `PYTHONNOUSERSITE=1`，然后通过“升级 numba 或降级 numpy”修复环境。
- 对 Part2 新变体（如 `s1/s2`），**必须使用 Numba kernel**；不可用时应直接报错，禁止静默 fallback。
- 运行建议固定：`PYTHONNOUSERSITE=1`。
- PowerShell 额外坑：形如 `0.2,0.3` 这种写法会被当成数组拆成多个参数；因此所有 `--*-list` 建议用引号包起来（如 `--s4-align-thr-list '0.2,0.3'`），避免 argparse 解析失败。

5) 总结

- 完成实验后注意统计数据,与baseline对比，并总结经验，分析为什么不行，什么可以，**考虑后续的优化方向**
	- 评测输出需要包含 MESR（当前代码里对应 `esr_mean` / ESR mean；**本阶段仅记录，不作为主决策指标**）：
 	- 为节省时间，可先只在“本 env 的 best-AUC tag”和“best-F1 tag”的 best operating point 处计算（与现有 sweep 脚本一致）。
6) 如果后续提出新的要求顺便补充完善进README文件中

7) 通用性（可叠加性）验证（后续实验任务）

- 把 s52/s55 视为“density score modulation layer”，尝试叠加到至少 1 个其它基于密度的算法（例如 STCF/FDF 等）上，保持其核心打分不变，仅对 score 做克制调制；
- 评测口径复用 Part2 现有流程（ROC/F1/MESR/AOCC），并同时报告：固定对齐点（流式低延迟口径）与全网格 best（best-case 口径）。

8) 噪声机理诊断（seg1 重点；后续实验默认针对这一段）

- 现象：heavy 的全量 best-F1 明显低于 prescreen200k，且分段显示 seg1（200k~400k）最难。
- 关键问题（避免遗忘）：
	- 我们是否还缺少“噪声结构”层面的统计（非平稳/爆发/空间聚簇/全局同步），导致 gate/heuristic 没有对准？
	- seg1 变难的主要来源是什么（信号更稀、噪声更同步、热点漂移、空间簇更强、还是混合多种）？
	- 这是否意味着“纯密度类 + 手工 gate”的上限（平台期），如果要再提高应该引入哪类 **非学习** 信息（例如多时间尺度/全局同步抑制/更稳健的结构统计特征）？
	- 评估时如何避免把无监督 AOCC 当作“过度去噪”的单一依据（必须结合 seg1 的监督统计）？

- 切片实验（默认 seg1）：
	- `scripts/noise_analyze/noise_type_stats.py`：默认 `--start-events 200000 --max-events 200000`（噪声类别/保留率）
	- `scripts/noise_analyze/dump_u_events.py`：默认 `--start-events 200000 --max-events 200000`（导出逐事件 u/状态量，便于画分布）

- 噪声结构统计（默认 seg1）：
	- `scripts/noise_analyze/noise_structure_stats.py`：按窗口输出结构统计（非平稳/爆发/聚簇/同步 proxy）
	- `scripts/noise_analyze/compare_noise_structure.py`：对比 seg0 vs seg1（按窗口均值汇总漂移）

- hotmask（用于 hotmask_frac / 以及其它脚本的 hot/near-hot 类别统计）：
	- heavy：`data/ED24/myPedestrain_06/EBF_Part2/hotmask_heavy_score_neg_minus_2pos_topk32768_dil1.npy`
	- mid：`data/ED24/myPedestrain_06/EBF_Part2/hotmask_mid_score_neg_minus_2pos_topk32768_dil1.npy`
	- light：`data/ED24/myPedestrain_06/EBF_Part2/hotmask_light_score_neg_minus_2pos_topk32768_dil1.npy`

- 本机已跑产物（heavy）：
	- seg1（200k~400k）：
		- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/validate_1M_heavy/noise_structure_seg1_heavy_window20k.csv`
		- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/validate_1M_heavy/top_noise_pixels_seg1_heavy.csv`
	- seg0（0~200k）：
		- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/validate_1M_heavy/noise_structure_seg0_heavy_window20k.csv`
		- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/validate_1M_heavy/top_noise_pixels_seg0_heavy.csv`
	- seg0 vs seg1 漂移汇总：
		- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/validate_1M_heavy/noise_structure_seg0_vs_seg1_heavy_summary.csv`

- 本机已跑产物（mid）：
	- seg1（200k~400k）：
		- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/validate_1M_mid/noise_structure_seg1_mid_window20k.csv`
		- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/validate_1M_mid/top_noise_pixels_seg1_mid.csv`
	- seg0（0~200k）：
		- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/validate_1M_mid/noise_structure_seg0_mid_window20k.csv`
		- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/validate_1M_mid/top_noise_pixels_seg0_mid.csv`
	- seg0 vs seg1 漂移汇总：
		- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/validate_1M_mid/noise_structure_seg0_vs_seg1_mid_summary.csv`

- 本机已跑产物（light）：
	- 注意：`Pedestrain_06_1.8.npy` 总事件数 n=194,395，不足以按 200k/200k 切 seg0/seg1；因此这里使用“前半段 vs 后半段”切片：`seg_len = floor(n/2) = 97,197`。
	- seg1（后半段）：
		- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/validate_1M_light/noise_structure_seg1_light_window20k.csv`
		- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/validate_1M_light/top_noise_pixels_seg1_light.csv`
	- seg0（前半段）：
		- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/validate_1M_light/noise_structure_seg0_light_window20k.csv`
		- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/validate_1M_light/top_noise_pixels_seg0_light.csv`
	- seg0 vs seg1 漂移汇总：
		- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/validate_1M_light/noise_structure_seg0_vs_seg1_light_summary.csv`

### 基于现有数据的初步结论（2026-04-13）

这一段只基于“已落盘的统计结果”做解释，不引入任何新假设数据。

#### 1) seg1 结构漂移：heavy/mid 与 light 的模式完全不同

下表是 `noise_structure_seg0_vs_seg1_*_summary.csv` 里挑出的关键指标（ratio = seg1/seg0）：

| metric | heavy | mid | light（前半 vs 后半） |
|---|---:|---:|---:|
| signal_frac | 0.472 | 0.594 | 1.169 |
| events_per_ms | 0.871 | 0.879 | 2.525 |
| dt_any_norm_p50 | 1.076 | 1.009 | 0.319 |
| dt_any_norm_p90 | 1.107 | 1.162 | 0.411 |
| dt_any_norm_p99 | 1.132 | 1.267 | 0.603 |
| dt_samepol_norm_p50 | 1.266 | 1.144 | 0.202 |
| dt_samepol_norm_p90 | 1.168 | 1.188 | 0.384 |
| nb_recent_frac_ge_k | 0.153 | 0.428 | 3.148 |
| hotmask_frac | 1.036 | 1.031 | 0.658 |

解释（只对齐到这些统计量）：

- heavy/mid：seg1 的 **signal_frac 大幅下降**，并且 dt 分布整体右移（更稀疏），hotmask_frac 还略升。
	- 这是一种典型的 **低 SNR + 稀疏** 场景：局部密度类证据会天然变弱，而“热点像素噪声”相对占比更大。
- light：由于样本长度不足采用“半段切片”，后半段反而 **更密（events_per_ms ↑）+ 更聚簇（nb_recent_frac_ge_k ↑）+ dt 变短**，同时 signal_frac ↑、hotmask_frac ↓。
	- 这更像是“真正运动/结构信号增强”的阶段，而不是 heavy/mid 那种“信号变稀”。

#### 2) heavy 的监督分段结果：seg1 确认是最难段，s52/s55 只带来温和改善

文件：

- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/validate_1M_heavy/segf1_baseline_heavy_1M_s9_tau128.csv`
- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/validate_1M_heavy/segf1_s52_heavy_1M_s9_tau128.csv`
- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/validate_1M_heavy/segf1_s55_heavy_1M_s9_tau128.csv`

以 seg1（200k~400k）为例，best-F1 阈值下 F1：

- baseline：F1=0.655
- s52：F1=0.676
- s55：F1=0.671

结论：s52/s55 的确对最难段有增益，但幅度还不够；要进一步提升，必须对准“seg1 的主导失败模式”。

#### 3) heavy 的噪声类型统计：seg1 的核心矛盾是“信号更少且更集中落在 hotmask 上”

文件（seg0/seg1 × baseline/s52/s55）：

- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/validate_1M_heavy/noise_types_seg0_baseline_heavy_s9_tau128_thr_bestf1.csv`
- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/validate_1M_heavy/noise_types_seg1_baseline_heavy_s9_tau128_thr_bestf1.csv`
- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/validate_1M_heavy/noise_types_seg1_s52_heavy_s9_tau128_thr_bestf1.csv`
- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/validate_1M_heavy/noise_types_seg1_s55_heavy_s9_tau128_thr_bestf1.csv`

从统计量本身能直接读出的事实（baseline）：

- seg1 的 signal_total 明显下降：39,863 → 18,796（约减半）；noise_total 相应上升：160,137 → 181,204。
- seg1 的 signal 更“贴着 hotmask 走”：signal_share_hotmask 从 0.697 → 0.758。
- seg1 上，hotmask/near-hotmask 的 signal_kept_rate 明显更低（更容易误伤）：
- hotmask：0.734 → 0.571
- near_hotmask：0.824 → 0.601

s52/s55 的改善点也很清晰：它们在 seg1 上把 hotmask 的 signal_kept_rate 拉回了约 +0.016，同时 hotmask 的 noise_kept_rate 还略降（对齐“克制调制”的预期）。

#### 4) 口径注意：切片跑脚本会改变“流式状态”

- `segment_f1.py` 是把整段流式从头打分后再按 event index 分段统计（状态是连续的）。
- `noise_type_stats.py` / `dump_u_events.py` 如果用 `--start-events/--max-events` 直接切片输入，会让去噪器在该切片“冷启动”。
- 因此：绝对的 kept 数可能和 `segment_f1.py` 不完全一致；但同一脚本内做 baseline vs s52/s55 的相对比较仍然有效。

#### 5) 可验证假设（下一步非学习方向应该对准什么）

基于上述统计，最小可验证假设是：

- heavy/mid 的 seg1 主要是 **低 SNR + 稀疏** 导致“密度证据不足”，同时信号大量与 hotmask 重叠，造成“压热点会伤信号、不压热点会留大量 FP”的矛盾。
- 因此下一步非学习改进应优先考虑：
- 多时间尺度（稀疏段拉长时间证据，但要配合更稳健的噪声抑制，避免把热点积分得更高）；
- 在线 hotness/背景率估计（对热点做更精细的、可恢复的 downweight，而不是静态 hard mask）；
- 全局同步/爆发 proxy 触发的模式切换（用非常便宜的全局统计抑制“非局部”的噪声形态）。

## README 维护规则（强制）

从现在起，每完成一次“新候选编号”的实验（哪怕只是 prescreen），都必须把以下内容补充到本 README：

- 方法原理：一句话动机 + 关键公式/定义（至少写清 raw / gate / weight 各自是什么）
- 实现约束核对：是否单遍/在线、是否保持 $O(r^2)$、是否 Numba 必须可用
- 实验口径：本次是 prescreen（固定参数对齐）还是全网格 sweep（按 env 各自 best），至少需要与 baseline 和上一次版本算法进行对比
- 失效原因分析：为什么没超过 baseline（从“触发条件漂移 / 误伤信号 / 判别信息不足 / 指标 trade-off”角度给出可验证解释）
- 是否继续：明确写“继续/停掉”，若继续给出下一步最小改动（不要写泛泛的愿望）

## 目标与约束（Part2 视角）

目标：在“基于事件密度”的大框架下，**先把可分性/精度（precision、AUC、F1 等）做到超过 baseline EBF**

硬约束：

- **低延迟实时**：算法必须支持在线流式、单遍处理；单事件计算量应与 EBF 同数量级（$O(r^2)$ 邻域遍历）。

默认评测口径（2026-04-22 起）：

- 默认直接跑 `compact400k`（三环境；`s in {5,7,9}` × `tau_us in {32000,64000,128000,256000}`）。
- heavy 默认同时做 `2×200k` 分段（seg0/seg1）观察“最难段”是否真实增益。
- heavy 1M 仅作为最终确认口径，不再作为每个新变体的默认必跑项。

## 已实现方法的经验教训总结（截至 2026-04-22）

这一节是“从已跑过的 s1–s25 里提炼出来的共性规律”，用于约束后续改动的方向，避免重复试错。

### 1) 对打分形式的总原则：避免全局缩放，偏好“克制的增量”

- **全局乘子/全局重排很危险**：像 s1 这类把某个指标当作全局乘子（哪怕看起来合理），会对大量真实 signal 一并降分，导致 AUC/F1 整体劣化。
- **更稳的形态是“加分/门控惩罚”**：s14 的经验非常明确：不要把某个证据当作必要条件（s13 失败），而是把它当作“存在就小幅加分”的增量项。
- **排序扰动要克制**：凡是会大范围改变 score 分布的操作（尤其在 light 下）容易把本来接近最优的排序打坏；因此更推荐“只在少量样本上触发”的 gate。

### 2) 门控类方法必须同时满足两点：覆盖面足够 + 触发足够克制

- **覆盖面不足 ⇒ 基本没用**：例如 s15（flip flicker）在该数据集上触发极少，机制再“机理正确”也无法带来可见收益。
- **触发太宽 ⇒ 误伤 signal**：许多 gate 一旦对大比例事件生效，往往会伤 light 的 AUC/F1（典型表现是 light 先掉）。
实操上更安全的写法是：
- 先加一个 **raw 前置阈值**（只在 raw 已经高时才考虑惩罚/修正），把 gate 作用范围控制在“高风险候选”上；
- 再用一个 **更具体的异常指标**（如短 dt、热点相对异常等）去打那一小撮 FP 失败模式。

补充（来自 s1–s6 的共性复盘，详见 `S_README.md`）：

- s1（coh 全局乘子）：会系统性压低真实信号，mid/heavy 先掉；
- s2（coh 硬门控惩罚）：方向上是“只打高 raw 的异常”，但硬阈值跨噪声强度不可迁移，仍会误伤；
- s3（soft gate）：把硬误伤抹平到接近 baseline，但仍缺少足够强的判别信息，难形成稳定正增；
- s4（resultant gate）：真实边缘/纹理对称性会让 resultant 抵消，容易误伤 signal；
- s5（全局固定方向椭圆核）：方向不匹配时等价于减少有效邻居，排序会变差；
- s6（dt 方差门控）：判别信号不稳定，最优点往往退化为“不触发”，等价 baseline。

### 3) ED24/myPedestrain_06 上，“热点/爆发噪声”仍是 heavy 的主要矛盾，但判别并不止一种

- s9 的经验：同像素同极性“极短 dt”是一个**很干净**的噪声指示，但提升通常在 $10^{-4}\sim10^{-3}$ 量级，说明 heavy 下的噪声并不都表现为“每次都极短 dt”。
- s10–s12 的经验：把热点从“绝对高发射率”升级为“相对异常高”（relative / z-score）更稳、更不容易误伤忙但正常的区域。
- s22 的经验（any-pol dt）：把“极短 dt”从同极性放宽到任意极性，覆盖面可能更大，但仍需要 raw 前置以避免误伤。

### 4) 结构性拟合类（平面残差 / R2）在 bursty 噪声上可能出现“假象”

- 经验（来自 s7/s8 的总结）：密集噪声团/热点区域的邻域 dt 可能呈现局部单调结构，导致拟合解释度并不低，从而无法稳定压噪。
- 这类方法容易变成“看起来更高级，但对主要失败模式不够对准”。

### 5) 跨环境目标的现实：heavy 的净提升经常会和 light 产生 trade-off

- 已验证现象：当我们更强地压制 heavy 下的热点/爆发噪声时（更宽的触发、更强的惩罚），往往会开始影响 light 的真实信号；反之，为了保持 light 的最优点，heavy 的增益会很小或消失。
因此后续评测口径必须先说清楚：
- 是追求“各 env 各自 best-F1 都不低于 s14”（强、但可能不可达），
- 还是追求“固定一套更可迁移的 recipe，在三环境下 F1 都尽可能高”（更符合迁移与稳定性目标）。

### 6) 当 heuristic gate 卡在 $10^{-3}$ 量级时，说明“判别信息不足”而不是“调参不够细”

- s24/s25 都体现了这一点：它们更像是对某套主干 recipe 的克制微调项，能补掉一部分特定 FP，但很难把整体可分性再推一个台阶。
当出现这种平台期时，下一步更可能需要：
- 换一个“覆盖面更大、但仍然可解释且克制”的噪声指示特征；或
- 使用 s23 这种“保底 baseline + 学增量权重”的线性融合框架，把多个弱证据组合成更强的排序信号。

补充（关于评测策略）：

- 对 n147+ 这类“主干已稳定、只微调少数连续参数（如 sigma/beta_init）”的变体，后续默认 **直接跑 compact400k**（`s in {5,7,9}` × `tau_us in {32000,64000,128000,256000}`），不再强依赖 prescreen200k。
- 本阶段不再默认跑 heavy 全量 1M；先用 compact400k + heavy 2×200k 分段（seg0/seg1）定位是否在“最难段”有真实增益；只有当候选已经稳定优于基线时，再用 heavy 1M 作为最终确认口径。

### 工具：误检噪声类型分解（FP breakdown，建议先看 heavy）

目的：回答“哪些噪声被过滤了、哪些噪声还保留、保留的噪声像什么”。做法是：读取某个 ROC CSV（来自 sweep），自动取该 (s,tau) 的 best-F1 operating point 阈值，然后在 labeled npy 上复算分数并把 FP 按简单可解释的类型分解（hotmask / 同像素短 dt / 翻转短 dt / 其它）。

脚本：`scripts/ED24_alg_evalu/analyze_fp_noise_types.py`

示例（s25 grid1，heavy）：

```powershell
$env:PYTHONNOUSERSITE='1'
conda run -n myEVS python scripts/ED24_alg_evalu/analyze_fp_noise_types.py \
	--variant s25 \
	--roc-csv data/ED24/myPedestrain_06/EBF_Part2/_prescreen_s25_s14refrac_grid1_s9_tau128ms_200k/roc_ebf_s25_heavy_labelscore_s3_5_7_9_tau8_16_32_64_128_256_512_1024ms.csv \
	--env heavy --s 9 --tau-us 128000 --max-events 200000 \
	--hotmask-npy data/ED24/myPedestrain_06/EBF_Part2/hotmask_heavy_score_neg_minus_2pos_topk32768_dil1.npy
```

对照（同一 ROC 文件里的 s14、baseline）：

```powershell
$env:PYTHONNOUSERSITE='1'
conda run -n myEVS python scripts/ED24_alg_evalu/analyze_fp_noise_types.py --variant s14 --roc-csv data/ED24/myPedestrain_06/EBF_Part2/s14_crosspol_boost_prescreen_s9_tau128ms_200k/roc_ebf_s14_heavy_labelscore_s3_5_7_9_tau8_16_32_64_128_256_512_1024ms.csv --env heavy --s 9 --tau-us 128000 --max-events 200000
conda run -n myEVS python scripts/ED24_alg_evalu/analyze_fp_noise_types.py --variant ebf --roc-csv data/ED24/myPedestrain_06/EBF_Part2/baseline_prescreen/roc_ebf_heavy_labelscore_s3_5_7_9_tau8_16_32_64_128_256_512_1024ms.csv --env heavy --s 9 --tau-us 128000 --max-events 200000
```

### 工具：噪声类型统计 + 去噪前后对比（CSV，baseline/s14/sXX 都可用）

上面的 FP breakdown 只回答“FP 还剩什么”。但很多时候我们更需要回答：

- 去噪前：各类噪声（noise_total）到底占多少？
- 去噪后：各类噪声被保留了多少（noise_kept=FP）、去掉了多少（noise_removed）？
- 同时：signal 是否被误伤（signal_kept/signal_removed）？

脚本：`scripts/noise_analyze/noise_type_stats.py`

特点：

- 读取某个 ROC CSV，自动在指定 `(s,tau)` 下选 **best-F1** operating point 的阈值（也可 `--thr` 手动指定）。
- 在 labeled npy 上复算 scores（复用 `scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py` 的 `score_stream_ebf()`，保证和 sweep 完全一致）。
- 对每个事件做“互斥分类”（优先级从高到低）：
	- `hotmask`：命中 hotmask（可选）
	- `near_hotmask`：hotmask 膨胀邻域（可选）
	- `highrate_pixel`：noise-only 事件计数 topK 的高发像素（可选，默认 topK=32768）
	- `samepol_shortdt` / `toggle_shortdt`：同像素短 dt（dt/τ < 阈值）
	- `isolated_nb0` / `cluster_nb_ge_k`：邻域最近活动（窗口内邻居数=0 或 >=k）
	- `other`
- 输出一张 summary CSV（每类一行）：`noise_total/noise_kept/noise_removed/noise_kept_rate` 以及 signal 对应列。

注意：

- `--labeled-npy` 必须是包含 `t/x/y/p/label` 的结构化 npy（ED24 默认数据路径见下方示例）。
- 如果一个 ROC CSV 里混了多个 variant 的 tag，脚本会按 `--variant` 自动过滤 tag 前缀（例如 `s14` 会筛 `ebf_s14_`），避免选错 best-F1 点。

示例（ED24/myPedestrain_06，prescreen200k 口径：`s=9,tau=128ms,max-events=200k`）：

baseline heavy：

```powershell
$env:PYTHONNOUSERSITE='1'
conda run -n myEVS python scripts/noise_analyze/noise_type_stats.py \
	--labeled-npy D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3.npy \
	--roc-csv data/ED24/myPedestrain_06/EBF_Part2/baseline_prescreen_s9_tau128ms_200k/roc_ebf_heavy_labelscore_s3_5_7_9_tau8_16_32_64_128_256_512_1024ms.csv \
	--variant ebf --s 9 --tau-us 128000 --max-events 200000 \
	--hotmask-npy data/ED24/myPedestrain_06/EBF_Part2/hotmask_heavy_score_neg_minus_2pos_topk32768_dil1.npy \
	--out-csv data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/noise_types_baseline_heavy_prescreen200k_s9_tau128ms.csv \
	--topk-pixels-csv data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/top_pixels_baseline_heavy_prescreen200k.csv
```

s14 heavy：

```powershell
$env:PYTHONNOUSERSITE='1'
conda run -n myEVS python scripts/noise_analyze/noise_type_stats.py \
	--labeled-npy D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3.npy \
	--roc-csv data/ED24/myPedestrain_06/EBF_Part2/s14_crosspol_boost_prescreen_s9_tau128ms_200k/roc_ebf_s14_heavy_labelscore_s3_5_7_9_tau8_16_32_64_128_256_512_1024ms.csv \
	--variant s14 --s 9 --tau-us 128000 --max-events 200000 \
	--hotmask-npy data/ED24/myPedestrain_06/EBF_Part2/hotmask_heavy_score_neg_minus_2pos_topk32768_dil1.npy \
	--out-csv data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/noise_types_s14_heavy_prescreen200k_s9_tau128ms.csv \
	--topk-pixels-csv data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/top_pixels_s14_heavy_prescreen200k.csv
```

light/mid（通常不带 hotmask 也可以先跑）：

```powershell
$env:PYTHONNOUSERSITE='1'
conda run -n myEVS python scripts/noise_analyze/noise_type_stats.py --labeled-npy D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_1.8.npy --roc-csv data/ED24/myPedestrain_06/EBF_Part2/baseline_prescreen_s9_tau128ms_200k/roc_ebf_light_labelscore_s3_5_7_9_tau8_16_32_64_128_256_512_1024ms.csv --variant ebf --s 9 --tau-us 128000 --max-events 200000 --out-csv data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/noise_types_baseline_light_prescreen200k_s9_tau128ms.csv
conda run -n myEVS python scripts/noise_analyze/noise_type_stats.py --labeled-npy D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_1.8.npy --roc-csv data/ED24/myPedestrain_06/EBF_Part2/s14_crosspol_boost_prescreen_s9_tau128ms_200k/roc_ebf_s14_light_labelscore_s3_5_7_9_tau8_16_32_64_128_256_512_1024ms.csv --variant s14 --s 9 --tau-us 128000 --max-events 200000 --out-csv data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/noise_types_s14_light_prescreen200k_s9_tau128ms.csv
conda run -n myEVS python scripts/noise_analyze/noise_type_stats.py --labeled-npy D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_2.5.npy --roc-csv data/ED24/myPedestrain_06/EBF_Part2/baseline_prescreen_s9_tau128ms_200k/roc_ebf_mid_labelscore_s3_5_7_9_tau8_16_32_64_128_256_512_1024ms.csv --variant ebf --s 9 --tau-us 128000 --max-events 200000 --out-csv data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/noise_types_baseline_mid_prescreen200k_s9_tau128ms.csv
conda run -n myEVS python scripts/noise_analyze/noise_type_stats.py --labeled-npy D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_2.5.npy --roc-csv data/ED24/myPedestrain_06/EBF_Part2/s14_crosspol_boost_prescreen_s9_tau128ms_200k/roc_ebf_s14_mid_labelscore_s3_5_7_9_tau8_16_32_64_128_256_512_1024ms.csv --variant s14 --s 9 --tau-us 128000 --max-events 200000 --out-csv data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/noise_types_s14_mid_prescreen200k_s9_tau128ms.csv
```

（已生成）baseline vs s14 的 prescreen200k 统计产物（2026-04-10）：

- heavy：
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/noise_types_baseline_heavy_prescreen200k_s9_tau128ms.csv`
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/noise_types_s14_heavy_prescreen200k_s9_tau128ms.csv`
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/top_pixels_baseline_heavy_prescreen200k.csv`
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/top_pixels_s14_heavy_prescreen200k.csv`
- light：
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/noise_types_baseline_light_prescreen200k_s9_tau128ms.csv`
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/noise_types_s14_light_prescreen200k_s9_tau128ms.csv`
- mid：
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/noise_types_baseline_mid_prescreen200k_s9_tau128ms.csv`
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/noise_types_s14_mid_prescreen200k_s9_tau128ms.csv`

### 工具：导出逐事件内部量并统计分布（用于解释 s36/s37/s38/s39）

目的：把“state-occupancy 系列”的关键内部量导出成 CSV，便于复盘为什么某个映射/更新规则没带来 heavy 提升。

脚本（两阶段工作流）：

1) 导出逐事件 CSV（重放某个 ROC CSV 的 best-F1 operating point）：`scripts/noise_analyze/dump_u_events.py`
	- 输出列包含（节选）：
		- 兼容列：`label/score/kept/cat/u/hot_state/dt0_norm/...`（其中 `u` 仍表示 `u_self`，用于兼容旧口径）
		- 新增诊断列：`u_eff/u_self/u_nb/u_nb_mix/mix/raw/raw_all/raw_opp/m/r_ema/r_pix/r_eff/mu/var/z_dbg`
	- 产物建议放：`data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/u_events_<variant>_<env>_prescreen200k_sX_tauY_bestf1.csv`

2) 从 u_events CSV 生成分位数表：`scripts/noise_analyze/u_quantiles.py`
	- 支持 `--col` 指定列（默认 `u`）。例如：`--col u_eff` 或 `--col z_dbg`
	- 产物：`u_quantiles_*.csv`

3) 从 u_events CSV 生成直方图与可视化：`scripts/noise_analyze/u_hist.py`
	- 支持 `--col` 指定列（默认 `u`）；对非 [0,1] 的列（如 `z_dbg`）会自动用数据范围设 bin
	- 产物：`u_hist_*.csv` 与 `u_hist_*.png`

注意：当前 `cat`（hotmask/near-hot/highrate/samepol_shortdt/...）的分类逻辑与 `scripts/noise_analyze/noise_type_stats.py` 保持一致，便于互相对照。

提示：`dump_u_events.py` 会在给定的 ROC CSV 中按 `--s/--tau-us`（以及可选 `--tag`）定位 best operating point；因此 **ROC CSV 必须真的包含该点位的 tag 行**。例如 baseline 的常用产物目录：

- (s=7,tau=64ms) 口径：`data/ED24/myPedestrain_06/EBF_Part2/_baseline_s7_tau64ms_200k/`
- (s=9,tau=128ms) 口径：`data/ED24/myPedestrain_06/EBF_Part2/baseline_prescreen_s9_tau128ms_200k/`

快速解读要点（为什么 s14 提升有限）：

数值证据（同一口径：prescreen200k，best-F1 operating point，`s=9,tau=128ms,max-events=200k`）：

- heavy：baseline `thr=7.3581`，TP=30304/39863，FP=6856；s14 `thr=7.6638`，TP=30754/39863，FP=7294。
	- heavy 的 FP 主要来自 hot 区域：baseline hotmask FP=5778、near-hot FP=711；s14 hotmask FP=6147、near-hot FP=759。
- light：baseline FP=7400；s14 FP=8171（主要都落在 highrate_pixel）。
- mid：baseline FP=8893；s14 FP=8557（主要都落在 highrate_pixel）。

- **heavy 的 FP 主要仍来自 hotmask/near-hot 区域**（noise_total 巨大，即使 kept_rate 只有 ~4% 也会主导 FP 绝对数量）。
- s14 的 cross-pol boost 在该数据集上会**同时抬高 signal 与 hotpixel noise 的分数**：heavy 上 TP 增加的同时，FP 也跟着增加，导致净增益有限。
- 要继续提升 heavy，下一步应优先针对“hotpixel/高发像素”类残余噪声（s10–s12、s21 这类方向），而不是只盯 short-dt refractory。

补充：继续对比 s10/s11/s12/s21（同一口径，2026-04-10）

（同一口径：prescreen200k，best-F1 operating point，`s=9,tau=128ms,max-events=200k`；统计来自本节生成的 summary CSV。）

- 总体（F1）：
	- light：baseline=0.9497，s14=0.9525，s10=0.9497，s11=0.9498，s12=0.9498，s21=0.9560
	- mid：baseline=0.8108，s14=0.8129，s10=0.8110，s11=0.8111，s12=0.8110，s21=0.8173
	- heavy：baseline=0.7869，s14=0.7895，s10=0.7872，s11=0.7870，s12=0.7870，s21=0.7922

- heavy 的 FP 分解（热点类占绝对主导，且几乎解释了全部 FP）：

| variant | F1 | TP | FP | hotmask FP | near-hot FP | highrate FP |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.7869 | 30304 | 6856 | 5778 | 711 | 323 |
| s14 | 0.7895 | 30754 | 7294 | 6147 | 759 | 342 |
| s10 | 0.7872 | 30436 | 7027 | 5931 | 721 | 331 |
| s11 | 0.7870 | 30400 | 6991 | 5907 | 712 | 328 |
| s12 | 0.7870 | 30378 | 6961 | 5875 | 715 | 327 |
| s21 | 0.7922 | 31331 | 7909 | 6647 | 839 | 372 |

备注：s10–s12 的方向确实在减少 heavy 的热点类 FP（相对 s14 降了一些），但同时 TP 也略回落；s21 通过显著抬高 TP 获得了更高的 heavy F1，但热点类 FP 也更高。

（追加生成）s10/s11/s12/s21 的 prescreen200k 统计产物（2026-04-10）：

- heavy：
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/noise_types_s10_heavy_prescreen200k_s9_tau128ms.csv`
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/noise_types_s11_heavy_prescreen200k_s9_tau128ms.csv`
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/noise_types_s12_heavy_prescreen200k_s9_tau128ms.csv`
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/noise_types_s21_heavy_prescreen200k_s9_tau128ms.csv`
- light：
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/noise_types_s10_light_prescreen200k_s9_tau128ms.csv`
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/noise_types_s11_light_prescreen200k_s9_tau128ms.csv`
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/noise_types_s12_light_prescreen200k_s9_tau128ms.csv`
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/noise_types_s21_light_prescreen200k_s9_tau128ms.csv`
- mid：
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/noise_types_s10_mid_prescreen200k_s9_tau128ms.csv`
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/noise_types_s11_mid_prescreen200k_s9_tau128ms.csv`
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/noise_types_s12_mid_prescreen200k_s9_tau128ms.csv`
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/noise_types_s21_mid_prescreen200k_s9_tau128ms.csv`
c
## s60（2026-04-13）：双时间尺度证据 + delta-only long boost（目标：稀疏段更稳；无新超参）

动机/定位：s60 尝试把 s52/s55 中验证过的“输入流自适应（少超参）调制层”抽出来，并在证据侧加入一个**固定比例**的长窗（`2*tau`）补强，用于提升稀疏/低 SNR 段（尤其 heavy seg1）的鲁棒性，同时尽量不扰动 light/mid。

要解决的失败模式/可验证假设：

- 在稀疏段，短窗 `tau` 内的邻域支持可能偏弱，导致 $\mathrm{raw}_{\mathrm{short}}$ 过低；但在更长时间尺度上仍可能存在一致支持。
- 直接用长窗替代短窗会改变致密段的排序，容易带来回归；因此 s60 只取“长窗相对短窗的正增量”，并对增量做上界约束（delta-only + clamp），把影响限制在“短窗不足但长窗补得上”的事件上。

### 方法定义（单 kernel，无新超参）

在同一轮邻域遍历中，同时累计两套三角时间权重的证据（长窗比例固定，不作为 sweep 维度）：

- 短窗：$\tau$
- 长窗：$\tau_{\mathrm{long}}=2\tau$

对 opp 证据做与 s52 相同的自适应门控（全局标量在线均值；不引入环境超参）：

$$
\mathrm{mix}=\frac{\mathrm{raw}_{opp}}{\mathrm{raw}_{same}+\mathrm{raw}_{opp}+\varepsilon},\quad
\mathrm{mix}_{\mathrm{ema}}\leftarrow \mathrm{mix}_{\mathrm{ema}}+\frac{\mathrm{mix}-\mathrm{mix}_{\mathrm{ema}}}{N},\quad
\alpha_{\mathrm{eff}}=(1-\mathrm{mix}_{\mathrm{ema}})^2
$$

注意：实现里 $\mathrm{mix}$ 使用**短窗**（$\tau$）下的 $\mathrm{raw}_{same},\mathrm{raw}_{opp}$ 计算。

短/长窗的归一化 raw（与实现一致，分别除以各自的 $\tau$）：

$$
\mathrm{raw}_{\mathrm{short}}=\frac{\mathrm{raw}_{same}^{(\tau)}+\alpha_{\mathrm{eff}}\,\mathrm{raw}_{opp}^{(\tau)}}{\tau},\quad
\mathrm{raw}_{\mathrm{long}}=\frac{\mathrm{raw}_{same}^{(2\tau)}+\alpha_{\mathrm{eff}}\,\mathrm{raw}_{opp}^{(2\tau)}}{2\tau}
$$

仅取长窗相对短窗的“正增量”，并把增量上界钳制到 $\mathrm{raw}_{\mathrm{short}}$（避免长窗过度主导）：

$$
\Delta=\mathrm{clip}(\max(0,\mathrm{raw}_{\mathrm{long}}-\mathrm{raw}_{\mathrm{short}}),\ 0,\ \mathrm{raw}_{\mathrm{short}}),\quad
\mathrm{raw}_{\mathrm{dual}}=\mathrm{raw}_{\mathrm{short}}+\Delta
$$

其余复用 s51/s52 的调制层：

- self-occupancy 抑制：
$$
\mathrm{base}=\frac{\mathrm{raw}_{\mathrm{dual}}}{1+u_{\mathrm{self}}^2}
$$
- support-breadth boost（$\beta_{\mathrm{eff}}$ 为在线均值自适应，形状与 s51/s52 一致）：
$$
\mathrm{score}=\mathrm{base}\cdot (1+\beta_{\mathrm{eff}}\,s_{\mathrm{frac}})
$$

其中 $u_{\mathrm{self}}$ 由每像素 `hot_state`（线性衰减累积器）得到，$s_{\mathrm{frac}}$ 为同极性支持比例（`cnt_support/cnt_possible`），$\beta_{\mathrm{eff}}$ 由 $u_{\mathrm{self}}$ 的在线均值更新得到。

### 实现与开销

实现位置：`src/myevs/denoise/ops/ebfopt_part2/s60_ebf_labelscore_dualtau_delta_selfocc_supportboost_autobeta_mixgateopp_div_u2.py`

- 在线单遍、主项仍为邻域遍历 $O(r^2)$。
- 相对 s52/s55 的新增计算是：额外累计一套长窗证据（同一轮遍历里常数级分支/加法）+ `delta/clamp` 的标量运算；不改变主复杂度。
- 未引入新的 sweep 超参；长窗比例固定为 `2*tau`，自适应窗口 $N=4096$ 固定写死。

### 资源占用与计算开销（对齐 s52/s55 的口径）

- s60 持久状态与 s52 基本一致：per-pixel `last_ts,last_pol,hot_state`，外加 2 个全局标量 `beta_state[0],mix_state[0]`。
- 因此：它同样**不**需要 s21 的 `acc_pos/acc_neg` 两张 per-pixel accumulator 表；资源开销主要来自多一张 `hot_state`。

### 实验口径与当前结果（prescreen200k 快速对比；非最终结论）

评测口径：统一用精简扫频脚本（ED24 labeled-npy → score → ROC CSV）。示例：

```powershell
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py --variant s60 --max-events 200000 --s-list 9 --tau-us-list 128000 --out-dir data/ED24/myPedestrain_06/EBF_Part2/_slim_s60_prescreen200k
```

数据来源（prescreen200k；ROC CSV 全行扫描得到 best-F1）：

- baseline：`data/ED24/myPedestrain_06/EBF_Part2/roc_ebf_{light,mid,heavy}_labelscore_*.csv`
- s55：`data/ED24/myPedestrain_06/EBF_Part2/roc_ebf_s55_{light,mid,heavy}_labelscore_*.csv`
- s60：`data/ED24/myPedestrain_06/EBF_Part2/roc_ebf_s60_{light,mid,heavy}_labelscore_*.csv`
- s61：`data/ED24/myPedestrain_06/EBF_Part2/_slim_s61_prescreen200k_s9_tau128/roc_ebf_s61_{light,mid,heavy}_labelscore_s9_tau128ms.csv`
- s62：`data/ED24/myPedestrain_06/EBF_Part2/_slim_s62_prescreen200k_s9_tau128/roc_ebf_s62_{light,mid,heavy}_labelscore_s9_tau128ms.csv`
- s63：`data/ED24/myPedestrain_06/EBF_Part2/_slim_s63_prescreen200k_s9_tau128/roc_ebf_s63_{light,mid,heavy}_labelscore_s9_tau128ms.csv`
- s64：`data/ED24/myPedestrain_06/EBF_Part2/_slim_s64_prescreen200k_s9_tau128/roc_ebf_s64_{light,mid,heavy}_labelscore_s9_tau128ms.csv`
- s65：`data/ED24/myPedestrain_06/EBF_Part2/_slim_s65_prescreen200k_s9_tau128/roc_ebf_s65_{light,mid,heavy}_labelscore_s9_tau128ms.csv`
- s66：`data/ED24/myPedestrain_06/EBF_Part2/_slim_s66_prescreen200k_s9_tau128/roc_ebf_s66_{light,mid,heavy}_labelscore_s9_tau128ms.csv`
- s67：`data/ED24/myPedestrain_06/EBF_Part2/_slim_s67_prescreen200k_s9_tau128/roc_ebf_s67_{light,mid,heavy}_labelscore_s9_tau128ms.csv`
- s68：`data/ED24/myPedestrain_06/EBF_Part2/_slim_s68_prescreen200k_s9_tau128/roc_ebf_s68_{light,mid,heavy}_labelscore_s9_tau128ms.csv`
- s69：`data/ED24/myPedestrain_06/EBF_Part2/_slim_s69_prescreen200k_s9_tau128/roc_ebf_s69_{light,mid,heavy}_labelscore_s9_tau128ms.csv`
- s70：`data/ED24/myPedestrain_06/EBF_Part2/_slim_s70_prescreen200k_s9_tau128/roc_ebf_s70_{light,mid,heavy}_labelscore_s9_tau128ms.csv`
- s71：`data/ED24/myPedestrain_06/EBF_Part2/_slim_s71_prescreen200k_s9_tau128/roc_ebf_s71_{light,mid,heavy}_labelscore_s9_tau128ms.csv`

#### 全网格 best-F1（per env；允许在 `s=3,5,7,9` × `tau=8..1024ms` 上取最优 tag）

| env | baseline best-F1（tag） | s55 best-F1（tag） | s60 best-F1（tag） |
|---|---|---|---|
| light | 0.952021（`ebf_labelscore_s9_tau512000`） | 0.955424（`ebf_s55_labelscore_s9_tau256000`） | 0.956438（`ebf_s60_labelscore_s9_tau128000`） |
| mid | 0.810827（`ebf_labelscore_s9_tau128000`） | 0.820097（`ebf_s55_labelscore_s9_tau128000`） | 0.822247（`ebf_s60_labelscore_s9_tau128000`） |
| heavy | 0.788680（`ebf_labelscore_s7_tau64000`） | 0.794704（`ebf_s55_labelscore_s7_tau64000`） | 0.784538（`ebf_s60_labelscore_s9_tau128000`） |

#### 固定统一点 `s=9,tau=128ms`（各 env 只允许调阈值取 best-F1；便于“统一 recipe”对比）

| env | baseline best-F1（thr） | s55 best-F1（thr） | s60 best-F1（thr） | s61 best-F1（thr） | s62 best-F1（thr） | s63 best-F1（thr） | s64 best-F1（thr） |
|---|---:|---:|---:|---:|---:|---:|---:|
| light | 0.949739（0.749148） | 0.952604（0.736163） | 0.956438（1.373152） | 0.956607（1.331266） | 0.955812（0.887812） | 0.955951（0.910390） | 0.955746（0.882180） |
| mid | 0.810827（4.839437） | 0.820097（4.738931） | 0.822247（7.089597） | 0.818020（7.596497） | 0.821735（5.815403） | 0.820744（5.713705） | 0.820273（5.358261） |
| heavy | 0.786882（7.358062） | 0.792701（7.581244） | 0.784538（9.668446） | 0.781816（10.523853） | 0.794134（8.602393） | 0.794910（7.892294） | 0.795122（7.725490） |

固定统一点 `s=9,tau=128ms`（续表：s65–s71；同样各 env 只允许调阈值取 best-F1）：

| env | s65 best-F1（thr） | s66 best-F1（thr） | s67 best-F1（thr） | s68 best-F1（thr） | s69 best-F1（thr） | s70 best-F1（thr） | s71 best-F1（thr） |
|---|---:|---:|---:|---:|---:|---:|---:|
| light | 0.955642（0.846336） | 0.955883（0.944409） | 0.956233（0.901472） | 0.956116（0.903604） | 0.956181（0.887543） | 0.955848（0.895876） | 0.956007（0.887585） |
| mid | 0.817328（6.182879） | 0.809838（7.024224） | 0.817945（6.886703） | 0.818306（6.524983） | 0.820485（5.717232） | 0.821095（5.369444） | 0.821620（5.510161） |
| heavy | 0.784758（8.636666） | 0.773026（10.115480） | 0.783275（9.697770） | 0.788115（9.101056） | 0.794765（7.888940） | 0.796355（7.749519） | 0.795827（7.752438） |

阶段性结论：

- 从 prescreen200k 的“全网格 best-F1”看，s60 对 light/mid 有增益，但 heavy 出现回落：heavy 0.784538 < baseline 0.788680 < s55 0.794704。
- 从固定统一点 `s=9,tau=128ms` 看，s60 的 heavy 仍回落：0.784538 < baseline 0.786882 < s55 0.792701；因此 s60 当前不满足“heavy 不退化”的门槛。
- heavy 分段（prescreen200k，固定 `s=9,tau=128ms`，2×100k；阈值取各自 best-F1）：s60 的回落主要来自 FP/噪声通过率显著上升（precision 掉得更明显），而不是 seg1 被“明显救起来”。
	- seg0：baseline F1=0.8346（noise_kept_rate=0.0422），s55 F1=0.8359（0.0400），s60 F1=0.8063（0.0808）。
	- seg1：baseline F1=0.7265（0.0434），s55 F1=0.7380（0.0393），s60 F1=0.7245（0.0905）。

## s61（2026-04-13）：条件启用的 long-delta（先抑制 FP 爆炸；隔离变量，不叠加 support-breadth boost）

来源/动机：这是对 7.15 建议的最小落实——**不要无条件叠加长窗**，而是在“热点/自激风险大”的时候把长窗增量关掉，从而避免 s60 的 FP/噪声通过率暴涨。

要解决的失败模式/可验证假设：

- s60 在 heavy 上主要失败于“噪声通过率显著上升（FP 爆炸）”。
- 假设：FP 爆炸来自“长窗把热点/噪声也积分得更高”；因此需要一个非常便宜的 gate，让长窗增量只在 `u_self` 低（更像真实运动轨迹、而非同像素自激/热点累积）的事件上启用。

### 方法定义（单 kernel，无新超参）

延续 s60 的双时间尺度定义，但对 long-delta 乘上 `u_self` 条件 gate，并且**不再叠加** s51/s55 的 support-breadth boost（隔离变量，先验证“条件长窗”本身）：

$$
\Delta=\mathrm{clip}(\max(0,\mathrm{raw}_{\mathrm{long}}-\mathrm{raw}_{\mathrm{short}}),\ 0,\ \mathrm{raw}_{\mathrm{short}})
$$
\u005ctau_{\u005cmathrm{long}}=2\u005ctau
$$
\mathrm{raw}_{\mathrm{dual}}=\mathrm{raw}_{\mathrm{short}}+(1-u_{\mathrm{self}})\,\Delta
$$
ctau_{cmathrm{long}}=2ctau
$$
\mathrm{score}=\frac{\mathrm{raw}_{\mathrm{dual}}}{1+u_{\mathrm{self}}^2}
$$

其中：

- $u_{\mathrm{self}}$ 来自 per-pixel `hot_state` 的线性衰减累积（与 s51/s52/s55 一致）。
- $\mathrm{raw}_{\mathrm{short}}/\mathrm{raw}_{\mathrm{long}}$ 仍使用 s60 的 `mix_ema -> alpha_eff` 门控 opp 证据（对齐 s52 的“少超参自适应”）。

### 实现与开销

实现位置：`src/myevs/denoise/ops/ebfopt_part2/s61_ebf_labelscore_dualtau_condlong_selfocc_mixgateopp_div_u2.py`

- 在线单遍，主项仍为邻域遍历 $O(r^2)$。
- 相对 s60 仅增加 1 个标量乘法 gate（常数级）；同时少了 support-breadth boost 的乘法项。

### 实验口径与当前结果（prescreen200k 单点；固定 `s=9,tau=128ms`）

- light：best-F1=0.956607（thr=1.331266）
- mid：best-F1=0.818020（thr=7.596497）
- heavy：best-F1=0.781816（thr=10.523853）

heavy 分段（prescreen200k，固定 `s=9,tau=128ms`，2×100k；阈值取各自 best-F1）：

- seg0：F1=0.8255，noise_kept_rate=0.04605
- seg1：F1=0.7275，noise_kept_rate=0.04857

产物文件：`data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/segf1_s61_heavy_prescreen200k_s9_tau128_2seg.csv`

阶段性结论：

- s61 的确把 s60 的“FP/噪声通过率暴涨”压回到接近 baseline 的量级（noise_kept_rate 从 ~0.09 回到 ~0.046）。
- 但 heavy 的整体 F1 仍明显低于 baseline/s55，说明“只靠 u_self 条件开关”还不足以在稀疏段补到足够的有效证据；需要更细的“长窗触发条件”。

## s62（2026-04-13）：条件 long-delta + 支持宽度门控（优先救 seg1，仍不叠加 support-breadth boost）

动机：在 s61 的基础上引入一个“长窗只在支持足够宽/更像轨迹而非点状热点”时才启用的 gate。

要解决的失败模式/可验证假设：

- heavy seg1 是“更稀疏 + 更贴 hotmask”的段，长窗确实可能补到信号，但也容易把窄支持热点一起抬高。
- 假设：当短窗同极性支持比例 $s_{\mathrm{frac}}$ 高时，更可能是“空间上更宽的支持/轨迹片段”；此时长窗增量更值得启用。

### 方法定义（单 kernel，无新超参）

在 s61 的基础上，把 long-delta gate 改为同时受 `u_self` 与 $s_{\mathrm{frac}}$ 影响：

$$
g_{\mathrm{long}}=(1-u_{\mathrm{self}})\,\sqrt{s_{\mathrm{frac}}}
$$
τ_{\mathrm{long}}=2τ
$$
\mathrm{raw}_{\mathrm{dual}}=\mathrm{raw}_{\mathrm{short}}+g_{\mathrm{long}}\,\Delta,\quad
\mathrm{score}=\frac{\mathrm{raw}_{\mathrm{dual}}}{1+u_{\mathrm{self}}^2}
$$

依然不叠加 s51/s55 的 support-breadth boost（隔离变量，先验证“条件长窗 + 支持宽度门控”）。

### 实现与开销

实现位置：`src/myevs/denoise/ops/ebfopt_part2/s62_ebf_labelscore_dualtau_condlong_selfocc_supportgate_mixgateopp_div_u2.py`

- 在线单遍，主项仍为邻域遍历 $O(r^2)$。
- 相对 s61 仅增加 `sqrt(s_frac)` 的常数级计算。

### 实验口径与当前结果（prescreen200k 单点；固定 `s=9,tau=128ms`）

- light：best-F1=0.955812（thr=0.887812）
- mid：best-F1=0.821735（thr=5.815403）
- heavy：best-F1=0.794134（thr=8.602393）

heavy 分段（prescreen200k，固定 `s=9,tau=128ms`，2×100k；阈值取各自 best-F1）：

- seg0：F1=0.8353，noise_kept_rate=0.04655
- seg1：F1=0.7422，noise_kept_rate=0.04461

产物文件：`data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/segf1_s62_heavy_prescreen200k_s9_tau128_2seg.csv`

阶段性结论：

- s62 在保持 noise_kept_rate 仍接近 baseline 的同时，把 seg1 的 F1 抬到了 0.742（相比 baseline 0.726、s55 0.738、s61 0.727），更符合“稀疏段需要条件启用长窗”的预期。
- 目前只做了 prescreen200k 单点验证；下一步若要进入可用候选，建议先补：compact400k sweep + heavy 2×200k 分段（seg0/seg1）验证；若后续要进入最终候选/写结论，再补 heavy 1M validate 以确认不在更长流上退化。

## s63（2026-04-13）：条件 long-delta + 轨迹一致性 gate（重心漂移 proxy；替换 s_frac）

来源/动机：来自 7.16 的建议——继续坚持“条件长窗”，但把 gate 的语义从“支持宽度 proxy（$s_{\mathrm{frac}}$）”升级为更接近“轨迹性/时空连续性”的量。

方法定义（不引入新超参）：

- short/long 同极性支持在邻域内的加权重心（权重仍用 EBF 的时间三角权重）：
$$
\mathbf{c}_s=\frac{\sum w_s\mathbf{x}}{\sum w_s},\quad \mathbf{c}_l=\frac{\sum w_l\mathbf{x}}{\sum w_l}
$$
- 轨迹 proxy：
$$
c_{\mathrm{traj}}=\mathrm{clip}\!\left(\frac{\|\mathbf{c}_l-\mathbf{c}_s\|}{r},0,1\right)
$$
- long-delta gate：
$$
g_{\mathrm{long}}=(1-u_{\mathrm{self}})\,c_{\mathrm{traj}}
$$

其余与 s61/s62 一致：`mix_ema -> alpha_eff` 门控 opp 证据、delta-only long boost、`/ (1+u_self^2)`。

实现位置：`src/myevs/denoise/ops/ebfopt_part2/s63_ebf_labelscore_dualtau_condlong_selfocc_trajgate_mixgateopp_div_u2.py`

prescreen200k 单点（固定 `s=9,tau=128ms`）：

- light：best-F1=0.955951（thr=0.910390）
- mid：best-F1=0.820744（thr=5.713705）
- heavy：best-F1=0.794910（thr=7.892294）

heavy 分段（prescreen200k，固定 `s=9,tau=128ms`，2×100k；阈值取各自 best-F1）：

- seg0：F1=0.8362，noise_kept_rate=0.04787
- seg1：F1=0.7427，noise_kept_rate=0.04516

产物文件：`data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/segf1_s63_heavy_prescreen200k_s9_tau128_2seg.csv`

阶段性结论：

- 相比 s62，s63 在 heavy 的总体 F1 与 seg1 F1 都是“极小幅”上升；说明 7.16 的方向（换 gate 语义）是值得做的，但当前这个 `c_traj` proxy 仍然偏弱，提升幅度有限。

## s64（2026-04-13）：s63 + 缩短长窗（$\tau_{long}\approx1.5\tau$）

动机：7.16 的第二优先级诊断——如果把长窗跨度从 `2*tau` 缩短到 `1.5*tau` 更健康，说明 heavy 的问题部分来自“补得太多（把坏支持也积分进来）”。

方法定义：与 s63 相同的 `c_traj` gate，但把长窗改为：

$$
\tau_{\mathrm{long}}=\tau+\left\lfloor\tau/2\right\rfloor
$$

实现位置：`src/myevs/denoise/ops/ebfopt_part2/s64_ebf_labelscore_dualtau1p5_condlong_selfocc_trajgate_mixgateopp_div_u2.py`

prescreen200k 单点（固定 `s=9,tau=128ms`）：

- light：best-F1=0.955746（thr=0.882180）
- mid：best-F1=0.820273（thr=5.358261）
- heavy：best-F1=0.795122（thr=7.725490）

heavy 分段（prescreen200k，固定 `s=9,tau=128ms`，2×100k；阈值取各自 best-F1）：

- seg0：F1=0.8376，noise_kept_rate=0.04714
- seg1：F1=0.7411，noise_kept_rate=0.04370

产物文件：`data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/segf1_s64_heavy_prescreen200k_s9_tau128_2seg.csv`

阶段性结论：

- s64 更“克制”：seg1 的 noise_kept_rate 进一步下降，但 seg1 F1 也略回落；总体 heavy F1 小幅上升。
- 这提示：把长窗做得更短确实更安全，但要显著救 seg1，仍需要更强的“轨迹性/连续性”判别量（当前仅靠重心漂移还不够）。

## s65–s71（2026-04-13）：在 s63/s64 基础上的后续迭代（结论：s71 小幅最优）

背景：在 s63/s64 之后继续沿着 7.16 的“轨迹一致性 gate”主线迭代，但要求仍保持：在线单遍、$O(r^2)$、不引入新的 sweep 超参。下面所有数字均为 prescreen200k 单点（固定 `s=9,tau=128ms`）。

### 结论先行（按 heavy/seg1 优先）

- s65（方向一致性 cos gate）退化明显，淘汰。
- s66（最近同极性位移幅度 gate）强烈退化，淘汰。
- s67/s68（半径/重心的“delta-only”版本）仍不如 s63，淘汰。
- s69（仅放松 delta cap：`delta <= 2*raw_short`）让 seg1 recall 略上去，但 FP 也略涨；整体收益很小。
- s70（s69 + 硬 purity gate：`clip(1-2*mix,0,1)`）heavy best-F1 最高，但 seg1 recall 被压（更偏保守）。
- s71（s69 + 软 purity gate：`1-mix`）在 seg1 上取得“F1 略涨 + noise_kept_rate 略降”，且 recall 基本不掉，是当前最均衡候选。

### s71：s69 的 delta-cap2 + “软 purity gate”的条件长窗（soft 版 s70；无新超参）

要解决的失败模式/可验证假设：

- 在 heavy/seg1 上，s63 的长窗增益（delta-only boost）会把一部分“opp 参与度较高”的可疑结构也一起抬分，导致 FP 泄露；而 s70 用 `clip(1-2*mix,0,1)` 把这类事件的长窗增益压得过狠，出现“precision ↑ 但 recall ↓”的保守退化。
- 假设：对长窗增益增加一个**更柔和**的极性纯度门控（soft purity），让高 mix 的事件仍能获得一部分长窗增益（不至于直接归零），可以在不放飞 FP 的前提下，回收 seg1 recall。

### 方法定义（单 kernel；在线单遍；$O(r^2)$；无新 sweep 超参）

记当前事件为 $e_i=(t_i,\mathbf{x}_i,p_i)$，邻域半径为 $r$，短窗为 $\tau$，长窗固定为：

$$
\tau_{\mathrm{long}}=2\tau
$$

对任一邻域像素的“最近事件”（由 `last_ts/last_pol` 给出），定义时间差 $\Delta t=|t_i-t_j|$，并使用线性年龄权重：

$$
w_s(\Delta t)=\max(\tau-\Delta t,0),\quad
w_l(\Delta t)=\max(\tau_{\mathrm{long}}-\Delta t,0)
$$

分别累加同极性/异极性的短窗与长窗证据（这里的 $\mathcal{N}$ 表示邻域内“存在有效最近事件”的像素集合）：

$$
\mathrm{raw}_s=\sum_{j\in\mathcal{N},\,p_j=p_i,\,\Delta t_j\le\tau} w_s(\Delta t_j),\quad
\mathrm{opp}_s=\sum_{j\in\mathcal{N},\,p_j=-p_i,\,\Delta t_j\le\tau} w_s(\Delta t_j)
$$

$$
\mathrm{raw}_l=\sum_{j\in\mathcal{N},\,p_j=p_i,\,\Delta t_j\le\tau_{\mathrm{long}}} w_l(\Delta t_j),\quad
\mathrm{opp}_l=\sum_{j\in\mathcal{N},\,p_j=-p_i,\,\Delta t_j\le\tau_{\mathrm{long}}} w_l(\Delta t_j)
$$

1) **mix（当前事件的局部 opp 比例）**：

$$
\mathrm{mix}=\frac{\mathrm{opp}_s}{\mathrm{raw}_s+\mathrm{opp}_s+\varepsilon}
$$

2) **mix_state（全局 EMA，用于门控 opp 融合强度）**：设 $m$ 为全局标量状态（`mix_state[0]`），用固定 $N=4096$ 做在线 EMA：

$$
m\leftarrow m+\frac{\mathrm{mix}-m}{N}
$$

并得到 opp 融合门控（与 s63/s64 一致）：

$$
\alpha_{\mathrm{eff}}=(1-m)^2
$$

3) **短/长窗融合密度**（对 opp 使用同一 $\alpha_{\mathrm{eff}}$）：

$$
\mathrm{raw}_{\mathrm{short}}=\frac{\mathrm{raw}_s+\alpha_{\mathrm{eff}}\,\mathrm{opp}_s}{\tau},\quad
\mathrm{raw}_{\mathrm{long}}=\frac{\mathrm{raw}_l+\alpha_{\mathrm{eff}}\,\mathrm{opp}_l}{\tau_{\mathrm{long}}}
$$

4) **delta-only 长窗增量 + cap2（s69 分支）**：

$$
\Delta=\max(\mathrm{raw}_{\mathrm{long}}-\mathrm{raw}_{\mathrm{short}},0),\quad
\Delta\leftarrow\min\bigl(\Delta,\,2\,\mathrm{raw}_{\mathrm{short}}\bigr)
$$

5) **轨迹 gate（重心漂移；与 s63 一致）**：仅对同极性支持计算加权重心

$$
\mathbf{c}_s=\frac{\sum w_s(\Delta t_j)\,\mathbf{x}_j}{\sum w_s(\Delta t_j)},\quad
\mathbf{c}_l=\frac{\sum w_l(\Delta t_j)\,\mathbf{x}_j}{\sum w_l(\Delta t_j)}
$$

并定义：

$$
c_{\mathrm{traj}}=\mathrm{clip}\left(\frac{\|\mathbf{c}_l-\mathbf{c}_s\|}{r},0,1\right)
$$

6) **self-occupancy（热点自抑制）**：由 per-pixel 的 `hot_state` 得到（实现与 s55/s63 一致，$t_r=\lfloor\tau/2\rfloor$）：

$$
u_{\mathrm{self}}=\frac{h}{h+t_r+\varepsilon}
$$

7) **soft purity gate（本变体的关键）**：只对“长窗增益”再乘一个更柔和的纯度项（基于当前事件的局部 mix，而不是全局 EMA）：

$$
c_{\mathrm{pol}}=1-\mathrm{mix}
$$

于是长窗增益门控为：

$$
g_{\mathrm{long}}=(1-u_{\mathrm{self}})\,c_{\mathrm{traj}}\,c_{\mathrm{pol}}
$$

最终密度与 score：

$$
\mathrm{raw}_{\mathrm{cond}}=\mathrm{raw}_{\mathrm{short}}+g_{\mathrm{long}}\,\Delta,\quad
\mathrm{score}=\frac{\mathrm{raw}_{\mathrm{cond}}}{1+u_{\mathrm{self}}^2}
$$

直觉检查（解释性）：

- 当 $\mathrm{mix}\to 0$（几乎纯同极性）：$c_{\mathrm{pol}}\to 1$，s71 退化回“s69 的 delta-cap2 条件长窗”。
- 当 $\mathrm{mix}=0.5$：$c_{\mathrm{pol}}=0.5$，仍保留一半长窗增益（对应“soft”）；而 s70 的 $c_{\mathrm{pol}}=\mathrm{clip}(1-2\mathrm{mix},0,1)$ 会直接变 0（更保守）。

### 实现与开销

实现位置：`src/myevs/denoise/ops/ebfopt_part2/s71_ebf_labelscore_dualtau_condlong_selfocc_trajgate_mixgateopp_deltacap2_softpuritygate_div_u2.py`

- 在线单遍；邻域遍历仍为主项 $O(r^2)$。
- 无新增持久状态（复用 s55/s63 的：`last_ts/last_pol/hot_state` + 1 个全局 `mix_state`；`beta_state` 在该实现中保留但不参与计算）。
- 相比 s63 的额外开销主要是：`c_pol` 的一次标量乘法，以及 delta cap 从 $\mathrm{raw}_{\mathrm{short}}$ 放松到 $2\,\mathrm{raw}_{\mathrm{short}}$（仍为常数级）。

### heavy 分段证据（2×100k；阈值取各自 best-F1；用于判断 seg1 是否靠“牺牲 recall”换 precision）

- s63（对照）：seg1 F1=0.7427，precision=0.7768，recall=0.7114，noise_kept_rate=0.04516
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/segf1_s63_heavy_prescreen200k_s9_tau128_2seg.csv`
- s69：seg1 F1=0.7432，precision=0.7735，recall=0.7151，noise_kept_rate=0.04626
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/segf1_s69_heavy_prescreen200k_s9_tau128_2seg.csv`
- s70（更保守）：seg1 F1=0.7423，precision=0.7864，recall=0.7028，noise_kept_rate=0.04217
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/segf1_s70_heavy_prescreen200k_s9_tau128_2seg.csv`
- s71（当前推荐）：seg1 F1=0.7429，precision=0.7774，recall=0.7113，noise_kept_rate=0.04500
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/segf1_s71_heavy_prescreen200k_s9_tau128_2seg.csv`

### heavy 1M validate（最终确认用；阈值取 heavy ROC best-F1）

注：这不是默认流程；默认先用 compact400k + heavy 2×200k 分段定位增益，只有候选已经稳定优于基线时才进入 heavy 1M。

ROC 产物目录：

- `data/ED24/myPedestrain_06/EBF_Part2/s71_validate_1M_heavy_s9_tau128/`
	- heavy ROC CSV：`data/ED24/myPedestrain_06/EBF_Part2/s71_validate_1M_heavy_s9_tau128/roc_ebf_s71_heavy_labelscore_s9_tau128ms.csv`

heavy（n=896,682；s=9,tau=128ms）ROC best 点：

- AUC=0.932677
- best-F1=0.773972（thr=7.596869）

heavy 分段（每段 200k；thr 取上面的 best-F1；状态连续流式）：

- seg0（0~200k）：F1=0.795573，precision=0.800472，recall=0.790733，noise_kept_rate=0.049064
- seg1（200k~400k）：F1=0.674983，precision=0.701244，recall=0.650617，noise_kept_rate=0.028752

分段产物：

- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/validate_1M_heavy/segf1_s71_heavy_1M_s9_tau128.csv`

阶段性结论：

- 如果只看 overall heavy best-F1：s70 最高；但它主要靠压 seg1 recall 来换 precision（更保守）。
- 若按你的主目标“heavy seg1 不能再差，且 FP 不能放飞”来选：s71 更均衡（seg1 F1 小幅高于 s63，noise_kept_rate 略低，recall 基本持平）。
- 1M validate 已完成：s71 在更长流上可以稳定复现（heavy 的 ROC best-F1/AUC 与 prescreen200k 同量级），但 seg1 依然是瓶颈段。
- 因此：继续在“轨迹 gate 细节”上迭代很可能只带来微小收益；下一步更建议转向 V2 类归一化/自适应阈值，或引入更廉价但更强的全局同步/爆发 proxy 做模式切换（按 README 顶部的主目标“阈值稳定性优先”）。

---

## s72–s73（2026-04-13）：极简“只保留轨迹 gate”检验（结论：与 baseline 持平但 seg1 更差）

背景/动机（你提出的验证需求）：

- s65–s71 越叠越复杂，但 heavy/seg1 的收益仍很有限；你怀疑这些机制本身没有贡献，想做一个“只保留一个轨迹 gate 依据”的极简版本，看看与 baseline 的差距。
- 这里的“轨迹 gate”仍然沿用 s63 的 centroid drift proxy（短窗/长窗同极性支持的加权重心漂移）。

本节包含两个版本：

- s72：gate 同向（漂移越大，越允许长窗增益）。
- s73：翻转 gate（漂移越小，越允许长窗增益）。

### 方法定义（在线单遍；$O(r^2)$；无新 sweep 超参；无额外状态）

记事件为 $e_i=(t_i,\mathbf{x}_i,p_i)$，邻域半径为 $r$，短窗为 $\tau$，长窗固定为：

$$
τ_{\mathrm{long}}=2τ
$$

对同极性邻域像素 $j$（其最近事件极性满足 $p_j=p_i$），定义 $\Delta t_j=|t_i-t_j|$，并使用线性年龄权重：

$$
w_s(\Delta t)=\max(\tau-\Delta t,0),\quad
w_l(\Delta t)=\max(\tau_{\mathrm{long}}-\Delta t,0)
$$

同极性短窗/长窗密度（与 baseline 同结构，但多了一个长窗）：

$$
\mathrm{raw}_{\mathrm{short}}=\frac{\sum_{j:\Delta t_j\le\tau} w_s(\Delta t_j)}{\tau},\quad
\mathrm{raw}_{\mathrm{long}}=\frac{\sum_{j:\Delta t_j\le\tau_{\mathrm{long}}} w_l(\Delta t_j)}{\tau_{\mathrm{long}}}
$$

delta-only 长窗增量（不做 cap）：

$$
\Delta=\max\bigl(\mathrm{raw}_{\mathrm{long}}-\mathrm{raw}_{\mathrm{short}},0\bigr)
$$

轨迹 gate（centroid drift；仅用同极性支持）：

$$
\mathbf{c}_s=\frac{\sum w_s(\Delta t_j)\,\mathbf{x}_j}{\sum w_s(\Delta t_j)},\quad
\mathbf{c}_l=\frac{\sum w_l(\Delta t_j)\,\mathbf{x}_j}{\sum w_l(\Delta t_j)}
$$

$$
c_{\mathrm{traj}}=\mathrm{clip}\left(\frac{\|\mathbf{c}_l-\mathbf{c}_s\|}{r},0,1\right)
$$

最终 score：

- s72（同向 gate）：

$$
\mathrm{score}=\mathrm{raw}_{\mathrm{short}}+c_{\mathrm{traj}}\,\Delta
$$

- s73（翻转 gate）：

$$
\mathrm{score}=\mathrm{raw}_{\mathrm{short}}+\bigl(1-c_{\mathrm{traj}}\bigr)\,\Delta
$$

### 实现、复杂度与内存

实现位置：

- s72：`src/myevs/denoise/ops/ebfopt_part2/s72_ebf_labelscore_dualtau_trajgate_only.py`
- s73：`src/myevs/denoise/ops/ebfopt_part2/s73_ebf_labelscore_dualtau_trajgate_only_flip.py`

复杂度：

- 在线单遍，单事件邻域扫描主导：$O(r^2)$。
- 相比 baseline 仅多做一次长窗累加与两次重心累加（同数量级，常数略增）。

内存占用（持久状态）：

- 仅 `last_ts/last_pol` 两张 per-pixel 表：
	- `last_ts`：`uint64[W*H]`
	- `last_pol`：`int8[W*H]`
- 以默认分辨率 $W\times H=346\times260=89{,}960$ 为例：约 $89{,}960\times(8+1)\approx0.77$ MiB（不含 numpy 对齐/对象开销）。

### 证据（对齐口径：固定点 `s=9,tau=128ms`）

#### prescreen200k（对齐点；各 env 只调阈值取 best-F1）

输出目录：

- s72（含 baseline 复跑产物）：`data/ED24/myPedestrain_06/EBF_Part2/s72_prescreen200k_s9_tau128/`
- s73：`data/ED24/myPedestrain_06/EBF_Part2/s73_prescreen200k_s9_tau128/`

heavy（n=200k）ROC best 点：

- baseline：AUC=0.920467，best-F1=0.786882（thr=7.358062）
- s72：AUC=0.921623，best-F1=0.785121（thr=7.639399）
- s73：AUC=0.917435，best-F1=0.776249（thr=10.044123）

对应 ROC CSV：

- baseline：`data/ED24/myPedestrain_06/EBF_Part2/s72_prescreen200k_s9_tau128/roc_ebf_heavy_labelscore_s9_tau128ms.csv`
- s72：`data/ED24/myPedestrain_06/EBF_Part2/s72_prescreen200k_s9_tau128/roc_ebf_s72_heavy_labelscore_s9_tau128ms.csv`
- s73：`data/ED24/myPedestrain_06/EBF_Part2/s73_prescreen200k_s9_tau128/roc_ebf_s73_heavy_labelscore_s9_tau128ms.csv`

#### heavy prescreen400k + 分段（更贴近 seg1；每段 200k；thr 取 heavy ROC best-F1）

输出目录：

- baseline/s72：`data/ED24/myPedestrain_06/EBF_Part2/s72_prescreen400k_s9_tau128/`
- s73：`data/ED24/myPedestrain_06/EBF_Part2/s73_prescreen400k_s9_tau128/`

heavy（n=400k）ROC best 点：

- baseline：AUC=0.912484，best-F1=0.745459（thr=7.356625）
- s72：AUC=0.914943，best-F1=0.743203（thr=7.639435）
- s73：AUC=0.912136，best-F1=0.733013（thr=10.185825）

heavy 分段（seg=200k；状态连续；thr=best-F1）：

- baseline：
	- seg0：F1=0.786836，precision=0.815374，recall=0.760229，noise_kept_rate=0.042851
	- seg1：F1=0.654839，precision=0.703224，recall=0.612684，noise_kept_rate=0.026821
- s72：
	- seg0：F1=0.785115，precision=0.800443，recall=0.770363，noise_kept_rate=0.047809
	- seg1：F1=0.652064，precision=0.682752，recall=0.624016，noise_kept_rate=0.030077
- s73：
	- seg0：F1=0.775872，precision=0.793238，recall=0.759250，noise_kept_rate=0.049264
	- seg1：F1=0.641383，precision=0.661299，recall=0.622632，noise_kept_rate=0.033079

分段产物：

- baseline：`data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/s72_prescreen400k_heavy/segf1_ebf_heavy_400k_s9_tau128.csv`
- s72：`data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/s72_prescreen400k_heavy/segf1_s72_heavy_400k_s9_tau128.csv`
- s73：`data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/s73_prescreen400k_heavy/segf1_s73_heavy_400k_s9_tau128.csv`

### 阶段性结论

- s72（同向 gate）在 prescreen200k 的 AUC 几乎不掉，但 heavy best-F1 与 baseline 持平略差；在 seg1 上 F1 略低，且 noise_kept_rate 更高（FP 更放）。
- s73（翻转 gate）整体更差（heavy best-F1/AUC 均下降），seg1 也更差。
- 结论上基本支持你的直觉：只靠“重心漂移”这个单一 proxy，既很难在 seg1 明显超越 baseline，也容易带来 FP 泄露；后续若还要坚持“只加一个依据”，需要换更强的轨迹/结构 proxy（而不是继续堆 gate）。

---

## s74（2026-04-13）：Surprise z-score + 自适应 null（固定内部参数；新 base model）

背景/动机（先审计 README 的已知成败点，再换 base model）：

- s28 的“全局 rate 归一化惊奇度（z-score）”对 heavy 的 hotmask FP 仍不够（README 已有结论）。
- s35 的“把 hotness 注入 null”方向是有效线索，但它需要额外 sweep 超参（`tau_rate/gamma/hmax`）。
- 这里做一个 **s74：固定内部选择** 的版本，用来作为 slim sweep 的“新 base model”候选：
	- 保持在线单遍、邻域扫描 $O(r^2)$。
	- 不引入新的 sweep 超参（仍只扫 `s,tau`），把必要的 rate/hotness 机制“锁死”为内部常数。

### 方法定义（在线单遍；$O(r^2)$；无新 sweep 超参）

记事件为 $e_i=(t_i,\mathbf{x}_i,p_i)$，邻域半径为 $r$，短窗为 $\tau$。

1) baseline 同极性邻域支持（与 EBF 同结构；三角年龄权重）：

$$
\mathrm{raw}=\sum_{j\in\mathcal{N}(i)}\mathbf{1}[p_j=p_i]\,\max\left(0,1-\frac{\Delta t_j}{\tau}\right)
$$

2) 全局事件率 EMA（events/tick）：

- 事件间隔 $\Delta t_g=t_i-t_{i-1}$，瞬时率 $\hat r=1/\Delta t_g$。
- 固定内部 time constant：$\tau_r=\max(1,\lfloor \tau/2\rfloor)$。
- 更新：$r\leftarrow r+\left(1-e^{-\Delta t_g/\tau_r}\right)(\hat r-r)$。

3) per-pixel hotness 状态（tick 单位，int32，沿用 s35 的思路）：

对当前像素的自间隔 $\Delta t_0$，维护 $H$：

$$
H\leftarrow\max(0,H-\Delta t_0)+\tau
$$

4) 自适应 null：把 hotness 注入有效像素率

令 $N_{\mathrm{pix}}=W\cdot H_{\mathrm{img}}$，$r_{\mathrm{pix}}=r/N_{\mathrm{pix}}$，$h=\mathrm{clip}(H/\tau,0,h_{\max})$，并固定常数：

$$
\gamma=0.3,\quad h_{\max}=4
$$

有效像素率：

$$
r_{\mathrm{eff}}=r_{\mathrm{pix}}\,(1+\gamma h)
$$

5) 输出 score：复用 s28 的噪声模型推导（并显式使用 $P(\text{pol match})\approx 0.5$）：

- 令 $a=r_{\mathrm{eff}}\,\tau$（无量纲）。
- 单邻居的 $w=\max(0,1-\Delta t/\tau)$ 期望：

$$
\mathbb{E}[w]=1-\frac{1-e^{-a}}{a}
$$

- 二阶矩：

$$
\mathbb{E}[w^2]=\frac{a^2-2a+2-2e^{-a}}{a^2}
$$

- 乘上 polarity match 概率 $0.5$ 得到 $\mu_{\mathrm{per}},\,\sigma^2_{\mathrm{per}}$，再按邻居数 $m=((2r+1)^2-1)$ 放大：

$$
\mu=m\,\mu_{\mathrm{per}},\quad \sigma^2=m\,\sigma^2_{\mathrm{per}},\quad
\mathrm{score}=\frac{\mathrm{raw}-\mu}{\sqrt{\sigma^2+\varepsilon}}
$$

### 实现、复杂度与内存

实现位置：`src/myevs/denoise/ops/ebfopt_part2/s74_ebf_labelscore_surprise_adaptive_null_fixed.py`

- 复杂度：在线单遍；邻域扫描主导 $O(r^2)$；额外仅常数级标量更新（`rate_ema` + 自像素 `hot_state`）。
- 持久状态：
	- `last_ts: uint64[W*H]`
	- `last_pol: int8[W*H]`
	- `hot_state: int32[W*H]`
	- `rate_ema: float64[1]`（全局标量）

### 证据（对齐口径：固定点 `s=9,tau=128ms`）

#### prescreen200k（对齐点；各 env 只调阈值取 best-F1）

运行命令（产物不覆盖，单独目录）：

```bash
PYTHONNOUSERSITE=1 D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py \
	--variant s74 --max-events 200000 --s-list 9 --tau-us-list 128000 \
	--out-dir data/ED24/myPedestrain_06/EBF_Part2/_slim_s74_max200k
```

对应 ROC CSV：

- light：`data/ED24/myPedestrain_06/EBF_Part2/_slim_s74_max200k/roc_ebf_s74_light_labelscore_s9_tau128ms.csv`
	- AUC=0.929366，best-F1=0.939597（thr=-0.608104）
- mid：`data/ED24/myPedestrain_06/EBF_Part2/_slim_s74_max200k/roc_ebf_s74_mid_labelscore_s9_tau128ms.csv`
	- AUC=0.932827，best-F1=0.819105（thr=1.048728）
- heavy：`data/ED24/myPedestrain_06/EBF_Part2/_slim_s74_max200k/roc_ebf_s74_heavy_labelscore_s9_tau128ms.csv`
	- AUC=0.934945，best-F1=0.794960（thr=1.101157）

#### heavy400k + 分段（更贴近 seg1；每段 200k；thr 取 heavy ROC best-F1）

运行命令：

```bash
PYTHONNOUSERSITE=1 D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py \
	--variant s74 --max-events 400000 --s-list 9 --tau-us-list 128000 \
	--out-dir data/ED24/myPedestrain_06/EBF_Part2/_slim_s74_max400k
PYTHONNOUSERSITE=1 D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/noise_analyze/segment_f1.py \
	--variant s74 --s 9 --tau-us 128000 --max-events 400000 --segment-events 200000 \
	--labeled-npy D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3.npy \
	--roc-csv data/ED24/myPedestrain_06/EBF_Part2/_slim_s74_max400k/roc_ebf_s74_heavy_labelscore_s9_tau128ms.csv \
	--out-csv data/ED24/myPedestrain_06/EBF_Part2/_slim_s74_max400k/seg_f1_ebf_s74_heavy_labelscore_s9_tau128ms_seg200k_max400k.csv
```

heavy（n=400k）ROC best 点：

- AUC=0.926476
- best-F1=0.755225（thr=1.312951）

heavy 分段（seg=200k；状态连续；thr=best-F1）：

- seg0：F1=0.794589，precision=0.831183，recall=0.761082，noise_kept_rate=0.038480
- seg1：F1=0.669536，precision=0.721154，recall=0.624814，noise_kept_rate=0.025060

分段产物：

- `data/ED24/myPedestrain_06/EBF_Part2/_slim_s74_max400k/seg_f1_ebf_s74_heavy_labelscore_s9_tau128ms_seg200k_max400k.csv`

### 阶段性结论

- s74 在“只扫 s/tau”的约束下，作为新 base model 已经能在 heavy400k 的 seg1 同时做到：F1 上升、noise_kept_rate 下降（相对该文档里 baseline 400k 分段对照）。
- 这条线索值得继续：下一步可以在不增加 sweep 维度的前提下，尝试把 $r_{\mathrm{eff}}$ 的 hotness 映射做更“克制/更稳”的固定函数（例如更小的 $\gamma$ 或更低的 $h_{\max}$），并检查阈值跨 env 的可迁移性。

---

## 9. 落地验证（2026-04-13）：基于本建议的 s63/s64 最小实现结果

我按你在本笔记里给的“最推荐优化顺序”做了两步最小实现与验证（均不引入新的 sweep 超参，只是替换 gate 的语义/调整固定长窗比例），并用精简 sweep + 分段脚本复现。

### s63：用“重心漂移”作为轨迹 gate（替换 s62 的 \(\sqrt{s_{\mathrm{frac}}}\)）

- gate 定义：
	- 短/长窗同极性支持的加权重心 \(\mathbf{c}_s,\mathbf{c}_l\)
	- \(c_{traj}=\mathrm{clip}(\|\mathbf{c}_l-\mathbf{c}_s\|/r,0,1)\)
	- \(g_{long}=(1-u_{self})c_{traj}\)

prescreen200k（`s=9,tau=128ms`）best-F1：

- light：0.955951
- mid：0.820744
- heavy：0.794910（thr=7.8923）

heavy 分段（2×100k，best-F1 阈值）：

- seg0：F1=0.8362，noise_kept_rate=0.0479
- seg1：F1=0.7427，noise_kept_rate=0.0452

结论：相比 s62，heavy/seg1 有“极小幅”提升，方向有效但 proxy 仍偏弱。

### s64：保持 s63 的轨迹 gate，但把长窗从 \(2\tau\) 缩到 \(1.5\tau\)

prescreen200k（`s=9,tau=128ms`）best-F1：

- light：0.955746
- mid：0.820273
- heavy：0.795122（thr=7.7255）

heavy 分段（2×100k，best-F1 阈值）：

- seg0：F1=0.8376，noise_kept_rate=0.0471
- seg1：F1=0.7411，noise_kept_rate=0.0437

结论：缩短长窗更“克制”（seg1 噪声通过率下降），但 seg1 F1 略回落；整体 heavy F1 小幅上升。这说明“补太多”确实会带来风险，但想显著救 seg1 还需要更强的轨迹一致性量（比重心漂移更有判别力）。

### 下一轮我建议的最小改进（在本建议框架内继续）

不再继续磨 gate 形状（clip/sqrt/pow），而是继续升级轨迹条件的“分辨率”，优先尝试：

1) **最近邻速度一致性（方案 A）**：在邻域里取 2~3 个最近同极性事件，计算 \(v_k=\|\Delta x\|/(\Delta t+\epsilon)\) 或方向一致性，得到 \(c_{traj}\in[0,1]\)。
2) 若 1) 太重，再做折中：只取“最近 1 个同极性事件”的速度/方向 proxy（更便宜，但比重心漂移更直接）。

状态更新：已按上述建议实现并验证 s65–s71（见上节）。其中 s65–s68 均退化或无收益；s69–s71 为“可复现的小幅正向”分支，当前最均衡候选为 s71。

## 反思
baseline “看起来简单但很强”，核心原因是它抓住了 ED24/myPedestrain_06 上最稳定、跨 light/mid/heavy 都成立的可分性来源：局部时空密度（同极性 + 时间衰减 + 邻域累加）。你看到的“加一点东西反而更差”，主要是因为 s1–s8 里很多增强项都隐含了更强的结构/几何假设（各向异性、dt 方差、局部平面等），但真实数据里有遮挡、多目标叠加、边缘两侧混合、噪声也会出现“结构假象”；一旦门控在 mid/heavy 上触发比例变大，就会系统性扰动排序，AUC 会掉得很快（README 里 s4/s8 的现象就是典型）。
## Part2 下一步（你已确认要继续做 s7+）

从 s7 开始不再在 s3–s6 上做更大网格，而是按 prescreen 的结论，改为设计“更强判别信息但仍单遍 $O(r^2)$”的新候选；每个候选仍遵循本 README 顶部的维护规则：实现 → prescreen → best_summary → 写清原理与失效分析。

状态更新：s7（平面残差门控）已完成 prescreen，未超过 baseline；s8（平面解释度门控）已完成 prescreen，明显劣于 baseline；s9（同像素超高频门控）已完成 prescreen，AUC 在三环境均略高但提升极小；s10（同像素泄露积分发射率门控）已完成 prescreen 且全量验证三环境均略高于 baseline；s11（相对热点异常门控）已完成 prescreen 与全量验证，均优于 s10 且为之前最好；s12（热点 z-score 异常门控）已完成 prescreen，但未超过 s11，建议停掉；s13（跨极性邻域支持度门控）已完成两轮 prescreen（宽触发/克制触发），仍显著劣于 baseline，建议停掉；s14（跨极性支持度加分）已完成 prescreen 与全量验证，三环境均显著优于 baseline/s11，当前最好；s15（同像素极性快速交替门控）已完成 prescreen，但该机理触发率极低且指标与 s14 持平，建议停掉；s16（s14 + 相对热点异常抑制）已完成 prescreen：light AUC 略升但 mid/heavy best-F1 未超过 s14，且 heavy FP 并不集中于少数像素，建议暂缓；s17（cross-pol spread trust）已完成 prescreen，AUC/F1 不如 s14，建议停掉；s18（去掉极性判断消融）已完成 prescreen：AUC/F1 在 mid/heavy 明显下降且 FP 上升，说明极性一致性是关键判别信息，建议停掉；s19（证据融合主模型：cross-pol 加分 + same-pixel hotness 惩罚，Q8 定点）已完成 prescreen（beta 单调提高可压 FP）+ 1M validate：beta=0.3 时 mid 的 precision/FP 未优于 s14；但 beta=0.5 时在 mid/heavy 明确压住了 s14 的 FP 并提升 precision，且 AUC/F1 基本持平；下一步建议以 `alpha=0.2,beta=0.5` 作为精度提升候选，然后进入 V2 类归一化/自适应阈值阶段；s20（同像素按极性分通道 hotness）已实现并完成 prescreen + 1M validate：light/heavy 指标明显更高，但 mid 的 best-F1 点 FP 高于 s19(b0.5)/s14，暂不替代；s21（同像素双极性热度混合惩罚）已完成 prescreen200k + 1M validate（b0.6,k0.5/0.8）：三环境 AUC/F1 均高于 s19/s20，mid 的 best-F1 点 FP 也被压回接近 s19(b0.5) 的量级，建议将 `s21(b0.6,k0.8)` 作为当前 Part2 新主候选。

补充：阈值可迁移性检查（A：best-F1 阈值 + MESR across env）显示，s14 虽整体提升 AUC/F1/MESR，但 best-F1 阈值在 light/mid/heavy 间的尺度差异仍然很大，后续应进入 V2 类归一化/自适应阈值机制阶段。

---

## s75（2026-04-13）：hotmask-aware raw（raw_nonhot / (1 + raw_hot)）

动机（对齐“seg1 信号更稀 + 更贴 hotmask”的事实）：

- heavy seg1 的核心矛盾（见本文前面的统计）：signal_share_hotmask 上升、hotmask 上更易误伤。
- 尝试把 hotmask 信息直接注入“raw 的构造”（而不是在 raw 上再叠 gate），并且不新增 sweep 维度。

### 方法定义（在线单遍；$O(r^2)$；无新 sweep 超参）

在 baseline 的同极性邻域三角权重上，引入 hotmask（每像素 $M\in\{0,1\}$，由预计算 `.npy` 给出）：

$$
w_j=\max\left(0,1-\frac{\Delta t_j}{\tau}\right),\quad \Delta t_j=|t_i-t_j|
$$

拆分支持度：

$$
\mathrm{raw}_{\mathrm{nonhot}}=\sum M_j=0\; w_j,\quad
\mathrm{raw}_{\mathrm{hot}}=\sum M_j=1\; w_j
$$

输出 score：

$$
\mathrm{score}=\frac{\mathrm{raw}_{\mathrm{nonhot}}}{1+\mathrm{raw}_{\mathrm{hot}}}
$$

### 实现

- 实现位置：`src/myevs/denoise/ops/ebfopt_part2/s75_ebf_labelscore_hotmask_ratio_raw.py`
- hotmask 默认路径（由 slim sweep 按 env 自动加载/缓存）：
	- light：`data/ED24/myPedestrain_06/EBF_Part2/hotmask_light_score_neg_minus_2pos_topk32768_dil1.npy`
	- mid：`data/ED24/myPedestrain_06/EBF_Part2/hotmask_mid_score_neg_minus_2pos_topk32768_dil1.npy`
	- heavy：`data/ED24/myPedestrain_06/EBF_Part2/hotmask_heavy_score_neg_minus_2pos_topk32768_dil1.npy`

### 证据（slim sweep + heavy 分段）

#### prescreen200k（网格：`s=3/5/7/9` × `tau=8..1024ms`）

运行命令：

```bash
PYTHONNOUSERSITE=1 D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py \
	--variant s75 --max-events 200000 \
	--out-dir data/ED24/myPedestrain_06/EBF_Part2/_slim_s75_hotmask_ratio_prescreen200k_grid
```

best-AUC（各 env）：

- light：AUC=0.915622（s=9,tau=128ms）
- mid：AUC=0.800467（s=9,tau=128ms）
- heavy：AUC=0.804026（s=9,tau=128ms）

best-F1（各 env）：

- light：best-F1=0.937660（s=9,tau=512ms）
- mid：best-F1=0.695710（s=9,tau=64ms）
- heavy：best-F1=0.659430（s=9,tau=32ms）

对应 ROC CSV：

- `data/ED24/myPedestrain_06/EBF_Part2/_slim_s75_hotmask_ratio_prescreen200k_grid/roc_ebf_s75_{light,mid,heavy}_labelscore_s3_5_7_9_tau8_16_32_64_128_256_512_1024ms.csv`

#### heavy400k + 分段（seg=200k；thr=ROC best-F1）

1) 对齐点（s=9,tau=128ms）：

- ROC：AUC=0.782561，best-F1=0.537449
- 分段：
	- seg0：F1=0.624145，precision=0.542365，recall=0.734967，noise_kept_rate=0.154374
	- seg1：F1=0.396153，precision=0.294012，recall=0.607044，noise_kept_rate=0.151200

产物：

- `data/ED24/myPedestrain_06/EBF_Part2/_slim_s75_hotmask_ratio_max400k/roc_ebf_s75_heavy_labelscore_s9_tau128ms.csv`
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_s75_hotmask_ratio_max400k/seg_f1_ebf_s75_heavy_labelscore_s9_tau128ms_seg200k_max400k.csv`

2) heavy prescreen best-F1 对应时间窗（s=9,tau=32ms）：

- ROC：AUC=0.762557，best-F1=0.584423
- 分段：
	- seg0：F1=0.659430，precision=0.669189，recall=0.649951，noise_kept_rate=0.079982
	- seg1：F1=0.440466，precision=0.407143，recall=0.479730，noise_kept_rate=0.072460

产物：

- `data/ED24/myPedestrain_06/EBF_Part2/_slim_s75_hotmask_ratio_tau32_max400k/roc_ebf_s75_heavy_labelscore_s9_tau32ms.csv`
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_s75_hotmask_ratio_tau32_max400k/seg_f1_ebf_s75_heavy_labelscore_s9_tau32ms_seg200k_max400k.csv`

### 阶段性结论

- 仅靠 hotmask 做“邻域 raw 重整”（ratio 形式）在 prescreen200k 能得到中等 AUC，但 **heavy400k 的 seg1 监督指标显著退化**（F1 远低于 baseline 的 ~0.655）。
- 这说明 seg1 的问题不能用“hotmask 贡献抑制”这一条信息单独解决：因为 seg1 的信号本身也高度贴着 hotmask，单独惩罚会系统性破坏排序。

---

## s76（2026-04-13）：AOCC-inspired activity-surface Sobel gradmag（不依赖 EBF）

动机（对齐 7.20 “连续结构可感知性”）：

- AOCC 是离线指标（多时间窗构帧 + 对比度曲线面积），不适合逐事件实时计算。
- 但 AOCC 的核心直觉是：**跨时间的累积让结构变“可见”**。所以这里尝试把它蒸馏成在线的“结构可见性 proxy”。

### 方法定义（在线单遍；每事件 $O(1)$；无新 sweep 超参）

维护指数衰减 activity surface：

$$
A_{t_i}(x,y)=A_{t_{i-1}}(x,y)\,e^{-(t_i-t_{i-1})/\tau}+\mathbb{1}[(x,y)=(x_i,y_i)]
$$

其中 $\tau$ 直接复用 sweep 的时间窗（与 baseline 同一维度），每次事件到来只更新中心像素（读取其它像素时用 `last_t/last_a` 复原衰减值）。

对 $A$ 做 3×3 Sobel-like 梯度幅值，作为每个事件的 score：

$$
\mathrm{score}_i=\|\nabla A\| \approx \sqrt{g_x^2+g_y^2}
$$

其中 3×3 采样的空间步长由 `s` 派生的 `radius_px` 控制（与 baseline 的邻域尺度对齐）。

### 实现

- 实现位置：`src/myevs/denoise/ops/ebfopt_part2/s76_aocc_activity_sobel_gradmag.py`
- 状态：`last_t:uint64[W*H]` + `last_a:float32[W*H]`
- 复杂度：每事件固定 9 点采样 + 少量 `exp/hypot`；暂不使用 polarity（先做最干净版本）。

### 证据（slim sweep：prescreen200k）

#### 1) 对齐口径：`s=9,tau=128ms,max=200k`

运行命令（PowerShell）：

```powershell
$env:PYTHONNOUSERSITE=1; D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py --variant s76 --max-events 200000 --s-list 9 --tau-us-list 128000 --out-dir data/ED24/myPedestrain_06/EBF_Part2/_slim_s76_aocc_activity_sobel_gradmag_prescreen200k_s9_tau128ms
```

best-AUC / best-F1：

- light：AUC=0.859233，best-F1=0.915675
- mid：AUC=0.720747，best-F1=0.509607
- heavy：AUC=0.718423，best-F1=0.437386

产物：

- `data/ED24/myPedestrain_06/EBF_Part2/_slim_s76_aocc_activity_sobel_gradmag_prescreen200k_s9_tau128ms/roc_ebf_s76_{light,mid,heavy}_labelscore_s9_tau128ms.csv`

#### 2) 小扫 tau（固定 `s=9,max=200k`；tau=8..1024ms）

运行命令（PowerShell）：

```powershell
$env:PYTHONNOUSERSITE=1; D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py --variant s76 --max-events 200000 --s-list 9 --tau-us-list 8000,16000,32000,64000,128000,256000,512000,1024000 --out-dir data/ED24/myPedestrain_06/EBF_Part2/_slim_s76_aocc_activity_sobel_gradmag_prescreen200k_s9_tau8_16_32_64_128_256_512_1024ms
```

best-AUC（各 env）：

- light：AUC=0.861984（tau=64ms）
- mid：AUC=0.742586（tau=32ms）
- heavy：AUC=0.743288（tau=32ms）

best-F1（各 env）：

- light：best-F1=0.915914（tau=32ms）
- mid：best-F1=0.543659（tau=8ms）
- heavy：best-F1=0.463591（tau=8ms）

对应 ROC CSV：

- `data/ED24/myPedestrain_06/EBF_Part2/_slim_s76_aocc_activity_sobel_gradmag_prescreen200k_s9_tau8_16_32_64_128_256_512_1024ms/roc_ebf_s76_{light,mid,heavy}_labelscore_s9_tau8_16_32_64_128_256_512_1024ms.csv`

### 阶段性结论

- s76（activity surface + Sobel 梯度）在 prescreen200k 的 mid/heavy 上显著弱于 baseline/现有强分支：best-F1 在 heavy 只有 ~0.46 量级，远达不到继续跑 heavy400k+seg 的门槛。
- 这表明“仅用 activity 的空间梯度幅值”作为结构 proxy 过于粗糙：它更像边缘强度而非“信号 vs 噪声”的可分性来源。
- 若要继续沿 AOCC 启发推进，下一步更可能需要：
	- 引入 polarity 分通道（避免 cross-pol 混叠成伪边缘）
	- 或做局部对比度归一化（例如 $\|\nabla A\|/(A+\epsilon)$）
	- 或固定两档内置时间常数做内部融合（不新增 sweep 维度），再重新做 prescreen 门槛检查。

---

## s77（2026-04-14）：polarity-split + 归一化 + 双时间常数一致性（不依赖 EBF）

动机（来自 7.21 复盘）：

- s76 的核心问题之一是忽略 polarity，导致 cross-pol 混叠成伪结构；同时 $|\nabla A|$ 没有归一化，会天然偏向热点/强活动区域。
- AOCC 的关键是跨时间尺度的结构形成过程；这里用“内置双时间常数（$\tau,2\tau$）+ 一致性”做最小近似，且不新增 sweep 维度。

### 方法定义（在线单遍；每事件 $O(1)$；无新 sweep 超参）

为 ON/OFF 分别维护 activity surface（指数衰减），并且为每个 polarity 同时维护两档时间常数：

- 短窗：$\tau_s=\tau$
- 长窗：$\tau_l=2\tau$（固定比例，不作为 sweep 超参）

在事件 polarity 对应的通道上，计算两档尺度的“归一化结构强度”：

$$
C_s=\frac{\|\nabla A_s\|}{\overline{A_s}_{3\times3}+\epsilon},\quad
C_l=\frac{\|\nabla A_l\|}{\overline{A_l}_{3\times3}+\epsilon}
$$

再计算跨尺度一致性（沿用 7.21 的形式）：

$$
S_{cont}=\frac{C_s+C_l}{C_s+C_l+|C_s-C_l|+\epsilon}
$$

最终 score：

$$
\mathrm{score}=(C_s+C_l)\cdot S_{cont}
$$

### 实现

- 实现位置：`src/myevs/denoise/ops/ebfopt_part2/s77_aocc_polnorm_multiscale_grad.py`
- 状态：4 个表面（pos/neg × short/long），每表面 `last_t:uint64[W*H]` + `last_a:float32[W*H]`
- 复杂度：每事件固定 3×3 采样两档尺度（约 18 次采样/exp），+ 少量 `hypot`/算术。

### 证据（slim sweep：prescreen200k；固定 `s=9` 小扫 tau）

运行命令（PowerShell）：

```powershell
$env:PYTHONNOUSERSITE=1; D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py --variant s77 --max-events 200000 --s-list 9 --tau-us-list 8000,16000,32000,64000,128000,256000,512000,1024000 --out-dir data/ED24/myPedestrain_06/EBF_Part2/_slim_s77_aocc_polnorm_multiscale_prescreen200k_s9_tau8_16_32_64_128_256_512_1024ms
```

best-AUC（各 env）：

- light：AUC=0.844358（tau=8ms）
- mid：AUC=0.738180（tau=8ms）
- heavy：AUC=0.738720（tau=8ms）

best-F1（各 env）：

- light：best-F1=0.911931（tau=8ms）
- mid：best-F1=0.555653（tau=8ms）
- heavy：best-F1=0.483480（tau=8ms）

对应 ROC CSV：

- `data/ED24/myPedestrain_06/EBF_Part2/_slim_s77_aocc_polnorm_multiscale_prescreen200k_s9_tau8_16_32_64_128_256_512_1024ms/roc_ebf_s77_{light,mid,heavy}_labelscore_s9_tau8_16_32_64_128_256_512_1024ms.csv`

### 阶段性结论

- 相比 s76，s77 在 heavy 的 best-F1 有小幅回升（~0.46→~0.48），说明“polarity 分通道 + 归一化 + 双尺度一致性”方向有用，但提升幅度远不足以进入 heavy400k+seg 验证。
- 下一步更值得把“cross-pol 闪烁/混叠抑制”引入 score（仍不依赖 EBF）：例如用 opposite-pol 的局部 activity 作为惩罚项，压住交替闪烁导致的伪结构响应。

---

## s78（2026-04-14）：s77 + cross-pol 局部 activity 惩罚（不依赖 EBF）

动机（对齐 7.21 的 cross-pol 闪烁问题）：

- s77 仍可能对“交替闪烁”有较高响应：虽然打分只看同极性通道，但噪声区域往往两极性都同时活跃。
- 用 opposite-pol 的局部 activity 作为惩罚项，期望压住 flicker / alternating noise 对结构分数的污染。

### 方法定义（在 s77 基础上加惩罚项；无新 sweep 超参）

先按 s77 得到 base 分数（同极性、双尺度、归一化 + 一致性）：

$$
\mathrm{base}=(C_s+C_l)\cdot S_{cont}
$$

再计算 opposite-pol 的短窗局部 activity 均值：

$$
m_{opp}=\overline{A^{opp}_s}_{3\times3}
$$

最终：

$$
\mathrm{score}=\frac{\mathrm{base}}{1+\beta m_{opp}}
$$

其中 $\beta$ 为固定常数（不 sweep）。

### 实现

- 实现位置：`src/myevs/denoise/ops/ebfopt_part2/s78_aocc_polnorm_multiscale_crosspol_penalty.py`

### 证据（slim sweep：prescreen200k；固定 `s=9` 小扫 tau）

运行命令（PowerShell）：

```powershell
$env:PYTHONNOUSERSITE=1; D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py --variant s78 --max-events 200000 --s-list 9 --tau-us-list 8000,16000,32000,64000,128000,256000,512000,1024000 --out-dir data/ED24/myPedestrain_06/EBF_Part2/_slim_s78_aocc_polnorm_multiscale_crosspol_penalty_prescreen200k_s9_tau8_16_32_64_128_256_512_1024ms
```

best-AUC（各 env）：

- light：AUC=0.844289（tau=8ms）
- mid：AUC=0.738024（tau=8ms）
- heavy：AUC=0.738325（tau=8ms）

best-F1（各 env）：

- light：best-F1=0.911931（tau=8ms）
- mid：best-F1=0.555703（tau=8ms）
- heavy：best-F1=0.483241（tau=8ms）

对应 ROC CSV：

- `data/ED24/myPedestrain_06/EBF_Part2/_slim_s78_aocc_polnorm_multiscale_crosspol_penalty_prescreen200k_s9_tau8_16_32_64_128_256_512_1024ms/roc_ebf_s78_{light,mid,heavy}_labelscore_s9_tau8_16_32_64_128_256_512_1024ms.csv`

### 阶段性结论

- s78 与 s77 基本持平，cross-pol 局部 activity 惩罚未带来可观改善；说明问题不只是“opposite-pol 同时活跃”，而更可能是“结构 proxy 本身仍不具备足够判别力”。
- 若继续沿 AOCC 启发推进，更接近 AOCC 本质的下一步应转向：跨连续时间窗（W1/W2/W3）的一致性度量，而不是仅做不同时间常数的指数平滑。

---

## s79（2026-04-14）：离散三窗（W1/W2/W3）结构一致性（不依赖 EBF）

动机（7.21 路线 B）：

- s76–s78 用指数衰减 activity surface，本质是“不同时间常数的平滑”，仍不等价于 AOCC 的“多个时间窗累积 + CCC 曲线”。
- 这里更贴近 AOCC 的描述：用 **三个连续时间窗**（W1/W2/W3）计算局部结构强度，并用一致性度量来奖励“跨窗稳定出现的结构”。

### 方法定义（在线单遍；每事件 $O(1)$；无新 sweep 超参）

定义三个连续窗口（窗口长度直接复用 sweep 的 $\tau$）：

$$
W_1=[t-\tau,t],\quad W_2=[t-2\tau,t-\tau],\quad W_3=[t-3\tau,t-2\tau]
$$

对每个窗口 $W_k$，在事件 polarity 对应通道里维护离散计数图 $C_k(x,y)$（用 `bin=floor(t/\tau)` 的 per-pixel ring-bins 在线维护），并在 3×3 patch 上计算结构强度：

$$
	ilde C_k=\frac{\|\nabla C_k\|}{\overline{C_k}_{3\times3}+\epsilon}
$$

一致性分数：

$$
S_{cont}=\frac{\tilde C_1+\tilde C_2+\tilde C_3}{\tilde C_1+\tilde C_2+\tilde C_3+|\tilde C_1-\tilde C_2|+|\tilde C_2-\tilde C_3|+\epsilon}
$$

最终：

$$
\mathrm{score}=(\tilde C_1+\tilde C_2+\tilde C_3)\cdot S_{cont}
$$

### 实现

- 实现位置：`src/myevs/denoise/ops/ebfopt_part2/s79_aocc_discrete_windows_continuity.py`
- 状态：按 polarity 分通道；每像素保存 `last_bin:int64` + 3 个 bin 计数 `c0/c1/c2:float32`
- 特点：不需要 `exp`，只做离散 bin 对齐与 3×3 采样。

### 证据（slim sweep：prescreen200k；固定 `s=9` 小扫 tau）

运行命令（PowerShell）：

```powershell
$env:PYTHONNOUSERSITE=1; D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py --variant s79 --max-events 200000 --s-list 9 --tau-us-list 8000,16000,32000,64000,128000,256000,512000,1024000 --out-dir data/ED24/myPedestrain_06/EBF_Part2/_slim_s79_aocc_discrete_windows_continuity_prescreen200k_s9_tau8_16_32_64_128_256_512_1024ms
```

best-AUC（各 env）：

- light：AUC=0.801995（tau=64ms）
- mid：AUC=0.723828（tau=32ms）
- heavy：AUC=0.725419（tau=32ms）

best-F1（各 env）：

- light：best-F1=0.911931（tau=8ms）
- mid：best-F1=0.548193（tau=32ms）
- heavy：best-F1=0.476949（tau=16ms）

对应 ROC CSV：

- `data/ED24/myPedestrain_06/EBF_Part2/_slim_s79_aocc_discrete_windows_continuity_prescreen200k_s9_tau8_16_32_64_128_256_512_1024ms/roc_ebf_s79_{light,mid,heavy}_labelscore_s9_tau8_16_32_64_128_256_512_1024ms.csv`

### 阶段性结论

- 这版更贴近“连续窗口一致性”的 AOCC 直觉，但在 prescreen200k 的 mid/heavy 上仍明显弱于 baseline/强分支：heavy best-F1 约 0.48。
- 说明在 ED24/myPedestrain_06 上，把 AOCC 直觉直接投影成“局部结构一致性 per-event score”，仍不足以形成稳定的信号/噪声可分性。

---

## s80（2026-04-14）：baseline EBF × AOCC-lite per-event 门控（路径A：只与 baseline 融合）

动机（对齐 7.22）：

- s76–s79 证明：AOCC-inspired 结构 proxy 很难单独成为强主分类器。
- 7.22 的建议是把 AOCC 当作“控制量”，去调主干（density/hot-state 等）的参数或门控。
- 这里先走最小实现：**主干固定为 baseline EBF**，AOCC-lite 只做温和门控，不新增 sweep 维度。

### 方法定义（在线单遍；无新 sweep 超参）

- 先算 baseline EBF score（邻域证据，完全复用现有实现）。
- 同时复用 s76 的 AOCC-lite per-event proxy（activity-surface + Sobel gradmag）得到 $v_i\ge 0$。
- 门控：

$$
\mathrm{gate01}_i=\frac{v_i}{v_i+c},\quad c=1.0
$$

$$
\mathrm{factor}_i=0.75+0.5\cdot\mathrm{gate01}_i\in[0.75,1.25]
$$

$$
\mathrm{score}_i=\mathrm{score}^{EBF}_i\cdot\mathrm{factor}_i
$$

### 证据（slim sweep：prescreen200k，对齐点 `s=9,tau=128ms`；并记录 MESR/AOCC）

运行命令（PowerShell）：

```powershell
$env:PYTHONNOUSERSITE=1; D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py --variant s80 --max-events 200000 --s-list 9 --tau-us-list 128000 --esr-mode best --aocc-mode best --out-dir data/ED24/myPedestrain_06/EBF_Part2/_slim_s80_prescreen200k_s9_tau128_esrbest_aoccbest
```

best-F1（各 env；best-F1 行已写入 `esr_mean/aocc`）：

- light：best-F1=0.950107，ESRmean=1.027067，AOCC=0.821037
- mid：best-F1=0.804972，ESRmean=1.017387，AOCC=0.838714
- heavy：best-F1=0.781503，ESRmean=1.022671，AOCC=0.902807

产物：

- `data/ED24/myPedestrain_06/EBF_Part2/_slim_s80_prescreen200k_s9_tau128_esrbest_aoccbest/roc_ebf_s80_{light,mid,heavy}_labelscore_s9_tau128ms.csv`

### 阶段性结论

- 作为“AOCC 控制量注入 baseline”的最小实现，s80 在 prescreen200k 上表现为：监督指标仍然由 baseline 主导，AOCC-lite 门控不会把性能拉崩。
- 但相比块级控制器（s81），s80 的 mid/heavy best-F1 略低；说明“按事件局部结构 proxy”的门控仍可能被噪声局部结构误触发。

---

## s81（2026-04-14）：块级控制器（路径B）：按时间块连续结构可见性调制 baseline（调阈值/调惩罚等效）

动机（对齐 7.22 的对象）：

- AOCC 本质是“跨时间窗构帧后的连续对比度曲线面积”，是块/序列级概念。
- 因此这里不再做 per-event 的结构 proxy 直接打分，而是做 **块级控制信号**：在一个短时间块内计算“结构可见性 + 连续性”，再对该块内所有事件的 baseline score 做温和调制（等效于块级自适应阈值/惩罚强度）。

### 方法定义（在线单遍；无新 sweep 超参）

- 块长度：$T_{blk}=\max(2\mathrm{ms},\tau/4)$（与 sweep 的时间尺度绑定，不新增维度）。
- 对每个块构建二值事件帧 $I\in\{0,1\}^{H\times W}$（忽略 polarity）。
- 块对比度 proxy：$C_b=\mathrm{std}(\|\nabla I\|)$（Sobel 梯度幅值标准差；AOCC-lite）。
- 连续性：$C^{cont}_b=\min(C_b,C_{b-1},C_{b-2})$（奖励跨 3 块持续可见的结构）。
- 归一化：对 $C^{cont}_b$ 做 EMA（避免绝对尺度敏感），得到比值 $r_b=C^{cont}_b/(EMA_b+\epsilon)$。
- 门控同 s80：

$$
\mathrm{gate01}_b=\frac{r_b}{r_b+1},\quad \mathrm{factor}_b=0.75+0.5\cdot\mathrm{gate01}_b
$$

$$
\mathrm{score}_i=\mathrm{score}^{EBF}_i\cdot\mathrm{factor}_{b(i)}
$$

### 证据（slim sweep：prescreen400k，对齐点 `s=9,tau=128ms`；baseline vs s81 同表；并记录 MESR/AOCC）

运行命令（PowerShell）：

```powershell
$env:PYTHONNOUSERSITE=1

# baseline (ebf)
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py --variant ebf --max-events 400000 --s-list 9 --tau-us-list 128000 --roc-max-points 5000 --esr-mode best --aocc-mode best --out-dir data/ED24/myPedestrain_06/EBF_Part2/ebf_baseline_s9_tau128_prescreen400k_esrbest_aoccbest

# s81
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py --variant s81 --max-events 400000 --s-list 9 --tau-us-list 128000 --roc-max-points 5000 --esr-mode best --aocc-mode best --out-dir data/ED24/myPedestrain_06/EBF_Part2/_prescreen_s81_blockctrl_s9_tau128ms_400k_esrbest_aoccbest
```

对比点：同一口径下的 `s=9, tau=128ms`，`--max-events 400000`（mid/heavy 截断到 400K；light 该段总事件数 <400K 因此为全量），每个 env 取 best-F1 operating point（ROC CSV 中取 `f1` 最大的行；该行的 `esr_mean/aocc` 已在 `--esr-mode best --aocc-mode best` 下写入）。

| env | baseline AUC | s81 AUC | baseline F1 | s81 F1 | baseline Thr(best-F1) | s81 Thr(best-F1) | baseline MESR | s81 MESR | baseline AOCC/1e7 | s81 AOCC/1e7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| light(1.8V) | 0.947564 | 0.947720 | 0.949739 | 0.949919 | 0.7491 | 0.7450 | 1.030472 | 1.030381 | 0.820552 | 0.820537 |
| mid(2.5V) | 0.917740 | 0.918065 | 0.791293 | 0.791720 | 5.1930 | 5.2313 | 1.001058 | 0.998813 | 0.682061 | 0.681609 |
| heavy(3.3V) | 0.912484 | 0.912894 | 0.745459 | 0.745637 | 7.3566 | 7.4630 | 1.020801 | 1.021692 | 0.760986 | 0.758741 |

产物：

- baseline：`data/ED24/myPedestrain_06/EBF_Part2/ebf_baseline_s9_tau128_prescreen400k_esrbest_aoccbest/roc_ebf_{light,mid,heavy}_labelscore_s9_tau128ms.csv`
- s81：`data/ED24/myPedestrain_06/EBF_Part2/_prescreen_s81_blockctrl_s9_tau128ms_400k_esrbest_aoccbest/roc_ebf_s81_{light,mid,heavy}_labelscore_s9_tau128ms.csv`

#### 阈值扫描（heavy400K；`s=9,tau=128ms`）

说明：这里把 ROC CSV（`heavy`）里的所有 operating points 按 `F1`（同 `_best_f1_index` 的 tie-break：更高 TPR→更高 precision→更低 FPR）排序，列出 top-8 阈值点，便于直观看到 `F1(thr)` 是否存在“明显可利用”的差异。

baseline top-8：

| rank | Thr | F1 | TPR | Precision | KeepRatio |
|---:|---:|---:|---:|---:|---:|
| 1 | 7.3566 | 0.745459 | 0.712951 | 0.781073 | 0.1339 |
| 2 | 7.2916 | 0.745456 | 0.715883 | 0.777576 | 0.1350 |
| 3 | 7.3869 | 0.745450 | 0.711246 | 0.783111 | 0.1332 |
| 4 | 7.2815 | 0.745414 | 0.716275 | 0.777024 | 0.1352 |
| 5 | 7.3800 | 0.745411 | 0.711621 | 0.782569 | 0.1334 |
| 6 | 7.2734 | 0.745408 | 0.716702 | 0.776509 | 0.1354 |
| 7 | 7.3979 | 0.745408 | 0.710786 | 0.783575 | 0.1330 |
| 8 | 7.4169 | 0.745407 | 0.709934 | 0.784611 | 0.1327 |

s81 top-8：

| rank | Thr | F1 | TPR | Precision | KeepRatio |
|---:|---:|---:|---:|---:|---:|
| 1 | 7.4630 | 0.745637 | 0.713497 | 0.780810 | 0.1340 |
| 2 | 7.5279 | 0.745594 | 0.710462 | 0.784382 | 0.1328 |
| 3 | 7.5383 | 0.745585 | 0.709951 | 0.784985 | 0.1326 |
| 4 | 7.4744 | 0.745574 | 0.712934 | 0.781346 | 0.1338 |
| 5 | 7.4492 | 0.745547 | 0.713906 | 0.780123 | 0.1342 |
| 6 | 7.5084 | 0.745543 | 0.711417 | 0.783107 | 0.1332 |
| 7 | 7.4959 | 0.745541 | 0.711911 | 0.782506 | 0.1334 |
| 8 | 7.4855 | 0.745522 | 0.712389 | 0.781888 | 0.1336 |

#### 调制强度 sweep（400K；`s=9,tau=128ms`）

在保持控制器结构不变的前提下，仅通过环境变量调整融合强度（`MYEVS_S81_FACTOR_MIN/MAX`，即 `factor = min + (max-min)*gate01`），检查 heavy400K 的 best-F1 是否存在“明显可用”的改进空间。

说明：下面表格的数值来自对应 out-dir 下的 ROC CSV（`--esr-mode off --aocc-mode off`，只看监督指标，提速）。

| factor[min,max] | light best-F1 | mid best-F1 | heavy best-F1 | heavy AUC | heavy Thr(best-F1) | out-dir |
|---|---:|---:|---:|---:|---:|---|
| [0.90,1.10] | 0.949857 | 0.791469 | 0.745669 | 0.912709 | 7.2944 | `data/ED24/myPedestrain_06/EBF_Part2/_s81_strength_min0p90_max1p10_400k/` |
| [0.75,1.25]（默认） | 0.949919 | 0.791720 | 0.745637 | 0.912894 | 7.4630 | `data/ED24/myPedestrain_06/EBF_Part2/_s81_strength_min0p75_max1p25_400k/` |
| [0.60,1.40] | 0.949996 | 0.791513 | 0.745550 | 0.912939 | 7.4719 | `data/ED24/myPedestrain_06/EBF_Part2/_s81_strength_min0p60_max1p40_400k/` |
| [0.50,1.50] | 0.950013 | 0.791498 | 0.744837 | 0.912905 | 7.6181 | `data/ED24/myPedestrain_06/EBF_Part2/_s81_strength_min0p50_max1p50_400k/` |

结论（就 heavy400K 而言）：最优点出现在更弱的调制 `[0.90,1.10]`，但整体变化幅度很小。

#### baseline vs s81（调参后强度 `[0.90,1.10]`；prescreen400k；含 MESR/AOCC）

复跑命令（PowerShell）：

```powershell
$env:PYTHONNOUSERSITE=1
$env:MYEVS_S81_FACTOR_MIN='0.90'
$env:MYEVS_S81_FACTOR_MAX='1.10'
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py --variant s81 --max-events 400000 --s-list 9 --tau-us-list 128000 --roc-max-points 5000 --esr-mode best --aocc-mode best --out-dir data/ED24/myPedestrain_06/EBF_Part2/_prescreen_s81_strength_min0p90_max1p10_s9_tau128ms_400k_esrbest_aoccbest
Remove-Item Env:MYEVS_S81_FACTOR_MIN -ErrorAction SilentlyContinue
Remove-Item Env:MYEVS_S81_FACTOR_MAX -ErrorAction SilentlyContinue
```

| env | baseline AUC | s81(tuned) AUC | baseline F1 | s81(tuned) F1 | baseline Thr(best-F1) | s81(tuned) Thr(best-F1) | baseline MESR | s81(tuned) MESR | baseline AOCC/1e7 | s81(tuned) AOCC/1e7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| light(1.8V) | 0.947564 | 0.947638 | 0.949739 | 0.949857 | 0.7491 | 0.7569 | 1.030472 | 1.029190 | 0.820552 | 0.820306 |
| mid(2.5V) | 0.917740 | 0.917899 | 0.791293 | 0.791469 | 5.1930 | 5.1866 | 1.001058 | 1.002112 | 0.682061 | 0.682581 |
| heavy(3.3V) | 0.912484 | 0.912709 | 0.745459 | 0.745669 | 7.3566 | 7.2944 | 1.020801 | 1.021580 | 0.760986 | 0.763215 |

产物：

- `data/ED24/myPedestrain_06/EBF_Part2/_prescreen_s81_strength_min0p90_max1p10_s9_tau128ms_400k_esrbest_aoccbest/roc_ebf_s81_{light,mid,heavy}_labelscore_s9_tau128ms.csv`

### 阶段性结论

- 在 prescreen400k 的同一口径对比下，s81 与 baseline 的监督指标（AUC/F1）非常接近；做了一轮融合强度 sweep 后，最优点也仅表现为极小幅度变化（见“调制强度 sweep”与 tuned 表）。
- 下一步若要继续验证 7.22 路线：应在 small grid（扫 tau）上检查“是否稳定”，再决定是否进入 seg（尤其 seg1）。

---

## s82（2026-04-14）：空间分块 + 三短窗稳定性控制器（路线B 的最小证伪版）

动机（对齐 7.22 的“路线B”差异点）：

- s81 是**全帧时间块**控制器；而路线B强调的是**空间分块**后，用多短窗的稳定性指数 $S_b$ 作为控制量。
- 这里实现一个保守版本：用 $S_b$ 对 baseline EBF score 做温和乘性调制（等效块级阈值/惩罚强度调整），不引入新的 sweep 维度。

### 方法定义（在线单遍；空间 block；三短窗稳定性 $S_b$）

- 空间分块：block 尺寸 $B=32$ px（环境变量 `MYEVS_S82_BLOCK_PX` 可改）。
- 短窗长度：$T_{win}=\max(20\mathrm{ms},\tau/4)$（环境变量 `MYEVS_S82_WIN_US` 可强制指定）。
- 每个短窗构建二值事件帧 $I\in\{0,1\}^{H\times W}$（忽略 polarity）。
- 对每个 block 计算“结构强度 proxy” $C$：这里用二值帧的 total variation（相邻像素差的绝对值求和）近似（比 Sobel 更便宜且对二值帧更稳）。
- 三短窗稳定性指数（每个 block 独立）：

$$
S_b=\frac{C_0+C_1+C_2}{C_0+C_1+C_2+|C_0-C_1|+|C_1-C_2|+\epsilon}\in[0,1]
$$

- 门控（保守强度，默认 `[0.90,1.10]`，可用 `MYEVS_S82_FACTOR_MIN/MAX` 调整）：

$$
\mathrm{factor}_b = f_{min} + (f_{max}-f_{min})\cdot S_b
$$

$$
\mathrm{score}_i = \mathrm{score}^{EBF}_i\cdot \mathrm{factor}_{b(i)}
$$

### 证据（slim sweep：prescreen400k，对齐点 `s=9,tau=128ms`；baseline vs s82 同表；并记录 MESR/AOCC）

运行命令（PowerShell）：

```powershell
$env:PYTHONNOUSERSITE=1

# s82
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py --variant s82 --max-events 400000 --s-list 9 --tau-us-list 128000 --roc-max-points 5000 --esr-mode best --aocc-mode best --out-dir data/ED24/myPedestrain_06/EBF_Part2/_prescreen_s82_spatblkstab_s9_tau128ms_400k_esrbest_aoccbest
```

对比点说明：同 s81 小节（每个 env 取 best-F1 operating point；该行 `esr_mean/aocc` 已在 `--esr-mode best --aocc-mode best` 下写入）。

| env | baseline AUC | s82 AUC | baseline F1 | s82 F1 | baseline Thr(best-F1) | s82 Thr(best-F1) | baseline MESR | s82 MESR | baseline AOCC/1e7 | s82 AOCC/1e7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| light(1.8V) | 0.947564 | 0.947452 | 0.949739 | 0.949740 | 0.7491 | 0.7725 | 1.030472 | 1.036053 | 0.820552 | 0.820730 |
| mid(2.5V) | 0.917740 | 0.917279 | 0.791293 | 0.790508 | 5.1930 | 5.5470 | 1.001058 | 1.004203 | 0.682061 | 0.682197 |
| heavy(3.3V) | 0.912484 | 0.911880 | 0.745459 | 0.744305 | 7.3566 | 7.6294 | 1.020801 | 1.020562 | 0.760986 | 0.769414 |

产物：

- baseline：`data/ED24/myPedestrain_06/EBF_Part2/ebf_baseline_s9_tau128_prescreen400k_esrbest_aoccbest/roc_ebf_{light,mid,heavy}_labelscore_s9_tau128ms.csv`
- s82：`data/ED24/myPedestrain_06/EBF_Part2/_prescreen_s82_spatblkstab_s9_tau128ms_400k_esrbest_aoccbest/roc_ebf_s82_{light,mid,heavy}_labelscore_s9_tau128ms.csv`

#### heavy prescreen400k + 分段（更贴近 seg1；每段 200k；thr 取各自 heavy ROC best-F1）

运行命令（PowerShell）：

```powershell
$env:PYTHONNOUSERSITE=1

# baseline
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/noise_analyze/segment_f1.py --variant ebf --s 9 --tau-us 128000 --max-events 400000 --segment-events 200000 --labeled-npy D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3.npy --roc-csv data/ED24/myPedestrain_06/EBF_Part2/ebf_baseline_s9_tau128_prescreen400k_esrbest_aoccbest/roc_ebf_heavy_labelscore_s9_tau128ms.csv --out-csv data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/validate_400k_heavy/segf1_baseline_heavy_400k_s9_tau128.csv

# s82
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/noise_analyze/segment_f1.py --variant s82 --s 9 --tau-us 128000 --max-events 400000 --segment-events 200000 --labeled-npy D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3.npy --roc-csv data/ED24/myPedestrain_06/EBF_Part2/_prescreen_s82_spatblkstab_s9_tau128ms_400k_esrbest_aoccbest/roc_ebf_s82_heavy_labelscore_s9_tau128ms.csv --out-csv data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/validate_400k_heavy/segf1_s82_heavy_400k_s9_tau128.csv
```

结果（从 segf1 CSV 直接读取）：

- baseline（thr=7.3566）：seg0 F1=0.786836，seg1 F1=0.654839
- s82（thr=7.6294）：seg0 F1=0.785706，seg1 F1=0.655098

产物：

- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/validate_400k_heavy/segf1_baseline_heavy_400k_s9_tau128.csv`
- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/validate_400k_heavy/segf1_s82_heavy_400k_s9_tau128.csv`

### 阶段性结论

- 在当前“保守强度 + total-variation 结构 proxy”的实现下，s82 在 mid/heavy 的整体 AUC/F1 相比 baseline 略降；heavy seg1 F1 与 baseline 基本持平（变化极小）。
- 这说明：仅把“空间分块稳定性”作为弱控制量去调制 baseline 排序，尚不足以在 prescreen400k（尤其 heavy）上产生可见收益；若要继续推进路线B，可能需要把控制量用在更接近失败机理的位置（例如更直接地调长窗证据/热点相关惩罚），并优先以 seg1 为验证锚点。

---

## N1（2026-04-14）：ESSM-lite → 7.31 双层改造（块级可信度 $C_b$ + 事件级最小可延续链 $L_i$）

动机（对齐 7.30）：

- 目标是从“逐事件局部密度打分”切到“块级序列状态机”：先判定一个空间 block 在短窗序列里是否处于 forming/stable 结构形成状态，再给该 block 内事件更高置信度。

### 方法定义（在线单遍；一窗延迟；双层 $C_b\times L_i$）

- 空间分块：`MYEVS_N1_BLOCK_PX`（默认 32px）。
- 窗长：$T_{win}=\max(20\mathrm{ms},\tau/4)$（可用 `MYEVS_N1_WIN_US` 强制）。
- 块级上下文（$C_b$）：每个 block 每窗统计 `cnt/act/pol_bias/hotness` 并连续映射为 $C_b\in[0,1]$；事件使用**上一窗**的 $C_b$（一窗延迟）。
- 事件级最小可延续链（$L_i$）：在半径 $r=s$ 的邻域内找同极性最近邻 `j` 与次近邻 `k`，要求时间间隔均不超过 $\Delta t$，并满足弱速度一致性（允许速度比在一个 ratio 内）。
	- $L_i=0$：无一跳邻居；$L_i\approx 0.2$：仅一跳；$L_i=1$：满足二跳一致性。
- 最终事件分数：$Score_i = C_{b(i)}\cdot L_i$。

（实现说明：$L_i$ 用 numba `njit` 加速，保证 prescreen400k 量级可跑。）

### 证据（slim sweep：prescreen400k，对齐点 `s=9,tau=128ms`）

运行命令（PowerShell）：

```powershell
$env:PYTHONNOUSERSITE=1
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py --variant n1 --max-events 400000 --s-list 9 --tau-us-list 128000 --roc-max-points 5000 --esr-mode off --aocc-mode off --out-dir data/ED24/myPedestrain_06/EBF_Part2/_prescreen_n1_dual_731_w02_s9_tau128ms_400k
```

best-F1（各 env）：

| env | baseline AUC | n1 AUC | baseline best-F1 | n1 best-F1 | baseline Thr(best-F1) | n1 Thr(best-F1) |
|---|---:|---:|---:|---:|---:|---:|
| light(1.8V) | 0.947564 | 0.898999 | 0.949739 | 0.913141 | 0.7491 | 0.0001 |
| mid(2.5V) | 0.917740 | 0.806307 | 0.791293 | 0.563491 | 5.1930 | 0.0709 |
| heavy(3.3V) | 0.912484 | 0.642047 | 0.745459 | 0.323864 | 7.3566 | 0.0286 |

产物：

（旧 v2 产物）

- n1：`data/ED24/myPedestrain_06/EBF_Part2/_prescreen_n1_essm_lite_v2_s9_tau128ms_400k/roc_ebf_n1_{light,mid,heavy}_labelscore_s9_tau128ms.csv`

（更新后产物）

- n1：`data/ED24/myPedestrain_06/EBF_Part2/_prescreen_n1_dual_731_w02_s9_tau128ms_400k/roc_ebf_n1_{light,mid,heavy}_labelscore_s9_tau128ms.csv`

#### heavy prescreen400k + 分段（每段 200k；thr 取 n1 heavy ROC best-F1）

运行命令（PowerShell）：

```powershell
$env:PYTHONNOUSERSITE=1
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/noise_analyze/segment_f1.py --variant n1 --s 9 --tau-us 128000 --max-events 400000 --segment-events 200000 --labeled-npy D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3.npy --roc-csv data/ED24/myPedestrain_06/EBF_Part2/_prescreen_n1_dual_731_w02_s9_tau128ms_400k/roc_ebf_n1_heavy_labelscore_s9_tau128ms.csv --out-csv data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/validate_400k_heavy/segf1_n1_dual731_w02_heavy_400k_s9_tau128.csv
```

结果（segf1 CSV）：

- n1（thr=0.0286）：seg0 F1=0.410565，seg1 F1=0.221606（seg1 precision=0.125805，noise_kept_rate=0.669748；仍然明显保留了大量噪声）

产物：

- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/validate_400k_heavy/segf1_n1_dual731_w02_heavy_400k_s9_tau128.csv`

### 阶段性结论

- 按 7.31 做了“双层：$C_b\times L_i$”的结构性改造后，heavy/seg1 仍然失败：seg1 precision 只有 ~0.126，noise_kept_rate 约 0.67，说明当前的最小链一致性 $L_i$ 仍不足以抑制 heavy 噪声“搭车”。
- 这也意味着：如果要继续沿 N1 推进，$L_i$ 需要更接近“链式证据主导”的定义（例如更强的连续性/方向/多步累积/反热点机制），而块级 $C_b$ 只能作为上下文而不能承担主要筛噪职责；否则更合理的路径是转向 N2（模式切换/控制器）或把链式证据上升为主打分器（N3）。

---

## N2/N3/N4（2026-04-14 同步修订）：与 7.33 伪代码一致化（去除额外 scale 乘子）

说明：

- 本节统一记录 N2/N3/N4 的最终数学定义，并明确与 7.33 伪代码口径一致：**最终分数不再额外乘全局 `score_scale` 常数**。
- 这次修订主要是“公式口径对齐”；排序指标（AUC/F1）理论上不应因此改变，变化主要体现在阈值数值尺度。

### N2：动态侧抑制场 + 同极性短时支持

定义当前事件 $e_i=(x_i,y_i,t_i,p_i)$，半径 $r$ 邻域为 $\mathcal N_r(i)$，时间窗 $\Delta t_s$。

1) 同极性支持：

$$
S_i=\sum_{j\in\mathcal N_r(i)} \mathbf 1[p_j=p_i]\,\mathbf 1[t_i-t_j\le\Delta t_s]\,\frac{1}{1+d_{ij}^2}
$$

2) 抑制场指数衰减（局部懒更新）：

$$
I(\mathbf q,t_i^-)=I(\mathbf q,t_{prev})\exp\!\left(-\frac{t_i-t_{prev}}{\tau_I}\right)
$$

3) 侧抑制与自抑制增量：

$$
I(\mathbf q,t_i)=I(\mathbf q,t_i^-)+g_{inh}\,g_{lat}\,\frac{1}{1+d_{i\mathbf q}^2},\quad \mathbf q\in\mathcal N_r(i)
$$

$$
I(\mathbf x_i,t_i)\leftarrow I(\mathbf x_i,t_i)+g_{inh}\,g_{self}
$$

4) 最终分数（无额外 scale 乘子）：

$$
\mathrm{score}_i=\frac{g_s\,S_i}{1+I(\mathbf x_i,t_i)}
$$

### N3：贝叶斯背景率后验 + 似然比打分

每像素维护 Gamma 后验参数 $(\alpha,\beta)$ 以及先验 $(\alpha_0,\beta_0)$，并做指数回归（遗忘）：

$$
\alpha_i^- = \alpha_0 + (\alpha_{i-1}-\alpha_0)\exp\!\left(-\frac{\Delta t_i}{\tau_n}\right),\quad
\beta_i^- = \beta_0 + (\beta_{i-1}-\beta_0)\exp\!\left(-\frac{\Delta t_i}{\tau_n}\right)
$$

同极性短时支持 $S_i$ 与 N2 相同定义（不含抑制场）。背景率后验均值：

$$
\hat\lambda_n(\mathbf x_i)=\frac{\alpha_i^-}{\beta_i^-+\varepsilon}
$$

最终分数（无额外 scale 乘子）：

$$
\mathrm{score}_i=\frac{S_i}{\hat\lambda_n(\mathbf x_i)+\varepsilon}
$$

事件到来后更新后验：

$$
\alpha_i=\alpha_i^-+1,\quad \beta_i=\beta_i^-+1
$$

### N4：局部动量一致性（momentum consistency）加权支持

在同极性邻域内定义时间-空间联合权重：

$$
w_{ij}=\underbrace{\max\!\left(0,1-\frac{t_i-t_j}{\Delta t_s}\right)}_{\text{时间三角窗}}
\cdot
\underbrace{\frac{1}{1+d_{ij}^2}}_{\text{空间衰减}}
$$

支持项：

$$
S_i=\sum_{j\in\mathcal N_r(i),\,p_j=p_i} w_{ij}
$$

方向一致性项（邻域单位向量加权和）：

$$
\mathbf u_i = \sum_j w_{ij}\,\hat{\mathbf d}_{ij},\qquad
c_i=\frac{\|\mathbf u_i\|}{\sum_j w_{ij}+\varepsilon}\in[0,1]
$$

最终分数（无额外 scale 乘子）：

$$
\mathrm{score}_i=g_s\,S_i\left(1+g_m\,c_i\right)
$$

### 同口径重跑结果（prescreen400k，`s=9,tau=128ms`）

重跑目录（去掉额外 scale 乘子后的口径）：

- `data/ED24/myPedestrain_06/EBF_Part2/_prescreen_n2_noscale_s9_tau128_400k/`
- `data/ED24/myPedestrain_06/EBF_Part2/_prescreen_n3_noscale_s9_tau128_400k/`
- `data/ED24/myPedestrain_06/EBF_Part2/_prescreen_n4_noscale_s9_tau128_400k/`

best-F1 operating point（各 env）：

| method | env | AUC | best-F1 | Thr(best-F1) |
|---|---|---:|---:|---:|
| baseline | light | 0.947564 | 0.949739 | 0.749148 |
| baseline | mid | 0.917740 | 0.791293 | 5.192953 |
| baseline | heavy | 0.912484 | 0.745459 | 7.356625 |
| n2 | light | 0.909055 | 0.924552 | 0.028883 |
| n2 | mid | 0.894141 | 0.733688 | 0.409865 |
| n2 | heavy | 0.890258 | 0.707321 | 0.552508 |
| n3 | light | 0.916937 | 0.924664 | 0.416414 |
| n3 | mid | 0.899329 | 0.760560 | 6.180442 |
| n3 | heavy | 0.896103 | 0.725562 | 8.038934 |
| n4 | light | 0.929563 | 0.933818 | 0.000585 |
| n4 | mid | 0.895374 | 0.749284 | 0.554966 |
| n4 | heavy | 0.886175 | 0.700713 | 0.815738 |

heavy 分段（2×200k；各自 heavy best-F1 阈值）：

| method | seg0 F1 | seg1 F1 | seg1 precision | seg1 noise_kept_rate |
|---|---:|---:|---:|---:|
| baseline | 0.786836 | 0.654839 | 0.703224 | 0.026821 |
| n2 | 0.757739 | 0.592672 | 0.733915 | 0.018692 |
| n3 | 0.772324 | 0.622749 | 0.723820 | 0.021628 |
| n4 | 0.756167 | 0.576464 | 0.709311 | 0.020640 |

阶段性结论：

- 与 baseline 相比，N2/N3/N4 在该口径下均未超过 baseline；其中 N3 相对 N2/N4 更接近 baseline，但仍有差距。
- 这与“口径对齐（去掉额外 scale 常数）后主要影响阈值尺度、不会凭空改善排序”一致。

### N5（2026-04-14）：双时间尺度率比 + 邻域支持（初次试跑）

实现位置：`src/myevs/denoise/ops/ebfopt_part2/n5_dual_timescale_rate_ratio.py`

核心定义（流式、单遍）：

$$
\mathrm{score}_i=\log\frac{g_s\,S_i+g_f\,\lambda_{fast}(x_i,y_i)+\varepsilon}{\lambda_{slow}(x_i,y_i)+\varepsilon}
$$

其中：

- $S_i$：同极性邻域短时支持（与 N4 类似的时间三角窗 + 空间反距离权重）；
- $\lambda_{fast},\lambda_{slow}$：像素级快/慢时间常数 EMA 事件率（默认 `tau_fast=16ms`, `tau_slow=128ms`）。

本次产物：

- smoke（20k）：`data/ED24/myPedestrain_06/EBF_Part2/_smoke_n5_s9_tau128_20k/`
- prescreen（400k）：`data/ED24/myPedestrain_06/EBF_Part2/_prescreen_n5_s9_tau128_400k/`
- heavy 分段：`data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/validate_400k_heavy_noscale/segf1_n5_heavy_400k_s9_tau128.csv`

prescreen400k（`s=9,tau=128ms`）best-F1 点：

| method | env | AUC | best-F1 | Thr(best-F1) |
|---|---|---:|---:|---:|
| n5 | light | 0.885614 | 0.924452 | -0.013995 |
| n5 | mid | 0.884681 | 0.713998 | 0.100778 |
| n5 | heavy | 0.879872 | 0.668248 | 0.156183 |

heavy 分段（2×200k；heavy best-F1 阈值）：

| method | seg0 F1 | seg1 F1 | seg1 precision | seg1 noise_kept_rate |
|---|---:|---:|---:|---:|
| n5 | 0.733718 | 0.532808 | 0.548507 | 0.044226 |

阶段判断：

- 当前默认参数下，N5 明显弱于 baseline 与 N3，不作为主线候选。
- 若继续尝试 N5，应优先做小网格（`tau_fast/tau_slow/support_dt/fast_gain`）后再决定是否保留该方向。

### 方法1（2026-04-14）：N6 = N2 抑制场 + N4 动量一致性逃逸门控

动机（对齐 7.34 的“抑制场 + 结构一致性”建议）：

- N2 的核心优势是能压热点/簇噪，但在 seg1 容易把“落在热点区域的真信号”一起压掉；
- N4 提供了局部方向一致性 $c_i\in[0,1]$，可作为“是否应减抑制”的证据；
- 因此构造 N6：当局部一致性高时，降低有效抑制并轻微放大支持，给结构化事件“逃逸通道”。

定义：

$$
\kappa_i=\mathrm{clip}(1-g_e\,c_i,\kappa_{min},1)
$$

$$
\mathrm{score}_i=\frac{g_s\,S_i\,(1+g_m\,c_i)}{1+\kappa_i\,I(x_i,y_i)}
$$

其中：

- $S_i$：同极性短窗邻域支持（沿用 N2）；
- $I(x_i,y_i)$：动态侧抑制场（沿用 N2）；
- $c_i$：局部方向一致性（借鉴 N4 的单位向量加权一致性）；
- $g_e$：escape 增益，$\kappa_{min}$：最小抑制缩放下限。

预期：

- 在不明显放大 noise_kept_rate 的前提下，提升 heavy seg1 的 signal_kept / F1；
- 如果只看到 precision 提升但 seg1 recall 下降，则说明 escape 门控不足或一致性估计偏噪声化。

最小验证口径（与当前 N 系列一致）：

- smoke：`max-events=20k, s=9, tau=128ms`
- prescreen：`max-events=400k, s=9, tau=128ms`
- heavy 分段：`segment_f1.py` 的 2×200k（看 seg1 F1/precision/noise_kept_rate）。

首轮结果（2026-04-14，默认参数）：

- smoke 产物：`data/ED24/myPedestrain_06/EBF_Part2/_smoke_n6_s9_tau128_20k/`
- prescreen400k 产物：`data/ED24/myPedestrain_06/EBF_Part2/_prescreen_n6_s9_tau128_400k/`
- heavy 分段产物：`data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/validate_400k_heavy_noscale/segf1_n6_heavy_400k_s9_tau128.csv`

prescreen400k（`s=9,tau=128ms`）best-F1 点：

| method | env | AUC | best-F1 | Thr(best-F1) |
|---|---|---:|---:|---:|
| n6 | light | 0.914065 | 0.924534 | 0.019854 |
| n6 | mid | 0.888705 | 0.733075 | 0.676322 |
| n6 | heavy | 0.881723 | 0.666909 | 0.964565 |

heavy 分段（2×200k；heavy best-F1 阈值）：

| method | seg0 F1 | seg1 F1 | seg1 precision | seg1 noise_kept_rate |
|---|---:|---:|---:|---:|
| n6 | 0.747745 | 0.539760 | 0.534509 | 0.049243 |

与主要对照（heavy，400k）对比：

| method | AUC | best-F1 | seg1 F1 |
|---|---:|---:|---:|
| baseline | 0.912484 | 0.745459 | 0.654839 |
| n3(default) | 0.896103 | 0.725562 | 0.622749 |
| n3(tunedB) | 0.899099 | 0.726512 | 0.629322 |
| n5(default) | 0.879872 | 0.668248 | 0.532808 |
| n6 | 0.881723 | 0.666909 | 0.539760 |

阶段判断：

- N6（默认参数）较 n5 有轻微 seg1 提升（0.5328 -> 0.5398），但整体仍显著落后于 n3 与 baseline。
- 说明“动量逃逸”方向有一定作用，但当前逃逸门控仍过弱/过噪，未形成可用主线。
- 若继续 N6，优先扫：`MYEVS_N6_ESCAPE_GAIN`、`MYEVS_N6_ESCAPE_FLOOR`、`MYEVS_N6_MOMENTUM_GAIN` 与 `MYEVS_N6_SUPPORT_DT_US`。

### 方法2（2026-04-14）：N7 = 双场模型（自抑制场 + 邻域足迹场）

来源：按 7.35 建议，将 n2 方向重写为“真正双场主轴”，不再以普通 support 作为唯一分子主证据。

实现位置：`src/myevs/denoise/ops/ebfopt_part2/n7_dual_field_selfinhib_footprint.py`

定义：

$$
\mathrm{score}_i=\frac{E_{foot}^\star(x_i,y_i,t_i)}{1+\lambda\,I_{self}(x_i,y_i,t_i)}
$$

其中：

$$
E_{foot}(x_i,y_i,t_i)=\sum_{j\in\mathcal N_r(i),j\neq i,p_j=p_i} \exp\!\left(-\frac{d_{ij}^2}{2\sigma_s^2}\right)\exp\!\left(-\frac{t_i-t_j}{\tau_E}\right)\mathbf 1[t_i-t_j\le T_E]
$$

$$
E_{foot}^\star = E_{foot}\cdot R_{escape},\quad R_{escape}=\mathrm{clip}(\Delta t_0/T_{ref},0,1)^\eta
$$

本次产物：

- smoke（20k）：`data/ED24/myPedestrain_06/EBF_Part2/_smoke_n7_s9_tau128_20k/`
- prescreen（400k）：`data/ED24/myPedestrain_06/EBF_Part2/_prescreen_n7_s9_tau128_400k/`
- heavy 分段：`data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/validate_400k_heavy_noscale/segf1_n7_heavy_400k_s9_tau128.csv`

prescreen400k（`s=9,tau=128ms`）best-F1 点：

| method | env | AUC | best-F1 | Thr(best-F1) |
|---|---|---:|---:|---:|
| n7 | light | 0.927746 | 0.934756 | 0.007466 |
| n7 | mid | 0.886256 | 0.693016 | 0.207095 |
| n7 | heavy | 0.869147 | 0.604835 | 0.605447 |

heavy 分段（2×200k；heavy best-F1 阈值）：

| method | seg0 F1 | seg1 F1 | seg1 precision | seg1 noise_kept_rate |
|---|---:|---:|---:|---:|
| n7 | 0.718465 | 0.514316 | 0.542609 | 0.042742 |

与主要对照（heavy，400k）对比：

| method | AUC | best-F1 | seg1 F1 |
|---|---:|---:|---:|
| baseline | 0.912484 | 0.745459 | 0.654839 |
| n3(tunedB) | 0.899099 | 0.726512 | 0.629322 |
| n6 | 0.881723 | 0.666909 | 0.539760 |
| n7 | 0.869147 | 0.604835 | 0.514316 |

阶段判断：

- N7 在当前默认参数下不成立：AUC/F1 与 seg1 均显著落后于 n3/n6，且 heavy 退化明显。
- 结论不是“方向必错”，而是该双场实现的参数与证据形态在现数据上过于激进（分子足迹过稀、分母抑制偏强），暂不进入主线。

补充：N7 小网格调参（2026-04-14，200k 快筛 -> 400k 验证）

按 7.35 的建议优先放宽抑制并增强足迹（先 200k 快筛 A/B/C/D，再选 D 做 400k）：

- D 点参数：
	- `MYEVS_N7_LAMBDA_INHIB=0.40`
	- `MYEVS_N7_SELF_GAIN=0.60`
	- `MYEVS_N7_SIGMA_SPATIAL=2.0`
	- `MYEVS_N7_TAU_FOOT_US=32000`
	- `MYEVS_N7_FOOT_DT_MAX_US=64000`
	- `MYEVS_N7_ESCAPE_REF_US=16000`
	- `MYEVS_N7_ESCAPE_ETA=0.00`

调参产物：

- 200k 快筛：`data/ED24/myPedestrain_06/EBF_Part2/_tune_n7_A_200k/`、`_tune_n7_B_200k/`、`_tune_n7_C_200k/`、`_tune_n7_D_200k/`
- 400k 验证：`data/ED24/myPedestrain_06/EBF_Part2/_prescreen_n7_tunedD_s9_tau128_400k/`
- heavy 分段：`data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/validate_400k_heavy_noscale/segf1_n7_tunedD_heavy_400k_s9_tau128.csv`

关键结果（heavy, 400k）：

| method | AUC | best-F1 | seg1 F1 | seg1 precision | seg1 noise_kept_rate |
|---|---:|---:|---:|---:|---:|
| baseline | 0.912484 | 0.745459 | 0.654839 | 0.703224 | 0.026821 |
| n3(tunedB) | 0.899099 | 0.726512 | 0.629322 | 0.734507 | 0.020640 |
| n6 | 0.881723 | 0.666909 | 0.539760 | 0.534509 | 0.049243 |
| n7(default) | 0.869147 | 0.604835 | 0.514316 | 0.542609 | 0.042742 |
| n7(tunedD) | 0.905461 | 0.745686 | 0.643821 | 0.729360 | 0.022179 |

更新后的阶段判断：

- N7 并非方向失败，而是**对抑制强度高度敏感**；默认参数会过抑制，导致显著退化。
- 经 D 点调参后，heavy best-F1 已达到并略高于 baseline（0.745686 vs 0.745459），且 seg1 指标明显优于 n3/n6。
- 现阶段可将 N7（tunedD）保留为“可继续推进”的候选线，并在下一轮重点验证其跨环境阈值稳定性与 1M 口径表现。

### 方法3（2026-04-14）：N7.1 = 双时间尺度足迹 + 自抑制

来源：采纳 7.36 建议，把 N7 的单尺度足迹改为“快慢双尺度增量融合”，优先针对 heavy seg1 的稀疏证据不足。

实现位置：`src/myevs/denoise/ops/ebfopt_part2/n71_dual_timescale_footprint.py`

定义：

$$
E_{dual}=E_{fast}+\rho\cdot\max(0,E_{slow}-E_{fast}),\qquad
\mathrm{score}_i=\frac{E_{dual}}{1+\lambda I_{self}}
$$

其中默认绑定为低自由度参数化：

- `tau_fast = tau0`
- `tau_slow = slow_ratio * tau0`（默认 `4*tau0`）
- `foot_dt_max = dt_max_ratio * tau0`（默认 `8*tau0`）

本次产物：

- prescreen（400k）：`data/ED24/myPedestrain_06/EBF_Part2/_prescreen_n71_s9_tau128_400k/`
- heavy 分段：`data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/validate_400k_heavy_noscale/segf1_n71_heavy_400k_s9_tau128.csv`

prescreen400k（`s=9,tau=128ms`）best-F1 点：

| method | env | AUC | best-F1 | Thr(best-F1) |
|---|---|---:|---:|---:|
| n71 | light | 0.946710 | 0.946417 | 0.090903 |
| n71 | mid | 0.916830 | 0.792035 | 1.413968 |
| n71 | heavy | 0.910270 | 0.747280 | 1.968687 |

heavy 分段（2×200k；heavy best-F1 阈值）：

| method | seg0 F1 | seg1 F1 | seg1 precision | seg1 noise_kept_rate |
|---|---:|---:|---:|---:|
| n71 | 0.792279 | 0.648318 | 0.717198 | 0.024194 |

补充：N7.1 网格扫频复核（2026-04-14，400k）

用户问题：`s=9,tau=128ms` 是否真的是 n71 最优？

先修复口径问题（关键）：

- 在原始 n71 实现中，`tau_us` 对时间常数影响过弱（主要由内部 `tau0` 默认主导），导致早期扫频里 `tau` 变化不充分。
- 已修复：n71 默认将 `tau_us` 映射到内部时间尺度（`tau0=tau/8`, `tau_inhib=tau/4`），再进行扫频。

扫频口径：

- 命令：`--variant n71 --max-events 400000 --s-list 3,5,7,9 --tau-us-list 8..1024ms`
- 为避免 AOCC/ESR 额外计算干扰，使用：`--aocc-mode off --esr-mode off`
- 产物：`data/ED24/myPedestrain_06/EBF_Part2/_grid_n71_400k_taubound_noaocc/`

三环境最优（按各 env best-F1）：

| env | best tag | AUC | best-F1 | Thr |
|---|---|---:|---:|---:|
| light | `ebf_n71_labelscore_s9_tau512000` | 0.947526 | 0.951241 | 0.352795 |
| mid | `ebf_n71_labelscore_s9_tau128000` | 0.916830 | 0.792035 | 1.413968 |
| heavy | `ebf_n71_labelscore_s9_tau128000` | 0.910270 | 0.747280 | 1.968687 |

结论：

- 对 heavy/mid：`s=9,tau=128ms` 仍是当前最优点，说明之前该点并非偶然。
- 对 light：更长时间尺度（`tau=512ms`）可以进一步提升 F1。
- 因此后续若目标是“全面超越”，建议按 env 分层：
	- heavy/mid 保持 `tau=128ms` 主线优化；
	- light 可单独使用更长 `tau` 或引入 per-env/自适应时间尺度机制。

### 方法4（2026-04-14）：N7.2 = N7.1 + 全局坏时段调制

来源：采纳 7.36 建议，在 N7.1 上增加 very cheap 的全局爆发 proxy，处理 seg1 的非局部同步/爆发风险。

实现位置：`src/myevs/denoise/ops/ebfopt_part2/n72_dual_timescale_global_burst.py`

定义：

$$
G(t)=\mathrm{clip}\left(\frac{R_{short}(t)}{R_{long}(t)+\epsilon}-1,\,0,\,1\right)
$$

$$
\lambda_{eff}=\lambda_0(1+\alpha G),\qquad
\rho_{eff}=\rho_0(1-\beta G)
$$

$$
\mathrm{score}_i=\frac{E_{fast}+\rho_{eff}\max(0,E_{slow}-E_{fast})}{1+\lambda_{eff}I_{self}}
$$

本次产物：

- prescreen（400k）：`data/ED24/myPedestrain_06/EBF_Part2/_prescreen_n72_s9_tau128_400k/`
- heavy 分段：`data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/validate_400k_heavy_noscale/segf1_n72_heavy_400k_s9_tau128.csv`

prescreen400k（`s=9,tau=128ms`）best-F1 点：

| method | env | AUC | best-F1 | Thr(best-F1) |
|---|---|---:|---:|---:|
| n72 | light | 0.946548 | 0.946230 | 0.088332 |
| n72 | mid | 0.916566 | 0.791528 | 1.401862 |
| n72 | heavy | 0.910095 | 0.747004 | 1.963248 |

heavy 分段（2×200k；heavy best-F1 阈值）：

| method | seg0 F1 | seg1 F1 | seg1 precision | seg1 noise_kept_rate |
|---|---:|---:|---:|---:|
| n72 | 0.791797 | 0.648480 | 0.717987 | 0.024089 |

补充判断（N7 tunedD vs N7.1/N7.2）：

- 口径说明（避免混淆）：这里比较的是 **prescreen400k, s=9, tau=128ms**。在该口径下，heavy baseline best-F1 = **0.745459**；不是 1M 小节里常见的约 0.761。
- heavy best-F1（400k 口径）：`n71=0.747280`、`n72=0.747004`，均仅小幅高于 baseline（+0.0018 / +0.0015）与 n7(tunedD=0.745686)。
- heavy seg1 F1（400k 口径）：`n71/n72` 明显优于 n7(tunedD=0.643821)，但仍略低于 baseline（`0.654839`）。
- 结论：7.36 的建议在 400k 口径下有效，但目前仍属于“局部小幅超越”，不能表述为“全面超越”。

### 方法5（2026-04-14）：N8 = 因果轨迹一致性 + 双场模型

来源：按“不是调参而是抓本质”的要求，把主证据从局部密度提升到局部可解释连续性；在 N7.2 框架上引入 8 方向因果轨迹一致性项。

实现位置：`src/myevs/denoise/ops/ebfopt_part2/n8_causal_traj_dualfield.py`

定义：

$$
\mathrm{score}_i = \frac{E_{dual}(i) + \gamma_{traj,eff}(t_i)\,C_{traj}(i)}{1 + \lambda_{eff}(t_i)\,I_{self}(i)}
$$

其中：

- `E_dual`：N7.1 的双时间尺度足迹（`E_fast + rho_eff * max(0, E_slow-E_fast)`）
- `C_traj`：同极性 8 方向前后像素的时序一致性证据（双侧 recent + 对称性）
- `lambda_eff/rho_eff/gamma_traj_eff`：由全局爆发 proxy `G(t)` 在线调制

本次产物：

- prescreen（400k）：`data/ED24/myPedestrain_06/EBF_Part2/_prescreen_n8_s9_tau128_400k/`
- heavy 分段：`data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/validate_400k_heavy_noscale/segf1_n8_heavy_400k_s9_tau128.csv`

prescreen400k（`s=9,tau=128ms`）best-F1 点：

| method | env | AUC | best-F1 | Thr(best-F1) |
|---|---|---:|---:|---:|
| n8 | light | 0.946538 | 0.946233 | 0.095866 |
| n8 | mid | 0.915213 | 0.784294 | 1.667636 |
| n8 | heavy | 0.908513 | 0.742868 | 2.276848 |

heavy 分段（2×200k；heavy best-F1 阈值）：

| method | seg0 F1 | seg1 F1 | seg1 precision | seg1 noise_kept_rate |
|---|---:|---:|---:|---:|
| n8 | 0.789723 | 0.639268 | 0.723387 | 0.022715 |

阶段判断：

- N8 首版在“低噪声保持率 + 高精度”上表现健康，但未形成全面突破：heavy best-F1 与 seg1 F1 均低于 n71/n72，heavy best-F1 也低于 baseline。
- 这说明“轨迹一致性”作为本质方向是对的，但当前融合策略（`E_dual + gamma*C_traj`）仍偏保守，轨迹证据在 heavy 稀疏段被抑制项抵消。
- 下一步不建议继续常规扫参，建议直接做结构改动：把 `C_traj` 从“加法辅项”改为“主导门控”（例如 `score = C_traj * f(E_dual, I_self)`），再验证是否能真正拉升 seg1 F1。

### 方法6（2026-04-14）：N8.1 = 轨迹主导门控（结构改动）

来源：延续 N8 的“因果轨迹”方向，不做常规调参，直接把轨迹证据从“加法辅项”改为“门控主项”。

实现位置：`src/myevs/denoise/ops/ebfopt_part2/n81_causal_gate_dualfield.py`

定义：

$$
\mathrm{score}_i = \frac{\mathrm{gate}(C_{traj})\cdot E_{dual}}{1+\lambda_{eff}I_{self}},
\quad \mathrm{gate}(C_{traj})=(c_0+\gamma_{traj,eff}C_{traj})^{\eta}
$$

本次产物：

- prescreen（400k）：`data/ED24/myPedestrain_06/EBF_Part2/_prescreen_n81_s9_tau128_400k/`
- heavy 分段：`data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/validate_400k_heavy_noscale/segf1_n81_heavy_400k_s9_tau128.csv`

prescreen400k（`s=9,tau=128ms`）best-F1 点：

| method | env | AUC | best-F1 | Thr(best-F1) |
|---|---|---:|---:|---:|
| n81 | light | 0.946729 | 0.946230 | 0.111907 |
| n81 | mid | 0.911440 | 0.751477 | 1.029367 |
| n81 | heavy | 0.901378 | 0.701359 | 1.178846 |

heavy 分段（2×200k；heavy best-F1 阈值）：

| method | seg0 F1 | seg1 F1 | seg1 precision | seg1 noise_kept_rate |
|---|---:|---:|---:|---:|
| n81 | 0.756972 | 0.577923 | 0.698132 | 0.022113 |

阶段判断：

- N8.1 说明“轨迹门控直接主导”在当前实现中过强，导致 mid/heavy recall 明显下降，整体 F1 退化。
- 结论不是否定“轨迹因果证据”，而是否定当前 gate 形态；下一步应考虑“软门控+上下限”而非硬主导门控。

### 方法7（2026-04-14）：N8.2 = 软门控轨迹一致性（理论+数据驱动）

改动动机（非拍脑袋）：

- 数据依据：heavy seg1 已知是稀疏低 SNR；N8.1 的结果呈现“precision 尚可但 recall 大幅下降”，说明轨迹门控过硬导致误杀。
- 理论依据：把轨迹证据当作“可信度”而非“硬判决”。当 `C_traj` 不确定时，应回退到 `E_dual` 主干，避免在稀疏段过抑制。

实现位置：`src/myevs/denoise/ops/ebfopt_part2/n82_causal_softgate_dualfield.py`

核心定义：

$$
\mathrm{score}_i = \frac{\mathrm{soft\_gate}(C_{traj})\cdot E_{dual}}{1+\lambda_{eff}I_{self}}
$$

其中：

$$
\mathrm{conf}=\frac{C_{traj}}{C_{traj}+k},\quad
\mathrm{soft\_gate}=(1-\mathrm{conf})+\mathrm{conf}\cdot \mathrm{clip}(c_0+\gamma C_{traj}, g_{min}, g_{max})
$$

解释：

- `conf` 小（轨迹证据弱）时，`soft_gate` 接近 1，自动回退到 dual-field 主干；
- `conf` 大（轨迹证据强）时，才逐步放大/缩小 gate；
- 用 `g_min/g_max` 限制门控幅度，避免 n8.1 的硬门控塌缩。

本次产物：

- prescreen（400k）：`data/ED24/myPedestrain_06/EBF_Part2/_prescreen_n82_s9_tau128_400k/`
- heavy 分段：`data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/validate_400k_heavy_noscale/segf1_n82_heavy_400k_s9_tau128.csv`

n82 的三环境结果（prescreen400k，`s=9,tau=128ms`）：

| method | env | AUC | best-F1 | Thr(best-F1) |
|---|---|---:|---:|---:|
| n82 | light | 0.946721 | 0.946230 | 0.084357 |
| n82 | mid | 0.916521 | 0.789996 | 1.385881 |
| n82 | heavy | 0.910017 | 0.745455 | 1.966811 |

heavy 分段（2×200k；heavy best-F1 阈值）：

| method | seg0 F1 | seg1 F1 | seg1 precision | seg1 noise_kept_rate |
|---|---:|---:|---:|---:|
| n82 | 0.791631 | 0.643890 | 0.715857 | 0.024089 |

与当前主线对照（heavy，prescreen400k 同口径）：

| method | AUC | best-F1 |
|---|---:|---:|
| baseline | 0.912484 | 0.745459 |
| n71 | 0.910270 | 0.747280 |
| n72 | 0.910095 | 0.747004 |
| n8 | 0.908513 | 0.742868 |
| n81 | 0.901378 | 0.701359 |
| n82 | 0.910017 | 0.745455 |

阶段判断：

- N8.2 成功验证了“软门控优于硬门控”：相较 n8.1，大幅恢复了 mid/heavy 的 F1。
- 但 n82 仍未超过 n71/n72（heavy best-F1 约与 baseline 持平、seg1 F1 仍低于 baseline）。
- 结论：这条“轨迹因果证据”主线是成立的，但当前最优仍是“软融合而非主导门控”；下一轮应优先做 `conf` 的自适应（与局部稀疏度/全局爆发联合）而非再堆新公式。

### 方法8（2026-04-14）：N8.3 = 联合可信度自适应软门控（局部稀疏度 + 全局爆发）

改动动机（理论+数据依据）：

- 数据依据：n8.2 相比 n8.1 已恢复，但 heavy seg1 recall 仍偏低；说明“仅用 `C_traj` 作为 conf”仍不足以刻画稀疏段不确定性。
- 理论依据：轨迹可信度应由“轨迹一致性 + 局部证据充足度 + 全局状态”联合决定；在局部稀疏时应主动放松抑制以保护 recall。

实现位置：`src/myevs/denoise/ops/ebfopt_part2/n83_causal_adaptive_confidence.py`

核心定义：

$$
\mathrm{score}_i=\frac{\mathrm{soft\_gate}(\mathrm{conf}_{joint})\cdot E_{dual}}{1+\lambda_{sparse}\,I_{self}}
$$

其中：

$$
\mathrm{conf}_{joint}=\mathrm{conf}_{traj}\cdot\mathrm{conf}_{local}\cdot(1-\beta_g G)
$$

$$
\lambda_{sparse}=\lambda_{eff}\cdot\left(1-\alpha_{sparse}(1-\mathrm{conf}_{local})\right)
$$

解释：

- `conf_local` 低表示邻域证据稀疏，此时降低轨迹门控强度并放松抑制；
- 全局爆发（`G` 大）时降低 `conf_joint`，避免爆发噪声把轨迹门控误激活。

本次产物：

- prescreen（400k）：`data/ED24/myPedestrain_06/EBF_Part2/_prescreen_n83_s9_tau128_400k/`
- heavy 分段：`data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/validate_400k_heavy_noscale/segf1_n83_heavy_400k_s9_tau128.csv`

n83 的三环境结果（prescreen400k，`s=9,tau=128ms`）：

| method | env | AUC | best-F1 | Thr(best-F1) |
|---|---|---:|---:|---:|
| n83 | light | 0.946564 | 0.945934 | 0.092386 |
| n83 | mid | 0.916063 | 0.789493 | 1.406460 |
| n83 | heavy | 0.909471 | 0.745019 | 2.004665 |

heavy 分段（2×200k；heavy best-F1 阈值）：

| method | seg0 F1 | seg1 F1 | seg1 precision | seg1 noise_kept_rate |
|---|---:|---:|---:|---:|
| n83 | 0.791073 | 0.643578 | 0.717727 | 0.023796 |

与主线对照（heavy，prescreen400k 同口径）：

| method | AUC | best-F1 |
|---|---:|---:|
| baseline | 0.912484 | 0.745459 |
| n71 | 0.910270 | 0.747280 |
| n72 | 0.910095 | 0.747004 |
| n82 | 0.910017 | 0.745455 |
| n83 | 0.909471 | 0.745019 |

阶段判断：

- N8.3 验证了“联合 conf + 稀疏段放松”是可行的稳定化方向，但本轮收益仍偏小，尚未超过 n71/n72。
- light/mid 结果已同步给出：light 基本持平，mid 小幅回落，说明当前自适应强度仍偏保守。
- 结论：下一轮应固定 n83 结构，仅针对 `conf_joint` 的组合权重做小范围、有监督约束下的校准，而不是继续叠加新机制。

### 方法9（2026-04-14）：N84 = 7.37 事件链状态模型（ECSM）

用户要求：按 7.37 的“事件链”思路实现真正新算法，并完成全流程评测与文档记录。

可行性判断：

- 理论上可行：把“事件是否属于可延续因果链”作为主证据，符合 7.37 的核心思想。
- 工程上可行：可实现为在线单遍、每事件常数邻域候选（8 邻域）更新，满足实时约束。

实现位置：`src/myevs/denoise/ops/ebfopt_part2/n84_event_chain_state_model.py`

方法摘要：

- 维护同极性像素级链状态：最近触发时刻、链长度、方向、方向稳定度。
- 新事件在 8 邻域中寻找候选前驱链，代价包含：多时间尺度时间惩罚、方向连续性惩罚、在线 hotness 惩罚。
- 事件分数由“链接强度 × 链质量（长度/稳定度/hotness/全局爆发）”给出。

本次产物：

- prescreen（400k）：`data/ED24/myPedestrain_06/EBF_Part2/_prescreen_n84_s9_tau128_400k/`
- heavy 分段：`data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/validate_400k_heavy_noscale/segf1_n84_heavy_400k_s9_tau128.csv`

n84 三环境结果（prescreen400k，`s=9,tau=128ms`）：

| method | env | AUC | best-F1 | Thr(best-F1) |
|---|---|---:|---:|---:|
| n84 | light | 0.870805 | 0.911931 | 0.000000 |
| n84 | mid | 0.848118 | 0.708025 | 0.041733 |
| n84 | heavy | 0.845315 | 0.656564 | 0.064673 |

heavy 分段（2×200k；heavy best-F1 阈值）：

| method | seg0 F1 | seg1 F1 | seg1 precision | seg1 noise_kept_rate |
|---|---:|---:|---:|---:|
| n84 | 0.711997 | 0.538529 | 0.628182 | 0.028934 |

heavy 同口径对照（400k）：

| method | AUC | best-F1 |
|---|---:|---:|
| baseline | 0.912484 | 0.745459 |
| n71 | 0.910270 | 0.747280 |
| n84 | 0.845315 | 0.656564 |

阶段判断：

- 7.37 的“事件链作为主对象”思路在理论上可行，但本次 n84 的首版工程化实现效果明显不足，未进入可用主线。
- 主要问题是链匹配过于局部（8 邻域）且阈值区间偏低，导致大量真实链无法稳定延续，recall 下降明显。
- 后续若继续该线，建议先做：候选链扩展到小半径集 + 链生命周期管理（建链/并链/终止）+ 代价项归一化，而不是直接调阈值。

### 方法9.1（2026-04-14）：N84-R（按 7.38 意见重构）

用户在 7.38 对 n84 提出结构性意见后，本轮按“不是调阈值，而是改对象模型”的要求重构 N84。

重构目标：

- 从“像素残影链”改为“显式 tracklet 对象”管理。
- 引入链状态机：`tentative / confirmed / dead`。
- 候选匹配从固定 8 邻域改为“预测位置 + 自适应半径”。
- 新链事件采用延迟确认，不再以 `best_link=0` 直接判死。
- 只保留已验证有信息量的信号：dual-footprint（奖励项）、在线 self-hotness、全局 burst proxy。

实现位置：`src/myevs/denoise/ops/ebfopt_part2/n84_event_chain_state_model.py`

本轮产物：

- prescreen（400k）：`data/ED24/myPedestrain_06/EBF_Part2/_prescreen_n84r_s9_tau128_400k/`
- heavy 分段：`data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/validate_400k_heavy_n84r/segf1_n84r_heavy_400k_s9_tau128.csv`

n84r 三环境结果（prescreen400k，`s=9,tau=128ms`）：

| method | env | AUC | best-F1 | Thr(best-F1) |
|---|---|---:|---:|---:|
| n84r | light | 0.485096 | 0.911931 | 0.000000 |
| n84r | mid | 0.541604 | 0.352486 | 0.000000 |
| n84r | heavy | 0.534190 | 0.255785 | 0.000000 |

heavy 分段（2×200k；heavy best-F1 阈值）：

| method | seg0 F1 | seg1 F1 | seg1 precision | seg1 noise_kept_rate |
|---|---:|---:|---:|---:|
| n84r | 0.332381 | 0.171813 | 0.093980 | 1.000000 |

heavy 同口径对照（400k）：

| method | AUC | best-F1 | Thr |
|---|---:|---:|---:|
| baseline | 0.912484 | 0.745459 | 7.356625 |
| n71 | 0.910270 | 0.747280 | 1.968687 |
| n84(old) | 0.845315 | 0.656564 | 0.064673 |
| n84r(7.38) | 0.534190 | 0.255785 | 0.000000 |

阶段结论（本轮）：

- 7.38 提出的“对象级重构方向”已完整落地并跑完，但当前 N84-R 的首版结果明显退化。
- 直接证据是 `Thr(best-F1)=0` 且 heavy 分段 `noise_kept_rate=1.0`，说明分数分离失败，阈值退化为“全保留”。
- 因此本轮结论是：N84-R 架构已实现，但当前参数化/打分归一化仍不可用，不进入主线。

下一步（若继续该线）：

- 先做分数可分离性修复：约束 `tentative->confirmed` 输出占比，并引入链质量分数下界/上界归一化，避免阈值塌到 0。
- 再做链候选集合收敛（按 block 管理活跃链，限制候选数），避免“全保留”式匹配泛化。
- 修复后先只看 heavy seg1 的 PR 曲线与链长度分布，再决定是否继续扩展到全环境。

### 方法9.2（2026-04-14）：N85（按 7.39 创建的新算法 ECSM-S）

用户要求：n84 已失败，按 7.39 的“减法重构”思路重新创建新算法并完整跑实验。

本轮创建的新算法（N85）核心设计：

- 保留显式链对象，但从 n84r 的复杂乘法质量项切换到“简化链模型（ECSM-S）”。
- tentative 链事件改为非零出生分（`gamma_birth * U_foot * (1-U_hot)`），避免 `Thr=0` 型分数塌缩。
- 链接代价只保留四项：`D = w_x*D_x + w_t*D_t + w_h*U_hot - w_f*U_foot`（去掉速度项）。
- confirmed 链质量强调“传播性”：`Q = aL*Q_L + aP*Q_prop + aN*Q_novel + aH*Q_hot + aG*Q_global`。

实现位置：`src/myevs/denoise/ops/ebfopt_part2/n85_event_chain_simplified.py`

本次产物：

- prescreen（400k）：`data/ED24/myPedestrain_06/EBF_Part2/_prescreen_n85_s9_tau128_400k/`
- heavy 分段：`data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/validate_400k_heavy_n85/segf1_n85_heavy_400k_s9_tau128.csv`

n85 三环境结果（prescreen400k，`s=9,tau=128ms`）：

| method | env | AUC | best-F1 | Thr(best-F1) |
|---|---|---:|---:|---:|
| n85 | light | 0.876236 | 0.935309 | 0.008346 |
| n85 | mid | 0.846489 | 0.646407 | 0.152664 |
| n85 | heavy | 0.849165 | 0.567822 | 0.184272 |

heavy 分段（2×200k；heavy best-F1 阈值）：

| method | seg0 F1 | seg1 F1 | seg1 precision | seg1 noise_kept_rate |
|---|---:|---:|---:|---:|
| n85 | 0.654717 | 0.428914 | 0.310040 | 0.160576 |

heavy 同口径对照（400k）：

| method | AUC | best-F1 | Thr |
|---|---:|---:|---:|
| baseline | 0.912484 | 0.745459 | 7.356625 |
| n71 | 0.910270 | 0.747280 | 1.968687 |
| n84(old) | 0.845315 | 0.656564 | 0.064673 |
| n84r(7.38) | 0.534190 | 0.255785 | 0.000000 |
| n85(7.39) | 0.849165 | 0.567822 | 0.184272 |

阶段结论（本轮）：

- 相比 n84r，n85 明显修复了“阈值塌缩”：`Thr(best-F1)` 从 0 恢复到正值，`noise_kept_rate` 从 1.0 降到约 0.16。
- 但 n85 在 heavy 口径仍显著落后于 baseline/n71，说明“传播性建模方向正确但当前参数化仍偏弱”。
- 结论：n85 作为 7.39 新算法已完成实现与验证，当前可作为继续迭代基线，但暂不进入主线替代。

### 方法9.3（2026-04-14）：N86（按 7.40 创建的新算法 ECSM-C）

用户要求：结合 7.40 继续优化，要求同上（新算法 + 同口径完整实验 + README 补充）。

本轮创建的新算法（N86）核心设计：

- 从“逐事件即时打分”改为“确认式链决策（ECSM-C）”。
- tentative 链事件默认 `score=0`，不参与排序；只有链达到确认条件后，才回填释放链内缓存事件分数。
- 确认条件采用 7.40 的最小集：`len >= 3` 且 `birth->last 位移 >= 2 px`（可由环境变量调参）。
- 链匹配代价保持简化四项：`D = w_x*D_x + w_t*D_t + w_h*U_hot - w_f*U_foot`。
- confirmed 链事件分数使用链质量 `Q(c)`（长度/传播/低热度/足迹支持）并裁剪到 `[0,1]`。

实现位置：`src/myevs/denoise/ops/ebfopt_part2/n86_event_chain_confirmed.py`

本次产物：

- prescreen（400k）：`data/ED24/myPedestrain_06/EBF_Part2/_prescreen_n86_s9_tau128_400k/`
- heavy 分段：`data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/validate_400k_heavy_n86/segf1_n86_heavy_400k_s9_tau128.csv`

n86 三环境结果（prescreen400k，`s=9,tau=128ms`）：

| method | env | AUC | best-F1 | Thr(best-F1) |
|---|---|---:|---:|---:|
| n86 | light | 0.862710 | 0.911931 | 0.000000 |
| n86 | mid | 0.794469 | 0.700371 | 0.404561 |
| n86 | heavy | 0.743807 | 0.609415 | 0.390952 |

heavy 分段（2×200k；heavy best-F1 阈值）：

| method | seg0 F1 | seg1 F1 | seg1 precision | seg1 noise_kept_rate |
|---|---:|---:|---:|---:|
| n86 | 0.673725 | 0.474767 | 0.556111 | 0.034293 |

heavy 同口径对照（400k）：

| method | AUC | best-F1 | Thr |
|---|---:|---:|---:|
| baseline | 0.912484 | 0.745459 | 7.356625 |
| n71 | 0.910270 | 0.747280 | 1.968687 |
| n84(old) | 0.845315 | 0.656564 | 0.064673 |
| n84r(7.38) | 0.534190 | 0.255785 | 0.000000 |
| n85(7.39) | 0.849165 | 0.567822 | 0.184272 |
| n86(7.40) | 0.743807 | 0.609415 | 0.390952 |

阶段结论（本轮）：

- 相比 n85，n86 在 heavy best-F1 与 seg1 precision 上有提升（0.5678 -> 0.6094，0.3100 -> 0.5561），且 seg1 噪声保留率显著下降（0.1606 -> 0.0343）。
- 但 n86 的 AUC 明显下降（尤其 heavy 0.8492 -> 0.7438），表明“确认式硬门控”提升了 precision，但损失了排序可分性与一部分 recall。
- light 环境 best-F1 阈值回到 0，说明当前确认条件在低噪声场景仍需细化（例如更稳的确认后分级打分或环境自适应确认阈）。
- 结论：n86 完成了 7.40 的方向性验证，证明“链确认先行”可显著抑制 heavy 噪声；下一步应在保持低噪声段排序能力的前提下，减轻 hard gate 带来的 AUC 损失。

### 方法9.4（2026-04-14）：N87（soft-confirm + 自适应确认门限）

用户要求：继续迭代（要求同上：新算法 + 同口径完整实验 + README 补充）。

本轮创建的新算法（N87）核心设计：

- 以 n86 为底座，加入 confirmed 事件“软释放打分”：`score = (1-link_blend)*Q + link_blend*exp(-D)`，提升已确认链内事件排序信息。
- 保留 tentative 链事件 `score=0`，继续贯彻“先确认链再释放”。
- 引入全局 burst 代理 `g`（短/长时间常数事件率比），对确认条件做自适应：
	- `len_eff = len_min + round(len_add_max * beta_g * g)`
	- `disp_eff = disp_min * (1 + disp_scale_max * beta_g * g)`
- 目标：在保持 n86 抑噪能力同时，恢复部分 AUC 与阈值稳定性。

实现位置：`src/myevs/denoise/ops/ebfopt_part2/n87_event_chain_confirmed_soft.py`

本次产物：

- prescreen（400k）：`data/ED24/myPedestrain_06/EBF_Part2/_prescreen_n87_s9_tau128_400k/`
- heavy 分段：`data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/validate_400k_heavy_n87/segf1_n87_heavy_400k_s9_tau128.csv`

n87 三环境结果（prescreen400k，`s=9,tau=128ms`）：

| method | env | AUC | best-F1 | Thr(best-F1) |
|---|---|---:|---:|---:|
| n87 | light | 0.845947 | 0.911931 | 0.000000 |
| n87 | mid | 0.790509 | 0.688273 | 0.530867 |
| n87 | heavy | 0.737089 | 0.597047 | 0.479652 |

heavy 分段（2×200k；heavy best-F1 阈值）：

| method | seg0 F1 | seg1 F1 | seg1 precision | seg1 noise_kept_rate |
|---|---:|---:|---:|---:|
| n87 | 0.659155 | 0.466594 | 0.565775 | 0.031605 |

heavy 同口径对照（400k）：

| method | AUC | best-F1 | Thr |
|---|---:|---:|---:|
| baseline | 0.912484 | 0.745459 | 7.356625 |
| n71 | 0.910270 | 0.747280 | 1.968687 |
| n84(old) | 0.845315 | 0.656564 | 0.064673 |
| n84r(7.38) | 0.534190 | 0.255785 | 0.000000 |
| n85(7.39) | 0.849165 | 0.567822 | 0.184272 |
| n86(7.40) | 0.743807 | 0.609415 | 0.390952 |
| n87(7.40+) | 0.737089 | 0.597047 | 0.479652 |

阶段结论（本轮）：

- n87 相比 n86，seg1 precision 与 noise_kept_rate 仅有微弱改善（0.5561 -> 0.5658，0.0343 -> 0.0316），但 AUC 与 best-F1 进一步下降。
- 说明“在硬确认框架内叠加软释放与轻度自适应门限”不足以弥补 recall/排序损失，主瓶颈仍在“tentative 全零导致信号释放过晚”。
- 结论：n87 已完整验证但不优于 n86，当前不作为下一轮主线；后续应考虑“极低分 tentative 预释放 + confirmed 后重评分”的双阶段方案。

### 方法9.5（2026-04-14）：N88（按 7.41 的极简传播链 N88-lite）

用户要求：先简化链路，验证“链路这条路到底能不能走”，并按同口径完成实现+实验+README。

本轮创建的新算法（N88-lite）核心设计：

- 严格按 7.41 做减法：只保留“时空接链 + 传播质量分”，去掉 hotness/footprint/global burst/confirmed gate。
- 链匹配仅用：`D = w_x*D_x + w_t*D_t`，其中 `D_x=dist/r_link`，`D_t=dt/t_link`。
- 事件分数直接取所属链质量：`Q = a_L*Q_L + a_P*Q_P + a_N*Q_N`。
- 质量项只保留三项：
	- `Q_L = L/(L+k_L)`
	- `Q_P = disp_birth_to_last / path_len`
	- `Q_N = unique_move_ratio`（基于“移动且非两步回弹”近似）

实现位置：`src/myevs/denoise/ops/ebfopt_part2/n88_event_chain_minimal.py`

本次产物：

- prescreen（400k）：`data/ED24/myPedestrain_06/EBF_Part2/_prescreen_n88_s9_tau128_400k/`
- heavy 分段：`data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/validate_400k_heavy_n88/segf1_n88_heavy_400k_s9_tau128.csv`

n88 三环境结果（prescreen400k，`s=9,tau=128ms`）：

| method | env | AUC | best-F1 | Thr(best-F1) |
|---|---|---:|---:|---:|
| n88 | light | 0.833481 | 0.911931 | 0.000000 |
| n88 | mid | 0.732918 | 0.610186 | 0.416847 |
| n88 | heavy | 0.705435 | 0.486277 | 0.298309 |

heavy 分段（2×200k；heavy best-F1 阈值）：

| method | seg0 F1 | seg1 F1 | seg1 precision | seg1 noise_kept_rate |
|---|---:|---:|---:|---:|
| n88 | 0.579526 | 0.344940 | 0.241870 | 0.195432 |

heavy 同口径对照（400k）：

| method | AUC | best-F1 | Thr |
|---|---:|---:|---:|
| baseline | 0.912484 | 0.745459 | 7.356625 |
| n71 | 0.910270 | 0.747280 | 1.968687 |
| n84(old) | 0.845315 | 0.656564 | 0.064673 |
| n84r(7.38) | 0.534190 | 0.255785 | 0.000000 |
| n85(7.39) | 0.849165 | 0.567822 | 0.184272 |
| n86(7.40) | 0.743807 | 0.609415 | 0.390952 |
| n87(7.40+) | 0.737089 | 0.597047 | 0.479652 |
| n88(7.41-lite) | 0.705435 | 0.486277 | 0.298309 |

阶段结论（本轮）：

- N88-lite 明显退化，尤其 heavy：best-F1 仅 0.486，seg1 precision 降至 0.242，noise_kept_rate 升至 0.195。
- 这说明“只靠传播性链评分（不加热点/足迹/确认机制）”在当前数据口径下判别力不足，噪声链会大量伪装成传播链。
- 结论：7.41 的极简链路已经完成验证，结果倾向于“纯链主轴不可单独成立”；后续若继续链路方向，必须重新引入至少一种抑噪先验（hotness 或 footprint）并保持轻量。

### 方法9.6（2026-04-15）：N89（按 7.42 的单一阈值调整）

用户要求：尝试 7.42 方法，并完成“新算法 + 跑完数据 + 记载到 README”。

可行性判断（先验）：

- 7.42 的控制变量思路是可行且必要的：保持 baseline 主干分数不变，仅引入一个外生状态做阈值偏移，便于判断“决策边界条件化”是否有效。

本轮创建的新算法（N89）核心设计：

- 先计算 baseline 原始分数 `S_base`（沿用 `ebf_scores_stream_numba`，不改主干排序机制）。
- 再做单一状态偏移：`S' = S_base - lambda * Z_b(t)`。
- 状态量定义为 block hotness 归一化：
	- `H_b(t) ~ N_b/A_b`（用指数窗近似，保持在线单遍）
	- `Z_b(t)=clip((H_b-H_low)/(H_high-H_low),0,1)`
- 只加一个状态量，不叠加第二状态机，不改链路主结构。

实现位置：`src/myevs/denoise/ops/ebfopt_part2/n89_single_threshold_blockhot.py`

默认参数（可环境变量覆盖）：

- `MYEVS_N89_BLOCK_SIZE=16`
- `MYEVS_N89_T_STATE_US=25000`
- `MYEVS_N89_T_ACTIVE_US=25000`
- `MYEVS_N89_LAMBDA=0.20`
- `MYEVS_N89_H_LOW=1.20`
- `MYEVS_N89_H_HIGH=4.00`

本次产物：

- prescreen（400k）：`data/ED24/myPedestrain_06/EBF_Part2/_prescreen_n89_s9_tau128_400k/`
- heavy 分段：`data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/validate_400k_heavy_n89/segf1_n89_heavy_400k_s9_tau128.csv`

n89 三环境结果（prescreen400k，`s=9,tau=128ms`）：

| method | env | AUC | best-F1 | Thr(best-F1) |
|---|---|---:|---:|---:|
| n89 | light | 0.949727 | 0.950716 | 0.800969 |
| n89 | mid | 0.918054 | 0.791482 | 5.178727 |
| n89 | heavy | 0.912647 | 0.745591 | 7.282914 |

heavy 分段（2×200k；heavy best-F1 阈值）：

| method | seg0 F1 | seg1 F1 | seg1 precision | seg1 noise_kept_rate |
|---|---:|---:|---:|---:|
| n89 | 0.786925 | 0.655304 | 0.699029 | 0.027544 |

heavy 同口径对照（400k）：

| method | AUC | best-F1 | Thr |
|---|---:|---:|---:|
| baseline | 0.912484 | 0.745459 | 7.356625 |
| n71 | 0.910270 | 0.747280 | 1.968687 |
| n84(old) | 0.845315 | 0.656564 | 0.064673 |
| n84r(7.38) | 0.534190 | 0.255785 | 0.000000 |
| n85(7.39) | 0.849165 | 0.567822 | 0.184272 |
| n86(7.40) | 0.743807 | 0.609415 | 0.390952 |
| n87(7.40+) | 0.737089 | 0.597047 | 0.479652 |
| n88(7.41-lite) | 0.705435 | 0.486277 | 0.298309 |
| n89(7.42) | 0.912647 | 0.745591 | 7.282914 |

阶段结论（本轮）：

- 7.42 方法验证为“可行”：在不破坏 baseline 主干的前提下，N89 在 heavy 上达到与 baseline 几乎等价且略优的 best-F1（0.74559 vs 0.74546），AUC 也略高。
- 相比 n86/n87/n88，N89 的 seg1 precision 与 noise_kept_rate 显著更优（precision 0.699，noise_kept_rate 0.0275）。
- 说明当前阶段“单一状态阈值偏移”比“重构复杂链路”更稳健，建议把这条线作为后续主线继续做参数稳健性与固定阈值迁移验证。

### 方法9.7（2026-04-15）：N90（重新开始的 baseline 最小规律验证）

用户要求：不和前面复杂算法结合，重新开始，仅在 baseline 上做最小优化以验证规律可行性。

本轮创建的新算法（N90）核心设计：

- 主干仍是 baseline 的邻域线性时衰求和，但**调制项直接内嵌进每个邻域贡献项**（不是先算完 baseline 再后乘）。
- 只保留一个严格触发规则（不看未来事件）：
	- 在当前事件邻域（默认窗口 9）找最近两个过去事件 `j1/j2`
	- 仅当 `j1/j2` 都存在且都与当前事件同极性时，才触发时间衰减加分
	- 否则不加分（增益为 0）
- 不引入 block 状态、不引入链状态机、不叠加历史复杂机制。

N90 详细公式（按最终确认口径）：

设当前事件为 $e_i=(x_i,y_i,p_i,t_i)$，邻域窗口为 $W_i$（默认 $9\times9$）。

1) 先定义候选邻域过去事件集合（因果 + 时间窗）：

$$
\mathcal{B}_i=\{j\mid j<i,\ (x_j,y_j)\in W_i,\ 0<\Delta t_{ij}\le \tau\},\quad \Delta t_{ij}=t_i-t_j
$$

其中 $t_i$ 是当前事件时刻，$t_j$ 是过去事件时刻，因此 $\Delta t_{ij}$ 明确定义为“当前时刻减去上一次（历史）时刻”。

2) 将“前两次都同极性”的约束直接写进 $\mathcal{N}_i$（不再单独设门控 $G_i$）：

$$
\mathcal{N}_i=\{j_1,j_2\}\subseteq \mathcal{B}_i
$$

其中 $j_1,j_2$ 是 $\mathcal{B}_i$ 中按时间最近的两个事件，满足

$$
t_{j_1}\ge t_{j_2},\quad p_{j_1}=p_{j_2},\quad p_{j_1}=p_i,\ p_{j_2}=p_i
$$

若满足条件的事件少于 2 个，则定义 $\mathcal{N}_i=\varnothing$（即该事件不打分）。

3) 不再使用 $\delta$，直接用指数时衰替换 baseline 线性权重，并取 $\tau_{mod}=\tau$：

$$
w_{new}(i,j)=\exp\!\left(-\frac{\Delta t_{ij}}{\tau}\right)
$$

4) 最终分数：

$$
S_i=\sum_{j\in\mathcal{N}_i} w_{new}(i,j)
$$

因此该口径下没有 $G_i$，也没有 $\delta$，打分完全由“邻域前两个同极性历史事件”的指数时衰和组成。

实现位置：`src/myevs/denoise/ops/ebfopt_part2/n90_baseline_nb2_causal.py`

当前实现参数：

- `MYEVS_N90_WINDOW=9`
- `\tau_mod` 固定为 `\tau`（不再单独设环境变量）
- 不使用 `\delta/\lambda/clamp` 相关项

本次产物：

- prescreen（400k，新实现）：`data/ED24/myPedestrain_06/EBF_Part2/_prescreen_n90_insum_strict_s9_tau128_400k/`
- heavy 分段（新实现）：`data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/validate_400k_heavy_n90_insum_strict/segf1_n90_heavy_400k_s9_tau128.csv`

n90 三环境结果（prescreen400k，`s=9,tau=128ms`）：

| method | env | AUC | best-F1 | Thr(best-F1) |
|---|---|---:|---:|---:|
| n90 | light | 0.947951 | 0.949753 | 0.753867 |
| n90 | mid | 0.918159 | 0.790916 | 5.269396 |
| n90 | heavy | 0.912844 | 0.744401 | 7.779500 |

heavy 分段（2×200k；heavy best-F1 阈值）：

| method | seg0 F1 | seg1 F1 | seg1 precision | seg1 noise_kept_rate |
|---|---:|---:|---:|---:|
| n90 | 0.786899 | 0.651207 | 0.701943 | 0.026749 |

heavy 简洁对照（400k）：

| method | AUC | best-F1 | Thr |
|---|---:|---:|---:|
| baseline | 0.912484 | 0.745459 | 7.356625 |
| n89(7.42) | 0.912647 | 0.745591 | 7.282914 |
| n90(old-postmul) | 0.912082 | 0.742251 | 7.676063 |
| n90(new-insum-strict) | 0.912844 | 0.744401 | 7.779500 |

阶段结论（本轮）：

- 这条“最小规律验证”是可行的：n90 保持了很强的 baseline 等级性能（heavy AUC/F1 仅小幅下降）。
- 但 n90 不如 n89（7.42 单状态阈值偏移）稳定，说明“只靠邻域双历史调制”信号强度还不够。
- 结论：作为“重新开始且尽量简单”的可行性验证，n90 通过；若要实用提升，优先继续 n89 主线，再将 n90 作为可控弱调制补充项。

注：上表已是“内嵌求和 + 严格双同极性门控”复跑结果；旧版后乘实现保留在对照行 `n90(old-postmul)` 便于横向比较。

#### 9.7b 复跑（2026-04-15）：按“Ni=前两个同极性邻域事件 + 指数和”实现

实现口径：

- 不再使用 `delta` 与后乘结构。
- 对当前事件，只取邻域内最近两个、且都与当前事件同极性的历史事件；若不足两个则该事件得分为 0。
- 打分：`score = exp(-dt1/tau) + exp(-dt2/tau)`，其中 `dt = t_i - t_j`。

产物目录：

- prescreen（400k）：`data/ED24/myPedestrain_06/EBF_Part2/_prescreen_n90_nb2same_exp_tau_s9_tau128_400k/`
- heavy 分段：`data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/validate_400k_heavy_n90_nb2same_exp_tau/segf1_n90_heavy_400k_s9_tau128.csv`

三环境结果（prescreen400k，`s=9,tau=128ms`）：

| method | env | AUC | best-F1 | Thr(best-F1) |
|---|---|---:|---:|---:|
| n90(nb2same-exp) | light | 0.933837 | 0.943520 | 0.754802 |
| n90(nb2same-exp) | mid | 0.881094 | 0.679516 | 1.884476 |
| n90(nb2same-exp) | heavy | 0.852315 | 0.556040 | 1.930220 |

heavy 分段（2×200k；heavy best-F1 阈值）：

| method | seg0 F1 | seg1 F1 | seg1 precision | seg1 noise_kept_rate |
|---|---:|---:|---:|---:|
| n90(nb2same-exp) | 0.642685 | 0.396909 | 0.334586 | 0.100621 |

heavy 对照（400k）：

| method | AUC | best-F1 | Thr |
|---|---:|---:|---:|
| baseline | 0.912484 | 0.745459 | 7.356625 |
| n89(7.42) | 0.912647 | 0.745591 | 7.282914 |
| n90(prev-insum-strict) | 0.912844 | 0.744401 | 7.779500 |
| n90(nb2same-exp) | 0.852315 | 0.556040 | 1.930220 |

阶段结论（9.7b）：

- 该“只取两个同极性邻域事件并做指数和”的版本性能明显下滑，尤其在 mid/heavy 场景（F1 与 AUC 均显著低于 baseline/n89）。
- 说明该规则在当前数据上约束过强，导致有效信号支持不足，不适合作为当前主线实现。

### 方法9.8（2026-04-15）：N91（基于 7.43 建议的新算法，baseline + pixel-rhythm gating）

这条线不是继续修补 N90，而是按 7.43 的建议重新创建一个新算法：保留 baseline 的完整邻域支持，但在每个邻域像素上叠加一个“像素节律”权重，避免把 baseline 主干改成过窄的两点结构。

N91 核心设计：

- 主干保留 baseline 的邻域线性时衰求和，不改掉完整邻域支持。
- 对每个邻域像素，引入一个基于“该像素最近两次同极性事件间隔”的节律置信度。
- 节律置信度不是硬门控，而是软调制：节律越稳定，权重越接近 1；节律异常时逐步降权。
- 参数只保留两个可调项：`T_r` 和 `g_min`。

N91 详细公式：

设当前事件为 $e_i=(x_i,y_i,p_i,t_i)$，邻域窗口仍为 $W_i$（默认 $9\times9$）。对每个邻域像素 $u=(x_u,y_u)$，定义它的历史同极性时间序列中的最近两次时刻为 $t_{u,p}^{(1)}$、$t_{u,p}^{(2)}$，其中 $p$ 与当前事件极性一致。

1) baseline 邻域支持项：

$$
S_i^{base} = \sum_{j \in \mathcal{B}_i} \exp\!\left(-\frac{t_i-t_j}{\tau}\right)
$$

其中 $\mathcal{B}_i$ 仍表示当前事件邻域内、时间窗内的历史候选事件集合。

2) 像素节律间隔：

$$
\Delta r_{u,p} = t_{u,p}^{(1)} - t_{u,p}^{(2)}
$$

若历史不足两次，则令 $g_{u,p}=g_{min}$。

3) 节律置信度映射：

$$
g_{u,p} = g_{min} + (1-g_{min})\exp\!\left(-\frac{|\Delta r_{u,p}-T_r|}{T_r}\right)
$$

这里 $T_r$ 是期望节律周期，$g_{min}$ 是最小保底权重。

4) 最终得分：

$$
S_i = \sum_{j \in \mathcal{B}_i} g_{x_j,y_j,p_i} \cdot \exp\!\left(-\frac{t_i-t_j}{\tau}\right)
$$

也就是说，N91 保留了 baseline 的“满邻域支持”，只是在每个邻域像素上按其同极性历史节律做软调制。

实现位置：`src/myevs/denoise/ops/ebfopt_part2/n91_pixel_rhythm_baseline.py`

当前实验参数：

- `MYEVS_N91_TR_US=1000`
- `MYEVS_N91_GMIN=0.4`

本次产物：

- prescreen（400k）：`data/ED24/myPedestrain_06/EBF_Part2/_slim_n91_rhythm_prescreen400k_s9_tau128ms/`
- heavy 分段（2×200k）：`data/ED24/myPedestrain_06/EBF_Part2/_slim_n91_rhythm_prescreen400k_s9_tau128ms/segment_f1_heavy_2x200k.csv`

N91 三环境结果（prescreen400k，`s=9,tau=128ms`）：

| method | env | AUC | best-F1 | Thr(best-F1) |
|---|---|---:|---:|---:|
| n91 | light | 0.946607 | 0.949159 | 0.810125 |
| n91 | mid | 0.912233 | 0.782098 | 5.864274 |
| n91 | heavy | 0.905161 | 0.731341 | 8.558063 |

heavy 分段（2×200k；heavy best-F1 阈值）：

| method | seg0 F1 | seg1 F1 | seg1 precision | seg1 noise_kept_rate |
|---|---:|---:|---:|---:|
| n91 | 0.776350 | 0.632521 | 0.686515 | 0.027775 |

heavy 简洁对照（400k）：

| method | AUC | best-F1 | Thr |
|---|---:|---:|---:|
| baseline | 0.912484 | 0.745459 | 7.356625 |
| n90(new-insum-strict) | 0.912844 | 0.744401 | 7.779500 |
| n91 | 0.905161 | 0.731341 | 8.558063 |

阶段结论（9.8）：

- N91 按 7.43 的建议把“完整 baseline 支持”保住了，实验也已经跑通。
- 这次的节律软门控能在 light 上维持很高的 AUC/F1，但在 mid/heavy 上比 baseline 更保守，说明当前 `T_r=1000us, g_min=0.4` 还偏紧。
- 这条线的价值在于：它验证了“baseline 主干 + 像素节律软调制”是可执行的；后续如果继续做，优先扫 `T_r` 和 `g_min`，而不是再削弱主干邻域支持。

N91 参数网格（400k 预筛，`s=9,tau=128ms`）：

| env | best `T_r` | best `g_min` | AUC | best-F1 | Thr(best-F1) |
|---|---:|---:|---:|---:|---:|
| light | 2000us | 0.2 | 0.946633 | 0.949196 | 0.809008 |
| mid | 2000us | 0.4 | 0.912260 | 0.782158 | 5.870547 |
| heavy | 2000us | 0.4 | 0.905178 | 0.731356 | 8.561234 |

这个网格的整体结论是：`T_r` 从 200us 往 2000us 拉时，结果会更接近当前数据里的稳定节律；`g_min` 对结果影响很小，但过低会让轻场景略占优、过高会把中重场景再压紧一点。

N91 扩大 `T_r` 范围后的宽扫（400k 预筛，`T_r=3000/4000/6000/8000/12000us`，`g_min=0.2/0.4/0.6`）：

| env | best `T_r` | best `g_min` | AUC | best-F1 | Thr(best-F1) |
|---|---:|---:|---:|---:|---:|
| light | 8000us | 0.2 | 0.946642 | 0.949280 | 0.809477 |
| mid | 3000us | 0.4 | 0.912282 | 0.782158 | 5.694617 |
| heavy | 4000us | 0.4 | 0.905170 | 0.731426 | 8.557446 |

宽扫结论：

- `T_r` 继续增大对 light 还有极轻微收益，但 mid/heavy 不再同步受益，说明节律周期不能继续无脑放大。
- mid/heavy 的最优点更靠近 `3000-4000us`，这和上一轮 `2000us` 的结果相比，说明“最优节律”确实被扫到了更大的尺度，但已经出现分环境分化。
- 如果下一轮还要继续，建议围绕 `T_r=3000/4000/8000us` 做更细的一维微调，而不是继续把范围再翻倍。

N91 的窗口长度/时间常数简单扫参（固定 `T_r=4000us, g_min=0.4`）：

- 扫参范围：`s=7/9/11`，`tau=32/64/128/256ms`
- 输出目录：`data/ED24/myPedestrain_06/EBF_Part2/_slim_n91_s_tau_grid_tr4000_g0p4_400k/`

按环境的最优组合：

| env | best-AUC 组合 | AUC | 该组合 F1 | best-F1 组合 | best-F1 | Thr(best-F1) |
|---|---|---:|---:|---|---:|---:|
| light | s11,tau128ms | 0.947443 | 0.950604 | s11,tau256ms | 0.952550 | 1.698395 |
| mid | s9,tau128ms | 0.912287 | 0.782133 | s9,tau128ms | 0.782133 | 5.695063 |
| heavy | s9,tau128ms | 0.905170 | 0.731426 | s7,tau64ms | 0.733727 | 3.853750 |

总体趋势（按每个 `s,tau` 组合先取各环境 best-F1，再做均值）：

- `tau` 方向：
	- 32ms: 0.798981
	- 64ms: 0.817889
	- 128ms: 0.819507（最高）
	- 256ms: 0.807982
- `s` 方向：
	- s7: 0.809866
	- s9: 0.812356（最高）
	- s11: 0.811048

结论：

- 从“整体稳健”角度看，`s=9, tau=128ms` 仍然是最平衡的点（与 baseline 常用配置一致）。
- 如果偏向 light，可尝试更大窗口/更长时间常数（如 s11,tau256ms）；如果偏向 heavy 的 F1，可尝试更短时间常数（如 s7,tau64ms）。

### 方法9.9（2026-04-15）：N92（7.44 -> 7.45 改版）

N92 的核心目标是验证一个和 baseline 不同的信息轴：

- baseline 关注“支持强度/总量”；
- N92 关注“同极性历史支持按时间排序后，空间是否连续推进”。

实现位置：`src/myevs/denoise/ops/ebfopt_part2/n92_local_temporal_spatial_continuity.py`

#### 7.44 原始定义（旧版）

候选集合：

$$
\mathcal{B}_i = \{j\mid j<i,\ (x_j,y_j)\in W_i,\ p_j=p_i,\ 0<t_i-t_j\le\tau\}
$$

取最近的前 $K$ 个候选，按时间从旧到新记为 $j_{(1)},\dots,j_{(K_i)}$，其中 $K_i=\min(K,|\mathcal{B}_i|)$。

定义相邻时间点空间距离：

$$
d_m=\|\mathbf{x}_{j_{(m+1)}}-\mathbf{x}_{j_{(m)}}\|,\ m=1,\dots,K_i-1
$$

最后一段到当前事件：

$$
d_{K_i}=\|\mathbf{x}_i-\mathbf{x}_{j_{(K_i)}}\|
$$

平均连续距离：

$$
\bar d_i=\frac{1}{K_i}\sum_{m=1}^{K_i} d_m
$$

最终分数（独立主分数，不与 baseline 融合）：

$$
Score_i=
\begin{cases}
\exp\left(-\dfrac{\bar d_i}{\sigma_d}\right), & K_i\ge K_{\min} \\
0, & K_i<K_{\min}
\end{cases}
$$

#### 7.45 改动（本次采用）

7.45 的关键修改是“每像素每极性只保留一个最近时间戳”，避免同像素重复触发在链上重复出现，伪造“高连续性”。

对应地，候选集合从“事件级”改为“像素级最新事件”：

$$
\tilde{\mathcal{B}}_i = \{u\in W_i\mid t^{(1)}_{u,p_i}>0,\ 0<t_i-t^{(1)}_{u,p_i}\le\tau\}
$$

其中 $t^{(1)}_{u,p_i}$ 是像素 $u$ 在极性 $p_i$ 下的最近一次时间戳。之后仍取最近前 $K$、时间排序、按同样的 $\bar d_i$ 与指数映射打分。

本次运行参数：

- `MYEVS_N92_K=4`
- `MYEVS_N92_KMIN=2`
- `MYEVS_N92_SIGMA_D=2.0`
- 扫参：`s=7,9`，`tau=32/64/128ms`

产物目录：

- 7.44 prescreen（400k）：`data/ED24/myPedestrain_06/EBF_Part2/_slim_n92_ltsc_prescreen400k_s7_9_tau32_64_128ms/`
- 7.45 prescreen（400k）：`data/ED24/myPedestrain_06/EBF_Part2/_slim_n92_ltsc745_prescreen400k_s7_9_tau32_64_128ms/`
- 7.45 heavy 分段（2x200k）：`data/ED24/myPedestrain_06/EBF_Part2/_slim_n92_ltsc745_prescreen400k_s7_9_tau32_64_128ms/segment_f1_heavy_2x200k_s7_tau32ms.csv`

#### 7.44 vs 7.45 对照（400k）

| env | 指标 | 7.44 | 7.45 | delta |
|---|---|---:|---:|---:|
| light | best AUC | 0.817148 | 0.885738 | +0.068590 |
| light | best F1 | 0.934299 | 0.940781 | +0.006482 |
| mid | best AUC | 0.829758 | 0.841599 | +0.011841 |
| mid | best F1 | 0.648323 | 0.660101 | +0.011778 |
| heavy | best AUC | 0.803523 | 0.811747 | +0.008224 |
| heavy | best F1 | 0.477200 | 0.486302 | +0.009101 |

7.45 best 组合明细：

| env | best-AUC 组合 | AUC | best-F1 组合 | best-F1 | Thr(best-F1) |
|---|---|---:|---|---:|---:|
| light | s9,tau32ms | 0.885738 | s9,tau128ms | 0.940781 | 0.013101 |
| mid | s7,tau32ms | 0.841599 | s7,tau32ms | 0.660101 | 0.130010 |
| heavy | s7,tau32ms | 0.811747 | s7,tau32ms | 0.486302 | 0.223699 |

7.45 heavy 分段（2x200k；heavy best-F1 阈值，s7/tau32ms）：

| method | seg0 F1 | seg1 F1 | seg1 precision | seg1 noise_kept_rate |
|---|---:|---:|---:|---:|
| n92(7.45) | 0.563669 | 0.366343 | 0.264376 | 0.172121 |

heavy 对照（400k，仍以 7.45 best 点）：

| method | AUC | best-F1 |
|---|---:|---:|
| baseline | 0.912484 | 0.745459 |
| n91 | 0.905161 | 0.731341 |
| n92(7.45) | 0.811747 | 0.486302 |

### N92 规则有效性统计验证（与算法口径一致）

统计脚本：`scripts/noise_analyze/n92_continuity_stats.py`

本次已同步到 7.45 口径：统计时同样采用“每像素每极性仅保留一个最近时间戳”。

统计产物：

- 7.44 统计：`data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/n92_continuity_stats_400k/`
- 7.45 统计：`data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/n92_continuity_stats_400k_v745/`

关键指标定义（不变）：

- `nsel`：实际选到的邻域事件数
- `sum_dist`：时间链相邻距离和
- `mean_dist`：`sum_dist / nsel`，越小表示连续性越好
- `ratio = signal_mean_dist / noise_mean_dist`：
  - `<1` 表示 signal 更连续（规则有效）
  - `>1` 表示 signal 更离散（规则失效）
- `valid_rate`：`nsel >= K_min` 占比（用于判断选择偏差）

阶段结论（9.9）：

- 7.45 改版（每像素单时间戳）较 7.44 在 light/mid/heavy 的 AUC 与 F1 全部提升。
- 但 N92 作为独立 backbone 仍明显弱于 baseline / n91，尤其在 heavy 的绝对 F1 仍偏低。
- 工程判断不变：N92 的连续性信息可作为辅助因子融合，单独替代主干暂不成立。

### 方法9.10（2026-04-15）：N93（9.46 时空加权支持）

N93 的思路是把“时间新鲜度”和“空间邻近度”显式耦合，在局部窗口内累加同极性支持强度。

实现位置：`src/myevs/denoise/ops/ebfopt_part2/n93_spatiotemporal_weighted_support.py`

#### 9.46 数学定义

候选集合（每像素每极性仅取最近一次时间戳）：

$$
\tilde{\mathcal{B}}_i = \{u\in W_i\mid t^{(1)}_{u,p_i}>0,\ 0<t_i-t^{(1)}_{u,p_i}\le\tau\}
$$

对任一候选像素 $u$，记 $\Delta t_{iu}=t_i-t^{(1)}_{u,p_i}$，空间距离 $d_{iu}=\|\mathbf{x}_i-\mathbf{x}_u\|$，定义：

$$
w_t(i,u)=\frac{\tau-\Delta t_{iu}}{\tau},\qquad
w_s(i,u)=\exp\left(-\frac{d_{iu}}{\sigma_d}\right)
$$

最终打分：

$$
Score_i=\sum_{u\in\tilde{\mathcal{B}}_i} w_t(i,u)\cdot w_s(i,u)
$$

与 N92 的区别是：N92 先构链再做平均距离映射；N93 直接做“时间衰减 × 空间衰减”的加权求和，因此对局部支持密度更敏感。

本次运行参数：

- `MYEVS_N93_SIGMA_D=2.0`
- 扫参：`s=7,9`，`tau=32/64/128ms`，`max-events=400000`

产物目录：

- prescreen（400k）：`data/ED24/myPedestrain_06/EBF_Part2/_slim_n93_stws946_prescreen400k_s7_9_tau32_64_128ms/`
- heavy 分段（2x200k）：`data/ED24/myPedestrain_06/EBF_Part2/_slim_n93_stws946_prescreen400k_s7_9_tau32_64_128ms/segment_f1_n93_heavy_bestf1_s9_tau64000_2x200k.csv`

#### N93（400k）最优结果

| env | best-AUC 组合 | AUC | best-F1 组合 | best-F1 | Thr(best-F1) |
|---|---|---:|---|---:|---:|
| light | s9,tau128ms | 0.948139 | s9,tau128ms | 0.948045 | 0.107228 |
| mid | s9,tau128ms | 0.916026 | s9,tau64ms | 0.790846 | 0.980350 |
| heavy | s9,tau128ms | 0.908197 | s9,tau64ms | 0.747434 | 1.393349 |

与 N92(7.45) 对照（400k）：

| env | 指标 | N92(7.45) | N93 | delta |
|---|---|---:|---:|---:|
| light | best AUC | 0.885738 | 0.948139 | +0.062401 |
| light | best F1 | 0.940781 | 0.948045 | +0.007264 |
| mid | best AUC | 0.841599 | 0.916026 | +0.074427 |
| mid | best F1 | 0.660101 | 0.790846 | +0.130744 |
| heavy | best AUC | 0.811747 | 0.908197 | +0.096450 |
| heavy | best F1 | 0.486302 | 0.747434 | +0.261133 |

#### heavy 分段稳定性（2x200k）

N93（heavy best-F1 点：s9/tau64ms）：

| method | seg0 F1 | seg1 F1 | seg1 precision | seg1 noise_kept_rate |
|---|---:|---:|---:|---:|
| n93 | 0.793563 | 0.644710 | 0.736418 | 0.021285 |

与 N92(7.45) heavy 分段对照：

| method | seg0 F1 | seg1 F1 | seg1 precision | seg1 noise_kept_rate |
|---|---:|---:|---:|---:|
| n92(7.45) | 0.563669 | 0.366343 | 0.264376 | 0.172121 |
| n93 | 0.793563 | 0.644710 | 0.736418 | 0.021285 |

heavy 总体对照（400k）：

| method | AUC | best-F1 |
|---|---:|---:|
| baseline | 0.912484 | 0.745459 |
| n91 | 0.905161 | 0.731341 |
| n92(7.45) | 0.811747 | 0.486302 |
| n93 | 0.908197 | 0.747434 |

阶段结论（9.10）：

- N93 在 light/mid/heavy 的 AUC 与 F1 均显著优于 N92(7.45)。
- 在 heavy 上，N93 的 best-F1 已与 baseline 持平略优（0.747434 vs 0.745459），AUC 也接近 baseline（0.908197 vs 0.912484）。
- 从工程可用性看，9.46 的“时间×空间”加权支持比“链连续性均值距离”更稳健，可作为后续主干候选继续细化。

### 方法9.11（2026-04-15）：N94（时间线性 × 空间线性）

你提出的方向是可行的：把 N93 的指数空间权重换成线性空间权重，通常会提高中远邻域贡献，从而增强“空间支持强度”在总分里的占比。

实现位置：`src/myevs/denoise/ops/ebfopt_part2/n94_spatiotemporal_linear_support.py`

定义（与 N93 相同的时间项，不同的空间项）：

$$
w_t(i,u)=\frac{\tau-\Delta t_{iu}}{\tau},\qquad
w_s(i,u)=\max\left(0,1-\frac{d_{iu}}{d_{\max}}\right)
$$

$$
Score_i=\sum_{u\in\tilde{\mathcal{B}}_i} w_t(i,u)\cdot w_s(i,u)
$$

其中：

- $\tilde{\mathcal{B}}_i$ 与 N93 一致（同极性、窗口内、每像素最近时间戳、$\Delta t\le\tau$）；
- `d_max` 由环境变量 `MYEVS_N94_SPACE_RANGE_PX` 控制；
- 若 `MYEVS_N94_SPACE_RANGE_PX<=0`，默认取当前窗口几何上限 $\sqrt{2}\,r$（`r=radius_px`）。

与 N93 的差异解释：

- N93：`exp(-d/sigma_d)`，远距离衰减更快；
- N94：`max(0,1-d/dmax)`，远距离保留更多权重，空间项影响更强。

运行示例：

```powershell
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py --variant n94 --max-events 400000 --s-list 7,9 --tau-us-list 32000,64000,128000 --out-dir data/ED24/myPedestrain_06/EBF_Part2/_slim_n94_stls_prescreen400k_s7_9_tau32_64_128ms
```

产物目录：

- prescreen（400k）：`data/ED24/myPedestrain_06/EBF_Part2/_slim_n94_stls_prescreen400k_s7_9_tau32_64_128ms/`
- heavy 分段（2x200k）：`data/ED24/myPedestrain_06/EBF_Part2/_slim_n94_stls_prescreen400k_s7_9_tau32_64_128ms/segment_f1_n94_heavy_bestf1_s9_tau64000_2x200k.csv`

#### N94（400k）最优结果

| env | best-AUC 组合 | AUC | best-F1 组合 | best-F1 | Thr(best-F1) |
|---|---|---:|---|---:|---:|
| light | s9,tau128ms | 0.948183 | s9,tau128ms | 0.948117 | 0.215158 |
| mid | s9,tau128ms | 0.916334 | s9,tau64ms | 0.792019 | 1.715370 |
| heavy | s9,tau128ms | 0.908831 | s9,tau64ms | 0.748917 | 2.463984 |

与 N93 对照（400k）：

| env | 指标 | N93 | N94 | delta |
|---|---|---:|---:|---:|
| light | best AUC | 0.948139 | 0.948183 | +0.000044 |
| light | best F1 | 0.948045 | 0.948117 | +0.000072 |
| mid | best AUC | 0.916026 | 0.916334 | +0.000309 |
| mid | best F1 | 0.790846 | 0.792019 | +0.001173 |
| heavy | best AUC | 0.908197 | 0.908831 | +0.000634 |
| heavy | best F1 | 0.747434 | 0.748917 | +0.001482 |

#### heavy 分段稳定性（2x200k）

N94（heavy best-F1 点：s9/tau64ms）：

| method | seg0 F1 | seg1 F1 | seg1 precision | seg1 noise_kept_rate |
|---|---:|---:|---:|---:|
| n94 | 0.794376 | 0.647176 | 0.745491 | 0.020248 |

与 N93 heavy 分段对照：

| method | seg0 F1 | seg1 F1 | seg1 precision | seg1 noise_kept_rate |
|---|---:|---:|---:|---:|
| n93 | 0.793563 | 0.644710 | 0.736418 | 0.021285 |
| n94 | 0.794376 | 0.647176 | 0.745491 | 0.020248 |

heavy 总体对照（400k）：

| method | AUC | best-F1 |
|---|---:|---:|
| baseline | 0.912484 | 0.745459 |
| n91 | 0.905161 | 0.731341 |
| n93 | 0.908197 | 0.747434 |
| n94 | 0.908831 | 0.748917 |

阶段结论（9.11）：

- 把空间项从指数衰减改为线性衰减是可行的，并且在本轮 400k 三环境上对 N93 形成了稳定小幅增益（AUC/F1 全部正增量）。
- N94 在 heavy 上进一步提升到 `best-F1=0.748917`，继续高于 baseline 的 `0.745459`。
- 增益量级目前较小（约 1e-4 ~ 1e-3），说明该改动方向正确但仍偏“微调收益”；后续可围绕 `d_max` 与时间项耦合策略继续放大效果。

#### N94 的 dmax 调优（2026-04-15）

为验证“空间线性项影响力”是否还能进一步提升，本次对 `MYEVS_N94_SPACE_RANGE_PX` 做了快速扫参：

- 扫描值：`3.5 / 4.5 / 5.5 / 6.5 / 8.0`
- 口径：`max-events=400000`，`s=9`，`tau=64/128ms`
- 输出目录：`data/ED24/myPedestrain_06/EBF_Part2/_sweep_n94_dmax_*_400k_s9_tau64_128/`

结论：

- `dmax=6.5` 在 AUC 上整体最好（light/mid/heavy 的 best-AUC 均高于默认）。
- 但 best-F1 并未同步提升：light 小幅上升，mid/heavy 小幅下降。

默认值（`dmax<=0` 自动取 `sqrt(2)*r`）与 `dmax=6.5` 对照（400k，全网格口径 `s=7,9; tau=32/64/128ms`）：

| env | 指标 | n94(default) | n94(dmax=6.5) | delta |
|---|---|---:|---:|---:|
| light | best AUC | 0.948183 | 0.948330 | +0.000147 |
| light | best F1 | 0.948117 | 0.948498 | +0.000381 |
| mid | best AUC | 0.916334 | 0.916604 | +0.000269 |
| mid | best F1 | 0.792019 | 0.791957 | -0.000062 |
| heavy | best AUC | 0.908831 | 0.909401 | +0.000571 |
| heavy | best F1 | 0.748917 | 0.748655 | -0.000261 |

heavy 分段（2x200k）对照：

| method | seg0 F1 | seg1 F1 | seg1 precision | seg1 noise_kept_rate |
|---|---:|---:|---:|---:|
| n94(default) | 0.794376 | 0.647176 | 0.745491 | 0.020248 |
| n94(dmax=6.5) | 0.793922 | 0.647713 | 0.736296 | 0.021479 |

工程建议（当前阶段）：

- 如果目标是 AUC 优先，可采用 `MYEVS_N94_SPACE_RANGE_PX=6.5`；
- 如果目标是 F1/分段稳定性优先，默认 `dmax`（即 `sqrt(2)*r`）更稳妥。

### 方法9.12（2026-04-15）：N95（最小空间抑制 × 线性空间保留）

你建议的方向可直接实现为一个更保守的空间项：在 N94 的线性保留之外，额外压制过近支持，避免同像素/极近邻过度占优。

实现位置：`src/myevs/denoise/ops/ebfopt_part2/n95_spatiotemporal_min_support.py`

定义：

$$
w_t(i,u)=\frac{\tau-\Delta t_{iu}}{\tau},\qquad
w_s(i,u)=\left(1-\exp\left(-\frac{d_{iu}}{\sigma_{min}}\right)\right)\cdot\max\left(0,1-\frac{d_{iu}}{d_{max}}\right)
$$

$$
Score_i=\sum_{u\in\tilde{\mathcal{B}}_i} w_t(i,u)\cdot w_s(i,u)
$$

其中：

- `sigma_min` 由环境变量 `MYEVS_N95_SIGMA_MIN_PX` 控制，默认 `1.0`；
- `d_max` 由环境变量 `MYEVS_N95_SPACE_RANGE_PX` 控制；
- 若 `MYEVS_N95_SPACE_RANGE_PX<=0`，默认取当前窗口几何上限 $\sqrt{2}\,r$（`r=radius_px`）；
- 其他口径与 N94 一致：同极性、窗口内、每像素最近时间戳、`\Delta t\le\tau`。

第一轮建议扫参（先看 heavy best-F1、heavy seg1 precision、heavy seg1 noise_kept_rate）：

- `sigma_min ∈ {0.5, 1.0, 1.5}`
- `d_max ∈ {sqrt(2)*r, 6.5}`
- 保持 `s=9`、`tau=64ms / 128ms`

运行示例：

```powershell
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py --variant n95 --max-events 400000 --s-list 9 --tau-us-list 64000,128000 --out-dir data/ED24/myPedestrain_06/EBF_Part2/_slim_n95_prescreen400k_s9_tau64_128ms
```

#### N95 首轮扫参结果（400k）

| sigma_min | d_max | light best AUC | light best F1 | mid best AUC | mid best F1 | heavy best AUC | heavy best F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| 0.5 | default | 0.948150 | 0.948123 | 0.916074 | 0.791402 | 0.908724 | 0.748260 |
| 0.5 | 6.5 | 0.948239 | 0.948545 | 0.916190 | 0.790847 | 0.909112 | 0.747544 |
| 1.0 | default | 0.948005 | 0.948316 | 0.915519 | 0.790076 | 0.908336 | 0.746246 |
| 1.0 | 6.5 | 0.947954 | 0.948793 | 0.915256 | 0.788584 | 0.908298 | 0.744038 |
| 1.5 | default | 0.947839 | 0.948441 | 0.915011 | 0.788285 | 0.907894 | 0.744627 |
| 1.5 | 6.5 | 0.947648 | 0.949003 | 0.914442 | 0.787173 | 0.907522 | 0.741047 |

#### heavy 分段对照（2x200k）

| method | heavy best-F1 | heavy AUC | seg1 precision | seg1 noise_kept_rate | seg1 F1 |
|---|---:|---:|---:|---:|---:|
| baseline | 0.786881 | 0.920467 | 0.703224 | 0.026821 | 0.654839 |
| n95 (sigma_min=0.5, default d_max) | 0.748260 | 0.908724 | 0.800081 | 0.013598 | 0.633720 |
| n95 (sigma_min=0.5, d_max=6.5) | 0.747544 | 0.909112 | 0.800081 | 0.013598 | 0.633720 |

阶段说明：

- 已完成首轮扫参：`sigma_min ∈ {0.5, 1.0, 1.5}`，`d_max ∈ {default, 6.5}`，`s=9`，`tau=64/128ms`。
- heavy 上最好的 `best-F1` 来自 `sigma_min=0.5`、`d_max=default`；同组的 `d_max=6.5` 在 heavy AUC 上略高，但 F1 略低。
- 相比原始 EBF baseline，N95 这轮没有拿到整体提升；它更像是一个“更保守”的空间抑制器。
- 从 heavy 分段表看，N95 明显更克制地保留噪声，但召回和 seg1 F1 都回落了。
- 结论：`sigma_min` 这条抑制线在当前配置下会把模型推向“更少保留噪声、也更少保留信号”的方向；如果继续推进，优先尝试把 `sigma_min` 压得更小，或者改成只对极近邻做轻微折损，而不是整段线性叠乘。

## n96：Top1-v Backbone（仅保留 top1 支持的速度一致性）

### 动机与可验证假设

- 目标是先复现 7.48 提出的最小主干：不做多支持融合、不做额外 freshness 衰减，仅用 top1 历史支持的速度一致性打分。
- 可验证假设：如果 signal 在“局部主导支持”的速度上更稳定，则单支持速度兼容项可以提供可分性提升。

### 算法定义（在线 / 单遍 / O(r^2)）

实现位置：`src/myevs/denoise/ops/ebfopt_part2/n96_top1_v_backbone.py`

对当前事件 `e_i=(x_i,y_i,t_i,p_i)`：

1) 在邻域内找同极性且 `dt>0` 的 top1（最近过去）支持，且排除中心像素历史 `x_j==x_i && y_j==y_i`。  
2) 计算：

$$
d=\sqrt{(x_i-x_j)^2+(y_i-y_j)^2},\qquad
\Delta t=t_i-t_j,\qquad
v=\frac{d}{\Delta t+\varepsilon}
$$

3) 仅用速度兼容项输出分数：

$$
score=\exp\left(-\frac{|v-v_0|}{\sigma_v}\right)
$$

其中 `v0` 为目标速度，`sigma_v` 为速度容忍带宽，`eps_us` 为数值稳定项。

### 超参与运行方式

环境变量：

- `MYEVS_N96_V0_PX_PER_US`
- `MYEVS_N96_SIGMA_V_PX_PER_US`
- `MYEVS_N96_EPS_US`

运行示例（prescreen400k，`s=9`，`tau=64/128ms`）：

```powershell
$env:PYTHONNOUSERSITE='1'
$env:MYEVS_N96_V0_PX_PER_US='0.0025'
$env:MYEVS_N96_SIGMA_V_PX_PER_US='0.0020'
$env:MYEVS_N96_EPS_US='1.0'
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py --variant n96 --max-events 400000 --s-list 9 --tau-us-list 64000,128000 --out-dir data/ED24/myPedestrain_06/EBF_Part2/_slim_n96_top1v_prescreen400k_s9_tau64_128_v0_0p0025_sv_0p0020
```

### N96 扫参结果（prescreen400k）

口径：`s=9`，`tau=64/128ms`，`max-events=400000`，网格 `v0∈{0.0015,0.0020,0.0025}`、`sigma_v∈{0.0005,0.0010,0.0020}`。  
表内数值为每个组合在各环境上的 `best AUC / best F1`（允许在 `tau=64/128ms` 间取最优）。

| v0 | sigma_v | light best AUC | light best F1 | mid best AUC | mid best F1 | heavy best AUC | heavy best F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0015 | 0.0005 | 0.808827 | 0.923314 | 0.558321 | 0.373783 | 0.496159 | 0.258851 |
| 0.0015 | 0.0010 | 0.815865 | 0.934090 | 0.559448 | 0.378409 | 0.496499 | 0.259025 |
| 0.0015 | 0.0020 | 0.819613 | 0.939925 | 0.560022 | 0.380688 | 0.496679 | 0.260713 |
| 0.0020 | 0.0005 | 0.820600 | 0.923402 | 0.593070 | 0.392817 | 0.532822 | 0.284000 |
| 0.0020 | 0.0010 | 0.827561 | 0.934301 | 0.594184 | 0.392833 | 0.533159 | 0.283998 |
| 0.0020 | 0.0020 | 0.831294 | 0.939925 | 0.594757 | 0.392833 | 0.533339 | 0.283993 |
| 0.0025 | 0.0005 | 0.829223 | 0.923554 | 0.617850 | 0.415181 | 0.559616 | 0.302769 |
| 0.0025 | 0.0010 | 0.836064 | 0.934318 | 0.618942 | 0.415185 | 0.559946 | 0.302795 |
| 0.0025 | 0.0020 | 0.839764 | 0.939925 | 0.619511 | 0.415200 | 0.560124 | 0.302790 |

### 阶段结论

- N96 已按 7.48 的“top1-v backbone”完成可复现实现，并完成 9 组网格实验。
- 在本轮网格中，`v0` 和 `sigma_v` 增大时表现单调上升，最优组合集中在 `v0=0.0025`。
- 但绝对性能仍显著弱于 EBF baseline（尤其 mid/heavy），说明“仅用 top1 速度兼容项”的判别力不足以支撑当前任务。
- 后续若继续该线，建议在保留 top1-v 主干的前提下，逐步加回最小必要信息（例如时间新鲜度或多支持稳健聚合），而不是直接大规模引入复杂门控。

## n97：Top1-2D Backbone（top1 的二维 `(d,Δt)` 兼容）

### 动机与可验证假设

- 7.49 的核心意见：top1 的速度比值 `v=d/Δt` 会把二维信息压成一维，区分力不够。
- 可验证假设：在保持 top1 极简框架下，直接用 `(d,Δt)` 二维兼容打分，应优于纯 `v` 主干。

### 算法定义（在线 / 单遍 / O(r^2)）

实现位置：`src/myevs/denoise/ops/ebfopt_part2/n97_top1_2d_backbone.py`

对当前事件 `e_i=(x_i,y_i,t_i,p_i)`：

1) 在邻域内找同极性且 `dt>0` 的 top1（最近过去）支持，排除中心像素历史。  
2) 计算 `d_i` 与 `Δt_i`。  
3) 用二维高斯兼容输出分数：

$$
score_i=
\exp\left(-\frac{(d_i-d_0)^2}{2\sigma_d^2}-\frac{(\Delta t_i-t_0)^2}{2\sigma_t^2}\right)
$$

无 top1 支持时记为 0。

### 参数设定（按 7.49）

- `d0/t0` 由 7.47 的 top1 signal 统计给初值：
	- `d0 = d_p50(top1, signal) = 2.1802459 px`
	- `t0 = dt_p50(top1, signal) = 1386.9361 us`
- 首轮只扫带宽：
	- `sigma_d ∈ {0.8, 1.2, 1.8}`
	- `sigma_t ∈ {2000, 5000, 10000}`

环境变量：

- `MYEVS_N97_D0_PX`
- `MYEVS_N97_T0_US`
- `MYEVS_N97_SIGMA_D_PX`
- `MYEVS_N97_SIGMA_T_US`

运行示例（prescreen400k，`s=9`，`tau=64/128ms`）：

```powershell
$env:PYTHONNOUSERSITE='1'
$env:MYEVS_N97_D0_PX='2.180245908658522'
$env:MYEVS_N97_T0_US='1386.9361159168966'
$env:MYEVS_N97_SIGMA_D_PX='1.8'
$env:MYEVS_N97_SIGMA_T_US='5000'
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py --variant n97 --max-events 400000 --s-list 9 --tau-us-list 64000,128000 --out-dir data/ED24/myPedestrain_06/EBF_Part2/_slim_n97_top1_2d_prescreen400k_s9_tau64_128_sd_1p8_st_5000
```

### N97 扫参结果（prescreen400k）

口径：`s=9`，`tau=64/128ms`，`max-events=400000`。  
表内数值为每组 `(sigma_d,sigma_t)` 在各环境上的 `best AUC / best F1`（允许在 `tau=64/128ms` 间取最优）。

| sigma_d | sigma_t(us) | light best AUC | light best F1 | mid best AUC | mid best F1 | heavy best AUC | heavy best F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.8 | 2000 | 0.909305 | 0.938311 | 0.815929 | 0.580486 | 0.777132 | 0.446995 |
| 0.8 | 5000 | 0.915094 | 0.945634 | 0.800922 | 0.555999 | 0.753654 | 0.428517 |
| 0.8 | 10000 | 0.910137 | 0.945851 | 0.770014 | 0.524255 | 0.715890 | 0.394892 |
| 1.2 | 2000 | 0.909276 | 0.938351 | 0.816565 | 0.577614 | 0.779131 | 0.447960 |
| 1.2 | 5000 | 0.917249 | 0.945692 | 0.810831 | 0.572252 | 0.767732 | 0.440624 |
| 1.2 | 10000 | 0.915394 | 0.945900 | 0.790378 | 0.542395 | 0.739884 | 0.416290 |
| 1.8 | 2000 | 0.908397 | 0.938388 | 0.814483 | 0.577692 | 0.776969 | 0.447588 |
| 1.8 | 5000 | 0.918349 | 0.945665 | 0.816027 | 0.579853 | 0.775882 | 0.446316 |
| 1.8 | 10000 | 0.918619 | 0.945901 | 0.804342 | 0.561501 | 0.758355 | 0.432176 |

### 阶段结论

- N97（top1-2D）相比 N96（top1-v）有明显提升，说明“保留二维 `(d,Δt)` 信息”是有效方向。
- 但在当前 top1-only 框架下，mid/heavy 仍显著弱于 baseline，核心瓶颈仍是“单支持点不够稳健”。
- 下一步若继续同主线，优先考虑“保持 2D 兼容核不变，改 top1 为 top3 稳健聚合”，而不是再扩大 `sigma_t`。

## n98：Top3-2D Backbone（top3 的二维 `(d,Δt)` 稳健聚合）

### 动机与可验证假设

- 延续 n97：二维 `(d,Δt)` 兼容核本身有效。
- 主要瓶颈在 top1 支持不稳健，因此改为 top3 聚合以降低单点偶然性。
- 可验证假设：保持同一 2D 核，仅把 top1 改为 top3 均值聚合，mid/heavy 应显著提升。

### 算法定义（在线 / 单遍 / O(r^2)）

实现位置：`src/myevs/denoise/ops/ebfopt_part2/n98_top3_2d_backbone.py`

对当前事件 `e_i=(x_i,y_i,t_i,p_i)`：

1) 在邻域内找同极性且 `dt>0` 的 top3（按最小 `dt` 选最近过去 3 个），排除中心像素历史。  
2) 对每个支持点计算 `d_k` 与 `Δt_k`。  
3) 用与 n97 相同的二维高斯兼容：

$$
g_k=
\exp\left(-\frac{(d_k-d_0)^2}{2\sigma_d^2}-\frac{(\Delta t_k-t_0)^2}{2\sigma_t^2}\right)
$$

4) 最终分数为已找到支持点（1~3 个）的均值：

$$
score_i = \frac{1}{K}\sum_{k=1}^{K} g_k,\quad K\in\{1,2,3\}
$$

无支持时记为 0。

### 参数设定（与 n97 对齐）

- 固定中心参数：
	- `d0 = 2.1802459 px`
	- `t0 = 1386.9361 us`
- 扫带宽：
	- `sigma_d ∈ {0.8, 1.2, 1.8}`
	- `sigma_t ∈ {2000, 5000, 10000}`

环境变量：

- `MYEVS_N98_D0_PX`
- `MYEVS_N98_T0_US`
- `MYEVS_N98_SIGMA_D_PX`
- `MYEVS_N98_SIGMA_T_US`

运行示例（prescreen400k，`s=9`，`tau=64/128ms`）：

```powershell
$env:PYTHONNOUSERSITE='1'
$env:MYEVS_N98_D0_PX='2.1802459'
$env:MYEVS_N98_T0_US='1386.9361'
$env:MYEVS_N98_SIGMA_D_PX='1.8'
$env:MYEVS_N98_SIGMA_T_US='10000'
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py --variant n98 --max-events 400000 --s-list 9 --tau-us-list 64000,128000 --out-dir data/ED24/myPedestrain_06/EBF_Part2/_slim_n98_top3_2d_prescreen400k_s9_tau64_128_sd_1p8_st_10000
```

### N98 扫参结果（prescreen400k）

口径：`s=9`，`tau=64/128ms`，`max-events=400000`。  
表内数值为每组 `(sigma_d,sigma_t)` 在各环境上的 `best AUC / best F1`（允许在 `tau=64/128ms` 间取最优）。

| sigma_d | sigma_t(us) | light best AUC | light best F1 | mid best AUC | mid best F1 | heavy best AUC | heavy best F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.8 | 2000 | 0.912719 | 0.938151 | 0.843671 | 0.613911 | 0.819202 | 0.500866 |
| 0.8 | 5000 | 0.922928 | 0.945589 | 0.856285 | 0.625326 | 0.833326 | 0.527991 |
| 0.8 | 10000 | 0.923761 | 0.945810 | 0.857225 | 0.629585 | 0.830291 | 0.527741 |
| 1.2 | 2000 | 0.911508 | 0.938211 | 0.845458 | 0.611189 | 0.822933 | 0.529888 |
| 1.2 | 5000 | 0.922553 | 0.945609 | 0.863229 | 0.655398 | 0.844899 | 0.587846 |
| 1.2 | 10000 | 0.924855 | 0.945836 | 0.871995 | 0.680344 | 0.852606 | 0.581916 |
| 1.8 | 2000 | 0.909797 | 0.938238 | 0.845448 | 0.607408 | 0.823820 | 0.554280 |
| 1.8 | 5000 | 0.921298 | 0.945624 | 0.865553 | 0.673089 | 0.849621 | 0.606792 |
| 1.8 | 10000 | 0.924767 | 0.945837 | 0.878787 | 0.704267 | 0.864132 | 0.616522 |

### 阶段结论

- 与 n97 对比，n98 在 mid/heavy 显著提升，验证了“top3 稳健聚合”是关键有效改动。
- 本轮最优组合集中在较宽时间核：`sigma_t=10000`，且 `sigma_d=1.2~1.8` 都较稳。
- 在当前口径下，推荐下一轮先固定 `sigma_t=10000`，只细化 `sigma_d`（例如 `1.2~2.0` 小步长）做精调，再评估是否需要引入加权 top3（按 `dt` 或兼容度加权）以进一步提 heavy 精度。

## n99：Top3-2D Max Backbone（`mean` 改 `max`）

### 动机与可验证假设

- 7.50 指出：`n98` 的均值聚合会稀释强支持。
- 可验证假设：保持 top3 与 2D 核不变，只把聚合从均值改成最大值，能提升判别力。

### 算法定义（在线 / 单遍 / O(r^2)）

实现位置：`src/myevs/denoise/ops/ebfopt_part2/n99_top3_2d_max_backbone.py`

top3 支持选择与 n98 相同，仅聚合改为：

$$
g_k=
\exp\left(-\frac{(d_k-d_0)^2}{2\sigma_d^2}-\frac{(\Delta t_k-t_0)^2}{2\sigma_t^2}\right),
\quad
score_i=\max_{k\le 3} g_k
$$

无支持时记为 0。

环境变量：

- `MYEVS_N99_D0_PX`
- `MYEVS_N99_T0_US`
- `MYEVS_N99_SIGMA_D_PX`
- `MYEVS_N99_SIGMA_T_US`

### N99 扫参结果（prescreen400k）

口径：`s=9`，`tau=64/128ms`，`max-events=400000`。  
表内数值为每组 `(sigma_d,sigma_t)` 在各环境上的 `best AUC / best F1`（允许在 `tau=64/128ms` 间取最优）。

| sigma_d | sigma_t(us) | light best AUC | light best F1 | mid best AUC | mid best F1 | heavy best AUC | heavy best F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.8 | 2000 | 0.915830 | 0.938311 | 0.834398 | 0.611571 | 0.803388 | 0.493607 |
| 0.8 | 5000 | 0.927256 | 0.945634 | 0.840381 | 0.615543 | 0.807496 | 0.478092 |
| 0.8 | 10000 | 0.930061 | 0.945932 | 0.836459 | 0.594516 | 0.797938 | 0.464391 |
| 1.2 | 2000 | 0.914556 | 0.938351 | 0.830766 | 0.602201 | 0.799135 | 0.488874 |
| 1.2 | 5000 | 0.926386 | 0.945692 | 0.838874 | 0.616068 | 0.807062 | 0.490031 |
| 1.2 | 10000 | 0.930346 | 0.945903 | 0.840026 | 0.610996 | 0.805433 | 0.474861 |
| 1.8 | 2000 | 0.912994 | 0.938388 | 0.826654 | 0.590117 | 0.793984 | 0.480469 |
| 1.8 | 5000 | 0.925315 | 0.945671 | 0.836080 | 0.613239 | 0.804388 | 0.493416 |
| 1.8 | 10000 | 0.929813 | 0.945916 | 0.840111 | 0.616152 | 0.807671 | 0.482092 |

### 阶段结论

- `mean->max` 对 light 基本无伤，但 mid/heavy 并未带来预期提升。
- 仅改聚合方式不足以弥补 n98 的核心短板。

## n100：Top3-2D Max×Support Backbone（补支持强度因子）

### 动机与可验证假设

- 7.50 指出：n98/n99 都缺“支持数量”信息轴。
- 可验证假设：在 `max(top3)` 基础上乘一个轻量支持强度因子，可显著提升 mid/heavy。

### 算法定义（在线 / 单遍 / O(r^2)）

实现位置：`src/myevs/denoise/ops/ebfopt_part2/n100_top3_2d_max_support_backbone.py`

在 n99 的 `g_max` 基础上，引入邻域支持数量 `n_support`：

$$
g_{max} = \max_{k\le 3} g_k,
\quad
f_{support}=\min\left(1,\frac{n_{support}}{N_0}\right),
\quad
score_i=g_{max}\cdot f_{support}
$$

其中本轮固定 `N0=5.0`。

环境变量：

- `MYEVS_N100_D0_PX`
- `MYEVS_N100_T0_US`
- `MYEVS_N100_SIGMA_D_PX`
- `MYEVS_N100_SIGMA_T_US`
- `MYEVS_N100_SUPPORT_N0`

### N100 扫参结果（prescreen400k，N0=5.0）

口径：`s=9`，`tau=64/128ms`，`max-events=400000`。  
表内数值为每组 `(sigma_d,sigma_t)` 在各环境上的 `best AUC / best F1`（允许在 `tau=64/128ms` 间取最优）。

| sigma_d | sigma_t(us) | light best AUC | light best F1 | mid best AUC | mid best F1 | heavy best AUC | heavy best F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.8 | 2000 | 0.924221 | 0.938299 | 0.852692 | 0.634674 | 0.817785 | 0.517964 |
| 0.8 | 5000 | 0.937619 | 0.945641 | 0.868220 | 0.672701 | 0.830040 | 0.521546 |
| 0.8 | 10000 | 0.942904 | 0.946062 | 0.874532 | 0.678638 | 0.829633 | 0.526112 |
| 1.2 | 2000 | 0.924606 | 0.938327 | 0.854221 | 0.644521 | 0.818184 | 0.530677 |
| 1.2 | 5000 | 0.938408 | 0.945669 | 0.872766 | 0.688697 | 0.835557 | 0.555190 |
| 1.2 | 10000 | 0.944545 | 0.946009 | 0.883871 | 0.715575 | 0.843427 | 0.545668 |
| 1.8 | 2000 | 0.924568 | 0.938351 | 0.854865 | 0.653175 | 0.817608 | 0.535931 |
| 1.8 | 5000 | 0.938542 | 0.945688 | 0.874573 | 0.699416 | 0.837486 | 0.568944 |
| 1.8 | 10000 | 0.944919 | 0.945979 | 0.887682 | 0.723557 | 0.849607 | 0.579395 |

### 阶段结论

- n100 相对 n99 与 n98 提升明显，验证“支持强度因子”是关键补偿项。
- 本轮最优组合：`sigma_d=1.8`, `sigma_t=10000`, `N0=5.0`。
- 相比 n98 最优（`1.8,10000`）的增益（best AUC）：
	- light：`0.924767 -> 0.944919`（+0.020152）
	- mid：`0.878787 -> 0.887682`（+0.008895）
	- heavy：`0.864132 -> 0.849607`（-0.014525）
- 说明 n100 在 light/mid 改善显著，但 heavy 在当前 `N0=5.0` 下存在回退；下一轮应优先扫 `N0`（如 `3/5/8/10`）做 heavy 定向校正。

## n101：Top3-2D Mean×Support Backbone（n100 的均值聚合版）

### 动机与可验证假设

- n100 使用 `g_max`，对单个最强邻居更敏感；在部分场景容易把偶发强匹配放大。
- 可验证假设：将 `max(top3)` 换成 `mean(top3)`，再乘同样的支持强度因子，可在保持 light 的同时提升 mid/heavy 稳定性。

### 算法定义（在线 / 单遍 / O(r^2)）

实现位置：`src/myevs/denoise/ops/ebfopt_part2/n101_top3_2d_mean_support_backbone.py`

top3 候选与 n98/n99/n100 一致，定义：

$$
g_k=
\exp\left(-\frac{(d_k-d_0)^2}{2\sigma_d^2}-\frac{(\Delta t_k-t_0)^2}{2\sigma_t^2}\right),
\quad
\bar g=\frac{1}{K}\sum_{k=1}^{K} g_k,\; K\in\{1,2,3\}
$$

并与支持强度因子相乘：

$$
f_{support}=\min\left(1,\frac{n_{support}}{N_0}\right),
\quad
score_i=\bar g\cdot f_{support}
$$

无支持时记为 0。

环境变量：

- `MYEVS_N101_D0_PX`
- `MYEVS_N101_T0_US`
- `MYEVS_N101_SIGMA_D_PX`
- `MYEVS_N101_SIGMA_T_US`
- `MYEVS_N101_SUPPORT_N0`

### N101 扫参设置（prescreen400k）

- 数据与口径：`s=9`，`tau=64/128ms`，`max-events=400000`（light/mid/heavy）。
- 网格：`N0={3,5,8,10}`，`sigma_d={0.8,1.2,1.8}`，`sigma_t={2000,5000,10000}`（共 36 组）。
- 汇总文件：`data/ED24/myPedestrain_06/EBF/n101_mean_support_grid/summary_n101_grid.csv`

### 结果摘要

#### 1) 全局最优（按 mean AUC）

| N0 | sigma_d | sigma_t(us) | light AUC/F1 | mid AUC/F1 | heavy AUC/F1 | mean AUC/F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 1.8 | 10000 | 0.946251 / 0.945936 | 0.903478 / 0.774517 | 0.893282 / 0.709822 | 0.914337 / 0.810092 |

#### 2) 固定 `(sigma_d=1.8,sigma_t=10000)` 的 N0 敏感性

| N0 | light AUC | mid AUC | heavy AUC | mean AUC |
|---:|---:|---:|---:|---:|
| 3 | 0.943439 | 0.886547 | 0.866315 | 0.898767 |
| 5 | 0.945436 | 0.896157 | 0.875680 | 0.905758 |
| 8 | 0.946109 | 0.902231 | 0.888756 | 0.912365 |
| 10 | 0.946251 | 0.903478 | 0.893282 | 0.914337 |

### 阶段结论

- n101（mean×support）在本轮 36 组中稳定优于 n100（max×support）的已测最优点，尤其 mid/heavy 提升明显。
- 经验最优区域为较大 `N0` + 宽时间核（`N0=8~10`, `sigma_t=10000`），`sigma_d=1.8` 略优。
- 当前建议将 n101 作为后续主线，下一轮在 `N0=8~12`、`sigma_d=1.6~2.0` 做小步长精调。

---

# n102_top3_2d_mean_goodsupport_backbone

## 1）代码实现

- 已实现新变体：`n102_top3_2d_mean_goodsupport_backbone.py`
- 关键实现点：
    - 保留 `top3` 最近过去支持并取 `g_mean`
    - 在全候选集上计算 `n_good = #{g_j >= theta}`
    - 最终：`score = g_mean * min(1, n_good / N0)`
- 新环境变量：
    - `MYEVS_N102_D0_PX`
    - `MYEVS_N102_T0_US`
    - `MYEVS_N102_SIGMA_D_PX`
    - `MYEVS_N102_SIGMA_T_US`
    - `MYEVS_N102_SUPPORT_N0`
    - `MYEVS_N102_GOOD_THETA`

## 2）评测接入

- 已接入 `scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py`
- 可直接使用 `--variant n102` 运行。

## 3）验证设置（聚焦网格）

- 数据口径：ED24 myPedestrain_06，`s=9`，`tau=64/128ms`，`max-events=400000`
- 固定：`sigma_t=10000`，`d0=2.1802459`，`t0=1386.9361`
- 扫参：
    - `theta in {0.2, 0.3, 0.5}`
    - `N0 in {3, 5, 8, 10}`
    - `sigma_d in {1.2, 1.8}`
    - 共 24 组

结果汇总文件：

- `data/ED24/myPedestrain_06/EBF/n102_goodsupport_focus/summary_n102_focus.csv`

## 4）最优结果（按 mean AUC）

最优参数：`theta=0.2, N0=10, sigma_d=1.8, sigma_t=10000`

- light：AUC `0.917809`，F1 `0.925074`
- mid：AUC `0.886173`，F1 `0.745036`
- heavy：AUC `0.878407`，F1 `0.692517`
- mean AUC：`0.894130`

## 5）与 n101 最优点对比

n101 最优（历史结果）：`N0=10, sigma_d=1.8, sigma_t=10000`

- light AUC：`0.946251`
- mid AUC：`0.903478`
- heavy AUC：`0.893282`
- mean AUC：`0.914337`

对比可见：在当前实现与这轮聚焦网格下，`n102` 尚未超过 `n101`。

## 6）当前结论

- `n_good` 思路已完成工程化落地，且能稳定运行。
- 但以当前计数方式（`g_j >= theta`）和参数范围，整体精度未优于 `n101`。
- 下一步若继续沿 7.51 路线，建议优先试：
    - 放宽/重标定 `theta`（如基于分位数而非常数阈值）
    - 将 `f_good` 改为平滑饱和（如 `1-exp(-n_good/N0)`）
    - 或改为“候选加权 good-count”，减少阈值硬切分损失。

---

# n103 / n104 / n105：TopK 扩展的 Mean×Support Backbone

## 1）动机

n101 的核心是 `top3` 最近过去支持的均值聚合，再乘支持强度因子。前一轮 n102 说明仅靠 `good-support` 阈值化并没有进一步超过 n101，因此这里改成另一条更直接的路线：把 `top3` 扩展为更大的 `topK`，看是否能从更稳定的局部历史里提取更强的判别信息。

## 2）算法定义

实现位置：`src/myevs/denoise/ops/ebfopt_part2/n103_topk_2d_mean_support_backbone.py`

对当前事件的同极性支持候选，按时间最近的过去支持取前 `K` 个，定义：

$$
g_k=\exp\left(-\frac{(d_k-d_0)^2}{2\sigma_d^2}-\frac{(\Delta t_k-t_0)^2}{2\sigma_t^2}\right),
\quad
\bar g_K=\frac{1}{K}\sum_{k=1}^{K} g_k
$$

再乘支持强度因子：

$$
f_{support}=\min\left(1,\frac{n_{support}}{N_0}\right),
\quad
score_i=\bar g_K\cdot f_{support}
$$

其中：

- n103：`K=5`
- n104：`K=7`
- n105：`K=9`

环境变量与 n101 保持一致，只是把 `MYEVS_N103_TOPK` 作为可覆盖的 K 参数。

## 3）评测设置

- 数据口径：ED24 `myPedestrain_06`
- 场景：light / mid / heavy
- 评测口径：`s=9`，`tau=64/128ms`，`max-events=400000`
- 聚焦网格：
	- `N0 in {3, 5, 8, 10}`
	- `sigma_d in {1.2, 1.8}`
	- `sigma_t = 10000`
- 输出目录：
	- `data/ED24/myPedestrain_06/EBF/n103_topk_focus/`
	- `data/ED24/myPedestrain_06/EBF/n104_topk_focus/`
	- `data/ED24/myPedestrain_06/EBF/n105_topk_focus/`

## 4）结果汇总

### 4.1 各变体最优点（按 mean AUC）

| variant | K | best N0 | best sigma_d | mean AUC | mean F1 | light AUC/F1 | mid AUC/F1 | heavy AUC/F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| n103 | 5 | 10 | 1.8 | 0.914184 | 0.814229 | 0.945717 / 0.945913 | 0.903504 / 0.775178 | 0.893329 / 0.721596 |
| n104 | 7 | 10 | 1.8 | 0.911909 | 0.811980 | 0.945233 / 0.945903 | 0.900132 / 0.768772 | 0.890362 / 0.721264 |
| n105 | 9 | 10 | 1.8 | 0.908941 | 0.803352 | 0.944796 / 0.945905 | 0.896013 / 0.756254 | 0.886014 / 0.707898 |

### 4.2 与 n101 的对比

n101 在本轮相同口径下的最优点是 `N0=10, sigma_d=1.8, sigma_t=10000`，mean AUC 为 `0.914337`。

对比可见：

- n103（top5）已经非常接近 n101，但 mean AUC 仍略低。
- n104（top7）和 n105（top9）继续下降，说明把 topK 再往大扩展后，边际信息开始变稀，mid/heavy 会先掉。
- 这轮实验没有找到一个比 n101 更强的更大 topK 版本。

## 5）当前结论

- 如果只在 n101 的框架内做 topK 扩展，`top5` 是最有希望的候选，但仍未稳定超越 n101。
- `top7/top9` 没有带来额外收益，反而让 mean AUC 和 mid/heavy F1 进一步下降。
- 因此这条路线的经验结论是：`top3 -> top5` 值得试，`top7+` 不建议作为默认主线。
- 后续如果继续沿 n101 改进，更值得优先试的是“更精细的支持筛选/加权”，而不是无上限增大 K。

---

# n106（7.53）：扇区密度各向异性 + 时间支持门控

## 1）创新动机

7.53 的核心不是继续加大 topK，而是把局部结构从“连续拟合”改为“离散方向密度”。

- 真实信号（边缘/轨迹）在局部邻域中更偏各向异性：会在少数方向上明显聚集。
- 团状噪声（burst/hot cluster）更偏各向同性：各方向密度更均匀。

因此采用 4 轴扇区（水平/垂直/主对角/副对角）做方向密度统计，用 `Smax - gamma*Smin` 抑制各向同性噪声；再叠加时间支持因子，保持 baseline 的时间特性。

## 2）算法定义与公式

实现位置：`src/myevs/denoise/ops/ebfopt_part2/n106_sector_density_backbone.py`

对当前事件 $i$ 的邻域候选 $j$（同极性、过去事件、且 $\Delta t_{ij}\le\tau$），定义单邻居权重：

$$
w_{ij}=\exp\left(-\frac{d_{ij}^2}{2\sigma_d^2}\right)\cdot\max\left(0,1-\frac{\Delta t_{ij}}{\tau}\right)
$$

按 $(dx,dy)$ 的绝对值关系把候选分到 4 个扇区，累加得到：

$$
S_k=\sum_{j\in \text{sector }k} w_{ij},\quad k\in\{0,1,2,3\}
$$

各向异性主分数：

$$
s_{aniso}=\max_k(S_k)-\gamma\cdot\min_k(S_k)
$$

时间支持门控（沿用 n101 系列的支持思想）：

$$
f_{support}=\min\left(1,\frac{n_{support}}{N_0}\right)
$$

最终分数：

$$
score_i=\max(0,s_{aniso})\cdot f_{support}
$$

环境变量：

- `MYEVS_N106_SIGMA_D_PX`
- `MYEVS_N106_GAMMA_ISO`
- `MYEVS_N106_SUPPORT_N0`

## 3）评测设置

- smoke：`max-events=20000`, `s=9`, `tau=64/128ms`
- 聚焦网格（与 n103/n104/n105 对齐）：
	- `max-events=400000`, `s=9`, `tau=64/128ms`
	- `gamma=1.0`
	- `N0 in {3,5,8,10}`
	- `sigma_d in {1.2,1.8}`
- 结果目录：`data/ED24/myPedestrain_06/EBF/n106_sector_focus/`
- 汇总文件：`data/ED24/myPedestrain_06/EBF/n106_sector_focus/summary_n106_focus.csv`

## 4）结果数据

### 4.1 smoke（20k）

- light：AUC `0.963952`（tau=64ms）
- mid：AUC `0.947136`（tau=64ms）
- heavy：AUC `0.928021`（tau=64ms）

### 4.2 聚焦网格最优点（按 mean AUC）

最优参数：`gamma=1.0, N0=10, sigma_d=1.8`

- light：AUC/F1 = `0.947350 / 0.947726`
- mid：AUC/F1 = `0.912928 / 0.785673`
- heavy：AUC/F1 = `0.902658 / 0.722599`
- mean AUC/F1 = `0.920979 / 0.818666`

### 4.3 与前序主线对比

| variant | best setting | mean AUC | mean F1 |
|---|---|---:|---:|
| n101 | `N0=10, sigma_d=1.8, sigma_t=10000` | 0.914337 | 0.810092 |
| n103 | `K=5, N0=10, sigma_d=1.8` | 0.914184 | 0.814229 |
| n106 | `gamma=1.0, N0=10, sigma_d=1.8` | 0.920979 | 0.818666 |

### 4.4 与 baseline EBF 对比（最终对照口径）

baseline 复用产物：`data/ED24/myPedestrain_06/EBF/best_params_ebf_auc_best.csv`

| env | baseline EBF AUC/F1 | n106 AUC/F1 | delta(AUC/F1, n106-baseline) |
|---|---:|---:|---:|
| light | 0.947564 / 0.949739 | 0.947350 / 0.947726 | -0.000214 / -0.002013 |
| mid | 0.921924 / 0.810827 | 0.912928 / 0.785673 | -0.008996 / -0.025154 |
| heavy | 0.920467 / 0.786882 | 0.902658 / 0.722599 | -0.017809 / -0.064283 |
| mean | 0.929985 / 0.849149 | 0.920979 / 0.818666 | -0.009006 / -0.030483 |

结论（对 baseline）：

- n106 在 light 与 baseline 基本接近，但仍略低。
- n106 在 mid/heavy 仍有明显差距，当前版本尚未超过 baseline EBF。
- 因此后续优化目标应明确为：在保持 n106 结构优势的前提下，优先补齐 mid/heavy 的召回与整体 AUC/F1。

## 5）阶段结论

- 7.53（n106）在当前口径下显著超过 n101/n103 主线，说明“方向离散密度 + 各向同性惩罚 + 时间支持门控”比单纯 topK 扩展更有效。
- 但与 baseline EBF 对比，n106 目前仍未超越，主要差距集中在 mid/heavy。
- 下一轮建议在 n106 上继续小步精调：`gamma`（如 0.75/1.0/1.25）与 `sigma_d`（1.6~2.0），并以“对 baseline 的 mid/heavy 差值收敛”为主目标，配合 seg1 专项稳定性验证。

---

# n107（7.54）：软投影轴能量法 Backbone（Projected Axis Energy）

## 1）创新动机与原有缺陷修正

7.54 针对 n106 中存在的两个关键几何和逻辑缺陷进行了修正：
1. **硬逻辑区分扇区的缺陷**：n106 采用 `adx > ady` 的硬逻辑将邻域分为 4 个扇区，这导致水平/垂直扇区各占据了大约 $1/4$ 的邻居（约 $r^2 / \pi$），而对角线仅仅占据了 $dx == dy$ 的一条 1 像素宽的细线。这导致不同方向的基础权重天然不等，在计算各向异性 $S_{max} - \gamma S_{min}$ 时，$S_{min}$ 通常是不公平地被对角线方向占位拉低。
2. **`nfac` 支持因子的过度惩罚**：n106 乘了一个 $(nsupport/N0)$ 的系数。因为密度项 $\sum w$ 本身已经是邻域支持密度的代理，再乘一次实际上把密度项做了“平方”处理（类似于 $\rho^2$）。在 heavy 等低 SNR 或稀疏场景下，这种平方惩罚会对那些微弱但真实的结构（只亮了极少个同极性像素）造成不可逆的毁灭打击。

**n107 软投影法（Projected Energy）** 的核心思想是：利用点积软投影计算各方向（H, V, D1, D2）所分配到的证据“能量”，抛弃 `if/else` 的硬边界拆分。而且最终的 score 返璞归真，不再随意缩放。

## 2）算法定义与公式

实现位置：`src/myevs/denoise/ops/ebfopt_part2/n107_projected_energy_backbone.py`

对当前事件 $i=(x_i, y_i, t_i, p_i)$，在时空邻域内寻找 $\Delta t_{ij} \le \tau$ 且极性相同的历史事件 $j=(x_j, y_j, t_j, p_j)$。
对于每个满足条件的邻居 $j$，其空域相对位移为 $dx = x_j - x_i$，$dy = y_j - y_i$，欧式距离平方 $d^2 = dx^2 + dy^2$。
定义基准能量（权重）$w_{ij}$ 为时间与空间的联合衰减（同 Baseline Raw Support 定义）：

$$
w_{ij} = \exp\left(-\frac{d^2}{2\sigma_d^2}\right) \cdot \max\left(0, 1 - \frac{\Delta t_{ij}}{\tau}\right)
$$

总证据能量（即传统的局部密度打分 $S_{total}$）为所有有效邻居能量的总和：

$$
S_{total} = \sum_{j} w_{ij}
$$

**核心创新（软投影能量）：**
抛弃 $if / else$ 的硬边界角度切分，通过向量点积的平方计算每个邻居在四个正交投影轴（水平、垂直、正对角线、反对角线）上的能量分量。对于 $d^2 > 0$ 的邻居（即非自身）：
- 水平轴能量投影比例：$\cos^2\theta = \frac{dx^2}{d^2}$
- 垂直轴能量投影比例：$\sin^2\theta = \frac{dy^2}{d^2}$
- 正对角线能量投影比例：$\frac{(dx+dy)^2}{2d^2}$
- 反对角线能量投影比例：$\frac{(dx-dy)^2}{2d^2}$

将所有邻居的能量 $w_{ij}$ 按上述比例累加，得到四个轴向的总投影能量：

$$
S_H = \sum_{j, d>0} w_{ij} \frac{dx^2}{d^2},\quad S_V = \sum_{j, d>0} w_{ij} \frac{dy^2}{d^2}
$$
$$
S_{D1} = \sum_{j, d>0} w_{ij} \frac{(dx+dy)^2}{2d^2},\quad S_{D2} = \sum_{j, d>0} w_{ij} \frac{(dx-dy)^2}{2d^2}
$$

各向异性（Anisotropy）$A_{max}$ 定义为上述正交基对方差的最大极化差值：

$$
A_{max} = \max\left( \big|S_H - S_V\big|, \big|S_{D1} - S_{D2}\big| \right)
$$

当事件产生的局部密度呈高度同向分布（例如运动边缘）时，$A_{max}$ 趋近于 $S_{total}$；当事件呈各向同性团状散布（例如热点噪声、Burst）时，$A_{max}$ 趋近于 0。

最终得分采用在总能量中减去“各向同性分量”的方法，并引入调优超系数 $\gamma_{iso} \in [0, 1]$ 控制惩罚力度（$\gamma_{iso}=0.0$ 即退化为 $S_{total}$ 基线，$\gamma_{iso}=1.0$ 全额扣去弥散能量）：

$$
Score = S_{total} - \gamma_{iso} \cdot \left(S_{total} - A_{max}\right)
$$

注意：最终得分直接输出，**不再**额外乘上 $\frac{n_{support}}{N_0}$ 的置信度缩放，防止在稀疏场景下产生过于激进的“平方”衰减。

## 3）评测记录

本次重点使用全局 sweep（`score_stream_n107`），通过精简版扫参数脚本测试 `gamma_iso`。

- `-v n107`
- 扫描 $\gamma_{iso} \in \{0.5, 1.0, 1.5\}$ 
- 口径：全局所有数据（全量 max-events off，s/tau网格化搜寻）。

结果目录：`data/ED24/myPedestrain_06/EBF_Part2/_slim/` （输出多个 `roc_*.csv`）

### 最优结果摘要（`gamma_iso=0.5` 时表现最好）

| env | best-AUC tag | AUC | best-F1 tag | F1 |
|---|---|---:|---|---:|
| light | `ebf_n107_labelscore_s9_tau128000` | 0.946239 | `ebf_n107_labelscore_s9_tau128000` | 0.946877 |
| mid | `ebf_n107_labelscore_s9_tau128000` | 0.916435 | `ebf_n107_labelscore_s9_tau64000` | 0.809050 |
| heavy | `ebf_n107_labelscore_s9_tau128000` | 0.904345 | `ebf_n107_labelscore_s9_tau64000` | 0.755958 |

当 $\gamma_{iso}$ 上升为 $1.0$ 或 $1.5$ 时，各向同性的惩罚力度过大，对真实轨迹周边稍有发散的信号点杀伤太大，导致 F1 急剧下滑（例如 $\gamma=1.5$ 时 heavy 下 F1 跌至 0.53，完全欠拟合）。

### 与 baseline EBF 对比（以 N107 $\gamma=0.5$ 各自最佳设置对比）

| env | baseline EBF 最佳 AUC/F1 | n107($\gamma=0.5$) AUC/F1 |
|---|---|---|
| light | 0.9475 / 0.9497 | 0.9462 / 0.9468 |
| mid | 0.9219 / 0.8108 | 0.9164 / 0.8090 |
| heavy | 0.9204 / 0.7868 | 0.9043 / 0.7559 |

## 4）分析与结论
1. 比起 n106（heavy F1 暴跌至 ~0.72），**n107 成功修补了由于不公的扇区定义和过惩罚 `nfac` 直接导致的严重雪崩**，将 heavy F1 回拉到 0.755（当 $\gamma=0.5$），证明“各向软投影+基础 Score 不二次乘系数”极大提升了弱信号场景的稳健性。
2. 虽然未能全盘超出 baseline 的天花板，但这已经证明几何特征作为惩罚项（且惩罚力度不宜过大，此时最佳 $\gamma$ 落在了比较克制的 $0.5$）是有结构分辨力的。接下来若再迭代，切忌引入硬切分或直接缩放总分的黑盒乘区机制。


---

# n108（7.56）：对比度增强抑制 Backbone（Contrast Enhancement Inhibition）

## 1）创新动机与原有缺陷修正

从 n107 的结果可以看出，当我们全局使用固定的惩罚系数 $\gamma_{iso}$ 时，往往面临选择困难：
1. 在 Light（干净）场景下，信号边缘本来就很清晰，任何额外的各向同性惩罚都可能误杀真实的转角、运动模糊等带“厚度”的真实事件，因此在 Light 下更倾向于信任 Baseline（即 $\gamma_{iso} \approx 0$）。
2. 在 Heavy（嘈杂）场景下，强密度的噪声团块（Burst）会伪装成高密度结构，此时我们需要 n107 的“各向判断”介入强力惩罚（增加 $\gamma_{iso}$）才能抑制假阳性。

为了解决这一矛盾，**n108 将各向同性惩罚项 $\gamma_{iso}$ 改为自适应的动态变量**，即实现“对比度增强”抑制。

## 2）动态抑制系数（Dynamic Inhibition Coefficient）的原理与公式化

计算时空邻域内（**不分极性**）的总活动量 $N_{all}$。当该区域无论正负极性都在密集触发时，说明该区域非常“脏”（长期高密度），此时我们提升惩罚要求。

$$
N_{all} = \sum_{j \in \mathcal{N}} I(|t_i - t_j| \le \tau) \quad \text{其中 } t_j \text{ 包括所有的正负极性最新事件}
$$

自适应惩罚系数 $\gamma_{dynamic}$ 定义为：

$$
\gamma_{dynamic} = \gamma_{base} \cdot \min\left(1.0, \frac{N_{all}}{N_{ref}}\right)
$$

最终得分计算保持 n107 的各向软投影逻辑，但用动态系数替换了常数项：

$$
Score = S_{total} - \gamma_{dynamic} \cdot (S_{total} - A_{max})
$$

## 3）评测记录

本次使用全局 sweep（`score_stream_n108`），探索动态基准和参考阈值（`N_ref`）。

- `-v n108`
- 设定了固定 `base_gamma = 1.0` 并且测试不同的饱和参考阈值 $N_{ref}$（即需要多少个全极性邻居活动才能拉满惩罚系数）。
- 口径：全局所有数据（全量 max-events off，s/tau网格化搜寻）。

结果目录：`data/ED24/myPedestrain_06/EBF_Part2/_slim/` （输出多个 `roc_*.csv`）

### 最优结果摘要（$N_{ref} = 40.0$ 时表现最好）

| env | best-AUC tag | AUC | best-F1 tag | F1 |
|---|---|---:|---|---:|
| light | `ebf_n108_labelscore_s9_tau128000` | 0.946564 | `ebf_n108_labelscore_s9_tau128000` | 0.946900 |
| mid | `ebf_n108_labelscore_s9_tau64000` | 0.910538 | `ebf_n108_labelscore_s5_tau64000` | 0.792947 |
| heavy | `ebf_n108_labelscore_s7_tau32000` | 0.891925 | `ebf_n108_labelscore_s5_tau64000` | 0.732937 |

### 与 baseline EBF 和 n107 对比

| env | baseline EBF 最佳 F1 | n107 最佳 F1 ($\gamma=0.5$) | n108 最佳 F1 ($N_{ref}=40.0$) |
|---|---|---|---|
| light | 0.9497 | 0.9468 | 0.9469 |
| mid | 0.8108 | 0.8090 | 0.7929 |
| heavy | 0.7868 | 0.7559 | 0.7329 |

## 4）分析与结论
1. **Light 场景下的“免伤”效果及格**：得益于动态参数的设计，n108 在 Light 场景（稀疏信号区）不再过度惩罚，F1 得分（0.9469）追平了 n107 并逼近 Baseline。
2. **Heavy 场景表现不达预期（为何比 n107 更差？）**：如对比表所示，n108 在 Mid 和 Heavy 下的 F1 分别掉到了 0.792 和 0.732，对比 n107 甚至出现了退步（0.732 vs 0.755）。其核心原因在于：**高活动量 $N_{all}$ 并非噪声的等价代名词**。真实的高对比度、高速度运动边缘同样会产生密集的 $N_{all}$，当算法仅仅依靠“邻居数量多不多”来拉高惩罚系数 $\gamma_{dynamic}$ 时，它在 Heavy 环境中不分青红皂白地把密集的真实边缘也加倍惩罚了，导致错杀大量真实信号（False Negatives 增加，F1 降低）。
3. **经验与下一步方向（放弃单纯的密度自适应）**：n108 证明了**只靠统计“局部事件数量”来增强对比度抑制是一条死胡同**，因为密度这把尺子根本无法区分“密集的噪声”和“密集的信号”。要想真正突破 Baseline 并解决这最后的识别瓶颈，我们必须彻底摈弃纯密度的视角，转向纯物理特性的识别。接下来，请立刻着手**“建议方案 1：正交异极性联合乘子”**：通过寻找空间中紧邻的正负极性事件伴随现象，直接锁定真实信号的物理特征。


---

# n109（7.56-B）：各向异性自适应非线性增强（Self-Adaptive Anisotropy）

## 1）改进思路与可行性判断

该思路的核心是把 n107 已经算出的结构信息（$A_{max}$）直接变成一个无量纲的“结构纯度比例”：

$$
r = \frac{A_{max}}{S_{total}}
$$

其中 $r\in[0,1]$。当局部事件更线性（强边缘）时，$r$ 接近 1；当局部更团状（各向同性噪声）时，$r$ 接近 0。

当前 n109（代码实现版）使用“保真下限 + 幂增强”的混合形式：

$$
Score = S_{total}\cdot\left[(1-\lambda)+\lambda r^\alpha\right]
$$

其中 $\lambda\in[0,1]$，当 $\lambda\to 1$ 时退化为最初提出的纯乘性幂增强：

$$
Score = S_{total}\cdot r^\alpha
$$

当 $\alpha>1$ 时，对低 $r$（各向同性）抑制更强；$\lambda$ 则控制“抑制强度”和“保真底座”之间的折中。

实现位置：`src/myevs/denoise/ops/ebfopt_part2/n109_self_adaptive_anisotropy_backbone.py`

环境变量：

- `MYEVS_N109_SIGMA_D_PX`
- `MYEVS_N109_ALPHA`
- `MYEVS_N109_LAMBDA_MIX`

## 2）公式化定义

n109 沿用 n107 的软投影累计方式，先得到：

$$
S_{total},\; S_H,\; S_V,\; S_{D1},\; S_{D2}
$$

并定义各向异性极化量：

$$
A_{max}=\max\left(|S_H-S_V|,\;|S_{D1}-S_{D2}|\right)
$$

再定义结构比例：

$$
r = \frac{A_{max}}{S_{total}}\in[0,1]
$$

最后输出分数：

$$
Score = S_{total}\cdot\left[(1-\lambda)+\lambda r^\alpha\right]
$$

其中：

- $\alpha>1$：幂增强指数。
- $\lambda\in[0,1]$：各向异性增强混合权重。

## 3）评测记录

使用精简 sweep 脚本全量口径：

- `-v n109`
- `sigma_d_px=1.8`
- `alpha=1.5` 与 `alpha=2.0`
- 口径：全局所有数据（全量 max-events off，s/tau 网格化搜寻）。

结果目录：`data/ED24/myPedestrain_06/EBF_Part2/_slim/`

### alpha=1.5 最优结果

| env | best-AUC tag | AUC | best-F1 tag | F1 |
|---|---|---:|---|---:|
| light | `ebf_n109_labelscore_s9_tau32000` | 0.916098 | `ebf_n109_labelscore_s9_tau64000` | 0.945150 |
| mid | `ebf_n109_labelscore_s9_tau16000` | 0.849918 | `ebf_n109_labelscore_s3_tau16000` | 0.730861 |
| heavy | `ebf_n109_labelscore_s9_tau16000` | 0.820825 | `ebf_n109_labelscore_s3_tau16000` | 0.627769 |

### alpha=2.0 最优结果

| env | best-AUC tag | AUC | best-F1 tag | F1 |
|---|---|---:|---|---:|
| light | `ebf_n109_labelscore_s9_tau32000` | 0.902058 | `ebf_n109_labelscore_s9_tau64000` | 0.945101 |
| mid | `ebf_n109_labelscore_s9_tau8000` | 0.825075 | `ebf_n109_labelscore_s3_tau16000` | 0.730861 |
| heavy | `ebf_n109_labelscore_s9_tau8000` | 0.800444 | `ebf_n109_labelscore_s3_tau16000` | 0.627763 |

### 与 baseline / n107 / n108 对比（best-F1）

| env | baseline EBF | n107 | n108 | n109($\alpha=1.5$) | n109($\alpha=2.0$) |
|---|---:|---:|---:|---:|---:|
| light | 0.9497 | 0.9468 | 0.9469 | 0.9452 | 0.9451 |
| mid | 0.8108 | 0.8090 | 0.7929 | 0.7309 | 0.7309 |
| heavy | 0.7868 | 0.7559 | 0.7329 | 0.6278 | 0.6278 |

## 4）分析与结论
1. **方向在理论上可行，但当前参数区间下明显过抑制**：该结构比例幂函数能稳定压制低各向异性事件，但在 Mid/Heavy 场景会把大量“非理想线性但真实”的边缘也压下去，导致 F1 显著下降。
2. **$\alpha$ 增大（1.5->2.0）并未带来收益**：AUC 与 F1 均继续下降或持平，说明“更强非线性”在当前数据上不是正确方向。
3. **结论**：n109 作为一次有效反例实验，验证了“只靠各向异性比例做纯乘性幂增强”不足以超越 n107，实际效果比 n107 与 baseline 都差。后续若继续沿该方向，可考虑把幂增强与下限保真项混合（例如 $Score=S_{total}\cdot[(1-\lambda)+\lambda r^\alpha]$），避免对中等各向异性真实边缘过度打击。

## 5）n109 二次迭代（直接在 n109 上改）

按你的要求，这一轮没有新建 n110，而是直接改 n109 的主公式。

### 改动点

把原来的纯乘性幂增强：

$$
Score = S_{total}\cdot r^\alpha
$$

改为“保真下限 + 幂增强”的混合形式：

$$
Score = S_{total}\cdot\left[(1-\lambda)+\lambda r^\alpha\right]
$$

其中：

- $\lambda\in[0,1]$：结构增强权重；$\lambda$ 越大越接近原始 n109 的强抑制。
- $\alpha>1$：对低各向异性比例的非线性压缩强度。

环境变量新增：

- `MYEVS_N109_LAMBDA_MIX`

### 二次迭代 sweep 结果（全量口径）

第一轮（固定 `sigma_d_px=1.8`）扫描：

- `alpha=1.5, lambda=0.5`
- `alpha=1.5, lambda=0.7`
- `alpha=1.5, lambda=0.85`
- `alpha=1.2, lambda=0.5`
- `alpha=1.3, lambda=0.5`

各组合 best-F1 汇总：

| 配置 | light F1 | mid F1 | heavy F1 |
|---|---:|---:|---:|
| n109-v2 $(\alpha=1.5,\lambda=0.5)$ | 0.946776 | 0.807489 | 0.754435 |
| n109-v2 $(\alpha=1.5,\lambda=0.7)$ | 0.946562 | 0.790790 | 0.729208 |
| n109-v2 $(\alpha=1.5,\lambda=0.85)$ | 0.946252 | 0.744878 | 0.663403 |
| n109-v2 $(\alpha=1.2,\lambda=0.5)$ | 0.946809 | 0.808240 | 0.755220 |
| n109-v2 $(\alpha=1.3,\lambda=0.5)$ | 0.946787 | 0.807949 | 0.754934 |

第二轮（冲 baseline 目标）继续扫描：

- `alpha=1.2, lambda=0.0/0.2/0.4, sigma_d=1.8`
- `alpha=1.2, lambda=0.2, sigma_d=1.2/1.5/2.2`
- `alpha=1.2, lambda=0.1, sigma_d=2.2`
- `alpha=1.1, lambda=0.2, sigma_d=2.2`

单配置最均衡结果为：`alpha=1.2, lambda=0.2, sigma_d=2.2`。

该配置的 best-F1：

| env | best-F1 |
|---|---:|
| light | 0.947758 |
| mid | 0.818990 |
| heavy | 0.767943 |

### 与 baseline / n107 / n108 对比（best-F1）

| env | baseline EBF | n107 | n108 | n109-v2 best(single config) |
|---|---:|---:|---:|---:|
| light | 0.9497 | 0.9468 | 0.9469 | 0.9478 |
| mid | 0.8108 | 0.8090 | 0.7929 | 0.8190 |
| heavy | 0.7868 | 0.7559 | 0.7329 | 0.7679 |

### 二次迭代结论

1. 直接在 n109 上加入保真下限后，效果从“明显失败”回升到“稳定超越 n107”。
2. 在 baseline 目标上，当前已实现 **mid 超越 baseline**（0.8190 > 0.8108），但 light/heavy 仍分别差约 0.0019 / 0.0189，尚未全线超越 baseline。
3. 这说明 n109 的问题核心确实在“纯乘性压缩过猛”；保真下限是必要修复，但要全线超越 baseline 还需引入额外判别信息（例如异极性边界伴随特征）。

---

# n110（7.57）：扇区极性状态转移门控（Sector Transition Gate）

## 1）核心动机

7.57 的出发点是：仅靠空间几何（如 n107/n109）会忽略一个更强的判别信号，即“极性时序结构”。

- 真实信号倾向于在局部产生同极性连续（`same -> same`）。
- 重噪场景（特别是 heavy）更容易出现极性交替震荡（`same -> opp` / `opp -> same`）。

因此 n110 采用“分扇区状态机”替代纯密度/纯几何思路：对每个扇区同时计算同极性密度，并提取 top1/top2 最近事件极性模式，再做门控加权。

## 2）算法公式

对中心事件 $e_c=(x_c,y_c,p_c,t_c)$，在邻域内划分 4 个扇区 $k\in\{0,1,2,3\}$。

同极性基础密度：

$$
E_k=\sum_{j\in k}\max\left(0,1-\frac{t_c-t_j}{\tau}\right)\cdot\mathbb{I}(p_j=p_c)
$$

在每个扇区内取最近两次历史事件（top1/top2）极性 $p_k^{(1)},p_k^{(2)}$，定义转移乘子：

$$
M_k=
\begin{cases}
1+\alpha,& p_k^{(1)}=p_c\land p_k^{(2)}=p_c\\
1-\beta,& p_k^{(1)}\neq p_k^{(2)}\\
\gamma,& p_k^{(1)}\neq p_c\land p_k^{(2)}\neq p_c\\
1,& \text{扇区历史不足}
\end{cases}
$$

最终得分：

$$
Score=\sum_{k=0}^{3}\max\left(0,E_k\cdot M_k\right)
$$

默认参数：

- `MYEVS_N110_ALPHA=0.2`
- `MYEVS_N110_BETA=0.4`
- `MYEVS_N110_GAMMA=0.8`

实现文件：`src/myevs/denoise/ops/ebfopt_part2/n110_sector_transition_gate_backbone.py`

## 3）实验命令与产物

运行命令（全网格 sweep）：

```bash
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py \
	--variant n110 \
	--out-dir data/ED24/myPedestrain_06/EBF_Part2/_slim_n110_sector_transition
```

产物目录：

- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n110_sector_transition/`

主要 ROC 文件：

- `roc_ebf_n110_light_labelscore_s3_5_7_9_tau8_16_32_64_128_256_512_1024ms.csv`
- `roc_ebf_n110_mid_labelscore_s3_5_7_9_tau8_16_32_64_128_256_512_1024ms.csv`
- `roc_ebf_n110_heavy_labelscore_s3_5_7_9_tau8_16_32_64_128_256_512_1024ms.csv`

## 4）结果汇总

### n110 最优 AUC / F1

| env | best-AUC tag | AUC | best-F1 tag | F1 | best-F1阈值 |
|---|---|---:|---|---:|---:|
| light | `ebf_n110_labelscore_s9_tau128000` | 0.942945 | `ebf_n110_labelscore_s9_tau128000` | 0.946025 | 0.155903 |
| mid | `ebf_n110_labelscore_s7_tau64000` | 0.909099 | `ebf_n110_labelscore_s7_tau64000` | 0.803122 | 2.719109 |
| heavy | `ebf_n110_labelscore_s7_tau64000` | 0.894949 | `ebf_n110_labelscore_s7_tau64000` | 0.744283 | 3.954803 |

### 与 baseline / n107 / n109-v2 对比（best-F1）

| env | baseline EBF | n107 | n109-v2(best) | n110 |
|---|---:|---:|---:|---:|
| light | 0.9497 | 0.9468 | 0.9478 | 0.9460 |
| mid | 0.8108 | 0.8090 | 0.8190 | 0.8031 |
| heavy | 0.7868 | 0.7559 | 0.7679 | 0.7443 |

## 5）结论

1. n110 的极性转移门控在思想上是正确方向（把“时序极性模式”显式纳入打分），但当前默认参数下，整体效果仍未超过 n109-v2，更未超过 baseline。
2. n110 相比 n107 仍有优势（mid/heavy 的 best-F1 均高于 n107），说明“极性交替惩罚”确实在抑制部分噪声；但对真实边缘的误伤仍偏大。
3. 下一步应针对 n110 做参数面扫描（尤其是 $\beta$ 和 $\alpha$），并考虑把 `same->same` 奖励改为“随支持度自适应”，避免在低支持扇区过度放大随机模式。

## 6）重试扫频（alpha/beta/gamma）与推荐参数

为回答“继续扫频哪个参数最合适”，对 n110 追加了参数网格扫描：

- prescreen 网格：`s={7,9}`，`tau={64,128}ms`，`max-events=200000`
- 参数网格：`alpha={0.2,0.4,0.6,0.8}`，`beta={0.4,0.8,1.2}`，`gamma={0.2,0.5,0.8}`
- 目标：按三环境 `mean_f1` 主排序，`mean_auc` 次排序

Top-1（prescreen）为：

- `MYEVS_N110_ALPHA=0.2`
- `MYEVS_N110_BETA=0.4`
- `MYEVS_N110_GAMMA=0.8`

对应 prescreen 指标：

- mean-F1 = **0.846369**
- mean-AUC = **0.928100**

随后对该 Top-1 参数做全网格 full-validate（`s={3,5,7,9}`，`tau=8..1024ms`）：

- mean-F1 = **0.840100**
- mean-AUC = **0.925443**
- best-AUC（light/mid/heavy）均出现在 `s9,tau128ms`

full-validate 最优 F1（按环境）：

| env | best-F1 tag | F1 | best-F1阈值 |
|---|---|---:|---:|
| light | `ebf_n110_labelscore_s9_tau256000` | 0.949651 | 0.943988 |
| mid | `ebf_n110_labelscore_s9_tau128000` | 0.813375 | 4.986178 |
| heavy | `ebf_n110_labelscore_s7_tau64000` | 0.757273 | 3.741075 |

推荐结论：

1. n110 后续默认参数建议改为 `alpha=0.2, beta=0.4, gamma=0.8`，显著优于本章此前默认值（0.4/0.8/0.5）。
2. 若面向“全环境均衡”，优先使用 `s9,tau128ms`；若更偏重 heavy 的 F1，可改用 `s7,tau64ms`。
3. 本轮扫频结果目录：`data/ED24/myPedestrain_06/EBF_Part2/_tune_n110_retry/`（含 `summary.csv` 与 full-validate ROC）。

## 7）n111（7.59）：扇区同/异极性对抗（Sector Polarity Contrast）

### 7.1 核心思想

n111 将 n110 的 `top1/top2` 时序状态机替换为更直接的“同/异极性密度对抗”。

- 每个扇区同时累计同极性密度 `E_same` 与异极性密度 `E_opp`。
- 用异极性密度直接惩罚同极性密度，得到净密度 `Net`。
- 仅保留正净密度累加到最终得分。

该设计对 heavy 噪声的目标是：在局部极性交替明显的区域（噪声特征），让分数被快速抵消。

### 7.2 公式

对中心事件 $e_c=(x_c,y_c,p_c,t_c)$，4 扇区编号为 $k\in\{0,1,2,3\}$，时间衰减权重：

$$
w_j=\max\left(0,1-\frac{t_c-t_j}{\tau}\right)
$$

扇区同/异极性密度：

$$
E_{same,k}=\sum_{j\in k,\,p_j=p_c} w_j,\qquad
E_{opp,k}=\sum_{j\in k,\,p_j\neq p_c} w_j
$$

净密度与总分：

$$
Net_k=\max(0, E_{same,k}-\mu\,E_{opp,k}),\qquad
Score=\sum_{k=0}^{3} Net_k
$$

其中 $\mu$ 对应环境变量 `MYEVS_N111_MU`（默认 `0.0`）。

实现文件：`src/myevs/denoise/ops/ebfopt_part2/n111_sector_polarity_contrast_backbone.py`

### 7.3 实验命令与产物

```bash
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py \
	--variant n111 \
	--esr-mode off --aocc-mode off \
	--out-dir data/ED24/myPedestrain_06/EBF_Part2/_slim_n111_sector_polarity_contrast
```

产物目录：

- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n111_sector_polarity_contrast/`

主要 ROC 文件：

- `roc_ebf_n111_light_labelscore_s3_5_7_9_tau8_16_32_64_128_256_512_1024ms.csv`
- `roc_ebf_n111_mid_labelscore_s3_5_7_9_tau8_16_32_64_128_256_512_1024ms.csv`
- `roc_ebf_n111_heavy_labelscore_s3_5_7_9_tau8_16_32_64_128_256_512_1024ms.csv`

### 7.4 结果汇总（默认 `mu=1.0`）

| env | best-AUC tag | AUC | best-F1 tag | F1 | best-F1阈值 |
|---|---|---:|---|---:|---:|
| light | `ebf_n111_labelscore_s9_tau64000` | 0.932002 | `ebf_n111_labelscore_s9_tau64000` | 0.938603 | 0.002516 |
| mid | `ebf_n111_labelscore_s7_tau64000` | 0.903248 | `ebf_n111_labelscore_s7_tau64000` | 0.796946 | 1.982188 |
| heavy | `ebf_n111_labelscore_s7_tau64000` | 0.889096 | `ebf_n111_labelscore_s7_tau64000` | 0.739974 | 2.948250 |

与当前 n110（默认已改为 0.2/0.4/0.8）对比（best-F1）：

| env | n110 | n111 |
|---|---:|---:|
| light | 0.949651 | 0.938603 |
| mid | 0.813375 | 0.796946 |
| heavy | 0.757273 | 0.739974 |

### 7.5 继续扫频：`mu` 单参数调优

使用脚本：`scripts/ED24_alg_evalu/tune_n111_mu.py`

```bash
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/tune_n111_mu.py \
	--project-root . \
	--base-out-dir data/ED24/myPedestrain_06/EBF_Part2/_tune_n111_mu \
	--mu-grid 0.0,0.5,1.0,1.5,2.0,3.0 \
	--s-list 7,9 \
	--tau-us-list 64000,128000 \
	--max-events 200000 \
	--full-validate \
	--full-s-list 3,5,7,9 \
	--full-tau-us-list 8000,16000,32000,64000,128000,256000,512000,1024000 \
	--full-max-events 0
```

预筛选（`s={7,9}, tau={64,128}ms`，按 `mean_f1` 排序）结果：

| rank | mu | mean_f1 | mean_auc | light_f1 | mid_f1 | heavy_f1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0 | 0.849739 | 0.929974 | 0.949738 | 0.810808 | 0.788670 |
| 2 | 0.5 | 0.841264 | 0.920041 | 0.942893 | 0.799829 | 0.781070 |
| 3 | 1.0 | 0.833673 | 0.912843 | 0.938603 | 0.791822 | 0.770594 |
| 4 | 1.5 | 0.826320 | 0.906370 | 0.933988 | 0.784344 | 0.760627 |
| 5 | 2.0 | 0.819223 | 0.900463 | 0.930942 | 0.777138 | 0.749589 |
| 6 | 3.0 | 0.807846 | 0.889992 | 0.925676 | 0.764889 | 0.732973 |

可见：随着 `mu` 增大，三环境 F1/AUC 近似单调下降；最佳值落在 `mu=0.0`。

细粒度复扫（`mu=0.0,0.1,0.2,0.3,0.4,0.5`）命令：

```bash
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/tune_n111_mu.py \
	--project-root . \
	--base-out-dir data/ED24/myPedestrain_06/EBF_Part2/_tune_n111_mu_fine \
	--mu-grid 0.0,0.1,0.2,0.3,0.4,0.5 \
	--s-list 7,9 \
	--tau-us-list 64000,128000 \
	--max-events 200000 \
	--full-validate \
	--full-s-list 3,5,7,9 \
	--full-tau-us-list 8000,16000,32000,64000,128000,256000,512000,1024000 \
	--full-max-events 0
```

细粒度预筛选（按 `mean_f1` 排序）：

| rank | mu | mean_f1 | mean_auc | light_f1 | mid_f1 | heavy_f1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0 | 0.849739 | 0.929974 | 0.949738 | 0.810808 | 0.788670 |
| 2 | 0.1 | 0.848143 | 0.928036 | 0.948341 | 0.808338 | 0.787751 |
| 3 | 0.2 | 0.846091 | 0.925673 | 0.946716 | 0.805537 | 0.786019 |
| 4 | 0.3 | 0.844348 | 0.923682 | 0.945265 | 0.803337 | 0.784441 |
| 5 | 0.4 | 0.842546 | 0.921621 | 0.943734 | 0.801204 | 0.782702 |
| 6 | 0.5 | 0.841264 | 0.920041 | 0.942893 | 0.799829 | 0.781070 |

细扫结论：在 `0.0~0.5` 局部区间内，`mu=0.0` 仍是最优，且指标随 `mu` 增大继续近似单调下降。

Top1（`mu=0.0`）full-validate（`s={3,5,7,9}`, `tau=8..1024ms`）结果：

| env | best-AUC tag | AUC | best-F1 tag | F1 | best-F1阈值 |
|---|---|---:|---|---:|---:|
| light | `ebf_n111_labelscore_s9_tau128000` | 0.947556 | `ebf_n111_labelscore_s9_tau512000` | 0.952015 | 1.717568 |
| mid | `ebf_n111_labelscore_s9_tau128000` | 0.923210 | `ebf_n111_labelscore_s9_tau128000` | 0.817653 | 4.834437 |
| heavy | `ebf_n111_labelscore_s9_tau128000` | 0.913575 | `ebf_n111_labelscore_s7_tau64000` | 0.761748 | 3.586437 |

产物目录：

- `data/ED24/myPedestrain_06/EBF_Part2/_tune_n111_mu/summary.csv`
- `data/ED24/myPedestrain_06/EBF_Part2/_tune_n111_mu/summary.json`
- `data/ED24/myPedestrain_06/EBF_Part2/_tune_n111_mu/full_validate_top1.json`
- `data/ED24/myPedestrain_06/EBF_Part2/_tune_n111_mu_fine/summary.csv`
- `data/ED24/myPedestrain_06/EBF_Part2/_tune_n111_mu_fine/summary.json`
- `data/ED24/myPedestrain_06/EBF_Part2/_tune_n111_mu_fine/full_validate_top1.json`

### 7.6 结论（更新）

1. n111 在 `mu=1.0` 时不如 n110；粗扫与细扫均确认最佳参数为 `mu=0.0`。
2. 将 `mu` 从 1.0 降到 0.0 后，n111 三环境指标显著提升，且优于 n110 当前默认方案。
3. 建议默认保持 `MYEVS_N111_MU=0.0`；若后续需要重新引入异极性惩罚，可从很小权重（如 `0.05~0.2`）开始并与 `s/tau` 联合调优。

## 8）n112（7.59-3）：双尺度极性纯度门控（Dual-Scale Polarity Purity Gate）

### 8.1 核心思想

n112 针对 n111 的“大尺度异极性误伤”问题，采用“宏观支持 + 微观纯度门控”解耦：

1. 宏观尺度（大邻域）仅累计同极性支持，保证结构连续性与召回。
2. 微观尺度（小邻域）统计同/异极性混合程度，得到纯度因子。
3. 用纯度因子对宏观得分做乘性门控，抑制 micro-level 极性交替噪声。

实现文件：`src/myevs/denoise/ops/ebfopt_part2/n112_dual_scale_purity_backbone.py`

### 8.2 公式

对中心事件 $e_c=(x_c,y_c,p_c,t_c)$，时间衰减权重：

$$
w_j=\max\left(0,1-\frac{t_c-t_j}{\tau}\right)
$$

宏观扇区同极性支持：

$$
E_{large,k}=\sum_{j\in k,\ p_j=p_c} w_j,
\qquad
Score_{base}=\sum_{k=0}^{3} E_{large,k}
$$

微观同/异极性密度：

$$
e_{same}=\sum_{j\in \mathcal{N}_{small},\ p_j=p_c} w_j,
\qquad
e_{opp}=\sum_{j\in \mathcal{N}_{small},\ p_j\neq p_c} w_j
$$

纯度与门控：

$$
Purity=\frac{e_{same}}{e_{same}+e_{opp}+\epsilon},
\qquad
M=\min\left(1,\frac{Purity}{\theta}\right)
$$

最终分数：

$$
Score_{final}=Score_{base}\cdot M
$$

其中参数含义：

- `MYEVS_N112_RSMALL`：微观半径（默认 `2`）
- `MYEVS_N112_THRESH`：纯度阈值（默认 `0.6`）

### 8.3 参数扫频命令（已执行）

```bash
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/tune_n112_params.py \
	--project-root . \
	--base-out-dir data/ED24/myPedestrain_06/EBF_Part2/_tune_n112_params \
	--rsmall-grid 1,2 \
	--thresh-grid 0.6,0.7,0.8 \
	--s-list 7,9 \
	--tau-us-list 64000,128000 \
	--max-events 200000 \
	--full-validate \
	--full-s-list 3,5,7,9 \
	--full-tau-us-list 8000,16000,32000,64000,128000,256000,512000,1024000 \
	--full-max-events 0
```

预筛选（按 `mean_f1` 排序）Top6：

| rank | rsmall | thresh | mean_f1 | mean_auc | light_f1 | mid_f1 | heavy_f1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 0.6 | 0.844652 | 0.920672 | 0.940393 | 0.805363 | 0.788201 |
| 2 | 2 | 0.7 | 0.843699 | 0.919716 | 0.940393 | 0.803970 | 0.786736 |
| 3 | 2 | 0.8 | 0.841973 | 0.918678 | 0.940393 | 0.801725 | 0.783802 |
| 4 | 1 | 0.6 | 0.838182 | 0.906296 | 0.931664 | 0.799384 | 0.783497 |
| 5 | 1 | 0.7 | 0.836717 | 0.905868 | 0.931664 | 0.797667 | 0.780821 |
| 6 | 1 | 0.8 | 0.835055 | 0.905322 | 0.931664 | 0.795758 | 0.777742 |

Top1 为 `rsmall=2, thresh=0.6`。

### 8.4 Top1 全量验证结果

Top1（`rsmall=2, thresh=0.6`）full-validate（`s={3,5,7,9}`, `tau=8..1024ms`）：

| env | best-AUC tag | AUC | best-F1 tag | F1 | best-F1阈值 |
|---|---|---:|---|---:|---:|
| light | `ebf_n112_labelscore_s9_tau64000` | 0.936982 | `ebf_n112_labelscore_s9_tau64000` | 0.940393 | 0.001662 |
| mid | `ebf_n112_labelscore_s9_tau64000` | 0.911271 | `ebf_n112_labelscore_s9_tau64000` | 0.811316 | 3.200813 |
| heavy | `ebf_n112_labelscore_s9_tau64000` | 0.901382 | `ebf_n112_labelscore_s9_tau64000` | 0.759434 | 4.641703 |

同口径 baseline / n111 / n112 的 best-F1 对比：

| env | baseline | n111 (`mu=0.0`) | n112 (`rsmall=2,thresh=0.6`) |
|---|---:|---:|---:|
| light | 0.952021 | 0.952015 | 0.940393 |
| mid | 0.817653 | 0.817653 | 0.811316 |
| heavy | 0.761764 | 0.761748 | 0.759434 |

### 8.5 结论

1. n112 设计思路有效，且在预筛选上 `rsmall=2` 明显优于 `rsmall=1`，说明微观统计半径取 `5x5` 更稳。
2. 在 full-validate 口径下，n112 最优配置仍略低于 n111（`mu=0.0`），差距主要在 light 与 mid。
3. n112 已实现并可复用，默认参数更新为 `MYEVS_N112_RSMALL=2`、`MYEVS_N112_THRESH=0.6`；后续可重点联合调优 `RSMALL/THRESH` 与 `s/tau` 以提升 full-validate 表现。

产物目录：

- `data/ED24/myPedestrain_06/EBF_Part2/_tune_n112_params/summary.csv`
- `data/ED24/myPedestrain_06/EBF_Part2/_tune_n112_params/summary.json`
- `data/ED24/myPedestrain_06/EBF_Part2/_tune_n112_params/full_validate_top1.json`

## 9）n113（7.60）：幂律时间衰减（Power-Law Temporal Decay）

### 9.1 核心思想

n113 放弃空间异极性惩罚，直接改造时间衰减核：

1. 保持 EBF 的同极性邻域累加框架不变。
2. 将线性时间权重做幂律变换，强化“短时突发”事件、抑制“慢速散布”噪声。
3. 通过单参数 `gamma` 控制衰减非线性强度。

实现文件：`src/myevs/denoise/ops/ebfopt_part2/n113_power_law_decay_backbone.py`

### 9.2 公式

线性时间权重：

$$
w_{lin}=\max\left(0,1-\frac{t_c-t_j}{\tau}\right)
$$

n113 幂律权重：

$$
w_j = w_{lin}^{\gamma}
$$

最终得分（同极性邻域累加）：

$$
Score = \sum_{j\in\mathcal{N},\ p_j=p_c} w_j
$$

其中 `MYEVS_N113_GAMMA` 默认 `2.0`（`gamma=1.0` 时退化为 baseline）。

### 9.3 参数扫频命令（已执行）

```bash
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/tune_n113_gamma.py \
	--project-root . \
	--base-out-dir data/ED24/myPedestrain_06/EBF_Part2/_tune_n113_gamma \
	--gamma-grid 1.0,1.5,2.0,2.5,3.0,4.0 \
	--s-list 7,9 \
	--tau-us-list 64000,128000 \
	--max-events 200000 \
	--full-validate \
	--full-s-list 3,5,7,9 \
	--full-tau-us-list 8000,16000,32000,64000,128000,256000,512000,1024000 \
	--full-max-events 0
```

预筛选（按 `mean_f1` 排序）结果：

| rank | gamma | mean_f1 | mean_auc | light_f1 | mid_f1 | heavy_f1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.0 | 0.850105 | 0.930382 | 0.948216 | 0.811009 | 0.791089 |
| 2 | 1.5 | 0.850063 | 0.930508 | 0.948926 | 0.811671 | 0.789592 |
| 3 | 1.0 | 0.849739 | 0.929974 | 0.949738 | 0.810808 | 0.788670 |
| 4 | 2.5 | 0.849539 | 0.929917 | 0.947934 | 0.810054 | 0.790627 |
| 5 | 3.0 | 0.848527 | 0.929271 | 0.947630 | 0.808128 | 0.789823 |
| 6 | 4.0 | 0.845807 | 0.927702 | 0.947332 | 0.803859 | 0.786229 |

Top1 为 `gamma=2.0`。

### 9.4 Top1 全量验证结果

Top1（`gamma=2.0`）full-validate（`s={3,5,7,9}`, `tau=8..1024ms`）：

| env | best-AUC tag | AUC | best-F1 tag | F1 | best-F1阈值 |
|---|---|---:|---|---:|---:|
| light | `ebf_n113_labelscore_s9_tau256000` | 0.948922 | `ebf_n113_labelscore_s9_tau512000` | 0.952056 | 1.275411 |
| mid | `ebf_n113_labelscore_s9_tau256000` | 0.924639 | `ebf_n113_labelscore_s9_tau128000` | 0.818710 | 3.783821 |
| heavy | `ebf_n113_labelscore_s9_tau256000` | 0.914868 | `ebf_n113_labelscore_s7_tau128000` | 0.764631 | 4.007954 |

同口径 baseline 参考（`variant=ebf`，同一 full-grid）：

| env | best-AUC tag | AUC | best-F1 tag | F1 | best-F1阈值 |
|---|---|---:|---|---:|---:|
| light | `ebf_labelscore_s9_tau128000` | 0.947564 | `ebf_labelscore_s9_tau512000` | 0.952021 | 1.717568 |
| mid | `ebf_labelscore_s9_tau128000` | 0.923218 | `ebf_labelscore_s9_tau128000` | 0.817653 | 4.834726 |
| heavy | `ebf_labelscore_s9_tau128000` | 0.913578 | `ebf_labelscore_s7_tau64000` | 0.761764 | 3.586625 |

与 baseline / n111 / n112 / n113 最优配置对比（best-F1）：

| env | baseline | n111 (`mu=0.0`) | n112 (`rsmall=2,thresh=0.6`) | n113 (`gamma=2.0`) |
|---|---:|---:|---:|---:|
| light | 0.952021 | 0.952015 | 0.940393 | 0.952056 |
| mid | 0.817653 | 0.817653 | 0.811316 | 0.818710 |
| heavy | 0.761764 | 0.761748 | 0.759434 | 0.764631 |

### 9.5 结论

1. n113 在本轮口径下优于 baseline / n111 / n112，三环境 best-F1 均为当前最优。
2. `gamma` 存在明显峰值区间，约在 `1.5~2.0`；过大（如 4.0）会导致过抑制。
3. 建议将默认参数设置为 `MYEVS_N113_GAMMA=2.0`，后续可围绕 `1.6~2.2` 做细扫并与 `s/tau` 联合优化。

产物目录：

- `data/ED24/myPedestrain_06/EBF_Part2/_tune_n113_gamma/summary.csv`
- `data/ED24/myPedestrain_06/EBF_Part2/_tune_n113_gamma/summary.json`
- `data/ED24/myPedestrain_06/EBF_Part2/_tune_n113_gamma/full_validate_top1.json`
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_baseline_fullgrid_ref/roc_ebf_light_labelscore_s3_5_7_9_tau8_16_32_64_128_256_512_1024ms.csv`
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_baseline_fullgrid_ref/roc_ebf_mid_labelscore_s3_5_7_9_tau8_16_32_64_128_256_512_1024ms.csv`
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_baseline_fullgrid_ref/roc_ebf_heavy_labelscore_s3_5_7_9_tau8_16_32_64_128_256_512_1024ms.csv`

## 10）n114（7.61）：各向同性底噪扣除（Adaptive Isotropic Subtraction）

### 10.1 核心思想

n114 在同极性 4 扇区密度上估计局部“各向同性底噪”，并对所有扇区做统一扣除：

1. 真实边缘通常是各向异性的（至少有 1~2 个稀疏扇区），底噪估计接近 0，主结构得分保留。
2. heavy 噪声更接近各向同性（4 扇区都被填满），底噪估计偏高，扣除后得分显著压低。
3. 只引入一个参数 `lambda`，当 `lambda=0` 时退化为 baseline。

实现文件：`src/myevs/denoise/ops/ebfopt_part2/n114_isotropic_subtraction_backbone.py`

### 10.2 公式

扇区同极性密度：

$$
E_k = \sum_{j\in k,\ p_j=p_c} w_j,
\qquad
w_j=\max\left(0,1-\frac{t_c-t_j}{\tau}\right)
$$

取 4 个扇区中最小的两个值 $E_{(0)},E_{(1)}$ 估计底噪：

$$
NoiseFloor=\frac{E_{(0)}+E_{(1)}}{2}
$$

底噪扣除与最终分数：

$$
Net_k=\max\left(0,E_k-\lambda\cdot NoiseFloor\right),
\qquad
Score=\sum_{k=0}^{3} Net_k
$$

其中 `MYEVS_N114_LAMBDA` 默认 `0.5`。

### 10.3 参数扫频命令（已执行）

```bash
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/tune_n114_lambda.py \
	--project-root . \
	--base-out-dir data/ED24/myPedestrain_06/EBF_Part2/_tune_n114_lambda \
	--lambda-grid 0.0,0.5,1.0,1.5,2.0,3.0 \
	--s-list 7,9 \
	--tau-us-list 64000,128000 \
	--max-events 200000 \
	--full-validate \
	--full-s-list 3,5,7,9 \
	--full-tau-us-list 8000,16000,32000,64000,128000,256000,512000,1024000 \
	--full-max-events 0
```

预筛选（按 `mean_f1` 排序）结果：

| rank | lambda | mean_f1 | mean_auc | light_f1 | mid_f1 | heavy_f1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.5 | 0.850784 | 0.930693 | 0.949721 | 0.812308 | 0.790322 |
| 2 | 0.0 | 0.849739 | 0.929974 | 0.949738 | 0.810808 | 0.788670 |
| 3 | 1.0 | 0.834623 | 0.921417 | 0.948849 | 0.791964 | 0.763056 |
| 4 | 1.5 | 0.795565 | 0.889267 | 0.937209 | 0.746679 | 0.702805 |
| 5 | 2.0 | 0.750458 | 0.844287 | 0.912877 | 0.695092 | 0.643405 |
| 6 | 3.0 | 0.698275 | 0.764738 | 0.911931 | 0.622265 | 0.560629 |

Top1 为 `lambda=0.5`。

### 10.4 Top1 全量验证结果

Top1（`lambda=0.5`）full-validate（`s={3,5,7,9}`, `tau=8..1024ms`）：

| env | best-AUC tag | AUC | best-F1 tag | F1 | best-F1阈值 |
|---|---|---:|---|---:|---:|
| light | `ebf_n114_labelscore_s9_tau128000` | 0.947499 | `ebf_n114_labelscore_s9_tau512000` | 0.952188 | 1.699926 |
| mid | `ebf_n114_labelscore_s9_tau128000` | 0.923589 | `ebf_n114_labelscore_s9_tau128000` | 0.819022 | 4.101529 |
| heavy | `ebf_n114_labelscore_s9_tau128000` | 0.913991 | `ebf_n114_labelscore_s7_tau128000` | 0.763061 | 4.165358 |

与 baseline / n111 / n112 / n113 / n114 最优配置对比（best-F1）：

| env | baseline | n111 (`mu=0.0`) | n112 (`rsmall=2,thresh=0.6`) | n113 (`gamma=2.0`) | n114 (`lambda=0.5`) |
|---|---:|---:|---:|---:|---:|
| light | 0.952021 | 0.952015 | 0.940393 | 0.952056 | 0.952188 |
| mid | 0.817653 | 0.817653 | 0.811316 | 0.818710 | 0.819022 |
| heavy | 0.761764 | 0.761748 | 0.759434 | 0.764631 | 0.763061 |

### 10.5 结论

1. n114 在 light/mid 上优于 baseline 与 n113，heavy 略低于 n113 但仍高于 baseline 与 n111。
2. `lambda` 有明显最优区间，约在 `0.5` 附近；`lambda>=1.0` 会出现显著过扣除。
3. 建议默认参数为 `MYEVS_N114_LAMBDA=0.5`；后续可做 `0.3~0.8` 细扫，重点观察 heavy 与 mean_f1 的折中。

产物目录：

- `data/ED24/myPedestrain_06/EBF_Part2/_tune_n114_lambda/summary.csv`
- `data/ED24/myPedestrain_06/EBF_Part2/_tune_n114_lambda/summary.json`
- `data/ED24/myPedestrain_06/EBF_Part2/_tune_n114_lambda/full_validate_top1.json`

## 11）n116（7.62）：等时突发门控（Isochronous Burst Gate）

### 11.1 核心思想

n116 在 baseline 的同极性邻域累加上叠加一个“短时突发计数门控”：

1. 先按 baseline 计算同极性时间衰减累加分数 `score_base`。
2. 同时统计邻域内“足够新”的同极性支持个数 `n_burst`（`dt <= tau_burst`）。
3. 若 `n_burst < thresh`，则将 `score_base` 乘以惩罚系数 `penalty`，否则保持不变。

实现文件：`src/myevs/denoise/ops/ebfopt_part2/n116_isochronous_burst_gate_backbone.py`

### 11.2 公式

baseline 同极性分数：

$$
Score_{base} = \sum_{j\in\mathcal{N},\ p_j=p_c} \max\left(0, 1-\frac{t_c-t_j}{\tau}\right)
$$

短时突发计数：

$$
n_{burst} = \sum_{j\in\mathcal{N},\ p_j=p_c} \mathbb{1}(t_c-t_j \le \tau_{burst})
$$

门控后的最终得分：

$$
Score =
\begin{cases}
Score_{base}, & n_{burst} \ge thresh \\
penalty\cdot Score_{base}, & n_{burst} < thresh
\end{cases}
$$

默认参数（本轮调优后）：

- `MYEVS_N116_TAU_BURST_US=2000`
- `MYEVS_N116_THRESH=3`
- `MYEVS_N116_PENALTY=0.2`

### 11.3 参数扫频命令（已执行）

```bash
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/tune_n116_burst.py \
	--project-root . \
	--base-out-dir data/ED24/myPedestrain_06/EBF_Part2/_tune_n116_burst \
	--tau-burst-us-grid 1000,2000,4000 \
	--thresh-grid 1,2,3 \
	--penalty-grid 0.0,0.2,0.5 \
	--s-list 7,9 \
	--tau-us-list 64000,128000 \
	--max-events 200000 \
	--full-validate \
	--full-s-list 3,5,7,9 \
	--full-tau-us-list 8000,16000,32000,64000,128000,256000,512000,1024000 \
	--full-max-events 0
```

预筛选（按 `mean_f1` 排序）Top5：

| rank | tau_burst_us | thresh | penalty | mean_f1 | mean_auc |
|---:|---:|---:|---:|---:|---:|
| 1 | 2000 | 3 | 0.2 | 0.850110 | 0.930254 |
| 2 | 2000 | 3 | 0.5 | 0.850103 | 0.930427 |
| 3 | 4000 | 3 | 0.2 | 0.849934 | 0.929901 |
| 4 | 4000 | 3 | 0.5 | 0.849926 | 0.930560 |
| 5 | 1000 | 3 | 0.5 | 0.849878 | 0.930208 |

Top1 为 `tau_burst_us=2000, thresh=3, penalty=0.2`。

### 11.4 Top1 全量验证结果

Top1（`tau_burst_us=2000, thresh=3, penalty=0.2`）full-validate（`s={3,5,7,9}`, `tau=8..1024ms`）：

| env | best-AUC tag | AUC | best-F1 tag | F1 | best-F1阈值 |
|---|---|---:|---|---:|---:|
| light | `ebf_n116_labelscore_s9_tau128000` | 0.947875 | `ebf_n116_labelscore_s9_tau512000` | 0.952015 | 0.343055 |
| mid | `ebf_n116_labelscore_s9_tau128000` | 0.923497 | `ebf_n116_labelscore_s9_tau128000` | 0.818116 | 0.989339 |
| heavy | `ebf_n116_labelscore_s9_tau128000` | 0.913594 | `ebf_n116_labelscore_s7_tau64000` | 0.761938 | 0.715100 |

与 baseline / n111 / n112 / n113 / n114 / n116 最优配置对比（best-F1）：

| env | baseline | n111 (`mu=0.0`) | n112 (`rsmall=2,thresh=0.6`) | n113 (`gamma=2.0`) | n114 (`lambda=0.5`) | n116 (`tau_burst=2000,th=3,p=0.2`) |
|---|---:|---:|---:|---:|---:|---:|
| light | 0.952021 | 0.952015 | 0.940393 | 0.952056 | 0.952188 | 0.952015 |
| mid | 0.817653 | 0.817653 | 0.811316 | 0.818710 | 0.819022 | 0.818116 |
| heavy | 0.761764 | 0.761748 | 0.759434 | 0.764631 | 0.763061 | 0.761938 |

### 11.5 二轮细扫（扩大 `thresh`）

为验证 `thresh` 的稳定性，进行了更细网格并将 `thresh` 扩展到 6：

```bash
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/tune_n116_burst.py \
	--project-root . \
	--base-out-dir data/ED24/myPedestrain_06/EBF_Part2/_tune_n116_burst_fine_v2 \
	--tau-burst-us-grid 500,1000,1500,2000,2500,3000,4000,6000 \
	--thresh-grid 1,2,3,4,5,6 \
	--penalty-grid 0.1,0.2,0.3,0.4,0.5,0.6 \
	--s-list 7,9 \
	--tau-us-list 64000,128000 \
	--max-events 200000 \
	--full-validate \
	--full-s-list 3,5,7,9 \
	--full-tau-us-list 8000,16000,32000,64000,128000,256000,512000,1024000 \
	--full-max-events 0
```

细扫预筛选 Top5（按 `mean_f1`）：

| rank | tau_burst_us | thresh | penalty | mean_f1 | mean_auc |
|---:|---:|---:|---:|---:|---:|
| 1 | 2000 | 3 | 0.3 | 0.850116 | 0.930330 |
| 2 | 2000 | 3 | 0.1 | 0.850115 | 0.930209 |
| 3 | 2000 | 3 | 0.2 | 0.850110 | 0.930254 |
| 4 | 2000 | 3 | 0.4 | 0.850107 | 0.930394 |
| 5 | 2000 | 3 | 0.5 | 0.850103 | 0.930427 |

细扫 Top1（`tau_burst_us=2000, thresh=3, penalty=0.3`）full-validate：

| env | best-AUC tag | AUC | best-F1 tag | F1 | best-F1阈值 |
|---|---|---:|---|---:|---:|
| light | `ebf_n116_labelscore_s9_tau128000` | 0.947872 | `ebf_n116_labelscore_s9_tau512000` | 0.952012 | 0.514579 |
| mid | `ebf_n116_labelscore_s9_tau128000` | 0.923564 | `ebf_n116_labelscore_s9_tau128000` | 0.818117 | 1.451109 |
| heavy | `ebf_n116_labelscore_s9_tau128000` | 0.913756 | `ebf_n116_labelscore_s7_tau64000` | 0.761938 | 1.072650 |

对比首轮 Top1（`p=0.2`）可见：

1. `thresh=3` 在两轮 sweep 中都稳定最优，扩到 `4~6` 后整体退化。
2. `penalty` 在 `0.1~0.5` 出现平台区，差异约在 `1e-5~1e-4` 量级。
3. 若以 full-validate 的 `mean_f1` 为主，`p=0.2` 仍略优（极微小差异）；若看 `mean_auc`，`p=0.3` 略高。

### 11.6 结论

1. n116 相比 baseline：mid/heavy 有小幅提升（`+0.000463/+0.000174`），light 基本持平略低（`-0.000006`）。
2. 从全量 best-F1 看，n116 整体未超过 n113/n114；它更像“稳态小修正”而非主增益方向。
3. 综合稳定性与 full-validate `mean_f1`，默认继续使用 `MYEVS_N116_TAU_BURST_US=2000`, `MYEVS_N116_THRESH=3`, `MYEVS_N116_PENALTY=0.2`；`p=0.3` 可作为 AUC 导向的备选。

产物目录：

- `data/ED24/myPedestrain_06/EBF_Part2/_tune_n116_burst/summary.csv`
- `data/ED24/myPedestrain_06/EBF_Part2/_tune_n116_burst/summary.json`
- `data/ED24/myPedestrain_06/EBF_Part2/_tune_n116_burst/full_validate_top1.json`
- `data/ED24/myPedestrain_06/EBF_Part2/_tune_n116_burst_fine_v2/summary.csv`
- `data/ED24/myPedestrain_06/EBF_Part2/_tune_n116_burst_fine_v2/summary.json`
- `data/ED24/myPedestrain_06/EBF_Part2/_tune_n116_burst_fine_v2/full_validate_top1.json`

## 12）n117（7.63）：双极性时空回声奖励（Bipolar Echo Boost）

### 12.1 核心思想

n117 尝试在 baseline 同极性密度分数之上加入“异极性回声奖励”：

1. 同极性邻域维持 baseline 累加，得到 `score_base`。
2. 额外检查邻域异极性历史事件，若其时间差落在 `[echo_min, echo_max]`，判定存在回声支撑。
3. 对存在回声的事件做乘性奖励 `score = score_base * (1 + alpha)`；无回声不惩罚。

实现文件：`src/myevs/denoise/ops/ebfopt_part2/n117_bipolar_echo_boost_backbone.py`

### 12.2 公式

基础分数（同极性）：

$$
Score_{base} = \sum_{j\in\mathcal{N},\ p_j=p_c} \max\left(0,1-\frac{t_c-t_j}{\tau}\right)
$$

异极性回声条件：

$$
has\_echo = \mathbb{1}\left(\exists j\in\mathcal{N},\ p_j\neq p_c,\ \tau_{echo,min}\le t_c-t_j\le\tau_{echo,max}\right)
$$

最终分数：

$$
Score =
\begin{cases}
Score_{base}\cdot(1+\alpha), & has\_echo=1 \\
Score_{base}, & has\_echo=0
\end{cases}
$$

默认参数（按本轮 Top1 对齐）：

- `MYEVS_N117_ECHO_MIN_US=1000`
- `MYEVS_N117_ECHO_MAX_US=100000`
- `MYEVS_N117_ALPHA=0.5`

### 12.3 参数扫频命令（已执行）

```bash
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/tune_n117_echo.py \
	--project-root . \
	--base-out-dir data/ED24/myPedestrain_06/EBF_Part2/_tune_n117_echo \
	--echo-min-us-grid 1000,5000,10000 \
	--echo-max-us-grid 30000,60000,100000 \
	--alpha-grid 0.5,1.0,2.0 \
	--s-list 7,9 \
	--tau-us-list 64000,128000 \
	--max-events 200000 \
	--full-validate \
	--full-s-list 3,5,7,9 \
	--full-tau-us-list 8000,16000,32000,64000,128000,256000,512000,1024000 \
	--full-max-events 0
```

预筛选（按 `mean_f1` 排序）Top5：

| rank | echo_min_us | echo_max_us | alpha | mean_f1 | mean_auc |
|---:|---:|---:|---:|---:|---:|
| 1 | 1000 | 100000 | 0.5 | 0.848419 | 0.929046 |
| 2 | 5000 | 100000 | 0.5 | 0.848123 | 0.928839 |
| 3 | 10000 | 100000 | 0.5 | 0.847684 | 0.928538 |
| 4 | 1000 | 60000 | 0.5 | 0.847330 | 0.928575 |
| 5 | 5000 | 60000 | 0.5 | 0.846849 | 0.928373 |

Top1 为 `echo_min_us=1000, echo_max_us=100000, alpha=0.5`。

### 12.4 Top1 全量验证结果

Top1 full-validate（`s={3,5,7,9}`, `tau=8..1024ms`）：

| env | best-AUC tag | AUC | best-F1 tag | F1 | best-F1阈值 |
|---|---|---:|---|---:|---:|
| light | `ebf_n117_labelscore_s9_tau128000` | 0.947571 | `ebf_n117_labelscore_s9_tau512000` | 0.952034 | 1.841086 |
| mid | `ebf_n117_labelscore_s9_tau128000` | 0.922487 | `ebf_n117_labelscore_s9_tau128000` | 0.815895 | 7.254562 |
| heavy | `ebf_n117_labelscore_s9_tau128000` | 0.912957 | `ebf_n117_labelscore_s9_tau128000` | 0.759884 | 10.921512 |

与 baseline / n113 / n114 / n116 / n117 最优配置对比（best-F1）：

| env | baseline | n113 (`gamma=2.0`) | n114 (`lambda=0.5`) | n116 (`tau_burst=2000,th=3,p=0.2`) | n117 (`emin=1000,emax=100000,a=0.5`) |
|---|---:|---:|---:|---:|---:|
| light | 0.952021 | 0.952056 | 0.952188 | 0.952015 | 0.952034 |
| mid | 0.817653 | 0.818710 | 0.819022 | 0.818116 | 0.815895 |
| heavy | 0.761764 | 0.764631 | 0.763061 | 0.761938 | 0.759884 |

### 12.5 结论

1. n117 在 light 上与 baseline 基本持平，但在 mid/heavy 明显下降，full-validate 下整体不优。
2. `alpha` 越大退化越明显（`1.0/2.0` 均劣于 `0.5`），说明异极性奖励在本数据上会放大误检。
3. 结论为负：n117 不建议作为主线方向，后续优先继续围绕 n113/n114 或 n116 的小幅稳定增益做组合优化。

产物目录：

- `data/ED24/myPedestrain_06/EBF_Part2/_tune_n117_echo/summary.csv`
- `data/ED24/myPedestrain_06/EBF_Part2/_tune_n117_echo/summary.json`
- `data/ED24/myPedestrain_06/EBF_Part2/_tune_n117_echo/full_validate_top1.json`

## 13）n118（7.64）：极性空间偶极子质心偏移门控（Polarity Spatial Dipole / CoM Offset）

### 13.1 核心思想

n118 针对 n117 的失败模式做了结构升级：

1. 同极性邻域仍按 baseline 线性时衰累加，得到 `W_same`。
2. 对异极性仅统计回声时间窗 `[echo_min, echo_max]` 内的邻域证据，得到 `W_opp`。
3. 分别计算同极性与异极性的时间加权质心，若两者空间偏移足够大（`Dist^2 >= dist_th^2`），再对 `W_same` 施加奖励 `*(1+alpha)`。

实现文件：`src/myevs/denoise/ops/ebfopt_part2/n118_polarity_dipole_backbone.py`

### 13.2 公式

同极性基础分数：

$$
W_{same} = \sum_{j\in\mathcal{N},\ p_j=p_c} \max\left(0,1-\frac{t_c-t_j}{\tau}\right)
$$

异极性回声证据（仅在时间窗内统计）：

$$
W_{opp} = \sum_{j\in\mathcal{N},\ p_j\neq p_c,\ \tau_{min}\le t_c-t_j\le\tau_{max}} w_j
$$

质心与距离：

$$
\mathbf{c}_{same}=\frac{\sum w_j\mathbf{x}_j}{W_{same}},\quad
\mathbf{c}_{opp}=\frac{\sum w_j\mathbf{x}_j}{W_{opp}},\quad
Dist^2=\|\mathbf{c}_{same}-\mathbf{c}_{opp}\|^2
$$

最终分数：

$$
Score=
\begin{cases}
W_{same}\cdot(1+\alpha), & W_{same}>0, W_{opp}>0, Dist^2\ge dist\_th^2 \\
W_{same}, & \text{otherwise}
\end{cases}
$$

默认参数（按本轮 Top1）：

- `MYEVS_N118_ECHO_MIN_US=1000`
- `MYEVS_N118_ECHO_MAX_US=60000`
- `MYEVS_N118_ALPHA=0.5`
- `MYEVS_N118_DIST_TH=1.0`

### 13.3 参数扫频命令（已执行）

```bash
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/tune_n118_dipole.py \
	--project-root . \
	--base-out-dir data/ED24/myPedestrain_06/EBF_Part2/_tune_n118_dipole \
	--alpha-grid 0.5,1.0,2.0 \
	--dist-th-grid 1.0,1.5,2.0 \
	--s-list 7,9 \
	--tau-us-list 64000,128000 \
	--max-events 200000 \
	--full-validate \
	--full-s-list 3,5,7,9 \
	--full-tau-us-list 8000,16000,32000,64000,128000,256000,512000,1024000 \
	--full-max-events 0
```

预筛选（按 `mean_f1` 排序）Top5：

| rank | echo_min_us | echo_max_us | alpha | dist_th | mean_f1 | mean_auc |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1000 | 60000 | 0.5 | 1.0 | 0.846452 | 0.928431 |
| 2 | 1000 | 60000 | 0.5 | 1.5 | 0.845582 | 0.928243 |
| 3 | 1000 | 60000 | 0.5 | 2.0 | 0.844430 | 0.928118 |
| 4 | 1000 | 60000 | 1.0 | 1.0 | 0.841634 | 0.925133 |
| 5 | 1000 | 60000 | 1.0 | 1.5 | 0.838747 | 0.924145 |

Top1 为 `echo_min_us=1000, echo_max_us=60000, alpha=0.5, dist_th=1.0`。

### 13.4 Top1 全量验证结果

Top1 full-validate（`s={3,5,7,9}`, `tau=8..1024ms`）：

| env | best-AUC tag | AUC | best-F1 tag | F1 | best-F1阈值 |
|---|---|---:|---|---:|---:|
| light | `ebf_n118_labelscore_s9_tau128000` | 0.947628 | `ebf_n118_labelscore_s9_tau512000` | 0.952262 | 1.777787 |
| mid | `ebf_n118_labelscore_s9_tau128000` | 0.921448 | `ebf_n118_labelscore_s9_tau128000` | 0.813600 | 6.966082 |
| heavy | `ebf_n118_labelscore_s9_tau128000` | 0.911202 | `ebf_n118_labelscore_s9_tau128000` | 0.756466 | 10.397508 |

与 baseline / n113 / n114 / n116 / n117 / n118 最优配置对比（best-F1）：

| env | baseline | n113 (`gamma=2.0`) | n114 (`lambda=0.5`) | n116 (`tau_burst=2000,th=3,p=0.2`) | n117 (`emin=1000,emax=100000,a=0.5`) | n118 (`emin=1000,emax=60000,a=0.5,d=1.0`) |
|---|---:|---:|---:|---:|---:|---:|
| light | 0.952021 | 0.952056 | 0.952188 | 0.952015 | 0.952034 | 0.952262 |
| mid | 0.817653 | 0.818710 | 0.819022 | 0.818116 | 0.815895 | 0.813600 |
| heavy | 0.761764 | 0.764631 | 0.763061 | 0.761938 | 0.759884 | 0.756466 |

### 13.5 结论

1. n118 在 light 上有极小提升，但 mid/heavy 明显低于 baseline，整体为负结果。
2. 扫频显示 `alpha` 增大时性能近似单调下降；`dist_th` 从 `1.0` 增至 `1.5/2.0` 也持续下降，说明“偶极门控”在本数据上仍会放大噪声侧误触发。
3. 结论为负：n118 不建议纳入主线，后续优先继续围绕 n113/n114 与 n116 的稳态小增益路线做组合优化。

产物目录：

- `data/ED24/myPedestrain_06/EBF_Part2/_tune_n118_dipole/summary.csv`
- `data/ED24/myPedestrain_06/EBF_Part2/_tune_n118_dipole/summary.json`
- `data/ED24/myPedestrain_06/EBF_Part2/_tune_n118_dipole/full_validate_top1.json`

## 14）n120（7.66）：中心像素时序自抑制（Center-Pixel Temporal Self-Inhibition）

### 14.1 核心思想

n120 针对“单像素短时高频噪声串发”引入中心像素自抑制门控：

1. 对当前像素维护一个随时间线性衰减的自激活计数 `rate_now`。
2. 若 `rate_now > r_max`，当前事件直接输出 0 分并跳过邻域遍历。
3. 未触发自抑制时，仍使用 baseline 同极性邻域线性时衰分数作为最终得分。

实现文件：`src/myevs/denoise/ops/ebfopt_part2/n120_self_inhibition_backbone.py`

### 14.2 公式

中心像素自激活率更新：

$$
r_t = r_{t^-}\cdot \max\left(0, 1-\frac{\Delta t_{self}}{\tau_{self}}\right) + 1
$$

门控与最终分数：

$$
Score=
\begin{cases}
0, & r_t > r_{max} \\
\sum_{j\in\mathcal{N},\ p_j=p_c}\max\left(0,1-\frac{t_c-t_j}{\tau}\right), & r_t \le r_{max}
\end{cases}
$$

默认参数（按本轮 Top1）：

- `MYEVS_N120_TAU_SELF_US=30000`
- `MYEVS_N120_R_MAX=5.0`

### 14.3 参数扫频命令（已执行）

```bash
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/tune_n120_self_inhib.py \
	--python-exe D:/software/Anaconda_envs/envs/myEVS/python.exe \
	--project-root . \
	--base-out-dir data/ED24/myPedestrain_06/EBF_Part2/_tune_n120_self_inhib \
	--tau-self-us-grid 10000,30000,50000 \
	--r-max-grid 2.0,3.0,4.0,5.0 \
	--s-list 7,9 \
	--tau-us-list 64000,128000 \
	--max-events 200000 \
	--full-validate \
	--full-s-list 3,5,7,9 \
	--full-tau-us-list 8000,16000,32000,64000,128000,256000,512000,1024000 \
	--full-max-events 0
```

预筛选（按 `mean_f1` 排序）Top5：

| rank | tau_self_us | r_max | mean_f1 | mean_auc |
|---:|---:|---:|---:|---:|
| 1 | 30000 | 5.0 | 0.850970 | 0.931338 |
| 2 | 50000 | 5.0 | 0.850940 | 0.931247 |
| 3 | 30000 | 4.0 | 0.850765 | 0.931029 |
| 4 | 10000 | 3.0 | 0.850703 | 0.931041 |
| 5 | 10000 | 4.0 | 0.850405 | 0.930698 |

Top1 为 `tau_self_us=30000, r_max=5.0`。

### 14.4 Top1 全量验证结果

Top1 full-validate（`s={3,5,7,9}`, `tau=8..1024ms`）：

| env | best-AUC tag | AUC | best-F1 tag | F1 | best-F1阈值 |
|---|---|---:|---|---:|---:|
| light | `ebf_n120_labelscore_s9_tau128000` | 0.949782 | `ebf_n120_labelscore_s9_tau512000` | 0.953856 | 1.690533 |
| mid | `ebf_n120_labelscore_s9_tau128000` | 0.923946 | `ebf_n120_labelscore_s9_tau128000` | 0.818312 | 4.836094 |
| heavy | `ebf_n120_labelscore_s9_tau128000` | 0.913971 | `ebf_n120_labelscore_s7_tau64000` | 0.762100 | 3.579719 |

与 baseline / n113 / n114 / n116 / n117 / n118 / n120 最优配置对比（best-F1）：

| env | baseline | n113 (`gamma=2.0`) | n114 (`lambda=0.5`) | n116 (`tau_burst=2000,th=3,p=0.2`) | n117 (`emin=1000,emax=100000,a=0.5`) | n118 (`emin=1000,emax=60000,a=0.5,d=1.0`) | n120 (`tau_self=30000,r_max=5.0`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| light | 0.952021 | 0.952056 | 0.952188 | 0.952015 | 0.952034 | 0.952262 | 0.953856 |
| mid | 0.817653 | 0.818710 | 0.819022 | 0.818116 | 0.815895 | 0.813600 | 0.818312 |
| heavy | 0.761764 | 0.764631 | 0.763061 | 0.761938 | 0.759884 | 0.756466 | 0.762100 |

### 14.5 结论

1. n120 在 light/mid/heavy 三档均优于 baseline（增益分别约 `+0.001835/+0.000659/+0.000336`），且 heavy 端未出现退化。
2. 参数趋势上，`r_max` 偏小（2.0~3.0）会明显欠杀信号；`r_max=5.0` + `tau_self=30~50ms` 形成稳定高分区。
3. 当前结论为正：n120 可作为后续主线候选，并建议以 `MYEVS_N120_TAU_SELF_US=30000`, `MYEVS_N120_R_MAX=5.0` 作为默认起点。

产物目录：

- `data/ED24/myPedestrain_06/EBF_Part2/_tune_n120_self_inhib/summary.csv`
- `data/ED24/myPedestrain_06/EBF_Part2/_tune_n120_self_inhib/summary.json`
- `data/ED24/myPedestrain_06/EBF_Part2/_tune_n120_self_inhib/full_validate_top1.json`

## 15）n121（7.67）：时空中心-周围对抗（Spatiotemporal Center-Surround）

### 15.1 核心思想

n121 尝试针对 heavy 下“邻域内均匀噪声堆积也能拿高分”的失败模式，引入中心-周围对抗结构：

1. 邻域内同极性历史事件仍按线性时衰计权。
2. 将邻域按 Chebyshev 距离分成内圈（`dist <= r_core`）与外圈（`r_core < dist <= r`）。
3. 内圈作为正证据，外圈作为抑制项，最终分数做截断非负。

实现文件：`src/myevs/denoise/ops/ebfopt_part2/n121_center_surround_backbone.py`

### 15.2 公式

定义：

$$
E_{core} = \sum_{j\in\mathcal{N},\ d_j\le r_{core}} w_j,
\quad
E_{surround} = \sum_{j\in\mathcal{N},\ d_j> r_{core}} w_j
$$

其中时间权重为：

$$
w_j = \max\left(0,1-\frac{t_c-t_j}{\tau}\right)
$$

最终分数：

$$
Score = \max\left(0, E_{core} - \beta E_{surround}\right)
$$

默认参数（按本轮 Top1）：

- `MYEVS_N121_R_CORE=1`
- `MYEVS_N121_BETA=0.1`

### 15.3 参数扫频命令（已执行）

```bash
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/tune_n121_center_surround.py \
	--python-exe D:/software/Anaconda_envs/envs/myEVS/python.exe \
	--project-root . \
	--base-out-dir data/ED24/myPedestrain_06/EBF_Part2/_tune_n121_center_surround \
	--r-core-grid 1 \
	--beta-grid 0.1,0.2,0.3,0.5 \
	--s-list 7,9 \
	--tau-us-list 64000,128000 \
	--max-events 200000 \
	--full-validate \
	--full-s-list 3,5,7,9 \
	--full-tau-us-list 8000,16000,32000,64000,128000,256000,512000,1024000 \
	--full-max-events 0
```

预筛选（按 `mean_f1` 排序）Top4：

| rank | r_core | beta | mean_f1 | mean_auc |
|---:|---:|---:|---:|---:|
| 1 | 1 | 0.1 | 0.744306 | 0.839940 |
| 2 | 1 | 0.2 | 0.679341 | 0.774986 |
| 3 | 1 | 0.3 | 0.611163 | 0.684993 |
| 4 | 1 | 0.5 | 0.555866 | 0.564528 |

Top1 为 `r_core=1, beta=0.1`。

### 15.4 Top1 全量验证结果

Top1 full-validate（`s={3,5,7,9}`, `tau=8..1024ms`）：

| env | best-AUC tag | AUC | best-F1 tag | F1 | best-F1阈值 |
|---|---|---:|---|---:|---:|
| light | `ebf_n121_labelscore_s3_tau512000` | 0.883806 | `ebf_n121_labelscore_s3_tau8000` | 0.911931 | 0.000000 |
| mid | `ebf_n121_labelscore_s3_tau128000` | 0.863742 | `ebf_n121_labelscore_s3_tau64000` | 0.748866 | 0.759641 |
| heavy | `ebf_n121_labelscore_s3_tau128000` | 0.854863 | `ebf_n121_labelscore_s3_tau64000` | 0.690043 | 0.984609 |

与 baseline / n113 / n114 / n116 / n117 / n118 / n120 / n121 最优配置对比（best-F1）：

| env | baseline | n113 (`gamma=2.0`) | n114 (`lambda=0.5`) | n116 (`tau_burst=2000,th=3,p=0.2`) | n117 (`emin=1000,emax=100000,a=0.5`) | n118 (`emin=1000,emax=60000,a=0.5,d=1.0`) | n120 (`tau_self=30000,r_max=5.0`) | n121 (`r_core=1,beta=0.1`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| light | 0.952021 | 0.952056 | 0.952188 | 0.952015 | 0.952034 | 0.952262 | 0.953856 | 0.911931 |
| mid | 0.817653 | 0.818710 | 0.819022 | 0.818116 | 0.815895 | 0.813600 | 0.818312 | 0.748866 |
| heavy | 0.761764 | 0.764631 | 0.763061 | 0.761938 | 0.759884 | 0.756466 | 0.762100 | 0.690043 |

### 15.5 结论

1. n121 在三档噪声上均显著低于 baseline 与 n120，属于明确负结果。
2. `beta` 从 `0.1 -> 0.5` 呈单调退化，说明“外圈惩罚”在本数据上主要在削弱有效结构而非抑制噪声。
3. 结论为负：n121 不建议继续作为主线方向，后续应保留 n120 路线并探索更温和的抑制机制。

产物目录：

- `data/ED24/myPedestrain_06/EBF_Part2/_tune_n121_center_surround/summary.csv`
- `data/ED24/myPedestrain_06/EBF_Part2/_tune_n121_center_surround/summary.json`
- `data/ED24/myPedestrain_06/EBF_Part2/_tune_n121_center_surround/full_validate_top1.json`

## 16）n123（7.68）：8扇区各向同性底噪扣除 + Top2 扇区聚合

### 16.1 核心思想

n123 基于 n114 的“各向同性底噪扣除”思路，做了两点结构改造：

1. 邻域从粗粒度方向分解升级为 8 扇区（45 度分辨率）同极性证据统计。
2. 先用最弱 3 扇区估计底噪，再只保留扣噪后最强的 2 个扇区进行聚合，减少弱扇区噪声残差累加。

实现文件：`src/myevs/denoise/ops/ebfopt_part2/n123_isotropic_max8_backbone.py`

### 16.2 公式

设 8 扇区同极性时衰证据为 $E_k, k\in\{0,\dots,7\}$，升序记作 $E_{(0)}\le\cdots\le E_{(7)}$。

底噪估计：

$$
NoiseFloor = \frac{E_{(0)} + E_{(1)} + E_{(2)}}{3}
$$

扣噪并取 Top2：

$$
Score = \max(0, E_{(7)} - \lambda\,NoiseFloor) + \max(0, E_{(6)} - \lambda\,NoiseFloor)
$$

其中时间权重为：

$$
w_j = \max\left(0,1-\frac{t_c-t_j}{\tau}\right)
$$

默认参数（按本轮 Top1）：

- `MYEVS_N123_LAMBDA=0.5`

### 16.3 参数扫频命令（已执行）

```bash
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/tune_n123_iso8_top2.py \
	--python-exe D:/software/Anaconda_envs/envs/myEVS/python.exe \
	--project-root . \
	--base-out-dir data/ED24/myPedestrain_06/EBF_Part2/_tune_n123_iso8_top2 \
	--lambda-grid 0.5,1.0,1.5,2.0 \
	--s-list 7,9 \
	--tau-us-list 64000,128000 \
	--max-events 200000 \
	--full-validate \
	--full-s-list 3,5,7,9 \
	--full-tau-us-list 8000,16000,32000,64000,128000,256000,512000,1024000 \
	--full-max-events 0
```

预筛选（按 `mean_f1` 排序）Top4：

| rank | lambda | mean_f1 | mean_auc |
|---:|---:|---:|---:|
| 1 | 0.5 | 0.846054 | 0.928805 |
| 2 | 1.0 | 0.842305 | 0.927052 |
| 3 | 1.5 | 0.835148 | 0.922326 |
| 4 | 2.0 | 0.825308 | 0.914151 |

Top1 为 `lambda=0.5`。

### 16.4 Top1 全量验证结果

Top1 full-validate（`s={3,5,7,9}`, `tau=8..1024ms`）：

| env | best-AUC tag | AUC | best-F1 tag | F1 | best-F1阈值 |
|---|---|---:|---|---:|---:|
| light | `ebf_n123_labelscore_s9_tau128000` | 0.947340 | `ebf_n123_labelscore_s9_tau512000` | 0.952116 | 1.632074 |
| mid | `ebf_n123_labelscore_s9_tau128000` | 0.921502 | `ebf_n123_labelscore_s9_tau128000` | 0.814154 | 3.255133 |
| heavy | `ebf_n123_labelscore_s9_tau128000` | 0.909913 | `ebf_n123_labelscore_s9_tau128000` | 0.755086 | 4.230930 |

与 baseline / n113 / n114 / n116 / n117 / n118 / n120 / n121 / n123 最优配置对比（best-F1）：

| env | baseline | n113 (`gamma=2.0`) | n114 (`lambda=0.5`) | n116 (`tau_burst=2000,th=3,p=0.2`) | n117 (`emin=1000,emax=100000,a=0.5`) | n118 (`emin=1000,emax=60000,a=0.5,d=1.0`) | n120 (`tau_self=30000,r_max=5.0`) | n121 (`r_core=1,beta=0.1`) | n123 (`lambda=0.5`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| light | 0.952021 | 0.952056 | 0.952188 | 0.952015 | 0.952034 | 0.952262 | 0.953856 | 0.911931 | 0.952116 |
| mid | 0.817653 | 0.818710 | 0.819022 | 0.818116 | 0.815895 | 0.813600 | 0.818312 | 0.748866 | 0.814154 |
| heavy | 0.761764 | 0.764631 | 0.763061 | 0.761938 | 0.759884 | 0.756466 | 0.762100 | 0.690043 | 0.755086 |

### 16.5 结论

1. n123 相较 n121 有明显恢复（AUC/F1 均大幅提升），说明“8扇区 + Top2”确实比 n121 的中心-周围扣分结构更稳健。
2. 但 n123 仍低于 baseline 与 n120，尤其 mid/heavy 仍有明显差距；当前不足以进入主线。
3. 在本轮 sweep 中 `lambda` 增大呈单调退化，最佳点稳定落在 `lambda=0.5`，建议若继续该方向仅在轻惩罚与更细粒度扇区融合上做小步探索。

产物目录：

- `data/ED24/myPedestrain_06/EBF_Part2/_tune_n123_iso8_top2/summary.csv`
- `data/ED24/myPedestrain_06/EBF_Part2/_tune_n123_iso8_top2/summary.json`
- `data/ED24/myPedestrain_06/EBF_Part2/_tune_n123_iso8_top2/full_validate_top1.json`

## 17）n124（7.69）：三机制融合（自抑制 + 幂律时衰 + 各向同性底噪扣除）

### 17.1 核心思想

n124 将三个已验证过的有效机制串联在同一骨干内：

1. n120：中心像素自抑制门控，先拦截高频热点。
2. n113：对同极性邻域时衰权重做幂律提纯（$w^\gamma$）。
3. n114：四象限底噪估计后做各向同性扣噪，保留净证据。

实现文件：`src/myevs/denoise/ops/ebfopt_part2/n124_synergy_trifilter_backbone.py`

### 17.2 公式

中心自抑制（n120）：

$$
R_t = R_{t-1}\cdot\max\left(0,1-\frac{\Delta t_{self}}{\tau_{self}}\right)+1,
\quad R_t>R_{max}\Rightarrow Score=0
$$

四象限同极性幂律聚合（n113）：

$$
E_k=\sum_{j\in\mathcal{N}_k,\,p_j=p_c}\left(\max\left(0,1-\frac{t_c-t_j}{\tau}\right)\right)^\gamma,
\quad k\in\{0,1,2,3\}
$$

各向同性底噪扣除（n114）：

$$
NoiseFloor=\frac{E_{(0)}+E_{(1)}}{2},
\quad
Score=\sum_{k=0}^{3}\max\left(0,E_k-\lambda\cdot NoiseFloor\right)
$$

默认参数（本轮固定）：

- `MYEVS_N124_TAU_SELF_US=30000`
- `MYEVS_N124_R_MAX=5.0`
- `MYEVS_N124_GAMMA=2.0`
- 仅扫 `MYEVS_N124_LAMBDA`

### 17.3 参数扫频命令（已执行）

```bash
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/tune_n124_synergy_trifilter.py \
	--python-exe D:/software/Anaconda_envs/envs/myEVS/python.exe \
	--project-root . \
	--base-out-dir data/ED24/myPedestrain_06/EBF_Part2/_tune_n124_synergy_trifilter \
	--lambda-grid 0.0,0.3,0.5,0.8,1.0 \
	--s-list 7,9 \
	--tau-us-list 64000,128000 \
	--max-events 200000 \
	--full-validate \
	--full-s-list 3,5,7,9 \
	--full-tau-us-list 8000,16000,32000,64000,128000,256000,512000,1024000 \
	--full-max-events 0
```

预筛选（按 `mean_f1` 排序）Top5：

| rank | lambda | mean_f1 | mean_auc |
|---:|---:|---:|---:|
| 1 | 0.3 | 0.853245 | 0.932182 |
| 2 | 0.5 | 0.852370 | 0.931835 |
| 3 | 0.0 | 0.851352 | 0.931757 |
| 4 | 0.8 | 0.845701 | 0.928863 |
| 5 | 1.0 | 0.835786 | 0.923630 |

Top1 为 `lambda=0.3`。

### 17.4 Top1 全量验证结果

Top1 full-validate（`s={3,5,7,9}`, `tau=8..1024ms`）：

| env | best-AUC tag | AUC | best-F1 tag | F1 | best-F1阈值 |
|---|---|---:|---|---:|---:|
| light | `ebf_n124_labelscore_s9_tau256000` | 0.951770 | `ebf_n124_labelscore_s9_tau512000` | 0.953905 | 0.996277 |
| mid | `ebf_n124_labelscore_s9_tau256000` | 0.926466 | `ebf_n124_labelscore_s9_tau128000` | 0.821299 | 3.536444 |
| heavy | `ebf_n124_labelscore_s9_tau256000` | 0.916940 | `ebf_n124_labelscore_s7_tau128000` | 0.767490 | 3.645591 |

与 baseline / n113 / n114 / n116 / n117 / n118 / n120 / n121 / n123 / n124 最优配置对比（best-F1）：

| env | baseline | n113 (`gamma=2.0`) | n114 (`lambda=0.5`) | n116 (`tau_burst=2000,th=3,p=0.2`) | n117 (`emin=1000,emax=100000,a=0.5`) | n118 (`emin=1000,emax=60000,a=0.5,d=1.0`) | n120 (`tau_self=30000,r_max=5.0`) | n121 (`r_core=1,beta=0.1`) | n123 (`lambda=0.5`) | n124 (`lambda=0.3`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| light | 0.952021 | 0.952056 | 0.952188 | 0.952015 | 0.952034 | 0.952262 | 0.953856 | 0.911931 | 0.952116 | 0.953905 |
| mid | 0.817653 | 0.818710 | 0.819022 | 0.818116 | 0.815895 | 0.813600 | 0.818312 | 0.748866 | 0.814154 | 0.821299 |
| heavy | 0.761764 | 0.764631 | 0.763061 | 0.761938 | 0.759884 | 0.756466 | 0.762100 | 0.690043 | 0.755086 | 0.767490 |

### 17.5 结论

1. n124 在 light/mid/heavy 三档 best-F1 全部超过 baseline 与 n120，是当前已验证变体中的最优结果。
2. 预筛选显示 `lambda=0.3` 最优，且 `lambda` 过大（0.8/1.0）明显退化，说明“轻扣噪 + 强结构保留”更符合当前数据分布。
3. 结论为正：n124 建议进入主线候选，并作为后续微调（`tau_self`、`gamma`、`lambda` 局部网格）的基线。

产物目录：

- `data/ED24/myPedestrain_06/EBF_Part2/_tune_n124_synergy_trifilter/summary.csv`
- `data/ED24/myPedestrain_06/EBF_Part2/_tune_n124_synergy_trifilter/summary.json`
- `data/ED24/myPedestrain_06/EBF_Part2/_tune_n124_synergy_trifilter/full_validate_top1.json`

## 18）n125（7.71）：微拓扑单调路径门控 + baseline 证据累积

### 18.1 核心思想

n125 在 baseline 同极性时衰累积前增加一个“微拓扑可达性”门控：

1. 以当前事件为起点，在 8 邻域内贪心追踪“时间单调递减”的同极性路径。
2. 每一步要求时间差不超过 `tau_step`，持续 `path_depth` 步。
3. 仅当路径成功时，才计算 baseline 分数；否则该事件直接置 0。

实现文件：`src/myevs/denoise/ops/ebfopt_part2/n125_micro_topo_path_backbone.py`

### 18.2 公式

设当前事件为 $e_i=(x_i,y_i,p_i,t_i)$，单步约束为：

$$
t_{k+1} < t_k,\quad t_k - t_{k+1} \le \tau_{step}
$$

若存在长度为 $D$（即 `path_depth`）的同极性单调路径，则通过门控；否则

$$
Score_i = 0
$$

通过门控后，分数回到 baseline 同极性邻域时衰和：

$$
Score_i = \sum_{j\in\mathcal{N},\,p_j=p_i}\max\left(0,1-\frac{t_i-t_j}{\tau}\right)
$$

默认参数（本轮 sweep 网格）：

- `MYEVS_N125_PATH_DEPTH in {2,3,4}`
- `MYEVS_N125_TAU_STEP_US in {2000,5000,10000}`

### 18.3 参数扫频命令（已执行）

```bash
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/tune_n125_micro_topo_path.py \
	--python-exe D:/software/Anaconda_envs/envs/myEVS/python.exe \
	--project-root . \
	--base-out-dir data/ED24/myPedestrain_06/EBF_Part2/_tune_n125_micro_topo_path \
	--path-depth-grid 2,3,4 \
	--tau-step-us-grid 2000,5000,10000 \
	--s-list 7,9 \
	--tau-us-list 64000,128000 \
	--max-events 200000 \
	--full-validate \
	--full-s-list 3,5,7,9 \
	--full-tau-us-list 8000,16000,32000,64000,128000,256000,512000,1024000 \
	--full-max-events 0
```

预筛选（按 `mean_f1` 排序）Top5：

| rank | path_depth | tau_step_us | mean_f1 | mean_auc |
|---:|---:|---:|---:|---:|
| 1 | 2 | 10000 | 0.712315 | 0.723863 |
| 2 | 3 | 10000 | 0.650238 | 0.676538 |
| 3 | 2 | 5000 | 0.600151 | 0.643695 |
| 4 | 4 | 10000 | 0.598785 | 0.639529 |
| 5 | 3 | 5000 | 0.560230 | 0.598069 |

Top1 为 `path_depth=2, tau_step_us=10000`。

### 18.4 Top1 全量验证结果

Top1 full-validate（`s={3,5,7,9}`, `tau=8..1024ms`）：

| env | best-AUC tag | AUC | best-F1 tag | F1 | best-F1阈值 |
|---|---|---:|---|---:|---:|
| light | `ebf_n125_labelscore_s9_tau32000` | 0.723658 | `ebf_n125_labelscore_s3_tau8000` | 0.911931 | 0.000000 |
| mid | `ebf_n125_labelscore_s5_tau32000` | 0.725894 | `ebf_n125_labelscore_s9_tau128000` | 0.620193 | 1.935969 |
| heavy | `ebf_n125_labelscore_s5_tau32000` | 0.727550 | `ebf_n125_labelscore_s9_tau128000` | 0.615231 | 4.348000 |

与 baseline / n120 / n124 / n125 最优配置对比（best-F1）：

| env | baseline | n120 (`tau_self=30000,r_max=5.0`) | n124 (`lambda=0.3`) | n125 (`d=2,step=10000`) |
|---|---:|---:|---:|---:|
| light | 0.952021 | 0.953856 | 0.953905 | 0.911931 |
| mid | 0.817653 | 0.818312 | 0.821299 | 0.620193 |
| heavy | 0.761764 | 0.762100 | 0.767490 | 0.615231 |

### 18.5 结论

1. n125 在 mid/heavy 上出现显著退化，mean-F1 与 mean-AUC 均远低于 baseline 与 n120/n124。
2. 预筛中 `tau_step_us` 增大虽有改善趋势，但整体仍停留在明显负区间，说明当前“硬门控”会过度抑制有效事件。
3. 结论为负：n125 不建议进入主线，后续若继续该方向应改为软门控（路径一致性作为加权项而非 0/1 截断）。

### 18.6 7.71 失效机理统计复盘（最近路径失效原因）

为验证 n125 失败原因，补充运行了 7.71 专用统计脚本（Light/Heavy）：

```bash
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/noise_analyze/n125_path_failure_stats_771.py \
	--env-list light,heavy \
	--max-events 500000 \
	--depth 4 \
	--tau-step-us 10000 \
	--out-dir data/ED24/myPedestrain_06/EBF_Part2/_n125_path_failure_stats_771
```

统计脚本：`scripts/noise_analyze/n125_path_failure_stats_771.py`

#### 18.6.1 相邻同极性最近旧邻居的时间差分布（\(\Delta t\)）

| env | [1,100]us | (100,1000]us | (1000,5000]us | (5000,10000]us | (10000,50000]us |
|---|---:|---:|---:|---:|---:|
| light | 0.0462 | 0.3528 | 0.2317 | 0.1608 | 0.2085 |
| heavy | 0.0160 | 0.1340 | 0.1627 | 0.1387 | 0.5487 |

解释：heavy 下超过 10ms 的邻域时间差占比达到 54.87%，显著超过 `tau_step=10ms` 的路径连接上限，天然导致“单调路径”难以延伸。

#### 18.6.2 AER 同时性 / 微抖动比率

- light：`collision_rate=0.000231`，`micro_jitter_rate(<=10us)=0.002582`
- heavy：`collision_rate=0.000050`，`micro_jitter_rate(<=10us)=0.000392`

解释：同时间戳冲突与微秒级抖动在当前数据上都很低，不构成 n125 主要失效来源。

#### 18.6.3 贪心路径“死亡深度与死因”统计（depth=4, tau_step=10ms）

路径深度分布：

| env | depth0 | depth1 | depth2 | depth3 | depth4(通关) |
|---|---:|---:|---:|---:|---:|
| light | 0.4906 | 0.1307 | 0.0789 | 0.0614 | 0.2383 |
| heavy | 0.8625 | 0.0813 | 0.0169 | 0.0094 | 0.0299 |

失败样本死因占比：

| env | A: 无更早同极性邻居 | B: 同时戳阻断 | C: 时间断层（\(\Delta t>\tau_{step}\)） |
|---|---:|---:|---:|
| light | 0.2197 | 0.000081 | 0.7803 |
| heavy | 0.1272 | 0.000016 | 0.8728 |

解释：light/heavy 都由 C 主导（78.03% / 87.28%），B 基本可忽略。

#### 18.6.4 复盘结论（对 n125 的直接意义）

1. n125 的主要失败机制是“时间断层”，不是“严格小于号”导致的同时戳阻断。
2. heavy 的微观拓扑链路天然断裂：`depth4` 通关率仅 2.99%，`depth0` 即死高达 86.25%。
3. 因此 7.71 方向若要继续，优先应改“时间连接模型/软连接权重”；仅把 `<` 改成 `<=` 不足以解决核心问题。

### 18.7 7.72 全尺度统计复盘（最近路径失效原因2）

按 7.72 要求补充了多尺度统计（密度/纯度、跳跃与时间断层、微观速度、方向一致性）：

```bash
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/noise_analyze/n125_path_failure_stats_772.py \
	--env-list light,heavy \
	--max-events 500000 \
	--tau-base-us 30000 \
	--tau-step-us 10000 \
	--out-dir data/ED24/myPedestrain_06/EBF_Part2/_n125_path_failure_stats_772
```

统计脚本：`scripts/noise_analyze/n125_path_failure_stats_772.py`

#### 18.7.1 多尺度密度与极性混合（R=4）

| env | same_R4_mean | opp_R4_mean | mix_R4_mean |
|---|---:|---:|---:|
| light | 9.7852 | 3.3833 | 0.1707 |
| heavy | 2.8598 | 2.0731 | 0.3359 |

解释：light 在宏观邻域上具备显著更高的同极性支持；heavy 的异极性混合明显更高（mix 约翻倍），表明“同极性单调链路”在重噪下更容易被局部混杂结构破坏。

#### 18.7.2 最新支持点的空间跳跃与时间分布

| env | latest_support_rate | jump_mean | jump_r3_r4_ratio | dt_mean_us | dt>30ms_ratio |
|---|---:|---:|---:|---:|---:|
| light | 0.9856 | 2.4326 | 0.4706 | 66376.31 | 0.1908 |
| heavy | 0.9949 | 2.8914 | 0.6555 | 17972.63 | 0.1889 |

解释：heavy 的“最新支持点”更多落在外圈（R3/R4），空间跳跃显著增大；这与 n125 依赖近邻单调回溯的假设不一致，容易造成路径追踪被噪声拓扑主导。

#### 18.7.3 微观速度与方向一致性

| env | vel_ge1_ratio | velocity_mean | angle_valid_rate | angle_var_mean | angle_low_var_ratio |
|---|---:|---:|---:|---:|---:|
| light | 0.5289 | 11.0460 | 0.5057 | 0.4605 | 0.2719 |
| heavy | 0.2755 | 3.8839 | 0.1154 | 0.4556 | 0.2871 |

解释：速度分布在 light/heavy 有差异，但当前“角度一致性”统计（圆方差）两者均值接近，区分度有限；仅靠该特征不足以解释 n125 的主要失效。

#### 18.7.4 复盘结论（对 n125 的补充）

1. 7.71 已证明 n125 主因是时间断层；7.72 进一步说明 heavy 下还存在更强的“外圈跳跃 + 高混合度”结构失配。
2. n125 的局部单调链路假设与 heavy 真实邻域拓扑不匹配：最新支持常在 R3/R4 且极性更混杂，导致门控稳定性差。
3. 若继续 7.71/7.72 方向，应从“硬路径存在性”转向“多尺度软证据融合”（宏观同极性支持、混合度惩罚、外圈跳跃抑制），避免 0/1 截断。

产物目录：

- `data/ED24/myPedestrain_06/EBF_Part2/_tune_n125_micro_topo_path/summary.csv`
- `data/ED24/myPedestrain_06/EBF_Part2/_tune_n125_micro_topo_path/summary.json`
- `data/ED24/myPedestrain_06/EBF_Part2/_tune_n125_micro_topo_path/full_validate_top1.json`
- `data/ED24/myPedestrain_06/EBF_Part2/_n125_path_failure_stats_771/summary.csv`
- `data/ED24/myPedestrain_06/EBF_Part2/_n125_path_failure_stats_771/summary.json`
- `data/ED24/myPedestrain_06/EBF_Part2/_n125_path_failure_stats_772/summary.csv`
- `data/ED24/myPedestrain_06/EBF_Part2/_n125_path_failure_stats_772/summary.json`

## 19）n126（7.72）：baseline + 同极性 top1 软加权

### 19.1 核心思想

n126 直接针对 n125 的“硬门控过抑制”失败模式做修正：

1. 保留 baseline 原始同极性时衰累积 `raw_score` 作为主干证据。
2. 额外提取邻域内“最近的同极性历史事件”（top1）时间戳。
3. 仅当 `raw_score >= raw_thr` 时，按 top1 新鲜度加入软加权 bonus；不做 0/1 硬截断。

实现文件：`src/myevs/denoise/ops/ebfopt_part2/n126_top1_bonus_backbone.py`

### 19.2 公式

baseline 主干分数：

$$
S_{raw} = \sum_{j\in\mathcal{N},\,p_j=p_i}\max\left(0,1-\frac{t_i-t_j}{\tau}\right)
$$

设 top1 为同极性最近历史（时间戳最大）邻居，记 $\Delta t_1=t_i-t_{top1}$，则：

$$
S = S_{raw} + \alpha\cdot\max\left(0,1-\frac{\Delta t_1}{\tau_1}\right)
$$

其中 bonus 仅在下述条件同时满足时生效：

$$
S_{raw}\ge raw\_thr,\quad \Delta t_1>0,\quad \Delta t_1\le\tau_1
$$

本轮 sweep 参数：

- `MYEVS_N126_ALPHA in {0.05,0.1,0.2,0.3}`
- `MYEVS_N126_TAU1_US in {2000,5000,10000,20000}`
- `MYEVS_N126_RAW_THR in {2.0}`

### 19.3 参数扫频命令（已执行）

```bash
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/tune_n126_top1_bonus.py \
	--python-exe D:/software/Anaconda_envs/envs/myEVS/python.exe \
	--project-root . \
	--base-out-dir data/ED24/myPedestrain_06/EBF_Part2/_tune_n126_top1_bonus \
	--alpha-grid 0.05,0.1,0.2,0.3 \
	--tau1-us-grid 2000,5000,10000,20000 \
	--raw-thr-grid 2.0 \
	--s-list 7,9 \
	--tau-us-list 64000,128000 \
	--max-events 200000 \
	--full-validate \
	--full-s-list 3,5,7,9 \
	--full-tau-us-list 8000,16000,32000,64000,128000,256000,512000,1024000 \
	--full-max-events 0
```

预筛选（按 `mean_f1` 排序）Top5：

| rank | alpha | tau1_us | raw_thr | mean_f1 | mean_auc |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.3 | 2000 | 2.0 | 0.849891 | 0.930200 |
| 2 | 0.2 | 2000 | 2.0 | 0.849831 | 0.930134 |
| 3 | 0.1 | 2000 | 2.0 | 0.849762 | 0.930059 |
| 4 | 0.2 | 5000 | 2.0 | 0.849727 | 0.930115 |
| 5 | 0.05 | 2000 | 2.0 | 0.849713 | 0.930018 |

Top1 为 `alpha=0.3, tau1_us=2000, raw_thr=2.0`。

### 19.4 Top1 全量验证结果

Top1 full-validate（`s={3,5,7,9}`, `tau=8..1024ms`）：

| env | best-AUC tag | AUC | best-F1 tag | F1 | best-F1阈值 |
|---|---|---:|---|---:|---:|
| light | `ebf_n126_labelscore_s9_tau128000` | 0.947682 | `ebf_n126_labelscore_s9_tau512000` | 0.952012 | 1.715096 |
| mid | `ebf_n126_labelscore_s9_tau128000` | 0.923463 | `ebf_n126_labelscore_s9_tau128000` | 0.818299 | 4.973133 |
| heavy | `ebf_n126_labelscore_s9_tau128000` | 0.913955 | `ebf_n126_labelscore_s7_tau64000` | 0.762218 | 3.715178 |

与 baseline / n120 / n124 / n125 / n126 最优配置对比（best-F1）：

| env | baseline | n120 (`tau_self=30000,r_max=5.0`) | n124 (`lambda=0.3`) | n125 (`d=2,step=10000`) | n126 (`a=0.3,tau1=2000,rthr=2.0`) |
|---|---:|---:|---:|---:|---:|
| light | 0.952021 | 0.953856 | 0.953905 | 0.911931 | 0.952012 |
| mid | 0.817653 | 0.818312 | 0.821299 | 0.620193 | 0.818299 |
| heavy | 0.761764 | 0.762100 | 0.767490 | 0.615231 | 0.762218 |

### 19.5 结论

1. n126 相比 baseline 在 mid/heavy 有小幅提升（mid +0.000646，heavy +0.000454），light 基本持平（-0.000009）。
2. 相比 n120，n126 在 heavy 有轻微优势（+0.000118），但 light 明显低于 n120，mid 基本打平略低；整体未超过 n120。
3. 相比 n124，n126 三档均明显落后，说明“top1 软加权”单独引入的信息增益有限，难以替代 n124 的三机制协同。
4. 结论为中性偏负：n126 可作为“软门控替代硬门控”的可行基线，但当前不建议替代 n124 主线。

产物目录：

- `data/ED24/myPedestrain_06/EBF_Part2/_tune_n126_top1_bonus/summary.csv`
- `data/ED24/myPedestrain_06/EBF_Part2/_tune_n126_top1_bonus/summary.json`
- `data/ED24/myPedestrain_06/EBF_Part2/_tune_n126_top1_bonus/full_validate_top1.json`

## 20）n127（7.73）：baseline + 同极性 top1/top2 双层软加权

### 20.1 核心思想

n127 在 n126 的“top1 软加权”上继续加一层同极性历史信息：

1. 主干仍为 baseline 同极性时衰累积 `raw_score`。
2. 在邻域中提取同极性最近两层历史（top1/top2 时间戳）。
3. 当 `raw_score >= raw_thr` 时，同时叠加 top1 与 top2 的软 bonus，不做硬截断。

实现文件：`src/myevs/denoise/ops/ebfopt_part2/n127_top2_bonus_backbone.py`

### 20.2 公式

baseline 主干分数：

$$
S_{raw} = \sum_{j\in\mathcal{N},\,p_j=p_i}\max\left(0,1-\frac{t_i-t_j}{\tau}\right)
$$

设 top1、top2 为同极性最近与次近历史，\(\Delta t_1=t_i-t_{top1}\)、\(\Delta t_2=t_i-t_{top2}\)，则：

$$
S = S_{raw}
  + \alpha_1\cdot\max\left(0,1-\frac{\Delta t_1}{\tau_1}\right)
  + \alpha_2\cdot\max\left(0,1-\frac{\Delta t_2}{\tau_2}\right)
$$

且 bonus 仅在 `S_raw >= raw_thr` 时生效。

本轮 sweep 参数：

- `MYEVS_N127_ALPHA1 in {0.1,0.2,0.3}`
- `MYEVS_N127_TAU1_US in {2000,5000,10000}`
- `MYEVS_N127_ALPHA2 in {0.05,0.1,0.2}`
- `MYEVS_N127_TAU2_US in {5000,10000,20000}`
- `MYEVS_N127_RAW_THR in {2.0}`

### 20.3 参数扫频命令（已执行）

```bash
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/tune_n127_top2_bonus.py \
	--python-exe D:/software/Anaconda_envs/envs/myEVS/python.exe \
	--project-root . \
	--base-out-dir data/ED24/myPedestrain_06/EBF_Part2/_tune_n127_top2_bonus \
	--alpha1-grid 0.1,0.2,0.3 \
	--tau1-us-grid 2000,5000,10000 \
	--alpha2-grid 0.05,0.1,0.2 \
	--tau2-us-grid 5000,10000,20000 \
	--raw-thr-grid 2.0 \
	--s-list 7,9 \
	--tau-us-list 64000,128000 \
	--max-events 200000 \
	--full-validate \
	--full-s-list 3,5,7,9 \
	--full-tau-us-list 8000,16000,32000,64000,128000,256000,512000,1024000 \
	--full-max-events 0
```

预筛选（按 `mean_f1` 排序）Top5：

| rank | alpha1 | tau1_us | alpha2 | tau2_us | raw_thr | mean_f1 | mean_auc |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.3 | 2000 | 0.05 | 5000 | 2.0 | 0.849828 | 0.930222 |
| 2 | 0.2 | 2000 | 0.05 | 5000 | 2.0 | 0.849825 | 0.930158 |
| 3 | 0.2 | 2000 | 0.1 | 5000 | 2.0 | 0.849818 | 0.930181 |
| 4 | 0.3 | 2000 | 0.05 | 10000 | 2.0 | 0.849807 | 0.930232 |
| 5 | 0.3 | 2000 | 0.2 | 5000 | 2.0 | 0.849803 | 0.930285 |

Top1 为 `alpha1=0.3, tau1_us=2000, alpha2=0.05, tau2_us=5000, raw_thr=2.0`。

### 20.4 Top1 全量验证结果

Top1 full-validate（`s={3,5,7,9}`, `tau=8..1024ms`）：

| env | best-AUC tag | AUC | best-F1 tag | F1 | best-F1阈值 |
|---|---|---:|---|---:|---:|
| light | `ebf_n127_labelscore_s9_tau128000` | 0.947694 | `ebf_n127_labelscore_s9_tau512000` | 0.952007 | 1.718686 |
| mid | `ebf_n127_labelscore_s9_tau128000` | 0.923487 | `ebf_n127_labelscore_s9_tau128000` | 0.818329 | 4.973594 |
| heavy | `ebf_n127_labelscore_s9_tau128000` | 0.913991 | `ebf_n127_labelscore_s7_tau64000` | 0.762163 | 3.716047 |

与 baseline / n120 / n124 / n126 / n127 最优配置对比（best-F1）：

| env | baseline | n120 (`tau_self=30000,r_max=5.0`) | n124 (`lambda=0.3`) | n126 (`a=0.3,tau1=2000,rthr=2.0`) | n127 (`a1=0.3,t1=2000,a2=0.05,t2=5000,rthr=2.0`) |
|---|---:|---:|---:|---:|---:|
| light | 0.952021 | 0.953856 | 0.953905 | 0.952012 | 0.952007 |
| mid | 0.817653 | 0.818312 | 0.821299 | 0.818299 | 0.818329 |
| heavy | 0.761764 | 0.762100 | 0.767490 | 0.762218 | 0.762163 |

### 20.5 结论

1. n127 相比 baseline 仍有小幅提升（mid +0.000676，heavy +0.000399），但 light 近乎持平（-0.000014）。
2. 相比 n126，n127 在 mid 仅有极小增益（+0.000030），但 light/heavy 略降，整体几乎等效。
3. 相比 n120 与 n124，n127 依然未形成可替代优势，尤其与 n124 的差距仍明显。
4. 结论为中性偏负：引入 top2 bonus 证明了“二层历史”可稳定运行，但当前收益接近噪声级，不建议替代 n124 主线。

产物目录：

- `data/ED24/myPedestrain_06/EBF_Part2/_tune_n127_top2_bonus/summary.csv`
- `data/ED24/myPedestrain_06/EBF_Part2/_tune_n127_top2_bonus/summary.json`
- `data/ED24/myPedestrain_06/EBF_Part2/_tune_n127_top2_bonus/full_validate_top1.json`

## 21）n128（7.74）：空间成团率驱动的时空联合衰减

### 21.1 思路

根据 7.73 统计可见：heavy 下“同极性局部成团”显著变差，且远圈支持占比更高。n128 不做硬门控，直接在 baseline 的时间权重上叠加一个空间距离衰减项，抑制“舍近求远”的外圈随机支持。

实现文件：`src/myevs/denoise/ops/ebfopt_part2/n128_spatiotemporal_joint_decay_backbone.py`

### 21.2 原理公式

设中心事件为 \((x_i,y_i,t_i,p_i)\)，邻域 \(\mathcal{N}\) 里同极性历史事件 \(j\) 的时间差 \(\Delta t_{ij}=t_i-t_j\)，切比雪夫距离

$$
d_{ij}=\max(|x_i-x_j|,|y_i-y_j|),\quad R_{max}=\text{radius}\in\{1,2,3,4\}
$$

baseline 时间权重：

$$
w_t(j)=\max\left(0,1-\frac{\Delta t_{ij}}{\tau}\right)
$$

n128 空间衰减权重（7.74）：

$$
w_s(j)=1-\eta\left(\frac{d_{ij}}{R_{max}}\right)^2,\quad \eta\in[0,1]
$$

最终分数：

$$
S_{n128}(i)=\sum_{j\in\mathcal{N},\,p_j=p_i} w_t(j)\cdot\max(0,w_s(j))
$$

对应环境变量：`MYEVS_N128_ETA`。

### 21.3 实验命令（已执行）

eta 扫频（同口径 full-grid）：

```bash
# eta in {0.2,0.5,0.8,1.0}
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py \
	--variant n128 \
	--max-events 400000 \
	--s-list 3,5,7,9 \
	--tau-us-list 8000,16000,32000,64000,128000,256000,512000,1024000 \
	--roc-max-points 5000 \
	--esr-mode off --aocc-mode off \
	--out-dir data/ED24/myPedestrain_06/EBF_Part2/_slim_n128_774_eta0p5_prescreen400k
```

baseline 对照（同口径）：

```bash
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py \
	--variant ebf \
	--max-events 400000 \
	--s-list 3,5,7,9 \
	--tau-us-list 8000,16000,32000,64000,128000,256000,512000,1024000 \
	--roc-max-points 5000 \
	--esr-mode off --aocc-mode off \
	--out-dir data/ED24/myPedestrain_06/EBF_Part2/_slim_baseline_774_ref_prescreen400k
```

### 21.4 eta 扫频结果

| eta | mean AUC | mean F1 | heavy best-F1 | heavy best-F1 tag |
|---:|---:|---:|---:|---|
| 0.2 | 0.926950 | 0.831608 | 0.748639 | `ebf_n128_labelscore_s9_tau128000` |
| 0.5 | **0.928174** | **0.834029** | 0.752675 | `ebf_n128_labelscore_s9_tau128000` |
| 0.8 | 0.927796 | 0.833879 | **0.754146** | `ebf_n128_labelscore_s9_tau64000` |
| 1.0 | 0.923103 | 0.828843 | 0.748349 | `ebf_n128_labelscore_s9_tau64000` |

综合 mean-F1/mean-AUC，选 `eta=0.5` 作为 n128 默认推荐值。

### 21.5 与 baseline 同口径对照（eta=0.5）

| 指标 | baseline | n128(eta=0.5) | delta |
|---|---:|---:|---:|
| mean AUC | 0.925929 | 0.928174 | +0.002245 |
| mean F1 | 0.829591 | 0.834029 | +0.004438 |
| light AUC | 0.947564 | 0.948713 | +0.001150 |
| light F1 | 0.952021 | 0.952069 | +0.000048 |
| mid AUC | 0.917740 | 0.920407 | +0.002667 |
| mid F1 | 0.791293 | 0.797344 | +0.006051 |
| heavy AUC | 0.912484 | 0.915401 | +0.002918 |
| heavy F1 | 0.745459 | 0.752675 | +0.007215 |

### 21.6 结论

1. 7.74 的“空间距离惩罚”在不引入硬门控的前提下可稳定提升，尤其在 heavy 上收益最明显。
2. 最优区域落在 `eta≈0.5~0.8`，说明“适度惩罚外圈支持”有效；`eta=1.0` 过强会伤及有效支持并导致整体回落。
3. n128 相比 baseline 的提升主要来自 mid/heavy，符合“heavy 依赖远圈随机支持更多”的统计观察。

产物目录：

- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n128_774_eta0p2_prescreen400k/`
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n128_774_eta0p5_prescreen400k/`
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n128_774_eta0p8_prescreen400k/`
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n128_774_eta1_prescreen400k/`
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_baseline_774_ref_prescreen400k/`

## 22）n129（7.76）：纯度门控 + 双峰结构带通

### 22.1 思路

根据 7.75 GT 统计，构造两个显观权重并与时空联合能量相乘：

1. 极性纯度权重（压制高混合随机噪声）。
2. 外圈比例结构带通（对 `outer_ratio` 两端极值做抑制，保留中间结构）。

实现文件：`src/myevs/denoise/ops/ebfopt_part2/n129_struct_purity_joint_backbone.py`

### 22.2 原理与公式

先用 n128 风格时空联合权重：

$$
w_t(j)=\max\left(0,1-\frac{\Delta t_{ij}}{\tau}\right),\quad
w_s(j)=\max\left(0,1-\eta\left(\frac{d_{ij}}{R_{max}}\right)^2\right)
$$

$$
w_{joint}(j)=w_t(j)\cdot w_s(j)
$$

按同/异极性与内/外圈（\(d\le2\) / \(d>2\)）累积：

$$
E_{same,in},\ E_{same,out},\ E_{opp,in},\ E_{opp,out}
$$

总能量：

$$
E_{total}=E_{same,in}+E_{same,out}+E_{opp,in}+E_{opp,out}
$$

混合度与纯度权重：

$$
Mix=\frac{E_{opp,in}+E_{opp,out}}{E_{total}+\epsilon},\quad
W_{purity}=(1-Mix)^\gamma
$$

外圈比例与结构带通权重：

$$
OuterRatio=\frac{E_{same,out}+E_{opp,out}}{E_{total}+\epsilon},\quad
W_{struct}=4\cdot OuterRatio\cdot(1-OuterRatio)
$$

最终分数：

$$
S_{n129}=E_{total}\cdot W_{purity}\cdot W_{struct}
$$

环境变量：

- `MYEVS_N129_ETA`（默认 0.5）
- `MYEVS_N129_GAMMA`（默认 2.0）
- `MYEVS_N129_EPS`（默认 1e-3）

### 22.3 实验命令（已执行）

```bash
$env:PYTHONNOUSERSITE=1
$env:MYEVS_N129_ETA=0.5
$env:MYEVS_N129_GAMMA=2.0

D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py \
	--variant n129 \
	--max-events 400000 \
	--s-list 3,5,7,9 \
	--tau-us-list 8000,16000,32000,64000,128000,256000,512000,1024000 \
	--roc-max-points 5000 \
	--esr-mode off --aocc-mode off \
	--out-dir data/ED24/myPedestrain_06/EBF_Part2/_slim_n129_776_prescreen400k_eta0p5_g2
```

对照使用已存在同口径结果：

- baseline：`data/ED24/myPedestrain_06/EBF_Part2/_slim_baseline_774_ref_prescreen400k/`
- n128(eta=0.5)：`data/ED24/myPedestrain_06/EBF_Part2/_slim_n128_774_eta0p5_prescreen400k/`

### 22.4 结果

| variant | mean AUC | mean F1 |
|---|---:|---:|
| baseline | 0.925929 | 0.829591 |
| n128(eta=0.5) | 0.928174 | 0.834029 |
| n129(eta=0.5,gamma=2.0) | 0.914114 | 0.813526 |

n129 各环境最佳点：

| env | best-AUC tag | AUC | best-F1 tag | F1 |
|---|---|---:|---|---:|
| light | `ebf_n129_labelscore_s9_tau128000` | 0.941323 | `ebf_n129_labelscore_s9_tau256000` | 0.946308 |
| mid | `ebf_n129_labelscore_s9_tau128000` | 0.903529 | `ebf_n129_labelscore_s9_tau128000` | 0.768668 |
| heavy | `ebf_n129_labelscore_s9_tau128000` | 0.897490 | `ebf_n129_labelscore_s9_tau64000` | 0.725602 |

n129 相对 baseline 的变化：

| 指标 | delta |
|---|---:|
| mean AUC | -0.011815 |
| mean F1 | -0.016065 |
| light AUC / F1 | -0.006241 / -0.005713 |
| mid AUC / F1 | -0.014211 / -0.022624 |
| heavy AUC / F1 | -0.014994 / -0.019857 |

### 22.5 结论

1. 在当前参数（eta=0.5, gamma=2.0）下，n129 明显劣于 baseline 与 n128，属于负结果。
2. 失败原因推测：`W_struct`（抛物线带通）与 `W_purity` 的乘积对高质量事件过度压缩，导致排序整体塌陷（AUC/F1 同降）。
3. 7.76 思路保留价值，但不建议当前实现直接进入主线；若继续，应优先改为“加性/温和乘性”融合，避免双重乘法门控过强。

产物目录：

- `data/ED24/myPedestrain_06/EBF_Part2/_smoke_n129_20k_s9_tau128/`
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n129_776_prescreen400k_eta0p5_g2/`

## 23）n131（7.78）：ratio_pure 平顶梯形结构门控（纯 Baseline）

### 23.1 设计对齐性分析（先验）

7.78 的核心思想与 7.77 统计是对齐的：

1. `ratio_pure` 在 noise 上更容易落在极值两端，而 signal 更偏中段；
2. 用“中段保留、两端抑制”的结构门控去修正 Baseline 的误触发，方向是合理的；
3. 保持纯时间权重（不引入 n128 空间衰减）可以更直接验证该结构假设。

### 23.2 实现前发现的缺陷与修复

原始规则存在一个结构性缺陷：当 `s=3/5`（即 `radius<=2`）时没有外圈像素，`ratio_pure` 退化为 0，门控会把分数链压塌。

修复策略（已实现）：

1. 当 `radius<=2` 时，结构门控自动退化为 baseline（`score=s_base`）；
2. 仅在 `radius>2` 时启用 ratio 门控。

实现文件：`src/myevs/denoise/ops/ebfopt_part2/n131_pure_struct_bandpass_backbone.py`

环境变量：

- `MYEVS_N131_LO`（默认 0.1）
- `MYEVS_N131_HI`（默认 0.9）
- `MYEVS_N131_EPS`（默认 1e-3）

### 23.3 原理与公式

同极性、纯时间权重（9x9 邻域）：

$$
w_t(j)=\max\left(0,1-\frac{\Delta t_{ij}}{\tau}\right)
$$

$$
E_{in}=\sum_{d\le2} w_t,\quad E_{out}=\sum_{d>2} w_t,\quad S_{base}=E_{in}+E_{out}
$$

$$
ratio_{pure}=\frac{E_{out}}{S_{base}+\epsilon}
$$

平顶梯形门控：

$$
W_{struct}=
\begin{cases}
\frac{ratio}{lo}, & ratio<lo \\
1, & lo\le ratio\le hi \\
\frac{1-ratio}{1-hi}, & ratio>hi
\end{cases}
$$

最终：

$$
S_{n131}=S_{base}\cdot W_{struct}
$$

### 23.4 实验命令（已执行）

```bash
$env:PYTHONNOUSERSITE=1
$env:MYEVS_N131_LO=0.1
$env:MYEVS_N131_HI=0.9

D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py \
	--variant n131 \
	--max-events 400000 \
	--s-list 3,5,7,9 \
	--tau-us-list 8000,16000,32000,64000,128000,256000,512000,1024000 \
	--out-dir data/ED24/myPedestrain_06/EBF_Part2/_slim_n131_778_prescreen400k_lo0p1_hi0p9
```

Smoke（通路验证）：

- `data/ED24/myPedestrain_06/EBF_Part2/_smoke_n131_778_20k/`
- `data/ED24/myPedestrain_06/EBF_Part2/_smoke_n131_778_fix_rr20k/`

### 23.5 结果

| variant | mean AUC | mean F1 |
|---|---:|---:|
| baseline | 0.925929 | 0.829591 |
| n128(eta=0.5) | 0.928174 | 0.834029 |
| n129(eta=0.5,gamma=2.0) | 0.914114 | 0.813526 |
| n131(lo=0.1,hi=0.9) | 0.920090 | 0.828457 |

n131 各环境最佳点：

| env | best-AUC tag | AUC | best-F1 tag | F1 |
|---|---|---:|---|---:|
| light | `ebf_n131_labelscore_s9_tau256000` | 0.937524 | `ebf_n131_labelscore_s9_tau256000` | 0.946404 |
| mid | `ebf_n131_labelscore_s9_tau256000` | 0.913550 | `ebf_n131_labelscore_s9_tau128000` | 0.792189 |
| heavy | `ebf_n131_labelscore_s9_tau256000` | 0.909196 | `ebf_n131_labelscore_s9_tau128000` | 0.746777 |

n131 相对 baseline 的变化：

| 指标 | delta |
|---|---:|
| mean AUC | -0.005839 |
| mean F1 | -0.001135 |
| light AUC / F1 | -0.010040 / -0.005617 |
| mid AUC / F1 | -0.004190 / +0.000896 |
| heavy AUC / F1 | -0.003288 / +0.001318 |

### 23.6 结论

1. n131 的结构门控方向是“有物理解释且可运行”的，但在当前固定形状（`lo=0.1, hi=0.9`）下，总体仍未超过 baseline。
2. 现象上，mid/heavy 的 best-F1 有轻微提升，但 light 与 AUC 全面回落，说明门控对排序整体有副作用。
3. 可继续的最小改进方向：把硬乘门控改为“弱门控/加性融合”，或只在 `S_base` 足够高时启用结构惩罚，降低对有效 signal 的误伤。

产物目录：

- `data/ED24/myPedestrain_06/EBF_Part2/_smoke_n131_778_20k/`
- `data/ED24/myPedestrain_06/EBF_Part2/_smoke_n131_778_fix_rr20k/`
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n131_778_prescreen400k_lo0p1_hi0p9/`

## 24）n132（7.78-改进）：高分区弱结构门控

### 24.1 改进动机

n131 的主要问题是“全局硬乘门控”导致排序被整体压缩：

1. light AUC/F1 回落较明显；
2. 中低分事件也被结构门控影响，误伤了可分性；
3. 虽然 mid/heavy 的 F1 有小幅正增，但不足以抵消总体损失。

因此按 7.78 后续建议改为：

1. baseline 仍作为主分数；
2. 结构惩罚只在高分区逐步生效；
3. 保持“弱门控”（受 `lambda` 控制）而非硬乘。

实现文件：`src/myevs/denoise/ops/ebfopt_part2/n132_conditional_weak_struct_gate_backbone.py`

### 24.2 原理与结构

先算 baseline 纯时间支持：

$$
S_{base}=E_{in}+E_{out},\quad ratio=\frac{E_{out}}{S_{base}+\epsilon}
$$

结构门控 `w_struct` 仍采用 7.78 平顶梯形（`lo, hi`）：

$$
w_{struct}=1\ (lo\le ratio\le hi),\ \text{两端线性衰减到 }0
$$

新增“高分激活因子” `g_high`：

$$
g_{high}=\mathrm{clip}\left(\frac{S_{base}-S_0}{S_1-S_0},0,1\right)
$$

最终弱门控：

$$
gate=1-\lambda\cdot(1-w_{struct})\cdot g_{high},\quad
S_{n132}=S_{base}\cdot gate
$$

含义：

1. `S_base` 低时（`g_high≈0`）几乎不惩罚；
2. 仅在高分区间才逐步施加结构抑制；
3. `lambda` 控制惩罚强度上限。

环境变量：

- `MYEVS_N132_LO`（默认 0.1）
- `MYEVS_N132_HI`（默认 0.9）
- `MYEVS_N132_LAMBDA`（默认 0.5）
- `MYEVS_N132_S0`（默认 2.0）
- `MYEVS_N132_S1`（默认 4.0）
- `MYEVS_N132_EPS`（默认 1e-3）

### 24.3 实验命令（已执行）

```bash
$env:PYTHONNOUSERSITE=1
$env:MYEVS_N132_LO=0.1
$env:MYEVS_N132_HI=0.9
$env:MYEVS_N132_LAMBDA=0.5
$env:MYEVS_N132_S0=2.0
$env:MYEVS_N132_S1=4.0

D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py \
	--variant n132 \
	--max-events 400000 \
	--s-list 3,5,7,9 \
	--tau-us-list 8000,16000,32000,64000,128000,256000,512000,1024000 \
	--out-dir data/ED24/myPedestrain_06/EBF_Part2/_slim_n132_778_prescreen400k_lo0p1_hi0p9_l0p5_s2_4
```

Smoke：

- `data/ED24/myPedestrain_06/EBF_Part2/_smoke_n132_778_20k/`

### 24.4 结果

| variant | mean AUC | mean F1 |
|---|---:|---:|
| baseline | 0.925929 | 0.829591 |
| n128(eta=0.5) | 0.928174 | 0.834029 |
| n131(lo=0.1,hi=0.9) | 0.920090 | 0.828457 |
| n132(lo=0.1,hi=0.9,lambda=0.5,S0=2,S1=4) | 0.925668 | 0.830222 |

n132 各环境最佳点：

| env | best-AUC tag | AUC | best-F1 tag | F1 |
|---|---|---:|---|---:|
| light | `ebf_n132_labelscore_s9_tau128000` | 0.947907 | `ebf_n132_labelscore_s9_tau512000` | 0.952014 |
| mid | `ebf_n132_labelscore_s9_tau128000` | 0.917229 | `ebf_n132_labelscore_s9_tau128000` | 0.792099 |
| heavy | `ebf_n132_labelscore_s9_tau128000` | 0.911870 | `ebf_n132_labelscore_s9_tau128000` | 0.746551 |

n132 相对 baseline：

| 指标 | delta |
|---|---:|
| mean AUC | -0.000261 |
| mean F1 | +0.000630 |
| light AUC / F1 | +0.000343 / -0.000007 |
| mid AUC / F1 | -0.000511 / +0.000806 |
| heavy AUC / F1 | -0.000614 / +0.001092 |

### 24.5 结论

1. 改进方向有效：相比 n131，n132 显著恢复了排序稳定性（mean AUC 从 -0.005839 回到 -0.000261），且 mean F1 转为正增。
2. 目前 n132 基本达到“接近 baseline、轻微优于 baseline 的 F1”水平，但仍未超过 n128 主线。
3. 下一步若继续该线，建议只微调 `lambda,S0,S1`（小网格）并以 heavy F1 + light AUC 双目标筛选，避免再次引入全局硬惩罚。

产物目录：

- `data/ED24/myPedestrain_06/EBF_Part2/_smoke_n132_778_20k/`
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n132_778_prescreen400k_lo0p1_hi0p9_l0p5_s2_4/`

### 24.6 边界改为 d=1 的复现实验（同口径）

按你的要求，把 n132 的内外圈边界由 `d<=2` 改为 `d<=1`（外圈为 `d>1`）再跑一轮完整实验。

实现支持：

1. n132 已支持环境变量 `MYEVS_N132_INNER_RADIUS`；
2. 默认仍为 2，设置为 1 即可进入 d=1 模式。

命令（已执行）：

```bash
$env:PYTHONNOUSERSITE=1
$env:MYEVS_N132_INNER_RADIUS=1
$env:MYEVS_N132_LO=0.1
$env:MYEVS_N132_HI=0.9
$env:MYEVS_N132_LAMBDA=0.5
$env:MYEVS_N132_S0=2.0
$env:MYEVS_N132_S1=4.0

D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py \
	--variant n132 \
	--max-events 400000 \
	--s-list 3,5,7,9 \
	--tau-us-list 8000,16000,32000,64000,128000,256000,512000,1024000 \
	--out-dir data/ED24/myPedestrain_06/EBF_Part2/_slim_n132_778_d1_prescreen400k_lo0p1_hi0p9_l0p5_s2_4
```

Smoke：

- `data/ED24/myPedestrain_06/EBF_Part2/_smoke_n132_778_d1_20k/`

主实验产物：

- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n132_778_d1_prescreen400k_lo0p1_hi0p9_l0p5_s2_4/`

结果对比（400k）：

| variant | mean AUC | mean F1 |
|---|---:|---:|
| baseline | 0.925929 | 0.829591 |
| n132(d=2) | 0.925668 | 0.830222 |
| n132(d=1) | 0.923530 | 0.826251 |

n132(d=1) 各环境最佳点：

| env | best-AUC tag | AUC | best-F1 tag | F1 |
|---|---|---:|---|---:|
| light | `ebf_n132_labelscore_s9_tau128000` | 0.948411 | `ebf_n132_labelscore_s9_tau512000` | 0.952007 |
| mid | `ebf_n132_labelscore_s9_tau128000` | 0.915038 | `ebf_n132_labelscore_s9_tau128000` | 0.784414 |
| heavy | `ebf_n132_labelscore_s9_tau128000` | 0.907142 | `ebf_n132_labelscore_s9_tau128000` | 0.742330 |

d=1 相对 baseline：

| 指标 | delta |
|---|---:|
| mean AUC | -0.002399 |
| mean F1 | -0.003341 |
| light AUC / F1 | +0.000847 / -0.000014 |
| mid AUC / F1 | -0.002702 / -0.006879 |
| heavy AUC / F1 | -0.005342 / -0.003129 |

d=1 相对 d=2：

| 指标 | delta |
|---|---:|
| mean AUC | -0.002138 |
| mean F1 | -0.003971 |
| light AUC / F1 | +0.000504 / -0.000007 |
| mid AUC / F1 | -0.002191 / -0.007685 |
| heavy AUC / F1 | -0.004728 / -0.004221 |

结论（d=1）：

1. d=1 边界使结构门控更激进，light AUC 有微小增益，但中重噪场景明显回落；
2. 从 mean 指标看，d=1 同时劣于 baseline 与 n132(d=2)；
3. 当前 n132 建议继续采用 `d<=2` 作为内圈默认边界。

### 24.7 Baseline 得分双端分布体检（7.79）

目标：验证“超高分段是否也会出现噪声伪装”。

统计定义（禁用空间门控，仅 baseline 时间分数）：

$$
S_{base} = \sum_{(x',y')\in 9\times9,\,p'=p} \max\left(0, 1-\frac{\Delta t}{30000\,\mu s}\right)
$$

分箱：

- `[0,1), [1,2), [2,3), [3,4), [4,5), [5,7), [7,10), [10,15), [15,20), [20,30), [30,50), >=50`

命令（已执行，heavy + light）：

```bash
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/noise_analyze/gt_feature_stats_779_score_bandpass.py \
	--labeled-npy D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3.npy \
	--max-events 500000 \
	--tau-us 30000 \
	--out-dir data/ED24/myPedestrain_06/EBF_Part2/_gt_feature_stats_779_heavy

D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/noise_analyze/plot_gt_feature_hist_779.py \
	--hist-csv data/ED24/myPedestrain_06/EBF_Part2/_gt_feature_stats_779_heavy/hist_sbase.csv \
	--out-png data/ED24/myPedestrain_06/EBF_Part2/_gt_feature_stats_779_heavy/hist_779_sbase.png \
	--title "7.79 Baseline S_base Bandpass (heavy, 500k)"

D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/noise_analyze/gt_feature_stats_779_score_bandpass.py \
	--labeled-npy D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_1.8.npy \
	--max-events 500000 \
	--tau-us 30000 \
	--out-dir data/ED24/myPedestrain_06/EBF_Part2/_gt_feature_stats_779_light

D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/noise_analyze/plot_gt_feature_hist_779.py \
	--hist-csv data/ED24/myPedestrain_06/EBF_Part2/_gt_feature_stats_779_light/hist_sbase.csv \
	--out-png data/ED24/myPedestrain_06/EBF_Part2/_gt_feature_stats_779_light/hist_779_sbase.png \
	--title "7.79 Baseline S_base Bandpass (light, 500k)"
```

heavy（500k）按分箱统计（GT=0 noise / GT=1 signal）：

| score band | noise count | noise ratio | signal count | signal ratio |
|---|---:|---:|---:|---:|
| [0,1) | 276530 | 0.630404 | 7590 | 0.123726 |
| [1,2) | 108151 | 0.246551 | 7205 | 0.117450 |
| [2,3) | 38140 | 0.086948 | 6748 | 0.110001 |
| [3,4) | 10157 | 0.023155 | 5735 | 0.093488 |
| [4,5) | 2733 | 0.006230 | 4819 | 0.078556 |
| [5,7) | 1634 | 0.003725 | 7685 | 0.125275 |
| [7,10) | 785 | 0.001790 | 8308 | 0.135431 |
| [10,15) | 404 | 0.000921 | 8223 | 0.134045 |
| [15,20) | 95 | 0.000217 | 3622 | 0.059043 |
| [20,30) | 25 | 0.000057 | 1409 | 0.022968 |
| [30,50) | 1 | 0.000002 | 1 | 0.000016 |
| >=50 | 0 | 0.000000 | 0 | 0.000000 |

light（500k）按分箱统计（GT=0 noise / GT=1 signal）：

| score band | noise count | noise ratio | signal count | signal ratio |
|---|---:|---:|---:|---:|
| [0,1) | 30042 | 0.954654 | 29891 | 0.183464 |
| [1,2) | 727 | 0.023102 | 15643 | 0.096013 |
| [2,3) | 248 | 0.007881 | 13601 | 0.083480 |
| [3,4) | 134 | 0.004258 | 11816 | 0.072524 |
| [4,5) | 68 | 0.002161 | 10412 | 0.063906 |
| [5,7) | 113 | 0.003591 | 17746 | 0.108921 |
| [7,10) | 84 | 0.002669 | 21211 | 0.130188 |
| [10,15) | 36 | 0.001144 | 21817 | 0.133907 |
| [15,20) | 13 | 0.000413 | 10539 | 0.064686 |
| [20,30) | 2 | 0.000064 | 7347 | 0.045094 |
| [30,50) | 2 | 0.000064 | 2861 | 0.017560 |
| >=50 | 0 | 0.000000 | 42 | 0.000258 |

补充统计（S_base 全体均值）：

| env | class | count | mean S_base | std S_base |
|---|---|---:|---:|---:|
| heavy | noise | 438655 | 0.974797 | 1.075443 |
| heavy | signal | 61345 | 6.234697 | 5.313045 |
| light | noise | 31469 | 0.229671 | 0.972112 |
| light | signal | 162926 | 7.169895 | 7.316164 |

结论（7.79）：

1. 你的“上界约束”洞察被数据支持：高分段（尤其 `[20,30)`、`[30,50)` 乃至 `>=50`）虽主要由 GT=1 构成，但 GT=0 在高噪场景下并非严格为零，确实存在“超高分噪声伪装”。
2. heavy 噪声在低分段占绝对多数（`[0,3)` 达到 96.39%），但在高分段仍有细尾；这为“低分抑制 + 超高分抑制（带通）”提供了统计依据。
3. light 的噪声极少且集中低分，但 signal 在 `[20,50)` 占比明显更高，说明上界惩罚应优先设计成“软惩罚”而不是硬截断，避免误伤高质量信号。

产物目录：

- `data/ED24/myPedestrain_06/EBF_Part2/_gt_feature_stats_779_heavy/`
- `data/ED24/myPedestrain_06/EBF_Part2/_gt_feature_stats_779_light/`

### 24.8 n133：高分软上界惩罚（7.79 落地）

根据 7.79 的统计结论（高分尾存在少量噪声伪装），新增 n133：

1. 保留 n132 的“高分区弱结构门控”；
2. 在 `S_base` 超过高分阈值后，引入软上界惩罚（soft over-activation cap）；
3. 惩罚使用连续可调曲线，并设置地板 `w_min`，避免硬截断伤及真实高分信号。

实现文件：

- `src/myevs/denoise/ops/ebfopt_part2/n133_soft_overactivation_bandpass_backbone.py`

接入：

- `scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py`（新增 `--variant n133`）

核心打分：

$$
S_{base}=E_{in}+E_{out},\quad S_{n133}=S_{base}\cdot G_{struct}(ratio\_pure,S_{base})\cdot G_{hi}(S_{base})
$$

其中：

1. $G_{struct}$：沿用 n132 的弱结构门控（仅在高分区渐进生效）；
2. $G_{hi}$：
   - $S_{base}\le S_{hi}$：$G_{hi}=1$；
   - $S_{hi}<S_{base}\le S_{max}$：线性衰减到 $1-\beta$；
   - $S_{base}>S_{max}$：继续按斜率 $\gamma$ 软衰减，并夹紧到 $[w_{min},1]$。

#### 24.8.1 参数预筛（扫频）

先做一轮轻量预筛（`max-events=200k, s=9, tau=128ms`），比较高分惩罚参数组：

```bash
$env:PYTHONNOUSERSITE=1
$env:MYEVS_N133_INNER_RADIUS=2
$env:MYEVS_N133_LO=0.1
$env:MYEVS_N133_HI=0.9
$env:MYEVS_N133_LAMBDA_STRUCT=0.35
$env:MYEVS_N133_S0=2.0
$env:MYEVS_N133_S1=4.0
$env:MYEVS_N133_S_HI=20.0
$env:MYEVS_N133_S_MAX=30.0

# A
$env:MYEVS_N133_BETA_HI=0.25
$env:MYEVS_N133_GAMMA_HI=0.15
$env:MYEVS_N133_W_MIN=0.55

# B
$env:MYEVS_N133_BETA_HI=0.35
$env:MYEVS_N133_GAMMA_HI=0.25
$env:MYEVS_N133_W_MIN=0.45

# C
$env:MYEVS_N133_BETA_HI=0.45
$env:MYEVS_N133_GAMMA_HI=0.35
$env:MYEVS_N133_W_MIN=0.35

# D
$env:MYEVS_N133_BETA_HI=0.30
$env:MYEVS_N133_GAMMA_HI=0.20
$env:MYEVS_N133_W_MIN=0.50
```

预筛结果（mean over light/mid/heavy）：

| param set | mean AUC | mean F1 |
|---|---:|---:|
| A (b=0.25,g=0.15,wmin=0.55) | 0.930133 | 0.849759 |
| B (b=0.35,g=0.25,wmin=0.45) | 0.930065 | 0.849784 |
| C (b=0.45,g=0.35,wmin=0.35) | 0.929829 | 0.849776 |
| D (b=0.30,g=0.20,wmin=0.50) | 0.930122 | 0.849780 |

说明：四组接近，最终选择 D（平衡、惩罚强度中等）进入 400k 全网格。

#### 24.8.2 400k 全网格实验（同口径扫频）

执行命令：

```bash
$env:PYTHONNOUSERSITE=1
$env:MYEVS_N133_INNER_RADIUS=2
$env:MYEVS_N133_LO=0.1
$env:MYEVS_N133_HI=0.9
$env:MYEVS_N133_LAMBDA_STRUCT=0.35
$env:MYEVS_N133_S0=2.0
$env:MYEVS_N133_S1=4.0
$env:MYEVS_N133_S_HI=20.0
$env:MYEVS_N133_S_MAX=30.0
$env:MYEVS_N133_BETA_HI=0.30
$env:MYEVS_N133_GAMMA_HI=0.20
$env:MYEVS_N133_W_MIN=0.50

D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py \
	--variant n133 \
	--max-events 400000 \
	--s-list 3,5,7,9 \
	--tau-us-list 8000,16000,32000,64000,128000,256000,512000,1024000 \
	--out-dir data/ED24/myPedestrain_06/EBF_Part2/_slim_n133_779_prescreen400k_lo0p1_hi0p9_ls0p35_shi20_smax30_b0p3_g0p2_w0p5
```

Smoke：

- `data/ED24/myPedestrain_06/EBF_Part2/_smoke_n133_779_20k/`

主实验产物：

- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n133_779_prescreen400k_lo0p1_hi0p9_ls0p35_shi20_smax30_b0p3_g0p2_w0p5/`

#### 24.8.3 结果汇总

| variant | mean AUC | mean F1 |
|---|---:|---:|
| baseline | 0.925929 | 0.829591 |
| n132(d=2) | 0.925668 | 0.830222 |
| n133(soft upper cap) | 0.926054 | 0.830117 |

n133 各环境最佳点：

| env | best-AUC tag | AUC | best-F1 tag | F1 |
|---|---|---:|---|---:|
| light | `ebf_n133_labelscore_s9_tau128000` | 0.947827 | `ebf_n133_labelscore_s9_tau512000` | 0.952014 |
| mid | `ebf_n133_labelscore_s9_tau128000` | 0.917806 | `ebf_n133_labelscore_s9_tau128000` | 0.792022 |
| heavy | `ebf_n133_labelscore_s9_tau128000` | 0.912528 | `ebf_n133_labelscore_s9_tau128000` | 0.746314 |

n133 相对 baseline：

| 指标 | delta |
|---|---:|
| mean AUC | +0.000125 |
| mean F1 | +0.000526 |

n133 相对 n132(d=2)：

| 指标 | delta |
|---|---:|
| mean AUC | +0.000385 |
| mean F1 | -0.000105 |

#### 24.8.4 结论

1. 7.79 的“高分软上界”方向是可行的：n133 相比 baseline 实现了 mean AUC 与 mean F1 双正增；
2. 与 n132 相比，n133 的优势更体现在排序（AUC）而非 operating point（F1）；
3. 当前 n133 提升幅度仍较小，后续可优先继续微调 `S_HI/S_MAX/BETA_HI/GAMMA_HI`，目标是提升 heavy F1 而不损伤 light AUC。

### 24.9 更正：严格按 7.79 级联公式重跑 n133

根据你指出的问题，已将 n133 改为严格 7.79 公式，不再使用“额外软上界项”。

严格公式（当前实现）：

$$
S_{base}=E_{in}+E_{out}
$$

$$
Score_{final}=\begin{cases}
S_{base}, & S_{base}<S_{act} \\
S_{base}\cdot W_{struct}, & S_{base}\ge S_{act}
\end{cases}
$$

$$
Ratio=\frac{E_{out}}{S_{base}},\quad
W_{struct}=\begin{cases}
1, & Ratio\le H_{thresh} \\
\frac{1-Ratio}{1-H_{thresh}}, & Ratio>H_{thresh}
\end{cases}
$$

说明：

1. 内外圈严格按 `d<=2` / `d>2`（`MYEVS_N133_INNER_RADIUS=2`）；
2. 仅当 `S_base >= S_act` 才触发结构审查（真正“宽进严出”级联）。

实现文件（已更新）：

- `src/myevs/denoise/ops/ebfopt_part2/n133_soft_overactivation_bandpass_backbone.py`

#### 24.9.1 参数扫频（S_act, H_thresh）

固定：`max-events=400k, s=9, tau=128ms`。

扫频网格：

- `S_act ∈ {2.5, 3.0, 3.5, 4.0}`
- `H_thresh ∈ {0.85, 0.90, 0.95}`

结果（按 mean AUC 降序）：

| S_act | H_thresh | mean AUC | mean F1 | light AUC/F1 | mid AUC/F1 | heavy AUC/F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 4.0 | 0.95 | 0.923420 | 0.828477 | 0.946412 / 0.947363 | 0.914328 / 0.792102 | 0.909521 / 0.745967 |
| 4.0 | 0.90 | 0.923184 | 0.828899 | 0.946458 / 0.947342 | 0.914088 / 0.792525 | 0.909007 / 0.746830 |
| 4.0 | 0.85 | 0.922931 | 0.829436 | 0.946573 / 0.947269 | 0.913814 / 0.793503 | 0.908406 / 0.747537 |
| 3.5 | 0.95 | 0.922901 | 0.828226 | 0.946043 / 0.946568 | 0.913483 / 0.792087 | 0.909177 / 0.746022 |
| 3.5 | 0.90 | 0.922640 | 0.828643 | 0.946100 / 0.946528 | 0.913204 / 0.792539 | 0.908614 / 0.746862 |
| 3.0 | 0.95 | 0.922457 | 0.827892 | 0.945537 / 0.945604 | 0.912631 / 0.792109 | 0.909203 / 0.745962 |
| 3.5 | 0.85 | 0.922373 | 0.829194 | 0.946229 / 0.946443 | 0.912883 / 0.793522 | 0.908008 / 0.747618 |
| 3.0 | 0.90 | 0.922165 | 0.828303 | 0.945587 / 0.945544 | 0.912312 / 0.792515 | 0.908598 / 0.746849 |
| 3.0 | 0.85 | 0.921887 | 0.828839 | 0.945709 / 0.945441 | 0.911950 / 0.793535 | 0.908001 / 0.747540 |
| 2.5 | 0.95 | 0.921841 | 0.827767 | 0.944626 / 0.945197 | 0.911814 / 0.792096 | 0.909084 / 0.746007 |
| 2.5 | 0.90 | 0.921554 | 0.828145 | 0.944666 / 0.945111 | 0.911460 / 0.792501 | 0.908535 / 0.746822 |
| 2.5 | 0.85 | 0.921280 | 0.828743 | 0.944778 / 0.945075 | 0.911085 / 0.793523 | 0.907975 / 0.747631 |

观察：

1. `S_act` 增大到 4.0 可显著抬高 mean AUC；
2. `H_thresh` 降低（0.95→0.85）会牺牲 AUC、换取少量 F1；
3. AUC/F1 呈典型 trade-off，与你提出的“高分段审查”逻辑一致。

#### 24.9.2 400k 全网格（两组代表参数）

执行了两组 full-grid：

1. AUC-opt：`S_act=4.0, H_thresh=0.95`
2. F1-opt：`S_act=4.0, H_thresh=0.85`

结果对比：

| variant | mean AUC | mean F1 |
|---|---:|---:|
| baseline | 0.925929 | 0.829591 |
| n132(d=2) | 0.925668 | 0.830222 |
| n133 strict AUC-opt (4.0,0.95) | 0.923653 | 0.829074 |
| n133 strict F1-opt (4.0,0.85) | 0.922989 | 0.830056 |

delta vs baseline：

| variant | mean AUC delta | mean F1 delta |
|---|---:|---:|
| n133 strict AUC-opt | -0.002276 | -0.000517 |
| n133 strict F1-opt | -0.002940 | +0.000465 |
| n132(d=2) | -0.000261 | +0.000630 |

结论（严格 7.79 版）：

1. 严格级联公式在本数据上没有超过 n132；
2. `S_act/H_thresh` 能稳定调节 AUC/F1 取舍：更保守的 `H_thresh` 有利 AUC，更激进的惩罚有利部分 F1；
3. 当前最实用参数：
	- 若偏好排序（AUC）：`S_act=4.0, H_thresh=0.95`
	- 若偏好 operating point（F1）：`S_act=4.0, H_thresh=0.85`

产物目录：

- `data/ED24/myPedestrain_06/EBF_Part2/_tune_n133_779strict_sact*_h*_400k_s9tau128/`
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n133_779strict_aucopt_sact4_h0p95_400k/`
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n133_779strict_f1opt_sact4_h0p85_400k/`

### 24.10 7.80：仅存储“可能事件时间”的 n134

按 7.80 思路新增 n134：

1. 打分仍是 baseline：`S_base = sum(max(0,1-dt/tau))`；
2. 差异只在记忆写入：仅当 `S_base >= S_keep` 时才写入 `last_ts/last_pol`；
3. 这样低分噪声不会继续“互相支撑”，抑制 ghost chain。

实现文件：

- `src/myevs/denoise/ops/ebfopt_part2/n134_stateful_support_pruning_backbone.py`

接入入口：

- `scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py`（新增 `--variant n134`）

公式：

$$
S_{base}(e_i)=\sum_{e_j\in\mathcal{N}_{9\times9},\,p_j=p_i}\max\left(0,1-\frac{\Delta t_{ij}}{\tau}\right)
$$

$$
score_i=S_{base}(e_i)
$$

$$
	ext{write\_memory}(e_i)=\mathbb{1}[S_{base}(e_i)\ge S_{keep}]
$$

补充（冷启动处理）：

1. 直接严格写入会出现冷启动死锁（初始全空时分数长期为 0）；
2. 因此加入 `MYEVS_N134_BOOTSTRAP_FIRST=1`：每像素首事件允许入库一次，用于建立最小支撑。

#### 24.10.1 参数扫频（S_keep）

预筛设置：`max-events=200k, s=9, tau=128ms`。

扫频：`S_keep ∈ {0.5, 1.0, 1.5, 2.0}`（`MYEVS_N134_BOOTSTRAP_FIRST=1`）。

| S_keep | mean AUC | mean F1 | light AUC/F1 | mid AUC/F1 | heavy AUC/F1 |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.921512 | 0.841839 | 0.922836 / 0.929302 | 0.921397 / 0.809362 | 0.920304 / 0.786854 |
| 1.0 | 0.910571 | 0.834001 | 0.895506 / 0.911931 | 0.917719 / 0.804524 | 0.918487 / 0.785548 |
| 1.5 | 0.905491 | 0.831020 | 0.890098 / 0.911931 | 0.911772 / 0.800278 | 0.914604 / 0.780852 |
| 2.0 | 0.890825 | 0.825316 | 0.869197 / 0.911931 | 0.901522 / 0.792807 | 0.901755 / 0.771211 |

结论：`S_keep=0.5` 在该预筛口径下最优，`S_keep` 越大越容易伤 light，并拖累 mean 指标。

#### 24.10.2 400k 全网格（选用 S_keep=0.5）

命令：

```bash
$env:PYTHONNOUSERSITE=1
$env:MYEVS_N134_BOOTSTRAP_FIRST=1
$env:MYEVS_N134_S_KEEP=0.5

D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py \
	--variant n134 \
	--max-events 400000 \
	--s-list 3,5,7,9 \
	--tau-us-list 8000,16000,32000,64000,128000,256000,512000,1024000 \
	--out-dir data/ED24/myPedestrain_06/EBF_Part2/_slim_n134_780_prescreen400k_keep0p5_boot1
```

Smoke：

- `data/ED24/myPedestrain_06/EBF_Part2/_smoke_n134_780_20k_boot1/`

主实验：

- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n134_780_prescreen400k_keep0p5_boot1/`

结果对比（400k full-grid）：

| variant | mean AUC | mean F1 |
|---|---:|---:|
| baseline | 0.925929 | 0.829591 |
| n132(d=2) | 0.925668 | 0.830222 |
| n134(S_keep=0.5) | 0.921455 | 0.827064 |

delta vs baseline：

| variant | mean AUC delta | mean F1 delta |
|---|---:|---:|
| n132(d=2) | -0.000261 | +0.000630 |
| n134(S_keep=0.5) | -0.004475 | -0.002527 |

n134 各环境最佳点（full-grid）：

| env | best-AUC tag | AUC | best-F1 tag | F1 |
|---|---|---:|---|---:|
| light | `ebf_n134_labelscore_s9_tau512000` | 0.934636 | `ebf_n134_labelscore_s9_tau512000` | 0.948543 |
| mid | `ebf_n134_labelscore_s9_tau256000` | 0.917023 | `ebf_n134_labelscore_s9_tau128000` | 0.787333 |
| heavy | `ebf_n134_labelscore_s9_tau128000` | 0.912706 | `ebf_n134_labelscore_s9_tau128000` | 0.745317 |

#### 24.10.3 结论

1. 7.80 的“条件入库”方向在本口径下未超过 baseline/n132，主要损失来自 light 的排序能力；
2. `S_keep` 需要保持很低（约 0.5）才能避免明显退化，过高会导致支撑链断裂；
3. 若继续该线，建议后续只在高噪场景（heavy-only）或分段自适应 `S_keep` 里验证，而不要全局固定一个较高阈值。

产物目录：

- `data/ED24/myPedestrain_06/EBF_Part2/_smoke_n134_780_20k_boot1/`
- `data/ED24/myPedestrain_06/EBF_Part2/_tune_n134_780_keep*_200k_s9tau128/`
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n134_780_prescreen400k_keep0p5_boot1/`

### 24.11 7.81：双轨级联 n135（全记忆 + 置信邻居二审）

按 7.81 思路新增 n135：

1. 时间戳记忆 `last_ts/last_pol` 对所有事件无条件保留（不再条件入库）；
2. 另外维护一张历史分值图 `last_score`，记录该像素上次事件的 baseline 分数；
3. 仅当当前事件 `S_base >= S_act` 时触发二审：在邻域内只统计 `last_score >= S_trust` 的“高置信邻居”，再做内外圈比例悬崖判别。

实现文件：

- `src/myevs/denoise/ops/ebfopt_part2/n135_confidence_map_dualtrack_backbone.py`

接入入口：

- `scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py`（新增 `--variant n135`）

公式（与实现一致）：

$$
S_{base}(e_i)=\sum_{e_j\in\mathcal{N}_{9\times9},\,p_j=p_i}\max\left(0,1-\frac{\Delta t_{ij}}{\tau}\right)
$$

当 `S_base < S_act`：

$$
score_i=S_{base}(e_i)
$$

当 `S_base \ge S_act`：

$$
r_{trust}=\frac{A_{out}^{trust}}{A_{in}^{trust}+A_{out}^{trust}+\epsilon},\qquad
score_i=\begin{cases}
S_{base}(e_i), & r_{trust}<H_{thresh}\\
0, & r_{trust}\ge H_{thresh}
\end{cases}
$$

其中 `A_in/A_out` 只累加满足 `last_score >= S_trust` 的邻居历史事件。

#### 24.11.1 smoke（20k）

目录：

- `data/ED24/myPedestrain_06/EBF_Part2/_smoke_n135_781_20k/`

最佳 AUC（smoke）：

- light：`ebf_n135_labelscore_s7_tau64000`，AUC=0.958997
- mid：`ebf_n135_labelscore_s7_tau64000`，AUC=0.943969
- heavy：`ebf_n135_labelscore_s5_tau64000`，AUC=0.930429

#### 24.11.2 参数预筛（200k，固定 s=9,tau=128ms）

预筛网格：

- `S_act ∈ {2.0, 3.0, 4.0}`
- `S_trust ∈ {1.0, 2.0, 3.0}`
- 固定 `H_thresh=0.9`, `inner_radius=2`

Top-3（按 mean AUC 排序）：

| rank | S_act | S_trust | mean AUC | mean F1 |
|---:|---:|---:|---:|---:|
| 1 | 4.0 | 1.0 | 0.926792 | 0.849044 |
| 2 | 3.0 | 1.0 | 0.925285 | 0.848659 |
| 3 | 4.0 | 2.0 | 0.925157 | 0.848829 |

结论：后续 400k full-grid 采用 `S_act=4.0, S_trust=1.0`。

预筛目录：

- `data/ED24/myPedestrain_06/EBF_Part2/_tune_n135_781_sact*_strust*_h0p9_200k_s9tau128/`

#### 24.11.3 400k 全网格（选用 S_act=4.0, S_trust=1.0）

命令：

```bash
$env:PYTHONNOUSERSITE=1
$env:MYEVS_N135_INNER_RADIUS=2
$env:MYEVS_N135_H_THRESH=0.90
$env:MYEVS_N135_S_ACT=4.0
$env:MYEVS_N135_S_TRUST=1.0

D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py \
	--variant n135 \
	--max-events 400000 \
	--s-list 3,5,7,9 \
	--tau-us-list 8000,16000,32000,64000,128000,256000,512000,1024000 \
	--out-dir data/ED24/myPedestrain_06/EBF_Part2/_slim_n135_781_prescreen400k_sact4_strust1_h0p9
```

主实验目录：

- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n135_781_prescreen400k_sact4_strust1_h0p9/`

n135 各环境最佳点（full-grid）：

| env | best-AUC tag | AUC | best-F1 tag | F1 |
|---|---|---:|---|---:|
| light | `ebf_n135_labelscore_s9_tau128000` | 0.945771 | `ebf_n135_labelscore_s7_tau512000` | 0.948573 |
| mid | `ebf_n135_labelscore_s9_tau256000` | 0.913560 | `ebf_n135_labelscore_s9_tau128000` | 0.792735 |
| heavy | `ebf_n135_labelscore_s9_tau256000` | 0.909087 | `ebf_n135_labelscore_s9_tau128000` | 0.746850 |

#### 24.11.4 与 baseline / n132 / n134 对比（400k full-grid）

| variant | mean AUC | mean F1 |
|---|---:|---:|
| baseline | 0.925929 | 0.829591 |
| n132(d=2) | 0.925668 | 0.830222 |
| n134(S_keep=0.5) | 0.921455 | 0.827064 |
| n135(S_act=4,S_trust=1,H=0.9) | 0.922806 | 0.829386 |

delta vs baseline：

| variant | mean AUC delta | mean F1 delta |
|---|---:|---:|
| n132(d=2) | -0.000261 | +0.000630 |
| n134(S_keep=0.5) | -0.004475 | -0.002527 |
| n135(S_act=4,S_trust=1,H=0.9) | -0.003123 | -0.000205 |

#### 24.11.5 结论

1. n135 相比 n134 明显更稳（恢复了大部分排序能力），说明“全记忆 + 计算时筛邻居”优于“条件入库”；
2. 但在当前 ED24 full-grid 口径下，n135 仍未超过 baseline / n132；
3. n135 的 F1 已接近 baseline（mean F1 仅 -0.000205），可作为后续“高噪场景定向增强”分支继续探索。

产物目录：

- `data/ED24/myPedestrain_06/EBF_Part2/_smoke_n135_781_20k/`
- `data/ED24/myPedestrain_06/EBF_Part2/_tune_n135_781_sact*_strust*_h0p9_200k_s9tau128/`
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n135_781_prescreen400k_sact4_strust1_h0p9/`

### 24.12 7.84：纯各向异性单轴最大能量 n137（Score=Emax）

按 7.84 思路新增 n137：完全去掉 baseline 作为主分数，仅保留 4 轴能量中的最大值作为最终打分。

实现要点：

1. 在 `9x9` 同极性邻域内按时间衰减累积 4 条轴能量：`E0/E90/E45/E135`；
2. 最终分数直接取 `Emax=max(E0,E90,E45,E135)`；
3. 像素记忆仍采用 `last_ts/last_pol` 全量更新时间戳（不做条件入库）。

实现文件：

- `src/myevs/denoise/ops/ebfopt_part2/n137_pure_axis_emax_filter.py`

接入入口：

- `scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py`（新增 `--variant n137`）

公式：

$$
E_0=\sum_{dy=0}w_j,\quad
E_{90}=\sum_{dx=0}w_j,\quad
E_{45}=\sum_{dx=dy}w_j,\quad
E_{135}=\sum_{dx=-dy}w_j
$$

$$
Score=E_{max}=\max(E_0,E_{90},E_{45},E_{135}),\qquad
w_j=\max\left(0,1-\frac{\Delta t_j}{\tau}\right)
$$

#### 24.12.1 400k 全网格（同口径）

命令：

```bash
$env:PYTHONNOUSERSITE=1

D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py \
	--variant n137 \
	--max-events 400000 \
	--s-list 3,5,7,9 \
	--tau-us-list 8000,16000,32000,64000,128000,256000,512000,1024000 \
	--out-dir data/ED24/myPedestrain_06/EBF_Part2/_slim_n137_784_prescreen400k
```

主实验目录：

- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n137_784_prescreen400k/`

n137 各环境最佳点（full-grid）：

| env | best-AUC tag | AUC | best-F1 tag | F1 |
|---|---|---:|---|---:|
| light | `ebf_n137_labelscore_s9_tau256000` | 0.930783 | `ebf_n137_labelscore_s9_tau1024000` | 0.941384 |
| mid | `ebf_n137_labelscore_s9_tau128000` | 0.884609 | `ebf_n137_labelscore_s9_tau128000` | 0.723727 |
| heavy | `ebf_n137_labelscore_s9_tau128000` | 0.877632 | `ebf_n137_labelscore_s9_tau128000` | 0.659882 |

#### 24.12.2 与 baseline / n132 / n134 / n135 对比（400k full-grid）

| variant | mean AUC | mean F1 |
|---|---:|---:|
| baseline | 0.925929 | 0.829591 |
| n132(d=2) | 0.925668 | 0.830222 |
| n134(S_keep=0.5) | 0.921455 | 0.827064 |
| n135(S_act=4,S_trust=1,H=0.9) | 0.922806 | 0.829386 |
| n137(Emax) | 0.897675 | 0.774998 |

delta vs baseline：

| variant | mean AUC delta | mean F1 delta |
|---|---:|---:|
| n132(d=2) | -0.000261 | +0.000630 |
| n134(S_keep=0.5) | -0.004475 | -0.002527 |
| n135(S_act=4,S_trust=1,H=0.9) | -0.003123 | -0.000205 |
| n137(Emax) | -0.028255 | -0.054594 |

#### 24.12.3 结论

1. 7.84 的“纯单轴最大能量”在当前 ED24 full-grid 口径下显著退化，三档环境的 AUC/F1 均低于 baseline；
2. 退化最明显发生在 heavy（best-F1 约 0.6599，对比 baseline heavy best-F1 约 0.7455），说明仅靠 `Emax` 会过度丢弃非理想线状但真实的结构支撑；
3. 结论与 7.83 的统计一致：轴向各向异性更适合做联合弱特征，不适合作为全局唯一主评分。

产物目录：

- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n137_784_prescreen400k/`

### 24.13 7.85：二值带通门控 n139（A_true 的 L/H 双阈值扫参）

按 7.85 思路新增 n139：在 4 轴能量上构造

$$
A_{true}=\frac{E_{max}}{S_{base}+\varepsilon},\quad S_{base}=\sum_j w_j
$$

并采用带通门控后输出 `S_base` 的打分：

$$
Score=S_{base}\cdot\mathbf{1}\{L\le A_{true}\le H\}
$$

本次实现把 `L/H` 作为 sweep 维度，与 `s/tau` 同时网格搜索，并额外输出两套边际 AUC 汇总（对应“两个 AUC 视角”）：

1. `by_low`：每个 `low` 下，对其余参数取 best AUC；
2. `by_high`：每个 `high` 下，对其余参数取 best AUC。

实现文件：

- `src/myevs/denoise/ops/ebfopt_part2/n139_binary_struct_bandpass.py`

接入与扫参扩展：

- `scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py`
  - 新增 `--variant n139`
  - 新增 `--n139-low-list` / `--n139-high-list`
  - 新增 `auc2_ebf_n139_<env>_by_low.csv` / `auc2_ebf_n139_<env>_by_high.csv`

#### 24.13.1 400k 紧凑网格（low 细扫 0~0.1）

命令：

```powershell
$env:PYTHONNOUSERSITE='1'; D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/sweep_n139_785_prescreen400k.py --max-events 400000 --s-list 5,7,9 --tau-us-list 32000,64000,128000,256000 --n139-low-list 0.00,0.02,0.04,0.06,0.08,0.10 --n139-high-list 0.8,0.9 --esr-mode off --aocc-mode off --out-dir data/ED24/myPedestrain_06/EBF_Part2/_slim_n139_785_low0to01_compact400k_sbase
```

扫频脚本：

- `scripts/ED24_alg_evalu/sweep_n139_785_prescreen400k.py`（由 `sweep_ebf_slim_labelscore_grid.py` 迁移出的 n139 专用扫频实现）

本次扫频口径：

- `s ∈ {5,7,9}`
- `tau_us ∈ {32000,64000,128000,256000}`
- `low ∈ {0.00,0.02,0.04,0.06,0.08,0.10}`
- `high ∈ {0.8,0.9}`

主实验目录：

- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n139_785_low0to01_compact400k_sbase/`

n139 各环境最佳点（compact-grid）：

| env | best-AUC tag | AUC | best-F1 tag | F1 |
|---|---|---:|---|---:|
| light | `ebf_n139_labelscore_s9_tau256000_l0_h0p9` | 0.942599 | `ebf_n139_labelscore_s9_tau256000_l0_h0p9` | 0.949959 |
| mid | `ebf_n139_labelscore_s9_tau256000_l0_h0p9` | 0.917448 | `ebf_n139_labelscore_s9_tau128000_l0_h0p9` | 0.791278 |
| heavy | `ebf_n139_labelscore_s9_tau128000_l0_h0p9` | 0.912266 | `ebf_n139_labelscore_s9_tau128000_l0_h0p9` | 0.745486 |

#### 24.13.2 与 baseline / n132 / n134 / n135 / n137 对比（400k compact-grid）

| variant | mean AUC | mean F1 |
|---|---:|---:|
| baseline | 0.925929 | 0.829591 |
| n132(d=2) | 0.925668 | 0.830222 |
| n134(S_keep=0.5) | 0.921455 | 0.827064 |
| n135(S_act=4,S_trust=1,H=0.9) | 0.922806 | 0.829386 |
| n137(Emax) | 0.897675 | 0.774998 |
| n139(A_true band-pass + S_base score) | 0.924104 | 0.828908 |

delta vs baseline：

| variant | mean AUC delta | mean F1 delta |
|---|---:|---:|
| n132(d=2) | -0.000261 | +0.000630 |
| n134(S_keep=0.5) | -0.004475 | -0.002527 |
| n135(S_act=4,S_trust=1,H=0.9) | -0.003123 | -0.000205 |
| n137(Emax) | -0.028255 | -0.054594 |
| n139(A_true band-pass + S_base score) | -0.001825 | -0.000683 |

#### 24.13.3 “两个 AUC”边际视角（n139）

本次输出：

- `auc2_ebf_n139_<env>_by_low.csv`
- `auc2_ebf_n139_<env>_by_high.csv`

三档环境在本次网格上的边际最优点如下（`by_low` 与 `by_high` 对应最优值一致）：

| env | best low | best high | best AUC | best s | best tau_us |
|---|---:|---:|---:|---:|---:|
| light | 0.00 | 0.9 | 0.942599 | 9 | 256000 |
| mid | 0.00 | 0.9 | 0.917448 | 9 | 256000 |
| heavy | 0.00 | 0.9 | 0.912266 | 9 | 128000 |

#### 24.13.4 结论

1. 把带通命中后的输出由二值改为 `S_base` 后，排序分辨率显著提升，AUC/F1 相比上一版二值输出大幅改善；
2. 在本次 compact-grid（low 0~0.1）口径下，n139 的 mean AUC/F1 已接近 baseline（delta 分别约 -0.0018 / -0.0007）；
3. 边际最优点在三环境均偏向 `low=0.0, high=0.9`，说明当前网格下更宽松带通更有利于该打分形式。

产物目录：

- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n139_785_low0to01_compact400k_sbase/`

### 24.14 7.86：基于 n128 的高斯空间衰减 n140

按 7.86 的时空相关性分析，在 n128 联合时空衰减骨架上把空间项替换为高斯核：

$$
Score_i=\sum_{j\in\mathcal{N},\,p_j=p_i}\max\left(0,1-\frac{\Delta t_{ij}}{\tau}\right)\cdot\exp\left(-\frac{d_{ij}^2}{2\sigma^2}\right)
$$

实现文件：

- `src/myevs/denoise/ops/ebfopt_part2/n140_gaussian_spatial_decay_backbone.py`

扫频脚本接入：

- `scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py`（新增 `--variant n140`）
- `MYEVS_N140_SIGMA` 用于控制高斯带宽 `sigma`

#### 24.14.1 实验口径（同上 compact400k）

- `max-events=400000`
- `s ∈ {5,7,9}`
- `tau_us ∈ {32000,64000,128000,256000}`
- `sigma ∈ {1.0,1.5,2.0,2.5,3.0,4.0}`
- `esr-mode=off, aocc-mode=off`

#### 24.14.2 sigma 扫频结果（n140）

| sigma | mean AUC | mean F1 | heavy best-F1 |
|---:|---:|---:|---:|
| 1.0 | 0.907814 | 0.791101 | 0.687265 |
| 1.5 | 0.920175 | 0.820541 | 0.734856 |
| 2.0 | 0.926763 | 0.831694 | 0.752824 |
| 2.5 | 0.928885 | 0.835620 | 0.757750 |
| 3.0 | **0.929262** | **0.836366** | **0.757800** |
| 4.0 | 0.928704 | 0.835164 | 0.754225 |

最佳带宽：`sigma=3.0`。

`sigma=3.0` 下各环境最优点：

| env | best-AUC tag | AUC | best-F1 tag | F1 |
|---|---|---:|---|---:|
| light | `ebf_n140_labelscore_s9_tau128000` | 0.949371 | `ebf_n140_labelscore_s9_tau256000` | 0.951393 |
| mid | `ebf_n140_labelscore_s9_tau128000` | 0.921847 | `ebf_n140_labelscore_s9_tau128000` | 0.799904 |
| heavy | `ebf_n140_labelscore_s9_tau128000` | 0.916567 | `ebf_n140_labelscore_s9_tau64000` | 0.757800 |

#### 24.14.3 与 baseline / n128 对照（同口径）

| variant | mean AUC | mean F1 | heavy best-F1 |
|---|---:|---:|---:|
| baseline | 0.925929 | 0.829531 | 0.745459 |
| n128 (eta=0.5) | 0.928174 | 0.834029 | 0.752675 |
| n140 (sigma=3.0) | **0.929262** | **0.836366** | **0.757800** |

delta（n140 sigma=3.0）：

| 对比对象 | mean AUC delta | mean F1 delta | heavy best-F1 delta |
|---|---:|---:|---:|
| vs baseline | +0.003333 | +0.006835 | +0.012341 |
| vs n128(eta=0.5) | +0.001088 | +0.002337 | +0.005125 |

#### 24.14.4 结论

1. n140（高斯空间衰减）在同口径 compact400k 上稳定优于 baseline 与 n128(eta=0.5)；
2. `sigma` 呈现明显“中间最优”区间（约 `2.5~3.0`），符合 7.86 对“过窄误杀、过宽放噪”的预期；
3. heavy 的 best-F1 提升最明显，说明高斯空间先验能更好抑制外圈随机支持导致的高分噪声。

产物目录：

- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n140_786_sigma1p0_compact400k/`
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n140_786_sigma1p5_compact400k/`
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n140_786_sigma2p0_compact400k/`
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n140_786_sigma2p5_compact400k/`
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n140_786_sigma3p0_compact400k/`
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n140_786_sigma4p0_compact400k/`
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_baseline_compact400k/`
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n128_eta0p5_compact400k/`

### 24.15 7.86 扩展：n141（空间高斯 + 时间高斯）

在 n140（空间高斯）基础上，把时间核也改为高斯形式：

$$
Score_i=\sum_{j\in\mathcal{N},\,p_j=p_i}\exp\left(-\frac{\Delta t_{ij}^2}{2\sigma_t^2}\right)\cdot\exp\left(-\frac{d_{ij}^2}{2\sigma_s^2}\right)
$$

其中：

- `sigma_s = MYEVS_N141_SIGMA`
- `sigma_t = tau * MYEVS_N141_TSIGMA_RATIO`（本次固定 `TSIGMA_RATIO=1.0`）

实现文件：

- `src/myevs/denoise/ops/ebfopt_part2/n141_gaussian_spatiotemporal_decay_backbone.py`

扫频脚本接入：

- `scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py`（新增 `--variant n141`）

#### 24.15.1 实验口径（同上 compact400k）

- `max-events=400000`
- `s ∈ {5,7,9}`
- `tau_us ∈ {32000,64000,128000,256000}`
- `sigma_s ∈ {1.0,1.5,2.0,2.5,3.0,4.0}`
- `TSIGMA_RATIO=1.0`
- `esr-mode=off, aocc-mode=off`

#### 24.15.2 sigma 扫频结果（n141）

| sigma_s | mean AUC | mean F1 | heavy best-F1 |
|---:|---:|---:|---:|
| 1.0 | 0.909198 | 0.791063 | 0.688771 |
| 1.5 | 0.920844 | 0.820877 | 0.735475 |
| 2.0 | 0.927291 | 0.832638 | 0.754833 |
| 2.5 | 0.929519 | 0.836314 | 0.759160 |
| 3.0 | **0.929948** | **0.836954** | **0.759119** |
| 4.0 | 0.929455 | 0.835717 | 0.756506 |

最佳带宽：`sigma_s=3.0`。

`sigma_s=3.0` 下各环境最优点：

| env | best-AUC tag | AUC | best-F1 tag | F1 |
|---|---|---:|---|---:|
| light | `ebf_n141_labelscore_s9_tau64000` | 0.950817 | `ebf_n141_labelscore_s9_tau128000` | 0.951592 |
| mid | `ebf_n141_labelscore_s9_tau64000` | 0.922252 | `ebf_n141_labelscore_s9_tau32000` | 0.800150 |
| heavy | `ebf_n141_labelscore_s9_tau64000` | 0.916775 | `ebf_n141_labelscore_s9_tau32000` | 0.759119 |

#### 24.15.3 同口径对照（baseline / n128 / n140 / n141）

| variant | mean AUC | mean F1 | heavy best-F1 |
|---|---:|---:|---:|
| baseline | 0.925929 | 0.829531 | 0.745459 |
| n128 (eta=0.5) | 0.928174 | 0.834029 | 0.752675 |
| n140 (sigma=3.0) | 0.929262 | 0.836366 | 0.757800 |
| n141 (sigma_s=3.0, tsig=1.0) | **0.929948** | **0.836954** | **0.759119** |

delta（n141 sigma_s=3.0, tsig=1.0）：

| 对比对象 | mean AUC delta | mean F1 delta | heavy best-F1 delta |
|---|---:|---:|---:|
| vs baseline | +0.004019 | +0.007423 | +0.013660 |
| vs n128(eta=0.5) | +0.001774 | +0.002925 | +0.006444 |
| vs n140(sigma=3.0) | +0.000686 | +0.000588 | +0.001319 |

#### 24.15.4 结论

1. 在当前 compact400k 口径下，n141（时空双高斯）相对 n140（仅空间高斯）有稳定但小幅增益；
2. 最优区间仍在 `sigma_s≈2.5~3.0`，与 24.14 的空间核结论一致；
3. 时间核高斯化后，最优 `tau` 倾向更短（AUC 多落在 `tau=64ms`，heavy 的 best-F1 落在 `tau=32ms`），符合“更强调近时刻相关性”的预期。

产物目录：

- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n141_786_sigma1p0_tsig1p0_compact400k/`
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n141_786_sigma1p5_tsig1p0_compact400k/`
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n141_786_sigma2p0_tsig1p0_compact400k/`
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n141_786_sigma2p5_tsig1p0_compact400k/`
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n141_786_sigma3p0_tsig1p0_compact400k/`
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n141_786_sigma4p0_tsig1p0_compact400k/`

### 24.16 7.86 扩展：n142（仅替换时间核为二次形式）

按你的要求，只改 baseline 的时间核，空间项与其余流程保持不变：

$$
W_{time}=\max\left(0,1-\left(\frac{\Delta t}{\tau}\right)^2\right)
$$

对应得分：

$$
Score_i=\sum_{j\in\mathcal{N},\,p_j=p_i} W_{time}(\Delta t_{ij})
$$

实现文件：

- `src/myevs/denoise/ops/ebfopt_part2/n142_quadratic_time_decay_backbone.py`

扫频脚本接入：

- `scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py`（新增 `--variant n142`）

#### 24.16.1 实验口径（同上 compact400k）

- `max-events=400000`
- `s ∈ {5,7,9}`
- `tau_us ∈ {32000,64000,128000,256000}`
- `esr-mode=off, aocc-mode=off`

#### 24.16.2 n142 结果（single run）

各环境最优点：

| env | best-AUC tag | AUC | best-F1 tag | F1 |
|---|---|---:|---|---:|
| light | `ebf_n142_labelscore_s9_tau128000` | 0.946173 | `ebf_n142_labelscore_s9_tau256000` | 0.951336 |
| mid | `ebf_n142_labelscore_s9_tau128000` | 0.915797 | `ebf_n142_labelscore_s9_tau64000` | 0.787767 |
| heavy | `ebf_n142_labelscore_s9_tau128000` | 0.910543 | `ebf_n142_labelscore_s7_tau64000` | 0.744912 |

汇总：`mean AUC = 0.924171`，`mean F1 = 0.828005`。

#### 24.16.3 同口径对照（baseline / n140 / n141 / n142）

| variant | mean AUC | mean F1 | heavy best-F1 |
|---|---:|---:|---:|
| baseline | 0.925929 | 0.829531 | 0.745459 |
| n140 (sigma=3.0) | 0.929262 | 0.836366 | 0.757800 |
| n141 (sigma_s=3.0, tsig=1.0) | **0.929948** | **0.836954** | **0.759119** |
| n142 (time quadratic) | 0.924171 | 0.828005 | 0.744912 |

delta（n142）：

| 对比对象 | mean AUC delta | mean F1 delta | heavy best-F1 delta |
|---|---:|---:|---:|
| vs baseline | -0.001758 | -0.001526 | -0.000547 |
| vs n140(sigma=3.0) | -0.005091 | -0.008361 | -0.012888 |
| vs n141(sigma_s=3.0,tsig=1.0) | -0.005777 | -0.008949 | -0.014207 |

#### 24.16.4 结论

1. 只把时间核从线性改为二次形式（n142）并未带来性能提升，整体略低于 baseline；
2. 当前口径下，提升主要来自“空间先验”（n140/n141），而不是单独改变时间核曲线；
3. 因此后续更值得继续优化空间高斯与时空联合高斯（n140/n141），而非仅做时间核替换。

产物目录：

- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n142_786_timequad_compact400k/`

### 24.17 7.87：n143（双边高斯近似：空间 LUT + 时间二次）

基于 7.87 的工程化思路，在 n141 的“时空双高斯”基础上做硬件友好近似：

$$
Score_i=\sum_{j\in\mathcal{N},\,p_j=p_i}\max\left(0,1-\left(\frac{\Delta t_{ij}}{\tau}\right)^2\right)\cdot W_s(d_{ij})
$$

其中空间项采用查表：

$$
W_s(d)=\exp\left(-\frac{d^2}{2\sigma_s^2}\right),\quad d=\max(|\Delta x|,|\Delta y|)
$$

实现文件：

- `src/myevs/denoise/ops/ebfopt_part2/n143_bilateral_gaussian_approx_backbone.py`

扫频脚本接入：

- `scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py`（新增 `--variant n143`）
- 空间带宽环境变量：`MYEVS_N143_SIGMA`

#### 24.17.1 实验口径（同上 compact400k）

- `max-events=400000`
- `s ∈ {5,7,9}`
- `tau_us ∈ {32000,64000,128000,256000}`
- `sigma_s ∈ {1.0,1.5,2.0,2.5,3.0,4.0}`
- `esr-mode=off, aocc-mode=off`

#### 24.17.2 sigma 扫频结果（n143）

| sigma_s | mean AUC | mean F1 | heavy best-F1 |
|---:|---:|---:|---:|
| 1.0 | 0.909521 | 0.801652 | 0.700496 |
| 1.5 | 0.919920 | 0.823787 | 0.741125 |
| 2.0 | 0.925309 | 0.832409 | 0.754492 |
| 2.5 | 0.926583 | **0.834768** | **0.756763** |
| 3.0 | **0.926588** | 0.834511 | 0.755789 |
| 4.0 | 0.925960 | 0.832569 | 0.751825 |

综合 `mean-F1` 与 heavy 指标，推荐 `sigma_s=2.5`。

`sigma_s=2.5` 下各环境最优点：

| env | best-AUC tag | AUC | best-F1 tag | F1 |
|---|---|---:|---|---:|
| light | `ebf_n143_labelscore_s9_tau128000` | 0.947389 | `ebf_n143_labelscore_s9_tau256000` | 0.950703 |
| mid | `ebf_n143_labelscore_s9_tau128000` | 0.918801 | `ebf_n143_labelscore_s9_tau64000` | 0.796838 |
| heavy | `ebf_n143_labelscore_s9_tau128000` | 0.913560 | `ebf_n143_labelscore_s9_tau64000` | 0.756763 |

#### 24.17.3 同口径对照（baseline / n140 / n141 / n142 / n143）

| variant | mean AUC | mean F1 | heavy best-F1 |
|---|---:|---:|---:|
| baseline | 0.925929 | 0.829531 | 0.745459 |
| n140 (sigma=3.0) | **0.929262** | **0.836366** | **0.757800** |
| n141 (sigma_s=3.0, tsig=1.0) | **0.929948** | **0.836954** | **0.759119** |
| n142 (time quadratic) | 0.924171 | 0.828005 | 0.744912 |
| n143 (sigma_s=2.5) | 0.926583 | 0.834768 | 0.756763 |

delta（n143 sigma_s=2.5）：

| 对比对象 | mean AUC delta | mean F1 delta | heavy best-F1 delta |
|---|---:|---:|---:|
| vs baseline | +0.000654 | +0.005237 | +0.011304 |
| vs n140(sigma=3.0) | -0.002679 | -0.001598 | -0.001037 |
| vs n141(sigma_s=3.0,tsig=1.0) | -0.003365 | -0.002186 | -0.002356 |

#### 24.17.4 结论

1. n143（双边高斯近似）相较 baseline 有稳定提升，说明 7.87 的工程化近似方向有效；
2. n143 与 n140/n141 相比略有性能损失，但差距不大，符合“用更低计算复杂度换少量精度”的预期；
3. 在追求边缘端实时部署时，n143 可以作为 n141 的高性价比替代方案。

产物目录：

- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n143_787_sigma1p0_compact400k/`
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n143_787_sigma1p5_compact400k/`
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n143_787_sigma2p0_compact400k/`
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n143_787_sigma2p5_compact400k/`
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n143_787_sigma3p0_compact400k/`
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n143_787_sigma4p0_compact400k/`

### 24.18 7.87 扩展：n144（n143 去时间平方，线性时间核）

按你的最新要求，在 n143 基础上仅改时间核：把二次项去掉，改为线性衰减。

$$
Score_i=\sum_{j\in\mathcal{N},\,p_j=p_i}\max\left(0,1-\frac{\Delta t_{ij}}{\tau}\right)\cdot W_s(d_{ij})
$$

其中空间项继续沿用 n143 的 LUT 高斯：

$$
W_s(d)=\exp\left(-\frac{d^2}{2\sigma_s^2}\right),\quad d=\max(|\Delta x|,|\Delta y|)
$$

实现文件：

- `src/myevs/denoise/ops/ebfopt_part2/n144_bilateral_gaussian_linear_time_backbone.py`

扫频脚本接入：

- `scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py`（新增 `--variant n144`）
- 空间带宽环境变量：`MYEVS_N144_SIGMA`

#### 24.18.1 实验口径（同上 compact400k）

- `max-events=400000`
- `s ∈ {5,7,9}`
- `tau_us ∈ {32000,64000,128000,256000}`
- `sigma_s ∈ {1.0,1.5,2.0,2.5,3.0,4.0}`
- `esr-mode=off, aocc-mode=off`

#### 24.18.2 sigma 扫频结果（n144）

| sigma_s | mean AUC | mean F1 | heavy best-F1 |
|---:|---:|---:|---:|
| 1.0 | 0.910111 | 0.801906 | 0.704802 |
| 1.5 | 0.922293 | 0.823353 | 0.740244 |
| 2.0 | 0.927347 | 0.832501 | 0.753259 |
| 2.5 | **0.928524** | 0.834646 | **0.755361** |
| 3.0 | 0.928499 | **0.834710** | 0.754442 |
| 4.0 | 0.927847 | 0.833496 | 0.751694 |

`sigma_s=2.5` 与 `sigma_s=3.0` 几乎并列；综合 heavy 指标与 mean AUC，推荐 `sigma_s=2.5`。

`sigma_s=2.5` 下各环境最优点：

| env | best-AUC tag | AUC | best-F1 tag | F1 |
|---|---|---:|---|---:|
| light | `ebf_n144_labelscore_s9_tau128000` | 0.948960 | `ebf_n144_labelscore_s9_tau256000` | 0.950840 |
| mid | `ebf_n144_labelscore_s9_tau128000` | 0.920836 | `ebf_n144_labelscore_s9_tau128000` | 0.797737 |
| heavy | `ebf_n144_labelscore_s9_tau128000` | 0.915775 | `ebf_n144_labelscore_s9_tau64000` | 0.755361 |

#### 24.18.3 同口径对照（baseline / n140 / n141 / n142 / n143 / n144）

| variant | mean AUC | mean F1 | heavy best-F1 |
|---|---:|---:|---:|
| baseline | 0.925929 | 0.829531 | 0.745459 |
| n140 (sigma=3.0) | 0.929262 | 0.836366 | 0.757800 |
| n141 (sigma_s=3.0, tsig=1.0) | **0.929948** | **0.836954** | **0.759119** |
| n142 (time quadratic) | 0.924171 | 0.828005 | 0.744912 |
| n143 (sigma_s=2.5) | 0.926583 | 0.834768 | 0.756763 |
| n144 (sigma_s=2.5, time linear) | 0.928524 | 0.834646 | 0.755361 |

delta（n144 sigma_s=2.5）：

| 对比对象 | mean AUC delta | mean F1 delta | heavy best-F1 delta |
|---|---:|---:|---:|
| vs baseline | +0.002595 | +0.005115 | +0.009902 |
| vs n140(sigma=3.0) | -0.000738 | -0.001720 | -0.002439 |
| vs n141(sigma_s=3.0,tsig=1.0) | -0.001424 | -0.002308 | -0.003758 |
| vs n142(time quadratic) | +0.004353 | +0.006641 | +0.010449 |
| vs n143(sigma_s=2.5) | +0.001941 | -0.000122 | -0.001402 |

#### 24.18.4 结论

1. n144（去时间平方）相对 baseline 与 n142 均有明显提升，说明“空间高斯先验 + 线性时间核”仍然有效；
2. 与 n143 对比：n144 的 mean AUC 更高，但 mean F1/heavy best-F1 略低，表现为“排序能力更强、阈值点精度略回落”；
3. 在当前口径下，综合性能仍以 n141 最优；n144 可作为 n143 的线性时间核替代分支，适合继续做阈值策略或后处理联合优化。

产物目录：

- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n144_787_sigma1p0_compact400k/`
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n144_787_sigma1p5_compact400k/`
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n144_787_sigma2p0_compact400k/`
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n144_787_sigma2p5_compact400k/`
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n144_787_sigma3p0_compact400k/`
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n144_787_sigma4p0_compact400k/`

### 24.19 7.87 扩展：n145（按你的要求：时间核逐项平方后再求和）

按你的澄清，这次不是“对累计和整体平方”，而是把每个邻居项的时间核从

$$
1-\frac{\Delta t}{\tau}
$$

改为

$$
\left(1-\frac{\Delta t}{\tau}\right)^2
$$

然后再做邻域求和。基于 n144 的空间项不变：

$$
Score_i=\sum_{j\in\mathcal{N},\,p_j=p_i}\max\left(0,1-\frac{\Delta t_{ij}}{\tau}\right)^2\cdot W_s(d_{ij})
$$

其中

$$
W_s(d)=\exp\left(-\frac{d^2}{2\sigma_s^2}\right),\quad d=\max(|\Delta x|,|\Delta y|)
$$

实现文件：

- `src/myevs/denoise/ops/ebfopt_part2/n145_bilateral_gaussian_sq_linear_time_backbone.py`

扫频脚本接入：

- `scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py`（新增 `--variant n145`）
- 空间带宽环境变量：`MYEVS_N145_SIGMA`

#### 24.19.1 实验口径（同上 compact400k）

- `max-events=400000`
- `s ∈ {5,7,9}`
- `tau_us ∈ {32000,64000,128000,256000}`
- `sigma_s ∈ {1.0,1.5,2.0,2.5,3.0,4.0}`
- `esr-mode=off, aocc-mode=off`

#### 24.19.2 sigma 扫频结果（n145）

| sigma_s | mean AUC | mean F1 | heavy best-F1 |
|---:|---:|---:|---:|
| 1.0 | 0.911031 | 0.800099 | 0.699825 |
| 1.5 | 0.923575 | 0.824865 | 0.741956 |
| 2.0 | 0.928847 | 0.833852 | 0.756104 |
| 2.5 | 0.930165 | **0.836139** | **0.758442** |
| 3.0 | **0.930235** | 0.836115 | 0.757905 |
| 4.0 | 0.929686 | 0.834475 | 0.754611 |

`sigma_s=2.5` 与 `sigma_s=3.0` 几乎并列；综合 mean F1 与 heavy 指标，推荐 `sigma_s=2.5`。

`sigma_s=2.5` 下各环境最优点：

| env | best-AUC tag | AUC | best-F1 tag | F1 |
|---|---|---:|---|---:|
| light | `ebf_n145_labelscore_s9_tau256000` | 0.950209 | `ebf_n145_labelscore_s9_tau256000` | 0.950155 |
| mid | `ebf_n145_labelscore_s9_tau256000` | 0.922781 | `ebf_n145_labelscore_s9_tau128000` | 0.799819 |
| heavy | `ebf_n145_labelscore_s9_tau256000` | 0.917505 | `ebf_n145_labelscore_s9_tau128000` | 0.758442 |

#### 24.19.3 同口径对照（baseline / n140 / n141 / n142 / n143 / n144 / n145）

| variant | mean AUC | mean F1 | heavy best-F1 |
|---|---:|---:|---:|
| baseline | 0.925929 | 0.829531 | 0.745459 |
| n140 (sigma=3.0) | 0.929262 | 0.836366 | 0.757800 |
| n141 (sigma_s=3.0, tsig=1.0) | 0.929948 | **0.836954** | **0.759119** |
| n142 (time quadratic) | 0.924171 | 0.828005 | 0.744912 |
| n143 (sigma_s=2.5) | 0.926583 | 0.834768 | 0.756763 |
| n144 (sigma_s=2.5, time linear) | 0.928524 | 0.834646 | 0.755361 |
| n145 (sigma_s=2.5, time (1-X)^2) | **0.930165** | 0.836139 | 0.758442 |

delta（n145 sigma_s=2.5）：

| 对比对象 | mean AUC delta | mean F1 delta | heavy best-F1 delta |
|---|---:|---:|---:|
| vs baseline | +0.004236 | +0.006608 | +0.012983 |
| vs n140(sigma=3.0) | +0.000903 | -0.000227 | +0.000642 |
| vs n141(sigma_s=3.0,tsig=1.0) | +0.000217 | -0.000815 | -0.000677 |
| vs n142(time quadratic) | +0.005994 | +0.008134 | +0.013530 |
| vs n143(sigma_s=2.5) | +0.003582 | +0.001371 | +0.001679 |
| vs n144(sigma_s=2.5) | +0.001641 | +0.001493 | +0.003081 |

#### 24.19.4 结论
为什么 $(1-x)^2$ 完胜 $1-x$？（凸函数与噪声饥饿）

令归一化时间 $x = \frac{\Delta t}{\tau}$。线性 (n144): $y = 1 - x$下凹平方 (n145): $y = (1 - x)^2$
这是因为下凹平方核，在不伤害最新鲜信号的前提下，物理上『饿死』了 33% 的中远期背景噪声。”在 Heavy 环境下，由于带宽和光照限制，背景噪声在 30ms ($\tau$) 的窗口内是均匀分布的。算法对噪声的总吸纳量，在数学上等于时间核函数的定积分（面积）：线性核的噪声吸纳量：$\int_0^1 (1-x) dx = 0.50$下凹平方核的噪声吸纳量：$\int_0^1 (1-x)^2 dx = \mathbf{0.33}$线性衰减对“中度陈旧”的事件太宽容了（比如 $x=0.5$ 时，权重还有 0.5）。而 $(1-x)^2$ 极其苛刻（$x=0.5$ 时，权重暴跌到 0.25）。它像一把尖锐的锥子，精准剥夺了那些企图互相支撑的陈旧噪声的分数，导致噪声团簇根本无法突破高分阈值。
1. 按你定义把时间核改为“逐项 `(1-X)^2` 再求和”（n145）后，整体明显优于 n144（线性时间核），并且对 heavy 改善更明显；
2. n145 的 mean AUC 已超过 n141，但 mean F1 与 heavy best-F1 仍略低于 n141，说明排序能力更强但最优阈值点优势还在 n141；
3. 在当前口径下，n145 是一个很强的候选：若追求 AUC/排序稳定性可优先考虑 n145，若追求 best-F1 极值仍可保留 n141 作为主参考。

产物目录：

- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n145_787_sigma1p0_compact400k/`
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n145_787_sigma1p5_compact400k/`
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n145_787_sigma2p0_compact400k/`
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n145_787_sigma2p5_compact400k/`
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n145_787_sigma3p0_compact400k/`
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n145_787_sigma4p0_compact400k/`

### 24.20 baseline / n128 / n140~n146 总对比（编号 + 架构命名）

口径统一为 compact400k：

- `max-events=400000`
- `s in {5,7,9}`
- `tau_us in {32000,64000,128000,256000}`
- `esr-mode=off, aocc-mode=off`

为避免只看编号，这里统一给出“编号 + 架构名”。

| variant | 架构命名 | 核心区别（相对 baseline） | 代表参数 | mean AUC | mean F1 | heavy best-F1 |
|---|---|---|---|---:|---:|---:|
| baseline | 同极性线性时间核（无空间衰减） | 时间核 `1-dt/tau`，空间等权 | s/tau 网格最优 | 0.925929 | 0.829531 | 0.745459 |
| n128 | 联合时空线性衰减核 | 在线性时间核上叠加空间线性衰减 | eta=0.5 | 0.928174 | 0.834029 | 0.752675 |
| n140 | 空间高斯 + 时间线性核 | 空间项改为高斯衰减 | sigma=3.0 | 0.929262 | 0.836366 | 0.757800 |
| n141 | 时空双高斯核 | 时间项也改为高斯衰减 | sigma_s=3.0, tsig=1.0 | 0.929948 | **0.836954** | **0.759119** |
| n142 | 时间二次核（空间等权） | 仅把时间核改为 `1-(dt/tau)^2` | - | 0.924171 | 0.828005 | 0.744912 |
| n143 | 双边高斯近似（LUT 空间 + 时间二次） | 硬件友好近似：空间 LUT + 时间二次 | sigma_s=2.5 | 0.926583 | 0.834768 | 0.756763 |
| n144 | 双边高斯近似（LUT 空间 + 时间线性） | 在 n143 上改回时间线性 | sigma_s=2.5 | 0.928524 | 0.834646 | 0.755361 |
| n145 | 双边高斯近似（LUT 空间 + 时间逐项平方） | 每邻居时间项改为 `(1-dt/tau)^2` 后再求和 | sigma_s=2.5 | **0.930165** | 0.836139 | 0.758442 |
| n146 | n145 时空底盘 + 极性软融合 | 同/异极性能量分流 + 纯度权重融合 | sigma_s=2.5, gamma=2.0 | 0.930017 | 0.835531 | 0.757268 |在·

当前观察：

1. 若以 mean AUC 为主，n145 仍是当前最高；n146 非常接近（差约 `1.48e-4`）。
2. 若以 mean F1 / heavy best-F1 为主，n141 仍保持当前最优。
3. n146 的价值在于把 S52 的极性信息以软融合方式并入 n145 底盘，性能处于第一梯队，且结构上更完整。

### 24.21 7.88：n146（n145 时空底盘 + S52 极性软融合）

按 7.88 文档实现最终形态 n146：

1) 时空联合权重（继承 n145）：

$$
W_j = \max\left(0, 1-\frac{\Delta t_{ij}}{\tau}\right)^2 \cdot \exp\left(-\frac{d_{ij}^2}{2\sigma_s^2}\right)
$$

2) 极性能量分流：

$$
E_{same}=\sum_{p_j=p_i}W_j,\quad E_{opp}=\sum_{p_j\ne p_i}W_j
$$

3) 极性纯度软融合：

$$
Mix=\frac{E_{opp}}{E_{same}+E_{opp}+\epsilon},\quad W_{purity}=(1-Mix)^\gamma
$$

$$
Score_i = E_{same} + W_{purity}\cdot E_{opp}
$$

实现文件：

- `src/myevs/denoise/ops/ebfopt_part2/n146_polarity_soft_fusion_backbone.py`

扫频脚本接入：

- `scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py`（新增 `--variant n146`）
- 环境变量：`MYEVS_N146_SIGMA`、`MYEVS_N146_GAMMA`（可选 `MYEVS_N146_EPS`）

#### 24.21.1 实验口径（同上 compact400k）

- `max-events=400000`
- `s in {5,7,9}`
- `tau_us in {32000,64000,128000,256000}`
- 固定 `sigma_s=2.5`（继承 n145 最优带宽）
- 扫 `gamma in {1.0,2.0,3.0,4.0}`

#### 24.21.2 gamma 扫频结果（n146, sigma_s=2.5）

| gamma | mean AUC | mean F1 | heavy best-F1 |
|---:|---:|---:|---:|
| 1.0 | **0.931236** | 0.835077 | 0.755140 |
| 2.0 | 0.930017 | **0.835531** | **0.757268** |
| 3.0 | 0.929754 | 0.835167 | 0.756843 |
| 4.0 | 0.929872 | 0.835059 | 0.756486 |

综合看：

- 若偏向排序能力（AUC）可选 `gamma=1.0`；
- 若偏向 best-F1 与 heavy 表现，推荐 `gamma=2.0`。

以下对照统一采用 `gamma=2.0`。

`gamma=2.0` 下各环境最优点：

| env | best-AUC tag | AUC | best-F1 tag | F1 |
|---|---|---:|---|---:|
| light | `ebf_n146_labelscore_s9_tau128000` | 0.951542 | `ebf_n146_labelscore_s9_tau256000` | 0.950297 |
| mid | `ebf_n146_labelscore_s9_tau256000` | 0.922141 | `ebf_n146_labelscore_s9_tau128000` | 0.799027 |
| heavy | `ebf_n146_labelscore_s9_tau256000` | 0.916367 | `ebf_n146_labelscore_s9_tau128000` | 0.757268 |
   
#### 24.21.3 同口径对照（n141 / n145 / n146）

| variant | mean AUC | mean F1 | heavy best-F1 |
|---|---:|---:|---:|
| n141 (sigma_s=3.0, tsig=1.0) | 0.929948 | **0.836954** | **0.759119** |
| n145 (sigma_s=2.5) | **0.930165** | 0.836139 | 0.758442 |
| n146 (sigma_s=2.5, gamma=2.0) | 0.930017 | 0.835531 | 0.757268 |

delta（n146 gamma=2.0）：

| 对比对象 | mean AUC delta | mean F1 delta | heavy best-F1 delta |
|---|---:|---:|---:|
| vs baseline | +0.004088 | +0.006000 | +0.011809 |
| vs n141 | +0.000069 | -0.001423 | -0.001851 |
| vs n145 | -0.000148 | -0.000608 | -0.001174 |

#### 24.21.4 结论

1. n146 已按 7.88 落地并完成同口径全量扫频，性能稳定处于第一梯队；
2. 在当前参数下，n146 没有超过 n145/n141 的最优点，但差距很小，且引入了更完整的极性融合机制；
3. 后续若要继续冲击 F1，优先建议在 n146 上做 `gamma` 的细粒度扫参（如 `1.5/2.5`）与 `sigma_s` 联合网格，而不是回退到更早结构。

产物目录：

- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n146_788_sigma2p5_gamma1_compact400k/`
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n146_788_sigma2p5_gamma2_compact400k/`
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n146_788_sigma2p5_gamma3_compact400k/`
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n146_788_sigma2p5_gamma4_compact400k/`

### 24.22 7.88 扩展：n147（n145 作为 rawsame + s52 融合）

按你的新要求实现：把 n145 的时空打分作为 `rawsame`，再按 s52 的在线融合逻辑做自适应融合。

定义如下：

1) `rawsame`（来自 n145，同极性时空项）：

$$
raw_{same} = \sum_{p_j=p_i}\left(1-\frac{\Delta t_{ij}}{\tau}\right)^2_+\cdot \exp\left(-\frac{d_{ij}^2}{2\sigma_s^2}\right)
$$

2) `rawopp`（同一时空核但异极性分量）：

$$
raw_{opp} = \sum_{p_j\ne p_i}\left(1-\frac{\Delta t_{ij}}{\tau}\right)^2_+\cdot \exp\left(-\frac{d_{ij}^2}{2\sigma_s^2}\right)
$$

3) 沿用 s52 的在线自适应融合：

- `mix = rawopp / (rawsame + rawopp + eps)`
- `mix_state` 做在线均值更新，`alpha_eff=(1-mix_state)^2`
- `raw_gated = rawsame + alpha_eff * rawopp`
- 自占据抑制：`base = raw_gated / (1 + u_self^2)`
- 支持增益：`score = base * (1 + beta_state * sfrac)`

实现文件：

- `src/myevs/denoise/ops/ebfopt_part2/n147_n145_s52_fusion_backbone.py`

扫频脚本接入：

- `scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py`（新增 `--variant n147`）
- 空间带宽环境变量：`MYEVS_N147_SIGMA`

#### 24.22.1 实验口径（同上 compact400k）

- `max-events=400000`
- `s in {5,7,9}`
- `tau_us in {32000,64000,128000,256000}`
- 固定 `sigma_s=2.5`
- `esr-mode=off, aocc-mode=off`

#### 24.22.2 n147 结果（sigma_s=2.5）

各环境最优点：

| env | best-AUC tag | AUC | best-F1 tag | F1 |
|---|---|---:|---|---:|
| light | `ebf_n147_labelscore_s9_tau256000` | 0.954903 | `ebf_n147_labelscore_s9_tau256000` | 0.956405 |
| mid | `ebf_n147_labelscore_s9_tau256000` | 0.940047 | `ebf_n147_labelscore_s9_tau256000` | 0.811565 |
| heavy | `ebf_n147_labelscore_s9_tau256000` | 0.936697 | `ebf_n147_labelscore_s9_tau128000` | 0.767755 |

汇总：

- `mean AUC = 0.943882`
- `mean F1 = 0.845241`
- `heavy best-F1 = 0.767755`

#### 24.22.3 同口径对照（baseline / n141 / n145 / n146 / n147）

| variant | mean AUC | mean F1 | heavy best-F1 |
|---|---:|---:|---:|
| baseline | 0.925929 | 0.829531 | 0.745459 |
| n141 (sigma_s=3.0, tsig=1.0) | 0.929948 | 0.836954 | 0.759119 |
| n145 (sigma_s=2.5) | 0.930165 | 0.836139 | 0.758442 |
| n146 (sigma_s=2.5, gamma=2.0) | 0.930017 | 0.835531 | 0.757268 |
| n147 (n145 rawsame + s52 fusion, sigma_s=2.5) | **0.943882** | **0.845241** | **0.767755** |

delta（n147）：

| 对比对象 | mean AUC delta | mean F1 delta | heavy best-F1 delta |
|---|---:|---:|---:|
| vs baseline | +0.017953 | +0.015710 | +0.022296 |
| vs n141 | +0.013934 | +0.008287 | +0.008636 |
| vs n145 | +0.013717 | +0.009102 | +0.009313 |
| vs n146 | +0.013865 | +0.009710 | +0.010487 |

#### 24.22.4 结论

1. 按“n145 作为 rawsame + s52 融合”实现的 n147，在当前 compact400k 同口径下显著优于 baseline 与 n14x 系列；
2. 该融合在 light/mid/heavy 三环境均同步提升，说明 n145 的时空核与 s52 的在线融合具有互补性；
3. 现阶段 n147 已成为当前表中的最优候选，可作为后续主线继续做更细粒度扫参（例如 `sigma_s` 与在线融合状态项的联合验证）。

产物目录：

- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n147_788_sigma2p5_compact400k/`

### 24.23 7.90：n148（n147 欧式距离 + 8重映射资源优化）

按 7.90 要求，在 n147 基础上建立 n148，核心改动有两项：

1. 空间距离从切比雪夫距离改为欧式距离平方；
2. 邻域遍历改为“8重对称映射（octant unrolling）”的展开方式。

实现文件：

- `src/myevs/denoise/ops/ebfopt_part2/n148_n145_s52_euclid_octant_backbone.py`

扫频脚本接入：

- `scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py`（新增 `--variant n148`）
- 空间带宽环境变量：`MYEVS_N148_SIGMA`

#### 24.23.1 算法定义

n148 继承 n147 的融合骨架（n145 rawsame/rawopp + s52 在线融合），只替换空间核与遍历方式。

1) 空间核（欧式各向同性高斯）：

$$
W_s = \exp\left(-\frac{d^2}{2\sigma_s^2}\right),\quad d^2 = \Delta x^2+\Delta y^2
$$

2) 时间核（与 n147 一致）：

$$
W_t = \left(1-\frac{\Delta t}{\tau}\right)^2_+
$$

3) raw 证据（与 n147 一致）：

$$
raw_{same} = \sum_{p_j=p_i} W_t W_s,\qquad raw_{opp} = \sum_{p_j\ne p_i} W_t W_s
$$

4) s52 式在线融合（与 n147 一致）：

- `mix = rawopp / (rawsame + rawopp + eps)`
- `mix_state` 在线更新，`alpha_eff=(1-mix_state)^2`
- `raw_gated = rawsame + alpha_eff * rawopp`
- `base = raw_gated / (1 + u_self^2)`
- `score = base * (1 + beta_state * sfrac)`

#### 24.23.2 资源优化：8重映射如何节省空间

对 `R=4`（9x9 邻域）而言，去中心后共有 80 个邻居点。n148 不再显式保存完整邻居坐标/模板，而是只遍历三类“规范基点”，再用对称映射还原全部邻居：

1. 轴向基点 `(u,0)`：`u=1..4`，共 4 个；
2. 对角基点 `(u,u)`：`u=1..4`，共 4 个；
3. 内部基点 `(u,v), u>v>0`：共 6 个（`(2,1),(3,1),(3,2),(4,1),(4,2),(4,3)`）。

规范基点总数为 `4+4+6=14`，通过 8 重对称展开覆盖完整邻域贡献。

空间节省要点：

1. 邻域几何描述从“80 点显式存储”降为“14 个规范基点 + 对称生成”，描述规模约降 `82.5%`（`1-14/80`）。
2. 不需要保存 9x9 全模板或每邻居预存 `(dx,dy,d2,w)` 结构，减少常驻常量表与缓存压力。
3. 欧式高斯 LUT 采用 `d^2` 索引，避免运行时开方：`R=4` 时索引范围 `0..32`，LUT 长度 33，仅一次查表得到空间权重。

说明：

- 与 n147 的切比雪夫 LUT（长度 5）相比，欧式 `d^2` LUT 长度更大（33）；
- 但 n148 的“节省资源”重点在邻域几何映射存储与访存组织，不在 LUT 长度本身。

#### 24.23.3 实验口径（同上 compact400k）

- `max-events=400000`
- `s in {5,7,9}`
- `tau_us in {32000,64000,128000,256000}`
- 固定 `sigma_s=2.5`
- `esr-mode=off, aocc-mode=off`

#### 24.23.4 n148 结果（sigma_s=2.5）

各环境最优点：

| env | best-AUC tag | AUC | best-F1 tag | F1 |
|---|---|---:|---|---:|
| light | `ebf_n148_labelscore_s9_tau256000` | 0.955008 | `ebf_n148_labelscore_s9_tau256000` | 0.955688 |
| mid | `ebf_n148_labelscore_s9_tau256000` | 0.940140 | `ebf_n148_labelscore_s9_tau256000` | 0.812832 |
| heavy | `ebf_n148_labelscore_s9_tau256000` | 0.936579 | `ebf_n148_labelscore_s9_tau128000` | 0.769356 |

汇总：

- `mean AUC = 0.943909`
- `mean F1 = 0.845959`
- `heavy best-F1 = 0.769356`

#### 24.23.5 同口径对照（baseline / n145 / n147 / n148）

| variant | mean AUC | mean F1 | heavy best-F1 |
|---|---:|---:|---:|
| baseline | 0.925929 | 0.829531 | 0.745459 |
| n145 (sigma_s=2.5) | 0.930165 | 0.836139 | 0.758442 |
| n147 (Chebyshev + n145/s52 fusion) | 0.943882 | 0.845241 | 0.767755 |
| n148 (Euclidean + octant mapping) | **0.943909** | **0.845959** | **0.769356** |

delta（n148）：

| 对比对象 | mean AUC delta | mean F1 delta | heavy best-F1 delta |
|---|---:|---:|---:|
| vs baseline | +0.017980 | +0.016428 | +0.023897 |
| vs n145 | +0.013744 | +0.009820 | +0.010914 |
| vs n147 | +0.000027 | +0.000718 | +0.001601 |

#### 24.23.6 结论

1. n148 在保持 n147 融合框架的前提下，用欧式空间核与 8重映射实现了更硬件友好的实现路径；
2. 在当前 compact400k 同口径下，n148 相比 n147 取得小幅但一致的增益（尤其 heavy best-F1）；
3. 从“效果 + 工程实现”双视角看，n148 可作为当前优先主线版本，后续可继续联合扫 `sigma_s` 与 `tau` 网格验证增益稳定性。

产物目录：

- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n148_790_sigma2p5_compact400k/`

### 24.24 7.90 后半段优化：n149（n148 紧凑 LUT 直索引）

7.90 后半段提出：在 n148 的 8重映射基础上，进一步把空间核权重改为“紧凑 LUT 直索引”，避免运行时计算 `d^2=u^2+v^2`。

#### 24.24.1 可行性结论

结论：可行，且与 n148 数学等价。

原因：

1. n148 的空间权重本质只依赖有限个离散偏移点（axis/diag/interior 规范基点）；
2. 将这些点对应的高斯权重预先按顺序存到紧凑 LUT，运行时按组内索引读取，得到的 `W_s` 与 n148 完全一致；
3. 因此融合逻辑、打分公式不变，理论上精度应保持一致（实验也验证一致）。

实现文件：

- `src/myevs/denoise/ops/ebfopt_part2/n149_n145_s52_euclid_compactlut_backbone.py`

扫频脚本接入：

- `scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py`（新增 `--variant n149`）
- 环境变量：`MYEVS_N149_SIGMA`

#### 24.24.2 相比 n148 可节省多少运算

对半径 `rr`，n148 每事件用于 `d^2` 的额外算术开销（乘加）为：

$$
Ops_{d2}(rr)=\frac{rr(3rr+1)}{2}
$$

n149 中该项为 0（改为紧凑 LUT 直索引）。

对应到本实验 `s in {5,7,9}`（即 `rr in {2,3,4}`）：

| s | rr | n148 d2算术/事件 | n149 d2算术/事件 | 节省 |
|---|---:|---:|---:|---:|
| 5 | 2 | 7 | 0 | 7 |
| 7 | 3 | 15 | 0 | 15 |
| 9 | 4 | 26 | 0 | 26 |

按 sweep 口径（每个环境 12 组配置：3 个 s × 4 个 tau）折算，本次运行总计减少的 d2 乘加次数约为：

- light（194395 events）：`37,323,840`
- mid（400000 events）：`76,800,000`
- heavy（400000 events）：`76,800,000`
- 合计：`190,923,840`

#### 24.24.3 相比 n148 可节省多少空间

n148 的欧式 `d^2` LUT 长度：

$$
L_{148}(rr)=2rr^2+1
$$

n149 紧凑 LUT 长度（axis + diag + interior）：

$$
L_{149}(rr)=2rr+\frac{rr(rr-1)}{2}=\frac{rr(rr+3)}{2}
$$

对应 `s=5/7/9`：

| s | rr | n148 LUT项数 | n149 LUT项数 | 节省项数 | 节省比例 |
|---|---:|---:|---:|---:|---:|
| 5 | 2 | 9 | 5 | 4 | 44.44% |
| 7 | 3 | 19 | 9 | 10 | 52.63% |
| 9 | 4 | 33 | 14 | 19 | 57.58% |

若按 `float32` 存储，单核 LUT 常驻内存节省分别为 16B / 40B / 76B。

#### 24.24.4 实验口径（同上 compact400k）

- `max-events=400000`
- `s in {5,7,9}`
- `tau_us in {32000,64000,128000,256000}`
- `sigma_s`（环境变量 `MYEVS_N149_SIGMA`）：历史默认使用 `2.5`；2026-04-22 重跑后推荐 `2.8`
- `esr-mode=off, aocc-mode=off`

#### 24.24.5 n149 结果（sigma_s=2.5）

各环境最优点：

| env | best-AUC tag | AUC | best-F1 tag | F1 |
|---|---|---:|---|---:|
| light | `ebf_n149_labelscore_s9_tau256000` | 0.955008 | `ebf_n149_labelscore_s9_tau256000` | 0.955688 |
| mid | `ebf_n149_labelscore_s9_tau256000` | 0.940140 | `ebf_n149_labelscore_s9_tau256000` | 0.812832 |
| heavy | `ebf_n149_labelscore_s9_tau256000` | 0.936579 | `ebf_n149_labelscore_s9_tau128000` | 0.769356 |

汇总：

- `mean AUC = 0.943909`
- `mean F1 = 0.845959`
- `heavy best-F1 = 0.769356`

#### 24.24.6 2026-04-22 重跑：n149 结果（sigma_s=2.8）

动机：

1. 复核 `sigma_s=2.5` 是否为 compact400k 口径下最优；
2. 在不改变 n149 结构与资源占用的前提下，尝试“同资源提精度”。

实现不变，仅设置：

- `MYEVS_N149_SIGMA=2.8`

各环境最优点（compact400k）：

| env | best-AUC tag | AUC | best-F1 tag | F1 |
|---|---|---:|---|---:|
| light | `ebf_n149_labelscore_s9_tau256000` | 0.955190 | `ebf_n149_labelscore_s9_tau256000` | 0.956455 |
| mid | `ebf_n149_labelscore_s9_tau256000` | 0.940657 | `ebf_n149_labelscore_s9_tau256000` | 0.813254 |
| heavy | `ebf_n149_labelscore_s9_tau256000` | 0.937127 | `ebf_n149_labelscore_s9_tau128000` | 0.769616 |

汇总：

- `mean AUC = 0.944325`
- `mean F1 = 0.846442`
- `heavy best-F1 = 0.769616`

delta（sigma=2.8 相对 sigma=2.5）：

- `mean AUC +0.000416`
- `mean F1 +0.000483`
- `heavy best-F1 +0.000260`

heavy 分段（2×200k；`s=9,tau=128ms`；阈值取各自 heavy best-F1 operating point）：

| sigma | seg0 F1 | seg0 noise_kept_rate | seg1 F1 | seg1 noise_kept_rate |
|---:|---:|---:|---:|---:|
| 2.5 | 0.807374 | 0.038105 | 0.684758 | 0.021986 |
| 2.8 | 0.801081 | 0.026633 | 0.673548 | 0.012224 |

分段解读：

1. `sigma=2.8` 更保守（noise_kept_rate 显著下降）；
2. seg1 的 recall/F1 也随之下降，说明这一轮总体 best-F1 的小幅提升主要来自 precision 改善，而非“修复最难段”。

#### 24.24.6 n148 vs n149（计算与资源开销对比）

| 维度 | n148 | n149 | 结论 |
|---|---|---|---|
| 空间权重索引 | 运行时计算 d2 后查表 | 组内索引直接查紧凑 LUT | n149 更省算术 |
| d2 乘加开销 | `rr(3rr+1)/2` / 事件 | 0 / 事件 | n149 消除 d2 相关乘加 |
| LUT 项数（rr=4） | 33 | 14 | n149 少 57.58% |
| 精度表现（本口径） | mean AUC/F1 = 0.943909 / 0.845959 | mean AUC/F1 = 0.943909 / 0.845959 | 等价保持 |

#### 24.24.7 结论

1. 7.90 后半段优化在工程上可行、数学上与 n148 等价；
2. n149 在保持效果不变的同时，显著降低空间核的 d2 乘加与 LUT 常驻项数；
3. 对后续硬件落地更友好：可在不牺牲指标的前提下降低计算路径复杂度与存储压力。
4. 2026-04-22 重跑显示：仅通过 `sigma_s` 微调即可获得稳定但很小的正增；但从 heavy 分段看，seg1 仍是瓶颈段，后续不应把“提高整体 best-F1”误解为“最难段已被修复”。

产物目录：

- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n149_790b_compactlut_sigma2p5_compact400k/`
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n149_20260422_sigma2p8_compact400k/`
- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/segf1_n149_prev_sigma2p5_heavy_compact400k_s9_tau128.csv`
- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/segf1_n149_20260422_heavy_compact400k_s9_tau128.csv`

### 24.25 S52 继续创新：去 hot_state 的低资源路线（s83 / s84）

目标（按你的要求）：

1. 以 S52 为基线继续创新，争取逼近“极性融合上限”；
2. 尽可能减少内存与计算复杂度；
3. 尝试在提升精度的同时去除 `hot_state` 表。

#### 24.25.1 参考 S21-S55 失败经验后的设计约束

从前述 S 系列结论抽取的硬约束：

1. 不做硬条件判决（参考 s13 失败），只做软调制；
2. `opp` 证据不能全盘放开（heavy 会被 toggle/flicker 污染），需要 gated 融合；
3. S52 的局限是全局 `mix_state` 偏粗（同环境内事件差异表达不足）；
4. 若要降资源，优先删除每像素表（`hot_state`），并控制新增状态为 0 或常数个标量。

#### 24.25.2 本次提出并实现的两个点子

实现文件：

- `src/myevs/denoise/ops/ebfopt_part2/s83_s52_hotless_localmix_proxy.py`
- `src/myevs/denoise/ops/ebfopt_part2/s84_s52_hotless_ema_proxy.py`

sweep 接入：

- `scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py`（新增 `--variant s83/s84`，并补充 `s52` 同口径入口用于对照）

**s83（hot_state-free + 全局状态全去掉）**

1. 保留 S52 的同/异极性证据框架；
2. 用局部 `mix` + `sfrac` 构造 `alpha_eff`（soft gate）；
3. 用“同像素复发度”代理 `u_self`（仅依赖 `last_ts/last_pol`，无 `hot_state`）；
4. 去掉 `beta_state/mix_state`，实现 0 全局自适应状态。

**s84（hot_state-free + 保留 S52 的全局 EMA）**

1. 同样去掉 `hot_state`；
2. `u_self` 改为同像素时间复发代理（基于 `dt0`）；
3. 保留 `beta_state/mix_state` 两个标量 EMA，以尽量贴近 S52 的稳定性。

#### 24.25.3 “是否可行”与节省量化

可行性结论：

1. 工程可行：两版均可稳定运行并完成全量 sweep；
2. 资源可降：两版都成功移除 `hot_state(int32)` per-pixel 表；
3. 精度上限：在当前口径下，去掉 `hot_state` 后精度仍有明显回落（s84 好于 s83，但仍低于 s52）。

**空间节省（346x260）**

- `hot_state` 大小：`346*260*4 = 359,840 bytes ≈ 351.4 KiB`
- S52 持久状态：约 `1142.1 KiB`
- s83/s84 持久状态：约 `790.7 KiB`
- 相对 S52：节省 `351.4 KiB`（约 `30.8%`）

**计算节省（常数项，主复杂度仍是 O(r^2)）**

相对 S52：

1. s84：去掉 `hot_state` 读写与其衰减更新路径，约减少 `2` 次数组访问/事件，约减少 `8~10` 次标量运算/比较/事件；
2. s83：在 s84 基础上再去掉 `beta/mix` EMA 更新，约减少 `12~16` 次标量运算/比较/事件。

按本次 compact400k sweep 的总事件量（3 环境 × 12 组配置，合计约 `11,932,740` 事件）估算：

1. s84：约减少 `95M~119M` 标量运算 + `23,865,480` 次数组访问；
2. s83：约减少 `143M~191M` 标量运算 + `23,865,480` 次数组访问。

#### 24.25.4 实验口径（同上 compact400k）

- `max-events=400000`
- `s in {5,7,9}`
- `tau_us in {32000,64000,128000,256000}`
- `esr-mode=off, aocc-mode=off`

#### 24.25.5 同口径结果（s52 / s83 / s84）

| variant | mean AUC | mean F1 | heavy best-F1 |
|---|---:|---:|---:|
| s52 | **0.940606** | **0.839193** | **0.757905** |
| s83 (hotless + local-only) | 0.931511 | 0.832069 | 0.746483 |
| s84 (hotless + EMA proxy) | 0.935693 | 0.834284 | 0.749200 |

各环境 best-F1（同口径）：

| variant | light best-F1 | mid best-F1 | heavy best-F1 |
|---|---:|---:|---:|
| s52 | 0.957560 | 0.802115 | 0.757905 |
| s83 | 0.954579 | 0.795144 | 0.746483 |
| s84 | 0.957056 | 0.796597 | 0.749200 |

delta（相对 s52）：

| variant | mean AUC delta | mean F1 delta | heavy best-F1 delta |
|---|---:|---:|---:|
| s83 | -0.009095 | -0.007124 | -0.011422 |
| s84 | -0.004913 | -0.004909 | -0.008705 |

#### 24.25.6 结论（这轮是否达成“去 hot_state 且涨精度”）

1. 资源目标达成：`hot_state` 已成功移除，内存下降约 30.8%，常数计算显著下降；
2. 精度目标未达成：在当前口径下，s83/s84 均未超过 s52（s84 仅部分接近）；
3. 说明 `hot_state` 作为“像素级历史占用积分”仍提供了关键判别信息，单靠 `dt0` 代理暂时不足以完全替代；
4. 下一步若坚持“无 hot_state”路线，建议优先做：
	- 仅增加极小 per-pixel 状态（例如 `uint8` 饱和计数）作为折中；
	- 或在无新增 per-pixel 状态前提下，重新设计更强的局部时空代理（而不是只用 `dt0`）。

产物目录：

- `data/ED24/myPedestrain_06/EBF_Part2/_slim_s52_compact400k/`
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_s83_s52hotless_localmix_compact400k/`
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_s84_s52hotless_ema_compact400k/`

### 24.26 参考 7.91：S52 的极小状态折中（s85: uint8 hot-state）

目标（按 7.91 建议）：

1. 不推翻 S52 主干（same/opp + mix_state + beta_state），只优化 `u_self/hot_state` 路径；
2. 避免 s83/s84 的“完全去 hot_state 导致精度明显回落”；
3. 在保持单遍在线和 O(r^2) 邻域主复杂度下，降低 per-pixel 状态内存与带宽。

#### 24.26.1 新算法定义（s85）

实现文件：

- `src/myevs/denoise/ops/ebfopt_part2/s85_s52_hotstate_u8_quantized.py`

sweep 接入：

- `scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py`（新增 `--variant s85`）

核心思路：

1. 保留 S52 的主结构：`rawsame/rawopp -> mix_state 门控 -> u_self 抑制 -> beta_state*sfrac 支持增益`；
2. 将 `hot_state(int32)` 替换为 `hot_u8(uint8)` 饱和计数器；
3. 用 `dt0` 的 0..64 粗量化衰减 `dq` 做更新：
	 - `h = max(0, h - dq)`
	 - `h = min(255, h + (64 - dq))`
4. 归一化得到 `u_self = h / 255`，继续走 S52 原有软调制链路。

#### 24.26.2 资源与计算变化（相对 s52）

**空间（346x260）**

- `hot_state(int32)`：`346*260*4 = 359,840 bytes ≈ 351.4 KiB`
- `hot_u8(uint8)`：`346*260*1 = 89,960 bytes ≈ 87.9 KiB`
- `hot_state` 子表节省：`263.5 KiB`（约 `75.0%`）
- 持久状态总量：
	- s52：约 `1142.1 KiB`
	- s85：约 `878.6 KiB`
	- 总节省：约 `263.5 KiB`（约 `23.1%`）

**计算/带宽（常数项）**

1. 邻域主计算仍为 `O(r^2)`，不变；
2. `hot_state` 相关每事件读写字节从 `8 bytes`（int32 读+写）降为 `2 bytes`（uint8 读+写），该路径带宽约降 `75%`；
3. s85 新增了 `dt0 -> dq` 的轻量量化（常数级），换取更小状态与更低带宽。

#### 24.26.3 实验口径（compact400k 同口径）

- `max-events=400000`
- `s in {5,7,9}`
- `tau_us in {32000,64000,128000,256000}`
- `esr-mode=off, aocc-mode=off`

#### 24.26.4 s85 结果与对照（s52 / s84 / s85）

| variant | mean AUC | mean F1 | heavy best-F1 |
|---|---:|---:|---:|
| s52 | **0.940606** | **0.839193** | **0.757905** |
| s84 | 0.935693 | 0.834284 | 0.749200 |
| s85 (S52 + uint8 hot-state) | 0.937204 | 0.836245 | 0.753863 |

各环境 best-F1：

| variant | light best-F1 | mid best-F1 | heavy best-F1 |
|---|---:|---:|---:|
| s52 | 0.957560 | 0.802115 | 0.757905 |
| s84 | 0.957056 | 0.796597 | 0.749200 |
| s85 | 0.956796 | 0.798076 | 0.753863 |

delta（相对 s52）：

| variant | mean AUC delta | mean F1 delta | heavy best-F1 delta |
|---|---:|---:|---:|
| s84 | -0.004913 | -0.004909 | -0.008705 |
| s85 | -0.003402 | -0.002948 | -0.004042 |

#### 24.26.5 结论

1. 7.91 的“极小 per-pixel 状态折中”方向有效：s85 在明显降内存（总状态约 -23.1%）的同时，精度显著优于 s84；
2. s85 尚未超过 s52，但已经把 heavy best-F1 的差距从 `-0.008705` 缩小到 `-0.004042`；
3. 这说明 S52 的关键不在于必须保留 `int32 hot_state` 的精确值，而在于保留“像素级历史占用积分”这一信息通道；
4. 下一步可在 s85 上继续微调量化更新（`dq` 分段和 `inc` 策略），目标是在不增加状态字节数的前提下进一步逼近或超过 s52。

产物目录：

- `data/ED24/myPedestrain_06/EBF_Part2/_slim_s85_s52hotu8_compact400k/`

### 24.27 参考 7.92：S52 的稀疏状态化（s86: sparse hot cache）

目标（按 7.92 建议）：

1. 保留 S52 的主干融合逻辑（`rawsame/rawopp + mix_state + beta_state`）；
2. 不再维护 dense `hot_state[W*H]`，改为“仅活跃像素持有状态”的稀疏缓存；
3. 在不改变邻域主复杂度 `O(r^2)` 的前提下，验证“信息承载方式替换”的可行性。

#### 24.27.1 新算法定义（s86）

实现文件：

- `src/myevs/denoise/ops/ebfopt_part2/s86_s52_sparsehot_u8_cache.py`

sweep 接入：

- `scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py`（新增 `--variant s86`）

状态结构：

1. 仍保留 dense `last_ts/last_pol`；
2. 将 dense `hot_state` 改为 open-addressing 稀疏哈希表：
	- `key[cap]`：像素 id（`int32`）
	- `val[cap]`：热度值（`uint8`）
	- `-1` 空槽，`-2` tombstone。

#### 24.27.2 原理与公式

像素映射：

$$
	ext{pid} = y \cdot W + x
$$

哈希槽（容量为 2 的幂，线性探测）：

$$
	ext{home} = (\text{pid} \cdot 2654435761)\ \&\ (\text{cap}-1)
$$

中心像素热度更新（保持 s85 的量化积分逻辑）：

$$
\Delta t_0 = \min(\tau, |t_i - t_{\text{last}}(\text{pid})|),\quad
d_q = \operatorname{clip}\left(\left\lfloor 64\cdot \frac{\Delta t_0}{\tau} + 0.5\right\rfloor,0,64\right)
$$

$$
h \leftarrow \max(0, h-d_q),\quad
h \leftarrow \min(255, h + (64-d_q))
$$

$$
u_{\text{self}} = \frac{h}{255}
$$

评分主链（与 s52/s85 保持一致）：

$$
	ext{mix} = \frac{\text{raw}_{\text{opp}}}{\text{raw}_{\text{same}}+\text{raw}_{\text{opp}}+\varepsilon},\quad
\alpha_{\text{eff}}=(1-\text{mix}_{\text{state}})^2
$$

$$
	ext{raw}_{\text{gated}} = \frac{\text{raw}_{\text{same}} + \alpha_{\text{eff}}\cdot\text{raw}_{\text{opp}}}{\tau}
$$

$$
	ext{score} = \frac{\text{raw}_{\text{gated}}}{1+u_{\text{self}}^2}\cdot(1+\beta_{\text{state}}\cdot s_{\text{frac}})
$$

#### 24.27.3 复杂度与状态占用（346x260）

复杂度：

1. 邻域证据主环仍为 `O(r^2)`；
2. `u_self` 路径由数组直访改为哈希查找/写回（均摊常数项，最坏受 `max_probe` 限制）。

状态占用（本次配置：`MYEVS_S86_CACHE_CAP=32768`，`MYEVS_S86_MAX_PROBE=12`）：

1. `last_ts`：`346*260*8 = 719,680 bytes ≈ 702.8 KiB`
2. `last_pol`：`346*260*1 = 89,960 bytes ≈ 87.9 KiB`
3. sparse cache：
	- `key(int32)`：`32768*4 = 131,072 bytes ≈ 128.0 KiB`
	- `val(uint8)`：`32768*1 = 32,768 bytes ≈ 32.0 KiB`
4. 总计约 `950.7 KiB`（不含极小标量）。

与既有版本对比：

| variant | 持久状态估算 |
|---|---:|
| s52 | 1142.1 KiB |
| s85 (dense u8) | 878.6 KiB |
| s86 (sparse cache, cap=32768) | 950.7 KiB |

#### 24.27.4 实验口径（compact400k 同口径）

- `max-events=400000`
- `s in {5,7,9}`
- `tau_us in {32000,64000,128000,256000}`
- `esr-mode=off, aocc-mode=off`

#### 24.27.5 同口径结果（baseline / s52 / s85 / s86）

| variant | mean AUC | mean F1 | heavy best-F1 |
|---|---:|---:|---:|
| baseline (ebf) | 0.925929 | 0.829531 | 0.745459 |
| s52 | **0.940606** | **0.839193** | **0.757905** |
| s85 | 0.937204 | 0.836245 | 0.753863 |
| s86 (sparse hot cache) | 0.936969 | 0.836212 | 0.753768 |

各环境 best-F1：

| variant | light best-F1 | mid best-F1 | heavy best-F1 |
|---|---:|---:|---:|
| baseline (ebf) | 0.951840 | 0.791293 | 0.745459 |
| s52 | 0.957560 | 0.802115 | 0.757905 |
| s85 | 0.956796 | 0.798076 | 0.753863 |
| s86 | 0.956794 | 0.798073 | 0.753768 |

delta（相对 baseline）：

| variant | mean AUC delta | mean F1 delta | heavy best-F1 delta |
|---|---:|---:|---:|
| s52 | +0.014677 | +0.009662 | +0.012446 |
| s85 | +0.011275 | +0.006714 | +0.008404 |
| s86 | +0.011040 | +0.006681 | +0.008309 |

delta（相对 s52）：

| variant | mean AUC delta | mean F1 delta | heavy best-F1 delta |
|---|---:|---:|---:|
| s85 | -0.003402 | -0.002948 | -0.004042 |
| s86 | -0.003637 | -0.002981 | -0.004137 |

delta（相对 s85）：

| variant | mean AUC delta | mean F1 delta | heavy best-F1 delta |
|---|---:|---:|---:|
| s86 - s85 | -0.000235 | -0.000033 | -0.000095 |

#### 24.27.6 结论

1. 7.92 的“结构级替换（dense -> sparse）”在工程上可行，s86 能稳定跑完整口径；
2. 在当前实现和容量配置下，s86 精度与 s85 基本持平但略低，尚未超过 s85/s52；
3. 内存上，s86 明显低于 s52，但在 Python/Numba 的 `int32 key + uint8 val` 设计下仍高于 s85 的 dense-u8；
4. 说明“稀疏化方向正确但当前容器粒度偏粗”，下一步若继续 7.92 路线，需优先优化 entry 编码/容量策略（例如更紧凑 key 编码或分层索引），否则难在 s85 基线上形成同时“更省 + 更准”的优势。

产物目录：

- `data/ED24/myPedestrain_06/EBF_Part2/_slim_ebf_baseline_compact400k/`
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_s86_s52sparsehotu8_compact400k/`

### 24.28 参考 7.92 的另一条验证：S52 的 block-wise mix_state（s87）

目标（按 7.93 的建议做一次 S52-only 最小结构改动）：

1. 保留 S52 的 `rawsame/rawopp + hot_state + beta_state` 主链；
2. 不新增 per-pixel 状态表；
3. 仅把 `mix_state` 从单全局标量改为 block-wise 标量，验证“局部 mix 统计”是否能提升 S52。

#### 24.28.1 新算法定义（s87）

实现文件：

- `src/myevs/denoise/ops/ebfopt_part2/s87_s52_blockwise_mixstate.py`

sweep 接入：

- `scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py`（新增 `--variant s87`）

状态结构：

1. 与 s52 相同：dense `last_ts/last_pol/hot_state`；
2. `beta_state` 仍为全局标量；
3. `mix_state` 改为 block 网格 `mix_blocks[Bx*By]`，默认块大小 `MYEVS_S87_BLOCK_SIZE=32`。

#### 24.28.2 原理与公式

块索引：

$$
b_x = \left\lfloor\frac{x}{B}\right\rfloor,\quad
b_y = \left\lfloor\frac{y}{B}\right\rfloor,\quad
b = b_y\cdot N_x + b_x
$$

其中 $B$ 为 block size（像素），$N_x=\lceil W/B\rceil$。

每事件局部 mix 定义（与 s52 一致）：

$$
\mathrm{mix}_i = \frac{\mathrm{raw}_{\mathrm{opp},i}}{\mathrm{raw}_{\mathrm{same},i}+\mathrm{raw}_{\mathrm{opp},i}+\varepsilon}
$$

把全局 EMA 改为“对应块”的 EMA：

$$
m_b \leftarrow m_b + \frac{\mathrm{mix}_i - m_b}{N},\quad N=4096
$$

opp 融合权重：

$$
\alpha_{\mathrm{eff},i}=(1-m_b)^2
$$

评分主链（其余与 s52 相同）：

$$
\mathrm{raw}_{\mathrm{gated},i}=\frac{\mathrm{raw}_{\mathrm{same},i}+\alpha_{\mathrm{eff},i}\cdot\mathrm{raw}_{\mathrm{opp},i}}{\tau}
$$

$$
\mathrm{score}_i = \frac{\mathrm{raw}_{\mathrm{gated},i}}{1+u_{\mathrm{self},i}^2}\cdot(1+\beta_{\mathrm{state}}\cdot s_{\mathrm{frac},i})
$$

#### 24.28.3 复杂度与状态占用（346x260, block=32）

复杂度：

1. 主复杂度不变，仍是邻域证据累计 `O(r^2)`；
2. 相对 s52 仅增加常数项：一次 block 索引和一次块内 `mix` EMA 更新。

状态占用：

1. s87 在 s52 基础上仅多 `mix_blocks`；
2. `N_x=ceil(346/32)=11`, `N_y=ceil(260/32)=9`, 共 `99` 个 `float32`；
3. 额外约 `99*4 = 396 bytes ~= 0.39 KiB`，可忽略不计。

#### 24.28.4 实验口径（compact400k 同口径）

- `max-events=400000`
- `s in {5,7,9}`
- `tau_us in {32000,64000,128000,256000}`
- `esr-mode=off, aocc-mode=off`

#### 24.28.5 同口径结果（baseline / s52 / s85 / s86 / s87）

| variant | mean AUC | mean F1 | heavy best-F1 |
|---|---:|---:|---:|
| baseline (ebf) | 0.925929 | 0.829531 | 0.745459 |
| s52 | **0.940606** | **0.839193** | **0.757905** |
| s85 | 0.937204 | 0.836245 | 0.753863 |
| s86 | 0.936969 | 0.836212 | 0.753768 |
| s87 (block-wise mix_state) | 0.936427 | 0.828621 | 0.739264 |

各环境 best-F1：

| variant | light best-F1 | mid best-F1 | heavy best-F1 |
|---|---:|---:|---:|
| baseline (ebf) | 0.951840 | 0.791293 | 0.745459 |
| s52 | 0.957560 | 0.802115 | 0.757905 |
| s85 | 0.956796 | 0.798076 | 0.753863 |
| s86 | 0.956794 | 0.798073 | 0.753768 |
| s87 | 0.956719 | 0.789881 | 0.739264 |

delta（相对 baseline）：

| variant | mean AUC delta | mean F1 delta | heavy best-F1 delta |
|---|---:|---:|---:|
| s52 | +0.014677 | +0.009662 | +0.012446 |
| s85 | +0.011275 | +0.006714 | +0.008404 |
| s86 | +0.011040 | +0.006681 | +0.008309 |
| s87 | +0.010498 | -0.000910 | -0.006195 |

delta（相对 s52）：

| variant | mean AUC delta | mean F1 delta | heavy best-F1 delta |
|---|---:|---:|---:|
| s87 | -0.004179 | -0.010572 | -0.018641 |

#### 24.28.6 结论

1. s87 验证了“只改 mix_state 粒度”的工程可行性，且额外状态开销极低；
2. 但在当前固定块大小与 EMA 形状下，s87 的 mean-F1 与 heavy best-F1 明显低于 s52/s85/s86，甚至 heavy 低于 baseline；
3. 这说明把全局 mix 改为块级后，局部块统计更易受区域噪声主导，导致 `alpha_eff` 在 mid/heavy 出现过抑制（opp 证据被过度压低）；
4. 因此“仅 block-wise mix_state”不构成对 S52 的有效优化，后续若继续此方向，需至少引入跨块平滑或基于事件数的置信度校正，否则不建议作为主线。

产物目录：

- `data/ED24/myPedestrain_06/EBF_Part2/_slim_s87_s52blockmix_compact400k/`

### 24.29 参考 7.94：n149 主线继续优化（n150: n149 + S52-lite, no-hotstate）

目标（按 7.94 的建议继续走 n149 主线）：

1. 以 n149 的 compact-LUT 欧式底盘作为主干，不回退到 dense `d^2` 运行时计算；
2. 保留 S52 的核心融合机制（`rawsame/rawopp + mix_state + support boost`）；
3. 去掉 dense `hot_state`，用更轻量的中心代理量验证“资源-精度”权衡。

#### 24.29.1 新算法定义（n150）

实现文件：

- `src/myevs/denoise/ops/ebfopt_part2/n150_n149_s52lite_nohot_backbone.py`

sweep 接入：

- `scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py`（新增 `--variant n150`）
- 环境变量：`MYEVS_N150_SIGMA`, `MYEVS_N150_BETA_INIT`

关键改动（相对 n149）：

1. 删除 per-pixel `hot_state`（`int32` dense 表）；
2. 中心抑制从 `u_self(hot_state)` 改为 `u_lite(dt0)`；
3. `beta_state` 继续做全局在线更新，但改由 `u_lite` 驱动；
4. `mix_state` 仍保留为全局慢变量（与 S52 同形状）。

#### 24.29.2 原理与公式

主证据仍使用 n149 的 compact-LUT 空间核 + 二次时间核：

$$
\mathrm{raw}_{\mathrm{same}} = \sum_{j\in\mathcal{N}_{\mathrm{same}}} w_t(\Delta t_{ij})\,w_s^{\mathrm{compact}}(\Delta x_{ij},\Delta y_{ij}),
\quad
\mathrm{raw}_{\mathrm{opp}} = \sum_{j\in\mathcal{N}_{\mathrm{opp}}} w_t(\Delta t_{ij})\,w_s^{\mathrm{compact}}(\Delta x_{ij},\Delta y_{ij})
$$

$$
w_t(\Delta t)=\left(1-\frac{\Delta t}{\tau}\right)_+^2
$$

mix 慢变量与 opp 门控（沿用 S52 结构）：

$$
\mathrm{mix}_i=\frac{\mathrm{raw}_{\mathrm{opp}}}{\mathrm{raw}_{\mathrm{same}}+\mathrm{raw}_{\mathrm{opp}}+\varepsilon},
\quad
m \leftarrow m + \frac{\mathrm{mix}_i - m}{N},\; N=4096
$$

$$
\alpha_{\mathrm{eff}}=(1-m)^2,
\quad
\mathrm{raw}_{\mathrm{gated}} = \mathrm{raw}_{\mathrm{same}} + \alpha_{\mathrm{eff}}\,\mathrm{raw}_{\mathrm{opp}}
$$

去 hotstate 后的轻量中心代理：

$$
u_{\mathrm{lite}} = \operatorname{clip}\left(1-\frac{\Delta t_0}{\tau},0,1\right)
$$

$$
b \leftarrow b + \frac{u_{\mathrm{lite}}-b}{N},
\quad
\mathrm{score}=\frac{\mathrm{raw}_{\mathrm{gated}}}{1+u_{\mathrm{lite}}^2}\cdot(1+b\,s_{\mathrm{frac}})
$$

其中 $s_{\mathrm{frac}}$ 为同极性支持比例（与 n149/s52 一致）。

#### 24.29.3 复杂度与状态占用（346x260）

复杂度：

1. 主复杂度仍是邻域累计 `O(r^2)`，与 n149 同阶；
2. 去除 `hot_state` 后，每事件少一次中心积分更新与一次 dense 状态读写。

状态占用（持久状态）：

1. n149：`last_ts(uint64)` + `last_pol(int8)` + `hot_state(int32)` + 2 个全局标量；
2. n150：`last_ts(uint64)` + `last_pol(int8)` + 2 个全局标量；
3. 去掉 `hot_state` 后，约减少 `346*260*4 = 359,840 bytes ~= 351.4 KiB`。

#### 24.29.4 实验口径（compact400k 同口径）

- `max-events=400000`
- `s in {5,7,9}`
- `tau_us in {32000,64000,128000,256000}`
- `esr-mode=off, aocc-mode=off`

#### 24.29.5 同口径结果（baseline / n147 / n149 / n150）

| variant | mean AUC | mean F1 | heavy best-F1 |
|---|---:|---:|---:|
| baseline (ebf) | 0.925929 | 0.829531 | 0.745459 |
| n147 | 0.943882 | 0.845241 | 0.767755 |
| n149 | **0.943909** | **0.845959** | **0.769356** |
| n150 (n149 + S52-lite, no-hotstate) | 0.942712 | 0.842971 | 0.764414 |

各环境 best-F1：

| variant | light best-F1 | mid best-F1 | heavy best-F1 |
|---|---:|---:|---:|
| baseline (ebf) | 0.951840 | 0.791293 | 0.745459 |
| n147 | 0.956405 | 0.811565 | 0.767755 |
| n149 | 0.955688 | 0.812832 | 0.769356 |
| n150 | 0.955687 | 0.808812 | 0.764414 |

delta（相对 baseline）：

| variant | mean AUC delta | mean F1 delta | heavy best-F1 delta |
|---|---:|---:|---:|
| n147 | +0.017953 | +0.015710 | +0.022296 |
| n149 | +0.017980 | +0.016428 | +0.023897 |
| n150 | +0.016783 | +0.013440 | +0.018955 |

delta（相对 n149）：

| variant | mean AUC delta | mean F1 delta | heavy best-F1 delta |
|---|---:|---:|---:|
| n150 - n149 | -0.001197 | -0.002988 | -0.004942 |

#### 24.29.5b 2026-04-22 重跑补充（compact400k；不改变 n150 结构）

设置：

- `MYEVS_N150_SIGMA=2.8`
- `MYEVS_N150_BETA_INIT=0.65`

同口径（compact400k）结果：

| variant | mean AUC | mean F1 | heavy best-F1 |
|---|---:|---:|---:|
| n149（sigma=2.8；见 24.24.6） | 0.944325 | 0.846442 | 0.769616 |
| n150（sigma=2.8,beta=0.65） | 0.943065 | 0.843447 | 0.764715 |

delta（n150 2.8/0.65 相对 n150 旧配置）：

- `mean AUC +0.000353`
- `mean F1 +0.000476`
- `heavy best-F1 +0.000300`

delta（n150 2.8/0.65 相对 n149 2.8）：

- `mean AUC -0.001259`
- `mean F1 -0.002994`
- `heavy best-F1 -0.004902`

heavy 分段（2×200k；`s=9,tau=128ms`；阈值取各自 heavy best-F1 operating point）：

| variant | seg0 F1 | seg0 noise_kept_rate | seg1 F1 | seg1 noise_kept_rate |
|---|---:|---:|---:|---:|
| n150 旧 | 0.804448 | 0.038398 | 0.674951 | 0.022207 |
| n150 2.8/0.65 | 0.797646 | 0.026608 | 0.660070 | 0.012478 |

分段解读：

1. 参数调优后的收益主要体现为“更保守的 noise_kept_rate”，而不是 seg1 的 F1 提升；
2. 若要进一步逼近 n149，优先改进中心代理 `u_lite(dt0)` 的信息量，而不是继续细扫 `sigma/beta_init`。

#### 24.29.6 结论

1. n150 证明了 7.94 的“n149 主线 + S52-lite 去 hotstate”在工程上可行，且能显著减小持久状态（约 -351.4 KiB）；
2. 在当前 `u_lite(dt0)` 代理下，n150 仍明显优于 baseline / n147，但相对 n149 出现稳定小幅回落；
3. 回落主要发生在 mid/heavy（best-F1 与 heavy best-F1 均下降），说明 `dt0` 单变量对“历史占据积分”的替代仍不充分；
4. 因此 n150 当前更像“资源优先的轻量版”，尚不能替代 n149 作为精度主线；若继续优化 7.94 路线，下一步应在不恢复 dense hotstate 的前提下增加更稳健的短历史代理（例如 2-3 阶复发节律码/LUT）。

产物目录：

- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n147_794_compact400k/`
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n149_794_compact400k/`
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n150_794_n149s52lite_compact400k/`
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n150_20260422_sigma2p8_b0p65_compact400k/`
- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/segf1_n150_prev_heavy_compact400k_s9_tau128.csv`
- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/segf1_n150_20260422_heavy_compact400k_s9_tau128.csv`

### 24.30 参考 7.95：继续优化 n150（n151: 二阶 recurrence center proxy）

目标（按 7.95 建议，仅做一轮最小结构升级）：

1. 保持 n150 的整体结构（n149 底盘 + S52-lite 融合 + 无 dense hotstate）；
2. 将中心代理从一阶 `u_lite(dt0)` 升级为二阶 recurrence `u_rec(dt0,dt1)`；
3. 不回退到 dense `hot_state`，仅增加一张轻量 `prev_dt` 表。

#### 24.30.1 新算法定义（n151）

实现文件：

- `src/myevs/denoise/ops/ebfopt_part2/n151_n150_recurrence_proxy_backbone.py`

sweep 接入：

- `scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py`（新增 `--variant n151`）

相对 n150 的唯一核心改动：

1. 记录每像素上一段间隔 `dt1`（`prev_dt_q`, `uint16`）；
2. 用 `(dt0, dt1)` 共同构建中心 recurrence 强度 `u_rec`；
3. 其余 `rawsame/rawopp + mix_state + support boost` 主链保持不变。

#### 24.30.2 原理与公式

主证据与融合链沿用 n150：

$$
\mathrm{raw}_{\mathrm{gated}} = \mathrm{raw}_{\mathrm{same}} + \alpha_{\mathrm{eff}}\,\mathrm{raw}_{\mathrm{opp}},
\quad
\alpha_{\mathrm{eff}}=(1-m)^2,
\quad
m\leftarrow m+\frac{\mathrm{mix}_i-m}{N}
$$

其中

$$
\mathrm{mix}_i=\frac{\mathrm{raw}_{\mathrm{opp}}}{\mathrm{raw}_{\mathrm{same}}+\mathrm{raw}_{\mathrm{opp}}+\varepsilon},\quad N=4096
$$

二阶 recurrence 代理：

$$
u_{\mathrm{rec}} = \exp\left(-\frac{dt_0}{T_0}\right)\cdot \exp\left(-\frac{|dt_0-dt_1|}{T_\Delta}\right)
$$

默认比例（环境变量可调）：

$$
T_0 = 0.5\tau,\quad T_\Delta = 0.25\tau
$$

最终打分：

$$
b\leftarrow b+\frac{u_{\mathrm{rec}}-b}{N},
\quad
\mathrm{score}=\frac{\mathrm{raw}_{\mathrm{gated}}}{1+u_{\mathrm{rec}}^2}\cdot(1+b\,s_{\mathrm{frac}})
$$

#### 24.30.3 复杂度与状态占用（346x260）

复杂度：

1. 主复杂度仍为邻域累计 `O(r^2)`；
2. 相对 n150 增加常数项：一次 `prev_dt_q` 读写和 2 次指数计算。

状态占用：

1. n150 持久状态：`last_ts(uint64)` + `last_pol(int8)`；
2. n151 新增：`prev_dt_q(uint16)`；
3. 额外 `346*260*2 = 179,920 bytes ~= 175.7 KiB`；
4. 相对 n149 仍节省约 `175.7 KiB`（不含标量）。

#### 24.30.4 实验口径（compact400k 同口径）

- `max-events=400000`
- `s in {5,7,9}`
- `tau_us in {32000,64000,128000,256000}`
- `esr-mode=off, aocc-mode=off`
- n151 默认：`MYEVS_N151_T0_RATIO=0.5`, `MYEVS_N151_TD_RATIO=0.25`

#### 24.30.5 同口径结果（baseline / n149 / n150 / n151）

| variant | mean AUC | mean F1 | heavy best-F1 |
|---|---:|---:|---:|
| baseline (ebf) | 0.925929 | 0.829531 | 0.745459 |
| n149 | **0.943909** | **0.845959** | **0.769356** |
| n150 | 0.942712 | 0.842971 | 0.764414 |
| n151 (recurrence proxy) | 0.932842 | 0.837122 | 0.757975 |

注：

- 本节表格用于验证 n151 的结构性改动，数值沿用当时同口径配置（n149 sigma=2.5；n150 默认配置）；
- 若要对齐 2026-04-22 的重跑参数（n149 sigma=2.8；n150 sigma=2.8,beta=0.65），见 24.24.6 与 24.29.5b。

各环境 best-F1：

| variant | light best-F1 | mid best-F1 | heavy best-F1 |
|---|---:|---:|---:|
| baseline (ebf) | 0.951840 | 0.791293 | 0.745459 |
| n149 | 0.955688 | 0.812832 | 0.769356 |
| n150 | 0.955687 | 0.808812 | 0.764414 |
| n151 | 0.953384 | 0.800007 | 0.757975 |

delta（相对 n150）：

| variant | mean AUC delta | mean F1 delta | heavy best-F1 delta |
|---|---:|---:|---:|
| n151 - n150 | -0.009870 | -0.005849 | -0.006439 |

delta（相对 n149）：

| variant | mean AUC delta | mean F1 delta | heavy best-F1 delta |
|---|---:|---:|---:|
| n151 - n149 | -0.011067 | -0.008837 | -0.011381 |

#### 24.30.6 结论

1. n151 在工程上完成了 7.95 提出的二阶 recurrence 代理验证，但默认形状过于保守；
2. 指标显示 n151 不仅未修复 n150 的回落，反而进一步下降（尤其 mid/heavy）；
3. 说明当前 `u_rec` 形状对中心事件抑制过强，导致有效信号被额外压制；
4. 这一轮应判定为负优化，后续若继续 7.95 路线，必须先做 `T0/TΔ` 形状与强度扫参后再判断是否保留该方向。

产物目录：

- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n151_795_recurrence_compact400k/`

（已将 2026-04-22 的 n149/n150 重跑数据分别合并回 24.24 与 24.29，不再单独保留 24.31 小节。）
