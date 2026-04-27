# myEVS 去噪算法对比实验总控 README

本 README 仅用于 `src/myevs/denoise/ops` 维度的跨算法对比实验管理，约束后续：
- 数据集路径
- 结果存储路径
- 扫频/运行脚本入口
- 指标统计与汇总表格式

当前状态（2026-04-23）：
- 已集成算法：`BAF`, `STCF(stc)`, `EBF`, `EBF_OPTIMIZED`, `KNOISE`, `EVFLOW`, `YNOISE`, `TS`, `MLPF`, `PFD`。
- `n175` 演化暂停，先做跨数据集横向对比。

## 1. 路径定义（已核验）

项目根目录：
- `D:\hjx_workspace\scientific_reserach\projects\myEVS`

本 README：
- `D:\hjx_workspace\scientific_reserach\projects\myEVS\src\myevs\denoise\ops\README.md`

ED24 数据集（已存在）：
- `D:\hjx_workspace\scientific_reserach\dataset\ED24\myPedestrain_06`

Driving 数据集（已存在）：
- `D:\hjx_workspace\scientific_reserach\dataset\DND21\mydriving\driving_noise_light_slomo_shot_withlabel`
- `D:\hjx_workspace\scientific_reserach\dataset\DND21\mydriving\driving_noise_mid_slomo_shot_withlabel`
- `D:\hjx_workspace\scientific_reserach\dataset\DND21\mydriving\driving_noise_heavy_slomo_shot_withlabel`

脚本目录（已存在）：
- `D:\hjx_workspace\scientific_reserach\projects\myEVS\scripts\ED24_alg_evalu`
- `D:\hjx_workspace\scientific_reserach\projects\myEVS\scripts\driving_alg_evalu`
- `D:\hjx_workspace\scientific_reserach\projects\myEVS\scripts\noise_analyze`

结果根目录：
- `D:\hjx_workspace\scientific_reserach\projects\myEVS\data`

## 2. 输出目录规范

统一规范：
- `data/<DATASET>/<SCENE>/<ALGORITHM>/roc_<algorithm>_<level>.csv`
- `data/<DATASET>/<SCENE>/<ALGORITHM>/roc_<algorithm>_<level>.png`

已约定示例：
- `data/ED24/myPedestrain_06/BAF/roc_baf_light.csv`
- `data/ED24/myPedestrain_06/KNOISE/roc_knoise_mid.csv`
- `data/DND21/mydriving/light/EVFLOW/roc_evflow_light.csv`

命名要求：
- 算法目录统一大写（如 `KNOISE`），文件名前缀统一小写（如 `roc_knoise_*.csv`）。
- `tag` 字段必须体现关键超参（如 `evflow_r2_tau3000`）。

## 3. 新增算法接入说明

算法 method token（`myevs.cli denoise/roc/sweep --method`）：
- `knoise`（id=12）
- `evflow`（id=13）
- `ynoise`（id=14）
- `ts`（id=15）
- `mlpf`（id=16）
- `pfd`（id=17）

参数映射（统一沿用 myEVS 通用参数）：
- `time-us` -> 算法中的时间窗/衰减参数（duration/decay）
- `radius-px` -> 空间搜索半径（search radius）
- `min-neighbors` -> 阈值（intThreshold/floatThreshold）

说明：
- `mlpf` 支持两种模式：  
1. 真实模型推理模式（推荐）：传 `--mlpf-model`，加载 TorchScript `.pt`；  
2. 代理评分模式（兼容）：未传模型时回退到 MLP-inspired 特征评分。

### 3.1 对比算法原理与公式（论文写作口径）

以下公式按当前代码实现给出（不是通用文献形式）。统一记号：
- 事件 \(e_i=(x_i,y_i,t_i,p_i)\)，\(p_i\in\{+1,-1\}\)。
- 邻域 \(\mathcal{N}_r(i)\) 为 \([x_i-r,x_i+r]\times[y_i-r,y_i+r]\) 的裁剪窗口。
- 时间窗 \(\tau\) 对应 `time-us` 转换到 ticks。

`BAF`（`src/myevs/denoise/ops/baf.py`）：
- 判据是“邻域是否存在任意近期事件”（不看极性）。
$$
\text{keep}_i=\mathbf{1}\!\left[\exists j\in \mathcal{N}_r(i)\setminus\{i\},\ t_j\in[t_i-\tau,t_i]\right].
$$
- 说明：这版是布尔门限，不是计数阈值版本。

`STCF/STC`（`src/myevs/denoise/ops/stc.py`）：
- 统计同极性、近期邻居数（不含中心点）：
$$
C_i^{(p)}=\sum_{j\in \mathcal{N}_r(i)\setminus\{i\}}
\mathbf{1}[t_j\in[t_i-\tau,t_i]]\cdot \mathbf{1}[p_j=p_i].
$$
- 判决：
$$
\text{keep}_i=\mathbf{1}\!\left[C_i^{(p)}\ge \theta_n\right],\quad \theta_n=\texttt{min-neighbors}.
$$

`EBF`（`src/myevs/denoise/ops/ebf.py`）：
- 你指出的是对的：当前实现是“时间线性核 + 极性门控 + 空间等权求和”，没有空间高斯核。
- 对每个邻居 \(j\)（不含中心点）：
$$
w_t(i,j)=\max\!\left(0,\,1-\frac{|t_i-t_j|}{\tau}\right),\qquad
w_p(i,j)=\mathbf{1}[p_j=p_i].
$$
- 评分：
$$
S_i=\sum_{j\in \mathcal{N}_r(i)\setminus\{i\}} w_t(i,j)\,w_p(i,j).
$$
- 判决：
$$
\text{keep}_i=\mathbf{1}[S_i>\theta_f],\quad \theta_f=\texttt{min-neighbors}.
$$

`N149`（`src/myevs/denoise/ops/ebfopt_part2/n149_n145_s52_euclid_compactlut_backbone.py`）：
- 基底项：同/反极性分别累计，时间核是平方线性衰减，空间核是欧式高斯 LUT。
$$
w_t(i,j)=\left(\max\!\left(0,1-\frac{\Delta t_{ij}}{\tau}\right)\right)^2,\qquad
w_s(i,j)=\exp\!\left(-\frac{\|q_i-q_j\|_2^2}{2\sigma^2}\right).
$$
$$
R_i^{+}=\sum_j w_t w_s\mathbf{1}[p_j=p_i],\qquad
R_i^{-}=\sum_j w_t w_s\mathbf{1}[p_j=-p_i].
$$
- 自激活与混合门控（代码里的 `u_self`, `b`, `mstate`）：
$$
u_i=\frac{h_i}{h_i+\tau/2+\varepsilon},\quad
\alpha_i=(1-m_i)^2,\quad
\widetilde{R}_i=R_i^{+}+\alpha_i R_i^{-}.
$$
- 主分数与支持率增益：
$$
B_i=\frac{\widetilde{R}_i}{1+u_i^2},\qquad
s_i=\frac{\#\text{support}}{\#\text{possible}},\qquad
S_i=B_i\,(1+b_i s_i).
$$
- 最终由外部阈值扫频决策（脚本里对 `S_i` 取阈值）。

`KNOISE`（`src/myevs/denoise/ops/knoise.py`）：
- 该实现是“行/列最近事件索引”模型，不是完整邻域累和。
- 在 \(x\pm1,x\) 三列与 \(y\pm1,y\) 三行检查最近事件，满足“同极性 + \(\Delta t\le\tau\) + 邻接几何约束”则记 1 支持：
$$
S_i=\sum_{k\in\mathcal{C}_i}\mathbf{1}[\text{time\_ok}\land\text{pol\_ok}\land\text{adj\_ok}],
$$
$$
\text{keep}_i=\mathbf{1}[S_i\ge \theta_n].
$$

`EVFLOW`（`src/myevs/denoise/ops/evflow.py`）：
- 在局部窗口内最小二乘拟合平面 \(t=ax+by+c\)，再映射为流幅值：
$$
\min_{a,b,c}\sum_{j\in\mathcal{N}_r(i)}\left((t_j-t_i)\cdot10^{-3}-(ax_j+by_j+c)\right)^2.
$$
$$
v_i=\sqrt{\left(-\frac{1}{a}\right)^2+\left(-\frac{1}{b}\right)^2}.
$$
- 判决方向是“小于等于阈值保留”：
$$
\text{keep}_i=\mathbf{1}[v_i\le \theta_f].
$$

`YNOISE`（`src/myevs/denoise/ops/ynoise.py`）：
- 统计邻域内“近期且同极性”事件数（当前实现包含中心像素位置）：
$$
D_i=\sum_{j\in\mathcal{N}_r(i)}
\mathbf{1}[|t_i-t_j|\le\tau]\cdot\mathbf{1}[p_j=p_i].
$$
$$
\text{keep}_i=\mathbf{1}[D_i\ge \theta_n].
$$

`TS`（`src/myevs/denoise/ops/ts.py`）：
- 同极性时间表面的邻域均值：
$$
T_i(j)=\exp\!\left(-\frac{|t_i-t_j|}{\tau}\right),\qquad
S_i=\frac{1}{|\Omega_i|}\sum_{j\in\Omega_i}T_i(j),
$$
其中 \(\Omega_i=\{j\in\mathcal{N}_r(i)\mid t_j>0\}\)。
$$
\text{keep}_i=\mathbf{1}[S_i\ge \theta_f].
$$

`MLPF`（`src/myevs/denoise/ops/mlpf.py`）：
- 当前实现是固定 \(7\times7\)（\(r=3\)）的轻量代理，不是实际神经网络推理。
- 对同极性且 \(\Delta t\le\tau\) 的像素累加线性 recency：
$$
\text{recency}_{ij}=1-\frac{|t_i-t_j|}{\tau},\qquad
S_i=\sum_{j\in\mathcal{N}_{r=3}(i)} \text{recency}_{ij}\,\mathbf{1}[p_j=p_i].
$$
$$
\text{keep}_i=\mathbf{1}[S_i\ge \theta_f].
$$

`PFD`（`src/myevs/denoise/ops/pfd.py`）：
- 当前实现对齐 `PFDs`（event-by-event 版本），含两级判决。
- Stage-1（同极性时域支持）：
$$
C_i=\sum_{j\in \mathcal{N}_r(i)\setminus\{i\}}
\mathbf{1}[p_j=p_i]\cdot\mathbf{1}[|t_i-t_j|<\tau],\qquad
\text{pass1}_i=\mathbf{1}[C_i\ge v].
$$
其中 \(v\) 对应 `refractory-us`（本项目默认固定为 1）。
- Stage-2（极性翻转一致性）：
设 \(F_i(\tau)\) 为像素 \(i\) 在时间窗 \(\tau\) 内的极性翻转次数，\
\(A_i=\sum_{j\in\mathcal{N}_r(i)\setminus\{i\}}\mathbf{1}[|t_i-t_j|\le\tau]\) 为活跃邻居数。
$$
S_i=\left|F_i(\tau)-\frac{1}{A_i}\sum_{j\in\mathcal{N}_r(i)\setminus\{i\}}F_j(\tau)\right|.
$$
$$
\text{keep}_i=\mathbf{1}\big[\text{pass1}_i\land (A_i>\theta_n)\land (S_i\le 1)\big],
$$
其中 \(\theta_n=\texttt{min-neighbors}\)。
- 模式说明（已实现）：
1. `pfd_mode=a`（默认，PFD-A）  
   \(S_i=\left|F_i(\tau)-\frac{1}{A_i}\sum_j F_j(\tau)\right|\)
1. `pfd_mode=b`（PFD-B）  
   \(S_i=\left|\frac{1}{A_i}\sum_j F_j(\tau)\right|\)
1. 当前 ED24 仅跑 `pfd_mode=a`；`pfd_mode=b` 已实现但未在本轮数据中启用（留给闪烁噪声数据集）。

评估指标定义（统一）：
$$
\mathrm{TPR}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}},\qquad
\mathrm{FPR}=\frac{\mathrm{FP}}{\mathrm{FP}+\mathrm{TN}}.
$$
$$
\mathrm{Precision}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}},\qquad
\mathrm{Recall}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}.
$$
$$
\mathrm{F1}=\frac{2\cdot \mathrm{Precision}\cdot \mathrm{Recall}}{\mathrm{Precision}+\mathrm{Recall}}.
$$
$$
\mathrm{AUC}=\int_0^1 \mathrm{TPR}(u)\,d u,\quad (u=\mathrm{FPR}).
$$

`MESR` 与 `AOCC` 为无参考结构指标：统一在 `best-AUC` 与 `best-F1` 两个工作点各记录一次。

### 3.2 复杂度、实时性与资源占用结论（实现视角）

结论先行：
- 你感觉 `EVFLOW` 慢是合理的。当前实现每个事件都要遍历时间窗内 `deque` 做局部筛选，再做一次最小二乘，平均复杂度不再是固定邻域常数级，数据密集时会明显变慢。
- `YNOISE/EBF/TS` 同属局部窗口打分，复杂度主项都是 \(O(K)\)（\(K=(2r+1)^2\)），但常数项不同：`TS` 有 `exp`，`EBF` 有浮点线性权重，`YNOISE` 主要是整数计数。
- `KNOISE` 是最轻量的对比算法（固定方向检查，近似 \(O(1)\) / event），最有利于实时。

统一记号：
$$
P = W\times H,\quad K=(2r+1)^2,\quad N=\text{事件总数}.
$$

| 算法 | 单事件时间复杂度 | 全序列复杂度 | 状态内存（粗估） | 单事件操作量（粗估） | 实时风险 |
|---|---:|---:|---:|---:|---|
| BAF | \(O(K)\)（有早停） | \(O(NK)\) | \(8P\) bytes | 最坏约 \(K-1\) 次邻居时间比较 | 中 |
| STCF | \(O(K)\)（有早停） | \(O(NK)\) | \(16P\) bytes | 最坏约 \(K-1\) 次 + 同极性判断 | 中-高 |
| EBF | \(O(K)\) | \(O(NK)\) | \(9P\) bytes | 约 \(K-1\) 次；含线性时间权重与累加 | 中 |
| YNOISE | \(O(K)\) | \(O(NK)\) | \(9P\) bytes | 约 \(K\) 次；硬窗 + 同极性计数 | 中 |
| TS | \(O(K)\) | \(O(NK)\) | \(16P\) bytes | 约 \(K\) 次 + `exp` | 中-高 |
| PFD | \(O(K)\)（含两级门控） | \(O(NK)\) | \(\approx 33P\) bytes | 邻域支持 + 翻转计数统计 | 中-高 |
| MLPF(real/proxy) | \(O(49)+O(\text{MLP forward})\) | \(O(N(49+\text{forward}))\) | \(9P\) bytes + 模型参数 | 固定49点特征 + 前向推理 | 中 |
| KNOISE | 近似 \(O(1)\) | \(O(N)\) | \(\approx 13(W+H)\) bytes | 固定行列方向检查（常数次） | 低 |
| EVFLOW | \(O(Q)+O(M)\) | 常见 \(O(N\bar Q)\)，最坏接近 \(O(N^2)\) | \(O(Q)\)（deque） | 遍历窗口事件 + 最小二乘拟合 | 高 |
| N149 | \(O(K)\)（常数较大） | \(O(NK)\) | \(\approx 13P\) bytes + LUT | 约 \(K\) 次 + 多门控状态更新 | 中-高 |

说明：
- `Q` 表示 `EVFLOW` 时间窗内候选事件数，`M` 为拟合矩阵构建与求解代价；在高事件率场景，`Q` 增大是主要瓶颈。
- 上表为实现复杂度与状态量估算，用于实时性预判；最终仍以实测 `events/s` 和 `runtime_sec` 为准。

#### 3.2.1 排序（按对比指标）

1. 按时间复杂度（快 -> 慢）  
`KNOISE` > `MLPF` > `BAF ≈ STCF ≈ YNOISE ≈ EBF ≈ TS ≈ PFD ≈ N149` > `EVFLOW`

1. 按单事件操作量（少 -> 多）  
`KNOISE` > `BAF` > `YNOISE` > `EBF` > `MLPF` > `STCF ≈ TS ≈ PFD` > `N149` > `EVFLOW`

1. 按状态内存占用（少 -> 多）  
`KNOISE` > `EBF ≈ YNOISE ≈ MLPF` > `BAF` > `STCF ≈ TS` > `PFD` > `N149` > `EVFLOW(随Q增长)`

1. 按实时风险（低 -> 高）  
`KNOISE` < `MLPF` < `BAF ≈ YNOISE ≈ EBF` < `STCF ≈ TS ≈ PFD ≈ N149` < `EVFLOW`

#### 3.2.2 为什么 EVFLOW 慢（本工程实现）

`EVFLOW` 在 `src/myevs/denoise/ops/evflow.py` 中采用：
1. 维护时间窗 `deque`；  
1. 对每个新事件遍历窗内事件并按半径筛选局部样本；  
1. 对局部样本执行一次最小二乘平面拟合（`np.linalg.lstsq`）；  
1. 由拟合参数换算流幅值再判决。  

因此它不是固定邻域常数代价，而是“与时间窗候选规模相关”的代价模型，密集场景下明显更慢。

## 4. 一键实验脚本（新增）

### 4.1 ED24

总入口（按算法参数自动扫频并输出 ROC）：
- `scripts/ED24_alg_evalu/run_slomo_alg.ps1`

单算法入口：
- `scripts/ED24_alg_evalu/run_slomo_knoise.ps1`
- `scripts/ED24_alg_evalu/run_slomo_evflow.ps1`
- `scripts/ED24_alg_evalu/run_slomo_ynoise.ps1`
- `scripts/ED24_alg_evalu/run_slomo_ts.ps1`
- `scripts/ED24_alg_evalu/run_slomo_mlpf.ps1`
- `scripts/ED24_alg_evalu/run_slomo_pfd.ps1`

运行示例：
```powershell
cd D:\hjx_workspace\scientific_reserach\projects\myEVS
powershell -ExecutionPolicy Bypass -File .\scripts\ED24_alg_evalu\run_slomo_knoise.ps1
```

### 4.2 Driving

总入口：
- `scripts/driving_alg_evalu/run_driving_alg.ps1`

单算法入口：
- `scripts/driving_alg_evalu/run_driving_knoise.ps1`
- `scripts/driving_alg_evalu/run_driving_evflow.ps1`
- `scripts/driving_alg_evalu/run_driving_ynoise.ps1`
- `scripts/driving_alg_evalu/run_driving_ts.ps1`
- `scripts/driving_alg_evalu/run_driving_mlpf.ps1`
- `scripts/driving_alg_evalu/run_driving_pfd.ps1`

运行示例：
```powershell
cd D:\hjx_workspace\scientific_reserach\projects\myEVS
powershell -ExecutionPolicy Bypass -File .\scripts\driving_alg_evalu\run_driving_evflow.ps1
```

Driving 脚本会在每个噪声级目录自动查找：
- `*signal_only*.npy` 或 `*clean*.npy` 作为 clean
- 其余 `.npy` 作为 noisy

## 5. scripts 目录功能说明（补充）

### 5.1 根目录脚本

- `scripts/v2e_labeled_txt_to_npy.py`：v2e 标注 txt -> npy
- `scripts/split_labeled_events.py`：从 labeled npy 拆分 clean(signal_only)/noise
- `scripts/ED24csv_to_npy.py`：ED24 csv -> npy
- `scripts/openeb_csv_to_evtq.py`：OpenEB csv -> evtq
- `scripts/run_experiment.py`：通用实验驱动脚本（历史）
- `scripts/eval_bestpoint_mesr_aocc.py`：从已有 ROC CSV 自动回算 `best-AUC`/`best-F1` 点的 `MESR/AOCC`（支持单算法或多算法）

### 5.2 ED24_alg_evalu

- `run_slomo_baf.ps1`：ED24 上 BAF ROC 扫频
- `run_slomo_stcf.ps1`：ED24 上 STCF ROC 扫频
- `run_slomo_ebf.ps1`：ED24 上 EBF ROC 扫频
- `run_slomo_ebf_paper_s_tau.ps1`：EBF 论文风格参数扫频
- `run_slomo_fdf.ps1`：FastDecayFilter 相关扫频
- `summarize_best_params_ed24.py`：汇总 ED24 最优参数
- `summarize_best_params_ebf_optimized_ed24.py`：汇总 EBF_OPTIMIZED 最优参数
- `summarize_fixed_threshold_ebf_optimized_ed24.py`：固定阈值统计
- `select_best_tag_ebfopt_ed24.py`：从 tag 选择最佳
- `summarize_horizontal_round1.py`：Round1（BAF/STCF/EBF/N149）横向汇总
- `summarize_horizontal_round2_new_methods.py`：Round2（KNOISE/EVFLOW/YNOISE/TS/MLPF/PFD）横向汇总
- `summarize_runtime_ed24.py`：ED24 运行时统一口径汇总（按 level 最新记录统计）
- `sweep_*.py`/`prescreen_*.py`/`tune_*.py`：各版本候选公式预筛与调参
- `run_slomo_alg.ps1`（新增）：KNOISE/EVFLOW/YNOISE/TS/MLPF/PFD 统一入口
- `run_slomo_{knoise|evflow|ynoise|ts|mlpf|pfd}.ps1`（新增）：单算法入口

### 5.3 driving_alg_evalu

- `run_driving_alg.ps1`（新增）：Driving 数据集统一入口
- `run_driving_{knoise|evflow|ynoise|ts|mlpf|pfd}.ps1`（新增）：单算法入口

### 5.4 noise_analyze

用于噪声结构分析、特征统计、分布可视化和误检类型分析（FP/transition/pattern 等），为算法改进提供先验证据。

## 6. 统一指标与汇总表

目标指标：
- `AUC`
- `F1`
- `MESR`
- `AOCC`
- `runtime_sec`（运行时）

建议总汇 CSV（后续维护）：
- `data/summary/alg_compare_master.csv`

列定义（最小必需）：
- `dataset, scene, level, algorithm, tag, auc, f1, mesr, aocc, time_us, radius_px, threshold, csv_path, fig_path, note`

要求：
- 每次新增实验必须写入总汇 CSV；`csv_path/fig_path` 必填。
- `tag` 与 `time_us/radius_px/threshold` 必须可互相还原。
- 每次实验必须写入运行时（`runtime_sec` 或 `runtime_min`）。
- 对支持的脚本，`best-AUC` 与 `best-F1` 两个点都要给出 `MESR/AOCC`。

## 7. 推荐执行顺序

1. 先跑 ED24：`BAF/STCF/EBF + KNOISE/EVFLOW/YNOISE/TS/MLPF/PFD`
2. 再跑 Driving：同样算法集合
3. 每个算法先看三档噪声（light/mid/heavy）的 AUC 稳定性
4. 再做跨数据集总表排序，筛选论文主结果算法

## 8. 约束与后续补充规则

- 所有新增结果必须进入本 README 对应章节与 `data/summary/alg_compare_master.csv`。
- 任何脚本新增后，必须在“scripts 目录功能说明”登记用途。
- 若改动数据路径或命名规则，先改本 README，再跑实验。

## 9. radius-px 统一口径（2026-04-23）

- `radius-px` 统一定义为“半径”，不是直径。
- 直径与半径关系：`diameter = 2 * radius + 1`。
- 若脚本使用 `s` 或 `d`（窗口直径），必须先转换为 `radius` 再传给 `--radius-px`。
- 已完成统一的脚本：
`scripts/ED24_alg_evalu/run_slomo_baf.ps1`、`scripts/ED24_alg_evalu/run_slomo_stcf.ps1`、`scripts/ED24_alg_evalu/run_slomo_ebf_paper_s_tau.ps1`。

## 10. ED24 横向结果汇总（Round1 + Round2）

说明：以下三张表已经把两轮结果合并在一起。
- Round1：`BAF / STCF / EBF / N149`
- Round2：`KNOISE / EVFLOW / YNOISE / TS / MLPF / PFD`

数据来源：
- `data/ED24/myPedestrain_06/horizontal_summary_all.csv`

### 10.1 light

| Algorithm | Round | Best AUC | Best F1 Tag | Best F1 | F1 Threshold | MESR@Best-F1 | AOCC@Best-F1 |
|---|---|---:|---|---:|---:|---:|---:|
| BAF | Round1 | 0.902368 | baf_r4 | 0.941875 | 64000.000000 | 1.204377 | 0.831391 |
| STCF | Round1 | 0.946044 | stcf_r4 | 0.947306 | 256000.000000 | 1.186034 | 0.826256 |
| EBF | Round1 | 0.933627 | ebf_r5_tau512000 | 0.951967 | 2.000000 | 1.138791 | 0.825984 |
| N149 | Round1 | 0.953997 | n149_r5_tau512000_light | 0.959438 | 0.787335 | 0.996116 | 0.823566 |
| KNOISE | Round2 | 0.716790 | knoise_tau16000 | 0.911931 | 0.000000 | 1.670778 | 0.849902 |
| EVFLOW | Round2 | 0.819366 | evflow_r5_tau64000 | 0.918792 | 80.000000 | 1.883983 | 0.792855 |
| YNOISE | Round2 | 0.933149 | ynoise_r5_tau256000 | 0.948630 | 3.000000 | 1.159098 | 0.821053 |
| TS | Round2 | 0.870558 | ts_r4_decay128000 | 0.930806 | 0.050000 | 1.434201 | 0.826768 |
| MLPF | Round2 | 0.866857 | mlpf_tau32000 | 0.943647 | 0.500000 | 1.021404 | 0.824071 |
| PFD | Round2 | 0.902708 | pfd_r3_tau64000_m1 | 0.908376 | 1.000000 | 0.915288 | 0.793225 |

### 10.2 mid

| Algorithm | Round | Best AUC | Best F1 Tag | Best F1 | F1 Threshold | MESR@Best-F1 | AOCC@Best-F1 |
|---|---|---:|---|---:|---:|---:|---:|
| BAF | Round1 | 0.839146 | baf_r1 | 0.686279 | 16000.000000 | 1.059620 | 0.843543 |
| STCF | Round1 | 0.895906 | stcf_r2 | 0.783758 | 32000.000000 | 1.019594 | 0.830355 |
| EBF | Round1 | 0.918293 | ebf_r5_tau128000 | 0.812238 | 6.000000 | 1.028522 | 0.795376 |
| N149 | Round1 | 0.946041 | n149_r5_tau256000_mid | 0.839209 | 2.928605 | 0.973047 | 0.800048 |
| KNOISE | Round2 | 0.662463 | knoise_tau16000 | 0.495071 | 1.000000 | 0.864277 | 0.590452 |
| EVFLOW | Round2 | 0.796419 | evflow_r4_tau16000 | 0.712599 | 80.000000 | 2.038876 | 0.798920 |
| YNOISE | Round2 | 0.908342 | ynoise_r4_tau64000 | 0.808960 | 6.000000 | 1.036820 | 0.783468 |
| TS | Round2 | 0.852829 | ts_r2_decay32000 | 0.714195 | 0.100000 | 1.600892 | 0.825788 |
| MLPF | Round2 | 0.814019 | mlpf_tau32000 | 0.659253 | 0.300000 | 1.135691 | 0.817674 |
| PFD | Round2 | 0.888911 | pfd_r3_tau32000_m3 | 0.794614 | 1.000000 | 0.919803 | 0.800480 |

### 10.3 heavy

| Algorithm | Round | Best AUC | Best F1 Tag | Best F1 | F1 Threshold | MESR@Best-F1 | AOCC@Best-F1 |
|---|---|---:|---|---:|---:|---:|---:|
| BAF | Round1 | 0.816080 | baf_r1 | 0.553047 | 8000.000000 | 0.931767 | 0.835285 |
| STCF | Round1 | 0.879051 | stcf_r2 | 0.700514 | 16000.000000 | 0.946833 | 0.819898 |
| EBF | Round1 | 0.890842 | ebf_r3_tau64000 | 0.756961 | 4.000000 | 0.985282 | 0.749912 |
| N149 | Round1 | 0.934184 | n149_r5_tau128000_heavy | 0.788696 | 2.965526 | 0.951670 | 0.784665 |
| KNOISE | Round2 | 0.641662 | knoise_tau16000 | 0.430141 | 1.000000 | 0.803600 | 0.584283 |
| EVFLOW | Round2 | 0.779147 | evflow_r2_tau16000 | 0.619358 | 80.000000 | 2.388692 | 0.730146 |
| YNOISE | Round2 | 0.897144 | ynoise_r3_tau64000 | 0.752252 | 6.000000 | 1.015795 | 0.791337 |
| TS | Round2 | 0.846531 | ts_r2_decay16000 | 0.647695 | 0.100000 | 1.113610 | 0.758223 |
| MLPF | Round2 | 0.772873 | mlpf_tau32000 | 0.518727 | 0.200000 | 0.949902 | 0.780568 |
| PFD | Round2 | 0.875718 | pfd_r3_tau32000_m3 | 0.727360 | 5.000000 | 0.931863 | 0.777345 |

### 10.4 MESR/AOCC 统一口径（ED24）

数据来源：
- `data/ED24/myPedestrain_06/horizontal_summary_all.csv`（用于论文主表，统一采用 `best-f1` 对应的 `MESR/AOCC`）
- `data/ED24/myPedestrain_06/bestpoint_mesr_aocc_summary.csv`（保留 `best-auc` 与 `best-f1` 两个工作点明细）

当前完整性检查（2026-04-27）：
- ED24 横向汇总应有 `10 algorithms × 3 levels = 30` 行，当前已齐全。
- `MESR/AOCC` 在 `horizontal_summary_all.csv` 中已全部可用（无空缺）。

### 10.5 PFD 数据结论（ED24）

- 本轮扫频口径：固定 `r=3`，扫 `Δt(time-us)`、`λ(min-neighbors)`、`m(refractory-us)`。
- 与论文默认口径不一致说明：论文主实现强调 `3x3(r=1)` 邻域；本工程在 ED24 上 `r=3` 的 AUC/F1 更优，故主表先采用 `r=3` 工程最优口径，同时保留论文口径说明用于复现实验章节。
- AUC 对比（PFD）：`light=0.902708`, `mid=0.888911`, `heavy=0.875718`。  
在 `mid/heavy` 上明显高于 `EVFLOW/TS/KNOISE/MLPF`，但低于 `YNOISE/EBF/N149`。
- F1 对比（PFD）：`light=0.908376`, `mid=0.794614`, `heavy=0.727360`。  
在 `mid/heavy` 上稳定高于 `TS/EVFLOW/KNOISE/MLPF`，与 `STCF` 接近或略优。
- 代价侧：旧版纯 Python 扫频确实很慢；本轮改为 `numba` 后，PFD 三档平均运行时已降到约 `76.418s/档`（coarse 网格）。论文中的“实时”来自 C++/FPGA 实现，本工程当前仍是 Python + numba 版本。

## 11. N147 补充结果（EBF_Part2 / compact400k）

数据来源：
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n147_794_compact400k/roc_ebf_n147_{light,mid,heavy}_labelscore_s5_7_9_tau32_64_128_256ms.csv`

| Algorithm | Level | Best AUC | Best AUC Tag | Best F1 | Best F1 Tag | Threshold |
|---|---|---:|---|---:|---|---:|
| N147 | light | 0.954903 | ebf_n147_labelscore_s9_tau256000 | 0.956405 | ebf_n147_labelscore_s9_tau256000 | 0.568591 |
| N147 | mid | 0.940047 | ebf_n147_labelscore_s9_tau256000 | 0.811565 | ebf_n147_labelscore_s9_tau256000 | 3.291225 |
| N147 | heavy | 0.936697 | ebf_n147_labelscore_s9_tau256000 | 0.765521 | ebf_n147_labelscore_s9_tau256000 | 4.651618 |

## 12. 运行时统计（ED24）

统一口径（避免“分钟/秒混用”）：
1. 每个算法读取各自 `runtime_*.csv`；
2. 对 `light/mid/heavy` 每个 level 只取“最新一条”记录；
3. 统计 `avg_sec_per_level`（每档平均秒数）和 `sum_sec_3levels`（三档总秒数）。

统一汇总文件：
- `data/ED24/myPedestrain_06/runtime_unified_ed24.csv`

生成命令：
```powershell
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/summarize_runtime_ed24.py
```

| Algorithm | light(s) | mid(s) | heavy(s) | avg_sec_per_level | sum_sec_3levels |
|---|---:|---:|---:|---:|---:|
| EBF | 14.035 | 17.324 | 20.545 | 17.301 | 51.904 |
| N149 | 55.374 | 61.798 | 67.643 | 61.605 | 184.815 |
| TS | 16.654 | 24.341 | 30.248 | 23.748 | 71.243 |
| PFD | 34.418 | 72.914 | 121.923 | 76.418 | 229.255 |
| BAF | 55.613 | 172.831 | 265.363 | 164.602 | 493.807 |
| STCF |  |  |  |  |  |
| KNOISE | 85.460 | 238.140 | 376.325 | 233.308 | 699.925 |
| EVFLOW | 174.066 | 548.679 | 1239.166 | 653.970 | 1961.911 |
| YNOISE | 275.119 | 777.092 | 1234.050 | 762.087 | 2286.261 |
| MLPF | 428.342 | 1239.780 | 578.035 | 748.719 | 2246.157 |

注：
- `STCF` 当前缺 `runtime_stcf.csv`，因此留空；补跑 `run_slomo_stcf.ps1` 后可自动纳入统一表。
- 在这个统一口径下，`PFD` 不是最慢算法，明显慢于 PFD 的是 `EVFLOW/YNOISE/MLPF`。

后续新增实验必须同步记录 runtime，推荐写入以下脚本的 runtime CSV：
- `scripts/ED24_alg_evalu/run_slomo_alg.ps1`
- `scripts/driving_alg_evalu/run_driving_alg.ps1`
- `scripts/ED24_alg_evalu/run_slomo_baf.ps1`
- `scripts/ED24_alg_evalu/run_slomo_stcf.ps1`
- `scripts/ED24_alg_evalu/run_slomo_ebf.ps1`
- `scripts/ED24_alg_evalu/run_n149_labelscore_grid.py`

## 13. MESR / AOCC 统计规则（新约束）

从现在开始，所有“最终汇报结果”都按以下规则执行：
1. 在 `best-AUC` 对应参数点计算并记录 `MESR/AOCC`。
2. 在 `best-F1` 对应参数点计算并记录 `MESR/AOCC`。
3. 若脚本支持，优先打开：
   - `--esr-mode best`
   - `--aocc-mode best`
4. 在 README 表格中明确区分：`AUC@best-AUC`, `F1@best-F1`, `MESR/AOCC@best-AUC`, `MESR/AOCC@best-F1`。

推荐执行方式：
1. EBF Part2 变体：继续使用原 sweep 脚本原生开关。
- `scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py`
- `scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py`
2. 横向对比算法（BAF/STCF/EBF/N149/KNOISE/EVFLOW/YNOISE/TS/MLPF/PFD）：统一使用后评估脚本。
- `scripts/eval_bestpoint_mesr_aocc.py`
- 该脚本从 ROC CSV 自动选 `best-AUC`/`best-F1` 参数点并计算 `MESR/AOCC`。

## 14. 两个 EBF 扫频脚本的区别（明确说明）

- `scripts/ED24_alg_evalu/run_slomo_ebf.ps1`
1. 用途：横向对比优先（和 BAF/STCF/N149 放在同一标准下比较）。
2. 扫频方式：`radius × tau × threshold` 常规网格。
3. 结果文件：`roc_ebf_{light,mid,heavy}.csv/.png`。

- `scripts/ED24_alg_evalu/run_slomo_ebf_paper_s_tau.ps1`
1. 用途：论文复现和消融分析。
2. 扫频方式：两步法。
   - Step A：固定 `tau=64ms`，扫描 `s`（窗口直径，脚本内部换算为 `radius`）。
   - Step B：固定 `s=5`，扫描 `tau`。
3. 结果文件：`roc_ebf_*_paper_*`。

建议：
- 做跨算法排名：优先用 `run_slomo_ebf.ps1`。
- 做机理说明/论文图：优先用 `run_slomo_ebf_paper_s_tau.ps1`。

横向对比绘图降采样规则（2026-04-23 起）：
- `EBF`：每个 `r` 仅绘制 AUC 最优 `3` 条曲线。
- `EVFLOW`：每个 `r` 仅绘制 AUC 最优 `3` 条曲线。
- `YNOISE`：每个 `r` 仅绘制 AUC 最优 `3` 条曲线。
- `TS`：每个 `r` 仅绘制 AUC 最优 `3` 条曲线。
- `PFD`：每个 `r` 仅绘制 AUC 最优 `3` 条曲线。
- `MLPF`：全局仅绘制 AUC 最优 `4` 条（按 `tau` 标签筛选）。

### 14.1 连续扫频（Dense）策略

`run_slomo_alg.ps1` 与 `run_driving_alg.ps1` 新增：
- `-SweepProfile coarse|dense`

说明：
- `coarse`：保持原有稀疏网格（速度快，适合首轮横向对比）。
- `dense`：更细阈值网格（近连续扫频，适合最终参数定点）。

建议：
- `EVFLOW`：优先使用 `dense`，原因是其 ROC 对阈值/时间窗更敏感，稀疏阈值容易导致曲线点过于集中。
- `TS`、`YNOISE`、`MLPF`、`PFD`：当你发现 best 点贴近阈值边界、或曲线锯齿明显时，再切 `dense`。
- `KNOISE`：通常 `coarse` 已足够；只有在论文图需要精细局部曲线时再用 `dense`。

## 15. 脚本使用方法（完整）

### 15.1 ED24 新算法总入口（可选一个或多个）

`run_slomo_alg.ps1` 支持全量、单算法、多算法：

```powershell
# 默认：跑全部（knoise/evflow/ynoise/ts/mlpf/pfd）
powershell -ExecutionPolicy Bypass -File ./scripts/ED24_alg_evalu/run_slomo_alg.ps1

# 单算法（兼容参数）
powershell -ExecutionPolicy Bypass -File ./scripts/ED24_alg_evalu/run_slomo_alg.ps1 -Algorithm knoise

# 多算法
powershell -ExecutionPolicy Bypass -File ./scripts/ED24_alg_evalu/run_slomo_alg.ps1 -Algorithms knoise,ts,mlpf,pfd

# 显式全量
powershell -ExecutionPolicy Bypass -File ./scripts/ED24_alg_evalu/run_slomo_alg.ps1 -Algorithms all

# EVFLOW 连续扫频（推荐）
powershell -ExecutionPolicy Bypass -File ./scripts/ED24_alg_evalu/run_slomo_alg.ps1 -Algorithms evflow -SweepProfile dense

# 多算法连续扫频
powershell -ExecutionPolicy Bypass -File ./scripts/ED24_alg_evalu/run_slomo_alg.ps1 -Algorithms evflow,ts,mlpf,pfd -SweepProfile dense
```

### 15.1.1 Driving 新算法总入口（支持 dense）

```powershell
# 单算法（coarse，默认）
powershell -ExecutionPolicy Bypass -File ./scripts/driving_alg_evalu/run_driving_alg.ps1 -Algorithm evflow

# EVFLOW 连续扫频（推荐）
powershell -ExecutionPolicy Bypass -File ./scripts/driving_alg_evalu/run_driving_alg.ps1 -Algorithm evflow -SweepProfile dense

# 其他算法 dense 示例
powershell -ExecutionPolicy Bypass -File ./scripts/driving_alg_evalu/run_driving_alg.ps1 -Algorithm ts -SweepProfile dense
powershell -ExecutionPolicy Bypass -File ./scripts/driving_alg_evalu/run_driving_alg.ps1 -Algorithm pfd -SweepProfile dense
```

### 15.2 ED24 传统算法入口

```powershell
powershell -ExecutionPolicy Bypass -File ./scripts/ED24_alg_evalu/run_slomo_baf.ps1
powershell -ExecutionPolicy Bypass -File ./scripts/ED24_alg_evalu/run_slomo_stcf.ps1
powershell -ExecutionPolicy Bypass -File ./scripts/ED24_alg_evalu/run_slomo_ebf.ps1
powershell -ExecutionPolicy Bypass -File ./scripts/ED24_alg_evalu/run_slomo_ebf_paper_s_tau.ps1
powershell -ExecutionPolicy Bypass -File ./scripts/ED24_alg_evalu/run_slomo_n149.ps1
```

### 15.3 横向汇总脚本

```powershell
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/summarize_horizontal_all.py
```
说明：
- 上面一条命令会自动依次运行：
`summarize_horizontal_round1.py` + `summarize_horizontal_round2_new_methods.py`
- 并输出合并文件：
`data/ED24/myPedestrain_06/horizontal_summary_all.csv`
- 同时会自动读取：
`data/ED24/myPedestrain_06/bestpoint_mesr_aocc_summary.csv`
并把 `best-f1` 对应的 `mesr/aocc` 回填到总表。
- 回填策略是“有新值才覆盖、无值不改”，避免把已有算法结果误清空。

STCF 口径确认（按代码）：
- `src/myevs/denoise/ops/stc.py` 中使用 `last_on` / `last_off` 两张图分别计时。
- 当前事件只在“同极性”邻域上计数（并且不包含中心像素），不是异极性累计。

### 15.4 MESR/AOCC 后评估脚本（支持一个或多个算法）

```powershell
# ED24：默认所有算法、三档噪声、best-AUC+best-F1，同时计算 MESR/AOCC
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/eval_bestpoint_mesr_aocc.py --dataset ed24

# ED24：只算一个算法（例如 baf）
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/eval_bestpoint_mesr_aocc.py --dataset ed24 --algorithms baf

# ED24：一次算多个算法
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/eval_bestpoint_mesr_aocc.py --dataset ed24 --algorithms baf,stcf,ebf,knoise

# ED24：只算 PFD
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/eval_bestpoint_mesr_aocc.py --dataset ed24 --algorithms pfd

# ED24：PFD-B（实现已支持，后续闪烁噪声数据集使用）
D:/software/Anaconda_envs/envs/myEVS/python.exe -m myevs.cli roc --clean <clean.npy> --noisy <noisy.npy> --assume npy --width 346 --height 260 --method pfd --pfd-mode b --engine numba --radius-px 1 --time-us 32000 --refractory-us 1 --param min-neighbors --values 1,2,3 --out-csv <out.csv>

# ED24：真实 MLPF（按 level 自动套模型）
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/eval_bestpoint_mesr_aocc.py --dataset ed24 --algorithms mlpf --mlpf-model-pattern "data/ED24/myPedestrain_06/MLPF/mlpf_torch_{level}.pt"

# Driving：只算 mid/heavy，且只算 best-F1
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/eval_bestpoint_mesr_aocc.py --dataset driving --levels mid,heavy --points best-f1

# 指标可选：只算 MESR 或只算 AOCC
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/eval_bestpoint_mesr_aocc.py --dataset ed24 --metrics mesr
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/eval_bestpoint_mesr_aocc.py --dataset ed24 --metrics aocc
```

默认输出：
- ED24：`data/ED24/myPedestrain_06/bestpoint_mesr_aocc_summary.csv`
- Driving：`data/DND21/mydriving/bestpoint_mesr_aocc_summary.csv`

写入策略（避免互相覆盖）：
- 采用“增量更新”模式：默认保留历史结果。
- 仅替换本次运行涉及到的行（键：`dataset + algorithm + level + point`）。
- 例如先跑 `baf` 再跑 `stcf`，两者会同时保留；再次跑 `stcf` 才会覆盖旧 `stcf` 行。

### 15.5 MLPF 训练与真实推理

```powershell
# 1) 训练（按 light / mid / heavy 分别训练）
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/train_mlpf_torch.py --clean "D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_1.8_signal_only.npy" --noisy "D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_1.8.npy" --out-model "data/ED24/myPedestrain_06/MLPF/mlpf_torch_light.pt" --out-meta "data/ED24/myPedestrain_06/MLPF/mlpf_torch_light.json" --width 346 --height 260 --patch 7 --epochs 5 --batch-size 4096 --max-events 120000
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/train_mlpf_torch.py --clean "D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_2.5_signal_only.npy" --noisy "D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_2.5.npy" --out-model "data/ED24/myPedestrain_06/MLPF/mlpf_torch_mid.pt" --out-meta "data/ED24/myPedestrain_06/MLPF/mlpf_torch_mid.json" --width 346 --height 260 --patch 7 --epochs 5 --batch-size 4096 --max-events 120000
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/train_mlpf_torch.py --clean "D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3_signal_only.npy" --noisy "D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3.npy" --out-model "data/ED24/myPedestrain_06/MLPF/mlpf_torch_heavy.pt" --out-meta "data/ED24/myPedestrain_06/MLPF/mlpf_torch_heavy.json" --width 346 --height 260 --patch 7 --epochs 5 --batch-size 4096 --max-events 120000

# 2) 用真实模型跑 ED24 MLPF ROC
powershell -ExecutionPolicy Bypass -File ./scripts/ED24_alg_evalu/run_slomo_alg.ps1 -Algorithm mlpf -SweepProfile coarse -MlpfModelPattern "data/ED24/myPedestrain_06/MLPF/mlpf_torch_{level}.pt"
```

### 15.6 N147（EBF_Part2）示例

```powershell
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py --variant n147 --max-events 400000 --s-list 5,7,9 --tau-us-list 32000,64000,128000,256000 --out-dir data/ED24/myPedestrain_06/EBF_Part2/_slim_n147_794_compact400k --esr-mode off --aocc-mode off
```
### 15.7 N147 增加了通用绘图脚本（直接吃 ROC CSV）
```powershell
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/plot_roc_with_auc.py `
  --in-csv data/ED24/myPedestrain_06/EBF_Part2/_slim_n147_794_compact400k/roc_ebf_n147_heavy_labelscore_s5_7_9_tau32_64_128_256ms.csv `
  --out-png data/ED24/myPedestrain_06/EBF_Part2/_slim_n147_794_compact400k/roc_ebf_n147_heavy_auclegend.png `
  --title "N147 ROC" `
  --strip-env-suffix
```
## 16. 参数调优位置（TUNE_HERE）

注：`run_slomo_alg.ps1` 与 `run_driving_alg.ps1` 可先用 `-SweepProfile coarse` 快速定位，再用 `-SweepProfile dense` 精细搜索。

- `scripts/ED24_alg_evalu/run_slomo_baf.ps1`：`$RADIUS_LIST`, `$TAU_LIST`
- `scripts/ED24_alg_evalu/run_slomo_stcf.ps1`：`$RADIUS_LIST`, `$TAU_LIST`
- `scripts/ED24_alg_evalu/run_slomo_ebf.ps1`：`$EBF_RADIUS_LIST`, `$EBF_TAU_LIST`, `$EBF_THR_LIST`
- `scripts/ED24_alg_evalu/run_slomo_alg.ps1`：每个算法块中的 `thr/r/tau` 网格
- `scripts/ED24_alg_evalu/run_slomo_n149.ps1`：`--radius-list`, `--tau-us-list`
- `scripts/driving_alg_evalu/run_driving_alg.ps1`：同 ED24 的参数区
