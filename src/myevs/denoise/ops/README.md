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
- `D:\hjx_workspace\scientific_reserach\dataset\DND21\mydriving\driving_noise_light_mid_slomo_shot_withlabel`
- `D:\hjx_workspace\scientific_reserach\dataset\DND21\mydriving\driving_noise_mid_slomo_shot_withlabel`

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
- `data/DND21/mydriving/EVFLOW/roc_evflow_light.csv`

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
cd D:\hjx_workspace\scientific_re                                           serach\projects\myEVS
powershell -ExecutionPolicy Bypass -File .\scripts\ED24_alg_evalu\run_slomo_knoise.ps1
```

### 4.2 Driving

总入口：
- `scripts/driving_alg_evalu/run_driving_alg.ps1`
- `scripts/driving_alg_evalu/run_driving_alg_paper.ps1`（论文口径数据入口）

单算法入口：
- `scripts/driving_alg_evalu/run_driving_knoise.ps1`
- `scripts/driving_alg_evalu/run_driving_evflow.ps1`
- `scripts/driving_alg_evalu/run_driving_ynoise.ps1`
- `scripts/driving_alg_evalu/run_driving_ts.ps1`
- `scripts/driving_alg_evalu/run_driving_mlpf.ps1`
- `scripts/driving_alg_evalu/run_driving_pfd.ps1`
- `scripts/driving_alg_evalu/run_driving_n149.ps1`（N149 独立入口，调用 labelscore 网格脚本）

运行示例：
```powershell
cd D:\hjx_workspace\scientific_reserach\projects\myEVS
powershell -ExecutionPolicy Bypass -File .\scripts\driving_alg_evalu\run_driving_evflow.ps1
```

Driving 脚本会在每个噪声级目录自动查找：
- `*signal_only*.npy` 或 `*clean*.npy` 作为 clean
- 其余 `.npy` 作为 noisy

#### 4.2.1 Driving 论文口径数据生成（1/3/5 Hz）

生成脚本：
- `scripts/driving_alg_evalu/generate_driving_paper_noise.py`

输入（paper clean）：
- `D:\hjx_workspace\scientific_reserach\dataset\DND21\mydriving_paper\v2e-dvs-events.txt`

输出目录：
- `D:\hjx_workspace\scientific_reserach\dataset\DND21\mydriving_paper\driving_noise_light_paper_withlabel`
- `D:\hjx_workspace\scientific_reserach\dataset\DND21\mydriving_paper\driving_noise_light_mid_paper_withlabel`
- `D:\hjx_workspace\scientific_reserach\dataset\DND21\mydriving_paper\driving_noise_mid_paper_withlabel`

生成命令：
```powershell
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/driving_alg_evalu/generate_driving_paper_noise.py `
  --clean-txt "D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_paper/v2e-dvs-events.txt" `
  --out-root "D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_paper" `
  --width 346 --height 260 --sigma-decades 0.5 --seed 12345
```

直接跑论文口径 Driving（默认仍输出到 `data/DND21/mydriving/{ALG}`）：
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\driving_alg_evalu\run_driving_alg_paper.ps1 -Algorithm ebf -MaxEvents 200000
powershell -ExecutionPolicy Bypass -File .\scripts\driving_alg_evalu\run_driving_alg_paper.ps1 -Algorithm all -MaxEvents 200000
```

口径说明（重要）：
1. 当前生成方式不是“直接调用 JAER 可执行流程”，而是“参考 JAER `NoiseTesterFilter` 思路在 Python 中实现”：
   - clean 基线来自 `mydriving_paper/v2e-dvs-events.txt`
   - 噪声按像素独立 Poisson 过程生成
   - 像素噪声率引入 log-normal 空间不均匀性（FPN）
2. `sigma-decades=0.5` 的含义：
   - 先在 \(\log_{10}\) 域采样像素噪声率扰动，标准差为 `0.5`
   - 直观上表示像素噪声率存在约“半个 decade 量级”的离散
3. `FPN`（Fixed Pattern Noise）含义：
   - 传感器不同像素的固有噪声率不同，且这种差异在时间上相对固定
   - 代码中体现为每个像素固定一个噪声率 \(r_{x,y}\)，再在时域按 Poisson 采样
4. 参数选择依据：
   - `1/3/5 Hz`：对齐 EBF/DND21 常用 driving 噪声档位
   - `sigma-decades=0.5`：对齐 JAER `NoiseTesterFilter` 与 DND21 文献中常见 FPN 设定
5. 本版噪声类型：
   - 当前仅添加 **shot noise**
   - 未添加 leak noise、leak jitter、记录噪声回放等更完整 JAER 分支
6. 与“严格 JAER 一致”的边界：
   - 本版属于“统计口径对齐”的复现，不是二进制级别同实现
   - 若要完全一致，需走 JAER 原链路导出（含其内部事件调度与可选 leak/noise-recording 分支）

JAER 原链路导出数据转换（2026-04-29 补充）：

生成脚本：
- `scripts/driving_alg_evalu/convert_jaer_driving_to_npy.py`

输入：
- clean: `D:\hjx_workspace\scientific_reserach\dataset\DND21\mydriving_paper\driving.aedat`
- noisy: `D:\hjx_workspace\scientific_reserach\dataset\DND21\mydriving_jaer\driving_noise_{light,light_mid,mid}\driving_jaer_shot_{1,3,5}hz.aedat`

转换命令：
```powershell
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/driving_alg_evalu/convert_jaer_driving_to_npy.py `
  --clean-aedat "D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_paper/driving.aedat" `
  --jaer-root "D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_jaer" `
  --out-root "D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_jaer" `
  --width 346 --height 260 --tick-ns 12.5 --overwrite
```

转换质量检查：

| Level | Target Hz/pixel | Noisy events | Matched signal | Estimated noise | Estimated Hz/pixel |
|---|---:|---:|---:|---:|---:|
| light | 1 | 4,285,026 | 3,748,579 | 536,447 | 0.996 |
| light_mid | 3 | 5,357,184 | 3,748,579 | 1,608,605 | 2.988 |
| mid | 5 | 6,431,569 | 3,748,578 | 2,682,991 | 4.983 |

说明：
1. `matched signal` 通过 `(t,x,y,p)` 精确匹配 clean AEDAT 恢复，约覆盖 clean 的 `99.81%`。
2. `Estimated Hz/pixel` 与目标 `1/3/5Hz` 基本一致，说明 JAER 导出的加噪强度是正确的。
3. 少量 unmatched clean 事件来自 JAER filtered logging 起止边界或重构细节，对整体 AUC 影响很小。

ED24 外部 Driving 加噪数据转换（2026-04-29 补充）：

源目录：
- `D:\hjx_workspace\scientific_reserach\dataset\DND21\mydriving_ED24`

每个噪声档位目录包含的文件含义：

| 文件 | 含义 | 是否用于本次 EBF |
|---|---|---|
| `1hz.txt` ... `7hz.txt` | v2e 从黑色视频生成的纯噪声事件，时间戳为秒，无 label | 否，仅用于理解噪声来源 |
| `mix_result.txt` | 另一份混合结果，带 epoch-like 微秒时间戳，`label=0` 数量恒定约 53 万；不像当前 DND21 driving 主序列 | 否 |
| `driving_mix_result.txt` | Driving 主序列混合结果，`label=0` 数量恒定约 273 万，`label=1` 随噪声档位增加 | 是 |
| `v2e-args.txt` | v2e 生成参数记录 | 仅作为元数据参考 |
| `*.avi` | 可视化预览文件，部分档位存在 | 否 |

标签口径：
- ED24 外部文件：`label=0` 表示 signal，`label=1` 表示 noise。
- myEVS 统一口径：`label=1` 表示 signal，`label=0` 表示 noise。
- 因此转换时必须翻转 label，否则 ROC/F1 会完全错误。

转换脚本：
- `scripts/driving_alg_evalu/convert_ed24_driving_to_npy.py`

转换命令：
```powershell
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/driving_alg_evalu/convert_ed24_driving_to_npy.py --overwrite
```

输出目录：
- `D:\hjx_workspace\scientific_reserach\dataset\DND21\mydriving_ED24\driving_noise_{1hz...9hz}_ed24_withlabel`

转换质量检查：

| Level | Events | Signal | Noise | Duration (s) | Estimated Hz/pixel |
|---|---:|---:|---:|---:|---:|
| 1hz | 3,081,599 | 2,732,313 | 349,286 | 5.976212 | 0.650 |
| 2hz | 3,429,942 | 2,732,313 | 697,629 | 5.976212 | 1.298 |
| 3hz | 3,777,023 | 2,732,313 | 1,044,710 | 5.976212 | 1.943 |
| 4hz | 4,126,122 | 2,732,313 | 1,393,809 | 5.976212 | 2.593 |
| 5hz | 4,478,144 | 2,732,313 | 1,745,831 | 5.976212 | 3.247 |
| 6hz | 4,822,218 | 2,732,313 | 2,089,905 | 5.976212 | 3.887 |
| 7hz | 5,169,768 | 2,732,313 | 2,437,455 | 5.976212 | 4.534 |
| 8hz | 5,515,979 | 2,732,313 | 2,783,666 | 5.976212 | 5.178 |
| 9hz | 5,867,956 | 2,732,313 | 3,135,643 | 5.976212 | 5.832 |

注意：
1. 这批数据的目录名是 `1...9hz`，但按事件数和时长估算的实际噪声率约为标称值的 `0.65` 倍。
2. 因此它不适合直接当作论文 `1/3/5 Hz/pixel` 绝对对齐数据，但适合验证“外部生成 Driving 加噪数据”下 EBF 的相对表现。

ED24 外部 Driving 数据上的 EBF 结果：

运行设置：
- EBF 参数：`radius=2`（论文 `s=5`）、`\(\tau=32000us\)`。
- 时间戳单位：`--tick-ns 1000`，因为 `driving_mix_result.txt` 时间戳为微秒整数。
- 阈值：`0.0:0.1:15.0`。
- 为避免 Python EBF 对每个阈值重复运行，本次使用 myEVS Numba EBF score 内核一次性计算 score，再按阈值生成 ROC；计算口径等价于 `score > threshold`。

结果文件：
- `data/DND21/mydriving_ED24/EBF/roc_ebf_{1hz...9hz}_ed24_paperfix.csv`
- `data/DND21/mydriving_ED24/EBF/ebf_ed24_driving_1to9hz_summary.csv`
- `data/DND21/mydriving_ED24/EBF/ebf_ed24_driving_allhz_summary.csv`

| Level | Estimated Hz/pixel | AUC step=0.1 | Exact score AUC | Best-F1 threshold | TPR | FPR | Precision | Best F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1hz | 0.650 | 0.950013 | 0.950048 | 0.4 | 0.973967 | 0.295572 | 0.962654 | 0.968278 |
| 2hz | 1.298 | 0.948619 | 0.948665 | 0.9 | 0.960809 | 0.226274 | 0.943281 | 0.951964 |
| 3hz | 1.943 | 0.946503 | 0.946547 | 1.4 | 0.940665 | 0.174585 | 0.933738 | 0.937189 |
| 4hz | 2.593 | 0.944914 | 0.944962 | 1.8 | 0.926896 | 0.148886 | 0.924266 | 0.925579 |
| 5hz | 3.247 | 0.943417 | 0.943472 | 2.1 | 0.915241 | 0.138341 | 0.911926 | 0.913581 |
| 6hz | 3.887 | 0.942176 | 0.942268 | 2.3 | 0.910690 | 0.135378 | 0.897905 | 0.904252 |
| 7hz | 4.534 | 0.940643 | 0.940747 | 2.7 | 0.895203 | 0.118098 | 0.894705 | 0.894954 |
| 8hz | 5.178 | 0.939436 | 0.939534 | 2.8 | 0.894398 | 0.122407 | 0.877630 | 0.885935 |
| 9hz | 5.832 | 0.938252 | 0.938358 | 3.2 | 0.878133 | 0.106354 | 0.877969 | 0.878051 |

初步结论：
1. 在这批 ED24 外部 Driving 加噪数据上，EBF AUC 随实际噪声率增加单调下降，趋势正常。
2. `1hz/3hz/5hz` 三档的 AUC 接近或高于论文表格中的 `1/3/5 Hz` 数值，但该数据实际噪声率低于目录名标称值，因此不能直接说明已经复现论文结果。
3. 该结果说明：之前 JAER 数据与论文差距不太可能来自 EBF 公式或 AUC 积分本身，更可能来自数据生成/标签口径/事件区间差异。

ED24 外部 Driving 数据上的 N149 结果：

运行设置：
- 数据：同上 `driving_noise_{1hz...9hz}_ed24_withlabel`。
- N149 网格：`radius={2,3,4,5}`，`\(\tau={16,32,64,128,256,512}ms\)`。
- ROC：使用 N149 连续 score 构造 ROC，并统计每个 `(radius,\tau)` 的 AUC；下表的 `N149 AUC` 是每档噪声下所有网格中的最佳 AUC。

结果文件：
- `data/DND21/mydriving_ED24/N149/roc_n149_{1hz...9hz}_ed24.csv`
- `data/DND21/mydriving_ED24/N149/n149_ed24_driving_1to9hz_summary.csv`
- `data/DND21/mydriving_ED24/N149/n149_ed24_driving_allhz_summary.csv`
- `data/DND21/mydriving_ED24/n149_vs_ebf_ed24_driving_1to9hz_summary.csv`

| Level | Estimated Hz/pixel | EBF AUC | N149 AUC | N149-EBF AUC | EBF Best F1 | N149 Best F1 | N149 best-AUC `(r,tau)` |
|---|---:|---:|---:|---:|---:|---:|---|
| 1hz | 0.650 | 0.950013 | 0.940363 | -0.009650 | 0.968278 | 0.966613 | `(2,32ms)` |
| 2hz | 1.298 | 0.948619 | 0.940156 | -0.008463 | 0.951964 | 0.946244 | `(2,32ms)` |
| 3hz | 1.943 | 0.946503 | 0.940568 | -0.005935 | 0.937189 | 0.931406 | `(2,32ms)` |
| 4hz | 2.593 | 0.944914 | 0.941424 | -0.003490 | 0.925579 | 0.919301 | `(2,32ms)` |
| 5hz | 3.247 | 0.943417 | 0.942468 | -0.000950 | 0.913581 | 0.909426 | `(2,32ms)` |
| 6hz | 3.887 | 0.942176 | 0.940074 | -0.002103 | 0.904252 | 0.896588 | `(2,32ms)` |
| 7hz | 4.534 | 0.940643 | 0.940966 | +0.000323 | 0.894954 | 0.889008 | `(3,32ms)` |
| 8hz | 5.178 | 0.939436 | 0.943954 | +0.004518 | 0.885935 | 0.884242 | `(5,16ms)` |
| 9hz | 5.832 | 0.938252 | 0.939164 | +0.000911 | 0.878051 | 0.871728 | `(3,32ms)` |

N149 对比结论：
1. 低噪声到中等噪声（实际约 `0.65~3.89 Hz/pixel`）下，EBF 的 AUC 和 Best-F1 均优于 N149。
2. 高噪声端（实际约 `4.53~5.83 Hz/pixel`）下，N149 的最佳 AUC 开始接近或略高于 EBF，尤其 `8hz` 档高出约 `0.0045`。
3. 但 N149 的 Best-F1 仍低于 EBF，说明 N149 在高噪声端可能改善整体排序能力（AUC），但最佳二分类工作点的 precision/recall 折中仍不如 EBF 稳定。
4. N149 最佳 AUC 多数落在 `tau=32ms`，说明在这批数据上时间尺度仍与 EBF 论文参数接近；高噪声 `8hz` 档偏向更大空间邻域 `r=5` 和更短时间窗 `16ms`。

#### 4.2.2 EBF + N149（按 EBF 论文最优参数重跑，paper-driving）

论文参数依据（Guo 2025）：
1. 在 Driving 上，作者给出的最优组合为 `s=5`、`\(\tau=32ms\)`（其中 `s=5` 对应 `radius=2`）。
2. 论文 Table II 报告 EBF AUC（Driving）：`1Hz=0.948`, `3Hz=0.942`, `5Hz=0.936`。

本工程重跑设置：
1. 数据：`mydriving_paper` 三档（`light/light_mid/mid`）；
2. EBF：固定 `radius=2`, `tau=32000us`，扫阈值（`min-neighbors`）得到 ROC 与 AUC；
3. N149：固定 `radius=2`, `tau=32000us`, `sigma=1.5`，阈值直接使用“同档 EBF 的 best-F1 阈值”。

执行命令（EBF）：
```powershell
D:/software/Anaconda_envs/envs/myEVS/python.exe -m myevs.cli roc --clean "D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_paper/driving_noise_light_paper_withlabel/driving_noise_light_signal_only.npy" --noisy "D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_paper/driving_noise_light_paper_withlabel/driving_noise_light_labeled.npy" --assume npy --width 346 --height 260 --tick-ns 1000 --engine python --method ebf --radius-px 2 --time-us 32000 --param min-neighbors --values "0,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.5,3,3.5,4,5,6,8" --match-us 0 --match-bin-radius 0 --tag "ebf_paper_r2_tau32000_light" --out-csv data/DND21/mydriving_paper_eval/EBF/roc_ebf_light_paperfix.csv
```

结果文件：
1. `data/DND21/mydriving_paper_eval/EBF/roc_ebf_{light,light_mid,mid}_paperfix.csv`
2. `data/DND21/mydriving_paper_eval/ebf_n149_fixed_by_ebf_threshold.csv`

EBF AUC 与论文对比：

| Level | Noise (Hz/pixel) | Paper EBF AUC | Python-noise EBF AUC | JAER-noise EBF AUC |
|---|---:|---:|---:|---:|
| light | 1 | 0.948 | 0.926983 | 0.928231 |
| light_mid | 3 | 0.942 | 0.921316 | 0.923244 |
| mid | 5 | 0.936 | 0.916600 | 0.919252 |

JAER-noise EBF 运行设置：
```powershell
# values 为 0.0 到 15.0，步长 0.1；固定 s=5(radius=2), tau=32ms
D:/software/Anaconda_envs/envs/myEVS/python.exe -m myevs.cli roc `
  --clean "D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_jaer/driving_noise_light_jaer_withlabel/driving_noise_light_signal_only.npy" `
  --noisy "D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_jaer/driving_noise_light_jaer_withlabel/driving_noise_light_labeled.npy" `
  --assume npy --width 346 --height 260 --tick-ns 12.5 `
  --method ebf --radius-px 2 --time-us 32000 `
  --param min-neighbors --values "0.0,0.1,...,15.0" `
  --tag ebf_r2_tau32000_jaer `
  --roc-convention paper --match-us 0 --match-bin-radius 0 `
  --out-csv data/DND21/mydriving_jaer/EBF/roc_ebf_light_jaer_paperfix.csv --progress
```

JAER-noise best-F1 点：

| Level | AUC | Best-F1 threshold | TPR | FPR | Precision | Best F1 |
|---|---:|---:|---:|---:|---:|---:|
| light | 0.928231 | 0.6 | 0.971682 | 0.373343 | 0.947881 | 0.959634 |
| light_mid | 0.923244 | 1.7 | 0.929725 | 0.235392 | 0.902001 | 0.915653 |
| mid | 0.919252 | 2.4 | 0.902238 | 0.199822 | 0.863174 | 0.882274 |

EBF ROC/AUC 统计口径排查（2026-04-29）：

1. 官方 EBF 代码最终也是把每个事件的 `sumfeature` 当作连续 score，再用 `sklearn.metrics.roc_auc_score(y_true, y_score)` 计算 AUC；因此理论上不应强依赖手工阈值步长。
2. 官方 EBF score 构成是：同极性邻域事件的线性时间核累加，中心像素置零，不包含额外空间高斯核。

$$
s_i=\sum_{j\in\mathcal{N}_r(i),\,j\ne i}
\mathbf{1}(p_j=p_i)\cdot
\max\left(0,1-\frac{|t_i-t_j|}{\tau}\right)
$$

3. 当前 myEVS 的 `method=ebf` 与上述公式一致：按事件流顺序维护每个像素最近时间戳和极性；对同极性邻居累加线性时间核；自身事件不参与当前 score；score 计算后再更新自身状态。
4. DND21/EBF 辅助脚本中也存在从 CSV 的 `(FPR,TPR)` 点排序后用梯形积分 `auc(fpr,tpr)` 的路径。因此本次同时比较了三种口径：
   - `step=0.1`：原始阈值网格。
   - `step=0.01`：更细阈值网格，模拟“连续扫频”。
   - `exact score AUC`：不手动指定阈值，按所有不同 score 的排序直接构造 ROC，等价于 `roc_auc_score` 口径。

| Level | Paper AUC | step=0.1 AUC | step=0.01 AUC | Exact score AUC | step 0.01 gain | Paper - Exact |
|---|---:|---:|---:|---:|---:|---:|
| light | 0.948000 | 0.928231 | 0.928250 | 0.928296 | +0.000020 | 0.019704 |
| light_mid | 0.942000 | 0.923244 | 0.923270 | 0.923320 | +0.000028 | 0.018680 |
| mid | 0.936000 | 0.919252 | 0.919277 | 0.919328 | +0.000028 | 0.016672 |

排查结论：

1. 把阈值从 `0.1` 加密到 `0.01` 只提升约 `2e-5` 到 `3e-5` AUC。
2. 完全连续的 exact score AUC 也只比 `0.01` 阈值网格高约 `5e-5`。
3. 因此当前 `0.0167~0.0197` 的论文差距不是由阈值扫频太粗或梯形积分精度导致。
4. 后续应优先继续排查：clean 数据是否与论文完全同版、JAER 输出事件的起止边界/重复事件处理、DND21 官方 label 生成口径、坐标/极性编码方向，以及官方实验是否在某些预处理步骤中裁剪或过滤了事件区间。

固定阈值对比（N149 使用 EBF 同档 best-F1 阈值）：

| Level | EBF Thr (best-F1) | Method | TPR | FPR | Precision | F1 | Accuracy |
|---|---:|---|---:|---:|---:|---:|---:|
| light | 0.50 | EBF | 0.975002 | 0.396886 | 0.944884 | 0.959707 | 0.928391 |
| light | 0.50 | N149 | 0.940655 | 0.318305 | 0.953752 | 0.947158 | 0.908197 |
| light_mid | 1.75 | EBF | 0.927291 | 0.228449 | 0.904169 | 0.915584 | 0.880444 |
| light_mid | 1.75 | N149 | 0.765692 | 0.096433 | 0.948603 | 0.847390 | 0.807166 |
| mid | 2.50 | EBF | 0.897832 | 0.193437 | 0.866288 | 0.881778 | 0.859737 |
| mid | 2.50 | N149 | 0.644994 | 0.048619 | 0.948764 | 0.767930 | 0.772877 |

结论：
1. 按 `s=5, tau=32ms` 重跑后，AUC 走势仍是 `1Hz > 3Hz > 5Hz`，趋势与论文一致。
2. 换成 JAER 原链路导出的噪声后，AUC 只比 Python-noise 版本提高约 `0.001~0.003`，仍低于论文约 `0.017~0.020`。
3. 因此“不直接调用 JAER 加噪”不是主要差异来源；下一步应优先检查 DND21/EBF 原始 ROC 统计协议、clean 数据版本、坐标/极性编码口径，以及 EBF 官方代码的阈值扫频/积分方式。
4. 在“同阈值”约束下，N149 的 FPR 更低但 TPR 下降更明显，导致 F1 普遍低于 EBF；这符合“阈值跨算法迁移不一定公平”的预期。

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
- `run_driving_alg_paper.ps1`：Driving 论文口径数据统一入口（`mydriving_paper`）
- `run_driving_{knoise|evflow|ynoise|ts|mlpf|pfd}.ps1`（新增）：单算法入口
- `run_driving_n149.ps1`：Driving 上 N149 独立扫频入口（当前不并入 `run_driving_alg.ps1`）
- `generate_driving_paper_noise.py`：从 paper clean txt 生成 1/3/5Hz 的标注噪声 npy（FPN: sigma-decades=0.5）

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
3. 每个算法先看三档噪声（light/light_mid/mid）的 AUC 稳定性
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

### 10.6 Driving 横向结果汇总（200k, light/light_mid/mid）

数据来源：
- `data/DND21/mydriving/horizontal_summary_all.csv`
- `data/DND21/mydriving/bestpoint_mesr_aocc_summary.csv`
- `data/DND21/mydriving/{ALG}/runtime_*.csv` 与 `data/DND21/mydriving/N149/runtime_n149.csv`

说明：
1. 该汇总已覆盖 `BAF/STCF/EBF/N149/KNOISE/EVFLOW/YNOISE/TS/MLPF/PFD` 全部算法。
2. 按你的要求，`EVFLOW` 的 `MESR/AOCC` 暂停计算，先留空。
3. `Runtime(s)` 为当前口径下各算法各档最新一次记录。
4. Driving 结果目录已统一为“算法优先”布局：`data/DND21/mydriving/{ALG}/roc_{alg}_{level}.csv`，与 ED24 的组织方式一致。

#### 10.6.1 light

| Algorithm | Best AUC | Best AUC Tag | F1@Best-AUC | Thr@Best-AUC | MESR@Best-AUC | AOCC@Best-AUC | Best F1 | Best F1 Tag | Thr@Best-F1 | AUC@Best-F1 | MESR@Best-F1 | AOCC@Best-F1 | Runtime(s) |
|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| BAF | 0.799350 | baf_r1_tau4000 | 0.905169 | 1.000000 | 0.788445 | 2.379989 | 0.977395 | baf_r2_tau32000 | 1.000000 | 0.593953 | 0.717649 | 2.321786 | 22.235 |
| STCF | 0.935075 | stcf_r1 | 0.977808 | 128000.000000 | 0.733109 | 2.429605 | 0.987283 | stcf_r4 | 16000.000000 | 0.902610 | 0.726550 | 2.392844 | 61.093 |
| EBF | 0.937124 | ebf_r2_tau16000 | 0.987250 | 0.000000 | 0.730291 | 2.430151 | 0.987540 | ebf_r3_tau16000 | 0.500000 | 0.916800 | 0.727634 | 2.407034 | 14.627 |
| N149 | 0.932978 | n149_r2_tau32000_light | 0.981619 | 0.259854 | 0.725054 | 2.402036 | 0.982559 | n149_r3_tau64000_light | 0.476634 | 0.926168 | 0.718161 | 2.335562 | 26.308 |
| KNOISE | 0.597576 | knoise_tau4000 | 0.979814 | 0.000000 | 0.711667 | 2.234047 | 0.979814 | knoise_tau1000 | 0.000000 | 0.596042 | 0.711667 | 2.234047 | 85.306 |
| EVFLOW | 0.822085 | evflow_r2_tau8000 | 0.920775 | 64.000000 |  |  | 0.970408 | evflow_r3_tau32000 | 64.000000 | 0.752090 |  |  | 684.152 |
| YNOISE | 0.936610 | ynoise_r2_tau8000 | 0.981020 | 1.000000 | 0.740235 | 2.461512 | 0.983444 | ynoise_r2_tau16000 | 1.000000 | 0.927710 | 0.730039 | 2.428665 | 300.664 |
| TS | 0.840665 | ts_r2_decay32000 | 0.981704 | 0.200000 | 0.761003 | 2.597591 | 0.981704 | ts_r2_decay32000 | 0.200000 | 0.840665 | 0.761003 | 2.597591 | 8.854 |
| MLPF | 0.472022 | mlpf_tau32000 | 0.972319 | 0.800000 | 0.711261 | 2.244642 | 0.972323 | mlpf_tau64000 | 0.800000 | 0.430472 | 0.705434 | 2.236060 | 446.578 |
| PFD | 0.901287 | pfd_r3_tau8000_m2 | 0.975540 | 1.000000 | 0.742263 | 2.483932 | 0.978808 | pfd_r3_tau8000_m1 | 1.000000 | 0.894581 | 0.737233 | 2.457937 | 31.352 |

#### 10.6.2 light_mid

| Algorithm | Best AUC | Best AUC Tag | F1@Best-AUC | Thr@Best-AUC | MESR@Best-AUC | AOCC@Best-AUC | Best F1 | Best F1 Tag | Thr@Best-F1 | AUC@Best-F1 | MESR@Best-F1 | AOCC@Best-F1 | Runtime(s) |
|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| BAF | 0.881811 | baf_r1 | 0.948786 | 16000.000000 | 0.723033 | 2.306037 | 0.948786 | baf_r1 | 16000.000000 | 0.881811 | 0.723033 | 2.306037 | 22.991 |
| STCF | 0.936965 | stcf_r1 | 0.961349 | 64000.000000 | 0.730580 | 2.376526 | 0.970314 | stcf_r2 | 16000.000000 | 0.933485 | 0.732522 | 2.415924 | 63.392 |
| EBF | 0.937867 | ebf_r2_tau16000 | 0.970612 | 1.000000 | 0.742812 | 2.461226 | 0.971160 | ebf_r2_tau32000 | 1.500000 | 0.922190 | 0.731829 | 2.436520 | 14.687 |
| N149 | 0.936847 | n149_r2_tau32000_light_mid | 0.958057 | 0.747070 | 0.720106 | 2.356677 | 0.958849 | n149_r3_tau32000_light_mid | 0.870674 | 0.936600 | 0.710368 | 2.272940 | 25.970 |
| KNOISE | 0.591022 | knoise_tau2000 | 0.939629 | 0.000000 | 0.681278 | 1.981033 | 0.939629 | knoise_tau1000 | 0.000000 | 0.590283 | 0.681278 | 1.981033 | 86.387 |
| EVFLOW | 0.828349 | evflow_r2_tau8000 | 0.911603 | 64.000000 |  |  | 0.941888 | evflow_r2_tau32000 | 64.000000 | 0.763634 |  |  | 677.403 |
| YNOISE | 0.935314 | ynoise_r2_tau8000 | 0.957535 | 1.000000 | 0.725979 | 2.306716 | 0.962252 | ynoise_r2_tau16000 | 2.000000 | 0.927591 | 0.732927 | 2.413554 | 296.672 |
| TS | 0.802150 | ts_r2_decay32000 | 0.943836 | 0.300000 | 0.784064 | 2.581381 | 0.943836 | ts_r2_decay32000 | 0.300000 | 0.802150 | 0.784064 | 2.581381 | 8.741 |
| MLPF | 0.439639 | mlpf_tau32000 | 0.916399 | 0.100000 | 0.681240 | 1.981077 | 0.916399 | mlpf_tau32000 | 0.100000 | 0.439639 | 0.681240 | 1.981077 | 448.517 |
| PFD | 0.900137 | pfd_r3_tau8000_m2 | 0.958203 | 1.000000 | 0.731202 | 2.368465 | 0.958203 | pfd_r3_tau8000_m2 | 1.000000 | 0.900137 | 0.731202 | 2.368465 | 31.951 |

#### 10.6.3 mid

| Algorithm | Best AUC | Best AUC Tag | F1@Best-AUC | Thr@Best-AUC | MESR@Best-AUC | AOCC@Best-AUC | Best F1 | Best F1 Tag | Thr@Best-F1 | AUC@Best-F1 | MESR@Best-F1 | AOCC@Best-F1 | Runtime(s) |
|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| BAF | 0.868373 | baf_r1 | 0.916932 | 8000.000000 | 0.735027 | 2.221840 | 0.916932 | baf_r1 | 8000.000000 | 0.868373 | 0.735027 | 2.221840 | 23.325 |
| STCF | 0.935924 | stcf_r1 | 0.946620 | 32000.000000 | 0.742287 | 2.392254 | 0.954822 | stcf_r2 | 8000.000000 | 0.930659 | 0.751749 | 2.389156 | 65.307 |
| EBF | 0.937166 | ebf_r2_tau16000 | 0.956357 | 1.500000 | 0.749396 | 2.433484 | 0.956357 | ebf_r2_tau16000 | 1.500000 | 0.937166 | 0.749396 | 2.433484 | 14.778 |
| N149 | 0.938152 | n149_r2_tau32000_mid | 0.937040 | 1.071131 | 0.718285 | 2.311529 | 0.937725 | n149_r3_tau32000_mid | 1.153651 | 0.938016 | 0.700190 | 2.158709 | 25.899 |
| KNOISE | 0.584814 | knoise_tau2000 | 0.899398 | 0.000000 | 0.657992 | 1.796615 | 0.899398 | knoise_tau1000 | 0.000000 | 0.584477 | 0.657992 | 1.796615 | 87.263 |
| EVFLOW | 0.826195 | evflow_r2_tau8000 | 0.898333 | 64.000000 |  |  | 0.915064 | evflow_r2_tau16000 | 64.000000 | 0.793410 |  |  | 682.711 |
| YNOISE | 0.933513 | ynoise_r2_tau8000 | 0.940422 | 2.000000 | 0.751525 | 2.384611 | 0.942359 | ynoise_r2_tau16000 | 3.000000 | 0.926730 | 0.743120 | 2.426991 | 293.449 |
| TS | 0.777210 | ts_r2_decay32000 | 0.904080 | 0.500000 | 0.851542 | 2.388471 | 0.904080 | ts_r2_decay32000 | 0.500000 | 0.777210 | 0.851542 | 2.388471 | 8.731 |
| MLPF | 0.498857 | mlpf_tau256000 | 0.853742 | 0.100000 | 0.666617 | 1.777021 | 0.861026 | mlpf_tau512000 | 0.200000 | 0.415801 | 0.673816 | 1.883022 | 450.010 |
| PFD | 0.897118 | pfd_r3_tau8000_m3 | 0.936814 | 1.000000 | 0.741823 | 2.358040 | 0.936814 | pfd_r3_tau8000_m3 | 1.000000 | 0.897118 | 0.741823 | 2.358040 | 33.339 |

结论更新（N149 vs EBF，Driving 200k）：
1. 旧版“N149 低于 EBF”的主要原因是 N149 当时只扫了 `r=3/4/5`，结构参数覆盖不完整。
2. 本轮加入 `r=2` 并做 `sigma` 扫描后，N149 的 `best-AUC` 变为 `0.932978/0.936847/0.938152`（light/light_mid/mid），已与 EBF 接近，且在 `mid` 上略高于当前 EBF 汇总值。
3. 对 driving 这类数据，优先补全结构参数覆盖（`r`、`sigma`），再细化阈值，收益更稳定。

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

# 真实 MLPF（按 level 自动套模型）
powershell -ExecutionPolicy Bypass -File ./scripts/driving_alg_evalu/run_driving_alg.ps1 -Algorithm mlpf -SweepProfile coarse -MlpfModelPattern "data/DND21/mydriving/MLPF/mlpf_torch_{level}.pt"

# N149（独立脚本，200k）
powershell -ExecutionPolicy Bypass -File ./scripts/driving_alg_evalu/run_driving_n149.ps1
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

# Driving：只算 light/light_mid/mid，且只算 best-F1
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/eval_bestpoint_mesr_aocc.py --dataset driving --levels light,light_mid,mid --points best-f1

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
# 1) 训练 ED24（按 light / mid / heavy）
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/train_mlpf_torch.py --clean "D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_1.8_signal_only.npy" --noisy "D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_1.8.npy" --width 346 --height 260 --tick-ns 1000 --duration-us 128000 --patch 7 --epochs 8 --batch-size 512 --max-events 200000 --out-ts "data/ED24/myPedestrain_06/MLPF/mlpf_torch_light.pt" --out-meta "data/ED24/myPedestrain_06/MLPF/mlpf_torch_light.json"
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/train_mlpf_torch.py --clean "D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_2.5_signal_only.npy" --noisy "D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_2.5.npy" --width 346 --height 260 --tick-ns 1000 --duration-us 128000 --patch 7 --epochs 8 --batch-size 512 --max-events 200000 --out-ts "data/ED24/myPedestrain_06/MLPF/mlpf_torch_mid.pt" --out-meta "data/ED24/myPedestrain_06/MLPF/mlpf_torch_mid.json"
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/train_mlpf_torch.py --clean "D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3_signal_only.npy" --noisy "D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3.npy" --width 346 --height 260 --tick-ns 1000 --duration-us 128000 --patch 7 --epochs 8 --batch-size 512 --max-events 200000 --out-ts "data/ED24/myPedestrain_06/MLPF/mlpf_torch_heavy.pt" --out-meta "data/ED24/myPedestrain_06/MLPF/mlpf_torch_heavy.json"

# 2) 用真实模型跑 ED24 MLPF ROC
powershell -ExecutionPolicy Bypass -File ./scripts/ED24_alg_evalu/run_slomo_alg.ps1 -Algorithm mlpf -SweepProfile coarse -MlpfModelPattern "data/ED24/myPedestrain_06/MLPF/mlpf_torch_{level}.pt"

# 3) 训练 Driving（按 light / light_mid / mid）
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/train_mlpf_torch.py --clean "D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving/driving_noise_light_slomo_shot_withlabel/driving_noise_light_signal_only.npy" --noisy "D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving/driving_noise_light_slomo_shot_withlabel/driving_noise_light_labeled.npy" --width 346 --height 260 --tick-ns 1000 --duration-us 128000 --patch 7 --epochs 8 --batch-size 512 --max-events 200000 --out-ts "data/DND21/mydriving/MLPF/mlpf_torch_light.pt" --out-meta "data/DND21/mydriving/MLPF/mlpf_torch_light.json"
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/train_mlpf_torch.py --clean "D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving/driving_noise_light_mid_slomo_shot_withlabel/driving_noise_light_mid_signal_only.npy" --noisy "D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving/driving_noise_light_mid_slomo_shot_withlabel/driving_noise_light_mid_labeled.npy" --width 346 --height 260 --tick-ns 1000 --duration-us 128000 --patch 7 --epochs 8 --batch-size 512 --max-events 200000 --out-ts "data/DND21/mydriving/MLPF/mlpf_torch_light_mid.pt" --out-meta "data/DND21/mydriving/MLPF/mlpf_torch_light_mid.json"
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/train_mlpf_torch.py --clean "D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving/driving_noise_mid_slomo_shot_withlabel/driving_noise_mid_signal_only.npy" --noisy "D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving/driving_noise_mid_slomo_shot_withlabel/driving_noise_mid_labeled.npy" --width 346 --height 260 --tick-ns 1000 --duration-us 128000 --patch 7 --epochs 8 --batch-size 512 --max-events 200000 --out-ts "data/DND21/mydriving/MLPF/mlpf_torch_mid.pt" --out-meta "data/DND21/mydriving/MLPF/mlpf_torch_mid.json"

# 4) 用真实模型跑 Driving MLPF ROC
powershell -ExecutionPolicy Bypass -File ./scripts/driving_alg_evalu/run_driving_alg.ps1 -Algorithm mlpf -SweepProfile coarse -MlpfModelPattern "data/DND21/mydriving/MLPF/mlpf_torch_{level}.pt"
```

### 15.5.1 Driving MLPF（真实模型）重跑结果（2026-04-28）

数据来源：
- `data/DND21/mydriving/MLPF/roc_mlpf_{level}.csv`
- `data/DND21/mydriving/MLPF/runtime_mlpf.csv`（按 level 取最新一条）
- `data/DND21/mydriving/bestpoint_mesr_aocc_summary.csv`（best-AUC / best-F1）

| Level | Best AUC | Best AUC Tag | Best AUC Threshold | MESR@Best-AUC | AOCC@Best-AUC | Best F1 | Best F1 Tag | Best F1 Threshold | MESR@Best-F1 | AOCC@Best-F1 | Runtime(s) |
|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| light | 0.472022 | mlpf_tau32000 | 0.8 | 0.711261 | 2.244642 | 0.972323 | mlpf_tau64000 | 0.8 | 0.705434 | 2.236060 | 446.578 |
| light_mid | 0.439639 | mlpf_tau32000 | 0.1 | 0.681240 | 1.981077 | 0.916399 | mlpf_tau32000 | 0.1 | 0.681240 | 1.981077 | 448.517 |
| mid | 0.498857 | mlpf_tau256000 | 0.1 | 0.666617 | 1.777021 | 0.861026 | mlpf_tau512000 | 0.2 | 0.673816 | 1.883022 | 450.010 |

说明：
1. 该表是“真实模型推理模式”结果（已传 `--mlpf-model`），不是 proxy 回退模式。
2. 当前 AUC 偏低的直接现象是：大量阈值点靠近“全保留”区（FPR/TPR 同时较高），说明该训练流程在 driving 上判别力不足，后续需继续优化训练策略（样本均衡/损失重加权/更强特征）。

### 15.5.2 Driving N149（200k）补跑结果（2026-04-28）

数据来源：
- `data/DND21/mydriving/N149_sigma_scan/summary_sigma_scan.csv`
- `data/DND21/mydriving/N149_sigma_scan/selected_config.csv`
- `data/DND21/mydriving/N149_selected/roc_n149_{light,light_mid,mid}.csv`
- `data/DND21/mydriving/N149/runtime_n149.csv`（由 `N149_selected` 同步）
- `data/DND21/mydriving/bestpoint_mesr_aocc_summary.csv`（已补 n149 的 best-AUC / best-F1）

| Level | Best AUC | Best AUC Tag | Threshold@Best-AUC | F1@Best-AUC | MESR@Best-AUC | AOCC@Best-AUC | Best F1 | Best F1 Tag | Threshold@Best-F1 | AUC@Best-F1 | MESR@Best-F1 | AOCC@Best-F1 | Runtime(s) |
|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| light | 0.932978 | n149_r2_tau32000_light | 0.259854 | 0.981619 | 0.725054 | 2.402036 | 0.982559 | n149_r3_tau64000_light | 0.476634 | 0.926168 | 0.718161 | 2.335562 | 26.308 |
| light_mid | 0.936847 | n149_r2_tau32000_light_mid | 0.747070 | 0.958057 | 0.720106 | 2.356677 | 0.958849 | n149_r3_tau32000_light_mid | 0.870674 | 0.936600 | 0.710368 | 2.272940 | 25.970 |
| mid | 0.938152 | n149_r2_tau32000_mid | 1.071131 | 0.937040 | 0.718285 | 2.311529 | 0.937725 | n149_r3_tau32000_mid | 1.153651 | 0.938016 | 0.700190 | 2.158709 | 25.899 |

说明：
1. N149 在 driving 三档下 AUC 更新为 `0.933~0.938`（本轮 sigma 扫描后选优，`sigma=1.5`）。
2. 与旧版仅 `r=3/4/5` 的结论相比，本轮加入 `r=2` 后，三档 AUC 均提升，说明之前主要不是阈值没扫够，而是半径覆盖不足。
3. `MESR/AOCC` 已按统一规则补齐：`best-AUC` 与 `best-F1` 两个工作点都已记录。

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
