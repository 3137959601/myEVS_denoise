# EBF_optimized 变体（EBFV1/EBFV2/EBFV3/EBFV4/EBFV41/EBFV5/EBFV6/EBFV7/EBFV8/EBFV9）

本目录用于**离线实验 / sweep 对比**的 EBF_optimized 改进算法集合。

- 统一入口：`create_ebfopt_variant(variant_id, dims, cfg, tb)`
- sweep 脚本： `scripts/ED24_alg_evalu/sweep_ebf_optimized_labelscore_grid.py --variant ...`

> 说明：如果你仍在使用 denoise pipeline/CLI，那么工程里还保留了一个兼容入口 `myevs.denoise.ops.ebf_optimized.EbfOptimizedOp`。
> 但做离线对比时，推荐直接用本目录的变体（更清晰）。

---

## 共同点（所有版本都一样）

这些版本都遵循同一个框架：

1. **raw score**：局部邻域的 EBF 打分（与原始 EBF 相同）
2. **global noise proxy**：用全局 EMA 估计事件到达率（以 `1/Δt` 为基本量）
3. **scale**：由全局到达率、`tau` 窗口、邻域大小推导的“期望噪声 raw-score 尺度”
4. **score_norm**：`score_norm = raw_score / (scale^α)`（warm-up 时会退化为 raw_score）

其中 `α`（scale exponent）默认 `α=1.0`，与历史实现完全一致；可用环境变量控制：

- `MYEVS_EBFOPT_SCALE_ALPHA=1.1`（范围会 clamp 到 `[0,4]`）

此外，全局噪声率的 EMA 平滑系数也可调（注意：这是**噪声率 EMA 的 α**，不是上面的 scale exponent）：

- `MYEVS_EBFOPT_RATE_EMA_ALPHA=0.002`（默认 `0.01`；范围会 clamp 到 `[1e-6, 1.0]`）

如果你希望把 EMA 的“响应速度”从“按更新次数”改成“按真实时间常数”，则可用 V9：

- `MYEVS_EBFOPT_RATE_EMA_TAU_US=100000`（仅 V9 使用；默认 `100000`，即 100ms）

直觉：该值越小，rate proxy 的方差越低但响应越慢（更“平滑/滞后”）；越大则更新更快但波动更大。

你要对比的关键差异就在 2) 噪声率 EMA 的更新权重，以及 3) scale 的公式。

---

## 版本对比

### EBFV1（`equalw_linear`）
- **噪声率更新**：所有事件等权更新（`w=1`）
- **scale**：线性近似
  - `scale ≈ 0.25 * (λ_pix * τ) * N_neigh`
- **用途**：baseline 对照。展示“只做全局归一化但不抑制 signal 污染”的效果/问题。

对应实现：`v1_equalw_linear.py`

### EBFV2（`softw_linear`）
- **噪声率更新**：按 raw-score 软降权（raw-score 越大越像信号，对噪声率贡献越小）
  - `w = 1 / (1 + s/k)`，并截断到 `[w_min, 1]`
- **scale**：仍是 V1 的线性近似
- **用途**：解决 V1 在 light 场景容易把信号当噪声、导致 scale 偏大/阈值漂移的问题。

对应实现：`v2_softw_linear.py`

### EBFV3（`softw_recent`）
- **噪声率更新**：同 V2（soft-weight）
- **scale**：改为“最近事件饱和模型”（更贴合实现：每像素只保留最近一次事件）
  - 设 `m = λ_pix * τ`
  - 单邻居期望贡献：`per_neigh = 0.5 * (1 - (1-exp(-m))/m)`
  - `scale = N_neigh * per_neigh`
- **用途**：在 heavy/高到达率场景下，避免线性 scale 过粗糙；并保持与 V2 一致的噪声率估计策略。

对应实现：`v3_softw_recent.py`

### EBFV4（`softw_linear_block`）

V4 以“硬件部署友好”为主要目标：在不引入复杂统计/模型的前提下，进一步增强阈值稳定性。

- **保持**：V2 的 soft-weight（噪声率估计的信号降权）
- **保持**：V2 的线性 scale（公式简单，便于硬件实现）
- **改动**：噪声率 proxy 从“全局 1 条 EMA”换成“block/tile 级 EMA”
  - 每个 block 只维护 `last_t_block` 与 `ema_inv_dt_block`
  - scale 使用 block 的 rate + block 的 area
- **默认 block**：32×32（2 的幂，硬件友好；边缘块按实际面积计算）

对应实现：`v4_softw_linear_block.py`

### EBFV4.1（`softw_linear_blockmix`）

V4.1 是对 V4 的“纠偏版”，目标仍然是硬件友好，但更强调“不掉精度”。

- **保持**：V2 的 soft-weight（噪声率估计的信号降权）
- **保持**：线性 scale（公式简单，便于硬件实现）
- **改动**：噪声率 proxy 采用 block/global 混合（hybrid）而不是 pure block：
  - 同时维护 `global_ema_inv_dt` 与 `block_ema_inv_dt`
  - 用 per-pixel rate 线性混合：
    - `λ_pix_eff = (1-β)*λ_pix_global + β*λ_pix_block`
  - 默认 `β=0.1`（固定常数，不引入新配置项）

对应实现：`v41_softw_linear_blockmix.py`

### EBFV5（`softw_linear_same_minus_opp`）

V5 是在 V2 的框架下，只改 raw score（不再动 scale / rate proxy）的尝试。

- **保持**：V2 的 soft-weight（噪声率估计的信号降权）
- **保持**：V2 的线性 scale（公式简单，便于硬件实现）
- **改动**：raw score 从“只累计同极性邻居”改为：
  - `S_raw = sum_same(Aw) - gamma * sum_opp(Aw)`
  - `gamma>=0`，`gamma=0` 退化为 V2
- **实现注意**：raw score 允许为负；噪声率估计的 soft-weight 对 raw score 取 `max(0, S_raw)`（避免负值破坏降权逻辑）。

对应实现：`v5_softw_linear_same_minus_opp.py`

### EBFV6（`softw_linear_purity`）

V6 是路线 B（更温和的 score 归一化）的实现：不再改 rate proxy / scale，只用一个“纯度”系数对 raw score 做调制，直觉是“同极性显著占优时更可信”。

- **保持**：V2 的 soft-weight（噪声率估计的信号降权）
- **保持**：V2 的线性 scale（公式简单，便于硬件实现）
- **改动**：在 score_norm 里引入 purity
  - `same = sum_same(Aw)`、`opp = sum_opp(Aw)`
  - `purity = same / (same + opp + eps)`
  - `score_norm = (same / scale_global) * purity`

对应实现：`v6_softw_linear_purity.py`

### EBFV7（`softw_linear_polrate`）

V7 尝试解决一个潜在的系统偏差：raw-score 只累计“同极性邻居”，但 V2/V3 的全局噪声率 proxy 是把 ON/OFF 混在一起估计的。若不同噪声环境下 ON/OFF 比例变化明显，用“混合 rate”归一化“同极性 raw-score”可能造成尺度漂移。

- **保持**：raw-score 与 V2 完全一致（不改排序信息）
- **保持**：soft-weight 噪声率更新权重仍用 V2 的 w(score_raw)
- **保持**：线性 scale
- **改动**：把全局噪声率 EMA 拆成两条（ON/OFF），归一化时按当前事件极性选择对应 rate

对应实现：`v7_softw_linear_polrate.py`

### EBFV8（`softw_linear_binrate`）

V8 不改 raw-score 的排序逻辑，仍然保持 V2 的 soft-weight 与线性 scale，但把“全局噪声率 proxy”从逐事件的 `1/Δt` EMA，改成**定长时间 bin**上的计数率（`sum_w / Δt_bin`）再做 EMA。

- **保持**：V2 的 soft-weight（噪声率估计的信号降权）
- **保持**：V2 的线性 scale（公式简单）
- **改动**：全局噪声率 proxy 改为 bin-rate（降低 `Δt` 抖动对 scale 的方差影响）
  - 在 bin 内累计 `sum_w`（soft-weight 累加）
  - bin 结束时用 `sum_w / Δt_bin` 更新 EMA rate

bin 大小用环境变量控制（单位 us）：

- `MYEVS_EBFOPT_BINRATE_US=15000` 表示 15ms bin

对应实现：`v8_softw_linear_binrate.py`

### EBFV9（`softw_linear_timeconst_rateema`）

V9 在 V2 的框架下只改一件事：把全局噪声率 proxy 的 EMA 从“固定 alpha（按更新次数）”改成“固定时间常数 tau_rate（按真实时间）”。

- **保持**：V2 的 soft-weight（噪声率估计的信号降权）
- **保持**：V2 的线性 scale（公式简单）
- **改动**：全局噪声率 EMA 的更新系数随 `dt` 自适应
  - `alpha(dt) = 1 - exp(-dt / tau_rate)`
  - 用环境变量指定 `tau_rate`（单位 us）：`MYEVS_EBFOPT_RATE_EMA_TAU_US`

直觉：

- old（固定 alpha）：事件越密，单位真实时间内更新次数越多，EMA 越“快”，rate proxy 方差也更大
- new（固定 tau_rate）：不管输入采样/事件间隔如何抖动，滤波器的时间尺度更一致

对应实现：`v9_softw_linear_timeconst_rateema.py`

---

## 如何调用（推荐写法）

### sweep

- `--variant equalw_linear` 或 `--variant EBFV1`
- `--variant softw_linear` 或 `--variant EBFV2`
- `--variant softw_recent` 或 `--variant EBFV3`
- `--variant softw_linear_block` 或 `--variant EBFV4`
- `--variant softw_linear_blockmix` 或 `--variant EBFV41`
- `--variant softw_linear_same_minus_opp` 或 `--variant EBFV5`
- `--variant softw_linear_purity` 或 `--variant EBFV6`
- `--variant softw_linear_polrate` 或 `--variant EBFV7`（alias: `ebfv7`）
- `--variant softw_linear_binrate` 或 `--variant EBFV8`（alias: `ebfv8`）
- `--variant softw_linear_timeconst_rateema` 或 `--variant EBFV9`（alias: `ebfv9`）

例如：

```powershell
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/sweep_ebf_optimized_labelscore_grid.py --variant EBFV3 --out-dir data/ED24/myPedestrain_06/EBFOPT_cmp
```

V8（bin=15ms）示例：

```powershell
$env:MYEVS_EBFOPT_BINRATE_US='15000'
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/sweep_ebf_optimized_labelscore_grid.py --variant EBFV8 --out-dir data/ED24/myPedestrain_06/EBF_optimized_V8_binrate
```

V9（rate EMA 时间常数=100ms）示例：

```powershell
$env:MYEVS_EBFOPT_RATE_EMA_TAU_US='100000'
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/sweep_ebf_optimized_labelscore_grid.py --variant EBFV9 --out-dir data/ED24/myPedestrain_06/EBF_optimized_V9_rateema_timeconst
```

α（scale exponent）示例：

```powershell
$env:MYEVS_EBFOPT_SCALE_ALPHA='1.1'
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/sweep_ebf_optimized_labelscore_grid.py --variant EBFV2 --out-dir data/ED24/myPedestrain_06/EBF_optimized_V2scale_linear
```

### 直接在 Python 里创建

```python
from myevs.denoise.ops.ebfopt_variants import create_ebfopt_variant
op = create_ebfopt_variant("EBFV2", dims, cfg, tb)
```

---

## 已跑完结果（ED24 / myPedestrain_06）

本节用于**对比记录**：把已经跑完的结果（CSV 产物）整理成一张表，方便后续复盘。

### 评测设置（来自现有产物）

- 数据集：ED24 / `myPedestrain_06`
- 三个噪声环境：`light(1.8V) / mid(2.5V) / heavy(3.3V)`
- sweep 网格：`s ∈ {3,5,7,9}`，`tau ∈ {8,16,32,64,128,256,512,1024} ms`
- 选最优点策略：
  - 先按 **AUC** 选择最佳 `(s, tau)` tag
  - 再在该 tag 内按 **F1 最大**选择阈值（并记录对应的 TPR/FPR/Precision/Accuracy）

### MESR（Mean ESR；无参考指标）

本次在原有 `AUC/F1` 之后，额外加入一个无参考指标 **MESR**（Mean Event Structural Ratio）。

实现口径：

- 先用 best-F1 阈值得到 “kept events”（即 `score >= Thr_bestF1`）
- 将 kept events 按时间顺序切成固定大小的块（默认每块 30000 events；不足一块的尾巴忽略）
- 每块计算一次 ESR，再对所有块取均值，得到 MESR

ESR 的单块公式（与论文工程 `cuke-emlb` 的 `EventStructuralRatio` 一致；仅使用每像素计数图，不依赖 reference）：

设该块事件在每个像素上的计数为 $n_{i}$（把 $(x,y)$ 展平成 $i\in[1,K]$），其中：

- $K = W\times H$（分辨率像素数）
- $N = \sum_i n_i$（该块事件总数）
- $M = \lfloor \tfrac{2}{3}N \rfloor$

则：

$$
\mathrm{NTSS} = \frac{\sum_i n_i(n_i-1)}{N(N-1)}
$$

$$
\mathrm{LN} = K - \sum_i \left(1 - \frac{M}{N}\right)^{n_i}
$$

$$
\mathrm{ESR} = \sqrt{\mathrm{NTSS}\cdot\mathrm{LN}}
$$

最后：

$$
\mathrm{MESR} = \frac{1}{T}\sum_{t=1}^{T} \mathrm{ESR}_t
$$

其中 $T$ 是块数。

备注：

- ESR/MESR 默认忽略 polarity（只看事件计数表面）
- 在 ROC CSV 里新增的列名是 `esr_mean`，它对应的就是 MESR；为了控制计算量，该列**只在每个 tag 的 best-F1 那一行写入**（同一 tag 其余阈值行为空）

### 结果汇总（最优 tag + 最优阈值）

> 注：下表中的 “Thr” 是 `score_norm` 的阈值。

| 方法 | env | best (s,tau) | Thr | AUC | F1 | TPR | FPR |
|---|---|---:|---:|---:|---:|---:|---:|
| 原始 EBF（未归一化） | light | (9, 128ms) | 0.7491 | 0.9476 | 0.9497 | 0.9454 | 0.2352 |
| 原始 EBF（未归一化） | mid | (9, 128ms) | 4.8347 | 0.9232 | 0.8177 | 0.7981 | 0.0623 |
| 原始 EBF（未归一化） | heavy | (9, 128ms) | 7.2797 | 0.9136 | 0.7610 | 0.7320 | 0.0426 |
| EBFV1（equalw_linear；早期等权全事件率） | light | (9, 64ms) | 0.0017 | 0.9286 | 0.9460 | 0.9454 | 0.2760 |
| EBFV1（equalw_linear；早期等权全事件率） | mid | (9, 128ms) | 0.7342 | 0.9047 | 0.7870 | 0.7471 | 0.0612 |
| EBFV1（equalw_linear；早期等权全事件率） | heavy | (9, 128ms) | 0.7885 | 0.8974 | 0.7355 | 0.6958 | 0.0436 |
| EBFV2（softw_linear） | light | (9, 128ms) | 0.1833 | 0.9356 | 0.9459 | 0.9584 | 0.3519 |
| EBFV2（softw_linear） | mid | (9, 128ms) | 1.2280 | 0.9181 | 0.8088 | 0.7808 | 0.0606 |
| EBFV2（softw_linear） | heavy | (9, 128ms) | 1.4929 | 0.9102 | 0.7537 | 0.7242 | 0.0439 |
| EBFV3（softw_recent） | light | (9, 128ms) | 0.1861 | 0.9361 | 0.9460 | 0.9586 | 0.3526 |
| EBFV3（softw_recent） | mid | (9, 128ms) | 1.3142 | 0.9185 | 0.8096 | 0.7813 | 0.0602 |
| EBFV3（softw_recent） | heavy | (9, 128ms) | 1.6221 | 0.9106 | 0.7546 | 0.7239 | 0.0433 |
| EBFV4（softw_linear_block） | light | (9, 64ms) | 0.0008 | 0.9080 | 0.9460 | 0.9454 | 0.2761 |
| EBFV4（softw_linear_block） | mid | (9, 128ms) | 0.4128 | 0.8903 | 0.7509 | 0.7282 | 0.0854 |
| EBFV4（softw_linear_block） | heavy | (9, 64ms) | 0.5509 | 0.8812 | 0.6908 | 0.6386 | 0.0467 |
| EBFV4.1（softw_linear_blockmix） | light | (9, 128ms) | 0.0363 | 0.9340 | 0.9453 | 0.9646 | 0.3943 |
| EBFV4.1（softw_linear_blockmix） | mid | (9, 128ms) | 1.0159 | 0.9149 | 0.8031 | 0.7699 | 0.0595 |
| EBFV4.1（softw_linear_blockmix） | heavy | (9, 128ms) | 1.1962 | 0.9066 | 0.7472 | 0.7174 | 0.0450 |
| EBFV6（softw_linear_purity） | light | (9, 64ms) | 0.0000558 | 0.9288 | 0.9460 | 0.9454 | 0.2761 |
| EBFV6（softw_linear_purity） | mid | (7, 64ms) | 1.2634 | 0.9022 | 0.7834 | 0.7257 | 0.0513 |
| EBFV6（softw_linear_purity） | heavy | (7, 64ms) | 1.3410 | 0.8916 | 0.7316 | 0.6730 | 0.0370 |
| EBFV8（softw_linear_binrate；bin=15ms） | light | (9, 128ms) | 1.3069 | 0.9458 | 0.9484 | 0.9446 | 0.2459 |
| EBFV8（softw_linear_binrate；bin=15ms） | mid | (9, 128ms) | 2.7131 | 0.9227 | 0.8140 | 0.7900 | 0.0610 |
| EBFV8（softw_linear_binrate；bin=15ms） | heavy | (9, 128ms) | 2.7056 | 0.9125 | 0.7532 | 0.7269 | 0.0451 |

#### 2026-04-08 关键结果（新增 MESR；全量重跑）

下表汇总了本次新生成的关键结果（在 AUC/F1 之外加入 MESR），并包含你特别点名的 V10：

| 方法 | env | best (s,tau) | Thr(best-F1) | AUC | F1 | MESR |
|---|---|---:|---:|---:|---:|---:|
| 原始 EBF（未归一化） | light | (9, 128ms) | 0.7491 | 0.9476 | 0.9497 | 1.0305 |
| 原始 EBF（未归一化） | mid | (9, 128ms) | 4.8347 | 0.9232 | 0.8177 | 1.0155 |
| 原始 EBF（未归一化） | heavy | (9, 128ms) | 7.2797 | 0.9136 | 0.7610 | 1.0085 |
| 原始 EBF：V10（空间距离线性权重；不做权重和归一化） | light | (9, 128ms) | 0.004780 | 0.9428 | 0.9458 | 1.0823 |
| 原始 EBF：V10（空间距离线性权重；不做权重和归一化） | mid | (9, 128ms) | 1.2979 | 0.9196 | 0.8145 | 1.0289 |
| 原始 EBF：V10（空间距离线性权重；不做权重和归一化） | heavy | (9, 128ms) | 1.8679 | 0.9090 | 0.7603 | 0.9750 |
| EBF_optimized：V2（softw_linear；归一化） | light | (9, 128ms) | 0.1833 | 0.9356 | 0.9459 | 1.1400 |
| EBF_optimized：V2（softw_linear；归一化） | mid | (9, 128ms) | 1.2280 | 0.9181 | 0.8088 | 1.0143 |
| EBF_optimized：V2（softw_linear；归一化） | heavy | (9, 128ms) | 1.4929 | 0.9102 | 0.7537 | 1.0076 |

对应产物（可直接打开 CSV 查看 `esr_mean` 列）：

- 原始 EBF：`data/ED24/myPedestrain_06/EBF/best_params_ebf_auc_best_esr.csv`
- V10：`data/ED24/myPedestrain_06/EBF/best_params_ebf_v10_auc_best_esr.csv`
- EBF_optimized V2：`data/ED24/myPedestrain_06/EBF_optimized_V2_rerun_esr/best_params_ebfopt_v2_esr.csv`

### 阈值稳定性（同一 tag 跨 env 对比）

归一化的主要目标之一，是让**不同噪声环境下的阈值尺度更可比**。

- 原始 EBF（未归一化）：Thr 量级随噪声显著漂移（约 `0.75 → 7.28`，range≈`6.53`）
- EBFV2（softw_linear）：Thr_light/mid/heavy = `0.1833 / 1.2280 / 1.4929`，range=`1.3096`，std=`0.5653`
- EBFV3（softw_recent）：Thr_light/mid/heavy = `0.1861 / 1.3142 / 1.6221`，range=`1.4360`，std=`0.6173`
- EBFV4（softw_linear_block；固定 tag=s9,tau=128ms）：Thr_light/mid/heavy = `0.0042 / 0.4128 / 0.4866`，range=`0.4824`，std=`0.2122`
- EBFV4.1（softw_linear_blockmix；固定 tag=s9,tau=128ms）：Thr_light/mid/heavy = `0.0363 / 1.0159 / 1.1962`，range=`1.1599`，std=`0.5096`
- EBFV6（softw_linear_purity；固定 tag=s9,tau=128ms）：Thr_light/mid/heavy = `2.20e-05 / 0.8537 / 1.0217`，range=`1.0217`，std=`0.4473`
- EBFV8（softw_linear_binrate；bin=15ms；固定 tag=s9,tau=128ms）：Thr_light/mid/heavy = `1.3069 / 2.7131 / 2.7056`，range=`1.4062`，std=`0.6611`

> 注：EBFV1 这一版在 light 上最佳 tag 变成了 `tau=64ms`（与 mid/heavy 不一致），因此“跨 env 的同 tag 稳定性”本身就不成立；这正是它作为早期 baseline 的主要问题之一。

### 固定阈值可迁移性（同一阈值跨 env 对比）

上面的 best_params 是“每个环境各自挑 best 阈值”。为了回答“能否用**同一个阈值**跑三环境”，我们额外做了固定阈值评测：

- 先固定 tag：`ebfopt_labelscore_s9_tau128000`（V4 对应 tag 为 `ebfopt_softw_linear_block_labelscore_s9_tau128000`）
- 再选择一个**全局固定阈值**：扫描候选阈值，最大化三环境的 mean-F1（用每个环境 ROC 表中 *nearest threshold* 近似）

结果（best-global-f1；s=9,tau=128ms 对齐对比）：

| 版本 | thr_fixed | F1(light) | F1(mid) | F1(heavy) | mean-F1 |
|---|---:|---:|---:|---:|---:|
| EBFV2（softw_linear） | 1.3177 | 0.9060 | 0.8071 | 0.7467 | 0.8200 |
| EBFV3（softw_recent） | 1.4348 | 0.9029 | 0.8069 | 0.7482 | 0.8193 |
| EBFV4（softw_linear_block） | 0.4246 | 0.8496 | 0.7508 | 0.6757 | 0.7587 |
| EBFV4.1（softw_linear_blockmix） | 1.0661 | 0.8956 | 0.8017 | 0.7397 | 0.8123 |
| EBFV6（softw_linear_purity） | 0.9053 | 0.8948 | 0.7812 | 0.7190 | 0.7983 |
| EBFV8（softw_linear_binrate；bin=15ms） | 2.6475 | 0.9387 | 0.8139 | 0.7529 | 0.8352 |

对应产物：

- V2：`data/ED24/myPedestrain_06/EBF_optimized_V2scale_linear/fixed_thr_eval_ebfopt.csv`
- V3：`data/ED24/myPedestrain_06/EBF_optimized_V3scale_recent/fixed_thr_eval_ebfopt.csv`
- V4：`data/ED24/myPedestrain_06/EBF_optimized_V4block_linear/fixed_thr_eval_ebfopt.csv`
- V4.1：`data/ED24/myPedestrain_06/EBF_optimized_V41blockmix_linear/fixed_thr_eval_ebfopt.csv`
- V6：`data/ED24/myPedestrain_06/EBF_optimized_V6purity_linear/fixed_thr_eval_ebfopt_v6_purity_s9_tau128ms_best_global_f1.csv`
- V8：`data/ED24/myPedestrain_06/EBF_optimized_V8_binrate/fixed_thr_eval_ebfopt_v8_binrate_bin15000us_bestglobalf1.csv`

> 补充：V4 的全环境 mean-AUC 最优 tag 实际是 `s=9,tau=64ms`（light/heavy 的 AUC 最优点），对应固定阈值评测也已单独保存：
> `data/ED24/myPedestrain_06/EBF_optimized_V4block_linear/fixed_thr_eval_ebfopt_tau64ms.csv`（mean-F1≈0.7612）。

### 产物位置（便于追溯）

- 原始 EBF 汇总：`data/ED24/myPedestrain_06/算法最优参数汇总_auc优先_f1次之.csv`
- EBFV1（早期）：`data/ED24/myPedestrain_06/EBF_optimized_V1equalw_linear/back_1.../best_params_ebf_optimized.csv`
- EBFV2：`data/ED24/myPedestrain_06/EBF_optimized_V2scale_linear/best_params_ebf_optimized_linear.csv` + `thr_stability_ebf_optimized_linear.csv`
- EBFV3：`data/ED24/myPedestrain_06/EBF_optimized_V3scale_recent/best_params_ebf_optimized_recent.csv` + `thr_stability_ebf_optimized_recent.csv`
- EBFV4：`data/ED24/myPedestrain_06/EBF_optimized_V4block_linear/best_params_ebf_optimized_block_besttag.csv`（按 env 选 best tag）
  + `best_params_ebf_optimized_block_tau128ms.csv` + `thr_stability_ebf_optimized_block_tau128ms.csv`（固定 tag=s9,tau=128ms）
- EBFV4.1：`data/ED24/myPedestrain_06/EBF_optimized_V41blockmix_linear/best_params_ebf_optimized_blockmix_besttag.csv`
  + `thr_stability_ebf_optimized_blockmix_besttag.csv` + `fixed_thr_eval_ebfopt.csv`
- EBFV6：`data/ED24/myPedestrain_06/EBF_optimized_V6purity_linear/best_params_ebfopt_v6_purity_perenvbest.csv`
  + `best_params_ebfopt_v6_purity_forced_s9_tau128000.csv` + `thr_stability_ebfopt_v6_purity_forced_s9_tau128000.csv`
  + `fixed_thr_eval_ebfopt_v6_purity_best_global_f1.csv`（tag 按 mean-AUC 自动选；当前为 s=7,tau=64ms）

- EBFV8（binrate；bin=15000us）：`data/ED24/myPedestrain_06/EBF_optimized_V8_binrate/roc_ebf_optimized_softw_linear_binrate_{env}_labelscore_s3_5_7_9_tau8_..._1024ms.(csv|png)`
  + `best_params_ebfopt_v8_binrate_bin15000us.csv` + `thr_stability_ebfopt_v8_binrate_bin15000us.csv`
  + `fixed_thr_eval_ebfopt_v8_binrate_bin15000us_bestglobalf1.csv`

> 注：V6 在全量 sweep 下的 per-env best tag 为：light `s=9,tau=64ms`；mid/heavy `s=7,tau=64ms`。

---

## 预筛结论（单 tag / 快速扫参）

本节用于记录“快速扫参”的结论：固定 tag（通常取 `s=9,tau=128ms`）+ 限制事件数（如 `max-events=200k`），先判断方向值不值得继续。

### V4.1：β sweep（block/global 混合系数）

固定：`s=9, tau=128ms, max-events=200k`

结论（AUC）：

- 随着 `β` 增大，`light/mid/heavy` 三环境的 AUC **单调下降**。
- 最优点落在 `β=0`，此时 V4.1 **退化为 V2**（在该预筛口径下，与 EBFV2 的 AUC 完全一致）。
- 因此可以认为：在“local(block) rate → scale”的这条具体路线里，混入 block 信息是**负贡献**，继续调参收益极低。

对应汇总 CSV：

- `data/ED24/myPedestrain_06/_prescreen_beta_v41_s9_tau128ms/beta_prescreen_auc_summary.csv`

### V5：γ sweep（same - gamma*opp raw score）

固定：`s=9, tau=128ms, max-events=200k`

结论（AUC）：

- `gamma` 只要 >0，AUC 即出现明显下降；且三环境趋势一致。
- 最优点落在 `gamma=0`（退化为 V2）。

这说明：在本数据上，“把异极性邻居作为反证项扣分”会误伤真实运动/边缘结构，对排序判别（AUC）是负贡献；该 raw-score 路线不建议继续深挖。

对应汇总 CSV：

- `data/ED24/myPedestrain_06/_prescreen_gamma_v5_s9_tau128ms/gamma_prescreen_auc_summary.csv`

### 原始 EBF：V10（空间距离权重；线性衰减；不做归一化）

实现：`src/myevs/denoise/ops/ebf_v10_spatialw_linear.py`

固定：`s=9, tau=128ms, max-events=200k`

运行命令（对照）：

```powershell
# baseline
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant ebf --max-events 200000 --s-list 9 --tau-us-list 128000 --out-dir data/ED24/myPedestrain_06/EBF

# V10
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant EBFV10 --max-events 200000 --s-list 9 --tau-us-list 128000 --out-dir data/ED24/myPedestrain_06/EBF
```

结论（该预筛口径下）：V10 的 AUC/F1 **未提升**，三环境均小幅下降。

| env | AUC (EBF) | AUC (V10) | best-F1 (EBF) | best-F1 (V10) |
|---|---:|---:|---:|---:|
| light | 0.9476 | 0.9428 | 0.9497 | 0.9458 |
| mid | 0.9219 | 0.9187 | 0.8108 | 0.8069 |
| heavy | 0.9205 | 0.9173 | 0.7869 | 0.7839 |

说明：

- 按你的要求，V10 **不除以权重和**（不归一化），因此分数尺度变小，最佳阈值会明显漂移。
  - 本次预筛的 best-F1 阈值（thr=value）约为：
    - EBF：`0.749 / 4.839 / 7.358`（light/mid/heavy）
    - V10：`0.00478 / 1.293 / 1.834`
- 产物（ROC 曲线 + 全阈值表）：
  - baseline：`data/ED24/myPedestrain_06/EBF/roc_ebf_{env}_labelscore_s3_5_7_9_tau8_16_32_64_128_256_512_1024ms.(csv|png)`
  - V10：`data/ED24/myPedestrain_06/EBF/roc_ebf_v10_{env}_labelscore_s3_5_7_9_tau8_16_32_64_128_256_512_1024ms.(csv|png)`

### V7：polrate（按极性拆分 rate EMA）

固定：`s=9, tau=128ms, max-events=200k`

结论（预筛口径）：

- AUC 基本不掉太多，但阈值稳定性反而变差：Thr_light/mid/heavy = `0.0749 / 1.9642 / 2.2616`，range=`2.1868`，std=`0.9684`
- fixed-threshold（best-global-f1）：thr_fixed≈`2.0447`，mean-F1≈`0.8249`

直观解释：把事件按 ON/OFF 分流后，每条 EMA 的有效样本数变少，rate proxy 的方差会变大；当 ON/OFF 比例变化并不是主要误差源时，这种“拆分”更像是在给归一化引入额外噪声，因此稳定性可能更差。

对应产物：

- `data/ED24/myPedestrain_06/_prescreen_v7_polrate_s9_tau128ms/roc_ebf_optimized_softw_linear_polrate_{env}_labelscore_s9_tau128ms.(csv|png)`
- `data/ED24/myPedestrain_06/_prescreen_v7_polrate_s9_tau128ms/best_params_v7_polrate_forced_s9_tau128ms.csv`
- `data/ED24/myPedestrain_06/_prescreen_v7_polrate_s9_tau128ms/thr_stability_v7_polrate_forced_s9_tau128ms.csv`
- `data/ED24/myPedestrain_06/_prescreen_v7_polrate_s9_tau128ms/fixed_thr_eval_ebfopt_v7_polrate_s9_tau128ms_bestglobalf1.csv`

### V8：bin sweep（binrate 全局噪声率 proxy）

固定：`s=9, tau=128ms`

结论（全量口径，单 tag 对齐，优先看阈值稳定性）：

- bin=10ms：Thr_range=`1.5248`，Thr_std=`0.7061`
- bin=15ms：Thr_range=`1.4062`，Thr_std=`0.6611`（本次对比里最稳）
- bin=20ms：Thr_range=`1.4896`，Thr_std=`0.6993`

说明：bin 继续增大并不会单调改善稳定性（过大 bin 会把 rate 估计变“慢/滞后”，在不同噪声环境下的偏差也会变大），因此需要在“方差下降”和“偏差/滞后上升”之间折中；本数据上 15ms 是当前测试到的最优点。

对应产物（全量单-tag 对齐）：

- `data/ED24/myPedestrain_06/EBF_optimized_V8_binrate/_fulltag_v8_binrate_10000us_s9_tau128ms/`
- `data/ED24/myPedestrain_06/EBF_optimized_V8_binrate/_fulltag_v8_binrate_15000us_s9_tau128ms/`
- `data/ED24/myPedestrain_06/EBF_optimized_V8_binrate/_fulltag_v8_binrate_20000us_s9_tau128ms/`

对应汇总 CSV：

- `data/ED24/myPedestrain_06/EBF_optimized_V8_binrate/thr_stability_fulltag_v8_binrate_10000us_s9_tau128ms.csv`
- `data/ED24/myPedestrain_06/EBF_optimized_V8_binrate/thr_stability_fulltag_v8_binrate_15000us_s9_tau128ms.csv`
- `data/ED24/myPedestrain_06/EBF_optimized_V8_binrate/thr_stability_fulltag_v8_binrate_20000us_s9_tau128ms.csv`

### V2/V8：α sweep（scale exponent；`score_norm = raw / scale^α`）

固定（预筛口径）：

- `s=9, tau=128ms, max-events=200k`
- V8 额外固定：`MYEVS_EBFOPT_BINRATE_US=15000`（bin=15ms）

**V2（EBFV2 / softw_linear）**：

| α | AUC_mean | Thr_std | fixed mean-F1 |
|---:|---:|---:|---:|
| 0.8 | 0.9247 | 0.7917 | 0.8240 |
| 0.9 | 0.9238 | 0.6659 | 0.8257 |
| 1.0 | 0.9229 | 0.5706 | 0.8271 |
| 1.1 | 0.9219 | 0.4926 | 0.8280 |
| 1.2 | 0.9208 | 0.4218 | 0.8285 |

结论：增大 `α` 会显著提升阈值稳定性（Thr_std 下降）并小幅提升 fixed mean-F1，但 AUC 会随之缓慢下降；若“稳定性优先且 AUC 不明显掉”，本口径下推荐 `α≈1.1`。

**V8（EBFV8 / softw_linear_binrate；bin=15ms）**：

| α | AUC_mean | Thr_std | fixed mean-F1 |
|---:|---:|---:|---:|
| 0.8 | 0.9312 | 0.6622 | 0.8452 |
| 0.9 | 0.9311 | 0.5178 | 0.8466 |
| 1.0 | 0.9310 | 0.3985 | 0.8471 |
| 1.1 | 0.9309 | 0.2838 | 0.8467 |
| 1.2 | 0.9307 | 0.2717 | 0.8454 |

结论：V8 的 fixed mean-F1 在 `α=1.0` 略高，但 `α=1.1` 在几乎不掉 AUC 的情况下能大幅提升阈值稳定性（Thr_std 下降显著）；若稳定性优先，推荐 `α≈1.1`。

对应汇总 CSV：

- `data/ED24/myPedestrain_06/_prescreen_v2_alpha_s9_tau128ms/alpha_sweep_summary_v2_s9_tau128ms.csv`
- `data/ED24/myPedestrain_06/_prescreen_v8_bin15ms_alpha_s9_tau128ms/alpha_sweep_summary_v8_bin15ms_s9_tau128ms.csv`

### V2：全局噪声率 EMA α sweep（rate_ema_alpha；`MYEVS_EBFOPT_RATE_EMA_ALPHA`）

固定（预筛口径）：

- V2（EBFV2 / softw_linear）
- `s=9, tau=128ms, max-events=200k`
- 固定阈值评测口径：best-global-f1（同一阈值跨 env mean-F1 最大）

结果：

| rate_ema_alpha | AUC_mean | Thr_std | Thr_range | thr_fixed | fixed mean-F1 |
|---:|---:|---:|---:|---:|---:|
| 0.002 | 0.9247 | 0.5478 | 1.2489 | 1.3441 | 0.8296 |
| 0.005 | 0.9236 | 0.5544 | 1.2857 | 1.3310 | 0.8284 |
| 0.010 | 0.9229 | 0.5706 | 1.3116 | 1.3210 | 0.8271 |
| 0.020 | 0.9220 | 0.5835 | 1.3575 | 1.3454 | 0.8247 |
| 0.050 | 0.9198 | 0.6261 | 1.4439 | 1.3859 | 0.8186 |

结论：

- `rate_ema_alpha` 越小，AUC / fixed mean-F1 / 阈值稳定性整体越好；在本 sweep 范围内，最优点落在 `0.002`。
- 继续减小可能会引入“rate proxy 适应过慢”的风险（尤其是噪声分布快速变化时），因此若更偏稳健默认值，`0.005` 也是较安全的折中点。

对应汇总 CSV：

- `data/ED24/myPedestrain_06/_prescreen_v2_rateema_s9_tau128ms/rateema_alpha_sweep_summary_v2_s9_tau128ms.csv`

### V9：按真实时间常数的全局噪声率 EMA sweep（rate_ema_tau；`MYEVS_EBFOPT_RATE_EMA_TAU_US`）

固定（预筛口径）：

- V9（EBFV9 / softw_linear_timeconst_rateema）
- `s=9, tau=128ms, max-events=200k`
- 固定阈值评测口径：best-global-f1（同一阈值跨 env mean-F1 最大）

结果：

| rate_ema_tau_us | AUC_mean | Thr_std | Thr_range | thr_fixed | fixed mean-F1 |
|---:|---:|---:|---:|---:|---:|
| 20000 | 0.9229 | 1.2571 | 2.7726 | 3.1999 | 0.8363 |
| 50000 | 0.9256 | 1.2266 | 2.6600 | 3.2066 | 0.8399 |
| 100000 | 0.9417 | 0.7954 | 1.7074c | 3.1804 | 0.8941 |
| 200000 | 0.9280 | 0.9929 | 2.1431 | 3.1482 | 0.8437 |
| 500000 | 0.9290 | 0.8766 | 1.8677 | 3.1451 | 0.8453 |

结论：

- 在本 sweep 范围内，`rate_ema_tau_us=100000`（100ms）同时给出最高的 AUC_mean / fixed mean-F1，并且阈值稳定性也最好（Thr_std / Thr_range 最低）。
- 经验上，tau_rate 太小（20ms/50ms）会导致 rate proxy 偏“躁动”，scale 波动变大；太大（200ms/500ms）则适应变慢，整体指标回落。

对应汇总 CSV：

- `data/ED24/myPedestrain_06/_prescreen_v9_rateema_tau_s9_tau128ms/rateema_tau_sweep_summary_v9_s9_tau128ms.csv`
