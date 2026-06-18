# LLEF 热状态可靠性折扣验证计划

## 注意
- 所有实验脚本和数据需要有记录
- 本脚本第一章介绍计划，脚本文件和输出产物路径，第二章记录数据和结果及分析
- 脚本和实验运行完成之后请在后续或者对应章节更新或者补充实验数据结果和分析结论

## 1. 目标

验证 LLEF 中热状态可靠性折扣不是经验性后处理，而是针对低光照高噪声事件流中“中心像素持续异常触发”的可靠性建模。

核心问题：

- 低光照 ED24 中，噪声事件是否比真实事件更容易出现在高频重复触发像素上？
- 中心热状态 \(H_i^{\mathrm{pre}}\) 或归一化热度 \(q_i\) 是否能区分真实事件和噪声事件？
- 默认热状态更新参数 \(\lambda=2\) 和默认折扣函数是否在 AUC/F1 上稳定？
- 去除热状态折扣后，错误是否主要增加在 hot/high-rate 像素区域？

## 2. 固定输出目录

所有中间文件、CSV、NPZ、图片和汇总表统一放在：

```text
D:\hjx_workspace\scientific_reserach\projects\myEVS\data\hot_state_validation\
```

目录结构：

```text
data\hot_state_validation\
  raw_stats\       # 原始事件率、像素触发频率、标签比例统计
  hot_trace\       # 每事件诊断数据：H_pre/H_post/discount/raw_same/raw_opp/score/label
  figures\         # 论文候选图：分布图、CDF、箱线图、消融图
  tables\          # AUC/F1/分位数/kept-rate 汇总 CSV 和 Markdown 表
  logs\            # stdout、运行参数 JSON、环境变量快照
```

建议每次运行使用时间戳或版本号，例如：

```text
data\hot_state_validation\hot_trace\ed24_ped06_3p3v_lleffinal_202606xx.parquet
data\hot_state_validation\tables\hot_state_quantiles_ed24_202606xx.csv
data\hot_state_validation\figures\hot_state_cdf_ed24_202606xx.pdf
```

## 3. 数据集优先级

必做数据集：

| 数据集 | 噪声等级 | 目的 |
|---|---:|---|
| ED24 Pedestrian06 | 1.8 V, 2.5 V, 3.3 V | 行人弱结构、低光真实 BA 噪声，重点验证热状态 |
| ED24 Bicycle02 | 1.8 V, 2.5 V, 3.3 V | 边缘结构更清楚，用于验证结论不是 Pedestrian 特例 |

可选数据集：

| 数据集 | 噪声等级 | 用途 |
|---|---:|---|
| DND21 Driving | 10 Hz | 验证仿真 shot noise 下热状态作用较弱 |
| MAH00447 | ratio100 | 验证受控真实噪声混合场景 |

不纳入主验证：

- LED：该数据集极稀疏，已有实验表明热状态可能误伤低频信号事件。可作为讨论中的边界案例，不作为 LLEF 主线证据。

## 4. 实验 A：像素触发频率与事件间隔分布

目的：证明低光高噪声下，噪声事件更集中于高频重复触发像素。

统计内容：

- 每个像素的总触发次数；
- 每个像素的真实事件触发次数与噪声事件触发次数；
- 每个事件与同像素上一次事件的间隔 \(\Delta t_i^0\)；
- valid/noise 两类事件的 log-frequency 或 inter-event interval 分布。

输出文件：

```text
raw_stats\pixel_rate_<dataset>_<level>.csv
raw_stats\inter_event_interval_<dataset>_<level>.csv
figures\pixel_rate_hist_<dataset>_<level>.pdf
figures\iei_cdf_<dataset>_<level>.pdf
tables\pixel_rate_summary_ed24.csv
```

建议图表：

- valid/noise 的 \(\log_{10}\) pixel event rate 直方图；
- valid/noise 的 \(\Delta t_i^0\) CDF；
- top 1%, top 5% 高频像素贡献的噪声事件比例。

可写入论文的结论模板：

> 在 ED24 低光高噪声子集中，噪声事件在高频重复触发像素上更集中，说明中心像素近期活跃度可作为局部邻域证据之外的可靠性线索。

## 5. 实验 B：热状态分布验证

目的：证明 LLEF 的中心热状态直接对准噪声事件，而不是任意经验变量。

需要导出的每事件字段：

```text
x, y, t, p, label, score, H_pre, H_post, q_pre, q_post, discount, raw_same, raw_opp
```

其中：

- `label=1` 表示真实事件，`label=0` 表示噪声事件；
- `H_pre` 是当前事件到来前的中心像素热状态；
- `H_post` 是当前事件更新后的热状态；
- `q_pre=H_pre/Q`，`q_post=H_post/Q`；
- `discount=(H_post+Q)/(2H_post+Q)`。

如果当前 `N149Native.score_batch` 只能输出最终 score，则新增诊断脚本或诊断绑定，不改变主算法逻辑。建议新增诊断入口：

```text
scripts\n149_ablation\dump_hot_state_trace.py
```

建议命令形式：

```powershell
$env:MYEVS_N149_SIGMA="2.75"
$env:MYEVS_N149_ALPHA_FIXED="0.25"
$env:MYEVS_N149_HOT_BITS="8"
$env:MYEVS_N149_HOT_DECAY_K="2"
python scripts\n149_ablation\dump_hot_state_trace.py `
  --dataset ed24_ped06 `
  --level 3.3v `
  --radius 3 `
  --tau-us 256000 `
  --out-dir data\hot_state_validation\hot_trace
```

输出文件：

```text
hot_trace\trace_<dataset>_<level>.npz
hot_trace\trace_<dataset>_<level>.parquet
tables\hot_state_quantiles_<dataset>_<level>.csv
figures\hot_state_cdf_<dataset>_<level>.pdf
figures\hot_state_box_<dataset>_<level>.pdf
```

统计表字段：

```text
dataset, level, class, count, mean, median, p75, p90, p95, p99
```

可写入论文的结论模板：

> 噪声事件的 \(H_i^{\mathrm{pre}}\) 分布整体高于真实事件，尤其在 Pedestrian 2.5 V 和 3.3 V 中更明显，说明中心热状态能够捕捉低光 BA 噪声中的持续异常触发。

## 6. 实验 C：\(\lambda\) 与折扣函数消融

目的：解释为什么热状态更新采用默认 \(\lambda=2\)，以及为什么默认折扣函数合理。

优先复用脚本：

```text
D:\hjx_workspace\scientific_reserach\projects\myEVS\scripts\n149_ablation\run_hot_k_f1_20260605.py
D:\hjx_workspace\scientific_reserach\projects\myEVS\scripts\n149_ablation\run_hot_func_sweep_20260605.py
```

新增或改造要求：

- 所有输出必须支持 `--out-dir data\hot_state_validation\...`；
- 若脚本当前写死输出路径，增加 `--out-dir` 参数；
- 保存运行参数 JSON 到 `logs\`。

扫描设置：

| 参数 | 扫描值 |
|---|---|
| \(\lambda\) / `MYEVS_N149_HOT_DECAY_K` | 0, 1, 2, 3, 4 |
| 折扣函数 | rational/default, exp, hill, power, linear |
| `HOT_BITS` | 固定 8 |
| ED24 参数 | Ped/Bike: \(r=3,\tau=256\) ms, \(\sigma=2.75,\alpha=0.25\) |

输出文件：

```text
tables\hot_lambda_auc_f1_ed24.csv
tables\hot_func_auc_f1_ed24.csv
figures\hot_lambda_sweep_ed24.pdf
figures\hot_func_sweep_ed24.pdf
logs\hot_lambda_sweep_<date>.json
logs\hot_func_sweep_<date>.json
```

评价指标：

- AUC；
- AUC 最优阈值附近的 F1；
- 不同噪声等级上的平均值与最差值。

可写入论文的结论模板：

> \(\lambda=2\) 在 Pedestrian/Bicycle 的多个低光噪声等级上取得稳定的 AUC/F1，\(\lambda=1\) 对持续高频噪声抑制不足，而更大的 \(\lambda\) 会增加对真实高频结构的压制。

## 7. 实验 D：No hot 错误类型对比

目的：证明热状态折扣主要减少 hot/high-rate 像素区域中的噪声误保留。

对比方法：

- LLEF 完整版本；
- No hot：设置 `MYEVS_N149_NO_HOT=1`；
- 可选：Time only。

统计对象：

- 全局 TP、FP、TN、FN；
- hot/high-rate 像素区域内的 TP/FP；
- near-hot 区域内的 TP/FP；
- signal kept rate 与 noise kept rate。

hot/high-rate 区域定义：

- 方案一：按训练/测试事件流中像素触发次数 top 1%；
- 方案二：按噪声事件计数 top 1%；
- 方案三：按已有 hotmask 诊断文件，如果数据集中已有稳定 hotmask。

输出文件：

```text
tables\no_hot_error_breakdown_ed24.csv
tables\hot_region_kept_rate_ed24.csv
figures\hot_region_error_bar_ed24.pdf
figures\qualitative_nohot_vs_llef_<dataset>_<level>.pdf
```

可写入论文的结论模板：

> 与 No hot 相比，完整 LLEF 在 hot/high-rate 区域减少了更多噪声误保留，同时保持主要真实结构事件，说明热状态折扣主要作用于持续异常触发像素。

## 8. 实施顺序

1. 创建输出目录：

```powershell
New-Item -ItemType Directory -Force `
  data\hot_state_validation\raw_stats, `
  data\hot_state_validation\hot_trace, `
  data\hot_state_validation\figures, `
  data\hot_state_validation\tables, `
  data\hot_state_validation\logs
```

2. 先完成实验 A：无需改 C++，只依赖事件和标签统计。
3. 再完成实验 B：若现有接口不能导出 `H_pre/H_post`，新增诊断导出。
4. 复用或改造实验 C 的 sweep 脚本。
5. 最后完成实验 D 的错误类型统计和定性图。

## 9. 最终交付物

至少生成以下可直接进入论文或补充材料的产物：

```text
figures\hot_state_cdf_ed24.pdf
figures\hot_lambda_sweep_ed24.pdf
figures\hot_region_error_bar_ed24.pdf
tables\hot_state_quantiles_ed24.csv
tables\hot_lambda_auc_f1_ed24.csv
tables\no_hot_error_breakdown_ed24.csv
```

主文优先使用：

- 1 张热状态 CDF 或箱线图；
- 1 个简短 \(\lambda\) 消融表；
- 1 段 No hot 错误类型解释。

补充材料可展开全部子集和完整表格。

## 10. 已完成实验记录与阶段结论（2026-06-12）

### 10.1 本轮运行范围

本轮已完成 ED24 主验证数据集的热状态折扣有效性验证：

| 数据集 | 噪声等级 | 实验 |
|---|---:|---|
| ED24 Pedestrian06 | 1.8 V, 2.5 V, 3.3 V | A/B/C/D |
| ED24 Bicycle02 | 1.8 V, 2.5 V, 3.3 V | A/B/C/D |

统一输出目录：

```text
D:\hjx_workspace\scientific_reserach\projects\myEVS\data\hot_state_validation\
```

已生成核心交付物：

```text
figures\hot_state_cdf_ed24.pdf
figures\hot_lambda_sweep_ed24.pdf
figures\hot_func_sweep_ed24.pdf
figures\hot_region_error_bar_ed24.pdf
tables\hot_state_quantiles_ed24.csv
tables\hot_lambda_auc_f1_ed24.csv
tables\hot_func_auc_f1_ed24.csv
tables\no_hot_error_breakdown_ed24.csv
tables\hot_region_kept_rate_ed24.csv
tables\pixel_rate_summary_ed24.csv
```

每个子集均已导出每事件诊断 trace：

```text
hot_trace\trace_<dataset>_<level>.npz
hot_trace\trace_<dataset>_<level>.parquet
hot_trace\trace_<dataset>_<level>_nohot.npz
hot_trace\trace_<dataset>_<level>_nohot.parquet
```

本轮使用的主要脚本：

```text
scripts\n149_ablation\hot_state_validation.py
scripts\n149_ablation\run_hot_k_f1_20260605.py
scripts\n149_ablation\run_hot_func_sweep_20260605.py
```

说明：当前 `hot_state_validation.py` 是统一入口，便于一次性复现实验 A/B/C/D。后续建议将其拆分为 `dump_hot_state_trace.py`、`run_pixel_rate_stats.py`、`run_hot_lambda_sweep.py`、`run_nohot_error_breakdown.py` 等模块化脚本，以便单独运行和维护。

### 10.2 实验 A 结论：高频重复触发像素与噪声集中性

实验 A 已导出每像素事件率、真实/噪声计数、同像素事件间隔分布以及 top 高频像素贡献统计。

阶段结论：

- ED24 低光高噪声子集中，噪声事件在高频重复触发像素上更集中，特别是在 Pedestrian06 2.5 V/3.3 V 和 Bicycle02 3.3 V 中更明显。
- 该现象支持“中心像素近期活跃度”作为局部邻域证据之外的可靠性线索，而不是任意经验惩罚项。
- 高频像素区域可作为后续 No-hot 错误归因的重点区域。

### 10.3 实验 B 结论：热状态分布能够表征持续异常触发

`hot_state_quantiles_ed24.csv` 显示，Pedestrian06 中噪声事件的 \(H_i^{\mathrm{pre}}\) 分布整体高于真实事件，且高分位差异更明显：

| 数据集 | 等级 | signal mean | noise mean | signal p95 | noise p95 |
|---|---:|---:|---:|---:|---:|
| Pedestrian06 | 1.8 V | 23.40 | 45.89 | 112 | 254 |
| Pedestrian06 | 2.5 V | 28.36 | 40.01 | 131 | 243 |
| Pedestrian06 | 3.3 V | 34.31 | 48.96 | 154 | 244 |

Bicycle02 中真实边缘结构本身更连续，因此 signal 的平均热状态不总是低于 noise；但 noise 的高分位仍长期接近饱和区间：

| 数据集 | 等级 | signal mean | noise mean | signal p95 | noise p95 |
|---|---:|---:|---:|---:|---:|
| Bicycle02 | 1.8 V | 47.78 | 41.71 | 194 | 254 |
| Bicycle02 | 2.5 V | 51.25 | 37.46 | 201 | 241 |
| Bicycle02 | 3.3 V | 55.97 | 45.10 | 211 | 240 |

阶段结论：

- Pedestrian06 的结果直接支持“噪声事件更容易对应高热状态中心像素”。
- Bicycle02 的结果说明热状态不是单独分类器；它需要与邻域同极性/反极性证据联合使用，避免误伤真实连续边缘结构。
- 因此，热状态折扣的合理解释应写成“对中心像素持续异常触发进行可靠性降权”，而不是“高热状态必然等于噪声”。

### 10.4 实验 C 结论：\(\lambda\) 与折扣函数稳定性

`hot_lambda_auc_f1_ed24.csv` 的 6 个 ED24 子集均值如下：

| \(\lambda\) | mean AUC | mean F1 | min AUC | min F1 |
|---:|---:|---:|---:|---:|
| 0 | 0.953599 | 0.882997 | 0.920723 | 0.765522 |
| 1 | 0.959043 | 0.885764 | 0.931567 | 0.775222 |
| 2 | 0.958479 | 0.885668 | 0.931275 | 0.770703 |
| 3 | 0.957549 | 0.884794 | 0.929486 | 0.767580 |
| 4 | 0.956750 | 0.884503 | 0.927768 | 0.769699 |

阶段结论：

- \(\lambda=1\) 在本轮 6 子集均值上略高于 \(\lambda=2\)，但两者差距很小。
- \(\lambda=2\) 的 mean AUC/mean F1 与最优均值几乎持平，且保留了更强的短时热状态更新强度，作为默认值仍然稳定合理。
- \(\lambda=0\) 明显较弱，说明完全不衰减/不形成有效短时动态的热状态不够稳定。
- \(\lambda=3,4\) 相比 \(\lambda=2\) 略降，提示过强衰减会削弱热状态对持续异常触发的表达。

折扣函数对比结果：

| 折扣函数 | mean AUC | mean F1 | min AUC | min F1 |
|---|---:|---:|---:|---:|
| exp | 0.958486 | 0.885556 | 0.931060 | 0.769637 |
| rational/default | 0.958479 | 0.885668 | 0.931275 | 0.770703 |
| linear | 0.958457 | 0.882808 | 0.930646 | 0.765597 |
| hill | 0.957807 | 0.885807 | 0.930053 | 0.769256 |
| power | 0.957807 | 0.885807 | 0.930053 | 0.769256 |

阶段结论：

- 默认 rational 折扣在 mean AUC 上与 exp 几乎相同，在 mean F1 和最差 F1 上略优。
- hill/power 的 F1 均值略高，但 AUC 与最差 AUC 稍低；线性函数 F1 明显偏弱。
- 默认 rational 函数可作为论文主版本：形式简单、单调、有界、数值稳定，并且在 AUC/F1 上不劣于其他候选。

### 10.5 实验 D 结论：No-hot 错误主要体现在 hot/high-rate 区域噪声误保留

`no_hot_error_breakdown_ed24.csv` 使用全局最佳 F1 阈值，统计完整 LLEF 与 `MYEVS_N149_NO_HOT=1` 的错误分布。3.3 V hot top-1% 区域结果如下：

| 数据集 | 方法 | hot 区域 FP | hot 区域 noise kept rate | hot 区域 signal kept rate |
|---|---|---:|---:|---:|
| Pedestrian06 3.3 V | LLEF | 2530 | 0.033631 | 0.728752 |
| Pedestrian06 3.3 V | No hot | 3737 | 0.049676 | 0.771858 |
| Bicycle02 3.3 V | LLEF | 345 | 0.008778 | 0.840841 |
| Bicycle02 3.3 V | No hot | 459 | 0.011678 | 0.854855 |

阶段结论：

- 在最高噪声 3.3 V 下，No-hot 在 hot top-1% 区域保留了更多噪声事件。
- LLEF 通过热状态折扣降低 hot/high-rate 区域 FP；代价是对部分真实高频结构事件也会更保守。
- 这与热状态折扣的设计目标一致：它不是单独提高全局保留率，而是在持续异常触发像素上降低局部证据的可靠性。

### 10.6 当前可写入论文的综合结论

本轮 ED24 Pedestrian06/Bicycle02 多噪声等级实验表明：

> 热状态折扣因子针对低光高噪声事件流中的中心像素持续异常触发建模。Pedestrian06 中噪声事件的 \(H_i^{\mathrm{pre}}\) 高分位显著高于真实事件；No-hot 对照在 hot/high-rate 区域保留更多噪声；默认 rational 折扣与 \(\lambda=2\) 在 AUC/F1 上保持稳定。Bicycle02 结果进一步说明热状态应作为邻域证据的可靠性调制项，而不是独立噪声判别变量。

### 10.7 后续工作

- 将当前统一脚本拆分为按实验独立运行的模块化脚本。
- 为每个脚本增加明确的命令示例、输入检查和 `--out-dir` 参数。
- 每次新增实验后继续在本计划文件中追加“实验日期、运行范围、关键数值、结论和局限性”。
- 可选扩展到 DND21 Driving 与 MAH00447，验证真实 BA 噪声与仿真 shot noise 下热状态贡献的差异。

### 10.8 图表含义与当前模型解释

#### 10.8.1 计划是否完整实施

本轮实验已经完整覆盖原计划中的 ED24 主验证部分：

- 实验 A：已完成每像素触发频率、同像素事件间隔和 top 高频像素噪声比例统计。
- 实验 B：已完成每事件 `H_pre/H_post/discount/score/label` 诊断导出，并生成热状态分布图、CDF 图和分位数表。
- 实验 C：已完成 \(\lambda\) 扫描和折扣函数扫描，输出 AUC/F1 表和曲线。
- 实验 D：已完成完整 LLEF 与 No-hot 的错误分解，并统计 hot top-1% 区域的 signal/noise kept rate。

当前结论已经足够支撑“ED24 低光高噪声场景下热状态折扣的合理性”。但可选数据集 DND21 Driving 和 MAH00447 尚未纳入本轮验证，因此若论文中只讨论热状态机制，建议把实验范围明确限定为 ED24 低光真实 BA 噪声，而不要泛化为所有数据集都具有相同热状态分布。

#### 10.8.2 能否从噪声特性推出合适的去噪函数

当前数据不支持把某一个具体函数形式写成严格理论最优，例如不能说 rational/default 折扣是唯一正确模型。更稳妥的理论叙事是：

1. 低光 BA 噪声中存在持续异常触发像素，高触发率像素区域在 2.5 V/3.3 V 下明显富集噪声事件。
2. 中心热状态 \(H_i^{\mathrm{pre}}\) 是一个带泄露的近期重复触发计数器，用于估计当前中心像素的可靠性风险。
3. 热状态并不能单独区分真实事件与噪声事件，特别是 Bicycle02 中真实边缘结构也可能具有较高热状态。因此，热状态不应作为硬删除规则，而应作为有界、单调的可靠性折扣。
4. 多种单调有界折扣函数的 AUC/F1 接近，说明真正有效的是“基于中心像素近期活跃度进行可靠性降权”这一建模约束，而不是某个函数被偶然试出来。
5. 默认 rational 折扣的优势在于形式简单、单调、有界、数值稳定，并且在当前 ED24 扫描中与 exp 等候选函数性能接近，可作为满足上述约束的简洁实现。

因此，论文中建议这样表述：

> 热状态折扣不是独立的噪声分类器，而是对邻域支撑评分进行中心像素可靠性调制。低光高噪声下，高频重复触发像素更容易富集背景活动噪声；因此，折扣函数只需满足随热状态单调下降且保持有界，便可降低持续异常触发事件对最终评分的影响，同时避免直接删除真实高频结构事件。

#### 10.8.3 CDF 的含义以及 `H_pre` CDF 图代表什么

CDF 是 cumulative distribution function，即累积分布函数。对热状态 \(H_i^{\mathrm{pre}}\) 而言，

\[
F(h)=P(H_i^{\mathrm{pre}}\le h)
\]

表示有多少比例的事件，其到来前的中心热状态不超过 \(h\)。

读图方法：

- 曲线越靠左，说明该类事件的 \(H_i^{\mathrm{pre}}\) 通常越小。
- 曲线越靠右，或在高 \(H\) 区间上升更慢，说明该类事件具有更重的高热状态尾部。
- 如果 noise 曲线在高 \(H\) 区间明显比 signal 更靠右，说明噪声事件更容易出现在近期重复触发较强的中心像素上。
- 如果两条曲线重叠明显，说明热状态不能单独作为分类器，只能作为可靠性调制项。

`H_pre` 表示当前事件到来之前该中心像素已经积累的热状态。使用 `H_pre` 而不是只看当前更新后的状态，是为了说明当前事件的判决可以由历史触发模式提供可靠性线索，而不是事后利用当前事件标签。

#### 10.8.4 为什么 \(\lambda=1\) 和 \(\lambda=2\) 效果接近

\(\lambda\) 控制热状态的时间泄露强度，可理解为近期重复触发计数器的记忆长度。当前实验中，\(\lambda=1\) 和 \(\lambda=2\) 的 AUC/F1 很接近，原因是：

- ED24 中持续异常触发像素的时间尺度较长，在 \(\lambda=1\) 和 \(\lambda=2\) 下都会保持较高热状态。
- 孤立或低频事件在两种设置下都会保持较低热状态，因此事件排序差异有限。
- AUC 由阈值扫描得到，对全局评分尺度变化不敏感；只要事件排序基本不变，AUC 就会接近。
- 8-bit 热状态存在饱和区间，许多高频噪声在两种 \(\lambda\) 下都会进入高热状态尾部，使二者差异进一步变小。

因此，\(\lambda=1\) 和 \(\lambda=2\) 接近不是坏事，而是说明热状态折扣存在稳定平台。论文中不应强调 \(\lambda=2\) 是严格最优，而应写成：

> \(\lambda=1\) 和 \(\lambda=2\) 均位于稳定性能平台；本文采用 \(\lambda=2\)，以在保持稳定 AUC/F1 的同时对短时间内重复触发的中心像素给出更快的可靠性响应。

#### 10.8.5 top 1%、p95 和 `hot_state_box_*` 图的含义

`top 1%` 指按像素总触发次数排序后，选出触发最频繁的前 1% 像素。它用于回答：高频触发像素区域中，噪声事件是否更集中。当前结果显示，在 2.5 V/3.3 V 下 top 1% 像素中的噪声比例约为 95%--98%，这是热状态折扣最直接的统计依据之一。

`p95` 是 95th percentile，即第 95 百分位数。若 noise 的 `p95=244`，表示 95% 的噪声事件 \(H_i^{\mathrm{pre}}\le244\)，剩余 5% 的噪声事件高于 244。`p95` 和 `p99` 主要用于观察高热状态尾部，比均值更适合描述“少量但很强的热点/高频异常触发”。

`hot_state_box_*` 是热状态箱线图。一般读法如下：

- 中间线表示中位数 median。
- 箱体表示 25%--75% 分位区间，即中间一半事件的分布范围。
- 须线和离群点表示更高或更低的尾部样本。

在当前实验中，箱线图可以展示 signal/noise 的整体重叠情况，但由于很多事件的热状态集中在低值，同时高噪声事件又存在接近饱和的长尾，箱线图不一定最直观。论文或补充材料中更推荐使用 CDF 图和 `p95/p99` 分位数表来说明高热状态尾部差异。

#### 10.8.6 当前结果的写作边界

可以写：

> ED24 低光高噪声子集中，高频重复触发像素显著富集噪声事件；噪声事件在 \(H_i^{\mathrm{pre}}\) 的高分位上更容易接近饱和。该结果说明中心像素近期活跃度能够提供邻域支撑之外的可靠性线索。由于真实连续边缘也可能产生较高热状态，LLEF 将热状态用于有界折扣，而不是直接删除高热状态事件。

不建议写：

- “热状态可以直接区分真实事件和噪声事件。”
- “\(\lambda=2\) 是理论最优。”
- “默认折扣函数是唯一最优函数。”
- “所有数据集上噪声事件的热状态整体都高于真实事件。”

这些说法都比当前数据更强，容易被审稿人抓住反例。
