# myEVS 去噪算法对比实验总控 README

本 README 仅用于 `src/myevs/denoise/ops` 维度的跨算法对比实验管理，约束后续：
- 数据集路径
- 结果存储路径
- 扫频/运行脚本入口
- 指标统计与汇总表格式

当前状态（2026-04-23）：
- 已集成算法：`BAF`, `STCF(stc)`, `EBF`, `EBF_OPTIMIZED`, `KNOISE`, `EVFLOW`, `YNOISE`, `TS`, `MLPF`。
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

参数映射（统一沿用 myEVS 通用参数）：
- `time-us` -> 算法中的时间窗/衰减参数（duration/decay）
- `radius-px` -> 空间搜索半径（search radius）
- `min-neighbors` -> 阈值（intThreshold/floatThreshold）

说明：
- `mlpf` 在本工程为轻量代理实现（MLP-inspired 特征评分），用于统一框架对比，不依赖外部 TorchScript 模型。

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
- `sweep_*.py`/`prescreen_*.py`/`tune_*.py`：各版本候选公式预筛与调参
- `run_slomo_alg.ps1`（新增）：KNOISE/EVFLOW/YNOISE/TS/MLPF 统一入口
- `run_slomo_{knoise|evflow|ynoise|ts|mlpf}.ps1`（新增）：单算法入口

### 5.3 driving_alg_evalu

- `run_driving_alg.ps1`（新增）：Driving 数据集统一入口
- `run_driving_{knoise|evflow|ynoise|ts|mlpf}.ps1`（新增）：单算法入口

### 5.4 noise_analyze

用于噪声结构分析、特征统计、分布可视化和误检类型分析（FP/transition/pattern 等），为算法改进提供先验证据。

## 6. 统一指标与汇总表

目标指标：
- `AUC`
- `F1`
- `MESR`
- `AOCC`

建议总汇 CSV（后续维护）：
- `data/summary/alg_compare_master.csv`

列定义（最小必需）：
- `dataset, scene, level, algorithm, tag, auc, f1, mesr, aocc, time_us, radius_px, threshold, csv_path, fig_path, note`

要求：
- 每次新增实验必须写入总汇 CSV；`csv_path/fig_path` 必填。
- `tag` 与 `time_us/radius_px/threshold` 必须可互相还原。

## 7. 推荐执行顺序

1. 先跑 ED24：`BAF/STCF/EBF + KNOISE/EVFLOW/YNOISE/TS/MLPF`
2. 再跑 Driving：同样算法集合
3. 每个算法先看三档噪声（light/mid/heavy）的 AUC 稳定性
4. 再做跨数据集总表排序，筛选论文主结果算法

## 8. 约束与后续补充规则

- 所有新增结果必须进入本 README 对应章节与 `data/summary/alg_compare_master.csv`。
- 任何脚本新增后，必须在“scripts 目录功能说明”登记用途。
- 若改动数据路径或命名规则，先改本 README，再跑实验。
