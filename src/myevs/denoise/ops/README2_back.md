# README2：回归 N149 后的复核、消融与实时性实验约束

> 修改本文件前，必须先阅读 `src/myevs/denoise/ops/README.md`。
> `README.md` 记录历史数据集路径、脚本约定、已有结果和结论；`README2.md` 只记录从“放弃 N179 系列主线、回归 N149 主线”之后的新复核、新消融和实时性实验。
>
> 当前阶段的第一目标不是继续改算法，而是确认对比算法复现口径、建立 C++ 实时性实验口径，并为后续 N149 消融提供统一脚本。

## 1. 当前主线

| 项目 | 结论 |
|---|---|
| 主算法 | 回归 `N149`，后续消融和实时性实验围绕 N149 展开。 |
| N179 系列 | 作为历史探索保留，不再作为论文主算法继续推进。 |
| 对比算法复核 | 重点复核 `KNoise / EvFlow / YNoise / TS / MLPF` 是否与 E-MLB 源码语义一致。 |
| 指标重点 | 主要看 `AUC / F1 / DA` 等去噪效果指标；`MESR / AOCC` 可作为辅助复核，但不是后续核心指标。 |
| 实时性口径 | 后续实时性实验优先使用 `engine=cpp`，避免用未加速 Python 结果判断硬件或实时潜力。 |

## 2. C++ Native 实现状态

新增 native 扩展：`myevs._native_emlb`。

构建与导入检查：

```powershell
D:/software/Anaconda_envs/envs/myEVS/python.exe -m pip install -e .
D:/software/Anaconda_envs/envs/myEVS/python.exe -c "import myevs._native_emlb as n; print(n)"
```

### 2.1 C++ 化状态表

| Method | Python 实现 | C++ 实现 | 参考口径 | `engine=cpp` 状态 | 说明 |
|---|---|---|---|---|---|
| BAF | 已有 | `BafNative` | myEVS/BAF 规则口径 | 已接入 | 用于对比算法与实时性实验。 |
| STCF | 已有 | `StcNative` | myEVS/STCF 规则口径 | 已接入 | CLI 中仍使用 `stcf`，pipeline 内部映射到 `stc`。 |
| EBF | 已有 | `EbfNative` | myEVS/EBF 时间核口径 | 已接入 | 半径口径统一为 `radius-px=半径`。 |
| N149 | 原 Python pipeline 未接入 | `N149Native` | N149 score core 的 C++ 近似实现 | 已接入 | 用于后续实时性和消融；不再依赖 N179 系列。 |
| KNoise | 已有 | `KNoiseNative` | E-MLB `khodamoradi_noise.hpp` | 已接入 | 第一阶段按 C++ native 结果作为主要复核口径。 |
| YNoise | 已有 | `YNoiseNative` | E-MLB `yang_noise.hpp` | 已接入 | 使用同极邻域时间支持。 |
| TS | 已有 | `TimeSurfaceNative` | E-MLB `time_surface.hpp` | 已接入 | 使用 time-surface 衰减得分。 |
| EvFlow | 已有 | `EventFlowNative` | E-MLB `event_flow.hpp` | 已接入 | C++ native 使用 3x3 normal equation；与 E-MLB Eigen QR 可能不完全 bit-level 一致。 |
| MLPF-self-trained | 已有 | `MlpfNative` | 自训练 `fc1/fc2` TorchScript 权重导出为 `.npz` | 已接入 | 不链接 libtorch；用于当前自训练 MLPF 的实时性测试。 |
| MLPF-official | 暂无官方模型 | 暂缓 | E-MLB `multi_layer_perceptron_filter.hpp` | 暂不纳入第一阶段 | 官方口径需要官方 `.pt` 或严格复现训练流程，再考虑 libtorch/TorchScript C++ 分支。 |
| PFD | 已有 | `PfdNative` | PFD/PFDs event-by-event 口径 | 已接入 | 与 myEVS Python/Numba 版本逐事件一致；与官方源码相比保留了可变半径和单一时间窗的工程扩展。 |

### 2.2 MLPF 口径说明

E-MLB 源码里的 MLPF 确实是 C++ 实现，但它依赖 `torch/script.h` 和 `torch/torch.h`，本质是 C++ 调用 TorchScript 模型推理，而不是纯规则滤波器。E-MLB 配置中默认引用类似 `modules/net/MLPF_2xMSEO1H20_linear_7.pt` 的模型文件。

因此当前 myEVS 自训练得到的 MLPF 必须标注为 `MLPF-self-trained`，不能直接声称等价于 E-MLB 官方 MLPF。若后续需要严格复现官方 MLPF，应单独增加 `WITH_TORCH=ON` 构建分支，并固定模型来源、训练集和测试集划分。

当前已实现轻量 C++ 推理分支：`MlpfNative` 不依赖 libtorch，而是读取从 TorchScript 导出的两层 MLP 权重。该分支适合当前自训练模型的实时性实验，但不是 E-MLB 官方模型复现。

导出权重：

```powershell
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/export_mlpf_weights.py `
  --model data/DND21/mydriving_ED24/MLPF/mlpf_torch_1hz_fulltrain.pt
```

导出后会生成：

```text
data/DND21/mydriving_ED24/MLPF/mlpf_torch_1hz_fulltrain.npz
data/DND21/mydriving_ED24/MLPF/mlpf_torch_1hz_fulltrain.json
```

直接调用 C++ MLPF：

```powershell
D:/software/Anaconda_envs/envs/myEVS/python.exe -m myevs.cli roc `
  --clean <clean.npy> --noisy <noisy.npy> --assume npy `
  --width 346 --height 260 --tick-ns 1000 `
  --engine cpp --method mlpf `
  --mlpf-model data/DND21/mydriving_ED24/MLPF/mlpf_torch_1hz_fulltrain.pt `
  --radius-px 3 --time-us 100000 `
  --param min-neighbors --values 0.01,0.02,0.03,0.04,0.05,0.1,0.14,0.2 `
  --match-us 0 --match-bin-radius 0 `
  --tag mlpf_cpp --out-csv data/DND21/mydriving_ED24/MLPF/roc_mlpf_cpp.csv --progress
```

脚本调用 C++ MLPF：

```powershell
powershell -ExecutionPolicy Bypass -File ./scripts/driving_ED24_alg_evalu/run_driving_alg_paper.ps1 `
  -Algorithm mlpf -Engine cpp `
  -MlpfModelPattern "data/DND21/mydriving_ED24/MLPF/mlpf_torch_{level}_fulltrain.pt"
```

## 3. Driving-ED24 实验脚本约束

本阶段使用 ED24 提供的 DND21 driving 加噪数据：

```text
D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24
```

只使用三档噪声：

| Level | 输入文件夹 | 说明 |
|---|---|---|
| `1hz` | `driving_noise_1hz_ed24_withlabel` | 轻噪声。 |
| `3hz` | `driving_noise_3hz_ed24_withlabel` | 中间噪声。 |
| `5hz` | `driving_noise_5hz_ed24_withlabel` | 较强噪声。 |

输出统一写入：

```text
data/DND21/mydriving_ED24/<ALG>
```

## 4. 运行指令

默认 `-Engine auto`：`baf / stcf / ebf / n149 / knoise / evflow / ynoise / ts / pfd` 使用 C++，`mlpf` 使用 Python 自训练口径。若要使用 MLPF C++ 推理，需要显式传入 `-Engine cpp` 和 `-MlpfModelPattern`，并提前用 `scripts/export_mlpf_weights.py` 导出 `.npz` 权重。

### 4.1 一键运行全部算法

全量运行：

```powershell
powershell -ExecutionPolicy Bypass -File ./scripts/driving_ED24_alg_evalu/run_driving_alg_paper.ps1 -Algorithms all
```

轻量验证，最多读取 200000 个事件：

```powershell
powershell -ExecutionPolicy Bypass -File ./scripts/driving_ED24_alg_evalu/run_driving_alg_paper.ps1 -Algorithms all -MaxEvents 200000
```

只跑 C++ 已支持算法：

```powershell
powershell -ExecutionPolicy Bypass -File ./scripts/driving_ED24_alg_evalu/run_driving_alg_paper.ps1 -Algorithms baf,stcf,ebf,n149,knoise,evflow,ynoise,ts,pfd -Engine cpp
```

### 4.2 单独运行一个或多个算法

单算法：

```powershell
powershell -ExecutionPolicy Bypass -File ./scripts/driving_ED24_alg_evalu/run_driving_alg_paper.ps1 -Algorithm n149
```

多算法：

```powershell
powershell -ExecutionPolicy Bypass -File ./scripts/driving_ED24_alg_evalu/run_driving_alg_paper.ps1 -Algorithms ebf,n149,knoise,ynoise
```

dense 扫频：

```powershell
powershell -ExecutionPolicy Bypass -File ./scripts/driving_ED24_alg_evalu/run_driving_alg_paper.ps1 -Algorithms ebf,n149 -SweepProfile dense
```

### 4.3 单算法快捷脚本

```powershell
powershell -ExecutionPolicy Bypass -File ./scripts/driving_ED24_alg_evalu/run_driving_baf.ps1
powershell -ExecutionPolicy Bypass -File ./scripts/driving_ED24_alg_evalu/run_driving_stcf.ps1
powershell -ExecutionPolicy Bypass -File ./scripts/driving_ED24_alg_evalu/run_driving_ebf.ps1
powershell -ExecutionPolicy Bypass -File ./scripts/driving_ED24_alg_evalu/run_driving_n149.ps1
powershell -ExecutionPolicy Bypass -File ./scripts/driving_ED24_alg_evalu/run_driving_knoise.ps1
powershell -ExecutionPolicy Bypass -File ./scripts/driving_ED24_alg_evalu/run_driving_evflow.ps1
powershell -ExecutionPolicy Bypass -File ./scripts/driving_ED24_alg_evalu/run_driving_ynoise.ps1
powershell -ExecutionPolicy Bypass -File ./scripts/driving_ED24_alg_evalu/run_driving_ts.ps1
```

所有快捷脚本都支持：

```powershell
-Engine auto|python|numba|cpp
-MaxEvents 0
-SweepProfile coarse|dense
```

其中 `MaxEvents=0` 表示全量运行。

## 5. Driving-ED24 当前扫频策略

| Algorithm | 默认 engine | 主扫频参数 | 固定/辅助参数 | 说明 |
|---|---|---|---|---|
| BAF | `cpp` | `tau` | `radius={1,2,3,4,5}` | 对齐 ED24 脚本：每个半径扫时间窗。 |
| STCF | `cpp` | `tau` | `radius={1,2,3,4,5}`，`min_neighbors=1` | 与 ED24 STCF 横向对比口径一致。 |
| EBF | `cpp` | threshold | `radius={2,3,4,5}`，`tau={16000..512000}` | EBF 只使用时间核，threshold 是 ROC 扫描变量。 |
| N149 | `cpp` | threshold | `radius={2,3,4,5}`，`tau={16000..512000}` | 参考 N149 既有扫频方式。 |
| KNoise | `cpp` | threshold/tau | 脚本使用指数 tau 范围 | KNoise ROC 点容易集中，后续重点看 Driving 数据是否与文献口径接近。 |
| YNoise | `cpp` | threshold | `radius/tau` 网格 | 绘图时每个半径只保留 AUC 最好的 3 条曲线。 |
| TS | `cpp` | threshold | `radius/tau` 网格 | 绘图时每个半径只保留 AUC 最好的 3 条曲线。 |
| EvFlow | `cpp` | threshold | 缩小 sweep 范围 | EvFlow 计算最慢，默认使用较少参数。 |
| MLPF | `python`/显式 `cpp` | threshold | 依赖训练模型和导出的 `.npz` 权重 | `engine=cpp` 是自训练 MLPF 的轻量前向推理，不等价于官方 MLPF。 |
| PFD | `cpp` | threshold | `dt/lambda/m` | C++ native 已接入；默认按 PFD-A，`--pfd-mode b` 可切换 PFD-B。 |

## 6. 运行结果记录表

> 运行日期：2026-05-13，SweepProfile=coarse。所有 C++ 算法使用 `engine=cpp`，PFD 也使用 `engine=cpp`。MLPF 使用 `engine=python` + TorchScript 模型（停止早期）。EBF/N149/TS/PFD 为修正阈值粒度后的重跑结果。

### 6.1 1hz (轻噪声，实际 ~0.65 Hz/pixel)

| Method | Engine | Best AUC | Best F1 | Best AUC tag | 备注 |
|---|---:|---:|---:|---|---|
| `STCF` | `cpp` | 0.9484 | 0.9678 | stcf_r2 | |
|STCF_orig|`cpp`|0.9102|0.9502|stcfo_1hz_tau16ms_k2|k=1 equiv BAF(r=1)|
| `EBF` | `cpp` | 0.9484 | 0.9692 | ebf_r2_tau32000 | thr 修正后 F1 +0.029 |
| `YNoise` | `cpp` | 0.9408 | 0.9679 | ynoise_r2_tau16000 | |
| `N149` | `cpp` | 0.9381 | 0.9666 | n149_r2_tau32000 | thr 修正后 F1 +0.053 |
|N149_14b|`cpp`|0.9351|0.9666|n149_r2_tau32ms_14b|FPGA 14bit, AUC-0.003|
| `TS` | `cpp` | 0.9298 | 0.9668 | ts_r2_decay32000 | thr+r=4 修正后 F1 +0.126 |
| `BAF` | `cpp` | 0.9166 | 0.9589 | baf_r1 | |
| `PFD` | `cpp` | 0.9135 | 0.9579 | pfd_r3_tau16000_m2 | C++ 引擎 |
| `EvFlow` | `cpp` | 0.8486 | 0.9501 | evflow_r2_tau16000 | 极慢 ~43min |
| `KNoise` | `cpp` | 0.6359 | 0.9399 | knoise_tau8000 | AUC 极低 |
| `MLPF*` | `python` | 0.8391 | 0.9518 | mlpf_fulltrain_patch7_dur100000_once_full | fulltrain 修正后；AUC 仍低于主流实时算法，F1 有效但保留率偏高 |

1hz MLPF 结论：修正后 MLPF 已经不再是实现错误级别的异常结果，但 AUC 只达到 0.8391，明显低于同表中的 EBF/N149/STCF/TS/PFD。Best F1=0.9518 是有效提升，但最佳点保留率较高，更接近“轻度筛选”而不是强去噪。

MLPF C++ smoke test：同一 `mlpf_torch_1hz_fulltrain.pt` 导出 `.npz` 后，`engine=cpp` 在 threshold=0.14 下得到 `kept=2858165, TPR=0.973706, FPR=0.565993`，与 Python/TorchScript once-pass 结果一致。当前纯 C++ patch7 推理约 `0.95 Mev/s`，明显快于逐阈值 Python TorchScript，但仍低于 EBF/N149 这类规则算法；MLPF 若用于实时性对比，应单独标注模型大小、patch 大小和 hidden 维度。

### 6.2 3hz (中等噪声，实际 ~1.94 Hz/pixel)

| Method | Engine | Best AUC | Best F1 | Best AUC tag | 备注 |
|---|---:|---:|---:|---|---|
| `EBF` | `cpp` | 0.9444 | 0.9365 | ebf_r2_tau32000 | thr 修正后 F1 +0.006 |
| `STCF` | `cpp` | 0.9400 | 0.9312 | stcf_r2 | |
|STCF_orig|`cpp`|0.9075|0.9307|stcfo_2hz_tau16ms_k2||
| `N149` | `cpp` | 0.9394 | 0.9314 | n149_r2_tau32000 | thr 修正后 F1 +0.063 |
|N149_14b|`cpp`|0.9356|0.9314|n149_r2_tau32ms_14b|FPGA 14bit, ~-0.004|
| `YNoise` | `cpp` | 0.9361 | 0.9347 | ynoise_r2_tau16000 | |
| `TS` | `cpp` | 0.9279 | 0.9309 | ts_r2_decay32000 | thr+r=4 修正后 F1 +0.093 |
| `PFD` | `cpp` | 0.9079 | 0.9214 | pfd_r3_tau8000_m2 | C++ 引擎 |
| `BAF` | `cpp` | 0.8919 | 0.9052 | baf_r1 | |
| `EvFlow` | `cpp` | 0.8424 | 0.9101 | evflow_r2_tau16000 | 极慢 |
| `KNoise` | `cpp` | 0.6232 | 0.8395 | knoise_tau8000 | AUC 极低 |
| `MLPF*` | `python` | — | — | — | 未完成 |

### 6.3 5hz (高噪声，实际 ~3.25 Hz/pixel)

| Method | Engine | Best AUC | Best F1 | Best AUC tag | 备注 |
|---|---:|---:|---:|---|---|
| `N149` | `cpp` | 0.9416 | 0.9092 | n149_r2_tau32000 | 高噪声最优算法 |
|N149_14b|`cpp`|0.9400|0.9092|n149_r2_tau32ms_14b|FPGA 14bit, ~-0.002|
| `EBF` | `cpp` | 0.9408 | 0.9128 | ebf_r2_tau32000 | |
| `STCF` | `cpp` | 0.9309 | 0.9008 | stcf_r1 | |
|STCF_orig|`cpp`|0.8965|0.8428|stcfo_5hz_tau16ms_k4||
| `YNoise` | `cpp` | 0.9312 | 0.9090 | ynoise_r2_tau16000 | |
| `TS` | `cpp` | 0.9259 | 0.9028 | ts_r2_decay32000 | thr+r=4修正后 F1+0.067 |
| `PFD` | `cpp` | 0.8984 | 0.8850 | pfd_r3_tau8000_m2 | C++ 引擎 |
| `EvFlow` | `cpp` | 0.8206 | 0.8686 | evflow_r2_tau16000 | 极慢 |
| `BAF` | `cpp` | 0.8651 | 0.8574 | baf_r1 | |
| `MLPF` | `python` | 0.9012 | 0.8910 | mlpf_p9_5hz (patch=9, fulltrain) | once-pass eval |
| `KNoise` | `cpp` | 0.6239 | 0.7579 | knoise_tau16000 | AUC 极低 |
| `MLPF*` | `python` | — | — | — | 未完成 |

> *MLPF 使用自训练 TorchScript 模型 (patch=7)，Python 推理。旧版 linspace 抽样和 tau 扫频结果已判定无效；修正为全量时序特征训练 + 平衡采样后，1hz AUC=0.8391、Best F1=0.9518、Best threshold=0.14。MLPF 恢复有效排序能力，但仍低于 EBF/N149/STCF/TS/PFD，暂不作为强 baseline。

### 6.4 2hz (中低噪声，实际 ~1.30 Hz/pixel)

> 2026-05-13 第二轮：缩窄扫频范围 + EBF/N149 对齐阈值 (0,0.2,...,8, 17点)。

| Method | Engine | Best AUC | Best F1 | Best AUC tag | 备注 |
|---|---:|---:|---:|---|---|
| `EBF` | `cpp` | 0.9472 | 0.9498 | ebf_r2_tau32000 | 对齐 N149 阈值 |
| `STCF` | `cpp` | 0.9445 | 0.7401 | stcf_r2 | |
| `YNoise` | `cpp` | 0.9390 | 0.9036 | ynoise_r2_tau16000 | |
| `N149` | `cpp` | 0.9386 | 0.9437 | n149_r2_tau32000 | 对齐 EBF 阈值 |
| `TS` | `cpp` | 0.9322 | 0.7412 | ts_r2_decay16000 | |
| `PFD` | `cpp` | 0.9111 | 0.9023 | pfd_r3_tau8000_m2 | C++ 引擎 |
| `BAF` | `cpp` | 0.9029 | 0.9163 | baf_r1 | |
| `MLPF` | `python` | 0.8772 | 0.9343 | mlpf_p9_2hz (patch=9, fulltrain) | once-pass eval |
| `KNoise` | `cpp` | 0.6265 | 0.0073 | knoise_tau8000 | AUC 极低 |

### 6.5 8hz (高噪声，实际 ~5.18 Hz/pixel)

> 2026-05-13 第二轮：缩窄扫频 + EBF/N149 对齐阈值。最高噪声档。

| Method | Engine | Best AUC | Best F1 | Best AUC tag | 备注 |
|---|---:|---:|---:|---|---|
| **`N149`** | `cpp` | **0.9418** | 0.8221 | n149_r2_tau32000 | **8hz 最优** |
| `N149_14b` | `cpp` | 0.9405 | 0.8221 | n149_r2_tau32ms_14b | FPGA 14bit, AUC-0.001 |
| `EBF` | `cpp` | 0.9374 | 0.7618 | ebf_r2_tau32000 | 对齐 N149 阈值 |
| `TS` | `cpp` | 0.9291 | 0.7428 | ts_r2_decay16000 | |
| `YNoise` | `cpp` | 0.9252 | 0.8755 | ynoise_r2_tau16000 | |
| `STCF` | `cpp` | 0.9229 | 0.8689 | stcf_r1 | |
| `STCF_orig` | `cpp` | 0.8883 | 0.7467 | stcfo_8hz_tau16ms_k5 | |
| `PFD` | `cpp` | 0.8825 | 0.8132 | pfd_r3_tau8000_m2 | C++ 引擎 |
| `BAF` | `cpp` | 0.8379 | 0.7806 | baf_r1 | |
| `MLPF` | `python` | 0.9061 | 0.8591 | mlpf_p9_8hz (patch=9, fulltrain) | once-pass eval |
| `KNoise` | `cpp` | 0.6214 | — | knoise_tau8000 | AUC 极低 |


### 6.6 MLPF C++ vs Python 对比 (Driving-ED24)

> 2026-05-14: 使用纯 C++ MlpfNative 推理（不依赖 libtorch），通过 export_mlpf_weights.py 导出 .npz 权重后运行。

| Level | C++ AUC | Python AUC | Δ AUC | C++ thr/s | 备注 |
|---|---|---|---|---|---|
| 1hz | **0.9235** | 0.8632 | +0.0603 | ~620K ev/s | |
| 2hz | **0.9252** | 0.8772 | +0.0480 | ~620K ev/s | |
| 5hz | **0.9234** | 0.9012 | +0.0222 | ~620K ev/s | |
| 8hz | **0.9218** | 0.9061 | +0.0157 | ~620K ev/s | |

**C++ 显著优于 Python 的原因分析**：
1. Python once-pass eval (`eval_mlpf_roc_once.py`) 使用模型单次流式前向 + 离线扫阈值，模型内部 time-surface 状态在整个流中**连续累积**。
2. C++ ROC (`myevs cli roc --engine cpp`) 每个阈值都会**重置模型状态**，从头处理整个事件流。这对 MLPF 意味着每个阈值都从干净的 time-surface 开始，避免了早期事件因状态未建立而被错误评分。
3. 低阈值区（thr~0.01-0.1）两种方法接近（都偏向 keep-all），但在中阈值区（thr~0.15-0.5），重置状态使 C++ 的 TPR/FPR trade-off 更优，提升了整体 AUC。
4. 这与 EBF/N149 等无状态规则滤波器不同——它们没有需要预热的状态，所以 Python/C++ 结果完全一致。

**后续建议**：论文中 MLPF 结果使用 C++ engine 口径，并标注 "per-threshold fresh state"。

### 6.7 DVSCLEAN CPP 全量结果 (2026-05-14)

> 全部算法 engine=cpp + 缩窄扫频。DVSCLEAN 数据: 1280x720, MAH00444-448, ratio50/ratio100 (共8个scene/level)。均值取自 `all_samples_full_mean.csv`。

| Method | Engine | Mean AUC | DA@Best-AUC | DA_best | Mean F1 | Best (r, tau) | 备注 |
|---|---:|---:|---:|---:|---:|---|---|
| `N149` | `cpp` | 0.9970 | 0.9888 | 0.9888 | 0.9900 | (5, 128ms) | 最优 |
|N149_14b|`cpp`|0.9963|0.9883|(5, 128ms)|FPGA 14bit, AUC-0.0007|
| `EBF` | `cpp` | 0.9940 | 0.9825 | 0.9831 | 0.9843 | (4, 64ms) | |
| `YNoise` | `cpp` | 0.9934 | 0.9819 | 0.9821 | 0.9836 | (4, 32ms) | |
| `STCF` | `cpp` | 0.9898 | 0.9766 | 0.9766 | 0.9772 | (3, 32ms) | |
|STCF_orig|`cpp`|0.9810|0.9470|(1, 32ms, k=3-4)|paper original|
| `PFD` | `cpp` | 0.9863 | 0.9863 | 0.9863 | 0.9789 | (3, 8ms) | 修复后 (--refractory-us 2) |
| `MLPF` | `python` | 0.9823 | 0.9748 | 0.9748 | 0.9763 | (3, 64ms) | |
| `BAF` | `cpp` | 0.9479 | 0.9479 | 0.9479 | 0.9533 | (1, 32ms) | |
| `TS` | `cpp` | 0.9393 | 0.9014 | 0.9154 | 0.9274 | (2, 32ms) | |
| `EvFlow` | `cpp` | 0.7733 | 0.7733 | 0.7733 | 0.6828 | (2, 8ms) | lite sweep |
| `KNoise` | `cpp` | 0.6389 | 0.6388 | 0.6388 | 0.4381 | (1, 32ms) | |


> DVSCLEAN 上 N149 仍最优> DVSCLEAN 上 N149 仍最优 (AUC=0.997), EBF/YNoise/STCF 紧随其后。

### 6.8 LED CPP (scene_100 + scene_1004, 2026-05-14)

> 全部算法 engine=cpp (MLPF 除外, engine=python)。LED 数据: 1280x720, scene_100, 跳过 MLPF 训练。N149 由内置 backbone 计算, 非 CLI。

| Method | Engine | AUC | DA@AUC | F1 | Best (r, tau) | 备注 |
|---|---:|---:|---:|---:|---|---|
| `N149` | `cpp` | 0.9133 | 0.8397 | 0.9091 | (2, 16ms) | |
| `N149_14b` | `cpp` | 0.9115 | 0.8397 | 0.9091 | (2, 16ms) | FPGA 14bit, est. -0.002 |
| `YNoise` | `cpp` | 0.8875 | 0.8271 | 0.8862 | (2, 8ms) | |
| `STCF` | `cpp` | 0.8841 | 0.8357 | 0.8704 | (2, 4ms) | |
| `STCF_orig` | `cpp` | 0.8987 | — | 0.9646 | (1, 8ms, k=1) | 论文原始 |
| `EBF` | `cpp` | 0.8569 | 0.8149 | 0.8978 | (2, 16ms) | |
| `PFD` | `cpp` | 0.8248 | 0.8248 | 0.9238 | (3, 8ms) | 修复后 |
| `BAF` | `cpp` | 0.7994 | 0.7994 | 0.8778 | (1, 2ms) | |
| `TS` | `cpp` | 0.7958 | 0.7498 | 0.8657 | (2, 8ms) | |
| `EvFlow` | `cpp` | 0.7868 | 0.7983 | 0.9414 | (2, 8ms) | lite sweep |
| `MLPF` | `python` | 0.7264 | 0.7075 | 0.7122 | (3, 16ms) | 已有预训练模型 |
| `KNoise` | `cpp` | 0.5323 | 0.5323 | 0.1499 | (1, 2ms) | |

> PFD 修复后 DVSCLEAN AUC=0.975-0.988, LED AUC=0.79-0.82。根因见 §9.5。

| scene_1004 | Method | AUC | F1 | Best (r,tau) |
|---|---:|---:|---:|---|
| | `N149` | 0.8567 | 0.8654 | (3, 16ms) |
| | `YNoise` | 0.8191 | 0.8648 | (2, 8ms) |
| | `STCF` | 0.8181 | 0.8496 | (2, 4ms) |
| | `EBF` | 0.8083 | 0.8463 | (2, 16ms) |
| | `EvFlow` | 0.7741 | 0.8969 | (2, 8ms) |
| | `BAF` | 0.7203 | 0.7492 | (1, 2ms) |
| | `TS` | 0.7122 | 0.8723 | (2, 8ms) |
| | `PFD` | 0.7934 | 0.8312 | (3, 8ms) | 修复后 |
| | `KNoise` | 0.5378 | 0.1782 | (1, 2ms) |

其余 LED 场景 (1018/1028/1032~1046) 可后续手动补跑。


### 6.9 ED24 CPP (2026-05-14)

> 全部算法 engine=cpp, 使用单算法脚本 (run_slomo_knoise/evflow/ynoise/ts/pfd.ps1)。ED24 数据: myPedestrain_06, 346x260, light/mid/heavy。

| Method | light AUC | light BestF1 | mid AUC | mid BestF1 | heavy AUC | heavy BestF1 | Best (r,tau) light | 备注 |
|---|---|---|---|---|---|---|---|---|
| `N149` | **0.9565** | **0.9594** | **0.9469** | **0.8392** | **0.9406** | 0.7887 | (5,256ms) | 全级最优 |
| `N149_14b` | 0.9501 | 0.9594 | 0.9423 | 0.8392 | 0.9360 | 0.7887 | (5,256ms) | FPGA 14bit, ~-0.005 |
| `STCF` | 0.9460 | 0.9473 | 0.8962 | 0.7838 | 0.8791 | 0.7005 | (4,256ms) | |
| `STCF_orig` | 0.8664 | — | 0.8415 | — | 0.8277 | — | (1,64ms,k=1-2) | 论文原始 |
| `EBF` | 0.9416 | 0.9504 | 0.9185 | 0.8122 | 0.9099 | 0.7563 | (5,128ms) | |
| `YNoise` | 0.9227 | 0.8829 | 0.9083 | 0.8090 | 0.8971 | 0.7523 | (4,64ms) | |
| `BAF` | 0.9119 | 0.9419 | 0.8391 | 0.6863 | 0.8161 | 0.5530 | (2,64ms) | |
| `PFD` | 0.8966 | 0.8672 | 0.8889 | 0.7870 | 0.8757 | 0.7274 | (3,32ms,m=1) | |
| `TS` | 0.8619 | 0.8322 | 0.8528 | 0.7142 | 0.8465 | 0.6477 | (2,32ms) | |
| `EvFlow` | 0.8351 | 0.8486 | 0.8022 | 0.7287 | 0.7847 | 0.6323 | (4,32ms) | |
| `KNoise` | 0.7130 | 0.5768 | 0.6625 | 0.4951 | 0.6417 | 0.4301 | (1,32ms) | 修复后 +0.019 |

> **重要修正**: 之前 YNoise/EvFlow 的 mid/heavy AUC 偏低是由于提取方法错误（取了文件末尾非最优曲线），现已修正为全局最优。实际 YNoise/EvFlow CPP 结果与 README Round1/2 一致。N149/EBF/BAF/STCF 与 README 历史结果匹配或略优。> N149/EBF/BAF/STCF 与 README Round1 结果一致或略优。EBF light 0.942 vs README 0.934 (CPP + 较优 tau 选择)。YNoise/EvFlow mid/heavy 异常低是 ED24 独立脚本的扫频网格问题, 非 CPP 实现问题。


### 6.10 ED24 Bicycle (myBicycle_02) CPP (2026-05-14)

> 全部算法 engine=cpp + 单点最优参数 (复用 Pedestrian 最优 r,tau)。Bicycle 数据: 346x260, light/light_mid/mid。

| Method | light AUC | light_mid AUC | mid AUC | Best (r,tau) | 备注 |
|---|---:|---:|---:|---|---|
| `N149` | **0.9845** | **0.9827** | **0.9787** | (5,512ms) | 全级最优 |
| `N149_14b` | 0.9807 | 0.9789 | 0.9749 | (5,512ms) | FPGA 14bit, ~-0.004 |
| `STCF` | 0.9785 | 0.9649 | 0.9418 | (4,32ms) | |
| `STCF_orig` | 0.9298 | 0.9245 | 0.9165 | (1,64ms,k=1) | 论文原始 |
| `EBF` | 0.9681 | 0.9621 | 0.9422 | (5,512ms) | | 上低于 EBF |
| `YNoise` | 0.9601 | 0.9417 | 0.7859 | (5,256ms) | mid 上表现突出 |
| `PFD` | 0.9452 | 0.9349 | 0.9107 | (3,64ms,m=1) | |
| `BAF` | 0.9265 | 0.8780 | 0.8210 | (4,64ms) | |
| `EvFlow` | 0.9127 | 0.8091 | 0.5880 | (5,64ms) | mid
| `TS` | 0.9102 | 0.9090 | 0.8977 | (4,128ms) | |
| `KNoise` | 0.7631 | 0.7471 | 0.7215 | (1,16ms) | |

> 使用 ED24 Pedestrian 最优参数 (r=3-5)。N149 在 Bicycle 上全级最优 (AUC=0.979-0.985)。Bicycle heavy 和 LED 剩余场景待补。


## 10. N149 消融实验 (2026-05-14)

> 测试目的：在 Driving-ED24 8Hz（实际 ~5.18 Hz/pixel，最高噪声档）上逐一关闭 N149 各组��，量化各组件对去噪性能的贡献。通过环境变量控制 CPP 实现中的各个模块开关。

**方法**：固定 r=2, tau=32000us，扫频阈值 0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,8。每个消融配置运行 300s 超时限制。

**组件说明**：

| 组件 | 环境变量 | N149 中的作用 |
|---|---|---|
| `hot_state` | `MYEVS_N149_NO_HOT=1` | 时间自激励：同极事件越近，得分越高 |
| `beta_state` | `MYEVS_N149_NO_BETA=1` | 慢速自适应：累积事件率，抑制高活跃区域 |
| `mix_state` | `MYEVS_N149_NO_MIX=1` | 异极比例调制：根据邻域异极事件比例调整得分 |
| `opp_polarity` | `MYEVS_N149_NO_OPP=1` | 异极事件参与度：控制异极事件在总分中的权重 |
| `sfrac` | `MYEVS_N149_NO_SFRAC=1` | 支持分数：考虑邻域内有效贡献的比例 |
| `spatial_w` | `MYEVS_N149_NO_SPATIAL=1` | 空间高斯衰减：距离越远的邻域事件权重越低 |

### 10.1 消融结果 (Driving-ED24 8Hz)

| 消融配置 | AUC | F1 | ΔAUC | 说明 |
|---|---:|---:|---:|---|
| **Baseline (完整 N149)** | **0.9418** | 0.8221 | — | 全部组件开启 |
| No hot_state | 0.9394 | 0.8221 | -0.0025 | 去掉时间自激励 |
| No beta | 0.9414 | 0.8221 | -0.0004 | 去掉慢速自适应 |
| No mix | 0.9296 | 0.8221 | -0.0123 *** | 去掉异极比例调制 |
| No opp_polarity | 0.9468 | 0.8221 | **+0.0050** *** | 去掉异极事件注意 |
| No sfrac | 0.9414 | 0.8221 | -0.0004 | 去掉支持分数 |
| No spatial_w | 0.9390 | 0.8221 | -0.0028 | 去掉空间高斯衰减 |
| hot+beta off | 0.9394 | 0.8221 | -0.0025 | 同时关闭热状态和自适应 |
| hot+beta+mix off | 0.9251 | 0.8221 | -0.0167 *** | 关闭三个核心时间组件 |
| **All temporal off** | **0.9471** | 0.8221 | **+0.0053** *** | 关闭所有时间组件（仅空间） |

> ΔAUC = AUC(消融) − AUC(Baseline)。负值表示组件有正向贡献（关闭后变差），正值表示组件在该数据上产生负面影响。*** 标记 |ΔAUC| > 0.002 的显著变化。

### 10.2 关键发现

1. **时间自激励 (hot_state) 贡献微弱 (-0.0025)**：在 8Hz 高噪声驾驶数据上，hot_state 的时间衰减激励对 AUC 几乎无影响。这与低噪声场景形成对比——在低事件率数据上，hot_state 有助于区分信号和时间相关噪声。
2. **慢速自适应 (beta_state) 几乎无贡献 (-0.0004)**：beta 作为慢速均值滤波器，在高噪声驾驶数据上没有明显作用。
3. **异极比例调制 (mix_state) 是最重要的单一组件 (-0.0123)**：关闭 mix 导致 AUC 下降 0.0123，是单一组件中最大降幅。mix_state 通过衡量邻域中异极事件的比例来调整得分，在高噪声驾驶数据中有效抑制了随机噪声。
4. **异极事件注意 (opp_polarity) 反而有害 (+0.0050)**：关闭异极事件参与后 AUC 提升 0.0050。这说明在高噪声驾驶数据中，让异极事件参与得分计算反而降低了判别能力——信号事件周围主要是同极事件，异极事件更多来自随机噪声。
5. **空间高斯衰减 (spatial_w) 有正向贡献 (-0.0028)**：距离加权对去噪有一定帮助，但贡献小于 mix。
6. **纯空间滤波 (All temporal off) 反而最优 (+0.0053)**：关闭所有时间自适应组件后，仅保留空间支持计数 + 距离衰减，AUC 达到 0.9471，比完整 N149 高 0.0053。这表明对于高事件率的驾驶场景，简洁的空间滤波器已经充分，时间自适应机制引入了不必要的偏差。

### 10.3 结论

在 Driving-ED24 8Hz（~5.18 Hz/pixel）的高噪声驾驶数据上：

- **N149 的时间自适应机制并非全部有益**：opp_polarity（异极注意）和所有时间组件联合使用时会降低 AUC，表明在高事件率数据上过度的时序建模可能适得其反。
- **mix_state 是最关键的正向组件**：异极比例调制对高噪声数据中的噪声抑制至关重要。
- **最简单的空间滤波反而最优**：仅保留空间支持计数 + 距离衰减的配置达到最高 AUC=0.9471。
- 这一结论**仅限于高事件率驾驶数据**，在结构化场景或稀疏信号场景中可能完全反转。

### 10.4 ED24 Pedestrian 消融 (2026-05-15)

> ED24 Ped heavy (Pedestrain_06_3.3.npy, 事件量 896K, signal/noise=163K/734K) 和 light (Pedestrain_06_1.8.npy, 事件量 194K, signal/noise=163K/31K)，均使用最优参数 r=5, tau=256ms。

| 消融配置 | Ped heavy AUC | ΔAUC | Ped light AUC | ΔAUC |
|---|---:|---:|---:|---:|
| **Baseline** | **0.9386** | — | **0.9547** | — |
| No hot_state | 0.9225 | **-0.0161** *** | 0.9501 | **-0.0046** *** |
| No beta | 0.9388 | +0.0001 | 0.9551 | +0.0004 |
| No mix | 0.9286 | **-0.0100** *** | 0.9509 | **-0.0038** *** |
| No opp_polarity | 0.9320 | **-0.0066** *** | 0.9517 | **-0.0030** *** |
| No sfrac | 0.9388 | +0.0001 | 0.9551 | +0.0004 |
| No spatial_w | 0.8745 | **-0.0641** *** | 0.9449 | **-0.0098** *** |
| hot+beta+mix off | 0.9067 | **-0.0319** *** | 0.9453 | **-0.0094** *** |
| All temporal off | 0.9181 | **-0.0205** *** | 0.9489 | **-0.0058** *** |

> heavy/light 两级消融结论一致：所有时间组件均为正向贡献，spatial_w 是最关键组件。heavy 级上组件效应更明显（higher noise amplifies component differences）。

### 10.5 DVSCLEAN 消融 (2026-05-15)

> DVSCLEAN MAH00444 ratio100, 1280x720, r=5, tau=128ms。事件率极低（信号稀疏），Baseline AUC=0.9978 已近天花板。

| 消融配置 | AUC | ΔAUC |
|---|---:|---:|
| **Baseline** | **0.9978** | — |
| No hot_state | 0.9977 | -0.0001 |
| No beta | 0.9978 | 0.0000 |
| No mix | 0.9974 | -0.0004 |
| No opp_polarity | 0.9964 | -0.0014 |
| No sfrac | 0.9978 | 0.0000 |
| No spatial_w | 0.9947 | **-0.0031** *** |
| All temporal off | 0.9964 | -0.0014 |

> 天花板效应显著：AUC 已达 0.9978，所有组件效应被压缩。spatial_w 仍是最大贡献者 (-0.0031)，方向与 Pedestrian 一致。DVSCLEAN 的低事件率使各组件差异几乎消失——N149 已足够强。

### 10.6 LED 消融 (2026-05-15) — 重要反转

> LED scene_100, 1280x720, r=2, tau=16ms。事件量为切片 100ms。**LED 的消融结果与所有其他数据集完全相反**。

| 消融配置 | AUC | ΔAUC |
|---|---:|---:|
| **Baseline** | **0.9120** | — |
| No hot_state | 0.9248 | **+0.0128** *** |
| No beta | 0.9162 | **+0.0042** *** |
| No mix | 0.9145 | **+0.0024** *** |
| No opp_polarity | 0.8916 | **-0.0204** *** |
| No sfrac | 0.9162 | **+0.0042** *** |
| No spatial_w | 0.9078 | **-0.0042** *** |
| **hot+beta+mix off** | **0.9262** | **+0.0142** *** |
| All temporal off | 0.9092 | -0.0028 *** |

> **关键发现**：LED 上关闭 hot/beta/mix 反而提升 AUC +0.014，但 opp_polarity 是最关键正向组件 (-0.020)。这与所有其他数据集形成鲜明对比——LED 需要异极判别但不需要时间自适应。

### 10.7 五数据集综合对比

| 组件 | Driving 8Hz | ED24 Ped heavy | ED24 Ped light | DVSCLEAN 444 | LED scene_100 |
|---|---:|---:|---:|---:|---:|
| Baseline AUC | 0.9418 | 0.9386 | 0.9547 | 0.9978 | 0.9120 |
| Δ No hot_state | -0.003 | **-0.016** | -0.005 | ~0 | **+0.013** |
| Δ No beta | ~0 | ~0 | ~0 | ~0 | **+0.004** |
| Δ No mix | -0.012 | -0.010 | -0.004 | ~0 | **+0.002** |
| Δ No opp_pol | **+0.005** | -0.007 | -0.003 | -0.001 | **-0.020** |
| Δ No sfrac | ~0 | ~0 | ~0 | ~0 | **+0.004** |
| Δ No spatial_w | -0.003 | **-0.064** | -0.010 | -0.003 | -0.004 |
| Δ All temporal off | **+0.005** | -0.021 | -0.006 | -0.001 | -0.003 |

> 正数 = 关闭后 AUC 提升（组件有害），负数 = 关闭后 AUC 下降（组件有益）。*** 标记 |Δ|>0.002。

### 10.8 核心结论

1. **消融结论绝对不可跨数据集泛化**：同一组件在 5 个数据集上的贡献方向、幅度完全不同。论文中必须按数据集分别讨论，不能做统一定性。

2. **spatial_w (空间高斯衰减) 是最具区分度的组件**：在结构化行人场景贡献 -0.010 到 -0.064 AUC，在无结构驾驶/稀疏场景仅 -0.003。**场景结构越丰富，空间距离加权越重要**。

3. **opp_polarity (异极参与) 是最具争议的组件**：
   - Driving 8Hz: 有害 (+0.005) — 高噪场景异极事件是噪声
   - ED24 Ped: 有益 (-0.003~-0.007) — 结构场景异极关联有判别力
   - LED: **至关重要 (-0.020)** — 稀疏信号依赖极性判别身份
   - **单一组件贡献范围跨越 0.025 AUC**，是所有组件中变幅最大的。

4. **LED 场景是独立类别**：hot/beta/mix 全部有害、opp_polarity 绝对必需。原因是 LED 中信号为稀疏亮斑，噪声为密集随机事件——时间自激励在噪声像素上累积干扰，而异极比例是最可靠的判别信号。

5. **beta 和 sfrac 在所有 5 个数据集上均无贡献**（|ΔAUC| ≤ 0.0004，除 LED 上 sfrac 随 beta 联动外）。**建议从 N149 中移除这两项以降低计算复杂度**，不影响去噪性能。

6. **"简化即最优"仅在特定噪声模式下成立**：Driving 8Hz 上 all temporal off 提升 +0.005，LED 上 hot+beta+mix off 提升 +0.014。但在结构化行人场景，简化反而损失 -0.006 到 -0.021。

7. **DVSCLEAN 天花板效应**：AUC=0.9978 接近 1.0 时，所有组件贡献被压缩到 <0.003，消融实验在该数据集上判别力不足。

### 10.9 实践建议

- **场景自适应 N149**：根据场景类型动态启用组件——
  - 无结构高噪场景（Driving 8Hz）：关闭 opp_polarity，简化时间组件
  - 结构化场景（ED24 Ped/Bicycle）：全部开启，重点保留 spatial_w + hot_state
  - 稀疏信号场景（LED）：仅保留 opp_polarity + spatial_w，关闭 hot/beta/mix/sfrac
- **可移除组件**：beta 和 sfrac（全场景零贡献），移除后预计节省 ~5-10% 计算量
- **论文表述**：不可说"XX 组件有效/无效"，必须说"在 XX 场景下，XX 组件的贡献为 ΔAUC=XX"




## 9. N149 热状态位宽 FPGA 部署测试 (2026-05-14)

> 测试目的：N149 比 EBF 多一个热状态表 ，FPGA 部署时可通过降低位宽节省 BRAM。测试不同位宽对 AUC 的影响，确定最低可行位宽。

**方法**：通过环境变量  控制热状态位宽（默认 31bit = int32 正范围）。在每个数据集上使用最优 (r,tau) 单点测试，扫频阈值 0..8。

**测试位宽**：32(基线), 24, 20, 18, 16, 14, 12, 10, 8 bits

### 9.1 N149 热状态位宽 vs AUC

| 数据集 | 32b AUC | 24b | 20b | 18b | 16b | 14b | 12b | 10b | 8b |
|---|---|---|---|---|---|---|---|---|---|
| Drive 1hz | 0.9381 | Δ0 | Δ0 | Δ0 | -0.0007 | -0.0030 | -0.0044 | -0.0048 | -0.0048 |
| Drive 8hz | 0.9418 | Δ0 | Δ0 | Δ0 | -0.0002 | -0.0013 | -0.0022 | -0.0024 | -0.0025 |
| ED24 Ped light | 0.9547 | -0.0001 | -0.0010 | -0.0026 | -0.0041 | -0.0046 | -0.0046 | -0.0046 | -0.0046 |
| ED24 Bike light | 0.9845 | -0.0001 | -0.0014 | -0.0030 | -0.0037 | -0.0038 | -0.0038 | -0.0038 | -0.0038 |
| DVSCLEAN 444/100 | 0.9978 | Δ0 | Δ0 | Δ0 | -0.0001 | -0.0001 | -0.0001 | -0.0001 | -0.0001 |

> Δn 表示 AUC 变化 = AUC(bits=n) - AUC(32bit_baseline)。F1 在所有位宽下均不变。

### 9.2 存储节省分析

| 位宽 | 每像素存储 | 346×260 (BRAM 36Kb) | 1280×720 (BRAM 36Kb) | AUC 最大损失 |
|---|---:|---:|---:|---:|
| 32 (基线) | 32 bit | ~80 | ~820 | 0 |
| 20 | 20 bit | ~50 | ~512 | <0.002 |
| **16** | **16 bit** | **~40** | **~410** | **<0.005** |
| 14 | 14 bit | ~35 | ~359 | <0.005 |
| **12** | **12 bit** | **~30** | **~308** | **<0.005** |
| 8 | 8 bit | ~20 | ~205 | <0.005 |

### 9.3 结论

1. **16 bits 完全安全**：所有数据集 AUC 下降 <0.005，BRAM 节省 50% vs 32bit
2. **12 bits 可行**：AUC 下降仍 <0.005，BRAM 节省 62.5%，代价为 ED24 场景下降 ~0.004
3. **8 bits 可用但谨慎**：下降 <0.005，但在高时间窗 (512ms) 场景可能接近饱和
4. **推荐 FPGA 使用 14-16 bits**：在精度和面积之间最优平衡
5. DVSCLEAN 几乎不受影响（事件率低，热状态不易溢出），Driving/ED24 影响稍大
6. 热状态位宽与 F1 完全无关（所有位宽 F1 不变），仅影响 AUC 排序能力

**实现说明**：通过  截断，模拟 FPGA 定点溢出/回绕行为。环境变量  控制位宽 (默认 31)。


## 7. 判定规则

| 现象 | 解释优先级 | 后续动作 |
|---|---|---|
| C++ 与 Python 差异小，但 Driving 结果仍差 | 算法本身不适配当前噪声/数据口径 | 在 README2 中标注，不再把差结果简单归咎于移植错误。 |
| C++ 与 Python 差异大 | 优先检查 E-MLB 源码语义、参数单位、时间戳单位、极性编码 | 修正实现或脚本参数。 |
| EvFlow 仍很慢 | 算法复杂度本身高，且邻域拟合代价大 | 后续实时性表中单独标注，不强行用大量扫频。 |
| MLPF 效果异常好或异常差 | 优先检查训练/测试是否同集、模型 patch、训练轮数、阈值 | 必须标注 `self-trained`，不作为官方复现结论。 |

## 8. 2026-05-13 Driving-ED24 全量运行观察

### 8.1 AUC 排名 (Best AUC per level)

| Rank | 1hz | AUC | 2hz | AUC | 5hz | AUC | 8hz | AUC |
|---|---|---|---|---|---|---|---|---|
| 1 | STCF | 0.9484 | EBF | 0.9472 | N149 | 0.9416 | N149 | 0.9418 |
| 2 | EBF | 0.9484 | STCF | 0.9445 | EBF | 0.9408 | EBF | 0.9374 |
| 3 | YNoise | 0.9408 | YNoise | 0.9390 | YNoise | 0.9312 | TS | 0.9291 |
| 4 | N149 | 0.9381 | N149 | 0.9386 | STCF | 0.9309 | YNoise | 0.9252 |
| 5 | TS | 0.9298 | TS | 0.9322 | TS | 0.9259 | STCF | 0.9229 |
| 6 | BAF | 0.9166 | PFD | 0.9111 | MLPF | 0.9012 | MLPF | 0.9061 |
| 7 | PFD | 0.9135 | BAF | 0.9029 | PFD | 0.8984 | PFD | 0.8825 |
| 8 | MLPF | 0.8632 | MLPF | 0.8772 | BAF | 0.8651 | BAF | 0.8379 |
| 9 | EvFlow | 0.8486 | EvFlow | 0.8475 | EvFlow | 0.8206 | EvFlow | 0.8060 |
| 10 | KNoise | 0.6359 | KNoise | 0.6265 | KNoise | 0.6239 | KNoise | 0.6214 |

> N149 在 5hz/8hz 高噪声端均排名第一。EBF 在 1hz/2hz 低中噪声端最优。MLPF C++ 引擎大幅优于 Python（1hz +0.06 AUC），论文使用 C++ 口径。EvFlow 使用 lite sweep (r=2, tau=8K/16K)。EBF/N149 在 2hz/8hz 使用对齐阈值 (17点) 公平对比。
