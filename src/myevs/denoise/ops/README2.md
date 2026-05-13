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
| MLPF-official | 自训练 Python 版 | 暂缓 | E-MLB `multi_layer_perceptron_filter.hpp` | 暂不纳入第一阶段 | E-MLB MLPF 是 C++/libtorch/TorchScript 推理，需要官方或明确训练得到的 `.pt` 模型。 |
| PFD | 已有 | 暂缓 | PFD 论文口径 | 仍使用 `numba` | 不是本阶段 E-MLB C++ 化重点。 |

### 2.2 MLPF 口径说明

E-MLB 源码里的 MLPF 确实是 C++ 实现，但它依赖 `torch/script.h` 和 `torch/torch.h`，本质是 C++ 调用 TorchScript 模型推理，而不是纯规则滤波器。E-MLB 配置中默认引用类似 `modules/net/MLPF_2xMSEO1H20_linear_7.pt` 的模型文件。

因此当前 myEVS 自训练得到的 MLPF 必须标注为 `MLPF-self-trained`，不能直接声称等价于 E-MLB 官方 MLPF。若后续需要严格复现官方 MLPF，应单独增加 `WITH_TORCH=ON` 构建分支，并固定模型来源、训练集和测试集划分。

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

默认 `-Engine auto`：`baf / stcf / ebf / n149 / knoise / evflow / ynoise / ts` 使用 C++，`pfd` 使用 numba，`mlpf` 使用 Python 自训练口径。

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
powershell -ExecutionPolicy Bypass -File ./scripts/driving_ED24_alg_evalu/run_driving_alg_paper.ps1 -Algorithms baf,stcf,ebf,n149,knoise,evflow,ynoise,ts -Engine cpp
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
| MLPF | `python` | threshold | 依赖训练模型 | 当前不作为 E-MLB 官方 MLPF 复现结论。 |
| PFD | `numba` | threshold | `dt/lambda/m` | 非 E-MLB 第一阶段重点。 |

## 6. 运行结果记录表

实验跑完后，把结果填入本表，并同步保留 CSV 原始文件。

| Level | Method | Engine | Best AUC | Best F1 | Best tag | Runtime(s) | 备注 |
|---|---|---|---:|---:|---|---:|---|
| `1hz` | `BAF` | `cpp` |  |  |  |  |  |
| `3hz` | `BAF` | `cpp` |  |  |  |  |  |
| `5hz` | `BAF` | `cpp` |  |  |  |  |  |
| `1hz` | `STCF` | `cpp` |  |  |  |  |  |
| `3hz` | `STCF` | `cpp` |  |  |  |  |  |
| `5hz` | `STCF` | `cpp` |  |  |  |  |  |
| `1hz` | `EBF` | `cpp` |  |  |  |  |  |
| `3hz` | `EBF` | `cpp` |  |  |  |  |  |
| `5hz` | `EBF` | `cpp` |  |  |  |  |  |
| `1hz` | `N149` | `cpp` |  |  |  |  |  |
| `3hz` | `N149` | `cpp` |  |  |  |  |  |
| `5hz` | `N149` | `cpp` |  |  |  |  |  |
| `1hz` | `KNoise` | `cpp` |  |  |  |  |  |
| `3hz` | `KNoise` | `cpp` |  |  |  |  |  |
| `5hz` | `KNoise` | `cpp` |  |  |  |  |  |
| `1hz` | `YNoise` | `cpp` |  |  |  |  |  |
| `3hz` | `YNoise` | `cpp` |  |  |  |  |  |
| `5hz` | `YNoise` | `cpp` |  |  |  |  |  |
| `1hz` | `TS` | `cpp` |  |  |  |  |  |
| `3hz` | `TS` | `cpp` |  |  |  |  |  |
| `5hz` | `TS` | `cpp` |  |  |  |  |  |
| `1hz` | `EvFlow` | `cpp` |  |  |  |  |  |
| `3hz` | `EvFlow` | `cpp` |  |  |  |  |  |
| `5hz` | `EvFlow` | `cpp` |  |  |  |  |  |

## 7. 判定规则

| 现象 | 解释优先级 | 后续动作 |
|---|---|---|
| C++ 与 Python 差异小，但 Driving 结果仍差 | 算法本身不适配当前噪声/数据口径 | 在 README2 中标注，不再把差结果简单归咎于移植错误。 |
| C++ 与 Python 差异大 | 优先检查 E-MLB 源码语义、参数单位、时间戳单位、极性编码 | 修正实现或脚本参数。 |
| EvFlow 仍很慢 | 算法复杂度本身高，且邻域拟合代价大 | 后续实时性表中单独标注，不强行用大量扫频。 |
| MLPF 效果异常好或异常差 | 优先检查训练/测试是否同集、模型 patch、训练轮数、阈值 | 必须标注 `self-trained`，不作为官方复现结论。 |

## 8. 后续占位

### 8.1 N149 消融实验

待 Driving-ED24 复核完成后展开。目标是围绕 `N149` 做可解释、可复现的消融，而不是继续推进 `N179` 系列。

### 8.2 N149 与对比算法实时性实验

待 C++ native 口径稳定后展开。实时性表必须统一记录：

```text
events_in / runtime_sec / Mev/s / dataset_event_rate_Mev/s
```

不要用未加速 Python 结果直接判断硬件实时性。
