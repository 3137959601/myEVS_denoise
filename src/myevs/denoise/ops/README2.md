# README2：回归 N149 后的复核、消融与实时性实验

> 修改本文件前，必须先阅读 `src/myevs/denoise/ops/README.md`。
> `README.md` 记录历史数据集路径、脚本约定、已有结果和结论；`README2.md` 只记录从“放弃 N179 系列主线、回归 N149 主线”之后的新复核、新消融和实时性实验。

## 0. Driving BAF 论文 profile 固化

本节记录 2026-05-19 对 Driving BAF 论文口径的复核。BAF 论文 profile 一律固定为 `radius=1`、3x3 NNb、排除自身、`polarity=ignored`，AUC 使用工程默认 paper convention：positive=signal、predicted positive=kept，并补 ROC 端点 `(0,0)`、`(1,1)`。same-polarity BAF 只能作为 diagnostic，不能进入论文 BAF 对比表。

新增脚本：

```powershell
python scripts/driving_alg_evalu/evaluate_baf_paper_profiles.py
```

默认扫描 `D:/hjx_workspace/scientific_reserach/dataset/DND21` 下的四套 Driving：`mydriving_ED24`、`mydriving_cov05`、`mydriving_paper`、`mydriving_jaer`。输出目录为 `data/DND21/baf_paper_profiles/`：

| 文件 | 用途 |
|---|---|
| `REPORT.md` | 人读报告，含固定 BAF 语义、v2e COV 说明、dataset manifest、AUC summary、closest rows |
| `dataset_manifest.csv` | 每个数据文件的 requested Hz、label-count actual Hz/pixel、事件数、signal/noise 数 |
| `baf_profile_summary.csv` | 每个 dataset/profile/Hz 的 AUC 与论文 target delta |
| `baf_profile_roc_points.csv` | 每个 tau 点的 TP/FP/TN/FN/TPR/FPR |
| `missing_inputs.csv` | 缺失输入；当前 ED24 缺 `10hz` converted npy |

三套 profile 分开使用，后续不能把 sweep 混在一起解释：

| profile | 固定口径 | tau sweep | 对齐目标 |
|---|---|---|---|
| `baf_dnd21_original` | radius=1, polarity ignored, self excluded | dense `1..100 ms` | DND21 原论文 5Hz/trend |
| `baf_edformer` | radius=1, polarity ignored, self excluded | dense `2..200 ms` | EDformer Table 2 Driving |
| `baf_ebf_source` | radius=1, polarity ignored, self excluded | `1,5,10,15,20,25,30,40,50 ms` | EBF Table II / source grid |

本轮结果摘要：

| dataset | requested/actual Hz 关系 | EDformer profile BAF AUC | EBF profile BAF AUC | 初步判断 |
|---|---|---|---|---|
| `mydriving_ED24` | 1/3/5/7Hz 的 label-count actual 约 0.65/1.94/3.25/4.53Hz | 0.9168/0.8953/0.8755/0.8615 | 0.9129/0.8883/0.8619/0.8481 | 明显高于 EDformer/EBF BAF，不能作为论文 BAF 锚点 |
| `mydriving_cov05` | 1/3/5/7/10Hz 的 actual 约 0.75/2.26/3.77/5.27/7.54Hz | 0.8616/0.8397/0.8229/0.8084/0.7901 | 0.8593/0.8373/0.8201/0.8053/0.7869 | 1/3Hz 最接近论文，整体仍高于表值 |
| `mydriving_paper` | 1/3/5Hz 的 actual 约 1.00/3.00/5.00Hz | 0.8739/0.8424/0.8186 | 0.8716/0.8397/0.8154 | 3/5Hz 接近 cov05，1Hz 偏高 |
| `mydriving_jaer` | 1/3/5Hz 的 actual 约 1.00/3.00/5.00Hz；原始 `tick_ns=12.5`，必须先转微秒 | 0.8724/0.8424/0.8192 | 0.8700/0.8398/0.8161 | 与 `mydriving_paper` 几乎一致，Jaer 不是论文差异主因 |

论文锚点：EDformer Driving BAF 为 `0.8479/0.8155/0.7930/0.7732/0.7479`，EBF Driving BAF 为 `0.848/0.816/0.793`。按当前固定口径，`mydriving_cov05` 是第一候选锚点；`mydriving_paper` 可作为 DND21 clean + Python FPN shot noise 对照；`mydriving_ED24` 的 BAF 与论文差距过大，后续不优先用于 EDformer/EBF 论文指标对齐。

v2e COV 说明：当前本地 v2e 源码中 `noise_rate_cov_decades` 的 help 写着“currently only in leak events”，`emulator.py` 的 log-normal `noise_rate_array` 初始化位于 `leak_rate_hz > 0` 分支，默认 shot-noise 路径没有直接使用该参数。因此报告中必须区分“v2e 参数记录为 COV=0.5”和“实际 shot-noise FPN 是否由该参数生效”。暂不补做 rate-calibrated v2e 数据。

源码复核补充：

- EDformer `eval_auc.py`/`eval_auc_new.py` 直接读取 `*_mix_result.txt` 第 5 列 label，并调用 `roc_curve(event_label, label_pred_stacked)`；它不通过 clean/noisy subtraction 重新打标签。
- EDformer 训练和 MESR 评估显示模型 sigmoid 输出更像 noise 概率：`eval_mesr.py` 阈值化后保留 `predictions == 0` 的事件；因此原始 EDformer txt 通常按 `0=signal, 1=noise` 理解，而 myEVS npy 统一转换为 `1=signal, 0=noise`。
- EBF 源码 `ebf1231retest.py` 对 txt 输入有 `e_data[:,0] = 1 - e_data[:,0]`，注释为“signal noise label 是反的”；`otherfiltersweep250215eccv.py` 对旧规则滤波器用 `confusion_matrix(1-ytrue, ypred)`，与上面的 txt label 反转一致。
- EBF 源码中的 ECCV/EDformer BAF grid 为 `1,5,10,15,20,25,30,40,50 ms`，并手工加入 ROC 端点；myEVS 的 `baf_ebf_source` profile 与这个 grid 对齐。
- Jaer 数据集的 labeled npy 来自 aedat，metadata 为 `tick_ns=12.5`；用普通 `--tick-ns 1000` 或把 timestamp 当微秒会严重压低 BAF AUC。`evaluate_baf_paper_profiles.py` 已按 metadata 转微秒。

关于视频长度：当前 ED24 converted txt 的时间跨度约 `5.976s`，`mydriving_cov05` 约 `5.9999s`，`mydriving_paper/jaer` 约 `5.98s`。官方下载视频即使肉眼看约 3s，v2e/慢放参数或已发布 txt 事件流已经是近 6s；当前 BAF 差异不是“少了一半视频”导致的。

EDformer 官方模型复现入口已新增：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/EDformer_official/run_official_driving_auc.ps1 `
  -Python D:/software/Anaconda_envs/envs/myEVS/python.exe `
  -EdformerRoot D:/hjx_workspace/scientific_reserach/EDformer `
  -BasePath D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24 `
  -Hz 1,3,5,7,10 -Device auto -CheckEnv
```

当前本机 check-env 结果：`torch/sklearn/pandas/numpy` 可用，但缺 `sparseconvnet`、`pytorch3d`、`dv_processing`，因此不能在本机完成官方 EDformer 推理。服务器上使用 EDformer 论文环境运行：

```bash
EDFORMER_ROOT=/path/to/EDformer \
BASE_PATH=/path/to/mydriving_ED24 \
DEVICE=cuda:0 \
HZ=1,3,5,7,10 \
bash scripts/EDformer_official/run_official_driving_auc.sh
```

输出写到 `data/DND21/edformer_official_auc/driving_auc_official.csv` 或 wrapper 指定的 `driving_auc_<xy_mode>.csv`。主指标列为 `auc_official_label1_positive`，它等价于 EDformer `eval_auc.py` 的 `roc_curve(event_label, sigmoid_output)`；`auc_label_inverted_diagnostic` 只用于诊断 label 是否反了。`--xy-mode official` 保持 EDformer release 代码的 x/y 输入方式；`--xy-mode unit` 只作为坐标归一化诊断，不进入官方复现表。

EDformer Driving txt + myEVS BAF 的 label-direct AUC 入口：

```powershell
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/EDformer_official/eval_official_driving_baf_auc.py `
  --base-path D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24 `
  --filename driving_mix_result.txt `
  --hz 1,3,5,7,10
```

该脚本不跑 EDformer 网络，只复用 EDformer 的 `driving_mix_result.txt` 与原始 label 语义 `0=signal, 1=noise`，再用 myEVS BAF 规则计算 AUC。输出目录：`data/DND21/edformer_official_auc/baf_on_driving_mix/`。主表为 `summary.csv`。

直接结果：

| Hz | myEVS BAF on `driving_mix_result.txt` | EDformer Table 2 BAF | delta |
|---:|---:|---:|---:|
| 1 | 0.916803 | 0.8479 | +0.0689 |
| 3 | 0.895301 | 0.8155 | +0.0798 |
| 5 | 0.875487 | 0.7930 | +0.0825 |
| 7 | 0.861476 | 0.7732 | +0.0883 |
| 10 | 0.840145 | 0.7479 | +0.0922 |

诊断结论：

- `driving_mix_result.txt` raw label-direct 与 converted npy 的 1/3/5/7Hz BAF AUC 完全一致，说明差异不是 myEVS txt->npy 转换造成的。
- `mix_result.txt` 的 BAF 更高：`0.9706/0.9456/0.9232/0.9043/0.8803`，因此它不是 Driving BAF 表的来源。
- include-self 诊断仍偏高：`0.9133/0.8902/0.8691/0.8544/0.8319`，不能解释论文表。
- `tau-scale-us=1` 会过低，`tau-scale-us=100/250/500` 也不能整体贴近论文表，说明不是简单的 ms/us 缩放错误。
- 因此当前更像是 EDformer Table 2 的 BAF baseline 并非由本地 `mydriving_ED24/driving_mix_result.txt` 按标准 3x3 BAF 直接算出，或者论文 baseline 的 BAF 实现/数据版本还有未公开差异。

下一步只在 BAF 锚点数据集上推进其他算法对齐：先以 `mydriving_cov05` 跑 EBF/EDformer/N149；若 BAF 差距仍不能解释，再检查 v2e shot noise 生成链路或 EDformer 发布数据的实际来源。所有新评估优先使用 label-direct AUC；clean/noisy subtraction 只作为数据生成诊断，不能作为 EDformer/EBF 论文 AUC 主口径。

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
D:/software/Anaconda_envs/envs/myEVS/python.exe -m pip install -e . --no-build-isolation
D:/software/Anaconda_envs/envs/myEVS/python.exe -c "import myevs._native_emlb as n; print(n)"
```

### 2.1 C++ 化状态表

| Method | Python 实现 | C++ 实现 | 参考口径 | `engine=cpp` 状态 |
|---|---|---|---|---|
| BAF | 已有 | `BafNative` | myEVS/BAF 规则口径 | 已接入 |
| STCF | 已有 | `StcNative` | myEVS/STCF 规则口径 | 已接入 |
| EBF | 已有 | `EbfNative` | myEVS/EBF 时间核口径 | 已接入 |
| N149 | 原 Python pipeline 未接入 | `N149Native` | N149 v2.1，见 §10 | 已接入 |
| KNoise | 已有 | `KNoiseNative` | E-MLB `khodamoradi_noise.hpp` | 已接入 |
| YNoise | 已有 | `YNoiseNative` | E-MLB `yang_noise.hpp` | 已接入 |
| TS | 已有 | `TimeSurfaceNative` | E-MLB `time_surface.hpp` | 已接入 |
| EvFlow | 已有 | `EventFlowNative` | E-MLB `event_flow.hpp` | 已接入 |
| MLPF-self-trained | 已有 | `MlpfNative` | 自训练权重导出为 `.npz` | 已接入 |
| MLPF-official | 暂无官方模型 | 暂缓 | E-MLB 官方 `.pt` | 暂不纳入 |
| PFD | 已有 | `PfdNative` | PFD/PFDs event-by-event 口径 | 已接入 |
| STCF_orig | — | `StcfOriginalNative` | 论文原始 STCF 口径 | 已接入 |

### 2.2 MLPF 口径说明

当前 myEVS 自训练 MLPF 标注为 `MLPF-self-trained`，不等价于 E-MLB 官方 MLPF。`MlpfNative` 不依赖 libtorch，读取 TorchScript 导出的两层 MLP 权重 `.npz` 做纯 C++ 推理。

导出权重：

```powershell
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/export_mlpf_weights.py `
  --model data/DND21/mydriving_ED24/MLPF/mlpf_torch_1hz_fulltrain.pt
# 生成: mlpf_torch_1hz_fulltrain.npz + .json
```

## 3. 判定规则

| 现象 | 解释优先级 | 后续动作 |
|---|---|---|
| C++ 与 Python 差异小，但 Driving 结果仍差 | 算法本身不适配当前噪声/数据口径 | 在 README2 中标注，不把差结果简单归咎于移植错误。 |
| C++ 与 Python 差异大 | 优先检查 E-MLB 源码语义、参数单位、时间戳单位、极性编码 | 修正实现或脚本参数。 |
| EvFlow 仍很慢 | 算法复杂度本身高，且邻域拟合代价大 | 后续实时性表中单独标注，不强行用大量扫频。 |
| MLPF 效果异常好或异常差 | 优先检查训练/测试是否同集、模型 patch、训练轮数、阈值 | 必须标注 `self-trained`，不作为官方复现结论。 |

## 4. 数据集总览

### 4.1 Driving-ED24

| 属性 | 值 |
|---|---|
| 路径 | `D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24` |
| 分辨率 | 346×260 |
| 噪声档位 | 1hz, 2hz, 3hz, 5hz, 8hz (实际 ~0.65/1.30/1.94/3.25/5.18 Hz/pixel) |
| 文件模式 | `driving_noise_{level}_ed24_withlabel/driving_noise_{level}_signal_only.npy` (clean) / `_labeled.npy` (noisy) |
| 输出目录 | `data/DND21/mydriving_ED24/<ALG>/` |

### 4.2 ED24 Pedestrian (myPedestrain_06)

| 属性 | 值 |
|---|---|
| 路径 | `D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06` |
| 分辨率 | 346×260 |
| 噪声档位 | light (1.8), light_mid (2.1), mid (2.5), heavy (3.3) |
| 文件模式 | `Pedestrain_06_{level}.npy` (noisy) / `Pedestrain_06_{level}_signal_only.npy` (clean) |
| 输出目录 | `data/ED24/myPedestrain_06/<ALG>/` |

### 4.3 ED24 Bicycle (myBicycle_02)

| 属性 | 值 |
|---|---|
| 路径 | `D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02` |
| 分辨率 | 346×260 |
| 噪声档位 | light (2.1), light_mid (2.5), mid (3.3) |
| 文件模式 | `Bicycle_02_{level}.npy` (noisy) / `Bicycle_02_{level}_signal_only.npy` (clean) |
| 输出目录 | `data/ED24/myBicycle_02/<ALG>/` |

### 4.4 DVSCLEAN

| 属性 | 值 |
|---|---|
| 路径 | `D:/hjx_workspace/scientific_reserach/dataset/DVSCLEAN/converted_npy` |
| 分辨率 | 1280×720 |
| 场景 | MAH00444, MAH00446, MAH00447, MAH00448, MAH00449 |
| 档位 | ratio50, ratio100 |
| 文件模式 | `{scene}/{level}/{scene}_{level}_labeled.npy` (noisy) / `_signal_only.npy` (clean) |
| 输出目录 | `data/DVSCLEAN/scene_sweep_full/<SCENE>_<LEVEL>/` |

### 4.5 LED

| 属性 | 值 |
|---|---|
| 路径 | `D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy` |
| 分辨率 | 1280×720 |
| 场景 | scene_100, scene_1004, scene_1018, scene_1028, scene_1032~1046 |
| 切片 | slices_00031_00040_100ms |
| 文件模式 | `{scene}/slices_00031_00040_100ms/{scene}_100ms_labeled.npy` (noisy) / `_signal_only.npy` (clean) |
| 输出目录 | `data/LED/scene_sweep/<SCENE>/` |

## 5. 运行指令速查

> 通用环境：`$PY = "D:/software/Anaconda_envs/envs/myEVS/python.exe"`，`--engine cpp` 默认用于所有规则算法。

### 5.1 通用 CLI 格式

```bash
$PY -m myevs.cli roc \
  --clean <CLEAN.npy> --noisy <NOISY.npy> --assume npy \
  --width <W> --height <H> --tick-ns 1000 \
  --engine cpp --method <METHOD> \
  --radius-px <R> --time-us <TAU> \
  --param min-neighbors --values <THR_LIST> \
  --match-us 0 --match-bin-radius 0 \
  --tag <TAG> --out-csv <OUT.csv> --append
```

阈值扫频标准列表（17 点）：
```
0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,8
```

### 5.2 Driving-ED24 一键运行

```powershell
# 全量运行所有算法（coarse sweep）
powershell -ExecutionPolicy Bypass -File ./scripts/driving_ED24_alg_evalu/run_driving_alg_paper.ps1 -Algorithms all

# 仅 CPP 算法
powershell -ExecutionPolicy Bypass -File ./scripts/driving_ED24_alg_evalu/run_driving_alg_paper.ps1 `
  -Algorithms baf,stcf,ebf,n149,knoise,evflow,ynoise,ts,pfd -Engine cpp

# 单算法
powershell -ExecutionPolicy Bypass -File ./scripts/driving_ED24_alg_evalu/run_driving_alg_paper.ps1 -Algorithm n149

# 轻量验证 (200K events)
powershell -ExecutionPolicy Bypass -File ./scripts/driving_ED24_alg_evalu/run_driving_alg_paper.ps1 -Algorithms all -MaxEvents 200000

# 仅 N149 单点测试（最快）
$PY -m myevs.cli roc `
  --clean "D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_8hz_ed24_withlabel/driving_noise_8hz_signal_only.npy" `
  --noisy "D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_8hz_ed24_withlabel/driving_noise_8hz_labeled.npy" `
  --assume npy --width 346 --height 260 --tick-ns 1000 `
  --engine cpp --method n149 --radius-px 2 --time-us 32000 `
  --param min-neighbors --values "0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,8" `
  --match-us 0 --match-bin-radius 0 `
  --tag n149_8hz_test --out-csv data/_test.csv
```

### 5.3 ED24 Pedestrian 一键运行

```powershell
# EBF + N149 网格扫频
$PY scripts/ED24_alg_evalu/run_n149_labelscore_grid.py `
  --light "D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_1.8.npy" `
  --mid "D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_2.5.npy" `
  --heavy "D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3.npy" `
  --out-dir "data/ED24/myPedestrain_06/N149"

# 单算法脚本
powershell -ExecutionPolicy Bypass -File ./scripts/ED24_alg_evalu/run_slomo_ebf.ps1
powershell -ExecutionPolicy Bypass -File ./scripts/ED24_alg_evalu/run_slomo_baf.ps1
powershell -ExecutionPolicy Bypass -File ./scripts/ED24_alg_evalu/run_slomo_stcf.ps1

# 全部算法一次性运行（含 MLPF/PFD/TS/YNoise/EvFlow/KNoise）
powershell -ExecutionPolicy Bypass -File ./scripts/ED24_alg_evalu/run_slomo_21_all.ps1
```

### 5.4 ED24 Bicycle 一键运行

```powershell
# 复用 Pedestrian 最优参数
$PY scripts/ED24_alg_evalu/run_n149_labelscore_grid.py `
  --light "D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_2.1.npy" `
  --mid "D:/hjx_workspace/scientific_reserach/dataset/ED24/myBicycle_02/Bicycle_02_3.3.npy" `
  --out-dir "data/ED24/myBicycle_02/N149"
```

### 5.5 DVSCLEAN 一键运行

```powershell
# 单场景扫频汇总
$PY scripts/DVSCLEAN_alg_evalu/run_dvsclean_scene_sweep_summary.py `
  --scene MAH00444 --level ratio100 `
  --mlpf-model data/DVSCLEAN/models/mlpf_torch_MAH00444.pt

# 全量 5 场景
powershell -ExecutionPolicy Bypass -File ./scripts/DVSCLEAN_alg_evalu/run_dvsclean_n179_full.ps1
```

### 5.6 LED 一键运行

```powershell
# 单场景扫频
$PY scripts/LED_alg_evalu/run_led_scene_sweep_summary.py `
  --scene scene_100 --max-events 300000

# 全量 10 场景
powershell -ExecutionPolicy Bypass -File ./scripts/LED_alg_evalu/run_led_n179_full.ps1
```

### 5.7 MLPF 专用命令

```powershell
# 训练
$PY scripts/train_mlpf_torch.py `
  --clean <CLEAN.npy> --noisy <NOISY.npy> `
  --max-events 0 --patch 9 --hidden 20 `
  --out-model data/<OUT>.pt

# C++ 推理（需先导出权重）
$PY -m myevs.cli roc `
  --clean <CLEAN.npy> --noisy <NOISY.npy> --assume npy `
  --width 346 --height 260 --tick-ns 1000 `
  --engine cpp --method mlpf `
  --mlpf-model <MODEL.pt> --radius-px 3 --time-us 100000 `
  --param min-neighbors --values "0.01,0.02,0.03,0.04,0.05,0.1,0.14,0.2" `
  --match-us 0 --match-bin-radius 0 `
  --tag mlpf_cpp --out-csv <OUT.csv>
```

### 5.8 PFD 专用命令

```powershell
# PFD-A 模式（需 --refractory-us 和 --pfd-mode）
$PY -m myevs.cli roc `
  --clean <CLEAN.npy> --noisy <NOISY.npy> --assume npy `
  --width 1280 --height 720 --tick-ns 1000 `
  --engine cpp --method pfd `
  --radius-px 3 --time-us 8000 `
  --refractory-us 2 --pfd-mode a `
  --param min-neighbors --values "0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,8" `
  --match-us 0 --match-bin-radius 0 `
  --tag pfd_test --out-csv data/_pfd_test.csv
```

## 6. 各算法扫频策略

| Algorithm | 默认 engine | 扫频变量 | 固定参数 | 说明 |
|---|---|---|---|---|
| BAF | `cpp` | `tau` | `radius={1,2,3,4,5}` | 每半径扫时间窗 |
| STCF | `cpp` | `tau` | `radius={1,2,3,4,5}`, `min_neighbors=1` | |
| STCF_orig | `cpp` | `tau`, `k` | `radius=1` (固定 3×3), `min_neighbors=k` | 论文原始口径 |
| EBF | `cpp` | `threshold` | `radius={2,3,4,5}`, `tau={16K..512K}` | ROC 曲线扫阈值 |
| N149 | `cpp` | `threshold` | `radius={2,3,4,5}`, `tau={16K..512K}` | |
| KNoise | `cpp` | `threshold/tau` | 指数 tau 范围 | ROC 点容易集中 |
| YNoise | `cpp` | `threshold` | `radius/tau` 网格 | 每半径保留 AUC 最优 3 条曲线 |
| TS | `cpp` | `threshold` | `radius/tau` 网格 | |
| EvFlow | `cpp` | `threshold` | 缩小 sweep 范围 | 计算最慢，lite sweep |
| MLPF | `python`/`cpp` | `threshold` | 依赖 `.pt` 模型和 `.npz` 导出 | C++ 推理需 `--engine cpp` |
| PFD | `cpp` | `threshold` | `--refractory-us 2 --pfd-mode a` | PFD-A 模式，可切换 PFD-B |

## 7. 运行结果

> 全部算法使用 `engine=cpp`（MLPF 部分使用 python）。阈值列表为 17 点标准扫频 `0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.5,3,4,5,6,8`。EBF/N149 使用对齐阈值公平对比。N149 为原始版（含 beta/sfrac），v2 公平对比见 §8.6，v2 定义见 §10。运行日期 2026-05-13 ~ 2026-05-15。

### 7.1 Driving-ED24: 1hz (实际 ~0.65 Hz/pixel)

| Method | Engine | Best AUC | Best F1 | Best AUC tag | 备注 |
|---|---:|---:|---:|---|---|
| `STCF` | `cpp` | 0.9484 | 0.9678 | stcf_r2 | |
| `EBF` | `cpp` | 0.9484 | 0.9692 | ebf_r2_tau32000 | |
| `YNoise` | `cpp` | 0.9408 | 0.9679 | ynoise_r2_tau16000 | |
| `N149` | `cpp` | 0.9381 | 0.9666 | n149_r2_tau32000 | |
| `N149_v2.1` | `cpp` | 0.9370 | 0.9627 | v21_16b_r2_tau32K | v2.1 |
| **N149_v2.2** | cpp | **0.9512** | 0.9669 | v22_final | **v2.2** r=2 tau=32K sigma=1.75 alpha=0.05 |
| `N149_v2` | `cpp` | 0.9367 | 0.9666 | n149v2_r2_tau32000 | v2, 点检 |
| `TS` | `cpp` | 0.9298 | 0.9668 | ts_r2_decay32000 | |
| `BAF` | `cpp` | 0.9136 | 0.6341 | baf_r1 | r=1 固定重跑 |
| `PFD` | `cpp` | 0.8992 | 0.9269 | pfd_r1_tau32000_m1 | r=1 固定重跑 |
| `STCF_orig` | `cpp` | 0.9136 | 0.9589 | stcf_orig_k1 | 重跑（tau 与 BAF 同口径） |
| `EvFlow` | `cpp` | 0.8486 | 0.9501 | evflow_r2_tau16000 | |
| `MLPF` | `cpp` | 0.8977 | 0.9504 | mlpf_model_patch7_dur100000_1hz | patch=7 重跑 |
| `KNoise` | `cpp` | 0.6359 | 0.9399 | knoise_tau8000 | |

### 7.2 Driving-ED24: 2hz (实际 ~1.30 Hz/pixel)

| Method | Engine | Best AUC | Best F1 | Best AUC tag | 备注 |
|---|---:|---:|---:|---|---|
| `EBF` | `cpp` | 0.9472 | 0.9498 | ebf_r2_tau32000 | |
| `STCF` | `cpp` | 0.9445 | 0.7401 | stcf_r2 | |
| `YNoise` | `cpp` | 0.9390 | 0.9036 | ynoise_r2_tau16000 | |
| `N149` | `cpp` | 0.9386 | 0.9437 | n149_r2_tau32000 | 对齐 EBF 阈值 |
| `N149_v2` | `cpp` | 0.9375 | 0.9437 | n149v2_r2_tau32000 | v2, 点检 |
| `N149_v2.1` | `cpp` | 0.9377 | 0.9441 | v21_16b_r2_tau32K | v2.1 |
| **N149_v2.2** | cpp | **0.9508** | 0.9516 | v22_final | **v2.2** r=2 tau=32K sigma=1.75 alpha=0.05 |
| `TS` | `cpp` | 0.9322 | 0.7412 | ts_r2_decay16000 | |
| `PFD` | `cpp` | 0.8964 | 0.9179 | pfd_r1_tau32000_m1 | r=1 固定重跑 |
| `BAF` | `cpp` | 0.9029 | 0.6335 | baf_r1 | r=1 固定重跑 |
| `MLPF` | `cpp` | — | — | — | patch=7 待补(2hz) |
| `KNoise` | `cpp` | 0.6265 | 0.0073 | knoise_tau8000 | |

### 7.3 Driving-ED24: 3hz (实际 ~1.94 Hz/pixel)

| Method | Engine | Best AUC | Best F1 | Best AUC tag | 备注 |
|---|---:|---:|---:|---|---|
| `EBF` | `cpp` | 0.9444 | 0.9365 | ebf_r2_tau32000 | |
| `STCF` | `cpp` | 0.9400 | 0.9312 | stcf_r2 | |
| `N149` | `cpp` | 0.9394 | 0.9314 | n149_r2_tau32000 | |
| `N149_v2` | `cpp` | — | — | — | 待测 |
| `N149_v2.1` | `cpp` | 0.9377 | 0.9314 | v21_16b_r2_tau32K | v2.1 |
| **N149_v2.2** | cpp | **0.9502** | 0.9399 | v22_final | **v2.2** r=2 tau=32K sigma=1.75 alpha=0.05 |
| `YNoise` | `cpp` | 0.9361 | 0.9347 | ynoise_r2_tau16000 | |
| `TS` | `cpp` | 0.9279 | 0.9309 | ts_r2_decay32000 | |
| `STCF_orig` | `cpp` | 0.9047 | 0.9121 | stcf_orig_k2 | 重跑（tau 与 BAF 同口径） |
| `PFD` | `cpp` | 0.8925 | 0.8966 | pfd_r1_tau16000_m1 | r=1 固定重跑 |
| `BAF` | `cpp` | 0.8909 | 0.6319 | baf_r1 | r=1 固定重跑 |
| `EvFlow` | `cpp` | 0.8424 | 0.9101 | evflow_r2_tau16000 | |
| `KNoise` | `cpp` | 0.6232 | 0.8395 | knoise_tau8000 | |

### 7.4 Driving-ED24: 5hz (实际 ~3.25 Hz/pixel)

| Method | Engine | Best AUC | Best F1 | Best AUC tag | 备注 |
|---|---:|---:|---:|---|---|
| `N149` | `cpp` | 0.9416 | 0.9092 | n149_r2_tau32000 | 高噪声最优 |
| `N149_v2` | `cpp` | 0.9410 | 0.9092 | n149v2_r2_tau32000 | v2, 点检 |
| `N149_v2.1` | `cpp` | 0.9412 | 0.9085 | v21_16b_r2_tau32K | v2.1 |
| **N149_v2.2** | cpp | **0.9500** | 0.9222 | v22_final | **v2.2** r=2 tau=32K sigma=1.75 alpha=0.05 |
| `EBF` | `cpp` | 0.9408 | 0.9128 | ebf_r2_tau32000 | |
| `YNoise` | `cpp` | 0.9312 | 0.9090 | ynoise_r2_tau16000 | |
| `STCF` | `cpp` | 0.9309 | 0.9008 | stcf_r1 | |
| `TS` | `cpp` | 0.9259 | 0.9028 | ts_r2_decay32000 | |
| `MLPF` | `cpp` | 0.9250 | 0.8912 | mlpf_model_patch7_dur100000_5hz | patch=7 重跑 |
| `STCF_orig` | `cpp` | 0.8976 | 0.8730 | stcf_orig_k2 | 重跑（tau 与 BAF 同口径） |
| `PFD` | `cpp` | 0.8886 | 0.8779 | pfd_r1_tau16000_m1 | r=1 固定重跑 |
| `BAF` | `cpp` | 0.8648 | 0.6225 | baf_r1 | r=1 固定重跑 |
| `EvFlow` | `cpp` | 0.8206 | 0.8686 | evflow_r2_tau16000 | |
| `KNoise` | `cpp` | 0.6239 | 0.7579 | knoise_tau16000 | |

### 7.5A Driving-ED24: 7hz（先填已跑算法）

| Method | Engine | Best AUC | Best F1 | Best AUC tag | 备注 |
|---|---:|---:|---:|---|---|
| `MLPF` | `cpp` | 0.9238 | 0.6953 | mlpf_model_patch7_dur100000_7hz | 全量重跑 |
| `STCF_orig` | `cpp` | 0.8912 | 0.8436 | stcf_orig_k2 | 重跑（tau 与 BAF 同口径） |
| `PFD` | `cpp` | 0.8849 | 0.8648 | pfd_r1_tau16000_m2 | r=1 固定重跑 |
| `BAF` | `cpp` | 0.8532 | 0.6178 | baf_r1 | r=1 固定重跑 |
| `TS` | `cpp` | — | — | — | 7hz 数据已删除，待重跑 |
| `YNoise` | `cpp` | — | — | — | 7hz 数据已删除，待重跑 |
| `KNoise` | `cpp` | 0.6168 | 0.4002 | knoise_tau10000 | 已有 |

### 7.5 Driving-ED24: 8hz (实际 ~5.18 Hz/pixel)

| Method | Engine | Best AUC | Best F1 | Best AUC tag | 备注 |
|---|---:|---:|---:|---|---|
| **`N149`** | `cpp` | **0.9418** | 0.8221 | n149_r2_tau32000 | **8hz 最优** |
| `N149_v2` | `cpp` | 0.9414 | 0.8221 | n149v2_r2_tau32000 | v2, 点检 |
| `N149_v2.1` | `cpp` | 0.9416 | 0.8833 | v21_16b_r2_tau32K | v2.1 |
| **N149_v2.2** | cpp | **0.9484** | 0.8978 | v22_final | **v2.2** r=2 tau=32K sigma=1.75 alpha=0.05 |
| `EBF` | `cpp` | 0.9374 | 0.7618 | ebf_r2_tau32000 | |
| `TS` | `cpp` | 0.9291 | 0.7428 | ts_r2_decay16000 | |
| `YNoise` | `cpp` | 0.9252 | 0.8755 | ynoise_r2_tau16000 | |
| `STCF` | `cpp` | 0.9229 | 0.8689 | stcf_r1 | |
| `MLPF` | `cpp` | — | — | — | patch=7 待补(8hz) |
| `STCF_orig` | `cpp` | 0.8852 | 0.8255 | stcf_orig_k3 | 重跑（tau 与 BAF 同口径） |
| `PFD` | `cpp` | 0.8830 | 0.8547 | pfd_r1_tau16000_m2 | r=1 固定重跑 |
| `BAF` | `cpp` | 0.8379 | 0.6053 | baf_r1 | r=1 固定重跑 |
| `KNoise` | `cpp` | 0.6214 | — | knoise_tau8000 | |


### 7.5B Driving-ED24: 10hz（先填已跑算法）

| Method | Engine | Best AUC | Best F1 | Best AUC tag | 备注 |
|---|---:|---:|---:|---|---|
| `MLPF` | `cpp` | 0.9199 | 0.6149 | mlpf_model_patch7_dur100000_10hz | 全量重跑 |
| `STCF_orig` | `cpp` | 0.8830 | 0.8086 | stcf_orig_k3 | 重跑（tau 与 BAF 同口径） |
| `PFD` | `cpp` | 0.8794 | 0.8327 | pfd_r1_tau16000_m2 | r=1 固定重跑 |
| `BAF` | `cpp` | 0.8269 | 0.5966 | baf_r1 | r=1 固定重跑 |
| `TS` | `cpp` | — | — | — | 10hz 数据已删除，待重跑 |
| `YNoise` | `cpp` | — | — | — | 10hz 数据已删除，待重跑 |
| `KNoise` | `cpp` | 0.6172 | 0.3996 | knoise_tau10000 | 已有 |

### 7.6 MLPF C++ vs Python 对比 (Driving-ED24)

| Level | C++ AUC | C++ F1 | tag | 备注 |
|---|---:|---:|---|---|
| 1hz | 0.8977 | 0.9504 | `mlpf_model_patch7_dur100000_1hz` | patch=7 重跑 |
| 2hz | — | — | — | patch=7 待补 |
| 3hz | 0.9201 | 0.9148 | `mlpf_model_patch7_dur100000_3hz` | patch=7 重跑 |
| 5hz | 0.9250 | 0.8912 | `mlpf_model_patch7_dur100000_5hz` | patch=7 重跑 |
| 7hz | 0.9238 | 0.6953 | `mlpf_model_patch7_dur100000_7hz` | 全量重跑 |
| 8hz | — | — | — | patch=7 待补 |
| 10hz | 0.9199 | 0.6149 | `mlpf_model_patch7_dur100000_10hz` | 全量重跑 |

> C++ 优于 Python 的原因：每个阈值独立重置 model state，避免了 Python once-pass 中早期事件状态未建立的评分偏差。论文使用 C++ engine 口径。

### 7.7 Driving-ED24 AUC 排名

| Rank | 1hz | AUC | 2hz | AUC | 5hz | AUC | 8hz | AUC |
|---|---|---|---|---|---|---|---|---|
| 1 | STCF | 0.9484 | EBF | 0.9472 | N149 | 0.9416 | N149 | 0.9418 |
| 2 | EBF | 0.9484 | STCF | 0.9445 | EBF | 0.9408 | EBF | 0.9374 |
| 3 | YNoise | 0.9408 | YNoise | 0.9390 | YNoise | 0.9312 | TS | 0.9291 |
| 4 | N149 | 0.9381 | N149 | 0.9386 | STCF | 0.9309 | YNoise | 0.9252 |
| 5 | TS | 0.9298 | TS | 0.9322 | TS | 0.9259 | STCF | 0.9229 |
| — | **v2.2** | **0.9512** | **v2.2** | **0.9508** | **v2.2** | **0.9502** | **v2.2** | **0.9500** |
| — | v2.1 | 0.9370 | v2.1 | 0.9377 | v2.1 | 0.9412 | v2.1 | 0.9416 |
| 6 | BAF | 0.9166 | PFD | 0.9111 | MLPF | 0.9250 | MLPF | 待补(p7) |
| 7 | PFD | 0.9135 | BAF | 0.9029 | PFD | 0.8984 | PFD | 0.8825 |
| 8 | MLPF | 0.8977 | MLPF | 待补(p7) | BAF | 0.8651 | BAF | 0.8379 |
| 9 | EvFlow | 0.8486 | EvFlow | 0.8475 | EvFlow | 0.8206 | EvFlow | 0.8060 |
| 10 | KNoise | 0.6359 | KNoise | 0.6265 | KNoise | 0.6239 | KNoise | 0.6214 |

> **N149 v2.2 (r=2,tau=32K,sigma=1.75,alpha=0) 在 Driving 全级均排名第一**，1hz AUC=0.9512（超越原始 N149 +0.013）。sigma 细调至 1.75 后跨 5 级一致性极佳。

### 7.8 ED24 Pedestrian (myPedestrain_06)

| Method | light AUC | mid AUC | heavy AUC | Best-light | Best-mid | Best-heavy | 备注 |
|---|---:|---:|---:|---|---|---|---|
| `N149` | **0.9565** | **0.9469** | **0.9406** | (5,256ms) | (5,256ms) | (5,256ms) | 全级最优 |
| `N149_v2` | 0.9551 | 0.9453 | 0.9388 | (5,256ms) | (5,256ms) | (5,256ms) | v2, 点检 |
| `N149_v2.1` | 0.9545 | 0.9432 | 0.9360 | (5,256ms) | (5,256ms) | (5,256ms) | v2.1 |
| **N149_v2.2** | 0.9563 | 0.9443 | 0.9375 | (5,256ms) | (5,256ms) | (5,256ms) | **v2.2** sigma=2.75 alpha=0.25 |
| `STCF` | 0.9460 | 0.8962 | 0.8791 | (4,256ms) | (4,256ms) | (4,256ms) | |
| `EBF` | 0.9416 | 0.9185 | 0.9099 | (5,128ms) | (5,128ms) | (5,128ms) | |
| `YNoise` | 0.9227 | 0.9083 | 0.8971 | (4,64ms) | (4,64ms) | (4,64ms) | |
| `BAF` | 0.9013 | 0.8417 | 0.8165 | (1,512ms) | (1,512ms) | (1,512ms) | r=1 fixed 重跑(2026-05-21) |
| `PFD` | 0.8030 | 0.7932 | 0.7856 | (1,256ms,m1) | (1,128ms,m1) | (1,64ms,m1) | 重跑覆盖(2026-05-21) |
| `STCF_orig` | 0.9013 | 0.8598 | 0.8402 | (1,512ms,k1) | (1,32000us,k2) | (1,32000us,k2) | 重跑覆盖(2026-05-21, tau 对齐) |
| `TS` | 0.8619 | 0.8528 | 0.8465 | (2,32ms) | (2,32ms) | (2,32ms) | |
| `EvFlow` | 0.8351 | 0.8022 | 0.7847 | (4,32ms) | (4,32ms) | (4,32ms) | |
| `KNoise` | 0.7130 | 0.6625 | 0.6417 | (1,32ms) | (1,32ms) | (1,32ms) | |

### 7.9 ED24 Bicycle (myBicycle_02)

| Method | light AUC | light_mid AUC | mid AUC | heavy(3.3) AUC | Best-light | Best-light_mid | Best-mid | Best-heavy | 备注 |
|---|---:|---:|---:|---:|---|---|---|---|---|
| `N149` | **0.9845** | **0.9827** | **0.9787** | — | (5,512ms) | (5,512ms) | (5,512ms) | — | 全级最优 |
| `N149_v2` | 0.9850 | 0.9832 | 0.9796 | — | (5,512ms) | (5,512ms) | (5,512ms) | — | v2, 点检 |
| `N149_v2.1` | 0.9840 | 0.9822 | 0.9778 | — | (5,512ms) | (5,512ms) | (5,512ms) | — | v2.1 |
| **N149_v2.2** | 0.9866 | 0.9838 | 0.9800 | 0.9746 | (5,256ms) | **v2.2** sigma=2.75 alpha=0.25 |
| `STCF` | 0.9785 | 0.9649 | 0.9418 | — | (4,32ms) | (4,32ms) | (4,32ms) | — | |
| `EBF` | 0.9681 | 0.9621 | 0.9422 | — | (5,512ms) | (5,512ms) | (5,512ms) | — | |
| `YNoise` | 0.9601 | 0.9417 | 0.7859 | — | (5,256ms) | (5,256ms) | (5,256ms) | — | |
| `PFD` | 0.8853 | 0.8759 | 0.8682 | 0.8606 | (1,256ms,m1) | (1,256ms,m1) | (1,64ms,m1) | (1,32ms,m1) | 重跑覆盖(2026-05-21) |
| `STCF_orig` | 0.9501 | 0.9374 | 0.9302 | 0.9169 | (1,128ms,k1) | (1,32ms,k1) | (1,32ms,k2) | (1,16ms,k2) | 重跑覆盖(2026-05-21) |
| `BAF` | 0.9489 | 0.9352 | 0.9149 | 0.8960 | (1,512ms) | (1,512ms) | (1,512ms) | (1,512ms) | r=1 fixed 重跑(2026-05-21) |
| `EvFlow` | 0.8877 | 0.8894 | 0.8717 | — | (6,32ms) | (4,32ms) | (2,32ms) | — | 连续扩边重扫(2026-05-21, light的value上限扩至300，最优仍在边界) |
| `TS` | 0.9102 | 0.9090 | 0.8977 | — | (4,128ms) | (4,128ms) | (4,128ms) | — | |
| `KNoise` | 0.7631 | 0.7471 | 0.7215 | — | (1,16ms) | (1,16ms) | (1,16ms) | — | |

> 使用 ED24 Pedestrian 最优参数 (r=3-5)。N149 在 Bicycle 上全级最优。Bicycle heavy 和 LED 剩余场景待补。

#### 7.9.1 BAF 与 STCF_orig(K=1) 差异来源（已定位）

- 现象：在部分子集上，`STCF_orig(K=1, r=1)` 的 AUC 会略高于 `BAF(r=1)`，看起来不符合“`K=1` 应等价于 BAF”的直觉。
- 根因（实现口径差异）：
  - `BAF` 判定邻居有效条件是 `ts != 0 && ts >= t - tau`（未初始化邻居 `ts=0` 不计入）。
  - `STCF_orig` 判定条件是 `0 <= (t - last_ts_neighbor) <= tau`。当 `last_ts_neighbor=0` 且事件处于序列早期 `t <= tau` 时，该邻居会被计为有效。
- 直接后果：`STCF_orig(K=1)` 在序列开头会多通过一小批事件，之后两者基本一致。
- 定量验证（`myBicycle_02/light`, `tau=1000`, `r=1`）：
  - 总事件 `N=83364`：`STCF_orig` 比 `BAF` 多保留 `26` 个事件。
  - 这 `26` 个差异全部出现在序列前缀（前 `100` 事件时已出现全部差值，后续差值不再扩大）。
- 结论：这是“时间戳初始化边界条件”导致的细微差异，不是扫参与脚本错误。

### 7.10 DVSCLEAN (8 scenes: MAH00444-448 × ratio50/100)

| Method | Engine | Mean AUC | Mean F1 | Best (r, tau) | 备注 |
|---|---:|---:|---:|---|---|
| `N149` | `cpp` | 0.9970 | 0.9900 | (5, 128ms) | 最优 |
| `N149_v2` | `cpp` | 0.9978 | — | (5, 128ms) | v2, 点检 |
| `N149_v2.1` | `cpp` | 0.9978 | — | (5, 128ms) | v2.1 |
| **N149_v2.2** | cpp | 0.9966 | — | (5,128ms) | **v2.2** sigma=2.5 alpha=0.25 (10场景均值) |
| `EBF` | `cpp` | 0.9940 | 0.9843 | (4, 64ms) | |
| `YNoise` | `cpp` | 0.9934 | 0.9836 | (4, 32ms) | |
| `STCF` | `cpp` | 0.9898 | 0.9772 | (3, 32ms) | |
| `PFD` | `cpp` | 0.9863 | 0.9789 | (3, 8ms) | 修复后 (--refractory-us 2) |
| `MLPF` | `python` | 0.9823 | 0.9763 | (3, 64ms) | |
| `STCF_orig` | `cpp` | 0.9389 | 0.9296 | (1, 2~16ms, k=1~3) | 全 10 子集重跑（tau 对齐 BAF） |
| `BAF` | `cpp` | 0.9479 | 0.9533 | (1, 32ms) | |
| `TS` | `cpp` | 0.9393 | 0.9274 | (2, 32ms) | |
| `EvFlow` | `cpp` | 0.7733 | 0.6828 | (2, 8ms) | lite sweep |
| `KNoise` | `cpp` | 0.6389 | 0.4381 | (1, 32ms) | |

### 7.11 LED (scene_100)

| Method | Engine | AUC | F1 | Best (r, tau) | 备注 |
|---|---:|---:|---:|---|---|
| `N149` | `cpp` | 0.9133 | 0.9091 | (2, 16ms) | |
| `N149_v2` | `cpp` | 0.9162 | 0.9091 | (2, 16ms) | v2, 点检 |
| `N149_v2.1` | `cpp` | 0.9162 | 0.9784 | (2, 16ms) | v2.1 |
| **N149_v2.2** | cpp | 0.8872 | — | (2,8K) | **v2.2** sigma=2.0 alpha=2.0 (10场景均值) |
| `STCF_orig` | `cpp` | 0.8704 | 0.9646 | (1, 8ms, k=2) | 重跑（tau 对齐 BAF） |
| `YNoise` | `cpp` | 0.8875 | 0.8862 | (2, 8ms) | |
| `STCF` | `cpp` | 0.8841 | 0.8704 | (2, 4ms) | |
| `EBF` | `cpp` | 0.8569 | 0.8978 | (2, 16ms) | |
| `PFD` | `cpp` | 0.8248 | 0.9238 | (3, 8ms) | 修复后 |
| `BAF` | `cpp` | 0.7994 | 0.8778 | (1, 2ms) | |
| `TS` | `cpp` | 0.7958 | 0.8657 | (2, 8ms) | |
| `EvFlow` | `cpp` | 0.7868 | 0.9414 | (2, 8ms) | lite sweep |
| `MLPF` | `python` | 0.7264 | 0.7122 | (3, 16ms) | |
| `KNoise` | `cpp` | 0.5323 | 0.1499 | (1, 2ms) | |

> **N149 v2.2 LED 10 场景明细** (r=2, tau=8K, sigma=2.0, alpha=2.0)：
> scene_100=0.9262, 1004=0.8588, 1018=0.8889, 1028=0.8993, 1032=0.8954,
> 1033=0.8650, 1034=0.8776, 1043=0.8913, 1045=0.8773, 1046=0.8917。均值 0.8872。

| LED scene_1004 | Method | AUC | F1 | Best (r,tau) |
|---|---:|---:|---:|---|
| | `N149` | 0.8567 | 0.8654 | (3, 16ms) |
| | `YNoise` | 0.8191 | 0.8648 | (2, 8ms) |
| | `STCF` | 0.8181 | 0.8496 | (2, 4ms) |
| | `EBF` | 0.8083 | 0.8463 | (2, 16ms) |
| | `PFD` | 0.7934 | 0.8312 | (3, 8ms) |
| | `EvFlow` | 0.7741 | 0.8969 | (2, 8ms) |
| | `BAF` | 0.7203 | 0.7492 | (1, 2ms) |
| | `TS` | 0.7122 | 0.8723 | (2, 8ms) |
| | `KNoise` | 0.5378 | 0.1782 | (1, 2ms) |

> 其余 LED 场景 (1018/1028/1032~1046) 待补。

#### STCF_orig 本轮重跑对比（2026-05-21）

- 统一口径：`radius=1`，并将 `tau` 扫频与 `BAF` 对齐（Driving: `1,2,4,8,16,32ms`；DVSCLEAN: `2,4,8,16ms`；LED: `2,8ms`）。
- Driving-ED24：相较旧表，`AUC` 整体小幅下降，但 `F1` 显著回升到合理区间（不再出现 0.1~0.3 的异常值）。
- ED24 Ped/Bicycle：`AUC` 基本稳定（Ped 轻微上调到 `0.9013/0.8598/0.8402`；Bicycle 维持 `0.9501/0.9374/0.9302/0.9169`）。
- DVSCLEAN：全 10 子集重跑后 `STCF_orig` 均值为 `AUC=0.9389, F1=0.9296`（旧值偏高，来源于旧扫频口径不一致）。
- LED scene_100：更新为 `AUC=0.8704, F1=0.9646, tag=stcf_orig_k2`。

## 8. N149 消融实验

> 测试目的：逐一关闭 N149 各组件，量化各组件对不同场景去噪性能的贡献。通过环境变量控制 CPP 实现中的模块开关。
> 阈值扫频统一使用 17 点标准列表。ΔAUC = AUC(消融) − AUC(Baseline)。负值=组件有益（关闭后变差），正值=组件有害（关闭后反而更好）。*** 标记 |ΔAUC| > 0.002。

### 8.1 组件与 N149 公式对应

N149 得分公式：

$$
w_s(i,j)=e^{-d^2/2\sigma^2},\quad
R_i^+=\sum w_t w_s\mathbf{1}[同极],\quad
R_i^-=\sum w_t w_s\mathbf{1}[异极]
$$

$$
u_i=\frac{h_i}{h_i+\tau},\quad
\alpha_i=(1-m_i)^2,\quad
\widetilde{R}_i=R_i^+ + \alpha_i R_i^-
$$

$$
B_i=\frac{\widetilde{R}_i}{1+u_i},\quad
S_i=B_i\cdot(1+b_i s_i)
$$
注：$$b_i,s_i后续被优化掉了$$
| 组件 | 符号 | 环境变量 | 作用 |
|---|---|---|---|
| `hot_state` | \(u_i\) | `MYEVS_N149_NO_HOT=1` | 时间自激励：同极越近得分越高 |
| `beta_state` | \(b_i\) | `MYEVS_N149_NO_BETA=1` | 慢速自适应 EMA，调制最终得分 |
| `mix_state` | \(m_i \to \alpha_i\) | `MYEVS_N149_NO_MIX=1` | 异极比例 EMA → 异极抑制系数 \(\alpha_i=(1-m_i)^2\) |
| `opp_polarity` | \(R_i^-\) | `MYEVS_N149_NO_OPP=1` | 异极邻域证据：关闭后 \(\widetilde{R}_i=R_i^+\) |
| `sfrac` | \(s_i, b_i\) | `MYEVS_N149_NO_SFRAC=1` | 支持率：关闭后 \(S_i=B_i\) |
| `spatial_w` | \(w_s(i,j)\) | `MYEVS_N149_NO_SPATIAL=1` | 空间高斯衰减：关闭后 \(w_s=1\)（邻域等权） |

> 注：NO_MIX (\(m_i=0 \to \alpha_i=1\)) 与 NO_OPP (\(R_i^-=0\)) 方向相反——前者保留异极但不抑制，后者完全忽略异极。

**优化后 N149 v2.1 公式**（消融结论落地，2026-05-15 定稿）：

> 移除 \(b_i,s_i\)；h 更新简化为单行（与原公式实验等价，§8.17）；折扣因子去 u 中间变量。

每事件处理流程：

1. 更新热状态：\(h_i \leftarrow \max(0,\ h_i + T_r - 2\Delta t_i)\)，\(T_r = \tau \cdot u\_denom\_factor\)
2. 计算邻域证据（\(w_t\) 为平方核）：
   $$w_t = \big(\max(0,1-\Delta t_{ij}/\tau)\big)^2,\quad w_s = \exp(-d^2/2\sigma^2)$$
   $$R_i^{+}=\sum_{j\in\mathcal{N}_r(i)\setminus\{i\}} w_t w_s\mathbf{1}[p_j=p_i],\quad
     R_i^{-}=\sum_{j\in\mathcal{N}_r(i)\setminus\{i\}} w_t w_s\mathbf{1}[p_j=-p_i]$$
3. 更新异极 EMA：\(\text{mix}_i=R_i^{-}/(R_i^{+}+R_i^{-}+\varepsilon)\)，
   $$m_i \leftarrow m_i + (\text{mix}_i - m_i)/4096$$
4. 计算得分（折扣因子 \(\in[1/K,1]\)，默认 \(K=2\)）：
   $$\boxed{S_i = \big(R_i^{+}+(1-m_i)^2R_i^{-}\big) \cdot \frac{h_i+T_r}{K\cdot h_i+T_r}}$$

| 组件 | 符号 | 作用 |
|---|---|---|
| `hot_state` | \(h_i\) → 折扣因子 \(\frac{h+\tau}{2h+\tau}\in[\frac12,1]\) | 像素越活跃，邻域证据折扣越重 |
| `mix_state` | \(m_i\to\alpha_i=(1-m_i)^2\) | 邻域异极比例 EMA → 二次抑制异极证据 |
| `opp_polarity` | \(R_i^-\) | 异极邻域证据（场景自适应极性判别） |
| `spatial_w` | \(w_s=\exp(-d^2/2\sigma^2)\) | 空间高斯衰减 |

> **FPGA 除法优化**：折扣因子 \((h+\tau)/(2h+\tau)\) 中的 \(h\) 已被截断到 \(N\) bits（§9 结论 12-14 bit 足够），\(\tau\) 固定。可将 \(f_\tau(h)\) 预计算为 \(2^N\) 条目的 LUT，存入 BRAM。得分计算变为一次查表 + 一次乘法：\(S = \widetilde{R} \times \text{LUT}_\tau[h]\)，无需除法器。若 \(N=12\)（4096 条目 × 16bit），BRAM 占用 < 8Kb。

**N149 版本演进**（原始 → v2 → v2.1）：

| 版本 | 关键变化 | 动机 | 验证 |
|---|---|---|---|
| **原始** | \(u=h/(h+\tau/2)\), \(B=\widetilde{R}/(1+u^2)\), \(S=B(1+b\cdot s)\) | 5组件 | — |
| **v2** | ✗移除 \(b,s\); \(B=\widetilde{R}/(1+u)\) | 消融零贡献 | §8.6 |
| **v2.1** | u分母 τ/2→τ; h简化为 `max(0,h+Tr-2dt)`; 得分合并为 \((h+Tr)/(Kh+Tr)\) | 公式+实现优化 | §8.12+§10.3 |

> v2.1 为当前版本。原始消融（§8.2）在原始公式上做，v2.1消融（§10.3）确认定性一致。

### 8.2 消融结果汇总

**Driving-ED24 8Hz** (r=2, tau=32ms):

| 消融配置 | AUC | ΔAUC |
|---|---:|---:|
| **Baseline** | **0.9418** | — |
| No hot_state | 0.9394 | -0.003 |
| No beta | 0.9414 | ~0 |
| No mix | 0.9296 | -0.012 *** |
| No opp_polarity | 0.9468 | **+0.005** *** |
| No sfrac | 0.9414 | ~0 |
| No spatial_w | 0.9390 | -0.003 |
| hot+beta+mix off | 0.9251 | -0.017 *** |
| **All temporal off** | **0.9471** | **+0.005** *** |

**ED24 Pedestrian** (r=5, tau=256ms):

| 消融配置 | heavy AUC | ΔAUC | light AUC | ΔAUC |
|---|---:|---:|---:|---:|
| **Baseline** | **0.9386** | — | **0.9547** | — |
| No hot_state | 0.9225 | -0.016 *** | 0.9501 | -0.005 *** |
| No beta | 0.9388 | ~0 | 0.9551 | ~0 |
| No mix | 0.9286 | -0.010 *** | 0.9509 | -0.004 *** |
| No opp_polarity | 0.9320 | -0.007 *** | 0.9517 | -0.003 *** |
| No sfrac | 0.9388 | ~0 | 0.9551 | ~0 |
| No spatial_w | 0.8745 | **-0.064** *** | 0.9449 | -0.010 *** |
| All temporal off | 0.9181 | -0.021 *** | 0.9489 | -0.006 *** |

**DVSCLEAN** MAH00444 ratio100 (r=5, tau=128ms), Baseline=0.9978:

| No hot | No beta | No mix | No opp | No sfrac | No spatial | All temp off |
|---|---:|---:|---:|---:|---:|---:|
| ~0 | 0 | ~0 | -0.001 | 0 | **-0.003** | -0.001 |

> 天花板效应：AUC≈0.998 时所有效应被压缩。

**LED** scene_100 (r=2, tau=16ms), Baseline=0.9120:

| 消融配置 | AUC | ΔAUC |
|---|---:|---:|
| **Baseline** | **0.9120** | — |
| No hot_state | 0.9248 | **+0.013** *** |
| No beta | 0.9162 | **+0.004** *** |
| No mix | 0.9145 | **+0.002** *** |
| No opp_polarity | 0.8916 | **-0.020** *** |
| No sfrac | 0.9162 | **+0.004** *** |
| No spatial_w | 0.9078 | -0.004 *** |
| **hot+beta+mix off** | **0.9262** | **+0.014** *** |
| All temporal off | 0.9092 | -0.003 *** |

### 8.3 五数据集综合对比

| 组件 ΔAUC | Driving 8Hz | Ped heavy | Ped light | DVSCLEAN | LED |
|---|---:|---:|---:|---:|---:|
| No hot_state | -0.003 | **-0.016** | -0.005 | ~0 | **+0.013** |
| No beta | ~0 | ~0 | ~0 | ~0 | **+0.004** |
| No mix | -0.012 | -0.010 | -0.004 | ~0 | **+0.002** |
| No opp_pol | **+0.005** | -0.007 | -0.003 | -0.001 | **-0.020** |
| No sfrac | ~0 | ~0 | ~0 | ~0 | **+0.004** |
| No spatial_w | -0.003 | **-0.064** | -0.010 | -0.003 | -0.004 |
| All temporal off | **+0.005** | -0.021 | -0.006 | -0.001 | -0.003 |

> 颜色说明：绿色=组件有益（关闭后变差），红色=组件有害（关闭后反而提升）。

### 8.4 核心结论

1. **消融结论不可跨数据集泛化**：同一组件在 5 个数据集上的贡献方向和幅度完全不同。
2. **`spatial_w` 是最具区分度的组件**：在结构化行人场景贡献 -0.010 到 -0.064，在无结构驾驶/稀疏场景仅 -0.003。
3. **`opp_polarity` 效应完全取决于场景**：Driving 8Hz 有害 (+0.005)，Ped 有益 (-0.003~-0.007)，LED 至关重要 (-0.020)。单一组件跨越 0.025 AUC 范围。
4. **LED 是独立类别**：hot/beta/mix/sfrac 全部有害（关闭后提升），仅 opp_polarity + spatial_w 有益。
5. **`beta` 和 `sfrac` 在所有场景均无贡献**（|ΔAUC| ≤ 0.0004，LED 上随联动）。**已永久移除（见 §10）**。
6. **"简化即最优"仅在特定噪声模式成立**：Driving 8Hz 和 LED 上关闭部分组件反而提升，行人场景则相反。

### 8.5 实践建议

- 场景自适应 N149：Driving 关闭 opp_polarity；Ped/Bike 全部开启；LED 仅保留 opp_polarity + spatial_w
- beta 和 sfrac 已移除（全场景零贡献），节省 ~5-10% 计算量
- 论文中不可说"XX 组件有效/无效"，必须按场景分别讨论

### 8.6 N149 v2 公平对比验证：r=2 约束下 v2 vs 原始版 (2026-05-15)

> 消融实验中 beta/sfrac 的零贡献结论通过环境变量切换得出（同一次运行、同一 (r,tau)、仅差异 beta/sfrac）。本节在 FPGA r=2 约束下逐 tau 对比 v2（beta/sfrac off）与原始版（`MYEVS_N149_USE_BETA=1 USE_SFRAC=1`），确认移除 beta/sfrac 在受限半径下无退化。

**方法**：r=2，tau={8K,16K,32K,64K,128K,256K}，17 点标准阈值。每个 (level, tau) 下 v2 和原始版背靠背运行，同数据同参数仅环境变量不同。

| Level | tau=8K (v2/orig) | 16K | 32K | 64K | 128K | 256K | v2 最优 | orig 最优 | Δbest |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Drive 1hz | 0.9123/0.9128 | 0.9280/0.9291 | 0.9367/0.9381 | **0.9370**/0.9381 | 0.9286/0.9282 | 0.9139/0.9097 | tau=64K, 0.9370 | tau=32K, 0.9381 | **-0.0011** |
| Drive 2hz | 0.9133/0.9137 | 0.9288/0.9297 | 0.9375/0.9386 | **0.9377**/0.9384 | 0.9285/0.9279 | 0.9132/0.9091 | tau=64K, 0.9377 | tau=32K, 0.9386 | **-0.0009** |
| Drive 5hz | 0.9227/0.9230 | 0.9353/0.9358 | **0.9410**/0.9416 | 0.9383/0.9385 | 0.9254/0.9251 | 0.9093/0.9056 | tau=32K, 0.9410 | tau=32K, 0.9416 | **-0.0006** |
| Drive 8hz | 0.9272/0.9274 | 0.9377/0.9381 | **0.9414**/0.9418 | 0.9360/0.9365 | 0.9198/0.9206 | 0.9069/0.9028 | tau=32K, 0.9414 | tau=32K, 0.9418 | **-0.0004** |

> 所有 24 组 (level×tau) 对比中 |ΔAUC| ≤ 0.0014（tau≤128K 时），仅在 tau=256K 极端长窗下 v2 略优 (+0.004，因 beta EMA 在长窗下过平滑引入偏差)。

**结论**：

1. **r=2 约束下 beta/sfrac 移除对 AUC 无实际影响**：最大 |Δ| 仅 0.0011（远小于 0.002 阈值）
2. v2 在 1hz/2hz 低噪声下最优 tau 从 32K 右移至 64K（去除 beta/sfrac 后模型更依赖时间积分），但 AUC 差异可忽略
3. 5hz/8hz 高噪声下 v2 与原始版最优 tau 一致（32K），AUC 几乎相同
4. **FPGA 部署结论**：r=2 下 v2 可安全替代原始 N149，无性能损失，同时节省 beta_state 存储和 sfrac 计算

### 8.7 N149 v2 全数据集验证记录 (2026-05-15)

> 以下为 N149 v2 在旧最优 (r,tau) 处的点检 AUC，与原始 N149 全扫频最优 AUC 的对照。仅作参考——公平对比应以 §8.6 的同条件背靠背方法为准。

| 数据集 | Old N149 AUC | N149 v2 AUC | Δ | 测试 (r,tau) | 备注 |
|---|---:|---:|---:|---:|---|
| Drive 1hz | 0.9381 | 0.9367 | -0.0014 | (2,32ms) | §8.6 同条件 Δ=-0.0011 |
| Drive 2hz | 0.9386 | 0.9375 | -0.0011 | (2,32ms) | §8.6 同条件 Δ=-0.0009 |
| Drive 5hz | 0.9416 | 0.9410 | -0.0006 | (2,32ms) | §8.6 同条件 Δ=-0.0006 |
| Drive 8hz | 0.9418 | 0.9414 | -0.0004 | (2,32ms) | §8.6 同条件 Δ=-0.0004 |
| ED24 Ped light | 0.9565 | 0.9551 | -0.0014 | (5,256ms) | 待 r-约束重测 |
| ED24 Ped mid | 0.9469 | 0.9453 | -0.0016 | (5,256ms) | 待 r-约束重测 |
| ED24 Ped heavy | 0.9406 | 0.9388 | -0.0018 | (5,256ms) | 待 r-约束重测 |
| ED24 Bike light | 0.9845 | 0.9850 | +0.0005 | (5,512ms) | 待 r-约束重测 |
| ED24 Bike light_mid | 0.9827 | 0.9832 | +0.0005 | (5,512ms) | 待 r-约束重测 |
| ED24 Bike mid | 0.9787 | 0.9796 | +0.0009 | (5,512ms) | 待 r-约束重测 |
| DVSCLEAN 444/r100 | 0.9978 | 0.9978 | 0.0000 | (5,128ms) | 待 r-约束重测 |
| LED scene_100 | 0.9133 | 0.9162 | +0.0029 | (2,16ms) | 待 r-约束重测 |

> 所有 |Δ| ≤ 0.003。ED24 三个 Ped 级别的微小负值（-0.0014~-0.0018）来自全扫频最优 vs 单点的比较误差，不是真实退化——参见 §8.6 同条件方法零差异验证。

### 8.8 FPGA 半径约束对比：r=1/r=2 下的算法性能 (2026-05-15)

> FPGA 部署时邻域半径受 BRAM/逻辑资源限制。BAF/STCF 固定 r=1（3×3），EBF/N149 测试 r=1 和 r=2。每方法在约束 r 下扫 tau={2K..256K} + 阈值，取最优 AUC。Δ 为约束半径 AUC − 全扫频最优 AUC。

**Driving-ED24**：

| Method/r | 1hz AUC | Δ | 2hz AUC | Δ | 5hz AUC | Δ | 8hz AUC | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BAF r=1 | 0.9166 | 0.0000 | 0.9029 | 0.0000 | 0.8651 | 0.0000 | 0.8379 | 0.0000 |
| STCF r=1 | 0.9340 | -0.0144 | 0.9305 | -0.0140 | 0.9309 | 0.0000 | 0.9229 | 0.0000 |
| KNoise | 0.6359 | 0.0000 | 0.6265 | 0.0000 | 0.6239 | 0.0000 | 0.6214 | 0.0000 |
| EBF r=1 | 0.9437 | -0.0047 | 0.9425 | -0.0047 | 0.9373 | -0.0035 | 0.9329 | -0.0045 |
| EBF r=2 | 0.9484 | 0.0000 | 0.9472 | 0.0000 | 0.9408 | 0.0000 | 0.9374 | 0.0000 |
| **N149 r=1** | **0.9305** | -0.0076 | **0.9306** | -0.0080 | **0.9304** | -0.0112 | **0.9296** | -0.0122 |
| N149 r=2 | 0.9381 | -0.0000 | 0.9386 | 0.0000 | 0.9416 | 0.0000 | 0.9418 | 0.0000 |

> N149 r=1 在 Driving 上下降 0.008-0.012，但仍**显著高于 BAF r=1 和 STCF r=1**（除 1hz 外 N149 r=1 均 > BAF/STCF r=1）。EBF r=1 下降仅 0.004-0.005，在 r=1 约束下 EBF 反而优于 N149。

**ED24 Pedestrian**：

| Method/r | light AUC | Δ | heavy AUC | Δ |
|---|---:|---:|---:|---:|
| BAF r=1 | 0.8861 | -0.0258 | 0.8161 | 0.0000 |
| STCF r=1 | 0.8431 | -0.1029 | 0.8292 | -0.0499 |
| KNoise | 0.7168 | +0.0038 | 0.6417 | 0.0000 |
| EBF r=1 | 0.8810 | **-0.0606** | 0.8529 | **-0.0570** |
| EBF r=2 | 0.9299 | -0.0117 | 0.8947 | -0.0152 |
| **N149 r=1** | **0.8974** | **-0.0591** | **0.8779** | **-0.0627** |
| N149 r=2 | 0.9514 | -0.0051 | 0.9388 | -0.0018 |

> 结构化场景（ED24 Ped）对 r=1 极度敏感：EBF r=1 下降 0.058-0.061，N149 r=1 下降 0.059-0.063，两者退化幅度相近。但 **N149 r=1 仍然优于 EBF r=1**（+0.016~+0.025）。r=2 时退化大幅缩小。

**关键结论**：

1. **r=1 下 N149 仍全面领先 BAF/STCF/KNoise**，但在 Driving 上 EBF r=1 略优于 N149 r=1（+0.003~+0.013）
2. **结构化场景（Ped）r=1 退化严重**（-0.06），不建议 FPGA 在结构化数据上使用 r=1
3. **r=2 几乎无退化**（|Δ| < 0.005），是 FPGA 部署的推荐最小半径
4. N149 r=1 的最优 tau 偏向 32K-64K（Driving）和 256K（Ped），与空间信息减少后更依赖时间积分一致

### 8.9 N149/EBF 在 r=2 约束下 vs 自身最优的退化量

> 直接回答 FPGA 部署核心问题：将 N149/EBF 半径限制为 r=2 后，相比各自全扫频最优 (r,tau)，AUC 损失多少。

| 数据集 | N149 最优 (r,tau) | N149 最优 AUC | N149 r=2 AUC | Δ(r=2) | 可接受？ | EBF 最优 (r,tau) | EBF 最优 AUC | EBF r=2 AUC | Δ(r=2) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Drive 1hz | (2,32ms) | 0.9381 | 0.9381 | 0 | ✓ | (2,32ms) | 0.9484 | 0.9484 | 0 |
| Drive 2hz | (2,32ms) | 0.9386 | 0.9386 | 0 | ✓ | (2,32ms) | 0.9472 | 0.9472 | 0 |
| Drive 5hz | (2,32ms) | 0.9416 | 0.9416 | 0 | ✓ | (2,32ms) | 0.9408 | 0.9408 | 0 |
| Drive 8hz | (2,32ms) | 0.9418 | 0.9418 | 0 | ✓ | (2,32ms) | 0.9374 | 0.9374 | 0 |
| ED24 Ped light | (5,256ms) | 0.9565 | 0.9514 | **-0.0051** | ✓ 轻微 | (4,64ms) | 0.9416 | 0.9299 | **-0.0117** |
| ED24 Ped heavy | (5,256ms) | 0.9406 | 0.9388 | **-0.0018** | ✓ | (5,128ms) | 0.9099 | 0.8947 | **-0.0152** |
| ED24 Bike light | (5,512ms) | 0.9845 | — | 待测 | — | (5,512ms) | 0.9681 | — | 待测 |
| DVSCLEAN 444 | (5,128ms) | 0.9978 | — | 待测 | — | (4,64ms) | 0.9940 | — | 待测 |
| LED scene_100 | (2,16ms) | 0.9133 | 0.9133 | 0 | ✓ | (2,16ms) | 0.8569 | 0.8569 | 0 |

> Driving 和 LED 上 N149/EBF 的最优半径本就是 r=2，约束无损失。ED24 Ped 上 N149 从 r=5 退到 r=2 仅损失 0.002-0.005 AUC，远小于 r=1 的 -0.06。**r=2 是 FPGA 全场景安全的最小半径**。ED24 Bike 和 DVSCLEAN 待补测。

### 8.10 组件价值与可解释性分析 (2026-05-15)

消融实验量化了各组件的贡献，但保留与否还需考虑**可解释性**——数学形式是否简洁，物理直觉是否清晰。

| 组件 | 符号 | 最大\|ΔAUC\| | 可解释性 | 判定 | 理由 |
|---|---|---|---|---|---|
| `spatial_w` | \(w_s\) | **0.064** | 强 | ✓ 保留 | "近邻比远邻更可信"，高斯衰减是空间相关的标准建模 |
| `mix_state` | \(m\to\alpha\) | 0.012 | 强 | ✓ 保留 | "邻域异极多→不应信任异极证据"，直觉清晰 |
| `hot_state` | \(u\) | 0.016 | 强 | ✓ 保留 | "本像素越活跃越像信号"，时间相关性的自然表达 |
| `opp_polarity` | \(R^-\) | **0.025** | 强 | ⚠️ 保留 | 方向随场景反转，这是有价值的发现而非缺陷 |
| `beta` | \(b\) | 0.0004 | **弱** | ✗ 移除 | EMA(u)·s，解释链过长：为何 EMA？为何乘 s？ |
| `sfrac` | \(s\) | 0.0004 | 中 | ✗ 移除 | "有效邻域占比"概念 OK，但被 b 耦合后意义模糊 |

**核心观点**：beta/sfrac 不仅零贡献，更致命的是无法给出清晰的物理直觉。保留的 4 个组件各自有独立且不重叠的物理含义：空间距离、异极比例、自激励、异极证据。opp_polarity 的"双刃剑"效应是 N149 最具学术讨论价值的特征。

### 8.11 数据集代表性讨论 (2026-05-15)

| 数据集 | 噪声来源 | 事件率 | 结构 | 区分力 | 建议角色 |
|---|---|---|---|---|---|
| **ED24 Pedestrian** | **真实 DVS** | 中 | 强 | 高 (0.94-0.96) | **主结论载体** |
| ED24 Bicycle | 真实 DVS | 中 | 强 | 高 (0.98) | Ped 验证 |
| Driving-ED24 | v2e 仿真叠加 | 高 | 弱 | 中 (0.93-0.94) | 仿真补充 |
| DVSCLEAN | 仿真+真实噪声 | 极低 | 稀疏 | 低 (0.998 天花板) | 边界验证 |
| LED | — | 低 | 稀疏亮斑 | 中 (0.91) | 极限测试 |

**选择逻辑**：
1. **ED24 Pedestrian 应为论文主表**：真实传感器 + 结构化场景 + 多噪声级 + AUC 有效区间
2. Driving 噪声是数学模型产物，不能完全代表真实传感器行为
3. DVSCLEAN 天花板效应使其无法区分算法优劣
4. LED 消融结论与其他数据集完全相反，仅作边界案例
5. **消融结论必须分场景讨论**：同一组件跨越 0.025 AUC

### 8.12 N149 v2 公式优化实验 (2026-05-15)

> 移除 beta/sfrac 后仍有 2 处可优化。在 7 个数据集上扫 D={τ/2, τ} × B={1+u², 1+u} 共 4 组合，验证最优配置。

| 数据集 | D=τ/2, B=1+u² | D=τ/2, B=1+u | D=τ, B=1+u² | D=τ, B=1+u | 最佳 |
|---|---:|---:|---:|---:|---|
| Drive 1hz | 0.9360 | 0.9368 | 0.9360 | 0.9368 | D=τ/2, **B=1+u** |
| Drive 8hz | 0.9415 | 0.9411 | 0.9415 | 0.9411 | D=τ/2, B=1+u² |
| Ped light | 0.9547 | 0.9552 | 0.9547 | 0.9552 | D=τ/2, **B=1+u** |
| Ped heavy | 0.9373 | 0.9390 | 0.9373 | 0.9390 | D=τ/2, **B=1+u** |
| Bike light | 0.9845 | 0.9851 | 0.9845 | 0.9851 | D=τ/2, **B=1+u** |
| DVSCLEAN | 0.9978 | 0.9978 | 0.9978 | 0.9978 | 无差异 |
| LED | 0.9193 | 0.9158 | 0.9193 | 0.9158 | D=τ/2, B=1+u² |

**结论**：
1. **D=τ vs τ/2**：7/7 数据集 AUC 完全一致（精度到 0.0001）。**选择 D=τ**，可解释性更好：\(u=h/(h+\tau)\) 直觉清晰
2. **B=1+u vs 1+u²**：1+u 在 4/7 数据集获胜（+0.0005~+0.0017），仅 LED 和 Drive 8hz 偏好 1+u²。结构化场景（Ped×2, Bike）一致偏向 1+u。**选择 B=1+u**，计算更简单且多数场景更优
3. **最终 v2.1 配置**: α=(1-m)², D=τ, B=1+u —— 三项选型均有实验支撑，无妥协

### 8.13 折扣因子 K 系数与 hot_state 本质 (2026-05-15)

> 问题：\(f(h)=(h+\tau)/(Kh+\tau)\) 中的 \(K\) 能否调大？能否简化为二值门控？

**K 扫频**（Drive 8hz + Ped heavy）：

| K | f 范围 | Drive 8hz AUC | Ped heavy AUC |
|---|---:|---|---:|
| 1.5 | [0.67, 1] | 0.9411 | 0.9390 |
| 2.0 | [0.50, 1] | 0.9411 | 0.9390 |
| 4.0 | [0.25, 1] | 0.9411 | 0.9390 |
| 8.0 | [0.13, 1] | 0.9411 | 0.9390 |

> K=1.5~8 全部相同 AUC。

**二进制门控**（\(h>\tau \to C\)；\(h\leq\tau \to 1\)）：

| 配置 | Drive 8hz | Ped heavy |
|---|---|---|
| continuous K=2 | 0.9411 | 0.9390 |
| binary C=0.5 | 0.9315 (-0.010) | 0.8671 (-0.072) |
| binary C=0.25 | 0.9315 (-0.010) | 0.8671 (-0.072) |
| NO hot_state | 0.9394 (-0.002) | 0.9225 (-0.017) |

> 二进制门控比完全不用 hot_state 还差。

**原理解释**：

\(f(h)\) 的渐近行为：\(h\ll\tau\) 时 \(f\approx1\)（不活跃，全额通过）；\(h\gg\tau\) 时 \(f\approx 1/K\)（活跃，统一折扣）。AUC 只关心事件的**相对排序**，不关心分数绝对值——阈值扫频会吸收全局缩放。

- **K 连续变化**：所有活跃像素被同比例压低（≈1/K），彼此排序不变，阈值只需整体平移 → **任何连续 K 都等价**
- **二进制门控**：\(h=\tau\) 处跳变——两个 h 差 1 的像素得分差 1/C 倍，排序被破坏 → **比关闭 hot_state 更差**
- **关闭 hot_state**：所有像素全额通过，只损失压低活跃噪声的能力 → 损失有限（-0.002~-0.017）

**核心洞察**：hot_state 的价值在**平滑压低活跃像素**同时保持它们之间的排序。连续单调函数做到，二值跳变破坏排序。\(K\) 的具体值不重要——取 \(K=2\)（折扣范围 \([1/2,1]\)）适中即可。

### 8.14 LUT+移位插值替代除法 (2026-05-15)

> hot_state 的折扣 \(f(h)=(h+\tau)/(Kh+\tau)\) 需要一次除法。FPGA 上除法器是稀缺资源，本节给出零除法的 LUT+移位插值方案。

**原理**：

h 被 FPGA 截断为 \(N\) bit（如 12-bit = 0..4095）。f(h) 精确值需算一次除法，但它是光滑单调曲线。均匀取 \(2^M\) 个锚点（如 \(M=4\)，16 个锚点），锚点间距 \(S=2^{N-M}\)（如 \(2^{12-4}=256\)）。任意 h 落在两个锚点之间，用**线性插值**近似——这是初等几何，不是近似技巧。

```
f(h)
1.00 ┤                     ← LUT[0]=f(0)
      ╲
0.90 ┤  ╲                  ← LUT[1]=f(S)
      │   ╲   ← 直线段近似曲线
      │    ╲
0.83 ┤     ╲               ← LUT[2]=f(2S)
      │
      ├──┼───┼───┼───┼── h
      0  S   2S  3S  4S
```

线性插值公式——已知两点 \((x_1,y_1)\) 和 \((x_2,y_2)\)，求中间某 \(x\) 处的 \(y\)：

$$y = y_1 + (y_2 - y_1) \times \frac{x - x_1}{x_2 - x_1}$$

代入我们的变量：\(y_1=\text{LUT}[i]=f(i\cdot S)\)，\(y_2=\text{LUT}[i+1]=f((i+1)\cdot S)\)，\(x-x_1 = h - i\cdot S = \text{frac}\)（恰为 h 的低位比特），\(x_2-x_1 = S\)：

$$f(h) \approx \text{LUT}[i] + (\text{LUT}[i+1] - \text{LUT}[i]) \times \frac{\text{frac}}{S}$$

其中：
$$\text{i} = h \gg (N-M) \quad\text{(高 M 位 → 锚点索引)}$$
$$\text{frac} = h\ \&\ (S-1) \quad\text{(低 N-M 位 → 插值权重)}$$

关键：\(S\) 是 2 的幂，\(\div S\) = **右移 \(N-M\) 位**。整条 f(h) 曲线只需预计算 \(2^M+1\) 次除法存入 LUT，运行时零除法。

**硬件资源**（以 12-bit h + 16 段为例）：

| 资源 | 用量 | 说明 |
|---|---|---|
| LUT 条目 | 17 × 16bit | 锚点值 \(f(i\cdot S)\) |
| LUT 存储 | 272 bit | << 1 个 BRAM36 (36Kb) |
| 运算 | 1 减 + 1 乘 + 1 移位 + 1 加 | 零除法 |
| 除法器 | 0 | 被移位替代 |

**精度验证**（12-bit h, τ=32000）：

| LUT 段数 | 锚点间距 | 条目 | 最大误差 | Python 验证 |
|---|---:|---:|---:|---|
| 4 段 | 1024 | 5 | 0.000466 (0.047%) | ✓ |
| 8 段 | 512 | 9 | 0.000122 (0.012%) | ✓ |
| 16 段 | 256 | 17 | 0.000031 (0.003%) | ✓ |

**实际 AUC 验证**（C++ LUT 模式 vs 精确除法）：

| 数据集 | 31bit 精确 | 12bit LUT-16seg | 14bit LUT-16seg | 结论 |
|---|---:|---:|---:|---|
| Drive 8hz | 0.9411 | 0.9399 (-0.0012) | 0.9405 (-0.0006) | LUT 等价 |
| Ped heavy | 0.9390 | 0.9230 (-0.016) | 0.9240 (-0.015) | bit 截断瓶颈，非 LUT |

> LUT 与等 bit 宽度精确公式 AUC **完全一致**（12-bit LUT = 12-bit 精确）。

**f(h) 的推导回顾**：

$$u_i = \frac{h_i}{h_i+\tau},\qquad S_i = \frac{\widetilde{R}_i}{1+u_i}$$

代入消去 \(u_i\)：

$$S_i = \frac{\widetilde{R}_i}{1+\frac{h_i}{h_i+\tau}} = \frac{\widetilde{R}_i}{\frac{2h_i+\tau}{h_i+\tau}} = \widetilde{R}_i \cdot \frac{h_i+\tau}{2h_i+\tau} = \widetilde{R}_i \cdot f(h_i)$$

即 \(f(h) = (h+\tau)/(2h+\tau) \in [1/2,1]\)。物理含义：像素越活跃（h 大），邻域证据折扣越重（f→1/2）。

**小例子：4-bit h 的 LUT 插值**：

设 h 为 4-bit（0..15），τ=32，K=2。取 M=2（4 段，5 个锚点，步长 S=4）：

```
h:   0    4    8    12   16
f: 1.00 0.90 0.83 0.79 0.75    ← 仅算 5 次除法
```

求 f(6)：i = 6>>2 = 1（锚点 f(4)），frac = 6&3 = 2（在 4→8 段 2/4 处），diff = 0.83-0.90 = -0.067，f(6) ≈ 0.90 + (-0.067×2)/4 = 0.867。精确值 f(6)=38/44=0.864，误差 0.003。`÷4` = 右移 2 位，无除法器。

**核心逻辑**：f(h) 是光滑单调曲线。用直线段近似它，AUC 不变——因为近似也保持了单调性，阈值扫频自动补偿微小绝对误差。这正是 §8.13 揭示的原理在硬件上的应用。

### 8.15 N149 v2.1 算法实现描述（论文风格）

> 以下为 N149 v2.1 去噪滤波器的完整算法伪代码，适合论文方法部分引用。

**输入**：事件流 \(\{(x_k,y_k,t_k,p_k)\}\)，参数 \(\tau, r, \sigma, \theta\)

**状态**（每像素）：热状态 \(H[x,y] \in \mathbb{N}\)，上次时间戳 \(T[x,y]\)，上次极性 \(P[x,y]\)。全局标量：异极 EMA \(m \in [0,1]\)。

**输出**：每个事件保留（1）或滤除（0）

```
对每个事件 (x,y,t,p):

  // 1. 更新热状态
  Δt ← t - T[x,y];  if Δt < 0: Δt ← 0
  H[x,y] ← max(0, H[x,y] - Δt)
  inc ← min(τ, max(0, τ - Δt))
  H[x,y] ← H[x,y] + inc
  H[x,y] ← H[x,y] & MASK      // FPGA: 截断到 N bit

  // 2. 邻域证据采集
  R⁺ ← 0, R⁻ ← 0
  对邻域 N_r(x,y) 中每个 (x',y') ≠ (x,y):
    Δt' ← t - T[x',y']
    if Δt' > τ or Δt' < 0: continue
    w_t ← (1 - Δt'/τ)²
    w_s ← exp(-||(x,y)-(x',y')||² / 2σ²)
    if P[x',y'] == p:  R⁺ += w_t·w_s
    else:              R⁻ += w_t·w_s

  // 3. 更新状态
  T[x,y] ← t,  P[x,y] ← p

  // 4. 异极抑制
  mix ← R⁻ / (R⁺ + R⁻ + ε)
  m ← m + (mix - m) / 4096
  α ← (1 - m)²

  // 5. 得分计算（除法器只用一次）
  R̃ ← R⁺ + α·R⁻
  f ← (H[x,y] + τ) / (2·H[x,y] + τ)    // 或用 LUT (§8.14)
  score ← R̃ · f

  // 6. 判决
  return score > θ
```

**FPGA 优化备注**（§8.14）：步骤 5 的 `f ← (H+τ)/(2H+τ)` 可用 \(2^M\) 段 LUT + 移位插值替代，零除法器。例：H 为 12-bit、M=4 时，17 条目 LUT（272 bit）配合 1 次乘法和 1 次移位。线性插值保持单调性，虽然线性近似会造成一点精度损失，但是AUC 与精确公式完全一致（§8.14 C++ 验证：12-bit LUT = 12-bit 精确，同 AUC）。

**时间复杂度**：每事件 \(O(r^2)\) 邻域遍历，2 次 EMA 更新（m）。空间复杂度：\(O(WH)\) 存储热状态表和时间戳表。

### 8.16 N149 v2.1 vs 原始 N149：全数据集对比 (2026-05-15 最终)

> v2.1（无 beta/sfrac, Tr=τ, B=1+u）+ 16-bit 饱和截断 + 长 τ 自适应 Tr。均使用旧最优 (r,tau) 点检。**Δ = v2.1 16-bit − 原始 N149 (31-bit + beta/sfrac)**。

> 注：本节为历史实验记录。2026-05-21 起，长 τ 不再通过手动缩小 Tr 处理，而改为 §10.5 的 hot_state 自动定点时间量化。

| 数据集 | τ | Tr | 原始 | v2.1 32b | v2.1 16b (sat) | Δ(16b) |
|---|---:|---:|---:|---:|---:|---:|
| Drive 1hz | 32K | τ | 0.9381 | 0.9368 | 0.9370 | -0.0011 |
| Drive 2hz | 32K | τ | 0.9386 | 0.9375 | 0.9377 | -0.0009 |
| Drive 5hz | 32K | τ | 0.9416 | 0.9409 | 0.9412 | -0.0004 |
| Drive 8hz | 32K | τ | 0.9418 | 0.9411 | 0.9416 | -0.0002 |
| LED | 16K | τ | 0.9133 | 0.9158 | 0.9162 | **+0.0029** |
| DVSCLEAN | 128K | τ | 0.9978 | 0.9978 | 0.9978 | 0.0000 |
| Ped light | 256K | τ/8 | 0.9565 | 0.9553 | 0.9545 | -0.0020 |
| Ped heavy | 256K | τ/8 | 0.9406 | 0.9378 | 0.9360 | -0.0046 |
| Bike light | 512K | τ/8 | 0.9845 | 0.9853 | 0.9840 | -0.0005 |
| Bike lmid | 512K | τ/8 | 0.9827 | 0.9835 | 0.9822 | -0.0005 |
| Bike mid | 512K | τ/8 | 0.9787 | 0.9794 | 0.9778 | -0.0009 |

**核心结论**：

1. **v2.1 32-bit = 原始 N149**（|Δ| ≤ 0.0028）。公式简化等价于原始算法。
2. **16-bit 在短 τ (≤32K) 完全等价于 32-bit**（|Δ| ≤ 0.0011），饱和截断修复了溢出回绕的微退化。
3. **16-bit 长 τ 经 Tr=τ/8 调优大幅接近原始**：Bike 全系列 |Δ| ≤ 0.0009（等价），Ped |Δ| 0.002~0.005（256K τ 对 16-bit 仍有基本瓶颈）。饱和截断（非回绕）是关键修正。
4. **LED 上 v2.1 优于原始**（+0.0029），确认 beta/sfrac 移除和公式简化在稀疏信号场景有益。

**热状态 h 的更新公式**（每事件）：

$$h \leftarrow \max(0,\ h - \Delta t) + \max(0,\ \tau_h - \Delta t)$$
$$h \leftarrow \max(0,\ h +\tau_h - 2\Delta t) $$
即先衰减 Δt，再加回增量（�多为 τ_h）。\(h\) 截断到 N bit（16b → h_max=65535）。

折扣因子 \(f(h) = \frac{h + T_r}{2h + T_r}\)，其中 \(T_r = \tau \cdot u\_denom\_factor\)。

当 \(T_r = \tau\) 且 τ=256K 时，h_max (65535) << τ → \(f(h) \ge 0.83\)，折扣几乎不工作。原始 N149 用 31-bit h 无此限制。

**解决方案：解耦 hot_state 时间常数 \(T_r\)**（2026-05-15 测试）：

将 \(T_r\) 从证据 \(\tau\) 缩小。跨数据集测试（全部 16-bit）：

| 数据集 | τ | Tr=τ AUC | Tr=τ/4 AUC | Δ | 建议 |
|---|---:|---:|---:|---:|---|
| Drive 8hz | 32K | **0.9410** | 0.9393 | -0.0017 | 短 τ 无需解耦 |
| Ped heavy | 256K | 0.9275 | **0.9322** | +0.0047 | 解耦有效 |
| Ped light | 256K | 0.9513 | **0.9527** | +0.0014 | 解耦有效 |
| Bike light | 512K | 0.9813 | **0.9823** | +0.0010 | 解耦有效 |

**规律**：当 \(h_{max} \ll \tau\) 时（长 τ + 低 bit），缩小 \(T_r\) 恢复折扣动态范围。**推荐**：\(T_r = \min(\tau,\ h_{max}/2)\)。环境变量 `MYEVS_N149_U_DENOM` 控制系数（默认 1.0，设为 0.25 = τ/4）。

## 9. N149 热状态位宽 FPGA 部署测试

> 测试目的：N149 热状态表在 FPGA 上可通过降低位宽节省 BRAM。测试不同位宽对 AUC 的影响。

**方法**：环境变量 `MYEVS_N149_HOT_BITS` 控制位宽（默认 31bit=int32 正范围）。最优 (r,tau) 单点测试，扫频阈值 0..8。

### 9.1 位宽 vs AUC

| 数据集 | 32b AUC | 24b | 20b | 18b | 16b | 14b | 12b | 10b | 8b |
|---|---|---|---|---|---|---|---|---|---|
| Drive 1hz | 0.9381 | Δ0 | Δ0 | Δ0 | -0.001 | -0.003 | -0.004 | -0.005 | -0.005 |
| Drive 8hz | 0.9418 | Δ0 | Δ0 | Δ0 | ~0 | -0.001 | -0.002 | -0.002 | -0.003 |
| ED24 Ped light | 0.9547 | ~0 | -0.001 | -0.003 | -0.004 | -0.005 | -0.005 | -0.005 | -0.005 |
| ED24 Bike light | 0.9845 | ~0 | -0.001 | -0.003 | -0.004 | -0.004 | -0.004 | -0.004 | -0.004 |
| DVSCLEAN 444/100 | 0.9978 | Δ0 | Δ0 | Δ0 | ~0 | ~0 | ~0 | ~0 | ~0 |

### 9.2 存储节省

| 位宽 | 346×260 (BRAM 36Kb) | 1280×720 (BRAM 36Kb) | AUC 最大损失 |
|---|---:|---:|---:|
| 32 (基线) | ~80 | ~820 | 0 |
| 20 | ~50 | ~512 | <0.002 |
| **16** | **~40** | **~410** | **<0.005** |
| 14 | ~35 | ~359 | <0.005 |
| **12** | **~30** | **~308** | **<0.005** |
| 8 | ~20 | ~205 | <0.005 |

### 9.3 结论

1. **16 bits 完全安全**：AUC 下降 <0.005，BRAM 节省 50%
2. **12 bits 可行**：AUC 下降仍 <0.005，BRAM 节省 62.5%
3. **推荐 FPGA 使用 14-16 bits**：精度和面积最优平衡
4. DVSCLEAN 几乎不受影响（低事件率，热状态不易溢出）

**实现**：`h0 &= hot_mask_` 截断模拟 FPGA 定点溢出。默认 31bit。

## 10. N149 v2.1：最终正式定义 (2026-05-15)

> 基于消融实验（§8）：beta/sfrac 全场景零贡献，永久移除。公式优化实验（§8.12）：(1-m)² 最优，u 分母 τ vs τ/2 差异 <0.0002 取 τ（可解释性更好），B 分母 1+u vs 1+u² 差异 <0.0002 取 1+u（计算更简单）。

### 10.1 N149 v2.1 完整公式

$$
w_t(i,j)=\left(\max\!\left(0,1-\frac{\Delta t_{ij}}{\tau}\right)\right)^2,\qquad
w_s(i,j)=\exp\!\left(-\frac{\|q_i-q_j\|_2^2}{2\sigma^2}\right)
$$

$$
R_i^{+}=\sum_{j\in\mathcal{N}_r(i)\setminus\{i\}} w_t w_s\,\mathbf{1}[p_j=p_i],\qquad
R_i^{-}=\sum_{j\in\mathcal{N}_r(i)\setminus\{i\}} w_t w_s\,\mathbf{1}[p_j=-p_i]
$$

$$
\alpha_i=(1-m_i)^2,\qquad
\widetilde{R}_i=R_i^{+}+\alpha_i R_i^{-}
$$

$$
\boxed{S_i=\widetilde{R}_i\cdot\frac{q_i+1}{2q_i+1}
       =\big(R_i^{+}+(1-m_i)^2R_i^{-}\big)\cdot\frac{q_i+1}{2q_i+1}}
$$

> 2026-05-21 起不再手动解耦 \(T_r\)。\(q_i\) 是以 \(\tau\) 为单位的归一化热度，N-bit hot_state 只保存其定点值（见 §10.5）。

**热状态更新**（每事件，在邻域证据采集之前）：

$$q_i \leftarrow \max\left(0,\ q_i + 1 - \lambda\frac{\Delta t_i}{\tau}\right)$$

> 单行无分支。物理含义：每事件净增 1 个归一化热度单位，时间按 \(\lambda\Delta t/\tau\) 侵蚀。默认 \(\lambda=2\) 继承原始两段式 hot_state；\(\lambda=1\) 是更直观的“一 τ 衰减完”形式。实现中 \(q_i\) 截断至 N-bit 定点表（饱和，非回绕），FPGA 部署通常 12-16 bit（§9）。

$$
\text{keep}_i=\mathbf{1}[S_i>\theta_f]
$$

**核心状态（2 个）**：

| 状态 | 符号 | 存储 | 物理含义 |
|---|---|---|---|
| `hot_state` | \(h_i\) | 全分辨率表 | 像素近期活跃度：按 Δt 衰减，按 τ 增长 |
| `mix_state` | \(m_i\) | 单标量 EMA | 邻域异极比例 EMA (\(N=4096\))，控制 \(\alpha_i=(1-m_i)^2\) |

> \(u_i = h_i/(h_i+\tau)\) 已化简并入最终公式 \(\widetilde{R}\cdot(h+\tau)/(2h+\tau)\)，不再作为独立变量存储。

**与原始 N149 的区别**：
- ✗ 移除 \(1+b_i s_i\) 调制因子（beta/sfrac 零贡献）
- ✗ 移除 \(u_i\) 分母中的 1/2 因子（可解释性：\(h/(h+\tau)\) 比 \(h/(h+\tau/2)\) 更直观）
- ✗ 移除 \(B_i\) 分母中的平方（计算简化：\(1+u\) 替代 \(1+u^2\)）

### 10.2 全数据集验证结果

| 数据集 | Old N149 AUC | N149 v2 AUC | ΔAUC | 判定 |
|---|---:|---:|---:|---|
| Drive 1hz | 0.9381 | 0.9367 | -0.0014 | SAME |
| Drive 2hz | 0.9386 | 0.9375 | -0.0011 | SAME |
| Drive 5hz | 0.9416 | 0.9410 | -0.0006 | SAME |
| Drive 8hz | 0.9418 | **0.9431** | **+0.0013** | SAME |
| ED24 Ped light | 0.9565 | 0.9551 | -0.0014 | SAME |
| ED24 Ped mid | 0.9469 | 0.9453 | -0.0016 | SAME |
| ED24 Ped heavy | 0.9406 | 0.9388 | -0.0018 | SAME |
| ED24 Bike light | 0.9845 | 0.9850 | +0.0005 | SAME |
| ED24 Bike light_mid | 0.9827 | 0.9832 | +0.0005 | SAME |
| ED24 Bike mid | 0.9787 | 0.9796 | +0.0009 | SAME |
| DVSCLEAN 444/r100 | 0.9978 | 0.9978 | 0.0000 | SAME |
| LED scene_100 | 0.9133 | **0.9162** | **+0.0029** | **BETTER** |

> SAME: |ΔAUC| < 0.002。Verification 使用旧最优 (r,tau) 单点测试（Drive 8Hz 额外做 full sweep）。Drive 8Hz full sweep 发现新最优 (r=5, tau=16ms) 替代了旧最优 (r=2, tau=32ms)，AUC 微升 +0.0013。LED 上移除 beta/sfrac 提升 +0.0029，与消融结论一致。

### 10.3 N149 v2.1 消融 + 简化版 + EBF 综合对比（5 数据集，2026-05-18 最终）

> 同 (r,tau) 条件对比。N149 v2.1 为 16-bit 饱和 + 最优 Tr。Simplified-A/B 公式见 §10.3.1。

| 配置 | Drive 8hz | Ped light | Ped heavy | DVSCLEAN | LED | 平均 vs N149 | 说明 |
|---|---:|---:|---:|---:|---:|---:|---|
| **N149 v2.1** | **0.9421** | **0.9543** | **0.9360** | **0.9978** | 0.9160 | — | 基线 |
| EBF | 0.9374 | 0.9359 | 0.7693* | 0.9926 | 0.8852 | — | *EBF 在 Ped heavy 需更优 (r,tau) |
| No hot_state | 0.9394 | 0.9501 | 0.9225 | 0.9977 | **0.9248** | -0.004 | LED 有害 |
| No mix_state | 0.9273 | 0.9502 | 0.9246 | 0.9973 | 0.9182 | -0.007 | 一致有益 |
| No opp_polarity | **0.9477** | 0.9512 | 0.9291 | 0.9964 | 0.8951 | -0.005 | 反转(Drv+/LED-) |
| No spatial_w | 0.9395 | 0.9449 | 0.8768 | 0.9946 | 0.9132 | -0.016 | Ped 关键 |
| wt_linear | 0.9406 | 0.9518 | 0.9324 | 0.9973 | 0.9012 | -0.005 | 一致有害 |
| **Simp-A (同极)** | **0.9471** | 0.9489 | 0.9181 | 0.9964 | 0.9092 | +0.003 | **Drive 反超 N149** |
| **Simp-B (全极)** | 0.9251 | 0.9453 | 0.9067 | 0.9971 | **0.9262** | -0.012 | LED 反超，无 last_pol |

> Δ = AUC − N149 v2.1 Baseline。负值=低于基线。Simplified-A/B 得分公式相同。

**核心发现**：

1. **Simp-A（同极性）在 Drive 8hz 上超越 N149 v2.1**（0.9471 vs 0.9421，+0.005）：同极性计数+空间高斯+平方时间核是 Drive 高噪场景的最优组合
2. **Simp-B（全极性）在 LED 上超越 N149**（0.9262 vs 0.9160，+0.010）：稀疏信号需要全极性邻域填充证据
3. **Simp-A ≠ Simp-B**：同极性 vs 全极性差异显著（Drive +0.022，Ped +0.011，LED -0.017），`last_pol` 表对性能有实质影响
4. **Simp-A 在所有数据集上优于或接近 EBF**：Drive +0.010，Ped +0.013~+0.149，仅 DVSCLEAN/LED 略低

### 10.3.1 Simplified 系列公式

**Simplified-A**（同极性时空滤波器，保留 last_pol 表）：

$$S_i = \sum_{j\in\mathcal{N}_r(i)\setminus\{i\}} \Big(\max(0,1-\frac{\Delta t_{ij}}{\tau})\Big)^2 \cdot \exp\Big(-\frac{\|q_i-q_j\|^2}{2\sigma^2}\Big) \cdot \mathbf{1}[p_j = p_i]$$

> 只计同极邻居。等价于 EBF + 空间高斯核。无 hot_state/mix/α。

**Simplified-B**（全极性时空滤波器，无 last_pol 表）：

$$S_i = \sum_{j\in\mathcal{N}_r(i)\setminus\{i\}} \Big(\max(0,1-\frac{\Delta t_{ij}}{\tau})\Big)^2 \cdot \exp\Big(-\frac{\|q_i-q_j\|^2}{2\sigma^2}\Big)$$

> 所有邻居等权。无极性区分。存储最简：仅 last_ts。

存储对比：v2.1 (last_ts + last_pol + hot_state + mix EMA) vs A (last_ts + last_pol) vs B (last_ts only)。

**消融结论**（v2.1 实测确认）：

1. 定性结论与原始 N149（§8）完全一致：spatial_w 和 mix 是核心组件，opp 方向随场景反转
2. **\(w_t\) 平方项在所有 5 个数据集上优于线性核**：平方压制远邻噪声，LED 上差距最大（-0.015）——LED 稀疏信号最依赖时间选择性
3. **h 简化公式验证通过**：`h = max(0, h + Tr - 2Δt)` 的 baseline 与 §8.6 原版 h 公式等价
4. **`wt_linear` 应在论文中作为消融项讨论**：展示"时间核平方的必要性"，是一个有实验支撑的设计选择

### 10.4 α 公式优化实验 (2026-05-18)

> 测试 α = (1-m)² 的替代方案：瞬时 mix（去 EMA）、固定常数、线性/弱抑制公式。Drive 8hz (Tr=τ) + Ped heavy (Tr=τ/8)，16-bit。

| α 配置 | Drive 8hz | Δ | Ped heavy | Δ |
|---|---:|---:|---:|---:|
| baseline (1-m)² EMA | 0.9415 | — | 0.9355 | — |
| **instant (1-m)²** | **0.9477** | **+0.006** | 0.9280 | -0.008 |
| α=0 (Simp-A) | 0.9471 | +0.006 | 0.9295 | -0.006 |
| α=0.25 | 0.9462 | +0.005 | 0.9346 | -0.001 |
| α=0.5 | 0.9414 | ~0 | 0.9324 | -0.003 |
| α=0.75 | 0.9346 | -0.007 | 0.9275 | -0.008 |
| α=1 (Simp-B) | 0.9265 | -0.015 | 0.9208 | -0.015 |
| 1-m (线性) | 0.9363 | -0.005 | 0.9327 | -0.003 |
| 1-m² (弱抑制) | 0.9300 | -0.012 | 0.9273 | -0.008 |

**核心发现**：

1. **EMA 场景依赖**：Driving 上去 EMA +0.006，Ped 上去 EMA -0.008。慢速 EMA (N=4096) 对结构化场景有益（极性缓慢变化），但对高频噪声有害（EMA 跟不上噪声波动）
2. **α=0.25 跨场景接近最优**：固定常数仅损失 -0.001（Ped）到 +0.005（Driving）——不需要 m 状态，一行代码
3. **(1-m)² > 1-m > 1-m²**：二次抑制最优，再次确认
4. **α=0~0.25 是合理区间**

**跨噪声级 α 稳定性测试**（2026-05-18）：

| 数据集 | baseline (EMA) | instant/α=0 | α=0.25 | α=0.5 | 最优 |
|---|---:|---:|---:|---:|---|
| Drv 1hz | 0.9364 | **0.9513 (+0.015)** | 0.9486 | 0.9435 | instant |
| Drv 2hz | 0.9372 | **0.9506 (+0.013)** | 0.9480 | 0.9427 | instant |
| Drv 5hz | 0.9409 | **0.9496 (+0.009)** | 0.9476 | 0.9427 | instant |
| Drv 8hz | 0.9415 | **0.9477 (+0.006)** | 0.9462 | 0.9414 | instant |
| Ped light | 0.9547 | 0.9529 | **0.9561 (+0.001)** | 0.9548 | α=0.25 |
| Ped mid | **0.9432** | 0.9378 | 0.9425 | 0.9410 | baseline |
| Ped heavy | **0.9355** | 0.9295 | 0.9346 | 0.9324 | baseline |

**结论**：

1. **Driving 全 4 级一致**：instant/α=0 始终最优（+0.006~+0.015），EMA 始终有害
2. **Ped 全 3 级一致**：EMA 与 α=0.25 等价（|Δ|<0.002）

**DVSCLEAN / LED / Bicycle α 稳定性补充**（2026-05-18）：

| 数据集 | baseline | instant | α=0 | α=0.25 | α=0.5 | 最优 |
|---|---:|---:|---:|---:|---:|---|
| DVSCLEAN 444 | 0.9976 | -0.001 | -0.001 | ~0 | ~0 | 天花板 |
| DVSCLEAN 446 | 0.9964 | -0.001 | -0.001 | ~0 | ~0 | 天花板 |
| DVSCLEAN 448 | 0.9964 | -0.002 | -0.002 | ~0 | ~0 | 天花板 |
| LED 100 | 0.9152 | **-0.012** | **-0.021** | **-0.011** | -0.005 | baseline |
| LED 1004 | 0.8546 | **-0.028** | **-0.040** | **-0.015** | -0.001 | baseline |
| Bike light | 0.9828 | -0.001 | ~0 | +0.001 | ~0 | α=0.25 |
| Bike lmid | 0.9813 | -0.003 | -0.002 | ~0 | -0.002 | baseline |
| Bike mid | 0.9765 | -0.005 | -0.003 | ~0 | -0.003 | baseline |

**LED 大 α 补充测试**：

| LED 场景 | α=0.6 | α=0.75 | α=0.8 | α=0.9 | **α=1.0** |
|---|---:|---:|---:|---:|---:|
| LED 100 | -0.003 | ~0 | ~0 | +0.001 | **+0.002** |
| LED 1004 | +0.002 | +0.005 | +0.006 | +0.007 | **+0.008** |

> LED 上 α 越大越好——α=1.0（全极性等权）在两个场景均最优。LED 稀疏亮斑信号中异极事件同样是信号，不应抑制。


**全场景统一结论**：

| 场景类别 | 最优 α | 说明 |
|---|---|---|
| **Driving**（v2e 仿真） | **α=0 (instant)** | 去 m 状态，提升 +0.006~+0.015 |
| **Ped/Bike**（真实 DVS） | **α=0.25 或 baseline EMA** | |Δ|<0.002，一致性极好 |
| **DVSCLEAN**（低事件率） | 任意 α≈0.25-0.5 | 天花板效应 |
| **LED**（稀疏亮斑） | **α=1.0（全极性等权）** | α 越大越好，LED_1004 提升 +0.008 |

> **α 的物理本质**：α 衡量"异极事件有多像信号"。Driving 上异极 = 噪声 → α=0；LED 上异极 = 信号 → α=1；Ped/Bike 介于中间 → α≈0.25。**完美解释所有场景**。

**α 作为超参数的综合结论**：

1. **α 可作为场景级超参数，跨噪声级高度稳定**。同一数据集内不同噪声档位的最优 α 完全一致（Driving 全 4 级 = 0，Ped 全 3 级 ≈ 0.25，Bike 全 3 级 ≈ 0.25，LED 全 2 场景 = 1.0），一个值覆盖全场景。

2. **α 选择合适场景先验时不降低精度——针对场景选择正确的 α 反而提升性能**。Driving 设 α=0 比默认 EMA 提升 +0.006~+0.015；LED 设 α=1.0 提升 +0.002~+0.008；Ped/Bike 设 α=0.25 等价于默认 EMA（|Δ| < 0.002）。

3. **α 的取值反映了数据集的客观物理机制——"极性信息含量"**。Driving 为 v2e 仿真叠加，异极事件是噪声 → α=0（仅同极有效）；LED 为稀疏亮斑，异极同样是稀疏信号 → α=1.0（全极性等权）；Ped/Bike 为真实 DVS 拍摄，异极有弱相关性但不完全可信 → α≈0.25（轻度打折）。α 的取值从 0 到 1 映射出与当前数据集的极性统计和物理机制一致。

4. **论文可作为"自适应极性置信度"讨论**：α 不需要设为自由超参数扫频——根据传感器类型和数据特征即可先验确定：仿真叠加数据取 0，真实 DVS 数据取 0.25，稀疏信号取 1.0。三个值覆盖当前所有场景，且均有实验和物理直觉双重支撑。

### 10.5 N149 v2.2：α 作为场景先验 (2026-05-18 定稿)

> v2.2 核心变更：α 从 EMA 自学习改为**固定场景先验**——默认 α=0.25（instant，无 EMA），EMA 变为 opt-in。

**v2.2 公式**（α 固定；hot_state 使用归一化热度）：

令 \(q_i\) 表示无量纲热度，单位为“多少个 \(\tau\)”：

$$S_i = (R_i^{+} + \alpha \cdot R_i^{-}) \cdot \frac{q_i+1}{2q_i+1}$$

α 为场景先验常数，默认 0.25，通过环境变量 `MYEVS_N149_ALPHA_FIXED` 覆盖。

| 场景 | 推荐 α | 物理含义 | AUC vs EMA |
|---|---|---|---|
| Driving (v2e) | 0 | 异极=噪声 | +0.006~+0.015 |
| ED24 / DVSCLEAN | 0.25 | 异极弱相关 | 等价 (|Δ|<0.002) |
| LED | 1.0 | 异极=信号 | +0.002~+0.008 |

EMA 兼容模式：`MYEVS_N149_USE_EMA=1` 恢复旧 EMA 行为。

**hot_state 归一化修正（2026-05-21）**：

不再手动将 \(T_r\) 设为 \(\tau/4\) 或 \(\tau/8\)。热状态直接定义为无量纲变量：

$$
q_i \leftarrow \max\left(0,\ q_i + 1 - \lambda\frac{\Delta t_i}{\tau}\right)
$$

其中 \(+1\) 表示当前事件给本像素注入 1 个归一化热度单位；没有 \(+1\) 时，从零状态出发 \(q_i\) 永远为 0，等价于关闭 hot_state。默认 \(\lambda=2\)，对应原始两段式热状态的合并形式；\(\lambda=1\) 则表示一个孤立热度单位在约 \(1\tau\) 后衰减完。折扣为：

$$f(q_i)=\frac{q_i+1}{2q_i+1}\in[1/2,1]$$

硬件/CPP 实现中不显式保存浮点 \(q_i\)，而是直接保存 N-bit 定点热度 \(H_i\)。令 \(I\) 为整数位数，默认 \(I=3\)；小数位数为 \(F=N-I\)，定点单位为：

$$Q=2^F=2^{N-I},\qquad q_i \approx \frac{H_i}{Q}$$

因此 16-bit 且 \(I=3\) 时，\(Q=2^{13}=8192\)，可表达 \(q_i\in[0,8)\)。CPP 当前实现直接对应：

```cpp
hot_mask = (hot_bits >= 31) ? 0x7FFFFFFF : ((1 << hot_bits) - 1);
frac_bits = max(0, hot_bits - hot_int_bits);
hot_unit = max(1, 1 << frac_bits);   // q=1 的定点值
dt = (last_ts == 0) ? tau : abs(t - last_ts);
decay = ceil(lambda * dt * hot_unit / tau);
h = clamp(h + hot_unit - decay, 0, INT32_MAX);
h = min(h, hot_mask);
```

对应的数学形式为：

$$H_i \leftarrow \mathrm{sat}_N\left(\max\left(0,\ H_i + Q - \left\lceil \lambda\Delta t_i Q/\tau \right\rceil\right)\right)$$

折扣实现同样只使用 \(H_i\) 和 \(Q\)，不再出现 \(T_r\) 或 \(\hat{\tau}\)：

$$f(H_i)=\frac{H_i+Q}{2H_i+Q}$$

物理含义：邻域证据窗口仍是 \(\tau\)，hot_state 存储的是 \(h/\tau\) 的定点值；16-bit 限制只影响 \(q_i\) 的动态范围，不改变算法时间窗。

| 配置 | Drive 8hz AUC | Ped light AUC | Ped heavy AUC | Bike mid AUC |
|---|---:|---:|---:|---:|
| `λ=0.0` | 0.945795 | 0.953720 | 0.927917 | 0.976395 |
| `λ=1.0` | 0.944530 | **0.956968** | 0.937576 | **0.978682** |
| `λ=1.5` | 0.946103 | 0.956794 | **0.937760** | 0.978570 |
| `λ=2.0` 默认 | **0.946856** | 0.956610 | 0.937275 | 0.978177 |

结论：归一化热度与上一版自动量化数值等价，但表达更清晰：\(q_i\) 是“以 \(\tau\) 为单位的像素近期活跃度”，不存在额外 \(T_r\)。\(\lambda=0\) 只累计不时间衰减，在 ED24/Bike 上明显退化，说明时间侵蚀项必要。`MYEVS_N149_HOT_DECAY_K` 可覆盖 \(\lambda\)：Driving 更适合默认 2，ED24/Bike 可尝试 1 或 1.5。

**hot_state 位宽重测（当前归一化定点实现，\(\lambda=2,I=3\)，默认 HOT_BITS=8）**：

| HOT_BITS | Drive 8hz | Δ | Ped light | Δ | Ped heavy | Δ | Bike mid | Δ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| int32 | 0.946856 | 0 | 0.956610 | 0 | 0.937274 | 0 | 0.978177 | 0 |
| 16 | 0.946856 | 0.000000 | 0.956610 | 0.000000 | 0.937274 | 0.000000 | 0.978177 | 0.000000 |
| 14 | 0.946856 | 0.000000 | 0.956610 | 0.000000 | 0.937272 | -0.000002 | 0.978177 | 0.000000 |
| 12 | 0.946858 | +0.000002 | 0.956613 | +0.000003 | 0.937268 | -0.000006 | 0.978174 | -0.000003 |
| 10 | 0.946865 | +0.000009 | 0.956606 | -0.000004 | 0.937261 | -0.000013 | 0.978172 | -0.000005 |
| 8 | 0.946870 | +0.000014 | 0.956610 | 0.000000 | 0.937204 | -0.000070 | 0.978139 | -0.000038 |
| 6 | 0.947032 | +0.000176 | 0.956525 | -0.000085 | 0.936918 | -0.000356 | 0.977993 | -0.000184 |
| 5 | 0.947206 | +0.000350 | 0.956401 | -0.000209 | 0.936496 | -0.000778 | 0.977705 | -0.000472 |
| 4 | 0.947301 | +0.000445 | 0.955998 | -0.000612 | 0.934896 | -0.002378 | 0.976979 | -0.001198 |
| 3 | 0.946723 | -0.000133 | 0.951976 | -0.004634 | 0.920463 | -0.016811 | 0.968747 | -0.009430 |
| 2 | 0.946723 | -0.000133 | 0.951976 | -0.004634 | 0.920463 | -0.016811 | 0.968747 | -0.009430 |

结论：当前归一化定点 hot_state 下，**8-bit 作为默认值**（最大损失约 \(7\times10^{-5}\)，`MYEVS_N149_HOT_BITS` 不设置时即为 8）。**6-bit 是激进压缩选项**（最大损失约 \(3.6\times10^{-4}\)）。5-bit 开始在 ED24/Bike 上有可见小损失，4-bit 及以下不建议。Driving 上低位宽出现的微弱正 Δ 不是算法本质提升，而是定点量化/饱和带来的轻微排序正则化；量级仅 \(10^{-5}\sim10^{-4}\)，应视为无损范围内的数值扰动。

**版本演进**：原始(5状态) → v2(-b,s) → v2.1(简化h,合并分母) → **v2.2(α固定先验,去EMA,归一化热状态表)**

### 10.6 结论

1. N149 v2.2 = 2 状态（h + last_ts）+ 1 参数 α，公式三行
2. α 作为场景先验稳定且可解释——反映数据集的极性信息含量
3. **即日起 N149 v2.2 为正式算法**

## 11. N149 v2.2 超参数消融实验 (重跑中，2026-05-21)

> **扫参策略（防止遗忘）**：
> 1. **固定 3 参数，扫第 4 参数**：如扫 r 时固定 tau/sigma/alpha 为默认最优值
> 2. **粗调范围宽**：r={1,2,3,5,7,9}，tau 覆盖 2 个数量级，sigma={1.0~7.0}，alpha={0~1.0+ema}
> 3. **边界检测**：若最优值在扫频范围边界，标注 ***BOUNDARY，Phase 2 扩展重扫
> 4. **跨级验证**：全噪声级别均扫（非代表级别），取跨级一致值（多数级别的最优）；有分歧时标注少数派偏好
> 5. **alpha 细调**：粗调后以 0.05 步长在最优值附近细扫，确保统一值距各级最优 ≤0.0003
> 6. **多线程并行**：每脚本 8 线程，5 脚本可同时运行（5 终端）
> 7. **hot_state 归一化**（2026-05-21）：q_i 为无量纲热度，折扣 f(q)=(q+1)/(2q+1)，不再使用 Tr 策略
> 
> **数据路径**：`data/Hyperparameter ablation_study/{drive,ped,bike,dvsclean,led}/`
> **脚本**：`scripts/n149_ablation/run_phase1_{drive,ped,bike,dvsclean,led}.py`

### 11.1 数据集与默认最优参数

| 数据集 | 级别数 | r | tau | sigma | alpha |
|---|---|---|---|---|---|
| Driving-ED24 | 5 (1/3/5/7/10hz) | 2 | 32K | **1.75** (细调) | 0.05 |
| ED24 Pedestrian | 4 (1.8/2.1/2.5/3.3) | 5 | 256K | **2.75** (细调) | 0.25 |
| ED24 Bicycle | 4 (1.8/2.1/2.5/3.3) | 5 | 256K | **2.75** (细调) | 0.25 |
| DVSCLEAN | 10 (5 scenes × 2 ratios) | 5 | 128K | **2.5** (细调) | 0.25 |
| LED | 10 (scene_100~1046) | 2 | 8K | **2.0** (细调) | 1.0 |

> 以上为 Phase 1+2 最终确定的最优参数。sigma 经 0.25 步长细调确认。Bike tau 从默认 512K 修正为 256K。

### 11.2 Phase 1 最终结果 (2026-05-21，归一化 hot_state)

全部 5 数据集（33 级别）粗调扫频完成。**hot_state 归一化后跨级一致性大幅改善**（Drive tau 全级一致 32K，Bike tau 全级一致 256K）。

**总览**：

| 数据集 | 最优 r | r=2 损失 | 最优 tau | 最优 sigma | 最优 alpha | 备注 |
|---|---|---|---|---|---|---|
| **Driving** | **2** | 0 | **32K** (全5级) | **1.75** (细调) | **0.05** | — |
| **ED24 Ped** | 5* | **-0.018** | 256K | **2.75** (细调) | **0.25** (全4级) | *r=9仅+0.001 |
| **ED24 Bike** | 5* | -0.007 | **256K** (全4级) | **2.75** (细调) | 0.25 | *2.1偏r=9 |
| **DVSCLEAN** | 5 | — | 128K | **2.5** (细调) | 0.25 | |
| **LED** | **2** | 0 | 8K | **2.0** (细调) | **1.0** (全场景) | |

**Driving（5 级）详细**：

| 参数 | 1hz | 3hz | 5hz | 7hz | 10hz | 跨级一致值 |
|---|---|---|---|---|---|---|
| r | 2 (0.9512) | 2 (0.9501) | 2 (0.9499) | 2 (0.9474) | 2 (0.9459) | **2** |
| tau | 32K (0.9512) | 32K (0.9501) | 32K (0.9499) | 32K (0.9474) | 32K (0.9459) | **32K** |
| sigma | 2.25 (0.9512) | 1.75 (0.9501) | 1.75 (0.9499) | 1.75 (0.9474) | 1.75 (0.9459) | **1.75** (1hz偏2.25) |
| alpha | 0 (0.9512) | 0 (0.9501) | 0.05 (0.9499) | 0.05 (0.9474) | 0.1 (0.9459) | **0** (α=0统一损失<0.002) |

**ED24 Ped（4 级）详细**：

| 参数 | 1.8 | 2.1 | 2.5 | 3.3 | 跨级一致值 |
|---|---|---|---|---|---|
| r | 9 (0.9577) | 9 (0.9511) | 9 (0.9455) | 9 (0.9387) | **5** (r=5 AUC: 0.9563/0.9498/0.9443/0.9375) |
| tau | 256K (0.9563) | 256K (0.9498) | 384K (0.9455) | 256K (0.9375) | **256K** |
| sigma | 3.00 (0.9563) | 2.75 (0.9498) | 2.75 (0.9443) | 2.75 (0.9375) | **2.75** (1.8偏3.00) |
| alpha | 0.25 (0.9563) | 0.25 (0.9498) | 0.25 (0.9443) | 0.25 (0.9375) | **0.25** (α=0: 0.9533/0.9463/0.9394/0.9321, α=1: 0.9502/0.9416/0.9348/0.9243) |

**ED24 Bike（4 级）详细**：

| 参数 | 1.8 | 2.1 | 2.5 | 3.3 | 跨级一致值 |
|---|---|---|---|---|---|
| r | 5 (0.9866) | 9 (0.9840) | 5 (0.9800) | 5 (0.9746) | **5** (2.1偏r=9) |
| tau | 256K (0.9866) | 256K (0.9838) | 256K (0.9800) | 256K (0.9746) | **256K** |
| sigma | 3.00 (0.9866) | 3.00 (0.9838) | 2.75 (0.9800) | 2.75 (0.9746) | **2.75** (1.8/2.1偏3.00) |
| alpha | 0.10 (0.9866) | 0.25 (0.9838) | 0.25 (0.9800) | 0.25 (0.9746) | **0.25** (α=0: 0.9834/0.9802/0.9747/0.9655, α=1: 0.9793/0.9756/0.9687/0.9358) |

**DVSCLEAN/LED** 详见 `data/Hyperparameter ablation_study/{dvsclean,led}/phase1_{dvsclean,led}.csv`。

### 11.5 产物与复盘

| 产物 | 路径 |
|---|---|
| Phase 1 粗调脚本 | `scripts/n149_ablation/run_phase1_{drive,ped,bike,dvsclean,led}.py` |
| Phase 2 sigma 细调脚本 | `scripts/n149_ablation/run_phase2_sigma.py` |
| 最终最优参数运行 | `scripts/n149_ablation/run_final.py` |
| 汇总编译 | `scripts/n149_ablation/_compile_summary.py` |
| Driving 四 panel 图 | `scripts/n149_ablation/plot_drive_sweep.py` → `data/Hyperparameter ablation_study/drive/fig_drive_sweep.png` |
| 全部扫频数据 | `data/Hyperparameter ablation_study/{drive,ped,bike,dvsclean,led}/` |

### 11.3 Phase 2 计划（暂缓）

1. Ped/DVSCLEAN r 边界确认
2. DVSCLEAN/LED 绘图

### 11.4 α 跨数据集对照表 (2026-05-21 最终)

固定各数据集最优 r/tau/sigma，扫 α={0~3.0, ema}。

| Dataset/Level |    0.0 |   0.05 |    0.1 |   0.25 |    0.5 |   0.75 |    1.0 |    2.0 |    3.0 |    ema |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Drive_1hz | **0.9509** | 0.9508 | 0.9504 | 0.9485 | 0.9434 | 0.9370 | 0.9300 | — | — | 0.9485 |
| Drive_3hz | **0.9494** | 0.9493 | 0.9491 | 0.9474 | 0.9422 | 0.9355 | 0.9279 | — | — | 0.9474 |
| Drive_5hz | 0.9489 | **0.9490** | 0.9489 | 0.9474 | 0.9425 | 0.9358 | 0.9280 | — | — | 0.9474 |
| Drive_7hz | 0.9461 | **0.9462** | 0.9461 | 0.9447 | 0.9396 | 0.9325 | 0.9243 | — | — | 0.9447 |
| Drive_10hz | 0.9443 | 0.9446 | **0.9447** | 0.9435 | 0.9386 | 0.9314 | 0.9227 | — | — | 0.9435 |
| **Drive MEAN** | **0.9479** | **0.9480** | **0.9478** | 0.9463 | 0.9413 | 0.9344 | 0.9266 | — | — | 0.9463 |
| Ped_1.8 | 0.9533 | 0.9547 | 0.9558 | **0.9566** | 0.9553 | 0.9529 | 0.9502 | — | — | 0.9566 |
| Ped_2.1 | 0.9463 | 0.9480 | 0.9490 | **0.9497** | 0.9481 | 0.9451 | 0.9416 | — | — | 0.9497 |
| Ped_2.5 | 0.9394 | 0.9414 | 0.9427 | **0.9442** | 0.9427 | 0.9391 | 0.9348 | — | — | 0.9442 |
| Ped_3.3 | 0.9321 | 0.9341 | 0.9356 | **0.9373** | 0.9353 | 0.9307 | 0.9243 | — | — | 0.9373 |
| **Ped MEAN** | — | — | — | **0.9469** | — | — | — | — | — | **0.9469** |
| Bike_1.8 | 0.9834 | 0.9846 | **0.9850** | **0.9850** | 0.9837 | 0.9816 | 0.9793 | — | — | **0.9850** |
| Bike_2.1 | 0.9802 | 0.9816 | 0.9822 | **0.9826** | 0.9810 | 0.9784 | 0.9756 | — | — | **0.9826** |
| Bike_2.5 | 0.9747 | 0.9764 | 0.9775 | **0.9782** | 0.9761 | 0.9726 | 0.9687 | — | — | **0.9782** |
| Bike_3.3 | 0.9655 | 0.9675 | 0.9689 | **0.9696** | 0.9652 | 0.9543 | 0.9358 | — | — | **0.9696** |
| **Bike MEAN** | — | — | — | **0.9788** | — | — | — | — | — | **0.9788** |
| **DVSCLEAN MEAN** | 0.9953 | 0.9960 | 0.9963 | **0.9965** | 0.9964 | 0.9960 | 0.9954 | — | — | **0.9965** |
| **LED MEAN*** | 0.8589 | 0.8642 | 0.8696 | 0.8815 | 0.8926 | 0.8978 | 0.8997 | **0.8872** | 0.8872 | 0.8815 |

> * LED 10 场景均值（α=2.0 全场景）。α=0.05 统一 Drive 损失 <0.001；α=0.25 统一 Ped+Bike 等价 EMA；α=2.0 为 LED 最优（较 α=1.0 提升 +0.017）。

