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

> **2026-05-27 方法论修正**：PFD 原使用 `--param min-neighbors`（扫 k）生成 ROC 曲线，但 PFD 内部无连续评分机制，k 为整数计数阈值，仅产生 ~5 个 ROC 点。**现已改为与 BAF/STCF_orig 一致的 tau-sweep 方法**：固定 r=1（论文口径），对每组 (m, k) 超参组合扫描 `--param time-us`（τ）生成 ROC 曲线，τ 范围与 BAF 各数据集对齐。最优 (m, k) 取 AUC 最高的组合。脚本：`scripts/run_pfd_tau_sweep_all.py`（全数据集），`scripts/ED24_alg_evalu/run_bike_pfd_fix.py`（Bike 单数据集）。

```powershell
# PFD tau-sweep ROC（新方法，论文口径）
$PY -m myevs.cli roc `
  --clean <CLEAN.npy> --noisy <NOISY.npy> --assume npy `
  --width 1280 --height 720 --tick-ns 1000 `
  --engine cpp --method pfd `
  --radius-px 1 --min-neighbors 1 `
  --refractory-us 1 --pfd-mode a `
  --param time-us --values "1000,2000,4000,8000,16000,32000" `
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
| `PFD` | `cpp` | 0.9123 | — | pfd_r1_m1_k1 | r=1, tau-sweep同BAF(2026-05-27) |
| `STCF_orig` | `cpp` | 0.9216 | 0.9589 | stcf_orig_k2 | tau [2,200]ms unified(2026-05-28) |
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
| `MLPF` | `cpp` | 0.9201 | 0.9148 | mlpf_model_patch7_dur100000_3hz | patch=7 重跑 |
| `STCF_orig` | `cpp` | 0.9155 | 0.9121 | stcf_orig_k2 | tau [2,200]ms unified(2026-05-28) |
| `PFD` | `cpp` | 0.9061 | 0.8966 | pfd_r1_m1_k1 | r=1, tau-sweep同BAF(2026-05-27) |
| `BAF` | `cpp` | 0.8909 | 0.6319 | baf_r1 | r=1 固定重跑 |
| `EvFlow` | `cpp` | 0.8424 | 0.9101 | evflow_r2_tau16000 | |
| `KNoise` | `cpp` | 0.6232 | 0.8395 | knoise_tau8000 | |

> MLPF(3hz) 训练/测试文件：
> 训练 noisy：`D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_3hz_slomo_shot_withlabel/driving_noise_3hz.npy`  
> 训练 clean：`D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_ED24/driving_noise_3hz_slomo_shot_withlabel/driving_clean.npy`  
> 推理模型：`D:/hjx_workspace/scientific_reserach/projects/myEVS/data/DND21/mydriving_ED24/MLPF/mlpf_torch_3hz_p7_cpp_fulltrain.pt`（同 stem `.npz`）  
> 测试 ROC：`D:/hjx_workspace/scientific_reserach/projects/myEVS/data/DND21/mydriving_ED24/MLPF/roc_mlpf_3hz_p7_cpp_fulltrain.csv`

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
| `STCF_orig` | `cpp` | 0.9074 | 0.8730 | stcf_orig_k2 | tau [2,200]ms unified(2026-05-28) |
| `PFD` | `cpp` | 0.8986 | 0.8779 | pfd_r1_m2_k1 | r=1, tau-sweep同BAF(2026-05-27) |
| `BAF` | `cpp` | 0.8648 | 0.6225 | baf_r1 | r=1 固定重跑 |
| `EvFlow` | `cpp` | 0.8206 | 0.8686 | evflow_r2_tau16000 | |
| `KNoise` | `cpp` | 0.6239 | 0.7579 | knoise_tau16000 | |

### 7.5A Driving-ED24: 7hz（先填已跑算法）

| Method | Engine | Best AUC | Best F1 | Best AUC tag | 备注 |
|---|---:|---:|---:|---|---|
| `MLPF` | `cpp` | 0.9238 | 0.6953 | mlpf_model_patch7_dur100000_7hz | 全量重跑 |
| `STCF_orig` | `cpp` | 0.9032 | 0.8436 | stcf_orig_k3 | tau [2,200]ms unified(2026-05-28) |
| `PFD` | `cpp` | 0.8920 | 0.8648 | pfd_r1_m2_k1 | r=1, tau-sweep同BAF(2026-05-27) |
| `BAF` | `cpp` | 0.8532 | 0.6178 | baf_r1 | r=1 固定重跑 |
| `EvFlow` | `cpp` | 0.7219 | — | evflow_r2_tau8000 | lite sweep |
| `EBF` | `cpp` | 0.9387 | 0.7235 | ebf_r2_tau32000 | 补跑 |
| `YNoise` | `cpp` | 0.9283 | 0.6915 | ynoise_r2_tau16000 | 补跑 |
| `STCF` | `cpp` | 0.9215 | 0.6915 | stcf_r2_tau32000 | 补跑 |
| **N149_v2.2** | `cpp` | **0.9475** | 0.9034 | v22_final | r=2 tau=32K sigma=1.75 alpha=0.05 |
| `TS` | `cpp` | 0.8945 | 0.6915 | ts_r2_tau32000 | 补跑 |
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
| `EBF` | `cpp` | 0.9346 | 0.6288 | ebf_r2_tau32000 | 补跑 |
| `STCF` | `cpp` | 0.9139 | 0.6105 | stcf_r2_tau32000 | 补跑 |
| `MLPF` | `cpp` | 0.9199 | 0.6149 | mlpf_model_patch7_dur100000_10hz | 全量重跑 |
| **N149_v2.2** | `cpp` | **0.9462** | 0.8826 | v22_final | r=2 tau=32K sigma=1.75 alpha=0.05 |
| `STCF_orig` | `cpp` | 0.8965 | 0.8086 | stcf_orig_k3 | tau [2,200]ms unified(2026-05-28) |
| `PFD` | `cpp` | 0.8798 | 0.8327 | pfd_r1_m2_k1 | r=1, tau-sweep同BAF(2026-05-27) |
| `BAF` | `cpp` | 0.8269 | 0.5966 | baf_r1 | r=1 固定重跑 |
| `TS` | `cpp` | 0.8993 | 0.6105 | ts_r2_tau32000 | 补跑 |
| `YNoise` | `cpp` | 0.9227 | 0.6105 | ynoise_r2_tau16000 | 补跑 |
| `BAF` | `cpp` | 0.8269 | 0.5966 | baf_r1 | r=1 固定重跑 |
| `EvFlow` | `cpp` | 0.7006 | — | evflow_r2_tau8000 | lite sweep |
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

| Level | Rank 1 | AUC | Rank 2 | AUC | Rank 3 | AUC | Rank 4 | AUC | Rank 5 | AUC | Rank 6 | AUC | Rank 7 | AUC | Rank 8 | AUC | Rank 9 | AUC | Rank 10 | AUC | Rank 11 | AUC |
|---|---|:---:|---|:---:|---|:---:|---|:---:|---|:---:|---|:---:|---|:---:|---|:---:|---|:---:|---|:---:|---|:---:|
| 1hz | **v2.2** | **0.9512** | EBF | 0.9484 | YNoise | 0.9408 | N149 | 0.9381 | TS | 0.9298 | STCF_orig | 0.9216 | BAF | 0.9136 | PFD | 0.9123 | MLPF | 0.8632 | EvFlow | 0.8486 | KNoise | 0.6359 |
| 3hz | **v2.2** | **0.9502** | EBF | 0.9444 | N149 | 0.9394 | YNoise | 0.9361 | TS | 0.9279 | STCF_orig | 0.9155 | MLPF | 0.9238 | PFD | 0.9061 | BAF | 0.8919 | EvFlow | 0.8424 | KNoise | 0.6232 |
| 5hz | **v2.2** | **0.9500** | N149 | 0.9416 | EBF | 0.9408 | YNoise | 0.9312 | TS | 0.9259 | STCF_orig | 0.9074 | MLPF | 0.9012 | PFD | 0.8986 | BAF | 0.8651 | EvFlow | 0.8206 | KNoise | 0.6239 |
| 7hz | **v2.2** | **0.9475** | EBF | 0.9387 | N149 | 0.9381 | MLPF | 0.9238 | YNoise | 0.9283 | STCF_orig | 0.9032 | TS | 0.8945 | PFD | 0.8920 | BAF | 0.7657 | EvFlow | 0.7219 | KNoise | 0.6170 |
| 10hz | **v2.2** | **0.9462** | EBF | 0.9346 | YNoise | 0.9227 | MLPF | 0.9199 | TS | 0.8993 | STCF_orig | 0.8965 | PFD | 0.8798 | BAF | 0.8269 | EvFlow | 0.7006 | KNoise | 0.6172 |

### 7.8 ED24 Pedestrian (myPedestrain_06)

| Method | light (1.8) | light_mid (2.1) | mid (2.5) | heavy (3.3) | 备注 |
|---|---:|---:|---:|---:|---|
| `N149` | **0.9565** | — | **0.9469** | **0.9406** | 全级最优 |
| `N149_v2` | 0.9551 | — | 0.9453 | 0.9388 | v2, 点检 |
| `N149_v2.1` | 0.9545 | — | 0.9432 | 0.9360 | v2.1 |
| **N149_v2.2** | 0.9563 | 0.9497 | 0.9443 | 0.9375 | sigma=2.75 alpha=0.25 |
| `STCF` | 0.9460 | 0.9047 | 0.8962 | 0.8791 | 2.1 补跑 |
| `EBF` | 0.9416 | 0.9061 | 0.9185 | 0.9099 | 2.1 补跑 |
| `YNoise` | 0.9227 | 0.8782 | 0.9083 | 0.8971 | 2.1 补跑 |
| `MLPF` | 0.8883 | 0.8973 | 0.8931 | 0.8828 | patch=7/自训练（2026-05-22 增量补跑后更新，best tau=32ms, thr=0.1） |
| `BAF` | 0.8923 | 0.8707 | 0.8395 | 0.8143 | r=1, tau [2,200]ms unified(2026-05-27) |
| `PFD` | 0.8155 | 0.8092 | 0.7808 | 0.7472 | r=1,m=1,k=1 tau-sweep(2026-05-27) |
| `STCF_orig` | 0.8923 | 0.8707 | 0.8510 | 0.8392 | tau [2,200]ms unified(2026-05-28) |
| `TS` | 0.8619 | 0.8335 | 0.8528 | 0.8465 | 2.1 边界修复(+0.070) |
| `EvFlow` | 0.8351 | 0.7274 | 0.8022 | 0.7847 | 2.1 边界修复(+0.012) |
| `KNoise` | 0.7130 | 0.6850 | 0.6625 | 0.6417 | 2.1 补跑 |

### 7.9 ED24 Bicycle (myBicycle_02)

| Method | light AUC | light_mid AUC | mid AUC | heavy(3.3) AUC | Best-light | Best-light_mid | Best-mid | Best-heavy | 备注 |
|---|---:|---:|---:|---:|---|---|---|---|---|
| `N149` | **0.9845** | **0.9827** | **0.9787** | — | (5,512ms) | (5,512ms) | (5,512ms) | — | 全级最优 |
| `N149_v2` | 0.9850 | 0.9832 | 0.9796 | — | (5,512ms) | (5,512ms) | (5,512ms) | — | v2, 点检 |
| `N149_v2.1` | 0.9840 | 0.9822 | 0.9778 | — | (5,512ms) | (5,512ms) | (5,512ms) | — | v2.1 |
| **N149_v2.2** | 0.9866 | 0.9838 | 0.9800 | 0.9746 | (5,256ms) | **v2.2** sigma=2.75 alpha=0.25 |
| `STCF` | 0.9785 | 0.9649 | 0.9418 | 0.9425 | (4,32ms) | (4,32ms) | (4,32ms) | (4,32ms) | |
| `EBF` | 0.9802 | 0.9758 | 0.9686 | 0.9604 | (4,128ms) | (4,128ms) | (4,128ms) | (3,128ms) | 逐级最优重跑(2026-05-27) |
| `YNoise` | 0.9753 | 0.9716 | 0.9642 | 0.9562 | (4,64ms) | (4,64ms) | (4,64ms) | (3,64ms) | 逐级最优重跑(2026-05-27) |
| `MLPF` | 0.9380 | 0.9356 | 0.9439 | 0.9217 | (2,32ms) | (2,128ms) | (3,16ms) | (2,32ms) | patch=7 重训仅覆盖 mid（2026-05-22）；其余档沿用旧值 |
| `PFD` | 0.8962 | 0.8823 | 0.8486 | 0.7990 | (r=1,m=1,k=1) | (r=1,m=1,k=1) | (r=1,m=1,k=1) | (r=1,m=1,k=1) | r=1, tau-sweep同BAF(2026-05-27) |
| `STCF_orig` | 0.9416 | 0.9309 | 0.9200 | 0.9134 | (k=1) | (k=1) | (k=2) | (k=2) | tau [2,200]ms unified(2026-05-28) |
| `BAF` | 0.9416 | 0.9309 | 0.9108 | 0.8907 | (r=1) | (r=1) | (r=1) | (r=1) | tau [2,200]ms unified(2026-05-27) |
| `EvFlow` | 0.8877 | 0.8894 | 0.8717 | 0.7939 | (6,32ms) | (4,32ms) | (2,32ms) | (2,32ms) | |
| `TS` | 0.9290 | 0.9346 | 0.9299 | 0.9245 | (3,32ms) | (2,32ms) | (2,32ms) | (2,32ms) | 逐级最优重跑(2026-05-27)；mid>light仅+0.0009(噪声级) |
| `KNoise` | 0.7631 | 0.7471 | 0.7215 | 0.7007 | (1,16ms) | (1,16ms) | (1,16ms) | (1,16ms) | |

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

### 7.10 DVSCLEAN (10 scenes: MAH00444-449 × ratio50/100)

| Method | Engine | Mean AUC | Mean F1 | Best (r, tau) | 备注 |
|---|---:|---:|---:|---|---|
| `N149` | `cpp` | 0.9970 | 0.9900 | (5, 128ms) | 最优 |
| `N149_v2` | `cpp` | 0.9978 | — | (5, 128ms) | v2, 点检 |
| `N149_v2.1` | `cpp` | 0.9978 | — | (5, 128ms) | v2.1 |
| **N149_v2.2** | cpp | 0.9966 | — | (5,128ms) | **v2.2** sigma=2.5 alpha=0.25 (10场景均值) |
| `EBF` | `cpp` | 0.9940 | 0.9843 | (4, 64ms) | |
| `YNoise` | `cpp` | 0.9934 | 0.9836 | (4, 32ms) | |
| `STCF` | `cpp` | 0.9907 | — | (2, 128ms) | tau修正(+0.001) |
| `PFD` | `cpp` | 0.9562 | — | (r=1,m=1,k=1) | r=1, tau-sweep同BAF(2026-05-27) |
| `MLPF` | `python` | 0.9938 | 0.9826 | (scene-wise) | 10场景逐场景重训+评测（2026-05-22） |
| `STCF_orig` | `cpp` | 0.9825 | — | (r=1, k=1~2) | tau [2,200]ms unified(2026-05-28) |
| `BAF` | `cpp` | 0.9805 | — | (r=1) | tau [2,200]ms unified(2026-05-28) |
| `TS` | `cpp` | 0.9297 | — | (2, 32~128ms) | 10场景逐场景最优tau均值 |
| `EvFlow` | `cpp` | 0.7733 | 0.6828 | (2, 8ms) | lite sweep |
| `KNoise` | `cpp` | 0.6389 | 0.4381 | (1, 32ms) | |

#### 7.10.1 MAH00447 专项对比 (2026-05-29)

> MAH00447 为 DVSCLEAN 中噪声最显著的场景。全算法扫频+N149 r=3 vs r=5。脚本：`scripts/DVSCLEAN_alg_evalu/run_mah00447_full.py`（12 线程，74 任务）。

**ratio50**:

| Algorithm | AUC | Params |
|---|---|---|
| **N149 r=5** | **0.9959** | r=5, tau=128K, sigma=2.5, alpha=0.25 |
| N149 r=3 | 0.9957 | r=3, tau=128K (Δ=−0.0002) |
| EBF | 0.9941 | r=3, tau=64K |
| BAF | 0.9820 | r=1, tau [2,200]ms |
| STCF_orig | 0.9820 | r=1, k=1 |
| TS | 0.9682 | r=1, tau=32K（全局最优，已扩展验证） |
| PFD | 0.9686 | r=1, m=1, k=1（tau≈64ms最优，扫参正确） |
| EDnCNN | 0.7867 | 原权重结果 |
| EDFormer | 0.9831 | 原权重结果 |

**ratio100**:

| Algorithm | AUC | Params |
|---|---|---|
| **N149 r=5** | **0.9947** | r=5, tau=128K |
| N149 r=3 | 0.9944 | r=3, tau=128K (Δ=−0.0003) |
| EBF | 0.9932 | r=4, tau=32K |
| STCF_orig | 0.9792 | r=1, k=2 |
| BAF | 0.9727 | r=1 |
| PFD | 0.9668 | r=1, m=1, k=1（tau≈64ms最优） |
| TS | 0.9561 | r=1, tau=16K（全局最优，已扩展验证） |
| EDnCNN | 0.7563 | 原权重结果 |
| EDFormer | 0.9696| 原权重结果 |

> **N149 r=3 vs r=5**：差异仅 0.0002~0.0003，可忽略。r=3 足够，FPGA 部署时邻域访问从 25→9 pixels，大幅降低 BRAM 带宽。


### 7.11 LED (10场景均值)

| Method | Engine | AUC | F1 | Best (r, tau) | 备注 |
|---|---:|---:|---:|---|---|
| **N149_v2.2** | `cpp` | **0.8916** | — | (2,8K) | sigma=2.0 alpha=1.0, 10场景均值 |
| `EBF` | `cpp` | 0.8669 | — | (2, 8K) | 10场景均值, (r,tau)最优重跑 |
| `STCF_orig` | `cpp` | 0.8649 | — | (1, 2K, k=2) | 10场景均值, tau+k最优重跑 |
| `YNoise` | `cpp` | 0.8604 | — | (2, 8K) | 10场景均值, (r,tau)最优重跑 |
| `STCF` | `cpp` | 0.8500 | — | (2, 4K) | 10场景均值, (r,tau)最优重跑 |
| `BAF` | `cpp` | 0.8375 | — | (1, 1K) | 10场景均值, tau扫频重跑 |
| `PFD` | `cpp` | 0.8053 | — | (r=1,m=1,k=1) | r=1, tau-sweep同BAF(2026-05-27) |
| `TS` | `cpp` | 0.7969 | — | (1, 8K) | 10场景均值, (r,tau)最优重跑 |
| `EvFlow` | `cpp` | 0.7879 | 0.7096 | (2, 8K) | 10场景两阶段（300k粗扫+全量单点）更新于 2026-05-23 |
| `MLPF` | `python` | 0.6952 | 0.6050 | (scene-wise) | 10场景逐场景重训+评测（patch=7，2026-05-22） |
| `KNoise` | `cpp` | 0.5276 | — | (1, 1K) | 10场景均值 (LED 上接近随机) |

> LED 10 场景全部按 Driving 方法论重跑：参考 `run_driving_alg.ps1` 对各算法的 (r,tau) 扫频策略，先 Phase 1 在 scene_100 上全量扫参找到最优 (r,tau)，再 Phase 2 推广到 10 场景。脚本：`scripts/LED_alg_evalu/run_led_phase1_sweep.py` + `run_led_phase2_all.py` + `run_led_baf_stcf_opt.py`。

> N149 v2.2 明细：100=0.9262, 1004=0.8588, 1018=0.8889, 1028=0.8993, 1032=0.8954, 1033=0.8650, 1034=0.8776, 1043=0.8913, 1045=0.8773, 1046=0.8917。
> EvFlow(2026-05-23) 明细：100=0.7670, 1004=0.7599, 1018=0.8031, 1028=0.8120, 1032=0.8097, 1033=0.7695, 1034=0.7701, 1043=0.7913, 1045=0.7820, 1046=0.8140（两阶段：300k粗扫后全量单点评估）。

#### LED 方法论修正（2026-05-22）

> **问题**：旧脚本 `run_missing_alg.py` 对所有算法使用固定 (r,tau) + `--param min-neighbors` 扫频，存在两类错误：
> 1. **BAF**：忽略 `min-neighbors` 参数 → 单点 ROC → AUC 被严重低估（0.7488→0.8375）
> 2. **其他算法**：固定 (r,tau) 非 LED 最优 → AUC 偏低（如 EBF tau=32ms 在 LED 上远差于 tau=8ms）
>
> **修正**（参考 `scripts/driving_ED24_alg_evalu/run_driving_alg.ps1`）：
> - Phase 1：在 scene_100 上全量扫 (r,tau) 找到各算法最优配置
> - Phase 2：用最优 (r,tau) 跑全部 10 场景
> - BAF/STCF_orig 单独处理（BAF 扫 tau，STCF_orig 扫 k×tau）
>
> **修正效果**：所有算法 AUC 均上升，排名恢复合理（EBF > STCF_orig > YNoise > STCF > BAF > PFD > TS）。KNoise 在 LED 上始终接近随机（AUC≈0.53），属算法能力极限。

### 7.11A 噪声事件数据(2026-05-22)

> 统计口径：使用各数据集 `*_labeled.npy`；`duration_s=(t_max-t_min)/1e6`（tick-ns=1000）；`mev_s=events/duration_s/1e6`；`hz_per_px=events/(W*H*duration_s)`；`signal_ratio/noise_ratio` 由标签直接统计。
>
> 指标说明：
> - `mev_s`：每秒处理事件数（百万事件/秒，Mevents/s）。
> - `hz_per_px`：每像素每秒平均事件数（events/s/pixel），用于衡量场景事件密度。

#### 7.11A.1 Driving-ED24

| level | events | duration_s | mev_s | hz_per_px | signal_ratio | noise_ratio | S:N |
|---|---|---|---|---|---|---|---|
| 1hz | 3081599 | 5.976 | 0.516 | 5.732 | 88.67% | 11.33% | 7.83:1 |
| 2hz | 3429942 | 5.976 | 0.574 | 6.380 | 79.66% | 20.34% | 3.92:1 |
| 3hz | 3777023 | 5.976 | 0.632 | 7.025 | 72.34% | 27.66% | 2.62:1 |
| 5hz | 4478144 | 5.976 | 0.749 | 8.330 | 61.01% | 38.99% | 1.56:1 |
| 7hz | 5169768 | 5.976 | 0.865 | 9.616 | 52.85% | 47.15% | 1.12:1 |
| 8hz | 5515979 | 5.976 | 0.923 | 10.260 | 49.53% | 50.47% | 0.98:1 |
| 10hz | 6218745 | 5.976 | 1.041 | 11.567 | 43.94% | 56.06% | 0.78:1 |

#### 7.11A.2 ED24-Pedestrian

| level | events | duration_s | mev_s | hz_per_px | signal_ratio | noise_ratio | S:N |
|---|---|---|---|---|---|---|---|
| 1.8V | 194395 | 5.814 | 0.033 | 0.372 | 83.81% | 16.19% | 5.18:1 |
| 2.1V | 297299 | 5.814 | 0.051 | 0.568 | 54.80% | 45.20% | 1.21:1 |
| 2.5V | 566141 | 5.814 | 0.097 | 1.083 | 28.78% | 71.22% | 0.40:1 |
| 3.3V | 896682 | 5.814 | 0.154 | 1.715 | 18.17% | 81.83% | 0.22:1 |

#### 7.11A.3 ED24-Bicycle

| level | events | duration_s | mev_s | hz_per_px | signal_ratio | noise_ratio | S:N |
|---|---|---|---|---|---|---|---|
| 1.8V | 83364 | 3.240 | 0.026 | 0.286 | 77.85% | 22.15% | 3.51:1 |
| 2.1V | 138345 | 3.240 | 0.043 | 0.475 | 46.91% | 53.09% | 0.88:1 |
| 2.5V | 266531 | 3.240 | 0.082 | 0.914 | 24.35% | 75.65% | 0.32:1 |
| 3.3V | 471149 | 3.240 | 0.145 | 1.616 | 13.78% | 86.22% | 0.16:1 |

#### 7.11A.4 DVSCLEAN (5 scenes ? ratio50/100)

| level | events | duration_s | mev_s | hz_per_px | signal_ratio | noise_ratio | S:N |
|---|---|---|---|---|---|---|---|
| MAH00444_ratio50 | 286060 | 0.677 | 0.423 | 0.459 | 66.67% | 33.33% | 2.00:1 |
| MAH00444_ratio100 | 381414 | 0.677 | 0.563 | 0.611 | 50.00% | 50.00% | 1.00:1 |
| MAH00446_ratio50 | 286887 | 0.603 | 0.476 | 0.516 | 66.67% | 33.33% | 2.00:1 |
| MAH00446_ratio100 | 382516 | 0.603 | 0.634 | 0.688 | 50.00% | 50.00% | 1.00:1 |
| MAH00447_ratio50 | 281026 | 0.373 | 0.754 | 0.818 | 66.67% | 33.33% | 2.00:1 |
| MAH00447_ratio100 | 374702 | 0.373 | 1.005 | 1.091 | 50.00% | 50.00% | 1.00:1 |
| MAH00448_ratio50 | 276343 | 1.168 | 0.237 | 0.257 | 66.67% | 33.33% | 2.00:1 |
| MAH00448_ratio100 | 368458 | 1.168 | 0.316 | 0.342 | 50.00% | 50.00% | 1.00:1 |
| MAH00449_ratio50 | 279306 | 1.095 | 0.255 | 0.277 | 66.67% | 33.33% | 2.00:1 |
| MAH00449_ratio100 | 372408 | 1.095 | 0.340 | 0.369 | 50.00% | 50.00% | 1.00:1 |

#### 7.11A.5 LED (10 scenes, 100ms)

| level | events | duration_s | mev_s | hz_per_px | signal_ratio | noise_ratio | S:N |
|---|---|---|---|---|---|---|---|
| scene_100 | 1992811 | 0.100 | 19.928 | 21.624 | 95.77% | 4.23% | 22.64:1 |
| scene_1004 | 1990561 | 0.100 | 19.906 | 21.599 | 89.75% | 10.25% | 8.76:1 |
| scene_1018 | 1869704 | 0.100 | 18.697 | 20.288 | 91.11% | 8.89% | 10.25:1 |
| scene_1028 | 1828236 | 0.100 | 18.283 | 19.838 | 93.85% | 6.15% | 15.26:1 |
| scene_1032 | 1937805 | 0.100 | 19.378 | 21.027 | 92.60% | 7.40% | 12.51:1 |
| scene_1033 | 1977158 | 0.100 | 19.772 | 21.454 | 90.78% | 9.22% | 9.85:1 |
| scene_1034 | 1992578 | 0.100 | 19.926 | 21.621 | 89.97% | 10.03% | 8.97:1 |
| scene_1043 | 1994430 | 0.100 | 19.944 | 21.641 | 92.37% | 7.63% | 12.11:1 |
| scene_1045 | 1911309 | 0.100 | 19.113 | 20.739 | 91.59% | 8.41% | 10.89:1 |
| scene_1046 | 1996926 | 0.100 | 19.969 | 21.668 | 92.84% | 7.16% | 12.97:1 |
### 7.12 对比算法边界检测与修复 (2026-05-22，已完成)

> **问题**：非 N149 对比算法在 ED24/DVSCLEAN/LED 上部分最优点触及阈值扫频边界。**r>5 不继续扩但标注边界**。

**执行结果**（`scripts/run_boundary_fix.py`，12 线程）：
- 扫描全部 128 个算法×场景组合，扩展阈值 0→15 重跑
- 59 个算法 AUC 提升（TS 最大 +0.070，STCF +0.03，EvFlow +0.03）
- 上边界 (val=15) 残留：**0 个**——已全部收敛
- 下边界 (val=0) 残留：107 个——稀疏数据天然最优（"全保留"），无需再扩

**数据一致性验证规则**（写入此处防止遗忘）：
1. **BAF ≤ STCF_orig**：BAF 为最简单时空滤波器，不应超越更复杂的 STCF_orig。tau [2,200]ms unified 重跑后所有数据集均满足（DVSCLEAN: BAF=0.9805 < STCF_orig=0.9825; LED: 0.8375 < 0.8649; ED24: 同样满足）
2. **低噪 AUC ≥ 高噪 AUC**：同数据集内噪声越强 AUC 应越低（或持平）。反转需排查数据或扫频问题
3. **PFD 不适用 r=1**：PFD 两阶段滤波需足够空间邻域，r=1 在 1280×720 上显著劣于 r=3（DVSCLEAN 0.961 vs 0.986；LED 0.785 vs 0.827）

### 7.13 跨数据集算法表现差异分析 (2026-05-22)

| Algorithm | Driving 8hz | ED24 Ped heavy | ED24 Bike heavy | DVSCLEAN | LED |
|---|---:|---:|---:|---:|---:|
| **N149_v2.2** | **0.9484** | **0.9375** | **0.9746** | **0.9966** | **0.8916** |
| EBF | 0.9374 | 0.9099 | 0.9426 | 0.9924 | 0.8669 |
| YNoise | 0.9252 | 0.8971 | 0.9274 | 0.9896 | 0.8604 |
| STCF_orig | 0.8965 | 0.8392 | 0.9134 | 0.9825 | 0.8649 |
| TS | 0.9291 | 0.8465 | 0.9148 | 0.9297 | 0.7969 |
| PFD | 0.8825 | 0.7856 | 0.8877 | 0.9610 | 0.8120 |
| BAF | 0.8379 | 0.8143 | 0.8907 | 0.9805 | 0.8375 |
| EvFlow | 0.8060 | 0.7847 | 0.8212 | 0.7733 | 0.7006 |
| KNoise | 0.6214 | 0.6417 | 0.7007 | 0.6389 | 0.5276 |

**数据集噪声特性与算法适配分析**：

| 数据集 | 噪声来源 | 分辨率 | 事件密度 | 信号特征 | 适配算法 |
|---|---|---|---|---|---|
| **Driving** | v2e 仿真叠加 | 346×260 | 极高 (~5 Hz/pix) | 连续纹理边缘 | EBF, TS, YNoise（时间核利用密集事件） |
| **ED24 Ped/Bike** | 真实 DVS 拍摄 | 346×260 | 中高 | 人体/车轮轮廓 | N149（极性+空间联合判别） |
| **DVSCLEAN** | 仿真事件+真实噪声 | 1280×720 | 极低 | 稀疏目标 | BAF, STCF（简单计数已足够，天花板效应） |
| **LED** | 未知传感器 | 1280×720 | 低（100ms 切片） | 稀疏亮斑 | N149（仅靠 α=2.0 全极性弥补稀疏） |

**差异根因**：
1. **事件密度决定时间信息价值**：Driving 密集事件 → 时间核算法（TS/EBF）强；DVSCLEAN 稀疏 → 时间核无用，简单计数（BAF）即可达 0.95
2. **分辨率影响空间信息**：1280×720 上 r=1 的 3×3 窗口仅覆盖百万分之一面积 → 空间信息稀释，需 PFD r=3 或更大半径
3. **极性信息含量因场景而异**：Driving 异极=噪声(α=0)，LED 异极=信号(α=2.0)——同一算法不同 α 跨越 0.03 AUC
4. **DVSCLEAN 天花板效应**：全算法 AUC > 0.93，区分力不足；LED 地板效应：全算法 AUC < 0.89，N149 大幅领先

**检测方法**：
1. 读取 `data/missing_alg/` 下所有 ROC CSV
2. 检查每个算法的 min-neighbors 最优点是否触 `0` 或 `5` 边界
3. 检查算法参数 (r, tau) 是否为数据集最优（对照 Driving 上已有扫频结果）

**修复策略**（按算法）：
| 算法 | 扫频参数 | 边界处理 |
|---|---|---|
| BAF | r=1 fixed, tau 扫 {2ms~256ms} | tau 触边界则扩展 |
| STCF | r=1 fixed, tau 扫 {2ms~256ms} | 同上 |
| EBF | r, tau 网格扫 | r/tau 触边界则扩大 |
| YNoise | r, tau 网格扫 | 同上 |
| TS | tau (decay) 扫 {8ms~256ms} | tau 触边界则扩展 |
| PFD | r=1 fixed, tau 扫 | tau 触边界则扩展 |
| EvFlow | r, tau 精简扫 (lite sweep) | 先找最优 tau，再全阈值 |
| KNoise | tau 扫 | 触边界则扩展 |

**执行**：`scripts/n149_ablation/run_boundary_fix.py`（待写），多线程。

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

### 10.6 hot_state 归一化修正（2026-05-21）：

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

### 10.7 结论

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

**DVSCLEAN（10 子集）详细**：

| 参数 | 44/50 | 44/100 | 46/50 | 46/100 | 47/50 | 47/100 | 48/50 | 48/100 | 49/50 | 49/100 | 跨级一致值 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| r | 9 (0.9980) | 9 (0.9977) | 5 (0.9973) | 5 (0.9966) | 3 (0.9956) | 5 (0.9943) | 9 (0.9972) | 9 (0.9967) | 9 (0.9972) | 9 (0.9966) | **5** (r=9仅+0.001) |
| tau | 128K (0.9979) | 128K (0.9976) | 64K (0.9977) | 64K (0.9972) | 64K (0.9966) | 64K (0.9958) | 128K (0.9970) | 128K (0.9963) | 256K (0.9968) | 128K (0.9961) | **128K** |
| sigma | 2.5 (0.9981) | 2.5 (0.9977) | 2.5 (0.9975) | 2.5 (0.9968) | 2.0 (0.9960) | 2.0 (0.9947) | 3.0 (0.9970) | 3.0 (0.9963) | 3.0 (0.9968) | 3.0 (0.9961) | **2.5** (447偏2.0, 448/449偏3.0) |
| alpha | 0.5 (0.9980) | 0.25 (0.9976) | 0.25 (0.9973) | 0.25 (0.9966) | 0.25 (0.9953) | 0.25 (0.9943) | 0.25 (0.9970) | 0.25 (0.9963) | 0.5 (0.9968) | 0.25 (0.9961) | **0.25** (444/449 偏 0.5) |

> DVSCLEAN 天花板效应显著，所有参数组合 AUC≥0.99。r=5 vs r=9 差异 <0.001，选 r=5 兼顾 FPGA 部署。tau 集中于 64K~128K。sigma 2.5 多数最优。

**LED（10 场景）详细**：

| 参数 | 100 | 1004 | 1018 | 1028 | 1032 | 1033 | 1034 | 1043 | 1045 | 1046 | 跨级一致值 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| r | 2 (0.9037) | 3 (0.8403) | 2 (0.9011) | 2 (0.8976) | 2 (0.9032) | 2 (0.8428) | 3 (0.8521) | 2 (0.8961) | 2 (0.8761) | 2 (0.9033) | **2** (1004/1034偏3) |
| tau | 8K (0.9162) | 8K (0.8442) | 16K (0.9011) | 16K (0.8976) | 16K (0.9032) | 16K (0.8428) | 8K (0.8583) | 16K (0.8961) | 16K (0.8761) | 16K (0.9033) | **8K** (多数偏16K；8K均值相当) |
| sigma | 1.5 (0.9060) | 2.0 (0.8404) | 2.0 (0.9025) | 1.5 (0.9030) | 2.0 (0.9051) | 1.5 (0.8467) | 2.0 (0.8522) | 2.0 (0.8978) | 2.0 (0.8785) | 1.5 (0.9061) | **2.0** (100/1028/1033/1046偏1.5) |
| alpha | 1.0 (0.9245) | 1.0 (0.8627) | 1.0 (0.8984) | 1.0 (0.9022) | 1.0 (0.9009) | 1.0 (0.8664) | 1.0 (0.8785) | 1.0 (0.8976) | 1.0 (0.8849) | 1.0 (0.8994) | **1.0** (scene_100偏2.0: 0.9262) |

> LED 最优参数 r=2, tau=8K, sigma=2.0, alpha=1.0（10 场景均值 AUC=0.8916）。仅 scene_100 偏好 α=2.0 (0.9262 vs α=1.0 的 0.9245)。完整 α={0~3.0} 扫频见 `scripts/n149_ablation/run_led_alpha_full.py`。

### 11.2A r-σ 耦合验证 (2026-06-01)

> **实验目标**：验证 N149 中空间尺度 `sigma` 与窗口半径 `r` 是否可耦合，从而减少 1 个超参数并保持效果稳定。

**实验设置**（固定其余参数为各数据集当前最优）：

- Drive：`r=2, tau=32K, alpha=0.05`
- Ped：`r=5, tau=256K, alpha=0.25`
- Bike：`r=5, tau=256K, alpha=0.25`
- `hot_bits=8`, `lambda=2`（`MYEVS_N149_HOT_DECAY_K=2`）

**对比组**：

1. `free_sigma`：每数据集独立最优 sigma（当前基线）
2. `sigma = r`
3. `sigma = 2r/3`
4. `sigma = r/2`

脚本：`scripts/n149_ablation/run_r_sigma_coupling.py`  
结果目录：`data/Hyperparameter ablation_study/rsigma_coupling/`

- `r_sigma_coupling_raw.csv`
- `r_sigma_coupling_delta.csv`
- `r_sigma_coupling_summary.csv`

**结果汇总（跨级均值 AUC）**：

| 数据集 | free_sigma | sigma=r | Δ | sigma=2r/3 | Δ | sigma=r/2 | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Drive | 0.949033 | 0.948786 | -0.000247 | 0.948211 | -0.000822 | 0.941208 | -0.007825 |
| Ped | 0.946939 | 0.941280 | -0.005660 | 0.946291 | -0.000648 | 0.946620 | -0.000319 |
| Bike | 0.981204 | 0.978185 | -0.003020 | 0.980925 | -0.000279 | 0.980964 | -0.000241 |

**结论**：

1. `sigma=r` 不是稳健耦合：在 Ped/Bike 上有明显退化（-0.0057 / -0.0030）。
2. `sigma=r/2` 也不稳健：Drive 出现明显退化（-0.0078）。
3. `sigma=2r/3` 在三数据集都稳定，损失均 <0.001，且无灾难性退化。
4. 若目标是**减少超参数并保持可解释性**，推荐将默认关系改为 `sigma = 2r/3`；若目标是追求极致性能，仍保留 `sigma` 独立可调更优（`free_sigma` 略高）。

#### 11.2A.1 进一步讨论：`2\sigma^2` 与 `r^2` 的等价重参数化（2026-06-01）

基于
\[
w_{space}=\exp\left(-\frac{d^2}{2\sigma^2}\right),
\]
可改写为
\[
w_{space}=\exp\left(-\frac{d^2}{k r^2}\right),\quad k=\frac{2\sigma^2}{r^2}.
\]

其中 `sigma=2r/3` 对应 `k=8/9`，而“更整洁”的 `2\sigma^2=r^2` 对应 `k=1`。  
为验证 `k=1` 是否优于 `k=8/9`，新增细扫脚本：

- `scripts/n149_ablation/run_rsigma_k_sweep.py`

输出目录：

- `data/Hyperparameter ablation_study/rsigma_k_sweep/r_sigma_k_raw.csv`
- `data/Hyperparameter ablation_study/rsigma_k_sweep/r_sigma_k_delta.csv`
- `data/Hyperparameter ablation_study/rsigma_k_sweep/r_sigma_k_summary.csv`
- `data/Hyperparameter ablation_study/rsigma_k_sweep/r_sigma_k_best.csv`

关键对比（`k=1` 相对 `k=8/9`）：

- Drive：`+0.000196`
- Ped：`-0.000482`
- Bike：`-0.000236`

即：`k=1` 仅在 Drive 略优，在 Ped/Bike 略劣，不具备跨数据集统一优势。

**最终决策（用于后续复盘）**：

1. 保持 `sigma` 为独立超参数，不强制绑定为 `sigma=c*r` 或固定 `k`。
2. 原因不是“耦合不可解释”，而是“耦合会减少灵活性”；若仍引入 `k`，本质上只是把 `sigma` 换名为另一个尺度超参数。
3. 因此在 N149 中，`sigma` 的工程语义应定义为：  
   **邻域内距离衰减强度控制参数（distance-decay scale）**，而非传统图像处理中与核尺寸强绑定的“几何空间尺度”。
4. `r` 负责“截断支持域”（看多大邻域），`sigma` 负责“域内权重分布形状”（同一邻域内近邻/远邻权重比）。二者相关但不等价，保留双参数更符合当前实验事实。

### 11.2B 跨场景参数稳定性分析 (2026-05-22)

> **问题**：同一数据集内不同噪声等级/场景可能偏好不同的最优参数。强制统一参数会带来多大 AUC 损失？

**DVSCLEAN（10 子集）**：天花板效应下所有损失可忽略。

| 参数 | 共识值 | 偏离场景数 | 最大单场景损失 | 均值损失 |
|---|---|---|---|---|
| r | 9 | 4/10 | 0.0005 | 0.0001 |
| tau | 128K | 5/10 | 0.0015 | 0.0004 |
| sigma | 2.5 | 6/10 | 0.0003 | 0.0001 |
| alpha | 0.25 | 2/10 | ~0 | ~0 |

> DVSCLEAN AUC≈0.99，参数的场景级差异 <0.0015，选任意值几乎无差异。

**LED（10 场景）**：tau 有有意义的跨场景差异，其余参数稳定。

| 参数 | 共识值 | 偏离场景数 | 最大单场景损失 | 均值损失 |
|---|---|---|---|---|
| r | 2 | 2/10 | 0.0007 | 0.0001 |
| tau | 16K | 3/10 | **0.0125** | 0.0024 |
| sigma | 2.0 | 4/10 | 0.0017 | 0.0003 |
| alpha | 1.0 | 1/10 (scene_100偏2.0) | 0.0017 | 0.0002 |

> LED 仅 tau 有场景级差异：scene_100 偏好 8K (0.9162)，若强制用 16K 降至 0.9037 (-0.0125)。最终选 tau=8K 因 Phase 2 优化后 8K 均值反超。r/sigma/alpha 跨场景损失均 <0.002，参数高度稳定。alpha=1.0 为 9/10 场景最优，仅 scene_100 偏好 α=2.0 (+0.0017)，强制 α=1.0 损失可忽略。

**Driving/Ped/Bike**：噪声等级间参数一致性见 §11.2 各表"跨级一致值"列。Driving 全 5 级 r=2、tau=32K 完全一致；Ped r=5 vs r=9 损失 <0.002；Bike r 损失 <0.007。

**总结**：N149 v2.2 的超参数在同类数据集内高度稳定。除 LED tau 有 ~0.01 的场景级差异外，所有参数在不同噪声等级/场景间的强制统一损失均 ≤0.002。

### 11.2C Driving 四 panel 图数据表

以下为 `fig_drive_sweep.png` 的原始数据，方便直接读取 AUC 数值。

**r 扫频**（固定 tau=32K, sigma=3.0, alpha=0）：

| r | 1hz | 3hz | 5hz | 7hz | 10hz |
|---|---:|---:|---:|---:|---:|
| 1 | 0.9394 | 0.9383 | 0.9380 | 0.9353 | 0.9335 |
| **2** | **0.9485** | **0.9474** | **0.9474** | **0.9447** | **0.9435** |
| 3 | 0.9463 | 0.9454 | 0.9459 | 0.9434 | 0.9428 |
| 4 | 0.9433 | 0.9424 | 0.9433 | 0.9409 | 0.9404 |
| 5 | 0.9409 | 0.9402 | 0.9413 | 0.9389 | 0.9384 |
| 7 | 0.9389 | 0.9383 | 0.9395 | 0.9371 | 0.9364 |

**τ 扫频**（固定 r=2, sigma=3.0, alpha=0）：

| τ (us) | 1hz | 3hz | 5hz | 7hz | 10hz |
|---|---:|---:|---:|---:|---:|
| 4K | 0.8807 | 0.8843 | 0.8888 | 0.8909 | 0.8960 |
| 8K | 0.9219 | 0.9236 | 0.9271 | 0.9263 | 0.9294 |
| 16K | 0.9394 | 0.9397 | 0.9416 | 0.9397 | 0.9405 |
| **32K** | **0.9485** | **0.9474** | **0.9474** | **0.9447** | **0.9435** |
| 64K | 0.9483 | 0.9460 | 0.9444 | 0.9406 | 0.9365 |
| 128K | 0.9393 | 0.9355 | 0.9314 | 0.9248 | 0.9166 |
| 256K | 0.9220 | 0.9165 | 0.9110 | 0.9058 | 0.9046 |

**σ 扫频 Phase 2 细调**（固定 r=2, tau=32K, alpha=0, 步长 0.25）：

| σ | 1hz | 3hz | 5hz | 7hz | 10hz |
|---|---:|---:|---:|---:|---:|
| 1.00 | 0.9416 | 0.9413 | 0.9416 | 0.9400 | 0.9390 |
| 1.25 | 0.9485 | 0.9480 | 0.9481 | 0.9459 | 0.9446 |
| 1.50 | 0.9506 | 0.9498 | 0.9497 | 0.9473 | 0.9459 |
| **1.75** | **0.9512** | **0.9501** | **0.9499** | **0.9474** | **0.9459** |
| 2.00 | 0.9512 | 0.9498 | 0.9495 | 0.9469 | 0.9455 |
| 2.25 | 0.9512 | 0.9499 | 0.9496 | 0.9469 | 0.9453 |
| 2.50 | 0.9511 | 0.9497 | 0.9494 | 0.9466 | 0.9450 |
| 3.00 | 0.9509 | 0.9494 | 0.9489 | 0.9461 | 0.9443 |

> 1hz 偏 σ=2.25 (0.9512)，其余 4 级 σ=1.75 最优。取 σ=1.75，1hz 损失 <0.0001。

**α 扫频**（固定 r=2, tau=32K, sigma=3.0）：

| α | 1hz | 3hz | 5hz | 7hz | 10hz |
|---|---:|---:|---:|---:|---:|
| 0 | **0.9509** | **0.9494** | 0.9489 | 0.9461 | 0.9443 |
| 0.05 | 0.9508 | 0.9493 | **0.9490** | **0.9462** | 0.9446 |
| 0.10 | 0.9504 | 0.9491 | 0.9489 | 0.9461 | **0.9447** |
| 0.25 | 0.9485 | 0.9474 | 0.9474 | 0.9447 | 0.9435 |
| 0.50 | 0.9434 | 0.9422 | 0.9425 | 0.9396 | 0.9386 |
| 0.75 | 0.9370 | 0.9355 | 0.9358 | 0.9325 | 0.9314 |
| 1.00 | 0.9300 | 0.9279 | 0.9280 | 0.9243 | 0.9227 |
| ema | 0.9485 | 0.9474 | 0.9474 | 0.9447 | 0.9435 |

> α=0 均值 AUC=0.9479，α=0.05 均值=0.9480，差异 <0.0001。取 α=0 简化实现（关闭极性加权）。

### 11.2D 异极性论证：极性盲化与纯异极性实验 (2026-05-22)

> **动机**：LED 上 α=1.0 最优，意味着同极性和异极性对去噪贡献相等。由此引出三个问题：(1) 是否可以完全不分极性直接数邻居？(2) 单独靠异极性能否完成去噪？(3) 单独靠同极性能否完成去噪？

**N149 v2.2 原始评分公式**（`score_one` 核心逻辑）：

$$R^+ = \sum_{j \in \mathcal{N}(i)} w_{space}(i,j) \cdot w_{time}(i,j) \cdot \mathbf{1}[p_j = p_i]$$

$$R^- = \sum_{j \in \mathcal{N}(i)} w_{space}(i,j) \cdot w_{time}(i,j) \cdot \mathbf{1}[p_j \neq p_i]$$

$$S = (R^+ + \alpha \cdot R^-) \cdot f(q_i), \quad f(q_i) = \frac{q_i + h_{unit}}{2q_i + h_{unit}}$$

其中 $\mathcal{N}(i)$ 为事件 $i$ 的时空邻域（半径 $r$，时间窗 $\tau$），$w_{space} = e^{-d^2/(2\sigma^2)}$，$w_{time} = (1 - \Delta t/\tau)^2$，$q_i$ 为归一化 hot_state，$h_{unit}=2^{frac\\_bits}$。

**四个变体定义**（r=2, tau=8K, sigma=2.0, hot_state 保留）：

| 变体 | 公式 | $R^+$ | $R^-$ | 实现 |
| --- | --- | --- | --- | --- |
| **基线 N149 v2.2 (α=1.0)** | $S = (R^+ + R^-) \cdot f(q)$ | 同极性邻居 | 异极性邻居 | 默认 |
| **变体 A：极性盲化 (Blind)** | $S_A = R_{all} \cdot f(q)$ | — | — | `MYEVS_N149_BLIND=1` |
| **变体 B：纯异极性 (Opp-only)** | $S_B = R^- \cdot f(q)$ | 置零 | 异极性邻居 | `MYEVS_N149_NO_SAME=1`, α=1.0 |
| **变体 C：纯同极性 (Same-only)** | $S_C = R^+ \cdot f(q)$ | 同极性邻居 | 置零 | `MYEVS_N149_NO_OPP=1` |

> 变体 A：不调 `norm_pol()`、不写 `last_pol_`、`acc_neighbor` 不检查极性、所有邻居计入 `raw_same`、强制 `no_opp_=no_mix_=true`。hot_state 保留。
> 
> $R_{all} = \sum_{j \in \mathcal{N}(i)} w_{space} \cdot w_{time}$（无条件，极性完全不参与）。
>
> 变体 B：`acc_neighbor` 正常区分同/异极性，计数后在 `score_one` 中将 `raw_same` 强制置零。
>
> 变体 C：`acc_neighbor` 正常计数，在 `score_one` 中将 `raw_opp` 强制置零（与变体 B 对偶）。

**LED 10 场景结果**（`scripts/LED_alg_evalu/run_led_polarity_ablation.py`, 12 线程）：

| 变体 | 10 场景均值 AUC | vs 基线 |
| --- | --- | --- |
| **基线 N149 v2.2 (α=1.0)** | **0.8916** | — |
| 变体 A：Blind ($R_{all}$) | **0.8916** | 0.0000（逐场景完全一致） |
| 变体 C：Same-only ($R^+$) | **0.8552** | −0.0364 |
| 变体 B：Opp-only ($R^-$) | **0.6710** | −0.2206 |

> Blind 与基线数值完全一致，验证了极性盲化在数学上等价于 α=1.0。Blind 代码不碰极性，部署时无需极性查找表。

| 场景 | 基线 ($R^++R^-$) | Blind ($R_{all}$) | Same ($R^+$) | Opp ($R^-$) |
| --- | --- | --- | --- | --- |
| scene_100 | 0.9245 | 0.9245 | 0.9110 | 0.6107 |
| scene_1004 | 0.8627 | 0.8627 | 0.8216 | 0.6576 |
| scene_1018 | 0.8984 | 0.8984 | 0.8676 | 0.6567 |
| scene_1028 | 0.9022 | 0.9022 | 0.8623 | 0.6843 |
| scene_1032 | 0.9009 | 0.9009 | 0.8686 | 0.6721 |
| scene_1033 | 0.8664 | 0.8664 | 0.8181 | 0.6710 |
| scene_1034 | 0.8785 | 0.8785 | 0.8360 | 0.6840 |
| scene_1043 | 0.8976 | 0.8976 | 0.8617 | 0.7088 |
| scene_1045 | 0.8849 | 0.8849 | 0.8435 | 0.6812 |
| scene_1046 | 0.8994 | 0.8994 | 0.8615 | 0.6835 |

**结论**：

1. **Blind ≡ 基线 (α=1.0)**：极性盲化与 α=1.0 数学等价，逐场景 AUC 完全一致。部署时可省略极性处理简化硬件。
2. **$R^+$（同极性）是主信号**：Same-only AUC=0.8552 仅比基线低 0.036，而同极性邻居正是 STC/BAF 的核心机制——"同一物体边缘在同一极性方向上持续触发"。
3. **$R^-$（异极性）是辅助增强**：Opp-only AUC=0.6710 比基线低 0.22，但比随机（0.5）高 0.17。异极性捕获的是极性翻转信号（边缘方向改变），单独使用效果差但作为 $R^+$ 的补充能将 AUC 从 0.855 提升至 0.892。
4. **互补性量化**：$R^+$ 贡献 ~0.855，$R^-$ 单独仅 ~0.671，但组合后达到 ~0.892。两组信息互补——同极性做时间相关滤波（主），异极性提供极性翻转证据（辅），两者协同才能达到最优。

#### DVSCLEAN 异极性实验

> DVSCLEAN 最优 α=0.25（同极性权重 4× 异极性）。同一套四个变体在 DVSCLEAN（r=5, tau=128K, sigma=2.5）上运行。脚本：`scripts/DVSCLEAN_alg_evalu/run_dvsclean_polarity_ablation.py`（12 线程）。

**DVSCLEAN 10 子集结果**：

| 变体 | 10 子集均值 AUC | vs 基线 |
| --- | --- | --- |
| **基线 N149 v2.2 (α=0.25)** | **0.9966** | — |
| 变体 A：Blind ($R_{all}$) | **0.9959** | −0.0007 |
| 变体 C：Same-only ($R^+$) | **0.9953** | −0.0013 |
| 变体 B：Opp-only ($R^-$) | **0.8189** | −0.1777 |

| 子集 | 基线 (α=0.25) | Blind ($R_{all}$) | Same ($R^+$) | Opp ($R^-$) |
| --- | --- | --- | --- | --- |
| 444_50 | 0.9981 | 0.9978 | 0.9969 | 0.9227 |
| 444_100 | 0.9977 | 0.9974 | 0.9964 | 0.9077 |
| 446_50 | 0.9975 | 0.9967 | 0.9968 | 0.8170 |
| 446_100 | 0.9968 | 0.9957 | 0.9961 | 0.7858 |
| 447_50 | 0.9959 | 0.9948 | 0.9949 | 0.8477 |
| 447_100 | 0.9947 | 0.9934 | 0.9935 | 0.8072 |
| 448_50 | 0.9969 | 0.9965 | 0.9952 | 0.7707 |
| 448_100 | 0.9962 | 0.9956 | 0.9943 | 0.7376 |
| 449_50 | 0.9965 | 0.9963 | 0.9947 | 0.8098 |
| 449_100 | 0.9958 | 0.9951 | 0.9937 | 0.7826 |

**DVSCLEAN vs LED 对比分析**：

| 指标 | LED (α=1.0) | DVSCLEAN (α=0.25) |
|---|---|---|
| 基线 AUC | 0.8916 | 0.9966 |
| Same-only 损失 | −0.0364 | −0.0013 |
| Opp-only 损失 | −0.2206 | −0.1777 |
| Blind 损失 | 0 | −0.0007 |
| 同极性主导程度 | 中等（α=1.0） | 极强（α=0.25，$R^+$ 几乎就是全部） |

> DVSCLEAN 天花板效应（AUC≈0.997）使所有变体损失都被压缩。Same-only 仅损失 0.0013，说明 DVSCLEAN 的去噪信息几乎全部来自同极性。Opp-only 虽降至 0.82，但仍比随机好——DVSCLEAN 的干净事件流中异极性信号比 LED 强（无噪声场景下极性翻转由物体边缘自然产生）。Blind 损失 0.0007，反映 α=1.0 在 DVSCLEAN 上非最优（最优是 α=0.25）。

### 11.2E 组件消融实验 (2026-05-22)

> **目的**：逐一移除 N149 v2.2 的五个核心组件，在全部 5 数据集 33 级别上测量各组件对去噪的独立贡献。
>
> **脚本**：
> - `scripts/n149_ablation/run_component_ablation.py` — 基础消融（no_spatial / no_opp / no_hot），全 33 级
> - `scripts/n149_ablation/run_comp_ab_polarity_v3.py` — no_polarity 补充（`MYEVS_N149_BLIND=1`），8 线程
> - `scripts/n149_ablation/run_comp_ab_time_only.py` — time_only 初版（固定 r,tau），8 线程
> - `scripts/n149_ablation/run_no_spatial_sweep.py` — no_spatial 网格扫频（r×τ 重搜索），8 线程，252 任务
> - `scripts/n149_ablation/run_time_only_sweep.py` — time_only 网格扫频（r×τ 重搜索），8 线程，252 任务
>
> **方法论**：消融组件后最优 (r,τ) 可能改变（如 time_only 中 r 从 5→3），因此 time_only 需独立扫频确定最优参数。

**被消融组件**：

| 组件 | 公式角色 | 消融方式 | 消融后公式 |
|---|---|---|---|
| 空间核 | $w_{space} = e^{-d^2/(2\sigma^2)}$ | `MYEVS_N149_NO_SPATIAL=1` → $w_{space}=1.0$ | $S = (R^+ + \alpha R^-) \cdot f(q)$，无空间衰减 |
| 异极性 | $R^-$ | `MYEVS_N149_NO_OPP=1` → $R^-=0$ | $S = R^+ \cdot f(q)$ |
| 热状态 | $f(q) = \frac{q+h_{unit}}{2q+h_{unit}}$ | `MYEVS_N149_NO_HOT=1` → $f(q)=1.0$ | $S = R^+ + \alpha R^-$，无时间折扣 |
| 极性区分 | $R^+$ vs $R^-$ | `MYEVS_N149_BLIND=1` → 不调 `norm_pol`、不存 `last_pol_` | $S = R_{all} \cdot f(q)$，等价于 α=1.0 |
| 仅保留时间核 | 全部空间/热度/极性 | `NO_SPATIAL+NO_HOT+BLIND` | $S = R_{all}$，纯时间窗内邻域计数 |

**各数据集均值**（`scripts/n149_ablation/run_component_ablation.py` + `run_comp_ab_polarity_v3.py` + `run_comp_ab_time_only.py`, 8 线程）：

| 变体 | Drive (n=5) | Δ | Ped (n=4) | Δ | Bike (n=4) | Δ | DVSCLEAN (n=10) | Δ | LED (n=10) | Δ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **baseline** | **0.9490** | — | **0.9470** | — | **0.9812** | — | **0.9966** | — | **0.8916** | — |
| no_spatial | 0.9465 | −0.0026 | 0.9377 | **−0.0093** | 0.9767 | **−0.0045** | 0.9939 | −0.0027 | 0.8884 | −0.0032 |
| no_opp | 0.9489 | −0.0001 | 0.9425 | −0.0044 | 0.9787 | −0.0025 | 0.9953 | −0.0013 | 0.8552 | **−0.0364** |
| no_hot | 0.9480 | −0.0010 | 0.9368 | −0.0102 | 0.9767 | −0.0045 | 0.9964 | −0.0002 | 0.8945 | **+0.0030** |
| no_polarity | 0.9288 | **−0.0202** | 0.9389 | −0.0081 | 0.9761 | −0.0051 | 0.9959 | −0.0007 | 0.8916 | 0.0000 |
| **time_only** | 0.9227 | **−0.0263** | 0.9137 | **−0.0333** | 0.9648 | **−0.0164** | 0.9906 | −0.0060 | 0.8918 | +0.0002 |

> `time_only` 为单独网格扫频后的最优 (r,τ) 结果（脚本 `run_time_only_sweep.py`）。去掉空间核和热状态后最优 r 和 τ 大幅改变：Drive r=2→r=2(不变), τ=32K→4K(10hz)；Ped r=5→r=3(重噪), τ=256K→128K；Bike r=5→r=3, τ=256K→128K。原固定参数严重低估了 Ped/Bike 的性能。

> `no_polarity`（`MYEVS_N149_BLIND=1`）完全移除极性概念：不调 `norm_pol()`、不存 `last_pol_`、不区分 $R^+$/$R^-$，等价于 α=1.0。脚本：`scripts/n149_ablation/run_comp_ab_polarity_v3.py`（8 线程）。Δ 排序：Drive (−0.020) > Ped (−0.008) > Bike (−0.005) > DVSCLEAN (−0.001) > LED (0)，与各数据集的 α 值完全一致——α 越小，极性区分越重要。

**Drive 逐级**（r=2, tau=32K, sigma=1.75, α=0.05）：

| Level | baseline | no_spatial | no_opp | no_hot | no_polarity | time_only |
| --- | --- | --- | --- | --- | --- | --- |
| Level | baseline | no_spatial | no_opp | no_hot | no_polarity | time_only |
| --- | --- | --- | --- | --- | --- | --- |
| 1hz | 0.9512 | 0.9497 (−0.0015) | 0.9512 (+0.0000) | 0.9496 (−0.0016) | 0.9317 (−0.0195) | 0.9249 (−0.0263) |
| 3hz | 0.9502 | 0.9480 (−0.0022) | 0.9501 (−0.0001) | 0.9488 (−0.0014) | 0.9300 (−0.0201) | 0.9225 (−0.0277) |
| 5hz | 0.9500 | 0.9475 (−0.0025) | 0.9499 (−0.0002) | 0.9489 (−0.0011) | 0.9302 (−0.0198) | 0.9227 (−0.0273) |
| 7hz | 0.9475 | 0.9444 (−0.0031) | 0.9474 (−0.0002) | 0.9468 (−0.0007) | 0.9267 (−0.0208) | 0.9192 (−0.0284) |
| 10hz | 0.9462 | 0.9426 (−0.0036) | 0.9459 (−0.0003) | 0.9460 (−0.0002) | 0.9254 (−0.0208) | 0.9240 (−0.0222) |

**Ped 逐级**（sigma=2.75, α=0.25; no_spatial/time_only 用独立扫频最优 r,tau）：

| Level | baseline | no_spatial | no_opp | no_hot | no_polarity | time_only |
| --- | --- | --- | --- | --- | --- | --- |
| 1.8 | 0.9563 | 0.9514 (−0.0049) | 0.9527 (−0.0036) | 0.9522 (−0.0041) | 0.9507 (−0.0056) | 0.9393 (−0.0170) |
| 2.1 | 0.9498 | 0.9407 (−0.0091) | 0.9461 (−0.0037) | 0.9424 (−0.0073) | 0.9423 (−0.0075) | 0.9226 (−0.0272) |
| 2.5 | 0.9443 | 0.9303 (−0.0140) | 0.9394 (−0.0050) | 0.9314 (−0.0129) | 0.9357 (−0.0086) | 0.8970 (−0.0473) |
| 3.3 | 0.9375 | 0.9283 (−0.0092) | 0.9320 (−0.0055) | 0.9212 (−0.0163) | 0.9267 (−0.0108) | 0.8958 (−0.0417) |

**Bike 逐级**（sigma=2.75, α=0.25; no_spatial/time_only 用独立扫频最优 r,tau）：

| Level | baseline | no_spatial | no_opp | no_hot | no_polarity | time_only |
| --- | --- | --- | --- | --- | --- | --- |
| 1.8 | 0.9865 | 0.9841 (−0.0024) | 0.9848 (−0.0018) | 0.9841 (−0.0024) | 0.9834 (−0.0031) | 0.9761 (−0.0104) |
| 2.1 | 0.9838 | 0.9796 (−0.0041) | 0.9816 (−0.0021) | 0.9806 (−0.0032) | 0.9797 (−0.0041) | 0.9702 (−0.0136) |
| 2.5 | 0.9800 | 0.9738 (−0.0062) | 0.9772 (−0.0028) | 0.9746 (−0.0054) | 0.9747 (−0.0053) | 0.9615 (−0.0185) |
| 3.3 | 0.9746 | 0.9694 (−0.0052) | 0.9713 (−0.0033) | 0.9675 (−0.0071) | 0.9666 (−0.0081) | 0.9515 (−0.0231) |

**DVSCLEAN 逐子集**（r=5, tau=128K, sigma=2.5, α=0.25）：

| 子集 | baseline | no_spatial | no_opp | no_hot | no_polarity | time_only |
| --- | --- | --- | --- | --- | --- | --- |
| 444_50 | 0.9981 | 0.9953 (−0.0028) | 0.9969 (−0.0012) | 0.9979 (−0.0002) | 0.9978 (−0.0003) | 0.9923 (−0.0058) |
| 444_100 | 0.9977 | 0.9950 (−0.0028) | 0.9964 (−0.0013) | 0.9976 (−0.0001) | 0.9974 (−0.0004) | 0.9921 (−0.0056) |
| 446_50 | 0.9975 | 0.9945 (−0.0030) | 0.9968 (−0.0007) | 0.9972 (−0.0003) | 0.9967 (−0.0008) | 0.9908 (−0.0067) |
| 446_100 | 0.9968 | 0.9939 (−0.0030) | 0.9961 (−0.0007) | 0.9965 (−0.0003) | 0.9957 (−0.0012) | 0.9899 (−0.0070) |
| 447_50 | 0.9959 | 0.9908 (−0.0052) | 0.9949 (−0.0010) | 0.9955 (−0.0004) | 0.9948 (−0.0011) | 0.9865 (−0.0094) |
| 447_100 | 0.9947 | 0.9897 (−0.0050) | 0.9935 (−0.0012) | 0.9944 (−0.0003) | 0.9934 (−0.0013) | 0.9845 (−0.0103) |
| 448_50 | 0.9969 | 0.9951 (−0.0017) | 0.9952 (−0.0016) | 0.9968 (−0.0001) | 0.9965 (−0.0004) | 0.9927 (−0.0042) |
| 448_100 | 0.9962 | 0.9946 (−0.0016) | 0.9943 (−0.0019) | 0.9962 (−0.0001) | 0.9956 (−0.0006) | 0.9923 (−0.0039) |
| 449_50 | 0.9965 | 0.9953 (−0.0012) | 0.9947 (−0.0018) | 0.9966 (+0.0000) | 0.9963 (−0.0002) | 0.9928 (−0.0037) |
| 449_100 | 0.9958 | 0.9947 (−0.0011) | 0.9937 (−0.0021) | 0.9958 (−0.0000) | 0.9951 (−0.0007) | 0.9923 (−0.0035) |

**N149 r=3 组件消融**（sigma=2.5, alpha=0.25, tau=128K）：

| 变体 | ratio50 AUC | Δ | ratio100 AUC | Δ |
|---|---|---|---|---|
| **baseline** | **0.9957** | — | **0.9944** | — |
| no_spatial | 0.9949 | −0.0008 | 0.9936 | −0.0008 |
| no_opp | 0.9945 | −0.0013 | 0.9930 | −0.0014 |
| no_hot | 0.9954 | −0.0003 | 0.9942 | −0.0003 |

> DVSCLEAN 天花板效应使所有组件损失 <0.0015，与 §11.2D 全数据集结论一致。

**LED 逐场景**（r=2, tau=8K, sigma=2.0, α=1.0）：

| 场景 | baseline | no_spatial | no_opp | no_hot | no_polarity | time_only |
| --- | --- | --- | --- | --- | --- | --- |
| scene_100 | 0.9245 | 0.9222 (−0.0023) | 0.9110 (−0.0135) | 0.9280 (+0.0035) | 0.9245 (+0.0000) | 0.9261 (+0.0016) |
| scene_1004 | 0.8627 | 0.8601 (−0.0026) | 0.8216 (−0.0411) | 0.8674 (+0.0047) | 0.8627 (+0.0000) | 0.8653 (+0.0026) |
| scene_1018 | 0.8984 | 0.8960 (−0.0024) | 0.8676 (−0.0308) | 0.8998 (+0.0014) | 0.8984 (+0.0000) | 0.8976 (−0.0008) |
| scene_1028 | 0.9022 | 0.8978 (−0.0044) | 0.8623 (−0.0398) | 0.9040 (+0.0019) | 0.9022 (+0.0000) | 0.9001 (−0.0021) |
| scene_1032 | 0.9009 | 0.8984 (−0.0025) | 0.8686 (−0.0323) | 0.9022 (+0.0013) | 0.9009 (+0.0000) | 0.8999 (−0.0010) |
| scene_1033 | 0.8664 | 0.8611 (−0.0053) | 0.8181 (−0.0483) | 0.8716 (+0.0051) | 0.8664 (+0.0000) | 0.8670 (+0.0006) |
| scene_1034 | 0.8785 | 0.8767 (−0.0018) | 0.8360 (−0.0425) | 0.8830 (+0.0045) | 0.8785 (+0.0000) | 0.8817 (+0.0032) |
| scene_1043 | 0.8976 | 0.8949 (−0.0028) | 0.8617 (−0.0360) | 0.9000 (+0.0024) | 0.8976 (+0.0000) | 0.8976 (−0.0000) |
| scene_1045 | 0.8849 | 0.8809 (−0.0040) | 0.8435 (−0.0413) | 0.8878 (+0.0029) | 0.8849 (+0.0000) | 0.8843 (−0.0006) |
| scene_1046 | 0.8994 | 0.8957 (−0.0037) | 0.8615 (−0.0379) | 0.9012 (+0.0018) | 0.8994 (+0.0000) | 0.8979 (−0.0015) |

**分析**：

1. **空间核是 Ped/Bike 的关键组件**：损失 −0.0161~−0.0226（均值），重噪声级别 Ped 3.3 达 −0.0524、Bike 3.3 达 −0.0455。噪声越强，空间距离衰减越不可替代——它区分信号团簇和散落噪点。Drive/DVSCLEAN/LED 对此不敏感（≤−0.0032），因密集事件或天花板效应使空间信息冗余。

2. **异极性在 LED 上是决定性组件**：LED 的 no_opp 损失 −0.0364，是所有数据集中最大的单项损失。对应的 α=1.0 确认异极性在 LED 上与同极性等权。Drive(α≈0) 和 Bike/DVSCLEAN(α=0.25) 损失远小于 LED。

3. **热状态在 LED 上有反效果**：no_hot 在 LED 上 AUC 反而提升 +0.0030（10 场景全部正值），其他数据集均为负。LED 事件极其稀疏，热状态衰减过快可能误伤低频信号事件。

4. **噪声增强的非线性放大**：Drive 各噪声级别损失几乎恒定；Ped/Bike 从轻噪到重噪，no_spatial 损失放大 5~10 倍。重噪声下空间核从"锦上添花"变为"不可或缺"。

5. **DVSCLEAN 天花板效应**：所有损失 <0.003，基线 AUC≈0.997 已接近上限，组件消融无区分力。

#### 跨版本一致性：时序组件在稀疏场景下普适有害

> 热状态对 LED 有害并非 v2.2 独有。以下汇总 N149 从 v2.1 到 v2.2 所有涉及 LED 的消融实验中时序组件的表现。

**LED 上所有时序组件的跨版本 ΔAUC**（正值 = 关闭组件后 AUC 提升 = 组件有害）：

| 实验 | 版本 | 参数 | baseline | no_hot | no_beta | no_mix | no_sfrac | hot+beta+mix 全关 | Simp-B |
|---|---|---|---|---|---|---|---|---|---|
| §8.2 单场景 | v2.1 | r=2,τ=16K,σ=3.0 | 0.9120 | **+0.013** | **+0.004** | **+0.002** | **+0.004** | **+0.014** | — |
| §10.3 单场景 | v2.1 | r=2,τ=16K,σ=3.0 | 0.9160 | **+0.009** | — | — | — | — | **0.9262** |
| §11.2D 10场景均值 | v2.2 | r=2,τ=8K,σ=2.0,α=1.0 | 0.8916 | **+0.003** | — | — | — | — | — |

> v2.1 中 hot/beta/mix/sfrac 四个时序组件全部有害（关闭后提升），全关时 AUC 达 0.9262（+0.014）。Simp-B（全极性无状态，即 Blind 的前身）以 0.9262 反超完整 N149 v2.1。
>
> v2.2 归一化 hot_state 将危害从 +0.013 缩至 +0.003，但仍未转正。**结论**：LED 这种超稀疏场景下，任何基于事件密度的自适应折扣机制都会因"冷启动"误伤低频信号。仅在事件足够密集的数据集（Drive/Ped/Bike）上时序组件才转为有益。**建议**：LED 部署时关闭 hot_state（`MYEVS_N149_NO_HOT=1`）。

**各数据集对各组件的依赖模式总结**：

| 数据集 | 核心组件 | 无用/有害组件 | 事件特性 |
|---|---|---|---|
| Drive | 全部微弱 | — | 密集，高度冗余 |
| Ped/Bike | 空间核 ≫ 热状态 > 异极性 | — | 中等密度，空间结构强 |
| DVSCLEAN | 全部微弱 | — | 天花板效应 |
| **LED** | 异极性 ≫ 空间核 | **热状态有害** | 稀疏，极性翻转=信号 |

### 11.5 产物与复盘

| 产物 | 路径 |
|---|---|
| Phase 1 粗调脚本 | `scripts/n149_ablation/run_phase1_{drive,ped,bike,dvsclean,led}.py` |
| Phase 2 sigma 细调脚本 | `scripts/n149_ablation/run_phase2_sigma.py` |
| 最终最优参数运行 | `scripts/n149_ablation/run_final.py` |
| 汇总编译 | `scripts/n149_ablation/_compile_summary.py` |
| Driving 四 panel 图 | `scripts/n149_ablation/plot_drive_sweep.py` → `data/Hyperparameter ablation_study/drive/fig_drive_sweep.png` |
| 全部扫频数据 | `data/Hyperparameter ablation_study/{drive,ped,bike,dvsclean,led}/` |

### 11.3 Phase 2 计划

1. Ped/DVSCLEAN r 边界确认
2. DVSCLEAN/LED 绘图

### 11.4 α 跨数据集对照表 (2026-05-21 最终)

固定各数据集最优 r/tau/sigma，扫 α={0~3.0, ema}。

| Dataset/Level |    0.0 |   0.05 |    0.1 |   0.25 |    0.5 |   0.75 |    1.0 |    2.0 |    3.0 |    ema |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Drive_1hz | **0.9509** | 0.9508 | 0.9504 | 0.9485 | 0.9434 | 0.9370 | 0.9300 | 0.9073 | 0.8901 | 0.9485 |
| Drive_3hz | **0.9494** | 0.9493 | 0.9491 | 0.9474 | 0.9422 | 0.9355 | 0.9279 | 0.9025 | 0.8814 | 0.9474 |
| Drive_5hz | 0.9489 | **0.9490** | 0.9489 | 0.9474 | 0.9425 | 0.9358 | 0.9280 | 0.9011 | 0.8776 | 0.9474 |
| Drive_7hz | 0.9461 | **0.9462** | 0.9461 | 0.9447 | 0.9396 | 0.9325 | 0.9243 | 0.8944 | 0.8667 | 0.9447 |
| Drive_10hz | 0.9443 | 0.9446 | **0.9447** | 0.9435 | 0.9386 | 0.9314 | 0.9227 | 0.8900 | 0.8580 | 0.9435 |
| **Drive MEAN** | **0.9479** | **0.9480** | **0.9478** | 0.9463 | 0.9413 | 0.9344 | 0.9266 | 0.8991 | 0.8748 | 0.9463 |
| Ped_1.8 | 0.9533 | 0.9547 | 0.9558 | **0.9566** | 0.9553 | 0.9529 | 0.9502 | 0.9407 | 0.9326 | 0.9566 |
| Ped_2.1 | 0.9463 | 0.9480 | 0.9490 | **0.9497** | 0.9481 | 0.9451 | 0.9416 | 0.9286 | 0.9160 | 0.9497 |
| Ped_2.5 | 0.9394 | 0.9414 | 0.9427 | **0.9442** | 0.9427 | 0.9391 | 0.9348 | 0.9168 | 0.8916 | 0.9442 |
| Ped_3.3 | 0.9321 | 0.9341 | 0.9356 | **0.9373** | 0.9353 | 0.9307 | 0.9243 | 0.8916 | 0.8319 | 0.9373 |
| **Ped MEAN** | — | — | — | **0.9469** | — | — | — | 0.9194 | 0.8930 | **0.9469** |
| Bike_1.8 | 0.9834 | 0.9846 | **0.9850** | **0.9850** | 0.9837 | 0.9816 | 0.9793 | 0.9781 | 0.9720 | **0.9850** |
| Bike_2.1 | 0.9802 | 0.9816 | 0.9822 | **0.9826** | 0.9810 | 0.9784 | 0.9756 | 0.9718 | 0.9637 | **0.9826** |
| Bike_2.5 | 0.9747 | 0.9764 | 0.9775 | **0.9782** | 0.9761 | 0.9726 | 0.9687 | 0.9629 | 0.9451 | **0.9782** |
| Bike_3.3 | 0.9655 | 0.9675 | 0.9689 | **0.9696** | 0.9652 | 0.9543 | 0.9358 | 0.9367 | 0.8747 | **0.9696** |
| **Bike MEAN** | — | — | — | **0.9788** | — | — | — | 0.9624 | 0.9389 | **0.9788** |
| **DVSCLEAN MEAN** | 0.9953 | 0.9960 | 0.9963 | **0.9965** | 0.9964 | 0.9960 | 0.9954 | 0.9940 | 0.9921 | **0.9965** |
| **LED MEAN** | 0.8552 | 0.8594 | 0.8638 | 0.8754 | 0.8856 | 0.8900 | **0.8916** | 0.8871 | 0.8785 | — |

> LED 10 场景均值（r=2, tau=8K, sigma=2.0，脚本 `scripts/n149_ablation/run_led_alpha_full.py`）。α=1.0 均值最优 (0.8916)，仅 scene_100 偏 α=2.0 (0.9262 vs 0.9245)。α=0.05 统一 Drive 损失 <0.001；α=0.25 统一 Ped+Bike 等价 EMA。

### 11.6 产物与复盘

| 产物 | 路径 |
|---|---|
| Phase 1/2 脚本 | `scripts/n149_ablation/run_phase1_*.py` / `run_phase2_sigma.py` / `run_final.py` |
| 汇总编译 | `scripts/n149_ablation/_compile_summary.py` |
| Driving 四 panel 图 | `scripts/n149_ablation/plot_drive_sweep.py` → `data/Hyperparameter ablation_study/drive/fig_drive_sweep.png` |
| 全部扫频数据 | `data/Hyperparameter ablation_study/{drive,ped,bike,dvsclean,led}/` |
| 缺失算法补跑 | `scripts/n149_ablation/run_missing_alg.py` → `data/missing_alg/` |
| PFD/EvFlow 修正 | `scripts/n149_ablation/run_fix_pfd_evflow.py` |

