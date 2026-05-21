# EDformer AUC 评估结果

## 概述

使用预训练的 **EDformer** 模型（`pretrained_model.pth`）在多个数据集上评估事件去噪性能，测试不同频率/电压下的 AUC（Area Under ROC Curve）指标。

- **模型路径**: `/home/stu1/LJC/EVS/DL_ED/EDformer-main/pretrained_model.pth`
- **运行环境**: `edformer220` (conda)
- **评估日期**: 2026-05-19

---

## 0. 模型版本、训练设置与论文使用说明

本工程中所有实验都使用同一个 **EDformer 网络结构**，区别只在于权重来源、训练数据和坐标输入方式。微调或从零训练后的模型仍可称为 EDformer，但论文中应明确写成 “EDformer fine-tuned on LED” 或 “EDformer trained from scratch on LED”，不能与原论文发布的 ED24 预训练权重混为同一个实验设置。

### 模型版本定义

| 模型简称 | 结构是否改变 | 初始化方式 | 训练数据 | 坐标输入 | 主要用途 |
|----------|--------------|------------|----------|----------|----------|
| EDformer-ED24-pretrained | 否 | 原始 ED24 预训练权重 | 原作者/原工程 ED24 训练 | 原始输入 | 衡量原始 EDformer 的跨数据集 zero-shot 泛化 |
| EDformer-LED-finetuned | 否 | ED24 预训练权重 | LED 训练集中排除 10 个保留 scene 后的 590 个 scene | 原始 LED 坐标 | 衡量 EDformer 经 LED 域适配后的性能 |
| EDformer-LED-scratch-raw | 否 | 随机初始化 | LED 训练集中排除 10 个保留 scene 后的 590 个 scene | 原始 LED 坐标 | 衡量不依赖 ED24 预训练时，EDformer 结构在 LED 上能学到的性能 |
| EDformer-LED-scratch-norm | 否 | 随机初始化 | LED 训练集中排除 10 个保留 scene 后的 590 个 scene | x/y 按 1280×720 归一化到 [0,1] | 检验 LED 坐标尺度是否限制性能 |
| EDformer-ED24-retrained | 否 | 随机初始化 | ED24 排除 `Bicycle_02`、`Pedestrain_06` 后的 98 个 scene | 原始 ED24 坐标 | 检验重新划分 ED24 训练集后的泛化能力 |

### 三个 LED 新模型的训练细节

| 模型 | checkpoint | 初始化 | Epochs | Batch size | Seq len | Samples/segment | LR | Weight decay | Seed | 训练 scene | 保留测试 scene |
|------|------------|--------|--------|------------|---------|------------------|----|--------------|------|------------|----------------|
| EDformer-LED-finetuned | `led_finetune/checkpoints/led_finetune_best.pth` | ED24 pretrained | 10 | 16 | 4096 | 4 | 1e-4 | 1e-4 | 42 | 590 | 10 |
| EDformer-LED-scratch-raw | `led_finetune/experiments/scratch_raw_60ep/checkpoints/led_finetune_best.pth` | Scratch | 60 | 16 | 4096 | 4 | 1e-4 | 1e-4 | 42 | 590 | 10 |
| EDformer-LED-scratch-norm | `led_finetune/experiments/scratch_xy_normalized_60ep/checkpoints/led_finetune_best.pth` | Scratch | 60 | 16 | 4096 | 4 | 1e-4 | 1e-4 | 42 | 590 | 10 |

### ED24 重新训练实验设计

新增 `EDformer-ED24-retrained` 实验用于回答：原始 EDformer 在 ED24 上的高 AUC 是否依赖被测试 scene，重新排除测试 scene 训练后还能否泛化到 ED24、Driving、DVSCLEAN 和 LED。

- **工程目录**: `ed24_retrain/`
- **训练数据**: ED24 中排除 `Bicycle_02` 和 `Pedestrain_06` 后的 98 个 scene
- **测试数据**: `Bicycle_02`、`Pedestrain_06`、Driving Mix、DVSCLEAN、LED 10 个保留 scene
- **训练电压**: 为保持原 EDformer 训练设置一致，默认只使用 1.5V 到 3.5V
- **训练设置**: scratch 初始化，60 epochs，batch size 96，seq len 4096，learning rate 1e-3，weight decay 1e-2，seed 230086
- **checkpoint**: `ed24_retrain/checkpoints/ed24_retrain_best.pth`
- **评估结果目录**: `ed24_retrain/results/generalization/`

论文中建议写为：`EDformer trained on ED24 excluding Bicycle_02 and Pedestrain_06`。该模型仍是 EDformer 结构，但不是原始 released pretrained 权重。

保留测试 scene 固定为：`scene_100`, `scene_1004`, `scene_1018`, `scene_1028`, `scene_1032`, `scene_1033`, `scene_1034`, `scene_1043`, `scene_1045`, `scene_1046`。这些 scene 没有参与 LED 微调或 LED 从零训练。

### 关于 LED 效果偏低的解释

当前证据更支持 **LED 数据域与 ED24/Driving/DVSCLEAN 存在明显差异**，而不是简单说明 EDformer 结构“不行”。理由如下：

1. 原始 ED24 预训练模型在 ED24 上平均 AUC 为 0.982015，在 Driving 上为 0.940906，在 DVSCLEAN 上为 0.980808，说明该结构和原权重在其训练域或相近噪声域上有效。
2. 同一个 ED24 预训练模型直接 zero-shot 测 LED 只有 0.581471，但用 LED 训练集微调后提升到 0.789621，说明 LED 低分主要与域差异有关，模型经过 LED 域适配后可以显著改善。
3. LED 从零训练 raw 与 LED 微调几乎相同，平均 AUC 分别为 0.789326 和 0.789621；LED 从零训练 xy 归一化只小幅提高到 0.792972，说明问题不主要是 ED24 预训练偏置，也不主要是 1280×720 坐标尺度。
4. DVSCLEAN 分辨率同为 1280×720，但 ED24 预训练和 LED 训练模型在 DVSCLEAN 上都很高，说明分辨率本身不是 LED 困难的充分解释。
5. 更合理的解释是 LED 使用 Prophesee 相机，噪声来源、事件密度、背景活动、热像素/BA 噪声、标注生成方式和类别比例与 ED24/DAVIS 及 DVSCLEAN 模拟噪声不同，导致 ED24 预训练权重 zero-shot 迁移失败。

因此，LED 结果不应被简单表述为“EDformer 模型不行”。更准确的论文表述是：**原始 EDformer 在 LED/Prophesee 数据上存在明显跨传感器域差异；LED 域微调可以显著改善性能，但 LED 专用训练后的模型在 Driving 和 ED24 上泛化下降，说明其学习到的判别特征更偏 LED 域。**

### 对比实验可信度与推荐写法

1. 如果目标是比较“原始算法在新数据集上的泛化能力”，应使用 **EDformer-ED24-pretrained zero-shot** 结果。这是最接近原方法发布权重的设置，对比可信，但 LED 数值较低，需要说明存在跨相机域差异。
2. 如果目标是比较“同一结构在 LED 训练集参与训练后的上限性能”，可以报告 **EDformer-LED-finetuned** 或 **EDformer-LED-scratch**，但必须在表格或注释中标明 “trained/fine-tuned on LED training split”。这类结果不再是 zero-shot 泛化，而是 LED 域适配结果。
3. 微调后的模型仍是 EDformer，因为网络结构、输入序列形式和推理方式没有改；但它不是“原始 ED24 预训练 EDformer”，而是 “LED-finetuned EDformer”。只要论文明确区分训练设置，对比实验是可信的。
4. 推荐论文主表同时给出两列：`EDformer (ED24 pretrained, zero-shot)` 和 `EDformer (LED fine-tuned)`。前者说明泛化能力，后者说明 EDformer 结构在 LED 数据上经过域适配后的能力。
5. LED 从零训练两个模型主要作为消融实验：证明从零训练没有明显超过 ED24 预训练后微调，且 xy 归一化收益很小。这可以支持“LED 难点主要来自数据/噪声域差异，而非简单坐标尺度或预训练偏置”的结论。

---

## 1. driving_mix_result 多频率评估

**数据路径**: `/home/stu1/LJC/EVS/Datasets/ECCV2024_datasets/AUC_test/`
**评估脚本**: `scripts/eval_driving_mix.py`

| 频率 (Hz) | AUC | 数据量 (events) |
|-----------|------|-----------------|
| 1 Hz | **0.954152** | 3,081,598 |
| 3 Hz | **0.947175** | 3,777,022 |
| 5 Hz | **0.942438** | 4,478,143 |
| 7 Hz | **0.934371** | 5,169,767 |
| 10 Hz | **0.926393** | 6,218,744 |

**结论**: 所有频率下 AUC 均 > 0.92，性能优秀。频率升高 → AUC 逐步下降（1Hz: 0.954 → 10Hz: 0.926，下降约 2.8%），因为高频场景噪声更大。

---

## 2. ED24 数据集多电压评估

**数据路径**: `/home/stu1/LJC/EVS/Datasets/ECCV2024_datasets/ED24/`
**评估脚本**: `scripts/eval_ed24.py`

### Pedestrain_06

| 电压 (V) | AUC |
|----------|------|
| 1.8V | **0.977345** |
| 2.1V | **0.976287** |
| 2.5V | **0.971501** |
| 3.3V | **0.961457** |

### Bicycle_02

| 电压 (V) | AUC |
|----------|------|
| 1.8V | **0.995207** |
| 2.1V | **0.995429** |
| 2.5V | **0.993000** |
| 3.3V | **0.985895** |

**结论**:
1. **整体性能极佳**: 两个数据集的 AUC 均 > 0.96，Bicycle_02 更是接近 1.0（0.985~0.995），说明 EDformer 对这些场景几乎完美去噪。
2. **电压与性能呈负相关**: 电压越高，传感器噪声越大，AUC 略微下降。趋势在两个数据集上一致。
3. **Bicycle_02 表现优于 Pedestrain_06**: 可能因为 Bicycle 场景运动模式更简单，噪声/信号比更低。
4. **模型泛化能力好**: 从高频评估（driving_mix）到电压评估（ED24），EDformer 在不同噪声类型下都保持高 AUC。

---

## 3. DVSCLEAN 数据集多噪声评估

**数据路径**: `/home/stu1/LJC/EVS/Datasets/DVSCLEAN/simulated_data/`
**评估脚本**: `scripts/eval_dvsclean.py`
**数据格式**: HDF5 (.hdf5)

| Video | Noise=50 | Noise=100 |
|-------|----------|-----------|
| MAH00444 | **0.989261** | **0.978885** |
| MAH00446 | **0.993814** | **0.989393** |
| MAH00447 | **0.983080** | **0.969578** |
| MAH00448 | **0.988210** | **0.980195** |
| MAH00449 | **0.977526** | **0.958138** |

**结论**:
1. **所有场景 AUC > 0.95**: EDformer 在 DVSCLEAN 模拟噪声数据上表现极佳。
2. **Noise=50 全面优于 Noise=100**: 噪声越大，去噪更困难，AUC 下降约 0.01~0.02。符合预期。
3. **MAH00446 表现最佳**: Noise=50 时达 0.994，Noise=100 也保持 0.989。
4. **MAH00449 表现最弱**: Noise=100 时 AUC 降至 0.958，场景可能更复杂或运动更剧烈。

---

## 4. LED 数据集多场景评估

**数据路径**: `/home/stu1/LJC/EVS/Datasets/LED/LED_Train_upload/`
**评估脚本**: `scripts/eval_led.py`
**数据格式**: numpy (.npy), 每个场景拼接 10 个片段 (00031~00040)
**标签策略**: 精确集合匹配。`raw_events = noise_events + denoised_events` 已按多重集验证。为匹配 ED24 训练标签语义，label=1 为噪声（去除），label=0 为信号（保留）。

| Scene | AUC | 总事件数 | 信号比例 | 噪声比例 |
|-------|-----|----------|----------|----------|
| scene_100 | **0.620102** | 1,992,811 | 95.8% | 4.2% |
| scene_1004 | **0.579085** | 1,990,561 | 89.8% | 10.2% |
| scene_1018 | **0.577195** | 1,869,704 | 91.1% | 8.9% |
| scene_1028 | **0.551072** | 1,828,236 | 93.8% | 6.2% |
| scene_1032 | **0.617092** | 1,937,805 | 92.6% | 7.4% |
| scene_1033 | **0.559350** | 1,977,158 | 90.8% | 9.2% |
| scene_1034 | **0.574792** | 1,992,578 | 90.0% | 10.0% |
| scene_1043 | **0.610079** | 1,994,430 | 92.4% | 7.6% |
| scene_1045 | **0.564652** | 1,911,309 | 91.6% | 8.4% |
| scene_1046 | **0.561292** | 1,996,926 | 92.8% | 7.2% |

**结论**:
1. **原 AUC < 0.5 的主因是 LED 标签方向写反**: ED24 中 label=1 随电压/噪声增加而增加，说明模型训练时的正类是噪声；原 LED 转换脚本却把 label=1 写成信号。修正为 label=1 噪声后，LED AUC 为 0.55~0.62。
2. **原因分析**: 
   - EDformer 在 ED24（DAVIS相机，346×260）上训练，学到的噪声模式针对 DAVIS 传感器
   - DVSCLEAN 虽然分辨率不同（1280×720），但其噪声是仿真的，与训练分布接近
   - LED 使用 Prophesee 相机，噪声模式（BA噪声、热像素等）与 DAVIS 本质不同
   - 仅做坐标缩放到 DAVIS 尺度不能改善 LED，说明问题不是简单分辨率未对齐
3. **建议**: 修正标签后仍只略高于随机，作为跨相机零样本对比可保留；若论文要比较 LED 数据集上的算法能力，应使用 LED 训练集微调或重新训练后再测试。

**数据处理**: 转换脚本 `scripts/convert_led.py`，精确标签 CSV 在 `data/LED/` 目录。

---

## 5. EDformer-LED 微调模型

本节是新的模型分支，与第 1-4 节中的 ED24 预训练模型区分开。

- **模型定义**: EDformer-LED-finetuned
- **初始化权重**: `/home/stu1/LJC/EVS/DL_ED/EDformer-main/pretrained_model.pth`
- **微调工程**: `/home/stu1/HJX/eventcam/EDformer/led_finetune/`
- **训练数据**: LED 训练集中排除保留测试集后的 590 个 scene
- **保留测试集**: `scene_100`, `scene_1004`, `scene_1018`, `scene_1028`, `scene_1032`, `scene_1033`, `scene_1034`, `scene_1043`, `scene_1045`, `scene_1046`
- **测试集列表**: `led_finetune/test_scenes.txt`
- **标签语义**: 与 ED24 保持一致，`label=1` 为噪声，`label=0` 为信号
- **checkpoint 输出**: `led_finetune/checkpoints/`
- **LED 微调测试结果**: `led_finetune/results/led_finetune_auc.csv`

### LED 保留测试集结果

**训练完成时间**: 2026-05-20 16:22  
**训练设置**: 单卡 GPU 1，10 epochs，batch size 16，learning rate 1e-4  
**测试 checkpoint**: `led_finetune/checkpoints/led_finetune_best.pth`

| Scene | Zero-shot AUC | LED 微调 AUC | 总事件数 | 信号比例 | 噪声比例 |
|-------|---------------|--------------|----------|----------|----------|
| scene_100 | 0.620102 | **0.861779** | 1,990,656 | 95.8% | 4.2% |
| scene_1004 | 0.579085 | **0.762612** | 1,986,560 | 89.8% | 10.2% |
| scene_1018 | 0.577195 | **0.785972** | 1,867,776 | 91.1% | 8.9% |
| scene_1028 | 0.551072 | **0.778933** | 1,826,816 | 93.8% | 6.2% |
| scene_1032 | 0.617092 | **0.787554** | 1,937,408 | 92.6% | 7.4% |
| scene_1033 | 0.559350 | **0.772994** | 1,974,272 | 90.8% | 9.2% |
| scene_1034 | 0.574792 | **0.786111** | 1,990,656 | 90.0% | 10.0% |
| scene_1043 | 0.610079 | **0.817087** | 1,990,656 | 92.4% | 7.6% |
| scene_1045 | 0.564652 | **0.779512** | 1,908,736 | 91.6% | 8.4% |
| scene_1046 | 0.561292 | **0.763657** | 1,994,752 | 92.8% | 7.2% |

**平均 AUC**:

- ED24 预训练模型 zero-shot LED: 0.581471
- EDformer-LED 微调模型: **0.789621**
- 平均提升: **+0.208150**

### EDformer-LED 微调模型跨数据集泛化

使用 `led_finetune/checkpoints/led_finetune_best_state_dict.pth` 重新评估 Driving、ED24、DVSCLEAN，结果保存于：

`led_finetune/results/generalization/`

#### Driving Mix

| 频率 (Hz) | ED24 预训练 AUC | LED 微调 AUC | 变化 |
|-----------|------------------|--------------|------|
| 1 Hz | 0.954152 | **0.907497** | -0.046655 |
| 3 Hz | 0.947175 | **0.911401** | -0.035774 |
| 5 Hz | 0.942438 | **0.912638** | -0.029800 |
| 7 Hz | 0.934371 | **0.907347** | -0.027024 |
| 10 Hz | 0.926393 | **0.904193** | -0.022200 |

平均 AUC：ED24 预训练 0.940906，LED 微调 0.908615，变化 -0.032290。

#### ED24

| 数据集 | 电压 | ED24 预训练 AUC | LED 微调 AUC | 变化 |
|--------|------|------------------|--------------|------|
| Pedestrain_06 | 1.8V | 0.977345 | **0.935183** | -0.042162 |
| Pedestrain_06 | 2.1V | 0.976287 | **0.920398** | -0.055889 |
| Pedestrain_06 | 2.5V | 0.971501 | **0.888068** | -0.083433 |
| Pedestrain_06 | 3.3V | 0.961457 | **0.863186** | -0.098271 |
| Bicycle_02 | 1.8V | 0.995207 | **0.963758** | -0.031449 |
| Bicycle_02 | 2.1V | 0.995429 | **0.949706** | -0.045723 |
| Bicycle_02 | 2.5V | 0.993000 | **0.941625** | -0.051375 |
| Bicycle_02 | 3.3V | 0.985895 | **0.927686** | -0.058209 |

平均 AUC：ED24 预训练 0.982015，LED 微调 0.923701，变化 -0.058314。

#### DVSCLEAN

| Video | Noise | ED24 预训练 AUC | LED 微调 AUC | 变化 |
|-------|-------|------------------|--------------|------|
| MAH00444 | 50 | 0.989261 | **0.998620** | +0.009359 |
| MAH00444 | 100 | 0.978885 | **0.997500** | +0.018615 |
| MAH00446 | 50 | 0.993814 | **0.997333** | +0.003519 |
| MAH00446 | 100 | 0.989393 | **0.994772** | +0.005379 |
| MAH00447 | 50 | 0.983080 | **0.997992** | +0.014912 |
| MAH00447 | 100 | 0.969578 | **0.996558** | +0.026980 |
| MAH00448 | 50 | 0.988210 | **0.997750** | +0.009540 |
| MAH00448 | 100 | 0.980195 | **0.996441** | +0.016246 |
| MAH00449 | 50 | 0.977526 | **0.997095** | +0.019569 |
| MAH00449 | 100 | 0.958138 | **0.995208** | +0.037070 |

平均 AUC：ED24 预训练 0.980808，LED 微调 0.996927，变化 +0.016119。

**泛化结论**:

1. LED 微调显著改善 LED 保留测试集，但没有把 LED 提升到 ED24/DVSCLEAN 的水平，说明 LED/Prophesee 域仍更难。
2. LED 微调后 Driving 和 ED24 有下降，属于跨域微调后的遗忘现象；ED24 下降更明显。
3. DVSCLEAN 反而提升到接近 0.997，说明 LED 微调学到的噪声判别特征对 DVSCLEAN 的模拟噪声有正迁移。
4. 论文中建议把 ED24 预训练模型和 EDformer-LED 微调模型分开列：前者表示原模型泛化，后者表示 LED 域适配后的模型。

### LED 从零训练实验

为判断 LED 结果是否受 ED24 预训练权重或 LED 坐标尺度影响，新增两个从零训练实验。两者都排除同一组 10 个 LED 保留测试 scene。

| 实验名 | 初始化 | 坐标输入 | GPU | Epochs | 输出目录 |
|--------|--------|----------|-----|--------|----------|
| `scratch_raw_60ep` | 从零训练 | 原始 LED 坐标 | GPU 1 | 60 | `led_finetune/experiments/scratch_raw_60ep/` |
| `scratch_xy_normalized_60ep` | 从零训练 | x/y 按 1280×720 归一化到 [0,1] | GPU 2 | 60 | `led_finetune/experiments/scratch_xy_normalized_60ep/` |

两个实验均已完成。每个实验先训练，再自动用 best checkpoint 评估同一组 LED 保留测试 scene。最终结果分别保存为：

- `led_finetune/experiments/scratch_raw_60ep/results/led_auc.csv`
- `led_finetune/experiments/scratch_xy_normalized_60ep/results/led_auc.csv`

**LED 保留测试集平均 AUC 对比**:

| 模型/实验 | 初始化 | 坐标输入 | 平均 AUC | 相对 ED24 zero-shot | 相对 LED 微调 |
|-----------|--------|----------|----------|----------------------|---------------|
| ED24 预训练 zero-shot | ED24 预训练 | 原始输入 | 0.581471 | - | -0.208150 |
| ED24 预训练 + LED 微调 | ED24 预训练 | 原始 LED 坐标 | 0.789621 | +0.208150 | - |
| LED 从零训练 raw | 随机初始化 | 原始 LED 坐标 | 0.789326 | +0.207855 | -0.000295 |
| LED 从零训练 xy 归一化 | 随机初始化 | x/y 归一化到 [0,1] | **0.792972** | **+0.211501** | **+0.003351** |

**逐 scene AUC 对比**:

| Scene | ED24 zero-shot | ED24+LED 微调 | LED scratch raw | LED scratch xy 归一化 |
|-------|----------------|---------------|-----------------|------------------------|
| scene_100 | 0.620102 | 0.861779 | 0.854944 | **0.865609** |
| scene_1004 | 0.579085 | 0.762612 | 0.760449 | **0.763551** |
| scene_1018 | 0.577195 | 0.785972 | **0.789110** | 0.789007 |
| scene_1028 | 0.551072 | 0.778933 | 0.782311 | **0.783208** |
| scene_1032 | 0.617092 | 0.787554 | 0.789706 | **0.793584** |
| scene_1033 | 0.559350 | 0.772994 | 0.764715 | **0.778696** |
| scene_1034 | 0.574792 | 0.786111 | 0.786618 | **0.787991** |
| scene_1043 | 0.610079 | 0.817087 | 0.818596 | **0.821161** |
| scene_1045 | 0.564652 | 0.779512 | 0.779564 | **0.785075** |
| scene_1046 | 0.561292 | 0.763657 | **0.767249** | 0.761841 |

**从零训练结论**:

1. 完全使用 LED 从零训练确实可以把 LED 从 zero-shot 的 0.581471 提升到约 0.79，但并没有明显超过“ED24 预训练 + LED 微调”。
2. xy 坐标归一化是目前四组 LED 实验中的最高结果，平均 AUC 为 **0.792972**，但只比 LED 微调高 **0.003351**，提升很小，不能解释为质变。
3. raw 从零训练和 LED 微调几乎相同，说明 LED 上限不主要受 ED24 预训练偏置限制；主要瓶颈更可能来自 LED/Prophesee 数据域、噪声定义、标注方式、类别分布或 EDformer 对该数据形态的适配能力。
4. DVSCLEAN 同为 1280×720 但效果很好，结合 xy 归一化只带来小幅提升，说明分辨率/坐标尺度不是 LED 结果偏低的主因。
5. 若论文中要体现“原始 EDformer 的跨数据集泛化能力”，应报告 ED24 预训练 zero-shot 的 LED 结果；若要公平比较“在 LED 数据集上训练后的算法能力”，应优先报告 LED 微调或 LED 从零训练结果，并明确训练数据使用 LED 且保留 scene 未参与训练。
6. 当前 LED 从零训练结果有对比价值，但它不是一个很强的 LED 专用结果。若需要进一步冲 LED 性能，下一步更值得尝试的是类别不均衡处理、focal loss/weighted BCE、按 scene 验证集选择 checkpoint、阈值指标与 AUC 同时报、以及针对 Prophesee 事件分布的预处理，而不是单纯延长训练或只改坐标尺度。

### LED 从零训练模型跨数据集泛化

使用两个 LED 从零训练 best checkpoint 重新评估 Driving、ED24、DVSCLEAN。结果保存于：

- `led_finetune/experiments/scratch_raw_60ep/results/generalization/`
- `led_finetune/experiments/scratch_xy_normalized_60ep/results/generalization/`

说明：`scratch_raw_60ep` 泛化评估使用原始 x/y 坐标；`scratch_xy_normalized_60ep` 泛化评估按对应传感器尺寸归一化 x/y，Driving/ED24 使用 346×260，DVSCLEAN 使用 1280×720。

**平均 AUC 汇总**:

| 数据集 | ED24 预训练 | LED 微调 | LED scratch raw | LED scratch xy 归一化 |
|--------|-------------|----------|-----------------|------------------------|
| Driving Mix | 0.940906 | 0.908615 | 0.875710 | 0.872293 |
| ED24 | 0.982015 | 0.923701 | 0.870263 | 0.883698 |
| DVSCLEAN | 0.980808 | 0.996927 | **0.996976** | 0.996851 |

#### Driving Mix 泛化

| 频率 | ED24 预训练 | LED 微调 | scratch raw | scratch xy 归一化 |
|------|-------------|----------|-------------|--------------------|
| 1 Hz | 0.954152 | 0.907497 | 0.898318 | 0.898632 |
| 3 Hz | 0.947175 | 0.911401 | 0.885913 | 0.883394 |
| 5 Hz | 0.942438 | 0.912638 | 0.874945 | 0.868170 |
| 7 Hz | 0.934371 | 0.907347 | 0.865627 | 0.862207 |
| 10 Hz | 0.926393 | 0.904193 | 0.853749 | 0.849062 |

平均 AUC：ED24 预训练 0.940906，LED 微调 0.908615，scratch raw 0.875710，scratch xy 归一化 0.872293。

#### ED24 泛化

| 数据集 | 电压 | ED24 预训练 | LED 微调 | scratch raw | scratch xy 归一化 |
|--------|------|-------------|----------|-------------|--------------------|
| Pedestrain_06 | 1.8V | 0.977345 | 0.935183 | 0.849086 | 0.864724 |
| Pedestrain_06 | 2.1V | 0.976287 | 0.920398 | 0.869200 | 0.854395 |
| Pedestrain_06 | 2.5V | 0.971501 | 0.888068 | 0.866221 | 0.841457 |
| Pedestrain_06 | 3.3V | 0.961457 | 0.863186 | 0.853375 | 0.833847 |
| Bicycle_02 | 1.8V | 0.995207 | 0.963758 | 0.859745 | 0.924007 |
| Bicycle_02 | 2.1V | 0.995429 | 0.949706 | 0.883897 | 0.921457 |
| Bicycle_02 | 2.5V | 0.993000 | 0.941625 | 0.896855 | 0.918317 |
| Bicycle_02 | 3.3V | 0.985895 | 0.927686 | 0.883722 | 0.911377 |

平均 AUC：ED24 预训练 0.982015，LED 微调 0.923701，scratch raw 0.870263，scratch xy 归一化 0.883698。

#### DVSCLEAN 泛化

| Video | Noise | ED24 预训练 | LED 微调 | scratch raw | scratch xy 归一化 |
|-------|-------|-------------|----------|-------------|--------------------|
| MAH00444 | 50 | 0.989261 | 0.998620 | 0.998217 | 0.997765 |
| MAH00444 | 100 | 0.978885 | 0.997500 | 0.996814 | 0.996034 |
| MAH00446 | 50 | 0.993814 | 0.997333 | 0.998182 | 0.997487 |
| MAH00446 | 100 | 0.989393 | 0.994772 | 0.996838 | 0.995885 |
| MAH00447 | 50 | 0.983080 | 0.997992 | 0.997335 | 0.997829 |
| MAH00447 | 100 | 0.969578 | 0.996558 | 0.995619 | 0.996559 |
| MAH00448 | 50 | 0.988210 | 0.997750 | 0.997897 | 0.997525 |
| MAH00448 | 100 | 0.980195 | 0.996441 | 0.997282 | 0.995720 |
| MAH00449 | 50 | 0.977526 | 0.997095 | 0.996730 | 0.997557 |
| MAH00449 | 100 | 0.958138 | 0.995208 | 0.994848 | 0.996144 |

平均 AUC：ED24 预训练 0.980808，LED 微调 0.996927，scratch raw 0.996976，scratch xy 归一化 0.996851。

**泛化结论**:

1. LED 从零训练模型在 LED 保留测试集上略高或接近 LED 微调，但在 Driving 和 ED24 上泛化明显更差，说明完全从 LED 学到的特征更偏 LED 域。
2. ED24 预训练模型仍是 Driving 和 ED24 上最强的模型；如果论文强调跨数据集泛化，这个结果应作为主对比。
3. LED 微调模型在 Driving/ED24 上比从零训练更稳，说明 ED24 预训练权重保留了一部分通用事件去噪能力。
4. DVSCLEAN 上三个 LED 训练/微调模型都接近 0.997，说明 LED 域训练对 DVSCLEAN 的模拟噪声仍有强正迁移；但这不能代表对 ED24/Driving 也有同样泛化。
5. 论文表述建议：LED 从零训练结果可作为“LED 专用训练”对比；ED24 预训练 zero-shot 可作为“原始模型跨相机泛化”对比；不要把 LED 从零训练模型作为 EDformer 的通用泛化结果。

**训练命令**:

```bash
# scratch_raw_60ep
cd /tmp
CUDA_VISIBLE_DEVICES=1 /home/stu1/.conda/envs/edformer220/bin/python \
  /home/stu1/HJX/eventcam/EDformer/led_finetune/train_led_finetune.py \
  --init scratch \
  --output_dir /home/stu1/HJX/eventcam/EDformer/led_finetune/experiments/scratch_raw_60ep \
  --exclude_scenes /home/stu1/HJX/eventcam/EDformer/led_finetune/test_scenes.txt \
  --coordinate_mode raw \
  --epochs 60 \
  --batch_size 16 \
  --samples_per_segment 4 \
  --lr 1e-4 \
  --device cuda:0 \
  --save_every 5

# scratch_xy_normalized_60ep
cd /tmp
CUDA_VISIBLE_DEVICES=2 /home/stu1/.conda/envs/edformer220/bin/python \
  /home/stu1/HJX/eventcam/EDformer/led_finetune/train_led_finetune.py \
  --init scratch \
  --output_dir /home/stu1/HJX/eventcam/EDformer/led_finetune/experiments/scratch_xy_normalized_60ep \
  --exclude_scenes /home/stu1/HJX/eventcam/EDformer/led_finetune/test_scenes.txt \
  --coordinate_mode normalized \
  --sensor_width 1280 \
  --sensor_height 720 \
  --epochs 60 \
  --batch_size 16 \
  --samples_per_segment 4 \
  --lr 1e-4 \
  --device cuda:0 \
  --save_every 5
```

**评估命令**:

```bash
# scratch_raw_60ep
cd /tmp
CUDA_VISIBLE_DEVICES=1 /home/stu1/.conda/envs/edformer220/bin/python \
  /home/stu1/HJX/eventcam/EDformer/led_finetune/eval_led_checkpoint.py \
  --checkpoint /home/stu1/HJX/eventcam/EDformer/led_finetune/experiments/scratch_raw_60ep/checkpoints/led_finetune_best.pth \
  --scenes /home/stu1/HJX/eventcam/EDformer/led_finetune/test_scenes.txt \
  --coordinate_mode raw \
  --output_csv /home/stu1/HJX/eventcam/EDformer/led_finetune/experiments/scratch_raw_60ep/results/led_auc.csv \
  --device cuda:0

# scratch_xy_normalized_60ep
cd /tmp
CUDA_VISIBLE_DEVICES=2 /home/stu1/.conda/envs/edformer220/bin/python \
  /home/stu1/HJX/eventcam/EDformer/led_finetune/eval_led_checkpoint.py \
  --checkpoint /home/stu1/HJX/eventcam/EDformer/led_finetune/experiments/scratch_xy_normalized_60ep/checkpoints/led_finetune_best.pth \
  --scenes /home/stu1/HJX/eventcam/EDformer/led_finetune/test_scenes.txt \
  --coordinate_mode normalized \
  --sensor_width 1280 \
  --sensor_height 720 \
  --output_csv /home/stu1/HJX/eventcam/EDformer/led_finetune/experiments/scratch_xy_normalized_60ep/results/led_auc.csv \
  --device cuda:0
```

---

## 文件结构

```
/home/stu1/HJX/eventcam/EDformer/
├── Readme.md                          # 本文件（评估结果与结论）
├── scripts/
│   ├── eval_driving_mix.py            # driving_mix 频率评估脚本
│   ├── eval_ed24.py                   # ED24 电压评估脚本
│   ├── eval_dvsclean.py               # DVSCLEAN 噪声评估脚本
│   ├── eval_led.py                    # LED 场景评估脚本
│   └── convert_led.py                 # LED npy→CSV 转换脚本
├── led_finetune/
│   ├── README.md                      # EDformer-LED 微调说明
│   ├── led_dataset.py                 # LED 微调数据集
│   ├── train_led_finetune.py          # 加载 ED24 pretrained 后微调
│   ├── eval_led_checkpoint.py         # 评估 LED 微调 checkpoint
│   ├── test_scenes.txt                # 保留测试 scene
│   ├── checkpoints/                   # 微调 checkpoint
│   ├── logs/                          # 后台训练日志
│   ├── experiments/                    # LED 从零训练实验
│   └── results/                       # 微调模型评估结果
├── data/
│   └── LED/
│       ├── scene_100.csv              # LED 精确标签数据
│       ├── scene_1004.csv
│       ├── scene_1018.csv
│       ├── scene_1028.csv
│       ├── scene_1032.csv
│       ├── scene_1033.csv
│       ├── scene_1034.csv
│       ├── scene_1043.csv
│       ├── scene_1045.csv
│       └── scene_1046.csv
└── results/
    ├── driving_mix_auc.csv            # driving_mix 频率 AUC 汇总
    ├── roc_data_1hz.csv               # 1Hz ROC 曲线数据 (FPR, TPR)
    ├── roc_data_3hz.csv               # 3Hz ROC 曲线数据
    ├── roc_data_5hz.csv               # 5Hz ROC 曲线数据
    ├── roc_data_7hz.csv               # 7Hz ROC 曲线数据
    ├── roc_data_10hz.csv              # 10Hz ROC 曲线数据
    ├── ed24_auc.csv                   # ED24 电压 AUC 汇总
    ├── dvsclean_auc.csv               # DVSCLEAN 噪声 AUC 汇总
    └── led_auc.csv                    # LED 场景 AUC 汇总
```

## 复现命令

```bash
# LED 数据转换 (npy → CSV 精确标签)
cd /tmp && /home/stu1/.conda/envs/edformer220/bin/python \
  /home/stu1/HJX/eventcam/EDformer/scripts/convert_led.py

# driving_mix 多频率评估
cd /tmp && /home/stu1/.conda/envs/edformer220/bin/python \
  /home/stu1/HJX/eventcam/EDformer/scripts/eval_driving_mix.py

# ED24 多电压评估
cd /tmp && /home/stu1/.conda/envs/edformer220/bin/python \
  /home/stu1/HJX/eventcam/EDformer/scripts/eval_ed24.py

# DVSCLEAN 多噪声评估
cd /tmp && /home/stu1/.conda/envs/edformer220/bin/python \
  /home/stu1/HJX/eventcam/EDformer/scripts/eval_dvsclean.py

# LED 多场景评估
cd /tmp && /home/stu1/.conda/envs/edformer220/bin/python \
  /home/stu1/HJX/eventcam/EDformer/scripts/eval_led.py
```
