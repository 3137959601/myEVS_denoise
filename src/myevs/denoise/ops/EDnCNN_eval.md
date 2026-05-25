# EDnCNN 预训练模型 AUC 评估

本工程独立于原始 EdnCNN 代码目录，不覆盖 `/home/stu1/LJC/EVS/DL_ED/edncnn/`。

## 设置

- 原始工程：`/home/stu1/LJC/EVS/DL_ED/edncnn`
- 预训练模型：`/home/stu1/LJC/EVS/DL_ED/edncnn/allData_v8_preTrained.mat`
- MATLAB：`/home/stu1/MATLAB/R2022b/bin/matlab`
- 结果目录：`/home/stu1/HJX/eventcam/EDnCNN/results`

## 推理方式说明

EdnCNN 原始流程依赖 DVSNOISE20 的 APS frame/IMU 生成 EPM 标签；当前 ED24、Driving、DVSCLEAN、LED 数据没有对应 APS/IMU，因此这里只使用预训练 CNN 的事件 time-surface 特征推理，不重新生成 EPM，也不重新训练。

网络输出类别为 `[false, true]`，其中 `true` 对应原始 EPM 概率大于 0.5 的有效事件。本评估使用 `P(noise)=1-P(valid)` 作为噪声分数，并与各数据集标签 `label=1` 为噪声计算 AUC。

## 运行命令

```bash
cd /home/stu1/HJX/eventcam/EDnCNN
/home/stu1/MATLAB/R2022b/bin/matlab -nodisplay -nosplash -r "addpath('scripts'); run_edncnn_pretrained_auc; exit"
```

## 结果文件

- `results/driving_mix_auc.csv`
- `results/ed24_auc.csv`
- `results/dvsclean_auc.csv`
- `results/led_auc.csv`

## 当前结果摘要

- Driving Mix：平均 AUC = 0.868087，结果文件 `/home/stu1/HJX/eventcam/EDnCNN/results/driving_mix_auc.csv`。
- ED24 held-out：平均 AUC = 0.883300，结果文件 `/home/stu1/HJX/eventcam/EDnCNN/results/ed24_auc.csv`。
- DVSCLEAN：平均 AUC = 0.763251，结果文件 `/home/stu1/HJX/eventcam/EDnCNN/results/dvsclean_auc.csv`。
- LED：平均 AUC = 0.799510，结果文件 `/home/stu1/HJX/eventcam/EDnCNN/results/led_auc.csv`。

## 完整结果

### Driving Mix

| 频率 | AUC |
|------|-----|
| 1Hz | 0.881746 |
| 3Hz | 0.873734 |
| 5Hz | 0.866095 |
| 7Hz | 0.864302 |
| 10Hz | 0.854558 |

平均 AUC：**0.868087**。

### ED24 held-out

| 数据集 | 电压 | AUC |
|--------|------|-----|
| Pedestrain_06 | 1.8V | 0.898486 |
| Pedestrain_06 | 2.1V | 0.883143 |
| Pedestrain_06 | 2.5V | 0.846444 |
| Pedestrain_06 | 3.3V | 0.809764 |
| Bicycle_02 | 1.8V | 0.939236 |
| Bicycle_02 | 2.1V | 0.929164 |
| Bicycle_02 | 2.5V | 0.903195 |
| Bicycle_02 | 3.3V | 0.856968 |

平均 AUC：**0.883300**。

### DVSCLEAN

| Video | Noise | AUC |
|-------|-------|-----|
| MAH00444 | 50 | 0.914565 |
| MAH00444 | 100 | 0.900174 |
| MAH00446 | 50 | 0.574157 |
| MAH00446 | 100 | 0.537788 |
| MAH00447 | 50 | 0.786672 |
| MAH00447 | 100 | 0.756345 |
| MAH00448 | 50 | 0.781409 |
| MAH00448 | 100 | 0.751384 |
| MAH00449 | 50 | 0.830663 |
| MAH00449 | 100 | 0.799350 |

平均 AUC：**0.763251**。

### LED

| Scene | AUC |
|-------|-----|
| scene_100 | 0.838795 |
| scene_1004 | 0.699925 |
| scene_1018 | 0.802020 |
| scene_1028 | 0.865969 |
| scene_1032 | 0.840968 |
| scene_1033 | 0.776307 |
| scene_1034 | 0.710526 |
| scene_1043 | 0.829283 |
| scene_1045 | 0.797333 |
| scene_1046 | 0.833971 |

平均 AUC：**0.799510**。

## 与 EDformer 的对比结论

| 数据集 | EDnCNN 预训练 | EDformer ED24 预训练 | 结论 |
|--------|---------------|----------------------|------|
| Driving Mix | 0.868087 | 0.940906 | EDformer 明显更强 |
| ED24 held-out | 0.883300 | 0.982015 | EDformer 明显更强 |
| DVSCLEAN | 0.763251 | 0.980808 | EDformer 明显更强，EDnCNN 在 MAH00446 上失效明显 |
| LED | 0.799510 | 0.581471 | EDnCNN zero-shot 明显更强 |

结论：

1. **EDnCNN 预训练模型在 LED 上表现最好**，平均 AUC 为 0.799510，已经接近此前 EDformer-LED 微调和 LED 从零训练的约 0.79，明显优于原始 EDformer ED24 预训练模型的 LED zero-shot 结果 0.581471。
2. **EDnCNN 在 ED24、Driving、DVSCLEAN 上明显弱于 EDformer**。这说明 EdnCNN 的预训练权重并不是一个更强的通用事件去噪模型，但它对 LED/Prophesee 数据的 zero-shot 适配反而更好。
3. **DVSCLEAN 结果波动很大**。MAH00444 可达到 0.90 以上，但 MAH00446 只有 0.54-0.57，说明 EDnCNN 对不同视频/噪声组合不稳定，不建议把它作为强泛化 baseline。
4. **论文使用建议**：如果论文重点讨论 LED 数据集，EDnCNN 预训练结果有较强对比价值，因为它不使用 LED 训练集却达到 0.80 左右；如果讨论跨数据集平均泛化，EDformer 仍更强。
5. **方法限制说明**：本评估没有使用 APS/IMU 重新生成 EPM，而是只使用预训练 CNN 对事件 time-surface 特征推理。因此论文中应写清楚是 `pretrained EDnCNN CNN inference without EPM re-estimation`，避免与原论文完整 EPM+EDnCNN 流程混淆。
