# 论文定性去噪图生成模块

这个目录用于维护论文定性图的生成流程，包括候选时间窗筛选、规则算法去噪、事件图渲染、图片拼版，以及后续 EDnCNN / EDFormer 服务器结果导入。

## 目标

- 在当前 `myEVS` 工程内完成定性图生成，不另起工程。
- 复用已有事件读取、C++ 去噪算子和红蓝事件渲染风格。
- 将实验配置、候选图、最终渲染图、拼版图和进度记录分开管理。
- Driving 明确不使用已弃用的 `2hz` 和 `8hz`。
- Pedestrian 固定使用 `Pedestrain_06_3.3`。
- Bicycle 暂不进入最终定性图，保留配置但默认关闭。
- DVSCLEAN 使用 `MAH00444/ratio100` 作为当前定性图候选。
- DVSNOISE20 stairs 只做定性分析，因为它没有噪声标签，不进入 AUC/F1 等定量指标。

## 目录和文件

源码和脚本：

| 路径 | 作用 |
|---|---|
| `src/myevs/qualitative/` | 定性图核心模块。 |
| `scripts/qualitative/` | 可直接运行的命令行脚本。 |

实验产物默认放在 `data/qualitative/`：

| 路径 | 内容 |
|---|---|
| `qualitative_cases.yaml` | 数据集路径、case 开关、扫描窗口、渲染默认值。 |
| `algorithm_params.yaml` | 各数据集组的算法出图参数。 |
| `candidate_manifest.csv` | 候选窗口列表，记录 start_us、窗口长度、事件数、缩略图路径等。 |
| `candidates/<case_id>/` | 候选时间窗缩略图，用来人工挑选适合论文展示的窗口。 |
| `rendered/<case_id>/` | 每个算法单独渲染出的 PNG 图，以及 `render_manifest.csv`。 |
| `events/<case_id>/` | 每个算法输出的事件流 `.npz`，用于复查或重新渲染。 |
| `panels/` | 已拼好的论文图，包含 PNG 和 PDF。 |
| `converted/dvsnoise20/` | DVSNOISE20 `.aedat4` 转换后的 `.npz`。 |
| `external/` | 后续导入的 EDnCNN / EDFormer 结果。 |

## 当前默认 case

| case_id | 状态 | 说明 |
|---|---|---|
| `driving_5hz` | 启用 | Driving 主候选；候选窗口为 30/50/80/100 ms，默认 50 ms。 |
| `driving_3hz` | 关闭 | Driving 备选。 |
| `driving_7hz` | 关闭 | Driving 备选。 |
| `driving_10hz` | 关闭 | Driving 备选。 |
| `ped_3p3` | 启用 | Pedestrian 3.3。 |
| `bike_2p5` | 关闭 | Bicycle 2.5V，暂不进入最终拼版。 |
| `dvsclean_444_ratio100` | 启用 | DVSCLEAN MAH00444 ratio100。 |
| `stairs_125854` | 启用 | DVSNOISE20 stairs，仅定性，需要先转换 `.aedat4`。 |
| `stairs_130316` | 启用 | DVSNOISE20 stairs，仅定性，需要先转换 `.aedat4`。 |
| `stairs_130353` | 启用 | DVSNOISE20 stairs，仅定性，需要先转换 `.aedat4`。 |
| `labfast_115638` | 启用 | DVSNOISE20 labFast，仅定性，需要先转换 `.aedat4`。 |
| `labslow_124009` | 启用 | DVSNOISE20 labSlow，仅定性，需要先转换 `.aedat4`。 |

## 渲染设置

当前默认使用白底、饱和红蓝事件图：

- `scheme = 0`
- `binary = true`
- `deadzone = 0`
- `raw_step = 127`

这样 ON/OFF 事件会直接显示为顶格红/蓝，不再使用淡色渐变。若后续觉得点太硬，可以把 `binary` 改回 `false`，或降低 `raw_step`。

## N149 / Ours 参数

| 数据集组 | r | tau | sigma | alpha |
|---|---:|---:|---:|---:|
| Driving | 2 | 32K us | 1.75 | 0.05 |
| Pedestrian | 5 | 256K us | 2.75 | 0.25 |
| Bicycle | 5 | 256K us | 2.75 | 0.25 |
| DVSCLEAN | 5 | 128K us | 2.5 | 0.25 |
| DVSNOISE20 | 2 | 32K us | 1.75 | 0.05 |
| stairs | 2 | 32K us | 1.75 | 0.05 |

BAF、STCF、EBF、TS、PFD 的出图参数集中在 `algorithm_params.yaml`。论文图中显示为 `STCF`，但底层调用工程方法名 `stcf_original`，不是 `stc`；其中 `min_neighbors` 表示 K 值，空间半径按原始实现固定为 1。部分阈值是视觉默认单点，因为 README2 中记录的是 sweep/AUC 结果，不是唯一强制出图阈值。

## 常用命令

初始化配置：

```powershell
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/qualitative/init_configs.py
```

扫描候选窗口：

```powershell
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/qualitative/scan_candidates.py --case-id driving_5hz
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/qualitative/scan_candidates.py --case-id ped_3p3
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/qualitative/scan_candidates.py --case-id dvsclean_444_ratio100
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/qualitative/scan_candidates.py --case-id stairs_125854
```

渲染一个 case。若不传 `--window-ms`，会使用配置里的默认窗口：

```powershell
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/qualitative/render_case.py --case-id driving_5hz
```

指定候选窗口渲染：

```powershell
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/qualitative/render_case.py --case-id driving_5hz --start-us 3015292 --window-ms 150
```

单独拼版：

```powershell
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/qualitative/make_panel.py `
  --images data/qualitative/rendered/driving_5hz/Noisy.png data/qualitative/rendered/driving_5hz/Ours.png `
  --labels Noisy Ours `
  --cols 2 `
  --out data/qualitative/panels/example_panel.png
```

DVSNOISE20 `.aedat4` 转换。当前本机如果没有 `dv_processing`，脚本会提示需要安装或到服务器转换：

```powershell
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/qualitative/convert_aedat4.py `
  --in D:/hjx_workspace/scientific_reserach/dataset/DVSNOISE20/stairs-2019_10_10_12_58_54.aedat4 `
  --out data/qualitative/converted/dvsnoise20/stairs-2019_10_10_12_58_54.npz
```

批量转换 DVSNOISE20 当前配置中的全部 `.aedat4`：

```powershell
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/qualitative/convert_dvsnoise20.py
```

生成 EDnCNN / EDFormer 期望结果清单：

```powershell
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/qualitative/import_external.py expected `
  --case-id driving_5hz --case-id ped_3p3 --case-id dvsclean_444_ratio100
```

## IEEE 图排版建议

IEEEtran journal 模板中，单栏图使用 `figure`，跨双栏图使用 `figure*`。多算法定性对比图通常适合使用 `figure*`。

示例：

```latex
\begin{figure*}[!t]
\centering
\includegraphics[width=\textwidth]{figures/qualitative_driving_panel.pdf}
\caption{Qualitative denoising comparison on the Driving sequence.}
\label{fig:qual_driving}
\end{figure*}
```

如果使用 `subfig`，注意总宽度不要超过 `\textwidth`，总图的 `\label` 放在 `\caption` 之后，图内文字尽量短。

## 进度记录

| 日期 | 状态 | 说明 |
|---|---|---|
| 2026-05-28 | 初版完成 | 添加定性图模块、脚本、默认配置。 |
| 2026-05-28 | 渲染调整 | 渲染改为饱和红蓝；Driving 重新使用 30/50/80/100 ms 候选窗口，默认 50 ms。 |
| 2026-05-28 | TS 修正 | TS 的 `min_neighbors` 是 time-surface 分数阈值，不是邻居数；默认从 1.0 改为 0.2，避免几乎删空事件。 |
| 2026-05-28 | Ours 出图阈值修正 | Ped/Bike 的 N149 结构参数仍用最优 r/tau/sigma/alpha，但定性图阈值从 1.0 调到 Ped=4.0、Bike=2.0，减少孤立噪声。 |
| 2026-05-28 | 全算法出图阈值修正 | 对 Driving/Ped/Bike 当前窗口用 clean reference 重新 sweep STCF/EBF/TS/PFD/Ours，写回 `algorithm_params.yaml`。 |
| 2026-05-28 | STCF 映射修正 | 论文列名显示为 `STCF`，实际调用工程 `stcf_original`；Pedestrian 原窗口为 `start_us=0, window=100ms`，但该窗口不利于展示 Ours 相对 EBF。 |
| 2026-05-28 | Pedestrian 选图与 TS 修正 | Pedestrian 切换到 `start_us=950000, window=150ms`；该窗口下 Ours F1=0.763，高于 EBF F1=0.744。TS 改为 `r=1, tau=16ms, threshold=0.3`，相对 F1 最优点少保留约 1100 个背景事件，更适合定性图。 |
| 2026-05-28 | DVSCLEAN 替换 Bicycle | Bicycle 默认关闭；新增 `dvsclean_444_ratio100`，选中 `start_us=97688, window=100ms`，并在拼版中按固定 tile 缩放混合分辨率图片。该图仅供参考，因为各算法差异较小。 |
| 2026-05-28 | DVSNOISE20 接入 | 新增 stairs/labFast/labSlow 共 5 个 `.aedat4` case；新增批量转换脚本 `convert_dvsnoise20.py`。本机缺 `dv_processing`，需先在含该依赖的环境转换为 `.npz` 后再扫描/渲染。 |
| 2026-05-28 | stairs 渲染完成 | 安装 `dv-processing` 后成功解析全部 5 个 `.aedat4`；`stairs_125854`（17.4M events, 25.1s）选定窗口 `start_us=7400000, window=50ms`（~101K events），7 算法全量渲染 + 拼版完成。参数参考 Driving（346×260, r=2, tau=32K, sigma=1.75, alpha=0.05）。 |

## 选图记录

人工查看 `candidate_manifest.csv` 和 `candidates/<case_id>/` 后，把最终窗口记录在这里。

| Case | 最终 start_us | 窗口 ms | 选择理由 |
|---|---:|---:|---|
| `driving_5hz` | 1015292 | 50 | 用户认可的 `driving_5hz_w50ms_rank02_t1015292us`。 |
| `ped_3p3` | 950000 | 150 | 原 `0us/100ms` 窗口中 Ours 最优 F1 低于 EBF；切到该窗口后人物仍可辨，且 Ours 高于 EBF。 |
| `dvsclean_444_ratio100` | 97688 | 100 | `w100ms_rank01_t97688us` 飞机轮廓清楚，背景噪声明显，适合定性展示。 |
| `bike_2p5` | 0 | 100 | 已暂时从最终拼版移除，配置保留便于后续恢复。 |
| `stairs_125854` | 7400000 | 50 | DVSNOISE20 stairs 场景，~101K events，人物清晰，无噪声适合展示算法去噪精度差异。 |
| `labfast_115638` | TBD | 50 | DVSNOISE20，仅定性，待转换后选择。 |
| `labslow_124009` | TBD | 50 | DVSNOISE20，仅定性，待转换后选择。 |

## 待办

- 人工确认 BAF/STCF/EBF/TS/PFD 的视觉阈值是否需要微调。
- 在有 `dv_processing` 的环境转换 DVSNOISE20 全部 `.aedat4`。
- 服务器跑完 EDnCNN / EDFormer 后导入结果并统一渲染。
