# myEVS（离线解析/去噪/显示/评估）

这个工程从你的 Qt 上位机保存格式出发，提供离线处理闭环：

- 解析输入：
  - `.evtq`（Qt 解析后事件流二进制）
  - `.csv`（Qt 解析后事件流文本：t,x,y,p）
  - `.hdf5/.h5`（支持 OpenEB / Prophesee HDF5：`/CD/events`）
  - `.aedat/.aedat2`（AEDAT-2.0：int32 address + int32 timestamp，常见 tick=1us；支持 v2e/jAER 风格 DVS 极性事件）
  - `usb_raw.raw`（USB 原始 EVT3 风格 32-bit word 流，按你的 `EvtParserWorker::parseWordColumn()` 位域解析）
- 去噪：实现与 Qt 对齐的 8 种方法（可参数化）
- 保存：把去噪后的事件流保存为 `.evtq` / `.csv` / `.hdf5`
- 显示：按“目标帧率”或“每帧事件数”输出预览，支持极性彩色/二值彩色/配色方案
- 评估：提供基础统计 + 可扩展的指标框架（TP/FP 需要你提供标签或参考真值）

## 安装

推荐使用 **Anaconda / Miniconda** 管理虚拟环境（对 Python 新手更友好）。

方式 A：一键按环境文件创建（推荐）

```powershell
cd D:\hjx_workspace\scientific_reserach\projects\myEVS
conda env create -f environment.yml
conda activate myevs
```

方式 B：手动创建环境

```powershell
conda create -n myevs python=3.11 -y
conda activate myevs
conda install -c conda-forge numpy opencv tqdm h5py -y
pip install -e .
```

说明：
- 本工程默认时间基准：**1 tick = 12.5ns**。
- 事件文件中的 `t` 是 **tick**，但所有去噪参数仍然按 Qt UI 的 **us** 输入；程序会自动转换。
- 对 OpenEB/Prophesee HDF5（通常 `t` 为 us）：程序会自动转成 tick 再进入后续算法。
- 注释：若读取 OpenEB 压缩 HDF5 失败，可设置 `HDF5_PLUGIN_PATH`，或直接在命令里传 `--hdf5-plugin-path`。

插件移植建议（Windows/Linux + conda）：
- OpenEB 压缩 HDF5 依赖 ECF 解码插件（Windows 常见：`H5Zecf.dll` / `hdf5_ecf_codec.dll`；Linux 常见：`libh5zecf.so` / `libhdf5_ecf_codec.so`）。
- Windows 推荐放到：`%CONDA_PREFIX%\Library\hdf5\plugin`。
- Linux 推荐放到：`${CONDA_PREFIX}/lib/hdf5/plugin`（或 `${CONDA_PREFIX}/lib64/hdf5/plugin`）。
- myEVS 会优先自动查找上述环境目录；找到后通常无需每次再写 `--hdf5-plugin-path`。
- 也可直接把插件随工程放在相对路径：`Library/hdf5/plugin`（仓库根目录下，Linux/Windows 都可）。
- 读取 HDF5 时，myevs 会尝试把该目录自动同步到当前环境插件目录（不可写时回退为直接使用工程目录）。
- 可选：通过环境变量 `MYEVS_HDF5_PLUGIN_DIR` 指定自定义插件目录。

Linux 服务器最小迁移步骤（推荐）：

```bash
# 1) 把 OpenEB 的 ECF .so 放进项目（便于和代码一起迁移）
mkdir -p Library/hdf5/plugin
cp /path/to/openeb/build/lib/hdf5/plugin/*.so Library/hdf5/plugin/

# 2) 激活环境后直接运行（myevs 会自动发现/同步）
conda activate myevs
myevs stats --in data/prophesee_hand/prophesee_hand.hdf5
```

若你的服务器目录权限受限，不能写 conda 环境目录，可显式指定：

```bash
export MYEVS_HDF5_PLUGIN_DIR=/abs/path/to/Library/hdf5/plugin
# 或
export HDF5_PLUGIN_PATH=/abs/path/to/Library/hdf5/plugin
```

## 快速使用

### 1) 解析 / 转换

- 读取 USB raw 并转成 evtq / hdf5：

```powershell
myevs convert-usb-raw --in usb_raw.raw --out out.evtq --width 640 --height 512
myevs convert-usb-raw --in usb_raw.raw --out out.hdf5 --width 640 --height 512
myevs convert-usb-raw --in data\30WEVS_hand\hand2.raw --out data\30WEVS_hand\hand2.hdf5 --width 640 --height 512
myevs convert-usb-raw --in data/usb_raw_20260130_223313.raw --out data/usb_evtq.evtq --width 640 --height 512
```

说明：Python 解析器不依赖后缀名，`.raw/.bin` 都可以；只要文件内容是同一份 USB 原始 32-bit word 流即可。

- evtq/hdf5/csv 互转：

```powershell
myevs convert --in in.evtq --out out.csv
myevs convert --in in.evtq --out out.hdf5
myevs convert --in in.hdf5 --out out.evtq
myevs convert --in data\prophesee_hand\prophesee_hand.hdf5 --out data\prophesee_hand\prophesee_hand.hdf5.evtq

# AEDAT2（例如 DND21 数据集）
myevs stats --in D:\hjx_workspace\scientific_reserach\dataset\DND21\driving\driving.aedat --progress
myevs convert --in D:\hjx_workspace\scientific_reserach\dataset\DND21\driving\driving.aedat --out data\dnd21_driving.evtq --progress
myevs view --in D:\hjx_workspace\scientific_reserach\dataset\DND21\driving\driving.aedat --mode fps --fps 60 --color onoff --scheme 1 --no-hold
myevs view --in D:\hjx_workspace\scientific_reserach\dataset\DND21\driving\driving.aedat --style prophesee 
myevs view --in D:\hjx_workspace\scientific_reserach\dataset\DND21\hotel-bar\hotel-bar-segment.aedat --style prophesee 
myevs view --in D:\hjx_workspace\scientific_reserach\dataset\DND21\hotel-bar\Davis346mini-2017-04-29T16-04-39+0200-00000004-0_CNE_bar_2017.aedat --mode fps --fps 60 --color onoff --scheme 1 --no-hold
# 若自动推断分辨率失败，可显式指定（DND21 driving 为 346x260）
myevs stats --in D:\hjx_workspace\scientific_reserach\dataset\DND21\driving\driving.aedat --width 346 --height 260 --assume aedat2 --progress
```

- Prophesee raw 先转 hdf5，再直接用 myevs：

```powershell
# 1) OpenEB: raw -> hdf5
D:\software\openeb-new\build\bin\Release\metavision_file_to_hdf5.exe -i data\prophesee_hand\prophesee_hand.raw

# 2) myEVS 直接读取 hdf5（会自动处理 OpenEB 的 /CD/events）
myevs stats --in data\prophesee_hand\prophesee_hand.hdf5 --hdf5-plugin-path D:\software\openeb-new\build\lib\hdf5\plugin
myevs view --in data\prophesee_hand\prophesee_hand.hdf5 --hdf5-plugin-path D:\software\openeb-new\build\lib\hdf5\plugin --mode fps --fps 60 --color onoff --scheme 1 --no-hold

myevs view --in data\prophesee_hand\prophesee_hand.hdf5 --hdf5-plugin-path D:\software\openeb-new\build\lib\hdf5\plugin --mode fps --fps 30 --color onoff --scheme 1 --no-hold --out-video data\prophesee_hand\prophesee_hand.mp4 --no-gui

myevs view --in data\prophesee_hand\prophesee_hand.hdf5 --hdf5-plugin-path D:\software\openeb-new\build\lib\hdf5\plugin --mode fps --fps 30 --color onoff --scheme 1 --no-hold --rotate-180 --out-video data\prophesee_hand\prophesee_hand_rot180.mp4 --no-gui --rotate-180 #翻转180°

myevs view --in data\30WEVS_hand\hand2.hdf5 --hdf5-plugin-path D:\software\openeb-new\build\lib\hdf5\plugin --mode fps --fps 30 --color onoff --scheme 1 --no-hold
```

注释：
- 若画面左右/上下都反了，可加 `--rotate-180`（等价于 `--flip-x --flip-y`）。
- 若漏传 `--hdf5-plugin-path` 导致 HDF5 解码失败，导出会报错且不会保留有效视频文件；旧版本遗留的极小 mp4（如几百字节）通常不可播放。

### 2) 去噪并保存

```powershell
myevs denoise --in in.evtq --out denoised.evtq --method 3 --time-us 2000 --min-neighbors 50 --refractory-us 2000
myevs denoise --in in.hdf5 --out denoised.hdf5 --method 3 --time-us 2000 --min-neighbors 50 --refractory-us 2000
myevs denoise --in data\usb_evtq.evtq --out data\denoised_method3.evtq --method 3 --time-us 2000 --min-neighbors 50 --refractory-us 2000
```

说明（与 Qt 一致的 method id）：
- 0=关闭
- 1=STC（时空相关性）
- 2=Refractory（不应期）
- 3=HotPixel（热像素屏蔽）
- 4=BAF（背景活动滤波，简化版）
- 5=Combo（STC + Refractory）
- 6=RateLimit（每像素限流）
- 7=GlobalGate（全局突发门控/抗频闪）
- 8=DP（交替极性 + 同极性超时放行）

组合测试（自定义 pipeline，按顺序执行）：

```powershell
myevs denoise --in in.evtq --out denoised.evtq --pipeline "7,1,2" \
  --time-us 2000 --radius-px 1 --min-neighbors 2 --refractory-us 50
```

进度条（不影响默认性能，只有加了才显示）：

```powershell
myevs denoise --in in.evtq --out denoised.evtq --method 3 --time-us 2000 --min-neighbors 50 --refractory-us 2000 --progress
myevs convert --in in.evtq --out out.csv --progress
myevs stats --in in.evtq --progress
```

加速（Numba，可选）：
- 适用于 CPU 利用率很低（单核瓶颈、总 CPU 约 5%）且 STC 等算法很慢的情况。
- 第一次运行会触发 JIT 编译，可能慢几十秒；后续会明显变快。
- 当前 `--engine numba` 仅加速 STC（method=1）。

```powershell
myevs denoise --in in.evtq --out stc_numba.evtq --method 1 --time-us 2000 --radius-px 1 --min-neighbors 2 --engine numba --progress
myevs sweep --in in.evtq --method 1 --param time-us --values 200,500,1000,2000,5000,10000 --radius-px 1 --min-neighbors 2 --engine numba --out-csv sweep_stc.csv --progress
```

### 3) 可视化播放

- 按目标帧率：

```powershell
myevs view --in denoised.evtq --mode fps --fps 60 --color onoff --scheme 1
myevs view --in denoised.hdf5 --mode fps --fps 60 --color onoff --scheme 1

myevs view --in data\usb_raw_20260130_223313.raw --width 640 --height 512 --mode fps --fps 60 --color onoff --scheme 1
myevs view --in data\usb_evtq.evtq --mode fps --fps 60 --color onoff --scheme 1 --no-hold
```


如果你觉得“新画面叠在旧画面上、越来越模糊”，这是因为默认开启了 `hold`（灰度缓冲帧间保持，用于做“事件轨迹/拖影”效果）。
想要每一帧都更干净清晰，直接加 `--no-hold`：

播放更流畅的常用调参：
- 降低每个事件的强度：`--raw-step 3` 或 `--raw-step 5`（默认 10 会更“糊”也更容易饱和）
- 改用按事件数出帧：`--mode events --events-per-frame 50000`（通常更稳定）
- 需要尽量贴近 60fps 的“墙钟播放”时：加 `--realtime`（尽力按 --fps 节流）

- 按每帧事件数：

```powershell
myevs view --in denoised.evtq --mode events --events-per-frame 200000 --color onoff
myevs view --in data\usb_evtq.evtq --mode events --events-per-frame 20000 --color onoff --scheme 1 --no-hold
```

导出视频（不弹窗，直接生成 mp4/avi 文件）：

```powershell
myevs view --in data\usb_evtq.evtq --mode fps --fps 60 --color onoff --scheme 1 --no-hold --out-video data\usb_evtq.mp4 --no-gui
```

说明：导出视频使用的“出帧规则”和“颜色模式”与直接预览完全一致：
- 出帧规则：`--mode fps`（按事件时间切帧）或 `--mode events`（每 N 个事件一帧）
- 颜色模式：`--color onoff`（极性彩色）或 `--color gray`（灰度）
- 配色：`--scheme 0/1`，以及 `--binary`、`--deadzone`、`--no-hold` 都同样生效
- 方向修正：`--flip-x`（左右镜像）、`--flip-y`（上下镜像）、`--rotate-180`（旋转 180°）

与 Prophesee 官方播放观感不一致时（更想看“清晰 + 噪声明显”）：
- 一键参数：`--style prophesee`（只调整可视化风格，不改变时间采样；默认播放速度和体积更稳定）
- 建议参数：`--binary --deadzone 0 --raw-step 20 --scheme 0 --no-hold`
- HDF5 输入通常不需要时间戳解包，建议加 `--no-unwrap-ts` 提升播放流畅度
- 说明：该风格对噪声和边缘更敏感，视频压缩效率会变差，文件通常更大
- 若想在体积/流畅度间平衡，建议 `--fps 30 --video-fps 30`

如果你希望“噪声更明显/细节更密”，可以再手动切到事件分帧（会显著增大帧数与文件体积）：
- `--mode events --events-per-frame 50000`

```powershell
# 推荐：插件已按上文部署时，通常无需再传 --hdf5-plugin-path
myevs view --in data\prophesee_hand\prophesee_hand.hdf5 --style prophesee

# Linux（仅当自动发现失败时）
# myevs view --in data/prophesee_hand/prophesee_hand.hdf5 --hdf5-plugin-path /path/to/openeb/build/lib/hdf5/plugin --style prophesee

# Windows（仅当自动发现失败时）
# myevs view --in data\prophesee_hand\prophesee_hand.hdf5 --hdf5-plugin-path D:\software\openeb-new\build\lib\hdf5\plugin --style prophesee

myevs view --in data\prophesee_hand\prophesee_hand.hdf5 --hdf5-plugin-path D:\software\openeb-new\build\lib\hdf5\plugin --style prophesee --rotate-180 --fps 30 --out-video data\prophesee_hand\prophesee_hand_style.mp4 --no-gui --video-fps 30
```

注释：
- `--mode fps` 下，导出视频时长≈原始采集时长（本示例数据约 5 秒），不是按导出耗时计算。
- 若想慢放，请让 `--video-fps` 小于 `--fps`（例如下例约 3x 慢放）：

```powershell
myevs view --in data\prophesee_hand\prophesee_hand.hdf5 --hdf5-plugin-path D:\software\openeb-new\build\lib\hdf5\plugin --style prophesee --rotate-180 --fps 30 --out-video data\prophesee_hand\prophesee_hand_style_slow.mp4 --no-gui --video-fps 10
```

如果用 `--mode events` 导出视频，建议显式指定输出视频帧率（否则默认用 `--fps`）：

```powershell
myevs view --in data\usb_evtq.evtq --mode events --events-per-frame 200000 --color gray --no-hold --out-video data\usb_evtq_gray.mp4 --no-gui --video-fps 60
myevs view --in data\usb_evtq.evtq --mode fps --fps 60 --color gray --no-hold --out-video data\usb_evtq_gray.mp4 --no-gui
```

颜色一致性说明：
- `--scheme 0`：Qt scheme0（白底，ON红，OFF蓝）
- `--scheme 1`：Qt scheme1（深色底 + 自定义 on/off/off 颜色）
- `--binary` + `--deadzone`：二值彩色 + 死区规则与 Qt 一致

### 4) 统计/评估

```powershell
myevs stats --in denoised.evtq
```

更方便的“去噪前后数量对比”（不需要导出视频）：

```powershell
myevs compare-stats --in-a data\usb_evtq.evtq --in-b data\denoised_method3.evtq
```

快速扫参数（不保存输出文件，只统计 out 事件数/保留率）：

```powershell
# 例：method=3 hotpixel，扫阈值 min-neighbors
myevs sweep --in data\usb_evtq.evtq --method 3 --param min-neighbors --values 2,5,10,20,50,100 --time-us 2000 --refractory-us 2000 --out-csv data\sweep_hotpixel.csv
```

说明：`--out-csv` 的表里会包含：
- input 总事件数/ON/OFF（events_in/on_in/off_in）
- 每个 sweep 点的输出事件数/ON/OFF（events_out/on_out/off_out）
- kept/removed ratio

把 CSV 直接画成科研图表（推荐）：
以后给 sweep 多加一列指标（比如 on_removed_ratio、off_removed_ratio、某个 ROI 内的 removed_ratio），只要列名在 CSV 里出现，plot-csv --y 新列名 就能画。

以后做“多组实验”（比如不同 method、不同 time-us），只要把 CSV 里加一列 group（比如 method 或 run_name），就可以用 --group group 自动画多条曲线同图。

```powershell
# 1) 画“保留率/去除率 vs 阈值”的曲线（最常用）
myevs plot-csv --in data\sweep_hotpixel.csv --out data\sweep_hotpixel_ratio.png --x value --y kept_ratio removed_ratio --kind line --title "HotPixel sweep (time-us=2000, refractory-us=2000)"

# 2) 画“输出事件数 vs 阈值”（有时更直观）
myevs plot-csv --in data\sweep_hotpixel.csv --out data\sweep_hotpixel_events.png --x value --y events_out --kind line --title "HotPixel events_out"
```

如果扫完发现所有点 `removed_ratio` 都接近 0：
- 很可能你的数据里“热像素”本来就不明显；或
- 参数太“宽松/苛刻”（比如阈值太高）；或
- 你看的颜色/出帧方式让差异不容易肉眼观察（建议先用 `compare-stats` 看数值差异）。

## 关于 TP/FP、SNR

这一节解释本工程里 `myevs roc`/ROC CSV 中各项指标的**精确定义**，避免和不同论文/不同工具的口径混淆。

### 1) 评估对象：把“事件去噪”当作二分类

`myevs roc` 的输入是两条流：

- clean：参考“干净”事件流（理想情况下接近 signal-only）
- noisy：含噪事件流（signal + noise）

对 noisy（或 denoised 输出）里的每一个事件，会先贴一个 **ground truth 标签**：

- signal：能在 clean 中找到“匹配”的事件
- noise：在 clean 中找不到匹配

然后 denoiser 给出 **预测**（是否保留这个事件）：

- kept：算法保留（出现在输出流）
- dropped：算法丢弃（不出现在输出流）

最终会得到混淆矩阵（TP/FP/TN/FN）以及 ROC/AUC、precision/accuracy/F1 等指标。

### 2) 标签匹配：`--match-us` / `--match-bin-radius` 的含义

`myevs roc` 默认用 (t,x,y,p) 做**精确匹配**。如果你传了 `--match-us > 0`，则会启用“时间容忍匹配”（只影响评估标注，不影响去噪本身）：

- 对 noisy 事件 (x,y,p,t)，去 clean 里找 (x,y,p,tc)，并要求 |t-tc| 大约在 `match_us` 范围内。

实现上是“时间分箱 + 邻 bin 查询”的近似：

- `t_bin = t // match_ticks`（其中 `match_ticks = us_to_ticks(match_us)`）
- `--match-bin-radius` 控制是否额外检查相邻分箱（0 表示只查本 bin；1 表示查 ±1；以此类推）

注意：在事件密度很高（heavy 噪声）时，`match_us` 过大或 `match_bin_radius` 过大，会显著增加“偶然匹配”的概率，把 noise 误标成 signal，从而造成指标虚高。

### 3) ROC 口径：`--roc-convention paper` vs `noise-drop`

同一套 totals/kept 统计，可以有两种常见口径。本工程通过 `--roc-convention` 显式选择：

#### (A) `paper`（默认，推荐科研对齐）

- 正类（positive）= signal
- 预测为正（predicted positive）= kept

混淆矩阵含义：

- TP = signal_kept（信号被保留）
- FP = noise_kept（噪声被保留）
- TN = noise_dropped（噪声被去除）
- FN = signal_dropped（信号被误删）

对应指标：

- TPR = TP/(TP+FN) = signal_kept/signal_total（信号召回率，越大越好）
- FPR = FP/(FP+TN) = noise_kept/noise_total（噪声保留率，越小越好）
- precision = TP/(TP+FP)（“保留下来的事件里有多少是真信号”，越大越好）
- accuracy = (TP+TN)/(TP+FP+TN+FN)（总体分类准确率）
- F1 = 2*precision*TPR/(precision+TPR)（正类为 signal 的 F1；此处 recall=TPR）

在这个口径下，ROC 通常画的是：x=FPR（noise_kept/noise_total），y=TPR（signal_kept/signal_total），AUC 越大越好。

#### (B) `noise-drop`（旧口径，主要用于兼容/直觉“噪声去除率”）

- 正类（positive）= noise
- 预测为正（predicted positive）= dropped

混淆矩阵含义：

- TP = noise_dropped
- FP = signal_dropped
- TN = signal_kept
- FN = noise_kept

对应指标：

- TPR = noise_dropped/noise_total（噪声去除率/噪声拒绝率，越大越好）
- FPR = signal_dropped/signal_total（信号损失率，越小越好）

注意：在这个口径下，precision/F1 的“正类”也变成 noise（且预测正是 dropped），和 `paper` 完全不同。

### 4) ROC CSV 各列对应关系（`myevs roc --out-csv`）

`myevs roc` 输出的 ROC CSV 中常用列含义：

- `signal_total/noise_total`：noisy 流里被标注为 signal/noise 的总数
- `signal_kept/noise_kept`：denoised 输出里（被保留的）signal/noise 数
- `tp/fp/tn/fn`、`tpr/fpr/precision/accuracy/f1`：由 `--roc-convention` 决定具体语义（见上文）
- `auc`：对所有 sweep 点的 (fpr,tpr) 用梯形法积分得到的 AUC（默认会补齐 (0,0)/(1,1) 端点）

如果你要做“跨噪声等级”的公平对比，建议除了 AUC 以外，同时报告：固定 signal recall（TPR）下的 noise_kept（或 noise_kept/noise_total）。

## 一键实验脚本（推荐科研用）

当你需要频繁做“转换 → 去噪（多个方案）→ 对比（数量/保留率）→（可选）导出视频/画图”时，建议用配置驱动的实验脚本：

1) 复制并修改配置：
- experiments/example_experiment.toml

2) 运行：

```powershell
cd D:\hjx_workspace\scientific_reserach\projects\myEVS
python scripts\run_experiment.py --config experiments\example_experiment.toml
```

配置里你可以很直观地表达：
- 只跑 method3 或 method4
- 跑全套方法（1..8）：启用 `[denoise_all]`

```powershell
python scripts\run_experiment.py --config experiments\\example_experiment.toml
```

- method3 + method4 叠加（pipeline 一次完成）
- method3 先生成文件，再从头对 method3 输出做 method4（sequential / re-open）
- 是否生成视频（[[video]] enabled=true/false）
- 是否显示进度条：启用 `[progress] enabled=true`

脚本会在 output.dir 下生成：
- stats_all.csv（每个结果的事件数/on/off）
- compare.csv（input→各结果的 kept/removed ratio）
- （可选）compare.png（快速柱状图）

## Windows 操作案例（CMD，可直接复制粘贴）

```bat
cd /d D:\hjx_workspace\scientific_reserach\projects\myEVS
call conda activate myevs

REM ====== 0) 路径与尺寸（按实际修改）======
set "RAW_PATH=data\30WEVS_hand\hand2.raw"
set "evtq_PATH=data\30WEVS_hand\input2.evtq"
set "OUT_PATH=data\30WEVS_hand\refractory_us"
set "W=640"
set "H=512"

if not exist "%OUT_PATH%" mkdir "%OUT_PATH%"

REM ====== 1) raw -> evtq（只做一次）======
myevs convert-usb-raw --in "%RAW_PATH%" --out "%evtq_PATH%" --width %W% --height %H% --progress

REM 可选：先看一下输入基本信息
myevs stats --in "%evtq_PATH%" --progress
myevs stats --in "%RAW_PATH%" --progress
REM ====== 2) STC 宽范围扫参（分别扫 time_us / radius_px / min_neighbors）======
REM 2.1 扫 time_us（固定 r=1, k=2）
myevs sweep --in "%evtq_PATH%" --method 1 --param time-us --values 200,500,1000,2000,5000,10000 --radius-px 1 --min-neighbors 2 --out-csv "%OUT_PATH%\sweep_time_us_wide.csv" --progress
myevs sweep --in "%evtq_PATH%" --method 2 --param refractory-us --values 200,500,1000,2000,5000,10000  --out-csv "%OUT_PATH%\sweep_refractory_us_wide.csv" --progress

myevs plot-csv --in "%OUT_PATH%\sweep_time_us_wide.csv" --out "%OUT_PATH%\sweep_time_us_wide.png" --x value --y kept_ratio removed_ratio --kind line --title "STC sweep time_us (r=1,k=2)"

myevs plot-csv --in "%OUT_PATH%\sweep_refractory_us_wide.csv" --out "%OUT_PATH%\sweep_refractory_us_wide.png" --x value --y kept_ratio removed_ratio --kind line --title "STC sweep refractory_us"

REM 2.2 扫 radius_px（固定 time=2000us, k=2）
myevs sweep --in "%evtq_PATH%" --method 1 --param radius-px --values 0,1,2,3,4 --time-us 2000 --min-neighbors 2 --out-csv "%OUT_PATH%\sweep_radius_px_wide.csv" --progress
myevs plot-csv --in "%OUT_PATH%\sweep_radius_px_wide.csv" --out "%OUT_PATH%\sweep_radius_px_wide.png" --x value --y kept_ratio removed_ratio --kind line --title "STC sweep radius_px (time_us=2000,k=2)"

REM 2.3 扫 min_neighbors（固定 time=2000us, r=1）
myevs sweep --in "%evtq_PATH%" --method 1 --param min-neighbors --values 1,2,3,5,8,12,20 --time-us 2000 --radius-px 1 --out-csv "%OUT_PATH%\sweep_min_neighbors_wide.csv" --progress
myevs plot-csv --in "%OUT_PATH%\sweep_min_neighbors_wide.csv" --out "%OUT_PATH%\sweep_min_neighbors_wide.png" --x value --y kept_ratio removed_ratio --kind line --title "STC sweep min_neighbors (time_us=2000,r=1)"

REM ====== 3) 缩小范围二次扫参（按你从图上看到的敏感区间改 values）======
myevs sweep --in "%evtq_PATH%" --method 1 --param time-us --values 800,1000,1200,1500,1800,2000,2500,3000 --radius-px 1 --min-neighbors 2 --out-csv "%OUT_PATH%\sweep_time_us_narrow.csv" --progress
myevs plot-csv --in "%OUT_PATH%\sweep_time_us_narrow.csv" --out "%OUT_PATH%\sweep_time_us_narrow.png" --x value --y kept_ratio removed_ratio --kind line --title "STC sweep time_us (narrow)"

REM ====== 4) 选定参数，真正去噪产出 evtq（把下面参数改成你选中的）======
myevs denoise --in "%evtq_PATH%" --out "%OUT_PATH%\stc_best.evtq" --method 1 --time-us 2000 --radius-px 2 --min-neighbors 3 --progress
myevs compare-stats --in-a "%evtq_PATH%" --in-b "%OUT_PATH%\stc_best.evtq" --progress

REM ====== 5) 导出去噪后视频（不弹窗）======
myevs view --in "%OUT_PATH%\stc_best.evtq" --mode fps --fps 60 --color onoff --scheme 1 --no-hold --out-video "%OUT_PATH%\stc_best.mp4" --no-gui --progress
```

## Windows 操作案例（PowerShell，可直接复制粘贴，可选）

```powershell
cd D:\hjx_workspace\scientific_reserach\projects\myEVS
conda activate myevs

# ====== 0) 路径与尺寸（按实际修改）======
$RAW_PATH = "data\30WEVS_hand\hand.raw"
$OUT_PATH = "data\30WEVS_hand\stc"
$W = 640
$H = 512

New-Item -ItemType Directory -Force -Path $OUT_PATH | Out-Null

# ====== 1) raw -> evtq（只做一次）======
myevs convert-usb-raw --in $RAW_PATH --out "$OUT_PATH\input.evtq" --width $W --height $H --progress

# 可选：先看一下输入基本信息
myevs stats --in "$OUT_PATH\input.evtq" --progress

# ====== 2) STC 宽范围扫参（分别扫 time_us / radius_px / min_neighbors）======
# 2.1 扫 time_us（固定 r=1, k=2）
myevs sweep --in "$OUT_PATH\input.evtq" --method 1 --param time-us --values 200,500,1000,2000,5000,10000 --radius-px 1 --min-neighbors 2 --out-csv "$OUT_PATH\sweep_time_us_wide.csv" --progress
myevs plot-csv --in "$OUT_PATH\sweep_time_us_wide.csv" --out "$OUT_PATH\sweep_time_us_wide.png" --x value --y kept_ratio removed_ratio --kind line --title "STC sweep time_us (r=1,k=2)"

# 2.2 扫 radius_px（固定 time=2000us, k=2）
myevs sweep --in "$OUT_PATH\input.evtq" --method 1 --param radius-px --values 0,1,2,3,4 --time-us 2000 --min-neighbors 2 --out-csv "$OUT_PATH\sweep_radius_px_wide.csv" --progress
myevs plot-csv --in "$OUT_PATH\sweep_radius_px_wide.csv" --out "$OUT_PATH\sweep_radius_px_wide.png" --x value --y kept_ratio removed_ratio --kind line --title "STC sweep radius_px (time_us=2000,k=2)"

# 2.3 扫 min_neighbors（固定 time=2000us, r=1）
myevs sweep --in "$OUT_PATH\input.evtq" --method 1 --param min-neighbors --values 1,2,3,5,8,12,20 --time-us 2000 --radius-px 1 --out-csv "$OUT_PATH\sweep_min_neighbors_wide.csv" --progress
myevs plot-csv --in "$OUT_PATH\sweep_min_neighbors_wide.csv" --out "$OUT_PATH\sweep_min_neighbors_wide.png" --x value --y kept_ratio removed_ratio --kind line --title "STC sweep min_neighbors (time_us=2000,r=1)"

# ====== 3) 缩小范围二次扫参（按你从图上看到的敏感区间改 values）======
myevs sweep --in "$OUT_PATH\input.evtq" --method 1 --param time-us --values 800,1000,1200,1500,1800,2000,2500,3000 --radius-px 1 --min-neighbors 2 --out-csv "$OUT_PATH\sweep_time_us_narrow.csv" --progress
myevs plot-csv --in "$OUT_PATH\sweep_time_us_narrow.csv" --out "$OUT_PATH\sweep_time_us_narrow.png" --x value --y kept_ratio removed_ratio --kind line --title "STC sweep time_us (narrow)"

# ====== 4) 选定参数，真正去噪产出 evtq（把下面参数改成你选中的）======
myevs denoise --in "$OUT_PATH\input.evtq" --out "$OUT_PATH\stc_best.evtq" --method 1 --time-us 2000 --radius-px 2 --min-neighbors 3 --progress
myevs compare-stats --in-a "$OUT_PATH\input.evtq" --in-b "$OUT_PATH\stc_best.evtq" --progress

# ====== 5) 导出去噪后视频（不弹窗）======
myevs view --in "$OUT_PATH\stc_best.evtq" --mode fps --fps 60 --color onoff --scheme 1 --no-hold --out-video "$OUT_PATH\stc_best.mp4" --no-gui --progress
```

## Linux 操作案例（bash，可直接复制粘贴）

```bash
cd /path/to/myEVS
conda activate myevs

# ====== 0) 路径与尺寸（按实际修改）======
RAW_PATH="data/30WEVS_hand/hand.raw"
OUT_PATH="data/30WEVS_hand/stc"
W=640
H=512

mkdir -p "$OUT_PATH"

# ====== 1) raw -> evtq（只做一次）======
myevs convert-usb-raw --in "$RAW_PATH" --out "$OUT_PATH/input.evtq" --width $W --height $H --progress

# 可选：先看一下输入基本信息
myevs stats --in "$OUT_PATH/input.evtq" --progress

# ====== 2) STC 宽范围扫参（分别扫 time_us / radius_px / min_neighbors）======
# 2.1 扫 time_us（固定 r=1, k=2）
myevs sweep --in "$OUT_PATH/input.evtq" --method 1 --param time-us --values 200,500,1000,2000,5000,10000 --radius-px 1 --min-neighbors 2 --out-csv "$OUT_PATH/sweep_time_us_wide.csv" --progress
myevs plot-csv --in "$OUT_PATH/sweep_time_us_wide.csv" --out "$OUT_PATH/sweep_time_us_wide.png" --x value --y kept_ratio removed_ratio --kind line --title "STC sweep time_us (r=1,k=2)"

# 2.2 扫 radius_px（固定 time=2000us, k=2）
myevs sweep --in "$OUT_PATH/input.evtq" --method 1 --param radius-px --values 0,1,2,3,4 --time-us 2000 --min-neighbors 2 --out-csv "$OUT_PATH/sweep_radius_px_wide.csv" --progress
myevs plot-csv --in "$OUT_PATH/sweep_radius_px_wide.csv" --out "$OUT_PATH/sweep_radius_px_wide.png" --x value --y kept_ratio removed_ratio --kind line --title "STC sweep radius_px (time_us=2000,k=2)"

# 2.3 扫 min_neighbors（固定 time=2000us, r=1）
myevs sweep --in "$OUT_PATH/input.evtq" --method 1 --param min-neighbors --values 1,2,3,5,8,12,20 --time-us 2000 --radius-px 1 --out-csv "$OUT_PATH/sweep_min_neighbors_wide.csv" --progress
myevs plot-csv --in "$OUT_PATH/sweep_min_neighbors_wide.csv" --out "$OUT_PATH/sweep_min_neighbors_wide.png" --x value --y kept_ratio removed_ratio --kind line --title "STC sweep min_neighbors (time_us=2000,r=1)"

# ====== 3) 缩小范围二次扫参（按你从图上看到的敏感区间改 values）======
myevs sweep --in "$OUT_PATH/input.evtq" --method 1 --param time-us --values 800,1000,1200,1500,1800,2000,2500,3000 --radius-px 1 --min-neighbors 2 --out-csv "$OUT_PATH/sweep_time_us_narrow.csv" --progress
myevs plot-csv --in "$OUT_PATH/sweep_time_us_narrow.csv" --out "$OUT_PATH/sweep_time_us_narrow.png" --x value --y kept_ratio removed_ratio --kind line --title "STC sweep time_us (narrow)"

# ====== 4) 选定参数，真正去噪产出 evtq（把下面参数改成你选中的）======
myevs denoise --in "$OUT_PATH/input.evtq" --out "$OUT_PATH/stc_best.evtq" --method 1 --time-us 2000 --radius-px 2 --min-neighbors 3 --progress
myevs compare-stats --in-a "$OUT_PATH/input.evtq" --in-b "$OUT_PATH/stc_best.evtq" --progress

# ====== 5) 导出去噪后视频（不弹窗）======
myevs view --in "$OUT_PATH/stc_best.evtq" --mode fps --fps 60 --color onoff --scheme 1 --no-hold --out-video "$OUT_PATH/stc_best.mp4" --no-gui --progress
```

# prophesee相机数据采集指令及类型转换
```python
metavision_viewer.exe -o data.raw #输出为raw文件 空格键开始录制，再次空格停止录制
metavision_file_to_csv.exe -i data.raw -o data.csv # 转换为csv格式文件
```

# match-us的含义
--match-us 的作用不是“把一段时间内触发的所有事件都当成真实信号”，而是在做指标评估时，用来允许 clean/noisy 两条流之间存在时间抖动/偏移的“匹配容忍度”。

具体是这样用的（只影响 myevs roc 的标注/统计，不影响去噪本身）：
对 noisy（或 denoised）里的每一个事件 (x,y,p,t)，去 clean 里找有没有同一像素同一极性的事件 (x,y,p,tc)，并且时间“足够近”
如果存在这样的 clean 事件，就把这个 noisy 事件标成 signal；否则标成 noise。
“足够近”由 --match-us 决定：你可以把它理解成“允许时间对齐误差的阈值”。
举个例子：
clean 有事件：(x=10,y=20,p=ON,t=1,000,000 μs)
noisy 里对应事件因为 v2e 抖动变成：t=1,000,050 μs
--match-us 0（精确匹配）会认为它不在 clean 里 → 标成 noise（会把信号错当噪声）
--match-us 1000 会认为 50μs 的偏移可接受 → 标成 signal
再强调一次区别，避免混淆：
--time-us（比如 BAF 的时间窗）= 算法参数，决定“去噪怎么做”
--match-us = 评估参数，决定“算 TP/FP/ROC 时怎么把 noisy 事件贴 signal/noise 标签”
另外我当前实现是“时间分箱 + 相邻 bin（±1）”的快速近似，所以 match-us=1000 表示“约 1ms 量级的容忍”（边界附近可能会略放宽一点）。