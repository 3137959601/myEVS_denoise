可以，这条路是对的。

如果 `match-us=0` 之后现象还是不对，那就说明问题更可能出在：

* 你现在的 **signal/noise 标注方式**
* 以及 ROC 统计用的 **ground truth 构造**

而不是时间抖动容忍。

你要的“**不依赖 match-us 的 injected-noise 评测流程**”，核心思想其实很简单：

> **不要再事后拿 clean 去匹配 noisy。**
> **而是在生成 noisy 的那一刻，就把每个事件的来源身份记下来。**

也就是把事件天然分成两类：

* `signal`：来自 clean 事件流
* `noise`：后注入的噪声事件流

这样评估时根本不需要 `match-us`。

---

# 一、整体流程

我建议你把评测流程改成 5 步。

## Step 1：生成 clean 事件流

从 `video_orig.avi` 生成 clean 事件流：

* `clean_events = [(x, y, p, t), ...]`

这批事件全部标记为：

* `label = 1`，表示 signal

---

## Step 2：单独生成 injected noise 事件流

不要把 noise 直接混进去就丢了身份。
而是单独生成：

* `noise_events = [(x, y, p, t), ...]`

这批事件全部标记为：

* `label = 0`，表示 noise

注意这里的噪声生成要和你的实验设置一致，比如：

* shot-only
* leak-only
* mixed noise

---

## Step 3：合并成 noisy 输入，但保留身份标签

把两批事件合并并按时间排序，形成：

* `noisy_events = clean_events ∪ noise_events`

但每个事件都带着来源标签：

```python
(event_id, x, y, p, t, label)
```

其中：

* `label = 1` 表示 signal
* `label = 0` 表示 injected noise

这一步非常关键。
你真正喂给滤波器的是 `(x, y, p, t)`，
但你自己另外保存一份“带标签版本”用于评估。

---

## Step 4：跑滤波器，同时保留事件 ID

滤波器输入仍然是：

```python
(x, y, p, t)
```

但你在程序里不要把 `event_id` 丢掉。

也就是说，对每个输入事件：

```python
(event_id, x, y, p, t, label)
```

滤波器输出时记录：

```python
(event_id, kept=True/False, label)
```

这样你就知道：

* 哪些 signal 被保留
* 哪些 signal 被删除
* 哪些 noise 被保留
* 哪些 noise 被删除

整个过程完全不需要 `match-us`。

---

## Step 5：直接统计 ROC 所需量

按论文定义：

* 正类 = signal
* 预测正类 = kept

那么：

* `TP = signal_kept`
* `FN = signal_dropped`
* `FP = noise_kept`
* `TN = noise_dropped`

于是：

[
TPR = \frac{signal_kept}{signal_total}
]

[
FPR = \frac{noise_kept}{noise_total}
]

这就是最干净、最不含糊的 ROC 评测。

---

# 二、为什么这个流程比 match-us 更合理

因为它彻底避免了“匹配误差”。

你之前那套 `match-us` 的问题在于：

* noisy 里的事件是不是 signal，要靠事后匹配推断
* 一旦时间、极性、邻域关系有点偏差，就会错标
* 而且容易出现一个 clean 事件匹配多个 noisy 事件的问题

现在这套 injected-noise 流程没有这些问题，因为：

> **signal/noise 不是推断出来的，而是生成时就知道的。**

所以它更适合和你提到的那种论文 benchmark 思路对齐。

---

# 三、你最该怎么实现

我建议你不要从 `.aedat` 文件后处理开始搞，那样麻烦。
最稳的是在你自己的数据生成脚本阶段就把标签带上。

---

## 方案 A：最推荐

你自己写一个 Python 脚本，生成三份东西：

### 1. clean 事件表

```python
clean_events = [
    (event_id, x, y, p, t, 1),   # 1 = signal
    ...
]
```

### 2. noise 事件表

```python
noise_events = [
    (event_id, x, y, p, t, 0),   # 0 = noise
    ...
]
```

### 3. noisy 输入表

```python
noisy_events = sorted(clean_events + noise_events, key=lambda e: e.t)
```

然后滤波器只吃：

```python
(event_id, x, y, p, t)
```

评估时再回头查 label。

---

## 方案 B：如果你现在必须用 `.aedat`

那就也行，但你要在生成 `.aedat` 之前，同时另外保存一个 sidecar 文件，比如：

* `driving_noisy_labels.csv`
* `driving_noisy_labels.npz`

里面记录每个事件的：

* `event_id`
* `label`

然后读入滤波器时保持顺序一致。

这个要求你的事件流处理不能随便重排事件顺序。
如果你的滤波器是在线处理并保持输入顺序，那就没问题。

---

# 四、最小化实现格式

我建议你每个事件保存成这样：

```python
event = {
    "id": int,
    "x": int,
    "y": int,
    "p": int,
    "t": int,
    "label": int,   # 1 signal, 0 noise
}
```

如果你嫌 dict 慢，就用 numpy 结构化数组或 dataclass。

例如：

```python
from dataclasses import dataclass

@dataclass
class LabeledEvent:
    id: int
    x: int
    y: int
    p: int
    t: int
    label: int   # 1=signal, 0=noise
```

---

# 五、评测代码怎么写

逻辑非常简单。

## 输入

滤波后的结果记录：

```python
results = [
    (event_id, kept),
    ...
]
```

以及原始标签表：

```python
labels[event_id] = 0 or 1
```

## 统计

```python
signal_total = 0
noise_total = 0
signal_kept = 0
noise_kept = 0

for event_id, kept in results:
    label = labels[event_id]
    if label == 1:
        signal_total += 1
        if kept:
            signal_kept += 1
    else:
        noise_total += 1
        if kept:
            noise_kept += 1

signal_dropped = signal_total - signal_kept
noise_dropped = noise_total - noise_kept

tpr = signal_kept / signal_total if signal_total else 0.0
fpr = noise_kept / noise_total if noise_total else 0.0
```

这就是你要的论文定义。

---

# 六、你怎么扫 ROC

这里要分清楚：

## 对 EBF 这类有连续 score 的方法

直接扫 score threshold，很自然。

## 对 BAF / STCF / FastDecay 这类传统规则滤波

没有天然连续 score 时，就用：

* `radius`
* `tau`
* `min_neighbors`
* `threshold`

这些参数组合产生一组 operating points。

然后：

* 用这些点画离散 ROC
* 必要时补 `(0,0)` 和 `(1,1)` 作为锚点

这就可以了。

---

# 七、为了和论文更对齐，你最好这样设计噪声实验

不要一开始就混很多变量。

## 实验 1：只扫 shot noise

固定：

* leak = 0
* sigma_thres 固定
* cutoff 固定

只改：

* `shot_noise_rate_hz = 1, 5, 10`

这样最容易和论文里“shot noise 越大”对齐。

## 实验 2：只扫 leak noise

固定：

* shot = 0

只改：

* `leak_rate_hz = 0.05, 0.1, 0.2`

## 实验 3：mixed noise

再做：

* leak + shot

这样你能分清楚到底是谁在影响 BAF 曲线。

---

# 八、你现在这套 injected-noise 评测流程的优点

它会直接解决你目前最烦的 4 个问题：

### 1. 不再需要 `match-us`

没有时序匹配歧义。

### 2. 不会一个 clean 事件喂多个 noisy 事件

因为标签天然绑定在事件本体上。

### 3. ROC 定义完全清楚

signal 就是 signal，noise 就是 noise。

### 4. 可以真正和论文那种“已知 injected noise”思路对齐

这点最关键。

---

# 九、你现在最值得做的一个最小实验

别一下重构所有代码。
先做一个最小化验证：

## 先只做一套数据

* `driving_clean_slomo`
* `shot_noise_rate_hz = 5`
* `leak_rate_hz = 0`

然后：

1. clean 事件生成后保存 label=1
2. noise 单独生成后保存 label=0
3. 合并
4. 跑 BAF
5. 算 `TPR/FPR`

只要这一套跑通，你后面整个实验链就顺了。

---

# 十、我对这套流程的判断

> **合理，而且比你现在的 match-us 方案更靠谱。**

尤其你现在已经开始怀疑评估协议影响结果了，那最正确的应对不是继续调 `match-us`，而是：

> **彻底取消匹配式标注，改成生成时显式带标签。**

这才是从根上解决问题。

你下一步最适合做的是：我直接帮你把这个 injected-noise 评测流程写成一个 **可落地的数据结构和脚本框架**，包括：

* 如何生成带标签的 noisy 流
* 如何让滤波器输出保留/删除结果
* 如何自动导出 CSV。


可以，这条路是对的。

如果 `match-us=0` 之后现象还是不对，那就说明问题更可能出在：

* 你现在的 **signal/noise 标注方式**
* 以及 ROC 统计用的 **ground truth 构造**

而不是时间抖动容忍。

你要的“**不依赖 match-us 的 injected-noise 评测流程**”，核心思想其实很简单：

> **不要再事后拿 clean 去匹配 noisy。**
> **而是在生成 noisy 的那一刻，就把每个事件的来源身份记下来。**

也就是把事件天然分成两类：

* `signal`：来自 clean 事件流
* `noise`：后注入的噪声事件流

这样评估时根本不需要 `match-us`。


---
---

问题：但是问题是Noise事件表怎么生成呢，可以直接通过v2e生成吗
---

# 一、整体流程

我建议你把评测流程改成 5 步。

## Step 1：生成 clean 事件流

从 `video_orig.avi` 生成 clean 事件流：

* `clean_events = [(x, y, p, t), ...]`

这批事件全部标记为：

* `label = 1`，表示 signal

---

## Step 2：单独生成 injected noise 事件流

不要把 noise 直接混进去就丢了身份。
而是单独生成：

* `noise_events = [(x, y, p, t), ...]`

这批事件全部标记为：

* `label = 0`，表示 noise

注意这里的噪声生成要和你的实验设置一致，比如：

* shot-only
* leak-only
* mixed noise

---

## Step 3：合并成 noisy 输入，但保留身份标签

把两批事件合并并按时间排序，形成：

* `noisy_events = clean_events ∪ noise_events`

但每个事件都带着来源标签：

```python
(event_id, x, y, p, t, label)
```

其中：

* `label = 1` 表示 signal
* `label = 0` 表示 injected noise

这一步非常关键。
你真正喂给滤波器的是 `(x, y, p, t)`，
但你自己另外保存一份“带标签版本”用于评估。

---

## Step 4：跑滤波器，同时保留事件 ID

滤波器输入仍然是：

```python
(x, y, p, t)
```

但你在程序里不要把 `event_id` 丢掉。

也就是说，对每个输入事件：

```python
(event_id, x, y, p, t, label)
```

滤波器输出时记录：

```python
(event_id, kept=True/False, label)
```

这样你就知道：

* 哪些 signal 被保留
* 哪些 signal 被删除
* 哪些 noise 被保留
* 哪些 noise 被删除

整个过程完全不需要 `match-us`。

---

## Step 5：直接统计 ROC 所需量

按论文定义：

* 正类 = signal
* 预测正类 = kept

那么：

* `TP = signal_kept`
* `FN = signal_dropped`
* `FP = noise_kept`
* `TN = noise_dropped`

于是：

[
TPR = \frac{signal_kept}{signal_total}
]

[
FPR = \frac{noise_kept}{noise_total}
]

这就是最干净、最不含糊的 ROC 评测。

---

# 二、为什么这个流程比 match-us 更合理

因为它彻底避免了“匹配误差”。

你之前那套 `match-us` 的问题在于：

* noisy 里的事件是不是 signal，要靠事后匹配推断
* 一旦时间、极性、邻域关系有点偏差，就会错标
* 而且容易出现一个 clean 事件匹配多个 noisy 事件的问题

现在这套 injected-noise 流程没有这些问题，因为：

> **signal/noise 不是推断出来的，而是生成时就知道的。**

所以它更适合和你提到的那种论文 benchmark 思路对齐。

---

# 三、你最该怎么实现

我建议你不要从 `.aedat` 文件后处理开始搞，那样麻烦。
最稳的是在你自己的数据生成脚本阶段就把标签带上。

---

## 方案 A：最推荐

你自己写一个 Python 脚本，生成三份东西：

### 1. clean 事件表

```python
clean_events = [
    (event_id, x, y, p, t, 1),   # 1 = signal
    ...
]
```

### 2. noise 事件表

```python
noise_events = [
    (event_id, x, y, p, t, 0),   # 0 = noise
    ...
]
```

### 3. noisy 输入表

```python
noisy_events = sorted(clean_events + noise_events, key=lambda e: e.t)
```

然后滤波器只吃：

```python
(event_id, x, y, p, t)
```

评估时再回头查 label。

---

## 方案 B：如果你现在必须用 `.aedat`

那就也行，但你要在生成 `.aedat` 之前，同时另外保存一个 sidecar 文件，比如：

* `driving_noisy_labels.csv`
* `driving_noisy_labels.npz`

里面记录每个事件的：

* `event_id`
* `label`

然后读入滤波器时保持顺序一致。

这个要求你的事件流处理不能随便重排事件顺序。
如果你的滤波器是在线处理并保持输入顺序，那就没问题。

---

# 四、最小化实现格式

我建议你每个事件保存成这样：

```python
event = {
    "id": int,
    "x": int,
    "y": int,
    "p": int,
    "t": int,
    "label": int,   # 1 signal, 0 noise
}
```

如果你嫌 dict 慢，就用 numpy 结构化数组或 dataclass。

例如：

```python
from dataclasses import dataclass

@dataclass
class LabeledEvent:
    id: int
    x: int
    y: int
    p: int
    t: int
    label: int   # 1=signal, 0=noise
```

---

# 五、评测代码怎么写

逻辑非常简单。

## 输入

滤波后的结果记录：

```python
results = [
    (event_id, kept),
    ...
]
```

以及原始标签表：

```python
labels[event_id] = 0 or 1
```

## 统计

```python
signal_total = 0
noise_total = 0
signal_kept = 0
noise_kept = 0

for event_id, kept in results:
    label = labels[event_id]
    if label == 1:
        signal_total += 1
        if kept:
            signal_kept += 1
    else:
        noise_total += 1
        if kept:
            noise_kept += 1

signal_dropped = signal_total - signal_kept
noise_dropped = noise_total - noise_kept

tpr = signal_kept / signal_total if signal_total else 0.0
fpr = noise_kept / noise_total if noise_total else 0.0
```

这就是你要的论文定义。

---

# 六、你怎么扫 ROC

这里要分清楚：

## 对 EBF 这类有连续 score 的方法

直接扫 score threshold，很自然。

## 对 BAF / STCF / FastDecay 这类传统规则滤波

没有天然连续 score 时，就用：

* `radius`
* `tau`
* `min_neighbors`
* `threshold`

这些参数组合产生一组 operating points。

然后：

* 用这些点画离散 ROC
* 必要时补 `(0,0)` 和 `(1,1)` 作为锚点

这就可以了。

---

# 七、为了和论文更对齐，你最好这样设计噪声实验

不要一开始就混很多变量。

## 实验 1：只扫 shot noise

固定：

* leak = 0
* sigma_thres 固定
* cutoff 固定

只改：

* `shot_noise_rate_hz = 1, 5, 10`

这样最容易和论文里“shot noise 越大”对齐。

## 实验 2：只扫 leak noise

固定：

* shot = 0

只改：

* `leak_rate_hz = 0.05, 0.1, 0.2`

## 实验 3：mixed noise

再做：

* leak + shot

这样你能分清楚到底是谁在影响 BAF 曲线。

---

# 八、你现在这套 injected-noise 评测流程的优点

它会直接解决你目前最烦的 4 个问题：

### 1. 不再需要 `match-us`

没有时序匹配歧义。

### 2. 不会一个 clean 事件喂多个 noisy 事件

因为标签天然绑定在事件本体上。

### 3. ROC 定义完全清楚

signal 就是 signal，noise 就是 noise。

### 4. 可以真正和论文那种“已知 injected noise”思路对齐

这点最关键。

---

# 九、你现在最值得做的一个最小实验

别一下重构所有代码。
先做一个最小化验证：

## 先只做一套数据

* `driving_clean_slomo`
* `shot_noise_rate_hz = 5`
* `leak_rate_hz = 0`

然后：

1. clean 事件生成后保存 label=1
2. noise 单独生成后保存 label=0
3. 合并
4. 跑 BAF
5. 算 `TPR/FPR`

只要这一套跑通，你后面整个实验链就顺了。

---

# 十、我对这套流程的判断

> **合理，而且比你现在的 match-us 方案更靠谱。**

尤其你现在已经开始怀疑评估协议影响结果了，那最正确的应对不是继续调 `match-us`，而是：

> **彻底取消匹配式标注，改成生成时显式带标签。**

这才是从根上解决问题。

你下一步最适合做的是：我直接帮你把这个 injected-noise 评测流程写成一个 **可落地的数据结构和脚本框架**，包括：

* 如何生成带标签的 noisy 流
* 如何让滤波器输出保留/删除结果
* 如何自动导出 CSV。
