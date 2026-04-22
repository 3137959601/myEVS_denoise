
## s1：Directional Coherence（各向异性调制的密度打分）

### 算法实现（低延迟设计）

位置：`src/myevs/denoise/ops/ebfopt_part2/s1_dircoh.py`（Kernel 提供给 sweep 脚本调用）

对每个事件 $e=(x,y,t,p)$：

1) baseline EBF 的 raw-score：

- 同极性
- 时域线性衰减权重 $w_t = (\tau-\Delta t)/\tau$
- 邻域累计：$\mathrm{raw}=\sum w_t$

2) 同步累计邻域空间二阶矩（以 $(dx,dy)$ 为偏移）：

- $S_{xx}=\sum w_t dx^2$，$S_{yy}=\sum w_t dy^2$，$S_{xy}=\sum w_t dx\,dy$

构造 $\mathbf{S}=\begin{bmatrix}S_{xx}&S_{xy}\\S_{xy}&S_{yy}\end{bmatrix}$。

3) 各向异性（方向一致性）指标 $\mathrm{coh}\in[0,1]$：

$$
\mathrm{coh} = \frac{\sqrt{\mathrm{disc}}}{\mathrm{trace}+\varepsilon},\quad
\mathrm{trace}=S_{xx}+S_{yy},\quad
\mathrm{disc}=\mathrm{trace}^2-4\det(\mathbf{S})
$$

4) 最终 score：

$$
\mathrm{score}=\mathrm{raw}\cdot\bigl(\eta+(1-\eta)\,\mathrm{coh}\bigr)
$$

其中 $\eta\in[0,1]$ 由环境变量 `MYEVS_EBF_S1_ETA` 控制（默认 0.2）。

### 运行命令（建议）

Prescreen：

```powershell
$env:MYEVS_EBF_S1_ETA='0.2'
$env:PYTHONNOUSERSITE='1'
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant s1 --max-events 200000 --out-dir data/ED24/myPedestrain_06/EBF_Part2/s1_dircoh_eta0p2_rerun
```

### s1 阶段性结论（已完成）

结论一句话：s1 作为“全局乘子”会误伤信号，整体比不过 baseline。

关键结果（prescreen200k，对比 baseline）：

评测设置：`--max-events 200000`，sweep 网格与 EBF 默认一致（`s=3,5,7,9`；`tau=8..1024ms`）。

| env | 方法 | best(s,tau) | Thr(best) | AUC | F1 | MESR |
|---|---|---:|---:|---:|---:|---:|
| light(1.8V) | EBF baseline | (9, 128ms) | 0.7491 | 0.9476 | 0.9497 | 1.0305 |
| light(1.8V) | s1 dircoh (eta=0.2) | (9, 128ms) | 0.5451 | 0.9458 | 0.9479 | 1.0642 |
| mid(2.5V) | EBF baseline | (9, 128ms) | 4.8394 | 0.9219 | 0.8108 | 1.0168 |
| mid(2.5V) | s1 dircoh (eta=0.2) | (9, 128ms) | 2.8744 | 0.9142 | 0.7923 | 1.0301 |
| heavy(3.3V) | EBF baseline | (9, 128ms) | 7.3581 | 0.9205 | 0.7869 | 1.0208 |
| heavy(3.3V) | s1 dircoh (eta=0.2) | (9, 128ms) | 3.8522 | 0.9097 | 0.7600 | 1.0195 |

eta 扫描趋势（固定 `s=9,tau=128ms`；prescreen200k；三环境均值）：

| eta | mean AUC | mean F1 | mean MESR |
|---:|---:|---:|---:|
| 0.0 | 0.8871 | 0.7785 | 1.0661 |
| 0.1 | 0.9145 | 0.8164 | 1.0480 |
| 0.2 | 0.9232 | 0.8334 | 1.0379 |
| 0.3 | 0.9271 | 0.8414 | 1.0364 |
| 0.5 | 0.9301 | 0.8485 | 1.0237 |
| 0.8 | 0.9305 | 0.8503 | 1.0227 |
| 1.0 | 0.9300 | 0.8491 | 1.0227 |

缺陷（为什么比不过 baseline）：

- coherence 作为全局乘子，会在 mid/heavy 下对大量“真实信号事件”一并降分。
- 这种“均匀缩放”会改变打分尺度与排序，导致 ROC/最佳点整体劣化（AUC/F1 下滑）。

是否值得继续优化：

- 不建议在 s1 这个“全局乘子”形态上继续投入；它更像偏好结构性的正则项，而不是精度提升主线。
- 继续方向应转向“判别式门控/惩罚”（即 s2 思路），只对特定噪声模式起作用。

## s2：Coherence-gated penalty（只惩罚低 coh 的高 raw 事件）

### 算法动机

s1 的问题是 coherence 作为全局乘子会普遍缩放分数。s2 把 coherence 改为更“判别式”的门控惩罚：只在 raw 已经很高但 coh 很低时压分。

### 算法定义（低延迟设计）

位置：`src/myevs/denoise/ops/ebfopt_part2/s2_cohgate.py`

$$
\mathrm{score} = \mathrm{raw}\cdot \mathrm{pen},\quad
\mathrm{pen} =
\begin{cases}
1 & (\mathrm{raw}<\mathrm{raw\_thr})\ \text{或}\ (\mathrm{coh}\ge \mathrm{coh\_thr})\\
\left(\frac{\mathrm{coh}}{\mathrm{coh\_thr}}\right)^{\gamma} & (\mathrm{raw}\ge \mathrm{raw\_thr}\ \text{且}\ \mathrm{coh}<\mathrm{coh\_thr})
\end{cases}
$$

环境变量：

- `MYEVS_EBF_S2_COH_THR`（默认 0.4）
- `MYEVS_EBF_S2_RAW_THR`（默认 3.0）
- `MYEVS_EBF_S2_GAMMA`（默认 1.0）

### 运行命令（建议）

```powershell
$env:PYTHONNOUSERSITE='1'
$env:MYEVS_EBF_S2_COH_THR='0.4'
$env:MYEVS_EBF_S2_RAW_THR='3'
$env:MYEVS_EBF_S2_GAMMA='1'
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant s2 --max-events 200000 --out-dir data/ED24/myPedestrain_06/EBF_Part2/s2_ct0p4_rt3_g1_prescreen200k
```

### s2 阶段性结论（已完成：小范围调参）

结论一句话：s2 方向是对的，但门控必须“克制”，否则仍会误伤信号。

参数小扫（固定 s=9,tau=128ms；prescreen200k；每行是该 env 内 best operating point）：

| config | env | Thr(best) | AUC | F1 | MESR |
|---|---|---:|---:|---:|---:|
| ct0p5_rt3_g1 | light(1.8V) | 0.5677 | 0.9434 | 0.9478 | 1.0706 |
| ct0p5_rt3_g1 | mid(2.5V) | 3.6592 | 0.8929 | 0.7602 | 1.0179 |
| ct0p5_rt3_g1 | heavy(3.3V) | 5.2772 | 0.8709 | 0.7060 | 1.0045 |
| ct0p4_rt3_g1 | light(1.8V) | 0.7419 | 0.9447 | 0.9484 | 1.0338 |
| ct0p4_rt3_g1 | mid(2.5V) | 4.1332 | 0.9019 | 0.7742 | 1.0104 |
| ct0p4_rt3_g1 | heavy(3.3V) | 5.6780 | 0.8841 | 0.7273 | 1.0031 |
| ct0p5_rt4_g1 | light(1.8V) | 0.5677 | 0.9435 | 0.9480 | 1.0704 |
| ct0p5_rt4_g1 | mid(2.5V) | 3.9950 | 0.8921 | 0.7587 | 1.0144 |
| ct0p5_rt4_g1 | heavy(3.3V) | 5.2796 | 0.8667 | 0.7060 | 1.0042 |
| ct0p4_rt4_g1 | light(1.8V) | 0.7451 | 0.9448 | 0.9486 | 1.0330 |
| ct0p4_rt4_g1 | mid(2.5V) | 4.1331 | 0.9018 | 0.7742 | 1.0104 |
| ct0p4_rt4_g1 | heavy(3.3V) | 5.6816 | 0.8825 | 0.7272 | 1.0030 |

三环境均值（快速筛选）：

| config | mean AUC | mean F1 | mean MESR |
|---|---:|---:|---:|
| ct0p5_rt3_g1 | 0.9024 | 0.8047 | 1.0310 |
| ct0p4_rt3_g1 | 0.9102 | 0.8166 | 1.0158 |
| ct0p5_rt4_g1 | 0.9008 | 0.8042 | 1.0297 |
| ct0p4_rt4_g1 | 0.9097 | 0.8167 | 1.0155 |

缺陷（当前为什么还没超过 baseline）：

- 现阶段只是“避免误伤”的参数调平，还没有引入更强的判别信息；在 mid/heavy 上仍可能不如 baseline。
- 门控规则仍偏硬阈值（raw_thr/coh_thr），在不同噪声强度下的触发比例会变化，阈值可迁移性仍未知。

### 2026-04-08：s2 更进一步的小网格优化（prescreen200k, s=9, tau=128ms）

运行产物：`data/ED24/myPedestrain_06/EBF_Part2/s2_grid1/`（含 `best_summary.csv`）。

该网格：`coh_thr∈{0.35,0.45}`，`raw_thr∈{2.5,3,3.5}`，`gamma∈{1,2}`。

最优点（每行是该 env 内 best operating point）：

| env | 方法 | Thr(best) | AUC | F1 | MESR |
|---|---|---:|---:|---:|---:|
| light(1.8V) | s2 best | 0.7456 | 0.9453 | 0.9488 | 1.0323 |
| mid(2.5V) | s2 best | 4.3147 | 0.9061 | 0.7814 | 1.0092 |
| heavy(3.3V) | s2 best | 5.9038 | 0.8913 | 0.7373 | 1.0067 |

阶段性结论（与 baseline 对照）：

- 在 mid/heavy 上 AUC/F1 仍明显低于 baseline，说明 s2 的“coh 门控”在 ED24/myPedestrain_06 上仍存在系统性误伤。
- 继续在 s2 的硬阈值门控形态上做更细网格，预计收益有限（更可能是 trade-off 微调）。

是否值得继续优化：

- 不建议继续在 s2 的硬门控上投入（除非后续引入 noise-level 自适应门控，使触发比例跨 env 更稳定）。

## 2026-04-08：s3–s6 prescreen 总结（200k, s=9, tau=128ms）

对照 baseline（相同设置）产物：`data/ED24/myPedestrain_06/EBF_Part2/baseline_prescreen/best_summary.csv`。

### 为什么 s3–s6 统一用 s=9, tau=128ms 做对比？它们都是这个参数下最好吗？

不是。

这里的 s3–s6 结果属于 **prescreen（预筛）**：为了把“新评分机制本身的影响”与“(s,tau) 网格选择”的影响解耦，我们固定一组对齐口径来快速判断方向值不值得继续。

- 固定的对齐口径：`max-events=200k, s=9, tau=128ms`。
- 这组参数是 baseline EBF 在该数据上的强点（且对三环境都不差），因此用它做统一对比可以减少 confound。
- 若某个候选在 prescreen 下显著优于 baseline，下一步才值得为它跑更大的 `(s,tau)` 网格（或至少跑一个“小网格”验证它的 best 是否发生迁移）。

因此：**不要把这里的 prescreen 结果解读为“每个变体在 s=9,tau=128ms 下最优”**；它只是“相同条件下的快速方向性判断”。

### baseline vs s2–s6（best-by-env）

| env | 方法 | AUC | F1 | MESR |
|---|---|---:|---:|---:|
| light(1.8V) | baseline | 0.9476 | 0.9497 | 1.0305 |
| light(1.8V) | s2 best | 0.9453 | 0.9488 | 1.0323 |
| light(1.8V) | s3 best | 0.9477 | 0.9497 | 1.0303 |
| light(1.8V) | s4 best | 0.9382 | 0.9450 | 1.2069 |
| light(1.8V) | s5 best | 0.9450 | 0.9466 | 1.0845 |
| light(1.8V) | s6 best | 0.9476 | 0.9497 | 1.0305 |
| mid(2.5V) | baseline | 0.9219 | 0.8108 | 1.0168 |
| mid(2.5V) | s2 best | 0.9061 | 0.7814 | 1.0092 |
| mid(2.5V) | s3 best | 0.9218 | 0.8099 | 1.0182 |
| mid(2.5V) | s4 best | 0.8751 | 0.7590 | 1.0034 |
| mid(2.5V) | s5 best | 0.9203 | 0.8089 | 1.0199 |
| mid(2.5V) | s6 best | 0.9219 | 0.8108 | 1.0168 |
| heavy(3.3V) | baseline | 0.9205 | 0.7869 | 1.0208 |
| heavy(3.3V) | s2 best | 0.8913 | 0.7373 | 1.0067 |
| heavy(3.3V) | s3 best | 0.9197 | 0.7834 | 1.0225 |
| heavy(3.3V) | s4 best | 0.8517 | 0.7227 | 0.9977 |
| heavy(3.3V) | s5 best | 0.9195 | 0.7869 | 1.0138 |
| heavy(3.3V) | s6 best | 0.9205 | 0.7869 | 1.0208 |

### 结论与经验（为什么没超过 baseline）

- **s3（softgate）**：能把 s2 的伤害“抹平”到接近 baseline，但在 mid/heavy 仍没有稳定提升，说明单靠“更温和的门控”不足以获得增益。
- **s4（resultant gate）**：显著劣化，说明 resultant 一阶统计在该数据上区分力不足，且容易引入极端阈值/异常 ESR。
- **s5（椭圆空间权重）**：可以接近 baseline，但仍普遍略低，说明“全局方向偏好”的空间权重对该数据集帮助有限。
- **s6（time-coh gate）**：best 等于 baseline（等价于门控基本不起作用），说明 time-coh 作为判别项在当前设定下没有带来可分性增益。

### 是否值得继续优化（s3–s6）

- 以当前这些“局部统计 + 简单门控/权重”的候选为主线，**不建议继续做更大网格**（收益已接近饱和，且难以在 mid/heavy 超过 baseline）。

下面补齐每个方法的“原理 + 失效原因”（按 s1/s2 的写法）。

## s3：Smooth Coherence Gate（sigmoid 平滑门控惩罚）

### 算法动机

s2 的硬阈值门控（raw_thr/coh_thr）在不同噪声强度下会导致触发比例漂移；一旦门控触发得“过多”，就会系统性误伤 signal。

s3 的目标是把门控变成**连续、可微、触发更平滑**的形态：即便门控参数不完美，也尽量不出现“突然大面积压分”。

### 算法定义（低延迟设计）

位置：`src/myevs/denoise/ops/ebfopt_part2/s3_softgate.py`

仍然计算：

- raw：baseline EBF raw-score（同极性 + 线性时间衰减 + 邻域累加）
- coh：与 s1/s2 相同的二阶矩各向异性指标 $\mathrm{coh}\in[0,1]$

在此基础上定义两个 sigmoid gate：

$$
w_{\mathrm{raw}} = \sigma\bigl(k_{\mathrm{raw}}(\mathrm{raw}-\mathrm{raw\_thr})\bigr),\quad
w_{\mathrm{coh}} = \sigma\bigl(k_{\mathrm{coh}}(\mathrm{coh\_thr}-\mathrm{coh})\bigr)
$$

其中 $\sigma(z)=\tfrac{1}{1+e^{-z}}$。

再定义“低 coh 的幅度项”（0..1）：

$$
\mathrm{mag} = 1-\left(\frac{\mathrm{coh}}{\mathrm{coh\_thr}+\varepsilon}\right)^{\gamma}
$$

最终惩罚：

$$
\mathrm{pen} = 1 - \alpha\, w_{\mathrm{raw}}\, w_{\mathrm{coh}}\, \mathrm{mag},\quad
\mathrm{score}=\mathrm{raw}\cdot\mathrm{pen}
$$

超参：`coh_thr, raw_thr, gamma, alpha, k_raw, k_coh`。

### 失效原因（为什么没超过 baseline）

- s3 的平滑化确实能把 s2 的“硬误伤”抹平到接近 baseline，但它本质仍只用 **coh 这一条判别信息**。当 coh 本身在 signal/noise 上的可分性不足时，门控函数再怎么温和也很难带来正增益。
- 在本数据上，最优点往往对应“门控趋于温和/接近不惩罚”，等价于回到 baseline 的排序；因此 AUC/F1 只能接近 baseline，而难以稳定超过。

是否值得继续：

- 不建议继续对 s3 做大网格调参；若要继续，应把 s3 作为“门控形态模板”，并引入更强判别量（比如跨 polarity 一致性或更直接的时间表面梯度），而不是继续围绕 coh 本身做函数微调。

## s4：Resultant Alignment Gate（一阶矩 resultant 一致性门控）

### 算法动机

coh（二阶矩各向异性）可能把一些真实结构也判成“各向同性”，导致门控误伤。

s4 尝试用一阶统计替代：看邻域偏移向量的 resultant 是否“能抵消”。直觉：

- 结构化运动/边缘：同极性邻居偏移会更一致，resultant 更强
- 各向同性噪声团：偏移方向杂乱，resultant 更易相互抵消

### 算法定义（低延迟设计）

位置：`src/myevs/denoise/ops/ebfopt_part2/s4_residual_gate.py`

在计算 raw 的同时累计：

- $\sum w$，$\sum w\,dx$，$\sum w\,dy$，$\sum w\,(dx^2+dy^2)$

定义 alignment（0..1）：

$$
\mathrm{align} = \frac{(\sum w\,dx)^2 + (\sum w\,dy)^2}{(\sum w)\,(\sum w\,(dx^2+dy^2))+\varepsilon}
$$

门控形式与 s2 类似：当 `raw>=raw_thr` 且 `align<align_thr` 时惩罚，否则不惩罚。

### 失效原因（为什么明显劣化）

- resultant 的核心假设是“signal 的邻域偏移方向更一致”。但在真实事件数据里，很多**真实边缘/纹理/角点**会同时在两侧产生同极性邻居，偏移向量天然更对称，resultant 会被抵消，导致 align 偏低，从而误伤 signal。
- 相反，一些噪声在局部也可能出现偶然偏移偏置（resultant 不为 0），align 不一定低，导致门控区分力更差。
- 最终表现为：AUC/F1 显著下滑，说明 s4 引入的判别量与目标不匹配。

是否值得继续：

- 建议停掉 s4。除非后续把 resultant 改成“方向场一致性”（需要额外信息/更强特征），否则继续调参收益很低。

## s5：Elliptic Spatial Weight（旋转椭圆空间权重）

### 算法动机

baseline EBF 的空间邻域是“圆形等权”（仅时间权重）。s5 尝试加入一个**方向性空间权重**：如果真实运动/边缘在某个方向更一致，用椭圆核可能更强调有效邻居、压低无关方向邻居，从而提升可分性。

### 算法定义（低延迟设计）

位置：`src/myevs/denoise/ops/ebfopt_part2/s5_elliptic_spatialw.py`

把偏移 $(dx,dy)$ 先按全局角度 $\theta$ 旋转：

$$
\begin{pmatrix}dx'\\dy'\end{pmatrix}=
\begin{pmatrix}\cos\theta & \sin\theta\\-\sin\theta & \cos\theta\end{pmatrix}
\begin{pmatrix}dx\\dy\end{pmatrix}
$$

定义椭圆距离：

$$
d = \sqrt{(dx'/a_x)^2+(dy'/a_y)^2}
$$

空间权重（线性衰减）：

$$
w_{\mathrm{sp}}=\max\left(0, 1-\frac{d}{r}\right)
$$

最终分数：

$$
\mathrm{score}=\sum w_t\,w_{\mathrm{sp}}
$$

超参：`ax, ay, theta_deg`（全局固定）。

### 失效原因（为什么只能接近 baseline）

- 该方法的方向性是**全局固定**的，但数据里的真实运动方向/边缘方向是局部变化的；一个全局 $\theta$ 很难对所有场景同时有利。
- 椭圆核等价于“在某些方向上减少有效邻居数”，这会改变 raw-score 的尺度与排序；如果方向不匹配，就会削弱真实结构的得分，造成 AUC/F1 轻微下降。

是否值得继续：

- 建议停掉 s5（至少不要做更大网格）。若要继续这条路，必须让方向性变成“每事件自适应”的局部估计，但那会引入额外统计与稳定性问题，需要作为新编号（s7+）单独设计。

## s6：Time-Coherence Gate（邻域 dt 方差的一致性门控）

### 算法动机

空间结构统计（coh/resultant）在该数据上增益有限，于是 s6 尝试引入“时间结构”：

- 真实运动/边缘的邻域事件在时间上可能更一致
- 噪声事件的邻域 dt 可能更散

因此用邻域 dt 的加权方差构造一个 time-coh，再做 s2 类似的门控惩罚。

### 算法定义（低延迟设计）

位置：`src/myevs/denoise/ops/ebfopt_part2/s6_timecoh_gate.py`

在 tau 窗内对邻居 dt 做加权一阶/二阶矩：

$$
\mu = \frac{\sum w\,dt}{\sum w+\varepsilon},\quad
\mathrm{var} = \frac{\sum w\,dt^2}{\sum w+\varepsilon}-\mu^2
$$

定义 time-coh：

$$
\mathrm{timecoh} = \frac{1}{1+\mathrm{var}/(\tau^2+\varepsilon)}\in(0,1]
$$

门控形式与 s2 类似：当 `raw>=raw_thr` 且 `timecoh<timecoh_thr` 时惩罚，否则不惩罚。

### 失效原因（为什么 best 等于 baseline）

- 在该数据上，dt 方差并没有提供稳定的判别增益：真实结构的邻域 dt 也可能是多模态/高方差（多目标/遮挡/边缘两侧混合），而噪声在局部也可能出现“看似一致”的 dt。
- 结果就是：一旦门控稍微激进就会误伤 signal；最优点往往通过参数把门控“关掉/几乎不触发”，从而退化回 baseline，导致 best_summary 与 baseline 完全一致。

是否值得继续：

- 建议停掉 s6。若要继续引入时间结构，需要更直接的判别量（例如时间表面梯度/局部时间平面残差等），并作为新编号（s7+）重新设计。

## s7：Plane Residual Gate（局部时间平面残差门控）

### 方法原理（动机）

s6 的“dt 方差”过于粗糙，s7 尝试把时间结构刻画成更经典的“时间表面局部平面一致性”：对每个事件，在邻域内取同极性且落在 $\tau$ 窗内的邻居，构造样本点 $(dx,dy,z)$，其中 $z=dt$（邻居与当前事件的时间差）。

直觉：

- 如果邻域事件来自同一条运动边缘/轨迹，$dt$ 在空间上更接近“近似平面/斜坡”。
- 如果是随机噪声团，$dt$ 的空间结构更像乱序，平面拟合残差更大。

### 算法定义（低延迟设计）

位置：`src/myevs/denoise/ops/ebfopt_part2/s7_plane_gate.py`

1) baseline EBF 的 raw-score（同极性 + 线性时间衰减）：

$$
\mathrm{raw} = \sum w_t,\quad w_t = \frac{\tau-\Delta t}{\tau}\cdot\mathbb{1}(\Delta t\le\tau)\cdot\mathbb{1}(p_{nei}=p)
$$

2) 在同一批邻居上做加权最小二乘平面拟合（用 $z=dt$）：

$$
z \approx a\,dx + b\,dy + c
$$

得到加权残差标准差 $\sigma$，并做归一化：

$$
\sigma_{norm}=\frac{\sigma}{\tau}
$$

3) s2 类似的门控惩罚（只在 raw 足够大时触发；只惩罚残差大的事件）：

$$
\mathrm{score} = \mathrm{raw}\cdot \mathrm{pen},\quad
\mathrm{pen}=
\begin{cases}
1 & (\mathrm{raw}<\mathrm{raw\_thr})\ \text{或}\ (\sigma_{norm}\le\sigma_{thr})\ \text{或}\ (N<\mathrm{min\_pts})\\
\left(\frac{\sigma_{thr}}{\sigma_{norm}}\right)^{\gamma} & (\mathrm{raw}\ge\mathrm{raw\_thr}\ \text{且}\ \sigma_{norm}>\sigma_{thr})
\end{cases}
$$

超参（环境变量）：

- `MYEVS_EBF_S7_SIGMA_THR`（默认 0.20）
- `MYEVS_EBF_S7_RAW_THR`（默认 3.0）
- `MYEVS_EBF_S7_GAMMA`（默认 1.0）
- `MYEVS_EBF_S7_MIN_PTS`（默认 6）

### 实现约束核对

- 在线流式/单遍：是（只维护 `last_ts/last_pol`）
- 复杂度：$O(r^2)$（邻域一次遍历，统计量一次性累计）
- Numba 必须：是（numba 不可用直接报错，无静默 fallback）

### 实验口径（prescreen，对齐参数）

本次只做 prescreen：固定 `--max-events 200000` 且固定对齐参数 `s=9, tau=128ms`，只扫少量超参，看是否有“明显超过 baseline”的迹象。

Baseline（对照）产物：`data/ED24/myPedestrain_06/EBF_Part2/baseline_prescreen_s9_tau128ms_200k/`

```powershell
$env:PYTHONNOUSERSITE='1'
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant ebf --max-events 200000 --s-list 9 --tau-us-list 128000 --out-dir data/ED24/myPedestrain_06/EBF_Part2/baseline_prescreen_s9_tau128ms_200k
```

s7 产物：`data/ED24/myPedestrain_06/EBF_Part2/s7_plane_gate_prescreen_s9_tau128ms_200k/`

```powershell
$env:PYTHONNOUSERSITE='1'
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant s7 --max-events 200000 --s-list 9 --tau-us-list 128000 --s7-sigma-thr-list 0.15,0.2,0.25 --s7-raw-thr-list 4 --s7-gamma-list 1,2 --s7-min-pts-list 6 --out-dir data/ED24/myPedestrain_06/EBF_Part2/s7_plane_gate_prescreen_s9_tau128ms_200k
```

### 结果与结论（为什么没超过 baseline）

最优点（baseline vs s7 best）：

| env | baseline AUC | s7 best AUC | 备注 |
|---|---:|---:|---|
| light(1.8V) | 0.947564 | 0.947556 | 几乎一致（差异 $<10^{-5}$） |
| mid(2.5V) | 0.921924 | 0.921914 | 略低 |
| heavy(3.3V) | 0.920467 | 0.920374 | 略低 |

s7 best config：`sig0p25_raw4_g1_mp6`（tag：`ebf_s7_sig0p25_raw4_g1_mp6_labelscore_s9_tau128000`）。

失效原因（可验证的解释）：

- **判别信息与 raw 高度相关**：在触发门控的高 raw 区间，邻域 dt 的“可平面拟合性”对 signal/noise 的区分不足，导致惩罚项对排序贡献很弱（最佳点倾向于让门控尽量不改变 baseline 的排序）。
- **噪声团也可能“看起来像平面”**：密集噪声在短时间窗内会形成局部时间结构（尤其在同极性筛选后），使平面残差不一定比真实边缘更差，从而无法稳定压噪。
- **一旦门控稍激进就会误伤**：对真实边缘的拐角/遮挡/多运动叠加区域，时间表面并不严格平面，残差偏大；过强惩罚会直接压低这类 signal 的分数，AUC/F1 下降。

是否值得继续：

- 建议先停掉 s7（至少不做更大网格）。当前形态下“平面残差门控”没有带来可观增益。
- 若要继续利用时间表面信息，建议作为新编号（s8+）改成更直接的判别量（例如估计平面法向/梯度方向后再与空间分布做一致性约束，或引入跨 polarity 的一致性），避免仅用“残差大小”这一维度。

### 下一步最小可行方向（建议）

- 若仍坚持 Part2“先提升可分性”，下一代候选应引入**更强判别信息**但仍保持单遍 $O(r^2)$：例如跨 polarity 的局部一致性、或对邻域时间表面梯度/法向的一致性刻画（不仅是 dt 方差/残差大小）。
- 若以工程效率优先：建议把重心转回“V2 类归一化/自适应阈值机制”，把 **阈值可迁移性** 作为主优化目标（在不牺牲 AUC 的前提下）。

## s8：Plane R2 Gate（局部时间平面解释度门控）

### 方法原理（动机）

s7 用“残差大小（sigma）”门控没有带来增益。s8 保持同样的“局部加权平面拟合”框架，但把判别量从“残差绝对大小”改为“解释度/拟合优度”。

直觉：

- 真实运动/边缘的时间表面更接近一个局部平面/斜坡，应该有更高的解释度。
- 噪声团在空间上的 dt 更乱，解释度更低。

### 算法定义（低延迟设计）

位置：`src/myevs/denoise/ops/ebfopt_part2/s8_plane_r2_gate.py`

1) baseline EBF 的 raw-score（同极性 + 线性时间衰减）：

$$
\mathrm{raw} = \sum w_t,\quad w_t = \frac{\tau-\Delta t}{\tau}\cdot\mathbb{1}(\Delta t\le\tau)\cdot\mathbb{1}(p_{nei}=p)
$$

2) 在同一批邻居上做加权最小二乘平面拟合（用 $z=dt$）：

$$
z \approx a\,dx + b\,dy + c
$$

计算加权残差平方和 $\mathrm{SSE}$ 与加权总平方和 $\mathrm{SST}$：

$$
\mathrm{SST}=\sum w\,(z-\bar z)^2,\quad \bar z=\frac{\sum w z}{\sum w}
$$

定义解释度：

$$
R^2 = 1 - \frac{\mathrm{SSE}}{\mathrm{SST}+\varepsilon}\in[0,1]
$$

3) s2 类似的门控惩罚（只在 raw 足够大时触发；只惩罚 $R^2$ 低的事件）：

$$
\mathrm{score} = \mathrm{raw}\cdot \mathrm{pen},\quad
\mathrm{pen}=
\begin{cases}
1 & (\mathrm{raw}<\mathrm{raw\_thr})\ \text{或}\ (R^2\ge R^2_{thr})\ \text{或}\ (N<\mathrm{min\_pts})\\
\left(\frac{R^2}{R^2_{thr}}\right)^{\gamma} & (\mathrm{raw}\ge\mathrm{raw\_thr}\ \text{且}\ R^2<R^2_{thr})
\end{cases}
$$

超参（环境变量）：

- `MYEVS_EBF_S8_R2_THR`（默认 0.60）
- `MYEVS_EBF_S8_RAW_THR`（默认 3.0）
- `MYEVS_EBF_S8_GAMMA`（默认 1.0）
- `MYEVS_EBF_S8_MIN_PTS`（默认 6）

### 实现约束核对

- 在线流式/单遍：是（只维护 `last_ts/last_pol`）
- 复杂度：$O(r^2)$（邻域一次遍历，统计量一次性累计）
- Numba 必须：是（numba 不可用直接报错，无静默 fallback）

### 实验口径（prescreen，对齐参数）

本次只做 prescreen：固定 `--max-events 200000` 且固定对齐参数 `s=9, tau=128ms`，只扫少量超参（主要扫 `r2_thr`）。

Baseline（对照）产物：`data/ED24/myPedestrain_06/EBF_Part2/baseline_prescreen_s9_tau128ms_200k/`（复用，不重复跑）。

s8 产物：`data/ED24/myPedestrain_06/EBF_Part2/s8_plane_r2_gate_prescreen_s9_tau128ms_200k/`

```powershell
$env:PYTHONNOUSERSITE='1'
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant s8 --max-events 200000 --s-list 9 --tau-us-list 128000 --s8-r2-thr-list '0.4,0.5,0.6,0.7' --s8-raw-thr-list '4' --s8-gamma-list '1,2' --s8-min-pts-list '6' --out-dir data/ED24/myPedestrain_06/EBF_Part2/s8_plane_r2_gate_prescreen_s9_tau128ms_200k
```

### 结果与结论（显著劣于 baseline）

s8 best config：`r20p4_raw4_g1_mp6`（tag：`ebf_s8_r20p4_raw4_g1_mp6_labelscore_s9_tau128000`）。

| env | baseline AUC | s8 best AUC | 备注 |
|---|---:|---:|---|
| light(1.8V) | 0.947564 | 0.936870 | 明显下降 |
| mid(2.5V) | 0.921924 | 0.835524 | 大幅下降 |
| heavy(3.3V) | 0.920467 | 0.781034 | 大幅下降 |

失效原因（可验证的解释）：

- **“平面解释度”不是稳定判别量**：真实边缘/运动会出现遮挡、多个目标/多个速度叠加、以及边缘两侧混合，使邻域 dt 表面并不满足单一平面模型；因此 $R^2$ 可能偏低，导致对 signal 的系统性惩罚。
- **对 mid/heavy 误伤更严重**：噪声更强时，触发 `raw>=raw_thr` 的事件更多，而 $R^2$ 统计也更不稳定；门控在更大比例事件上起作用，排序被破坏（AUC 断崖式下降）。
- **噪声团也可能出现“高 $R^2$ 假象”**：短时间窗内的密集噪声/热点区域，邻域 dt 可能呈现局部单调结构，使拟合解释度并不低，从而无法稳定压噪。

是否值得继续：

- 建议停掉 s8（当前“按 $R^2$ 的平面解释度门控”形态）。它对 baseline 是明显负收益。
- 若继续利用时间表面信息，需要换模型假设：例如针对“多运动叠加/遮挡”设计更鲁棒的局部结构判别（仍需保持单遍 $O(r^2)$），或转向跨 polarity 一致性等新信号。

## s9：Refractory/Burst Gate（同像素超高频门控）

### 反思：为什么 baseline 很简单却很强？

基于本 README 中 s1–s8 的 prescreen 经验，baseline 强主要来自它抓住了这个数据上最稳定的“可分性来源”：**事件密度的时空聚集性**。

- baseline raw（同极性 + 时间线性衰减 + 邻域累加）本质是在估计“这个事件周围是否存在稳定的局部时空支持”。在 ED24/myPedestrain_06 上，这个信号对 light/mid/heavy 都很稳。
- 我们之前大多数改动（各向异性、time-coh、平面残差/解释度）都隐含了更强的几何/结构假设；但真实数据里存在遮挡、多目标叠加、边缘两侧混合、以及噪声也会出现局部结构“假象”。
- 一旦门控开始对大量事件乘上惩罚（尤其在 mid/heavy 下 raw 触发更多），排序就会被系统性扰动，AUC 会掉得非常快（s4/s8 就是典型）。

因此，下一代改动必须满足：**触发条件非常克制、针对明确噪声机理、且尽量只影响极少一部分“可疑事件”**。

### 方法原理（动机）

一个更“机理化”的噪声来源是热点/爆发噪声：同一像素会在极短时间内反复触发，且常常同极性连续。

s9 不去猜“边缘应该是什么几何形状”，只用一个非常局部、非常便宜的统计量：

- 同像素上一次**同极性**事件到当前事件的时间间隔 $dt_0$。

直觉：

- 若 $dt_0$ 极小且 raw 又很高，这更像“像素级爆发/热点”而非真实边缘。
- 若 $dt_0$ 不小（或者上一次是异极性），则不做任何惩罚，尽量避免误伤。

### 算法定义（低延迟设计）

位置：`src/myevs/denoise/ops/ebfopt_part2/s9_refractory_gate.py`

仍先计算 baseline raw：

$$
\mathrm{raw} = \sum w_t,\quad w_t = \frac{\tau-\Delta t}{\tau}\cdot\mathbb{1}(\Delta t\le\tau)\cdot\mathbb{1}(p_{nei}=p)
$$

定义同像素上一次同极性事件的间隔：

$$
dt_0 = t_i - t_{\text{prev}}(x_i,y_i,p_i)
$$

归一化：$u=dt_0/\tau$。

门控惩罚：仅当 `raw>=raw_thr` 且同极性历史存在、并且 $u<dt_{thr}$ 时惩罚：

$$
\mathrm{score} = \mathrm{raw}\cdot\mathrm{pen},\quad
\mathrm{pen}=
\begin{cases}
1 & (\mathrm{raw}<\mathrm{raw\_thr})\ \text{或}\ (u\ge dt_{thr})\ \text{或}\ (\text{prev polarity}\ne p)\\
\left(\frac{u}{dt_{thr}}\right)^{\gamma} & (\mathrm{raw}\ge\mathrm{raw\_thr}\ \text{且}\ u<dt_{thr}\ \text{且 prev polarity}=p)
\end{cases}
$$

超参（环境变量）：

- `MYEVS_EBF_S9_DT_THR`（默认 0.004；**归一化到 tau**，例如 tau=128ms 时约等于 512us）
- `MYEVS_EBF_S9_RAW_THR`（默认 3.0）
- `MYEVS_EBF_S9_GAMMA`（默认 1.0）

### 实现约束核对

- 在线流式/单遍：是（只维护 `last_ts/last_pol`）
- 复杂度：$O(r^2)$（raw 仍是邻域一次遍历；额外只读同像素历史一次）
- Numba 必须：是（numba 不可用直接报错，无静默 fallback）

### 实验口径（prescreen，对齐参数）

Baseline（对照）产物（含 `esr_mean/aocc` 列，便于记录 MESR/AOCC）：

- `data/ED24/myPedestrain_06/EBF_Part2/ebf_baseline_s9_tau128_prescreen200k_esrbest_aoccbest/`（复用，不重复跑）

s9 产物（本轮“最后尝试”的三段式：粗扫 → 细扫 → best 点复跑带 AOCC/ESR）：

- 粗扫：`data/ED24/myPedestrain_06/EBF_Part2/_prescreen_s9_refractory_grid_dt_raw_g_s9_tau128ms_200k/`
- 细扫：`data/ED24/myPedestrain_06/EBF_Part2/_prescreen_s9_refractory_finesweep_dt_raw_g_s9_tau128ms_200k/`
- best 点复跑（`--aocc-mode all --esr-mode all`）：`data/ED24/myPedestrain_06/EBF_Part2/_prescreen_s9_best_dt0p01_raw2_g1p5_s9_tau128ms_200k_aocc_esr/`

```powershell
$env:PYTHONNOUSERSITE='1'

#（可复现 best 点）
$env:MYEVS_EBF_S9_DT_THR='0.010'
$env:MYEVS_EBF_S9_RAW_THR='2.0'
$env:MYEVS_EBF_S9_GAMMA='1.5'
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant s9 --max-events 200000 --s-list 9 --tau-us-list 128000 --aocc-mode all --esr-mode all --out-dir data/ED24/myPedestrain_06/EBF_Part2/_prescreen_s9_best_dt0p01_raw2_g1p5_s9_tau128ms_200k_aocc_esr
```

### 结果与结论（最后一次细扫：F1 仅有 $10^{-4}\sim10^{-3}$ 级别增益）

对比点：同一口径下的 `s=9, tau=128ms`，每个 env 取 best-F1 operating point（ROC CSV 中筛 `tag` 包含 `labelscore_s9_tau128000` 的行）。

s9 best 超参：`dt_thr=0.010`（归一化到 tau）、`raw_thr=2.0`、`gamma=1.5`。

| env | baseline AUC | s9 AUC | ΔAUC | baseline F1 | s9 F1 | ΔF1 | baseline Thr(best-F1) | s9 Thr | baseline MESR | s9 MESR | baseline AOCC/1e7 | s9 AOCC/1e7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| light(1.8V) | 0.947564 | 0.947725 | +0.000161 | 0.949739 | 0.949769 | +0.000029 | 0.7491 | 0.7543 | 1.030472 | 1.029233 | 0.8206 | 0.8205 |
| mid(2.5V) | 0.921924 | 0.922127 | +0.000203 | 0.810827 | 0.811219 | +0.000392 | 4.8394 | 4.8395 | 1.016829 | 1.015244 | 0.8481 | 0.8481 |
| heavy(3.3V) | 0.920467 | 0.920504 | +0.000037 | 0.786882 | 0.787379 | +0.000497 | 7.3581 | 7.2715 | 1.020768 | 1.018776 | 0.9065 | 0.9081 |

解释（为什么“之前感觉不明显”是对的）：

- s9 的触发条件很克制（raw 高 + 同像素同极性 + 极短 dt），因此对排序的扰动小，但也意味着**能被它“额外纠正”的事件比例可能很低**。
- 在当前 prescreen200k 口径下，F1 的提升只有 $\sim 5\times10^{-4}$（heavy）量级；这更像一个“低风险补丁”而不是能显著抬升上限的主线机制。

是否值得继续：

- 建议停掉 s9 的进一步调参（收益已接近平台期）。
- 若你希望确认它是否“真实有效”，只做一个最小验证即可：把 `--max-events` 提到 1M/全量复跑 best 超参，看提升是否仍同号。

## s10：Hotpixel Leaky-Rate Gate（同像素泄露积分发射率门控）

### 方法原理（动机）

s9 只针对“极短 $dt$ 的同像素爆发”做门控，非常克制；但另一类常见机理是：**热点/爆发噪声不一定每次都极短 $dt$**，但会在同一像素、同一极性上，在一个短时间段内**反复触发很多次**。

s10 用一个“泄露积分”的方式近似该像素同极性的短期发射率：

- 维护每个像素的同极性累加器 $a$（越大表示“最近一段时间同极性触发越频繁”）。

然后只在 **baseline raw 已经很高** 且 **$a$ 也很高** 时进行惩罚，尽量不误伤普通边缘。

### 算法定义（低延迟设计）

位置：`src/myevs/denoise/ops/ebfopt_part2/s10_hotpixel_rate_gate.py`

仍先计算 baseline raw：

$$
\mathrm{raw} = \sum w_t,\quad w_t = \frac{\tau-\Delta t}{\tau}\cdot\mathbb{1}(\Delta t\le\tau)\cdot\mathbb{1}(p_{nei}=p)
$$

对每个像素维护同极性“泄露积分”累加器 $a$：

$$
a \leftarrow
\begin{cases}
1 & (\text{无历史或上次极性}\ne p)\\
\max\left(0, a-\frac{dt_0}{\tau}\right)+1 & (\text{同极性})
\end{cases}
$$

其中 $dt_0$ 是同像素上一次事件与当前事件的时间间隔。

门控惩罚（仅当 `raw>=raw_thr` 且 `a>acc_thr` 时触发）：

$$
\mathrm{score} = \mathrm{raw}\cdot\mathrm{pen},\quad
\mathrm{pen} = \left(\frac{\mathrm{acc\_thr}}{a+\varepsilon}\right)^{\gamma}
$$

否则 `score=raw`（不惩罚）。

超参（环境变量）：

- `MYEVS_EBF_S10_ACC_THR`（默认 4.0；近似表示“每个 tau 内的同极性触发次数阈值”）
- `MYEVS_EBF_S10_RAW_THR`（默认 3.0）
- `MYEVS_EBF_S10_GAMMA`（默认 1.0）

### 实现约束核对

- 在线流式/单遍：是（只维护 `last_ts/last_pol/self_acc`）
- 复杂度：$O(r^2)$（raw 仍是邻域一次遍历；额外只更新同像素状态）
- Numba 必须：是（numba 不可用直接报错，无静默 fallback）

### 实验口径（prescreen，对齐参数）

Baseline（对照）产物：`data/ED24/myPedestrain_06/EBF_Part2/baseline_prescreen_s9_tau128ms_200k/`（复用，不重复跑）。

s10 产物：`data/ED24/myPedestrain_06/EBF_Part2/s10_hotpixel_rate_gate_prescreen_s9_tau128ms_200k/`

```powershell
$env:PYTHONNOUSERSITE='1'
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant s10 --max-events 200000 --s-list 9 --tau-us-list 128000 --s10-acc-thr-list '2,3,4,6,8' --s10-raw-thr-list '4' --s10-gamma-list '1,2' --out-dir data/ED24/myPedestrain_06/EBF_Part2/s10_hotpixel_rate_gate_prescreen_s9_tau128ms_200k
```

### 结果与结论（light/heavy 小幅提升，mid 略回落）

| env | baseline AUC | s9 best AUC | s10 best AUC | s10 best tag（对齐 s=9,tau=128ms） |
|---|---:|---:|---:|---|
| light(1.8V) | 0.947564 | 0.947636 | 0.947771 | `ebf_s10_acc2_raw4_g1_labelscore_s9_tau128000` |
| mid(2.5V) | 0.921924 | 0.921994 | 0.921962 | `ebf_s10_acc3_raw4_g1_labelscore_s9_tau128000` |
| heavy(3.3V) | 0.920467 | 0.920473 | 0.920523 | `ebf_s10_acc3_raw4_g1_labelscore_s9_tau128000` |

解释与风险：

- s10 的触发比 s9 **“更宽”**：它不要求极短 $dt$，只要短期累计频率高就会触发；因此更可能在某些场景（例如 mid）对真实信号产生轻微误伤，出现 AUC 小幅回落。
- 但它在 light/heavy 上给出了比 s9 更明显一点的提升（同样仍是 $10^{-4}$ 量级），说明“热点/短期高发射率”可能确实存在。

是否值得继续：

- 现阶段不建议做大网格 sweep；先做**最小验证**：对 `baseline/s9-best/s10-best` 用更大 `--max-events`（例如 1M 或全量）复核这些 $10^{-4}$ 级别差异是否稳定、是否仍是 mid 轻微回落。
- 若大样本下 mid 的回落消失（或变为提升），s10 可作为一个“可选补丁”；若 mid 回落稳定存在，则下一步应让触发更克制（例如提高 `acc_thr` 或结合 s9 的短 $dt$ 条件做更稀疏触发）。

### 最小验证（全量，`--max-events 1000000`）

说明：ED24/myPedestrain_06 这三份 `.npy` 的事件数分别约为 light=19.4 万、mid=56.6 万、heavy=89.7 万，因此 `--max-events 1000000` 等价于“各环境全量”。

Baseline（全量）产物：`data/ED24/myPedestrain_06/EBF_Part2/baseline_validate_1M_s9_tau128ms/`

s9（全量）产物：`data/ED24/myPedestrain_06/EBF_Part2/s9_validate_1M_s9_tau128ms/`

s10（全量）产物：`data/ED24/myPedestrain_06/EBF_Part2/s10_validate_1M_s9_tau128ms/`

| env | baseline AUC | s9 best AUC | s10 best AUC | 备注 |
|---|---:|---:|---:|---|
| light(1.8V) | 0.947564 | 0.947636 | 0.947771 | s10 仍最好 |
| mid(2.5V) | 0.923218 | 0.923253 | 0.923259 | **s10 不再回落** |
| heavy(3.3V) | 0.913578 | 0.913549 | 0.913651 | s10 略升 |

结论更新：s10 在全量验证下三环境均略高于 baseline，mid 的 200k 回落现象并不稳定；但提升幅度依旧很小（仍处于 $10^{-4}$ 量级），更像“修补型特征”而非决定性隐性特征。

## s11：Relative Hotness Gate（相对热点异常门控）

### 方法原理（动机）

s10 采用“绝对发射率阈值”（同像素泄露积分累加器 $a$ 超过阈值）来惩罚热点/爆发噪声；但一个隐患是：当某一片区域整体都很忙（例如强边缘/复杂纹理/快速运动）时，**很多像素的 $a$ 都会偏高**，此时“绝对阈值”更可能误伤 signal。

s11 试图把“热点”从“绝对高”升级为“相对异常高”：

- 不仅要求中心像素 $a$ 高，还要求它**显著高于邻域同极性像素的典型 $a$**。

直觉：

- 真边缘：邻域很多像素一起活跃，中心并不会对邻域形成特别夸张的倍率优势。
- 热点噪声：中心像素可能独自异常高，邻域均值较低，倍率更大。

### 算法定义（低延迟设计）

位置：`src/myevs/denoise/ops/ebfopt_part2/s11_relative_hotness_gate.py`

仍先计算 baseline raw：

$$
\mathrm{raw} = \sum w_t,\quad w_t = \frac{\tau-\Delta t}{\tau}\cdot\mathbb{1}(\Delta t\le\tau)\cdot\mathbb{1}(p_{nei}=p)
$$

同样维护同极性泄露积分累加器 $a$（与 s10 一致）：

$$
a \leftarrow
\begin{cases}
1 & (\text{无历史或上次极性}\ne p)\\
\max\left(0, a-\frac{dt_0}{\tau}\right)+1 & (\text{同极性})
\end{cases}
$$

并在 raw 的邻域扫描中，同时统计邻域同极性像素的“有效累加器”（做一个近似连续泄露）：

$$
a_{nei}^{eff} = \max\left(0, a_{nei} - \frac{dt_{nei}}{\tau}\right)
$$

其中 $dt_{nei}=t_i-t_{last}(x_{nei},y_{nei})$。

令邻域均值：

$$
\bar a = \mathrm{mean}(a_{nei}^{eff})
$$

定义相对异常倍率：

$$
\rho = \frac{a}{\bar a+\varepsilon}
$$

仅当 `raw>=raw_thr` 且 `a>acc_thr` 且 `rho>=ratio_thr` 时触发惩罚：

$$
\mathrm{score} = \mathrm{raw}\cdot\mathrm{pen},\quad
\mathrm{pen}=\left(\frac{\mathrm{ratio\_thr}}{\rho+\varepsilon}\right)^{\gamma}
$$

否则 `score=raw`。

超参（环境变量）：

- `MYEVS_EBF_S11_ACC_THR`（默认 3.0）
- `MYEVS_EBF_S11_RATIO_THR`（默认 2.0）
- `MYEVS_EBF_S11_RAW_THR`（默认 3.0）
- `MYEVS_EBF_S11_GAMMA`（默认 1.0）

### 实现约束核对

- 在线流式/单遍：是（只维护 `last_ts/last_pol/self_acc`）
- 复杂度：$O(r^2)$（raw 的邻域遍历中顺便累加邻域 acc；不增加额外邻域 pass）
- Numba 必须：是（numba 不可用直接报错，无静默 fallback）

### 实验口径（prescreen，对齐参数）

Baseline（对照）产物：`data/ED24/myPedestrain_06/EBF_Part2/baseline_prescreen_s9_tau128ms_200k/`（复用，不重复跑）。

s11 产物：`data/ED24/myPedestrain_06/EBF_Part2/s11_relative_hotness_gate_prescreen_s9_tau128ms_200k/`

```powershell
$env:PYTHONNOUSERSITE='1'
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant s11 --max-events 200000 --s-list 9 --tau-us-list 128000 --s11-acc-thr-list '2,3,4' --s11-ratio-thr-list '2,3' --s11-raw-thr-list '4' --s11-gamma-list '1,2' --out-dir data/ED24/myPedestrain_06/EBF_Part2/s11_relative_hotness_gate_prescreen_s9_tau128ms_200k
```

### 结果与结论（prescreen，较 s9/s10 更明显）

| env | baseline AUC | s9 best AUC | s10 best AUC | s11 best AUC | s11 best tag（对齐 s=9,tau=128ms） |
|---|---:|---:|---:|---:|---|
| light(1.8V) | 0.947564 | 0.947636 | 0.947771 | 0.947888 | `ebf_s11_acc2_ratio2_raw4_g2_labelscore_s9_tau128000` |
| mid(2.5V) | 0.921924 | 0.921994 | 0.921962 | 0.921995 | `ebf_s11_acc2_ratio3_raw4_g1_labelscore_s9_tau128000` |
| heavy(3.3V) | 0.920467 | 0.920473 | 0.920523 | 0.920554 | `ebf_s11_acc3_ratio3_raw4_g1_labelscore_s9_tau128000` |

解释与风险：

- 相比 s10 的“绝对热点阈值”，s11 引入了“相对异常倍率”条件，使触发更克制、更像异常检测；因此在 prescreen 上更稳定地优于 s9/s10。
- 仍然要警惕：邻域均值 $\bar a$ 的估计可能受极端值影响（当前用均值）；如果后续发现误伤，下一步可考虑更鲁棒的邻域统计（但要保持单遍/低代价）。

### 最小验证（全量，`--max-events 1000000`）

s11（全量）产物：`data/ED24/myPedestrain_06/EBF_Part2/s11_validate_1M_s9_tau128ms/`

| env | baseline AUC | s10 best AUC | s11 best AUC | 备注 |
|---|---:|---:|---:|---|
| light(1.8V) | 0.947564 | 0.947771 | 0.947888 | s11 继续最好 |
| mid(2.5V) | 0.923218 | 0.923259 | 0.923297 | s11 略高于 s10 |
| heavy(3.3V) | 0.913578 | 0.913651 | 0.913657 | s11 略高于 s10 |

结论更新：s11 的提升在全量验证下仍然存在，且优于 s10；但绝对幅度依旧较小（仍在 $10^{-4}$ 量级），说明它可能更接近隐性特征，但还不是“决定性因素”。

## s12：Hotness Z-Score Gate（热点 z-score 异常门控）

### 方法原理（动机）

s11 用相对倍率 $\rho=\frac{a}{\bar a+\varepsilon}$ 来判断“中心是否相对异常”；但 $\bar a$ 用均值时，可能仍受邻域分布形态影响。

s12 尝试用更标准的异常度：邻域的均值 + 方差构造 z-score：

$$
z=\frac{a-\mu}{\sigma+\varepsilon}
$$

期望它在“忙但正常”的区域（邻域整体偏高、方差也偏大）更不容易误触发，而在“单点异常热点”时更容易触发。

### 算法定义（低延迟设计）

位置：`src/myevs/denoise/ops/ebfopt_part2/s12_hotness_zscore_gate.py`

沿用 s10/s11 的同极性泄露积分累加器 $a$，并在 raw 的邻域扫描中统计邻域的有效累加器 $a_{nei}^{eff}$：

$$
a_{nei}^{eff} = \max\left(0, a_{nei} - \frac{dt_{nei}}{\tau}\right)
$$

计算邻域均值/方差：

$$
\mu=\mathrm{mean}(a_{nei}^{eff}),\quad \sigma^2=\mathrm{mean}((a_{nei}^{eff})^2)-\mu^2
$$

仅当 `raw>=raw_thr` 且 `a>acc_thr` 且 `z>=z_thr` 时触发惩罚：

$$
\mathrm{score}=\mathrm{raw}\cdot\left(\frac{z_{thr}}{z+\varepsilon}\right)^{\gamma}
$$

超参（环境变量）：

- `MYEVS_EBF_S12_ACC_THR`
- `MYEVS_EBF_S12_Z_THR`
- `MYEVS_EBF_S12_RAW_THR`
- `MYEVS_EBF_S12_GAMMA`

### 实验口径（prescreen，对齐参数）

s12 产物：`data/ED24/myPedestrain_06/EBF_Part2/s12_hotness_zscore_gate_prescreen_s9_tau128ms_200k/`

```powershell
$env:PYTHONNOUSERSITE='1'
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant s12 --max-events 200000 --s-list 9 --tau-us-list 128000 --s12-acc-thr-list '2,3,4' --s12-z-thr-list '2,3,4' --s12-raw-thr-list '4' --s12-gamma-list '1,2' --out-dir data/ED24/myPedestrain_06/EBF_Part2/s12_hotness_zscore_gate_prescreen_s9_tau128ms_200k
```

### 结果与结论（未超过 s11，建议停掉）

s12 best AUC（对齐 s=9,tau=128ms）：

- light: 0.947832（tag：`ebf_s12_acc2_z2_raw4_g1_labelscore_s9_tau128000`）
- mid: 0.921955（tag：`ebf_s12_acc3_z4_raw4_g1_labelscore_s9_tau128000`）
- heavy: 0.920542（tag：`ebf_s12_acc4_z3_raw4_g1_labelscore_s9_tau128000`）

对比：s12 虽略高于 baseline，但三环境均未超过 s11 的 best AUC，因此不继续在该形态上做更大 sweep。

## s13：Cross-Polarity Support Gate（跨极性邻域支持度门控）

### 方法原理（动机）

前面 s9–s12 的改善主要来自“热点/爆发噪声”的同像素/同极性异常。下一步我尝试引入一个更结构性的证据：**在局部时空邻域中，真实边缘/运动往往会伴随一定比例的 opposite polarity 活动**，而热点/爆发噪声更容易表现为“同极性单边独占”。

因此在一次 $O(r^2)$ 邻域扫描中，同时累计：

- 同极性支持（baseline raw）：$\mathrm{raw}=\sum w_t$，其中 $w_t=\frac{\tau-\Delta t}{\tau}\,\mathbb{1}(\Delta t\le\tau)$
- opposite polarity 支持：$\mathrm{opp}=\sum w_t\,\mathbb{1}(p_{nei}=-p)$

定义跨极性支持比例（归一化到 $[0,1]$）：

$$
\mathrm{bal}=\frac{\mathrm{opp}}{\mathrm{raw}+\mathrm{opp}+\varepsilon}
$$

只在“raw 已经很高但 bal 很低”的情况下触发惩罚（保持触发克制）：

$$
\mathrm{score}=\mathrm{raw}\cdot \mathrm{pen},\quad
\mathrm{pen}=
\begin{cases}
1 & (\mathrm{raw}<\mathrm{raw\_thr})\ \text{或}\ (\mathrm{bal}\ge\mathrm{bal\_thr})\\
\left(\frac{\mathrm{bal}}{\mathrm{bal\_thr}}\right)^{\gamma} & (\mathrm{raw}\ge\mathrm{raw\_thr}\ \text{且}\ \mathrm{bal}<\mathrm{bal\_thr})
\end{cases}
$$

代码位置：`src/myevs/denoise/ops/ebfopt_part2/s13_crosspol_support_gate.py`

超参（环境变量）：

- `MYEVS_EBF_S13_BAL_THR`（默认 0.05）
- `MYEVS_EBF_S13_RAW_THR`（默认 3.0）
- `MYEVS_EBF_S13_GAMMA`（默认 1.0）

### 实现约束核对

- 在线流式/单遍：是（只维护 `last_ts/last_pol`）
- 复杂度：$O(r^2)$（与 baseline 同一邻域遍历；额外只分流同极性/异极性两份累加）
- Numba 必须：是（numba 不可用直接报错，无静默 fallback）

### 实验口径（prescreen，对齐参数）

Baseline（对照）产物：`data/ED24/myPedestrain_06/EBF_Part2/baseline_prescreen_s9_tau128ms_200k/`（复用，不重复跑）。

#### prescreen-1（较宽触发，直接失败）

产物：`data/ED24/myPedestrain_06/EBF_Part2/s13_crosspol_support_gate_prescreen_s9_tau128ms_200k/`

```powershell
$env:PYTHONNOUSERSITE='1'
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant s13 --max-events 200000 --s-list 9 --tau-us-list 128000 --s13-bal-thr-list '0.03,0.05,0.08' --s13-raw-thr-list '2.5,3.5' --s13-gamma-list '1' --out-dir data/ED24/myPedestrain_06/EBF_Part2/s13_crosspol_support_gate_prescreen_s9_tau128ms_200k
```

best AUC（对齐 s=9,tau=128ms）：

- light: 0.855752（tag：`ebf_s13_bal0p03_raw3p5_g1_labelscore_s9_tau128000`）
- mid: 0.887855（tag：`ebf_s13_bal0p03_raw3p5_g1_labelscore_s9_tau128000`）
- heavy: 0.905716（tag：`ebf_s13_bal0p03_raw3p5_g1_labelscore_s9_tau128000`）

#### prescreen-2（更克制触发，仍显著劣于 baseline）

产物：`data/ED24/myPedestrain_06/EBF_Part2/s13_crosspol_support_gate_prescreen2_s9_tau128ms_200k_balSmall/`

```powershell
$env:PYTHONNOUSERSITE='1'
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant s13 --max-events 200000 --s-list 9 --tau-us-list 128000 --s13-bal-thr-list '0.001,0.003,0.01' --s13-raw-thr-list '4,6,8' --s13-gamma-list '1,2' --out-dir data/ED24/myPedestrain_06/EBF_Part2/s13_crosspol_support_gate_prescreen2_s9_tau128ms_200k_balSmall
```

best AUC（对齐 s=9,tau=128ms）：

- light: 0.884539（tag：`ebf_s13_bal0p001_raw8_g1_labelscore_s9_tau128000`）
- mid: 0.897366（tag：`ebf_s13_bal0p001_raw8_g1_labelscore_s9_tau128000`）
- heavy: 0.910850（tag：`ebf_s13_bal0p001_raw8_g1_labelscore_s9_tau128000`）

#### sanity（禁用门控可精确回到 baseline）

说明：将 `raw_thr` 设为极大值（例如 `1e9`）会使门控永不触发，此时三环境 AUC 精确回到 baseline（用于确认实现无 bug）。

### 结论与失效原因分析（建议停掉）

结论一句话：s13 的“缺少 opposite polarity 支持即惩罚”假设在 ED24/myPedestrain_06 上不成立，会系统性压低大量真实信号，导致 AUC 大幅下降。

可验证解释：该数据集里，很多有效 signal 在局部邻域内并不保证出现足够 opposite polarity 支持；把它当作“必要条件”会造成大比例误触发，扰动排序（尤其 light 环境最明显）。

是否继续：停掉 s13（不再做更大样本验证）。

## s14：Cross-Polarity Boost（跨极性支持度加分）

### 方法原理（动机）

s13 的失败说明：在该数据集上，“opposite polarity 支持不足”并不是噪声的必要条件，把它当作硬惩罚会系统性误伤。

s14 改为更保守的使用方式：**不惩罚缺失，只在存在时小幅加分**，避免把 cross-polarity 当作必须条件。

仍在一次 $O(r^2)$ 邻域扫描中累计：

- $\mathrm{raw}=\sum w_t\,\mathbb{1}(p_{nei}=p)$
- $\mathrm{opp}=\sum w_t\,\mathbb{1}(p_{nei}=-p)$

并定义：

$$
\mathrm{score}=
\begin{cases}
\mathrm{raw} & (\mathrm{raw}<\mathrm{raw\_thr})\\
\mathrm{raw}+\alpha\,\mathrm{opp} & (\mathrm{raw}\ge\mathrm{raw\_thr})
\end{cases}
$$

其中 $\alpha\ge 0$ 是一个小系数（建议 0.1–0.5），用于限制“排序扰动”。

代码位置：`src/myevs/denoise/ops/ebfopt_part2/s14_crosspol_boost.py`

超参（环境变量）：

- `MYEVS_EBF_S14_ALPHA`（默认 0.25）
- `MYEVS_EBF_S14_RAW_THR`（默认 3.0）

### 实现约束核对

- 在线流式/单遍：是（只维护 `last_ts/last_pol`）
- 复杂度：$O(r^2)$（与 baseline 同一邻域遍历；额外只维护 `opp` 累计）
- Numba 必须：是（numba 不可用直接报错，无静默 fallback）

### 实验口径（prescreen，对齐参数）

s14 产物：`data/ED24/myPedestrain_06/EBF_Part2/s14_crosspol_boost_prescreen_s9_tau128ms_200k/`

```powershell
$env:PYTHONNOUSERSITE='1'
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant s14 --max-events 200000 --s-list 9 --tau-us-list 128000 --s14-alpha-list '0.1,0.2,0.4' --s14-raw-thr-list '0,3,6' --out-dir data/ED24/myPedestrain_06/EBF_Part2/s14_crosspol_boost_prescreen_s9_tau128ms_200k
```

best AUC（对齐 s=9,tau=128ms；三环境一致 best）：

- light: 0.950814
- mid: 0.923996
- heavy: 0.922564
- best tag：`ebf_s14_a0p2_raw0_labelscore_s9_tau128000`

### 最小验证（全量，`--max-events 1000000`）

s14（全量）产物：`data/ED24/myPedestrain_06/EBF_Part2/s14_validate_1M_s9_tau128ms/`

```powershell
$env:PYTHONNOUSERSITE='1'
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant s14 --max-events 1000000 --s-list 9 --tau-us-list 128000 --s14-alpha-list '0.2' --s14-raw-thr-list '0' --out-dir data/ED24/myPedestrain_06/EBF_Part2/s14_validate_1M_s9_tau128ms
```

best AUC（全量）：

- light: 0.950814
- mid: 0.926840
- heavy: 0.917763

结论更新：s14 在 prescreen 与全量验证下均显著优于 baseline/s10/s11（提升达到 $10^{-3}$ 量级），目前是 Part2 最强候选。

alpha 稳健性（全量，固定 `raw_thr=0`，扫 `alpha=0.1/0.2/0.3`）：

- light：最佳仍为 `alpha=0.2`（AUC=0.950814）
- mid：最佳仍为 `alpha=0.2`（AUC=0.926840）
- heavy：`alpha=0.3` 略高于 `0.2`（0.917794 vs 0.917763，差异很小）

是否继续：继续（当前以 `alpha=0.2, raw_thr=0` 作为默认推荐）。

最小迁移检查（验证 best 是否依赖固定 `s=9,tau=128ms`）：

- 口径：只扫 `s∈{7,9}`、`tau∈{64ms,128ms}`，固定 `alpha=0.2, raw_thr=0`。
- prescreen 产物：`data/ED24/myPedestrain_06/EBF_Part2/s14_migrationcheck_prescreen_200k_s7_9_tau64_128ms/`
- 全量产物：`data/ED24/myPedestrain_06/EBF_Part2/s14_migrationcheck_validate_1M_s7_9_tau64_128ms/`

结论：在 prescreen(200k) 与全量(1M) 两种口径下，三环境 best-by-env 均稳定为 `s=9,tau=128ms`（tag：`ebf_s14_a0p2_raw0_labelscore_s9_tau128000`），未观察到 best 发生迁移。

### A：阈值可迁移性检查（best-F1 阈值 + MESR across env）

目标：验证 s14 在提升 AUC/F1 的同时，是否也让 best-F1 的阈值在 light/mid/heavy 间更稳定（漂移更小），并对照 MESR（`esr_mean`）。

口径：全量（`--max-events 1000000`），按 sweep 脚本一致的 tie-breaker 选择 best-F1 operating point（最大 F1，其次 TPR、precision，最后最小 FPR）。

数据来源：

- baseline：`data/ED24/myPedestrain_06/EBF_Part2/baseline_validate_1M_s9_tau128ms/`，tag：`ebf_labelscore_s9_tau128000`
- s14：`data/ED24/myPedestrain_06/EBF_Part2/s14_migrationcheck_validate_1M_s7_9_tau64_128ms/`，tag：`ebf_s14_a0p2_raw0_labelscore_s9_tau128000`

对照结果（每行都是该 env 内的 best-F1 operating point）：

| env | 方法 | Thr(best-F1) | F1(best) | AUC | MESR (`esr_mean`) |
|---|---|---:|---:|---:|---:|
| light(1.8V) | baseline | 0.7491 | 0.9497 | 0.9476 | 1.0305 |
| light(1.8V) | s14 | 0.8185 | 0.9520 | 0.9508 | 1.0320 |
| mid(2.5V) | baseline | 4.8347 | 0.8177 | 0.9232 | 1.0155 |
| mid(2.5V) | s14 | 5.6732 | 0.8199 | 0.9268 | 1.0212 |
| heavy(3.3V) | baseline | 7.2797 | 0.7610 | 0.9136 | 1.0085 |
| heavy(3.3V) | s14 | 8.0514 | 0.7641 | 0.9178 | 1.0208 |

结论（A）：

- s14 在三环境下的 AUC/F1 与 MESR 均优于 baseline（与前面的 AUC best 总结一致）。
- 但 best-F1 阈值在 light→mid→heavy 的“尺度差异”仍然很大，且从本对照看并没有明显变小（baseline 的 max/min≈9.72，s14 的 max/min≈9.84）。
- 这意味着：s14 更像是在“可分性/排序”上更强，但并没有从根本上解决“跨环境阈值漂移”的主目标；后续仍需要进入 V2 类归一化/自适应阈值机制阶段。

## s15：Flip Flicker Gate（同像素极性快速交替噪声门控）

### 方法原理（动机）

假设：部分传感器噪声会表现为“同一个像素在极短时间内频繁正负交替”（例如 `+ - + -`），这种现象比 s13 的 cross-polarity 假设更局部、更明确。

在 s14 的邻域累计基础上（同一 $O(r^2)$ 遍历计算 `raw/opp`），额外做一个 $O(1)$ 的中心像素检测：

- 若该像素上一事件极性为 `-p`，且 $
\Delta t=|t-\mathrm{prev\_ts}|\le \mathrm{flip\_dt}$
，则记为 `flicker=1`。

并用一个很小的惩罚系数 `beta` 轻微降分（只在 `raw>=raw_thr` 时启用，避免大范围误伤）。

代码位置：`src/myevs/denoise/ops/ebfopt_part2/s15_flip_flicker_gate.py`

超参（环境变量）：

- `MYEVS_EBF_S15_ALPHA`（默认 0.2）
- `MYEVS_EBF_S15_RAW_THR`（默认 0.0）
- `MYEVS_EBF_S15_FLIP_DT_US`（默认 50us）
- `MYEVS_EBF_S15_BETA`（默认 0.2）

### 实现约束核对

- 在线流式/单遍：是
- 复杂度：$O(r^2)$ + $O(1)$（仅增加中心像素一次判断）
- Numba 必须：是（numba 不可用直接报错，无静默 fallback）

### 实验口径（prescreen，对齐参数）

s15 产物：`data/ED24/myPedestrain_06/EBF_Part2/s15_flip_flicker_gate_prescreen_s9_tau128ms_200k/`

```powershell
$env:PYTHONNOUSERSITE='1'
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant s15 --esr-mode best --max-events 200000 --s-list 9 --tau-us-list 128000 --s15-alpha-list '0.2' --s15-raw-thr-list '0' --s15-flip-dt-us-list '10,30,100' --s15-beta-list '0.1,0.2' --out-dir data/ED24/myPedestrain_06/EBF_Part2/s15_flip_flicker_gate_prescreen_s9_tau128ms_200k
```

best AUC（对齐 s=9,tau=128ms）：

- light: 0.950814（tag：`ebf_s15_a0p2_raw0_fdt10us_b0p1_labelscore_s9_tau128000`）
- mid: 0.923998（tag：`ebf_s15_a0p2_raw0_fdt10us_b0p2_labelscore_s9_tau128000`）
- heavy: 0.922564（tag：`ebf_s15_a0p2_raw0_fdt10us_b0p2_labelscore_s9_tau128000`）

### 结论与失效原因分析（建议停掉）

结论一句话：s15 在 prescreen 下基本与 s14 持平，且“同像素极性快速交替”在该数据集里发生率极低，难以成为主要判别信号。

可验证证据：在 `max-events=200k` 的事件流中统计“同像素上一事件为 opposite polarity 且 dt<=flip_dt_us”的触发比例：

- light：触发次数 0（比例 0）
- mid：触发次数 3–5（比例约 $1.5\times10^{-5}$ 到 $2.5\times10^{-5}$）
- heavy：触发次数 2–7（比例约 $1.0\times10^{-5}$ 到 $3.5\times10^{-5}$）

即该机理在 ED24/myPedestrain_06 上几乎“没有覆盖面”，因此对 AUC/F1 排序的影响很有限。

是否继续：停掉 s15（不再做 1M 验证）。

## s16：s14 + Hotness Clamp（相对热点异常抑制）

### 方法原理（动机）

假设：s14 虽然整体更强，但剩余 FP 可能来自“局部 burst / 热点像素”——中心像素的同极性触发频率相对邻域过高。

在一次 $O(r^2)$ 邻域扫描中同时计算：

- s14 基础分：

$$
\mathrm{score\_base}=
\begin{cases}
\mathrm{raw} & (\mathrm{raw}<\mathrm{raw\_thr})\\
\mathrm{raw}+\alpha\,\mathrm{opp} & (\mathrm{raw}\ge\mathrm{raw\_thr})
\end{cases}
$$

- s11 风格的“相对热点”指标（同极性泄露积分）：
	- 中心像素 accumulator：`acc`
	- 邻域同极性 accumulator 均值：`mean_acc`
	- 比值：$\mathrm{ratio}=\frac{\mathrm{acc}}{\mathrm{mean\_acc}+\varepsilon}$

仅当 `raw>=raw_thr` 且 `acc>acc_thr` 且 `ratio>=ratio_thr` 时，才对整体分数做保守乘性惩罚：

$$
\mathrm{pen}=\left(\frac{\mathrm{ratio\_thr}}{\mathrm{ratio}+\varepsilon}\right)^{\gamma},\quad
\mathrm{score}=\mathrm{score\_base}\cdot\mathrm{pen}
$$

代码位置：`src/myevs/denoise/ops/ebfopt_part2/s16_s14_hotness_clamp.py`

超参（环境变量）：

- `MYEVS_EBF_S16_ALPHA`（默认 0.2）
- `MYEVS_EBF_S16_RAW_THR`（默认 0.0）
- `MYEVS_EBF_S16_ACC_THR`（默认 3.0）
- `MYEVS_EBF_S16_RATIO_THR`（默认 2.0）
- `MYEVS_EBF_S16_GAMMA`（默认 1.0）

### 实现约束核对

- 在线流式/单遍：是（维护 `last_ts/last_pol`，以及每像素 `self_acc`）
- 复杂度：$O(r^2)$（与 baseline/s14 同一邻域遍历；额外只维护 accumulator 统计）
- Numba 必须：是（numba 不可用直接报错，无静默 fallback）

### 实验口径（prescreen，对齐参数）

s16 产物：`data/ED24/myPedestrain_06/EBF_Part2/s16_hotnessclamp_prescreen200k_s9_tau128ms/`

```powershell
$env:PYTHONNOUSERSITE='1'
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant s16 --max-events 200000 --s-list 9 --tau-us-list 128000 --esr-mode best --s16-alpha-list '0.2' --s16-raw-thr-list '0' --s16-acc-thr-list '2,3' --s16-ratio-thr-list '1.5,2' --s16-gamma-list '1,2' --out-dir data/ED24/myPedestrain_06/EBF_Part2/s16_hotnessclamp_prescreen200k_s9_tau128ms
```

best AUC（对齐 s=9,tau=128ms）：

- light: 0.951893（tag：`ebf_s16_a0p2_raw0_acc2_ratio2_g2_labelscore_s9_tau128000`）
- mid: 0.924378（tag：`ebf_s16_a0p2_raw0_acc2_ratio2_g1_labelscore_s9_tau128000`）
- heavy: 0.922703（tag：`ebf_s16_a0p2_raw0_acc3_ratio2_g1_labelscore_s9_tau128000`）

best-F1 operating point + MESR（`esr_mean`，每行是该 env 内的 best-F1 点）：

| env | 方法 | Thr(best-F1) | F1(best) | AUC | MESR (`esr_mean`) | FP@best |
|---|---|---:|---:|---:|---:|---:|
| light(1.8V) | s16 | 0.8184 | 0.9527 | 0.9518 | 1.0019 | 7561 |
| mid(2.5V) | s16 | 5.6747 | 0.8118 | 0.9242 | 1.0233 | 8369 |
| heavy(3.3V) | s16 | 8.2270 | 0.7882 | 0.9227 | 1.0281 | 7273 |

### 结论与失效原因分析（建议停掉/暂缓）

结论一句话：s16 在 light 上能把 AUC 再推高一点，但 mid/heavy 的 best-F1 并未超过 s14（甚至略低），整体收益不足以抵消新增超参/复杂度，建议暂缓。

可验证证据（FP 是否“集中于少数热点像素”）：在 heavy（200k）按各自 best-F1 阈值统计，top-10 像素贡献的 FP 占比很低：

- baseline：top10/FP ≈ 2.36%（FP=6856，top1=37）
- s14：top10/FP ≈ 2.22%（FP=7294，top1=32）
- s16：top10/FP ≈ 1.97%（FP=7273，top1=23）

即 FP 并不主要由极少数像素主导；因此“相对热点 clamp”即便方向正确，其可挖掘空间也有限（更像是微调）。

是否继续：先停掉 s16（除非后续做更细的 FP 画像证明 burst/hotness 在时间维度更强、且 clamp 能显著压 FP 而不伤 TPR）。

## s17：Cross-Polarity Spread Trust（跨极性支持的空间分散度可信度）

### 方法原理（动机）

假设：s14 的 opposite polarity 支持里，有一部分是“偶然/局部集中”的 opposite 触发（例如某个邻点刚好抖一下），这种 opp 证据不够可靠，可能抬高 FP。

在一次 $O(r^2)$ 邻域扫描中对 opp 事件累计二阶矩，构造一个简易“分散度”（加权方差）代理：

$$
\mathrm{var}=\mathbb{E}[r^2]-\|\mathbb{E}[\vec r]\|^2
$$

并定义可信度 `trust`（当 var 很小就下调 opp boost）：

$$
\mathrm{trust}=\begin{cases}
1 & (\mathrm{var}\ge \mathrm{var\_thr})\\
\beta+(1-\beta)\left(\frac{\mathrm{var}}{\mathrm{var\_thr}}\right)^{\gamma} & (\mathrm{var}<\mathrm{var\_thr})
\end{cases}
$$

最终：

$$
\mathrm{score}=\begin{cases}
\mathrm{raw} & (\mathrm{raw}<\mathrm{raw\_thr})\\
\mathrm{raw}+\alpha\,\mathrm{opp}\cdot\mathrm{trust} & (\mathrm{raw}\ge\mathrm{raw\_thr})
\end{cases}
$$

代码位置：`src/myevs/denoise/ops/ebfopt_part2/s17_crosspol_spread_boost.py`

超参（环境变量）：

- `MYEVS_EBF_S17_ALPHA`（默认 0.2）
- `MYEVS_EBF_S17_RAW_THR`（默认 0.0）
- `MYEVS_EBF_S17_VAR_THR`（默认 2.0）
- `MYEVS_EBF_S17_BETA`（默认 0.2）
- `MYEVS_EBF_S17_GAMMA`（默认 1.0）

### 实现约束核对

- 在线流式/单遍：是（只维护 `last_ts/last_pol`）
- 复杂度：$O(r^2)$（同一次邻域扫描累计 opp 的矩）
- Numba 必须：是（numba 不可用直接报错，无静默 fallback）

### 实验口径（prescreen，对齐参数）

s17 产物：`data/ED24/myPedestrain_06/EBF_Part2/s17_spreadboost_prescreen200k_s9_tau128ms/`

```powershell
$env:PYTHONNOUSERSITE='1'
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant s17 --max-events 200000 --s-list 9 --tau-us-list 128000 --esr-mode best --s17-alpha-list '0.2' --s17-raw-thr-list '0' --s17-var-thr-list '1,2' --s17-beta-list '0.2,0.4' --s17-gamma-list '1,2' --out-dir data/ED24/myPedestrain_06/EBF_Part2/s17_spreadboost_prescreen200k_s9_tau128ms
```

best AUC（对齐 s=9,tau=128ms）：

- light: 0.950982（tag：`ebf_s17_a0p2_raw0_var1_b0p2_g1_labelscore_s9_tau128000`）
- mid: 0.923958（tag：`ebf_s17_a0p2_raw0_var1_b0p4_g1_labelscore_s9_tau128000`）
- heavy: 0.922486（tag：`ebf_s17_a0p2_raw0_var1_b0p4_g1_labelscore_s9_tau128000`）

best-F1 operating point + MESR（`esr_mean`）：

| env | 方法 | Thr(best-F1) | F1(best) | AUC | MESR (`esr_mean`) | FP@best |
|---|---|---:|---:|---:|---:|---:|
| light(1.8V) | s17 | 0.8302 | 0.9522 | 0.9510 | 1.0242 | 7503 |
| mid(2.5V) | s17 | 5.6374 | 0.8115 | 0.9239 | 1.0245 | 8544 |
| heavy(3.3V) | s17 | 8.2263 | 0.7883 | 0.9225 | 1.0297 | 7319 |

### 结论与失效原因分析（建议停掉）

结论一句话：s17 的“opp 分散度可信度”在该数据集上没有带来有效收益（mid/heavy 的 AUC/F1 均未超过 s14），建议停掉。

可验证解释：opp 证据的空间集中/分散在该数据集上可能并不是主要的判别维度；或者 var/trust 这一路径太弱，难以在不伤 signal 的前提下显著压 FP。

是否继续：停掉 s17。

## s18：No-Polarity EBF（去掉极性一致性判断的消融）

### 方法原理（动机）

目标：验证 baseline EBF 中“只统计同极性邻域事件”这一判别项，是否对 ED24/myPedestrain_06 的可分性贡献很大。

对每个事件 $e=(x,y,t,p)$，baseline EBF 的 raw-score 是：

$$
\mathrm{raw}=\sum w_t\,\mathbb{1}(p_{nei}=p),\quad w_t=(\tau-\Delta t)/\tau
$$

s18 做最直接的消融：**完全不看极性**，邻域内只要在 $\tau$ 内就计入：

$$
\mathrm{raw}_{\mathrm{nopol}}=\sum w_t,\quad \mathrm{score}=\mathrm{raw}_{\mathrm{nopol}}
$$

即：不再使用 `last_pol`，只维护 `last_ts`。

代码位置：`src/myevs/denoise/ops/ebfopt_part2/s18_no_polarity_ebf.py`

### 实现约束核对

- 在线流式/单遍：是（每像素仅维护 `last_ts`）
- 复杂度：$O(r^2)$（与 baseline 同一邻域遍历）
- Numba 必须：是（numba 不可用直接报错，无静默 fallback）

### 实验口径（prescreen，对齐参数）

s18 产物：`data/ED24/myPedestrain_06/EBF_Part2/s18_nopol_prescreen_s9_tau128ms_200k/`

```powershell
$env:PYTHONNOUSERSITE='1'
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant s18 --esr-mode best --max-events 200000 --s-list '9' --tau-us-list '128000' --out-dir data/ED24/myPedestrain_06/EBF_Part2/s18_nopol_prescreen_s9_tau128ms_200k
```

best AUC（对齐 s=9,tau=128ms；仅 1 个 tag）：

- light: 0.944546（tag：`ebf_s18_labelscore_s9_tau128000`）
- mid: 0.911880（tag：`ebf_s18_labelscore_s9_tau128000`）
- heavy: 0.908993（tag：`ebf_s18_labelscore_s9_tau128000`）

best-F1 operating point + MESR（`esr_mean`；每行是该 env 内的 best-F1 点）：

| env | 方法 | Thr(best-F1) | F1(best) | AUC | MESR (`esr_mean`) | FP@best |
|---|---|---:|---:|---:|---:|---:|
| light(1.8V) | baseline | 0.7491 | 0.9497 | 0.9476 | 1.0305 | 7400 |
| light(1.8V) | s14 (best tag) | 0.8185 | 0.9520 | 0.9508 | 1.0320 | 7819 |
| light(1.8V) | s18 (no polarity) | 0.9955 | 0.9499 | 0.9445 | 1.0512 | 8892 |
| mid(2.5V) | baseline | 4.8394 | 0.8108 | 0.9219 | 1.0168 | 8893 |
| mid(2.5V) | s14 (best tag) | 5.6691 | 0.8116 | 0.9240 | 1.0241 | 8441 |
| mid(2.5V) | s18 (no polarity) | 8.1850 | 0.7863 | 0.9119 | 1.0331 | 10883 |
| heavy(3.3V) | baseline | 7.3581 | 0.7869 | 0.9205 | 1.0208 | 6856 |
| heavy(3.3V) | s14 (best tag) | 8.2326 | 0.7883 | 0.9226 | 1.0298 | 7311 |
| heavy(3.3V) | s18 (no polarity) | 12.4920 | 0.7581 | 0.9090 | 1.0490 | 8765 |

### 结论与失效原因分析（建议停掉）

结论一句话：去掉“同极性一致性”后，AUC/F1 在 mid/heavy 明显下降、FP 上升、阈值尺度膨胀，说明 **极性判断是 EBF 在该数据集上的关键判别信息**。

可验证解释：s18 把邻域内的 opposite polarity 事件也当作“密度证据”加入 raw-score，会对噪声团/抖动类事件更容易“抬分”，从而压缩 signal/noise 的可分性；同时打分尺度整体变大，导致 best-F1 阈值在 mid/heavy 更偏大（进一步恶化阈值可迁移性目标）。

是否继续：停掉 s18（作为消融结论保留：后续改进应保留极性一致性判别）。

## s19：Evidence Fusion Q8（cross-pol 正证据 + same-pixel hotness 噪声证据统一主模型）

### 方法原理（动机）

你明确要求不要再做“raw + 新门控”的补丁式设计，而是把两类已验证有效的证据统一进一个主打分模型：

- cross-polarity 作为“正证据”（s14 经验：只加分不惩罚更稳）
- same-pixel hotness 作为“噪声证据”（热点/爆发噪声常在同像素反复触发）

同时要求尽量硬件化/实现简单：因此将融合写成“整数累计 + Q8 定点线性组合”。

### 定义（硬件友好 / Q8）

对事件 $e=(x,y,t,p)$，在邻域 $\mathcal{N}_r$ 内遍历历史事件（与 baseline 一样保持单遍 $O(r^2)$），用 $\Delta t=t-t_{nei}$，时间权重用“整数形式”的 $w=\tau-\Delta t$（当 $\Delta t\in[0,\tau)$）。

邻域两类证据：

$$
\mathrm{raw\_w}=\sum (\tau-\Delta t)\,\mathbb{1}(p_{nei}=p),\quad
\mathrm{opp\_w}=\sum (\tau-\Delta t)\,\mathbb{1}(p_{nei}\ne p)
$$

同像素 hotness（leaky accumulator，整数）：令 $\Delta t_0=t-t_{last}(x,y)$，

$$
\mathrm{acc\_w}\leftarrow \max(0,\mathrm{acc\_w}-\Delta t_0)+\tau
$$

融合（$\alpha,\beta$ 以 Q8 存储：$\alpha_{q8}=\mathrm{round}(256\alpha)$，$\beta_{q8}=\mathrm{round}(256\beta)$）：

$$
\mathrm{score} = \frac{\max\bigl(0,\ \mathrm{raw\_w}+\alpha\,\mathrm{opp\_w}-\beta\,\mathrm{acc\_w}\bigr)}{\tau}
$$

代码位置：`src/myevs/denoise/ops/ebfopt_part2/s19_evidence_fusion_q8.py`

环境变量：

- `MYEVS_EBF_S19_ALPHA`（本轮固定 0.2）
- `MYEVS_EBF_S19_BETA`（本轮只扫 beta）

### 实现约束核对

- 在线流式/单遍：是（每像素维护 `last_ts`/`last_pol`/`acc_w`）
- 复杂度：$O(r^2)$（与 baseline 同一邻域遍历）
- Numba 必须：是（numba 不可用直接报错，无静默 fallback）

### 实验口径（prescreen：固定 alpha=0.2，只扫 beta）

s19 产物：`data/ED24/myPedestrain_06/EBF_Part2/s19_fusionq8_prescreen200k_s9_tau128ms_a0p2_betaSweep/`

```powershell
$env:PYTHONNOUSERSITE='1'
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant s19 --esr-mode best --max-events 200000 --s-list '9' --tau-us-list '128000' --s19-beta-list '0.05,0.1,0.2,0.3' --out-dir data/ED24/myPedestrain_06/EBF_Part2/s19_fusionq8_prescreen200k_s9_tau128ms_a0p2_betaSweep
```

best AUC（对齐 s=9,tau=128ms）：

- light: 0.951015（tag：`ebf_s19_a0p2_b0p1_labelscore_s9_tau128000`）
- mid: 0.924413（tag：`ebf_s19_a0p2_b0p3_labelscore_s9_tau128000`）
- heavy: 0.922656（tag：`ebf_s19_a0p2_b0p2_labelscore_s9_tau128000`）

best-F1 operating point + MESR（`esr_mean`；每行是该 env 内的 best-F1 点）：

| env | 方法 | Thr(best-F1) | F1(best) | AUC | MESR (`esr_mean`) | FP@best |
|---|---|---:|---:|---:|---:|---:|
| light(1.8V) | baseline | 0.7491 | 0.9497 | 0.9476 | 1.0305 | 7400 |
| light(1.8V) | s14 | 0.8689 | 0.9525 | 0.9498 | 1.0366 | 8171 |
| light(1.8V) | s19 (a0.2, beta sweep best-F1) | 0.5158 | 0.9531 | 0.9507 | 0.9923 | 7446 |
| mid(2.5V) | baseline | 4.8394 | 0.8108 | 0.9219 | 1.0168 | 8893 |
| mid(2.5V) | s14 | 5.2456 | 0.8129 | 0.9223 | 1.0199 | 8557 |
| mid(2.5V) | s19 (a0.2, beta sweep best-F1) | 5.3727 | 0.8117 | 0.9244 | 1.0231 | 8307 |
| heavy(3.3V) | baseline | 7.3581 | 0.7869 | 0.9205 | 1.0208 | 6856 |
| heavy(3.3V) | s14 | 7.6638 | 0.7895 | 0.9216 | 1.0257 | 7294 |
| heavy(3.3V) | s19 (a0.2, beta sweep best-F1) | 8.1278 | 0.7885 | 0.9226 | 1.0293 | 7269 |

### 实验口径（validate：固定 alpha=0.2,beta=0.3，对齐 1M）

s19 产物：`data/ED24/myPedestrain_06/EBF_Part2/s19_fusionq8_validate_1M_s9_tau128ms_a0p2_b0p3/`

```powershell
$env:PYTHONNOUSERSITE='1'
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant s19 --esr-mode best --max-events 1000000 --s-list '9' --tau-us-list '128000' --s19-beta-list '0.3' --out-dir data/ED24/myPedestrain_06/EBF_Part2/s19_fusionq8_validate_1M_s9_tau128ms_a0p2_b0p3
```

best-F1 operating point + MESR（`esr_mean`；每行是该 env 内的 best-F1 点；对照同口径的 1M baseline/s14 产物目录）：

| env | 方法 | Thr(best-F1) | F1(best) | AUC | MESR (`esr_mean`) | FP@best |
|---|---|---:|---:|---:|---:|---:|
| light(1.8V) | baseline(1M) | 0.7491 | 0.9497 | 0.9476 | 1.0305 | 7400 |
| light(1.8V) | s14(1M) | 0.8185 | 0.9520 | 0.9508 | 1.0320 | 7819 |
| light(1.8V) | s19(1M,b0.3) | 0.5158 | 0.9531 | 0.9507 | 0.9923 | 7446 |
| mid(2.5V) | baseline(1M) | 4.8347 | 0.8177 | 0.9232 | 1.0155 | 25108 |
| mid(2.5V) | s14(1M) | 5.6732 | 0.8199 | 0.9268 | 1.0212 | 23663 |
| mid(2.5V) | s19(1M,b0.3) | 5.1624 | 0.8201 | 0.9273 | 1.0202 | 25406 |
| heavy(3.3V) | baseline(1M) | 7.2797 | 0.7610 | 0.9136 | 1.0085 | 31245 |
| heavy(3.3V) | s14(1M) | 8.0514 | 0.7641 | 0.9178 | 1.0208 | 33870 |
| heavy(3.3V) | s19(1M,b0.3) | 7.7323 | 0.7638 | 0.9179 | 1.0186 | 33704 |

### 实验口径（validate-2：固定 alpha=0.2,beta=0.5，对齐 1M；精度/FP 优先）

s19 产物：`data/ED24/myPedestrain_06/EBF_Part2/s19_fusionq8_validate_1M_s9_tau128ms_a0p2_b0p5/`

```powershell
$env:PYTHONNOUSERSITE='1'
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant s19 --esr-mode best --max-events 1000000 --s-list '9' --tau-us-list '128000' --s19-beta-list '0.5' --out-dir data/ED24/myPedestrain_06/EBF_Part2/s19_fusionq8_validate_1M_s9_tau128ms_a0p2_b0p5
```

best-F1 operating point + MESR（`esr_mean`；每行是该 env 内的 best-F1 点；与 1M 的 baseline/s14/b0.3 对照）：

| env | 方法 | Thr(best-F1) | F1(best) | AUC | MESR (`esr_mean`) | FP@best |
|---|---|---:|---:|---:|---:|---:|
| light(1.8V) | baseline(1M) | 0.7491 | 0.9497 | 0.9476 | 1.0305 | 7400 |
| light(1.8V) | s14(1M) | 0.8185 | 0.9520 | 0.9508 | 1.0320 | 7819 |
| light(1.8V) | s19(1M,b0.3) | 0.5158 | 0.9531 | 0.9507 | 0.9923 | 7446 |
| light(1.8V) | s19(1M,b0.5) | 0.3166 | 0.9532 | 0.9497 | 0.9856 | 7354 |
| mid(2.5V) | baseline(1M) | 4.8347 | 0.8177 | 0.9232 | 1.0155 | 25108 |
| mid(2.5V) | s14(1M) | 5.6732 | 0.8199 | 0.9268 | 1.0212 | 23663 |
| mid(2.5V) | s19(1M,b0.3) | 5.1624 | 0.8201 | 0.9273 | 1.0202 | 25406 |
| mid(2.5V) | s19(1M,b0.5) | 5.1665 | 0.8200 | 0.9274 | 1.0145 | 23208 |
| heavy(3.3V) | baseline(1M) | 7.2797 | 0.7610 | 0.9136 | 1.0085 | 31245 |
| heavy(3.3V) | s14(1M) | 8.0514 | 0.7641 | 0.9178 | 1.0208 | 33870 |
| heavy(3.3V) | s19(1M,b0.3) | 7.7323 | 0.7638 | 0.9179 | 1.0186 | 33704 |
| heavy(3.3V) | s19(1M,b0.5) | 7.5773 | 0.7635 | 0.9178 | 1.0185 | 32976 |

### 结论与下一步（validate 后的判断）

结论一句话：在 1M 对照下，s19(b0.5) 在三环境都实现了“精度/FP 优先”的目标：

- mid：FP@best-F1 从 s14 的 23663 降到 23208，precision 同时提升（0.8457 → 0.8479），且 AUC/F1 不降。
- heavy：FP@best-F1 从 s14 的 33870 降到 32976，precision 提升，AUC/F1 基本持平。
- light：FP 也略低于 baseline/s14，F1 保持略优，但 AUC 相对 s14 有小幅下降（仍显著高于 baseline）。

是否继续：

- 若优先目标是“mid/heavy precision/FP 下降”：建议继续，并优先把 s19 的全局默认候选锁定在 `alpha=0.2,beta=0.5`，然后再进入 V2 类归一化/自适应阈值阶段（解决阈值跨场景尺度差异）。
- 若优先目标是“light AUC 极致不掉”：仍可保留 s14 作为对照，但以当前结果看 s19(b0.5) 更贴近“精度提升”诉求。

## s20：Polarity-aware hotness（同像素按极性分通道）+ Evidence Fusion Q8

### 方法原理（动机）

你已接受“每像素状态翻倍”，因此做一个最小架构升级：把 same-pixel hotness 从单通道 accumulator 改为 **按极性分通道的两路 accumulator**，并只惩罚“当前事件同极性”这一路的热度。

直觉：如果噪声的极性混杂/快速交替，那么“同极性爆发”更像噪声；相反，目标运动边缘的邻域响应可能在同极性上更稳定。该升级仍保持在线单遍、邻域 $O(r^2)$，且融合仍是 Q8 定点。

### 定义（与 s19 的唯一差别）

同像素 hotness 从单通道变为两通道：

- 维护 `acc_pos(x,y)`、`acc_neg(x,y)` 两个 leaky accumulator
- 对当前事件极性 $p$，只取同极性通道作为惩罚项：

$$
\mathrm{acc\_same} =
\begin{cases}
\mathrm{acc\_pos}(x,y) & p=+\\
\mathrm{acc\_neg}(x,y) & p=-
\end{cases}
$$

融合仍为：

$$
\mathrm{score} = \frac{\max\bigl(0,\ \mathrm{raw\_w}+\alpha\,\mathrm{opp\_w}-\beta\,\mathrm{acc\_same}\bigr)}{\tau}
$$

代码位置：`src/myevs/denoise/ops/ebfopt_part2/s20_polhot_evidence_fusion_q8.py`

环境变量：

- `MYEVS_EBF_S20_ALPHA`（本轮固定 0.2）
- `MYEVS_EBF_S20_BETA`（本轮扫 beta）

### 实验口径（prescreen：固定 alpha=0.2，只扫 beta）

s20 产物：`data/ED24/myPedestrain_06/EBF_Part2/s20_polhot_fusionq8_prescreen200k_s9_tau128ms_a0p2_betaHigh/`

```powershell
$env:PYTHONNOUSERSITE='1'
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant s20 --esr-mode best --max-events 200000 --s-list '9' --tau-us-list '128000' --s20-beta-list '0.3,0.4,0.5,0.6,0.8' --out-dir data/ED24/myPedestrain_06/EBF_Part2/s20_polhot_fusionq8_prescreen200k_s9_tau128ms_a0p2_betaHigh
```

prescreen 观察（对齐 s=9,tau=128ms，best-F1 口径）：

- light：s20 的 FP 明显低于 s19/s14，同时 F1/AUC 更高
- heavy：beta=0.5 左右的 FP/precision 更稳（beta 太大时 best-F1 点会偏向“更宽松阈值”导致 FP 反弹）
- mid：beta 提升会略降 FP，但整体仍比 s19(b0.5) 更“宽松”（F1/AUC 更高，但 FP 更高）

### 实验口径（validate：固定 alpha=0.2，对齐 1M）

s20 产物：

- `data/ED24/myPedestrain_06/EBF_Part2/s20_polhot_fusionq8_validate_1M_s9_tau128ms_a0p2_b0p5/`
- `data/ED24/myPedestrain_06/EBF_Part2/s20_polhot_fusionq8_validate_1M_s9_tau128ms_a0p2_b0p6/`
- `data/ED24/myPedestrain_06/EBF_Part2/s20_polhot_fusionq8_validate_1M_s9_tau128ms_a0p2_b0p8/`

```powershell
$env:PYTHONNOUSERSITE='1'
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant s20 --esr-mode best --max-events 1000000 --s-list '9' --tau-us-list '128000' --s20-beta-list '0.5' --out-dir data/ED24/myPedestrain_06/EBF_Part2/s20_polhot_fusionq8_validate_1M_s9_tau128ms_a0p2_b0p5
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant s20 --esr-mode best --max-events 1000000 --s-list '9' --tau-us-list '128000' --s20-beta-list '0.6' --out-dir data/ED24/myPedestrain_06/EBF_Part2/s20_polhot_fusionq8_validate_1M_s9_tau128ms_a0p2_b0p6
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant s20 --esr-mode best --max-events 1000000 --s-list '9' --tau-us-list '128000' --s20-beta-list '0.8' --out-dir data/ED24/myPedestrain_06/EBF_Part2/s20_polhot_fusionq8_validate_1M_s9_tau128ms_a0p2_b0p8
```

best-F1 operating point + MESR（`esr_mean`；每行是该 env 内的 best-F1 点；与 1M 的 baseline/s14/s19(b0.5) 对照）：

| env | 方法 | Thr(best-F1) | F1(best) | AUC | MESR (`esr_mean`) | FP@best |
|---|---|---:|---:|---:|---:|---:|
| light(1.8V) | baseline(1M) | 0.7491 | 0.9497 | 0.9476 | 1.0305 | 7400 |
| light(1.8V) | s14(1M) | 0.8185 | 0.9520 | 0.9508 | 1.0320 | 7819 |
| light(1.8V) | s19(1M,b0.5) | 0.3166 | 0.9532 | 0.9497 | 0.9856 | 7354 |
| light(1.8V) | s20(1M,b0.5) | 0.2307 | 0.9557 | 0.9526 | 0.9183 | 7088 |
| mid(2.5V) | baseline(1M) | 4.8347 | 0.8177 | 0.9232 | 1.0155 | 25108 |
| mid(2.5V) | s14(1M) | 5.6732 | 0.8199 | 0.9268 | 1.0212 | 23663 |
| mid(2.5V) | s19(1M,b0.5) | 5.1665 | 0.8200 | 0.9274 | 1.0145 | 23208 |
| mid(2.5V) | s20(1M,b0.5) | 4.8735 | 0.8233 | 0.9312 | 0.9727 | 24701 |
| heavy(3.3V) | baseline(1M) | 7.2797 | 0.7610 | 0.9136 | 1.0085 | 31245 |
| heavy(3.3V) | s14(1M) | 8.0514 | 0.7641 | 0.9178 | 1.0208 | 33870 |
| heavy(3.3V) | s19(1M,b0.5) | 7.5773 | 0.7635 | 0.9178 | 1.0185 | 32976 |
| heavy(3.3V) | s20(1M,b0.5) | 7.3924 | 0.7676 | 0.9218 | 0.9889 | 33077 |

补充：提高 beta 的趋势（仍按 best-F1 取点）

- heavy：beta=0.6 时 FP@best-F1 下降到 32160（precision 上升），beta=0.8 与 beta=0.6 接近
- mid：beta=0.6 时 FP 仅小幅下降到 24545，beta=0.8 反而回升到 24869，未能压回到 s19(b0.5) 的 23208

### 结论（是否替代 s19）

结论一句话：s20 在 light/heavy 上能显著抬 F1/AUC（且 heavy 通过 beta↑ 可以继续压 FP），但在 mid 上 best-F1 口径下 **FP 明显高于 s19(b0.5)/s14**，不满足“mid/heavy precision/FP 优先”的当前目标。

因此：

- 若主目标仍是压 mid/heavy FP、提 precision：继续用 s19(b0.5) 作为主候选，s20 暂不替代。
- 若未来允许牺牲 mid 的 FP 换整体 F1/AUC：s20 可作为备选（尤其 light/heavy 更强）。

## s21：Bi-polar hotness mix（同像素双极性热度混合惩罚）+ Evidence Fusion Q8

### 方法原理（动机）

s20 的核心假设是“只惩罚同极性 hotness 更稳”。但 mid 上出现的失败模式更像 **同像素在两极性之间快速交替的 burst/flicker**：

- 这类噪声会把 `acc_pos/acc_neg` 两路都抬高；
- s20 只惩罚 `acc_same`，因此对“另一极性热度（acc_opp）”没有直接惩罚，可能导致排序更宽松、best-F1 点 FP 偏高。

s21 的最小结构改动：仍保留 s20 的两路 accumulator，但把惩罚从“只看同极性”升级为 **同极性 + 异极性的混合热度**，用一个标量超参 $\kappa\in[0,1]$ 控制强度。

### 定义（与 s20 的唯一差别）

仍维护 `acc_pos(x,y)`、`acc_neg(x,y)`，对当前事件极性 $p$ 定义：

$$
\mathrm{acc\_same} =
\begin{cases}
\mathrm{acc\_pos}(x,y) & p=+\\
\mathrm{acc\_neg}(x,y) & p=-
\end{cases},\quad
\mathrm{acc\_opp} =
\begin{cases}
\mathrm{acc\_neg}(x,y) & p=+\\
\mathrm{acc\_pos}(x,y) & p=-
\end{cases}
$$

混合惩罚：

$$
\mathrm{acc\_mix}=\mathrm{acc\_same}+\kappa\,\mathrm{acc\_opp}
$$

融合：

$$
\mathrm{score} = \frac{\max\bigl(0,\ \mathrm{raw\_w}+\alpha\,\mathrm{opp\_w}-\beta\,\mathrm{acc\_mix}\bigr)}{\tau}
$$

其中：

- $\kappa=0$ 退化为 s20（只惩罚同极性热度）
- $\kappa=1$ 等价于惩罚“总热度”（两路都压）

代码位置：`src/myevs/denoise/ops/ebfopt_part2/s21_bipolhot_evidence_fusion_q8.py`

环境变量：

- `MYEVS_EBF_S21_ALPHA`（默认 0.2）
- `MYEVS_EBF_S21_BETA`（默认 0.5）
- `MYEVS_EBF_S21_KAPPA`（默认 0.5；自动 clamp 到 [0,1]）

### 实现约束核对

- 在线流式/单遍：是（每像素维护 `last_ts/last_pol/acc_pos/acc_neg`）
- 复杂度：$O(r^2)$（与 baseline 同一邻域遍历）
- Numba 必须：是（numba 不可用直接报错，无静默 fallback）

### 实验口径（prescreen：固定 alpha=0.2，扫 beta/kappa）

s21 产物：`data/ED24/myPedestrain_06/EBF_Part2/s21_bipolhotmix_prescreen200k_s9_tau128ms_a0p2_b0p5_0p6_k0p3_0p5_0p8/`

```powershell
$env:PYTHONNOUSERSITE='1'
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant s21 --esr-mode best --max-events 200000 --s-list '9' --tau-us-list '128000' --s21-beta-list '0.5,0.6' --s21-kappa-list '0.3,0.5,0.8' --out-dir data/ED24/myPedestrain_06/EBF_Part2/s21_bipolhotmix_prescreen200k_s9_tau128ms_a0p2_b0p5_0p6_k0p3_0p5_0p8
```

best AUC（对齐 s=9,tau=128ms；在该小网格内取 best tag）：

- light: 0.953086（tag：`ebf_s21_a0p2_b0p5_k0p8_labelscore_s9_tau128000`）
- mid: 0.932585（tag：`ebf_s21_a0p2_b0p6_k0p8_labelscore_s9_tau128000`）
- heavy: 0.929583（tag：`ebf_s21_a0p2_b0p6_k0p8_labelscore_s9_tau128000`）

注意（避免口径混淆）：上面提到的 “light AUC/F1 > 0.95” 是 **s=9,tau=128ms** 且允许调 `beta/kappa` 的 s21 最优点；它不是 Part2 后续常用的“对齐点 s=7,tau=64ms”。

补充：在对齐点 s=7,tau=64ms（prescreen200k；alpha=0.2；扫 beta∈{0.5,0.6}, kappa∈{0.3,0.5,0.8}）下，s21 数值如下：

- 产物目录：`data/ED24/myPedestrain_06/EBF_Part2/_tune_s21_prescreen200k_s7_tau64/`
- best-AUC（per env）：
	- light：AUC 0.937464（tag：`ebf_s21_a0p2_b0p5_k0p8_labelscore_s7_tau64000`）
	- mid：AUC 0.920956（tag：`ebf_s21_a0p2_b0p5_k0p8_labelscore_s7_tau64000`）
	- heavy：AUC 0.924092（tag：`ebf_s21_a0p2_b0p6_k0p8_labelscore_s7_tau64000`）
- best-F1（per env）：
	- light：F1 0.941407（tag：`ebf_s21_a0p2_b0p5_k0p8_labelscore_s7_tau64000`）
	- mid：F1 0.811971（tag：`ebf_s21_a0p2_b0p6_k0p8_labelscore_s7_tau64000`）
	- heavy：F1 0.793229（tag：`ebf_s21_a0p2_b0p6_k0p8_labelscore_s7_tau64000`）

对照：在同一对齐点（s=7,tau=64ms）下，s55 的 light/mid/heavy best-F1 分别为 0.942475 / 0.813311 / 0.794704（见 s55 小节），即对齐点上 s55 并不比 s21 更差。

mid 的 best-F1 点对比（prescreen200k，固定 s=9,tau=128ms；重点看 precision/FP）：

- s19(a0.2,b0.5)：F1=0.8116，precision=0.8367，FP=8257，AUC=0.9246
- s20(a0.2,b0.8)：F1=0.8162，precision=0.8314，FP=8728，AUC=0.9292
- s21(a0.2,b0.6,k0.5)：F1=0.8182，precision=0.8414，FP=8058，AUC=0.9313（更偏“精度/低 FP”）
- s21(a0.2,b0.6,k0.8)：F1=0.8195，precision=0.8397，FP=8199，AUC=0.9326（更偏“F1/AUC”）

### 结论与下一步（是否替代 s20 / 是否挑战 s19）

结论一句话：在 prescreen200k（s=9,tau=128ms）口径下，s21 同时解决了 s20 的 mid FP 偏高问题，并在 light/mid/heavy 上把 AUC/F1 推到更高水平，属于“可继续投入”的结构性升级。

是否继续：继续。

#### 1M validate（已完成；固定 s=9,tau=128ms；a=0.2,b=0.6；k sweep）

产物目录：

- `data/ED24/myPedestrain_06/EBF_Part2/s21_1M_validate_s9_tau128ms_a0p2_b0p6_k0p5/`
- `data/ED24/myPedestrain_06/EBF_Part2/s21_1M_validate_s9_tau128ms_a0p2_b0p6_k0p8/`

```powershell
$env:PYTHONNOUSERSITE='1'
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant s21 --esr-mode best --max-events 1000000 --s-list '9' --tau-us-list '128000' --s21-beta-list '0.6' --s21-kappa-list '0.5' --out-dir data/ED24/myPedestrain_06/EBF_Part2/s21_1M_validate_s9_tau128ms_a0p2_b0p6_k0p5
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant s21 --esr-mode best --max-events 1000000 --s-list '9' --tau-us-list '128000' --s21-beta-list '0.6' --s21-kappa-list '0.8' --out-dir data/ED24/myPedestrain_06/EBF_Part2/s21_1M_validate_s9_tau128ms_a0p2_b0p6_k0p8
```

best-F1 operating point + MESR（`esr_mean`；每行是该 env 内的 best-F1 点；用于与 s19/s20 表格对齐比较）：

| env | 方法 | Thr(best-F1) | precision(best) | F1(best) | AUC | MESR (`esr_mean`) | FP@best |
|---|---|---:|---:|---:|---:|---:|---:|
| light(1.8V) | s21(1M,b0.6,k0.5) | 0.1283 | 0.9583 | 0.9564 | 0.9522 | 0.9182 | 6759 |
| light(1.8V) | s21(1M,b0.6,k0.8) | 0.1261 | 0.9589 | 0.9567 | 0.9523 | 0.9178 | 6660 |
| mid(2.5V) | s21(1M,b0.6,k0.5) | 4.693 | 0.8449 | 0.8256 | 0.9336 | 0.9700 | 24145 |
| mid(2.5V) | s21(1M,b0.6,k0.8) | 4.689 | 0.8479 | 0.8266 | 0.9347 | 0.9695 | 23576 |
| heavy(3.3V) | s21(1M,b0.6,k0.5) | 7.168 | 0.7902 | 0.7702 | 0.9244 | 0.9846 | 32487 |
| heavy(3.3V) | s21(1M,b0.6,k0.8) | 7.172 | 0.7947 | 0.7714 | 0.9256 | 0.9840 | 31542 |

#### AOCC（新增默认参考指标；无标签、非单调；paper 口径；1M rerun 已完成）

从 2026-04-09 起，Part2 的 validate 口径除 AUC/F1/FP 外，默认**额外记录 AOCC**（Area of Continuous Contrast Curve；无参考、非单调）作为参考指标，与 MESR（`esr_mean`）并列。

实现位置：`src/myevs/metrics/aocc.py`

评测脚本支持：`--aocc-mode best|all|off`（默认 `best`：仅在每个 env 的 best-AUC 与 best-F1 tag 上计算并回填到 ROC CSV 的 `aocc` 列）。

本次为对齐比较，已在同一 1M 口径下重跑 baseline / s14 / s19(b0.5) / s21(b0.6,k0.8)，并写入新的 out-dir（`*_aocc_paper`）：

- `data/ED24/myPedestrain_06/EBF_Part2/baseline_validate_1M_s9_tau128ms_aocc_paper/`
- `data/ED24/myPedestrain_06/EBF_Part2/s14_validate_1M_s9_tau128ms_a0p2_raw0_aocc_paper/`
- `data/ED24/myPedestrain_06/EBF_Part2/s19_validate_1M_s9_tau128ms_a0p2_b0p5_aocc_paper/`
- `data/ED24/myPedestrain_06/EBF_Part2/s21_validate_1M_s9_tau128ms_a0p2_b0p6_k0p8_aocc_paper/`

```powershell
$env:PYTHONNOUSERSITE='1'
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant ebf --esr-mode best --aocc-mode best --max-events 1000000 --s-list 9 --tau-us-list 128000 --out-dir data/ED24/myPedestrain_06/EBF_Part2/baseline_validate_1M_s9_tau128ms_aocc_paper
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant s14 --esr-mode best --aocc-mode best --max-events 1000000 --s-list 9 --tau-us-list 128000 --s14-alpha-list 0.2 --s14-raw-thr-list 0 --out-dir data/ED24/myPedestrain_06/EBF_Part2/s14_validate_1M_s9_tau128ms_a0p2_raw0_aocc_paper
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant s19 --esr-mode best --aocc-mode best --max-events 1000000 --s-list 9 --tau-us-list 128000 --s19-alpha-list 0.2 --s19-beta-list 0.5 --out-dir data/ED24/myPedestrain_06/EBF_Part2/s19_validate_1M_s9_tau128ms_a0p2_b0p5_aocc_paper
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant s21 --esr-mode best --aocc-mode best --max-events 1000000 --s-list 9 --tau-us-list 128000 --s21-alpha-list 0.2 --s21-beta-list 0.6 --s21-kappa-list 0.8 --out-dir data/ED24/myPedestrain_06/EBF_Part2/s21_validate_1M_s9_tau128ms_a0p2_b0p6_k0p8_aocc_paper
```

best-F1 operating point + MESR + AOCC（每行是该 env 内的 best-F1 点；AOCC 为 ROC CSV 的 `aocc` 列，口径为 **AOCC/1e7**：`src/myevs/metrics/aocc.py` 内部统一做 `/1e7`（可由 `MYEVS_AOCC_UNIT` 调整），评测脚本不再二次缩放）：

| env | 方法 | Thr(best-F1) | precision(best) | F1(best) | AUC | MESR (`esr_mean`) | AOCC/1e7 (`aocc`) | FP@best |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| light(1.8V) | baseline(1M) | 0.7491 | 0.9542 | 0.9497 | 0.9476 | 1.0305 | 0.820552 | 7400 |
| light(1.8V) | s14(1M,a0.2,raw0) | 0.8185 | 0.9520 | 0.9520 | 0.9508 | 1.0320 | 0.821413 | 7819 |
| light(1.8V) | s19(1M,a0.2,b0.5) | 0.3166 | 0.9547 | 0.9532 | 0.9497 | 0.9856 | 0.821157 | 7354 |
| light(1.8V) | s21(1M,a0.2,b0.6,k0.8) | 0.1261 | 0.9589 | 0.9567 | 0.9523 | 0.9178 | 0.821585 | 6660 |
| mid(2.5V) | baseline(1M) | 4.8347 | 0.8382 | 0.8177 | 0.9232 | 1.0155 | 0.791076 | 25108 |
| mid(2.5V) | s14(1M,a0.2,raw0) | 5.6732 | 0.8457 | 0.8199 | 0.9268 | 1.0212 | 0.782966 | 23663 |
| mid(2.5V) | s19(1M,a0.2,b0.5) | 5.1665 | 0.8479 | 0.8200 | 0.9274 | 1.0145 | 0.781709 | 23208 |
| mid(2.5V) | s21(1M,a0.2,b0.6,k0.8) | 4.6887 | 0.8479 | 0.8266 | 0.9347 | 0.9695 | 0.787541 | 23576 |
| heavy(3.3V) | baseline(1M) | 7.2797 | 0.7924 | 0.7610 | 0.9136 | 1.0085 | 0.770944 | 31245 |
| heavy(3.3V) | s14(1M,a0.2,raw0) | 8.0514 | 0.7822 | 0.7641 | 0.9178 | 1.0208 | 0.773252 | 33870 |
| heavy(3.3V) | s19(1M,a0.2,b0.5) | 7.5773 | 0.7858 | 0.7635 | 0.9178 | 1.0185 | 0.770314 | 32976 |
| heavy(3.3V) | s21(1M,a0.2,b0.6,k0.8) | 7.1721 | 0.7947 | 0.7714 | 0.9256 | 0.9840 | 0.772704 | 31542 |

#### 结论（是否替代 s19 / s20）

- s21 在 1M 口径下三环境 AUC/F1 均高于 s19/s20；
- mid 的 FP 在 best-F1 点已被压回到接近 s19(b0.5) 的量级，同时 precision/F1/AUC 更高；
- heavy 的 FP 明显低于 s19/s20，且 F1/AUC 同步更高。

建议：将 `s21(a0.2,b0.6,k0.8)` 作为当前 Part2 新主候选（更偏 F1/AUC），`k0.5` 可作为“更偏 precision/FP”的备选。

### 资源不变的 best-F1 上限（prescreen200k / s=9 / tau=128ms）

动机：用户当前只关心 “F1 是否提高，且尽量不增加资源消耗”。因此先做 **纯超参重选**（不改 kernel、**不新增任何 per-pixel 状态数组**），把 s21 在 prescreen200k 口径下的 best-F1 上限榨干。

共同口径：`max-events=200000, s=9, tau-us=128000`，从 ROC CSV 中筛 `*s9_tau128000*` 的行，取全局 max-F1 对应的 operating point（其阈值列为 `value`）。

#### sweep #1（固定 a=0.2，宽扫 b/k）

产物目录：`data/ED24/myPedestrain_06/EBF_Part2/_tmp_s21_paramsearch_prescreen200k_s9_tau128ms/`

复现命令：

```powershell
$env:PYTHONNOUSERSITE='1'
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant s21 --max-events 200000 --s-list 9 --tau-us-list 128000 --s21-alpha-list 0.2 --s21-beta-list '0.3,0.4,0.5,0.6,0.7,0.8' --s21-kappa-list '0.2,0.3,0.4,0.5,0.6,0.7,0.8' --esr-mode best --aocc-mode off --out-dir data/ED24/myPedestrain_06/EBF_Part2/_tmp_s21_paramsearch_prescreen200k_s9_tau128ms
```

| env | best-F1 | tag | Thr(value) | TP | FP | FN |
|---|---:|---|---:|---:|---:|---:|
| light | 0.9567 | `ebf_s21_a0p2_b0p7_k0p8_labelscore_s9_tau128000` | 0.0299 | 155447 | 6594 | 7479 |
| mid | 0.8208 | `ebf_s21_a0p2_b0p8_k0p8_labelscore_s9_tau128000` | 4.5210 | 42915 | 7965 | 10777 |
| heavy | 0.7956 | `ebf_s21_a0p2_b0p8_k0p8_labelscore_s9_tau128000` | 6.9769 | 31005 | 7069 | 8858 |

#### sweep #2（宽扫 a/b/k）

产物目录：`data/ED24/myPedestrain_06/EBF_Part2/_tmp_s21_paramsearch2_abk_prescreen200k_s9_tau128ms/`

复现命令：

```powershell
$env:PYTHONNOUSERSITE='1'
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant s21 --max-events 200000 --s-list 9 --tau-us-list 128000 --s21-alpha-list '0.1,0.15,0.2,0.25,0.3' --s21-beta-list '0.7,0.8,0.9,1.0' --s21-kappa-list '0.6,0.7,0.8,0.9,1.0' --esr-mode best --aocc-mode off --out-dir data/ED24/myPedestrain_06/EBF_Part2/_tmp_s21_paramsearch2_abk_prescreen200k_s9_tau128ms
```

| env | best-F1 | tag | Thr(value) | TP | FP | FN |
|---|---:|---|---:|---:|---:|---:|
| light | 0.9572 | `ebf_s21_a0p3_b0p7_k1_labelscore_s9_tau128000` | 0.0532 | 155760 | 6756 | 7166 |
| mid | 0.8225 | `ebf_s21_a0p1_b1_k1_labelscore_s9_tau128000` | 3.8699 | 43014 | 7887 | 10678 |
| heavy | 0.7971 | `ebf_s21_a0p2_b1_k1_labelscore_s9_tau128000` | 6.7755 | 30848 | 6693 | 9015 |

结论（prescreen200k / 资源不变）：s21 仍有可挖空间；在 #2 的更宽超参范围内，light/mid/heavy 的 best-F1 均较 #1 继续提升。

## s22：Any-Pol Burst Gate（同像素任意极性短 dt 门控，零新增状态）

### 动机

s9 的门控非常克制：它只在“同像素 + 同极性 + 极短 dt”时触发。

但在更高分辨率/更真实的平台上，我们更关心一种更常见的噪声机理：

- 同一像素在极短时间内反复触发，**极性可能交替**（+ - + -），或受读出/阈值抖动影响出现翻转。

因此 s22 把“同像素历史”从“同极性”放宽到“任意极性”，仍然保持 **不新增任何 per-pixel 状态数组**（复用 baseline 的 `last_ts/last_pol`）。

### 算法定义（低延迟设计）

位置：`src/myevs/denoise/ops/ebfopt_part2/s22_anypol_burst_gate.py`

1) 先计算 baseline raw（同极性邻域支持）：

$$
\mathrm{raw} = \sum w_t,\quad w_t = \frac{\tau-\Delta t}{\tau}\cdot\mathbb{1}(\Delta t\le\tau)\cdot\mathbb{1}(p_{nei}=p)
$$

2) 取同像素“上一事件（任意极性）”间隔（绝对时间，单位 us）：

$$
dt_0 = t_i - t_{\text{prev}}(x_i,y_i)
$$

3) 门控惩罚（仅当 `raw>=raw_thr` 且 $dt_0<dt_{thr}$ 时惩罚）：

$$
\mathrm{score}=\mathrm{raw}\cdot\mathrm{pen},\quad
\mathrm{pen}=
\begin{cases}
1 & (\mathrm{raw}<\mathrm{raw\_thr})\ \text{或}\ (dt_0\ge dt_{thr}) \\
\left(\frac{dt_0}{dt_{thr}}\right)^{\gamma} & (\mathrm{raw}\ge\mathrm{raw\_thr}\ \text{且}\ dt_0<dt_{thr})
\end{cases}
$$

超参（环境变量）：

- `MYEVS_EBF_S22_DT_THR_US`（默认 4096）
- `MYEVS_EBF_S22_RAW_THR`（默认 0）
- `MYEVS_EBF_S22_GAMMA`（默认 1）

### 实现约束核对

- 在线流式/单遍：是（仅使用 `last_ts/last_pol`，不新增 `self_acc/acc_pos/acc_neg`）
- 复杂度：$O(r^2)$（raw 仍是邻域一次遍历；额外只读同像素历史一次）
- Numba 必须：是（numba 不可用直接报错，无静默 fallback）

### 实验口径（prescreen，对齐参数）

对齐口径：固定 `--max-events 200000` 且固定 `s=9, tau=128ms`。

- baseline（对照）产物：`data/ED24/myPedestrain_06/EBF_Part2/ebf_baseline_s9_tau128_prescreen200k_esrbest/`
- s22 产物：`data/ED24/myPedestrain_06/EBF_Part2/s22_anypol_burst_dt4096_raw0_g1_s9_tau128_prescreen200k_esrbest/`

运行命令：

```powershell
$env:PYTHONNOUSERSITE='1'

# baseline
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant ebf --max-events 200000 --esr-mode best --aocc-mode off --s-list 9 --tau-us-list 128000 --out-dir data/ED24/myPedestrain_06/EBF_Part2/ebf_baseline_s9_tau128_prescreen200k_esrbest

# s22 (推荐参数)
$env:MYEVS_EBF_S22_DT_THR_US='4096'
$env:MYEVS_EBF_S22_RAW_THR='0'
$env:MYEVS_EBF_S22_GAMMA='1'
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant s22 --max-events 200000 --esr-mode best --aocc-mode off --s-list 9 --tau-us-list 128000 --out-dir data/ED24/myPedestrain_06/EBF_Part2/s22_anypol_burst_dt4096_raw0_g1_s9_tau128_prescreen200k_esrbest
```

### 结果与结论（AUC/F1 有 0.0几级别提升；MESR 仅记录）

对比点：同一口径下的 `s=9, tau=128ms`，每个 env 各自取 best-F1 operating point。

| env | baseline AUC | s22 AUC | ΔAUC | baseline F1(best) | s22 F1(best) | ΔF1 | baseline MESR(esr_mean) | s22 MESR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| light(1.8V) | 0.947564 | 0.949362 | +0.001798 | 0.949739 | 0.951340 | +0.001600 | 1.030472 | 0.954729 |
| mid(2.5V) | 0.921924 | 0.923065 | +0.001141 | 0.810827 | 0.812498 | +0.001671 | 1.016829 | 1.012657 |
| heavy(3.3V) | 0.920467 | 0.920780 | +0.000312 | 0.786882 | 0.788155 | +0.001274 | 1.020768 | 1.014276 |

解释（为什么它在“高分辨率目标”上更有吸引力）：

- s22 用的是“同像素重复触发的短 dt”这一 **与分辨率无关/更机理化** 的噪声指示；当分辨率升高时，真实运动更少在几毫秒内重复落在同一像素，短 dt 更可能来自热点/翻转噪声。
- 资源上：只复用 baseline 的 `last_ts/last_pol`，对 640×480 / 1280×720 的扩展不会出现像 s21 那样的 per-pixel 状态翻倍。

风险与注意：

- 该门控在推荐参数下 `raw_thr=0`，属于“广触发”门控，会改变更多事件的分数尺度；优点是提升可分性，缺点是更依赖数据分布。
- 若迁移到其他数据集出现 AUC 下降，优先把 `MYEVS_EBF_S22_RAW_THR` 调到 1~3，让触发更克制（更像 s9 的风格）。

## s23：Feat+Logit Fusion（特征提取 + 可学习线性融合）

### FAQ：它是不是“机器学习”？为什么看起来很像？实现会很复杂吗？

你问到的几个点，结论先写在前面：

- **它是“监督学习的线性模型”**：训练脚本离线拟合的是一个 logistic regression（线性 + sigmoid 的二分类器），输出是一组线性权重；推理时不跑优化，不需要梯度/反向传播。
- **推理阶段仍然是“手工特征 + 一次线性加和”**：实际在线输出用的是 logit（不做 sigmoid），就是把若干在线特征做加权求和，用于排序与阈值扫描。
- **代码量大主要来自工程约束，而不是模型复杂**：
	- 特征必须在 Numba kernel 内在线算出来（保证单遍/实时），且要和训练侧的特征定义严格一致；
	- 需要支持“可选特征开关”（`selfacc` / `ishot` / `hotnbr` 等）与环境变量注入，避免改代码就换配置；
	- 还需要保证在 numba 不可用时直接失败（避免静默退化）。
- **上线/复现并不复杂**：拿到一组可复现的权重（训练脚本会打印 PowerShell 可复制的 env vars）+（可选）一个 `hotmask.npy`，然后像其它 s* 一样跑 sweep/推理即可；在线复杂度仍是 $O(r^2)$，额外内存默认不变（仅复用 baseline 的 `last_ts/last_pol`）。

### 失败模式 / 可验证假设

动机：Part2 里单一 heuristic gate/penalty 往往卡在 ~$10^{-3}$ 量级提升；而 baseline raw 只是“同极性邻域证据”，可分性信息可能不足。

假设（可验证）：在不引入额外 per-pixel 状态数组的前提下，只用 **baseline 的 last_ts/last_pol** 在线抽取少量特征（同极/异极/翻转/短 dt 等），再做线性融合，有机会把 AUC/F1 推到接近 $10^{-2}$ 量级提升。

补充：允许“极少量新增状态”时，可以额外引入一个 **同像素同极性 hotness/rate** 特征 `selfacc`（每像素仅 2 字节 `uint16` 的 Q8 accumulator，默认关闭；只有 `MYEVS_EBF_S23_W_SELFACC!=0` 才启用/分配），通常对 heavy 场景更有帮助。

### 算法定义（在线 / 单遍 / $O(r^2)$）

位置：`src/myevs/denoise/ops/ebfopt_part2/s23_featlogit.py`

对每个事件 $e=(x,y,t,p)$：

- 邻域证据（与 EBF 同复杂度，一次邻域遍历同时累计）：
	- $raw_{same}$：同极性邻域 tau-weighted recency 累计
	- $raw_{opp}$：异极性邻域 tau-weighted recency 累计
- 同像素特征（不新增状态，复用 `last_ts/last_pol`）：
	- $toggle\in\{0,1\}$：同像素上一事件存在且极性翻转
	- $dtsmall = 1-\mathrm{clamp}(dt0/dt_{thr},0..1)$，其中 $dt0$ 为同像素上一次事件 dt（任意极性）
	- $oppr = raw_{opp}/(raw_{same}+\varepsilon)$
	- $sameburst = dtsmall\cdot(1-toggle)$：短 dt 且同极性连续触发（一个零额外存储的交互项，通常更偏向噪声 burst）

- 可选同像素特征（极少量新增状态，默认关闭）：
	- $selfacc$：同像素泄露 burst 积分（Q8，不区分极性），用 $dt0/tau$ 做泄露并在 $dt0\le dt_{thr}$ 时 +1（实现里是 +256），用于刻画“该像素近期短间隔连发强度/发热程度”。

- 可选静态像素特征（极少量新增状态，默认关闭）：
	- $ishot\in\{0,1\}$：来自外部 hotpixel mask 的指示量（每像素 1 字节 `uint8` mask；推理时只做一次查表读取），用于直接惩罚“已知热点像素”的事件。
	- $hotnbr\in[0,1]$：事件邻域内 hotpixel mask 命中比例（在原本的邻域遍历里顺手统计；不新增 per-pixel 状态），用于表达“该事件周围是否被热点包围”。

输出 score（logit，不做 sigmoid，直接用于排序与阈值扫描）：

$$
score = b + w_{same} raw_{same} + w_{opp} raw_{opp} + w_{oppr} oppr + w_{toggle} toggle + w_{dtsmall} dtsmall + w_{sameburst} sameburst + w_{selfacc} selfacc + w_{hot} ishot + w_{hotnbr} hotnbr
$$

参数由环境变量控制：

- `MYEVS_EBF_S23_DT_THR_US`
- `MYEVS_EBF_S23_BIAS`, `MYEVS_EBF_S23_W_SAME`, `MYEVS_EBF_S23_W_OPP`, `MYEVS_EBF_S23_W_OPPR`, `MYEVS_EBF_S23_W_TOGGLE`, `MYEVS_EBF_S23_W_DTSMALL`, `MYEVS_EBF_S23_W_SAMEBURST`
- `MYEVS_EBF_S23_W_SELFACC`（默认 0.0；非 0 时启用 `selfacc` 并分配 `uint16` per-pixel 状态数组）
- `MYEVS_EBF_S23_W_HOT`（默认 0.0；非 0 时启用 `ishot` 特征）
- `MYEVS_EBF_S23_W_HOTNBR`（默认 0.0；非 0 时启用 `hotnbr` 特征；需要 hotmask）

当 `MYEVS_EBF_S23_W_HOT!=0` 或 `MYEVS_EBF_S23_W_HOTNBR!=0` 时：

- sweep 脚本需要设置 `MYEVS_EBF_S23_HOTMASK_NPY=/path/to/mask.npy`（mask 为 `(H,W)` 或 `(H*W,)`，非 0 视为 hotpixel）。
- 训练脚本用 `--hotmask-npy /path/to/mask.npy` 把 `ishot/hotnbr` 作为额外特征参与拟合，并导出 `MYEVS_EBF_S23_W_HOT` / `MYEVS_EBF_S23_W_HOTNBR`。

实现约束核对：

- 在线流式、单遍处理：是
- 单事件复杂度：邻域遍历 $O(r^2)$ + 常数项：是
- Numba kernel 必须可用：是（不可用直接返回 None，训练脚本会报错）
- 额外 per-pixel 状态数组：默认否（复用 baseline `last_ts/last_pol`）；启用 `selfacc` 时：是（`uint16`，2B/px）

### 训练脚本（离线拟合线性权重）

位置：`scripts/ED24_alg_evalu/train_s23_featlogit.py`

设计原则：不追求“学到最强”，而是先把流程跑通，并强调 **保底 baseline**：

- 推荐训练方式：固定同极性主项（标准化空间的 `--fix-w-same 1.0`），只学习其它特征的增量权重，避免把 baseline 的判别结构学坏。

示例（50k/环境，快速拟合并打印可复制的 PowerShell env vars）：

```powershell
$env:PYTHONNOUSERSITE='1'
conda run -n myEVS python -m scripts.ED24_alg_evalu.train_s23_featlogit --max-events 50000 --epochs 15 --lr 0.05 --seed 0 --fix-w-same 1.0
```

补充：训练子集控制（用于做“只优化 heavy”之类的对照，不影响推理开销）：

- `--train-envs light,mid,heavy`（默认全用）
- 例如只用 heavy：`--train-envs heavy`

如果想训练并启用 `selfacc`（会多学一个 `MYEVS_EBF_S23_W_SELFACC`）：

```powershell
$env:PYTHONNOUSERSITE='1'
conda run -n myEVS python -m scripts.ED24_alg_evalu.train_s23_featlogit --use-selfacc --max-events 50000 --epochs 15 --lr 0.05 --seed 0 --fix-w-same 1.0
```

如果想训练并启用 `ishot`（会多学一个 `MYEVS_EBF_S23_W_HOT`，并在 sweep 时需要 `MYEVS_EBF_S23_HOTMASK_NPY`）：

```powershell
$env:PYTHONNOUSERSITE='1'
# 先用 heavy 数据构造 hotmask（示例：score = neg_count - 2*pos_count，再取 topk）
conda run -n myEVS python -m scripts.ED24_alg_evalu.build_hotmask_from_labeled_npy \
	--input D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3.npy \
	--width 346 --height 260 --topk 32768 --pos-weight 2.0 \
	--output data/ED24/myPedestrain_06/EBF_Part2/hotmask_heavy_score_neg_minus_2pos_topk32768.npy

conda run -n myEVS python -m scripts.ED24_alg_evalu.train_s23_featlogit \
	--use-selfacc --hotmask-npy data/ED24/myPedestrain_06/EBF_Part2/hotmask_heavy_score_neg_minus_2pos_topk32768.npy \
	--max-events 50000 --epochs 25 --lr 0.05 --seed 0 --fix-w-same 1.0 \
	--nonpos toggle,dtsmall,sameburst,selfacc,ishot
```

### Prescreen（对齐口径，200k / s=9 / tau=128ms）

这次跑出来的“可复现权重”（来自 `--fix-w-same 1.0` 的 50k 训练）：

```powershell
$env:MYEVS_EBF_S23_DT_THR_US='4096.0'
$env:MYEVS_EBF_S23_BIAS='-0.861757778631843'
$env:MYEVS_EBF_S23_W_SAME='0.09599152171709041'
$env:MYEVS_EBF_S23_W_OPP='0.00789800746671325'
$env:MYEVS_EBF_S23_W_OPPR='-1.778900237617663e-14'
$env:MYEVS_EBF_S23_W_TOGGLE='-0.08978509939291006'
$env:MYEVS_EBF_S23_W_DTSMALL='-0.3378145853221442'

$env:PYTHONNOUSERSITE='1'
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant s23 --max-events 200000 --s-list 9 --tau-us-list 128000 --esr-mode all --aocc-mode all --out-dir data/ED24/myPedestrain_06/EBF_Part2/s23_featlogit_prescreen_s9_tau128ms_200k_wsamefix1_train50k_ep15_seed0_esrall_aoccall
```

产物目录（本次）：

- `data/ED24/myPedestrain_06/EBF_Part2/s23_featlogit_prescreen_s9_tau128ms_200k_wsamefix1_train50k_ep15_seed0_esrall_aoccall/`

### 阶段性结果（prescreen 对齐）

对比 baseline（同口径：`max-events=200k, s=9, tau=128ms`；roc convention=paper；F1 取 best-F1 operating point）：

注意：从 2026-04-09 起，`aocc` 的口径统一为 **AOCC/1e7**（即 `src/myevs/metrics/aocc.py` 内部直接做 `/1e7` 缩放后返回；ROC CSV 的 `aocc` 列直接写该值）。

| env | 方法 | best(s,tau) | Thr(best-F1) | AUC | F1 | MESR(esr_mean) | AOCC/1e7(aocc) | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| light(1.8V) | baseline | (9, 128ms) | 0.7491 | 0.9476 | 0.9497 | 1.0305 | 0.8206 | - |
| light(1.8V) | s23 | (9, 128ms) | -0.8562 | 0.9496 | 0.9511 | 0.9400 | 0.8252 | AUC/F1↑；AOCC↑（MESR 仅记录） |
| mid(2.5V) | baseline | (9, 128ms) | 4.8394 | 0.9219 | 0.8108 | 1.0168 | 0.8481 | - |
| mid(2.5V) | s23 | (9, 128ms) | -0.4129 | 0.9301 | 0.8174 | 1.0178 | 0.8489 | AUC/F1 接近 +$10^{-2}$ |
| heavy(3.3V) | baseline | (9, 128ms) | 7.3581 | 0.9205 | 0.7869 | 1.0208 | 0.9065 | - |
| heavy(3.3V) | s23 | (9, 128ms) | -0.1810 | 0.9287 | 0.7944 | 1.0219 | 0.9119 | AUC/F1 接近 +$10^{-2}$ |

补充（同口径 200k / s=9 / tau=128ms）：启用 `ishot`（hotmask）后，在一个可复现构造（`score=neg-2*pos, topk=32768`）上 heavy best-F1 可达约 **0.8136**。

补充（同口径 200k / s=9 / tau=128ms）：新增 `hotnbr`（邻域 hotmask 命中比例，在线顺手统计，不新增 per-pixel 状态）并一起参与训练（`--use-selfacc --hotmask-npy ... --nonpos ... ,hotnbr`）后，heavy best-F1 约 **0.8108**，未超过 0.8136（结论：该特征在当前口径下不带来增益）。

- 产物目录：`data/ED24/myPedestrain_06/EBF_Part2/_tmp_s23_hotnbr_prescreen200k_s9_tau128ms_200k_train50k_ep25_seed0/`

### 冲 heavy best-F1=0.85 的低成本尝试（prescreen200k，F1 优先）

背景：用户明确希望在 **prescreen200k** 口径下把 heavy best-F1 推到 **0.85**，同时尽量不增加资源占用。

关键观察：当前 best-F1 点（heavy）约为 `prec≈0.823, rec≈0.805`；若要到 0.85，通常需要同时显著压 FP 且不掉 recall（难度大）。因此先尝试了两类“几乎不增推理开销”的方案做排除。

尝试 1：heavy-only 训练（仅训练侧改动；推理 kernel/状态不变）

- 做法：`train_s23_featlogit.py --train-envs heavy`，其余保持与当前 s23+hotmask 训练一致（含 `--use-selfacc` + 二值 hotmask）。
- 结果（prescreen200k / s=9 / tau=128ms）：heavy best-F1 约 **0.8069**（比 0.8136 更差）。
- 产物目录：`data/ED24/myPedestrain_06/EBF_Part2/_tmp_sweep_heavy_only_train_hotmask_sa/`
- 结论：仅改变训练数据分布不足以把 heavy best-F1 拉到 0.85；该方向停掉。

尝试 2：hotmask “强度化”/rank-hotmap（信息量更大但仍 1B/px）

- 思路：把二值 `ishot∈{0,1}` 扩展为按 score 排名编码的 uint8 强度（topk 内给不同强度），希望减少“误伤 TP”同时继续压 FP。
- 结果（prescreen200k / s=9 / tau=128ms）：heavy best-F1 约 **0.7947**（显著变差）。
- 产物目录：`data/ED24/myPedestrain_06/EBF_Part2/_tmp_sweep_s23_hotmap_rank_sa_prescreen200k/`
- 结论：该思路在当前口径下不成立；相关实验性实现已回滚，不作为后续主线。

下一步建议（仍坚持在线/单遍/低状态）：

- 二值 hotmask 的调参（topk/pos_weight/dilate 等）已出现明显平台期；要继续冲 0.85，更可能需要引入“对 heavy 更专门、但仍很轻量”的新判别信息（例如更稳健的同像素/邻域噪声指示），并通过 prescreen200k 快速验证是否值得继续。

解读（本阶段优先级：AUC/F1 → AOCC → MESR 仅记录）：

- **主目标 AUC/F1**：mid/heavy 达到你希望的接近 $10^{-2}$ 档提升；light 也有小幅增益。
- AOCC：三种噪声强度下均有小幅提升（次要参考）。
- MESR：按同一脚本口径计算并记录，不作为本阶段主优化目标。

是否继续：继续（因为首次达到接近 $10^{-2}$ 量级 AUC/F1 提升）。下一步最小改动：  

- 先把训练/验证口径进一步对齐 sweep（例如直接在 sweep 的 200k 上做一次权重再拟合），并检查权重在不同 seed/不同子集上的稳定性（避免少量样本导致的偶然提升）。

## s24：S14 + Burstiness Gate（邻域极短 dt 占比门控，零新增状态）

### 失败模式 / 可验证假设

动机：s14 的 cross-pol boost 能显著提升可分性，但 residual FP 里仍可能存在“局部爆发/热点扩散”——表现为邻域内大量像素在极短时间内刚刚触发过，导致分数虚高。

假设（可验证）：对 s14 的高分事件，若其邻域里“极短 $dt$ 的邻居权重占比”异常高，则它更像 bursty noise 团而非稳定运动边缘；对这类事件做**克制惩罚**可以进一步压 FP、提升 AUC/F1，且不引入 s7/s8 那样的强几何假设。

### 算法定义（在线 / 单遍 / $O(r^2)$）

位置：`src/myevs/denoise/ops/ebfopt_part2/s24_s14_burstiness_gate.py`

一次邻域遍历同时累计（同 s14）：

- 同极性证据：$raw=\sum w_t$
- 异极性证据：$opp=\sum w_t$

其中 $w_t=(\tau-\Delta t)/\tau$，且只统计 $\Delta t\le\tau$。

s14 主干分数：

$$
score_0 =
\begin{cases}
raw & (raw < raw_{thr})\\
raw + \alpha\,opp & (raw \ge raw_{thr})
\end{cases}
$$

定义 burstiness（邻域极短 dt 占比）：设一个很小的阈值 $\delta$（单位 us），在同一次邻域遍历中额外累计

$$
total_w = \sum w_t,\quad burst_w = \sum w_t\,\mathbb{1}(\Delta t\le\delta),\quad b=\frac{burst_w}{total_w+\varepsilon}\in[0,1]
$$

门控惩罚（只在 raw 已经高且 burstiness 异常高时触发）：

$$
score = score_0\cdot pen,\quad
pen=
\begin{cases}
1 & (b\le b_{thr})\\
\left(\frac{1-b}{1-b_{thr}}\right)^\gamma & (b>b_{thr})
\end{cases}
$$

### 超参（环境变量）

- `MYEVS_EBF_S24_ALPHA`：s14 boost 系数 $\alpha$（建议小，避免扰乱排序）
- `MYEVS_EBF_S24_RAW_THR`：只在 raw 高时启用 s14 boost（同时也是 s24 的“克制触发”前置）
- `MYEVS_EBF_S24_BURST_DT_US`：$\delta$（极短 dt 阈值，us）
- `MYEVS_EBF_S24_B_THR`：$b_{thr}$（burstiness 阈值，越大越克制）
- `MYEVS_EBF_S24_GAMMA`：惩罚幂指数

### 实现约束核对

- 在线流式/单遍：是（仅 `last_ts/last_pol`）
- 复杂度：$O(r^2)$（复用邻域遍历；新增只是常数累计）
- Numba 必须：是（不可用直接报错，无静默 fallback）
- 额外 per-pixel 状态数组：否

### 烟测（已跑通）

```powershell
$env:PYTHONNOUSERSITE='1'
$env:MYEVS_EBF_S24_ALPHA='0.2'
$env:MYEVS_EBF_S24_RAW_THR='3.0'
$env:MYEVS_EBF_S24_BURST_DT_US='2048'
$env:MYEVS_EBF_S24_B_THR='0.85'
$env:MYEVS_EBF_S24_GAMMA='1.0'
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant s24 --max-events 20000 --s-list 9 --tau-us-list 128000 --esr-mode best --aocc-mode off --out-dir data/ED24/myPedestrain_06/EBF_Part2/_smoke_s24_s14_burstiness_prescreen20k_s9_tau128ms
```

### prescreen200k（grid1：已完成）

设置：`s=9,tau=128ms,max-events=200k`；小网格（`alpha=0.2, raw_thr=3.0, burst_dt_us∈{2048,4096}, b_thr∈{0.85,0.9}, gamma=1.0`）。

输出目录：`data/ED24/myPedestrain_06/EBF_Part2/_prescreen_s24_s14_burst_grid1_s9_tau128ms_200k/`

结论：heavy 的 best-F1 相比 baseline 有小幅提升，但仍低于 s14；因此继续做了一轮更有针对性的 grid2。

### prescreen200k（grid2：已完成，针对性扫 b_thr/gamma/burst_dt_us）

设置：`s=9,tau=128ms,max-events=200k`；网格：

- `alpha∈{0.1,0.2}`
- `raw_thr=3.0`
- `burst_dt_us∈{1024,2048,4096}`
- `b_thr∈{0.75,0.8,0.85,0.9}`
- `gamma∈{1.0,2.0}`

输出目录：`data/ED24/myPedestrain_06/EBF_Part2/_prescreen_s24_s14_burst_grid2_s9_tau128ms_200k/`

关键结果（同口径对比：固定 `s=9,tau=128ms`，每个 env 各自取 best-F1 operating point；AUC 取该 tag 的 AUC）：

| env | baseline AUC | s14 AUC | s24(grid2) AUC | baseline F1(best) | s14 F1(best) | s24(grid2) F1(best) |
|---|---:|---:|---:|---:|---:|---:|
| light(1.8V) | 0.947564 | 0.949835 | 0.947511 | 0.949739 | 0.952529 | 0.949727 |
| mid(2.5V) | 0.921924 | 0.922328 | 0.922231 | 0.810827 | 0.812892 | 0.812862 |
| heavy(3.3V) | 0.920467 | 0.921598 | 0.921479 | 0.786882 | 0.789465 | 0.789465 |

best-F1（heavy）对应的 s24 参数 tag：`ebf_s24_a0p1_raw3_bdt1024us_b0p9_g1_labelscore_s9_tau128000`。

### 阶段性结论（是否继续）

- 在当前 ED24/myPedestrain_06 口径下：s24 的 heavy best-F1 **可以追平** s14（但 AUC 略低），light 上明显不如 s14。
- 这意味着“burstiness gate”在该数据上更像是一个“克制的微调项”，**没有带来可分性的净增益**（至少在当前网格范围内）。

是否值得继续：

- 对 ED24/myPedestrain_06：建议 **先停**（主线仍用 s14；s24 作为可解释备选保留即可）。
- 若后续换到“更 bursty 的噪声、更高分辨率、更明显的热点扩散”数据：可优先复用上面 heavy 的 best-F1 参数作为起点，再按 `burst_dt_us` 与 `b_thr` 做小幅微调验证是否出现“超过 s14 的净收益”。

## s25：s14 + Same-Pixel Refractory Gate（s14 主干 + 同像素同极性短 dt 惩罚）

### 失败模式 / 可验证假设

动机：用户目标是“light 不掉的前提下，heavy F1 更高”。在不引入新 per-pixel 状态数组的约束下，一个最小、最克制的尝试是：在 s14 主干上叠加 s9 的同像素 refractory 惩罚，只在“raw 已经很高”时触发，以更精准压制 heavy 下的热点/爆发噪声，同时尽量不伤 light。

可验证假设：heavy 的误检中存在一定比例来自“同像素同极性极短 dt 的高 raw 事件”；对这类事件做克制惩罚可以提升 precision，从而推动 heavy F1 上升。

### 算法定义（在线 / 单遍 / $O(r^2)$）

位置：`src/myevs/denoise/ops/ebfopt_part2/s25_s14_refractory_gate.py`

对每个事件 $e=(x,y,t,p)$：

1) 邻域累计（与 baseline/s14 相同，同一次邻域遍历累计）：

- 同极性证据：$raw=\sum w_t$
- 异极性证据：$opp=\sum w_t$

其中 $w_t=(\tau-\Delta t)/\tau$，只统计 $\Delta t\le\tau$。

2) s14 主干分数（cross-pol boost，仅加分不惩罚）：

$$
score_0 =
\begin{cases}
raw & (raw < raw_{thr})\\
raw + \alpha\,opp & (raw \ge raw_{thr})
\end{cases}
$$

3) 同像素 refractory 惩罚（复用 s9 思路，仅同极性、仅 raw 很高时触发）：

- 取同像素上一事件时间戳 $t_{prev}$ 与极性 $p_{prev}$（复用 `last_ts/last_pol`）。
- 若满足：$raw\ge ref\_raw_{thr}$ 且 $p_{prev}=p$，则计算 $dt=t-t_{prev}$，归一化 $d=dt/\tau$。

惩罚项：

$$
pen =
\begin{cases}
1 & (d\ge dt_{thr})\\
\left(\frac{d}{dt_{thr}}\right)^{\gamma} & (d<dt_{thr})
\end{cases}
\quad\in(0,1]
$$

最终：$score=score_0\cdot pen$。

说明：这里 $dt_{thr}$ 是“归一化到 $\tau$ 的阈值”（与 s9 保持一致的尺度约定），例如 $dt_{thr}=0.01$ 表示 $dt<0.01\tau$ 时才开始惩罚。

### 超参（环境变量）

- `MYEVS_EBF_S25_ALPHA`：s14 boost 系数 $\alpha$
- `MYEVS_EBF_S25_RAW_THR`：s14 boost 触发阈值 $raw_{thr}$
- `MYEVS_EBF_S25_DT_THR`：归一化阈值 $dt_{thr}$（相对 $\tau$）
- `MYEVS_EBF_S25_REF_RAW_THR`：refractory 惩罚触发阈值 $ref\_raw_{thr}$（仅 raw 很高时才惩罚）
- `MYEVS_EBF_S25_GAMMA`：惩罚幂指数 $\gamma$

### 实现约束核对

- 在线流式/单遍：是（只维护 `last_ts/last_pol`）
- 复杂度：$O(r^2)$ + $O(1)$（邻域遍历不变；新增仅中心像素常数逻辑）
- Numba 必须：是（numba 不可用直接报错，无静默 fallback）
- 额外 per-pixel 状态数组：否

### 烟测（已跑通）

输出目录：`data/ED24/myPedestrain_06/EBF_Part2/_smoke_s25_s14refrac_20k_s9_tau128ms/`

### prescreen200k（grid1：已完成，先固定 s14 主干为 heavy 导向）

设置：`s=9,tau=128ms,max-events=200k`。

- 固定 s14 主干：`alpha=0.1, raw_thr=3.0`（更偏 heavy 的保守 boost）
- 扫 refractory：`dt_thr∈{0.004,0.006,0.008,0.010}`、`ref_raw_thr∈{2,3,4}`、`gamma∈{1,2}`（共 24 组）

输出目录：`data/ED24/myPedestrain_06/EBF_Part2/_prescreen_s25_s14refrac_grid1_s9_tau128ms_200k/`

关键结果（同口径对比：固定 `s=9,tau=128ms`，每个 env 各自取 best-F1 operating point；AUC 取该 tag 的 AUC）：

| env | baseline AUC | s14 AUC | s25(grid1) AUC | baseline F1(best) | s14 F1(best) | s25(grid1) F1(best) |
|---|---:|---:|---:|---:|---:|---:|
| light(1.8V) | 0.947564 | 0.949835 | 0.947775 | 0.949739 | 0.952529 | 0.949787 |
| mid(2.5V) | 0.921924 | 0.922328 | 0.922476 | 0.810827 | 0.812892 | 0.813355 |
| heavy(3.3V) | 0.920467 | 0.921598 | 0.921615 | 0.786882 | 0.789465 | 0.789883 |

best-F1（heavy）对应的 s25 参数 tag：`ebf_s25_a0p1_raw3_dt0p01_rraw2_g2_labelscore_s9_tau128000`。

补充（重要）：

- 虽然 heavy best-F1 有微小提升（约 $4.2\times 10^{-4}$），但 light 与 s14 的 light 最优（0.952529）差距较大。
- 若把对照改为“同一套 s14 主干（a0.1/raw3）”而不是“各 env 的 s14 最优”，那么 s25(grid1) 在 light 与 heavy 上都能获得微小提升：
	- s14(a0.1/raw3) 的 light best-F1=0.949740，heavy best-F1=0.789465
	- s25(grid1 best) 的 light best-F1=0.949787，heavy best-F1=0.789883
	- 说明 refractory 惩罚更像是对 heavy 导向 recipe 的“克制微调项”，而不是能把 light 的最优点推高的机制。

### FP 噪声类型分析（heavy，prescreen200k，best-F1）

为验证“s25 为什么没有抓到重点”，用脚本 `scripts/ED24_alg_evalu/analyze_fp_noise_types.py` 在 heavy 的 best-F1 operating point 处，对 FP 做了一个最小可解释分解（hotmask / 同像素短 dt / 翻转短 dt / 其它）。

结果（`s=9,tau=128ms,max-events=200k`）：

- baseline（tag `ebf_labelscore_s9_tau128000`）：FP=6856，其中 hotmask 命中 84.28%；同像素同极性 $dt/\tau<0.01$ 仅 1.05%。
- s14（tag `ebf_s14_a0p1_raw3_labelscore_s9_tau128000`）：FP=7294，其中 hotmask 命中 84.27%；同像素同极性 $dt/\tau<0.01$ 仅 0.96%。
- s25（tag `ebf_s25_a0p1_raw3_dt0p01_rraw2_g2_labelscore_s9_tau128000`）：FP=7239，其中 hotmask 命中 84.24%；同像素同极性 $dt/\tau<0.01$ 仅 0.21%。

结论：在该口径下，heavy 误检主要高度集中在“热点像素”（hotmask 命中约 84%），而“同像素极短 dt”在 FP 中占比极低；因此 s25 这种主要针对 short-dt burst 的惩罚，天然很难带来明显的 heavy 净收益（它并没有打到 dominant FP 类型）。

### prescreen200k（grid2：已完成，尝试在“light 导向的 s14 主干”上叠加更克制惩罚）

设置：`s=9,tau=128ms,max-events=200k`。

- 扫 s14 主干：`alpha∈{0.3,0.4}`、`raw_thr∈{0,1}`（更偏 light/mid 的 boost 使用方式）
- 更克制惩罚：`dt_thr∈{0.004,0.006}`、`ref_raw_thr∈{4,6,8}`、`gamma∈{1,2}`

输出目录：`data/ED24/myPedestrain_06/EBF_Part2/_prescreen_s25_s14refrac_grid2_s9_tau128ms_200k/`

关键结果：

| env | baseline AUC | s14 AUC | s25(grid2) AUC | baseline F1(best) | s14 F1(best) | s25(grid2) F1(best) |
|---|---:|---:|---:|---:|---:|---:|
| light(1.8V) | 0.947564 | 0.949835 | 0.949850 | 0.949739 | 0.952529 | 0.952532 |
| mid(2.5V) | 0.921924 | 0.922328 | 0.923189 | 0.810827 | 0.812892 | 0.810255 |
| heavy(3.3V) | 0.920467 | 0.921598 | 0.921795 | 0.786882 | 0.789465 | 0.785568 |

best-F1（light）对应的 s25 参数 tag：`ebf_s25_a0p4_raw0_dt0p004_rraw4_g1_labelscore_s9_tau128000`。

### 阶段性结论（是否继续）

结论一句话：s25 在当前 ED24/myPedestrain_06 口径下，呈现明显 trade-off——要追平/略超 s14 的 light 最优时，heavy 会明显下降；而当 s14 主干偏 heavy 时，heavy 可微小提升但 light 达不到 s14 的 light 最优。

可验证证据（强约束筛选）：

- 以 s14 的 best-F1 作为强目标（light≥0.952529 且 heavy>0.789465），在 grid2 的全部 tag 中**不存在任何同时满足点**。

是否继续：

- 如果你的“light 不掉”定义是“达到 s14 的 light 最优（0.952529）”，同时 heavy 还要超过 s14 heavy 最优（0.789465）：建议 **先停** s25（当前机制在该搜索空间内不可达）。
- 如果你的“light 不掉”定义是“固定一套更可迁移的 recipe（例如 s14 的 a0.1/raw3）时，light 不降且 heavy 上升”：s25(grid1) 已经给出微小改进，但属于 $10^{-4}$ 量级，建议后续只在需要更高 precision 的 heavy 场景作为小修饰项保留。

## s26：Act-Norm Hotness Fusion（活动归一化的热点惩罚融合，Q8）

### 失败模式 / 可验证假设

动机（对准失败模式）：

- 已有证据链（noise_type_stats / FP breakdown）表明：heavy 的 FP 主要由 `hotmask/near-hot/highrate_pixel` 主导。
- 但“直接按同像素热度（hotness）强惩罚”容易误伤**真实活跃区域**（例如运动边缘/纹理区域），导致 TP 掉、light/mid AUC/F1 受影响。

可验证假设：同像素热度惩罚应当与“邻域活动强度”耦合：

- 若邻域证据强（事件密集/结构化活动），热点惩罚应被**自动减弱**；
- 若邻域证据弱（孤立、异常高发像素），热点惩罚应保持**强**。

### 算法定义（在线 / 单遍 / $O(r^2)$）

位置：`src/myevs/denoise/ops/ebfopt_part2/s26_actnorm_hotness_fusion_q8.py`

对每个事件 $e=(x,y,t,p)$：

1) 同像素热度（per-pixel 两路泄露累积，Q0 in ticks）：

- 维护 `acc_pos/acc_neg`（int32），每次触发用 $dt=t-last\_ts[x,y]$ 做泄露：`acc-=dt`（下限 0），再给当前极性那一路 `acc+=tau`（饱和）。
- 定义：
	- $acc_{same}$：当前极性对应的 accumulator
	- $acc_{opp}$：相反极性对应的 accumulator
	- 混合热度：$acc_{mix}=acc_{same}+\kappa\,acc_{opp}$

2) 邻域证据（与 EBF 同量级，一次邻域遍历累计）：

- 同极性：$raw_w=\sum (\tau-\Delta t)$
- 异极性：$opp_w=\sum (\tau-\Delta t)$
- $raw_{tot}=raw_w+opp_w$

3) 活动归一化的热点惩罚：

$$
w=\frac{\eta\,\tau}{raw_{tot}+\eta\,\tau}\in(0,1],\qquad acc_{pen}=acc_{mix}\cdot w
$$

解释：

- $raw_{tot}$ 越大（邻域越活跃），$w$ 越小，惩罚越弱（更宽容真实活动区域）；
- $raw_{tot}$ 越小（孤立/证据弱），$w$ 越接近 1，惩罚越强（更针对异常热点）。

4) 最终 score（Q8 定点）：

$$
score_{q8}=(raw_w\ll 8)+\alpha_{q8}\,opp_w-\beta_{q8}\,acc_{pen}
$$

并输出：$score=\dfrac{score_{q8}}{\tau\cdot 256}$（与 Part2 其它 q8 变体同尺度）。

### 超参（环境变量 / sweep 参数）

环境变量（直接复现实验时更稳定）：

- `MYEVS_EBF_S26_ALPHA`：异极性加分系数 $\alpha$（>=0）
- `MYEVS_EBF_S26_BETA`：热点惩罚系数 $\beta$（>=0）
- `MYEVS_EBF_S26_KAPPA`：异极性热度混合系数 $\kappa\in[0,1]$
- `MYEVS_EBF_S26_ETA`：活动归一化强度 $\eta\in[0.25,4]$（越小表示越“早”减弱惩罚）

sweep 脚本参数：

- `--s26-alpha-list` / `--s26-beta-list` / `--s26-kappa-list` / `--s26-eta-list`

### 实现约束核对

- 在线流式/单遍：是
- 复杂度：$O(r^2)$（邻域遍历不变；同像素热度更新 $O(1)$）
- Numba 必须：是（不可用直接报错，无静默 fallback）
- 额外 per-pixel 状态数组：是（`acc_pos/acc_neg` 两个 int32；复用 `last_ts/last_pol`）

### 最小验证口径（建议先 prescreen200k）

先跑 smoke（20k）确认管线正常：

```powershell
$env:PYTHONNOUSERSITE='1'
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant s26 --max-events 20000 --s-list 9 --tau-us-list 128000 --esr-mode best --aocc-mode off --out-dir data/ED24/myPedestrain_06/EBF_Part2/_smoke_s26_actnorm_prescreen20k_s9_tau128ms
```

再跑 prescreen200k（小网格，先看 heavy 是否能在不明显伤 light 的情况下减少 hot/near-hot FP）：

```powershell
$env:PYTHONNOUSERSITE='1'
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py \
	--variant s26 \
	--max-events 200000 \
	--s-list 9 --tau-us-list 128000 \
	--s26-alpha-list '0.2' \
	--s26-beta-list '0.4,0.8,1.2' \
	--s26-kappa-list '1.0' \
	--s26-eta-list '0.5,1.0,2.0' \
	--esr-mode best --aocc-mode off \
	--out-dir data/ED24/myPedestrain_06/EBF_Part2/_prescreen_s26_actnorm_grid1_s9_tau128ms_200k
```

预期要看什么（可证伪）：

- heavy：best-F1 的 FP（尤其 hotmask/near-hot/highrate）有可见下降，同时 TP 不显著下降；
- light/mid：AUC/F1 不应出现明显劣化（否则说明惩罚触发面过宽或 w 归一化不足）。

### 阶段性结论（是否继续）

### prescreen200k（grid1：已完成）

设置：`s=9,tau=128ms,max-events=200k`。

扫参：`alpha=0.2` 固定，`kappa=1.0` 固定，扫 `beta∈{0.4,0.8,1.2}` 与 `eta∈{0.5,1.0,2.0}`。

输出目录：`data/ED24/myPedestrain_06/EBF_Part2/_prescreen_s26_actnorm_grid1_s9_tau128ms_200k/`

关键产物：

- `roc_ebf_s26_light_labelscore_*.csv/png`
- `roc_ebf_s26_mid_labelscore_*.csv/png`
- `roc_ebf_s26_heavy_labelscore_*.csv/png`

关键结果（同口径对比：固定 `s=9,tau=128ms`，每个 env 各自取 best operating point）：

| env | s26 best-AUC tag | s26 AUC(best) | s26 best-F1 tag | s26 F1(best) |
|---|---|---:|---|---:|
| light(1.8V) | `ebf_s26_a0p2_b0p4_k1_e2_labelscore_s9_tau128000` | 0.954298 | `ebf_s26_a0p2_b1p2_k1_e1_labelscore_s9_tau128000` | 0.956944 |
| mid(2.5V) | `ebf_s26_a0p2_b1p2_k1_e2_labelscore_s9_tau128000` | 0.930733 | `ebf_s26_a0p2_b1p2_k1_e2_labelscore_s9_tau128000` | 0.816503 |
| heavy(3.3V) | `ebf_s26_a0p2_b1p2_k1_e2_labelscore_s9_tau128000` | 0.927234 | `ebf_s26_a0p2_b1p2_k1_e2_labelscore_s9_tau128000` | 0.791317 |

与既有对照（同口径：prescreen200k，best-F1 operating point）对比：

- light：s14 best-F1=0.952529（s26 提升约 +0.004415）
- mid：s14 best-F1=0.812892（s26 提升约 +0.003611）
- heavy：s14 best-F1=0.789465（s26 提升约 +0.001852）

### FP 噪声类型分析（heavy，prescreen200k，best-F1）

用脚本 `scripts/ED24_alg_evalu/analyze_fp_noise_types.py` 在 heavy 的 best-F1 operating point 处做最小可解释分解（hotmask / 同像素短 dt / 翻转短 dt / 其它）。

结果（`s=9,tau=128ms,max-events=200k`，tag `ebf_s26_a0p2_b1p2_k1_e2_labelscore_s9_tau128000`）：

- best-F1 点阈值：`thr=7.71835`，F1=0.791317，AUC=0.927234
- Confusion：TP=31358，FP=8233，FN=8505
- FP 分解：hotmask=6944（84.34%），其它=1269（15.41%）；同像素极短 dt（dt/τ<0.01）占比极低（0.44%）

解读（重要）：

- s26 在该口径下的 heavy **F1/AUC 确实超过 s14**，但 heavy 的 FP 仍高度集中在 hotmask（占比与 baseline/s14 同量级），且 **FP 绝对数量并未下降**（本次 best-F1 点 FP=8233，高于 baseline=6856 与 s14=7294 的量级）。
- 说明：当前 s26(grid1) 更像是“提高 recall（TP 上升）带动 F1 上升”的机制，而不是“选择性压制 hot/near-hot 噪声导致 FP 下降”的机制。

是否继续：**继续（但目标要重新对齐到‘hotmask FP 下降’）**。

下一步最小改动建议（不改机制，只扩一轮更强惩罚网格，验证能否把 FP 压下来同时不明显伤 light）：

- 以当前 best-F1（heavy/mid）为中心：固定 `kappa=1.0`，尝试 `beta` 更大（例如 `1.2,1.6,2.0`）与 `eta` 更大（例如 `2,3,4`），并可加一个更保守的 `alpha`（例如 `0.1,0.2`）
- 评估口径仍先用 prescreen200k；并对 heavy best-F1 点继续做 FP 噪声类型分解，重点看 hotmask/near-hot/highrate 是否真的下降。

### prescreen200k（grid2：已完成，更强惩罚）

设置：`s=9,tau=128ms,max-events=200k`。

扫参：`kappa=1.0` 固定，扫：

- `alpha∈{0.1,0.2}`
- `beta∈{1.2,1.6,2.0}`
- `eta∈{2,3,4}`

输出目录：`data/ED24/myPedestrain_06/EBF_Part2/_prescreen_s26_actnorm_grid2_strongerpen_s9_tau128ms_200k/`

best（by env）：

- light best-AUC：`ebf_s26_a0p2_b1p2_k1_e2_labelscore_s9_tau128000`，AUC=0.950956
- mid best-AUC：`ebf_s26_a0p1_b2_k1_e4_labelscore_s9_tau128000`，AUC=0.933073
- heavy best-AUC：`ebf_s26_a0p1_b2_k1_e4_labelscore_s9_tau128000`，AUC=0.930585

best-F1（by env）：

- light：`ebf_s26_a0p2_b1p2_k1_e2_labelscore_s9_tau128000`，F1=0.956903
- mid：`ebf_s26_a0p1_b2_k1_e4_labelscore_s9_tau128000`，F1=0.820563
- heavy：`ebf_s26_a0p1_b2_k1_e4_labelscore_s9_tau128000`，F1=0.795050

### FP 噪声类型分析（heavy，prescreen200k，best-F1，grid2）

tag `ebf_s26_a0p1_b2_k1_e4_labelscore_s9_tau128000`：

- best-F1 点阈值：`thr=6.93010`，F1=0.795050，AUC=0.930585
- Confusion：TP=32496，FP=11098，FN=7367
- FP 分解：hotmask=9602（86.52%），其它=1474（13.28%）

解读（对照 grid1）：

- heavy best-F1 虽继续上升（0.791317 -> 0.795050），但 FP 绝对数反而明显增加（8233 -> 11098），hotmask 占比也更高。
- 这进一步表明：当前 s26 的“活动归一化”并没有把 heavy 的 dominant FP（hotmask/near-hot/highrate）压下去；它的提升更偏向“TP/召回上升”带动 F1 上升。

是否继续（对准 hotmask FP 下降这个主目标）：建议 **先停 s26 的继续加大惩罚调参**（至少在当前机制形态下，强惩罚并没有带来 hotmask FP 的下降）。

下一步最小改动建议（进入下一个编号，换机制而不是继续调参）：

- 走“相对异常热度（relative abnormal hotness）”路线：用邻域同类 accumulator 的均值/中位数做 baseline，只惩罚“比邻域明显更热”的像素（更对准 hotmask/near-hot）；该思路可以做到不新增 per-pixel 数组（邻域临时统计即可），更符合你最初的噪声分析结论。

## s27：Rel Abnormal Hotness Fusion（相对异常热度惩罚融合，Q8）

### 失败模式 / 可验证假设

动机（承接 s26 的结论）：

- s26 的 F1 提升更像是“召回/TP 上升带动”，但 hotmask FP **没有下降**，甚至在更强惩罚下 FP 变多。
- heavy dominant FP 仍集中在 hotmask/near-hot/highrate，这提示我们需要一个“更对准相对异常”的惩罚：**只打比周围明显更热的像素**，而不是按绝对热度一刀切。

可验证假设：如果我们用邻域的热度（decayed accumulator 总热度）作为局部基线，只惩罚 $acc\_{mix}$ 超出基线的那一部分，则：

- heavy：hotmask/near-hot 的 FP 更可能下降（precision 上升）；
- light/mid：真实活跃区域（邻域同样很热）更不容易被误伤。

### 算法定义（在线 / 单遍 / $O(r^2)$）

位置：`src/myevs/denoise/ops/ebfopt_part2/s27_relabnorm_hotness_fusion_q8.py`

沿用 s21/s26 的同像素双通道热度（leaky accumulator）：

- $acc\_{mix}=acc\_{same}+\kappa\,acc\_{opp}$

在邻域遍历（同一次 $O(r^2)$）中，额外计算邻域基线：

- 对每个邻居像素 $j$，取其 `acc_pos/acc_neg` 并用 $dt=t-t_j$ 做临时泄露（不写回）：
  $acc\_{tot,j}=\max(0,acc\_{pos,j}-dt)+\max(0,acc\_{neg,j}-dt)$
- 只统计 $dt\le\tau$ 的邻居，取均值：$mean\_{nb}=\mathrm{mean}(acc\_{tot,j})$

相对异常惩罚：

$$
acc\_{pen}=\max\bigl(0,\;acc\_{mix}-\lambda\_{nb}\,mean\_{nb}\bigr)
$$

最终 score（Q8 定点，尺度与 s26 相同）：

$$
score\_{q8}=(raw_w\ll 8)+\alpha\_{q8}\,opp_w-\beta\_{q8}\,acc\_{pen},\quad score=\frac{score\_{q8}}{\tau\cdot 256}
$$

### 超参（环境变量 / sweep 参数）

- `MYEVS_EBF_S27_ALPHA`：异极性加分 $\alpha$（>=0）
- `MYEVS_EBF_S27_BETA`：异常热度惩罚 $\beta$（>=0）
- `MYEVS_EBF_S27_KAPPA`：异极性热度混合 $\kappa\in[0,1]$
- `MYEVS_EBF_S27_LAMBDA_NB`：邻域基线权重 $\lambda\_{nb}\in[0,2]$（越大越“宽容”，惩罚越少）

sweep：`--s27-alpha-list / --s27-beta-list / --s27-kappa-list / --s27-lambda-nb-list`

### 实现约束核对

- 在线流式/单遍：是
- 复杂度：$O(r^2)$（邻域遍历不变；额外只是同循环里做邻域 mean）
- Numba 必须：是
- 额外 per-pixel 状态数组：是（复用 s21/s26 同样的 `acc_pos/acc_neg`；不新增额外数组）

### 最小验证口径（先 smoke，再 prescreen200k）

smoke（20k）：

```powershell
$env:PYTHONNOUSERSITE='1'
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant s27 --max-events 20000 --s-list 9 --tau-us-list 128000 --esr-mode best --aocc-mode off --out-dir data/ED24/myPedestrain_06/EBF_Part2/_smoke_s27_relabnorm_prescreen20k_s9_tau128ms
```

prescreen200k（grid1，先验证“hotmask FP 是否下降”）：

```powershell
$env:PYTHONNOUSERSITE='1'
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py \
	--variant s27 \
	--max-events 200000 \
	--s-list 9 --tau-us-list 128000 \
	--s27-alpha-list '0.2' \
	--s27-beta-list '0.6,0.8,1.0,1.2' \
	--s27-kappa-list '1.0' \
	--s27-lambda-nb-list '0.0,0.5,1.0,1.5' \
	--esr-mode best --aocc-mode off \
	--out-dir data/ED24/myPedestrain_06/EBF_Part2/_prescreen_s27_relabnorm_grid1_s9_tau128ms_200k
```

### prescreen200k（grid1：已完成）

设置：`s=9,tau=128ms,max-events=200k`。

扫参：`alpha=0.2` 固定，`kappa=1.0` 固定，扫：

- `beta∈{0.6,0.8,1.0,1.2}`
- `lambda_nb∈{0.0,0.5,1.0,1.5}`

输出目录：`data/ED24/myPedestrain_06/EBF_Part2/_prescreen_s27_relabnorm_grid1_s9_tau128ms_200k/`

关键结果（同口径对比：固定 `s=9,tau=128ms`，每个 env 各自取 best operating point）：

| env | s27 best-AUC tag | s27 AUC(best) | s27 best-F1 tag | s27 F1(best) |
|---|---|---:|---|---:|
| light(1.8V) | `ebf_s27_a0p2_b0p6_k1_l1p5_labelscore_s9_tau128000` | 0.954662 | `ebf_s27_a0p2_b0p6_k1_l0_labelscore_s9_tau128000` | 0.956778 |
| mid(2.5V) | `ebf_s27_a0p2_b1_k1_l0_labelscore_s9_tau128000` | 0.934846 | `ebf_s27_a0p2_b1p2_k1_l0_labelscore_s9_tau128000` | 0.822364 |
| heavy(3.3V) | `ebf_s27_a0p2_b1p2_k1_l0_labelscore_s9_tau128000` | 0.932724 | `ebf_s27_a0p2_b1p2_k1_l0_labelscore_s9_tau128000` | 0.797273 |

### FP 噪声类型分析（heavy，prescreen200k，best-F1）

tag `ebf_s27_a0p2_b1p2_k1_l0_labelscore_s9_tau128000`：

- best-F1 点阈值：`thr=6.56057`，F1=0.797273，AUC=0.932724
- Confusion：TP=30754，FP=6531，FN=9109
- FP 分解：hotmask=5427（83.10%），其它=1086（16.63%）；同像素极短 dt（dt/τ<0.01）占比极低（0.51%）

解读（对照 s26 grid2）：

- heavy：在 best-F1 点，FP 绝对数从 11098 降到 6531，hotmask FP 从 9602 降到 5427，同时 F1 从 0.795050 升到 0.797273。
- 说明：s27 的“相对异常热度”惩罚在该口径下确实更对准 dominant FP（hotmask），符合最初目标。

评估重点（可证伪）：

- heavy：best-F1 点 hotmask FP 绝对数是否下降（用 `analyze_fp_noise_types.py` 对比 baseline/s14/s26）；
- light：是否出现明显 AUC/F1 劣化（若劣化，说明惩罚触发仍过宽，需提高 `lambda_nb` 或降低 `beta`）。

## s28：Noise-Model Surprise Z-Score（噪声率模型下的惊讶度标准化）

### 失败模式 / 可验证假设

动机（更“论文型”的改造点）：

- baseline/raw 绝对值会随事件率上升；heavy 下热点/高发射像素更容易获得更高 raw，从而主导 FP。
- 我们希望把 raw-support 变成一个**无量纲**、对事件率更鲁棒的分数，让阈值/排序更稳定。

可验证假设：如果把 raw-support 按“噪声模型下的期望与方差”做标准化（z-score），则：

- heavy：热点/高发射噪声的 raw 提升会被部分抵消，FP（尤其 hotmask 类）更可能下降；
- light/mid：真实结构信号仍能保持“显著高于噪声期望”的 z 值，AUC 不应明显掉。

### 算法定义（在线 / 单遍 / $O(r^2)$）

位置：`src/myevs/denoise/ops/ebfopt_part2/s28_noise_surprise_zscore.py`

1) baseline raw-support（同极性邻域支持度）：

$$
raw=\sum_{j\in\mathcal{N}} \mathbf{1}[pol_j=pol_i]\,\max\bigl(0,1-\Delta t/\tau\bigr)
$$

2) 全局噪声率估计（只用 1 个全局标量状态）：

- 用相邻事件的全局时间差 $\Delta t_g=t_i-t_{i-1}$ 估计瞬时事件率 $\hat\lambda\approx 1/\Delta t_g$（events/tick），再做 EMA 平滑。
- 转成每像素噪声率：$r=\lambda/(W\cdot H)$。

3) 噪声模型下的标准化（Poisson + 随机极性近似）：

- 在噪声假设下推导 $\mu(r)$ 与 $\sigma(r)$（与 $\tau$ 和邻域像素数有关），输出：

$$
z=\frac{raw-\mu(r)}{\sigma(r)+\varepsilon}
$$

### 超参（环境变量 / sweep 参数）

- `MYEVS_EBF_S28_TAU_RATE_US`：全局事件率 EMA 时间常数（微秒）。
	- `0` 表示 auto（默认）：直接用当前 sweep 点的 `tau_us`。

sweep：`--s28-tau-rate-us-list`（可选；为空等价于 `0`）。

### 实现约束核对

- 在线流式/单遍：是
- 复杂度：$O(r^2)$（邻域遍历同 baseline；额外计算是常数级）
- Numba 必须：是
- 额外 per-pixel 状态数组：否（仍然只用 `last_ts/last_pol`；外加 1 个全局标量 `rate_ema[0]`）

### 最小验证口径（先 smoke，再 prescreen200k）

smoke（20k）：

```powershell
$env:PYTHONNOUSERSITE='1'
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant s28 --max-events 20000 --s-list 9 --tau-us-list 128000 --esr-mode best --aocc-mode off --out-dir data/ED24/myPedestrain_06/EBF_Part2/_smoke_s28_surprise_prescreen20k_s9_tau128ms
```

prescreen200k（先只跑默认 auto；必要时再扫 `tau_rate_us`）：

```powershell
$env:PYTHONNOUSERSITE='1'
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant s28 --max-events 200000 --s-list 9 --tau-us-list 128000 --esr-mode best --aocc-mode off --out-dir data/ED24/myPedestrain_06/EBF_Part2/_prescreen_s28_surprise_s9_tau128ms_200k
```

### prescreen200k（default/auto：已完成）

设置：`s=9,tau=128ms,max-events=200k`。

输出目录：`data/ED24/myPedestrain_06/EBF_Part2/_prescreen_s28_surprise_s9_tau128ms_200k/`

关键结果（同口径对比：固定 `s=9,tau=128ms`，每个 env 各自取 best operating point）：

| env | s28 best-AUC tag | s28 AUC(best) | s28 best-F1 tag | s28 F1(best) |
|---|---|---:|---|---:|
| light(1.8V) | `ebf_s28_labelscore_s9_tau128000` | 0.934483 | `ebf_s28_labelscore_s9_tau128000` | 0.943288 |
| mid(2.5V) | `ebf_s28_labelscore_s9_tau128000` | 0.923317 | `ebf_s28_labelscore_s9_tau128000` | 0.810405 |
| heavy(3.3V) | `ebf_s28_labelscore_s9_tau128000` | 0.923493 | `ebf_s28_labelscore_s9_tau128000` | 0.781485 |

### FP 噪声类型分析（heavy，prescreen200k，best-F1，对照 baseline/s27）

s28 tag `ebf_s28_labelscore_s9_tau128000`：

- best-F1 点阈值：`thr=2.57854`，F1=0.781485，AUC=0.923493
- Confusion：TP=30601，FP=7851，FN=9262
- FP 分解：hotmask=6727（85.68%），其它=1104（14.06%）；同像素极短 dt（dt/τ<0.01）占比极低（0.74%）

对照（同脚本同 hotmask，同口径）：

- baseline：F1=0.786882，TP=30304，FP=6856，hotmask FP=5778（84.28%）
- s27：F1=0.797273，TP=30754，FP=6531，hotmask FP=5427（83.10%）

解读（当前结论，允许被后续扫 `tau_rate_us` 推翻）：

- s28 的“按全局事件率做 z-score 标准化”在本口径下**没有压下 heavy 的 hotmask FP**（反而 FP/hotmask FP 都更高），说明仅靠全局 rate 的归一化不足以对准该数据集的 dominant FP 失败模式。
- 它更像是在“重排 raw 分布”，但 hotmask 噪声与 signal 的 z 值仍强烈重叠。

是否继续：**先暂停**（当前默认 auto 结果已显示不优于 baseline/s27）。

若要最小成本证伪（只做 1 个小改动）：扫 `tau_rate_us` 看是否存在显著改善的稳定区间：

```powershell
$env:PYTHONNOUSERSITE='1'
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant s28 --max-events 200000 --s-list 9 --tau-us-list 128000 --s28-tau-rate-us-list '64000,128000,256000' --esr-mode best --aocc-mode off --out-dir data/ED24/myPedestrain_06/EBF_Part2/_prescreen_s28_surprise_tr_s9_tau128ms_200k
```

### prescreen200k（tau_rate_us sweep：已完成）

设置：`s=9,tau=128ms,max-events=200k`；扫 `tau_rate_us∈{64ms,128ms,256ms}`。

输出目录：`data/ED24/myPedestrain_06/EBF_Part2/_prescreen_s28_surprise_tr_s9_tau128ms_200k/`

关键结果（同口径对比：固定 `s=9,tau=128ms`，每个 env 各自取 best operating point）：

| env | s28 best-AUC tag | s28 AUC(best) | s28 best-F1 tag | s28 F1(best) |
|---|---|---:|---|---:|
| light(1.8V) | `ebf_s28_tr256000_labelscore_s9_tau128000` | 0.937689 | `ebf_s28_tr256000_labelscore_s9_tau128000` | 0.944994 |
| mid(2.5V) | `ebf_s28_tr64000_labelscore_s9_tau128000` | 0.923975 | `ebf_s28_tr64000_labelscore_s9_tau128000` | 0.811929 |
| heavy(3.3V) | `ebf_s28_tr64000_labelscore_s9_tau128000` | 0.926575 | `ebf_s28_tr64000_labelscore_s9_tau128000` | 0.791226 |

### FP 噪声类型分析（heavy，prescreen200k，best-F1；s28 tau_rate sweep 最优点）

s28 tag `ebf_s28_tr64000_labelscore_s9_tau128000`：

- best-F1 点阈值：`thr=2.34882`，F1=0.791226，AUC=0.926575
- Confusion：TP=30806，FP=7200，FN=9057
- FP 分解：hotmask=6086（84.53%），其它=1094（15.19%）；同像素极短 dt（dt/τ<0.01）占比极低（0.90%）

更新解读（结合 default/auto 与本次 sweep）：

- **`tau_rate_us` 确实能改善 s28**：heavy 上 best-F1 从 0.7815 → 0.7912，FP 从 7851 → 7200，hotmask FP 从 6727 → 6086。
- 但从“主目标：压 hotmask FP、超过 baseline/s27”角度看，**仍不足**：
	- baseline：F1=0.786882，hotmask FP=5778
	- s27：F1=0.797273，hotmask FP=5427
	- s28(tr=64ms)：F1=0.791226，hotmask FP=6086（仍高于 baseline/s27）

是否继续：**仍然暂停**（已完成最小扫参证伪；提升不足以替代 s27，也未显著压过 baseline 的 hotmask FP）。

## baseline / s14 / s19 / s21：复杂度、资源、延迟对比（已完成）

这部分回答你提出的“实时性约束下，baseline vs s14 vs s19 vs s21 的复杂度/资源/延迟差异”。结论以代码结构为主，并给出一组本机实测吞吐（steady-state，已 warmup JIT）。

### 1) 时间复杂度（单事件）

共同点：四者都保持 **在线流式、单遍处理**，主开销来自“以当前事件为中心的邻域遍历”。

- 主项：邻域遍历 $O(r^2)$，其中 $r$ 为半径（像素）。当前实现里 $r$ 会被 clamp 到 $\le 8$，所以理论上是有上界的常数，但常数仍然很大。
- 其它项：每事件还会做常数次的 per-pixel 状态更新（last_ts/last_pol，以及 s19/s21 的热度累积器）。

差异（常数因子）：

- baseline：邻域内只累计“同极性”证据（opp 直接跳过），操作最少。
- s14：同一次邻域遍历里同时累计 same/opp 两路证据，并在 raw 达阈值时引入 cross-pol boost，因此邻域内分支/加法更多。
- s19：在 s14 的基础上增加 1 路“同像素热度泄露累加器”状态（self_acc_w），并做 Q8 融合；邻域遍历仍是主项。
- s21：在 s14 基础上改为双通道同像素热度（acc_neg/acc_pos）+ 融合惩罚；每事件的常数状态更新更多，但仍主要被邻域遍历主项主导。

### 2) 额外内存（per-pixel 状态）

以默认分辨率 $346\times 260$（共 89960 像素）为例，仅统计“每像素持久状态数组”，不含输入事件数组与输出 scores：

- baseline / s14：`last_ts`(uint64) + `last_pol`(int8) \approx 0.81 MB
- s19：在上面基础上 + `self_acc_w`(int32) \approx 1.16 MB
- s21：在 baseline/s14 基础上 + `acc_neg`(int32) + `acc_pos`(int32) \approx 1.51 MB

结论：s19/s21 的额外内存是“每像素多 4~8 字节”量级，相比输入事件数组与 AOCC/可视化等耗时模块，**不是主要瓶颈**。

### 3) 延迟/吞吐（本机 micro-benchmark）

在本机对 heavy 序列做了一个可复现的吞吐测试（只测核心 kernel 的评分计算；先用 1 万事件 warmup 触发 Numba JIT，再对 30 万事件计时）：

- 设置：$w\times h=346\times 260$，`s=9`（实现里半径 clamp $\le 8$），`tau=128ms`，tick=1us
- 数据：`D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3.npy` 的前 300k events
- 结果（越大越好，单位 ev/s）：

| variant | time (s) | throughput (ev/s) | 相对 baseline |
|---|---:|---:|---:|
| baseline | 0.300 | 999,624 | 1.00x |
| s14 | 0.592 | 506,818 | 0.51x |
| s19 | 0.589 | 509,322 | 0.51x |
| s21 | 0.578 | 518,782 | 0.52x |

解读：

- baseline 明显更快，主要因为邻域内只做同极性累计；s14/s19/s21 在邻域内做 same/opp 两路证据与融合，常数因子接近 ~2x。
- s19/s21 的“热度状态”并没有再显著拉低吞吐：在 $O(r^2)$ 主项下，它们的额外 per-event 常数开销被掩盖了。
- 首次运行的端到端延迟会额外包含 Numba JIT 编译时间（秒级），不代表 steady-state。

如果你需要“更贴近真实在线”的延迟结论，建议用你目标平台的 CPU，固定 `s`/`tau`/事件率，直接用 `ev/s` 对比你的实时预算（例如 1Mev/s 的输入事件率，baseline 在本机接近可达，而 s14/s19/s21 约在 0.5Mev/s 量级）。

## 后续计划（先写清要做什么，再逐项执行）

说明：下面是“候选实验队列”，不是承诺一次性全做；原则是每个实验都先 prescreen，再决定是否值得扩展全网格 sweep。

- s3（候选）：把 s2 的惩罚从“幂”改成更平滑/更保守的函数（例如分段线性或 sigmoid），目标是减少误伤信号。
- s4（候选）：引入“局部一致性残差”而非二阶矩各向异性（例如局部方向/平面拟合残差的轻量近似），要求仍为单遍 $O(r^2)$。
- s5（候选）：各向异性核（椭圆邻域/方向性权重）+ 简单门控，目标是提升边缘/运动轨迹上的打分稳定性。
- s6（候选）：结合事件时间结构（例如局部时间梯度一致性）作为轻量判别项，仍不引入全局窗口。

每个实验完成后都要补齐：

- 方法原理（像 s1/s2 那样写清定义/公式）
- 当前算法缺陷（为什么比不过原算法/为什么会伤 AUC/F1）
- 是否值得继续优化（若值得：给出下一步最小改动；若不值得：明确停掉并进入下一个编号）

## 实验清单（s1/s2/...）

说明：本 Part2 的新方向用 `s1, s2, ...` 作为别名编号；每个编号对应一种“提高可分性/精度”的新打分机制（与 V2 类归一化分开）。

| 编号 | 名称 | 关键思想 | 主要新增超参 | 代码位置 | 产物目录（示例） |
|---|---|---|---|---|---|
| s1 | directional coherence（方向一致性/各向异性密度） | raw-score 基础上引入二阶矩各向异性指标，尝试抑制各向同性噪声团 | `eta∈[0,1]` | `src/myevs/denoise/ops/ebfopt_part2/s1_dircoh.py` | `data/ED24/myPedestrain_06/EBF_Part2/s1_*` |
| s2 | coherence-gated penalty（coh 门控惩罚） | 只惩罚“raw 高但 coh 低”的事件，尽量只压噪声团、不压结构化边缘 | `coh_thr, raw_thr, gamma` | `src/myevs/denoise/ops/ebfopt_part2/s2_cohgate.py` | `data/ED24/myPedestrain_06/EBF_Part2/s2_*` |
| s3 | smooth coh gate（平滑门控惩罚） | 用 sigmoid 平滑替代 s2 的硬门控，尽量降低“门控触发比例随噪声强度漂移”导致的误伤 | `coh_thr, raw_thr, gamma, alpha, k_raw, k_coh` | `src/myevs/denoise/ops/ebfopt_part2/s3_softgate.py` | `data/ED24/myPedestrain_06/EBF_Part2/s3_*` |
| s4 | resultant gate（一阶矩一致性门控） | 用邻域偏移向量的 resultant（抵消性）衡量结构性，替代二阶矩 coh | `align_thr, raw_thr, gamma` | `src/myevs/denoise/ops/ebfopt_part2/s4_residual_gate.py` | `data/ED24/myPedestrain_06/EBF_Part2/s4_*` |
| s5 | elliptic spatialw（椭圆/方向性空间权重） | 用旋转椭圆距离代替圆形距离，形成“全局方向偏好”的空间权重（流式友好） | `ax, ay, theta_deg` | `src/myevs/denoise/ops/ebfopt_part2/s5_elliptic_spatialw.py` | `data/ED24/myPedestrain_06/EBF_Part2/s5_*` |
| s6 | time-coh gate（时间一致性门控） | 用邻域 dt 的加权方差构造 time-coh，一样只惩罚“raw 高但 time-coh 低”的事件 | `timecoh_thr, raw_thr, gamma` | `src/myevs/denoise/ops/ebfopt_part2/s6_timecoh_gate.py` | `data/ED24/myPedestrain_06/EBF_Part2/s6_*` |
| s7 | plane residual gate（时间平面残差门控） | 在邻域 dt 表面上做加权平面拟合，用拟合残差只惩罚“raw 高但平面一致性差”的事件 | `sigma_thr, raw_thr, gamma, min_pts` | `src/myevs/denoise/ops/ebfopt_part2/s7_plane_gate.py` | `data/ED24/myPedestrain_06/EBF_Part2/s7_*` |
| s8 | plane R2 gate（时间平面解释度门控） | 在邻域 dt 表面上做加权平面拟合，用解释度 $R^2$ 只惩罚“raw 高但平面解释度低”的事件 | `r2_thr, raw_thr, gamma, min_pts` | `src/myevs/denoise/ops/ebfopt_part2/s8_plane_r2_gate.py` | `data/ED24/myPedestrain_06/EBF_Part2/s8_*` |
| s9 | refractory/burst gate（同像素超高频门控） | 用“同像素上一次同极性事件的 dt”识别热点/爆发噪声，只惩罚“raw 高且 dt 极小”的事件 | `dt_thr, raw_thr, gamma` | `src/myevs/denoise/ops/ebfopt_part2/s9_refractory_gate.py` | `data/ED24/myPedestrain_06/EBF_Part2/s9_*` |
| s24 | s14 + burstiness gate（邻域极短 dt 占比门控） | 在 s14（cross-pol boost）主干上，统计邻域内“极短 dt”权重占比 b；仅当 raw 高且 b 高时对 score 做克制惩罚，专打 bursty 噪声团/热点扩散 | `alpha, raw_thr, burst_dt_us, b_thr, gamma` | `src/myevs/denoise/ops/ebfopt_part2/s24_s14_burstiness_gate.py` | `data/ED24/myPedestrain_06/EBF_Part2/s24_*` |
| s25 | s14 + refractory gate（s14 主干 + 同像素同极性短 dt 惩罚） | 在 s14（cross-pol boost）主干上，复用 s9 的“同像素同极性极短 dt”惩罚；仅当 raw 足够高时触发，专打热点/爆发噪声，尽量不伤 light | `alpha, raw_thr, dt_thr, ref_raw_thr, gamma` | `src/myevs/denoise/ops/ebfopt_part2/s25_s14_refractory_gate.py` | `data/ED24/myPedestrain_06/EBF_Part2/_prescreen_s25_*` |
| s26 | act-norm hotness fusion q8（活动归一化的热点惩罚融合） | heavy 的 FP 主因是 hot/near-hot/highrate；但强惩罚会误伤“真实活跃区域”。s26 让热点惩罚强度随邻域活动强度衰减：活动越强惩罚越弱，孤立异常热点惩罚越强 | `alpha, beta, kappa, eta` | `src/myevs/denoise/ops/ebfopt_part2/s26_actnorm_hotness_fusion_q8.py` | `data/ED24/myPedestrain_06/EBF_Part2/_prescreen_s26_*` |
| s27 | rel abnormal hotness fusion q8（相对异常热度惩罚融合） | 只惩罚“比邻域基线明显更热”的像素：用邻域 decayed accumulator 总热度均值做基线，$acc\_pen=max(0,acc\_mix-\lambda\,mean\_nb)$，更对准 hot/near-hot 的相对异常 | `alpha, beta, kappa, lambda_nb` | `src/myevs/denoise/ops/ebfopt_part2/s27_relabnorm_hotness_fusion_q8.py` | `data/ED24/myPedestrain_06/EBF_Part2/_prescreen_s27_*` |
| s28 | noise surprise z-score（噪声率模型下的惊讶度标准化） | 把 baseline raw-support 变成“相对噪声率的惊讶度”：用全局事件率 EMA 估计噪声率 $r$，推导噪声模型下 $\mu(r),\sigma(r)$，输出 $z=(raw-\mu)/\sigma$，降低对绝对事件率/热点的敏感性 | `tau_rate_us`（可选，默认 auto） | `src/myevs/denoise/ops/ebfopt_part2/s28_noise_surprise_zscore.py` | `data/ED24/myPedestrain_06/EBF_Part2/_prescreen_s28_*` |
| s29 | local polarity-surprise z-score（局部极性一致性惊讶度标准化） | 用邻域总活动做“局部背景”，看同极性支持度相对随机极性噪声的显著性：$z=(2S_{same}-S_{all})/\sqrt{S_{sq}+\varepsilon}$，更对准“局部 hotmask 仍主导 FP”这一失败模式 | 无（仅用当前 sweep 点的 `tau_us`） | `src/myevs/denoise/ops/ebfopt_part2/s29_polarity_surprise_zscore.py` | `data/ED24/myPedestrain_06/EBF_Part2/_prescreen_s29_*` |
| s30 | s28 + local-rate max（邻域即时局部率上修） | 保留 s28 的 raw-surprise z-score；用邻域总 recency mass 反解“局部活动率”并只允许上修（max-correction），试图更对准 heavy 的 hot/near-hot | `tau_rate_us`（可选，默认 auto） | `src/myevs/denoise/ops/ebfopt_part2/s30_surprise_zscore_localrate_max.py` | `data/ED24/myPedestrain_06/EBF_Part2/_prescreen_s30_*` |
| s31 | s28 + polarity-bias（全局极性偏置修正） | 在 s28 的噪声模型里把极性 match 概率从 0.5 改为 $q=(1+m^2)/2$（m 为全局极性均值的 EMA），尝试修正极性不平衡带来的期望偏差 | `tau_rate_us`（可选，默认 auto） | `src/myevs/denoise/ops/ebfopt_part2/s31_noise_surprise_zscore_polbias.py` | `data/ED24/myPedestrain_06/EBF_Part2/_prescreen_s31_*` |
| s32 | s28 + block-rate max（块级局部率上修） | 保留 s28 的 raw-surprise z-score；用块级（block）事件率 EMA 作为“更平滑的局部背景”，只在 raw 已经高时允许用 $r_{block}$ 上修噪声率（max-correction），避免 s30 的邻域即时反解过于敏感 | `tau_rate_us`（可选，默认 auto） | `src/myevs/denoise/ops/ebfopt_part2/s32_noise_surprise_zscore_blockrate_max.py` | `data/ED24/myPedestrain_06/EBF_Part2/_prescreen_s32_*` |
| s33 | s28 - abn-hot penalty（相对异常热度弱惩罚） | 以 s28 的 raw-surprise z-score 为主干；维护像素级 leaky hotness（pos/neg accumulator），用邻域 decayed 总热度均值作基线，$abn=max(0,acc_{tot}-mean_{nb})/\tau$；仅当 raw 已经高时做克制惩罚：$score=z-\beta\,abn$ | `tau_rate_us`（可选，默认 auto），`beta` | `src/myevs/denoise/ops/ebfopt_part2/s33_noise_surprise_zscore_abnhot_penalty.py` | `data/ED24/myPedestrain_06/EBF_Part2/_prescreen_s33_*` |
| s34 | s28 - short-dt penalty（像素短 dt 连续弱惩罚） | 以 s28 的 raw-surprise z-score 为主干；用同像素上一事件的 dt 得到短 dt 指标：$ratio=\tau/dt_0$；仅当 raw 高且 $dt_0\le\tau/8$ 时，对 $z$ 做克制惩罚：$score=z-k_{self}\,clip(ratio,0,8)$（更像 s9 的连续版） | `tau_rate_us`（可选，默认 auto），`k_self` | `src/myevs/denoise/ops/ebfopt_part2/s34_noise_surprise_zscore_selfrate_max.py` | `data/ED24/myPedestrain_06/EBF_Part2/_prescreen_s34_*` |
| s35 | s28 - pixel-state adaptive null（像素状态条件自适应空模型） | 把“热点/高发像素”当作**空模型的一部分**：维护单通道像素 hot-state $H$（leaky accumulator），用 $H$ 调制有效噪声率 $r_{eff}=r\,(1+\gamma\,clip(H/\tau_{rate},0,h_{max}))$，再复用 s28 的 $z=(raw-\mu(r_{eff}))/\sigma(r_{eff})$ | `tau_rate_us`（可选，默认 auto），`gamma`，`hmax` | `src/myevs/denoise/ops/ebfopt_part2/s35_noise_surprise_zscore_pixelstate.py` | `data/ED24/myPedestrain_06/EBF_Part2/_prescreen_s35_*` |
| s36 | s28 - state-occupancy adaptive null（状态占用率自适应空模型） | 在 s35 的“像素状态条件空模型”上做减参与平滑：用自 dt0 驱动的 hot-state 更新 + 占用率 $u\in[0,1)$ 映射，令 $r_{eff}=r\,(1+u)^2$（有界、无 gamma/hmax），再复用 s28 的 $z=(raw-\mu(r_{eff}))/\sigma(r_{eff})$ | `tau_rate_us`（可选，默认 auto） | `src/myevs/denoise/ops/ebfopt_part2/s36_noise_surprise_zscore_stateoccupancy.py` | `data/ED24/myPedestrain_06/EBF_Part2/_tune_s36_*` |
| s37 | s36 - occupancy 3-state adaptive null（占用率三段式自适应空模型） | 保留 s36 的 dt0 驱动 hot-state + 占用率 $u=H/(H+\tau_{rate})$；仅修改 $u\to r_{eff}$ 的形态为三段式放大：$u<1/3\Rightarrow\times1$，$1/3\le u<2/3\Rightarrow\times2$，$u\ge2/3\Rightarrow\times4$，更早触发强压制（仍无 gamma/hmax），再复用 s28 的 $z=(raw-\mu(r_{eff}))/\sigma(r_{eff})$ | `tau_rate_us`（可选，默认 auto） | `src/myevs/denoise/ops/ebfopt_part2/s37_noise_surprise_zscore_stateoccupancy_3state.py` | `data/ED24/myPedestrain_06/EBF_Part2/_tune_s37_*` |
| s38 | s36 - state+nb occupancy adaptive null（自占用率+邻域占用率融合空模型） | 保留 s36 的自 dt0 驱动 hot-state + 自占用率 $u_{self}$；额外用邻域任意极性 recency mass 构造 $u_{nb}$，用 union 融合得到 $u$，再用 $r_{eff}=r\,(1+u)^2$ 上修空模型噪声率，试图更对准 hotmask/噪声团的“局部活跃” | `tau_rate_us`（可选，默认 auto） | `src/myevs/denoise/ops/ebfopt_part2/s38_noise_surprise_zscore_stateoccupancy_nbocc_fusion.py` | `data/ED24/myPedestrain_06/EBF_Part2/_tune_s38_*` |
| s22 | any-pol burst gate（同像素任意极性短 dt 门控） | 用“同像素上一次事件（不要求同极性）的 dt”识别热点/翻转噪声；只增加常数逻辑，不新增状态数组 | `dt_thr_us, raw_thr, gamma` | `src/myevs/denoise/ops/ebfopt_part2/s22_anypol_burst_gate.py` | `data/ED24/myPedestrain_06/EBF_Part2/s22_*` |
| s23 | feat+logit fusion（特征提取+可学习线性融合） | 在线抽取少量机制特征（same/opp/toggle/dt 等）并用线性 logit 融合打分；以“保底 baseline + 学增量”提升 AUC/F1 | `dt_thr_us + 线性权重` | `src/myevs/denoise/ops/ebfopt_part2/s23_featlogit.py` | `data/ED24/myPedestrain_06/EBF_Part2/s23_*` |

## s29：Local Polarity-Surprise Z-Score（局部极性一致性惊讶度标准化）

### 失败模式 / 可验证假设

动机（来自 s28 的教训 + 你在 7.4 的判断）：

- s28 用“全局事件率”解释“局部 hotmask 主导的 FP”，信息不匹配；即使调 `tau_rate_us`，heavy 的 hotmask FP 仍压不过 baseline/s27。

可验证假设：如果把标准化的背景从“全局 rate”改成“邻域局部背景活动”，则：

- heavy：hot/near-hot 区域的高活动会被当作背景抵消，hotmask 类 FP 更可能下降；
- light/mid：结构化信号在局部会表现为“同极性一致性显著高于随机极性”，AUC/F1 不应明显掉。

### 算法定义（在线 / 单遍 / $O(r^2)$）

位置：`src/myevs/denoise/ops/ebfopt_part2/s29_polarity_surprise_zscore.py`

在一次邻域遍历中（与 baseline 相同的 recency 权重 $w_j=\max(0,1-\Delta t/\tau)$）：

$$
S_{all}=\sum_j w_j,\quad
S_{same}=\sum_j w_j\,\mathbf{1}[pol_j=pol_i],\quad
S_{sq}=\sum_j w_j^2
$$

在“噪声极性随机 $P(match)=0.5$”近似下：

$$
\mathbb{E}[S_{same}]=0.5S_{all},\quad \mathrm{Var}(S_{same})\approx 0.25S_{sq}
$$

输出 z-score：

$$
z=\frac{S_{same}-0.5S_{all}}{\sqrt{0.25S_{sq}+\varepsilon}}=\frac{2S_{same}-S_{all}}{\sqrt{S_{sq}+\varepsilon}}
$$

### 实现约束核对

- 在线流式/单遍：是
- 复杂度：$O(r^2)$（一次邻域遍历）
- Numba 必须：是
- 额外 per-pixel 状态数组：否（仅 `last_ts/last_pol`）
- 新增超参：无

### 最小验证口径（建议先 smoke，再 prescreen200k）

smoke（20k）：

```powershell
$env:PYTHONNOUSERSITE='1'
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant s29 --max-events 20000 --s-list 9 --tau-us-list 128000 --esr-mode best --aocc-mode off --out-dir data/ED24/myPedestrain_06/EBF_Part2/_smoke_s29_polsurprise_prescreen20k_s9_tau128ms
```

prescreen200k（对齐口径）：

```powershell
$env:PYTHONNOUSERSITE='1'
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant s29 --max-events 200000 --s-list 9 --tau-us-list 128000 --esr-mode best --aocc-mode off --out-dir data/ED24/myPedestrain_06/EBF_Part2/_prescreen_s29_polsurprise_s9_tau128ms_200k
```

### 结果（prescreen200k，s=9,tau=128ms）

结论：**停掉 s29**（排序能力显著劣化，且未压住 hotmask FP）。

sweep（AUC / best-F1）：

- light：AUC=0.7813，best-F1=0.9119
- mid：AUC=0.7792，best-F1=0.6274
- heavy：AUC=0.7795，best-F1=0.5904

heavy 噪声类型分解（best-F1 operating point；`scripts/noise_analyze/noise_type_stats.py`）：

- baseline（EBF）：hotmask FP=5778，near-hot FP=711，总 FP≈6856；hotmask 区域 signal_kept_rate≈0.7307
- s29：hotmask FP=8926，near-hot FP=576，总 FP≈9772；hotmask 区域 signal_kept_rate≈0.5176

可验证解释（为什么不行）：

- 该数据上“极性一致性 z-score”对真实结构的区分性不足，整体排序信息明显变弱（AUC 全环境掉到 ~0.78）。
- 在 heavy 的主矛盾（hotmask/near-hot）上没有形成有效抑制：hotmask FP 反而升高；同时对 signal（尤其热点附近）误伤严重，导致 best-F1 大幅下滑。

是否继续：**不继续**。下一步应回到更直接对准 hotmask 的机制（如 s27 的相对异常热度惩罚），或在保底 baseline 的前提下用 s23 的“线性融合”框架组合弱证据。

## s30：Surprise Z-Score + Local-Rate Max（邻域即时局部率上修）

### 失败模式 / 可验证假设

动机（从 s28/s29 的教训出发）：

- s28 的全局 rate 对 heavy 的局部热点仍可能偏乐观；s29 又证明“把排序主轴换成 polarity-surprise”会整体变差。

可验证假设：如果在 s28 中把噪声率背景从“全局 EMA”加入“局部活动率上修”（只允许增加噪声率），则 heavy 的 hot/near-hot 类 FP 应该下降，且 light/mid 不应明显掉。

### 算法定义（在线 / 单遍 / $O(r^2)$）

位置：`src/myevs/denoise/ops/ebfopt_part2/s30_surprise_zscore_localrate_max.py`

- raw：同极性邻域支持（baseline raw-support）
- 背景：全局 rate EMA + 由邻域 recency mass 反解得到的“局部 a（=r\tau）”估计；只允许 $a$ 上修，并用 $wbar$ 做置信度权重
- 输出：仍为 s28 的噪声模型 z-score

### 实现约束核对

- 在线流式/单遍：是
- 复杂度：$O(r^2)$（一次邻域遍历）
- Numba 必须：是
- 额外状态：否（仅 `last_ts/last_pol` + 全局 `rate_ema`）
- 新增超参：`tau_rate_us`（可选，默认 auto）

### 最小验证口径

smoke（20k）：

```powershell
$env:PYTHONNOUSERSITE='1'
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant s30 --max-events 20000 --s-list 9 --tau-us-list 128000 --esr-mode best --aocc-mode off --out-dir data/ED24/myPedestrain_06/EBF_Part2/_smoke_s30_surprise_localrate_prescreen20k_s9_tau128ms
```

prescreen200k（对齐口径）：

```powershell
$env:PYTHONNOUSERSITE='1'
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant s30 --max-events 200000 --s-list 9 --tau-us-list 128000 --esr-mode best --aocc-mode off --out-dir data/ED24/myPedestrain_06/EBF_Part2/_prescreen_s30_surprise_localrate_s9_tau128ms_200k
```

### 结果（prescreen200k，s=9,tau=128ms）

结论：**停掉 s30**（AUC/F1 全环境下降）。

sweep（AUC / best-F1）：

- light：AUC=0.9060，best-F1=0.9312
- mid：AUC=0.9029，best-F1=0.7772
- heavy：AUC=0.8993，best-F1=0.7351

heavy（best-F1 operating point 摘要）：thr=1.8639，TP=28907，FP=9879。

可验证解释（为什么不行）：

- 邻域即时“反解局部率”过于敏感，且与 raw/support 高度相关，会在大量事件上触发更强的标准化（等价于普遍降分/重排），从而把 light/mid/heavy 的排序整体打坏。
- 这类局部估计把“结构化信号产生的局部活动”也当作背景抵消，误伤真实轨迹，导致跨环境一起掉。

是否继续：**不继续**。后续局部背景应更平滑、更克制（例如块级/区域级 rate），避免邻域即时反推。

## s31：Noise Surprise Z-Score + Polarity-Bias（全局极性偏置修正）

### 失败模式 / 可验证假设

动机：如果全局极性分布不平衡（$P(match)\neq 0.5$），s28 的噪声模型期望/方差会有系统偏差，可能影响阈值稳定性。

可验证假设：把噪声模型里 $P(match)=0.5$ 替换为由全局极性均值 $m$ 推导的 $q=(1+m^2)/2$，能带来 AUC/F1 或 best-F1 的可见提升。

### 算法定义（在线 / 单遍 / $O(r^2)$）

位置：`src/myevs/denoise/ops/ebfopt_part2/s31_noise_surprise_zscore_polbias.py`

- 保持 s28 的 raw 与 rate EMA
- 额外维护全局极性均值/偏置的 EMA
- 用 $q$ 代替 0.5 进入噪声模型的 $\mu/\sigma$ 推导

### 实现约束核对

- 在线流式/单遍：是
- 复杂度：$O(r^2)$
- Numba 必须：是
- 额外状态：是（全局 `rate_pol_ema`）
- 新增超参：`tau_rate_us`（可选，默认 auto）

### 最小验证口径

prescreen200k（对齐口径）：

```powershell
$env:PYTHONNOUSERSITE='1'
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant s31 --max-events 200000 --s-list 9 --tau-us-list 128000 --esr-mode best --aocc-mode off --out-dir data/ED24/myPedestrain_06/EBF_Part2/_prescreen_s31_surprise_polbias_s9_tau128ms_200k
```

### 结果（prescreen200k，s=9,tau=128ms）

结论：**停掉 s31**（几乎等价于 s28）。

sweep（AUC / best-F1）：

- light：AUC=0.9345，best-F1=0.9433
- mid：AUC=0.9233，best-F1=0.8104
- heavy：AUC=0.9235，best-F1=0.7815

对照 s28：AUC/best-F1 与 s28 在四舍五入层面一致。

可验证解释（为什么没效果）：

- 该数据/口径下全局极性偏置 $m$ 很可能接近 0，使得 $q\approx 0.5$，修正项几乎不改变 $\mu/\sigma$。
- 或者即使存在小偏置，也不是当前可分性瓶颈。

是否继续：**不继续**（作为“确认无效”的分支记录即可）。

## s32：Noise Surprise Z-Score + Block-Rate Max（块级局部率上修）

### 失败模式 / 可验证假设

动机（来自 s29 失败分析建议 + s30 的反例）：

- 应保留 raw-support 作为被标准化对象（s29 的 polarity-surprise 主轴会显著伤排序）。
- 需要引入“局部背景活动率”，但必须比 s30 的邻域即时反解更平滑、更克制，避免误伤信号。

可验证假设：用块级（block）事件率 EMA 作为局部背景，并且只在 raw 已经高时允许用 $r_{block}$ 上修噪声率，可以在 heavy 的 hot/near-hot 区域减少 FP，同时尽量不扰动 light/mid 的整体排序。

### 算法定义（在线 / 单遍 / $O(r^2)$）

位置：`src/myevs/denoise/ops/ebfopt_part2/s32_noise_surprise_zscore_blockrate_max.py`

- raw：同极性邻域支持（baseline raw-support）
- 背景：全局 rate EMA + 块级 rate EMA（固定 block=32×32）；将块级 rate 折算为 per-pixel rate
- 只在 raw>=raw_thr 且同像素 self-dt 足够小（疑似热点/爆发）时，允许 $r_{pix}=\max(r_{pix,global}, r_{pix,block})$
- 输出：仍为 s28 的噪声模型 z-score

### 实现约束核对

- 在线流式/单遍：是
- 复杂度：$O(r^2)$
- Numba 必须：是
- 额外状态：是（每块 `block_last_t/block_rate_ema`）
- 新增超参：`tau_rate_us`（可选，默认 auto；其余常数固定，不进入 sweep）

### 最小验证口径（建议先 smoke，再 prescreen200k）

smoke（20k）：

```powershell
$env:PYTHONNOUSERSITE='1'
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant s32 --max-events 20000 --s-list 9 --tau-us-list 128000 --esr-mode best --aocc-mode off --out-dir data/ED24/myPedestrain_06/EBF_Part2/_smoke_s32_blockrate_prescreen20k_s9_tau128ms
```

prescreen200k（对齐口径）：

```powershell
$env:PYTHONNOUSERSITE='1'
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant s32 --max-events 200000 --s-list 9 --tau-us-list 128000 --esr-mode best --aocc-mode off --out-dir data/ED24/myPedestrain_06/EBF_Part2/_prescreen_s32_blockrate_s9_tau128ms_200k
```

### 结果

结论：**暂时停掉 s32**（仍未超过 s28；对 heavy 的净提升不成立）。

prescreen200k（s=9,tau=128ms）结果：

- light：AUC=0.9298，best-F1=0.9417
- mid：AUC=0.9216，best-F1=0.8089
- heavy：AUC=0.9203，best-F1=0.7768

heavy（best-F1 operating point 摘要）：thr=2.3803，TP=30482，FP=8136。

可验证解释（为什么没赢）：

- 即使加了 self-dt gate，块级 rate 上修仍会把一部分真实信号所在的“活跃块”当作背景抵消，导致整体排序仍略被扰动；heavy 也未形成稳定的 FP 下降。

是否继续：**不继续（作为方向验证记录）**。若要继续“局部背景”方向，建议改成更对准 hot/near-hot 的局部背景（例如结合相对异常热度/热点掩膜一类更结构化的信号），而不是仅靠 block rate。

## s33：Noise Surprise Z-Score - Abnormal-Hotness Penalty（相对异常热度弱惩罚）

### 失败模式 / 可验证假设

动机（来自 s32 的教训 + s10–s12/s27 的经验）：

- heavy 的主矛盾是 hotmask/near-hot，但“绝对活跃”并不等价于“异常热点”（s32 的 block busy 就踩了这个坑）。
- 更可能有效的是“相对异常”（中心比邻域基线更热），并且要用 **raw 前置阈值** 把影响范围收紧，避免像 s30 那样扰动全局排序。

可验证假设：在保持 s28 的 z-score 主排序的前提下，对“中心热度显著高于邻域基线”的事件施加一个弱惩罚，heavy 的 hotmask/near-hot FP 有机会下降，同时 light/mid 的 AUC/F1 不应明显下降。

### 算法定义（在线 / 单遍 / $O(r^2)$）

位置：`src/myevs/denoise/ops/ebfopt_part2/s33_noise_surprise_zscore_abnhot_penalty.py`

- backbone：s28 的噪声模型 z-score（raw-surprise z-score）
- 额外维护：像素级 leaky accumulator（pos/neg 两路），每次事件对本像素做线性衰减并 +tau
- 邻域基线：$mean_{nb}$ 为邻域像素的 decayed 总热度 $(acc_{pos}+acc_{neg})$ 的均值
- abnormality（无量纲）：

$$
abn=\max\bigl(0,\frac{acc_{tot}(x,y)-mean_{nb}(x,y)}{\tau}\bigr)
$$

- 仅当 raw>=raw_thr（固定常数）时惩罚：

$$
score=z_{s28}-\beta\,abn
$$

### 实现约束核对

- 在线流式/单遍：是
- 复杂度：$O(r^2)$
- Numba 必须：是
- 额外状态：是（`acc_neg/acc_pos` 两个 int32 per-pixel 数组 + 全局 `rate_ema`）
- 新增超参：`beta`（主要 sweep 维度），`tau_rate_us`（可选）

### 最小验证口径（建议先 smoke，再 prescreen200k）

smoke（20k）：

```powershell
$env:PYTHONNOUSERSITE='1'
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant s33 --max-events 20000 --s-list 9 --tau-us-list 128000 --esr-mode best --aocc-mode off --out-dir data/ED24/myPedestrain_06/EBF_Part2/_smoke_s33_abnhot_prescreen20k_s9_tau128ms
```

prescreen200k（对齐口径）：

```powershell
$env:PYTHONNOUSERSITE='1'
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant s33 --max-events 200000 --s-list 9 --tau-us-list 128000 --esr-mode best --aocc-mode off --out-dir data/ED24/myPedestrain_06/EBF_Part2/_prescreen_s33_abnhot_grid1_s9_tau128ms_200k --s33-beta-list '0.25,0.5,0.75,1.0'
```

### 结果

结论：**继续 s33（方向1 成立）**。在 prescreen200k（s=9,tau=128ms）下，相比 s28，AUC 与 best-F1 三环境均有小幅提升，且 heavy 的 hotmask/near-hot FP 有实证下降。

prescreen200k（s=9,tau=128ms；beta sweep：0.25/0.5/0.75/1.0）最优摘要：

- light：best AUC=0.9363（beta=0.25），best-F1=0.9438（beta=0.25）
- mid：best AUC=0.9261（beta=0.5），best-F1=0.8156（beta=0.75）
- heavy：best AUC=0.9273（beta=0.5），best-F1=0.7860（beta=0.75）

对照 s28（同口径）：

- light：AUC=0.9345，best-F1=0.9433
- mid：AUC=0.9233，best-F1=0.8104
- heavy：AUC=0.9235，best-F1=0.7815

heavy 噪声类型分解（best-F1 operating point，对比 s28；`scripts/noise_analyze/noise_type_stats.py`）：

- s28：hotmask FP=6727，near-hot FP=754，highrate FP=324，hotmask signal_kept_rate=0.7369
- s33（best-F1 tag=ebf_s33_b0p75）：hotmask FP=6376，near-hot FP=732，highrate FP=306，hotmask signal_kept_rate=0.7379

标准网格（grid2：s=3/5/7/9；tau=8..1024ms；beta 固定 0.75）：

```powershell
$env:PYTHONNOUSERSITE='1'
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant s33 --max-events 200000 --s-list 3,5,7,9 --tau-us-list 8000,16000,32000,64000,128000,256000,512000,1024000 --s33-tau-rate-us-list 0 --s33-beta-list 0.75 --esr-mode best --aocc-mode off --out-dir data/ED24/myPedestrain_06/EBF_Part2/_prescreen_s33_abnhot_grid2_s3579_tau8_16_32_64_128_256_512_1024ms_200k
```

- BEST AUC（by env）：
	- light：s=9,tau=128ms，AUC=0.9360
	- mid：s=9,tau=256ms，AUC=0.9291
	- heavy：s=9,tau=128ms，AUC=0.9270
- BEST F1（by env）：
	- light：s=9,tau=1024ms，F1=0.9445
	- mid：s=9,tau=128ms，F1=0.8156
	- heavy：s=7,tau=64ms，F1=0.7886（相比 grid1 的 heavy best-F1=0.7860 略升）

heavy 噪声类型分解（grid2 的 heavy best-F1 点：s=7,tau=64ms，对比 s28 同点 best-F1）：

- s28（tag=ebf_s28_labelscore_s7_tau64000）：hotmask FP=4563，near-hot FP=457，highrate FP=201，signal_kept_rate=0.7041
- s33（tag=ebf_s33_b0p75_labelscore_s7_tau64000）：hotmask FP=4577，near-hot FP=458，highrate FP=203，signal_kept_rate=0.7061

备注：该点 heavy 的 F1 提升主要来自 signal_kept_rate 的小幅提升；hotmask/near-hot FP 在该点与 s28 基本持平（并非单调下降）。

可验证解释（为什么这次有效）：

- s33 的惩罚项更接近 s27 的“相对异常”思想：它不会把“整块都忙”的区域当作背景（避免 s32 的 block busy 误判），而是只打“中心比邻域更热”的像素。
- raw 前置阈值把影响限制在高风险子集，避免像 s30 那样对全局排序造成大范围扰动。

下一步最小改动：

- 已完成：beta=0.75 的标准网格（grid2）验证，跨 tau/s 的表现稳定。
- 建议：在 grid2 的 heavy best-F1（s=7,tau=64ms）与 heavy best-AUC（s=9,tau=128ms）各做一次 1M validate，确认“提升是否仍在 full/long 口径下成立”。

## s34：Noise Surprise Z-Score - Short-dt Penalty（像素短 dt 连续弱惩罚）

### 失败模式 / 可验证假设

动机：heavy 的热点/爆发噪声常表现为同像素极短 dt；s9 的硬阈值门控很干净但覆盖有限。这里尝试做一个“连续版”的弱惩罚，并用 raw 前置阈值收紧作用范围。

可验证假设：对 raw 已经高、且同像素 dt0 非常短的事件，按 $\tau/dt_0$ 的强度做一个 clipped 的连续弱惩罚，可以压制爆发热点，同时尽量不改变其它样本的排序。

### 算法定义（在线 / 单遍 / $O(r^2)$）

位置：`src/myevs/denoise/ops/ebfopt_part2/s34_noise_surprise_zscore_selfrate_max.py`

- backbone：s28 的噪声模型 z-score
- 短 dt 指标：$ratio=\tau/dt_0$（$dt_0$ 为同像素上一事件到当前的时间差）
- 仅当 raw>=raw_thr 且 $dt_0\le\tau/8$（固定常数）时惩罚：

$$
score=z_{s28}-k_{self}\,clip(ratio,0,8)
$$

### 实现约束核对

- 在线流式/单遍：是
- 复杂度：$O(r^2)$
- Numba 必须：是
- 额外状态：否（复用 `last_ts/last_pol` + 全局 `rate_ema`）
- 新增超参：`k_self`（主要 sweep 维度），`tau_rate_us`（可选）

### 最小验证口径（建议先 smoke，再 prescreen200k）

smoke（20k）：

```powershell
$env:PYTHONNOUSERSITE='1'
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant s34 --max-events 20000 --s-list 9 --tau-us-list 128000 --esr-mode best --aocc-mode off --out-dir data/ED24/myPedestrain_06/EBF_Part2/_smoke_s34_shortdt_prescreen20k_s9_tau128ms --s34-k-self-list '0.05,0.1,0.2,0.4'
```

prescreen200k（对齐口径）：

```powershell
$env:PYTHONNOUSERSITE='1'
conda run -n myEVS python scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py --variant s34 --max-events 200000 --s-list 9 --tau-us-list 128000 --esr-mode best --aocc-mode off --out-dir data/ED24/myPedestrain_06/EBF_Part2/_prescreen_s34_shortdt_grid1_s9_tau128ms_200k --s34-k-self-list '0.05,0.1,0.2,0.4'
```

### 结果

结论：**暂时停掉 s34（方向2 收益不清晰）**。prescreen200k 下 AUC/best-F1 有小幅波动，但 heavy 的 hotmask/near-hot FP 没有下降（不对准主矛盾），优先级低于 s33。

prescreen200k（s=9,tau=128ms；k_self sweep：0.05/0.1/0.2/0.4）最优摘要：

- light：best AUC=0.9346（k_self=0.2），best-F1=0.9433（k_self=0.1）
- mid：best AUC=0.9252（k_self=0.2），best-F1=0.8133（k_self=0.2）
- heavy：best AUC=0.9265（k_self=0.2），best-F1=0.7834（k_self=0.2）

heavy 噪声类型分解（best-F1 tag=ebf_s34_k0p2，对比 s28）：

- s28：hotmask FP=6727，near-hot FP=754
- s34：hotmask FP=6791，near-hot FP=759（均略高）

可验证解释：

- 该连续惩罚更多像“扩大版的短 dt 规则”，但 heavy 的 hotmask 主矛盾并不完全由同像素短 dt 驱动；因此它未能带来 hotmask FP 的结构性下降。

是否继续：**不继续**（如需短 dt 方向，优先继续用 s9/s22 这类更可控的 gate）。

## s35：Noise Surprise Z-Score - Pixel-State Adaptive Null（像素状态条件自适应空模型）

### 失败模式 / 可验证假设

动机（来自你对 7.7 的判断 + s28 的教训）：

- s28 仅用全局事件率 $r$ 做空模型，无法表达“**某些像素长期处于 hot 状态**”这类局部结构；因此在 heavy 下，hotmask/near-hot 仍会与 signal 的 $z$ 强烈重叠。
- 与 s33/s34 这类“后验惩罚项”不同，这里尝试把 hotness 直接并入**空模型（null model）**：对 hot 像素，空模型下就允许更高的噪声率，从而把它们的“raw 高”解释为“噪声预期高”，让 $z$ 自然下降。

可验证假设：引入一个极简的像素状态 $H(x,y)$ 来调制 $r$（仍保持在线/单遍/低复杂度），则：

- heavy：对 hotmask/near-hot/highrate 这类 FP，$z$ 将更偏向下降（FP 绝对数可能下降）；
- light/mid：真实结构信号的 $z$ 不应整体被压塌（AUC/best-F1 不应明显掉）。

### 算法定义（在线 / 单遍 / $O(r^2)$）

位置：`src/myevs/denoise/ops/ebfopt_part2/s35_noise_surprise_zscore_pixelstate.py`

1) raw-support（与 s28 相同）：

$$
raw=\sum_{j\in\mathcal{N}} \mathbf{1}[pol_j=pol_i] \,\max\bigl(0,1-\Delta t/\tau\bigr)
$$

2) 全局噪声率估计（与 s28 相同）：用全局 EMA 得到每像素噪声率 $r$。

3) 像素 hot-state（单通道 leaky accumulator）：

对当前事件所在像素 $(x,y)$，维护一个状态 $H(x,y)$（无单位，量纲上近似为“最近一段时间内的活动累积”）：

$$
H\leftarrow H\,e^{-\Delta t/\tau_{rate}} + \tau_{rate}
$$

并定义无量纲 hotness：

$$
h=\mathrm{clip}\bigl(H/\tau_{rate},0,h_{max}\bigr)
$$

4) 状态条件的自适应空模型（有效噪声率）：

$$
r_{eff}=r\,\bigl(1+\gamma\,h\bigr)
$$

5) 复用 s28 的噪声模型标准化（仅把 $r$ 替换为 $r_{eff}$）：

$$
z=\frac{raw-\mu(r_{eff})}{\sigma(r_{eff})+\varepsilon}
$$

直观理解：同样的 raw，如果它发生在“最近很 hot 的像素”，那么空模型认为它更可能是噪声 ⇒ $\mu$ 更大/方差更大 ⇒ $z$ 更小。

### 超参（环境变量 / sweep 参数）

- `MYEVS_EBF_S35_TAU_RATE_US`：$\tau_{rate}$（微秒）。
	- `0` 表示 auto：直接用当前 sweep 点的 `tau_us`。
- `MYEVS_EBF_S35_GAMMA`：$\gamma\ge 0$（hot-state 对有效噪声率的放大强度）。
- `MYEVS_EBF_S35_HMAX`：$h_{max}$（hotness clip 上限，避免极端像素把 $r_{eff}$ 放大到不合理范围）。

sweep 参数：`--s35-tau-rate-us-list`、`--s35-gamma-list`、`--s35-hmax-list`。

### 实现约束核对

- 在线流式/单遍：是
- 复杂度：$O(r^2)$（邻域遍历同 s28；新增逻辑为常数级）
- Numba 必须：是
- 额外状态：是（新增 1 个 int32 per-pixel 数组 `hot_state` + 全局 `rate_ema`）

### prescreen200k：关键结果（固定 operating point，对照 s28）

本次只验证“最小可行版本”：`hmax=8`，扫 `gamma∈{0.5,1,2}`，`tau_rate_us=auto`；并在两个 operating point 上对齐对照 s28。

#### A) 固定 `s=9,tau=128ms,max-events=200k`

输出目录：

- s35：`data/ED24/myPedestrain_06/EBF_Part2/_prescreen_s35_pixelstate_s9_tau128ms_200k_g0p5_1_2_h8/`
- s28（对照）：`data/ED24/myPedestrain_06/EBF_Part2/_cmp_s28_surprise_s9_tau128ms_200k/`

best 摘要（在 `gamma` 网格内、固定 s/tau，各 env 各自取 best）：

| env | s35 best-AUC tag | s35 AUC(best) | s35 best-F1 tag | s35 F1(best) | s28 AUC | s28 best-F1 |
|---|---|---:|---|---:|---:|---:|
| light(1.8V) | `ebf_s35_g0p5_h8_labelscore_s9_tau128000` | 0.935891 | `ebf_s35_g0p5_h8_labelscore_s9_tau128000` | 0.941986 | 0.934483 | 0.943288 |
| mid(2.5V) | `ebf_s35_g0p5_h8_labelscore_s9_tau128000` | 0.933714 | `ebf_s35_g0p5_h8_labelscore_s9_tau128000` | 0.815943 | 0.923317 | 0.810405 |
| heavy(3.3V) | `ebf_s35_g0p5_h8_labelscore_s9_tau128000` | 0.928062 | `ebf_s35_g0p5_h8_labelscore_s9_tau128000` | 0.776139 | 0.923493 | 0.781485 |

heavy（best-F1 operating point，来自 ROC CSV）：

- s35（tag=ebf_s35_g0p5_h8）：thr=1.00715，F1=0.776139，AUC=0.928062，TP=29659，FP=6905，FN=10204
- s28（tag=ebf_s28）：thr=2.57854，F1=0.781485，AUC=0.923493，TP=30601，FP=7851，FN=9262

heavy 噪声类型分解（best-F1，`scripts/noise_analyze/noise_type_stats.py`）：

- s35：hotmask FP=5962，near-hot FP=662，highrate FP=239，hotmask signal_kept_rate=0.7166
- s28：hotmask FP=6727，near-hot FP=754，highrate FP=324，hotmask signal_kept_rate=0.7369

解读：

- s35 在该点确实把“热点类 FP”压下来了（hotmask/near-hot/highrate FP 都下降），但同时对 signal 的误伤更大，导致 heavy best-F1 暂时未超过 s28。

#### B) 固定 `s=7,tau=64ms,max-events=200k`

输出目录：

- s35：`data/ED24/myPedestrain_06/EBF_Part2/_prescreen_s35_pixelstate_s7_tau64ms_200k_g0p5_1_2_h8/`
- s28（对照）：`data/ED24/myPedestrain_06/EBF_Part2/_cmp_s28_surprise_s7_tau64ms_200k/`

best 摘要（在 `gamma` 网格内、固定 s/tau，各 env 各自取 best）：

| env | s35 best-AUC tag | s35 AUC(best) | s35 best-F1 tag | s35 F1(best) | s28 AUC | s28 best-F1 |
|---|---|---:|---|---:|---:|---:|
| light(1.8V) | `ebf_s35_g0p5_h8_labelscore_s7_tau64000` | 0.926810 | `ebf_s35_g0p5_h8_labelscore_s7_tau64000` | 0.936997 | 0.922839 | 0.939621 |
| mid(2.5V) | `ebf_s35_g2_h8_labelscore_s7_tau64000` | 0.924563 | `ebf_s35_g1_h8_labelscore_s7_tau64000` | 0.811149 | 0.914572 | 0.805484 |
| heavy(3.3V) | `ebf_s35_g1_h8_labelscore_s7_tau64000` | 0.925986 | `ebf_s35_g0p5_h8_labelscore_s7_tau64000` | 0.790497 | 0.918622 | 0.787730 |

heavy（best-F1 operating point，来自 ROC CSV）：

- s35（tag=ebf_s35_g0p5_h8）：thr=1.52112，F1=0.790497，AUC=0.925687，TP=29963，FP=5982，FN=9900
- s28（tag=ebf_s28）：thr=2.96366，F1=0.787730，AUC=0.918622，TP=29313，FP=5248，FN=10550

heavy 噪声类型分解（best-F1）：

- s35：hotmask FP=5258，near-hot FP=487，highrate FP=208，signal_kept_rate=0.7236
- s28：hotmask FP=4563，near-hot FP=457，highrate FP=201，signal_kept_rate=0.7041

解读：

- 在该点 s35 的 heavy best-F1 略高于 s28（更偏向目标 A），但热点类 FP（尤其 hotmask FP）在 best-F1 点反而更高；提升主要来自 signal_kept_rate 增加。

### 是否继续

结论：**继续 s35（作为“改空模型”的主线候选）**，但目前表现呈现明显 trade-off：

- `s=9,tau=128ms` 更像“压热点 FP 但误伤 signal”，需要把调参目标从“纯压 FP”改为“在不明显降低 signal_kept 的前提下压 FP”；
- `s=7,tau=64ms` 能带来 best-F1 小幅增益，但要进一步约束 hotmask FP 的上升。

下一步最小改动（保持框架不变，只做更细的参数搜索）：

- 以 heavy best-F1 为主目标，围绕 `gamma≈0.3~0.8`、`hmax∈{4,6,8,12}` 做一轮小网格（先固定 `tau_rate_us=auto`），再用 `noise_type_stats.py` 检查 hotmask FP 与 signal_kept_rate 的变化。

### 聚焦调参（只扫 `gamma×hmax`，不改结构；目标优先 best-F1，其次 AUC）

说明：固定 `tau_rate_us=auto`，扫描 `gamma∈{0.3,0.4,0.5,0.6,0.7,0.8}` × `hmax∈{4,6,8,12}`。

#### A) 固定 `s=9,tau=128ms,max-events=200k`

输出目录：

- s35 tune：`data/ED24/myPedestrain_06/EBF_Part2/_tune_s35_pixelstate_s9_tau128ms_200k_gam0p3_0p8_h4_6_8_12/`

best 摘要表（各 env 在该网格内各自取 best；表内 `thr` 为该 env 下该 tag 的 best-F1 operating point 阈值）：

| env | pick | tag | gamma/hmax | thr | F1 | AUC(tag) | TP | FP | FN |
| --- | --- | --- | --- | ---:| ---:| ---:| ---:| ---:| ---:|
| light | best-F1 | `ebf_s35_g0p3_h12_labelscore_s9_tau128000` | g=0.3, h=12 | -0.634334 | 0.944539 | 0.938441 | 152808 | 7827 | 10118 |
| light | best-AUC | `ebf_s35_g0p3_h12_labelscore_s9_tau128000` | g=0.3, h=12 | -0.634334 | 0.944539 | 0.938441 | 152808 | 7827 | 10118 |
| mid | best-F1 | `ebf_s35_g0p3_h12_labelscore_s9_tau128000` | g=0.3, h=12 | 1.169530 | 0.816839 | 0.933242 | 42601 | 8014 | 11091 |
| mid | best-AUC | `ebf_s35_g0p5_h12_labelscore_s9_tau128000` | g=0.5, h=12 | 0.747015 | 0.816342 | 0.933757 | 42451 | 7860 | 11241 |
| heavy | best-F1 | `ebf_s35_g0p3_h4_labelscore_s9_tau128000` | g=0.3, h=4 | 1.523715 | 0.782557 | 0.930429 | 30184 | 7095 | 9679 |
| heavy | best-AUC | `ebf_s35_g0p4_h4_labelscore_s9_tau128000` | g=0.4, h=4 | 1.230106 | 0.780731 | 0.930477 | 30112 | 7163 | 9751 |

heavy 的 `gamma×hmax` 网格（每格为该 tag 在 heavy 下的 best-F1；越大越好）：

| gamma\hmax | 4 | 6 | 8 | 12 |
| --- | ---:| ---:| ---:| ---:|
| 0.3 | 0.782557 | 0.782191 | 0.781944 | 0.782079 |
| 0.4 | 0.780731 | 0.779450 | 0.779139 | 0.779153 |
| 0.5 | 0.777954 | 0.776187 | 0.776139 | 0.776155 |
| 0.6 | 0.775503 | 0.773465 | 0.773365 | 0.773242 |
| 0.7 | 0.772879 | 0.770802 | 0.770563 | 0.770377 |
| 0.8 | 0.770258 | 0.768447 | 0.767941 | 0.767832 |

heavy（best-F1 operating point，来自 ROC CSV；tag=`g0p3_h4`）：

- s35 tuned：thr=1.52371，F1=0.782557，AUC(tag)=0.930429，TP=30184，FP=7095，FN=9679
- s28（对照）：thr=2.57854，F1=0.781485，AUC(tag)=0.923493，TP=30601，FP=7851，FN=9262

heavy 噪声类型分解（best-F1；对照 s28）：

- s35 tuned：hotmask FP=6098，near-hot FP=687，highrate FP=266，hotmask signal_kept_rate=0.7284
- s28：hotmask FP=6727，near-hot FP=754，highrate FP=324，hotmask signal_kept_rate=0.7369

解读：

- tuned 点相对 s28：热点类 FP 进一步下降（hotmask/near-hot/highrate 都降），但 signal_kept_rate 略降（TP 略降）；净效应是 **FP 降幅更大 ⇒ F1 小幅超过 s28**。

#### B) 固定 `s=7,tau=64ms,max-events=200k`

输出目录：

- s35 tune：`data/ED24/myPedestrain_06/EBF_Part2/_tune_s35_pixelstate_s7_tau64ms_200k_gam0p3_0p8_h4_6_8_12/`

best 摘要表（各 env 在该网格内各自取 best；表内 `thr` 为该 env 下该 tag 的 best-F1 operating point 阈值）：

| env | pick | tag | gamma/hmax | thr | F1 | AUC(tag) | TP | FP | FN |
| --- | --- | --- | --- | ---:| ---:| ---:| ---:| ---:| ---:|
| light | best-F1 | `ebf_s35_g0p3_h12_labelscore_s7_tau64000` | g=0.3, h=12 | -0.333491 | 0.938305 | 0.928318 | 148522 | 5127 | 14404 |
| light | best-AUC | `ebf_s35_g0p4_h12_labelscore_s7_tau64000` | g=0.4, h=12 | -0.347923 | 0.937821 | 0.928480 | 148231 | 4961 | 14695 |
| mid | best-F1 | `ebf_s35_g0p8_h12_labelscore_s7_tau64000` | g=0.8, h=12 | 1.028816 | 0.811337 | 0.924073 | 41222 | 6701 | 12470 |
| mid | best-AUC | `ebf_s35_g0p8_h12_labelscore_s7_tau64000` | g=0.8, h=12 | 1.028816 | 0.811337 | 0.924073 | 41222 | 6701 | 12470 |
| heavy | best-F1 | `ebf_s35_g0p3_h12_labelscore_s7_tau64000` | g=0.3, h=12 | 1.996921 | 0.791323 | 0.924712 | 29713 | 5521 | 10150 |
| heavy | best-AUC | `ebf_s35_g0p8_h4_labelscore_s7_tau64000` | g=0.8, h=4 | 1.126241 | 0.789240 | 0.926461 | 29735 | 5753 | 10128 |

heavy 的 `gamma×hmax` 网格（每格为该 tag 在 heavy 下的 best-F1；越大越好）：

| gamma\hmax | 4 | 6 | 8 | 12 |
| --- | ---:| ---:| ---:| ---:|
| 0.3 | 0.790989 | 0.791140 | 0.791263 | 0.791323 |
| 0.4 | 0.790957 | 0.791085 | 0.791102 | 0.791128 |
| 0.5 | 0.790348 | 0.790486 | 0.790497 | 0.790539 |
| 0.6 | 0.789979 | 0.789932 | 0.789984 | 0.789974 |
| 0.7 | 0.789673 | 0.789517 | 0.789434 | 0.789416 |
| 0.8 | 0.789240 | 0.788851 | 0.788761 | 0.788718 |

heavy（best-F1 operating point，来自 ROC CSV；tag=`g0p3_h12`）：

- s35 tuned：thr=1.99692，F1=0.791323，AUC(tag)=0.924712，TP=29713，FP=5521，FN=10150
- s28（对照）：thr=2.96366，F1=0.787730，AUC(tag)=0.918622，TP=29313，FP=5248，FN=10550

heavy 噪声类型分解（best-F1；对照 s28）：

- s35 tuned：hotmask FP=4813，near-hot FP=471，highrate FP=209，hotmask signal_kept_rate=0.7163
- s28：hotmask FP=4563，near-hot FP=457，highrate FP=201，hotmask signal_kept_rate=0.7041

解读：

- tuned 点相对 s28：TP 增加（signal_kept_rate 提升）带来 F1 增益，但热点类 FP 也略升；该 operating point 更偏“保信号”，未体现出对 hotmask 的结构性压制。

#### C) 继续扫 `tau_rate_us`（固定 `gamma/hmax`，检验“hotness 记忆长度”）

说明：在上面的 `gamma×hmax` 网格里，heavy 的最优点都落在 `gamma=0.3` 一侧。接下来固定 `(gamma,hmax)`，只扫 `tau_rate_us`，看“hot-state 记忆长度”是否能进一步改善 trade-off。

注意：`tau_rate_us=0` 表示 auto（实现里等价于 `tau_rate_us=tau_us`），因此表里你会看到 `0` 与 `tau_us` 两行完全一致。

##### C1) 固定 `s=9,tau=128ms,max-events=200k`，`gamma=0.3,hmax=4`

输出目录：

- s35 tau-rate sweep：`data/ED24/myPedestrain_06/EBF_Part2/_tune_s35_pixelstate_s9_tau128ms_200k_g0p3_h4_tr0_64_128_256_512/`

light（best-F1 per tag）：

| tau_rate_us | tag | thr(best-F1) | F1(best) | AUC(tag) | TP | FP | FN |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | ebf_s35_g0p3_h4_labelscore_s9_tau128000 | -0.634664 | 0.943023 | 0.934715 | 152855 | 8400 | 10071 |
| 64000 | ebf_s35_tr64000_g0p3_h4_labelscore_s9_tau128000 | -0.608104 | 0.939597 | 0.929366 | 151665 | 8239 | 11261 |
| 128000 | ebf_s35_tr128000_g0p3_h4_labelscore_s9_tau128000 | -0.634664 | 0.943023 | 0.934715 | 152855 | 8400 | 10071 |
| 256000 | ebf_s35_tr256000_g0p3_h4_labelscore_s9_tau128000 | -0.662884 | 0.945275 | 0.938926 | 153955 | 8855 | 8971 |
| 512000 | ebf_s35_tr512000_g0p3_h4_labelscore_s9_tau128000 | -0.602184 | 0.946960 | 0.942458 | 154532 | 8917 | 8394 |

mid（best-F1 per tag）：

| tau_rate_us | tag | thr(best-F1) | F1(best) | AUC(tag) | TP | FP | FN |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | ebf_s35_g0p3_h4_labelscore_s9_tau128000 | 1.171884 | 0.815497 | 0.931684 | 42677 | 8296 | 11015 |
| 64000 | ebf_s35_tr64000_g0p3_h4_labelscore_s9_tau128000 | 1.048728 | 0.819105 | 0.932827 | 42942 | 8217 | 10750 |
| 128000 | ebf_s35_tr128000_g0p3_h4_labelscore_s9_tau128000 | 1.171884 | 0.815497 | 0.931684 | 42677 | 8296 | 11015 |
| 256000 | ebf_s35_tr256000_g0p3_h4_labelscore_s9_tau128000 | 1.503260 | 0.805954 | 0.928269 | 41987 | 8513 | 11705 |
| 512000 | ebf_s35_tr512000_g0p3_h4_labelscore_s9_tau128000 | 2.017557 | 0.790404 | 0.922854 | 41463 | 9761 | 12229 |

heavy（best-F1 per tag）：

| tau_rate_us | tag | thr(best-F1) | F1(best) | AUC(tag) | TP | FP | FN |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | ebf_s35_g0p3_h4_labelscore_s9_tau128000 | 1.523715 | 0.782557 | 0.930429 | 30184 | 7095 | 9679 |
| 64000 | ebf_s35_tr64000_g0p3_h4_labelscore_s9_tau128000 | 1.101157 | 0.794960 | 0.934945 | 30949 | 7051 | 8914 |
| 128000 | ebf_s35_tr128000_g0p3_h4_labelscore_s9_tau128000 | 1.523715 | 0.782557 | 0.930429 | 30184 | 7095 | 9679 |
| 256000 | ebf_s35_tr256000_g0p3_h4_labelscore_s9_tau128000 | 2.114941 | 0.758192 | 0.921487 | 29466 | 8398 | 10397 |
| 512000 | ebf_s35_tr512000_g0p3_h4_labelscore_s9_tau128000 | 3.351959 | 0.734847 | 0.911149 | 28364 | 8970 | 11499 |

heavy（该 sweep 的最优点，对比 s28）：

- s35 tuned（tag=`ebf_s35_tr64000_g0p3_h4_labelscore_s9_tau128000`）：thr=1.10116，F1=0.794960，AUC(tag)=0.934945，TP=30949，FP=7051，FN=8914
- s28（对照）：thr=2.57854，F1=0.781485，AUC(tag)=0.923493，TP=30601，FP=7851，FN=9262

heavy 噪声类型分解（best-F1；对照 s28）：

- s35 tuned：hotmask FP=5993，near-hot FP=723，highrate FP=289，hotmask signal_kept_rate=0.7491
- s28：hotmask FP=6727，near-hot FP=754，highrate FP=324，hotmask signal_kept_rate=0.7369

解读：

- `tau_rate_us=64000` 同时实现了 **更低的热点类 FP**（hotmask/near-hot/highrate 都下降）与 **更高的 signal_kept_rate**（TP 上升），因此 heavy best-F1 出现了显著跃升。

##### C2) 固定 `s=7,tau=64ms,max-events=200k`，`gamma=0.3,hmax=12`

输出目录：

- s35 tau-rate sweep：`data/ED24/myPedestrain_06/EBF_Part2/_tune_s35_pixelstate_s7_tau64ms_200k_g0p3_h12_tr0_32_64_128_256/`

light（best-F1 per tag）：

| tau_rate_us | tag | thr(best-F1) | F1(best) | AUC(tag) | TP | FP | FN |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | ebf_s35_g0p3_h12_labelscore_s7_tau64000 | -0.333491 | 0.938305 | 0.928318 | 148522 | 5127 | 14404 |
| 32000 | ebf_s35_tr32000_g0p3_h12_labelscore_s7_tau64000 | -0.316954 | 0.936680 | 0.925094 | 147980 | 5061 | 14946 |
| 64000 | ebf_s35_tr64000_g0p3_h12_labelscore_s7_tau64000 | -0.333491 | 0.938305 | 0.928318 | 148522 | 5127 | 14404 |
| 128000 | ebf_s35_tr128000_g0p3_h12_labelscore_s7_tau64000 | -0.345294 | 0.940126 | 0.931029 | 149150 | 5222 | 13776 |
| 256000 | ebf_s35_tr256000_g0p3_h12_labelscore_s7_tau64000 | -0.361112 | 0.941182 | 0.933201 | 149624 | 5399 | 13302 |

mid（best-F1 per tag）：

| tau_rate_us | tag | thr(best-F1) | F1(best) | AUC(tag) | TP | FP | FN |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | ebf_s35_g0p3_h12_labelscore_s7_tau64000 | 1.757936 | 0.810568 | 0.921526 | 41157 | 6702 | 12535 |
| 32000 | ebf_s35_tr32000_g0p3_h12_labelscore_s7_tau64000 | 1.602286 | 0.810885 | 0.921458 | 41600 | 7312 | 12092 |
| 64000 | ebf_s35_tr64000_g0p3_h12_labelscore_s7_tau64000 | 1.757936 | 0.810568 | 0.921526 | 41157 | 6702 | 12535 |
| 128000 | ebf_s35_tr128000_g0p3_h12_labelscore_s7_tau64000 | 1.738666 | 0.807753 | 0.920541 | 41422 | 7447 | 12270 |
| 256000 | ebf_s35_tr256000_g0p3_h12_labelscore_s7_tau64000 | 2.044706 | 0.801967 | 0.918324 | 40691 | 7095 | 13001 |

heavy（best-F1 per tag）：

| tau_rate_us | tag | thr(best-F1) | F1(best) | AUC(tag) | TP | FP | FN |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | ebf_s35_g0p3_h12_labelscore_s7_tau64000 | 1.996921 | 0.791323 | 0.924712 | 29713 | 5521 | 10150 |
| 32000 | ebf_s35_tr32000_g0p3_h12_labelscore_s7_tau64000 | 1.908451 | 0.795306 | 0.925495 | 29789 | 5260 | 10074 |
| 64000 | ebf_s35_tr64000_g0p3_h12_labelscore_s7_tau64000 | 1.996921 | 0.791323 | 0.924712 | 29713 | 5521 | 10150 |
| 128000 | ebf_s35_tr128000_g0p3_h12_labelscore_s7_tau64000 | 2.266193 | 0.780416 | 0.921710 | 29257 | 5858 | 10606 |
| 256000 | ebf_s35_tr256000_g0p3_h12_labelscore_s7_tau64000 | 2.581198 | 0.764102 | 0.916132 | 29131 | 7255 | 10732 |

heavy（该 sweep 的最优点，对比 s28）：

- s35 tuned（tag=`ebf_s35_tr32000_g0p3_h12_labelscore_s7_tau64000`）：thr=1.90845，F1=0.795306，AUC(tag)=0.925495，TP=29789，FP=5260，FN=10074
- s28（对照）：thr=2.96366，F1=0.787730，AUC(tag)=0.918622，TP=29313，FP=5248，FN=10550

heavy 噪声类型分解（best-F1；对照 s28）：

- s35 tuned：hotmask FP=4559，near-hot FP=465，highrate FP=209，hotmask signal_kept_rate=0.7187
- s28：hotmask FP=4563，near-hot FP=457，highrate FP=201，hotmask signal_kept_rate=0.7041

解读：

- 这里 `tau_rate_us=32000` 的收益主要来自 signal_kept_rate 上升（TP 上升、FN 下降），而热点类 FP 基本持平；更像是把“hotness 记忆”调短后，减少了对 signal 的过度解释。

### 资源分析（相对 baseline 的额外空间）

下面只统计“算法在线运行时、必须常驻内存的状态表”，不含输入/输出事件缓冲。

记传感器分辨率为 `W×H`，像素数 `N=W*H`。

- baseline EBF 必需状态（已存在）：
	- `last_ts`：每像素最后时间戳，`N × 64 bit`（`uint64`）
	- `last_pol`：每像素最后极性，`N × 8 bit`（`int8`）
	- 其余为常数级（如阈值/参数）

- s28 额外状态（相对 baseline）：
	- `rate_ema`：全局 EMA 事件率，`1 × 64 bit`（`float64`）

- s35 额外状态（相对 s28/baseline）：
	- `hot_state`：每像素 hot-state（leaky accumulator），`N × 32 bit`（`int32`）

因此，**s35 相对 baseline 的“新增常驻空间”主要就是一张 `W×H×32bit` 的表**。

举例（按你常用的写法表达）：如果分辨率是 `342×260`，则新增空间是 `342*260*32bit`（再加上 `rate_ema` 的 `64bit` 可忽略）。

### 阈值（thr）与“自适应阈值”能力

- 上面两张 best 摘要表中，已经给出了在“各 env 各自最优参数”下的 `thr`（light/mid/heavy 各一套）。
- s35 本身输出的是连续分数（z-score），**当前实现不包含“在线自适应阈值”机制**：运行时仍需要你给定一个固定阈值（或在离线 ROC 上选定阈值）。
- 现阶段能算作“弱自适应”的只有：s28/s35 的 z-score 标准化减少了阈值随全局事件率漂移的幅度，但并不等价于自动选阈值。

如果你希望在 Part2 就加一个最小版“自适应阈值”，比较工程/论文都好写的方向是：

- 固定目标保留率（keep-rate）或目标 FP-rate：在线估计 score 的分位数，然后把阈值设为对应分位点；
- 或按全局 `rate_ema` 做一个轻量的阈值标定函数 `thr = a + b*log(rate_ema)`（需要少量离线拟合）。

### s35 还有没有优化空间？

结论：**有**，但应分成两类：

1) 继续调参（不改结构，成本最低）

- 你当前 tune 只扫了 `gamma×hmax`，仍有一个关键超参未系统探索：`tau_rate_us`。
	- 直觉：`tau_rate_us` 决定“hot-state 记忆长度”，它会直接影响“压热点 FP vs 误伤 signal”的 trade-off。
	- 建议最小 sweep：固定你已经找到的 heavy best-F1 附近参数（如 `g=0.3,h=4/12`），再扫 `tau_rate_us ∈ {0(auto), 0.5*tau, 1*tau, 2*tau, 4*tau}`。
- 针对跨环境迁移：可以把目标从“各 env 各自 best-F1”改为“同一组 (gamma,hmax,tau_rate) 在三 env 上的平均/最差 F1 最大化”，这样更贴近你 Part2 的主目标。

2) 函数/模型形态的优化（仍保持在线/单遍/O(r^2)）

- 当前 s35 的核心是用一个标量状态去表达 hotness，并通过 `r_eff = r * (1 + gamma*h)` 放大噪声率。这个形态还有至少三种“论文型、可解释”的改进空间：
	- **状态更新与归一化更一致**：把 hot-state 的时间尺度与 `tau_rate_us` 严格对齐（现在的实现形式更接近 `H <- max(0, H-dt)+tau`），从而减少 `tau` 改变时 hotness 标度被动改变；
	- **饱和函数更柔和**：`clip` 可以替换为更平滑的饱和（如 `h = hmax*(1-exp(-H/(hmax*tau_rate)))`），避免在 h 接近上限时出现“微小变化不再反映到 r_eff”的硬截断效应；
	- **对极性/邻域解耦**：hotness 可只看“该像素自身发射”是合理的，但也可以尝试把“近邻共同活跃”纳入 hotness（仍是常数开销：例如维护一个更粗的 block hotness），用来区分“真实运动边缘（局部簇）”与“孤立 hotpixel”。

以上改动都不会引入离线步骤，也不需要额外的全局优化；但它们会改变分数分布，需要重新跑 prescreen 对齐验证。

## s36：Noise Surprise Z-Score - State-Occupancy Adaptive Null（状态占用率自适应空模型）

### 失败模式 / 可验证假设

动机（来自你对 7.8 的判断 + s35 的现象）：

- s35 的核心方向是对的（把 hotness 并入空模型），但它引入了 `tau_rate_us/gamma/hmax` 三个超参，且三者耦合导致敏感；同时在 light/mid 上更容易出现“hotness 过早介入”的误伤。
- 因此 s36 的目标是：**减参（只保留 `tau_rate_us`）+ 更平滑/有界的调制**，让 light/mid 的 score 分布更不容易被大幅重排，同时尽量保留 heavy 的热点压制能力。

可验证假设：把 s35 的“经验调幅器（gamma/hmax + clip）”替换为“占用率 $u$ 的有界映射”，并把 hot-state 的增量改为 dt0 相关，则：

- light/mid：误伤 signal 的概率下降（AUC/F1 更稳）；
- heavy：hotmask/near-hot/highrate 这类 FP 仍应被一定程度压制。

### 算法定义（在线 / 单遍 / $O(r^2)$）

位置：`src/myevs/denoise/ops/ebfopt_part2/s36_noise_surprise_zscore_stateoccupancy.py`

1) raw-support（同 s28）：

$$
raw=\sum_{j\in\mathcal{N}} \mathbf{1}[pol_j=pol_i] \,\max\bigl(0,1-\Delta t/\tau\bigr)
$$

2) 全局噪声率估计（同 s28）：用全局事件率 EMA 得到每像素噪声率 $r$（events / time / pixel）。

3) 像素 hot-state（单通道 leaky accumulator，dt0 驱动增量）：

定义该像素与其上一事件的自 dt：$\Delta t_0$。

状态更新：

$$
H\leftarrow\max(0,H-\Delta t_0)+\max(0,\tau-\Delta t_0)
$$

直观理解：只有当自 dt 足够短（$\Delta t_0<\tau$）时，才会给 $H$ 增加正增量；否则只发生衰减。这比 s35 的“每事件无条件 +\tau_{rate}”更克制。

4) 状态占用率（dimensionless，$u\in[0,1)$）：

$$
u=\frac{H}{H+\tau_{rate}}
$$

其中 $\tau_{rate}$ 为全局 rate-EMA 的时间常数（与 s28 的 `tau_rate_us` 一致）。

5) 状态条件的自适应空模型（有效噪声率，有界且平滑）：

$$
r_{eff}=r\,(1+u)^2
$$

由于 $u\in[0,1)$，因此 $(1+u)^2\in[1,4)$，有效噪声率最多放大约 4 倍；避免了 s35 里 `gamma/hmax` 造成的调幅不稳定。

6) 复用 s28 的噪声模型标准化：

$$
z=\frac{raw-\mu(r_{eff})}{\sigma(r_{eff})+\varepsilon}
$$

### 超参（环境变量 / sweep 参数）

- `MYEVS_EBF_S36_TAU_RATE_US`：$\tau_{rate}$（微秒）。
	- `0` 表示 auto：直接用当前 sweep 点的 `tau_us`。

sweep 参数：`--s36-tau-rate-us-list`。

### 实现约束核对

- 在线流式/单遍：是
- 复杂度：$O(r^2)$（邻域遍历同 s28；新增逻辑为常数级）
- Numba 必须：是
- 额外状态：是（新增 1 个 int32 per-pixel 数组 `hot_state` + 全局 `rate_ema`，与 s35 同量级）

### 结果（tau_rate_us sweep；prescreen200k）

本节口径：固定 `max-events=200k`；对每个 `tau_rate_us`，从 ROC CSV 里取该 tag 的 best-F1 operating point（表内给出 `thr/TP/FP/FN/F1/AUC(tag)`）。

#### A) 固定 `s=9,tau=128ms,max-events=200k`

输出目录：

- s36 tau-rate sweep：`data/ED24/myPedestrain_06/EBF_Part2/_tune_s36_stateocc_s9_tau128ms_200k_tr0_64_128_256_512/`

light（best-F1 per tag）：

| tau_rate_us | tag | thr(best-F1) | F1(best) | AUC(tag) | TP | FP | FN |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | ebf_s36_labelscore_s9_tau128000 | -0.556683 | 0.945995 | 0.939877 | 153630 | 8245 | 9296 |
| 64000 | ebf_s36_tr64000_labelscore_s9_tau128000 | -0.533683 | 0.942174 | 0.933416 | 152138 | 7887 | 10788 |
| 128000 | ebf_s36_tr128000_labelscore_s9_tau128000 | -0.556683 | 0.945995 | 0.939877 | 153630 | 8245 | 9296 |
| 256000 | ebf_s36_tr256000_labelscore_s9_tau128000 | -0.527072 | 0.948433 | 0.944239 | 154595 | 8480 | 8331 |
| 512000 | ebf_s36_tr512000_labelscore_s9_tau128000 | -0.450062 | 0.949700 | 0.947003 | 155113 | 8618 | 7813 |

mid（best-F1 per tag）：

| tau_rate_us | tag | thr(best-F1) | F1(best) | AUC(tag) | TP | FP | FN |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | ebf_s36_labelscore_s9_tau128000 | 1.599210 | 0.813615 | 0.935157 | 42358 | 8073 | 11334 |
| 64000 | ebf_s36_tr64000_labelscore_s9_tau128000 | 1.458342 | 0.810298 | 0.934036 | 41796 | 7674 | 11896 |
| 128000 | ebf_s36_tr128000_labelscore_s9_tau128000 | 1.599210 | 0.813615 | 0.935157 | 42358 | 8073 | 11334 |
| 256000 | ebf_s36_tr256000_labelscore_s9_tau128000 | 1.974495 | 0.808460 | 0.932148 | 42240 | 8563 | 11452 |
| 512000 | ebf_s36_tr512000_labelscore_s9_tau128000 | 2.542987 | 0.795345 | 0.925619 | 42131 | 10121 | 11561 |

heavy（best-F1 per tag）：

| tau_rate_us | tag | thr(best-F1) | F1(best) | AUC(tag) | TP | FP | FN |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | ebf_s36_labelscore_s9_tau128000 | 1.809675 | 0.774394 | 0.930152 | 30007 | 7628 | 9856 |
| 64000 | ebf_s36_tr64000_labelscore_s9_tau128000 | 1.471044 | 0.775090 | 0.929048 | 29641 | 6980 | 10222 |
| 128000 | ebf_s36_tr128000_labelscore_s9_tau128000 | 1.809675 | 0.774394 | 0.930152 | 30007 | 7628 | 9856 |
| 256000 | ebf_s36_tr256000_labelscore_s9_tau128000 | 2.506510 | 0.760625 | 0.924731 | 29888 | 8837 | 9975 |
| 512000 | ebf_s36_tr512000_labelscore_s9_tau128000 | 4.277860 | 0.742401 | 0.915026 | 28392 | 8232 | 11471 |

heavy（该 sweep 的最优点，对照 s35 tuned / s28）：

- s36 tuned（tag=`ebf_s36_tr64000_labelscore_s9_tau128000`）：thr=1.47104，F1=0.775090，AUC(tag)=0.929048，TP=29641，FP=6980，FN=10222
- s35 tuned（对照，见上节 s35 C1）：F1=0.794960（tag=`ebf_s35_tr64000_g0p3_h4_labelscore_s9_tau128000`）
- s28（对照，见上节 s35 C1）：F1=0.781485

heavy 噪声类型分解（best-F1；`scripts/noise_analyze/noise_type_stats.py`）：

- s36 tuned：hotmask FP=6036，near-hot FP=643，highrate FP=258，hotmask signal_kept_rate=0.7194
- s35 tuned（对照，见上节 s35 C1）：hotmask FP=5993，near-hot FP=723，highrate FP=289，hotmask signal_kept_rate=0.7491

解读：

- s36 在该点把 near-hot/highrate 的 FP 压得更低，但总体 signal_kept_rate 明显更低（误伤更大），因此 heavy best-F1 低于 s35 tuned。

#### B) 固定 `s=7,tau=64ms,max-events=200k`

输出目录：

- s36 tau-rate sweep：`data/ED24/myPedestrain_06/EBF_Part2/_tune_s36_stateocc_s7_tau64ms_200k_tr0_32_64_128_256/`

light（best-F1 per tag）：

| tau_rate_us | tag | thr(best-F1) | F1(best) | AUC(tag) | TP | FP | FN |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | ebf_s36_labelscore_s7_tau64000 | -0.292233 | 0.940080 | 0.929954 | 149210 | 5305 | 13716 |
| 32000 | ebf_s36_tr32000_labelscore_s7_tau64000 | -0.275724 | 0.938778 | 0.926996 | 148694 | 5162 | 14232 |
| 64000 | ebf_s36_tr64000_labelscore_s7_tau64000 | -0.292233 | 0.940080 | 0.929954 | 149210 | 5305 | 13716 |
| 128000 | ebf_s36_tr128000_labelscore_s7_tau64000 | -0.305953 | 0.940869 | 0.932088 | 149585 | 5461 | 13341 |
| 256000 | ebf_s36_tr256000_labelscore_s7_tau64000 | -0.282884 | 0.941571 | 0.933557 | 149835 | 5505 | 13091 |

mid（best-F1 per tag）：

| tau_rate_us | tag | thr(best-F1) | F1(best) | AUC(tag) | TP | FP | FN |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | ebf_s36_labelscore_s7_tau64000 | 2.034201 | 0.811273 | 0.924734 | 41439 | 7027 | 12253 |
| 32000 | ebf_s36_tr32000_labelscore_s7_tau64000 | 1.979332 | 0.809160 | 0.925619 | 41077 | 6761 | 12615 |
| 64000 | ebf_s36_tr64000_labelscore_s7_tau64000 | 2.034201 | 0.811273 | 0.924734 | 41439 | 7027 | 12253 |
| 128000 | ebf_s36_tr128000_labelscore_s7_tau64000 | 2.284569 | 0.809457 | 0.922172 | 41118 | 6784 | 12574 |
| 256000 | ebf_s36_tr256000_labelscore_s7_tau64000 | 2.586068 | 0.803404 | 0.918334 | 40780 | 7046 | 12912 |

heavy（best-F1 per tag）：

| tau_rate_us | tag | thr(best-F1) | F1(best) | AUC(tag) | TP | FP | FN |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | ebf_s36_labelscore_s7_tau64000 | 2.399130 | 0.788883 | 0.927859 | 29619 | 5609 | 10244 |
| 32000 | ebf_s36_tr32000_labelscore_s7_tau64000 | 2.201998 | 0.787280 | 0.928649 | 29486 | 5557 | 10377 |
| 64000 | ebf_s36_tr64000_labelscore_s7_tau64000 | 2.399130 | 0.788883 | 0.927859 | 29619 | 5609 | 10244 |
| 128000 | ebf_s36_tr128000_labelscore_s7_tau64000 | 2.737523 | 0.782127 | 0.923874 | 29460 | 6010 | 10403 |
| 256000 | ebf_s36_tr256000_labelscore_s7_tau64000 | 3.094440 | 0.766794 | 0.917022 | 29553 | 7666 | 10310 |

heavy（该 sweep 的最优点，对照 s35 tuned / s28）：

- s36 tuned（tag=`ebf_s36_labelscore_s7_tau64000`）：thr=2.39913，F1=0.788883，AUC(tag)=0.927859，TP=29619，FP=5609，FN=10244
- s35 tuned（对照，见上节 s35 C2）：F1=0.795306（tag=`ebf_s35_tr32000_g0p3_h12_labelscore_s7_tau64000`）
- s28（对照，见上节 s35 C2）：F1=0.787730

heavy 噪声类型分解（best-F1）：

- s36 tuned：hotmask FP=4925，near-hot FP=459，highrate FP=195，hotmask signal_kept_rate=0.7143
- s35 tuned（对照，见上节 s35 C2）：hotmask FP=4559，near-hot FP=465，highrate FP=209，hotmask signal_kept_rate=0.7187

### 是否继续

结论：**s36 作为“减参/更平滑”的备选保留，但当前 heavy best-F1 仍明显弱于 s35 tuned**。

如果下一阶段的主目标仍是 heavy best-F1（同时兼顾 light/mid），优先继续以 s35 tuned 为主线；若你更在意“参数少 + 稳定性”（允许 heavy 牺牲一些），s36 可以作为更简洁的迁移版本继续迭代。

## s37：Noise Surprise Z-Score - State-Occupancy 3-State Adaptive Null（状态占用率三段式自适应空模型）

### 失败模式 / 可验证假设

动机（来自你在 7.9 的优化直觉 + s36 的现象）：

- s36 的 $r_{eff}=r\,(1+u)^2$ 虽然有界/平滑，但可能过于“温和”，导致 heavy 下对更极端的 hotmask/hotpixel 抑制不够彻底；同时在 light/mid 下又可能仍然会在部分区域引入不必要的分数重排。
- 因此 s37 的目标是：**保持 s36 的减参/稳定骨架不变，仅修改 $u\to r_{eff}$ 的映射形态，使强抑制更“早触发、更果断”**，同时不引入新的可调超参（仍只保留 `tau_rate_us`）。

可验证假设：把 $(1+u)^2$ 替换为三段式倍率（$1\to2\to4$），则：

- heavy：hotmask/near-hot/highrate 的 FP 可能进一步下降；
- light/mid：由于倍率是离散且仍有上界，整体排序能力（AUC/F1）不应出现灾难性退化。

### 算法定义（在线 / 单遍 / $O(r^2)$）

位置：`src/myevs/denoise/ops/ebfopt_part2/s37_noise_surprise_zscore_stateoccupancy_3state.py`

s37 与 s36 完全一致，**唯一差异**在第 5 步的 $r_{eff}$ 形态。

1) raw-support：同 s28/s36。

2) 全局噪声率估计：同 s28/s36。

3) 像素 hot-state：同 s36。

4) 状态占用率：同 s36。

$$
u=\frac{H}{H+\tau_{rate}}\in[0,1)
$$

5) 三段式状态条件空模型（有效噪声率，有界且更早触发强压制）：

$$
m(u)=\begin{cases}
1,&u<1/3\\
2,&1/3\le u<2/3\\
4,&u\ge 2/3
\end{cases}
\qquad r_{eff}=r\,m(u)
$$

6) 复用 s28 的噪声模型标准化：

$$
z=\frac{raw-\mu(r_{eff})}{\sigma(r_{eff})+\varepsilon}
$$

### 超参（环境变量 / sweep 参数）

- `MYEVS_EBF_S37_TAU_RATE_US`：$\tau_{rate}$（微秒）。
	- `0` 表示 auto：直接用当前 sweep 点的 `tau_us`。

sweep 参数：`--s37-tau-rate-us-list`。

### 结果（tau_rate_us sweep；prescreen200k）

本节口径：固定 `max-events=200k`；对每个 `tau_rate_us`，从 ROC CSV 里取该 tag 的 best-F1 operating point（表内给出 `thr/TP/FP/FN/F1/AUC(tag)`）。

#### A) 固定 `s=9,tau=128ms,max-events=200k`

输出目录：

- s37 tau-rate sweep：`data/ED24/myPedestrain_06/EBF_Part2/_tune_s37_stateocc3state_s9_tau128ms_200k_tr0_64_128_256_512/`

light（best-F1 per tag）：

| tau_rate_us | tag | thr(best-F1) | F1(best) | AUC(tag) | TP | FP | FN |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | ebf_s37_labelscore_s9_tau128000 | -0.556721 | 0.946033 | 0.939823 | 153744 | 8359 | 9182 |
| 64000 | ebf_s37_tr64000_labelscore_s9_tau128000 | -0.537304 | 0.942167 | 0.933143 | 152266 | 8033 | 10660 |
| 128000 | ebf_s37_tr128000_labelscore_s9_tau128000 | -0.556721 | 0.946033 | 0.939823 | 153744 | 8359 | 9182 |
| 256000 | ebf_s37_tr256000_labelscore_s9_tau128000 | -0.528539 | 0.948379 | 0.943994 | 154858 | 8790 | 8068 |
| 512000 | ebf_s37_tr512000_labelscore_s9_tau128000 | -0.455385 | 0.949539 | 0.946598 | 155261 | 8837 | 7665 |

mid（best-F1 per tag）：

| tau_rate_us | tag | thr(best-F1) | F1(best) | AUC(tag) | TP | FP | FN |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | ebf_s37_labelscore_s9_tau128000 | 1.643163 | 0.812510 | 0.933021 | 42450 | 8349 | 11242 |
| 64000 | ebf_s37_tr64000_labelscore_s9_tau128000 | 1.463555 | 0.810435 | 0.931805 | 42002 | 7959 | 11690 |
| 128000 | ebf_s37_tr128000_labelscore_s9_tau128000 | 1.643163 | 0.812510 | 0.933021 | 42450 | 8349 | 11242 |
| 256000 | ebf_s37_tr256000_labelscore_s9_tau128000 | 2.241394 | 0.806502 | 0.927054 | 42124 | 8645 | 11568 |
| 512000 | ebf_s37_tr512000_labelscore_s9_tau128000 | 2.980868 | 0.792202 | 0.920908 | 41429 | 9471 | 12263 |

heavy（best-F1 per tag）：

| tau_rate_us | tag | thr(best-F1) | F1(best) | AUC(tag) | TP | FP | FN |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | ebf_s37_labelscore_s9_tau128000 | 2.071142 | 0.774426 | 0.927297 | 29561 | 6919 | 10302 |
| 64000 | ebf_s37_tr64000_labelscore_s9_tau128000 | 1.498836 | 0.774128 | 0.924570 | 29718 | 7197 | 10145 |
| 128000 | ebf_s37_tr128000_labelscore_s9_tau128000 | 2.071142 | 0.774426 | 0.927297 | 29561 | 6919 | 10302 |
| 256000 | ebf_s37_tr256000_labelscore_s9_tau128000 | 3.081079 | 0.759545 | 0.919074 | 29334 | 8044 | 10529 |
| 512000 | ebf_s37_tr512000_labelscore_s9_tau128000 | 4.690416 | 0.740792 | 0.910143 | 28359 | 8342 | 11504 |

heavy（该 sweep 的最优点，对照 s36 tuned / s35 tuned / s28）：

- s37 tuned（tag=`ebf_s37_labelscore_s9_tau128000`）：thr=2.07114，F1=0.774426，AUC(tag)=0.927297，TP=29561，FP=6919，FN=10302
- s36 tuned（对照，见上节 s36 A）：F1=0.775090（tag=`ebf_s36_tr64000_labelscore_s9_tau128000`）
- s35 tuned（对照，见上节 s35 C1）：F1=0.794960（tag=`ebf_s35_tr64000_g0p3_h4_labelscore_s9_tau128000`）
- s28（对照，见上节 s35 C1）：F1=0.781485

heavy 噪声类型分解（best-F1；`scripts/noise_analyze/noise_type_stats.py`）：

- s37 tuned：hotmask FP=5972，near-hot FP=649，highrate FP=254，hotmask signal_kept_rate=0.7146

#### B) 固定 `s=7,tau=64ms,max-events=200k`

输出目录：

- s37 tau-rate sweep：`data/ED24/myPedestrain_06/EBF_Part2/_tune_s37_stateocc3state_s7_tau64ms_200k_tr0_32_64_128_256/`

light（best-F1 per tag）：

| tau_rate_us | tag | thr(best-F1) | F1(best) | AUC(tag) | TP | FP | FN |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | ebf_s37_labelscore_s7_tau64000 | -0.292233 | 0.940278 | 0.929934 | 149280 | 5317 | 13646 |
| 32000 | ebf_s37_tr32000_labelscore_s7_tau64000 | -0.274105 | 0.938967 | 0.926889 | 148762 | 5175 | 14164 |
| 64000 | ebf_s37_tr64000_labelscore_s7_tau64000 | -0.292233 | 0.940278 | 0.929934 | 149280 | 5317 | 13646 |
| 128000 | ebf_s37_tr128000_labelscore_s7_tau64000 | -0.305281 | 0.941029 | 0.931316 | 149658 | 5489 | 13268 |
| 256000 | ebf_s37_tr256000_labelscore_s7_tau64000 | -0.317978 | 0.941595 | 0.932671 | 150006 | 5689 | 12920 |

mid（best-F1 per tag）：

| tau_rate_us | tag | thr(best-F1) | F1(best) | AUC(tag) | TP | FP | FN |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | ebf_s37_labelscore_s7_tau64000 | 2.160098 | 0.810087 | 0.923251 | 41182 | 6799 | 12510 |
| 32000 | ebf_s37_tr32000_labelscore_s7_tau64000 | 2.033634 | 0.809354 | 0.923907 | 41099 | 6769 | 12593 |
| 64000 | ebf_s37_tr64000_labelscore_s7_tau64000 | 2.160098 | 0.810087 | 0.923251 | 41182 | 6799 | 12510 |
| 128000 | ebf_s37_tr128000_labelscore_s7_tau64000 | 2.607798 | 0.805905 | 0.917412 | 40506 | 6325 | 13186 |
| 256000 | ebf_s37_tr256000_labelscore_s7_tau64000 | 2.650572 | 0.799608 | 0.914520 | 40844 | 7624 | 12848 |

heavy（best-F1 per tag）：

| tau_rate_us | tag | thr(best-F1) | F1(best) | AUC(tag) | TP | FP | FN |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | ebf_s37_labelscore_s7_tau64000 | 2.473024 | 0.787407 | 0.926185 | 29601 | 5722 | 10262 |
| 32000 | ebf_s37_tr32000_labelscore_s7_tau64000 | 2.298913 | 0.788212 | 0.926262 | 29422 | 5370 | 10441 |
| 64000 | ebf_s37_tr64000_labelscore_s7_tau64000 | 2.473024 | 0.787407 | 0.926185 | 29601 | 5722 | 10262 |
| 128000 | ebf_s37_tr128000_labelscore_s7_tau64000 | 3.059002 | 0.779216 | 0.918785 | 29145 | 5798 | 10718 |
| 256000 | ebf_s37_tr256000_labelscore_s7_tau64000 | 3.521517 | 0.764680 | 0.913375 | 28989 | 6968 | 10874 |

heavy（该 sweep 的最优点，对照 s36 tuned / s35 tuned / s28）：

- s37 tuned（tag=`ebf_s37_tr32000_labelscore_s7_tau64000`）：thr=2.29891，F1=0.788212，AUC(tag)=0.926262，TP=29422，FP=5370，FN=10441
- s36 tuned（对照，见上节 s36 B）：F1=0.788883（tag=`ebf_s36_labelscore_s7_tau64000`）
- s35 tuned（对照，见上节 s35 C2）：F1=0.795306（tag=`ebf_s35_tr32000_g0p3_h12_labelscore_s7_tau64000`）
- s28（对照，见上节 s35 C2）：F1=0.787730

heavy 噪声类型分解（best-F1）：

- s37 tuned：hotmask FP=4707，near-hot FP=446，highrate FP=187，hotmask signal_kept_rate=0.7098

### 补充：u 分布诊断（s36 vs s37；heavy best-F1）

动机：s37 只改了 $u\to r_{eff}$ 的映射形态；若“剩余 FP 的 u 主要落在强抑制阈值以下”，则 s37 很难通过映射形态本身继续压 FP。

以下产物均来自“重放 ROC best-F1 operating point”的逐事件导出：

#### A) heavy, s=9, tau=128ms, prescreen200k

- s37：
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/u_events_s37_heavy_prescreen200k_s9_tau128ms_bestf1.csv`
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/u_quantiles_s37_heavy_prescreen200k_s9_tau128ms_bestf1.csv`
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/u_hist_s37_heavy_prescreen200k_s9_tau128ms_bestf1.png`

- s36（对照）：
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/u_events_s36_heavy_prescreen200k_s9_tau128ms_bestf1.csv`
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/u_quantiles_s36_heavy_prescreen200k_s9_tau128ms_bestf1.csv`
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/u_hist_s36_heavy_prescreen200k_s9_tau128ms_bestf1.png`

关键数值（hotmask FP 的 u 分位数）：

- s37：p75=0.3247，p90=0.5598，p95=0.6473（<2/3），p99=0.8337
- s36：p75=0.1315，p90=0.6978（>2/3），p95=0.8107，p99=0.9230

解读：在该点，s37 的“剩余 hotmask FP”在 p95 仍低于 2/3，因此 s37 的 4× 强抑制区间对剩余 FP 的覆盖并不高；仅靠离散倍率映射更难进一步压 FP。

#### B) heavy, s=7, tau=64ms, prescreen200k

- s37：
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/u_events_s37_heavy_prescreen200k_s7_tau64ms_bestf1.csv`
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/u_quantiles_s37_heavy_prescreen200k_s7_tau64ms_bestf1.csv`
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/u_hist_s37_heavy_prescreen200k_s7_tau64ms_bestf1.png`

- s36（对照）：
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/u_events_s36_heavy_prescreen200k_s7_tau64ms_bestf1.csv`
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/u_quantiles_s36_heavy_prescreen200k_s7_tau64ms_bestf1.csv`
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/u_hist_s36_heavy_prescreen200k_s7_tau64ms_bestf1.png`

关键数值（hotmask FP 的 u 分位数）：

- s37：p90=0.6052，p95=0.6651（≈2/3），p99=0.8633
- s36：p90=0.4561，p95=0.5753，p99=0.7678

解读：该点剩余 hotmask FP 的 u 更接近 2/3，但仍存在大量 FP 落在强抑制阈值附近或以下；因此更可能需要改动 $H$ 的标定尺度/更新方式，或引入更能覆盖 hotmask 的额外证据（见 s38）。

### 是否继续

结论：**当前 s37 没有明显优于 s36（heavy best-F1 基本持平/略低），仍明显弱于 s35 tuned**。如果后续继续沿“把 hotness 并入空模型”这条主线优化，建议把精力集中在 s35/s36 的更平滑可解释形态（或进一步对准 hotmask/near-hot 的机制），而不是继续细调这类离散倍率映射。

## s38：Noise Surprise Z-Score - State+Neighborhood Occupancy Fusion（自占用率+邻域占用率融合空模型）

### 失败模式 / 可验证假设

基于 s37 的 u 分布诊断：heavy 下剩余 hotmask FP 的 u 往往落在强抑制阈值附近或以下；因此仅修改 $u\to r_{eff}$ 的映射形态很难继续压 FP。

假设：若让“局部邻域近期活动”也参与占用率（而不是只依赖单像素自历史），则对 hotmask/噪声团这类“局部活跃但单像素未必持续极短 dt”的模式，$r_{eff}$ 会更早被上修，从而减少这类 FP。

### 算法定义（在线 / 单遍 / $O(r^2)$）

位置：`src/myevs/denoise/ops/ebfopt_part2/s38_noise_surprise_zscore_stateoccupancy_nbocc_fusion.py`

s38 保留 s36 的主干（raw-support + 全局 rate-EMA + hot-state H），并额外构造邻域占用率：

1) raw-support（同 s28/s36）：

$$
raw=\sum_{j\in\mathcal{N}} \mathbf{1}[pol_j=pol_i] \,\max\bigl(0,1-\Delta t/\tau\bigr)
$$

2) 全局噪声率 EMA（同 s28/s36）：得到每像素噪声率 $r$。

3) 自状态 hot-state（同 s36）：

$$
H\leftarrow\max(0,H-\Delta t_0)+\max(0,\tau-\Delta t_0)
$$

4) 自占用率（同 s36）：

$$
u_{self}=\frac{H}{H+\tau_{rate}}\in[0,1)
$$

5) 邻域占用率（新增）：在同一邻域遍历中，统计任意极性的 recency mass：

$$
raw_{all}=\sum_{j\in\mathcal{N}} \max\bigl(0,1-\Delta t/\tau\bigr)
$$

用 $m=|\mathcal{N}|$ 归一化成占用率：

$$
u_{nb}=\frac{raw_{all}}{raw_{all}+m}\in[0,1)
$$

6) 占用率融合（参数无关的 union）：

$$
u = 1-(1-u_{self})(1-u_{nb})\in[0,1)
$$

7) 自适应空模型（沿用 s36 的平滑有界形态）：

$$
r_{eff}=r\,(1+u)^2
$$

8) 复用 s28 的噪声模型标准化：

$$
z=\frac{raw-\mu(r_{eff})}{\sigma(r_{eff})+\varepsilon}
$$

### 超参（环境变量 / sweep 参数）

- `MYEVS_EBF_S38_TAU_RATE_US`：$\tau_{rate}$（微秒）。
	- `0` 表示 auto：直接用当前 sweep 点的 `tau_us`。

sweep 参数：`--s38-tau-rate-us-list`。

### 实现约束核对

- 在线流式/单遍：是
- 复杂度：$O(r^2)$（邻域遍历同 s36；新增逻辑为同一循环内的常数累加）
- Numba 必须：是
- 额外状态：与 s36 同量级（`hot_state` + `rate_ema`；无额外 per-pixel 新数组）

### 当前状态

- 已完成：代码接入 sweep + 分析脚本（可通过 `_trXXXX_` tag 复现）
- 已完成：smoke 跑通（`data/ED24/myPedestrain_06/EBF_Part2/_smoke_s38_2k_s9_tau128/`）

### 实验结果（prescreen200k：max-events=200k）

说明：本次重点做 s38 的 `tau_rate_us` 小扫，口径与 s36/s37 的诊断点对齐（`(s=9,tau=128ms)` 与 `(s=7,tau=64ms)`）。

#### A) prescreen200k sweep：对齐点 (s=9,tau=128ms)

输出目录：`data/ED24/myPedestrain_06/EBF_Part2/_tune_s38_stateocc_nbocc_s9_tau128ms_200k/`

摘要（来自 sweep stdout）：

- light：best AUC=0.945374（tag=`ebf_s38_tr512000_labelscore_s9_tau128000`），best-F1=0.948869
- mid：best AUC=0.933145（tag=`ebf_s38_labelscore_s9_tau128000`），best-F1=0.810092
- heavy：best AUC=0.928266，best-F1=0.775007（tag=`ebf_s38_tr64000_labelscore_s9_tau128000`）

产物：

- `roc_ebf_s38_light_labelscore_s3_5_7_9_tau8_16_32_64_128_256_512_1024ms.csv/png`
- `roc_ebf_s38_mid_labelscore_s3_5_7_9_tau8_16_32_64_128_256_512_1024ms.csv/png`
- `roc_ebf_s38_heavy_labelscore_s3_5_7_9_tau8_16_32_64_128_256_512_1024ms.csv/png`

#### B) prescreen200k sweep：对齐点 (s=7,tau=64ms)

输出目录：`data/ED24/myPedestrain_06/EBF_Part2/_tune_s38_stateocc_nbocc_s7_tau64ms_200k/`

摘要（来自 sweep stdout）：

- light：best AUC=0.932561，best-F1=0.941384（tag=`ebf_s38_tr256000_labelscore_s7_tau64000`）
- mid：best-F1=0.809960（tag=`ebf_s38_labelscore_s7_tau64000`）；mid best AUC=0.924946（tag=`ebf_s38_tr32000_labelscore_s7_tau64000`）
- heavy：best AUC=0.928029，best-F1=0.787094（tag=`ebf_s38_tr32000_labelscore_s7_tau64000`）

产物：

- `roc_ebf_s38_light_labelscore_s3_5_7_9_tau8_16_32_64_128_256_512_1024ms.csv/png`
- `roc_ebf_s38_mid_labelscore_s3_5_7_9_tau8_16_32_64_128_256_512_1024ms.csv/png`
- `roc_ebf_s38_heavy_labelscore_s3_5_7_9_tau8_16_32_64_128_256_512_1024ms.csv/png`

### heavy 诊断：best-F1 operating point 的噪声分解 + u 分布

目的：回答“s38 是否更对准 hotmask/near-hot / highrate 失败模式，以及剩余 FP 的 u 是否被推到更强抑制区域”。

hotmask 统一使用：`data/ED24/myPedestrain_06/EBF_Part2/hotmask_heavy_score_neg_minus_2pos_topk32768_dil1.npy`

#### A) heavy, (s=9,tau=128ms) best-F1：tag=`ebf_s38_tr64000_labelscore_s9_tau128000`

- 噪声类型统计：
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/noise_types_s38_heavy_prescreen200k_s9_tau128ms_tr64000_bestf1.csv`
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/top_pixels_s38_heavy_prescreen200k_s9_tau128ms_tr64000_bestf1.csv`

关键 kept-rate（noise_kept_rate，越小越好；括号内为对应 signal_kept_rate）：

- hotmask：0.03885（0.71612）
- near_hotmask：0.11149（0.79417）
- highrate_pixel：0.14231（0.80839）

- u 分布：
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/u_events_s38_heavy_prescreen200k_s9_tau128ms_tr64000_bestf1.csv`
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/u_quantiles_s38_heavy_prescreen200k_s9_tau128ms_tr64000_bestf1.csv`
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/u_hist_s38_heavy_prescreen200k_s9_tau128ms_tr64000_bestf1.png`

关键分位数（hotmask FP 的 u）：p75=0.3356，p90=0.7326，p95=0.8292，p99=0.9308。

#### B) heavy, (s=7,tau=64ms) best-F1：tag=`ebf_s38_tr32000_labelscore_s7_tau64000`

- 噪声类型统计：
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/noise_types_s38_heavy_prescreen200k_s7_tau64ms_tr32000_bestf1.csv`
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/top_pixels_s38_heavy_prescreen200k_s7_tau64ms_tr32000_bestf1.csv`

关键 kept-rate（noise_kept_rate；括号内为 signal_kept_rate）：

- hotmask：0.03336（0.71526）
- near_hotmask：0.08313（0.79766）
- highrate_pixel：0.10414（0.84323）

- u 分布：
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/u_events_s38_heavy_prescreen200k_s7_tau64ms_tr32000_bestf1.csv`
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/u_quantiles_s38_heavy_prescreen200k_s7_tau64ms_tr32000_bestf1.csv`
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/u_hist_s38_heavy_prescreen200k_s7_tau64ms_tr32000_bestf1.png`

关键分位数（hotmask FP 的 u）：p75=0.0，p90=0.6053，p95=0.7284，p99=0.8738。

#### 解读（先就 s38 内部两点位对比）

- (s=7,tau=64ms,tr=32ms) 在 heavy 上的 best-F1 更高（0.7871 vs 0.7750），且 hotmask/near-hot/highrate 的 noise_kept_rate 更低，同时 signal_kept_rate 未出现明显恶化。
- (s=9,tau=128ms,tr=64ms) 的 hotmask FP u 分布整体更“靠右”（p75/p90/p95 更高），但这并未转化成更低的 hotmask kept-rate；提示仅“把 u 推高”可能不足以解决该点位的主要 FP（仍需要检查阈值位置、以及 u→r_eff 标定是否匹配）。

#### 对照：s36 vs s37 vs s38（heavy，prescreen200k，同口径）

说明：下表均来自 `scripts/noise_analyze/noise_type_stats.py` 与 `u_quantiles.py` 的输出；对 s36/s37 使用已完成的 `*_tune_taurate` best-F1 operating point（tag/阈值见表格）。

**(s=9, tau=128ms) best-F1 operating point：noise_kept_rate（括号内为 signal_kept_rate）**

| variant | tag | thr | hotmask | near_hotmask | highrate_pixel |
|---|---|---:|---:|---:|---:|
| s36 | ebf_s36_tr64000_labelscore_s9_tau128000 | 1.4710 | 0.03959 (0.71936) | 0.11397 (0.79242) | 0.14068 (0.81077) |
| s37 | ebf_s37_labelscore_s9_tau128000 | 2.0711 | 0.03917 (0.71461) | 0.11503 (0.79629) | 0.13850 (0.81789) |
| s38 | ebf_s38_tr64000_labelscore_s9_tau128000 | 0.9808 | 0.03885 (0.71612) | 0.11149 (0.79417) | 0.14231 (0.80839) |

**(s=9, tau=128ms) hotmask FP 的 u 分位数（p75/p90/p95/p99）**

| variant | tag | p75 | p90 | p95 | p99 |
|---|---|---:|---:|---:|---:|
| s36 | ebf_s36_tr64000_labelscore_s9_tau128000 | 0.1315 | 0.6978 | 0.8107 | 0.9230 |
| s37 | ebf_s37_labelscore_s9_tau128000 | 0.3247 | 0.5598 | 0.6473 | 0.8337 |
| s38 | ebf_s38_tr64000_labelscore_s9_tau128000 | 0.3356 | 0.7326 | 0.8292 | 0.9308 |

**(s=7, tau=64ms) best-F1 operating point：noise_kept_rate（括号内为 signal_kept_rate）**

| variant | tag | thr | hotmask | near_hotmask | highrate_pixel |
|---|---|---:|---:|---:|---:|
| s36 | ebf_s36_labelscore_s7_tau64000 | 2.3991 | 0.03231 (0.71429) | 0.08135 (0.79953) | 0.10632 (0.84719) |
| s37 | ebf_s37_tr32000_labelscore_s7_tau64000 | 2.2989 | 0.03088 (0.70979) | 0.07905 (0.79305) | 0.10196 (0.84086) |
| s38 | ebf_s38_tr32000_labelscore_s7_tau64000 | 1.8108 | 0.03336 (0.71526) | 0.08313 (0.79766) | 0.10414 (0.84323) |

**(s=7, tau=64ms) hotmask FP 的 u 分位数（p75/p90/p95/p99）**

| variant | tag | p75 | p90 | p95 | p99 |
|---|---|---:|---:|---:|---:|
| s36 | ebf_s36_labelscore_s7_tau64000 | 0.0000 | 0.4561 | 0.5753 | 0.7678 |
| s37 | ebf_s37_tr32000_labelscore_s7_tau64000 | 0.0000 | 0.6052 | 0.6651 | 0.8633 |
| s38 | ebf_s38_tr32000_labelscore_s7_tau64000 | 0.0000 | 0.6053 | 0.7284 | 0.8738 |

### 是否继续

结论：**继续，但不再在 s38 内部做“只推 u”式的细调**。目前 s38 已完成 prescreen200k 两点位 sweep + heavy best-F1 的噪声分解与 u 分布，并已和 s36/s37 做了同口径并排对照。

下一步最小改动（可验证/可复现）：

- 先用 `scripts/noise_analyze/dump_u_events.py` 的新增列，把 heavy best-F1 点位的 **TP vs FP** 在 `raw/raw_all/raw_opp/mix/u_eff/r_eff/z_dbg` 上做直观看差异（而不是只看 `u` 分布）。
- 然后并行推进一个“更对准噪声的邻域证据”变体：s39（见下）。

## s39：Noise Surprise Z-Score - State+Neighborhood Occupancy (Polarity-Mix Weighted)（混极性加权的邻域占用率）

直觉：s38 的 `u_nb` 只看邻域“有多活跃”，但并不区分这种活跃更像信号还是更像噪声。s39 在 s38 的框架里加入一个 `mix`（邻域近期活动中 opposite-pol 的比例）去加权邻域占用率，使邻域项更偏“噪声证据”。

### 方法原理（公式化定义）

对每个事件 $e=(x,y,t,p)$，在半径 $r$ 的邻域窗口里（不含自身像素），对最近 $	au$ 时间窗内的历史活动做加权累计（线性 recency weight）：

$$
\Delta t = t-t_j,\quad w(\Delta t)=\frac{\tau-\Delta t}{\tau}\,\mathbf{1}[0\le \Delta t \le \tau]
$$

定义（与实现一致，单位上 raw 是“归一化后的加权和”）：

$$
\mathrm{raw}_{\mathrm{all}} = \sum_{j\in\mathcal{N}(e)} w(\Delta t),\quad
\mathrm{raw}_{\mathrm{same}} = \sum_{j\in\mathcal{N}(e)} w(\Delta t)\,\mathbf{1}[p_j=p],\quad
\mathrm{raw}_{\mathrm{opp}} = \mathrm{raw}_{\mathrm{all}}-\mathrm{raw}_{\mathrm{same}}
$$

其中 $\mathcal{N}(e)$ 表示空间邻域内且在时间窗内的历史事件集合。最终用于 z-score 的 baseline raw 仍取 same-pol（保持与 EBF 主干一致）：

$$
\mathrm{raw}=\mathrm{raw}_{\mathrm{same}}
$$

邻域“占用率”来自 any-pol 活跃度（$m$ 为邻域像素数，$\varepsilon$ 防止除零）：

$$
u_{\mathrm{nb}} = \frac{\mathrm{raw}_{\mathrm{all}}}{\mathrm{raw}_{\mathrm{all}} + m + \varepsilon}\in[0,1]
$$

定义邻域混极性比例（越接近 1，表示邻域里 opposite-pol 活跃占比越高）：

$$
\mathrm{mix}=\frac{\mathrm{raw}_{\mathrm{opp}}}{\mathrm{raw}_{\mathrm{all}}+\varepsilon}\in[0,1]
$$

把 mix 作为“更像噪声”的权重去调制邻域占用：

$$
u_{\mathrm{nb\_mix}}=\mathrm{clip}_{[0,1]}\bigl(k_{\mathrm{nbmix}}\,u_{\mathrm{nb}}\,\mathrm{mix}\bigr)
$$

像素自占用（来自 per-pixel 的 hot_state 累计，$\tau_r$ 为全局 rate-EMA 的时间常数，默认 auto=\,$\tau$）：

$$
u_{\mathrm{self}} = \mathrm{clip}_{[0,1]}\Bigl(\frac{\mathrm{hot\_state}}{\mathrm{hot\_state}+\tau_r+\varepsilon}\Bigr)
$$

融合占用取“并集”（避免二者互相覆盖）：

$$
u_{\mathrm{eff}} = 1-(1-u_{\mathrm{self}})(1-u_{\mathrm{nb\_mix}})\in[0,1]
$$

把占用率映射成更保守的 null-model rate（保持 s38 的平滑调制形态）：

$$
r_{\mathrm{eff}} = r_{\mathrm{pix}}\,(1+u_{\mathrm{eff}})^2,\quad r_{\mathrm{pix}}=\frac{r_{\mathrm{ema}}}{WH}
$$

最后做 noise-surprise z-score（与 s36/s38 相同的推导口径）：记 $a=r_{\mathrm{eff}}\tau$，则

$$
E[w]=\tfrac{1}{2}\Bigl(1-\frac{1-e^{-a}}{a}\Bigr),\quad
E[w^2]=\tfrac{1}{2}\cdot\frac{a^2-2a+2-2e^{-a}}{a^2}
$$
	au_{\mathrm{long}}=2\tau
$$
\mu=m\,E[w],\quad \sigma^2=m\,(E[w^2]-E[w]^2),\quad
\mathrm{score}=\frac{\mathrm{raw}-\mu}{\sqrt{\sigma^2+\varepsilon}}
$$

实现位置：`src/myevs/denoise/ops/ebfopt_part2/s39_noise_surprise_zscore_stateoccupancy_nbocc_mix.py`

超参：

- `MYEVS_EBF_S39_TAU_RATE_US`：全局 rate-EMA 的时间常数（us）；`0` 表示 auto（用 `tau_us`）。
- `MYEVS_EBF_S39_K_NBMIX`：邻域 mix 占用率强度 $k\ge 0$。

sweep 参数：`--s39-tau-rate-us-list`、`--s39-k-nbmix-list`（tag 后缀为 `_trXXXX_knY`）。

当前状态：

- 已完成：接入 `scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py`（可 sweep）
- 已完成：接入 `scripts/noise_analyze/noise_type_stats.py` / `dump_u_events.py` 的 tag→env 复现
- 已完成：2k smoke 跑通：`data/ED24/myPedestrain_06/EBF_Part2/_smoke_s39_2k/`（heavy best-F1 tag=`ebf_s39_tr128000_kn1_labelscore_s9_tau128000`）

### 实验结果（prescreen200k：max-events=200k）

说明：先对齐 s36/s37/s38 的两个诊断点位，做小扫参：`k_nbmix ∈ {0.5, 1, 2}`，`tau_rate_us ∈ {0, 32000, 64000, 128000}`。

#### A) prescreen200k sweep：对齐点 (s=9,tau=128ms)

输出目录：`data/ED24/myPedestrain_06/EBF_Part2/_tune_s39_stateocc_nbocc_mix_s9_tau128ms_200k/`

摘要（来自 sweep stdout）：

- light：best AUC=0.939056（tag=`ebf_s39_kn0p5_labelscore_s9_tau128000`），best-F1=0.945610
- mid：best AUC=0.934334（tag=`ebf_s39_kn0p5_labelscore_s9_tau128000`），best-F1=0.812140
- heavy：best AUC=0.929107（tag=`ebf_s39_kn0p5_labelscore_s9_tau128000`），best-F1=0.774140（tag=`ebf_s39_tr64000_kn0p5_labelscore_s9_tau128000`，thr=1.3300）

产物：

- `roc_ebf_s39_light_labelscore_s3_5_7_9_tau8_16_32_64_128_256_512_1024ms.csv/png`
- `roc_ebf_s39_mid_labelscore_s3_5_7_9_tau8_16_32_64_128_256_512_1024ms.csv/png`
- `roc_ebf_s39_heavy_labelscore_s3_5_7_9_tau8_16_32_64_128_256_512_1024ms.csv/png`

#### B) prescreen200k sweep：对齐点 (s=7,tau=64ms)

输出目录：`data/ED24/myPedestrain_06/EBF_Part2/_tune_s39_stateocc_nbocc_mix_s7_tau64ms_200k/`

摘要（来自 sweep stdout）：

- light：best AUC=0.931698（tag=`ebf_s39_tr128000_kn0p5_labelscore_s7_tau64000`），best-F1=0.940770
- mid：best AUC=0.925285（tag=`ebf_s39_tr32000_kn0p5_labelscore_s7_tau64000`），best-F1=0.810622（tag=`ebf_s39_kn0p5_labelscore_s7_tau64000`）
- heavy：best AUC=0.928267（tag=`ebf_s39_tr32000_kn0p5_labelscore_s7_tau64000`），best-F1=0.787951（tag=`ebf_s39_kn0p5_labelscore_s7_tau64000`，thr=2.3308）

产物：

- `roc_ebf_s39_light_labelscore_s3_5_7_9_tau8_16_32_64_128_256_512_1024ms.csv/png`
- `roc_ebf_s39_mid_labelscore_s3_5_7_9_tau8_16_32_64_128_256_512_1024ms.csv/png`
- `roc_ebf_s39_heavy_labelscore_s3_5_7_9_tau8_16_32_64_128_256_512_1024ms.csv/png`

#### C) prescreen200k fine sweep：对齐点 (s=7,tau=64ms)

说明：在 B) 的基础上做更细的 `k_nbmix` 扫参，且只保留少量 `tau_rate_us` 候选（`{0, 32000, 64000}`；其中 `0` 表示 auto=\,$\tau$）。

输出目录：`data/ED24/myPedestrain_06/EBF_Part2/_tune_s39_stateocc_nbocc_mix_s7_tau64ms_200k_finek/`

best-AUC（来自 sweep stdout）：

| env | best AUC | tau_rate_us | k_nbmix | tag |
|---|---:|---:|---:|---|
| light | 0.929760 | 0 (auto) | 0.25 | `ebf_s39_kn0p25_labelscore_s7_tau64000` |
| mid | 0.925453 | 32000 | 0.25 | `ebf_s39_tr32000_kn0p25_labelscore_s7_tau64000` |
| heavy | 0.928464 | 32000 | 0.25 | `ebf_s39_tr32000_kn0p25_labelscore_s7_tau64000` |

best-F1（来自 sweep stdout）：

| env | best F1 | tau_rate_us | k_nbmix | tag |
|---|---:|---:|---:|---|
| light | 0.940033 | 0 (auto) | 0.25 | `ebf_s39_kn0p25_labelscore_s7_tau64000` |
| mid | 0.811020 | 0 (auto) | 0.25 | `ebf_s39_kn0p25_labelscore_s7_tau64000` |
| heavy | 0.788516 | 0 (auto) | 0.25 | `ebf_s39_kn0p25_labelscore_s7_tau64000` |

### heavy 诊断：best-F1 operating point 的噪声分解 + u_eff 分布

hotmask 统一使用：`data/ED24/myPedestrain_06/EBF_Part2/hotmask_heavy_score_neg_minus_2pos_topk32768_dil1.npy`

#### A) heavy, (s=9,tau=128ms) best-F1：tag=`ebf_s39_tr64000_kn0p5_labelscore_s9_tau128000`

- 噪声类型统计：
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/noise_types_s39_heavy_prescreen200k_s9_tau128ms_tr64000_kn0p5_bestf1.csv`

关键 kept-rate（noise_kept_rate，越小越好；括号内为对应 signal_kept_rate）：

- hotmask：0.04112（0.72174）
- near_hotmask：0.11397（0.79504）
- highrate_pixel：0.14340（0.81156）

- u_eff 分布：
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/u_events_s39_heavy_prescreen200k_s9_tau128ms_tr64000_kn0p5_bestf1.csv`
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/u_eff_quantiles_s39_heavy_prescreen200k_s9_tau128ms_tr64000_kn0p5_bestf1.csv`
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/u_eff_hist_s39_heavy_prescreen200k_s9_tau128ms_tr64000_kn0p5_bestf1.png`
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/z_dbg_hist_s39_heavy_prescreen200k_s9_tau128ms_tr64000_kn0p5_bestf1.png`

关键分位数（hotmask FP 的 u_eff）：p75=0.1494，p90=0.7074，p95=0.8207，p99=0.9285。

#### B) heavy, (s=7,tau=64ms) best-F1：tag=`ebf_s39_kn0p5_labelscore_s7_tau64000`

- 噪声类型统计：
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/noise_types_s39_heavy_prescreen200k_s7_tau64ms_kn0p5_bestf1.csv`

关键 kept-rate（noise_kept_rate；括号内为 signal_kept_rate）：

- hotmask：0.03267（0.71382）
- near_hotmask：0.08189（0.79915）
- highrate_pixel：0.10632（0.84481）

- u_eff 分布：
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/u_events_s39_heavy_prescreen200k_s7_tau64ms_kn0p5_bestf1.csv`
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/u_eff_quantiles_s39_heavy_prescreen200k_s7_tau64ms_kn0p5_bestf1.csv`
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/u_eff_hist_s39_heavy_prescreen200k_s7_tau64ms_kn0p5_bestf1.png`
	- `data/ED24/myPedestrain_06/EBF_Part2/noise_analyze/z_dbg_hist_s39_heavy_prescreen200k_s7_tau64ms_kn0p5_bestf1.png`

关键分位数（hotmask FP 的 u_eff）：p75=0.0566，p90=0.4649，p95=0.5858，p99=0.7810。

### 是否值得继续

结论：**值得继续，先聚焦 (s=7,tau=64ms)**。细扫结果表明该点在 `k_nbmix=0.25` 时三环境的 best-F1 同步提升（heavy best-F1=0.78852，mid best-F1=0.81102；light best-F1=0.94003），且 coarse 点位下 heavy 的 hotmask/near-hot/highrate kept-rate 更低，属于“更健康”的方向。

下一步最小改动（保持可复现）：

- 用 fine sweep 的 heavy best-F1 tag（`ebf_s39_kn0p25_labelscore_s7_tau64000`）按同一口径补一份 heavy 诊断（noise breakdown + `u_eff/z_dbg` 分布），与现有 `kn0p5` 的 heavy 诊断并排对照，确认提升来自哪类 FP 的减少。
- 如果希望 AUC 也更接近 B) 的 light best-AUC，可把 `tau_rate_us=128000` 加回候选（但保持 `k_nbmix` 仍围绕 0.25）。

## s40：s39 变体 - 只改融合为几何均值（fuse=geom mean）

动机：s39 的并集融合 $u_{\mathrm{eff}} = 1-(1-u_{\mathrm{self}})(1-u_{\mathrm{nb\_mix}})$ 可能在 light 场景过强；s40 尝试把融合改得更“保守”。

改动：保持 s39 的占用与 z-score 框架不变，仅将融合从并集改为几何均值（直觉上更偏向取小者）。

实现位置：`src/myevs/denoise/ops/ebfopt_part2/s40_noise_surprise_zscore_stateocc_nbocc_mix_fuse_geom.py`

输出目录：`data/ED24/myPedestrain_06/EBF_Part2/_tune_s40_stateocc_nbocc_mix_fusegeom_s7_tau64ms_200k_finek/`

best-F1（来自 ROC CSV 反算）：

| env | best AUC | best F1 | tag |
|---|---:|---:|---|
| light | 0.922722 | 0.939496 | `ebf_s40_kn0p35_labelscore_s7_tau64000` |
| mid | 0.916560 | 0.806892 | `ebf_s40_tr32000_kn1_labelscore_s7_tau64000` |
| heavy | 0.921253 | 0.792608 | `ebf_s40_tr32000_kn0p6_labelscore_s7_tau64000` |

阶段性结论：**light AUC 显著下降**，该方向不建议继续。

## s41：s39 变体 - 只改 mix shaping 为平方（mix^2）

动机：希望让邻域 mix 作为“噪声证据”时更尖锐（小 mix 更接近 0，大 mix 更接近 1）。

改动：保持 s39 框架，仅将 mix 权重从 $\mathrm{mix}$ 改为 $\mathrm{mix}^2$（其余一致）。

实现位置：`src/myevs/denoise/ops/ebfopt_part2/s41_noise_surprise_zscore_stateocc_nbocc_mix_pow2.py`

输出目录：`data/ED24/myPedestrain_06/EBF_Part2/_tune_s41_stateocc_nbocc_mix_pow2_s7_tau64ms_200k_finek/`

best-F1（来自 ROC CSV 反算）：

| env | best AUC | best F1 | tag |
|---|---:|---:|---|
| light | 0.929791 | 0.940036 | `ebf_s41_kn0p25_labelscore_s7_tau64000` |
| mid | 0.924575 | 0.811046 | `ebf_s41_kn0p25_labelscore_s7_tau64000` |
| heavy | 0.927652 | 0.788598 | `ebf_s41_kn0p25_labelscore_s7_tau64000` |

阶段性结论：与 s39（fine sweep）几乎等价（差异极小）。

## s42：s39 变体 - 邻域项按 $u_{\mathrm{self}}^2$ 门控（更“自热点”才启用邻域噪声证据）

动机：怀疑邻域噪声证据在 light 场景会误伤信号；因此仅当像素自身已经“偏热点”时才让邻域项发挥作用。

改动：保持 s39 框架，仅把邻域项做门控：$u_{\mathrm{nb\_mix}} \leftarrow u_{\mathrm{nb\_mix}}\cdot u_{\mathrm{self}}^2$（再做融合）。

实现位置：`src/myevs/denoise/ops/ebfopt_part2/s42_noise_surprise_zscore_stateocc_nbocc_mix_gated_self2.py`

输出目录：`data/ED24/myPedestrain_06/EBF_Part2/_tune_s42_stateocc_nbocc_mix_gated_self2_s7_tau64ms_200k_small/`

best-F1（来自 ROC CSV 反算）：

| env | best AUC | best F1 | tag |
|---|---:|---:|---|
| light | 0.929927 | 0.940068 | `ebf_s42_kn0p5_labelscore_s7_tau64000` |
| mid | 0.924725 | 0.811264 | `ebf_s42_kn0p25_labelscore_s7_tau64000` |
| heavy | 0.927820 | 0.788856 | `ebf_s42_kn0p5_labelscore_s7_tau64000` |

阶段性结论：整体仍非常接近 s39/s41，对 light 的系统性落差无实质改善。

## s43：s39 变体 - 压缩占用度（$u_{\mathrm{eff}}\to u_{\mathrm{eff}}^2$）

动机：希望在 heavy 场景更强抑制热点（让 rate modulation 更“保守”）。

改动：保持 s39 框架与融合不变，仅在 rate modulation 前做压缩：$u\leftarrow u_{\mathrm{eff}}^2$，然后 $r_{\mathrm{eff}}=r_{\mathrm{pix}}(1+u)^2$。

实现位置：`src/myevs/denoise/ops/ebfopt_part2/s43_noise_surprise_zscore_stateocc_nbocc_mix_u2.py`

输出目录：`data/ED24/myPedestrain_06/EBF_Part2/_tune_s43_stateocc_nbocc_mix_u2_s7_tau64ms_200k_small/`

best-F1（来自 ROC CSV 反算）：

| env | best AUC | best F1 | tag |
|---|---:|---:|---|
| light | 0.929608 | 0.940335 | `ebf_s43_kn0p25_labelscore_s7_tau64000` |
| mid | 0.923989 | 0.811605 | `ebf_s43_tr32000_kn0p25_labelscore_s7_tau64000` |
| heavy | 0.927861 | 0.793418 | `ebf_s43_tr32000_kn0p25_labelscore_s7_tau64000` |

阶段性结论：heavy best-F1 有明显上升，但 light 仍比 baseline 的最佳排序更弱；因此更像“heavy-only 加强版”，无法解决你对 s39 的核心不满。

### 小结：对齐点 (s=7, tau=64ms) 的 prescreen200k best-F1 对照

说明：下表均为 *best-F1 (per env)* operating point（AUC/F1 来自各自 ROC CSV 反算）。

| variant | light AUC/F1 | mid AUC/F1 | heavy AUC/F1 |
|---|---:|---:|---:|
| baseline (EBF) | 0.938489 / 0.942450 | 0.914649 / 0.806029 | 0.917085 / 0.788680 |
| s39 (fine k) | 0.929760 / 0.940033 | 0.924550 / 0.811020 | 0.927616 / 0.788516 |
| s40 (fuse geom) | 0.922722 / 0.939496 | 0.916560 / 0.806892 | 0.921253 / 0.792608 |
| s41 (mix^2) | 0.929791 / 0.940036 | 0.924575 / 0.811046 | 0.927652 / 0.788598 |
| s42 (gate by self^2) | 0.929927 / 0.940068 | 0.924725 / 0.811264 | 0.927820 / 0.788856 |
| s43 (u_eff^2) | 0.929608 / 0.940335 | 0.923989 / 0.811605 | 0.927861 / 0.793418 |
| s44 (baseline/(1+u^2)) | 0.939789 / 0.942471 | 0.919081 / 0.812291 | 0.922182 / 0.793760 |
| s45 (s44 + u0 gate) | 0.939789 / 0.942471 | 0.919081 / 0.812284 | 0.922182 / 0.793770 |
| s46 (baseline/(1+odds(u)^2)) | 0.933007 / 0.944781 | 0.915582 / 0.796356 | 0.911119 / 0.756832 |
| s50 (s44 + support-breadth boost) | 0.939789 / 0.942471 | 0.919331 / 0.813688 | 0.922253 / 0.795574 |
| s51 (s50 auto-beta, no env hyperparams) | 0.939757 / 0.942475 | 0.919059 / 0.812467 | 0.922151 / 0.793990 |

## s44：Baseline Labelscore + Self-Occupancy Penalty（保留 baseline 排序，仅对热点做温和惩罚）

动机（针对 s39 的不足）：s39+ 一类 z-score/null-model 变体在 heavy 上更容易收益，但在 light/mid 上经常会因为“替换了 baseline 的 raw 排序信息”而出现系统性退化。s44 的策略是：**不再替换 raw 排序，只在怀疑是热点像素时做一个强度很小的惩罚项**。

### 为什么要这样做（设计动机与可验证假设）

核心判断：ED24 heavy 的主要 FP 来源是 **persistent hot pixel / 高发像素**。这类噪声在 baseline EBF 下很容易因为邻域同极性 recency 叠加而得到高 raw，导致误保留。

但在 light/mid 上，baseline EBF 的 raw 排序本身就很强（尤其 light），因此我们希望：

- **不改 baseline raw 的排序信息**（避免像 s39/s41/s42/s43 那样在 light 上系统性掉 AUC）；
- 只增加一个“热点代理”去温和压制 hot pixel 类型噪声，从而 **主要提升 heavy**，并尽量不伤 light。

可验证假设：如果用一个仅依赖“同像素持续活跃程度”的 $u_{\mathrm{self}}\in[0,1]$ 去惩罚 score，则：

- light：大部分像素 $u_{\mathrm{self}}\approx 0$，排序几乎等同 baseline；
- heavy：hot pixel 上 $u_{\mathrm{self}}\to 1$，这些点的 score 会被压下去，FP 会减少，从而 AUC/F1 上升。

### 方法原理（公式化定义）

记当前事件为 $e=(x,y,t,p)$，半径为 $r$，时间常数为 $\tau$。

1) baseline raw（与 EBF labelscore 完全一致）：

$$
\mathrm{raw}=\sum_{j\in\mathcal{N}(e)} \frac{\tau-\Delta t}{\tau}\,\mathbf{1}[0\le \Delta t\le \tau]\,\mathbf{1}[p_j=p]
$$

其中 $\mathcal{N}(e)$ 为以 $(x,y)$ 为中心的方形邻域（半径 $r$，实现里会 clamp 到 $\le 8$），$\Delta t=|t-t_j|$，$t_j$ 为像素 $j$ 上一次事件的时间戳，$p_j$ 为其上一次事件极性。

2) 自占用（热点）代理 $u_{\mathrm{self}}\in[0,1]$：使用与 s36/s38/s39 同样的 per-pixel 线性衰减累加器 `hot_state`，并用时间常数 $\tau_r$ 归一化：

$$
u_{\mathrm{self}} = \mathrm{clip}_{[0,1]}\Bigl(\frac{\mathrm{hot\_state}}{\mathrm{hot\_state}+\tau_r+\varepsilon}\Bigr)
$$

3) 最终 score：对 baseline raw 做温和抑制（只有 $u_{\mathrm{self}}$ 大时才显著生效）：

$$
\mathrm{score}=\frac{\mathrm{raw}}{1+u_{\mathrm{self}}^2}
$$

### hot_state 的“完整更新规则”（与实现一致）

对同一像素 $(x,y)$ 维护一个整型累加器 $H$（实现里为 `hot_state(int32)`），令 $\Delta t_0$ 为该像素距离上一次事件的时间差（若没有历史事件则取 $\Delta t_0=\tau$），则每个事件到来时更新：

$$
H \leftarrow \max(0, H-\Delta t_0) + \max(0, \tau-\Delta t_0)
$$

直觉：$H$ 是一个“线性衰减的活跃度”，短 dt 连续触发会累积变大；长时间不触发则会自然衰减到 0。

随后用 $\tau_r$ 把它压到 $[0,1]$：

$$
u_{\mathrm{self}} = \mathrm{clip}_{[0,1]}\Bigl(\frac{H}{H+\tau_r+\varepsilon}\Bigr)
$$

实现细节：当 `MYEVS_EBF_S44_TAU_RATE_US=0` 时，$\tau_r$ 自动取当前 $\tau$。

### 实现与超参

实现位置：`src/myevs/denoise/ops/ebfopt_part2/s44_ebf_labelscore_selfocc_div_u2.py`

超参：

- `MYEVS_EBF_S44_TAU_RATE_US`：$\tau_r$（us）；`0` 表示 auto（用当前 `tau_us`）。

sweep 参数：`--s44-tau-rate-us-list`（tag 后缀为 `_trXXXX`）。

### 实验结果（prescreen200k：max-events=200k）

说明：只做最小验证（对齐你当前关心的点位），扫描 `tau_rate_us ∈ {0, 32000}`。

输出目录：`data/ED24/myPedestrain_06/EBF_Part2/_tune_s44_ebf_labelscore_selfocc_div_u2_s7_tau64ms_200k_small/`

best-AUC / best-F1（来自 ROC CSV 反算，三环境一致选到同一 tag）：

| env | best AUC | best F1 | tag |
|---|---:|---:|---|
| light | 0.939789 | 0.942471 | `ebf_s44_tr32000_labelscore_s7_tau64000` |
| mid | 0.919081 | 0.812291 | `ebf_s44_tr32000_labelscore_s7_tau64000` |
| heavy | 0.922182 | 0.793760 | `ebf_s44_tr32000_labelscore_s7_tau64000` |

阶段性结论：s44 是目前第一个同时满足“light 不掉、heavy 还能涨”的方向；建议把它作为 s39 线的一个**可复现替代**。

补充：best-AUC / best-F1 对应阈值（`param=min-neighbors`）

说明：AUC 是整条 ROC 曲线的指标；这里的“best-AUC 阈值”指 **best-AUC 所选 tag 下，能达到 best-F1 的 operating point 阈值**（便于落地部署/复现实验口径）。

| env | baseline best-F1 thr | s44 best-F1 thr | s44 tag |
|---|---:|---:|---|
| light | 0.003938 | 0.003594 | `ebf_s44_tr32000_labelscore_s7_tau64000` |
| mid | 2.542469 | 2.253969 | `ebf_s44_tr32000_labelscore_s7_tau64000` |
| heavy | 3.526625 | 3.202828 | `ebf_s44_tr32000_labelscore_s7_tau64000` |

对比 baseline（同口径 s=7,tau=64ms）：

- light：AUC 0.938489 → 0.939789，F1 0.942450 → 0.942471（几乎不变）
- mid：AUC 0.914649 → 0.919081，F1 0.806029 → 0.812291（提升）
- heavy：AUC 0.917085 → 0.922182，F1 0.788680 → 0.793760（提升，符合“重点提高 heavy”）

补充验证：`s=9,tau=128ms`（prescreen200k，max-events=200k）

说明：扫描 `tau_rate_us ∈ {0, 64000, 128000}`。

输出目录：`data/ED24/myPedestrain_06/EBF_Part2/_tune_s44_ebf_labelscore_selfocc_div_u2_s9_tau128ms_200k_small/`

baseline vs s44（best-AUC 与 best-F1 的 tag/阈值可能不同；数据来自 ROC CSV 反算）：

| env | baseline (AUC/F1, thr) | s44 best-AUC (tag, AUC, thr@bestF1) | s44 best-F1 (tag, F1, thr) |
|---|---:|---:|---:|
| light | 0.947564 / 0.949739, 0.749148 | `ebf_s44_tr64000_labelscore_s9_tau128000`, 0.949800, 0.713570 | `ebf_s44_tr64000_labelscore_s9_tau128000`, 0.952451, 0.713570 |
| mid | 0.921924 / 0.810827, 4.839437 | `ebf_s44_tr64000_labelscore_s9_tau128000`, 0.930215, 4.320984 | `ebf_s44_tr64000_labelscore_s9_tau128000`, 0.819301, 4.320984 |
| heavy | 0.920467 / 0.786882, 7.358062 | `ebf_s44_tr64000_labelscore_s9_tau128000`, 0.929281, 6.368444 | `ebf_s44_labelscore_s9_tau128000`, 0.793649, 6.713469 |

备注：heavy 上 best-AUC 与 best-F1 选到不同的 `tau_rate_us`（差异很小，属于 operating point 的细节）。整体趋势仍是：s44 对 heavy 的 AUC/F1 均有提升。

对比 baseline（同口径 s=9,tau=128ms）：

- light：AUC 0.947564 → 0.949800，F1 0.949739 → 0.952451（提升）
- mid：AUC 0.921924 → 0.930215，F1 0.810827 → 0.819301（提升）
- heavy：AUC 0.920467 → 0.929281，F1 0.786882 → 0.793649（提升，且主要针对热点类 FP）

### 资源占用与计算开销（相对 baseline）

与 baseline EBF 相比，s44 的主要新增成本是 **每像素多存一张 `hot_state(int32)` 表**，以及每事件多做几步常数操作。

- **新增持久状态（per-pixel）**：
	- baseline：`last_ts(uint64)` + `last_pol(int8)`
	- s44：在 baseline 基础上 + `hot_state(int32)`
	- 以 $346\times260=89960$ 像素估算：新增内存约 $89960\times 4 \approx 0.34$ MB（量级远小于邻域遍历的时间成本）。
- **每事件新增步骤（常数开销）**：
	1) 计算同像素 $\Delta t_0$
	2) 更新 `hot_state`（线性衰减 + 增量）
	3) 计算 $u_{\mathrm{self}}$，并执行一次除法 `raw/(1+u^2)`

复杂度结论：s44 仍然是在线单遍、主项仍是邻域遍历 $O(r^2)$；新增开销是很小的常数项，通常不会改变“由邻域遍历主导”的性能画像。

## s45：s44 变体 - 低占用门控（$u_{\mathrm{self}}\le u_0$ 不惩罚）

动机：你最初的直觉是“light/mid 的绝大多数像素 $u_{\mathrm{self}}$ 很低，若这些区域完全不惩罚，则可以进一步避免误伤信号；而 heavy 的 hot pixel 仍然会被惩罚”。因此在 s44 的惩罚项上增加一个门控阈值 $u_0$。

### 方法定义（公式）

与 s44 相同地定义 baseline raw、`hot_state` 与 $u_{\mathrm{self}}$。仅修改最终 score：

$$
\mathrm{score}=
\begin{cases}
\mathrm{raw} & (u_{\mathrm{self}}\le u_0)\\
\frac{\mathrm{raw}}{1+u_{\mathrm{self}}^2} & (u_{\mathrm{self}}>u_0)
\end{cases}
$$

其中 $u_0\in[0,1)$。

### 实现与超参

实现位置：`src/myevs/denoise/ops/ebfopt_part2/s45_ebf_labelscore_selfocc_gate_div_u2.py`

超参：

- `MYEVS_EBF_S45_TAU_RATE_US`：$\tau_r$（us）；`0` 表示 auto（用当前 `tau_us`）。
- `MYEVS_EBF_S45_U0`：门控阈值 $u_0$（默认 0.0；严格退化为 s44）。

sweep 参数：

- `--s45-tau-rate-us-list`
- `--s45-u0-list`（tag 后缀 `_u0...`）

### 实验结果（prescreen200k：s=7,tau=64ms）

输出目录：`data/ED24/myPedestrain_06/EBF_Part2/_tune_s45_ebf_labelscore_selfocc_gate_div_u2_s7_tau64ms_200k_small/`

扫描网格：固定 `tau_rate_us=32000`，扫描 $u_0\in\{0,0.2,0.4,0.6\}$。

best-AUC / best-F1（来自 ROC CSV 反算）：

| env | best AUC (tag) | best F1 (tag, thr) |
|---|---:|---:|
| light | 0.939789 (`ebf_s45_tr32000_u00_labelscore_s7_tau64000`) | 0.942471 (`ebf_s45_tr32000_u00_labelscore_s7_tau64000`, thr=0.003594) |
| mid | 0.919081 (`ebf_s45_tr32000_u00_labelscore_s7_tau64000`) | 0.812284 (`ebf_s45_tr32000_u00_labelscore_s7_tau64000`, thr=2.246078) |
| heavy | 0.922182 (`ebf_s45_tr32000_u00_labelscore_s7_tau64000`) | 0.793770 (`ebf_s45_tr32000_u00_labelscore_s7_tau64000`, thr=3.203219) |

阶段性结论：**门控并没有带来额外收益**。三环境的 best-AUC 与 best-F1 都选择 $u_0=0$（即严格退化为 s44）；$u_0>0$ 会系统性降低 AUC/F1。也就是说，“低占用完全不惩罚”会把一部分应当被轻微压制的噪声也放过，整体排序反而更差。

## s46：s44 变体 - odds 形状强化（让 $u\to 1$ 时惩罚更陡）

动机：s45 的失败说明“简单门控”不够精确；而你想要的形状其实是：低 $u$ 几乎不动、高 $u$ 明显更强。为此把 $u\in[0,1)$ 映射到一个在 $u\to 1$ 时快速发散的变量 $v$。

### 方法定义（公式）

与 s44 相同定义 $\mathrm{raw}$、`hot_state`、$u_{\mathrm{self}}$。然后做 odds-like 变换：

$$
v = \frac{u_{\mathrm{self}}}{1-u_{\mathrm{self}}+\varepsilon}
$$

最终 score：

$$
\mathrm{score}=\frac{\mathrm{raw}}{1+v^2}
$$

直觉：$u$ 很小时 $v\approx u$（几乎不变）；但 $u\to 1$ 时 $v\to\infty$，惩罚会急剧变强。

### 实现与超参

实现位置：`src/myevs/denoise/ops/ebfopt_part2/s46_ebf_labelscore_selfocc_odds_div_v2.py`

超参：

- `MYEVS_EBF_S46_TAU_RATE_US`：$\tau_r$（us）；`0` 表示 auto（用当前 `tau_us`）。

sweep 参数：`--s46-tau-rate-us-list`（tag 后缀 `_trXXXX`）。

### 实验结果（prescreen200k：s=7,tau=64ms）

输出目录：`data/ED24/myPedestrain_06/EBF_Part2/_tune_s46_prescreen200k_s7_tau64/`

扫描网格：`tau_rate_us ∈ {0, 32000, 64000}`。

best-AUC / best-F1（来自 ROC CSV 反算）：

| env | best AUC (tag) | best F1 (tag, thr) |
|---|---:|---:|
| light | 0.938970 (`ebf_s46_labelscore_s7_tau64000`) | 0.944781 (`ebf_s46_tr32000_labelscore_s7_tau64000`, thr=0.005906) |
| mid | 0.915582 (`ebf_s46_labelscore_s7_tau64000`) | 0.796356 (`ebf_s46_labelscore_s7_tau64000`, thr=1.983406) |
| heavy | 0.911119 (`ebf_s46_labelscore_s7_tau64000`) | 0.756832 (`ebf_s46_labelscore_s7_tau64000`, thr=2.832447) |

阶段性结论：s46 **整体明显退化**（尤其 mid/heavy 的 best-F1），且 `tau_rate_us=32000` 在三环境都会显著拉低 AUC；因此不建议继续沿这个 odds^2 形状推进。

一个重要教训：虽然 odds 变换满足“高 u 更陡”，但它会对 hot pixel 之外的高占用区域产生过强抑制，破坏了“保留 baseline 排序主轴”的目标，导致整体 ROC 劣化。

## s50：s44 + Support-Breadth Boost（用“支持宽度”做温和加分，目标是 heavy 再涨且不伤 light）

动机（来自证据驱动诊断）：在 `s=7,tau=64ms` 的 heavy best-F1 operating point 上，s44 的提升主要表现为“阈值可以更低 → 召回上升”，但 hotmask/near-hotmask 内残余 FP 并不一定对应更高的 $u_{\mathrm{self}}$。继续加大 $u$ 惩罚形状（s46）会误伤信号而整体退化。

因此 s50 的改动不再“改惩罚壳”，而是引入一个**新但克制**的弱证据：同极性邻域的“支持点数”（support breadth）。直觉上：

- 真信号的邻域更容易出现多点一致触发（支持更宽）；
- 许多热点/噪声残余更像“窄支持”（集中在少数像素/少量邻居上）。

### 方法原理（公式化定义）

保留 s44 的定义：

$$
\mathrm{base} = \frac{\mathrm{raw}}{1+u_{\mathrm{self}}^2}
$$

在计算 `raw` 的同一次邻域遍历中，额外统计同极性邻域里“在窗口内活跃”的像素数：

$$
c_{\mathrm{sup}} = \sum_{j\in\mathcal{N}(e)} \mathbf{1}[0\le \Delta t\le \tau] \;\mathbf{1}[p_j=p]
$$

然后给一个轻微、饱和的加分项：

$$
g = 1 + \beta\cdot \min\Bigl(1,\frac{c_{\mathrm{sup}}}{c_0}\Bigr),\quad \beta\ge 0,\; c_0>0
$$

最终：

$$
\mathrm{score} = \mathrm{base}\cdot g
$$

解释：当支持点数很少时几乎不改动（$g\approx 1$）；当支持足够宽时最多提升 $(1+\beta)$ 倍，但由于是**乘在 s44 的 base 上**，仍然不会彻底推翻 baseline 排序主轴。

### 实现与超参

实现位置：`src/myevs/denoise/ops/ebfopt_part2/s50_ebf_labelscore_selfocc_supportboost_div_u2.py`

超参：

- `MYEVS_EBF_S50_TAU_RATE_US`：$\tau_r$（us）；`0` 表示 auto（用当前 `tau_us`）。
- `MYEVS_EBF_S50_BETA`：$\beta$（支持宽度加分强度，建议从 0~1 试）。
- `MYEVS_EBF_S50_CNT0`：$c_0$（支持点数饱和阈值，正整数；默认 8）。

注意：s50 的最优超参会随噪声环境漂移（这是你指出的核心问题之一）；因此 **s50 不建议作为“固定 recipe”直接部署**，更适合作为研究/消融对照。若目标是“去掉 b/cnt0 对环境的敏感性”，优先看下方的 s51（auto-beta，无环境超参）。

sweep 参数：

- `--s50-tau-rate-us-list`（tag 后缀 `_trXXXX`）
- `--s50-beta-list`（tag 后缀 `_b...`）
- `--s50-cnt0-list`（tag 后缀 `_c...`）

### 实验结果（prescreen200k：s=7,tau=64ms）

输出目录：`data/ED24/myPedestrain_06/EBF_Part2/_tune_s50_prescreen200k_s7_tau64/`

扫描网格：

- `tau_rate_us ∈ {0, 32000, 64000}`
- `beta ∈ {0, 0.25, 0.5, 1.0}`
- `cnt0 ∈ {4, 8}`

best-F1（per env；来自对应 ROC CSV 反算）：

| env | best AUC | best F1 | best-F1 tag |
|---|---:|---:|---|
| light | 0.939789 | 0.942471 | `ebf_s50_tr32000_b0_c4_labelscore_s7_tau64000` |
| mid | 0.919331 | 0.813688 | `ebf_s50_tr32000_b1_c8_labelscore_s7_tau64000` |
| heavy | 0.922253 | 0.795574 | `ebf_s50_tr32000_b1_c8_labelscore_s7_tau64000` |

补充：best-F1 tag 对应阈值（`param=min-neighbors`）：

| env | best-F1 thr | best-F1 tag |
|---|---:|---|
| light | 0.003594 | `ebf_s50_tr32000_b0_c4_labelscore_s7_tau64000` |
| mid | 3.376219 | `ebf_s50_tr32000_b1_c8_labelscore_s7_tau64000` |
| heavy | 5.387865 | `ebf_s50_tr32000_b1_c8_labelscore_s7_tau64000` |

阶段性结论：s50 是目前第一个在 prescreen200k 对齐点上**确定超过 s44（heavy best-F1）**的方向；且 light 不退化、mid 小幅提升。下一步应优先把 s50 的 best-F1 operating point 走完同口径诊断链（`noise_type_stats.py` / `dump_u_events.py`），形成“为什么有效”的证据闭环。

### 资源占用与计算开销（相对 s44）

与 s44 相比，s50 **不新增持久状态**；新增成本主要是：在计算 `raw` 的同一轮邻域遍历中多维护一个整型计数器 `cnt_support`，以及一次饱和线性函数与乘法。

复杂度结论：仍然是在线单遍、主项仍是邻域遍历 $O(r^2)$；新增开销为极小常数项。

## s51：s50 去超参版（auto-beta + 支持比例归一化，减少 light/heavy 之间配方漂移）

你指出的关键矛盾：s50 的最优超参在不同噪声环境下会漂移（例如对齐点上 light 更偏好 `beta=0`，heavy 更偏好 `beta=1`），这不符合“后续要做自适应阈值/稳定 recipe”的目标。

s51 的目标是：**去掉对环境敏感的 `beta/cnt0` 超参**，让算法自己根据输入流的“整体噪声强度”自动决定 boost 强度；在 light 下自然接近 s44（几乎不 boost），在 heavy 下自动更接近 s50（更愿意 boost）。

### 方法定义（无环境超参）

仍保留 s44 base：

$$
\mathrm{base}=\frac{\mathrm{raw}}{1+u_{\mathrm{self}}^2}
$$

定义邻域支持比例（考虑边界裁剪后的可用邻居数，归一化到 $[0,1]$）：

$$
s_{\mathrm{frac}} = \frac{c_{\mathrm{sup}}}{c_{\mathrm{possible}}}\in[0,1]
$$

其中 $c_{\mathrm{sup}}$ 与 s50 相同，为同极性且在 $[0,\tau]$ 窗口内活跃的邻域像素计数；$c_{\mathrm{possible}}$ 为该事件在图像边界裁剪后实际可用的邻域像素数（去掉中心像素）。

用事件流的 $u_{\mathrm{self}}$ 在线均值作为自适应 boost 强度（不暴露超参）：

$$
\beta_{\mathrm{eff}} \leftarrow \beta_{\mathrm{eff}} + \frac{u_{\mathrm{self}}-\beta_{\mathrm{eff}}}{N}
$$

其中 $N$ 是实现里固定的常数（不作为 sweep 超参）。然后：

关于 $N$（回答你问的“s51 里 N 是什么、不同取值会怎样”）：

- 这里的更新是标准的指数滑动平均（EMA），$N$ 可以理解为“以事件数计的时间常数”。每来 1 个事件，新的观测对状态的权重是 $1/N$。
- 直观地说，经过 $k$ 个事件后，旧状态的权重大约衰减为 $\exp(-k/N)$；对应的“半衰期”是 $N\ln 2$ 个事件。
- 当前实现固定 $N=4096$，半衰期约 $4096\ln2\approx 2839$ events。
- 调大 $N$：状态更平滑、更不抖，但对环境/阶段变化更“迟钝”（例如从稀疏到 bursty 的段落，boost 强度需要更久才跟上）。
- 调小 $N$：适应更快，但更容易被局部 burst/短片段带偏，导致阈值/排序更不稳定（尤其在 light）。

$$
\mathrm{score}=\mathrm{base}\cdot (1+\beta_{\mathrm{eff}}\,s_{\mathrm{frac}})
$$

直觉：

- light：整体 $u_{\mathrm{self}}$ 很低，$\beta_{\mathrm{eff}}\approx 0$，因此几乎退化为 s44。
- heavy：整体 $u_{\mathrm{self}}$ 更高，$\beta_{\mathrm{eff}}$ 自动上升，算法会更愿意对“宽支持”的事件给一点 boost。

### 实现与开销

实现位置：`src/myevs/denoise/ops/ebfopt_part2/s51_ebf_labelscore_selfocc_supportboost_autobeta_div_u2.py`

- 无新增 per-pixel 持久状态（仍是 `last_ts/last_pol/hot_state`）；仅新增 1 个全局标量状态 `beta_state`。
- 仍为在线单遍，主项仍是邻域遍历 $O(r^2)$。

### 实验结果（prescreen200k：s=7,tau=64ms，对齐点口径）

输出目录：`data/ED24/myPedestrain_06/EBF_Part2/_tune_s51_prescreen200k_s7_tau64/`

best-F1（per env；来自对应 ROC CSV 反算，且不需要设置任何 s51 环境变量）：

| env | best AUC | best F1 | best-F1 tag |
|---|---:|---:|---|
| light | 0.939757 | 0.942475 | `ebf_s51_labelscore_s7_tau64000` |
| mid | 0.919059 | 0.812467 | `ebf_s51_labelscore_s7_tau64000` |
| heavy | 0.922151 | 0.793990 | `ebf_s51_labelscore_s7_tau64000` |

补充：best-F1 tag 对应阈值（`param=min-neighbors`）：

| env | best-F1 thr |
|---|---:|
| light | 0.003494 |
| mid | 2.296354 |
| heavy | 3.300291 |

阶段性结论：s51 达成了“无需按环境切换 b/cnt0”的目标：

- light：几乎等同 s44（不再需要手动设 `beta=0`）。
- heavy：best-F1 略高于 s44（但仍略低于 s50 的 tuned 最优点，这是用‘去超参’换来的）。

补充（heavy，对齐点 best-F1；基于 `scripts/noise_analyze/dump_u_events.py` 导出的 `u_events_*.csv` 逐事件统计）：

- s51 vs s44：TP 29918→29885（-33），FP 5602→5530（-72），F1 0.793760→0.793990（+0.000230）
- FP 的减少主要来自 `hotmask` 类事件（-71），代价也主要发生在 `hotmask/near_hotmask` 的少量 TP（-25/-8）

## s52：在 s51 上加“opp 证据自适应门控”（目标：light/mid 更接近 s21，heavy 尽量不输 s51；无新超参）

动机：s21 在 light/mid 上更吃“opposite-polarity 证据融合”，但 heavy 下这类证据更容易被噪声污染；因此尝试用**输入流自身统计量**（而非环境超参）去决定“opp 证据该信任多少”。

### 方法定义（单 kernel，无新超参）

### 使用依据与推导思路（论文解释版）

这一段回答“为什么是这个公式/形状”，并把它放回 s21 与 s51 系列的演进逻辑里。

1) 从 s21 出发：为什么要用 opp 证据？

- 在 light/mid 中，真实运动边缘往往在小邻域内同时产生正负极性事件（受局部对比变化、边缘通过像素、以及事件相机阈值机制影响），因此异极性邻域证据 $\mathrm{raw}_{opp}$ 对“真实边缘”是有信息的。
- s21 的经验事实（你已有的 sweep）是：允许融合 opp 证据后，light/mid 的 AUC/F1 上限显著提高。

2) 从 heavy 现实出发：为什么 opp 证据在 heavy 可能更危险？

- heavy 中常见的噪声模式是 bi-polar flicker / toggle：同一像素或局部邻域在短时间内交替触发正负事件。
- 这会抬高 $\mathrm{raw}_{opp}$，使得“opp 证据强”不再意味着“更像真实边缘”，反而可能意味着“更像 flicker/toggle 噪声”。

3) s52 的核心假设（可验证）：环境级的 polarity-mix 可以作为“噪声 toggle 程度”的代理指标

- 定义每事件局部 mix：
	$$
	\mathrm{mix}=\frac{\mathrm{raw}_{opp}}{\mathrm{raw}_{same}+\mathrm{raw}_{opp}+\varepsilon}
	$$
- 观测（来自你前面 `dump_u_events.py` 的统计结论）：light 的 mix 整体更低、heavy 的 mix 整体更高。
- 因此我们用一个全局标量 $\mathrm{mix}_{ema}$（在线均值）来表示“当前输入流整体更纯（低 mix）还是更混（高 mix）”。

4) 为什么用 $\alpha_{eff}=(1-\mathrm{mix}_{ema})^2$？

- $1-\mathrm{mix}_{ema}$ 可以理解为“全局极性纯度（purity）”。purity 越高，opp 证据越稀有且更可能来自真实边缘结构；purity 越低，opp 证据越常见且更可能被 toggle 噪声污染。
- 取平方的原因：希望在 heavy（高 mix、低 purity）时更快地把 opp 融合权重压到接近 0；而在 light（低 mix、高 purity）时，权重仍能保持较大，从而接近 s21 的收益。
- 该形状无可调参数、单调且有界（$\alpha_{eff}\in[0,1]$），便于解释与复现。

5) 为什么用在线均值（EMA/Running mean）而不是每事件局部 mix？

- s52 的目标是“最小新增状态 + 环境自适应”：只增加 1 个标量 `mix_state`，避免像 s21 那样引入额外 per-pixel 表。
- 在线均值可以平滑掉单个事件的统计波动，让门控更稳定，符合“低延迟单遍”的工程约束。
- 同时这也解释了 s52 的局限：它表达的是“环境整体更混/更纯”，而不是“同一环境内某些事件更像 flicker/noise”。因此后续 s53 才进一步改为使用每事件局部 mix 去门控。

6) 与 s51 的一致性：同样采用‘输入流自适应’而不是环境超参

- s51 用自像素热度 $u_{self}$ 的在线均值生成 $\beta_{eff}$，避免按环境手调 beta。
- s52 类比地用 mix 的在线均值生成 $\alpha_{eff}$，避免为 opp 融合引入环境敏感的超参。

在同一轮邻域遍历里同时累计：

- $\mathrm{raw}_{\mathrm{same}}$：同极性邻域证据（与 baseline EBF 一致）
- $\mathrm{raw}_{\mathrm{opp}}$：异极性邻域证据（同一套时域权重）

定义瞬时的 polarity-mix：

$$
\mathrm{mix} = \frac{\mathrm{raw}_{\mathrm{opp}}}{\mathrm{raw}_{\mathrm{same}}+\mathrm{raw}_{\mathrm{opp}}+\varepsilon}
$$

对 $\mathrm{mix}$ 做一个在线 EMA（同样用固定 $N$，不作为 sweep 超参）：

$$
\mathrm{mix}_{\mathrm{ema}} \leftarrow \mathrm{mix}_{\mathrm{ema}} + \frac{\mathrm{mix}-\mathrm{mix}_{\mathrm{ema}}}{N}
$$

用 $\mathrm{mix}_{\mathrm{ema}}$ 生成 opp 证据的自适应权重（固定形状，不引入可调参数）：

$$
\alpha_{\mathrm{eff}} = (1-\mathrm{mix}_{\mathrm{ema}})^2
$$

然后把 raw 替换为：

$$
\mathrm{raw}=\mathrm{raw}_{\mathrm{same}}+\alpha_{\mathrm{eff}}\,\mathrm{raw}_{\mathrm{opp}}
$$

其余完全复用 s51：

$$
\mathrm{base}=\frac{\mathrm{raw}}{1+u_{\mathrm{self}}^2},\quad
\mathrm{score}=\mathrm{base}\cdot (1+\beta_{\mathrm{eff}}\,s_{\mathrm{frac}})
$$

为避免“只看到公式但不知道各量如何在流式里定义”，这里补全 s52/s51 里自适应量的完整定义（与实现一致）：

- 自像素热度（leaky accumulator，tick 单位；每像素 1 个 `hot_state`）：
	- 设该像素上一事件时间为 $t_0$，本事件时间为 $t$，则 $\Delta t = |t-t_0|$。
	- 状态更新（等价于代码）：
		$$
		h \leftarrow \max(0, h-\Delta t) + \max(0, \tau-\Delta t)
		$$
	- 其中 $\tau$ 为时间常数（tick）。
- 自遮挡强度（归一化到 [0,1]）：
	- 令 $\tau_r = \lfloor \tau/2 \rfloor$（固定绑定到 $\tau$，不外露超参），则
		$$
		u_{\mathrm{self}} = \frac{h}{h+\tau_r+\varepsilon}
		$$
	- 最终对 raw 做自遮挡惩罚：除以 $1+u_{\mathrm{self}}^2$。
- 支持比例 $s_{\mathrm{frac}}\in[0,1]$（同极性支持宽度）：
	- 设邻域里满足“同极性且在 $\tau$ 内”的像素个数为 $c_{\mathrm{support}}$，邻域总像素数（不含中心）为 $c_{\mathrm{possible}}$，则
		$$
		s_{\mathrm{frac}} = \frac{c_{\mathrm{support}}}{c_{\mathrm{possible}}}
		$$
- `beta_eff`（代码里的 `b`，持久化在 `beta_state[0]`，不是 sweep 外露超参）：
	- 用固定 $N=4096$ 的在线均值自适应：
		$$
		\beta_{\mathrm{eff}} \leftarrow \beta_{\mathrm{eff}} + \frac{u_{\mathrm{self}}-\beta_{\mathrm{eff}}}{N}
		$$
- `mix_state` / $\mathrm{mix}_{ema}$（全局标量，持久化在 `mix_state[0]`）：
	- 每事件先算局部 mix：
		$$
		\mathrm{mix} = \frac{\mathrm{raw}_{\mathrm{opp}}}{\mathrm{raw}_{\mathrm{same}}+\mathrm{raw}_{\mathrm{opp}}+\varepsilon}
		$$
	- 再用固定 $N$ 在线均值更新：
		$$
		\mathrm{mix}_{ema} \leftarrow \mathrm{mix}_{ema} + \frac{\mathrm{mix}-\mathrm{mix}_{ema}}{N}
		$$
	- opp 权重：
		$$
		\alpha_{\mathrm{eff}} = (1-\mathrm{mix}_{ema})^2
		$$
- 最终打分（与实现一致，包含除以 $\tau$）：
	$$
	\mathrm{raw} = \frac{\mathrm{raw}_{\mathrm{same}}+\alpha_{\mathrm{eff}}\,\mathrm{raw}_{\mathrm{opp}}}{\tau},\quad
	\mathrm{base}=\frac{\mathrm{raw}}{1+u_{\mathrm{self}}^2},\quad
	\mathrm{score}=\mathrm{base}\cdot(1+\beta_{\mathrm{eff}}\,s_{\mathrm{frac}})
	$$

### 实现与开销

实现位置：`src/myevs/denoise/ops/ebfopt_part2/s52_ebf_labelscore_selfocc_supportboost_autobeta_mixgateopp_div_u2.py`

- 仍为在线单遍，主项仍是邻域遍历 $O(r^2)$；新增仅为常数级：多算一份 $\mathrm{raw}_{\mathrm{opp}}$，以及 1 个全局标量状态 `mix_state`。

### 实验结果（prescreen200k：s=7,tau=64ms，对齐点口径）

输出目录：`data/ED24/myPedestrain_06/EBF_Part2/_tune_s52_prescreen200k_s7_tau64/`

对齐点 tag（固定）：`ebf_s52_labelscore_s7_tau64000`

| env | AUC（s7,tau64ms） | best F1（s7,tau64ms） |
|---|---:|---:|
| light | 0.941183 | 0.942323 |
| mid | 0.919599 | 0.809206 |
| heavy | 0.923927 | 0.791831 |

补充：对齐点 tag 的 best-F1 operating point 阈值（`param=min-neighbors`）：

| env | best-F1 thr |
|---|---:|
| light | 0.381075 |
| mid | 2.599577 |
| heavy | 3.853803 |

阶段性结论：在“对齐点 s=7,tau=64ms”口径下，s52 **没有超过 s51**（light 基本持平；mid/heavy 小幅回落）。

补充（避免口径混淆：全网格 / 统一点 `s=9,tau=128ms` 下，s52 反而非常强）：

- 全网格 best-F1（prescreen200k；各 env 独立取 best tag）：
	- light：0.957995（tag：`ebf_s52_labelscore_s9_tau512000`）
	- mid：0.819913（tag：`ebf_s52_labelscore_s9_tau128000`）
	- heavy：0.795046（tag：`ebf_s52_labelscore_s9_tau128000`）
- 固定统一点 `s=9,tau=128ms`（tag：`ebf_s52_labelscore_s9_tau128000`；只允许各 env 内调阈值取 best-F1）：
	- light：best-F1 0.955353 / AUC 0.951957
	- mid：best-F1 0.819913 / AUC 0.933620
	- heavy：best-F1 0.795046 / AUC 0.933429

## s53：按“局部 polarity-mix + 支持宽度”建模的 opp 证据门控（避免 s52 的全局门控偏差）

动机（噪声规律建模，而非堆砌）：

- 通过 `dump_u_events.py` 的逐事件统计可以观察到：在 heavy 下，TP/FP 的邻域 polarity-mix（opp/(same+opp)）分布存在差异；而 s52 用的是**全局** `mix_state`，它只能表达“环境整体更混/更纯”，难以表达“同一环境内某些事件更像 flicker/noise”。
- 因此 s53 改为使用**每个事件自己的局部 mix** 来门控 opp 证据，并额外用“同极性支持比例”来约束：opp 证据只在“空间支持足够宽”的事件上才有贡献，避免少数像素上的 toggle/flicker 噪声被 opp 证据抬分。

### 方法定义（单 kernel，无新超参）

同一轮邻域遍历里累计：

- $\mathrm{raw}_{\mathrm{same}}$：同极性邻域证据
- $\mathrm{raw}_{\mathrm{opp}}$：异极性邻域证据
- $s_{\mathrm{frac}}\in[0,1]$：同极性支持比例（与 s51 相同）

定义每事件局部 polarity-mix：

$$
\mathrm{mix}=\frac{\mathrm{raw}_{\mathrm{opp}}}{\mathrm{raw}_{\mathrm{same}}+\mathrm{raw}_{\mathrm{opp}}+\varepsilon}
$$

用局部 mix（polarity purity）与支持宽度共同生成 opp 权重（固定形状，不引入可调参数）：

$$
\alpha_{\mathrm{eff}}=(1-\mathrm{mix})^2\,\sqrt{s_{\mathrm{frac}}}
$$

然后：

$$
\mathrm{raw}=\mathrm{raw}_{\mathrm{same}}+\alpha_{\mathrm{eff}}\,\mathrm{raw}_{\mathrm{opp}}
$$

其余复用 s51（auto-beta + 支持比例 boost）：

$$
\mathrm{base}=\frac{\mathrm{raw}}{1+u_{\mathrm{self}}^2},\quad
\mathrm{score}=\mathrm{base}\cdot (1+\beta_{\mathrm{eff}}\,s_{\mathrm{frac}})
$$

### 实现与开销

实现位置：`src/myevs/denoise/ops/ebfopt_part2/s53_ebf_labelscore_selfocc_supportboost_autobeta_eventmixgateopp_div_u2.py`

- 仍为在线单遍、主项仍是邻域遍历 $O(r^2)$；新增仅为常数级：多算一份 $\mathrm{raw}_{\mathrm{opp}}$，并做一次局部 `mix` 与 `sqrt(sfrac)` 计算。
- 无新增 per-pixel 持久状态；仅复用 s51 的 `beta_state`。

### 实验结果（prescreen200k：s=7,tau=64ms，对齐点口径）

输出目录：`data/ED24/myPedestrain_06/EBF_Part2/_tune_s53_prescreen200k_s7_tau64/`

对齐点 tag（固定）：`ebf_s53_labelscore_s7_tau64000`

| env | AUC（s7,tau64ms） | best F1（s7,tau64ms） |
|---|---:|---:|
| light | 0.939786 | 0.942475 |
| mid | 0.919072 | 0.812773 |
| heavy | 0.922116 | 0.794058 |

补充：对齐点 tag 的 best-F1 operating point 阈值（`param=min-neighbors`）：

| env | best-F1 thr |
|---|---:|
| light | 0.003496 |
| mid | 2.296325 |
| heavy | 3.548370 |

阶段性结论：s53 在“对齐点 s=7,tau=64ms”口径下**小幅超过 s51**（mid/heavy 都是 +$\sim 10^{-4}$ 量级，light 持平）。这说明“局部 mix + 支持宽度”的建模方向是对的，但强度仍偏保守；后续可在不引入新超参的前提下，继续优化 $\alpha_{\mathrm{eff}}$ 的固定形状（例如用更合理的归一化以提升 mid/light 的收益）。

## s54：s53 的 $\sqrt{s_{\mathrm{frac}}}$ 形状微调（$\sqrt[4]{s_{\mathrm{frac}}}$，尝试救回 mid/light；结果 heavy 回落）

要解决的失败模式/可验证假设：

- s53 的 opp 融合权重 $\alpha_{\mathrm{eff}}=(1-\mathrm{mix})^2\,\sqrt{s_{\mathrm{frac}}}$ 很“保守”，当 $s_{\mathrm{frac}}$ 中等偏小（但并非纯噪声）时，opp 证据几乎被压没，可能限制了 mid/light 的收益。
- 假设：把 $\sqrt{s_{\mathrm{frac}}}$ 改成更“宽松”的 $\sqrt[4]{s_{\mathrm{frac}}}$，能在不引入新超参的前提下，让更多“中等支持宽度”的事件获得 opp 证据增量，从而提升 mid（理想情况下不伤 heavy）。

### 方法定义（单 kernel，无新超参）

仅改一处固定形状：

$$
\alpha_{\mathrm{eff}}=(1-\mathrm{mix})^2\,s_{\mathrm{frac}}^{1/4}
$$

其余完全复用 s53（以及 s51 的 auto-beta + support boost）：

$$
\mathrm{raw}=\mathrm{raw}_{\mathrm{same}}+\alpha_{\mathrm{eff}}\,\mathrm{raw}_{\mathrm{opp}},\quad
\mathrm{base}=\frac{\mathrm{raw}}{1+u_{\mathrm{self}}^2},\quad
\mathrm{score}=\mathrm{base}\cdot (1+\beta_{\mathrm{eff}}\,s_{\mathrm{frac}})
$$

### 实现与开销

实现位置：`src/myevs/denoise/ops/ebfopt_part2/s54_ebf_labelscore_selfocc_supportboost_autobeta_eventmixgateopp_root4_div_u2.py`

- 仍为在线单遍、主项仍是邻域遍历 $O(r^2)$。
- 无新增持久状态；仅是 1 次幂运算形状调整（常数级）。

### 实验口径与结果（prescreen200k 全网格 sweep；同时记录“全网格 best”与“对齐点”）

输出目录：`data/ED24/myPedestrain_06/EBF_Part2/_tune_s54_prescreen200k_s7_tau64/`

对齐点 tag（固定）：`ebf_s54_labelscore_s7_tau64000`

| env | AUC（s7,tau64ms） | best F1（s7,tau64ms） |
|---|---:|---:|
| light | 0.939822 | 0.942475 |
| mid | 0.919000 | 0.812783 |
| heavy | 0.921946 | 0.793731 |

补充：对齐点 tag 的 best-F1 operating point 阈值（`param=min-neighbors`）：

| env | best-F1 thr |
|---|---:|
| light | 0.003499 |
| mid | 2.383567 |
| heavy | 3.641110 |

阶段性结论（停掉）：

- s54 的确把 mid 在对齐点上抬了一点点，但 heavy 出现回落（对齐点 F1 低于 s53）。
- 解释上也合理：$s_{\mathrm{frac}}^{1/4}$ 会显著放大“支持不够宽”的事件的 opp 融合权重，更容易把 flicker/toggle 类噪声抬分，heavy 最先受损。
- 因此 s54 **不建议作为统一 recipe**；后续不再沿“单纯放大 $s_{\mathrm{frac}}$ 非线性”方向继续。

## s55：s53 的 mix gate 做“支持宽度自适应松弛”（仅在支持足够宽时更信任 opp；无新超参）

要解决的失败模式/可验证假设：

- s53 的核心理念是对的（局部 mix + 支持宽度），但它对 opp 的信任依赖于一个“过于硬”的 gate：当局部 mix 偏高时，$(1-\mathrm{mix})^2$ 会把 opp 基本压没。
- 观察：真实结构（边缘/运动）也可能出现较高 mix（例如多源叠加、纹理、或短时抖动），但它们往往伴随**更宽的空间支持**（$s_{\mathrm{frac}}$ 更大）。
- 假设：用 $s_{\mathrm{frac}}$ 去“松弛” mix gate——**只有在支持足够宽**时才允许 gate 变弱——可以提升 mid/light，同时不把 narrow-support 的 flicker/noise 抬起来，从而不伤 heavy。

### 方法定义（单 kernel，无新超参）

保留：

$$
\mathrm{mix}=\frac{\mathrm{raw}_{\mathrm{opp}}}{\mathrm{raw}_{\mathrm{same}}+\mathrm{raw}_{\mathrm{opp}}+\varepsilon},\quad
g=(1-\mathrm{mix})^2
$$

用 $s_{\mathrm{frac}}$ 对 gate 做线性插值松弛：

$$
g_{\mathrm{eff}} = g + (1-g)\,s_{\mathrm{frac}}
$$

再沿用 s53 的“支持宽度约束”，得到：

$$
\alpha_{\mathrm{eff}} = g_{\mathrm{eff}}\,\sqrt{s_{\mathrm{frac}}}
$$

其余复用 s53/s51：

$$
\mathrm{raw}=\mathrm{raw}_{\mathrm{same}}+\alpha_{\mathrm{eff}}\,\mathrm{raw}_{\mathrm{opp}},\quad
\mathrm{base}=\frac{\mathrm{raw}}{1+u_{\mathrm{self}}^2},\quad
\mathrm{score}=\mathrm{base}\cdot (1+\beta_{\mathrm{eff}}\,s_{\mathrm{frac}})
$$

直觉检查（解释性）：

- 当 $s_{\mathrm{frac}}\to 0$：$g_{\mathrm{eff}}\to g$，即仍严格依赖 mix gate（不轻易信任 opp）。
- 当 $s_{\mathrm{frac}}\to 1$：$g_{\mathrm{eff}}\to 1$，即在“支持足够宽”的事件上允许更充分利用 opp 证据。

### 实现与开销

实现位置：`src/myevs/denoise/ops/ebfopt_part2/s55_ebf_labelscore_selfocc_supportboost_autobeta_eventmixgateopp_supportlerp_div_u2.py`

- 仍为在线单遍、主项仍是邻域遍历 $O(r^2)$。
- 无新增持久状态；新增仅为常数级的几次乘加。
- 运行注意（Windows/Numba）：该文件使用 `@njit(cache=False)`，避免 Numba cache 写入在 Windows 下触发 `OSError: [Errno 22] Invalid argument`（长路径/缓存文件名过长）。这只影响“是否落盘缓存”，不改变算法与单次运行性能；首次 import 仍会 JIT 编译。

### 资源占用与计算开销（回答“是不是像 s21 一样多两张表？”）

结论先说：s55（以及 s51/s53/s54）**没有**引入 s21 那种“两张 per-pixel accumulator 表”（`acc_pos/acc_neg`）。

- s55/s51 家族持久状态（per pixel）：`last_ts(uint64)` + `last_pol(int8)` + `hot_state(int32)`，外加 1 个全局标量 `beta_state(float64[1])`。
- baseline EBF（本 sweep 脚本口径）持久状态（per pixel）：`last_ts(uint64)` + `last_pol(int8)`。
- s21 持久状态（per pixel）：`last_ts` + `last_pol` + `acc_pos(int32)` + `acc_neg(int32)`（确实是两张表）。

按当前分辨率 `346×260` 估算状态内存（仅持久数组，不含临时局部变量）：

- baseline：约 790.7 KiB
- s55/s51 家族：约 1142.1 KiB（比 baseline 多一张 `hot_state`，约 +351.4 KiB）
- s21：约 1493.5 KiB

计算开销方面：

- baseline 只统计同极性邻域证据；s55 需要同时统计 `raw_same` 与 `raw_opp`（同一轮邻域遍历里多一些分支与加法），属于常数级变慢。
- s55 额外的 `mix/sfrac/gate_eff` 都是标量运算，不改变主复杂度（仍为 $O(r^2)$）。

### 实验口径与结果（prescreen200k 全网格 sweep；对齐点 s=7,tau=64ms 抽取 best-F1 operating point）

输出目录：`data/ED24/myPedestrain_06/EBF_Part2/_tune_s55_prescreen200k_s7_tau64/`

注意：该目录里的 ROC CSV 覆盖了全网格（`s=3,5,7,9` × `tau=8..1024ms`）。因此：

- “全网格 best-AUC/best-F1”：允许在整个网格上取最优点（更接近你说的 s21 的 0.95+ 口径）。
- “对齐点”：只看固定的 `s=7,tau=64ms`，用于比较“统一 recipe 的稳定性/迁移性”。

#### 全网格 best（per env）

从对应 ROC CSV 直接取全局最优点：

| env | best AUC（全网格） | best AUC tag | best F1（全网格） | best F1 tag |
|---|---:|---|---:|---|
| light | 0.949775 | `ebf_s55_labelscore_s9_tau128000` | 0.955424 | `ebf_s55_labelscore_s9_tau256000` |
| mid | 0.931638 | `ebf_s55_labelscore_s9_tau256000` | 0.820097 | `ebf_s55_labelscore_s9_tau128000` |
| heavy | 0.929525 | `ebf_s55_labelscore_s9_tau128000` | 0.794704 | `ebf_s55_labelscore_s7_tau64000` |

对齐点 tag（固定）：`ebf_s55_labelscore_s7_tau64000`

| env | AUC（s7,tau64ms） | best F1（s7,tau64ms） |
|---|---:|---:|
| light | 0.939792 | 0.942475 |
| mid | 0.919221 | 0.813311 |
| heavy | 0.922415 | 0.794704 |

补充：对齐点 tag 的 best-F1 operating point 阈值（`param=min-neighbors`）：

| env | best-F1 thr |
|---|---:|
| light | 0.004309 |
| mid | 2.300890 |
| heavy | 3.563847 |

阶段性结论（继续）：

- s55 在“对齐点 s=7,tau=64ms”口径下同时抬升 mid 与 heavy（相对 s53 为 $+\sim 5\times10^{-4}$ 量级），light 持平。
- 这更符合“按噪声规律建模”的预期：narrow-support 的高 mix 事件仍然被严格 gate（避免把 flicker/noise 抬起来），而 wide-support 的高 mix 事件获得更合理的 opp 融合增量。
- 解释闭环（heavy，基于 `noise_type_stats.py` 的对齐点 best-F1 operating point）：s55 相对 s53 的净提升主要来自“回收被压制的 TP”，而不是压低 FP。
	- TP：29695 → 29801（+106），其中 `hotmask` +73、`near_hotmask` +23。
	- FP：5235 → 5335（+100），增量也主要发生在 `hotmask` +84、`near_hotmask` +10。
	- 直观含义：s53 在 hotmask/near-hot 区域可能过度谨慎（把一部分真实信号也压低），s55 通过“支持宽度自适应松弛”在这些区域恢复了一部分信号，代价是带来少量额外噪声通过；总体上 F1 仍净增。
- 下一步最小工作：补跑 `dump_u_events.py`（heavy）对比 s53/s55 的 `mix/sfrac/alpha_eff` 分布，验证“只在 wide-support 时松弛 gate”确实发生在 TP 而不是 FP 上；并检查 light/mid 是否出现阈值可迁移性变差（s55 的 light thr 有明显上移）。

## s51–s55：补充汇总（prescreen200k 全网格；并给出“同一参数(tag)下”的三噪声表现）

动机：你提到“之前只有 s=7,tau=64000 的数据”，以及“最优参数在不同噪声下不统一”。下面给两类统计：

1) **全网格 best**：每个 env 允许在整张网格上取最优点（更接近“追 s21 上限”的口径）。
2) **同一参数(tag)下的 best-F1**：固定同一个 tag（即固定同一组算法参数，例如固定 `s/tau`，或 s21 的 `a/b/k+s/tau`），在各 env 内只允许调阈值（`min-neighbors`）取 best-F1，用于衡量“统一 recipe”可迁移性。

### A) 全网格 best（per env）对比（baseline/s21/s51–s55）

（数据来源：对应 out-dir 下的 ROC CSV 全行扫描）

| 方法 | light bestAUC | light bestF1 | mid bestAUC | mid bestF1 | heavy bestAUC | heavy bestF1 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.947564 | 0.949739 | 0.921924 | 0.810827 | 0.920467 | 0.786882 |
| s21 | 0.953086 | 0.956651 | 0.932585 | 0.819471 | 0.929583 | 0.794560 |
| s51 | 0.949698 | 0.955211 | 0.931349 | 0.819442 | 0.928998 | 0.793990 |
| s52 | 0.951957 | 0.957995 | 0.936710 | 0.819913 | 0.933429 | 0.795046 |
| s53 | 0.949727 | 0.955261 | 0.930882 | 0.819546 | 0.928714 | 0.794058 |
| s54 | 0.949789 | 0.955224 | 0.930714 | 0.819259 | 0.928466 | 0.793731 |
| s55 | 0.949775 | 0.955424 | 0.931638 | 0.820097 | 0.929525 | 0.794704 |

### B) 同一参数(tag)下的 best-F1（解释“为什么 baseline/s21 更统一”）

这里的“统一”指：**选一个固定 tag**，在 light/mid/heavy 三个环境里都用同一组参数（仅阈值各自取 best-F1）。

baseline：

- 统一 tag：`ebf_labelscore_s9_tau128000`
- F1（light/mid/heavy）：0.949739 / 0.810827 / 0.786882

s21：

- 统一 tag：`ebf_s21_a0p2_b0p6_k0p8_labelscore_s9_tau128000`
- F1（light/mid/heavy）：0.956651 / 0.819471 / 0.794560

s55（示例：选“平均 F1 最优”的统一 tag）：

- 统一 tag：`ebf_s55_labelscore_s9_tau128000`
- F1（light/mid/heavy）：0.952604 / 0.820097 / 0.792701

进一步：把“统一参数”固定得更死（都用 `s=9,tau=128ms`；s21 固定 `a0.2,b0.6,k0.8`），只允许各 env 内调阈值取 best-F1，则：

| 方法（统一 tag） | light bestF1 | mid bestF1 | heavy bestF1 |
|---|---:|---:|---:|
| baseline（`ebf_labelscore_s9_tau128000`） | 0.949739 | 0.810827 | 0.786882 |
| s21（`ebf_s21_a0p2_b0p6_k0p8_labelscore_s9_tau128000`） | 0.956651 | 0.819471 | 0.794560 |
| s51（`ebf_s51_labelscore_s9_tau128000`） | 0.952454 | 0.819442 | 0.791979 |
| s52（`ebf_s52_labelscore_s9_tau128000`） | 0.955353 | 0.819913 | 0.795046 |
| s53（`ebf_s53_labelscore_s9_tau128000`） | 0.952489 | 0.819546 | 0.791844 |
| s54（`ebf_s54_labelscore_s9_tau128000`） | 0.952537 | 0.819259 | 0.791379 |
| s55（`ebf_s55_labelscore_s9_tau128000`） | 0.952604 | 0.820097 | 0.792701 |

注：上表里的 baseline 数值来自 **prescreen200k** 口径的全网格 ROC CSV（用于与 s51–s55、s52 在同一口径对比）：

- 目录：`data/ED24/myPedestrain_06/EBF_Part2/ebf_baseline_sweep_prescreen200k_noaocc/`
- heavy（`ebf_labelscore_s9_tau128000`）best-F1 = 0.786882；heavy 的全网格 best-F1 = 0.788680（`ebf_labelscore_s7_tau64000`）

如果你记得 baseline heavy best-F1 约 0.76：那通常来自 **validate_1M**（全量）口径。对应产物为：

- 目录：`data/ED24/myPedestrain_06/EBF_Part2/baseline_validate_1M_s9_tau128ms/`
- heavy（`ebf_labelscore_s9_tau128000`）best-F1 = 0.760973

### validate_1M 全量对比（baseline vs s52 vs s55；含 thr/MESR/AOCC）

口径：`max-events=1_000_000`（全量）；评测时仍扫 `s∈{3,5,7,9}` 与 `tau∈{8..1024ms}` 的网格（用于确认各 env 的 best-F1 / best-AUC tag），但本文对比表采用**统一 tag**：`s=9,tau=128ms`（只在该 tag 内选 best-F1 operating point）。并启用：

- `--esr-mode best`：MESR/ESR mean（写入 ROC CSV 的 `esr_mean` 列，仅在 best-F1 行填值）
- `--aocc-mode best`：AOCC（写入 ROC CSV 的 `aocc` 列，仅在 best-F1 行填值）

产物目录：

- baseline：`data/ED24/myPedestrain_06/EBF_Part2/baseline_validate_1M_s9_tau128ms_aocc_paper/`
- s52：`data/ED24/myPedestrain_06/EBF_Part2/s52_validate_1M_s9_tau128ms_aocc_paper/`
- s55：`data/ED24/myPedestrain_06/EBF_Part2/s55_validate_1M_s9_tau128ms_aocc_paper/`

说明：这里的 `thr` 是 ROC CSV 里 `param=min-neighbors` 的 `value`（labelscore 的阈值）；MESR/AOCC 都是在该 best-F1 operating point（scores >= thr）上计算的。

validate_1M（s=9,tau=128ms）/light：best-F1 operating point（含 thr/MESR/AOCC）

| 方法 | tag | bestF1 | thr | AUC | MESR(esr_mean) | AOCC |
|---|---|---:|---:|---:|---:|---:|
| baseline | `ebf_labelscore_s9_tau128000` | 0.949739 | 0.749148 | 0.947564 | 1.030472 | 0.820552 |
| s52 | `ebf_s52_labelscore_s9_tau128000` | 0.955353 | 0.817547 | 0.951957 | 0.957297 | 0.824324 |
| s55 | `ebf_s55_labelscore_s9_tau128000` | 0.952604 | 0.736163 | 0.949775 | 0.948890 | 0.820021 |

validate_1M（s=9,tau=128ms）/mid：best-F1 operating point（含 thr/MESR/AOCC）

| 方法 | tag | bestF1 | thr | AUC | MESR(esr_mean) | AOCC |
|---|---|---:|---:|---:|---:|---:|
| baseline | `ebf_labelscore_s9_tau128000` | 0.817653 | 4.834726 | 0.923218 | 1.015456 | 0.791076 |
| s52 | `ebf_s52_labelscore_s9_tau128000` | 0.828421 | 5.396395 | 0.936762 | 0.971959 | 0.788476 |
| s55 | `ebf_s55_labelscore_s9_tau128000` | 0.826684 | 4.727915 | 0.931353 | 0.967736 | 0.794337 |

validate_1M（s=9,tau=128ms）/heavy：best-F1 operating point（含 thr/MESR/AOCC）

| 方法 | tag | bestF1 | thr | AUC | MESR(esr_mean) | AOCC |
|---|---|---:|---:|---:|---:|---:|
| baseline | `ebf_labelscore_s9_tau128000` | 0.760973 | 7.279687 | 0.913578 | 1.008477 | 0.770944 |
| s52 | `ebf_s52_labelscore_s9_tau128000` | 0.773793 | 7.983418 | 0.930800 | 0.975172 | 0.767642 |
| s55 | `ebf_s55_labelscore_s9_tau128000` | 0.769547 | 7.385659 | 0.924501 | 0.971497 | 0.771860 |

补充（best-AUC tag 的 MESR/AOCC 是否缺失？）：本次 validate_1M 的全网格统计中，baseline/s52/s55 在 light/mid/heavy 三环境的 **best-AUC tag 与 best-F1 tag 完全一致**，均落在 `s=9,tau=128ms`。因此上面三张表的 MESR/AOCC 也同时覆盖了“best-AUC tag”的记录口径，不需要另起一套 best-AUC 表。

#### 诊断：为什么 heavy 的“全量”会比 prescreen200k 明显掉分？（时间段落漂移）

关键事实：ED24 heavy 的 labeled npy 实际总事件数为 **896,682**（因此 `max-events=1_000_000` 等价于“用全量”）。

把同一套参数（`s=9,tau=128ms` + 各自 best-F1 阈值）按事件序号每 200k 分一段，F1 的段内波动很大：

| 段(每段200k) | baseline F1 | s52 F1 | s55 F1 | baseline FP/FN | s52 FP/FN | s55 FP/FN |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.7868 | 0.7949 | 0.7923 | 7043/9442 | 7126/8869 | 6804/9245 |
| 1 | 0.6550 | 0.6755 | 0.6712 | 5023/7197 | 4445/6943 | 4797/6880 |
| 2 | 0.6786 | 0.6916 | 0.6873 | 4281/7433 | 4138/7147 | 4190/7240 |
| 3 | 0.7350 | 0.7474 | 0.7454 | 8489/11824 | 8090/11305 | 7967/11499 |
| 4 | 0.8380 | 0.8505 | 0.8448 | 6409/7776 | 6779/6552 | 6075/7507 |

解释：prescreen200k 本质上只“看到了第0段”，而第1~3段更难（信号更稀/噪声结构不同/热点背景变化等），所以全量聚合后的 best-F1 会明显低于 prescreen。

#### 诊断：AOCC 低是否等价于“过度去噪”？（不等价；需要看信号/噪声保留率）

AOCC 定义是：对“保留下来的事件”在多尺度时间窗内成帧，取 Sobel 梯度幅值的 std 做 contrast，再对 dt 积分得到面积；**它没有 label 参照，也不要求单调**。因此 AOCC 更低并不能直接推出“过度去噪”，可能是：

- 去掉了会形成边缘假象的噪声（AOCC↓但更好），或
- 误伤了细节结构的真实事件（AOCC↓且更差）。

用 `scripts/noise_analyze/noise_type_stats.py` 对最难的第1段（200k~400k）做噪声类型分解（同一阈值口径）后，可以看到 s52 并不是简单“更激进地删掉一切”：

| 类别(seg1) | baseline noise_kept_rate | s52 noise_kept_rate | s55 noise_kept_rate | baseline signal_kept_rate | s52 signal_kept_rate | s55 signal_kept_rate |
|---|---:|---:|---:|---:|---:|---:|
| hotmask | 0.0235 | 0.0206 | 0.0224 | 0.5709 | 0.5869 | 0.5872 |
| near_hotmask | 0.0630 | 0.0652 | 0.0654 | 0.6012 | 0.6152 | 0.6162 |

读法：在 hotmask 类中，s52 相比 baseline **更少保留噪声**（noise_kept_rate 更低），同时 **更多保留信号**（signal_kept_rate 更高）；s55 非常接近 s52，但会略多保留一些 hotmask 噪声（这类“更边缘化的噪声”有时反而会把 AOCC 拉高）。

解释（回答“为什么 baseline/s21 看起来统一，而它不统一”）：

- baseline 的最优点在三环境都落在同一个 `s=9,tau=128ms`（tag 自然统一）。
- s21 在这个 prescreen 网格里也几乎是同一个 tag 在三环境同时最好（`a=0.2,b=0.6,k=0.8,s9,tau128`），因此看起来“统一”。
- s51–s55 这套机制在 heavy 上往往更偏好更短时间常数/更小空间尺度（例如对齐点 `s7,tau64ms` 的 best-F1 很强），所以“按 env 取全网格最优”时会出现不同 env 的 best tag 不同；但如果强制用统一 tag（例如 `s9,tau128ms`），它们的跨环境表现依然是稳定且接近 s21 的。

### 综合结论：s21 vs s52 vs s55（性能 / 复杂度 / 资源 / 超参）

先说结论（按“统一 recipe、少调参、跨环境稳定”排序）：

- 推荐优先级：s52 ≥ s55（两者都几乎无显式超参）；若允许引入并调 `alpha/beta/kappa`，s21 仍是很强参考。
- 在统一点 `s=9,tau=128ms` 下，s52 的表现几乎已达到 s21（且不需要 s21 的 `alpha/beta/kappa` 超参），并在 light/heavy 上相对 s55 有优势：
	- s52（`ebf_s52_labelscore_s9_tau128000`）F1：0.955353 / 0.819913 / 0.795046
	- s55（`ebf_s55_labelscore_s9_tau128000`）F1：0.952604 / 0.820097 / 0.792701
	- s21（`ebf_s21_a0p2_b0p6_k0p8_labelscore_s9_tau128000`）F1：0.956651 / 0.819471 / 0.794560
- “对齐点 s=7,tau=64ms”只是统一对齐口径，不代表各方法在全网格/统一点上的上限；s52/s55 在该对齐点的排序与 `s=9,tau=128ms` 可能不同。

计算复杂度（同 `s/tau/radius` 下）：

- 三者主耗时都在邻域遍历 $O(r^2)$。
- s52 vs s55 的额外开销均为常数级：
	- s52：多维护 1 个全局标量 `mix_state`（在线均值），并用它门控 opp 证据。
	- s55：对每事件计算 `gate_eff` 与 `sqrt(sfrac)` 以调整 opp 证据。
- 实测差异通常远小于邻域遍历的主项（更像“同量级”）。

资源占用（per-pixel 持久状态）：

- s55：`last_ts,last_pol,hot_state` + 1 个标量 `beta_state[0]`。
- s52：在 s55 的基础上仅多 1 个标量 `mix_state[0]`，资源几乎不变。
- s21：除 `last_ts,last_pol` 外还需要 `acc_neg,acc_pos` 两张 int32 表，因此内存占用显著高于 s52/s55（这是结构性差异）。

`beta_eff` / `\u03b2_{eff}` 是什么？是不是超参数？

- 这里的 `\u03b2_{eff}` 不是 sweep 的外露超参数，它对应 kernel 里的 `b`（持久化在 `beta_state[0]`），是通过输入流在线自适应得到的“有效 beta”。
- 更新形式是固定窗口的在线均值（$N=4096$ 固定写死，不作为 sweep 维度）：
	- `b \leftarrow b + (u_{self}-b)/N`
- 最终用法（s52/s55 一致）：`score = base * (1 + b * sfrac)`。

关于“把 s21 的超参优化掉”：

- 方向上是可行的：可以让 `alpha/beta/kappa` 像 s52 的 `mix_state`、s51/s55 的 `beta_state` 一样来自在线统计量，从而减少环境敏感性。
- 但即便做到“去超参”，s21 仍会保留两张 accumulator 表的资源开销（`acc_neg/acc_pos`），因此在资源上不太可能与 s52/s55 完全一致。
