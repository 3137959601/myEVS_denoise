# myEVS 去噪算法对比实验总控 README

本 README 仅用于 `src/myevs/denoise/ops` 维度的跨算法对比实验管理，约束后续：
- 数据集路径
- 结果存储路径
- 扫频/运行脚本入口
- 指标统计与汇总表格式

当前状态（2026-04-23）：
- 已集成算法：`BAF`, `STCF(stc)`, `EBF`, `EBF_OPTIMIZED`, `KNOISE`, `EVFLOW`, `YNOISE`, `TS`, `MLPF`, `PFD`。
- `n175` 演化暂停，先做跨数据集横向对比。


## A. 总览

## 1. 统一指标与汇总表

目标指标：
- `AUC`
- `F1`
- `MESR`
- `AOCC`
- `runtime_sec`（运行时）

建议总汇 CSV（后续维护）：
- `data/summary/alg_compare_master.csv`

列定义（最小必需）：
- `dataset, scene, level, algorithm, tag, auc, f1, mesr, aocc, time_us, radius_px, threshold, csv_path, fig_path, note`

要求：
- 每次新增实验必须写入总汇 CSV；`csv_path/fig_path` 必填。
- `tag` 与 `time_us/radius_px/threshold` 必须可互相还原。
- 每次实验必须写入运行时（`runtime_sec` 或 `runtime_min`）。
- 对支持的脚本，`best-AUC` 与 `best-F1` 两个点都要给出 `MESR/AOCC`。

## 2. 约束与后续补充规则

- 所有新增结果必须进入本 README 对应章节与 `data/summary/alg_compare_master.csv`。
- 任何脚本新增后，必须在“scripts 目录功能说明”登记用途。
- 若改动数据路径或命名规则，先改本 README，再跑实验。

## 3. N179 结果并入规则（更新）

`n179` 不再单独汇总展示。  
从本版开始，`n179` 结果统一并入各数据集原有对比章节：

1. ED24 / Driving：并入 `10.*` 对比节；
2. DVSCLEAN：并入 `17.*` 与 `21.*`；
3. LED：并入 `18.*`。

全量复跑脚本保持不变：

```powershell
powershell -ExecutionPolicy Bypass -File ./scripts/ED24_alg_evalu/run_ed24_n179_full.ps1
powershell -ExecutionPolicy Bypass -File ./scripts/driving_alg_evalu/run_driving_n179_full.ps1
powershell -ExecutionPolicy Bypass -File ./scripts/DVSCLEAN_alg_evalu/run_dvsclean_n179_full.ps1
powershell -ExecutionPolicy Bypass -File ./scripts/LED_alg_evalu/run_led_n179_full.ps1
```

说明（防漏算）：`run_ed24_n179_full.ps1` 与 `run_driving_n179_full.ps1` 已内置 `MESR/AOCC` 后处理步骤，会自动更新对应 `bestpoint_mesr_aocc_summary.csv`。


## B. ED24 数据集

## 4. 新增算法接入说明

算法 method token（`myevs.cli denoise/roc/sweep --method`）：
- `knoise`（id=12）
- `evflow`（id=13）
- `ynoise`（id=14）
- `ts`（id=15）
- `mlpf`（id=16）
- `pfd`（id=17）

参数映射（统一沿用 myEVS 通用参数）：
- `time-us` -> 算法中的时间窗/衰减参数（duration/decay）
- `radius-px` -> 空间搜索半径（search radius）
- `min-neighbors` -> 阈值（intThreshold/floatThreshold）

说明：
- `mlpf` 支持两种模式：  
1. 真实模型推理模式（推荐）：传 `--mlpf-model`，加载 TorchScript `.pt`；  
2. 代理评分模式（兼容）：未传模型时回退到 MLP-inspired 特征评分。

### 3.1 对比算法原理与公式（论文写作口径）

以下公式按当前代码实现给出（不是通用文献形式）。统一记号：
- 事件 \(e_i=(x_i,y_i,t_i,p_i)\)，\(p_i\in\{+1,-1\}\)。
- 邻域 \(\mathcal{N}_r(i)\) 为 \([x_i-r,x_i+r]\times[y_i-r,y_i+r]\) 的裁剪窗口。
- 时间窗 \(\tau\) 对应 `time-us` 转换到 ticks。
- \(\mathbf{1}[\cdot]\) 为指示函数，条件成立取 1，否则取 0。
- \(\operatorname{clip}(x,a,b)=\min(\max(x,a),b)\)。
- \(p_{\text{prev}}(\text{idx}(i))\) 表示“当前像素上一次事件的极性”（来自 `last_pol`）。

`BAF`（`src/myevs/denoise/ops/baf.py`）：
- 判据是“邻域是否存在任意近期事件”（不看极性）。
$$
\text{keep}_i=\mathbf{1}\!\left[\exists j\in \mathcal{N}_r(i)\setminus\{i\},\ t_j\in[t_i-\tau,t_i]\right].
$$
- 说明：这版是布尔门限，不是计数阈值版本。

`STCF/STC`（`src/myevs/denoise/ops/stc.py`）：
- 统计同极性、近期邻居数（不含中心点）：
$$
C_i^{(p)}=\sum_{j\in \mathcal{N}_r(i)\setminus\{i\}}
\mathbf{1}[t_j\in[t_i-\tau,t_i]]\cdot \mathbf{1}[p_j=p_i].
$$
- 判决：
$$
\text{keep}_i=\mathbf{1}\!\left[C_i^{(p)}\ge \theta_n\right],\quad \theta_n=\texttt{min-neighbors}.
$$

`EBF`（`src/myevs/denoise/ops/ebf.py`）：
- 你指出的是对的：当前实现是“时间线性核 + 极性门控 + 空间等权求和”，没有空间高斯核。
- 对每个邻居 \(j\)（不含中心点）：
$$
w_t(i,j)=\max\!\left(0,\,1-\frac{|t_i-t_j|}{\tau}\right),\qquad
w_p(i,j)=\mathbf{1}[p_j=p_i].
$$
- 评分：
$$
S_i=\sum_{j\in \mathcal{N}_r(i)\setminus\{i\}} w_t(i,j)\,w_p(i,j).
$$
- 判决：
$$
\text{keep}_i=\mathbf{1}[S_i>\theta_f],\quad \theta_f=\texttt{min-neighbors}.
$$

`N149`（`src/myevs/denoise/ops/ebfopt_part2/n149_n145_s52_euclid_compactlut_backbone.py`）：
- 基底项：同/反极性分别累计，时间核是平方线性衰减，空间核是欧式高斯 LUT。
$$
w_t(i,j)=\left(\max\!\left(0,1-\frac{\Delta t_{ij}}{\tau}\right)\right)^2,\qquad
w_s(i,j)=\exp\!\left(-\frac{\|q_i-q_j\|_2^2}{2\sigma^2}\right).
$$
$$
R_i^{+}=\sum_{j\in\mathcal{N}_r(i)\setminus\{i\}} w_t(i,j)\, w_s(i,j)\,\mathbf{1}[p_j=p_i],\qquad
R_i^{-}=\sum_{j\in\mathcal{N}_r(i)\setminus\{i\}} w_t(i,j)\, w_s(i,j)\,\mathbf{1}[p_j=-p_i].
$$
- 状态定义（代码变量映射）：
1. \(h_i\)：中心像素热状态（`hot_state`）。
2. \(u_i\)：自状态归一化（`u_self`）。
3. \(m_i\)：反极混合状态 EMA（`mstate`）。
4. \(b_i\)：支持增益状态 EMA（`b`）。
5. \(s_i\)：支持率（`cnt_support/cnt_possible`）。
- 评分：
$$
u_i=\frac{h_i}{h_i+\tau/2+\varepsilon},\quad
\alpha_i=(1-m_i)^2,\quad
\widetilde{R}_i=R_i^{+}+\alpha_i R_i^{-}.
$$
$$
B_i=\frac{\widetilde{R}_i}{1+u_i^2},\qquad
S_i=B_i\,(1+b_i s_i).
$$
- 最终由外部阈值扫频决策（脚本里对 `S_i` 取阈值）。

`N179`（`src/myevs/denoise/ops/ebfopt_part2/n179_cpp_final_fast32_backbone.py`）：
- 版本更正（2026-05-07）：
1. `n179` 恢复为“原始 N179 定义”：即 `n176` 的最终封装（工程/C++ 优化口径），不包含 `pb/pr/pg` 的 `pi-proxy` 新分支。
2. 本轮新增的 `pi` 组合演化单独命名为 `n180`，文件为：
   - `src/myevs/denoise/ops/ebfopt_part2/n180_cpp_pi_proxy_final_backbone.py`
   - `src/myevs/denoise/ops/ebfopt_part2/n180_realtime_relief_pi_proxy_backbone.py`
   - `src/myevs/denoise/ops/ebfopt_part2/n180_s52lite_pi_proxy_gate_backbone.py`
3. 之后规则：`n179` 只做 baseline；`n180` 继续演化。

- `n179`（基线）核心式：
$$
\pi_i=\frac{\text{bad}_i+r_i}{2},\quad
u_i' = u_i(1-k_s s_i)(1+k_m\text{mix}_i)\left(1+0.5\pi_i-r_g\text{good}_i\right),
$$
$$
S_i=\frac{R_i^+ + (1-m_i)^2R_i^-}{1+(u_i')^2}\cdot\left(1+b_i s_i(1+c_{sg}\text{good}_i)\right).
$$

- `n180`（新分支）核心式：
$$
\pi_i^{proxy}=\operatorname{clip}\!\left(k_{bad}\,\text{bad}_i+k_r(1-s_i)-k_{good}\,\text{good}_i,\ 0,1\right),
$$
$$
u_i' = u_i(1-k_s s_i)(1+k_m\text{mix}_i)(1+\pi_i^{proxy}),
$$
$$
S_i=\frac{R_i^+ + (1-m_i)^2R_i^-}{1+(u_i')^2}\cdot(1+b_i s_i).
$$

- 最终优化版（当前默认参数）：
1. `k_sfrac=0.4`
2. `k_mix=0.0`
3. `rhythm_pressure_coeff (r_p)=0.0`
4. `rhythm_good_coeff (r_g)=0.75`
5. `support_good_coeff (c_{sg})=0.0`

- 这套默认值下，N179 的核心思想可概括为：  
“用同/反极邻域证据做主判别，用自状态 \(u\) 与支持率 \(s\) 做平滑抑制/增强；节律分支只保留对 \(u\) 的弱抑制，不再额外叠加复杂支持项。”

- 记号与代码映射（按计算顺序）：
1. 当前事件 \(e_i=(x_i,y_i,t_i,p_i)\)，中心像素索引为 \(\text{idx}(i)\)。
2. \(\Delta t_{ij}=|t_i-t_j|\)。
3. \(\Delta t_i^{(0)}\)：中心像素最近一次事件间隔（`dt0`）。
4. \(u_i\in[0,1]\)：中心短时活跃度（`u_lite`）：
$$
u_i=\operatorname{clip}\!\left(1-\frac{\Delta t_i^{(0)}}{\tau},0,1\right).
$$
5. \(\text{mix}_i\in[0,1]\)：当前邻域反极占比（`mix`）：
$$
\text{mix}_i=\frac{R_i^{-}}{R_i^{+}+R_i^{-}+\varepsilon}.
$$
6. \(m_i\in[0,1]\)：反极占比 EMA（`mstate`）：
$$
m_i=m_{i-1}+\frac{\text{mix}_i-m_{i-1}}{N_{\text{ema}}},\quad N_{\text{ema}}=4096.
$$
7. \(\text{bad}_i,\text{good}_i\in[0,1]\)：快速反极/同极节律强度（`rhythm_bad`,`rhythm_good`）：
$$
\text{bad}_i=
\begin{cases}
u_i,& p_{\text{prev}}(\text{idx}(i))=-p_i\\
0,& \text{otherwise}
\end{cases},
\quad
\text{good}_i=
\begin{cases}
u_i,& p_{\text{prev}}(\text{idx}(i))=p_i\\
0,& \text{otherwise}
\end{cases}
$$
8. \(r_i\in[0,1]\)：`bad` 的 EMA（`rstate`）：
$$
r_i=r_{i-1}+\frac{\text{bad}_i-r_{i-1}}{N_{\text{ema}}}.
$$
9. \(\pi_i\in[0,1]\)：原始 N179 的节律压力项，由快速反极翻转与其慢状态平均得到：
$$
\pi_i=\frac{\text{bad}_i+r_i}{2}.
$$
10. \(s_i\in[0,1]\)：支持率（`sfrac`）：
$$
s_i=\frac{\text{cnt\_support}_i}{\text{cnt\_possible}_i}.
$$
11. \(b_i\in[0,1]\)：支持增益状态（`b`）：
$$
b_i=b_{i-1}+\frac{u_i'-b_{i-1}}{N_{\text{ema}}}.
$$

- 邻域证据（同/反极）：
$$
w_t(i,j)=\left(\max\!\left(0,1-\frac{\Delta t_{ij}}{\tau}\right)\right)^2,\qquad
w_s(i,j)=\exp\!\left(-\frac{\|q_i-q_j\|_2^2}{2\sigma^2}\right),
$$
$$
R_i^{+}=\sum_{j\in\mathcal{N}_r(i)\setminus\{i\}} w_t w_s\mathbf{1}[p_j=p_i],\qquad
R_i^{-}=\sum_{j\in\mathcal{N}_r(i)\setminus\{i\}} w_t w_s\mathbf{1}[p_j=-p_i].
$$

- 第一级门控（反极抑制）：
$$
\alpha_i=(1-m_i)^2,\qquad
\alpha_i'=\alpha_i\,(1-r_p\,\pi_i).
$$
在当前默认 \(r_p=0\) 时，化简为：
$$
\alpha_i'=\alpha_i.
$$

- 第二级门控（自状态修正）：
$$
u_i' = u_i\cdot (1-k_s s_i)\cdot (1+k_m\cdot \text{mix}_i)\cdot \left(1+0.5\pi_i-r_g\text{good}_i\right),
$$
当前默认 \(k_m=0\) 时化简为：
$$
u_i' = u_i\cdot (1-k_s s_i)\cdot \left(1+0.5\pi_i-r_g\text{good}_i\right).
$$

- 融合与打分：
$$
\widetilde{R}_i=R_i^{+}+\alpha_i'R_i^{-},\qquad
B_i=\frac{\widetilde{R}_i}{1+(u_i')^2},
$$
$$
S_i=B_i\cdot\left(1+b_i s_i(1+c_{sg}\,\text{good}_i)\right).
$$
当前默认 \(c_{sg}=0\) 时化简为：
$$
S_i=B_i\cdot(1+b_i s_i).
$$

- 因而“最终优化版 N179”可写成更简洁的三段式：
$$
\widetilde{R}_i = R_i^{+} + (1-m_i)^2 R_i^{-},
$$
$$
u_i' = u_i(1-k_s s_i)\left(1+0.5\pi_i-r_g\text{good}_i\right),
$$
$$
S_i=\frac{\widetilde{R}_i}{1+(u_i')^2}\cdot(1+b_i s_i).
$$
- 参数影响与可去除性（当前实验结论）：
1. `k_sfrac`：有效，主要影响排序细节与中高噪分离。
2. `k_mix`：多数据集常趋向 `0`，可弱化。
3. `rp`：在 DVSCLEAN 基本无效、LED 多数场景偏好 `0`、ED24 差值很小；可固定为 `0`。
4. `rg`：有稳定但小幅影响；在 `rp=0` 下扩展扫频到 `1.0` 后，最佳区间在高值端（`0.375~1.0`），全局默认取 `0.75`。

`N179` 符号表（避免“有尾没头”）：

| 符号 | 含义 | 取值范围 | 代码变量 |
|---|---|---|---|
| \(u_i\) | 中心近期活跃度（由中心间隔归一化） | \([0,1]\) | `u_lite` |
| \(\text{mix}_i\) | 邻域反极占比 | \([0,1]\) | `mix` |
| \(m_i\) | 反极占比 EMA 状态 | \([0,1]\) | `mstate` |
| \(\text{bad}_i\) | 当前是否“快速反极翻转”强度 | \([0,1]\) | `rhythm_bad` |
| \(\text{good}_i\) | 当前是否“快速同极连续”强度 | \([0,1]\) | `rhythm_good` |
| \(r_i\) | \(\text{bad}\) 的 EMA 状态 | \([0,1]\) | `rstate` |
| \(\pi_i\) | 原始 N179 节律压力 \((\text{bad}_i+r_i)/2\) | \([0,1]\) | `rhythm_pressure` |
| \(s_i\) | 邻域支持率 | \([0,1]\) | `sfrac` |
| \(b_i\) | 支持增益 EMA 状态 | \([0,1]\) | `b` |
| \(k_s\) | 支持率调制系数 | \([0,1]\) 常用 | `k_sfrac` |
| \(k_m\) | 反极混合调制系数 | \([0,1]\) 常用 | `k_mix` |
| \(r_p\) | 节律压力抑制系数 | \([0,1]\) | `rhythm_pressure_coeff` |
| \(r_g\) | 同极节律抑制系数 | \([0,1]\) | `rhythm_good_coeff` |
| \(c_{sg}\) | 支持项中 `good` 增益系数 | \([0,2]\) | `support_good_coeff` |
| \(N_{\text{ema}}\) | EMA 时间常数 | 正数（当前 4096） | `N` |

`N181`（删项式简化分支，`src/myevs/denoise/ops/ebfopt_part2/n181_simplified_n179_backbone.py`）：

- 目的：不再向 N179 继续增加自由系数，而是验证删去弱收益项后能否保持接近 N179 的精度。
- `n181-conservative` 保留 `good_i` 对中心自激状态的抑制：
$$
\widetilde{R}_i=R_i^+ +(1-m_i)^2R_i^-,
$$
$$
u_i'=u_i(1-k_s s_i)(1-r_g\text{good}_i),
$$
$$
S_i=\frac{\widetilde{R}_i}{1+(u_i')^2}(1+b_i s_i).
$$
- `n181-minimal` 进一步删除 `good_i` 抑制，只保留支持率对中心自激的缓解：
$$
u_i'=u_i(1-k_s s_i),\qquad
S_i=\frac{R_i^+ +(1-m_i)^2R_i^-}{1+(u_i')^2}(1+b_i s_i).
$$
- 相比 N179，N181 删除了 `rp`、`k_mix`、`0.5\pi_i` 正向放大项、`support_good_coeff`。因此它不是为了追求历史最高点，而是用于验证公式简化是否成立。

`N181` 简化验证结果（2026-05-07）：

| Dataset | Scene/Level | Candidate | Best tag | AUC_best | F1@Best-AUC-param | MESR | AOCC | 备注 |
|---|---|---|---|---:|---:|---:|---:|---|
| ED24 | Pedestrain_06 light | minimal | `n181_min_s9_r4_tau256000` | 0.954425 | 0.956366 | 0.997947 | 0.821692 | 完整扫频 |
| ED24 | Pedestrain_06 light_mid | minimal | `n181_min_s9_r4_tau256000` | 0.947836 | 0.903433 | 0.956865 | 0.815926 | 完整扫频 |
| ED24 | Pedestrain_06 mid | conservative | `n181_cons_s9_r4_tau256000` | 0.943111 | 0.836211 | 0.995567 | 0.807941 | 完整扫频 |
| ED24 | Pedestrain_06 heavy | conservative | `n181_cons_s9_r4_tau256000` | 0.938695 | 0.784644 | 1.005212 | 0.797154 | 完整扫频 |
| ED24 | Bicycle_02 light | minimal | `n181_min_s9_r4_tau256000` | 0.985647 | 0.974496 | 1.074401 | 0.628614 | 完整扫频 |
| ED24 | Bicycle_02 light_mid | minimal | `n181_min_s9_r4_tau256000` | 0.982982 | 0.948604 | 1.017163 | 0.622483 | 完整扫频 |
| ED24 | Bicycle_02 mid | conservative | `n181_cons_s9_r4_tau256000` | 0.979357 | 0.914486 | 1.046768 | 0.616123 | 完整扫频 |
| DVSCLEAN | MAH00448 ratio100 | conservative | `n181_dvsclean_cons_s9_r4_tau128000` | 0.995465 | 0.986268 | 0.720824 | 0.781865 | 300K 快速确认 |
| LED | scene_100 100ms | conservative | `n181_led_cons_s5_r2_tau32000` | 0.856280 | 0.975955 | 0.213646 | 2.048802 | 300K 快速确认 |

结论：`minimal` 在 ED24 轻噪下略优，但中/重噪与 LED/DVSCLEAN 更偏向 `conservative`。这说明 `good_i` 抑制项不是无效项；如果论文主算法需要稳健性，优先保留 `n181-conservative` 作为候选，`minimal` 仅作为删项消融对照。

`N181` 与 `EBF / N149 / N179` 的直接对比（`conservative` 重跑，固定各 level 历史最优 `s/tau` 并重扫 `k_sfrac,r_g`，2026-05-08）：

ED24 行人：

| Level | EBF AUC/F1 | N149 AUC/F1 | N179 AUC/F1 | N181 best AUC/F1 | N181 vs EBF | N181 vs N149 | N181 vs N179 |
|---|---:|---:|---:|---:|---:|---:|---:|
| light | 0.933627 / 0.951967 | 0.953997 / 0.959438 | 0.954849 / 0.958507 | 0.951822 / 0.958689 | +0.018195 | -0.002175 | -0.003027 |
| light_mid | - | - | - | 0.948088 / 0.903615 | - | - | - |
| mid | 0.918293 / 0.812238 | 0.946041 / 0.839209 | 0.944215 / 0.838142 | 0.938436 / 0.838038 | +0.020143 | -0.007605 | -0.005779 |
| heavy | 0.890842 / 0.756961 | 0.934184 / 0.788696 | 0.938908 / 0.787984 | 0.933843 / 0.790154 | +0.043001 | -0.000341 | -0.005065 |

ED24 Bicycle：

| Level | EBF AUC/F1 | N149 AUC/F1 | N179 AUC/F1 | N181 best AUC/F1 | N181 vs EBF | N181 vs N149 | N181 vs N179 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1.8 light | 0.980557 / 0.973217 | 0.986759 / 0.975852 | 0.985747 / 0.975215 | 0.985188 / 0.975144 | +0.004631 | -0.001571 | -0.000559 |
| 2.1 light_mid | 0.975204 / 0.943114 | 0.984161 / 0.950202 | 0.983211 / 0.949102 | 0.983052 / 0.948827 | +0.007848 | -0.001109 | -0.000159 |
| 2.5 mid | 0.970035 / 0.904248 | 0.980782 / 0.917088 | 0.979715 / 0.914409 | 0.976939 / 0.914025 | +0.006904 | -0.003843 | -0.002776 |

`N181` 复核表（按你指定口径：`conservative-only`，固定各 level 历史最优 `s/tau`，重扫 `k_sfrac∈{0,0.1,0.2}`，`r_g∈{0,0.1,...,1.0}`）：

| Scene | Level | old N181 AUC/F1 | new N181 AUC/F1 | best \(k_{sfrac}\) | best \(r_g\) | \(\Delta\)AUC | \(\Delta\)F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| myPedestrain_06 | light | 0.954425 / 0.956366 | 0.951923 / 0.958738 | 0.0 | 0.1 | -0.002502 | +0.002372 |
| myPedestrain_06 | light_mid | 0.947836 / 0.903433 | 0.948233 / 0.903687 | 0.0 | 0.2 | +0.000397 | +0.000254 |
| myPedestrain_06 | mid | 0.943111 / 0.836211 | 0.938610 / 0.838299 | 0.0 | 0.5 | -0.004501 | +0.002088 |
| myPedestrain_06 | heavy | 0.938695 / 0.784644 | 0.934098 / 0.790474 | 0.0 | 0.6 | -0.004597 | +0.005830 |
| myBicycle_02 | light | 0.985647 / 0.974496 | 0.985283 / 0.975175 | 0.0 | 0.0 | -0.000364 | +0.000679 |
| myBicycle_02 | light_mid | 0.982982 / 0.948604 | 0.983106 / 0.949000 | 0.0 | 0.2 | +0.000124 | +0.000396 |
| myBicycle_02 | mid | 0.979357 / 0.914486 | 0.976994 / 0.914096 | 0.0 | 0.6 | -0.002363 | -0.000390 |

说明：本次复核结果对应汇总文件 `data/ED24/n181_rescan_ks0to02_rg/compare_vs_old_n181_table.csv`。

`N181` 精细扫频补充（`conservative`，固定 `s=9,tau=256000`）：

- 扫频范围：`k_sfrac=0.34:0.02:0.46`，`r_g=0.65:0.02:0.85`
- 最优参数（7 个子集）稳定落在：`k_sfrac≈0.34`，`r_g≈0.65`
- 与 old N181 对比后，AUC 变化量级约在 `[-3.6e-4, +2.1e-4]`，F1 变化量级约在 `[-7.1e-4, +4.5e-4]`

结论：

1. `k_sfrac` 与 `r_g` 对 `N181` 的影响属于“小幅微调”，不会带来大跨度性能变化。
2. 之前出现的大差异主要来自口径变化（`mode/tau` 或网格不含关键点），而非 `k_sfrac/r_g` 本身敏感。
3. 若以稳健默认为主，`N181-conservative` 可采用 `k_sfrac=0.34, r_g=0.65`；若追求完全对齐历史表，可继续保留 `k_sfrac=0.4, r_g=0.75`。

DVSCLEAN / LED 快速确认：

| Dataset | Method | AUC | F1 | 口径 |
|---|---|---:|---:|---|
| DVSCLEAN MAH00448 ratio100 | EBF | 0.992530 | - | full scene summary |
| DVSCLEAN MAH00448 ratio100 | N149 | 0.996266 | - | full scene summary |
| DVSCLEAN MAH00448 ratio100 | N179 | 0.995813 | 0.986674 | fullrun N179 summary |
| DVSCLEAN MAH00448 ratio100 | N181-conservative | 0.995465 | 0.986268 | 300K 快速确认 |
| DVSCLEAN MAH00448 ratio100 | N181-minimal | 0.995460 | 0.986216 | 300K 快速确认 |
| LED scene_100 | EBF | 0.834866 | 0.848575 | 300K |
| LED scene_100 | N149 | 0.873910 | 0.878399 | 300K |
| LED scene_100 | N181-conservative | 0.856280 | 0.975955 | 300K |
| LED scene_100 | N181-minimal | 0.849597 | 0.975955 | 300K |
| LED scene_100 | EBF | 0.856949 | 0.897817 | fullrun |
| LED scene_100 | N149 | 0.913320 | 0.909076 | fullrun |
| LED scene_100 | N179 | 0.919596 | 0.978401 | existing N179 full summary |

综合判断：

1. `N181` 相对 `EBF` 基本成立：在 ED24 / DVSCLEAN / LED 快速确认中都不低于 EBF，ED24 提升尤其明显。
2. `N181` 相对 `N149` 不稳定：ED24 行人 light/heavy 可接近或略高，但 Bicycle / DVSCLEAN / LED 多数低于 N149。
3. `N181` 相对 `N179` 多数小幅下降：ED24 上 AUC 下降通常在 `0.0001~0.0011`，属于小损失；DVSCLEAN / LED 仍需 fullrun 同口径确认。
4. 当前论文主算法仍建议以 `N179` 为稳健主版本；`N181` 更适合作为“删项式简化版 / 可解释性消融版”，除非后续 fullrun 证明其在更多数据集上与 N179 基本等价。

`N182`（`src/myevs/denoise/ops/ebfopt_part2/n182_pi_restored_simplified_backbone.py`）：

- 目的：验证从 `N179` 中只恢复一个可解释项 \(\pi_i\)，能否追回 `N181` 相对 `N179` 的精度损失。
- 不恢复 `k_mix/rp/c_sg/pb/pr/pg`，避免重新回到多自由系数公式。
- 本轮只扫固定候选 \(\lambda\in\{0.25,0.5\}\)。

公式：
$$
\pi_i=\frac{\text{bad}_i+r_i}{2},
$$
$$
\widetilde{R}_i=R_i^+ +(1-m_i)^2R_i^-,
$$
$$
u_i'=u_i(1-k_s s_i)(1-r_g\text{good}_i)(1+\lambda\pi_i),
$$
$$
S_i=\frac{\widetilde{R}_i}{1+(u_i')^2}(1+b_i s_i).
$$

`N182` 与 `EBF / N149 / N179 / N181` 的 AUC 对比（小范围确认，2026-05-08）：

| Dataset | Level/Scene | EBF AUC | N149 AUC | N179 AUC | N181 AUC | N182 AUC | N181-N179 | N182-N179 | N182-N181 |
|---|---|---|---|---|---|---|---|---|---|
| ED24 | Ped light | 0.933627 | 0.953997 | 0.954849 | 0.954425 | 0.954240 | -0.000424 | -0.000609 | -0.000185 |
| ED24 | Ped light_mid | - | - | - | 0.947836 | 0.947993 | - | - | +0.000157 |
| ED24 | Ped mid | 0.918293 | 0.946041 | 0.944215 | 0.943111 | 0.943987 | -0.001104 | -0.000228 | +0.000876 |
| ED24 | Ped heavy | 0.890842 | 0.934184 | 0.938908 | 0.938695 | 0.939567 | -0.000213 | +0.000659 | +0.000872 |
| ED24 | Bicycle light | 0.980557 | 0.986759 | 0.985747 | 0.985647 | 0.985383 | -0.000100 | -0.000364 | -0.000264 |
| ED24 | Bicycle light_mid | 0.975204 | 0.984161 | 0.983211 | 0.982982 | 0.982960 | -0.000229 | -0.000251 | -0.000022 |
| ED24 | Bicycle mid | 0.970035 | 0.980782 | 0.979715 | 0.979357 | 0.979637 | -0.000358 | -0.000078 | +0.000280 |
| DVSCLEAN | MAH00448 ratio100 | 0.992530 | 0.996266 | 0.995813 | 0.995465 | 0.995464 | -0.000348 | -0.000349 | -0.000001 |
| LED | scene_100 300K | 0.834866 | 0.873910 | - | 0.856280 | 0.855691 | - | - | -0.000589 |

`N182` 与 `EBF / N149 / N179 / N181` 的 F1 对比：

| Dataset | Level/Scene | EBF F1 | N149 F1 | N179 F1 | N181 F1 | N182 F1 | N181-N179 | N182-N179 | N182-N181 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ED24 | Ped light | 0.951967 | 0.959438 | 0.958507 | 0.956366 | 0.955760 | -0.002141 | -0.002747 | -0.000606 |
| ED24 | Ped light_mid | - | - | - | 0.903433 | 0.903041 | - | - | -0.000392 |
| ED24 | Ped mid | 0.812238 | 0.839209 | 0.838142 | 0.836211 | 0.836504 | -0.001931 | -0.001638 | +0.000293 |
| ED24 | Ped heavy | 0.756961 | 0.788696 | 0.787984 | 0.784644 | 0.784826 | -0.003340 | -0.003158 | +0.000182 |
| ED24 | Bicycle light | 0.973217 | 0.975852 | 0.975215 | 0.974496 | 0.974003 | -0.000719 | -0.001212 | -0.000493 |
| ED24 | Bicycle light_mid | 0.943114 | 0.950202 | 0.949102 | 0.948604 | 0.948707 | -0.000498 | -0.000395 | +0.000103 |
| ED24 | Bicycle mid | 0.904248 | 0.917088 | 0.914409 | 0.914486 | 0.914910 | +0.000077 | +0.000501 | +0.000424 |
| DVSCLEAN | MAH00448 ratio100 | - | - | 0.986674 | 0.986268 | 0.986271 | -0.000406 | -0.000403 | +0.000003 |
| LED | scene_100 300K | 0.848575 | 0.878399 | - | 0.975955 | 0.975955 | - | - | +0.000000 |

`N182` 结论：

1. `N182` 能追回一部分 `N181` 的损失，但不稳定：ED24 Ped mid/heavy、Bicycle mid 有提升，Ped light、Bicycle light、LED scene_100 下降。
2. `N182` 仍明显强于 `EBF`：在已对齐的 ED24 与 DVSCLEAN/LED 300K 口径中，AUC 均高于 EBF。
3. `N182` 不能稳定接近或超过 `N149`：ED24 Ped heavy 可超过 N149，但 Bicycle / DVSCLEAN / LED 仍低于 N149。
4. `N182` 相对 `N181` 的平均收益很小，且有正有负；这说明恢复 \(\pi_i\) 不是稳定有效的简化增强。
5. 当前建议停止继续优化简化分支：`N179` 继续作为论文主算法，`N181` 作为简化消融，`N182` 作为“恢复 \(\pi_i\) 不足以稳定追回精度”的反证实验。

`N183`（`src/myevs/denoise/ops/ebfopt_part2/n183_pi_source_ablation_backbone.py`）：

- 目标：在不扫 `s/tau` 的前提下，只验证 \(\pi_i\) 的来源（`bad` vs `r` vs `avg`）对精度的影响。
- 固定参数：`k_sfrac=0.4, k_mix=0, rp=0, rg=0.75, c_sg=0`，`lambda=0.5`，`mode=conservative`。
- 命名口径：`N179` 为原始主线；`N180` 为简化变体分支（本节已补齐 ED24 缺失结果并同表对比）。
- 记号约定：本节新增 `N179*`，表示“同口径锚点版 N179”（固定 `k_sfrac=0.4,k_mix=0,rp=0,c_sg=0` 且在 `N184` 等价点 `a=0.5,rg=0.75` 与 `N179` 一致）。代码方法名仍为 `n179`。
- 固定 `s/tau`（沿用此前各噪声环境最佳）：
  - Pedestrain_06: light `s9,tau=512000`; light_mid `s9,tau=256000`; mid/heavy `s9,tau=128000`
  - Bicycle_02: light `s9,tau=512000`; light_mid/mid `s9,tau=256000`

同一页公式对照（`N179*~N184`）：

| 方法 | \(u_i'\) 关键式 | 主要变化 |
|---|---|---|
| N179* | \(u_i'=u_i(1-k_s s_i)(1+k_m\,mix_i)(1+0.5\pi_i-r_g good_i)\) | 同口径锚点版（用于与 N184 公平比较） |
| N180 | \(u_i'=u_i(1-k_s s_i)(1+k_m\,mix_i)(1+\pi_i^{proxy}-r_g good_i)\) | \(\pi_i\) 改为 proxy 组合（`pb/pr/pg`） |
| N181 | \(u_i'=u_i(1-k_s s_i)(1-r_g good_i)\) | 删去 \(\pi_i\) 与 `mix` 增益，最简化 |
| N182 | \(u_i'=u_i(1-k_s s_i)(1-r_g good_i)(1+\lambda\pi_i)\) | 在 N181 上最小恢复 \(\pi_i\) 项 |
| N183 | \(u_i'=u_i(1-k_s s_i)(1-r_g good_i)(1+\lambda\pi_i^{src})\) | 仅扫 \(\pi_i\) 来源（`bad/r/avg`） |
| N184 | \(u_i'=u_i(1-k_s s_i)(1+a\pi_i-r_g good_i)\) | 把 `0.5` 改成可扫超参数 `a`，与 `r_g` 联扫 |

单结论表（ED24 同口径，固定各 level 最优 `s/tau`；保留原 `N179`，新增 `N179*` 锚点；`N184` 为扩扫 `a∈[1,3],步长0.2` 与 `r_g∈[0,1],步长0.1` 后 best）：

| Dataset | Level/Scene | EBF (AUC/F1) | N149 (AUC/F1) | N179(原表) (AUC/F1) | N179* (AUC/F1) | N180 (AUC/F1) | N181 (AUC/F1) | N182 (AUC/F1) | N183 (AUC/F1) | N184-best (AUC/F1) |
|---|---|---|---|---|---|---|---|---|---|---|
| ED24 Pedestrain_06 | light | 0.933627 / 0.951967 | 0.953997 / 0.959438 | 0.954849 / 0.958507 | 0.950942 / 0.957831 | 0.951641 / 0.958789 | 0.954425 / 0.956366 | 0.954240 / 0.955760 | 0.951075 / 0.957818 | 0.951721 / 0.958760 |
| ED24 Pedestrain_06 | light_mid | - | - | - | 0.947981 / 0.903010 | 0.948320 / 0.903281 | 0.947836 / 0.903433 | 0.947993 / 0.903041 | 0.947981 / 0.903010 | 0.948405 / 0.903301 |
| ED24 Pedestrain_06 | mid | 0.918293 / 0.812238 | 0.946041 / 0.839209 | 0.944215 / 0.838142 | 0.939424 / 0.838748 | 0.939364 / 0.836056 | 0.943111 / 0.836211 | 0.943987 / 0.836504 | 0.939508 / 0.838558 | 0.939784 / 0.838131 |
| ED24 Pedestrain_06 | heavy | 0.890842 / 0.756961 | 0.934184 / 0.788696 | 0.938908 / 0.787984 | 0.935270 / 0.790831 | 0.934152 / 0.783904 | 0.938695 / 0.784644 | 0.939567 / 0.784826 | 0.935320 / 0.790491 | 0.935499 / 0.789323 |
| ED24 Bicycle_02 | light | 0.980557 / 0.973217 | 0.986759 / 0.975852 | 0.985747 / 0.975215 | 0.984653 / 0.974512 | 0.985230 / 0.975201 | 0.985647 / 0.974496 | 0.985383 / 0.974003 | 0.984653 / 0.974512 | 0.985269 / 0.975198 |
| ED24 Bicycle_02 | light_mid | 0.975204 / 0.943114 | 0.984161 / 0.950202 | 0.983211 / 0.949102 | 0.982949 / 0.948644 | 0.983228 / 0.948576 | 0.982982 / 0.948604 | 0.982960 / 0.948707 | 0.982949 / 0.948644 | 0.983239 / 0.948598 |
| ED24 Bicycle_02 | mid | 0.970035 / 0.904248 | 0.980782 / 0.917088 | 0.979715 / 0.914409 | 0.977311 / 0.914325 | 0.977304 / 0.912447 | 0.979357 / 0.914486 | 0.979637 / 0.914910 | 0.979681 / 0.915065 | 0.977455 / 0.913731 |

`N183` 的 \(\pi_i\) 来源消融（固定 `s/tau`，AUC）：

| Dataset | Level/Scene | best source | bad | r | avg |
|---|---|---|---:|---:|---:|
| ED24 Pedestrain_06 | light | r | 0.950789 | 0.951075 | 0.950942 |
| ED24 Pedestrain_06 | light_mid | avg | 0.947904 | 0.947873 | 0.947981 |
| ED24 Pedestrain_06 | mid | bad | 0.939508 | 0.939018 | 0.939424 |
| ED24 Pedestrain_06 | heavy | bad | 0.935320 | 0.934869 | 0.935270 |
| ED24 Bicycle_02 | light | avg | 0.984652 | 0.984631 | 0.984653 |
| ED24 Bicycle_02 | light_mid | avg | 0.982939 | 0.982916 | 0.982949 |
| ED24 Bicycle_02 | mid | r | 0.979574 | 0.979681 | 0.979636 |

`N183` 结论（ED24 两数据集）：

1. `N183` 没有超过 `N179`，也没有稳定超过 `N181/N182`，说明“仅替换 \(\pi_i\) 来源”不足以形成新的主算法。
2. `bad` 与 `r` 都有作用，但优势随场景切换：Ped 中噪偏 `bad`，Bicycle 中噪偏 `r`，轻噪常为 `avg/r`，不存在全局单一最优来源。
3. `avg=(bad+r)/2` 作为折中项依然合理：它在多个场景接近最优且最稳定，支持继续把 `N179` 作为主线定义。
4. 在当前证据下，不建议继续在 `N183` 方向扩展复杂度；保留 `N183` 作为“\(\pi_i\) 来源可解释性”消融结果即可。

`N184` 的 \(a\) 与 \(r_g\) 联扫（固定 `s/tau`，ED24 两数据集，2026-05-08）：

实验形式：
$$
u_i'=u_i(1-k_s s_i)\left(1+a\pi_i-r_g\,good_i\right)
$$
其中 \(\pi_i\) 使用 `N179` 原始定义 \((bad_i+r_i)/2\)，本轮扩扫：
$$
a\in\{1.0,1.2,\dots,3.0\},\quad r_g\in\{0,0.1,\dots,1.0\}
$$

每个场景取 AUC 最优点（`data/ED24/n184_arg_expand/summary_best_per_level_1to3.csv`）：

| Scene | Level | best \(a\) | best \(r_g\) | AUC | F1 |
|---|---|---:|---:|---:|---:|
| myPedestrain_06 | light | 3.00 | 0.20 | 0.951721 | 0.958760 |
| myPedestrain_06 | light_mid | 3.00 | 0.30 | 0.948405 | 0.903301 |
| myPedestrain_06 | mid | 3.00 | 0.60 | 0.939784 | 0.838131 |
| myPedestrain_06 | heavy | 3.00 | 0.80 | 0.935499 | 0.789323 |
| myBicycle_02 | light | 3.00 | 0.10 | 0.985269 | 0.975198 |
| myBicycle_02 | light_mid | 3.00 | 0.20 | 0.983239 | 0.948598 |
| myBicycle_02 | mid | 3.00 | 0.60 | 0.977455 | 0.913731 |

影响结论：

1. 扩到 `a=3.0` 后最优点再次落在上边界（全子集 `a=3.0`），说明该方向仍未收敛，继续上探可能还有小增益。
2. `r_g` 最优仍集中在 `0.1/0.2/0.6/0.8`，说明 `good_i` 抑制需要按场景分档，不适合单一固定值。
3. 对比 `N179*`：`N184` 在 AUC 上 7/7 子集均小幅提升；但并未全面超过“原表 N179 最优值”。

`N184` 第二轮扩扫（你指定区间 `a=3..6, step=0.2`）：

结果文件：

1. `data/ED24/myPedestrain_06/N184_arg_expand_3to6/summary_n184_expand_3to6_*.csv`
2. `data/ED24/myBicycle_02/N184_arg_expand_3to6/summary_n184_expand_3to6_*.csv`

观察：

1. 最优 `a` 再次全部贴边到 `6.0`，说明在当前结构下，`a` 方向仍是“未收敛状态”。
2. `r_g` 最优分布仍呈分档（约 `0.1/0.2/0.6/0.8`），与子场景噪声形态有关。

`N185`（按你的要求：改为 `N181` 简化结构后再引入 `a,r_g` 扫描）：

- 文件：`src/myevs/denoise/ops/ebfopt_part2/n185_n181_alpha_rg_backbone.py`
- 结构：沿用 `N181` 简化主干，仅把节律项写为
$$
u'_i=u_i(1-k_s s_i)\left(1+a\pi_i-r_g\,good_i\right)
$$
并在同样范围 `a=3..6, step=0.2; r_g=0..1, step=0.1` 扫描。

对比汇总（`data/ED24/n185_scan_compare/compare_n184_n185_3to6_vs_baselines.csv`）：

| Scene | Level | N184 best (AUC/F1) | N185 best (AUC/F1) | N184 best (a,rg) | N185 best (a,rg) |
|---|---|---:|---:|---:|---:|
| myPedestrain_06 | light | 0.951721 / 0.958760 | 0.952201 / 0.959012 | (6.0, 0.2) | (6.0, 0.0) |
| myPedestrain_06 | light_mid | 0.948405 / 0.903301 | 0.948621 / 0.903449 | (6.0, 0.3) | (6.0, 0.6) |
| myPedestrain_06 | mid | 0.939784 / 0.838131 | 0.939966 / 0.838030 | (6.0, 0.6) | (6.0, 1.0) |
| myPedestrain_06 | heavy | 0.935499 / 0.789323 | 0.935132 / 0.786371 | (6.0, 0.8) | (6.0, 1.0) |
| myBicycle_02 | light | 0.985269 / 0.975198 | 0.985357 / 0.975214 | (6.0, 0.1) | (6.0, 0.0) |
| myBicycle_02 | light_mid | 0.983239 / 0.948598 | 0.983305 / 0.948650 | (6.0, 0.2) | (6.0, 0.6) |
| myBicycle_02 | mid | 0.977455 / 0.913731 | 0.977502 / 0.913541 | (6.0, 0.6) | (6.0, 1.0) |

结论（当前证据）：

1. `N185` 在大多数 light/light_mid 子集上略优于 `N184`，但 heavy 场景会回退。
2. `N184/N185` 两条线上 `a` 都持续贴边，说明“仅靠放大 \(\pi_i\)”还没有达到稳定最优。
3. 对论文主线建议不变：`N179` 继续作为主算法；`N184/N185` 作为“可解释调参分支”与消融结果保留。

`N180`（ED24 行人三档，固定 `s=5,k_sfrac=0.4,k_mix=0,rp=0,rg=0.75,c_sg=0`）新增小规模扫参（2026-05-07，pi-proxy 分支，非 N179 baseline）：

| Level | 最优 tau(us) | 最优 \(k_{bad}\) | 最优 \(k_r\) | 最优 \(k_{good}\) | AUC | F1 |
|---|---:|---:|---:|---:|---:|---:|
| light | 512000 | 0.3 | 0.7 | 0.5 | 0.941708 | 0.949600 |
| light_mid | 512000 | 0.7 | 0.7 | 0.5 | 0.936188 | 0.889278 |
| mid | 512000 | 0.3 | 0.3 | 0.5 | 0.929339 | 0.805762 |

跨三档均值 Top-1 组合为：`k_bad=0.3, k_r=0.5, k_good=0.5`（`AUC_mean=0.933102`）。  
建议：ED24 主线可先用该组合作为默认，再按单档做微调。

`N180`（DVSCLEAN / LED）`pi` 组合扫频补充（2026-05-07，pi-proxy 分支，非 N179 baseline）：

扫频口径：
1. 固定结构参数：`k_sfrac=0.4, k_mix=0, rp=0, rg=0.75, c_sg=0`。
2. 只扫 `k_bad/k_r/k_good`（即 `pb/pr/pg`）。
3. 事件量：`max-events=200000`（快速扫频口径）。

| Dataset | Scene | 固定 s | 固定 tau(us) | 最优 \(k_{bad}\) | 最优 \(k_r\) | 最优 \(k_{good}\) | AUC | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DVSCLEAN | MAH00448 ratio100 | 9 | 128000 | 0.3 | 0.3 | 0.9 | 0.994556 | 0.984913 |
| LED | scene_100 | 5 | 16000 | 0.3 | 0.3 | 0.6 | 0.887938 | 0.976527 |

结果文件：
1. `data/summary/n179_pi_combo_scan_dvsclean_led_20260507/summary_dvsclean_mah00448_ratio100.csv`
2. `data/summary/n179_pi_combo_scan_dvsclean_led_20260507/summary_led_scene100.csv`

`KNOISE`（`src/myevs/denoise/ops/knoise.py`）：
- 该实现是“行/列最近事件索引”模型，不是完整邻域累和。
- 其中 \(\mathcal{C}_i\) 表示当前事件在“三列 + 三行”上可检查到的候选最近事件集合。
- 在 \(x\pm1,x\) 三列与 \(y\pm1,y\) 三行检查最近事件，满足“同极性 + \(\Delta t\le\tau\) + 邻接几何约束”则记 1 支持：
$$
S_i=\sum_{k\in\mathcal{C}_i}\mathbf{1}[\text{time\_ok}\land\text{pol\_ok}\land\text{adj\_ok}],
$$
$$
\text{keep}_i=\mathbf{1}[S_i\ge \theta_n].
$$

`EVFLOW`（`src/myevs/denoise/ops/evflow.py`）：
- 在局部窗口内最小二乘拟合平面 \(t=ax+by+c\)，再映射为流幅值：
- 其中 \(a,b,c\) 为局部平面参数，\((x_j,y_j)\) 为邻域事件坐标（相对中心坐标系）。
$$
\min_{a,b,c}\sum_{j\in\mathcal{N}_r(i)}\left((t_j-t_i)\cdot10^{-3}-(ax_j+by_j+c)\right)^2.
$$
$$
v_i=\sqrt{\left(-\frac{1}{a}\right)^2+\left(-\frac{1}{b}\right)^2}.
$$
- 判决方向是“小于等于阈值保留”：
$$
\text{keep}_i=\mathbf{1}[v_i\le \theta_f].
$$

`YNOISE`（`src/myevs/denoise/ops/ynoise.py`）：
- 统计邻域内“近期且同极性”事件数（当前实现包含中心像素位置）：
$$
D_i=\sum_{j\in\mathcal{N}_r(i)}
\mathbf{1}[|t_i-t_j|\le\tau]\cdot\mathbf{1}[p_j=p_i].
$$
$$
\text{keep}_i=\mathbf{1}[D_i\ge \theta_n].
$$

`TS`（`src/myevs/denoise/ops/ts.py`）：
- 同极性时间表面的邻域均值：
$$
T_i(j)=\exp\!\left(-\frac{|t_i-t_j|}{\tau}\right),\qquad
S_i=\frac{1}{|\Omega_i|}\sum_{j\in\Omega_i}T_i(j),
$$
其中 \(\Omega_i=\{j\in\mathcal{N}_r(i)\mid t_j>0\}\)。
$$
\text{keep}_i=\mathbf{1}[S_i\ge \theta_f].
$$

`MLPF`（`src/myevs/denoise/ops/mlpf.py`）：
- 当前实现是固定 \(7\times7\)（\(r=3\)）的轻量代理，不是实际神经网络推理。
- 对同极性且 \(\Delta t\le\tau\) 的像素累加线性 recency：
$$
\text{recency}_{ij}=1-\frac{|t_i-t_j|}{\tau},\qquad
S_i=\sum_{j\in\mathcal{N}_{r=3}(i)} \text{recency}_{ij}\,\mathbf{1}[p_j=p_i].
$$
$$
\text{keep}_i=\mathbf{1}[S_i\ge \theta_f].
$$

`PFD`（`src/myevs/denoise/ops/pfd.py`）：
- 当前实现对齐 `PFDs`（event-by-event 版本），含两级判决。
- Stage-1（同极性时域支持）：
$$
C_i=\sum_{j\in \mathcal{N}_r(i)\setminus\{i\}}
\mathbf{1}[p_j=p_i]\cdot\mathbf{1}[|t_i-t_j|<\tau],\qquad
\text{pass1}_i=\mathbf{1}[C_i\ge v].
$$
其中 \(v\) 对应 `refractory-us`（本项目默认固定为 1）。
- Stage-2（极性翻转一致性）：
设 \(F_i(\tau)\) 为像素 \(i\) 在时间窗 \(\tau\) 内的极性翻转次数，\
\(A_i=\sum_{j\in\mathcal{N}_r(i)\setminus\{i\}}\mathbf{1}[|t_i-t_j|\le\tau]\) 为活跃邻居数。
$$
S_i=\left|F_i(\tau)-\frac{1}{A_i}\sum_{j\in\mathcal{N}_r(i)\setminus\{i\}}F_j(\tau)\right|.
$$
$$
\text{keep}_i=\mathbf{1}\big[\text{pass1}_i\land (A_i>\theta_n)\land (S_i\le 1)\big],
$$
其中 \(\theta_n=\texttt{min-neighbors}\)。
- 模式说明（已实现）：
1. `pfd_mode=a`（默认，PFD-A）  
   \(S_i=\left|F_i(\tau)-\frac{1}{A_i}\sum_j F_j(\tau)\right|\)
1. `pfd_mode=b`（PFD-B）  
   \(S_i=\left|\frac{1}{A_i}\sum_j F_j(\tau)\right|\)
1. 当前 ED24 仅跑 `pfd_mode=a`；`pfd_mode=b` 已实现但未在本轮数据中启用（留给闪烁噪声数据集）。

评估指标定义（统一）：
$$
\mathrm{TPR}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}},\qquad
\mathrm{FPR}=\frac{\mathrm{FP}}{\mathrm{FP}+\mathrm{TN}}.
$$
$$
\mathrm{Precision}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}},\qquad
\mathrm{Recall}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}.
$$
$$
\mathrm{F1}=\frac{2\cdot \mathrm{Precision}\cdot \mathrm{Recall}}{\mathrm{Precision}+\mathrm{Recall}}.
$$
$$
\mathrm{AUC}=\int_0^1 \mathrm{TPR}(u)\,d u,\quad (u=\mathrm{FPR}).
$$

`MESR` 与 `AOCC` 为无参考结构指标：统一在 `best-AUC` 与 `best-F1` 两个工作点各记录一次。

### 3.2 复杂度、实时性与资源占用结论（实现视角）

结论先行：
- 你感觉 `EVFLOW` 慢是合理的。当前实现每个事件都要遍历时间窗内 `deque` 做局部筛选，再做一次最小二乘，平均复杂度不再是固定邻域常数级，数据密集时会明显变慢。
- `YNOISE/EBF/TS` 同属局部窗口打分，复杂度主项都是 \(O(K)\)（\(K=(2r+1)^2\)），但常数项不同：`TS` 有 `exp`，`EBF` 有浮点线性权重，`YNOISE` 主要是整数计数。
- `KNOISE` 是最轻量的对比算法（固定方向检查，近似 \(O(1)\) / event），最有利于实时。

统一记号：
$$
P = W\times H,\quad K=(2r+1)^2,\quad N=\text{事件总数}.
$$

| 算法 | 单事件时间复杂度 | 全序列复杂度 | 状态内存（粗估） | 单事件操作量（粗估） | 实时风险 |
|---|---:|---:|---:|---:|---|
| BAF | \(O(K)\)（有早停） | \(O(NK)\) | \(8P\) bytes | 最坏约 \(K-1\) 次邻居时间比较 | 中 |
| STCF | \(O(K)\)（有早停） | \(O(NK)\) | \(16P\) bytes | 最坏约 \(K-1\) 次 + 同极性判断 | 中-高 |
| EBF | \(O(K)\) | \(O(NK)\) | \(9P\) bytes | 约 \(K-1\) 次；含线性时间权重与累加 | 中 |
| YNOISE | \(O(K)\) | \(O(NK)\) | \(9P\) bytes | 约 \(K\) 次；硬窗 + 同极性计数 | 中 |
| TS | \(O(K)\) | \(O(NK)\) | \(16P\) bytes | 约 \(K\) 次 + `exp` | 中-高 |
| PFD | \(O(K)\)（含两级门控） | \(O(NK)\) | \(\approx 33P\) bytes | 邻域支持 + 翻转计数统计 | 中-高 |
| MLPF(real/proxy) | \(O(49)+O(\text{MLP forward})\) | \(O(N(49+\text{forward}))\) | \(9P\) bytes + 模型参数 | 固定49点特征 + 前向推理 | 中 |
| KNOISE | 近似 \(O(1)\) | \(O(N)\) | \(\approx 13(W+H)\) bytes | 固定行列方向检查（常数次） | 低 |
| EVFLOW | \(O(Q)+O(M)\) | 常见 \(O(N\bar Q)\)，最坏接近 \(O(N^2)\) | \(O(Q)\)（deque） | 遍历窗口事件 + 最小二乘拟合 | 高 |
| N149 | \(O(K)\)（常数较大） | \(O(NK)\) | \(\approx 13P\) bytes + LUT | 约 \(K\) 次 + 多门控状态更新 | 中-高 |

说明：
- `Q` 表示 `EVFLOW` 时间窗内候选事件数，`M` 为拟合矩阵构建与求解代价；在高事件率场景，`Q` 增大是主要瓶颈。
- 上表为实现复杂度与状态量估算，用于实时性预判；最终仍以实测 `events/s` 和 `runtime_sec` 为准。

#### 3.2.1 排序（按对比指标）

1. 按时间复杂度（快 -> 慢）  
`KNOISE` > `MLPF` > `BAF ≈ STCF ≈ YNOISE ≈ EBF ≈ TS ≈ PFD ≈ N149` > `EVFLOW`

1. 按单事件操作量（少 -> 多）  
`KNOISE` > `BAF` > `YNOISE` > `EBF` > `MLPF` > `STCF ≈ TS ≈ PFD` > `N149` > `EVFLOW`

1. 按状态内存占用（少 -> 多）  
`KNOISE` > `EBF ≈ YNOISE ≈ MLPF` > `BAF` > `STCF ≈ TS` > `PFD` > `N149` > `EVFLOW(随Q增长)`

1. 按实时风险（低 -> 高）  
`KNOISE` < `MLPF` < `BAF ≈ YNOISE ≈ EBF` < `STCF ≈ TS ≈ PFD ≈ N149` < `EVFLOW`

#### 3.2.2 为什么 EVFLOW 慢（本工程实现）

`EVFLOW` 在 `src/myevs/denoise/ops/evflow.py` 中采用：
1. 维护时间窗 `deque`；  
1. 对每个新事件遍历窗内事件并按半径筛选局部样本；  
1. 对局部样本执行一次最小二乘平面拟合（`np.linalg.lstsq`）；  
1. 由拟合参数换算流幅值再判决。  

因此它不是固定邻域常数代价，而是“与时间窗候选规模相关”的代价模型，密集场景下明显更慢。

## 5. radius-px 统一口径（2026-04-23）

- `radius-px` 统一定义为“半径”，不是直径。
- 直径与半径关系：`diameter = 2 * radius + 1`。
- 若脚本使用 `s` 或 `d`（窗口直径），必须先转换为 `radius` 再传给 `--radius-px`。
- 已完成统一的脚本：
`scripts/ED24_alg_evalu/run_slomo_baf.ps1`、`scripts/ED24_alg_evalu/run_slomo_stcf.ps1`、`scripts/ED24_alg_evalu/run_slomo_ebf_paper_s_tau.ps1`。

## 6. N147 补充结果（EBF_Part2 / compact400k）

数据来源：
- `data/ED24/myPedestrain_06/EBF_Part2/_slim_n147_794_compact400k/roc_ebf_n147_{light,mid,heavy}_labelscore_s5_7_9_tau32_64_128_256ms.csv`

| Algorithm | Level | Best AUC | Best AUC Tag | Best F1 | Best F1 Tag | Threshold |
|---|---|---:|---|---:|---|---:|
| N147 | light | 0.954903 | ebf_n147_labelscore_s9_tau256000 | 0.956405 | ebf_n147_labelscore_s9_tau256000 | 0.568591 |
| N147 | mid | 0.940047 | ebf_n147_labelscore_s9_tau256000 | 0.811565 | ebf_n147_labelscore_s9_tau256000 | 3.291225 |
| N147 | heavy | 0.936697 | ebf_n147_labelscore_s9_tau256000 | 0.765521 | ebf_n147_labelscore_s9_tau256000 | 4.651618 |

## 7. MESR / AOCC 统计规则（新约束）

从现在开始，所有“最终汇报结果”都按以下规则执行：
1. 在 `best-AUC` 对应参数点计算并记录 `MESR/AOCC`。
2. 在 `best-F1` 对应参数点计算并记录 `MESR/AOCC`。
3. 若脚本支持，优先打开：
   - `--esr-mode best`
   - `--aocc-mode best`
4. 在 README 表格中明确区分：`AUC@best-AUC`, `F1@best-F1`, `MESR/AOCC@best-AUC`, `MESR/AOCC@best-F1`。

推荐执行方式：
1. EBF Part2 变体：继续使用原 sweep 脚本原生开关。
- `scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py`
- `scripts/ED24_alg_evalu/sweep_ebf_labelscore_grid.py`
2. 横向对比算法（BAF/STCF/EBF/N149/KNOISE/EVFLOW/YNOISE/TS/MLPF/PFD）：统一使用后评估脚本。
- `scripts/eval_bestpoint_mesr_aocc.py`
- 该脚本从 ROC CSV 自动选 `best-AUC`/`best-F1` 参数点并计算 `MESR/AOCC`。


## C. Driving 数据集

## 8. 路径定义（已核验）

项目根目录：
- `D:\hjx_workspace\scientific_reserach\projects\myEVS`

本 README：
- `D:\hjx_workspace\scientific_reserach\projects\myEVS\src\myevs\denoise\ops\README.md`

ED24 数据集（已存在）：
- `D:\hjx_workspace\scientific_reserach\dataset\ED24\myPedestrain_06`

Driving 数据集（已存在）：
- `D:\hjx_workspace\scientific_reserach\dataset\DND21\mydriving\driving_noise_light_slomo_shot_withlabel`
- `D:\hjx_workspace\scientific_reserach\dataset\DND21\mydriving\driving_noise_light_mid_slomo_shot_withlabel`
- `D:\hjx_workspace\scientific_reserach\dataset\DND21\mydriving\driving_noise_mid_slomo_shot_withlabel`

脚本目录（已存在）：
- `D:\hjx_workspace\scientific_reserach\projects\myEVS\scripts\ED24_alg_evalu`
- `D:\hjx_workspace\scientific_reserach\projects\myEVS\scripts\driving_alg_evalu`
- `D:\hjx_workspace\scientific_reserach\projects\myEVS\scripts\noise_analyze`

结果根目录：
- `D:\hjx_workspace\scientific_reserach\projects\myEVS\data`

## 9. 输出目录规范

统一规范：
- `data/<DATASET>/<SCENE>/<ALGORITHM>/roc_<algorithm>_<level>.csv`
- `data/<DATASET>/<SCENE>/<ALGORITHM>/roc_<algorithm>_<level>.png`

已约定示例：
- `data/ED24/myPedestrain_06/BAF/roc_baf_light.csv`
- `data/ED24/myPedestrain_06/KNOISE/roc_knoise_mid.csv`
- `data/DND21/mydriving/EVFLOW/roc_evflow_light.csv`

命名要求：
- 算法目录统一大写（如 `KNOISE`），文件名前缀统一小写（如 `roc_knoise_*.csv`）。
- `tag` 字段必须体现关键超参（如 `evflow_r2_tau3000`）。

## 10. 推荐执行顺序

1. 先跑 ED24：`BAF/STCF/EBF + KNOISE/EVFLOW/YNOISE/TS/MLPF/PFD`
2. 再跑 Driving：同样算法集合
3. 每个算法先看三档噪声（light/light_mid/mid）的 AUC 稳定性
4. 再做跨数据集总表排序，筛选论文主结果算法

## 11. ED24 横向结果汇总（Round1 + Round2）

说明：以下三张表已经把两轮结果合并在一起。
- Round1：`BAF / STCF / EBF / N149`
- Round2：`KNOISE / EVFLOW / YNOISE / TS / MLPF / PFD`

数据来源：
- `data/ED24/myPedestrain_06/horizontal_summary_all.csv`

### 10.1 light

| Algorithm | Round | Best AUC | Best F1 Tag | Best F1 | F1 Threshold | MESR@Best-F1 | AOCC@Best-F1 |
|---|---|---:|---|---:|---:|---:|---:|
| BAF | Round1 | 0.902368 | baf_r4 | 0.941875 | 64000.000000 | 1.204377 | 0.831391 |
| STCF | Round1 | 0.946044 | stcf_r4 | 0.947306 | 256000.000000 | 1.186034 | 0.826256 |
| EBF | Round1 | 0.933627 | ebf_r5_tau512000 | 0.951967 | 2.000000 | 1.138791 | 0.825984 |
| N149 | Round1 | 0.953997 | n149_r5_tau512000_light | 0.959438 | 0.787335 | 0.996116 | 0.823566 |
| KNOISE | Round2 | 0.716790 | knoise_tau16000 | 0.911931 | 0.000000 | 1.670778 | 0.849902 |
| EVFLOW | Round2 | 0.819366 | evflow_r5_tau64000 | 0.918792 | 80.000000 | 1.883983 | 0.792855 |
| YNOISE | Round2 | 0.933149 | ynoise_r5_tau256000 | 0.948630 | 3.000000 | 1.159098 | 0.821053 |
| TS | Round2 | 0.870558 | ts_r4_decay128000 | 0.930806 | 0.050000 | 1.434201 | 0.826768 |
| MLPF | Round2 | 0.866857 | mlpf_tau32000 | 0.943647 | 0.500000 | 1.021404 | 0.824071 |
| PFD | Round2 | 0.902708 | pfd_r3_tau64000_m1 | 0.908376 | 1.000000 | 0.915288 | 0.793225 |
| N179 | Round3 | 0.954849 | n179_s9_r4_tau512000 | 0.958507 | 0.842908 | 1.012425 | 0.823840 |
| N181-minimal | Simplify | 0.954425 | n181_min_s9_r4_tau256000 | 0.956366 | 0.594046 | 0.997947 | 0.821692 |

#### 10.1B light_mid（2.1，N181 简化验证）

| Algorithm | Round | Best AUC | Best F1 Tag | Best F1 | F1 Threshold | MESR@Best-F1 | AOCC@Best-F1 |
|---|---|---:|---|---:|---:|---:|---:|
| N181-conservative | Simplify | 0.947484 | n181_cons_s9_r4_tau256000 | 0.902743 | 1.613209 | 0.987330 | 0.816631 |
| N181-minimal | Simplify | 0.947836 | n181_min_s9_r4_tau256000 | 0.903433 | 1.584168 | 0.956865 | 0.815926 |

### 10.2 mid

| Algorithm | Round | Best AUC | Best F1 Tag | Best F1 | F1 Threshold | MESR@Best-F1 | AOCC@Best-F1 |
|---|---|---:|---|---:|---:|---:|---:|
| BAF | Round1 | 0.839146 | baf_r1 | 0.686279 | 16000.000000 | 1.059620 | 0.843543 |
| STCF | Round1 | 0.895906 | stcf_r2 | 0.783758 | 32000.000000 | 1.019594 | 0.830355 |
| EBF | Round1 | 0.918293 | ebf_r5_tau128000 | 0.812238 | 6.000000 | 1.028522 | 0.795376 |
| N149 | Round1 | 0.946041 | n149_r5_tau256000_mid | 0.839209 | 2.928605 | 0.973047 | 0.800048 |
| KNOISE | Round2 | 0.662463 | knoise_tau16000 | 0.495071 | 1.000000 | 0.864277 | 0.590452 |
| EVFLOW | Round2 | 0.796419 | evflow_r4_tau16000 | 0.712599 | 80.000000 | 2.038876 | 0.798920 |
| YNOISE | Round2 | 0.908342 | ynoise_r4_tau64000 | 0.808960 | 6.000000 | 1.036820 | 0.783468 |
| TS | Round2 | 0.852829 | ts_r2_decay32000 | 0.714195 | 0.100000 | 1.600892 | 0.825788 |
| MLPF | Round2 | 0.814019 | mlpf_tau32000 | 0.659253 | 0.300000 | 1.135691 | 0.817674 |
| PFD | Round2 | 0.888911 | pfd_r3_tau32000_m3 | 0.794614 | 1.000000 | 0.919803 | 0.800480 |
| N179 | Round3 | 0.944215 | n179_s9_r4_tau128000 | 0.838142 | 2.189939 | 0.978208 | 0.798098 |
| N181-conservative | Simplify | 0.943111 | n181_cons_s9_r4_tau256000 | 0.836211 | 3.188635 | 0.995567 | 0.807941 |

### 10.3 heavy

| Algorithm | Round | Best AUC | Best F1 Tag | Best F1 | F1 Threshold | MESR@Best-F1 | AOCC@Best-F1 |
|---|---|---:|---|---:|---:|---:|---:|
| BAF | Round1 | 0.816080 | baf_r1 | 0.553047 | 8000.000000 | 0.931767 | 0.835285 |
| STCF | Round1 | 0.879051 | stcf_r2 | 0.700514 | 16000.000000 | 0.946833 | 0.819898 |
| EBF | Round1 | 0.890842 | ebf_r3_tau64000 | 0.756961 | 4.000000 | 0.985282 | 0.749912 |
| N149 | Round1 | 0.934184 | n149_r5_tau128000_heavy | 0.788696 | 2.965526 | 0.951670 | 0.784665 |
| KNOISE | Round2 | 0.641662 | knoise_tau16000 | 0.430141 | 1.000000 | 0.803600 | 0.584283 |
| EVFLOW | Round2 | 0.779147 | evflow_r2_tau16000 | 0.619358 | 80.000000 | 2.388692 | 0.730146 |
| YNOISE | Round2 | 0.897144 | ynoise_r3_tau64000 | 0.752252 | 6.000000 | 1.015795 | 0.791337 |
| TS | Round2 | 0.846531 | ts_r2_decay16000 | 0.647695 | 0.100000 | 1.113610 | 0.758223 |
| MLPF | Round2 | 0.772873 | mlpf_tau32000 | 0.518727 | 0.200000 | 0.949902 | 0.780568 |
| PFD | Round2 | 0.875718 | pfd_r3_tau32000_m3 | 0.727360 | 5.000000 | 0.931863 | 0.777345 |
| N179 | Round3 | 0.938908 | n179_s9_r4_tau128000 | 0.787984 | 3.197062 | 0.959567 | 0.787410 |
| N181-conservative | Simplify | 0.938695 | n181_cons_s9_r4_tau256000 | 0.784644 | 4.685995 | 1.005212 | 0.797154 |

### 10.3B Bicycle_02（ED24，同组展示）

说明：`myBicycle_02` 与 `myPedestrain_06` 同属 ED24，这里统一并排展示，不再分散阅读。

#### 10.3B.1 light（1.8）

| Algorithm | Round | Best AUC | Best F1 Tag | Best F1 | F1 Threshold | MESR@Best-F1 | AOCC@Best-F1 |
|---|---|---:|---|---:|---:|---:|---:|
| BAF | Round1 | 0.957215 | baf_r2 | 0.955000 | 64000.000000 | - | - |
| STCF | Round1 | 0.981157 | stcf_r4 | 0.969471 | 64000.000000 | - | - |
| EBF | Round1 | 0.980557 | ebf_r5_tau256000 | 0.973217 | 2.000000 | - | - |
| N149 | Round1 | 0.986759 | n149_r5_tau512000_light | 0.975852 | 1.118799 | - | - |
| KNOISE | Round2 | 0.769042 | knoise_tau64000 | 0.715164 | 1.000000 | - | - |
| EVFLOW | Round2 | 0.881686 | evflow_r4_tau32000 | 0.922386 | 80.000000 | - | - |
| YNOISE | Round2 | 0.975253 | ynoise_r5_tau128000 | 0.971707 | 3.000000 | - | - |
| TS | Round2 | 0.928523 | ts_r3_decay32000 | 0.943306 | 0.050000 | - | - |
| MLPF | Round2 | 0.938006 | mlpf_tau32000 | 0.953709 | 0.400000 | - | - |
| PFD | Round2 | 0.945203 | pfd_r3_tau64000_m1 | 0.943050 | 1.000000 | - | - |
| N179 | Round3 | 0.985747 | n179_s9_r4_tau512000 | 0.975215 | 1.220967 | 1.147989 | 0.631207 |
| N181-minimal | Simplify | 0.985647 | n181_min_s9_r4_tau256000 | 0.974496 | 0.931867 | 1.074401 | 0.628614 |

#### 10.3B.2 light_mid（2.1）

| Algorithm | Round | Best AUC | Best F1 Tag | Best F1 | F1 Threshold | MESR@Best-F1 | AOCC@Best-F1 |
|---|---|---:|---|---:|---:|---:|---:|
| BAF | Round1 | 0.937188 | baf_r1 | 0.879775 | 32000.000000 | - | - |
| STCF | Round1 | 0.968653 | stcf_r2 | 0.926680 | 64000.000000 | - | - |
| EBF | Round1 | 0.975204 | ebf_r5_tau128000 | 0.943114 | 4.000000 | - | - |
| N149 | Round1 | 0.984161 | n149_r5_tau256000_light_mid | 0.950202 | 1.742941 | - | - |
| KNOISE | Round2 | 0.748371 | knoise_tau32000 | 0.672396 | 1.000000 | - | - |
| EVFLOW | Round2 | 0.888969 | evflow_r4_tau32000 | 0.884358 | 80.000000 | - | - |
| YNOISE | Round2 | 0.971597 | ynoise_r4_tau64000 | 0.938415 | 4.000000 | - | - |
| TS | Round2 | 0.933748 | ts_r2_decay32000 | 0.870577 | 0.200000 | - | - |
| MLPF | Round2 | 0.935633 | mlpf_tau32000 | 0.892941 | 0.200000 | - | - |
| PFD | Round2 | 0.935834 | pfd_r3_tau32000_m2 | 0.909414 | 1.000000 | - | - |
| N179 | Round3 | 0.983211 | n179_s9_r4_tau256000 | 0.949102 | 1.877606 | 1.019155 | 0.622939 |
| N181-minimal | Simplify | 0.982982 | n181_min_s9_r4_tau256000 | 0.948604 | 1.895287 | 1.017163 | 0.622483 |

#### 10.3B.3 mid（2.5）

| Algorithm | Round | Best AUC | Best F1 Tag | Best F1 | F1 Threshold | MESR@Best-F1 | AOCC@Best-F1 |
|---|---|---:|---|---:|---:|---:|---:|
| BAF | Round1 | 0.914945 | baf_r1 | 0.764380 | 8000.000000 | - | - |
| STCF | Round1 | 0.954248 | stcf_r2 | 0.861163 | 16000.000000 | - | - |
| EBF | Round1 | 0.970035 | ebf_r4_tau128000 | 0.904248 | 6.000000 | - | - |
| N149 | Round1 | 0.980782 | n149_r5_tau256000_mid | 0.917088 | 3.010741 | - | - |
| KNOISE | Round2 | 0.721558 | knoise_tau16000 | 0.601094 | 1.000000 | - | - |
| EVFLOW | Round2 | 0.867815 | evflow_r3_tau16000 | 0.809184 | 80.000000 | - | - |
| YNOISE | Round2 | 0.964211 | ynoise_r5_tau64000 | 0.901482 | 8.000000 | - | - |
| TS | Round2 | 0.928500 | ts_r2_decay32000 | 0.802157 | 0.300000 | - | - |
| MLPF | Round2 | 0.849260 | mlpf_tau32000 | 0.769603 | 0.200000 | - | - |
| PFD | Round2 | 0.923924 | pfd_r3_tau16000_m3 | 0.862512 | 1.000000 | - | - |
| N179 | Round3 | 0.979715 | n179_s9_r4_tau256000 | 0.914409 | 3.295680 | 1.016064 | 0.612406 |
| N181-conservative | Simplify | 0.979357 | n181_cons_s9_r4_tau256000 | 0.914486 | 3.329738 | 1.046768 | 0.616123 |

### 10.4 MESR/AOCC 统一口径（ED24）

数据来源：
- `data/ED24/myPedestrain_06/horizontal_summary_all.csv`（用于论文主表，统一采用 `best-f1` 对应的 `MESR/AOCC`）
- `data/ED24/myPedestrain_06/bestpoint_mesr_aocc_summary.csv`（保留 `best-auc` 与 `best-f1` 两个工作点明细）

当前完整性检查（2026-04-27）：
- ED24 横向汇总应有 `10 algorithms × 3 levels = 30` 行，当前已齐全。
- `MESR/AOCC` 在 `horizontal_summary_all.csv` 中已全部可用（无空缺）。

### 10.5 PFD 数据结论（ED24）

- 本轮扫频口径：固定 `r=3`，扫 `Δt(time-us)`、`λ(min-neighbors)`、`m(refractory-us)`。
- 与论文默认口径不一致说明：论文主实现强调 `3x3(r=1)` 邻域；本工程在 ED24 上 `r=3` 的 AUC/F1 更优，故主表先采用 `r=3` 工程最优口径，同时保留论文口径说明用于复现实验章节。
- AUC 对比（PFD）：`light=0.902708`, `mid=0.888911`, `heavy=0.875718`。  
在 `mid/heavy` 上明显高于 `EVFLOW/TS/KNOISE/MLPF`，但低于 `YNOISE/EBF/N149`。
- F1 对比（PFD）：`light=0.908376`, `mid=0.794614`, `heavy=0.727360`。  
在 `mid/heavy` 上稳定高于 `TS/EVFLOW/KNOISE/MLPF`，与 `STCF` 接近或略优。
- 代价侧：旧版纯 Python 扫频确实很慢；本轮改为 `numba` 后，PFD 三档平均运行时已降到约 `76.418s/档`（coarse 网格）。论文中的“实时”来自 C++/FPGA 实现，本工程当前仍是 Python + numba 版本。

### 10.6 Driving 横向结果汇总（200k, light/light_mid/mid）

数据来源：
- `data/DND21/mydriving/horizontal_summary_all.csv`
- `data/DND21/mydriving/bestpoint_mesr_aocc_summary.csv`
- `data/DND21/mydriving/{ALG}/runtime_*.csv` 与 `data/DND21/mydriving/N149/runtime_n149.csv`

说明：
1. 该汇总已覆盖 `BAF/STCF/EBF/N149/KNOISE/EVFLOW/YNOISE/TS/MLPF/PFD` 全部算法。
2. 按你的要求，`EVFLOW` 的 `MESR/AOCC` 暂停计算，先留空。
3. `Runtime(s)` 为当前口径下各算法各档最新一次记录。
4. Driving 结果目录已统一为“算法优先”布局：`data/DND21/mydriving/{ALG}/roc_{alg}_{level}.csv`，与 ED24 的组织方式一致。

#### 10.6.1 light

| Algorithm | Best AUC | Best AUC Tag | F1@Best-AUC | Thr@Best-AUC | MESR@Best-AUC | AOCC@Best-AUC | Best F1 | Best F1 Tag | Thr@Best-F1 | AUC@Best-F1 | MESR@Best-F1 | AOCC@Best-F1 | Runtime(s) |
|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| BAF | 0.799350 | baf_r1_tau4000 | 0.905169 | 1.000000 | 0.788445 | 2.379989 | 0.977395 | baf_r2_tau32000 | 1.000000 | 0.593953 | 0.717649 | 2.321786 | 22.235 |
| STCF | 0.935075 | stcf_r1 | 0.977808 | 128000.000000 | 0.733109 | 2.429605 | 0.987283 | stcf_r4 | 16000.000000 | 0.902610 | 0.726550 | 2.392844 | 61.093 |
| EBF | 0.937124 | ebf_r2_tau16000 | 0.987250 | 0.000000 | 0.730291 | 2.430151 | 0.987540 | ebf_r3_tau16000 | 0.500000 | 0.916800 | 0.727634 | 2.407034 | 14.627 |
| N149 | 0.932978 | n149_r2_tau32000_light | 0.981619 | 0.259854 | 0.725054 | 2.402036 | 0.982559 | n149_r3_tau64000_light | 0.476634 | 0.926168 | 0.718161 | 2.335562 | 26.308 |
| KNOISE | 0.597576 | knoise_tau4000 | 0.979814 | 0.000000 | 0.711667 | 2.234047 | 0.979814 | knoise_tau1000 | 0.000000 | 0.596042 | 0.711667 | 2.234047 | 85.306 |
| EVFLOW | 0.822085 | evflow_r2_tau8000 | 0.920775 | 64.000000 |  |  | 0.970408 | evflow_r3_tau32000 | 64.000000 | 0.752090 |  |  | 684.152 |
| YNOISE | 0.936610 | ynoise_r2_tau8000 | 0.981020 | 1.000000 | 0.740235 | 2.461512 | 0.983444 | ynoise_r2_tau16000 | 1.000000 | 0.927710 | 0.730039 | 2.428665 | 300.664 |
| TS | 0.840665 | ts_r2_decay32000 | 0.981704 | 0.200000 | 0.761003 | 2.597591 | 0.981704 | ts_r2_decay32000 | 0.200000 | 0.840665 | 0.761003 | 2.597591 | 8.854 |
| MLPF | 0.472022 | mlpf_tau32000 | 0.972319 | 0.800000 | 0.711261 | 2.244642 | 0.972323 | mlpf_tau64000 | 0.800000 | 0.430472 | 0.705434 | 2.236060 | 446.578 |
| PFD | 0.901287 | pfd_r3_tau8000_m2 | 0.975540 | 1.000000 | 0.742263 | 2.483932 | 0.978808 | pfd_r3_tau8000_m1 | 1.000000 | 0.894581 | 0.737233 | 2.457937 | 31.352 |
| N179 | 0.908441 | n179_s5_r2_tau32000 | 0.977785 | 0.144646 | 0.721843 | 2.371284 | 0.979041 | n179_s9_r4_tau32000 | 0.673025 | 0.902787 | 0.721980 | 2.378653 | 0.662 |

#### 10.6.2 light_mid

| Algorithm | Best AUC | Best AUC Tag | F1@Best-AUC | Thr@Best-AUC | MESR@Best-AUC | AOCC@Best-AUC | Best F1 | Best F1 Tag | Thr@Best-F1 | AUC@Best-F1 | MESR@Best-F1 | AOCC@Best-F1 | Runtime(s) |
|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| BAF | 0.881811 | baf_r1 | 0.948786 | 16000.000000 | 0.723033 | 2.306037 | 0.948786 | baf_r1 | 16000.000000 | 0.881811 | 0.723033 | 2.306037 | 22.991 |
| STCF | 0.936965 | stcf_r1 | 0.961349 | 64000.000000 | 0.730580 | 2.376526 | 0.970314 | stcf_r2 | 16000.000000 | 0.933485 | 0.732522 | 2.415924 | 63.392 |
| EBF | 0.937867 | ebf_r2_tau16000 | 0.970612 | 1.000000 | 0.742812 | 2.461226 | 0.971160 | ebf_r2_tau32000 | 1.500000 | 0.922190 | 0.731829 | 2.436520 | 14.687 |
| N149 | 0.936847 | n149_r2_tau32000_light_mid | 0.958057 | 0.747070 | 0.720106 | 2.356677 | 0.958849 | n149_r3_tau32000_light_mid | 0.870674 | 0.936600 | 0.710368 | 2.272940 | 25.970 |
| KNOISE | 0.591022 | knoise_tau2000 | 0.939629 | 0.000000 | 0.681278 | 1.981033 | 0.939629 | knoise_tau1000 | 0.000000 | 0.590283 | 0.681278 | 1.981033 | 86.387 |
| EVFLOW | 0.828349 | evflow_r2_tau8000 | 0.911603 | 64.000000 |  |  | 0.941888 | evflow_r2_tau32000 | 64.000000 | 0.763634 |  |  | 677.403 |
| YNOISE | 0.935314 | ynoise_r2_tau8000 | 0.957535 | 1.000000 | 0.725979 | 2.306716 | 0.962252 | ynoise_r2_tau16000 | 2.000000 | 0.927591 | 0.732927 | 2.413554 | 296.672 |
| TS | 0.802150 | ts_r2_decay32000 | 0.943836 | 0.300000 | 0.784064 | 2.581381 | 0.943836 | ts_r2_decay32000 | 0.300000 | 0.802150 | 0.784064 | 2.581381 | 8.741 |
| MLPF | 0.439639 | mlpf_tau32000 | 0.916399 | 0.100000 | 0.681240 | 1.981077 | 0.916399 | mlpf_tau32000 | 0.100000 | 0.439639 | 0.681240 | 1.981077 | 448.517 |
| PFD | 0.900137 | pfd_r3_tau8000_m2 | 0.958203 | 1.000000 | 0.731202 | 2.368465 | 0.958203 | pfd_r3_tau8000_m2 | 1.000000 | 0.900137 | 0.731202 | 2.368465 | 31.951 |
| N179 | 0.910494 | n179_s5_r2_tau32000 | 0.949475 | 1.023406 | 0.726019 | 2.421034 | 0.950730 | n179_s9_r4_tau32000 | 1.955368 | 0.905237 | 0.722295 | 2.384831 | 0.797 |

#### 10.6.3 mid

| Algorithm | Best AUC | Best AUC Tag | F1@Best-AUC | Thr@Best-AUC | MESR@Best-AUC | AOCC@Best-AUC | Best F1 | Best F1 Tag | Thr@Best-F1 | AUC@Best-F1 | MESR@Best-F1 | AOCC@Best-F1 | Runtime(s) |
|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| BAF | 0.868373 | baf_r1 | 0.916932 | 8000.000000 | 0.735027 | 2.221840 | 0.916932 | baf_r1 | 8000.000000 | 0.868373 | 0.735027 | 2.221840 | 23.325 |
| STCF | 0.935924 | stcf_r1 | 0.946620 | 32000.000000 | 0.742287 | 2.392254 | 0.954822 | stcf_r2 | 8000.000000 | 0.930659 | 0.751749 | 2.389156 | 65.307 |
| EBF | 0.937166 | ebf_r2_tau16000 | 0.956357 | 1.500000 | 0.749396 | 2.433484 | 0.956357 | ebf_r2_tau16000 | 1.500000 | 0.937166 | 0.749396 | 2.433484 | 14.778 |
| N149 | 0.938152 | n149_r2_tau32000_mid | 0.937040 | 1.071131 | 0.718285 | 2.311529 | 0.937725 | n149_r3_tau32000_mid | 1.153651 | 0.938016 | 0.700190 | 2.158709 | 25.899 |
| KNOISE | 0.584814 | knoise_tau2000 | 0.899398 | 0.000000 | 0.657992 | 1.796615 | 0.899398 | knoise_tau1000 | 0.000000 | 0.584477 | 0.657992 | 1.796615 | 87.263 |
| EVFLOW | 0.826195 | evflow_r2_tau8000 | 0.898333 | 64.000000 |  |  | 0.915064 | evflow_r2_tau16000 | 64.000000 | 0.793410 |  |  | 682.711 |
| YNOISE | 0.933513 | ynoise_r2_tau8000 | 0.940422 | 2.000000 | 0.751525 | 2.384611 | 0.942359 | ynoise_r2_tau16000 | 3.000000 | 0.926730 | 0.743120 | 2.426991 | 293.449 |
| TS | 0.777210 | ts_r2_decay32000 | 0.904080 | 0.500000 | 0.851542 | 2.388471 | 0.904080 | ts_r2_decay32000 | 0.500000 | 0.777210 | 0.851542 | 2.388471 | 8.731 |
| MLPF | 0.498857 | mlpf_tau256000 | 0.853742 | 0.100000 | 0.666617 | 1.777021 | 0.861026 | mlpf_tau512000 | 0.200000 | 0.415801 | 0.673816 | 1.883022 | 450.010 |
| PFD | 0.897118 | pfd_r3_tau8000_m3 | 0.936814 | 1.000000 | 0.741823 | 2.358040 | 0.936814 | pfd_r3_tau8000_m3 | 1.000000 | 0.897118 | 0.741823 | 2.358040 | 33.339 |
| N179 | 0.911717 | n179_s5_r2_tau32000 | 0.928930 | 1.623486 | 0.736428 | 2.445534 | 0.929641 | n179_s7_r3_tau32000 | 2.435610 | 0.909197 | 0.732736 | 2.433042 | 0.968 |

#### 10.6.4 N179（并入 Driving 对比）

| Level | N179 吞吐事件率(eps) | 数据集事件率(eps) | 裕量(吞吐/数据) |
|---|---:|---:|---:|
| light | 6,236,813 | 1,375,531 | 4.53x |
| light_mid | 5,652,045 | 1,501,901 | 3.76x |
| mid | 5,072,268 | 1,637,169 | 3.10x |

结论：Driving 三档下 N179 吞吐均高于数据事件率，满足实时处理要求。

ED24 的 N179 已并入 `10.1/10.2/10.3` 主对比表，不再单独列出。

#### 10.6.5 N179 固定 `s/tau` 扫 K（本轮新增）

目的：验证“固定 `s,tau` 先扫 K”是否可提升跨数据集稳定性。  
固定参数：`s=9, tau=128000us`，仅扫 `k_sfrac × k_mix`。

K 网格：
- `k_sfrac ∈ {0.67, 0.75, 0.80, 0.90}`
- `k_mix ∈ {0.08, 0.10, 0.125, 0.16}`
- `beta_init`: 使用默认（`bdef`）

代表集扫频结果（best-AUC）：

| Dataset | Split | Best Tag | AUC | F1 |
|---|---|---|---:|---:|
| ED24 | myPedestrain_06 light_mid | `n179_s9_r4_tau128000_ks670_km160_bdef` | 0.945199 | 0.900324 |
| ED24 | myBicycle_02 light_mid | `n179_s9_r4_tau128000_ks670_km160_bdef` | 0.980988 | 0.946166 |
| DVSCLEAN | MAH00448 ratio100 | `n179_s9_r4_tau128000_ks670_km80_bdef` | 0.474982 | 0.666667 |
| LED | scene_100 | `n179_s9_r4_tau128000_ks900_km80_bdef` | 0.556282 | 0.978401 |
| LED | scene_1004 | `n179_s9_r4_tau128000_ks900_km80_bdef` | 0.511859 | 0.945997 |

按代表集均值选定（用于最后 Driving）：`k_sfrac=0.90, k_mix=0.16`。

Driving（最后执行，固定 `k_sfrac=0.90,k_mix=0.16`）：

| Level | Best Tag | AUC | F1 | F1 Threshold | MESR@Best-F1 | AOCC@Best-F1 |
|---|---|---:|---:|---:|---:|---:|
| light | `n179_s9_r4_tau128000_ks900_km160_bdef` | 0.870823 | 0.978081 | 2.070620 | 0.719056 | 2.348723 |
| light_mid | `n179_s9_r4_tau128000_ks900_km160_bdef` | 0.871818 | 0.946542 | 5.733807 | 0.719141 | 2.349198 |
| mid | `n179_s9_r4_tau128000_ks900_km160_bdef` | 0.871914 | 0.923360 | 7.716200 | 0.721507 | 2.318535 |

说明：
1. 该实验是“固定 `s,tau` 的 K 扫频实验”，用于验证 K 敏感性，不替代主对比表（主表仍采用各算法原 sweep 口径）。
2. 脚本已支持传参：`--k-sfrac-list --k-mix-list --beta-init-list`，后续可继续扩展 `beta_init` 扫频。

结论更新（N149 vs EBF，Driving 200k）：
1. 旧版“N149 低于 EBF”的主要原因是 N149 当时只扫了 `r=3/4/5`，结构参数覆盖不完整。
2. 本轮加入 `r=2` 并做 `sigma` 扫描后，N149 的 `best-AUC` 变为 `0.932978/0.936847/0.938152`（light/light_mid/mid），已与 EBF 接近，且在 `mid` 上略高于当前 EBF 汇总值。
3. 对 driving 这类数据，优先补全结构参数覆盖（`r`、`sigma`），再细化阈值，收益更稳定。

## 12. 运行时统计（ED24）

统一口径（避免“分钟/秒混用”）：
1. 每个算法读取各自 `runtime_*.csv`；
2. 对 `light/mid/heavy` 每个 level 只取“最新一条”记录；
3. 统计 `avg_sec_per_level`（每档平均秒数）和 `sum_sec_3levels`（三档总秒数）。

统一汇总文件：
- `data/ED24/myPedestrain_06/runtime_unified_ed24.csv`

生成命令：
```powershell
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/summarize_runtime_ed24.py
```

| Algorithm | light(s) | mid(s) | heavy(s) | avg_sec_per_level | sum_sec_3levels |
|---|---:|---:|---:|---:|---:|
| EBF | 14.035 | 17.324 | 20.545 | 17.301 | 51.904 |
| N149 | 55.374 | 61.798 | 67.643 | 61.605 | 184.815 |
| TS | 16.654 | 24.341 | 30.248 | 23.748 | 71.243 |
| PFD | 34.418 | 72.914 | 121.923 | 76.418 | 229.255 |
| BAF | 55.613 | 172.831 | 265.363 | 164.602 | 493.807 |
| STCF |  |  |  |  |  |
| KNOISE | 85.460 | 238.140 | 376.325 | 233.308 | 699.925 |
| EVFLOW | 174.066 | 548.679 | 1239.166 | 653.970 | 1961.911 |
| YNOISE | 275.119 | 777.092 | 1234.050 | 762.087 | 2286.261 |
| MLPF | 428.342 | 1239.780 | 578.035 | 748.719 | 2246.157 |

注：
- `STCF` 当前缺 `runtime_stcf.csv`，因此留空；补跑 `run_slomo_stcf.ps1` 后可自动纳入统一表。
- 在这个统一口径下，`PFD` 不是最慢算法，明显慢于 PFD 的是 `EVFLOW/YNOISE/MLPF`。

后续新增实验必须同步记录 runtime，推荐写入以下脚本的 runtime CSV：
- `scripts/ED24_alg_evalu/run_slomo_alg.ps1`
- `scripts/driving_alg_evalu/run_driving_alg.ps1`
- `scripts/ED24_alg_evalu/run_slomo_baf.ps1`
- `scripts/ED24_alg_evalu/run_slomo_stcf.ps1`
- `scripts/ED24_alg_evalu/run_slomo_ebf.ps1`
- `scripts/ED24_alg_evalu/run_n149_labelscore_grid.py`

## 13. 两个 EBF 扫频脚本的区别（明确说明）

- `scripts/ED24_alg_evalu/run_slomo_ebf.ps1`
1. 用途：横向对比优先（和 BAF/STCF/N149 放在同一标准下比较）。
2. 扫频方式：`radius × tau × threshold` 常规网格。
3. 结果文件：`roc_ebf_{light,mid,heavy}.csv/.png`。

- `scripts/ED24_alg_evalu/run_slomo_ebf_paper_s_tau.ps1`
1. 用途：论文复现和消融分析。
2. 扫频方式：两步法。
   - Step A：固定 `tau=64ms`，扫描 `s`（窗口直径，脚本内部换算为 `radius`）。
   - Step B：固定 `s=5`，扫描 `tau`。
3. 结果文件：`roc_ebf_*_paper_*`。

建议：
- 做跨算法排名：优先用 `run_slomo_ebf.ps1`。
- 做机理说明/论文图：优先用 `run_slomo_ebf_paper_s_tau.ps1`。

横向对比绘图降采样规则（2026-04-23 起）：
- `EBF`：每个 `r` 仅绘制 AUC 最优 `3` 条曲线。
- `EVFLOW`：每个 `r` 仅绘制 AUC 最优 `3` 条曲线。
- `YNOISE`：每个 `r` 仅绘制 AUC 最优 `3` 条曲线。
- `TS`：每个 `r` 仅绘制 AUC 最优 `3` 条曲线。
- `PFD`：每个 `r` 仅绘制 AUC 最优 `3` 条曲线。
- `MLPF`：全局仅绘制 AUC 最优 `4` 条（按 `tau` 标签筛选）。

### 14.1 连续扫频（Dense）策略

`run_slomo_alg.ps1` 与 `run_driving_alg.ps1` 新增：
- `-SweepProfile coarse|dense`

说明：
- `coarse`：保持原有稀疏网格（速度快，适合首轮横向对比）。
- `dense`：更细阈值网格（近连续扫频，适合最终参数定点）。

建议：
- `EVFLOW`：优先使用 `dense`，原因是其 ROC 对阈值/时间窗更敏感，稀疏阈值容易导致曲线点过于集中。
- `TS`、`YNOISE`、`MLPF`、`PFD`：当你发现 best 点贴近阈值边界、或曲线锯齿明显时，再切 `dense`。
- `KNOISE`：通常 `coarse` 已足够；只有在论文图需要精细局部曲线时再用 `dense`。

## 14. 参数调优位置（TUNE_HERE）

注：`run_slomo_alg.ps1` 与 `run_driving_alg.ps1` 可先用 `-SweepProfile coarse` 快速定位，再用 `-SweepProfile dense` 精细搜索。

- `scripts/ED24_alg_evalu/run_slomo_baf.ps1`：`$RADIUS_LIST`, `$TAU_LIST`
- `scripts/ED24_alg_evalu/run_slomo_stcf.ps1`：`$RADIUS_LIST`, `$TAU_LIST`
- `scripts/ED24_alg_evalu/run_slomo_ebf.ps1`：`$EBF_RADIUS_LIST`, `$EBF_TAU_LIST`, `$EBF_THR_LIST`
- `scripts/ED24_alg_evalu/run_slomo_alg.ps1`：每个算法块中的 `thr/r/tau` 网格
- `scripts/ED24_alg_evalu/run_slomo_n149.ps1`：`--radius-list`, `--tau-us-list`
- `scripts/driving_alg_evalu/run_driving_alg.ps1`：同 ED24 的参数区


## D. DVSCLEAN 数据集

## 15. 一键实验脚本（新增）

### 4.1 ED24

总入口（按算法参数自动扫频并输出 ROC）：
- `scripts/ED24_alg_evalu/run_slomo_alg.ps1`

单算法入口：
- `scripts/ED24_alg_evalu/run_slomo_knoise.ps1`
- `scripts/ED24_alg_evalu/run_slomo_evflow.ps1`
- `scripts/ED24_alg_evalu/run_slomo_ynoise.ps1`
- `scripts/ED24_alg_evalu/run_slomo_ts.ps1`
- `scripts/ED24_alg_evalu/run_slomo_mlpf.ps1`
- `scripts/ED24_alg_evalu/run_slomo_pfd.ps1`

运行示例：
```powershell
cd D:\hjx_workspace\scientific_re                                           serach\projects\myEVS
powershell -ExecutionPolicy Bypass -File .\scripts\ED24_alg_evalu\run_slomo_knoise.ps1
```

### 4.2 Driving

总入口：
- `scripts/driving_alg_evalu/run_driving_alg.ps1`
- `scripts/driving_alg_evalu/run_driving_alg_paper.ps1`（论文口径数据入口）

单算法入口：
- `scripts/driving_alg_evalu/run_driving_knoise.ps1`
- `scripts/driving_alg_evalu/run_driving_evflow.ps1`
- `scripts/driving_alg_evalu/run_driving_ynoise.ps1`
- `scripts/driving_alg_evalu/run_driving_ts.ps1`
- `scripts/driving_alg_evalu/run_driving_mlpf.ps1`
- `scripts/driving_alg_evalu/run_driving_pfd.ps1`
- `scripts/driving_alg_evalu/run_driving_n149.ps1`（N149 独立入口，调用 labelscore 网格脚本）

运行示例：
```powershell
cd D:\hjx_workspace\scientific_reserach\projects\myEVS
powershell -ExecutionPolicy Bypass -File .\scripts\driving_alg_evalu\run_driving_evflow.ps1
```

Driving 脚本会在每个噪声级目录自动查找：
- `*signal_only*.npy` 或 `*clean*.npy` 作为 clean
- 其余 `.npy` 作为 noisy

#### 4.2.1 Driving 论文口径数据生成（1/3/5 Hz）

生成脚本：
- `scripts/driving_alg_evalu/generate_driving_paper_noise.py`

输入（paper clean）：
- `D:\hjx_workspace\scientific_reserach\dataset\DND21\mydriving_paper\v2e-dvs-events.txt`

输出目录：
- `D:\hjx_workspace\scientific_reserach\dataset\DND21\mydriving_paper\driving_noise_light_paper_withlabel`
- `D:\hjx_workspace\scientific_reserach\dataset\DND21\mydriving_paper\driving_noise_light_mid_paper_withlabel`
- `D:\hjx_workspace\scientific_reserach\dataset\DND21\mydriving_paper\driving_noise_mid_paper_withlabel`

生成命令：
```powershell
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/driving_alg_evalu/generate_driving_paper_noise.py `
  --clean-txt "D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_paper/v2e-dvs-events.txt" `
  --out-root "D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_paper" `
  --width 346 --height 260 --sigma-decades 0.5 --seed 12345
```

直接跑论文口径 Driving（默认仍输出到 `data/DND21/mydriving/{ALG}`）：
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\driving_alg_evalu\run_driving_alg_paper.ps1 -Algorithm ebf -MaxEvents 200000
powershell -ExecutionPolicy Bypass -File .\scripts\driving_alg_evalu\run_driving_alg_paper.ps1 -Algorithm all -MaxEvents 200000
```

口径说明（重要）：
1. 当前生成方式不是“直接调用 JAER 可执行流程”，而是“参考 JAER `NoiseTesterFilter` 思路在 Python 中实现”：
   - clean 基线来自 `mydriving_paper/v2e-dvs-events.txt`
   - 噪声按像素独立 Poisson 过程生成
   - 像素噪声率引入 log-normal 空间不均匀性（FPN）
2. `sigma-decades=0.5` 的含义：
   - 先在 \(\log_{10}\) 域采样像素噪声率扰动，标准差为 `0.5`
   - 直观上表示像素噪声率存在约“半个 decade 量级”的离散
3. `FPN`（Fixed Pattern Noise）含义：
   - 传感器不同像素的固有噪声率不同，且这种差异在时间上相对固定
   - 代码中体现为每个像素固定一个噪声率 \(r_{x,y}\)，再在时域按 Poisson 采样
4. 参数选择依据：
   - `1/3/5 Hz`：对齐 EBF/DND21 常用 driving 噪声档位
   - `sigma-decades=0.5`：对齐 JAER `NoiseTesterFilter` 与 DND21 文献中常见 FPN 设定
5. 本版噪声类型：
   - 当前仅添加 **shot noise**
   - 未添加 leak noise、leak jitter、记录噪声回放等更完整 JAER 分支
6. 与“严格 JAER 一致”的边界：
   - 本版属于“统计口径对齐”的复现，不是二进制级别同实现
   - 若要完全一致，需走 JAER 原链路导出（含其内部事件调度与可选 leak/noise-recording 分支）

JAER 原链路导出数据转换（2026-04-29 补充）：

生成脚本：
- `scripts/driving_alg_evalu/convert_jaer_driving_to_npy.py`

输入：
- clean: `D:\hjx_workspace\scientific_reserach\dataset\DND21\mydriving_paper\driving.aedat`
- noisy: `D:\hjx_workspace\scientific_reserach\dataset\DND21\mydriving_jaer\driving_noise_{light,light_mid,mid}\driving_jaer_shot_{1,3,5}hz.aedat`

转换命令：
```powershell
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/driving_alg_evalu/convert_jaer_driving_to_npy.py `
  --clean-aedat "D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_paper/driving.aedat" `
  --jaer-root "D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_jaer" `
  --out-root "D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_jaer" `
  --width 346 --height 260 --tick-ns 12.5 --overwrite
```

转换质量检查：

| Level | Target Hz/pixel | Noisy events | Matched signal | Estimated noise | Estimated Hz/pixel |
|---|---:|---:|---:|---:|---:|
| light | 1 | 4,285,026 | 3,748,579 | 536,447 | 0.996 |
| light_mid | 3 | 5,357,184 | 3,748,579 | 1,608,605 | 2.988 |
| mid | 5 | 6,431,569 | 3,748,578 | 2,682,991 | 4.983 |

说明：
1. `matched signal` 通过 `(t,x,y,p)` 精确匹配 clean AEDAT 恢复，约覆盖 clean 的 `99.81%`。
2. `Estimated Hz/pixel` 与目标 `1/3/5Hz` 基本一致，说明 JAER 导出的加噪强度是正确的。
3. 少量 unmatched clean 事件来自 JAER filtered logging 起止边界或重构细节，对整体 AUC 影响很小。

ED24 外部 Driving 加噪数据转换（2026-04-29 补充）：

源目录：
- `D:\hjx_workspace\scientific_reserach\dataset\DND21\mydriving_ED24`

每个噪声档位目录包含的文件含义：

| 文件 | 含义 | 是否用于本次 EBF |
|---|---|---|
| `1hz.txt` ... `7hz.txt` | v2e 从黑色视频生成的纯噪声事件，时间戳为秒，无 label | 否，仅用于理解噪声来源 |
| `mix_result.txt` | 另一份混合结果，带 epoch-like 微秒时间戳，`label=0` 数量恒定约 53 万；不像当前 DND21 driving 主序列 | 否 |
| `driving_mix_result.txt` | Driving 主序列混合结果，`label=0` 数量恒定约 273 万，`label=1` 随噪声档位增加 | 是 |
| `v2e-args.txt` | v2e 生成参数记录 | 仅作为元数据参考 |
| `*.avi` | 可视化预览文件，部分档位存在 | 否 |

标签口径：
- ED24 外部文件：`label=0` 表示 signal，`label=1` 表示 noise。
- myEVS 统一口径：`label=1` 表示 signal，`label=0` 表示 noise。
- 因此转换时必须翻转 label，否则 ROC/F1 会完全错误。

转换脚本：
- `scripts/driving_alg_evalu/convert_ed24_driving_to_npy.py`

转换命令：
```powershell
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/driving_alg_evalu/convert_ed24_driving_to_npy.py --overwrite
```

输出目录：
- `D:\hjx_workspace\scientific_reserach\dataset\DND21\mydriving_ED24\driving_noise_{1hz...9hz}_ed24_withlabel`

转换质量检查：

| Level | Events | Signal | Noise | Duration (s) | Estimated Hz/pixel |
|---|---:|---:|---:|---:|---:|
| 1hz | 3,081,599 | 2,732,313 | 349,286 | 5.976212 | 0.650 |
| 2hz | 3,429,942 | 2,732,313 | 697,629 | 5.976212 | 1.298 |
| 3hz | 3,777,023 | 2,732,313 | 1,044,710 | 5.976212 | 1.943 |
| 4hz | 4,126,122 | 2,732,313 | 1,393,809 | 5.976212 | 2.593 |
| 5hz | 4,478,144 | 2,732,313 | 1,745,831 | 5.976212 | 3.247 |
| 6hz | 4,822,218 | 2,732,313 | 2,089,905 | 5.976212 | 3.887 |
| 7hz | 5,169,768 | 2,732,313 | 2,437,455 | 5.976212 | 4.534 |
| 8hz | 5,515,979 | 2,732,313 | 2,783,666 | 5.976212 | 5.178 |
| 9hz | 5,867,956 | 2,732,313 | 3,135,643 | 5.976212 | 5.832 |

注意：
1. 这批数据的目录名是 `1...9hz`，但按事件数和时长估算的实际噪声率约为标称值的 `0.65` 倍。
2. 因此它不适合直接当作论文 `1/3/5 Hz/pixel` 绝对对齐数据，但适合验证“外部生成 Driving 加噪数据”下 EBF 的相对表现。

ED24 外部 Driving 数据上的 EBF 结果：

运行设置：
- EBF 参数：`radius=2`（论文 `s=5`）、`\(\tau=32000us\)`。
- 时间戳单位：`--tick-ns 1000`，因为 `driving_mix_result.txt` 时间戳为微秒整数。
- 阈值：`0.0:0.1:15.0`。
- 为避免 Python EBF 对每个阈值重复运行，本次使用 myEVS Numba EBF score 内核一次性计算 score，再按阈值生成 ROC；计算口径等价于 `score > threshold`。

结果文件：
- `data/DND21/mydriving_ED24/EBF/roc_ebf_{1hz...9hz}_ed24_paperfix.csv`
- `data/DND21/mydriving_ED24/EBF/ebf_ed24_driving_1to9hz_summary.csv`
- `data/DND21/mydriving_ED24/EBF/ebf_ed24_driving_allhz_summary.csv`

| Level | Estimated Hz/pixel | AUC step=0.1 | Exact score AUC | Best-F1 threshold | TPR | FPR | Precision | Best F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1hz | 0.650 | 0.950013 | 0.950048 | 0.4 | 0.973967 | 0.295572 | 0.962654 | 0.968278 |
| 2hz | 1.298 | 0.948619 | 0.948665 | 0.9 | 0.960809 | 0.226274 | 0.943281 | 0.951964 |
| 3hz | 1.943 | 0.946503 | 0.946547 | 1.4 | 0.940665 | 0.174585 | 0.933738 | 0.937189 |
| 4hz | 2.593 | 0.944914 | 0.944962 | 1.8 | 0.926896 | 0.148886 | 0.924266 | 0.925579 |
| 5hz | 3.247 | 0.943417 | 0.943472 | 2.1 | 0.915241 | 0.138341 | 0.911926 | 0.913581 |
| 6hz | 3.887 | 0.942176 | 0.942268 | 2.3 | 0.910690 | 0.135378 | 0.897905 | 0.904252 |
| 7hz | 4.534 | 0.940643 | 0.940747 | 2.7 | 0.895203 | 0.118098 | 0.894705 | 0.894954 |
| 8hz | 5.178 | 0.939436 | 0.939534 | 2.8 | 0.894398 | 0.122407 | 0.877630 | 0.885935 |
| 9hz | 5.832 | 0.938252 | 0.938358 | 3.2 | 0.878133 | 0.106354 | 0.877969 | 0.878051 |

初步结论：
1. 在这批 ED24 外部 Driving 加噪数据上，EBF AUC 随实际噪声率增加单调下降，趋势正常。
2. `1hz/3hz/5hz` 三档的 AUC 接近或高于论文表格中的 `1/3/5 Hz` 数值，但该数据实际噪声率低于目录名标称值，因此不能直接说明已经复现论文结果。
3. 该结果说明：之前 JAER 数据与论文差距不太可能来自 EBF 公式或 AUC 积分本身，更可能来自数据生成/标签口径/事件区间差异。

ED24 外部 Driving 数据上的 N149 结果：

运行设置：
- 数据：同上 `driving_noise_{1hz...9hz}_ed24_withlabel`。
- N149 网格：`radius={2,3,4,5}`，`\(\tau={16,32,64,128,256,512}ms\)`。
- ROC：使用 N149 连续 score 构造 ROC，并统计每个 `(radius,\tau)` 的 AUC；下表的 `N149 AUC` 是每档噪声下所有网格中的最佳 AUC。

结果文件：
- `data/DND21/mydriving_ED24/N149/roc_n149_{1hz...9hz}_ed24.csv`
- `data/DND21/mydriving_ED24/N149/n149_ed24_driving_1to9hz_summary.csv`
- `data/DND21/mydriving_ED24/N149/n149_ed24_driving_allhz_summary.csv`
- `data/DND21/mydriving_ED24/n149_vs_ebf_ed24_driving_1to9hz_summary.csv`

| Level | Estimated Hz/pixel | EBF AUC | N149 AUC | N149-EBF AUC | EBF Best F1 | N149 Best F1 | N149 best-AUC `(r,tau)` |
|---|---:|---:|---:|---:|---:|---:|---|
| 1hz | 0.650 | 0.950013 | 0.940363 | -0.009650 | 0.968278 | 0.966613 | `(2,32ms)` |
| 2hz | 1.298 | 0.948619 | 0.940156 | -0.008463 | 0.951964 | 0.946244 | `(2,32ms)` |
| 3hz | 1.943 | 0.946503 | 0.940568 | -0.005935 | 0.937189 | 0.931406 | `(2,32ms)` |
| 4hz | 2.593 | 0.944914 | 0.941424 | -0.003490 | 0.925579 | 0.919301 | `(2,32ms)` |
| 5hz | 3.247 | 0.943417 | 0.942468 | -0.000950 | 0.913581 | 0.909426 | `(2,32ms)` |
| 6hz | 3.887 | 0.942176 | 0.940074 | -0.002103 | 0.904252 | 0.896588 | `(2,32ms)` |
| 7hz | 4.534 | 0.940643 | 0.940966 | +0.000323 | 0.894954 | 0.889008 | `(3,32ms)` |
| 8hz | 5.178 | 0.939436 | 0.943954 | +0.004518 | 0.885935 | 0.884242 | `(5,16ms)` |
| 9hz | 5.832 | 0.938252 | 0.939164 | +0.000911 | 0.878051 | 0.871728 | `(3,32ms)` |

N149 对比结论：
1. 低噪声到中等噪声（实际约 `0.65~3.89 Hz/pixel`）下，EBF 的 AUC 和 Best-F1 均优于 N149。
2. 高噪声端（实际约 `4.53~5.83 Hz/pixel`）下，N149 的最佳 AUC 开始接近或略高于 EBF，尤其 `8hz` 档高出约 `0.0045`。
3. 但 N149 的 Best-F1 仍低于 EBF，说明 N149 在高噪声端可能改善整体排序能力（AUC），但最佳二分类工作点的 precision/recall 折中仍不如 EBF 稳定。
4. N149 最佳 AUC 多数落在 `tau=32ms`，说明在这批数据上时间尺度仍与 EBF 论文参数接近；高噪声 `8hz` 档偏向更大空间邻域 `r=5` 和更短时间窗 `16ms`。

### 4.3 DVSCLEAN（EBF vs N149，2026-04-29）

源目录：
- `D:\hjx_workspace\scientific_reserach\dataset\DVSCLEAN`

文件结构：
- `MAH00444/MAH00444_50.hdf5`, `MAH00444/MAH00444_100.hdf5`
- `MAH00446/MAH00446_50.hdf5`, `MAH00446/MAH00446_100.hdf5`
- `MAH00447/MAH00447_50.hdf5`, `MAH00447/MAH00447_100.hdf5`

HDF5 字段检查：
- group: `events`
- datasets: `events/timestamp`, `events/x`, `events/y`, `events/polarity`, `events/label`
- `label=0` 为 signal，`label=1` 为 noise
- `_50` 表示 `noise/signal=0.5`
- `_100` 表示 `noise/signal=1.0`
- 坐标范围为 `x=0..1279`, `y=0..719`，因此分辨率按 `1280x720`
- `timestamp` 为秒，转换时保存为微秒整数；后续评估使用 `--tick-ns 1000`
- `polarity` 为 `[0,1]` 浮点，本工程按 `polarity>0.5 -> +1`，否则 `-1` 二值化

脚本目录：
- `scripts/DVSCLEAN_alg_evalu`

转换脚本：
- `scripts/DVSCLEAN_alg_evalu/convert_dvsclean_to_npy.py`

运行命令：
```powershell
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/DVSCLEAN_alg_evalu/convert_dvsclean_to_npy.py --overwrite
```

输出目录：
- `D:\hjx_workspace\scientific_reserach\dataset\DVSCLEAN\converted_npy\<scene>\ratio50`
- `D:\hjx_workspace\scientific_reserach\dataset\DVSCLEAN\converted_npy\<scene>\ratio100`

转换统计：

| Scene | Level | Events | Signal | Noise | Noise/Signal | Estimated Hz/pixel |
|---|---|---:|---:|---:|---:|---:|
| MAH00444 | ratio50 | 286,060 | 190,707 | 95,353 | 0.500 | 0.152845 |
| MAH00444 | ratio100 | 381,414 | 190,707 | 190,707 | 1.000 | 0.305691 |
| MAH00446 | ratio50 | 286,887 | 191,258 | 95,629 | 0.500 | 0.172038 |
| MAH00446 | ratio100 | 382,516 | 191,258 | 191,258 | 1.000 | 0.344075 |
| MAH00447 | ratio50 | 281,026 | 187,351 | 93,675 | 0.500 | 0.272703 |
| MAH00447 | ratio100 | 374,702 | 187,351 | 187,351 | 1.000 | 0.545408 |

评测脚本：
- `scripts/DVSCLEAN_alg_evalu/run_dvsclean_ebf_n149.py`

运行命令：
```powershell
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/DVSCLEAN_alg_evalu/run_dvsclean_ebf_n149.py
```

评测设置：
- EBF: `radius=2`, `tau=32000us`, threshold `0.0:0.1:15.0`
- N149: `radius={2,3,4,5}`, `tau={16,32,64,128,256,512}ms`
- N149 表中取每个文件的 best-AUC 配置，同时记录 best-F1

结果文件：
- `data/DVSCLEAN/dvsclean_ebf_n149_summary.csv`
- `data/DVSCLEAN/<scene>/EBF/roc_ebf_<scene>_<level>.csv`
- `data/DVSCLEAN/<scene>/N149/roc_n149_<scene>_<level>.csv`

结果表：

| Scene | Level | EBF AUC | N149 AUC | N149-EBF AUC | EBF F1 | N149 F1 | N149 best-AUC `(r,tau)` |
|---|---|---:|---:|---:|---:|---:|---|
| MAH00444 | ratio50 | 0.990730 | 0.998073 | +0.007343 | 0.980285 | 0.993490 | `(5,128ms)` |
| MAH00444 | ratio100 | 0.989676 | 0.997742 | +0.008066 | 0.971272 | 0.990971 | `(5,128ms)` |
| MAH00446 | ratio50 | 0.990507 | 0.997454 | +0.006946 | 0.980851 | 0.992721 | `(5,64ms)` |
| MAH00446 | ratio100 | 0.989384 | 0.996841 | +0.007457 | 0.973906 | 0.989567 | `(5,64ms)` |
| MAH00447 | ratio50 | 0.991312 | 0.996694 | +0.005382 | 0.981440 | 0.990933 | `(5,64ms)` |
| MAH00447 | ratio100 | 0.989927 | 0.995988 | +0.006061 | 0.973551 | 0.987964 | `(5,64ms)` |

均值：
- EBF AUC: `0.990256`
- N149 AUC: `0.997132`
- N149 平均 AUC 提升: `+0.006876`
- EBF Best-F1: `0.976884`
- N149 Best-F1: `0.990941`
- N149 平均 F1 提升: `+0.014057`

DVSCLEAN 结论：
1. DVSCLEAN 的标签明确，且 `_50/_100` 噪声比例与文件命名完全一致，是适合做 EBF/N149 泛化验证的数据集。
2. 在 DVSCLEAN 上，N149 对 EBF 的提升非常稳定：6 个文件的 AUC 和 Best-F1 全部提升。
3. N149 最佳 AUC 都选择 `r=5`，时间窗集中在 `64ms/128ms`，说明该数据集更偏向“较大空间支撑 + 较长时间积分”的结构噪声场景。
4. 这与 ED24 上 N149 优于 EBF、Driving 低中噪声下 N149 不如 EBF 的结论并不矛盾：N149 对带标签混合噪声和结构性噪声更有效，但对低中强度 shot-noise Driving 不一定稳健。

### 4.4 EBF + N149（按 EBF 论文最优参数重跑，paper-driving）

论文参数依据（Guo 2025）：
1. 在 Driving 上，作者给出的最优组合为 `s=5`、`\(\tau=32ms\)`（其中 `s=5` 对应 `radius=2`）。
2. 论文 Table II 报告 EBF AUC（Driving）：`1Hz=0.948`, `3Hz=0.942`, `5Hz=0.936`。

本工程重跑设置：
1. 数据：`mydriving_paper` 三档（`light/light_mid/mid`）；
2. EBF：固定 `radius=2`, `tau=32000us`，扫阈值（`min-neighbors`）得到 ROC 与 AUC；
3. N149：固定 `radius=2`, `tau=32000us`, `sigma=1.5`，阈值直接使用“同档 EBF 的 best-F1 阈值”。

执行命令（EBF）：
```powershell
D:/software/Anaconda_envs/envs/myEVS/python.exe -m myevs.cli roc --clean "D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_paper/driving_noise_light_paper_withlabel/driving_noise_light_signal_only.npy" --noisy "D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_paper/driving_noise_light_paper_withlabel/driving_noise_light_labeled.npy" --assume npy --width 346 --height 260 --tick-ns 1000 --engine python --method ebf --radius-px 2 --time-us 32000 --param min-neighbors --values "0,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.5,3,3.5,4,5,6,8" --match-us 0 --match-bin-radius 0 --tag "ebf_paper_r2_tau32000_light" --out-csv data/DND21/mydriving_paper_eval/EBF/roc_ebf_light_paperfix.csv
```

结果文件：
1. `data/DND21/mydriving_paper_eval/EBF/roc_ebf_{light,light_mid,mid}_paperfix.csv`
2. `data/DND21/mydriving_paper_eval/ebf_n149_fixed_by_ebf_threshold.csv`

EBF AUC 与论文对比：

| Level | Noise (Hz/pixel) | Paper EBF AUC | Python-noise EBF AUC | JAER-noise EBF AUC |
|---|---:|---:|---:|---:|
| light | 1 | 0.948 | 0.926983 | 0.928231 |
| light_mid | 3 | 0.942 | 0.921316 | 0.923244 |
| mid | 5 | 0.936 | 0.916600 | 0.919252 |

JAER-noise EBF 运行设置：
```powershell
# values 为 0.0 到 15.0，步长 0.1；固定 s=5(radius=2), tau=32ms
D:/software/Anaconda_envs/envs/myEVS/python.exe -m myevs.cli roc `
  --clean "D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_jaer/driving_noise_light_jaer_withlabel/driving_noise_light_signal_only.npy" `
  --noisy "D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving_jaer/driving_noise_light_jaer_withlabel/driving_noise_light_labeled.npy" `
  --assume npy --width 346 --height 260 --tick-ns 12.5 `
  --method ebf --radius-px 2 --time-us 32000 `
  --param min-neighbors --values "0.0,0.1,...,15.0" `
  --tag ebf_r2_tau32000_jaer `
  --roc-convention paper --match-us 0 --match-bin-radius 0 `
  --out-csv data/DND21/mydriving_jaer/EBF/roc_ebf_light_jaer_paperfix.csv --progress
```

JAER-noise best-F1 点：

| Level | AUC | Best-F1 threshold | TPR | FPR | Precision | Best F1 |
|---|---:|---:|---:|---:|---:|---:|
| light | 0.928231 | 0.6 | 0.971682 | 0.373343 | 0.947881 | 0.959634 |
| light_mid | 0.923244 | 1.7 | 0.929725 | 0.235392 | 0.902001 | 0.915653 |
| mid | 0.919252 | 2.4 | 0.902238 | 0.199822 | 0.863174 | 0.882274 |

EBF ROC/AUC 统计口径排查（2026-04-29）：

1. 官方 EBF 代码最终也是把每个事件的 `sumfeature` 当作连续 score，再用 `sklearn.metrics.roc_auc_score(y_true, y_score)` 计算 AUC；因此理论上不应强依赖手工阈值步长。
2. 官方 EBF score 构成是：同极性邻域事件的线性时间核累加，中心像素置零，不包含额外空间高斯核。

$$
s_i=\sum_{j\in\mathcal{N}_r(i),\,j\ne i}
\mathbf{1}(p_j=p_i)\cdot
\max\left(0,1-\frac{|t_i-t_j|}{\tau}\right)
$$

3. 当前 myEVS 的 `method=ebf` 与上述公式一致：按事件流顺序维护每个像素最近时间戳和极性；对同极性邻居累加线性时间核；自身事件不参与当前 score；score 计算后再更新自身状态。
4. DND21/EBF 辅助脚本中也存在从 CSV 的 `(FPR,TPR)` 点排序后用梯形积分 `auc(fpr,tpr)` 的路径。因此本次同时比较了三种口径：
   - `step=0.1`：原始阈值网格。
   - `step=0.01`：更细阈值网格，模拟“连续扫频”。
   - `exact score AUC`：不手动指定阈值，按所有不同 score 的排序直接构造 ROC，等价于 `roc_auc_score` 口径。

| Level | Paper AUC | step=0.1 AUC | step=0.01 AUC | Exact score AUC | step 0.01 gain | Paper - Exact |
|---|---:|---:|---:|---:|---:|---:|
| light | 0.948000 | 0.928231 | 0.928250 | 0.928296 | +0.000020 | 0.019704 |
| light_mid | 0.942000 | 0.923244 | 0.923270 | 0.923320 | +0.000028 | 0.018680 |
| mid | 0.936000 | 0.919252 | 0.919277 | 0.919328 | +0.000028 | 0.016672 |

排查结论：

1. 把阈值从 `0.1` 加密到 `0.01` 只提升约 `2e-5` 到 `3e-5` AUC。
2. 完全连续的 exact score AUC 也只比 `0.01` 阈值网格高约 `5e-5`。
3. 因此当前 `0.0167~0.0197` 的论文差距不是由阈值扫频太粗或梯形积分精度导致。
4. 后续应优先继续排查：clean 数据是否与论文完全同版、JAER 输出事件的起止边界/重复事件处理、DND21 官方 label 生成口径、坐标/极性编码方向，以及官方实验是否在某些预处理步骤中裁剪或过滤了事件区间。

固定阈值对比（N149 使用 EBF 同档 best-F1 阈值）：

| Level | EBF Thr (best-F1) | Method | TPR | FPR | Precision | F1 | Accuracy |
|---|---:|---|---:|---:|---:|---:|---:|
| light | 0.50 | EBF | 0.975002 | 0.396886 | 0.944884 | 0.959707 | 0.928391 |
| light | 0.50 | N149 | 0.940655 | 0.318305 | 0.953752 | 0.947158 | 0.908197 |
| light_mid | 1.75 | EBF | 0.927291 | 0.228449 | 0.904169 | 0.915584 | 0.880444 |
| light_mid | 1.75 | N149 | 0.765692 | 0.096433 | 0.948603 | 0.847390 | 0.807166 |
| mid | 2.50 | EBF | 0.897832 | 0.193437 | 0.866288 | 0.881778 | 0.859737 |
| mid | 2.50 | N149 | 0.644994 | 0.048619 | 0.948764 | 0.767930 | 0.772877 |

结论：
1. 按 `s=5, tau=32ms` 重跑后，AUC 走势仍是 `1Hz > 3Hz > 5Hz`，趋势与论文一致。
2. 换成 JAER 原链路导出的噪声后，AUC 只比 Python-noise 版本提高约 `0.001~0.003`，仍低于论文约 `0.017~0.020`。
3. 因此“不直接调用 JAER 加噪”不是主要差异来源；下一步应优先检查 DND21/EBF 原始 ROC 统计协议、clean 数据版本、坐标/极性编码口径，以及 EBF 官方代码的阈值扫频/积分方式。
4. 在“同阈值”约束下，N149 的 FPR 更低但 TPR 下降更明显，导致 F1 普遍低于 EBF；这符合“阈值跨算法迁移不一定公平”的预期。

### 4.5 LED（10ms 切片拼接 100ms，EBF vs N149，2026-04-29）

源目录：
- `D:\hjx_workspace\scientific_reserach\dataset\LED`

数据结构：
- `raw_events/<scene>/<slice>.npy`
- `denoise_events/<scene>/<slice>.npy`
- `noise_events/<scene>/<slice>.npy`
- 当前检查的场景：`scene_100`, `scene_1004`, `scene_1018`
- 使用切片：`00031` 到 `00040`（共 10 片）

切片连续性确认（可拼接）：
1. 每片时间范围固定为 10ms：如 `t=300000..309999us`, `310000..319999us`, ... `390000..399999us`。
2. 相邻切片在 `raw_events`/`denoise_events` 中时间边界间隔稳定为 `+1us`，可视为严格连续。
3. `noise_events` 个别切片起始会晚 `1~9us`，但处于同一 10ms 窗口内，不影响 100ms 拼接。

数据格式确认：
- LED 原始 `.npy` 形状为 `(4, N)`，按实际取值确认字段顺序为：`[x, y, polarity(0/1), timestamp(us)]`。
- 因此无需跨格式转换，但需要重排为 myEVS 结构化数组并加 label。

转换脚本：
- `scripts/LED_alg_evalu/convert_led_to_npy.py`

转换命令：
```powershell
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/LED_alg_evalu/convert_led_to_npy.py --overwrite
```

转换规则：
1. `signal` 直接来自 `denoise_events`，label 记为 1。
2. `noise` 直接来自 `noise_events`，label 记为 0。
3. `noisy = signal + noise` 后按 `t` 稳定排序。
4. `raw` 仅用于一致性检查：`raw_events == denoise_events + noise_events`（事件数完全一致）。
5. 极性转换：`polarity>0 -> p=+1`，否则 `p=-1`（LED 的极性原本只有 0/1）。

输出目录：
- `D:\hjx_workspace\scientific_reserach\dataset\LED\converted_npy\<scene>\slices_00031_00040_100ms\`

转换统计（100ms）：

| Scene | Signal | Noise | Raw | Noise/Signal | Estimated Hz/pixel | Raw-(Signal+Noise) |
|---|---:|---:|---:|---:|---:|---:|
| scene_100 | 1,908,545 | 84,266 | 1,992,811 | 0.0442 | 0.984657 | 0 |
| scene_1004 | 1,786,584 | 203,977 | 1,990,561 | 0.1142 | 2.307351 | 0 |
| scene_1018 | 1,703,395 | 166,309 | 1,869,704 | 0.0976 | 1.881257 | 0 |

说明（关于“四种噪声程度”）：
1. 本地目录当前只有以上 3 个场景，不包含明确的 4 档噪声标签文件。
2. 因此本轮采用可量化的 `Noise/Signal` 与 `Hz/pixel` 作为噪声强度指标；后续若补齐完整 LED 场景可按该指标自动分档。

评测脚本：
- `scripts/LED_alg_evalu/run_led_ebf_n149.py`

评测命令：
```powershell
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/LED_alg_evalu/run_led_ebf_n149.py
```

评测设置：
- 分辨率：`1280x720`，时间基：`tick-ns=1000`。
- EBF：`radius=2`, `tau=32000us`, 阈值 `0.0:0.1:15.0`。
- N149：`radius={2,3,4,5}`, `tau={16,32,64,128,256,512}ms`，每场景取 best-AUC 和 best-F1。

结果文件：
- `data/LED/led_ebf_n149_summary.csv`
- `data/LED/<scene>/EBF/roc_ebf_<scene>_slices_00031_00040_100ms.csv`
- `data/LED/<scene>/N149/roc_n149_<scene>_slices_00031_00040_100ms.csv`

结果表：

| Scene | Noise/Signal | Hz/pixel | EBF AUC | N149 AUC | N149-EBF AUC | EBF F1 | N149 F1 | N149 best-AUC `(r,tau)` |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| scene_100 | 0.0442 | 0.984657 | 0.837446 | 0.913320 | +0.075874 | 0.976058 | 0.978704 | `(2,16ms)` |
| scene_1004 | 0.1142 | 2.307351 | 0.767038 | 0.856746 | +0.089709 | 0.941966 | 0.947916 | `(3,16ms)` |
| scene_1018 | 0.0976 | 1.881257 | 0.874173 | 0.910531 | +0.036358 | 0.950651 | 0.958325 | `(2,16ms)` |

均值：
- EBF AUC: `0.826219`
- N149 AUC: `0.893532`
- 平均 AUC 提升: `+0.067314`
- EBF Best-F1: `0.956225`
- N149 Best-F1: `0.961648`
- 平均 F1 提升: `+0.005424`

LED 结论：
1. 在已下载的 LED 三个场景（拼接 100ms）上，N149 对 EBF 的提升稳定且显著，尤其 AUC 提升明显。
2. N149 的最优时间窗统一偏短（`16ms`），与 DVSCLEAN 偏长时间窗（`64/128ms`）相反，说明噪声时序统计特性存在明显域差异。
3. LED 的 EBF AUC 相比其它数据集明显偏低，但 F1 仍较高，提示该数据集在可分排序（ROC形状）与最佳工作点之间存在更强 trade-off。
4. 该现象支持“跨数据集自适应参数或门控策略”的必要性，固定一套 `tau/r` 难以全域最优。

## 16. scripts 目录功能说明（补充）

### 5.1 根目录脚本

- `scripts/v2e_labeled_txt_to_npy.py`：v2e 标注 txt -> npy
- `scripts/split_labeled_events.py`：从 labeled npy 拆分 clean(signal_only)/noise
- `scripts/ED24csv_to_npy.py`：ED24 csv -> npy
- `scripts/openeb_csv_to_evtq.py`：OpenEB csv -> evtq
- `scripts/run_experiment.py`：通用实验驱动脚本（历史）
- `scripts/eval_bestpoint_mesr_aocc.py`：从已有 ROC CSV 自动回算 `best-AUC`/`best-F1` 点的 `MESR/AOCC`（支持单算法或多算法）

### 5.2 ED24_alg_evalu

- `run_slomo_baf.ps1`：ED24 上 BAF ROC 扫频
- `run_slomo_stcf.ps1`：ED24 上 STCF ROC 扫频
- `run_slomo_ebf.ps1`：ED24 上 EBF ROC 扫频
- `run_slomo_ebf_paper_s_tau.ps1`：EBF 论文风格参数扫频
- `run_slomo_fdf.ps1`：FastDecayFilter 相关扫频
- `summarize_best_params_ed24.py`：汇总 ED24 最优参数
- `summarize_best_params_ebf_optimized_ed24.py`：汇总 EBF_OPTIMIZED 最优参数
- `summarize_fixed_threshold_ebf_optimized_ed24.py`：固定阈值统计
- `select_best_tag_ebfopt_ed24.py`：从 tag 选择最佳
- `summarize_horizontal_round1.py`：Round1（BAF/STCF/EBF/N149）横向汇总
- `summarize_horizontal_round2_new_methods.py`：Round2（KNOISE/EVFLOW/YNOISE/TS/MLPF/PFD）横向汇总
- `summarize_runtime_ed24.py`：ED24 运行时统一口径汇总（按 level 最新记录统计）
- `sweep_*.py`/`prescreen_*.py`/`tune_*.py`：各版本候选公式预筛与调参
- `run_slomo_alg.ps1`（新增）：KNOISE/EVFLOW/YNOISE/TS/MLPF/PFD 统一入口
- `run_slomo_{knoise|evflow|ynoise|ts|mlpf|pfd}.ps1`（新增）：单算法入口

### 5.3 driving_alg_evalu

- `run_driving_alg.ps1`（新增）：Driving 数据集统一入口
- `run_driving_alg_paper.ps1`：Driving 论文口径数据统一入口（`mydriving_paper`）
- `run_driving_{knoise|evflow|ynoise|ts|mlpf|pfd}.ps1`（新增）：单算法入口
- `run_driving_n149.ps1`：Driving 上 N149 独立扫频入口（当前不并入 `run_driving_alg.ps1`）
- `generate_driving_paper_noise.py`：从 paper clean txt 生成 1/3/5Hz 的标注噪声 npy（FPN: sigma-decades=0.5）

### 5.4 noise_analyze

用于噪声结构分析、特征统计、分布可视化和误检类型分析（FP/transition/pattern 等），为算法改进提供先验证据。

### 5.5 DVSCLEAN_alg_evalu

- `convert_dvsclean_to_npy.py`：读取 DVSCLEAN HDF5，确认并转换 `label=0 signal / label=1 noise` 到 myEVS 的 labeled npy。
- `run_dvsclean_ebf_n149.py`：在 DVSCLEAN converted npy 上跑 EBF 与 N149，并输出逐场景 ROC CSV 与总汇总表。

### 5.6 LED_alg_evalu

- `convert_led_to_npy.py`：把 LED 的 `10 x 10ms` 切片（如 `00031..00040`）拼接成 100ms，并生成 myEVS labeled npy。
- `run_led_ebf_n149.py`：在 LED 100ms 拼接数据上跑 EBF 与 N149，并输出逐场景 ROC CSV 与总汇总表。

## 17. DVSCLEAN 全算法迁移实验（已完成）

实验目标：
1. 先在一个 DVSCLEAN 子集上对全部算法扫频，选每个算法的最优配置（以 AUC 优先，记录最佳 F1 阈值）。
1. 再把该配置迁移到其余 5 个子集，统一统计 AUC/F1。

执行脚本：
```powershell
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/DVSCLEAN_alg_evalu/run_dvsclean_all_alg_transfer.py
```

默认设置：
1. 调参子集：`MAH00446_ratio100`
1. 迁移评估子集：DVSCLEAN 剩余 5 个样本
1. 算法集合：`baf, stcf, ebf, knoise, evflow, ynoise, ts, mlpf, pfd, n149`

输出文件：
1. `data/DVSCLEAN/all_alg/tuning_all_alg_results.csv`
1. `data/DVSCLEAN/all_alg/tuning_best_config.csv`
1. `data/DVSCLEAN/all_alg/transfer_eval_all_alg.csv`
1. `data/DVSCLEAN/all_alg/transfer_eval_algorithm_summary.csv`
1. `data/DVSCLEAN/all_alg/transfer_eval_meta.json`

### 17.1 调参子集最优配置（MAH00446_ratio100）

| Algorithm | r | tau(us) | Best AUC | Best F1 | Best Threshold |
|---|---:|---:|---:|---:|---:|
| baf | 1 | 16000 | 0.943550 | 0.942840 | 1.0 |
| stcf | 3 | 32000 | 0.993422 | 0.983142 | 3.0 |
| ebf | 4 | 64000 | 0.995692 | 0.989147 | 3.0 |
| knoise | 1 | 8000 | 0.641571 | 0.666667 | 0.0 |
| evflow | 3 | 32000 | 0.975457 | 0.975035 | 64.0 |
| ynoise | 4 | 32000 | 0.995080 | 0.985747 | 4.0 |
| ts | 2 | 32000 | 0.939605 | 0.898773 | 0.2 |
| mlpf | 3 | 64000 | 0.994507 | 0.977250 | 3.0 |
| pfd | 4 | 64000 | 0.574169 | 0.258502 | 1.0 |
| n149 | 5 | 64000 | 0.996841 | 0.989040 | 1.323230 |

### 17.2 迁移到其余 5 个子集后的均值结果

| Rank | Algorithm | Mean AUC | Mean F1 |
|---:|---|---:|---:|
| 1 | n149 | 0.997021 | 0.989423 |
| 2 | n179 | 0.996602 | 0.989815 |
| 3 | ebf | 0.994311 | 0.984307 |
| 4 | mlpf | 0.994092 | 0.983355 |
| 5 | ynoise | 0.993882 | 0.981679 |
| 6 | stcf | 0.992940 | 0.982054 |
| 7 | evflow | 0.977279 | 0.977963 |
| 8 | baf | 0.954333 | 0.957899 |
| 9 | ts | 0.946160 | 0.925132 |
| 10 | knoise | 0.649253 | 0.746667 |
| 11 | pfd | 0.592245 | 0.296807 |

### 17.3 N179（并入 DVSCLEAN 对比）

| Scope | Mean AUC | Mean F1 | Mean Throughput(eps) | Mean EventRate(eps) | 裕量(吞吐/数据) |
|---|---:|---:|---:|---:|---:|
| DVSCLEAN 10 子集（best-AUC 点） | 0.996602 | 0.989815 | 2,352,538 | 500,251 | 4.70x |

说明：明细见 `data/summary/n179_fullrun_bestauc_20260506.csv` 中 `dataset=DVSCLEAN` 的 10 行；该口径满足实时处理要求。

结论（当前 DVSCLEAN 口径）：
1. `n149` 在迁移均值 AUC/F1 上均为第一，较 `ebf` 有稳定优势。
1. `ebf/mlpf/ynoise/stcf` 组成第二梯队，AUC 接近，差异主要体现在 F1 和阈值稳定性。
1. `knoise/pfd` 在该数据口径下表现明显偏弱，后续建议单独做参数空间扩展或实现口径核查。

## 18. LED 全算法迁移实验（10 场景）

说明：
1. 你确认 LED 共 10 个场景，本轮已全覆盖。
1. 为避免 `EVFLOW` 在单场景约 200 万事件上耗时过长，本轮采用轻量口径：每场景截断 `300k` 事件（100ms 拼接数据）。
1. 流程为：`2` 个调参场景 + `8` 个迁移评估场景。
1. 调参场景（seed=2026）：`scene_1004`, `scene_1033`。

执行命令：
```powershell
# 先补齐 10 场景 100ms 转换
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/LED_alg_evalu/convert_led_to_npy.py --src-root D:/hjx_workspace/scientific_reserach/dataset/LED --out-root D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy --slice-start 31 --slice-end 40 --overwrite

# 再跑全算法（2 调参 + 8 迁移，300k）
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/LED_alg_evalu/run_led_all_alg_transfer.py --npy-root D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy --out-root data/LED/all_alg_300k --num-scenes 10 --num-tune-scenes 2 --seed 2026 --max-events 300000
```

输出文件：
1. `data/LED/all_alg_300k/scene_stats.csv`
1. `data/LED/all_alg_300k/tuning_best_config.csv`
1. `data/LED/all_alg_300k/transfer_eval_all_alg.csv`
1. `data/LED/all_alg_300k/transfer_eval_algorithm_summary.csv`

### 18.1 LED 各场景噪声比例与 Hz/像素（100ms）

| Scene | noise/signal | noise Hz/pixel |
|---|---:|---:|
| scene_100 | 0.0442 | 0.9847 |
| scene_1004 | 0.1142 | 2.3074 |
| scene_1018 | 0.0976 | 1.8813 |
| scene_1028 | 0.0656 | 1.2727 |
| scene_1032 | 0.0800 | 1.6230 |
| scene_1033 | 0.1016 | 2.0629 |
| scene_1034 | 0.1115 | 2.2616 |
| scene_1043 | 0.0826 | 1.7207 |
| scene_1045 | 0.0918 | 1.8181 |
| scene_1046 | 0.0772 | 1.6180 |

### 18.2 调参后最优配置（2 调参场景均值，历史口径：LED-300k）

历史口径说明（必须注意）：
1. 本节来自早期 `300k` 截断实验（非全量），仅保留为过程记录，不作为当前最终结论依据。
1. 该批次包含旧版阈值网格与旧统计口径，尤其 MLPF/KNOISE 的阈值可分性在后续已修正。
1. 当前对外结论请优先使用 `18.6`（`scene_100` 全量复核）及后续“统一口径复跑”结果。

| Algorithm | r | tau(us) | tuned threshold | tune mean AUC | tune mean F1 |
|---|---:|---:|---:|---:|---:|
| baf | 1 | 2000 | 1.0 | 0.701507 | 0.722459 |
| stcf | 2 | 4000 | 1.0 | 0.775531 | 0.884095 |
| ebf | 2 | 16000 | 0.0 | 0.758830 | 0.898787 |
| knoise | 1 | 2000 | 0.0 | 0.535314 | 0.933208 |
| evflow | 2 | 8000 | 64.0 | 0.735036 | 0.807147 |
| ynoise | 2 | 8000 | 1.0 | 0.763457 | 0.898583 |
| ts | 2 | 8000 | 0.1 | 0.689276 | 0.900942 |
| mlpf | 3 | 8000 | 1.0 | 0.809618 | 0.915383 |
| pfd | 4 | 16000 | 1.0 | 0.501871 | 0.007748 |
| n149 | 2 | 16000 | 0.0 | 0.798983 | 0.933208 |

### 18.3 迁移到其余 8 场景结果（均值，历史口径：LED-300k）

历史口径说明（必须注意）：
1. 本节是“2 调参场景 + 8 迁移场景”的 `300k` 截断统计，受时间前缀采样偏差影响较大。
1. 该批次中的 `mean F1` 在 LED 这种信号占优场景会被放大，不能单独用于算法优劣排序。
1. 本节排名与 `18.6` 全量口径不一致是预期现象，不代表实现错误。

| Rank | Algorithm | Mean AUC | Mean F1 |
|---:|---|---:|---:|
| 1 | n149 | 0.865178 | 0.936704 |
| 2 | mlpf | 0.858065 | 0.935892 |
| 3 | stcf | 0.837178 | 0.906574 |
| 4 | ynoise | 0.835303 | 0.921767 |
| 5 | ebf | 0.833056 | 0.922723 |
| 6 | evflow | 0.782915 | 0.833812 |
| 7 | baf | 0.744689 | 0.758100 |
| 8 | ts | 0.736819 | 0.924623 |
| 9 | knoise | 0.526883 | 0.954161 |
| 10 | pfd | 0.510974 | 0.043414 |

结论（LED-300k 历史口径，仅供回溯）：
1. `n149` 与 `mlpf` 在 AUC 上为第一梯队，`n149` 略高。
1. `stcf/ynoise/ebf` 构成第二梯队，AUC 接近。
1. `knoise` 的 F1 高但 AUC 很低，说明其阈值可分性弱、ROC 拉伸能力不足。
1. `pfd` 在该口径下表现最差，建议后续单独扩展参数或切换到更匹配的噪声类型再评估。
1. 本节不用于当前最终结论；最终结论以全量与修正阈值口径为准。

### 18.4 LED 指标口径（按论文）

后续 LED 实验统一使用以下指标（替代仅看 F1）：
1. `AUC_best ↑`
1. `DA@Best-AUC ↑`
1. `DA_best ↑`
1. `AUC@Best-DA ↑`
1. `SR@Best-DA ↑`
1. `NR@Best-DA ↑`
1. `F1`（记录在 Best-DA 点）

其中（LED 论文口径）：
$$
SR=\frac{TP}{GP}=\mathrm{TPR},\quad
NR=\frac{TN}{GN}=1-\mathrm{FPR}
$$
$$
DA=\frac{1}{2}\left(\frac{TP}{GP}+\frac{TN}{GN}\right)=\frac{1}{2}(SR+NR)
$$
这里 \(GP\) / \(GN\) 分别是总信号数 / 总噪声数。

### 18.5 LED scene_100 轻量扫频复核（300k，含已训练 MLPF）

命令：
```powershell
# 先训练 scene_100 的 MLPF 模型
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/train_mlpf_torch.py --clean D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/scene_100/slices_00031_00040_100ms/scene_100_100ms_signal_only.npy --noisy D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/scene_100/slices_00031_00040_100ms/scene_100_100ms_labeled.npy --width 1280 --height 720 --tick-ns 1000 --duration-us 100000 --patch 7 --epochs 4 --batch-size 512 --max-events 300000 --out-ts data/LED/models/mlpf_torch_scene_100.pt --out-meta data/LED/models/mlpf_torch_scene_100.json

# 再跑全算法精简扫频 + 指标汇总
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/LED_alg_evalu/run_led_scene_sweep_summary.py --scene scene_100 --npy-root D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy --out-root data/LED/scene_sweep --max-events 300000
```

结果文件：
1. `data/LED/scene_sweep/scene_100/scene_sweep_summary.csv`

| Method | AUC_best | DA@Best-AUC | DA_best | AUC@Best-DA | SR@Best-DA | NR@Best-DA | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| n149 | 0.873910 | 0.806211 | 0.806211 | 0.873910 | 0.790019 | 0.822402 | 0.878399 |
| stcf | 0.858734 | 0.808082 | 0.808082 | 0.858734 | 0.774525 | 0.841638 | 0.869116 |
| ynoise | 0.846463 | 0.782648 | 0.782648 | 0.846463 | 0.776823 | 0.788472 | 0.869296 |
| ebf | 0.834866 | 0.781786 | 0.781786 | 0.834866 | 0.743512 | 0.820060 | 0.848575 |
| baf | 0.774406 | 0.774406 | 0.774406 | 0.774406 | 0.741032 | 0.807780 | 0.846650 |
| evflow | 0.773854 | 0.780230 | 0.780230 | 0.773854 | 0.799743 | 0.760718 | 0.882946 |
| ts | 0.736819 | 0.701289 | 0.723469 | 0.736244 | 0.816898 | 0.630040 | 0.890290 |
| knoise | 0.532107 | 0.532109 | 0.532109 | 0.532107 | 0.078272 | 0.985945 | 0.145088 |
| pfd | 0.501538 | 0.501538 | 0.501538 | 0.501538 | 0.003148 | 0.999929 | 0.006276 |
| mlpf | 0.500000 | 0.500000 | 0.500000 | 0.500000 | 0.000000 | 1.000000 | 0.000000 |

### 18.6 LED scene_100 全量复核（max-events=0，精简扫频）

执行命令：
```powershell
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/LED_alg_evalu/run_led_scene_sweep_summary.py --scene scene_100 --npy-root D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy --out-root data/LED/scene_sweep_full --max-events 0 --mlpf-patch 3
```

结果文件：
1. `data/LED/scene_sweep_full/scene_100/scene_sweep_summary.csv`

| Method | AUC_best | DA@Best-AUC | DA_best | AUC@Best-DA | SR@Best-DA | NR@Best-DA | F1 | BestRadius | BestTauUs | Threshold@Best-DA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| n149 | 0.913320 | 0.839652 | 0.839652 | 0.913320 | 0.839190 | 0.840113 | 0.909076 | 2 | 16000 | 2.758442 |
| ynoise | 0.887506 | 0.827097 | 0.827097 | 0.887506 | 0.800813 | 0.853381 | 0.886205 | 2 | 8000 | 6.000000 |
| stcf | 0.884103 | 0.835730 | 0.835730 | 0.884103 | 0.773957 | 0.897503 | 0.870356 | 2 | 4000 | 4.000000 |
| ebf | 0.856949 | 0.814909 | 0.814909 | 0.856949 | 0.821473 | 0.808345 | 0.897817 | 2 | 16000 | 4.500000 |
| baf | 0.799443 | 0.799443 | 0.799443 | 0.799443 | 0.788796 | 0.810089 | 0.877815 | 1 | 2000 | 1.000000 |
| ts | 0.795761 | 0.749767 | 0.749767 | 0.795761 | 0.772337 | 0.727197 | 0.865663 | 2 | 8000 | 0.400000 |
| evflow | 0.786815 | 0.798266 | 0.798266 | 0.786815 | 0.901163 | 0.695369 | 0.941353 | 2 | 8000 | 64.000000 |
| mlpf | 0.776736 | 0.751533 | 0.751533 | 0.776736 | 0.679460 | 0.823606 | 0.805406 | 3 | 16000 | 0.900000 |
| knoise | 0.532322 | 0.532316 | 0.532316 | 0.532322 | 0.081055 | 0.983576 | 0.149856 | 1 | 2000 | 1.000000 |
| pfd | 0.525451 | 0.525451 | 0.525451 | 0.525451 | 0.052742 | 0.998161 | 0.100192 | 2 | 16000 | 1.000000 |

说明（MLPF 前后差异的原因）：
1. 之前 10 场景结果里 MLPF 的阈值网格使用了 `1,2,3...`，这对“模型概率输出”不合理（概率范围本应在 `0~1`），会导致模型几乎全拒绝或全保留，统计失真。
1. 当前已修正：`LED/DVSCLEAN` 脚本中的 MLPF 阈值统一改为 `0.05~0.9` 概率网格。
1. 因此，旧结果与新结果不应直接横向比较；后续请以修正阈值后的新一轮结果为准。
1. `scene_100` 全量复核中，MLPF 在修正阈值后不再崩塌；`patch=7` 的最佳点为：`tag=mlpf_patch7_r3_tau16000_scene_100`，`threshold=0.9`，`AUC=0.776736`。
1. 对比：同口径下 `patch=7` 明显优于 `patch=3`（`AUC: 0.776736 > 0.726367`），与你之前“7 比 3 更好”的观察一致。

MLPF patch 对比（scene_100 全量，同口径）：

| patch | AUC_best | DA_best | F1 | BestTag | Threshold |
|---:|---:|---:|---:|---|---:|
| 7 | 0.776736 | 0.751533 | 0.805406 | mlpf_patch7_r3_tau16000_scene_100 | 0.9 |
| 9 | 0.772356 | 0.744163 | 0.766289 | mlpf_patch9_r3_tau16000_scene_100 | 0.9 |
| 5 | 0.742103 | 0.717436 | 0.695279 | mlpf_patch5_r3_tau16000_scene_100 | 0.9 |
| 3 | 0.726367 | 0.707544 | 0.712226 | mlpf_r3_tau16000_scene_100 | 0.9 |

### 18.7 LED 与 ED24 排名差异的排查结论（2026-05-01）

结论先行：
1. 当前代码与算法主实现未发现“会系统性颠倒排名”的明显错误；差异主要来自数据分布与统计口径差异。
1. `ED24` 排名更接近复杂度预期，`LED` 排名波动更大，属于数据与指标耦合导致，不是单一脚本 bug。

已确认的关键原因：
1. 数据分布差异：`LED` 的 `noise/signal` 普遍较低（约 `0.04~0.11`），信号主导；`ED24` 的噪声形态与强度分布不同。
1. 指标敏感性差异：在信号占优数据上，`F1` 易被“高保留率”策略放大，不能单独反映去噪能力；需以 `AUC + DA + SR/NR` 为主。
1. 采样方式差异：`LED-300k` 是时间前缀截断，不等价于全量分布；与单场景全量结果出现较大偏差是正常现象。
1. 历史口径影响：`18.2/18.3` 含旧阈值网格与旧统计方式，特别是 MLPF 的历史阈值口径已被后续修正，不能与新结果直接对比。
1. 算法假设不一致：`EVFLOW/TS` 等方法依赖局部时空结构假设，数据切片短且噪声形态变化时，未必优于简单门控类方法。

对“结果是否正常”的判断：
1. “同一算法在 ED24 与 LED 排名差异明显”在当前数据口径下是正常现象。
1. “LED 单场景全量 vs 多场景前 300k 差异大”是可解释现象，主要由采样与分布漂移导致。
1. 当前结论应以全量、统一指标口径结果为主；历史 `300k` 结果仅作过程记录。

### 18.8 LED scene_1004 全量重扫复核（覆盖旧迁移段）

目标：
1. 按 `scene_100` 同口径做 `scene_1004` 全量扫频（不是固定阈值迁移）。
1. `MLPF` 使用 `scene_1004` 专属重训模型，避免模型错配。
1. `EVFLOW` 使用轻量扫频（单点）以控制总时长，其余算法保持原有网格。

执行命令：
```powershell
# 1) 训练 scene_1004 的 MLPF（全量）
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/train_mlpf_torch.py --clean D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/scene_1004/slices_00031_00040_100ms/scene_1004_100ms_signal_only.npy --noisy D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/scene_1004/slices_00031_00040_100ms/scene_1004_100ms_labeled.npy --width 1280 --height 720 --tick-ns 1000 --duration-us 100000 --patch 7 --epochs 4 --batch-size 512 --max-events 0 --out-ts data/LED/models/mlpf_torch_scene_1004.pt --out-meta data/LED/models/mlpf_torch_scene_1004.json

# 2) scene_1004 全量重扫（EVFLOW 轻量）
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/LED_alg_evalu/run_led_scene_sweep_summary.py --scene scene_1004 --npy-root D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy --out-root data/LED/scene_sweep_full --max-events 0 --mlpf-model-pattern "data/LED/models/mlpf_torch_{scene}.pt" --mlpf-patch 7 --evflow-lite
```

结果文件：
1. `data/LED/scene_sweep_full/scene_1004/scene_sweep_summary.csv`

`scene_1004` 全量重扫结果：

| Method | AUC_best | DA_best | F1 | BestRadius | BestTauUs | Threshold@Best-DA |
|---|---:|---:|---:|---:|---:|---:|
| n149 | 0.856746 | 0.783714 | 0.865405 | 3 | 16000 | 2.441698 |
| ynoise | 0.819130 | 0.752756 | 0.864842 | 2 | 8000 | 3.0 |
| stcf | 0.818052 | 0.757306 | 0.849637 | 2 | 4000 | 2.0 |
| ebf | 0.808347 | 0.745625 | 0.846271 | 2 | 16000 | 2.5 |
| evflow | 0.776031 | 0.776031 | 0.896900 | 2 | 8000 | 64.0 |
| baf | 0.720325 | 0.720325 | 0.749180 | 1 | 2000 | 1.0 |
| ts | 0.712194 | 0.680049 | 0.872317 | 2 | 8000 | 0.2 |
| mlpf | 0.652276 | 0.642371 | 0.587136 | 3 | 16000 | 0.05 |
| knoise | 0.537806 | 0.537737 | 0.178172 | 1 | 2000 | 1.0 |
| pfd | 0.500265 | 0.500265 | 0.001080 | 2 | 16000 | 1.0 |

与 `scene_100` 全量结果对比（按 AUC/DA）：
1. 头部排序基本稳定：`n149` 仍为第一，`ynoise/stcf/ebf` 仍在第二梯队，未出现“完全反转”。
1. 中段发生变化：`evflow` 在 `scene_1004` 的 `DA` 提升明显，超过 `baf/ts/mlpf`，说明其在该场景噪声结构下更匹配。
1. `mlpf` 在 `scene_1004` 明显下滑（即使已重训），说明 LED 跨场景下 MLPF 的阈值与特征稳定性仍不足。
1. 结论更新后，旧的“固定迁移评估”段不再作为本节结论依据，以本次全量重扫结果为准。

### 18.9 LED scene_1032 全量重扫复核（中间噪声强度）

场景选择依据：
1. `scene_1032` 的噪声强度约 `noise/signal=0.0800`、`noise Hz/pixel=1.6230`，位于 `scene_100(0.0442)` 与 `scene_1004(0.1142)` 之间。

执行命令：
```powershell
# 1) 训练 scene_1032 的 MLPF（全量）
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/train_mlpf_torch.py --clean D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/scene_1032/slices_00031_00040_100ms/scene_1032_100ms_signal_only.npy --noisy D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy/scene_1032/slices_00031_00040_100ms/scene_1032_100ms_labeled.npy --width 1280 --height 720 --tick-ns 1000 --duration-us 100000 --patch 7 --epochs 4 --batch-size 512 --max-events 0 --out-ts data/LED/models/mlpf_torch_scene_1032.pt --out-meta data/LED/models/mlpf_torch_scene_1032.json

# 2) 全量扫频（EVFLOW 轻量）
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/LED_alg_evalu/run_led_scene_sweep_summary.py --scene scene_1032 --npy-root D:/hjx_workspace/scientific_reserach/dataset/LED/converted_npy --out-root data/LED/scene_sweep_full --max-events 0 --mlpf-model-pattern "data/LED/models/mlpf_torch_{scene}.pt" --mlpf-patch 7 --evflow-lite
```

结果文件：
1. `data/LED/scene_sweep_full/scene_1032/scene_sweep_summary.csv`

`scene_1032` 全量重扫结果：

| Method | AUC_best | DA_best | F1 | BestTagByDA | Threshold@Best-DA |
|---|---:|---:|---:|---|---:|
| n149 | 0.913109 | 0.841805 | 0.887694 | n149_r2_tau16000_scene_1032 | 1.382605 |
| ebf | 0.883409 | 0.816482 | 0.875212 | ebf_r2_tau16000_scene_1032 | 2.000000 |
| ynoise | 0.880812 | 0.816539 | 0.855735 | ynoise_r2_tau8000_scene_1032 | 3.000000 |
| stcf | 0.868663 | 0.815857 | 0.857076 | stcf_r2_tau8000_scene_1032 | 3.000000 |
| evflow | 0.833474 | 0.833474 | 0.895036 | evflow_r2_tau8000_scene_1032 | 64.000000 |
| baf | 0.777079 | 0.777079 | 0.930470 | baf_r1_tau8000_scene_1032 | 1.000000 |
| ts | 0.764852 | 0.719848 | 0.870726 | ts_r2_tau8000_scene_1032 | 0.200000 |
| mlpf | 0.654294 | 0.653435 | 0.519507 | mlpf_r3_tau16000_scene_1032 | 0.050000 |
| knoise | 0.524312 | 0.524316 | 0.100060 | knoise_r1_tau2000_scene_1032 | 1.000000 |
| pfd | 0.506476 | 0.506476 | 0.025654 | pfd_r2_tau16000_scene_1032 | 1.000000 |

与 `scene_100 / scene_1004` 的对比结论：
1. 头部稳定：`n149` 仍保持第一；`ebf/ynoise/stcf` 仍是紧邻第二梯队。
1. `scene_1032` 在多数方法上的结果位于 `scene_100` 与 `scene_1004` 之间或接近较优端，符合“中间噪声强度”预期。
1. `evflow` 在 `scene_1032` 提升明显（AUC `0.833`），但计算代价依然较高，因此继续保留 `--evflow-lite` 口径用于场景批量比较。
1. `mlpf` 仍明显落后于 `n149/ebf/stcf/ynoise`，说明当前训练设置与阈值策略在 LED 跨场景上仍需单独优化。

### 18.10 LED 其余 7 场景全量复核（本轮新增，逐场景独立脚本）

本轮新增独立脚本（一个场景一个入口）：
1. `scripts/LED_alg_evalu/run_led_scene_1018.ps1`
1. `scripts/LED_alg_evalu/run_led_scene_1028.ps1`
1. `scripts/LED_alg_evalu/run_led_scene_1033.ps1`
1. `scripts/LED_alg_evalu/run_led_scene_1034.ps1`
1. `scripts/LED_alg_evalu/run_led_scene_1043.ps1`
1. `scripts/LED_alg_evalu/run_led_scene_1045.ps1`
1. `scripts/LED_alg_evalu/run_led_scene_1046.ps1`

通用脚本：
1. `scripts/LED_alg_evalu/run_led_scene_full.ps1`（统一执行“MLPF 训练 + 全量扫频”；默认 `EVFLOW` 轻量）

执行口径：
1. 每个场景先训练对应 `MLPF`（`patch=7`, `max-events=0`），再跑 `run_led_scene_sweep_summary.py` 全量扫频。
1. `EVFLOW` 使用 `--evflow-lite`（`r=2, tau=8000, threshold=64` 单点）控制时长，其它算法保持原网格。

新增 7 场景结果（按 AUC 前四）：

| Scene | Top1 | AUC | Top2 | AUC | Top3 | AUC | Top4 | AUC | EVFLOW AUC | MLPF AUC |
|---|---|---:|---|---:|---|---:|---|---:|---:|---:|
| scene_1018 | n149 | 0.910531 | ebf | 0.880963 | ynoise | 0.877878 | stcf | 0.865144 | 0.823847 | 0.622571 |
| scene_1028 | n149 | 0.910285 | ebf | 0.876501 | ynoise | 0.875978 | stcf | 0.866415 | 0.835661 | 0.657940 |
| scene_1033 | n149 | 0.858952 | ynoise | 0.821549 | ebf | 0.818118 | stcf | 0.811112 | 0.788009 | 0.641480 |
| scene_1034 | n149 | 0.868648 | ynoise | 0.832110 | stcf | 0.832075 | ebf | 0.820762 | 0.789518 | 0.683596 |
| scene_1043 | n149 | 0.906577 | ebf | 0.878211 | ynoise | 0.875665 | stcf | 0.861756 | 0.821673 | 0.702376 |
| scene_1045 | n149 | 0.888328 | ebf | 0.854045 | ynoise | 0.852406 | stcf | 0.837719 | 0.802851 | 0.668571 |
| scene_1046 | n149 | 0.914644 | ebf | 0.883508 | ynoise | 0.880838 | stcf | 0.867658 | 0.834627 | 0.701646 |

结论：
1. 7 个新增场景中，`n149` 均为 AUC 第一，头部排序稳定。
1. `ebf/ynoise/stcf` 在不同场景有小幅换位，但整体属于同一梯队。
1. `mlpf` 在 LED 全量场景下仍明显落后于头部方法，说明当前训练流程仍需继续优化。

### 18.11 LED 10 场景全量汇总（统一口径）

汇总脚本：
```powershell
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/LED_alg_evalu/summarize_led_scene_sweep_full.py
```

输出文件：
1. `data/LED/scene_sweep_full/all_scenes_full_summary.csv`（10 场景逐算法明细）
1. `data/LED/scene_sweep_full/all_scenes_full_mean.csv`（10 场景均值）
1. `data/LED/scene_sweep_full/all_scenes_full_rank.csv`（场景内 AUC 排名）

10 场景均值（AUC/DA/F1）：

| Method | Mean AUC_best | Mean DA_best | Mean F1 |
|---|---:|---:|---:|
| n149 | 0.894114 | 0.820302 | 0.884181 |
| n179 | 0.894045 | 0.606241 | 0.960161 |
| ynoise | 0.860387 | 0.794133 | 0.859366 |
| ebf | 0.856081 | 0.791335 | 0.857951 |
| stcf | 0.851270 | 0.795682 | 0.857841 |
| evflow | 0.809054 | 0.810396 | 0.905818 |
| baf | 0.759172 | 0.759172 | 0.859402 |
| ts | 0.736103 | 0.703913 | 0.870614 |
| mlpf | 0.676149 | 0.669133 | 0.579657 |
| knoise | 0.527620 | 0.527604 | 0.121880 |
| pfd | 0.505619 | 0.505619 | 0.022113 |

总结合并结论（LED 全量）：
1. AUC 口径下，`n149` 在 10 场景均值与单场景排名都保持领先。
1. `ynoise/ebf/stcf` 为稳定第二梯队，场景间仅有小幅排序波动。
1. `EVFLOW` 的 `DA/F1` 可观，但 AUC 仍显著低于头部；其计算代价高，建议保留轻量扫频用于横评。
1. `MLPF` 当前在 LED 全量口径下泛化仍不足，后续需要重做训练策略（样本均衡、损失重加权、阈值校准）。

### 18.12 N179（并入 LED 对比）

| Scope | Mean AUC | Mean DA@Best-AUC | Mean F1 | Mean Throughput(eps) | Mean EventRate(eps) | 裕量(吞吐/数据) |
|---|---:|---:|---:|---:|---:|---:|
| LED 10 场景（best-AUC 点） | 0.894045 | 0.606241 | 0.960161 | 780,591 | 19,491,713 | 0.04x |

说明：`n179` 在 LED 上平均吞吐远低于场景事件率（约 18M~20M eps），当前口径不满足实时；明细见 `data/summary/n179_fullrun_bestauc_20260506.csv` 中 `dataset=LED`。

## 19. DVSCLEAN 全算法全子集复跑（2026-05-02）

本轮按你要求完成：
1. 每个子集单独训练 MLPF（不是复用其他数据集模型）。
1. 每个子集独立扫频（不是“第一子集调参后迁移”）。
1. 统计并汇总 `AUC/DA/F1/SNR`。

### 19.1 DVSCLEAN 片段时长（每个 scene 的 ratio50/ratio100 时长相同）

| Scene | Duration (s) |
|---|---:|
| MAH00444 | 0.6769 |
| MAH00446 | 0.6031 |
| MAH00447 | 0.3727 |
| MAH00448 | 1.1677 |
| MAH00449 | 1.0950 |

说明：时长由转换后的时间戳 `t_last - t_first` 计算，已写入各子集 `*_meta.json`。

### 19.2 运行脚本

单子集（会自动训练 MLPF 并扫频全部算法）：

```powershell
powershell -ExecutionPolicy Bypass -File ./scripts/DVSCLEAN_alg_evalu/run_dvsclean_one.ps1 -Scene MAH00448 -Level ratio50
```

全子集顺序跑：

```powershell
powershell -ExecutionPolicy Bypass -File ./scripts/DVSCLEAN_alg_evalu/run_dvsclean_all_full.ps1
```

汇总：

```powershell
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/DVSCLEAN_alg_evalu/summarize_dvsclean_scene_sweep_full.py
```

### 19.3 指标口径（本轮新增）

- `SR = TPR`
- `NR = 1 - FPR`
- `DA = 0.5 * (SR + NR)`
- `SNR_linear = SR / (1 - NR + eps) = TPR / FPR`
- `SNRdB = 10 * log10(SNR_linear)`

### 19.4 10 子集均值结果（统一口径：Best-AUC）

| Method | AUC_best | DA_best | SNRdB@Best-AUC | F1 |
|---|---:|---:|---:|---:|
| n149 | 0.996962 | 0.988846 | 20.724885 | 0.990038 |
| n179 | 0.996602 | 0.988204 | 19.993846 | 0.989815 |
| ebf | 0.993967 | 0.983092 | 20.377353 | 0.984271 |
| ynoise | 0.993411 | 0.982136 | 19.999665 | 0.983587 |
| stcf | 0.989767 | 0.976582 | 20.516487 | 0.977240 |
| mlpf | 0.982256 | 0.974781 | 18.905465 | 0.976296 |
| baf | 0.947943 | 0.947943 | 13.032464 | 0.953254 |
| ts | 0.939306 | 0.915388 | 8.452000 | 0.927443 |
| evflow | 0.773337 | 0.773337 | 39.868824 | 0.682832 |
| knoise | 0.638881 | 0.638770 | 17.599148 | 0.438068 |
| pfd | 0.561714 | 0.561714 | 47.466607 | 0.205158 |

### 19.5 结果文件

- 全部子集明细：`data/DVSCLEAN/scene_sweep_full/all_samples_full_summary.csv`
- 10 子集均值：`data/DVSCLEAN/scene_sweep_full/all_samples_full_mean.csv`
- 单子集明细：`data/DVSCLEAN/scene_sweep_full/<scene>_<level>/scene_sweep_summary.csv`
- MLPF 模型：`data/DVSCLEAN/models/mlpf_torch_<scene>_<level>.pt`

备注：`BestTagByAUC / BestTagByDA / Threshold@Best-DA` 已在 `all_samples_full_summary.csv` 中逐子集记录，便于后续按算法抽取“最优扫频参数”。

### 19.6 N179（DVSCLEAN 并入对比）

数据源：`data/summary/n179_fullrun_summary_20260506.csv`（`dataset=DVSCLEAN`，`point=best_auc`）。

| Method | AUC_best | DA@Best-AUC | SNRdB@Best-AUC | F1@Best-AUC |
|---|---:|---:|---:|---:|
| n179 | 0.996602 | 0.988204 | 19.993846 | 0.989815 |

说明：
1. 该行已并入 19.4 的全算法均值表，口径与本节一致（10 子集均值）。
2. 单子集明细见 `data/summary/n179_fullrun_summary_20260506.csv`，可与 21 节逐子集表对照。

## 20. 数据集聚合导航（重排入口）

为避免同一数据集内容分散，后续以本节作为统一入口：

- ED24：查看 11~18 相关章节（含横向对比、MESR/AOCC、runtime）。
- Driving：查看 12~16 相关章节（含 light/light_mid/mid、paper 口径、ED24 噪声版本）。
- DVSCLEAN：优先看 19~21（含 n179 并入结果，见 19.4/19.6）。
- LED：优先看 18.6~18.11（全量场景与汇总）。

说明：旧的分散记录先保留为“实验历史”，后续新增结果只往对应数据集聚合章节追加。

## 21. DVSCLEAN 单子集对比表（全算法）

数据源：data/DVSCLEAN/scene_sweep_full/all_samples_full_summary.csv

#### MAH00444_ratio100

| Method | AUC_best | DA_best | F1 | BestTagByDA | Threshold@Best-DA |
|---|---:|---:|---:|---|---:|
| n149 | 0.997742 | 0.991002 | 0.990983 | n149_r5_tau128000_MAH00444_ratio100 | 1.669879 |
| n179 | 0.997407 | 0.990475 | 0.990458 | n179_s9_r4_tau128000 | 1.707812 |
| ebf | 0.994709 | 0.984529 | 0.984413 | ebf_r4_tau64000_MAH00444_ratio100 | 2.500000 |
| ynoise | 0.993649 | 0.981747 | 0.981645 | ynoise_r4_tau32000_MAH00444_ratio100 | 3.000000 |
| stcf | 0.992759 | 0.980994 | 0.980698 | stcf_r3_tau32000_MAH00444_ratio100 | 3.000000 |
| mlpf | 0.989870 | 0.983365 | 0.983217 | mlpf_r3_tau64000_MAH00444_ratio100 | 0.200000 |
| baf | 0.955550 | 0.955550 | 0.955322 | baf_r1_tau16000_MAH00444_ratio100 | 1.000000 |
| ts | 0.948664 | 0.920451 | 0.922011 | ts_r2_tau16000_MAH00444_ratio100 | 0.100000 |
| evflow | 0.863857 | 0.863857 | 0.842402 | evflow_r2_tau8000_MAH00444_ratio100 | 64.000000 |
| knoise | 0.644415 | 0.644308 | 0.452412 | knoise_r1_tau16000_MAH00444_ratio100 | 1.000000 |
| pfd | 0.534983 | 0.534983 | 0.130799 | pfd_r4_tau64000_MAH00444_ratio100 | 1.000000 |

#### MAH00444_ratio50

| Method | AUC_best | DA_best | F1 | BestTagByDA | Threshold@Best-DA |
|---|---:|---:|---:|---|---:|
| n149 | 0.998073 | 0.991196 | 0.993014 | n149_r4_tau128000_MAH00444_ratio50 | 1.431657 |
| n179 | 0.997691 | 0.990207 | 0.993124 | n179_s9_r4_tau128000 | 1.276058 |
| ebf | 0.995439 | 0.986338 | 0.988248 | ebf_r3_tau64000_MAH00444_ratio50 | 1.500000 |
| ynoise | 0.994632 | 0.984518 | 0.988618 | ynoise_r4_tau64000_MAH00444_ratio50 | 3.000000 |
| stcf | 0.993576 | 0.984526 | 0.986975 | stcf_r3_tau32000_MAH00444_ratio50 | 2.000000 |
| mlpf | 0.991074 | 0.984841 | 0.988207 | mlpf_r3_tau64000_MAH00444_ratio50 | 0.200000 |
| baf | 0.967911 | 0.967911 | 0.977772 | baf_r1_tau32000_MAH00444_ratio50 | 1.000000 |
| ts | 0.962928 | 0.941237 | 0.955354 | ts_r2_tau16000_MAH00444_ratio50 | 0.100000 |
| evflow | 0.863862 | 0.863862 | 0.842416 | evflow_r2_tau8000_MAH00444_ratio50 | 64.000000 |
| knoise | 0.671487 | 0.671352 | 0.514106 | knoise_r1_tau16000_MAH00444_ratio50 | 1.000000 |
| pfd | 0.535172 | 0.535172 | 0.131459 | pfd_r4_tau64000_MAH00444_ratio50 | 1.000000 |

#### MAH00446_ratio100

| Method | AUC_best | DA_best | F1 | BestTagByDA | Threshold@Best-DA |
|---|---:|---:|---:|---|---:|
| n149 | 0.996841 | 0.989603 | 0.989586 | n149_r5_tau128000_MAH00446_ratio100 | 1.815941 |
| n179 | 0.996478 | 0.988842 | 0.988778 | n179_s9_r4_tau64000 | 1.407844 |
| ebf | 0.995692 | 0.989195 | 0.989147 | ebf_r4_tau64000_MAH00446_ratio100 | 3.000000 |
| ynoise | 0.995080 | 0.986665 | 0.986573 | ynoise_r4_tau64000_MAH00446_ratio100 | 6.000000 |
| stcf | 0.993422 | 0.983342 | 0.983142 | stcf_r3_tau32000_MAH00446_ratio100 | 3.000000 |
| mlpf | 0.985937 | 0.978712 | 0.978416 | mlpf_r3_tau64000_MAH00446_ratio100 | 0.200000 |
| baf | 0.943550 | 0.943550 | 0.942840 | baf_r1_tau16000_MAH00446_ratio100 | 1.000000 |
| ts | 0.939605 | 0.913183 | 0.915183 | ts_r2_tau16000_MAH00446_ratio100 | 0.100000 |
| evflow | 0.843886 | 0.843886 | 0.815018 | evflow_r2_tau8000_MAH00446_ratio100 | 64.000000 |
| knoise | 0.641571 | 0.641448 | 0.445209 | knoise_r1_tau8000_MAH00446_ratio100 | 1.000000 |
| pfd | 0.574169 | 0.574169 | 0.258502 | pfd_r4_tau64000_MAH00446_ratio100 | 1.000000 |

#### MAH00446_ratio50

| Method | AUC_best | DA_best | F1 | BestTagByDA | Threshold@Best-DA |
|---|---:|---:|---:|---|---:|
| n149 | 0.997454 | 0.990037 | 0.992445 | n149_r5_tau128000_MAH00446_ratio50 | 1.576275 |
| n179 | 0.997145 | 0.988748 | 0.991750 | n179_s9_r4_tau64000 | 0.900412 |
| ebf | 0.996279 | 0.990536 | 0.992285 | ebf_r4_tau64000_MAH00446_ratio50 | 2.500000 |
| ynoise | 0.995909 | 0.989130 | 0.991345 | ynoise_r4_tau64000_MAH00446_ratio50 | 4.000000 |
| stcf | 0.994363 | 0.985470 | 0.988380 | stcf_r3_tau32000_MAH00446_ratio50 | 2.000000 |
| mlpf | 0.986152 | 0.977729 | 0.980667 | mlpf_r3_tau64000_MAH00446_ratio50 | 0.300000 |
| baf | 0.957557 | 0.957557 | 0.967969 | baf_r1_tau32000_MAH00446_ratio50 | 1.000000 |
| ts | 0.955003 | 0.936622 | 0.951869 | ts_r2_tau16000_MAH00446_ratio50 | 0.100000 |
| evflow | 0.843852 | 0.843852 | 0.814983 | evflow_r2_tau8000_MAH00446_ratio50 | 64.000000 |
| knoise | 0.663555 | 0.663449 | 0.495914 | knoise_r1_tau8000_MAH00446_ratio50 | 1.000000 |
| pfd | 0.574334 | 0.574334 | 0.258985 | pfd_r4_tau64000_MAH00446_ratio50 | 1.000000 |

#### MAH00447_ratio100

| Method | AUC_best | DA_best | F1 | BestTagByDA | Threshold@Best-DA |
|---|---:|---:|---:|---|---:|
| n149 | 0.995988 | 0.988049 | 0.987994 | n149_r5_tau64000_MAH00447_ratio100 | 1.637956 |
| n179 | 0.995674 | 0.987820 | 0.987758 | n179_s9_r4_tau64000 | 1.747562 |
| ebf | 0.992858 | 0.982501 | 0.982358 | ebf_r4_tau64000_MAH00447_ratio100 | 3.500000 |
| ynoise | 0.991960 | 0.979811 | 0.979684 | ynoise_r4_tau32000_MAH00447_ratio100 | 4.000000 |
| stcf | 0.991389 | 0.979344 | 0.979208 | stcf_r3_tau32000_MAH00447_ratio100 | 3.000000 |
| mlpf | 0.949402 | 0.947139 | 0.944726 | mlpf_r3_tau64000_MAH00447_ratio100 | 0.050000 |
| baf | 0.939848 | 0.939848 | 0.940472 | baf_r1_tau16000_MAH00447_ratio100 | 1.000000 |
| ts | 0.921671 | 0.891170 | 0.889374 | ts_r2_tau8000_MAH00447_ratio100 | 0.100000 |
| evflow | 0.859518 | 0.859518 | 0.836619 | evflow_r2_tau8000_MAH00447_ratio100 | 64.000000 |
| pfd | 0.658499 | 0.658499 | 0.481677 | pfd_r4_tau64000_MAH00447_ratio100 | 1.000000 |
| knoise | 0.623476 | 0.623386 | 0.400936 | knoise_r1_tau16000_MAH00447_ratio100 | 1.000000 |

#### MAH00447_ratio50

| Method | AUC_best | DA_best | F1 | BestTagByDA | Threshold@Best-DA |
|---|---:|---:|---:|---|---:|
| n149 | 0.996694 | 0.988092 | 0.990625 | n149_r5_tau64000_MAH00447_ratio50 | 1.380958 |
| n179 | 0.996399 | 0.987550 | 0.990803 | n179_s9_r4_tau64000 | 1.242048 |
| ebf | 0.993894 | 0.985266 | 0.988367 | ebf_r4_tau64000_MAH00447_ratio50 | 2.500000 |
| ynoise | 0.993307 | 0.982597 | 0.986597 | ynoise_r4_tau32000_MAH00447_ratio50 | 3.000000 |
| stcf | 0.992612 | 0.982680 | 0.984220 | stcf_r3_tau32000_MAH00447_ratio50 | 3.000000 |
| mlpf | 0.967307 | 0.964868 | 0.967039 | mlpf_r3_tau64000_MAH00447_ratio50 | 0.050000 |
| baf | 0.956117 | 0.956117 | 0.965066 | baf_r1_tau16000_MAH00447_ratio50 | 1.000000 |
| ts | 0.943875 | 0.927201 | 0.954025 | ts_r2_tau16000_MAH00447_ratio50 | 0.100000 |
| evflow | 0.859582 | 0.859582 | 0.836782 | evflow_r2_tau8000_MAH00447_ratio50 | 64.000000 |
| pfd | 0.658238 | 0.658238 | 0.481113 | pfd_r4_tau64000_MAH00447_ratio50 | 1.000000 |
| knoise | 0.645709 | 0.645580 | 0.456074 | knoise_r1_tau16000_MAH00447_ratio50 | 1.000000 |

#### MAH00448_ratio100

| Method | AUC_best | DA_best | F1 | BestTagByDA | Threshold@Best-DA |
|---|---:|---:|---:|---|---:|
| n149 | 0.996266 | 0.987274 | 0.987239 | n149_r5_tau256000_MAH00448_ratio100 | 2.020751 |
| n179 | 0.995813 | 0.986750 | 0.986674 | n179_s9_r4_tau128000 | 1.479062 |
| ebf | 0.992530 | 0.978733 | 0.978460 | ebf_r4_tau64000_MAH00448_ratio100 | 2.000000 |
| ynoise | 0.992090 | 0.978985 | 0.978709 | ynoise_r4_tau64000_MAH00448_ratio100 | 4.000000 |
| mlpf | 0.987035 | 0.975074 | 0.974685 | mlpf_r3_tau64000_MAH00448_ratio100 | 0.400000 |
| stcf | 0.986225 | 0.970013 | 0.969558 | stcf_r3_tau32000_MAH00448_ratio100 | 2.000000 |
| baf | 0.938764 | 0.938764 | 0.938380 | baf_r1_tau32000_MAH00448_ratio100 | 1.000000 |
| ts | 0.935436 | 0.906353 | 0.909775 | ts_r2_tau32000_MAH00448_ratio100 | 0.100000 |
| evflow | 0.664909 | 0.664909 | 0.496084 | evflow_r2_tau8000_MAH00448_ratio100 | 64.000000 |
| knoise | 0.616030 | 0.615918 | 0.382390 | knoise_r1_tau32000_MAH00448_ratio100 | 1.000000 |
| pfd | 0.533887 | 0.533887 | 0.126963 | pfd_r4_tau64000_MAH00448_ratio100 | 1.000000 |

#### MAH00448_ratio50

| Method | AUC_best | DA_best | F1 | BestTagByDA | Threshold@Best-DA |
|---|---:|---:|---:|---|---:|
| n149 | 0.997011 | 0.987673 | 0.990265 | n149_r5_tau128000_MAH00448_ratio50 | 1.147876 |
| n179 | 0.996620 | 0.986880 | 0.990309 | n179_s9_r4_tau128000 | 1.047920 |
| ebf | 0.993905 | 0.981973 | 0.984973 | ebf_r4_tau64000_MAH00448_ratio50 | 1.500000 |
| ynoise | 0.993586 | 0.983005 | 0.985497 | ynoise_r4_tau64000_MAH00448_ratio50 | 3.000000 |
| mlpf | 0.989478 | 0.978339 | 0.982370 | mlpf_r3_tau64000_MAH00448_ratio50 | 0.400000 |
| stcf | 0.987834 | 0.974065 | 0.975209 | stcf_r3_tau32000_MAH00448_ratio50 | 2.000000 |
| baf | 0.952266 | 0.952266 | 0.958112 | baf_r1_tau32000_MAH00448_ratio50 | 1.000000 |
| ts | 0.950843 | 0.932491 | 0.952022 | ts_r2_tau32000_MAH00448_ratio50 | 0.100000 |
| evflow | 0.665007 | 0.665007 | 0.496317 | evflow_r2_tau8000_MAH00448_ratio50 | 64.000000 |
| knoise | 0.630134 | 0.630034 | 0.418635 | knoise_r1_tau32000_MAH00448_ratio50 | 1.000000 |
| pfd | 0.533963 | 0.533963 | 0.127248 | pfd_r4_tau64000_MAH00448_ratio50 | 1.000000 |

#### MAH00449_ratio100

| Method | AUC_best | DA_best | F1 | BestTagByDA | Threshold@Best-DA |
|---|---:|---:|---:|---|---:|
| n149 | 0.996419 | 0.987328 | 0.987294 | n149_r5_tau256000_MAH00449_ratio100 | 1.952435 |
| n179 | 0.996027 | 0.987135 | 0.987099 | n179_s9_r4_tau256000 | 2.058011 |
| ebf | 0.991299 | 0.973003 | 0.972535 | ebf_r4_tau64000_MAH00449_ratio100 | 2.000000 |
| ynoise | 0.991037 | 0.974681 | 0.974260 | ynoise_r4_tau64000_MAH00449_ratio100 | 4.000000 |
| mlpf | 0.988184 | 0.978019 | 0.977958 | mlpf_r3_tau64000_MAH00449_ratio100 | 0.200000 |
| stcf | 0.981578 | 0.960739 | 0.959760 | stcf_r3_tau32000_MAH00449_ratio100 | 2.000000 |
| baf | 0.925423 | 0.925423 | 0.923830 | baf_r1_tau32000_MAH00449_ratio100 | 1.000000 |
| ts | 0.902910 | 0.876614 | 0.878494 | ts_r2_tau32000_MAH00449_ratio100 | 0.100000 |
| evflow | 0.634479 | 0.634479 | 0.423947 | evflow_r2_tau8000_MAH00449_ratio100 | 64.000000 |
| knoise | 0.617642 | 0.617548 | 0.385716 | knoise_r1_tau32000_MAH00449_ratio100 | 1.000000 |
| pfd | 0.506968 | 0.506968 | 0.027490 | pfd_r4_tau64000_MAH00449_ratio100 | 1.000000 |

#### MAH00449_ratio50

| Method | AUC_best | DA_best | F1 | BestTagByDA | Threshold@Best-DA |
|---|---:|---:|---:|---|---:|
| n149 | 0.997136 | 0.988204 | 0.990935 | n149_r4_tau256000_MAH00449_ratio50 | 1.548153 |
| n179 | 0.996765 | 0.987632 | 0.991398 | n179_s9_r4_tau256000 | 1.467431 |
| ebf | 0.993063 | 0.978848 | 0.981925 | ebf_r4_tau64000_MAH00449_ratio50 | 1.500000 |
| ynoise | 0.992857 | 0.980218 | 0.982946 | ynoise_r4_tau64000_MAH00449_ratio50 | 3.000000 |
| mlpf | 0.988126 | 0.979727 | 0.985676 | mlpf_r3_tau64000_MAH00449_ratio50 | 0.200000 |
| stcf | 0.983909 | 0.964652 | 0.965253 | stcf_r3_tau32000_MAH00449_ratio50 | 2.000000 |
| baf | 0.942445 | 0.942445 | 0.962777 | baf_r2_tau32000_MAH00449_ratio50 | 1.000000 |
| ts | 0.932123 | 0.908555 | 0.946326 | ts_r2_tau64000_MAH00449_ratio50 | 0.100000 |
| knoise | 0.634795 | 0.634678 | 0.429285 | knoise_r1_tau32000_MAH00449_ratio50 | 1.000000 |
| evflow | 0.634414 | 0.634414 | 0.423755 | evflow_r2_tau8000_MAH00449_ratio50 | 64.000000 |
| pfd | 0.506931 | 0.506931 | 0.027343 | pfd_r4_tau64000_MAH00449_ratio50 | 1.000000 |


## E. LED 数据集

## 22. 快速导航（建议先看）

当前“可用于对外结论”的优先阅读顺序：
1. `18.6`：`scene_100` 全量复核（统一指标口径）
1. `18.8`：`scene_1004` 全量复核（已覆盖旧迁移实验）
1. `18.9`：`scene_1032` 全量复核（中间噪声强度）
1. `18.10`：其余 7 个 LED 场景全量复核（本轮新增）
1. `18.11`：LED 10 场景全量汇总总表（本轮新增）

历史过程记录（保留但不建议直接作为最终结论）：
1. `18.2`、`18.3`：`LED-300k` 历史口径。

## 23. 脚本使用方法（完整）

### 15.1 ED24 新算法总入口（可选一个或多个）

`run_slomo_alg.ps1` 支持全量、单算法、多算法：

```powershell
# 默认：跑全部（knoise/evflow/ynoise/ts/mlpf/pfd）
powershell -ExecutionPolicy Bypass -File ./scripts/ED24_alg_evalu/run_slomo_alg.ps1

# 单算法（兼容参数）
powershell -ExecutionPolicy Bypass -File ./scripts/ED24_alg_evalu/run_slomo_alg.ps1 -Algorithm knoise

# 多算法
powershell -ExecutionPolicy Bypass -File ./scripts/ED24_alg_evalu/run_slomo_alg.ps1 -Algorithms knoise,ts,mlpf,pfd

# 显式全量
powershell -ExecutionPolicy Bypass -File ./scripts/ED24_alg_evalu/run_slomo_alg.ps1 -Algorithms all

# EVFLOW 连续扫频（推荐）
powershell -ExecutionPolicy Bypass -File ./scripts/ED24_alg_evalu/run_slomo_alg.ps1 -Algorithms evflow -SweepProfile dense

# 多算法连续扫频
powershell -ExecutionPolicy Bypass -File ./scripts/ED24_alg_evalu/run_slomo_alg.ps1 -Algorithms evflow,ts,mlpf,pfd -SweepProfile dense
```

### 15.1.1 Driving 新算法总入口（支持 dense）

```powershell
# 单算法（coarse，默认）
powershell -ExecutionPolicy Bypass -File ./scripts/driving_alg_evalu/run_driving_alg.ps1 -Algorithm evflow

# EVFLOW 连续扫频（推荐）
powershell -ExecutionPolicy Bypass -File ./scripts/driving_alg_evalu/run_driving_alg.ps1 -Algorithm evflow -SweepProfile dense

# 其他算法 dense 示例
powershell -ExecutionPolicy Bypass -File ./scripts/driving_alg_evalu/run_driving_alg.ps1 -Algorithm ts -SweepProfile dense
powershell -ExecutionPolicy Bypass -File ./scripts/driving_alg_evalu/run_driving_alg.ps1 -Algorithm pfd -SweepProfile dense

# 真实 MLPF（按 level 自动套模型）
powershell -ExecutionPolicy Bypass -File ./scripts/driving_alg_evalu/run_driving_alg.ps1 -Algorithm mlpf -SweepProfile coarse -MlpfModelPattern "data/DND21/mydriving/MLPF/mlpf_torch_{level}.pt"

# N149（独立脚本，200k）
powershell -ExecutionPolicy Bypass -File ./scripts/driving_alg_evalu/run_driving_n149.ps1
```

### 15.2 ED24 传统算法入口

```powershell
powershell -ExecutionPolicy Bypass -File ./scripts/ED24_alg_evalu/run_slomo_baf.ps1
powershell -ExecutionPolicy Bypass -File ./scripts/ED24_alg_evalu/run_slomo_stcf.ps1
powershell -ExecutionPolicy Bypass -File ./scripts/ED24_alg_evalu/run_slomo_ebf.ps1
powershell -ExecutionPolicy Bypass -File ./scripts/ED24_alg_evalu/run_slomo_ebf_paper_s_tau.ps1
powershell -ExecutionPolicy Bypass -File ./scripts/ED24_alg_evalu/run_slomo_n149.ps1
```

### 15.3 横向汇总脚本

```powershell
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/summarize_horizontal_all.py
```
说明：
- 上面一条命令会自动依次运行：
`summarize_horizontal_round1.py` + `summarize_horizontal_round2_new_methods.py`
- 并输出合并文件：
`data/ED24/myPedestrain_06/horizontal_summary_all.csv`
- 同时会自动读取：
`data/ED24/myPedestrain_06/bestpoint_mesr_aocc_summary.csv`
并把 `best-f1` 对应的 `mesr/aocc` 回填到总表。
- 回填策略是“有新值才覆盖、无值不改”，避免把已有算法结果误清空。

STCF 口径确认（按代码）：
- `src/myevs/denoise/ops/stc.py` 中使用 `last_on` / `last_off` 两张图分别计时。
- 当前事件只在“同极性”邻域上计数（并且不包含中心像素），不是异极性累计。

### 15.4 MESR/AOCC 后评估脚本（支持一个或多个算法）

```powershell
# ED24：默认所有算法、三档噪声、best-AUC+best-F1，同时计算 MESR/AOCC
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/eval_bestpoint_mesr_aocc.py --dataset ed24

# ED24：只算一个算法（例如 baf）
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/eval_bestpoint_mesr_aocc.py --dataset ed24 --algorithms baf

# ED24：一次算多个算法
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/eval_bestpoint_mesr_aocc.py --dataset ed24 --algorithms baf,stcf,ebf,knoise

# ED24：只算 PFD
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/eval_bestpoint_mesr_aocc.py --dataset ed24 --algorithms pfd

# ED24：PFD-B（实现已支持，后续闪烁噪声数据集使用）
D:/software/Anaconda_envs/envs/myEVS/python.exe -m myevs.cli roc --clean <clean.npy> --noisy <noisy.npy> --assume npy --width 346 --height 260 --method pfd --pfd-mode b --engine numba --radius-px 1 --time-us 32000 --refractory-us 1 --param min-neighbors --values 1,2,3 --out-csv <out.csv>

# ED24：真实 MLPF（按 level 自动套模型）
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/eval_bestpoint_mesr_aocc.py --dataset ed24 --algorithms mlpf --mlpf-model-pattern "data/ED24/myPedestrain_06/MLPF/mlpf_torch_{level}.pt"

# Driving：只算 light/light_mid/mid，且只算 best-F1
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/eval_bestpoint_mesr_aocc.py --dataset driving --levels light,light_mid,mid --points best-f1

# 指标可选：只算 MESR 或只算 AOCC
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/eval_bestpoint_mesr_aocc.py --dataset ed24 --metrics mesr
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/eval_bestpoint_mesr_aocc.py --dataset ed24 --metrics aocc
```

默认输出：
- ED24：`data/ED24/myPedestrain_06/bestpoint_mesr_aocc_summary.csv`
- Driving：`data/DND21/mydriving/bestpoint_mesr_aocc_summary.csv`

写入策略（避免互相覆盖）：
- 采用“增量更新”模式：默认保留历史结果。
- 仅替换本次运行涉及到的行（键：`dataset + algorithm + level + point`）。
- 例如先跑 `baf` 再跑 `stcf`，两者会同时保留；再次跑 `stcf` 才会覆盖旧 `stcf` 行。

### 15.5 MLPF 训练与真实推理

```powershell
# 1) 训练 ED24（按 light / mid / heavy）
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/train_mlpf_torch.py --clean "D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_1.8_signal_only.npy" --noisy "D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_1.8.npy" --width 346 --height 260 --tick-ns 1000 --duration-us 128000 --patch 7 --epochs 8 --batch-size 512 --max-events 200000 --out-ts "data/ED24/myPedestrain_06/MLPF/mlpf_torch_light.pt" --out-meta "data/ED24/myPedestrain_06/MLPF/mlpf_torch_light.json"
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/train_mlpf_torch.py --clean "D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_2.5_signal_only.npy" --noisy "D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_2.5.npy" --width 346 --height 260 --tick-ns 1000 --duration-us 128000 --patch 7 --epochs 8 --batch-size 512 --max-events 200000 --out-ts "data/ED24/myPedestrain_06/MLPF/mlpf_torch_mid.pt" --out-meta "data/ED24/myPedestrain_06/MLPF/mlpf_torch_mid.json"
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/train_mlpf_torch.py --clean "D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3_signal_only.npy" --noisy "D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_3.3.npy" --width 346 --height 260 --tick-ns 1000 --duration-us 128000 --patch 7 --epochs 8 --batch-size 512 --max-events 200000 --out-ts "data/ED24/myPedestrain_06/MLPF/mlpf_torch_heavy.pt" --out-meta "data/ED24/myPedestrain_06/MLPF/mlpf_torch_heavy.json"

# 2) 用真实模型跑 ED24 MLPF ROC
powershell -ExecutionPolicy Bypass -File ./scripts/ED24_alg_evalu/run_slomo_alg.ps1 -Algorithm mlpf -SweepProfile coarse -MlpfModelPattern "data/ED24/myPedestrain_06/MLPF/mlpf_torch_{level}.pt"

# 3) 训练 Driving（按 light / light_mid / mid）
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/train_mlpf_torch.py --clean "D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving/driving_noise_light_slomo_shot_withlabel/driving_noise_light_signal_only.npy" --noisy "D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving/driving_noise_light_slomo_shot_withlabel/driving_noise_light_labeled.npy" --width 346 --height 260 --tick-ns 1000 --duration-us 128000 --patch 7 --epochs 8 --batch-size 512 --max-events 200000 --out-ts "data/DND21/mydriving/MLPF/mlpf_torch_light.pt" --out-meta "data/DND21/mydriving/MLPF/mlpf_torch_light.json"
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/train_mlpf_torch.py --clean "D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving/driving_noise_light_mid_slomo_shot_withlabel/driving_noise_light_mid_signal_only.npy" --noisy "D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving/driving_noise_light_mid_slomo_shot_withlabel/driving_noise_light_mid_labeled.npy" --width 346 --height 260 --tick-ns 1000 --duration-us 128000 --patch 7 --epochs 8 --batch-size 512 --max-events 200000 --out-ts "data/DND21/mydriving/MLPF/mlpf_torch_light_mid.pt" --out-meta "data/DND21/mydriving/MLPF/mlpf_torch_light_mid.json"
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/train_mlpf_torch.py --clean "D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving/driving_noise_mid_slomo_shot_withlabel/driving_noise_mid_signal_only.npy" --noisy "D:/hjx_workspace/scientific_reserach/dataset/DND21/mydriving/driving_noise_mid_slomo_shot_withlabel/driving_noise_mid_labeled.npy" --width 346 --height 260 --tick-ns 1000 --duration-us 128000 --patch 7 --epochs 8 --batch-size 512 --max-events 200000 --out-ts "data/DND21/mydriving/MLPF/mlpf_torch_mid.pt" --out-meta "data/DND21/mydriving/MLPF/mlpf_torch_mid.json"

# 4) 用真实模型跑 Driving MLPF ROC
powershell -ExecutionPolicy Bypass -File ./scripts/driving_alg_evalu/run_driving_alg.ps1 -Algorithm mlpf -SweepProfile coarse -MlpfModelPattern "data/DND21/mydriving/MLPF/mlpf_torch_{level}.pt"
```

### 15.5.1 Driving MLPF（真实模型）重跑结果（2026-04-28）

数据来源：
- `data/DND21/mydriving/MLPF/roc_mlpf_{level}.csv`
- `data/DND21/mydriving/MLPF/runtime_mlpf.csv`（按 level 取最新一条）
- `data/DND21/mydriving/bestpoint_mesr_aocc_summary.csv`（best-AUC / best-F1）

| Level | Best AUC | Best AUC Tag | Best AUC Threshold | MESR@Best-AUC | AOCC@Best-AUC | Best F1 | Best F1 Tag | Best F1 Threshold | MESR@Best-F1 | AOCC@Best-F1 | Runtime(s) |
|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| light | 0.472022 | mlpf_tau32000 | 0.8 | 0.711261 | 2.244642 | 0.972323 | mlpf_tau64000 | 0.8 | 0.705434 | 2.236060 | 446.578 |
| light_mid | 0.439639 | mlpf_tau32000 | 0.1 | 0.681240 | 1.981077 | 0.916399 | mlpf_tau32000 | 0.1 | 0.681240 | 1.981077 | 448.517 |
| mid | 0.498857 | mlpf_tau256000 | 0.1 | 0.666617 | 1.777021 | 0.861026 | mlpf_tau512000 | 0.2 | 0.673816 | 1.883022 | 450.010 |

说明：
1. 该表是“真实模型推理模式”结果（已传 `--mlpf-model`），不是 proxy 回退模式。
2. 当前 AUC 偏低的直接现象是：大量阈值点靠近“全保留”区（FPR/TPR 同时较高），说明该训练流程在 driving 上判别力不足，后续需继续优化训练策略（样本均衡/损失重加权/更强特征）。

### 15.5.2 Driving N149（200k）补跑结果（2026-04-28）

数据来源：
- `data/DND21/mydriving/N149_sigma_scan/summary_sigma_scan.csv`
- `data/DND21/mydriving/N149_sigma_scan/selected_config.csv`
- `data/DND21/mydriving/N149_selected/roc_n149_{light,light_mid,mid}.csv`
- `data/DND21/mydriving/N149/runtime_n149.csv`（由 `N149_selected` 同步）
- `data/DND21/mydriving/bestpoint_mesr_aocc_summary.csv`（已补 n149 的 best-AUC / best-F1）

| Level | Best AUC | Best AUC Tag | Threshold@Best-AUC | F1@Best-AUC | MESR@Best-AUC | AOCC@Best-AUC | Best F1 | Best F1 Tag | Threshold@Best-F1 | AUC@Best-F1 | MESR@Best-F1 | AOCC@Best-F1 | Runtime(s) |
|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| light | 0.932978 | n149_r2_tau32000_light | 0.259854 | 0.981619 | 0.725054 | 2.402036 | 0.982559 | n149_r3_tau64000_light | 0.476634 | 0.926168 | 0.718161 | 2.335562 | 26.308 |
| light_mid | 0.936847 | n149_r2_tau32000_light_mid | 0.747070 | 0.958057 | 0.720106 | 2.356677 | 0.958849 | n149_r3_tau32000_light_mid | 0.870674 | 0.936600 | 0.710368 | 2.272940 | 25.970 |
| mid | 0.938152 | n149_r2_tau32000_mid | 1.071131 | 0.937040 | 0.718285 | 2.311529 | 0.937725 | n149_r3_tau32000_mid | 1.153651 | 0.938016 | 0.700190 | 2.158709 | 25.899 |

说明：
1. N149 在 driving 三档下 AUC 更新为 `0.933~0.938`（本轮 sigma 扫描后选优，`sigma=1.5`）。
2. 与旧版仅 `r=3/4/5` 的结论相比，本轮加入 `r=2` 后，三档 AUC 均提升，说明之前主要不是阈值没扫够，而是半径覆盖不足。
3. `MESR/AOCC` 已按统一规则补齐：`best-AUC` 与 `best-F1` 两个工作点都已记录。

### 15.6 N147（EBF_Part2）示例

```powershell
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py --variant n147 --max-events 400000 --s-list 5,7,9 --tau-us-list 32000,64000,128000,256000 --out-dir data/ED24/myPedestrain_06/EBF_Part2/_slim_n147_794_compact400k --esr-mode off --aocc-mode off
```
### 15.7 N147 增加了通用绘图脚本（直接吃 ROC CSV）
```powershell
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24_alg_evalu/plot_roc_with_auc.py `
  --in-csv data/ED24/myPedestrain_06/EBF_Part2/_slim_n147_794_compact400k/roc_ebf_n147_heavy_labelscore_s5_7_9_tau32_64_128_256ms.csv `
  --out-png data/ED24/myPedestrain_06/EBF_Part2/_slim_n147_794_compact400k/roc_ebf_n147_heavy_auclegend.png `
  --title "N147 ROC" `
  --strip-env-suffix
```
## 24. ED24 新增 2.1v（light_mid）补跑记录（2026-05-03）

### 20.1 数据转换（CSV -> NPY）

原始文件：
- `D:\hjx_workspace\scientific_reserach\dataset\ED24\myPedestrain_06\Pedestrain_06_2.1.csv`

转换命令：

```powershell
D:/software/Anaconda_envs/envs/myEVS/python.exe scripts/ED24csv_to_npy.py `
  --in "D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_2.1.csv" `
  --out "D:/hjx_workspace/scientific_reserach/dataset/ED24/myPedestrain_06/Pedestrain_06_2.1.npy" `
  --tick-ns 1000 --timestamp-unit us --width 346 --height 260 `
  --signal-label-value 0 --overwrite
```

信号-only 文件：
- `D:\hjx_workspace\scientific_reserach\dataset\ED24\myPedestrain_06\Pedestrain_06_2.1_signal_only.npy`

本次统计（转换输出）：
- events_total=`297299`
- signal_total=`162926`
- noise_total=`134373`

### 20.2 运行入口（全算法扫频）

本次为单 level（`light_mid`）全算法补跑，入口脚本：
- `scripts/ED24_alg_evalu/run_slomo_21_all.ps1`
- `scripts/ED24_alg_evalu/run_slomo_21_remaining.ps1`
- `scripts/ED24_alg_evalu/run_slomo_21_tail6.ps1`

统一结果目录：
- `data/ED24/myPedestrain_06/{ALG}/roc_{alg}_light_mid.csv`
- `data/ED24/myPedestrain_06/{ALG}/roc_{alg}_light_mid.png`

汇总 CSV（本节主表来源）：
- `data/ED24/myPedestrain_06/horizontal_light_mid_summary.csv`

### 20.3 light_mid（2.1v）横向汇总（10算法）

| Algorithm | AUC_best | AUC_best_tag | AUC_best_threshold | F1_best | F1_best_tag | F1_best_threshold | Runtime_sec |
|---|---:|---|---:|---:|---|---:|---:|
| N149 | 0.950059 | n149_r5_tau256000_light_mid | inf | 0.905392 | n149_r5_tau256000_light_mid | 1.481252 | 54.625 |
| EBF | 0.930014 | ebf_r5_tau256000 | 0.0 | 0.887646 | ebf_r4_tau128000 | 2.0 | 14.021 |
| STCF | 0.925090 | stcf_r3 | 100.0 | 0.874169 | stcf_r3 | 64000.0 | 102.409 |
| YNOISE | 0.922690 | ynoise_r4_tau64000 | 1.0 | 0.882852 | ynoise_r5_tau128000 | 6.0 | 515.993 |
| MLPF | 0.896621 | mlpf_tau128000 | 2.0 | 0.862411 | mlpf_tau256000 | 4.0 | 486.952 |
| PFD | 0.896562 | pfd_r3_tau32000_m1 | 1.0 | 0.867869 | pfd_r3_tau32000_m2 | 1.0 | 42.084 |
| BAF | 0.877417 | baf_r2 | 20.0 | 0.837223 | baf_r2 | 32000.0 | 92.871 |
| TS | 0.861908 | ts_r2_decay32000 | 0.05 | 0.832228 | ts_r2_decay32000 | 0.05 | 18.068 |
| EVFLOW | 0.817695 | evflow_r4_tau32000 | 0.0 | 0.840140 | evflow_r5_tau32000 | 80.0 | 921.576 |
| KNOISE | 0.693640 | knoise_tau32000 | 1.0 | 0.576794 | knoise_tau32000 | 1.0 | 23.717 |

说明：
1. `Runtime_sec` 取各算法 `runtime_{alg}.csv` 中 `level=light_mid` 的最新记录。
2. `N149` 的 `AUC_best_threshold=inf` 来自其 score-threshold 曲线的端点（全拒绝基线点）；F1 最优点阈值见 `F1_best_threshold`。
## 25. ED24 Bicycle_02 清理说明

为避免 ED24 内容重复与分散，原 `Bicycle_02` 的独立章节已合并到：
- `10.3B Bicycle_02（ED24，同组展示）`

原始汇总 CSV 与脚本入口保持不变：
- `data/ED24/myBicycle_02/horizontal_bicycle02_summary.csv`
- `scripts/ED24_alg_evalu/run_bicycle02_alg.ps1`

## 26. N179 跨数据集总表（含 EBF/N149 对比）

说明：
1. 按数据集分开列出，不混在一张表中。
2. `k_sfrac / k_mix / beta_init` 仅对 N179 有效；`EBF/N149` 记为 `-`。
3. N179 若 tag 中未显式写 `ks/km`，表示使用默认：`k_sfrac=0.8, k_mix=0.125, beta_init=default`。
4. 本节主表只保留“与既有基线完全同口径”的结果；`27.*` 中的 N179-K 网格结果作为参数诊断，不直接替换主榜单。

### 26.1 ED24（myPedestrain_06）

| Split | Method | AUC_best | F1_best | BestTag | k_sfrac | k_mix | beta_init |
|---|---|---:|---:|---|---:|---:|---|
| light | EBF | 0.933627 | 0.951967 | ebf_r5_tau512000 | - | - | - |
| light | N149 | 0.953997 | 0.959438 | n149_r5_tau512000_light | - | - | - |
| light | N179 | 0.954849 | 0.956303 | n179_s9_r4_tau256000 | 0.800 | 0.125 | default |
| light_mid | N179 | 0.948602 | 0.903832 | n179_s9_r4_tau256000 | 0.800 | 0.125 | default |
| mid | EBF | 0.918293 | 0.812238 | ebf_r5_tau128000 | - | - | - |
| mid | N149 | 0.946041 | 0.839209 | n149_r5_tau256000_mid | - | - | - |
| mid | N179 | 0.944215 | 0.836136 | n179_s9_r4_tau256000 | 0.800 | 0.125 | default |
| heavy | EBF | 0.890842 | 0.756961 | ebf_r3_tau64000 | - | - | - |
| heavy | N149 | 0.934184 | 0.788696 | n149_r5_tau128000_heavy | - | - | - |
| heavy | N179 | 0.938908 | 0.780698 | n179_s9_r4_tau256000 | 0.800 | 0.125 | default |

### 26.2 ED24（myBicycle_02）

| Split | Method | AUC_best | F1_best | BestTag | k_sfrac | k_mix | beta_init |
|---|---|---:|---:|---|---:|---:|---|
| light | EBF | 0.980557 | 0.973217 | ebf_r5_tau128000 | - | - | - |
| light | N149 | 0.986759 | 0.975852 | n149_r5_tau256000_light | - | - | - |
| light | N179 | 0.985747 | 0.974581 | n179_s9_r4_tau256000 | 0.800 | 0.125 | default |
| light_mid | EBF | 0.975204 | 0.943114 | ebf_r5_tau128000 | - | - | - |
| light_mid | N149 | 0.984161 | 0.950202 | n149_r5_tau512000_light_mid | - | - | - |
| light_mid | N179 | 0.983211 | 0.949102 | n179_s9_r4_tau256000 | 0.800 | 0.125 | default |
| mid | EBF | 0.970035 | 0.904248 | ebf_r5_tau128000 | - | - | - |
| mid | N149 | 0.980782 | 0.917088 | n149_r5_tau256000_mid | - | - | - |
| mid | N179 | 0.979715 | 0.914409 | n179_s9_r4_tau256000 | 0.800 | 0.125 | default |

### 26.3 Driving（mydriving）

| Split | Method | AUC_best | F1_best | BestTag | k_sfrac | k_mix | beta_init |
|---|---|---:|---:|---|---:|---:|---|
| light | EBF | 0.937124 | 0.987540 | ebf_r2_tau16000 | - | - | - |
| light | N149 | 0.932978 | 0.982559 | n149_r2_tau32000_light | - | - | - |
| light | N179 | 0.908441 | 0.977785 | n179_s5_r2_tau32000 | 0.800 | 0.125 | default |
| light_mid | EBF | 0.937867 | 0.971160 | ebf_r2_tau16000 | - | - | - |
| light_mid | N149 | 0.936847 | 0.958849 | n149_r2_tau32000_light_mid | - | - | - |
| light_mid | N179 | 0.910494 | 0.949475 | n179_s5_r2_tau32000 | 0.800 | 0.125 | default |
| mid | EBF | 0.937166 | 0.956357 | ebf_r2_tau16000 | - | - | - |
| mid | N149 | 0.938152 | 0.937725 | n149_r2_tau32000_mid | - | - | - |
| mid | N179 | 0.911717 | 0.928930 | n179_s5_r2_tau32000 | 0.800 | 0.125 | default |

### 26.4 DVSCLEAN（10 子集均值）

| Split | Method | AUC_best | F1_best | BestTag | k_sfrac | k_mix | beta_init |
|---|---|---:|---:|---|---:|---:|---|
| all(mean10) | EBF | 0.993967 | 0.984271 | (mean) | - | - | - |
| all(mean10) | N149 | 0.996962 | 0.990038 | (mean) | - | - | - |
| all(mean10) | N179 | 0.996602 | 0.989815 | (mean from n179 summary) | 0.800 | 0.125 | default |

### 26.5 LED（10 场景均值）

| Split | Method | AUC_best | F1_best | BestTag | k_sfrac | k_mix | beta_init |
|---|---|---:|---:|---|---:|---:|---|
| all(mean10) | EBF | 0.856081 | 0.857951 | (mean) | - | - | - |
| all(mean10) | N149 | 0.894114 | 0.884181 | (mean) | - | - | - |
| all(mean10) | N179 | 0.894045 | 0.960161 | (mean from n179 summary) | 0.800 | 0.125 | default |

## 27. N179 K 扫频（已修正，仅保留同口径有效结果）

修正说明：
1. 原 27.2/27.3/27.4（Driving、DVSCLEAN、LED）使用的是参数诊断链路，与第26节主评测链路口径不一致，已从主文档移除，避免误导结论。
2. 第27节现在只保留 ED24（myPedestrain_06）的同口径 K 扫频结果，用于解释 K 对 N179 的影响。
3. 跨数据集对比与最终结论统一以第26节为准。

### 27.1 ED24（myPedestrain_06）固定历史最优 tau 的 K 扫频（s=9）

固定 tau：
- light=512000us
- light_mid=256000us
- mid=256000us
- heavy=128000us

扫频网格：
- k_sfrac = 0, 0.1, ..., 1.0
- k_mix = 0, 0.1, ..., 1.0

结果文件：
- `data/ED24/myPedestrain_06/N179/k_expand_tauprior_step01_20260506/k_expand_tauprior_step01_best_summary.csv`

| level | tau_us | BestTag | AUC_best | F1_best | threshold | k_sfrac | k_mix |
|---|---:|---|---:|---:|---:|---:|---:|
| light | 512000 | ebf_n179_labelscore_s9_tau512000_ks0_km1 | 0.952351 | 0.958846 | 0.834975 | 0.0 | 1.0 |
| light_mid | 256000 | ebf_n179_labelscore_s9_tau256000_ks0_km1 | 0.948783 | 0.903419 | 1.528535 | 0.0 | 1.0 |
| mid | 256000 | ebf_n179_labelscore_s9_tau256000_ks0p3_km0 | 0.944476 | 0.836372 | 3.129318 | 0.3 | 0.0 |
| heavy | 128000 | ebf_n179_labelscore_s9_tau128000_ks0p4_km0 | 0.934958 | 0.788114 | 3.140208 | 0.4 | 0.0 |

### 27.2 与固定K版本对照（同一数据集）

对照第26.1中固定K（k_sfrac=0.8, k_mix=0.125）的 N179：
- light：K-sweep 略低于固定K主结果。
- light_mid：K-sweep 与固定K主结果接近。
- mid/heavy：K-sweep 与固定K主结果基本一致。

结论：
1. ED24 行人集上，N179 对 K 存在“按噪声档位分化”的偏好，不同 level 的最优 K 不同。
2. 仅扫 K 并不能稳定带来全档位增益；主榜单仍应使用第26节统一口径结果。
3. 如果后续要把 K-tuned 作为正式结果，必须在每个数据集使用与第26节一致的 fullrun 评测链路重跑。

### 27.3 ED24 行人三阶段重跑（同口径，2026-05-06）

流程（仅 ED24/myPedestrain_06）：
1. Stage-1：固定 `k_sfrac=0.8, k_mix=0.125`，扫 `s∈{5,7,9}` 与 `tau∈{16k,32k,64k,128k,256k,512k}`，取 `AUC` 最优 `tau*`。
2. Stage-2：固定 `s,tau*`，扫 `k_sfrac∈{0.4,0.55,0.67,0.8,0.9}`、`k_mix∈{0,0.04,0.08,0.125,0.16,0.3,1.0}`。
3. Stage-3：用 Stage-2 最优组合重跑最终 ROC 与 bestpoint。

结果文件：
- `data/ED24/myPedestrain_06/N179/tune3/final_best_by_level.csv`

| level | s | tau_us | k_sfrac | k_mix | AUC_best | F1_best | BestTag |
|---|---:|---:|---:|---:|---:|---:|---|
| light | 9 | 256000 | 0.40 | 1.00 | 0.954964 | 0.956509 | n179s3_s9_r4_tau256000_ks400_km1000_bdef |
| light_mid | 9 | 256000 | 0.40 | 0.30 | 0.948746 | 0.903750 | n179s3_s9_r4_tau256000_ks400_km300_bdef |
| mid | 9 | 256000 | 0.40 | 0.00 | 0.944467 | 0.836497 | n179s3_s9_r4_tau256000_ks400_km0_bdef |
| heavy | 9 | 256000 | 0.55 | 0.00 | 0.939231 | 0.781546 | n179s3_s9_r4_tau256000_ks550_km0_bdef |

对比固定K（26.1）结论：
1. 本轮三阶段同口径重跑后，N179 相比固定K版在四档噪声均有小幅提升（AUC/F1均提升）。
2. 提升幅度不大，说明 N179 在 ED24 行人上主要性能由 `s/tau` 决定，K 只能做“微调增益”。
3. 即便修正后，N179 在 `light/mid` 上仍略低于 N149；`heavy` 仍优于 N149（AUC）。

### 27.4 N179 分数据集 K 推荐（用于“尽量接近 N149”）

结果汇总文件：
- `data/summary/n179_k_reco_mainchain_20260506.csv`

可信口径（可直接用于后续全量复跑）：

| Dataset | Split | s | tau_us | k_sfrac | k_mix | AUC_best | F1_best | 说明 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| ED24-Ped | light | 9 | 256000 | 0.40 | 1.00 | 0.954964 | 0.956509 | 同口径三阶段重跑 |
| ED24-Ped | light_mid | 9 | 256000 | 0.40 | 0.30 | 0.948746 | 0.903750 | 同口径三阶段重跑 |
| ED24-Ped | mid | 9 | 256000 | 0.40 | 0.00 | 0.944467 | 0.836497 | 同口径三阶段重跑 |
| ED24-Ped | heavy | 9 | 256000 | 0.55 | 0.00 | 0.939231 | 0.781546 | 同口径三阶段重跑 |
| ED24-Bicycle | light | 9 | 256000 | 0.40 | 1.00 | 0.986017 | 0.974624 | 同口径重跑 |
| ED24-Bicycle | light_mid | 9 | 256000 | 0.40 | 1.00 | 0.983353 | 0.948629 | 同口径重跑 |
| ED24-Bicycle | mid | 9 | 256000 | 0.40 | 0.00 | 0.979819 | 0.914508 | 同口径重跑 |
| Driving | light | 5 | 32000 | 0.40 | 0.30 | 0.909106 | 0.977784 | 同口径重跑 |
| Driving | light_mid | 5 | 32000 | 0.40 | 0.30 | 0.911118 | 0.949554 | 同口径重跑 |
| Driving | mid | 5 | 32000 | 0.40 | 0.30 | 0.912254 | 0.929187 | 同口径重跑 |

可执行的“统一K简化建议”（优先保证稳定）：
1. ED24 全部子集先用：`k_sfrac=0.40`；`k_mix` 取 `light/light_mid=1.00`，`mid/heavy=0.00`。
2. Driving 三档先统一：`k_sfrac=0.40, k_mix=0.30`。
3. 若追求最简部署（单组K）：ED24 可先试 `k_sfrac=0.40, k_mix=0.30`，再按档位微调 `k_mix`。

DVSCLEAN / LED（fullrun 同口径 K 扫频，已补齐正式推荐）：
1. 扫频口径：固定各子集已有 fullrun 最优 `s/tau`，仅扫 `k_sfrac∈{0.4,0.55,0.67,0.8,0.9}`、`k_mix∈{0,0.04,0.08,0.125,0.16,0.3,1.0}`。
2. 结果文件：
- `data/summary/n179_k_reco_dvsclean_led_fullrun_20260506.csv`
- `data/summary/n179_k_reco_dvsclean_led_recommended_20260506.csv`
- `data/summary/n179_k_reco_dvsclean_led_gain_20260506.csv`
3. DVSCLEAN 标签口径修正：当前 converted_npy 中 signal 标签值为 `1`，本轮已按该口径重跑。

正式推荐 K（可直接用于后续复跑）：

| Dataset | Split | reco_s | reco_tau_us | reco_k_sfrac | reco_k_mix | AUC_mean | F1_mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| DVSCLEAN | ratio50 | 9 | 64000 | 0.90 | 0.00 | 0.996935 | 0.991478 |
| DVSCLEAN | ratio100 | 9 | 64000 | 0.90 | 0.00 | 0.996289 | 0.988166 |
| LED | 100ms | 5 | 16000 | 0.90 | 0.00 | 0.895297 | 0.960188 |

相对原 fixed-K 主结果（同口径）平均增益：
1. DVSCLEAN：`ΔAUC=+0.000010`，`ΔF1=+0.000007`（基本持平，说明原 K 已很接近最优）。
2. LED：`ΔAUC=+0.001251`，`ΔF1=+0.000027`（小幅增益）。

与 N149 / 固定K-N179 的直接对比（你关心的“有没有提升”）：

对比文件：
- `data/summary/compare_dvsclean_n149_n179_fixed_tuned.csv`
- `data/summary/compare_led_n149_n179_fixed_tuned.csv`

### 27.5 DVSCLEAN 三方均值对比（10 子集）

| Method | AUC_mean | F1_mean |
|---|---:|---:|
| N149 | 0.984085 | 0.990941 |
| N179 (fixed K) | 0.996602 | 0.989815 |
| N179 (K-tuned, fullrun) | 0.996612 | 0.989822 |

结论：
1. 相比固定K，`N179(K-tuned)` 在 DVSCLEAN 上是小幅提升（AUC/F1 都略增）。
2. 相比 N149，N179 在 DVSCLEAN 上 AUC 更高，但 F1 略低。

### 27.6 LED 三方均值对比（10 场景）

| Method | AUC_mean | F1_mean |
|---|---:|---:|
| N149 | 0.892707 | 0.961648 |
| N179 (fixed K) | 0.894045 | 0.960161 |
| N179 (K-tuned, fullrun) | 0.895297 | 0.960188 |

结论：
1. 相比固定K，`N179(K-tuned)` 在 LED 上 AUC 有明确提升，F1 基本持平。
2. 相比 N149，N179 在 LED 上 AUC 更高，但 F1 仍略低。

### 27.7 加入 EBF baseline 的四方对比（你关心的“相对 baseline 提升”）

对比文件：
- `data/summary/compare_dvsclean_ebf_n149_n179_fixed_tuned.csv`
- `data/summary/compare_led_ebf_n149_n179_fixed_tuned.csv`

DVSCLEAN（10 子集均值）：

| Method | AUC_mean | F1_mean |
|---|---:|---:|
| EBF (baseline) | 0.990256 | 0.976884 |
| N149 | 0.984085 | 0.990941 |
| N179 (fixed K) | 0.996602 | 0.989815 |
| N179 (K-tuned, fullrun) | 0.996612 | 0.989822 |

相对 EBF 的提升（N179 K-tuned）：
1. `ΔAUC = +0.006356`
2. `ΔF1 = +0.012938`

LED（10 场景均值）：

| Method | AUC_mean | F1_mean |
|---|---:|---:|
| EBF (baseline) | 0.826219 | 0.956225 |
| N149 | 0.892707 | 0.961648 |
| N179 (fixed K) | 0.894045 | 0.960161 |
| N179 (K-tuned, fullrun) | 0.895297 | 0.960188 |

相对 EBF 的提升（N179 K-tuned）：
1. `ΔAUC = +0.069078`
2. `ΔF1 = +0.003963`

结论（关于 K 是否有意义）：
1. 对“超过 EBF baseline”这个目标，N179 本身已经显著成立（尤其 AUC）。
2. K-tuned 相对 fixed-K 的收益确实很小，属于微调级别；它的价值更多是“稳定压榨最后一点性能”，不是主增益来源。

### 27.8 全局默认 K=`k_sfrac=0.40,k_mix=0.00` 对比

目的：验证一个全局简化 K 是否足够接近 tuned-K，并与 `EBF / N149 / N179 fixed-K` 对齐比较。

结果文件：
- `data/summary/compare_all_k0400_vs_fixed_tuned_ebf_n149.csv`

按数据集均值：

| Dataset | EBF AUC | N149 AUC | N179 fixed-K AUC | N179 K=0.40/0.00 AUC | N179 tuned-K(扫频最优) AUC | EBF F1 | N149 F1 | N179 fixed-K F1 | N179 K=0.40/0.00 F1 | N179 tuned-K F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ED24-Ped | 0.918194 | 0.946070 | 0.946643 | 0.946796 | 0.946852 | 0.852203 | 0.873184 | 0.869242 | 0.869430 | 0.869575 |
| ED24-Bicycle | 0.975265 | 0.983901 | 0.982891 | 0.982953 | 0.983063 | 0.940193 | 0.947714 | 0.946031 | 0.946071 | 0.945920 |
| Driving | 0.937386 | 0.935992 | 0.910217 | 0.910526 | 0.910826 | 0.971686 | 0.959711 | 0.952063 | 0.952081 | 0.952175 |
| DVSCLEAN | 0.990256 | 0.984085 | 0.996602 | 0.996590 | 0.996612 | 0.976884 | 0.990941 | 0.989815 | 0.989825 | 0.989822 |
| LED | 0.826219 | 0.892707 | 0.894045 | 0.893267 | 0.895297 | 0.956225 | 0.961648 | 0.960161 | 0.960180 | 0.960188 |

结论：
1. `k_sfrac=0.40,k_mix=0.00` 在所有已有 K-sweep 数据中都有记录，不是插值或推断。
2. 它相对 tuned-K 的差距很小：ED24/Driving/DVSCLEAN 基本贴近，LED AUC 平均低约 `0.00203`。
3. 它相对 fixed-K 没有明显退化，多数数据集略高或接近。
4. 相对 EBF：除 Driving 外，N179 `0.40/0.00` 在 AUC 上均保持优势；Driving 仍是 N179 当前主要短板。
5. 相对 N149：N179 `0.40/0.00` 在 ED24-Ped、DVSCLEAN、LED 的 AUC 上接近或更高；在 ED24-Bicycle、Driving 上低于 N149。

### 27.9 N179 默认 K 固化与其他系数处理原则

代码默认值已同步：
1. `src/myevs/denoise/ops/ebfopt_part2/n179_cpp_final_fast32_backbone.py`
2. `scripts/noise_analyze/sweep_n179_pair.py`
3. `scripts/ED24_alg_evalu/sweep_ebf_slim_labelscore_grid.py`

默认参数：

| Parameter | Default | 原因 |
|---|---:|---|
| `k_sfrac` | 0.40 | ED24/Driving 稳定，DVSCLEAN/LED 对 K 不敏感 |
| `k_mix` | 0.00 | 多数据集最优退化到 0，说明 mix gain 不是稳定增益项 |
| `rhythm_pressure_coeff` | 0.00 | DVSCLEAN 基本不敏感，LED 10 场景整体更优，ED24 混合但差值很小 |
| `rhythm_good_coeff` | 0.75 | `rp=0` 固定下扩展扫到 `1.0`，全局 AUC 均值在 `rg=0.75` 最高 |
| `support_good_coeff` | 0.00 | ED24(light/light_mid/mid) 全子集扫频均最优为 0，可直接去掉该项提升公式简洁性 |

关于 `rhythm_bad/rhythm_good/rhythm_pressure/rhythm_scale` 等固定系数：

结论：不建议把它们全部开放成常规 sweep 超参。

理由：
1. 这些系数处在同一条耦合链路里：`rhythm_bad/rstate -> rhythm_pressure -> alpha_eff/rhythm_scale -> u_eff -> score`。单独扫某一个系数，结果很容易依赖另一个固定系数，解释性差。
2. 继续开放会把 N179 从“低资源公式近似”变成“多超参数调参器”，论文上更难自圆其说。
3. 当前实验已经显示 K 的收益很小；再扫这些内部系数大概率只会得到数据集特化增益，不适合作为默认算法。

推荐处理方式：
1. 主算法固定这些系数，不进入常规 sweep。
2. 若必须验证，可做一次小规模消融，而不是全数据集扫参：
   `rhythm_pressure` 系数只取 `{0, 1/6, 1/3, 1/2}`；
   `rhythm_good` 抑制系数只取 `{0, 1/8, 1/4, 3/8, 1/2}`；
   代表数据集只选 ED24-Ped-heavy、Driving-mid、LED-scene_1004。
3. 只有当消融在至少两类数据集上带来超过 `+0.002 AUC` 或 `+0.002 F1` 的稳定提升，才值得把它升级为可调参数。

### 27.10 N179 rhythm 系数小规模消融脚本

目的：只验证内部固定系数是否有必要继续保留为可调参数，不作为常规 sweep，也不替代第 26 节主对比结果。

脚本：

```powershell
powershell -ExecutionPolicy Bypass -File ./scripts/noise_analyze/run_n179_rhythm_ablation_small.ps1
```

只跑某一个或多个代表集：

```powershell
powershell -ExecutionPolicy Bypass -File ./scripts/noise_analyze/run_n179_rhythm_ablation_small.ps1 -Dataset ed24
powershell -ExecutionPolicy Bypass -File ./scripts/noise_analyze/run_n179_rhythm_ablation_small.ps1 -Dataset ed24,driving
```

默认消融设置：

| Item | Value |
|---|---|
| `k_sfrac` | `0.4` |
| `k_mix` | `0.0` |
| `rhythm_pressure_coeff` | `0, 1/6, 1/3, 1/2` |
| `rhythm_good_coeff` | `0, 1/8, 1/4, 3/8, 1/2` |
| ED24 代表集 | `myPedestrain_06 heavy`, `s=9`, `tau=256000us` |
| Driving 代表集 | `mydriving mid`, `s=5`, `tau=32000us` |
| LED 代表集 | `scene_1004 100ms`, `s=7`, `tau=16000us` |
| 输出目录 | `data/summary/n179_rhythm_ablation_small/` |

输出文件：

| File | 内容 |
|---|---|
| `*_roc.csv` | 每个阈值点的 ROC 明细 |
| `*_summary.csv` | 每组系数的 AUC / best-F1 / threshold / MESR / AOCC |
| `*_runtime.csv` | 每组系数的运行时间与事件吞吐 |

判定规则：

1. 若新系数只在单一数据集提升，视为数据集特化，不进入默认算法。
2. 若在至少两类代表数据集上同时带来 `+0.002 AUC` 或 `+0.002 F1`，再考虑升级为正式参数。
3. 若收益低于上述阈值，继续保持当前默认：`rhythm_pressure_coeff=0`，`rhythm_good_coeff=3/4`。

本轮实测结果（2026-05-07）：

结果文件：
1. `data/summary/n179_rhythm_ablation_small/ed24_ped_heavy_summary.csv`
2. `data/summary/n179_rhythm_ablation_small/driving_mid_summary.csv`
3. `data/summary/n179_rhythm_ablation_small/led_scene1004_summary.csv`
4. 汇总：`data/summary/n179_rhythm_ablation_small/ablation_default_vs_best.csv`

| Dataset | Level | Default `(rp,rg)` | Default AUC | Default F1 | Best `(rp,rg)` | Best AUC | Best F1 | `ΔAUC` | `ΔF1` |
|---|---|---|---:|---:|---|---:|---:|---:|---:|
| ED24 / myPedestrain_06 | heavy | `(1/3, 1/4)` | 0.939174 | 0.781058 | `(0, 1/4)` | 0.939303 | 0.781526 | +0.000129 | +0.000468 |
| Driving / mydriving | mid | `(1/3, 1/4)` | 0.912064 | 0.928938 | `(1/2, 0)` | 0.913124 | 0.929416 | +0.001060 | +0.000477 |
| LED / scene_1004 | 100ms | `(1/3, 1/4)` | 0.852368 | 0.947731 | `(0, 1/4)` | 0.853848 | 0.947762 | +0.001480 | +0.000031 |

结论：
1. 三个代表集均有提升，但增益均小于升级门槛（`+0.002 AUC` 或 `+0.002 F1`）。
2. 该结果不支持将 `rhythm_pressure_coeff/rhythm_good_coeff` 升级为常规扫频参数。
3. 后续默认改为：`rhythm_pressure_coeff=0`，`rhythm_good_coeff=3/4`（保留参数接口以便回滚）。

补充验证（完整数据集级，对照 `rp=0` vs `rp=1/3`，固定 `rg=0.75`，2026-05-07，修正版）：

结果文件：
1. `data/summary/n179_rp_rg_compare_fullsets_fixed_20260507/rp_allrows.csv`
2. `data/summary/n179_rp_rg_compare_fullsets_fixed_20260507/rp_pairwise.csv`
3. `data/summary/n179_rp_rg_compare_fullsets_fixed_20260507/rp_dataset_agg.csv`

| Dataset | 子集数 | `AUC(rp=0)-AUC(rp=1/3)` 平均 | 最小 | 最大 | `F1(rp=0)-F1(rp=1/3)` 平均 |
|---|---:|---:|---:|---:|---:|
| DVSCLEAN | 10 | +0.000001000 | -0.000005000 | +0.000010000 | -0.000008000 |
| ED24（行人+自行车） | 7 | -0.000063000 | -0.000209000 | +0.000216000 | +0.000005000 |
| LED（10 场景） | 10 | +0.001091000 | +0.000433000 | +0.001733000 | +0.000055000 |

说明：
1. DVSCLEAN：`rp` 影响极小（约 `1e-6` 量级），可视为无效项。
2. LED：`rp=0` 在 10 场景上 AUC 一致更高（虽为小幅）。
3. ED24：两者互有胜负，但绝对差值很小，未出现必须保留 `rp=1/3` 的强证据。
4. 说明：旧版 DVSCLEAN 异常低分来自错误口径（分辨率用成 `346x260`），本节已改为 `1280x720 + signal_label=1` 重新统计。

### 27.11 固定 `rp=0` 后扫 `rg`（完整数据集，扩展到 1.0）

目的：在 `rp=0` 固定后，确认 `rg` 的推荐默认值。  
口径：ED24（行人+自行车）、DVSCLEAN（10 子集）、LED（10 场景）；每个子集固定该子集既有 best-AUC 的 `s/tau`，并固定 `k_sfrac=0.4,k_mix=0.0`，扫 `rg∈{0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1.0}`。

结果文件：
1. `data/summary/n179_rp_rg_compare_fullsets_fixed_20260507/rg_allrows.csv`
2. `data/summary/n179_rp_rg_compare_fullsets_fixed_20260507/rg_dataset_by_value.csv`
3. `data/summary/n179_rp_rg_compare_fullsets_fixed_20260507/rg_recommended_by_dataset.csv`
4. `data/summary/n179_rp_rg_compare_fullsets_fixed_20260507/rg_global_rank.csv`

按数据集推荐 `rg`（以 AUC 均值主排序，F1 次排序）：

| Dataset | 子集数 | 推荐 `rg` | AUC_mean | F1_mean |
|---|---:|---:|---:|---:|
| DVSCLEAN | 10 | 0.625 | 0.996598 | 0.989818 |
| ED24（行人+自行车） | 7 | 0.375 | 0.962201 | 0.902541 |
| LED（10 场景） | 10 | 0.875 | 0.896692 | 0.960252 |

全局均值排序：

| `rg` | AUC_mean | F1_mean |
|---:|---:|---:|
| 0.750 | 0.950597 | 0.956203 |
| 0.875 | 0.950589 | 0.956184 |
| 0.625 | 0.950588 | 0.956235 |
| 1.000 | 0.950582 | 0.956168 |
| 0.500 | 0.950442 | 0.956247 |

结论：
1. 在 `rp=0` 固定下，`rg` 的最优确实落在高值端（`0.375~1.0`）。
2. 扩展到 `1.0` 后，LED 继续提升，DVSCLEAN 在 `0.625` 附近达到最优，ED24 仍偏 `0.375`。
3. 若用单一全局值，`rg=0.75` 的全局 AUC 均值最高；若强调稳健与保守，可继续使用 `rg=0.5`（与 `0.75` 差距很小）。
4. `rg` 仍保留为可选超参，但默认无需大范围扫频。

### 27.12 固定 `rp=0, rg=0.75` 后扫支持项系数 `c_{sg}`（原式中的 `0.25`）

目标：验证
$$
S_i=B_i\cdot\left(1+b_i s_i(1+c_{sg}\,\text{good}_i)\right)
$$
里的 `c_{sg}` 是否有必要保留为超参数。

口径（按你当前重点）：仅 ED24 的 light/light_mid/mid，覆盖行人与自行车两套场景；每个子集固定该子集既有 best-AUC 的 `s/tau`，固定 `k_sfrac=0.4,k_mix=0,rp=0,rg=0.75`，扫 `c_{sg}∈{0,0.125,0.25,0.375,0.5,0.75,1.0}`。

结果文件：
1. `data/summary/n179_supportgood_scan_ed24_20260507/support_good_scan_rows.csv`
2. `data/summary/n179_supportgood_scan_ed24_20260507/support_good_scan_best_by_subset.csv`
3. `data/summary/n179_supportgood_scan_ed24_20260507/support_good_scan_global_rank.csv`

子集最优结果（6/6 全部一致）：

| scene | level | 最优 `c_{sg}` | AUC | F1 |
|---|---|---:|---:|---:|
| myPedestrain_06 | light | 0.0 | 0.954230 | 0.955754 |
| myPedestrain_06 | light_mid | 0.0 | 0.947981 | 0.903005 |
| myPedestrain_06 | mid | 0.0 | 0.943993 | 0.836604 |
| myBicycle_02 | light | 0.0 | 0.985370 | 0.973955 |
| myBicycle_02 | light_mid | 0.0 | 0.982949 | 0.948594 |
| myBicycle_02 | mid | 0.0 | 0.979636 | 0.914774 |

全局均值（6 子集）：

| `c_{sg}` | AUC_mean | F1_mean |
|---:|---:|---:|
| 0.000 | 0.965693 | 0.922114 |
| 0.125 | 0.965678 | 0.922102 |
| 0.250 | 0.965664 | 0.922086 |
| 0.375 | 0.965649 | 0.922059 |
| 0.500 | 0.965634 | 0.922036 |
| 0.750 | 0.965604 | 0.922005 |
| 1.000 | 0.965573 | 0.921970 |

结论：
1. 对 ED24 轻噪到中噪，`c_{sg}` 的最优是 `0`，可直接去掉该项。
2. 因此默认固定 `support_good_coeff=0`，公式更简洁且精度不降。

补充：DVSCLEAN / LED 全子集验证（固定 `rp=0, rg=0.75`，2026-05-07）

结果文件：
1. `data/summary/n179_supportgood_scan_dvsclean_led_20260507/support_good_scan_rows.csv`
2. `data/summary/n179_supportgood_scan_dvsclean_led_20260507/support_good_scan_best_by_subset.csv`
3. `data/summary/n179_supportgood_scan_dvsclean_led_20260507/support_good_scan_dataset_rank.csv`
4. `data/summary/n179_supportgood_scan_dvsclean_led_20260507/support_good_scan_reco_by_dataset.csv`
5. `data/summary/n179_supportgood_scan_dvsclean_led_20260507/support_good_scan_global_rank.csv`

按数据集推荐：

| Dataset | 子集数 | 推荐 `c_{sg}` | AUC_mean | F1_mean |
|---|---:|---:|---:|---:|
| DVSCLEAN | 10 | 0.0 | 0.996599 | 0.989815 |
| LED | 10 | 1.0 | 0.896794 | 0.960253 |

解释与建议：
1. `c_{sg}` 的影响在 DVSCLEAN/LED 上都很小（量级约 `1e-4`），属于“微调项”。
2. 如果目标是“公式简洁 + ED24 优先”，保持 `c_{sg}=0` 最合适。
3. 如果目标是“跨 LED 略微提 AUC”，可用 `c_{sg}=1`，但会引入一个解释成本更高的项。
