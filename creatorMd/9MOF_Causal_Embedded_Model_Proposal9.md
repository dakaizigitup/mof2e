# MOF Causal Embedded Model Proposal V9

> 版本定位：**V9（文献收敛版）**，基于最新因果/OOD思路的三阶段方案。  
> 目标：**ID 不显著退化、OOD 明显提升、机制可解释且可验证**。

---

## 0. 总体原则

V9 采用渐进式三阶段训练，不一次性上满所有因果约束：

1. **Stage 1**：先把原子表示学“干净”（语义稳定）
2. **Stage 2**：再学“跨环境不变”（抗伪相关）
3. **Stage 3**：最后做“机制分解 + 反事实一致性”（可解释因果）

全程只预测一个最终能量：

$$
E_{pred}
$$

但中间表示逐步加入因果约束。

---

## 1. Stage 1：原子语义地基（Atom Semantics）

### 1.1 结构

- eSCN backbone（沿用现有实现）
- 原子级辅助头：
  - `atom_type` 分类头
  - `masked_atom_recovery`（随机 mask 部分原子属性并恢复）

### 1.2 损失函数

$$
L_{s1}=L_{energy}+\lambda_{type}L_{atom\_type}+\lambda_{mask}L_{mask\_recon}
$$

说明：Stage 1 不引入复杂机制级约束，目标是获得稳定、可迁移的原子语义表示。

### 1.3 训练策略

- 先使用稳定版（detach 路线）训练到收敛；
- 再小步放开（去 detach，`lambda_type` 取小值，如 0.01–0.02）短程微调；
- 以验证集 `energy_mae` 最优 checkpoint 进入 Stage 2。

---

## 2. Stage 2：跨环境不变 + 子结构稳健（Invariance / OOD）

### 2.1 结构

- 三分支编码：
  - $Z_{site}$
  - $Z_{motif}$
  - $Z_{path}$
- 环境标签：`pure_CO2 / pure_H2O / mixed`
- 可选 `env_infer` 小头（仅训练期辅助，推理可丢弃）

### 2.2 核心损失

$$
L_{s2}=L_{energy}+\lambda_{inv}L_{IRM}+\lambda_{dec}L_{decorrel}+\lambda_{sub}L_{subgraph\_consistency}
$$

其中：

- $L_{IRM}$：跨环境共享最优方向（主因果项）
- $L_{decorrel}$：约束 $Z_{site}, Z_{motif}, Z_{path}$ 不坍缩为同一种信息
- $L_{subgraph\_consistency}$：同一分子不同子图扰动下预测保持一致

### 2.3 训练策略

- 冻结 Stage 1 低层（前 6 层），高层可训练；
- 每个 batch 保证多环境混合（IRM 才有有效信号）；
- 早停优先看 **OOD val MAE**（不是只看 ID）。

---

## 3. Stage 3：机制分解 + 反事实一致性（Mechanism / Counterfactual）

### 3.1 结构

- 冻结 Stage 2 encoders；
- 训练 `Causal Mixer + Cooperative Head`；
- 显式机制分解：
  - $E_{site}$
  - $E_{motif}$
  - $E_{path}$
  - $E_{coop}(h,c)$

最终组合：

$$
E_{pred}=E_{site}+E_{motif}+E_{path}+E_{coop}
$$

### 3.2 硬约束

保留解析约束：

$$
E_{coop}(0,c)=0,\quad E_{coop}(h,0)=0
$$

### 3.3 损失函数

$$
L_{s3}=L_{energy}+\lambda_{cf}L_{counterfactual}+\lambda_{phys}L_{physics}+\lambda_{inv}L_{IRM}
$$

其中：

- $L_{counterfactual}$：对 motif/path 做可控替换，约束变化方向符合化学先验
- $L_{physics}$：`extensive + 体积去相关 + cooperative zero`
- 保留小权重 $L_{IRM}$：抑制机制头过拟合单一环境

---

## 4. 与当前 V7 的关键差别

1. Stage 2 不再只是“加全局特征”，而是 **显式三分支 + 不变性约束**；
2. Stage 3 不再只是 mask 分支，而是 **可分解机制项 + 反事实检验**；
3. 评估从单 MAE 升级到：**ID / OOD / 机制一致性** 三维。

---

## 5. 验收门槛（实用版）

- **Stage 1**：`val/energy_mae` 明显优于均值基线
- **Stage 2**：OOD split 相对 ERM 提升 **≥ 8%–12%**
- **Stage 3**：满足 cooperative 硬约束，且反事实方向正确率显著高于随机

---

## 6. 后续落地清单

下一步可直接进入工程落地：

1. 三阶段 yml 参数表（可直接运行）
2. trainer 中新增损失的最小改动清单
3. OOD/机制一致性评估脚本模板

---

## 7. 版本说明

- 本文档为 **V9**（你确认的“非纯 V8、文献收敛版”）
- 可与 Proposal7 / Proposal8 并行作为独立路线

---

## 8. V9.1 修订记录（实现侧反馈回写）

> 在 V9 工程落地（Plan_V9 / 代码 escn_v9_causal.py、causal_trainer_v9.py）阶段，
> 对最初 §1–§5 中存在歧义或会导致工程失败的几处技术决策做了修正。
> 本节是 V9 → V9.1 的差异说明，**不是新方案**。

### 8.1 三分支语义（修订 §2.1）

原文「三分支编码 z_site / z_motif / z_path」未指定输入子集。
V9.1 落地版**明确分割**：

- $z_{site}$：**金属原子**池化（Z≥11 且不在 {P,S,Cl,Br,I}）
- $z_{motif}$：**非金属原子**池化（有机骨架）
- $z_{path}$：**全部原子**池化（全图上下文）

三个分支都是**纯结构描述子**（不再吸收 condition），气体计数 $(n_{H_2O}, n_{CO_2})$ 在 fusion 层（S2 `tri_energy_head` / S3 `CausalMixer` 的 head + `CooperativeHead`）才注入。这样保证 IRM 的不变性对象是结构本身，env_infer 头才有意义。

### 8.2 IRM 环境定义（修订 §2.1 / §2.2）

原方案用 `pure_CO2 / pure_H2O / mixed` 三个环境。但 `mixed` 本身承载 **合作吸附机制**（V9 §3 `E_{coop}` 的建模目标），把它作为「需要被消除的环境扰动」与 S3 的 coop head「专门建模该机制」**目标冲突**。

V9.1 做法：**S2 的 IRM 仅在 `pure_CO2` 与 `pure_H2O` 两个环境上计算**，`mixed` 不进 IRM 项，留给 S3 coop head 学习。同时**要求 batch 内同时存在两环境样本**才计算 IRM penalty（否则当 batch 跳过，避免单环境退化）。

### 8.3 batch_size 与 IRM 有效性（修订 §2.3）

原 yml 默认 `batch_size=1`，IRM 在单样本 batch 上**梯度方差等价于 0**，约束失效。V9.1 yml：S2/S3 `batch_size: 8`。配合 §8.2 的 batch 内环境检查，IRM 在足够 mix 的 batch 上才生效。

### 8.4 反事实方向先验（修订 §3.3）

原 `L_{counterfactual}` 是「移除 motif/path 后能量变化方向 vs 全局 +1」hinge。化学上 motif 既可能是吸附位点（移除后能量变正）也可能是位阻（移除后能量变负），全局 +1 是噪声标签。

V9.1 调整：

1. λ_cf 初始权重从 0.3 降到 0.05；
2. trainer 增加 `cf_direction fallback ratio` 日志，第一次跑训练就能看到有多少样本落到 +1 fallback；
3. 后续若需要更强约束，可把 hinge 改为 sign-free 形式：`relu(cf_min_effect - |delta|)`，只要求 motif/path 重要，不强制方向。

### 8.5 S3 容量与冻结策略（修订 §3.1）

原方案完全冻结 backbone + tri_encoder，S3 可调参数只剩 ~100k。V9.1 在 yml 暴露 `freeze_base` / `freeze_tri_encoder` 两个开关：默认严冻结（论文原意），但 S3 拟合不动时可单独放开 tri_encoder 微调。

### 8.6 S3 各机制头是否吸收 condition

原 §3.1 写法 `E_{site} / E_{motif} / E_{path} / E_{coop}(h,c)` 给人「只有 coop 含气体计数」的暗示。但物理上同一 MOF 在 pure_CO2 与 pure_H2O 下能量并不相同——若结构性 head 完全不知道气体，模型在纯气场景**无法区分两种气体**。

V9.1 落地版：site/motif/path 各 head 都吃 condition（"gas-conditional 结构能量"），coop_head 保留 $h\cdot c$ 解析硬零（"超出独立贡献的协同项"）。这样在数学上仍满足：

$$
E_{coop}(h=0, c) = 0,\quad E_{coop}(h, c=0) = 0
$$

且总能量在纯气下能正确区分两种气体。

### 8.7 OOD 评估口径（修订 §5）

数据集 `causal_model/prep_data/ood_split.json` 已提供：

- `ood_topology`（按拓扑划出的 OOD 子集）
- `ood_metal`（按金属族划出的 OOD 子集）
- `ood_combined` / `standard_val`

V9.1 验收门槛对应：

- **ID** = `in_distribution_train` 划出的 val 子集
- **OOD-topology** ≥ 8% 提升
- **OOD-metal** ≥ 8% 提升
- **机制一致性**：cf_direction_accuracy + coop_zero_residual

评估管线尚未实现，列入 M3 交付物。

### 8.8 安全网（与 Plan_V9 同步）

- S3 trainer 启动时若 `tri_encoder` 关键权重未从 S2 ckpt 加载，**hard error**（防止冻结的 tri_encoder 实际是随机初始化）。
- S2 trainer 同名检查保持 warning。

---

修订后总损失（V9.1）：

$$
\begin{aligned}
L_{s1} &= L_{energy} + \lambda_{type} L_{atom\_type} + \lambda_{mask} L_{mask\_recon}\\
L_{s2} &= L_{energy} + \lambda_{inv} L_{IRM}^{\{0,1\}} + \lambda_{dec} L_{decorrel} + \lambda_{sub} L_{sub}\\
L_{s3} &= L_{energy} + \lambda_{cf}^{\text{small}} L_{cf}^{zero} + \lambda_{phys} L_{phys} + \lambda_{inv}^{\text{small}} L_{IRM}^{\{0,1\}}
\end{aligned}
$$

其中 $L_{IRM}^{\{0,1\}}$ 表示仅在 `pure_CO2` 与 `pure_H2O` 上计算的 IRM penalty。

---

## 9. 参考文献（V9 方法直接来源）

> 仅列「**没有这几篇就没有 V9 方法**」的核心来源。每条注明 V9 哪一个设计决策来自它。

### 9.1 Arjovsky et al. 2019 — IRM penalty 直接实现

**Arjovsky M, Bottou L, Gulrajani I, Lopez-Paz D.** *Invariant Risk Minimization*.
**arXiv preprint** [arXiv:1907.02893](https://arxiv.org/abs/1907.02893), 2019（v3: 2020-03-27）.

- 仅在 arXiv 发表，未走会议正式 proceedings；后续 IRM 文献全部以此为基础。
- V9 Stage 2 的 `irmv1_penalty`（dummy classifier 二阶梯度方案）**逐行照搬**该论文 §3.2。
- 代码见 `causal_modules.py:177-193`（注释里也直接标注了引用）。

### 9.2 EISG 2026 — Stage 2 整体架构蓝本

*Molecular graph-based invariant representation learning with environmental inference and subgraph generation for out-of-distribution generalization*. **J. Cheminformatics, 2026, 18:12**（accepted 2025-12, published 2026-01）.
DOI: [10.1186/s13321-025-01142-w](https://link.springer.com/article/10.1186/s13321-025-01142-w)

- V9 Stage 2 的 **env_infer_head + 子图扰动一致性** 组合直接来自 EISG 的「环境推断模块 + 子图提取器」设计。
- 信息瓶颈思想间接支撑了 V9.1 §8.1 三分支去耦的合理性。

### 9.3 MILI (KDD 2024) — 三分支语义切分的理论根

**Wang R, Dai H, Yang C, Song L, Shi C.** *Advancing Molecule Invariant Representation via Privileged Substructure Identification*.
**KDD 2024** —— *Proc. of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*, pp. 3188–3199.
DOI: [10.1145/3637528.3671886](https://dl.acm.org/doi/10.1145/3637528.3671886) ｜ Code: <https://github.com/BUPT-GAMMA/Advancing-Molecule-Invariant-Representation-via-Privileged-Substructure-Identification>
（北京邮电大学 BUPT-GAMMA Lab）

- 「核心子结构是不变量、其余是环境扰动」的形式化命题，是 V9.1 §8.1 把 z_site / z_motif / z_path 切分成「金属配位 / 非金属骨架 / 全图通道」的理论根。
- MILI 的 dual-head（identifier + task head）启发了 V9 Stage 3 显式机制分解的整体架构。

### 9.4 MMGCF 2025 — Stage 3 motif 级反事实

**Zhang X, Liu Q, Han R.** *MMGCF: Generating Counterfactual Explanations for Molecular Property Prediction via Motif Rebuild*.
**Journal of Computer and Communications**, 2025, Vol. 13, No. 1.
DOI: [10.4236/jcc.2025.131011](https://doi.org/10.4236/jcc.2025.131011)

- 该期刊为 SCIRP 开源期刊，方法思路可借鉴；正式投稿时建议**同时引用更权威的反事实工作**（例如 KDD 2024 RLHEX）作支撑。
- V9 Stage 3 的 **motif/path 反事实 zero+swap**（`escn_v9_causal.py:390-410`）是 MMGCF「motif rebuild → 测能量变化」思想的简化版。
- 后续若要把 V9 反事实方向先验从全局 +1 升级到 sample-specific，MMGCF 的 motif 重构 pipeline 是直接候选标签源。

### 9.5 V9 设计决策 ↔ 来源

| V9 设计 | 来源 | 对应类型 |
|---|---|---|
| `irmv1_penalty` | Arjovsky 2019 | **照搬** |
| Stage 2 env_infer + subgraph_consistency | EISG 2025 | **整体蓝本** |
| 三分支 z_site / z_motif / z_path | MILI 2024 | **理论根** |
| Stage 3 显式机制分解 + 反事实 | MILI 2024 + MMGCF 2025 | **概念融合** |
| `CooperativeHead` 解析硬零 $\psi(h{=}0)=0$ | **物理化学先验**（无论文引用） | **架构贡献，非借鉴** |

### 9.6 V9 自身贡献点（非借鉴）

- **`CooperativeHead` 的 $h\cdot c$ 解析硬零**是把吸附化学常识（纯气无协同效应）写进**架构**而非 loss 的设计。这是 V9 相对上述四篇文献的独立工程贡献，论文撰写时 Method 章应单独标注 *architectural prior, not from prior work*。
