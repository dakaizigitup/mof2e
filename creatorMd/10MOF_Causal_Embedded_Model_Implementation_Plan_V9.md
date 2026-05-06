# V9 实现计划（逐步落地版）

> 目标：在不破坏现有 V7/V8 与 legacy eSCN 流程的前提下，逐步实现 V9 三阶段因果嵌入方案。  
> 约束：并行注册、最小侵入、可随时回滚、每步可独立验证。

---

## 0. 实施边界与原则

1. **不修改旧主线行为**：`main.py`、既有 eSCN 配置可继续原样运行。  
2. **并行注册**：新增 V9 专属 model/trainer 名称，不复用 V7 名称。  
3. **阶段化落地**：先 S1 跑稳，再 S2，再 S3，避免一次引入过多不稳定因素。  
4. **评估先行**：每阶段都定义可量化验收门槛。

---

## 1. 命名与目录规划

### 1.1 新增命名

- 模型注册名：
  - `escn_v9_causal_s1`
  - `escn_v9_causal_s2`
  - `escn_v9_causal_s3`
- 训练器注册名：
  - `ocp_causal_v9`

### 1.2 新增文件

- `fairchem/causal_model/model_causal/escn_v9_causal.py`
- `fairchem/causal_model/trainer_causal/causal_trainer_v9.py`
- `fairchem/causal_model/config_causal/v9/eSCN_v9_s1.yml`
- `fairchem/causal_model/config_causal/v9/eSCN_v9_s2.yml`
- `fairchem/causal_model/config_causal/v9/eSCN_v9_s3.yml`

### 1.3 接入点修改

- `fairchem/causal_model/model_causal/__init__.py`：增加 `escn_v9_causal` 的懒加载导入
- `fairchem/causal_model/trainer_causal/__init__.py`：增加 `causal_trainer_v9` 的懒加载导入
- `fairchem/causal_main.py`：在 `causal.version == v9` 时强制 `trainer=ocp_causal_v9`

---

## 2. Stage 1 实现（原子语义地基）

### 2.1 模型结构

在 `EscnV9CausalS1` 中实现：

- 复用现有 `escn_weighted_energy_head` 作为 base
- 从 `sphere_values` 聚合 `per_atom` 特征
- 输出头：
  1. `atom_type_head`（分类）
  2. `mask_recon_head`（被 mask 属性恢复）

### 2.2 关键开关

- `detach_atom_aux: bool`（默认 `true`）
  - `true`：辅助头不回传 backbone（稳定优先）
  - `false`：小步放开辅助梯度（微调）

### 2.3 损失函数

\[
L_{s1}=L_{energy}+\lambda_{type}L_{atom\_type}+\lambda_{mask}L_{mask\_recon}
\]

### 2.4 训练策略

- Phase-A：`detach_atom_aux=true` 训练至收敛
- Phase-B：`detach_atom_aux=false` + 小 `lambda_type`（0.01~0.02）短程微调
- 用最优 `val/energy_mae` checkpoint 进入 Stage 2

### 2.5 验收门槛

- 无 NaN，无梯度爆炸
- `val/energy_mae` 显著低于均值基线

---

## 3. Stage 2 实现（跨环境不变 + 子结构稳健）

### 3.1 模型结构

在 `EscnV9CausalS2` 中实现：

- 加载 Stage1 checkpoint
- 冻结前 6 层（低层语义固定）
- 构建三分支表示（**V9.1：纯结构描述子，不吸收 condition**）：
  - `z_site`：金属原子池化（`Z>=11 & not in {15,16,17,35,53}`）
  - `z_motif`：非金属原子池化（有机骨架）
  - `z_path`：全部原子池化（全图上下文）
- `tri_energy_head` 输入 `[z_site, z_motif, z_path, condition]`（condition 在 fusion 层注入）
- 可选 `env_infer_head`（仅训练辅助；输入只有 `[z_site, z_motif, z_path]`，**不含 condition** —— 该头预测准确率低 = IRM 起效）

### 3.2 环境定义（在线构造）

根据 `condition=(n_h2o,n_co2)` 构造：

- `pure_CO2`
- `pure_H2O`
- `mixed`

不依赖外部机制分析标签。

### 3.3 损失函数

\[
L_{s2}=L_{energy}+\lambda_{inv}L_{IRM}^{\{0,1\}}+\lambda_{dec}L_{decorrel}+\lambda_{sub}L_{subgraph\_consistency}
\]

- `L_IRM`（V9.1）：**仅在 pure_CO2(0) 与 pure_H2O(1) 两个环境**上计算。`mixed`(2) 不进 IRM——它承载 S3 的合作机制目标，不该在 S2 被擦掉。
- `L_decorrel`：`z_site/z_motif/z_path` 去耦
- `L_subgraph_consistency`：子图扰动一致性（path 分支 node-dropout 后能量与原能量 MSE）

### 3.4 训练策略

- **batch_size ≥ 8**（V9.1）：`bs=1` 让 IRM penalty 在单样本上等价于 0，约束失效；
- trainer 在 batch 内若 `pure_CO2` 与 `pure_H2O` 不同时存在，**该 batch 跳过 IRM**（每 200 个跳过 batch 输出一次警告）；
- 早停指标优先 OOD val MAE。

### 3.5 验收门槛

- OOD split 相对 ERM 提升 ≥ 8%~12%
- ID 不显著退化

---

## 4. Stage 3 实现（机制分解 + 反事实一致性）

### 4.1 模型结构

在 `EscnV9CausalS3` 中实现：

- 冻结 Stage2 encoders（V9.1：通过 yml 的 `freeze_base` / `freeze_tri_encoder` 两开关控制；默认严冻结，需要时可放开 tri_encoder）
- 训练 `CausalMixer + CooperativeHead`
- 显式分解输出（V9.1：所有 head 都吸收 condition，coop head 保留 `h*c` 解析硬零）：
  - `E_site (z_site, condition)`
  - `E_motif(z_motif, condition)`
  - `E_path(z_path, condition)`
  - `E_coop(h, c, ctx)` —— `ψ = (h*c/max²) * sigmoid(MLP_gate) * MLP_amp`，纯气下解析为 0
- 启动期 hard error：若 `tri_encoder` 关键权重未从 S2 ckpt 加载，**直接 raise**（防止冻结的随机初始化）

组合：

\[
E_{pred}=E_{site}+E_{motif}+E_{path}+E_{coop}
\]

### 4.2 硬约束

\[
E_{coop}(0,c)=0,\quad E_{coop}(h,0)=0
\]

### 4.3 损失函数

\[
L_{s3}=L_{energy}+\lambda_{cf}^{\text{small}}L_{counterfactual}+\lambda_{phys}L_{physics}+\lambda_{inv}^{\text{small}}L_{IRM}^{\{0,1\}}
\]

- `L_counterfactual`（V9.1）：对 motif/path 做 zero-out 反事实，hinge 形式 `relu(cf_min_effect - dir·delta)`。`dir` 优先从 batch field `cf_motif_direction`/`cf_path_direction` 读 sample-specific 值；缺失时 fallback 到 yml 的全局 +1。trainer 每 100 step 输出 fallback 比例。**初始 λ_cf=0.05**（弱先验，避免噪声标签拉错方向）。
- `L_physics`：`coop_zero (pure 环境上 MSE(coop, 0))` + `extensive (E ≈ N·per_atom_mean)` + `volume decorrel (corr²(E, |det(cell)|))`。
- `L_IRM`（V9.1）：与 S2 同口径，**仅 pure_CO2/pure_H2O 两环境**，权重小（0.1）。

### 4.4 验收门槛

- cooperative 硬约束满足
- 反事实方向正确率显著高于随机

---

## 5. Trainer 设计（`ocp_causal_v9`）

### 5.1 配置读取

- 支持 `self.config["causal"]` 与 `self.config["model"]["causal"]` 双路径
- 关键字段：
  - `stage`
  - `lambda_type/lambda_mask/lambda_inv/lambda_dec/lambda_sub/lambda_cf/lambda_phys`
  - `init_from`

### 5.2 分阶段 loss 分派

- `stage==1`：`energy + atom_type + mask_recon`
- `stage==2`：`energy + IRM + decorrel + subgraph_consistency`
- `stage==3`：`energy + counterfactual + physics + small IRM`

### 5.3 日志规范

每阶段单独记录：

- 主 loss
- 每个子 loss
- `grad_norm`
- OOD 分组指标

---

## 6. 配置文件落地（YAML）

### 6.1 `eSCN_v9_s1.yml`

- `causal.version: v9`
- `causal.stage: 1`
- `detach_atom_aux: true`（首跑）
- `lambda_type`, `lambda_mask`

### 6.2 `eSCN_v9_s2.yml`

- `causal.stage: 2`
- `init_from: <stage1_best_checkpoint>`
- `lambda_inv`, `lambda_dec`, `lambda_sub`

### 6.3 `eSCN_v9_s3.yml`

- `causal.stage: 3`
- `init_from: <stage2_best_checkpoint>`
- `lambda_cf`, `lambda_phys`, `lambda_inv(small)`

---

## 7. 评估与Ablation计划

### 7.1 评估维度

1. **ID**：`in_distribution_train` 划出的 val 子集（来自 `causal_model/prep_data/ood_split.json`）→ MAE/MSE/R²
2. **OOD**：直接复用现有 split：
   - `ood_topology`（76 entries，按拓扑留出）
   - `ood_metal`（91 entries，按金属族留出 — Ag/Ce/Eu/La/Nd/Tb）
   - `ood_combined`（159 entries，topology∪metal）
   - `standard_val`（182 entries，原始 val）
3. **机制一致性**：
   - `coop_zero_residual` = `mean(|E_coop|)` on env∈{0,1}
   - `cf_direction_accuracy` = `mean(sign(delta) == sign(direction))` on `mixed`
   - `env_infer_accuracy` 应**显著低于** baseline（IRM 起效的判据）

### 7.2 最小 ablation 集合

- 去 `IRM`
- 去 `decorrel`
- 去 `counterfactual`
- 去 cooperative 硬约束

---

## 8. 执行顺序（推荐）

1. 先打通 S1（模型+trainer+配置+训练）
2. 再实现 S2（先上 IRM，再补 decorrel/subgraph）
3. 最后实现 S3（机制分解+反事实）
4. 统一评估与 ablation

---

## 9. 里程碑与交付物

### M1（S1完成）

- `escn_v9_causal_s1` 可训练
- `ocp_causal_v9(stage1)` 可用
- `eSCN_v9_s1.yml` 可复现实验

### M2（S2完成）

- 三分支表示输出可用
- IRM + decorrel + subgraph consistency 跑通
- OOD 指标优于 ERM 基线

### M3（S3完成）

- 机制分解输出完整
- 反事实一致性评估可运行
- 形成 V9 最终模型与 ablation 结果

---

## 10. 风险与缓解

1. **S1 欠拟合 / 均值塌缩**：降低 weight decay，先稳训后放开辅助梯度  
2. **S2 IRM 信号弱**：保证 batch 环境混合，提高 env 采样平衡  
3. **S3 机制项互相干扰**：先单独 warmup mixer，再叠加 counterfactual loss  
4. **ID/OOD tradeoff 过大**：使用小权重逐步引入因果项，按 OOD 优先早停

---

## 11. 备注

- 本计划为 V9 的工程实施蓝图，和 Proposal7/8 并行不冲突。  
- 后续可在此文档上直接追加“参数表 v1/v2”和“实验记录链接”。

---

## 12. V9.1 修订摘要（与 Proposal9 §8 对应）

| 编号 | 改动 | 文件 |
|---|---|---|
| **P-1** | `_TriBranchEncoder.is_metal` 排除 P/S/Cl/Br/I | `escn_v9_causal.py` |
| **P-2** | S3 trainer 在 `tri_encoder` 加载失败时 hard error | `causal_trainer_v9.py` |
| **P-5** | `cf_direction` fallback 比例每 100 step 输出 | `causal_trainer_v9.py` |
| **R-1** | S2/S3 `batch_size: 8`；trainer 单环境 batch 自动跳过 IRM | `eSCN_v9_s{2,3}.yml`, `causal_trainer_v9.py` |
| **R-2** | IRM penalty 仅在 env∈{0,1} 上计算，`mixed` 排除 | `causal_trainer_v9.py` |
| **R-3** | `z_motif` 改非金属池化；condition 下沉到 fusion 层；CausalMixer 各 head 都吃 condition | `escn_v9_causal.py` |
| **R-4** | S3 `lambda_cf: 0.3 → 0.05` | `eSCN_v9_s3.yml` |
| **R-5** | S3 暴露 `freeze_base` / `freeze_tri_encoder` 开关 | `escn_v9_causal.py`, `eSCN_v9_s3.yml` |

---

## 13. 参考文献速查

V9 方法的直接来源**只有 4 篇**，详细 stage 映射见 [Proposal9.md §9](9MOF_Causal_Embedded_Model_Proposal9.md#9-参考文献v9-方法直接来源)。

| 实现项 | 参考（完整引用见 Proposal9 §9） | 链接 |
|---|---|---|
| `irmv1_penalty` 实现 | Arjovsky, Bottou, Gulrajani, Lopez-Paz. *Invariant Risk Minimization*. arXiv 2019 | [arXiv:1907.02893](https://arxiv.org/abs/1907.02893) |
| Stage 2 env_infer + subgraph_consistency | EISG. *J. Cheminformatics* **2026, 18:12** | [DOI](https://link.springer.com/article/10.1186/s13321-025-01142-w) |
| 三分支语义切分 | Wang, Dai, Yang, Song, Shi. MILI. **KDD 2024**, pp.3188–3199 | [DOI](https://dl.acm.org/doi/10.1145/3637528.3671886) |
| Stage 3 motif 级反事实 | Zhang, Liu, Han. MMGCF. ***J. Computer and Communications* 2025, 13(1)** | [DOI](https://doi.org/10.4236/jcc.2025.131011) |
| `CooperativeHead` 解析硬零 | 物理先验（无论文引用，V9 自身架构贡献） | — |
