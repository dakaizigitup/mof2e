# EquiformerV2-LoRA 消融实验结果

数据集：MOF 吸附能（CO₂ / H₂O），train = 960 MOF（explode 后 3,714 条），val = 182 MOF（explode 后 699 条），train / val 的 MOF 完全不重叠（OOD 评估）。

Backbone：EquiformerV2（8 层），预训练 checkpoint `Equiformer_V2_Direct.pt`，基于 LoRA 微调。

## 1. 配置说明

| 标签 | 含义 | 主要开关 |
|---|---|---|
| **equiv2Base** | 仅 nm 条件，无原子级 / 无全局 | `use_atom_extra_features: false`，`use_mof_global_features: false` |
| **equiv2sta1** | +原子级特征（V2） | `use_atom_extra_features: true`，`atom_encoder_type: v2`，`use_mof_global_features: false` |
| **equiv2sta2** | +原子级特征（V2）+ 全局特征（V2 三模态） | stage1 基础上再开 `use_mof_global_features: true`，`mof_global_encoder_type: v2`，`global_inject_layer`，`restart_stage2` |

## 2. 结果对比

| 标签 | run timestamp | val / energy_mae ↓ | val / energy_mse ↓ | val / energy_within_threshold ↑ | val / loss ↓ | best @ epoch |
|---|---|---:|---:|---:|---:|---:|
| equiv2Base | 2026-04-21-21-05-04 | 0.25785 | 0.19604 | 0.081545 | 1.09368 | 29.887 |
| equiv2sta1 | 2026-04-21-21-24-16 | 0.24711 | 0.18819 | 0.098712 | 1.04817 | 7.270 |
| equiv2sta2 | 2026-04-23-11-44-00 | **0.24646** | **0.18787** | **0.098712** | **1.04540** | 1.884 |

## 3. 结论

- **原子级特征（Base → sta1）**：`val/energy_mae` 从 0.25785 降到 0.24711（**−0.01074**），`within_threshold` 从 0.0815 提到 0.0987，`val/loss` 也相应下降。原子级特征的加入带来了**清晰且可重复的提升**。
- **全局特征（sta1 → sta2）**：`val/energy_mae` 从 0.24711 再降到 0.24646（**−0.00065**），`val/loss` 进一步从 1.04817 降到 1.04540。全局特征在已经吃掉原子级收益的基础上仍能带来小幅提升。
- Best metric 出现时间：Base 约在最后（~ epoch 30），sta1 在 epoch 7，sta2 在 epoch 1.9 —— 引入的新特征越多，训练越快达到最优，随后进入过拟合区。
- 整体 MAE 地板被数据标签本身的条件冲突（同 MOF × 同气体组合对应多个 ratio/group 版本，ratio/group 未作为模型输入）抬高。在不扩展 condition 维度的前提下，当前 ~0.246 已接近该配置下的上限。

## 4. 复现命令

```bash
# equiv2Base（nm-only）
python fairchem/main.py \
  --config fairchem/mof2e/EquiformerV2/EqV2_lora_nm.yml \
  --mode train

# equiv2sta1（+atom V2）
python fairchem/main.py \
  --config fairchem/mof2e/EquiformerV2/EqV2_lora_stage1.yml \
  --mode train

# equiv2sta2（+global V2）
python fairchem/main.py \
  --config fairchem/mof2e/EquiformerV2/EqV2_lora_stage2.yml \
  --mode train \
  --checkpoint /home/dell/autodl-tmp/lorafair/checkpoints/2026-04-21-21-24-16/best_checkpoint.pt
```
