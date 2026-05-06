# 四流多模态融合模型 - 训练指南

## 配置文件说明

### 1. base.yml
基础配置，定义数据集和训练参数：
- 数据路径
- MOF embedding 路径
- 数据预处理（explode_y_relaxed）
- 损失函数和评估指标

### 2. MultiStream.yml
模型配置，定义四流架构参数：
- `hidden_dim`: 所有编码器的隐藏维度 (512)
- `mof_emb_dim`: MOF 全局 embedding 维度 (64)
- `escn_config`: eSCN 骨干网络配置
- 可选：`pretrained_escn_path` 预训练权重路径

## 训练命令

```bash
# 基础训练
python -m fairchem.core.scripts.main \
    --mode train \
    --config-yml /home/dell/autodl-tmp/lorafair/fairchem/mof2e/MultiStream/MultiStream.yml \
    --run-dir /home/dell/autodl-tmp/lorafair/runs/multistream_v1

# 使用预训练 eSCN
python -m fairchem.core.scripts.main \
    --mode train \
    --config-yml /home/dell/autodl-tmp/lorafair/fairchem/mof2e/MultiStream/MultiStream.yml \
    --run-dir /home/dell/autodl-tmp/lorafair/runs/multistream_pretrained \
    --model.pretrained_escn_path /path/to/escn_checkpoint.pt

# 冻结 eSCN 权重（只训练其他模块）
python -m fairchem.core.scripts.main \
    --mode train \
    --config-yml /home/dell/autodl-tmp/lorafair/fairchem/mof2e/MultiStream/MultiStream.yml \
    --run-dir /home/dell/autodl-tmp/lorafair/runs/multistream_frozen \
    --model.pretrained_escn_path /path/to/escn_checkpoint.pt \
    --model.freeze_escn true
```

## 验证命令

```bash
python -m fairchem.core.scripts.main \
    --mode validate \
    --config-yml /home/dell/autodl-tmp/lorafair/fairchem/mof2e/MultiStream/MultiStream.yml \
    --checkpoint /path/to/checkpoint.pt
```

## 关键参数说明

### 模型参数
- `hidden_dim`: 512 (所有编码器输出维度)
- `num_layers`: 8 (eSCN 层数，从12减少以节省显存)
- `batch_size`: 2 (由于模型较大，减小batch size)

### 优化器参数
- `lr_initial`: 0.0005 (初始学习率)
- `weight_decay`: 0.1 (权重衰减)
- `clip_grad_norm`: 10 (梯度裁剪)
- `warmup_steps`: 200 (预热步数)

### 数据参数
- `explode_y_relaxed`: 将 y_relaxed 字典展开为多条样本
- `mof_embedding_path`: MOF 全局特征 Excel 文件路径

## 注意事项

1. **显存管理**：
   - 四流模型参数量约 19M
   - 建议使用 `activation_checkpoint: true`
   - 如果显存不足，可以减少 `num_layers` 或 `batch_size`

2. **预训练权重**：
   - 如果有预训练的 eSCN，强烈建议使用
   - 可以选择冻结 eSCN 权重，只训练其他模块

3. **数据格式**：
   - 确保 LMDB 数据包含 `y_relaxed` 字典
   - 确保 MOF_embedding_all.xlsx 存在且包含所有 MOF 名称

4. **WandB 日志**：
   - 修改 `wandb.project` 和 `wandb.name` 以匹配您的项目
   - 如果不使用 WandB，设置 `logger: tensorboard`

## 故障排查

### 问题1：找不到 multi_stream_fusion 模型
**解决**：确保导入了 `multi_stream_wrapper.py`
```python
# 在 fairchem/core/models/__init__.py 中添加
from fairchem.core.models.multi_stream_wrapper import MultiStreamFusionWrapper
```

### 问题2：数据加载失败
**解决**：检查数据路径和 MOF embedding 路径是否正确

### 问题3：显存溢出
**解决**：
- 减少 `batch_size` (2 → 1)
- 减少 `num_layers` (8 → 6)
- 启用 `activation_checkpoint: true`
