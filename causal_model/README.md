# Causal-Embedded MOF GNN (V7 + V8)

> 与 `lora_model/` 平行的独立模块。**零侵入**：删掉本目录整个项目立即回到 V7/V8 之前的状态。

## 路线

- **V7 (CMG)** — 依托现有事后机制分析（Phase A→X），把 free-COOH cluster 4 / B3.1 motif contributions / D3 反事实作为 prior 与 supervision
- **V8 (AFCMG)** — 不依赖事后分析，纯架构-first 因果 GNN，仅用化学常识 + V2 预训练 chemistry encoder 作 prior

两版共用渐进式三阶段训练：atom → motif → mechanism（V7）/ atom → modular → causal-mixer（V8）。

## 与现有代码的关系

| 类别 | 行为 |
|---|---|
| 现有 `fairchem/mof2e/eSCN/eSCN.yml` | **不修改**，仍可正常 train/validate |
| 现有 `fairchem/core/models/escn/escn.py` | **不修改** |
| 现有 `fairchem/irm_trainer.py` | **不修改** |
| 现有 checkpoint | **不 warmstart**，作为 ERM baseline 进入论文 §4 对比 |

新模型/trainer 全部通过 `@registry.register_model("新名字")` 注册，不覆盖任何旧名字。

## 目录

```
causal_model/
├── prep_data/         # Phase 1 输出（数据层，不动模型）
├── config_causal/     # 新 yml（v7/, v8/）
├── model_causal/      # 新模型（包装现有 eSCN，不改源码）
├── trainer_causal/    # 新 trainer（继承 OCPTrainer）
├── data_causal/       # 数据 builder 脚本（生成 prep_data 内容）
├── eval_causal/       # 评估指标
└── scripts/           # 一键启动脚本 + 回归测试
```

## 验证回归

任何时候跑 `bash causal_model/scripts/verify_backward_compat.sh` 都应正常输出旧 yml 的 val MAE：
- eSCN base: 0.287
- eSCN_global: 0.266
- EqV2 base: 0.258
