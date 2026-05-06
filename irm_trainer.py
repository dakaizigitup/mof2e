"""
IRM (Invariant Risk Minimization) Trainer
在 OCPTrainer 基础上加入 IRM 正则化，其余逻辑完全不变。

环境划分方案（通过 yml 里 irm.env_scheme 控制）：
  - "oms"       : 按 has_OMS 分两组（0/1），从 Excel 读取
  - "condition" : 按 n_h2o > 0 分两组（纯CO₂ / 含水），直接从 batch.condition 读取
  - "metal"     : 按金属类型分三组（Cu / Zn / Others），从 Excel 读取

当 irm.lambda == 0 时，与原始 OCPTrainer 完全等价。
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import torch

from fairchem.core.common.registry import registry
from fairchem.core.trainers.ocp_trainer import OCPTrainer


def _irm_penalty(loss_per_env: list[torch.Tensor]) -> torch.Tensor:
    """
    计算 IRM penalty。
    对每个环境的 loss，计算 loss 对虚拟标量乘子 scale（=1.0）的梯度范数平方，求和。
    """
    scale = torch.ones(1, device=loss_per_env[0].device, requires_grad=True)
    penalty = torch.zeros(1, device=loss_per_env[0].device)
    for env_loss in loss_per_env:
        grad = torch.autograd.grad(
            env_loss * scale,
            scale,
            create_graph=True,
            retain_graph=True,
        )[0]
        penalty = penalty + grad.pow(2)
    return penalty.squeeze()


def _build_name_to_env(excel_path: str, scheme: str) -> dict[str, int]:
    """
    从 Excel 构建 {mof_base_name: env_id} 映射。
    scheme="oms"   : env_id = int(has_OMS == "Yes")
    scheme="metal" : env_id = 0(Cu) / 1(Zn) / 2(Others)
    """
    df = pd.read_excel(excel_path)
    name_to_env = {}

    if scheme == "oms":
        for _, row in df.iterrows():
            name = str(row["Name"]).strip()
            oms_val = str(row.get("Has_OMS", "")).strip().lower()
            env_id = 1 if oms_val == "yes" else 0
            name_to_env[name] = env_id

    elif scheme == "metal":
        for _, row in df.iterrows():
            name = str(row["Name"]).strip()
            metals_str = str(row.get("Metals", "")).strip()
            if "Cu" in metals_str:
                env_id = 0
            elif "Zn" in metals_str:
                env_id = 1
            else:
                env_id = 2
            name_to_env[name] = env_id

    return name_to_env


@registry.register_trainer("ocp_irm")
class IRMTrainer(OCPTrainer):
    """
    IRM-augmented OCP Trainer.
    yml 里新增 irm 配置块：

        irm:
          lambda: 1.0e-4      # IRM penalty 权重，0 = 关闭
          env_scheme: oms     # "oms" | "condition" | "metal"
          excel_path: /path/to/MOF_embedding_all.xlsx  # oms/metal 方案需要
          min_env_size: 4     # 每个 env 至少多少样本才计算 penalty（默认 4）
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        irm_cfg = self.config.get("model", {}).get("irm", {})
        self.irm_lambda = float(irm_cfg.get("lambda", 0.0))
        self.irm_scheme = irm_cfg.get("env_scheme", "condition")
        self.irm_min_env_size = int(irm_cfg.get("min_env_size", 4))
        self._name_to_env: dict[str, int] | None = None

        if self.irm_lambda > 0:
            logging.info(
                f"[IRM] lambda={self.irm_lambda}, scheme={self.irm_scheme}"
            )
            if self.irm_scheme in ("oms", "metal"):
                excel_path = irm_cfg.get(
                    "excel_path",
                    self.config["model"]["backbone"].get("mof_global_excel_path", ""),
                )
                self._name_to_env = _build_name_to_env(excel_path, self.irm_scheme)
                logging.info(
                    f"[IRM] Loaded env mapping for {len(self._name_to_env)} MOFs"
                )

    def _get_env_ids(self, batch) -> torch.Tensor:
        """
        根据 scheme 返回 batch 中每个样本的 env_id (LongTensor, shape=[batch_size])。
        """
        batch_size = batch.natoms.numel()

        if self.irm_scheme == "condition":
            # batch.condition 被 PyG 拼接为 [B*2]，需要 reshape 成 [B, 2]
            cond = batch.condition.to(self.device)
            if cond.dim() == 1:
                cond = cond.view(-1, 2)             # [B*2] → [B, 2]
            env_ids = (cond[:, 0] > 0).long()       # 0=纯CO₂, 1=含水

        elif self.irm_scheme in ("oms", "metal"):
            env_ids = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            names = batch.name if isinstance(batch.name, (list, tuple)) else [batch.name]
            for i, full_name in enumerate(names):
                base_name = str(full_name).split("_")[0]
                env_id = self._name_to_env.get(base_name, 0)
                env_ids[i] = env_id

        else:
            raise ValueError(f"Unknown IRM env_scheme: {self.irm_scheme}")

        return env_ids

    def _compute_loss(self, out, batch) -> torch.Tensor:
        # 原始 MAE loss（完全不变）
        base_loss = super()._compute_loss(out, batch)

        # IRM 关闭时直接返回
        if self.irm_lambda == 0.0 or not self.model.training:
            return base_loss

        # ── 计算 IRM penalty ──────────────────────────────────────────────────
        env_ids = self._get_env_ids(batch)
        unique_envs = env_ids.unique()

        # 如果 batch 里只有一个环境，penalty 无意义，跳过
        if unique_envs.numel() < 2:
            return base_loss

        # 获取归一化后的 target（与 base_loss 保持一致）
        target_name = "energy"
        target = batch[target_name].view(batch.natoms.numel(), -1)
        if target_name in self.normalizers:
            target = self.normalizers[target_name].norm(target)

        pred = out[target_name].view(batch.natoms.numel(), -1)

        loss_per_env = []
        for env_id in unique_envs:
            mask = env_ids == env_id
            if mask.sum() < self.irm_min_env_size:
                continue
            env_pred = pred[mask]
            env_target = target[mask]
            env_loss = torch.mean(torch.abs(env_pred - env_target))
            loss_per_env.append(env_loss)

        if len(loss_per_env) < 2:
            return base_loss

        penalty = _irm_penalty(loss_per_env)
        total_loss = base_loss + self.irm_lambda * penalty

        # 记录 penalty 值便于 wandb 监控
        if hasattr(self, "metrics"):
            self.metrics.setdefault("irm_penalty", {"total": 0.0, "numel": 0})
            self.metrics["irm_penalty"]["total"] += penalty.item()
            self.metrics["irm_penalty"]["numel"] += 1

        return total_loss
