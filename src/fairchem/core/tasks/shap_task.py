"""fairchem.core.tasks.shap_task

使用 padding + 黑盒 KernelSHAP 解释图模型。
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import List, Tuple
import os
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
from torch_geometric.data import Data

from fairchem.core.common.registry import registry
from fairchem.core.tasks.task import BaseTask


@registry.register_task("shap")
class ShapTask(BaseTask):
    # ---------------- util ---------------- #

    @staticmethod
    def _node_feat(g: Data) -> torch.Tensor | None:
        """Return node feature tensor or None if not available"""
        if getattr(g, "x", None) is not None:
            return g.x
        if getattr(g, "atomic_numbers", None) is not None:
            return g.atomic_numbers.unsqueeze(-1).float()
        if getattr(g, "z", None) is not None:
            return g.z.unsqueeze(-1).float()
        return None

    @classmethod
    def _has_node_feat(cls, g: Data) -> bool:
        return cls._node_feat(g) is not None

    @classmethod
    def _pad_and_flatten_x(cls, graphs: List[Data], nmax: int, f: int) -> np.ndarray:
        X = np.zeros((len(graphs), nmax, f), dtype=np.float32)
        for i, g in enumerate(graphs):
            xi = cls._node_feat(g).detach().cpu().numpy().astype(np.float32)
            ni = min(xi.shape[0], nmax)
            X[i, :ni, : xi.shape[1]] = xi[:ni, :]
        return X.reshape(len(graphs), nmax * f)

    @classmethod
    def _infer_feature_dims(cls, graphs: List[Data]) -> Tuple[int, int]:
        nmax = max(int(cls._node_feat(g).shape[0]) for g in graphs)
        f = int(cls._node_feat(graphs[0]).shape[1])
        return nmax, f

    # -------------- sampling -----------------
    def _sample_from_loader(self, loader, k: int) -> List[Data]:
        pool: List[Data] = []
        for batch in loader:
            pool.extend(batch.to_data_list())
            if len(pool) >= k:
                break
        return random.sample(pool, min(k, len(pool)))

    def _prepare_data(self) -> List[Data]:
        cfg = self.config.get("shap", {})
        train_k = cfg.get("train_sample", 50)
        val_k = cfg.get("val_sample", 50)

        assert self.trainer.train_loader and self.trainer.val_loader
        train_samples = self._sample_from_loader(self.trainer.train_loader, train_k)
        val_samples = self._sample_from_loader(self.trainer.val_loader, val_k)
        return train_samples + val_samples

    # -------------- main ------------------
    def run(self) -> None:
        cfg = self.config.get("shap", {})
        num_background = cfg.get("num_background", 5)
        num_explain = cfg.get("num_explain", 5)
        nsamples = cfg.get("nsamples", 100)

        # model
        model = self.trainer.model
        from torch.nn.parallel import DistributedDataParallel as DDP
        if isinstance(model, DDP):
            model = model.module
        model.eval()
        # 1. 选择 GPU（默认 cuda:0，可用环境变量指定）
        gpu_id = int(os.getenv("FAIRCHEM_GPU_ID", 0)) 
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

        # 2. 把模型搬过去
        model.to(device)
        # device = torch.device("cpu")
        # model.to(device)

        data_all = [g for g in self._prepare_data() if self._has_node_feat(g)]
        if len(data_all) < num_background + num_explain:
            raise RuntimeError(
                f"有效样本不足，仅 {len(data_all)} 条，需 >= {num_background+num_explain}."
            )

        bg_graphs = data_all[:num_background]
        ex_graphs = data_all[num_background : num_background + num_explain]

        nmax, f = self._infer_feature_dims(bg_graphs + ex_graphs)
        logging.info(f"Padding to Nmax={nmax}, F={f}")

        X_bg = self._pad_and_flatten_x(bg_graphs, nmax, f)
        X_ex = self._pad_and_flatten_x(ex_graphs, nmax, f)

        def model_predict(x_flat: np.ndarray) -> np.ndarray:
            # shap 可能传 object 数组 (len, ) or 普通 2D array
            if x_flat.ndim == 1:
                if x_flat.dtype == object:
                    x_flat = np.stack(x_flat, axis=0).astype(np.float32)
                else:
                    x_flat = x_flat.reshape(1, -1)
            outs = []
            from torch_geometric.data import Batch
            for idx in range(x_flat.shape[0]):
                v = x_flat[idx]
                g = ex_graphs[idx % len(ex_graphs)]  # 循环映射
                vec = torch.tensor(v, dtype=torch.float32, device=device)
                x_pad = vec.view(nmax, f)
                ni = g.num_nodes
                g2 = g.clone()
                g2.x = x_pad[:ni, :].clone()
                bgraph = Batch.from_data_list([g2]).to(device)
                with torch.no_grad():
                    energy = model(bgraph)
                    energy = energy if torch.is_tensor(energy) else energy["energy"]
                outs.append(float(energy.squeeze().cpu()))
            return np.asarray(outs)

        logging.info("Running KernelSHAP …")
        explainer = shap.KernelExplainer(model_predict, X_bg)
        shap_values = explainer.shap_values(X_ex, nsamples=nsamples)

        out_dir = Path(self.config.get("output"))
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"shap_values": shap_values, "nmax": nmax, "f": f}, out_dir / "shap_values.pt")
        try:
            shap.summary_plot(shap_values, X_ex, show=False)
            plt.tight_layout(); plt.savefig(out_dir / "summary_plot.png", dpi=300)
            plt.close()
        except Exception as e:
            logging.warning(f"无法绘图: {e}")
        logging.info("SHAP 完成")
