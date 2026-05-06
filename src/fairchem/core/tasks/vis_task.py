"""Visualization Task: generate 3D colored XYZ files and PNG screenshots from saved SHAP values."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from ase import Atoms
from ase.io import write
from torch_geometric.data import Data

from fairchem.core.common.registry import registry
from fairchem.core.tasks.task import BaseTask


@registry.register_task("vis")
class VisTask(BaseTask):
    """Load shap_values.pt produced by shap task and create per-sample 3D colored files & screenshots."""

    def _sample_graphs(self, k: int) -> List[Data]:
        assert self.trainer.val_loader is not None, "val_loader required for vis"
        graphs: List[Data] = []
        for batch in self.trainer.val_loader:
            graphs.extend(batch.to_data_list())
            if len(graphs) >= k:
                break
        return graphs[:k]

    def run(self):
        out_dir = Path(self.config.get("output"))
        shap_file = out_dir / "shap_values.pt"
        if not shap_file.exists():
            raise FileNotFoundError(shap_file)

        data_saved = torch.load(shap_file, map_location="cpu")
        phi = data_saved["shap_values"]  # [B, Nmax*F]
        nmax = int(data_saved["nmax"])
        f = int(data_saved["f"])
        num_explain = phi.shape[0]

        graphs = self._sample_graphs(num_explain)
        assert len(graphs) == num_explain, "Mismatch between explain samples and loaded graphs"

        vis_dir = out_dir / "vis"
        vis_dir.mkdir(parents=True, exist_ok=True)

        for j, g in enumerate(graphs):
            phi_nodes = phi[j][: g.num_nodes * f].reshape(g.num_nodes, f)[:, 0]
            pos = g.pos.cpu().numpy()
            z = (g.atomic_numbers if hasattr(g, "atomic_numbers") else g.z).cpu().numpy()
            atoms = Atoms(numbers=z, positions=pos)
            atoms.new_array("charges", phi_nodes)
            write(vis_dir / f"sample_{j}.xyz", atoms)

            # simple scatter screenshot
            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(111, projection="3d")
            p = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=phi_nodes, cmap="bwr", s=30)
            plt.colorbar(p, shrink=0.65, label="SHAP value")
            ax.set_axis_off()
            plt.tight_layout()
            plt.savefig(vis_dir / f"sample_{j}.png", dpi=300, transparent=True)
            plt.close()
            logging.info(f"vis: sample_{j} done")

        logging.info("VisTask complete.")
