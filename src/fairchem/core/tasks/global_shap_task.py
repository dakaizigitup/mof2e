"""GlobalShapTask

支持对 train/val/all 数据集进行全量或分批的 SHAP 分析，并内置了自动批处理和周期性保存功能。

YAML 配置示例:

shap:
  split: "val"
  num_explain: -1
  shap_batch_size: 4
  save_every_n_batches: 10  # ★ 每 N 批保存一次检查点和对应的图
  num_background: 20
  nsamples: 50
  global_xlsx_train: ...
  global_xlsx_val: ...
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data

from fairchem.core.common.registry import registry
from fairchem.core.tasks.task import BaseTask


def _setup_matplotlib_chinese_font():
    # ... (font setup logic remains the same)
    pass

def _graph_key(g: Data):
    if hasattr(g, "name") and g.name is not None:
        return str(g.name).split("_")[0].upper().strip()
    if hasattr(g, "sid") and g.sid is not None:
        return str(g.sid)
    return None


@registry.register_task("shap-global")
class GlobalShapTask(BaseTask):

    def _load_xlsx(self):
        if hasattr(self, "_xlsx_loaded"): return
        cfg = self.config.get("shap", {})
        paths = [cfg.get("global_xlsx_train"), cfg.get("global_xlsx_val")]
        dfs = [pd.read_excel(p) for p in paths if p and Path(p).exists()]
        if not dfs: raise FileNotFoundError("No valid xlsx path found.")
        df = pd.concat(dfs, ignore_index=True)
        self._xlsx_id_col = df.columns[0]
        self._df = df
        self._xlsx_loaded = True
        logging.info(f"Loaded {len(df)} total rows from {len(dfs)} xlsx files.")

    def _collect_all_graphs(self, loader: DataLoader) -> List[Data]:
        if loader is None: return []
        return [g for batch in loader for g in batch.to_data_list()]

    def _model_predict(self, x_batch: np.ndarray) -> np.ndarray:
        # Memory-safe, serial version
        outs = []
        graph_templates = getattr(self, "_current_graphs_for_batch", self.ex_graphs)

        for i in range(x_batch.shape[0]):
            g_template = graph_templates[i % len(graph_templates)].clone()
            raw_values = x_batch[i]
            
            emb = self.emb_module.encode_from_raw_values(raw_values, self._col_names)
            g_template.global_features = emb
            
            batch_g = Batch.from_data_list([g_template]).to(self.device)
            
            with torch.no_grad():
                y = self.model(batch_g)
                energy = y if torch.is_tensor(y) else y["energy"]
            
            outs.append(float(energy.squeeze().cpu()))
            
        return np.asarray(outs)

    def run(self):
        _setup_matplotlib_chinese_font()

        cfg = self.config.get("shap", {})
        B0 = cfg.get("num_background", 20)
        split = cfg.get("split", "val")
        start_idx = cfg.get("start_idx", 0)
        num_explain = cfg.get("num_explain", -1)
        nsamples = cfg.get("nsamples", 100)
        shap_batch_size = cfg.get("shap_batch_size", 4)
        save_every = cfg.get("save_every_n_batches", 10)

        self._load_xlsx()

        # -- Prepare Model and Encoders --
        self.model = self.trainer.model
        from torch.nn.parallel import DistributedDataParallel as DDP
        if isinstance(self.model, DDP): self.model = self.model.module

        self.emb_module = getattr(self.model, "mof_global_encoder", None) or getattr(self.model, "global_emb", None)
        assert self.emb_module is not None, "Model missing mof_global_encoder/global_emb"
        self.device = next(self.model.parameters()).device
        self.model.eval()

        # -- Define columns to be used in SHAP --
        numeric_cols = [c for c in getattr(self.emb_module, "numeric_features", []) if c in self._df.columns]
        cat_cols = [c for c in getattr(self.emb_module, "categorical_features", []) 
                    if c not in getattr(self.emb_module, "multi_label_features", set()) and c in self._df.columns]
        self._col_names = numeric_cols + cat_cols
        if not self._col_names: raise RuntimeError("No usable columns found.")

        # -- Prepare Data --
        id_col = self._xlsx_id_col
        raw_map: Dict[str, pd.Series] = {str(row[id_col]).upper().strip(): row for _, row in self._df.iterrows()}

        bg_graphs_all = self._collect_all_graphs(self.trainer.train_loader)
        bg_graphs_matched = [g for g in bg_graphs_all if _graph_key(g) in raw_map]
        if len(bg_graphs_matched) < B0: raise RuntimeError(f"Not enough background samples. Found {len(bg_graphs_matched)}, need {B0}.")
        bg_graphs = random.sample(bg_graphs_matched, B0)

        explain_graphs_raw = []
        if split in ("train", "all"): explain_graphs_raw.extend(bg_graphs_all)
        if split in ("val", "all"): explain_graphs_raw.extend(self._collect_all_graphs(self.trainer.val_loader))
        if not explain_graphs_raw: raise RuntimeError(f"No graphs for split '{split}'.")

        explain_graphs_matched = [g for g in explain_graphs_raw if _graph_key(g) in raw_map]
        
        if num_explain == -1:
            self.ex_graphs = explain_graphs_matched[start_idx:]
            end_idx = len(explain_graphs_matched)
        else:
            end_idx = start_idx + num_explain
            self.ex_graphs = explain_graphs_matched[start_idx:end_idx]
        
        if not self.ex_graphs: 
            logging.warning(f"No samples for split '{split}' in range {start_idx}-{end_idx}. Done."); return

        logging.info(f"Explaining {len(self.ex_graphs)} samples from split '{split}' (index {start_idx} to {end_idx}).")

        def row_to_vec(row: pd.Series) -> np.ndarray:
            vec = []
            for c in numeric_cols: vec.append(float(pd.to_numeric(row.get(c, 0), errors="coerce") or 0.0))
            for c in cat_cols:
                enc = self.emb_module.categorical_encoders.get(c)
                try: idx = int(enc.transform([str(row.get(c, ""))])[0])
                except: idx = 0
                vec.append(float(idx))
            return np.asarray(vec, dtype=np.float32)

        X_bg = np.stack([row_to_vec(raw_map[_graph_key(g)]) for g in bg_graphs])
        X_ex = np.stack([row_to_vec(raw_map[_graph_key(g)]) for g in self.ex_graphs])

        # -- Run SHAP with Automated Batching & Intermittent Saving --
        explainer = shap.KernelExplainer(self._model_predict, X_bg)
        all_phi = []
        num_total_explain = X_ex.shape[0]
        num_batches = (num_total_explain + shap_batch_size - 1) // shap_batch_size

        logging.info(f"Starting SHAP in {num_batches} batches of size {shap_batch_size}. Will save every {save_every} batches.")

        out_dir = Path(self.config.get("output"))
        out_dir.mkdir(parents=True, exist_ok=True)

        for i in range(0, num_total_explain, shap_batch_size):
            start, end = i, min(i + shap_batch_size, num_total_explain)
            X_ex_batch = X_ex[start:end]
            self._current_graphs_for_batch = self.ex_graphs[start:end]
            
            batch_idx = i // shap_batch_size
            logging.info(f"Processing batch {batch_idx + 1}/{num_batches} (samples {start}-{end-1})...")
            
            phi_batch = explainer.shap_values(X_ex_batch, nsamples=nsamples)
            all_phi.append(phi_batch)

            # --- Intermittent Saving Logic ---
            is_last_batch = (end >= num_total_explain)
            is_checkpoint_batch = ((batch_idx + 1) % save_every == 0)

            if (is_checkpoint_batch and not is_last_batch):
                phi_checkpoint = np.concatenate(all_phi, axis=0)
                X_ex_checkpoint = X_ex[:phi_checkpoint.shape[0]]
                
                checkpoint_filename_base = f"global_shap_{split}_{start_idx}-{start_idx + phi_checkpoint.shape[0] - 1}_checkpoint"
                
                torch.save({
                    "phi": phi_checkpoint, 
                    "cols": self._col_names, 
                    "X_ex": X_ex_checkpoint
                }, out_dir / f"{checkpoint_filename_base}.pt")
                logging.info(f"Saved checkpoint data to {checkpoint_filename_base}.pt")

                # Also save the beeswarm plot for the checkpoint
                try:
                    shap.summary_plot(
                        phi_checkpoint,
                        features=X_ex_checkpoint,
                        feature_names=self._col_names,
                        show=False,
                    )
                    plt.tight_layout()
                    plt.savefig(out_dir / f"{checkpoint_filename_base}_beeswarm.png", dpi=300)
                    plt.close()
                    logging.info(f"Saved checkpoint beeswarm plot to {checkpoint_filename_base}_beeswarm.png")
                except Exception as e:
                    logging.warning(f"Could not generate checkpoint plot: {e}")

        # --- Final Save and Plot ---
        phi = np.concatenate(all_phi, axis=0)

        filename_base = f"global_shap_{split}_{start_idx}-{end_idx-1}_final"
        torch.save({"phi": phi, "cols": self._col_names, "X_ex": X_ex}, out_dir / f"{filename_base}.pt")

        shap.summary_plot(phi, features=X_ex, feature_names=self._col_names, show=False)
        plt.tight_layout()
        plt.savefig(out_dir / f"{filename_base}_beeswarm.png", dpi=300)
        plt.close()

        logging.info(f"Global SHAP finished for split '{split}' [{start_idx}-{end_idx-1}].")
