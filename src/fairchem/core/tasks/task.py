"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import os

from fairchem.core.common.registry import registry
from fairchem.core.common.utils import get_checkpoint_format
from fairchem.core.common import distutils
from fairchem.core.trainers import OCPTrainer


class BaseTask:
    def __init__(self, config) -> None:
        self.config = config

    def setup(self, trainer) -> None:
        self.trainer = trainer

        # if `ignore_checkpoint` is set to True, we don't load any checkpoint.
        if self.config.get("task", {}).get("ignore_checkpoint", False):
            if distutils.is_master():
                logging.info("Found `ignore_checkpoint: true`, starting from scratch.")
            return

        format = get_checkpoint_format(self.config)
        if format == "pt":
            self.chkpt_path = os.path.join(
                self.trainer.config["cmd"]["checkpoint_dir"], "checkpoint.pt"
            )
        else:
            self.chkpt_path = self.trainer.config["cmd"]["checkpoint_dir"]

        # if the supplied checkpoint exists, then load that, ie: when user specifies the --checkpoint option
        # OR if the a job was preempted correctly and the submitit checkpoint function was called
        # (https://github.com/FAIR-Chem/fairchem/blob/main/src/fairchem/core/_cli.py#L44), then we should attempt to
        # load that checkpoint
        if self.config["checkpoint"] is not None:
            logging.info(
                f"Attemping to load user specified checkpoint at {self.config['checkpoint']}"
            )
            self.trainer.load_checkpoint(checkpoint_path=self.config["checkpoint"])
        # if the supplied checkpoint doesn't exist and there exists a previous checkpoint in the checkpoint path, this
        # means that the previous job didn't terminate "nicely" (due to node failures, crashes etc), then attempt
        # to load the last found checkpoint
        ############
        # else:
        #     logging.info("⚠️ No checkpoint found. Initializing a fresh model for evaluation.")
        #     # 重新实例化模型（使用 config）
        #     model_class = registry.get_model_class(self.config["model"]["name"])
        #     # 复制一份模型配置，因为直接修改原始config可能不是一个好习惯
        #     model_kwargs = self.config["model"].copy()
        #     # 移除 'name' 键，因为它不是模型构造函数的参数
        #     model_kwargs.pop("name", None)
        #     # 使用清理过的参数字典来实例化模型
        #     self.trainer.model = model_class(**model_kwargs)
        #
        #     # 把模型移动到 GPU
        #     self.trainer.model.to(self.trainer.device)
        #
        #     # 打印提示
        #     logging.info("✅ New model created (no checkpoint loaded). "
        #                  "Will use pretrained weights if defined inside model class.")
        #############自动恢复##############
        elif (
            os.path.isfile(self.chkpt_path)
            or (os.path.isdir(self.chkpt_path) and len(os.listdir(self.chkpt_path))) > 0
        ):
            logging.info(
                f"Previous checkpoint found at {self.chkpt_path}, resuming job from this checkecpoint"
            )
            self.trainer.load_checkpoint(checkpoint_path=self.chkpt_path)

    def run(self):
        raise NotImplementedError


@registry.register_task("train")
class TrainTask(BaseTask):
    def _process_error(self, e: RuntimeError) -> None:
        e_str = str(e)
        if (
            "find_unused_parameters" in e_str
            and "torch.nn.parallel.DistributedDataParallel" in e_str
        ):
            for name, parameter in self.trainer.model.named_parameters():
                if parameter.requires_grad and parameter.grad is None:
                    logging.warning(
                        f"Parameter {name} has no gradient. Consider removing it from the model."
                    )

    def run(self) -> None:
        try:
            self.trainer.train(
                disable_eval_tqdm=self.config.get("hide_eval_progressbar", False)
            )
        except RuntimeError as e:
            self._process_error(e)
            raise e


@registry.register_task("predict")
class PredictTask(BaseTask):
    def run(self) -> None:
        assert (
            self.trainer.test_loader is not None
        ), "Test dataset is required for making predictions"
        assert self.config["checkpoint"]
        results_file = "predictions"
        self.trainer.predict(
            self.trainer.test_loader,
            results_file=results_file,
            disable_tqdm=self.config.get("hide_eval_progressbar", False),
        )


@registry.register_task("validate")
class ValidateTask(BaseTask):
    def run(self) -> None:
        # Note that the results won't be precise on multi GPUs due to padding of extra images (although the difference should be minor)
        assert (
            self.trainer.val_loader is not None
        ), "Val dataset is required for making predictions"
        assert self.config["checkpoint"]
        self.trainer.validate(
            split="val",
            disable_tqdm=self.config.get("hide_eval_progressbar", False),
        )


@registry.register_task("run-relaxations")
class RelaxationTask(BaseTask):
    def run(self) -> None:
        assert isinstance(
            self.trainer, OCPTrainer
        ), "Relaxations are only possible for ForcesTrainer"
        assert (
            self.trainer.relax_dataset is not None
        ), "Relax dataset is required for making predictions"
        assert self.config["checkpoint"]
        self.trainer.run_relaxations()


@registry.register_task("vis_atten")
class VisAttenTask(BaseTask):
    """Extract and visualize EquiformerV2 attention weights for every sample in an LMDB dataset.

    Config keys expected:
      - checkpoint: path to checkpoint (required)
      - lmdb_dir: path to lmdb folder/file
      - output: output directory
      - batch_size: optional, default 1
      - num_workers: optional, default 0
      - max_samples: optional, limit number of samples

    Outputs:
      - <output>/atten/sample_<idx>.pt : raw attention tensors and metadata
      - <output>/atten/sample_<idx>.png: simple per-edge attention heat visualization (2D)

    Note: Attention weights are input-dependent; we run a forward pass per sample.
    """

    def run(self) -> None:
        import math
        from pathlib import Path

        import matplotlib.pyplot as plt
        import torch
        from torch_geometric.loader import DataLoader

        from fairchem.core.datasets.lmdb_dataset import LmdbDataset

        assert self.config.get("checkpoint"), "vis_atten requires config['checkpoint']"
        lmdb_dir = self.config.get("lmdb_dir")
        out_dir = self.config.get("output")
        assert lmdb_dir is not None, "vis_atten requires config['lmdb_dir']"
        assert out_dir is not None, "vis_atten requires config['output']"

        out_dir = Path(out_dir)
        atten_dir = out_dir / "atten"
        atten_dir.mkdir(parents=True, exist_ok=True)

        # --- dataset/dataloader ---
        ds = LmdbDataset({"src": lmdb_dir})
        max_samples = self.config.get("max_samples", None)
        if max_samples is not None:
            max_samples = int(max_samples)

        batch_size = int(self.config.get("batch_size", 1))
        num_workers = int(self.config.get("num_workers", 0))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # --- patch EquiformerV2 attention modules to cache alpha ---
        model = self.trainer.model
        device = self.trainer.device
        model.eval()

        def _patch_attention_modules(m: torch.nn.Module):
            """Monkey-patch SO2EquivariantGraphAttention.forward to cache attention weights."""
            cls_name = type(m).__name__
            if cls_name != "SO2EquivariantGraphAttention":
                return
            if getattr(m, "_vis_atten_patched", False):
                return

            orig_forward = m.forward

            def wrapped_forward(self, x, atomic_numbers, edge_distance, edge_index, node_offset: int = 0):
                # Call original forward, but re-compute alpha identically to cache it.
                # To avoid duplicating full compute graph, we do a no_grad recomputation
                # by reusing internal ops. This is safe because we are in eval/inference.
                #
                # We cannot access local variable `alpha` from orig_forward without editing upstream source.
                # So we re-run the relevant alpha path with torch.no_grad().

                out = orig_forward(x, atomic_numbers, edge_distance, edge_index, node_offset=node_offset)

                # Best-effort: try to use internal tensors if present (not guaranteed).
                # If fails, cache None.
                try:
                    with torch.no_grad():
                        # We re-execute the minimal alpha computation by duplicating the original forward logic.
                        # This requires importing the same utilities used in the module.
                        import torch_geometric
                        from fairchem.core.common import gp_utils
                        from fairchem.core.models.equiformer_v2.so3 import SO3_Embedding

                        # Compute edge scalar features
                        if self.use_atom_edge_embedding:
                            source_element = atomic_numbers[edge_index[0]]
                            target_element = atomic_numbers[edge_index[1]]
                            source_embedding = self.source_embedding(source_element)
                            target_embedding = self.target_embedding(target_element)
                            x_edge = torch.cat((edge_distance, source_embedding, target_embedding), dim=1)
                        else:
                            x_edge = edge_distance

                        x_source = x.clone()
                        x_target = x.clone()
                        if gp_utils.initialized():
                            x_full = gp_utils.gather_from_model_parallel_region(x.embedding, dim=0)
                            x_source.set_embedding(x_full)
                            x_target.set_embedding(x_full)
                        x_source._expand_edge(edge_index[0, :])
                        x_target._expand_edge(edge_index[1, :])

                        x_message_data = torch.cat((x_source.embedding, x_target.embedding), dim=2)
                        x_message = SO3_Embedding(
                            0,
                            x_target.lmax_list.copy(),
                            x_target.num_channels * 2,
                            device=x_target.device,
                            dtype=x_target.dtype,
                        )
                        x_message.set_embedding(x_message_data)
                        x_message.set_lmax_mmax(self.lmax_list.copy(), self.mmax_list.copy())

                        if self.use_m_share_rad:
                            x_edge_weight = self.rad_func(x_edge)
                            x_edge_weight = x_edge_weight.reshape(
                                -1, (max(self.lmax_list) + 1), 2 * self.sphere_channels
                            )
                            x_edge_weight = torch.index_select(x_edge_weight, dim=1, index=self.expand_index)
                            x_message.embedding = x_message.embedding * x_edge_weight

                        x_message._rotate(self.SO3_rotation, self.lmax_list, self.mmax_list)

                        # First conv
                        x_message, x_0_extra = self.so2_conv_1(x_message, x_edge)

                        # Activation and alpha features
                        x_alpha_num_channels = self.num_heads * self.attn_alpha_channels
                        if self.use_gate_act:
                            x_0_alpha = x_0_extra.narrow(1, 0, x_alpha_num_channels)
                        else:
                            if self.use_sep_s2_act:
                                x_0_alpha = x_0_extra.narrow(1, 0, x_alpha_num_channels)
                            else:
                                x_0_alpha = x_0_extra

                        x_0_alpha = x_0_alpha.reshape(-1, self.num_heads, self.attn_alpha_channels)
                        x_0_alpha = self.alpha_norm(x_0_alpha)
                        x_0_alpha = self.alpha_act(x_0_alpha)
                        alpha = torch.einsum("bik, ik -> bi", x_0_alpha, self.alpha_dot)
                        alpha = torch_geometric.utils.softmax(alpha, edge_index[1])
                        # cache as [E, H]
                        self.last_alpha = alpha.detach().cpu()
                        self.last_edge_index = edge_index.detach().cpu()
                except Exception:
                    self.last_alpha = None
                    self.last_edge_index = None

                return out

            m.forward = wrapped_forward.__get__(m, type(m))  # bind
            m._vis_atten_patched = True

        for mod in model.modules():
            _patch_attention_modules(mod)

        # helper to collect cached alphas
        def collect_attention(mod: torch.nn.Module):
            attn = []
            for m in mod.modules():
                if type(m).__name__ == "SO2EquivariantGraphAttention" and hasattr(m, "last_alpha"):
                    attn.append(
                        {
                            "alpha": m.last_alpha,
                            "edge_index": getattr(m, "last_edge_index", None),
                            "num_heads": getattr(m, "num_heads", None),
                        }
                    )
            return attn

        # --- run inference and dump per-sample attention ---
        seen = 0
        global_idx = 0
        for batch in loader:
            if max_samples is not None and seen >= max_samples:
                break

            batch = batch.to(device)
            with torch.no_grad():
                _ = model(batch)

            # split to per-graph
            graphs = batch.to_data_list()
            for g in graphs:
                if max_samples is not None and seen >= max_samples:
                    break

                # Attention cached is for the last batch forward; for simplicity we re-run per-sample
                g = g.to(device)
                with torch.no_grad():
                    _ = model(g)
                attn_layers = collect_attention(model)

                # save raw
                save_obj = {
                    "idx": global_idx,
                    "attn_layers": attn_layers,
                    "num_nodes": int(g.num_nodes),
                }
                if hasattr(g, "pos"):
                    save_obj["pos"] = g.pos.detach().cpu()
                if hasattr(g, "atomic_numbers"):
                    save_obj["atomic_numbers"] = g.atomic_numbers.detach().cpu()
                elif hasattr(g, "z"):
                    save_obj["atomic_numbers"] = g.z.detach().cpu()

                torch.save(save_obj, atten_dir / f"sample_{global_idx}.pt")

                # quick-and-dirty visualization: mean attention per edge (mean over heads, last layer)
                try:
                    # pick last layer that has alpha
                    last = None
                    for item in reversed(attn_layers):
                        if item.get("alpha") is not None:
                            last = item
                            break
                    if last is not None and last["alpha"] is not None:
                        alpha = last["alpha"]  # [E, H?] or [E]
                        if alpha.dim() == 2:
                            alpha_mean = alpha.mean(dim=1)
                        else:
                            alpha_mean = alpha
                        alpha_mean = alpha_mean.numpy()

                        # plot as a heatmap-like bar image
                        fig_w = 6
                        fig_h = max(2, min(8, math.ceil(len(alpha_mean) / 200)))
                        plt.figure(figsize=(fig_w, fig_h))
                        plt.imshow(alpha_mean[None, :], aspect="auto", cmap="viridis")
                        plt.yticks([])
                        plt.xlabel("edge index")
                        plt.colorbar(label="attention (mean over heads)")
                        plt.tight_layout()
                        plt.savefig(atten_dir / f"sample_{global_idx}.png", dpi=200)
                        plt.close()
                except Exception as e:
                    logging.warning(f"vis_atten: failed to plot sample {global_idx}: {e}")

                seen += 1
                global_idx += 1

        logging.info(f"vis_atten complete. saved {seen} samples to {atten_dir}")
