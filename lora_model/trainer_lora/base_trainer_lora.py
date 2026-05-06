"""
Modified BaseTrainer with integrated multi-parameter-group optimizer & scheduler support.
Added functionality:
  - Automatic detection of multi-group config (lora / mof_encoder / param_groups)
  - Parameter grouping by name substrings or explicit patterns
  - Separate LR / WD / betas per group
  - Scheduler support (ReduceLROnPlateau / StepLR / MultiStepLR / Null)
  - Per-group LR logging helper
  - Backwards-compatible fallback to original single-group (decay/no_decay) logic
"""

from __future__ import annotations

import copy
import datetime
import errno
import logging
import os
import random
import sys
from abc import ABC, abstractmethod
from functools import partial
from itertools import chain
from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np
import numpy.typing as npt
import torch
import yaml
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

from fairchem.core import __version__
from fairchem.core.common import distutils, gp_utils
from fairchem.core.common.data_parallel import BalancedBatchSampler
from fairchem.core.common.logger import WandBSingletonLogger
from fairchem.core.common.registry import registry
from fairchem.core.common.slurm import add_timestamp_id_to_submission_pickle
from fairchem.core.common.typing import assert_is_instance as aii
from fairchem.core.common.typing import none_throws
from fairchem.core.common.utils import (
    get_commit_hash,
    get_weight_table,
    load_state_dict,
    match_state_dict,
    save_checkpoint,
    update_config,
)
from fairchem.core.datasets import data_list_collater
from fairchem.core.datasets.base_dataset import create_dataset
from fairchem.core.modules.evaluator import Evaluator
from fairchem.core.modules.exponential_moving_average import ExponentialMovingAverage
from fairchem.core.modules.loss import DDPLoss
from fairchem.core.modules.normalization.element_references import (
    create_element_references,
    load_references_from_config,
)
from fairchem.core.modules.normalization.normalizer import (
    create_normalizer,
    load_normalizers_from_config,
)
from fairchem.core.modules.scaling.compat import load_scales_compat
from fairchem.core.modules.scaling.util import ensure_fitted
from fairchem.core.modules.scheduler import LRScheduler

if TYPE_CHECKING:
    from collections.abc import Sequence


# === Multi-Group LR MOD: Simple wrapper to mimic original LRScheduler interface ===
class _SimpleSchedulerWrapper:
    """
    Wrap a torch scheduler to provide:
      - scheduler (actual torch scheduler)
      - scheduler_type (string)
    """

    def __init__(self, scheduler: Any, scheduler_type: str):
        self.scheduler = scheduler
        self.scheduler_type = scheduler_type

    def step(self, *args, **kwargs):
        if self.scheduler_type == "ReduceLROnPlateau":
            # external should pass metric
            return self.scheduler.step(*args, **kwargs)
        else:
            return self.scheduler.step()


# === Multi-Group LR MOD: Null scheduler (no-op) ===
class _NullScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def state_dict(self):
        return {}

    def load_state_dict(self, _):
        pass

    def step(self, *args, **kwargs):
        pass


@registry.register_trainer("base_lora")
class BaseTrainer_lora(ABC):
    def __init__(
        self,
        task: dict[str, str | Any],
        model: dict[str, Any],
        outputs: dict[str, str | int],
        dataset: dict[str, str | float],
        optimizer: dict[str, str | float],
        loss_functions: dict[str, str | float],
        evaluation_metrics: dict[str, str],
        identifier: str,
        local_rank: int,
        timestamp_id: str | None = None,
        run_dir: str | None = None,
        is_debug: bool = False,
        print_every: int = 100,
        seed: int | None = None,
        logger: str = "wandb",
        amp: bool = False,
        cpu: bool = False,
        name: str = "ocp",
        slurm=None,
        gp_gpus: int | None = None,
        inference_only: bool = False,
    ) -> None:
        if slurm is None:
            slurm = {}
        self.name = name
        self.is_debug = is_debug
        self.cpu = cpu
        self.epoch = 0
        self.step = 0
        self.ema = None

        if torch.cuda.is_available() and not self.cpu:
            logging.info(f"local rank base: {local_rank}")
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device("cpu")
            self.cpu = True

        if run_dir is None:
            run_dir = os.getcwd()

        if timestamp_id is None:
            timestamp_id = self._get_timestamp(self.device, identifier)
        self.timestamp_id = none_throws(timestamp_id)

        commit_hash = get_commit_hash()
        logger_name = logger if isinstance(logger, str) else logger["name"]
        self.config = {
            "task": task,
            "trainer": name,
            "model": model,
            "outputs": outputs,
            "optim": optimizer,
            "loss_functions": loss_functions,
            "evaluation_metrics": evaluation_metrics,
            "logger": logger,
            "amp": amp,
            "gpus": distutils.get_world_size() if not self.cpu else 0,
            "cmd": {
                "identifier": identifier,
                "print_every": print_every,
                "seed": seed,
                "timestamp_id": self.timestamp_id,
                "commit": commit_hash,
                "version": __version__,
                "checkpoint_dir": os.path.join(run_dir, "checkpoints", self.timestamp_id),
                "results_dir": os.path.join(run_dir, "results", self.timestamp_id),
                "logs_dir": os.path.join(
                    run_dir, "logs", logger_name, self.timestamp_id
                ),
            },
            "slurm": slurm,
            "gp_gpus": gp_gpus,
        }

        # AMP
        self.scaler = torch.GradScaler("cuda") if amp and not self.cpu else None

        # SLURM
        if "SLURM_JOB_ID" in os.environ and "folder" in self.config["slurm"]:
            if "SLURM_ARRAY_JOB_ID" in os.environ:
                self.config["slurm"]["job_id"] = "{}_{}".format(
                    os.environ["SLURM_ARRAY_JOB_ID"],
                    os.environ["SLURM_ARRAY_TASK_ID"],
                )
            else:
                self.config["slurm"]["job_id"] = os.environ["SLURM_JOB_ID"]
            self.config["slurm"]["folder"] = self.config["slurm"]["folder"].replace(
                "%j", self.config["slurm"]["job_id"]
            )
            if distutils.is_master():
                add_timestamp_id_to_submission_pickle(
                    self.config["slurm"]["folder"],
                    self.config["slurm"]["job_id"],
                    self.timestamp_id,
                )

        # Dataset config normalize
        if isinstance(dataset, list):
            if len(dataset) > 0:
                self.config["dataset"] = dataset[0]
            if len(dataset) > 1:
                self.config["val_dataset"] = dataset[1]
            if len(dataset) > 2:
                self.config["test_dataset"] = dataset[2]
        elif isinstance(dataset, dict):
            self.config["dataset"] = dataset.get("train", {}) or {}
            self.config["val_dataset"] = dataset.get("val", {}) or {}
            self.config["test_dataset"] = dataset.get("test", {}) or {}
            self.config["relax_dataset"] = dataset.get("relax", {}) or {}
        else:
            self.config["dataset"] = dataset or {}

        for dataset_name in ("val_dataset", "test_dataset", "relax_dataset"):
            if dataset_name not in self.config:
                self.config[dataset_name] = {}

        if not is_debug and distutils.is_master():
            os.makedirs(self.config["cmd"]["checkpoint_dir"], exist_ok=True)
            os.makedirs(self.config["cmd"]["results_dir"], exist_ok=True)
            os.makedirs(self.config["cmd"]["logs_dir"], exist_ok=True)

        # Backward compat adjustments
        self.config = update_config(self.config)

        if distutils.is_master():
            logging.info(yaml.dump(self.config, default_flow_style=False))

        # Placeholders
        self.elementrefs = {}
        self.normalizers = {}
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.best_val_metric = None
        self.primary_metric = None
        self.ema = None

        self.load(inference_only)

    @abstractmethod
    def train(self, disable_eval_tqdm: bool = False) -> None:
        pass

    @staticmethod
    def _get_timestamp(device: torch.device, suffix: str | None) -> str:
        now = datetime.datetime.now().timestamp()
        timestamp_tensor = torch.tensor(now).to(device)
        distutils.broadcast(timestamp_tensor, 0)
        timestamp_str = datetime.datetime.fromtimestamp(
            timestamp_tensor.float().item()
        ).strftime("%Y-%m-%d-%H-%M-%S")
        if suffix:
            timestamp_str += "-" + suffix
        return timestamp_str

    def load(self, inference_only: bool) -> None:
        self.load_seed_from_config()
        self.load_logger()
        self.load_task()
        self.load_model()
        if not inference_only:
            self.load_datasets()
            self.load_references_and_normalizers()
            self.load_loss()
            # Multi-group 或 原逻辑
            self.load_optimizer()
            self.load_extras()

        if self.config["optim"].get("load_datasets_and_model_then_exit", False):
            sys.exit(0)

    # ==================================================================================
    # Seed / Logger
    # ==================================================================================
    @staticmethod
    def set_seed(seed) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def load_seed_from_config(self) -> None:
        if self.config["cmd"]["seed"] is None:
            return
        self.set_seed(self.config["cmd"]["seed"])

    def load_logger(self) -> None:
        self.logger = None
        if not self.is_debug and distutils.is_master():
            assert self.config["logger"] is not None, "Specify logger in config"
            logger = self.config["logger"]
            logger_name = logger if isinstance(logger, str) else logger["name"]
            assert logger_name, "Specify logger name"
            if logger_name == "wandb_singleton":
                WandBSingletonLogger.init_wandb(
                    config=self.config,
                    run_id=self.config["cmd"]["timestamp_id"],
                    run_name=self.config["cmd"]["identifier"],
                    log_dir=self.config["cmd"]["logs_dir"],
                    project=self.config["logger"]["project"],
                    entity=self.config["logger"]["entity"],
                    group=self.config["logger"].get("group", ""),
                )
                self.logger = WandBSingletonLogger.get_instance()
            else:
                self.logger = registry.get_logger_class(logger_name)(self.config)

    # ==================================================================================
    # Dataset
    # ==================================================================================
    def get_sampler(
        self, dataset, batch_size: int, shuffle: bool
    ) -> BalancedBatchSampler:
        balancing_mode = self.config["optim"].get("load_balancing", None)
        on_error = self.config["optim"].get("load_balancing_on_error", None)
        if balancing_mode is not None:
            if on_error is None:
                on_error = "raise"
        else:
            balancing_mode = "atoms"
        if on_error is None:
            on_error = "warn_and_no_balance"

        if gp_utils.initialized():
            num_replicas = gp_utils.get_dp_world_size()
            rank = gp_utils.get_dp_rank()
        else:
            num_replicas = distutils.get_world_size()
            rank = distutils.get_rank()
        return BalancedBatchSampler(
            dataset,
            batch_size=batch_size,
            num_replicas=num_replicas,
            rank=rank,
            device=self.device,
            mode=balancing_mode,
            shuffle=shuffle,
            on_error=on_error,
            seed=self.config["cmd"]["seed"],
        )

    def get_dataloader(self, dataset, sampler, workers=None) -> DataLoader:
        num_workers = (
            self.config["optim"]["num_workers"] if workers is None else workers
        )
        return DataLoader(
            dataset,
            collate_fn=self.collater,
            num_workers=num_workers,
            pin_memory=True,
            batch_sampler=sampler,
        )

    def load_datasets(self) -> None:
        self.collater = partial(
            data_list_collater, otf_graph=self.config["model"].get("otf_graph", False)
        )
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        def convert_settings_to_split_settings(config, split_name):
            config = copy.deepcopy(config)
            if f"{split_name}_split_settings" in config:
                config["splits"] = {
                    split_name: config.pop(f"{split_name}_split_settings")
                }
            return config

        if "src" in self.config["dataset"]:
            logging.info(
                f"Loading dataset: {self.config['dataset'].get('format', 'lmdb')}"
            )
            self.train_dataset = create_dataset(
                convert_settings_to_split_settings(self.config["dataset"], "train"),
                "train",
            )
            self.train_sampler = self.get_sampler(
                self.train_dataset,
                self.config["optim"].get("batch_size", 1),
                shuffle=True,
            )
            self.train_loader = self.get_dataloader(
                self.train_dataset,
                self.train_sampler,
            )

        if (
            "first_n" in self.config["dataset"]
            or "sample_n" in self.config["dataset"]
            or "max_atom" in self.config["dataset"]
        ):
            logging.warning(
                "Dataset attributes (first_n/sample_n/max_atom) applied to all splits. "
                "Prefer specifying them per-split to avoid unintended effects."
            )

        if "src" in self.config["val_dataset"]:
            if self.config["val_dataset"].get("use_train_settings", True):
                val_config = self.config["dataset"].copy()
                val_config.update(self.config["val_dataset"])
            else:
                val_config = self.config["val_dataset"]
            self.val_dataset = create_dataset(
                convert_settings_to_split_settings(val_config, "val"), "val"
            )
            self.val_sampler = self.get_sampler(
                self.val_dataset,
                self.config["optim"].get(
                    "eval_batch_size", self.config["optim"].get("batch_size", 1)
                ),
                shuffle=False,
            )
            self.val_loader = self.get_dataloader(
                self.val_dataset,
                self.val_sampler,
            )

        if "src" in self.config["test_dataset"]:
            if self.config["test_dataset"].get("use_train_settings", True):
                test_config = self.config["dataset"].copy()
                test_config.update(self.config["test_dataset"])
            else:
                test_config = self.config["test_dataset"]
            self.test_dataset = create_dataset(
                convert_settings_to_split_settings(test_config, "test"), "test"
            )
            self.test_sampler = self.get_sampler(
                self.test_dataset,
                self.config["optim"].get(
                    "eval_batch_size", self.config["optim"].get("batch_size", 1)
                ),
                shuffle=False,
            )
            self.test_loader = self.get_dataloader(
                self.test_dataset,
                self.test_sampler,
            )

        if self.config["relax_dataset"]:
            if self.config["relax_dataset"].get("use_train_settings", True):
                relax_config = self.config["dataset"].copy()
                relax_config.update(self.config["relax_dataset"])
            else:
                relax_config = self.config["relax_dataset"]
            self.relax_dataset = registry.get_dataset_class(
                relax_config.get("format", "lmdb")
            )(relax_config)
            self.relax_sampler = self.get_sampler(
                self.relax_dataset,
                self.config["optim"].get(
                    "eval_batch_size", self.config["optim"]["batch_size"]
                ),
                shuffle=False,
            )
            self.relax_loader = self.get_dataloader(
                self.relax_dataset,
                self.relax_sampler,
            )

    # ==================================================================================
    # References & Normalizers
    # ==================================================================================
    def load_references_and_normalizers(self):
        elementref_config = (
            self.config["dataset"].get("transforms", {}).get("element_references")
        )
        norms_config = self.config["dataset"].get("transforms", {}).get("normalizer")
        elementrefs, normalizers = {}, {}
        if distutils.is_master():
            if elementref_config is not None:
                elementrefs = load_references_from_config(
                    elementref_config,
                    dataset=self.train_dataset,
                    seed=self.config["cmd"]["seed"],
                    checkpoint_dir=(
                        self.config["cmd"]["checkpoint_dir"]
                        if not self.is_debug
                        else None
                    ),
                )
            if norms_config is not None:
                normalizers = load_normalizers_from_config(
                    norms_config,
                    dataset=self.train_dataset,
                    seed=self.config["cmd"]["seed"],
                    checkpoint_dir=(
                        self.config["cmd"]["checkpoint_dir"]
                        if not self.is_debug
                        else None
                    ),
                    element_references=elementrefs,
                )
                for output, normalizer in normalizers.items():
                    logging.info(
                        f"Normalization values for {output}: "
                        f"mean={normalizer.mean.item()}, rmsd={normalizer.rmsd.item()}."
                    )

        elementrefs, normalizers = [elementrefs], [normalizers]
        distutils.broadcast_object_list(
            object_list=elementrefs, src=0, device=self.device
        )
        distutils.broadcast_object_list(
            object_list=normalizers, src=0, device=self.device
        )
        self.elementrefs.update(
            {
                output: elementref.to(self.device)
                for output, elementref in elementrefs[0].items()
            }
        )
        self.normalizers.update(
            {
                output: normalizer.to(self.device)
                for output, normalizer in normalizers[0].items()
            }
        )

    # ==================================================================================
    # Task / Model
    # ==================================================================================
    def load_task(self):
        self.output_targets = {}
        for target_name in self.config["outputs"]:
            self.output_targets[target_name] = self.config["outputs"][target_name]
            if "decomposition" in self.config["outputs"][target_name]:
                for subtarget in self.config["outputs"][target_name]["decomposition"]:
                    self.output_targets[subtarget] = (
                        self.config["outputs"][target_name]["decomposition"]
                    )[subtarget]
                    self.output_targets[subtarget]["parent"] = target_name
                    if "level" not in self.output_targets[subtarget]:
                        self.output_targets[subtarget]["level"] = self.config[
                            "outputs"
                        ][target_name].get("level", "system")
                    if "train_on_free_atoms" not in self.output_targets[subtarget]:
                        self.output_targets[subtarget]["train_on_free_atoms"] = (
                            self.config["outputs"][target_name].get(
                                "train_on_free_atoms", True
                            )
                        )
                    if "eval_on_free_atoms" not in self.output_targets[subtarget]:
                        self.output_targets[subtarget]["eval_on_free_atoms"] = (
                            self.config["outputs"][target_name].get(
                                "eval_on_free_atoms", True
                            )
                        )

        self.evaluation_metrics = self.config.get("evaluation_metrics", {})
        self.evaluator = Evaluator(
            task=self.name,
            eval_metrics=self.evaluation_metrics.get(
                "metrics", Evaluator.task_metrics.get(self.name, {})
            ),
        )

    def load_model(self) -> None:
        if distutils.is_master():
            logging.info(f"Loading model: {self.config['model']['name']}")
        model_config_copy = copy.deepcopy(self.config["model"])
        model_name = model_config_copy.pop("name")
        self.model = registry.get_model_class(model_name)(
            **model_config_copy,
        ).to(self.device)

        num_params = sum(p.numel() for p in self.model.parameters())
        if distutils.is_master():
            logging.info(
                f"Loaded {self.model.__class__.__name__} with {num_params} parameters."
            )

        if self.logger is not None:
            if "watch" in self.config["logger"]:
                self.logger.watch(
                    self.model, log_freq=int(self.config["logger"]["watch"])
                )
            if hasattr(self.logger, "log_summary"):
                self.logger.log_summary({"num_params": num_params})

        if distutils.initialized():
            self.model = DistributedDataParallel(self.model)

    @property
    def _unwrapped_model(self):
        module = self.model
        while isinstance(module, DistributedDataParallel):
            module = module.module
        return module

    # ==================================================================================
    # Checkpoint
    # ==================================================================================
    def load_checkpoint(
        self,
        checkpoint_path: str,
        checkpoint: dict | None = None,
        inference_only: bool = False,
    ) -> None:
        map_location = torch.device("cpu") if self.cpu else self.device
        if checkpoint is None:
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(
                    errno.ENOENT, "Checkpoint file not found", checkpoint_path
                )
            logging.info(f"Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=map_location)

        if not inference_only:
            self.epoch = checkpoint.get("epoch", 0)
            self.step = checkpoint.get("step", 0)
            self.best_val_metric = checkpoint.get("best_val_metric", None)
            self.primary_metric = checkpoint.get("primary_metric", None)

            if "optimizer" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            if "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
                self.scheduler.scheduler.load_state_dict(checkpoint["scheduler"])
        else:
            logging.info(
                "Loading checkpoint in inference-only mode; skipping trainer state."
            )

        if "ema" in checkpoint and checkpoint["ema"] is not None and self.ema:
            self.ema.load_state_dict(checkpoint["ema"])
        else:
            self.ema = None

        new_dict = match_state_dict(self.model.state_dict(), checkpoint["state_dict"])
        strict = self.config.get("task", {}).get("strict_load", True)
        load_state_dict(self.model, new_dict, strict=strict)

        scale_dict = checkpoint.get("scale_dict", None)
        if scale_dict:
            logging.info("Overwriting scaling factors from checkpoint scale_dict.")
            load_scales_compat(self._unwrapped_model, scale_dict)

        for key, state_dict in checkpoint["normalizers"].items():
            if key == "target":
                target_key = "energy"
            elif key == "grad_target":
                target_key = "forces"
            else:
                target_key = key
            if target_key not in self.normalizers:
                self.normalizers[target_key] = create_normalizer(state_dict=state_dict)
            else:
                mkeys = self.normalizers[target_key].load_state_dict(state_dict)
                assert len(mkeys.missing_keys) == 0
                assert len(mkeys.unexpected_keys) == 0
            self.normalizers[target_key].to(map_location)

        for key, state_dict in checkpoint.get("elementrefs", {}).items():
            if key not in self.elementrefs:
                self.elementrefs[key] = create_element_references(state_dict=state_dict)
            else:
                mkeys = self.elementrefs[key].load_state_dict(state_dict)
                assert len(mkeys.missing_keys) == 0
                assert len(mkeys.unexpected_keys) == 0
            self.elementrefs[key].to(map_location)

        if self.scaler and checkpoint.get("amp"):
            self.scaler.load_state_dict(checkpoint["amp"])

    # ==================================================================================
    # Loss
    # ==================================================================================
    def load_loss(self) -> None:
        self.loss_functions = []
        for _idx, loss in enumerate(self.config["loss_functions"]):
            for target in loss:
                assert "fn" in loss[target], f"Missing 'fn' in {loss[target]}"
                loss_name = loss[target].get("fn")
                assert "coefficient" in loss[target], f"Missing 'coefficient' in {loss[target]}"
                coefficient = loss[target].get("coefficient")
                loss_reduction = loss[target].get("reduction")
                loss_fn = DDPLoss(loss_name, reduction=loss_reduction)
                self.loss_functions.append(
                    (target, {"fn": loss_fn, "coefficient": coefficient})
                )

    # ==================================================================================
    # === Multi-Group LR MOD: detection helper
    # ==================================================================================
    def _multi_group_mode_enabled(self) -> bool:
        optim_cfg = self.config.get("optim", {})
        if "param_groups" in optim_cfg and optim_cfg["param_groups"]:
            return True
        if "lora" in optim_cfg or "mof_encoder" in optim_cfg:
            return True
        return False

    # ==================================================================================
    # === Multi-Group LR MOD: implicit grouping (lora/mof/other)
    # ==================================================================================
    def setup_parameter_groups(self) -> Dict[str, List[tuple[str, torch.nn.Parameter]]]:
        lora_params, mof_encoder_params, other_params = [], [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "lora_A" in name or "lora_B" in name:
                lora_params.append((name, param))
            elif "mof_global_encoder" in name:
                mof_encoder_params.append((name, param))
            else:
                other_params.append((name, param))
        return {
            "lora_params": lora_params,
            "mof_encoder_params": mof_encoder_params,
            "other_params": other_params,
        }

    # ==================================================================================
    # === Multi-Group LR MOD: build optimizer & scheduler
    # ==================================================================================
    def setup_optimizer_and_scheduler(self):
        optim_cfg = self.config["optim"]
        base_optim_name = optim_cfg.get("optimizer", "AdamW")
        OptimClass = getattr(torch.optim, base_optim_name)
        optimizer_params = optim_cfg.get("optimizer_params", {})
        default_lr = optim_cfg.get("lr_initial", optim_cfg.get("lr", 1e-4))
        default_wd = optimizer_params.get(
            "weight_decay", optim_cfg.get("weight_decay", 0.0)
        )

        nowd_suffixes = set()
        if hasattr(self._unwrapped_model, "no_weight_decay"):
            try:
                nwd = self._unwrapped_model.no_weight_decay()
                if isinstance(nwd, (list, tuple, set)):
                    nowd_suffixes = set(nwd)
            except Exception:
                pass

        def split_decay(pairs, group_wd):
            decay_list, no_decay_list = [], []
            for name, param in pairs:
                if any(name.endswith(suf) for suf in nowd_suffixes):
                    no_decay_list.append(param)
                else:
                    decay_list.append(param)
            out = []
            if no_decay_list:
                out.append({"params": no_decay_list, "weight_decay": 0.0})
            if decay_list:
                out.append({"params": decay_list, "weight_decay": group_wd})
            return out

        optimizer_groups = []

        # 隐式分组 (lora / mof / other)
        raw_groups = self.setup_parameter_groups()
        
        # 获取组特定配置，使用lr_initial而不是lr
        lora_cfg = optim_cfg.get("lora", {})
        lora_lr = lora_cfg.get("lr_initial", lora_cfg.get("lr", default_lr))
        lora_wd = lora_cfg.get("optimizer_params", {}).get("weight_decay", default_wd)
        lora_betas = lora_cfg.get("optimizer_params", {}).get("betas", optimizer_params.get("betas", (0.9, 0.999)))
        
        mof_cfg = optim_cfg.get("mof_encoder", {})
        mof_lr = mof_cfg.get("lr_initial", mof_cfg.get("lr", default_lr))
        mof_wd = mof_cfg.get("optimizer_params", {}).get("weight_decay", default_wd)
        mof_betas = mof_cfg.get("optimizer_params", {}).get("betas", optimizer_params.get("betas", (0.9, 0.999)))

        def add_group(pairs, lr, wd, betas, tag):
            if not pairs:
                return
            split_g = split_decay(pairs, wd)
            for sg in split_g:
                sg["lr"] = lr
                sg["betas"] = betas
                sg["group_name"] = tag if sg["weight_decay"] != 0 else f"{tag}_no_wd"
                optimizer_groups.append(sg)

        add_group(raw_groups["lora_params"], lora_lr, lora_wd, lora_betas, "lora")
        add_group(raw_groups["mof_encoder_params"], mof_lr, mof_wd, mof_betas, "mof_encoder")
        add_group(
            raw_groups["other_params"],
            default_lr, default_wd, optimizer_params.get("betas", (0.9, 0.999)),
            "other",
        )

        torch_param_groups = []
        for g in optimizer_groups:
            name_tag = g.pop("group_name", "group")
            params_list = g.pop("params")
            lr_val = g.pop("lr")
            betas_val = g.pop("betas")
            weight_decay_val = g.pop("weight_decay")
            torch_param_groups.append(
                {
                    "params": params_list,
                    "lr": lr_val,
                    "betas": betas_val,
                    "weight_decay": weight_decay_val,
                    "name": name_tag,
                }
            )

        if not torch_param_groups:
            raise RuntimeError("Multi-group optimizer produced 0 groups.")

        opt_params_filtered = {
            k: v for k, v in optimizer_params.items() if k not in ["weight_decay", "betas"]
        }
        self.optimizer = OptimClass(torch_param_groups, **opt_params_filtered)

        if distutils.is_master():
            self.print_parameter_groups_info()

        self.setup_multi_group_scheduler()


    # ==================================================================================
    # === Multi-Group LR MOD: scheduler builder
    # ==================================================================================
    def setup_multi_group_scheduler(self):
        optim_cfg = self.config["optim"]
        
        # 检查是否有组特定的调度器配置
        lora_cfg = optim_cfg.get("lora", {})
        mof_cfg = optim_cfg.get("mof_encoder", {})
        
        # 优先使用组特定的调度器，否则使用全局调度器
        lora_sched = lora_cfg.get("scheduler", "LambdaLR")
        mof_sched = mof_cfg.get("scheduler", "LambdaLR")
        
        # 由于torch的调度器是全局的，我们需要创建一个统一的调度器
        # 这里我们使用LambdaLR来为不同的参数组设置不同的lambda函数
        
        if lora_sched == "LambdaLR" or mof_sched == "LambdaLR":
            self.setup_lambda_lr_scheduler()
        else:
            # 如果不是LambdaLR，回退到原来的逻辑
            sched_raw = optim_cfg.get("scheduler", None)
            if isinstance(sched_raw, dict):
                sched_type = sched_raw.get("type", "ReduceLROnPlateau")
                sched_conf = sched_raw
            else:
                sched_type = sched_raw
                sched_conf = optim_cfg

            if sched_type is None or sched_type in ("Null", "None"):
                self.scheduler = _SimpleSchedulerWrapper(
                    scheduler=_NullScheduler(self.optimizer), scheduler_type="Null"
                )
                return

            # 其他调度器的处理逻辑保持不变...
            self.setup_other_schedulers(sched_type, sched_conf, optim_cfg)
    
    def setup_lambda_lr_scheduler(self):
        """设置LambdaLR调度器，支持不同参数组的不同学习率策略"""
        optim_cfg = self.config["optim"]
        lora_cfg = optim_cfg.get("lora", {})
        mof_cfg = optim_cfg.get("mof_encoder", {})
        
        max_epochs = optim_cfg.get("max_epochs", 30)
        steps_per_epoch = len(self.train_loader) if hasattr(self, 'train_loader') and self.train_loader else 1000
        total_steps = max_epochs * steps_per_epoch
        
        def create_lambda_fn(group_cfg, group_name):
            """为特定参数组创建lambda函数"""
            sched_params = group_cfg.get("scheduler_params", {})
            lambda_type = sched_params.get("lambda_type", "cosine")
            warmup_factor = sched_params.get("warmup_factor", 0.1)
            warmup_epochs = sched_params.get("warmup_epochs", 0.01)
            lr_min_factor = sched_params.get("lr_min_factor", 0.1)
            
            warmup_steps = max(1, int(warmup_epochs * total_steps))
            if distutils.is_master():
                logging.info(f"Group {group_name}: warmup_steps={warmup_steps}, total_steps={total_steps}")
            def lambda_fn(step):
                if step < warmup_steps:
                    # Warmup阶段：从warmup_factor线性增长到1.0
                    return warmup_factor + (1.0 - warmup_factor) * (step / warmup_steps)
                else:
                    # 主训练阶段
                    if lambda_type == "cosine":
                        # 余弦退火
                        progress = (step - warmup_steps) / (total_steps - warmup_steps)
                        progress = min(progress, 1.0)  # 确保不超过1.0
                        cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
                        return lr_min_factor + (1.0 - lr_min_factor) * cosine_factor
                    else:
                        # 默认常数
                        return 1.0
            
            return lambda_fn
        
        # 为每个参数组创建对应的lambda函数
        lambda_fns = []
        for i, group in enumerate(self.optimizer.param_groups):
            group_name = group.get("name", f"group_{i}")
            
            if "lora" in group_name:
                lambda_fn = create_lambda_fn(lora_cfg, group_name)
            elif "mof_encoder" in group_name:
                lambda_fn = create_lambda_fn(mof_cfg, group_name)
            else:
                # 其他参数组使用默认配置
                default_cfg = {
                    "scheduler_params": {
                        "lambda_type": "cosine",
                        "warmup_factor": 0.1,
                        "warmup_epochs": 0.01,
                        "lr_min_factor": 0.1
                    }
                }
                lambda_fn = create_lambda_fn(default_cfg, group_name)
            
            lambda_fns.append(lambda_fn)
        
        # 创建LambdaLR调度器
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, 
            lr_lambda=lambda_fns,
            last_epoch=-1
        )
        
        self.scheduler = _SimpleSchedulerWrapper(scheduler, "LambdaLR")
        self.scheduler_step_based = True
        self.current_step = 0
        if distutils.is_master():
            logging.info("=" * 60)
            logging.info("LambdaLR Scheduler Configuration:")
            logging.info(f"Max epochs: {max_epochs}")
            logging.info(f"Steps per epoch: {steps_per_epoch}")
            logging.info(f"Total steps: {total_steps}")
            
            for i, group in enumerate(self.optimizer.param_groups):
                group_name = group.get("name", f"group_{i}")
                if "lora" in group_name:
                    cfg = lora_cfg
                elif "mof_encoder" in group_name:
                    cfg = mof_cfg
                else:
                    cfg = {"scheduler_params": {"warmup_epochs": 0.01}}
                
                warmup_epochs = cfg.get("scheduler_params", {}).get("warmup_epochs", 0.01)
                warmup_steps = max(1, int(warmup_epochs * total_steps))
                
                logging.info(f"Group {group_name}:")
                logging.info(f"  Warmup epochs: {warmup_epochs}")
                logging.info(f"  Warmup steps: {warmup_steps}")
                logging.info(f"  Initial LR: {group['lr']}")
            logging.info("=" * 60)


    def setup_other_schedulers(self, sched_type, sched_conf, optim_cfg):
        """处理其他类型的调度器"""
        stype = sched_type
        sch = None

        if stype == "ReduceLROnPlateau":
            factor = sched_conf.get("factor", optim_cfg.get("lr_gamma", 0.8))
            patience = sched_conf.get("patience", optim_cfg.get("lr_patience", 3))
            mode = sched_conf.get("mode", "min")
            min_lr = sched_conf.get("min_lr", 0.0)
            sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=mode,
                factor=factor,
                patience=patience,
                min_lr=min_lr,
                verbose=True,
            )

        elif stype == "StepLR":
            step_size = sched_conf.get("step_size", optim_cfg.get("lr_step_size", 50))
            gamma = sched_conf.get("gamma", optim_cfg.get("lr_gamma", 0.8))
            sch = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma
            )

        elif stype == "MultiStepLR":
            milestones = sched_conf.get(
                "milestones", optim_cfg.get("lr_milestones", [100, 200])
            )
            gamma = sched_conf.get("gamma", optim_cfg.get("lr_gamma", 0.8))
            sch = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=milestones, gamma=gamma
            )

        elif stype == "CosineAnnealingLR":
            T_max = sched_conf.get(
                "T_max",
                optim_cfg.get("T_max", optim_cfg.get("max_epochs", None))
            )
            if T_max is None:
                raise ValueError(
                    "[CosineAnnealingLR] 需要提供 T_max"
                )
            eta_min = sched_conf.get("eta_min", 0.0)
            last_epoch = sched_conf.get("last_epoch", -1)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=T_max,
                eta_min=eta_min,
                last_epoch=last_epoch,
            )

        else:
            if distutils.is_master():
                logging.warning(
                    f"[MultiGroup] Unknown scheduler type '{stype}', using Null."
                )
            sch = _NullScheduler(self.optimizer)
            stype = "Null"

        self.scheduler = _SimpleSchedulerWrapper(sch, stype)

    # ==================================================================================
    # === Multi-Group LR MOD: printing
    # ==================================================================================
    def print_parameter_groups_info(self):
        if not hasattr(self, "optimizer"):
            return
        print("=" * 68)
        print("Multi-Group Optimizer Parameter Groups:")
        total_params = 0
        for i, group in enumerate(self.optimizer.param_groups):
            pcount = sum(p.numel() for p in group["params"])
            total_params += pcount
            gname = group.get("name", f"group_{i}")
            print(f"Group {i} ({gname}):")
            print(f"  #Params      : {pcount:,}")
            print(f"  LR           : {group['lr']}")
            print(f"  Weight Decay : {group.get('weight_decay', 'NA')}")
        print(f"Total trainable parameters: {total_params:,}")
        print("=" * 68)

    # ==================================================================================
    # === Multi-Group LR MOD: scheduler step
    # ==================================================================================
    def step_scheduler(self, val_metrics=None):
        if not hasattr(self, "scheduler") or self.scheduler is None:
            return
            
        stype = getattr(self.scheduler, "scheduler_type", "Null")
        
        if stype == "ReduceLROnPlateau":
            if val_metrics is not None:
                if self.primary_metric is None:
                    self.primary_metric = self.evaluation_metrics.get(
                        "primary_metric", self.evaluator.task_primary_metric[self.name]
                    )
                metric_val = val_metrics[self.primary_metric]["metric"]
                self.scheduler.step(metric_val)
        elif stype == "LambdaLR":
            # LambdaLR可以按step或epoch步进
            if hasattr(self, 'scheduler_step_based') and self.scheduler_step_based:
                # 基于step的调度（每次训练步骤后调用）
                self.scheduler.step()
                self.current_step += 1
            else:
                # 基于epoch的调度（每个epoch结束时调用）
                self.scheduler.step()
        elif stype != "Null":
            self.scheduler.step()



    # ==================================================================================
    # Original optimizer (fallback) + extras
    # ==================================================================================
    def load_optimizer(self) -> None:
        if self._multi_group_mode_enabled():
            if distutils.is_master():
                logging.info("[MultiGroup] Activating multi-parameter-group optimizer.")
            self.setup_optimizer_and_scheduler()
            return

        optimizer_cls = getattr(
            torch.optim, self.config["optim"].get("optimizer", "AdamW")
        )
        optimizer_params = self.config["optim"].get("optimizer_params", {})
        weight_decay = optimizer_params.get("weight_decay", 0)
        if "weight_decay" in self.config["optim"]:
            weight_decay = self.config["optim"]["weight_decay"]
            logging.warning(
                "Using `optim.weight_decay` (deprecated soon). "
                "Move to `optim.optimizer_params.weight_decay`."
            )
        lr_initial = self.config["optim"]["lr_initial"]

        if weight_decay > 0:
            self.model_params_no_wd = {}
            if hasattr(self._unwrapped_model, "no_weight_decay"):
                self.model_params_no_wd = self._unwrapped_model.no_weight_decay()

            params_decay, params_no_decay, name_no_decay = [], [], []
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if any(name.endswith(skip_name) for skip_name in self.model_params_no_wd):
                    params_no_decay.append(param)
                    name_no_decay.append(name)
                else:
                    params_decay.append(param)

            if distutils.is_master():
                logging.info("Parameters without weight decay:")
                logging.info(name_no_decay)

            self.optimizer = optimizer_cls(
                params=[
                    {"params": params_no_decay, "weight_decay": 0},
                    {"params": params_decay, "weight_decay": weight_decay},
                ],
                lr=lr_initial,
                **optimizer_params,
            )
        else:
            self.optimizer = optimizer_cls(
                params=self.model.parameters(),
                lr=lr_initial,
                **optimizer_params,
            )

    def load_extras(self) -> None:
        if self._multi_group_mode_enabled():
            self.clip_grad_norm = aii(
                self.config["optim"].get("clip_grad_norm", None), (int, float)
            )
            self.ema_decay = aii(self.config["optim"].get("ema_decay"), float)
            if self.ema_decay:
                ema_params = [p for p in self.model.parameters() if p.requires_grad]
                self.ema = ExponentialMovingAverage(
                    ema_params,
                    self.ema_decay,
                )
            else:
                self.ema = None
            return

        self.scheduler = LRScheduler(self.optimizer, self.config["optim"])
        self.clip_grad_norm = aii(
            self.config["optim"].get("clip_grad_norm", None), (int, float)
        )
        self.ema_decay = aii(self.config["optim"].get("ema_decay"), float)
        if self.ema_decay:
            ema_params = [p for p in self.model.parameters() if p.requires_grad]
            self.ema = ExponentialMovingAverage(
                ema_params,
                self.ema_decay,
            )
        else:
            self.ema = None

    # ==================================================================================
    # Save / Update best
    # ==================================================================================
    def save(
        self,
        metrics=None,
        checkpoint_file: str = "checkpoint.pt",
        training_state: bool = True,
    ) -> str | None:
        if not self.is_debug and distutils.is_master():
            state = {
                "state_dict": self.model.state_dict(),
                "normalizers": {
                    key: value.state_dict() for key, value in self.normalizers.items()
                },
                "elementrefs": {
                    key: value.state_dict() for key, value in self.elementrefs.items()
                },
                "config": self.config,
                "val_metrics": metrics,
                "amp": self.scaler.state_dict() if self.scaler else None,
            }
            if training_state:
                state.update(
                    {
                        "epoch": self.epoch,
                        "step": self.step,
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": (
                            self.scheduler.scheduler.state_dict()
                            if getattr(self.scheduler, "scheduler_type", "Null") != "Null"
                            else None
                        ),
                        "config": self.config,
                        "ema": self.ema.state_dict() if self.ema else None,
                        "best_val_metric": self.best_val_metric,
                        "primary_metric": self.evaluation_metrics.get(
                            "primary_metric",
                            self.evaluator.task_primary_metric[self.name],
                        ),
                    },
                )
                ckpt_path = save_checkpoint(
                    state,
                    checkpoint_dir=self.config["cmd"]["checkpoint_dir"],
                    checkpoint_file=checkpoint_file,
                )
            else:
                if self.ema is not None:
                    self.ema.store()
                    self.ema.copy_to()
                ckpt_path = save_checkpoint(
                    state,
                    checkpoint_dir=self.config["cmd"]["checkpoint_dir"],
                    checkpoint_file=checkpoint_file,
                )
                if self.ema:
                    self.ema.restore()
            return ckpt_path
        return None

    def update_best(
        self,
        primary_metric,
        val_metrics,
        disable_eval_tqdm: bool = True,
    ) -> None:
        if (
            "mae" in primary_metric
            and val_metrics[primary_metric]["metric"] < self.best_val_metric
        ) or (
            "mae" not in primary_metric
            and val_metrics[primary_metric]["metric"] > self.best_val_metric
        ):
            self.best_val_metric = val_metrics[primary_metric]["metric"]
            self.save(
                metrics=val_metrics,
                checkpoint_file="best_checkpoint.pt",
                training_state=False,
            )
            if self.test_loader is not None:
                self.predict(
                    self.test_loader,
                    results_file="predictions",
                    disable_tqdm=disable_eval_tqdm,
                )

    # ==================================================================================
    # Metrics aggregation / Validation
    # ==================================================================================
    def _aggregate_metrics(self, metrics):
        aggregated_metrics = {}
        for k in metrics:
            aggregated_metrics[k] = {
                "total": distutils.all_reduce(
                    metrics[k]["total"], average=False, device=self.device
                ),
                "numel": distutils.all_reduce(
                    metrics[k]["numel"], average=False, device=self.device
                ),
            }
            aggregated_metrics[k]["metric"] = (
                aggregated_metrics[k]["total"] / aggregated_metrics[k]["numel"]
            )
        return aggregated_metrics

    @torch.no_grad()
    def validate(self, split: str = "val", disable_tqdm: bool = False):
        ensure_fitted(self._unwrapped_model, warn=True)
        if distutils.is_master():
            logging.info(f"Evaluating on {split}.")
        self.model.eval()
        if self.ema:
            self.ema.store()
            self.ema.copy_to()

        metrics = {}
        evaluator = Evaluator(
            task=self.name,
            eval_metrics=self.evaluation_metrics.get(
                "metrics", Evaluator.task_metrics.get(self.name, {})
            ),
        )
        rank = distutils.get_rank()
        loader = self.val_loader if split == "val" else self.test_loader

        for _i, batch in tqdm(
            enumerate(loader),
            total=len(loader),
            position=rank,
            desc=f"device {rank}",
            disable=disable_tqdm,
        ):
            with torch.autocast("cuda", enabled=self.scaler is not None):
                batch.to(self.device)
                out = self._forward(batch)
            loss = self._compute_loss(out, batch)
            metrics = self._compute_metrics(out, batch, evaluator, metrics)
            metrics = evaluator.update("loss", loss.item(), metrics)

        metrics = self._aggregate_metrics(metrics)
        log_dict = {k: metrics[k]["metric"] for k in metrics}
        log_dict.update({"epoch": self.epoch})
        if distutils.is_master():
            log_str = [f"{k}: {v:.4f}" for k, v in log_dict.items()]
            logging.info(", ".join(log_str))
        if self.logger is not None:
            self.logger.log(
                log_dict,
                step=self.step,
                split=split,
            )
        if self.ema:
            self.ema.restore()
        return metrics

    # ==================================================================================
    # Backward / Step
    # ==================================================================================
    def _backward(self, loss) -> None:
        self.optimizer.zero_grad()
        loss.backward()

        if hasattr(self.model, "shared_parameters"):
            for p, factor in self.model.shared_parameters:
                if hasattr(p, "grad") and p.grad is not None:
                    p.grad.detach().div_(factor)
                else:
                    if not hasattr(self, "warned_shared_param_no_grad"):
                        self.warned_shared_param_no_grad = True
                        logging.warning(
                            "Some shared parameters do not have gradients. "
                            "Please verify they are used and attached correctly."
                        )

        if self.scaler:
            self.scaler.unscale_(self.optimizer)
            if isinstance(self.config["logger"], dict):
                log_weight_frequency = self.config["logger"].get("log_weight_table", -1)
            else:
                log_weight_frequency = -1
            if (
                self.logger is not None
                and distutils.is_master()
                and log_weight_frequency > 0
                and self.step % log_weight_frequency == 1
            ):
                columns, data = get_weight_table(self.model)
                self.logger.log_table(name="weight_table", cols=columns, data=data)

        if self.clip_grad_norm:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.clip_grad_norm,
            )
            if self.logger is not None:
                self.logger.log({"grad_norm": grad_norm}, step=self.step, split="train")

        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        if self.ema:
            self.ema.update()
        if hasattr(self, 'scheduler') and hasattr(self, 'scheduler_step_based') and self.scheduler_step_based:
            self.step_scheduler()
        
        # 记录学习率（每隔一定步数）
        if self.step % 5000 == 0 and self.logger is not None:
            lr_dict = self.get_lr_dict()
            self.logger.log(lr_dict, step=self.step, split="train")

    # ==================================================================================
    # Results
    # ==================================================================================
    def save_results(
        self,
        predictions: dict[str, npt.NDArray],
        results_file: str | None,
        keys: Sequence[str] | None = None,
    ) -> None:
        if results_file is None:
            return
        if keys is None:
            keys = predictions.keys()
        results = distutils.gather_objects(predictions)
        distutils.synchronize()
        if distutils.is_master():
            gather_results = {
                key: list(chain(*(result[key] for result in results))) for key in keys
            }
            _, idx = np.unique(gather_results["ids"], return_index=True)
            for k in keys:
                if "chunk_idx" in k:
                    gather_results[k] = np.cumsum([gather_results[k][i] for i in idx])[
                        :-1
                    ]
                else:
                    if f"{k}_chunk_idx" in keys or k == "forces":
                        gather_results[k] = np.concatenate(
                            [gather_results[k][i] for i in idx]
                        )
                    else:
                        gather_results[k] = np.array(
                            [gather_results[k][i] for i in idx]
                        )

            full_path = os.path.join(
                self.config["cmd"]["results_dir"], f"{self.name}_{results_file}.npz"
            )
            logging.info(f"Writing results to {full_path}")
            np.savez_compressed(full_path, **gather_results)

    # ==================================================================================
    # === Multi-Group LR MOD: external helper
    # ==================================================================================
    def get_lr_dict(self) -> Dict[str, float]:
        lr_dict = {}
        if hasattr(self, "optimizer"):
            for i, group in enumerate(self.optimizer.param_groups):
                gname = group.get("name", f"group_{i}")
                lr_dict[f"lr_{gname}"] = group["lr"]
        return lr_dict
