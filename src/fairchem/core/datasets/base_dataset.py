"""
Copyright (c) Meta, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
from abc import ABCMeta
from functools import cached_property
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
)

import numpy as np
import torch
from torch import randperm
from torch.utils.data import Dataset
from torch.utils.data import Subset as Subset_

from fairchem.core.common.registry import registry

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike


T_co = TypeVar("T_co", covariant=True)


class UnsupportedDatasetError(ValueError):
    pass


class BaseDataset(Dataset[T_co], metaclass=ABCMeta):
    """Base Dataset class for all OCP datasets."""

    def __init__(self, config: dict):
        """Initialize

        Args:
            config (dict): dataset configuration
        """
        self.config = config
        self.paths = []

        if "src" in self.config:
            if isinstance(config["src"], str):
                self.paths = [Path(self.config["src"])]
            else:
                self.paths = tuple(Path(path) for path in sorted(config["src"]))

        self.lin_ref = None
        if self.config.get("lin_ref", False):
            lin_ref = torch.tensor(
                np.load(self.config["lin_ref"], allow_pickle=True)["coeff"]
            )
            self.lin_ref = torch.nn.Parameter(lin_ref, requires_grad=False)

    def __len__(self) -> int:
        return self.num_samples

    def metadata_hasattr(self, attr) -> bool:
        return attr in self._metadata

    @cached_property
    def indices(self):
        return np.arange(self.num_samples, dtype=int)

    @cached_property
    def _metadata(self) -> dict[str, ArrayLike]:
        # logic to read metadata file here
        metadata_npzs = []
        if self.config.get("metadata_path", None) is not None:
            metadata_npzs.append(
                np.load(self.config["metadata_path"], allow_pickle=True)
            )

        else:
            for path in self.paths:
                if path.is_file():
                    metadata_file = path.parent / "metadata.npz"
                else:
                    metadata_file = path / "metadata.npz"
                if metadata_file.is_file():
                    metadata_npzs.append(np.load(metadata_file, allow_pickle=True))

        if len(metadata_npzs) == 0:
            logging.warning(
                f"Could not find dataset metadata.npz files in '{self.paths}'"
            )
            return {}

        metadata = {
            field: np.concatenate([metadata[field] for metadata in metadata_npzs])
            for field in metadata_npzs[0]
        }

        assert np.issubdtype(
            metadata["natoms"].dtype, np.integer
        ), f"Metadata natoms must be an integer type! not {metadata['natoms'].dtype}"
        assert metadata["natoms"].shape[0] == len(
            self
        ), "Loaded metadata and dataset size mismatch."

        return metadata

    def get_metadata(self, attr, idx):
        if attr in self._metadata:
            metadata_attr = self._metadata[attr]
            if isinstance(idx, list):
                return [metadata_attr[_idx] for _idx in idx]
            return metadata_attr[idx]
        return None


class Subset(Subset_, BaseDataset):
    """A pytorch subset that also takes metadata if given."""

    def __init__(
        self,
        dataset: BaseDataset,
        indices: Sequence[int],
        metadata: dict[str, ArrayLike],
    ) -> None:
        super().__init__(dataset, indices)
        self.metadata = metadata
        self.indices = indices
        self.num_samples = len(indices)
        self.config = dataset.config

    @cached_property
    def _metadata(self) -> dict[str, ArrayLike]:
        return self.dataset._metadata

    def get_metadata(self, attr, idx):
        if isinstance(idx, list):
            return self.dataset.get_metadata(attr, [[self.indices[i] for i in idx]])
        return self.dataset.get_metadata(attr, self.indices[idx])


class ExplodeYRelaxedDataset(BaseDataset):
    """把每条样本中的 y_relaxed(dict) 展开成多条样本。

    底层 dataset 的每个 item：
      - MOF 图结构
      - y_relaxed: dict, key 为 nm=(n_h2o, n_co2)，value 为能量

    展开后 dataset 的每个 item：
      - 同一个 MOF 图结构
      - condition: Tensor[2] = [n_h2o, n_co2]
      - energy: Tensor[1] = y_relaxed[nm]
      - 可选删除 y_relaxed，避免 DataLoader collate dict 失败

    配置（写在 dataset.train / dataset.val 里）：
      explode_y_relaxed:
        enabled: true
        src_key: y_relaxed
        condition_key: condition
        target_key: energy
        drop_src_key: true
    """

    def __init__(self, dataset: Dataset, config: dict[str, Any]):
        # dataset 通常是 Subset(BaseDataset)，这里保留引用用于元数据代理
        self.dataset = dataset
        self.config = config or {}

        ex_cfg = (
            (self.config.get("explode_y_relaxed") or {})
            if isinstance(self.config, dict)
            else {}
        )
        self.enabled = bool(ex_cfg.get("enabled", True))
        self.src_key = ex_cfg.get("src_key", "y_relaxed")
        self.condition_key = ex_cfg.get("condition_key", "condition")
        self.target_key = ex_cfg.get("target_key", "energy")
        self.drop_src_key = bool(ex_cfg.get("drop_src_key", True))

        if not self.enabled:
            raise ValueError("ExplodeYRelaxedDataset instantiated but enabled=False")

        # 构建全局索引：global_idx -> (base_idx, nm_key)
        self._pairs: list[tuple[int, Any]] = []
        for base_i in range(len(self.dataset)):
            d = self.dataset[base_i]
            y = d[self.src_key] if self.src_key in d else getattr(d, self.src_key, None)
            if y is None:
                continue
            if not isinstance(y, dict):
                raise TypeError(
                    f"Expected {self.src_key} to be dict, got {type(y)} at base idx {base_i}"
                )
            for nm in y.keys():
                self._pairs.append((base_i, nm))

    def __len__(self):
        return len(self._pairs)

    # ---- 元数据代理：BalancedBatchSampler 需要 ----
    def metadata_hasattr(self, attr) -> bool:
        if hasattr(self.dataset, "metadata_hasattr"):
            return self.dataset.metadata_hasattr(attr)
        return False

    def get_metadata(self, attr, idx):
        # idx 是展开后的样本索引，需要映射回 base_i
        if isinstance(idx, list):
            base_indices = [self._pairs[i][0] for i in idx]
            if hasattr(self.dataset, "get_metadata"):
                return self.dataset.get_metadata(attr, base_indices)
            return None
        base_i, _ = self._pairs[idx]
        if hasattr(self.dataset, "get_metadata"):
            return self.dataset.get_metadata(attr, base_i)
        return None

    @staticmethod
    def _to_float_tensor(x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.detach().to(torch.float32)
        return torch.tensor(x, dtype=torch.float32)

    @staticmethod
    def _nm_to_condition(nm) -> torch.Tensor:
        if isinstance(nm, torch.Tensor):
            t = nm.detach().to(torch.float32).view(-1)
            if t.numel() != 2:
                raise TypeError(
                    f"nm tensor must have 2 elements, got shape {tuple(nm.shape)}"
                )
            return t
        if isinstance(nm, (tuple, list)):
            if len(nm) != 2:
                raise TypeError(f"nm tuple/list must have length 2, got {nm}")
            return torch.tensor([nm[0], nm[1]], dtype=torch.float32)
        if isinstance(nm, str):
            import re

            m1 = re.search(r"H2O\s*=\s*(\d+)", nm)
            m2 = re.search(r"CO2\s*=\s*(\d+)", nm)
            if m1 and m2:
                return torch.tensor(
                    [int(m1.group(1)), int(m2.group(1))], dtype=torch.float32
                )
            raise TypeError(f"Unsupported nm key format: {nm!r}")
        raise TypeError(f"Unsupported nm key type: {type(nm)}")

    def __getitem__(self, idx: int):
        base_i, nm = self._pairs[idx]
        data = self.dataset[base_i]

        y = data[self.src_key] if self.src_key in data else getattr(data, self.src_key)
        if not isinstance(y, dict):
            raise TypeError(
                f"Expected {self.src_key} to be dict, got {type(y)} at base idx {base_i}"
            )
        if nm not in y:
            raise KeyError(f"Key {nm} not found in {self.src_key} at base idx {base_i}")

        cond = self._nm_to_condition(nm)
        energy = self._to_float_tensor(y[nm]).view(1)

        data[self.condition_key] = cond
        data[self.target_key] = energy

        if self.drop_src_key and (self.src_key in data):
            try:
                del data[self.src_key]
            except Exception:
                data[self.src_key] = None

        return data


def create_dataset(config: dict[str, Any], split: str) -> Subset:
    """Create a dataset from a config dictionary

    Args:
        config (dict): dataset config dictionary
        split (str): name of split

    Returns:
        Subset: dataset subset class
    """
    # Initialize the dataset
    dataset_cls = registry.get_dataset_class(config.get("format", "lmdb"))
    assert issubclass(dataset_cls, Dataset), f"{dataset_cls} is not a Dataset"

    # remove information about other splits, only keep specified split
    # this may only work with the mt config not main config
    current_split_config = config.copy()
    if "splits" in current_split_config:
        current_split_config.pop("splits")
        current_split_config.update(config["splits"][split])

    seed = current_split_config.get("seed", 0)
    if split != "train":
        seed += (
            1  # if we use same dataset for train / val , make sure its diff sampling
        )

    g = torch.Generator()
    g.manual_seed(seed)

    dataset = dataset_cls(current_split_config)
    logging.info(f"DEBUG create_dataset: Created {dataset_cls.__name__}, len={len(dataset)}")

    # Get indices of the dataset
    indices = dataset.indices
    logging.info(f"DEBUG create_dataset: Initial indices shape={indices.shape}, len={len(indices)}")
    
    max_atoms = current_split_config.get("max_atoms", None)
    if max_atoms is not None:
        if not dataset.metadata_hasattr("natoms"):
            raise ValueError("Cannot use max_atoms without dataset metadata")
        indices = indices[dataset.get_metadata("natoms", indices) <= max_atoms]
        logging.info(f"DEBUG create_dataset: After max_atoms filter, indices len={len(indices)}")

    for subset_to in current_split_config.get("subset_to", []):
        if not dataset.metadata_hasattr(subset_to["metadata_key"]):
            raise ValueError(
                f"Cannot use {subset_to} without dataset metadata key {subset_to['metadata_key']}"
            )
        rhv = subset_to["rhv"]
        if isinstance(rhv, str):
            with open(rhv) as f:
                rhv = f.read().splitlines()
                rhv = [int(x) for x in rhv]
        if subset_to["op"] == "abs_le":
            indices = indices[
                np.abs(dataset.get_metadata(subset_to["metadata_key"], indices)) <= rhv
            ]
        elif subset_to["op"] == "in":
            indices = indices[
                np.isin(dataset.get_metadata(subset_to["metadata_key"], indices), rhv)
            ]
        logging.info(f"DEBUG create_dataset: After subset_to filter, indices len={len(indices)}")

    # Apply dataset level transforms
    # TODO is no_shuffle mutually exclusive though? or what is the purpose of no_shuffle?
    first_n = current_split_config.get("first_n")
    sample_n = current_split_config.get("sample_n")
    no_shuffle = current_split_config.get("no_shuffle")
    # this is true if at most one of the mutually exclusive arguments are set
    if sum(arg is not None for arg in (first_n, sample_n, no_shuffle)) > 1:
        raise ValueError(
            "sample_n, first_n, no_shuffle are mutually exclusive arguments. Only one can be provided."
        )
    if first_n is not None:
        max_index = first_n
    elif sample_n is not None:
        # shuffle by default, user can disable to optimize if they have confidence in dataset
        # shuffle all datasets by default to avoid biasing the sampling in concat dataset
        # TODO only shuffle if split is train
        max_index = sample_n
        indices = (
            indices
            if len(indices) == 1
            else indices[randperm(len(indices), generator=g)]
        )
    else:
        max_index = len(indices)
        indices = (
            indices
            if (no_shuffle or len(indices) == 1)
            else indices[randperm(len(indices), generator=g)]
        )

    if max_index > len(indices):
        msg = (
            f"Cannot take {max_index} data points from a dataset of only length {len(indices)}.\n"
            f"Make sure to set first_n or sample_n to a number =< the total samples in dataset."
        )
        if max_atoms is not None:
            msg = msg[:-1] + f"that are smaller than the given max_atoms {max_atoms}."
        raise ValueError(msg)

    indices = indices[:max_index]
    logging.info(f"DEBUG create_dataset: Final indices len={len(indices)}, max_index={max_index}")

    subset = Subset(dataset, indices, metadata=dataset._metadata)
    logging.info(f"DEBUG create_dataset: Created Subset, len={len(subset)}")

    # Optional: explode y_relaxed dict into multiple samples
    ex_cfg = current_split_config.get("explode_y_relaxed")
    if isinstance(ex_cfg, dict) and ex_cfg.get("enabled", False):
        logging.info(f"DEBUG create_dataset: Applying ExplodeYRelaxedDataset")
        subset = ExplodeYRelaxedDataset(subset, current_split_config)
        logging.info(f"DEBUG create_dataset: After ExplodeYRelaxedDataset, len={len(subset)}")

    logging.info(f"DEBUG create_dataset: Returning dataset with len={len(subset)}")
    return subset
