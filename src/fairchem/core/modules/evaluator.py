"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Callable, ClassVar

import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Hashable

"""
An evaluation module for use with the OCP dataset and suite of tasks. It should
be possible to import this independently of the rest of the codebase, e.g:

```
from fairchem.core.modules import Evaluator

evaluator = Evaluator(task="is2re")
perf = evaluator.eval(prediction, target)
```

task: "s2ef", "is2rs", "is2re".

We specify a default set of metrics for each task, but should be easy to extend
to add more metrics. `evaluator.eval` takes as input two dictionaries, one for
predictions and another for targets to check against. It returns a dictionary
with the relevant metrics computed.
"""

NONE_SLICE = slice(None)


class Evaluator:
    task_metrics: ClassVar[dict[str, str]] = {
        "s2ef": {
            "energy": ["mae"],
            "forces": [
                "forcesx_mae",
                "forcesy_mae",
                "forcesz_mae",
                "mae",
                "cosine_similarity",
                "magnitude_error",
                "energy_forces_within_threshold",
            ],
        },
        "is2rs": {
            "positions": [
                "average_distance_within_threshold",
                "mae",
                "mse",
            ]
        },
        "is2re": {
            "energy": [
                "mae",
                "mse",
                "energy_within_threshold",
                "r2",
                "pearsonr",
                "spearmanr",
            ]
        },
    }

    task_primary_metric: ClassVar[dict[str, str | None]] = {
        "s2ef": "energy_forces_within_threshold",
        "is2rs": "average_distance_within_threshold",
        "is2re": "energy_mae",
        "ocp": None,
    }

    def __init__(
        self, task: str | None = None, eval_metrics: dict | None = None
    ) -> None:
        if eval_metrics is None:
            eval_metrics = {}
        self.task = task
        self.target_metrics = (
            eval_metrics if eval_metrics else self.task_metrics.get(task, {})
        )

    def eval(
        self,
        prediction: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        prev_metrics: dict | None = None,
    ):
        prev_metrics = prev_metrics or {}
        metrics = prev_metrics

        for target_property in self.target_metrics:
            for fn in self.target_metrics[target_property]:
                metric_name = (
                    f"{target_property}_{fn}"
                    if target_property not in fn and target_property != "misc"
                    else fn
                )
                res = eval(fn)(prediction, target, target_property)
                metrics = self.update(metric_name, res, metrics)

        return metrics

    def update(self, key, stat, metrics):
        if key not in metrics:
            metrics[key] = {
                "metric": None,
                "total": 0,
                "numel": 0,
            }

        if isinstance(stat, dict):
            # If dictionary, we expect it to have `metric`, `total`, `numel`.
            metrics[key]["total"] += stat["total"]
            metrics[key]["numel"] += stat["numel"]
            # final metric = total / numel (see custom metrics like r2/pearsonr/spearmanr)
            metrics[key]["metric"] = (
                metrics[key]["total"] / metrics[key]["numel"]
                if metrics[key]["numel"] != 0
                else float("nan")
            )
        elif isinstance(stat, (float, int)):
            # If float or int, just add to the total and increment numel by 1.
            metrics[key]["total"] += stat
            metrics[key]["numel"] += 1
            metrics[key]["metric"] = metrics[key]["total"] / metrics[key]["numel"]
        elif torch.is_tensor(stat):
            raise NotImplementedError

        return metrics


def metrics_dict(metric_fun: Callable) -> Callable:
    """Wrap up the return of a metrics function"""

    @wraps(metric_fun)
    def wrapped_metrics(
        prediction: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        key: Hashable = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        error = metric_fun(prediction, target, key, **kwargs)
        return {
            "metric": torch.mean(error).item(),
            "total": torch.sum(error).item(),
            "numel": error.numel(),
        }

    return wrapped_metrics


@metrics_dict
def cosine_similarity(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = NONE_SLICE,
):
    # cast to float 32 to avoid 0/nan issues in fp16
    # https://github.com/pytorch/pytorch/issues/69512
    return torch.cosine_similarity(prediction[key].float(), target[key].float())


@metrics_dict
def mae(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = NONE_SLICE,
) -> torch.Tensor:
    return torch.abs(target[key] - prediction[key])


@metrics_dict
def mse(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = NONE_SLICE,
) -> torch.Tensor:
    return (target[key] - prediction[key]) ** 2


@metrics_dict
def per_atom_mae(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = NONE_SLICE,
) -> torch.Tensor:
    return torch.abs(target[key] - prediction[key]) / target["natoms"].unsqueeze(1)


@metrics_dict
def per_atom_mse(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = NONE_SLICE,
) -> torch.Tensor:
    return ((target[key] - prediction[key]) / target["natoms"].unsqueeze(1)) ** 2


@metrics_dict
def magnitude_error(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = NONE_SLICE,
    p: int = 2,
) -> torch.Tensor:
    assert prediction[key].shape[1] > 1
    return torch.abs(
        torch.norm(prediction[key], p=p, dim=-1) - torch.norm(target[key], p=p, dim=-1)
    )


def forcesx_mae(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = NONE_SLICE,
):
    return mae(prediction["forces"][:, 0], target["forces"][:, 0])


def forcesx_mse(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = NONE_SLICE,
):
    return mse(prediction["forces"][:, 0], target["forces"][:, 0])


def forcesy_mae(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = None,
):
    return mae(prediction["forces"][:, 1], target["forces"][:, 1])


def forcesy_mse(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = None,
):
    return mse(prediction["forces"][:, 1], target["forces"][:, 1])


def forcesz_mae(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = None,
):
    return mae(prediction["forces"][:, 2], target["forces"][:, 2])


def forcesz_mse(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = None,
):
    return mse(prediction["forces"][:, 2], target["forces"][:, 2])


def energy_forces_within_threshold(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = None,
) -> dict[str, float | int]:
    # Note that this natoms should be the count of free atoms we evaluate over.
    assert target["natoms"].sum() == prediction["forces"].size(0)
    assert target["natoms"].size(0) == prediction["energy"].size(0)

    # compute absolute error on per-atom forces and energy per system.
    # then count the no. of systems where max force error is < 0.03 and max
    # energy error is < 0.02.
    f_thresh = 0.03
    e_thresh = 0.02

    success = 0
    total = int(target["natoms"].size(0))

    error_forces = torch.abs(target["forces"] - prediction["forces"])
    error_energy = torch.abs(target["energy"] - prediction["energy"])

    start_idx = 0
    for i, n in enumerate(target["natoms"]):
        if (
            error_energy[i] < e_thresh
            and error_forces[start_idx : start_idx + n].max() < f_thresh
        ):
            success += 1
        start_idx += n

    return {
        "metric": success / total,
        "total": success,
        "numel": total,
    }


def energy_within_threshold(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = None,
) -> dict[str, float | int]:
    # compute absolute error on energy per system.
    # then count the no. of systems where max energy error is < 0.02.
    e_thresh = 0.02
    error_energy = torch.abs(target["energy"] - prediction["energy"])

    success = (error_energy < e_thresh).sum().item()
    total = target["energy"].size(0)

    return {
        "metric": success / total,
        "total": success,
        "numel": total,
    }


def average_distance_within_threshold(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = None,
) -> dict[str, float | int]:
    pred_pos = torch.split(prediction["positions"], prediction["natoms"].tolist())
    target_pos = torch.split(target["positions"], target["natoms"].tolist())

    mean_distance = []
    for idx, ml_pos in enumerate(pred_pos):
        mean_distance.append(
            np.mean(
                np.linalg.norm(
                    min_diff(
                        ml_pos.detach().cpu().numpy(),
                        target_pos[idx].detach().cpu().numpy(),
                        target["cell"][idx].detach().cpu().numpy(),
                        target["pbc"].tolist(),
                    ),
                    axis=1,
                )
            )
        )

    success = 0
    intv = np.arange(0.01, 0.5, 0.001)
    for i in intv:
        success += sum(np.array(mean_distance) < i)

    total = len(mean_distance) * len(intv)

    return {"metric": success / total, "total": success, "numel": total}


def min_diff(
    pred_pos: torch.Tensor,
    dft_pos: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
):
    pos_diff = pred_pos - dft_pos
    fractional = np.linalg.solve(cell.T, pos_diff.T).T

    for i, periodic in enumerate(pbc):
        # Yes, we need to do it twice
        if periodic:
            fractional[:, i] %= 1.0
            fractional[:, i] %= 1.0

    fractional[fractional > 0.5] -= 1

    return np.matmul(fractional, cell)


def rmse(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = None,
) -> dict[str, float | int]:
    error = torch.sqrt(((target[key] - prediction[key]) ** 2).sum(dim=-1))
    return {
        "metric": torch.mean(error).item(),
        "total": torch.sum(error).item(),
        "numel": error.numel(),
    }


# --------------------
# 新增指标: R^2 / Pearson / Spearman（batch 可聚合）
# --------------------

def _flatten_1d(x: torch.Tensor) -> torch.Tensor:
    x = x.view(-1)
    return x.float()


def r2(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = NONE_SLICE,
) -> dict[str, float | int]:
    """批级别 R^2，可跨 batch 通过 total/numel 聚合。
    这里返回 total = (SS_tot - SS_res)，numel = SS_tot，使得最终 metric = total/numel = 1 - SS_res/SS_tot。
    """
    y_pred = _flatten_1d(prediction[key])
    y_true = _flatten_1d(target[key])
    if y_pred.numel() == 0:
        return {"metric": float("nan"), "total": 0.0, "numel": 0.0}
    # 兼容 [B,1] 情形
    if y_true.dim() == 0:
        y_true = y_true.view(1)
    if y_pred.dim() == 0:
        y_pred = y_pred.view(1)

    eps = 1e-12
    ss_res = torch.sum((y_true - y_pred) ** 2)
    y_mean = torch.mean(y_true)
    ss_tot = torch.sum((y_true - y_mean) ** 2)
    ss_tot = ss_tot + eps  # 避免零方差导致除零

    total = (ss_tot - ss_res).item()
    numel = ss_tot.item()
    metric = total / numel if numel != 0 else float("nan")
    return {"metric": metric, "total": total, "numel": numel}


def pearsonr(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = NONE_SLICE,
) -> dict[str, float | int]:
    """计算当前 batch 的 Pearson 相关系数 r。
    为了跨 batch 聚合，返回 total = r * n, numel = n（n 为当前 batch 样本数），
    聚合后得到加权平均 r（近似全量 r）。
    """
    y_pred = _flatten_1d(prediction[key])
    y_true = _flatten_1d(target[key])
    n = y_true.numel()
    if n <= 1:
        return {"metric": float("nan"), "total": 0.0, "numel": 0.0}

    x = y_pred - y_pred.mean()
    y = y_true - y_true.mean()
    eps = 1e-12
    r = (x * y).sum() / (torch.sqrt((x * x).sum()) * torch.sqrt((y * y).sum()) + eps)
    r_item = r.item()
    return {"metric": r_item, "total": r_item * n, "numel": n}


def spearmanr(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = NONE_SLICE,
) -> dict[str, float | int]:
    """计算当前 batch 的 Spearman 等级相关。
    做法：对 pred/true 各自转为秩（近似处理 ties），再做 Pearson。
    聚合策略与 Pearson 相同：total = rho * n, numel = n。
    """
    y_pred = _flatten_1d(prediction[key])
    y_true = _flatten_1d(target[key])
    n = y_true.numel()
    if n <= 1:
        return {"metric": float("nan"), "total": 0.0, "numel": 0.0}

    # 近似 rank：双 argsort（ties 简单近似处理）
    def rankify(t: torch.Tensor) -> torch.Tensor:
        # 返回 0..n-1 的秩
        order = torch.argsort(t)
        ranks = torch.empty_like(order, dtype=torch.float32)
        ranks[order] = torch.arange(t.numel(), dtype=torch.float32, device=t.device)
        return ranks

    rx = rankify(y_pred)
    ry = rankify(y_true)

    x = rx - rx.mean()
    y = ry - ry.mean()
    eps = 1e-12
    rho = (x * y).sum() / (torch.sqrt((x * x).sum()) * torch.sqrt((y * y).sum()) + eps)
    rho_item = rho.item()
    return {"metric": rho_item, "total": rho_item * n, "numel": n}
