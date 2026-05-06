#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""inspect_aggregated_lmdb.py

用途：快速抽样检查你“聚合后”的 LMDB 是否符合预期。
- 输出若干条样本的 name
- 按 (n_H2O, m_CO2) 排序逐行打印 y_relaxed 字典：H2O=n CO2=m -> energy

说明：
- 路径写死（按你的要求）。
- 兼容 LmdbDataset 返回 torch_geometric.data.Data 或 dict。

运行：
PYTHONPATH=. python fairchem/lora_model/data_lora/inspect_aggregated_lmdb.py
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Tuple

import torch
from fairchem.core.common.utils import setup_logging
from fairchem.core.datasets import LmdbDataset


# ================== 配置区（写死） ==================
# SRC_LMDB = "/home/dell/autodl-tmp/lorafair/data/is2r/newODAC/mof_only_aggregate_yrelaxed.lmdb"
SRC_LMDB = "/home/dell/autodl-tmp/lorafair/data/is2r/newODAC/mof_val.lmdb"
# 打印多少条示例
NUM_SAMPLES_TO_PRINT = 10
# 从哪些索引打印（优先使用这些；不够则从头补齐）
PREFERRED_INDICES = [0, 1, 2, 3, 4]
# ====================================================


def _get(sample: Any, key: str, default=None):
    if isinstance(sample, dict):
        return sample.get(key, default)
    return getattr(sample, key, default)


def _format_energy(v: Any) -> str:
    try:
        if torch.is_tensor(v):
            v = v.detach().cpu().view(-1)[0].item()
        v = float(v)
        if math.isnan(v) or math.isinf(v):
            return str(v)
        return f"{v:.6f}"
    except Exception:
        return str(v)


def _iter_sorted_nm_items(y_map: Dict) -> List[Tuple[Tuple[int, int], Any]]:
    """把 y_relaxed 字典转成按 (n,m) 排序的列表。

    兼容：key 可能是 tuple，也可能是字符串（如果你后续改过）。
    """
    items = list(y_map.items())

    def _key(kv):
        k, _ = kv
        # tuple 形式：(n,m)
        if isinstance(k, tuple) and len(k) == 2:
            try:
                return int(k[0]), int(k[1])
            except Exception:
                return 10**9, 10**9
        # 字符串形式：尽量从里面抠出两个整数
        if isinstance(k, str):
            import re

            nums = re.findall(r"-?\d+", k)
            if len(nums) >= 2:
                return int(nums[0]), int(nums[1])
        return 10**9, 10**9

    items.sort(key=_key)
    return items


def main():
    setup_logging()

    dataset = LmdbDataset({"src": SRC_LMDB, "normalize_labels": False})
    n = len(dataset)
    print(f"✅ 加载成功: {SRC_LMDB}")
    print(f"数据集大小: {n}\n")

    # 组织要打印的索引列表
    idxs: List[int] = []
    for i in PREFERRED_INDICES:
        if 0 <= i < n and i not in idxs:
            idxs.append(i)
    i = 0
    while len(idxs) < min(NUM_SAMPLES_TO_PRINT, n):
        if i not in idxs:
            idxs.append(i)
        i += 1

    print(f"将打印 {len(idxs)} 条示例：{idxs}\n")

    # 统计全数据集中出现过的 (n_H2O, m_CO2) 组合
    nm_counter = {}
    for ii in range(n):
        s2 = dataset[ii]
        y2 = _get(s2, "y_relaxed", None)
        if not isinstance(y2, dict):
            continue
        for k in y2.keys():
            if isinstance(k, tuple) and len(k) == 2:
                try:
                    nm = (int(k[0]), int(k[1]))
                except Exception:
                    continue
            elif isinstance(k, str):
                import re

                nums = re.findall(r"-?\d+", k)
                if len(nums) >= 2:
                    nm = (int(nums[0]), int(nums[1]))
                else:
                    continue
            else:
                continue
            nm_counter[nm] = nm_counter.get(nm, 0) + 1

    nm_list = sorted(nm_counter.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    print("全数据集 nm=(n_H2O,m_CO2) 组合统计（出现于多少个 MOF 的 y_relaxed 中）：")
    if not nm_list:
        print("  （未发现任何 nm 组合，可能 y_relaxed 不是 dict）\n")
    else:
        print("  " + ", ".join([f"{nm}:{cnt}" for nm, cnt in nm_list]))
        print("")

    for rank, idx in enumerate(idxs):
        s = dataset[idx]
        name = _get(s, "name", None)
        y_relaxed = _get(s, "y_relaxed", None)
        tags = _get(s, "tags", None)

        print("=" * 80)
        print(f"[{rank:02d}] idx={idx}  name={name}")

        # 额外：确认是否还有吸附物（tags==2）
        if tags is not None and torch.is_tensor(tags):
            ads_n = int((tags == 2).sum().item())
            print(f"     tags==2(吸附物) 原子数: {ads_n}")

        if isinstance(y_relaxed, dict):
            items = _iter_sorted_nm_items(y_relaxed)
            print(f"     y_relaxed(dict) keys={len(items)}")
            for (nm, e) in items:
                if isinstance(nm, tuple) and len(nm) == 2:
                    n_h2o, m_co2 = nm
                else:
                    # 回退
                    n_h2o, m_co2 = nm, "?"
                print(f"       H2O={n_h2o}  CO2={m_co2}  ->  { _format_energy(e) }")
        else:
            print(f"     y_relaxed: {y_relaxed} (type={type(y_relaxed)})")

    print("\n✅ 检查结束")


if __name__ == "__main__":
    main()

