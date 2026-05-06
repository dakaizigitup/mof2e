#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""filter_lowest_adsorption_energy_to_lmdb.py

根据预设配置筛选吸附能最低的样本并保存为 lmdb。
无需命令行参数，直接运行即可。
"""

import os
import math
import pickle
from typing import List, Optional, Tuple

import lmdb
from tqdm import tqdm

from fairchem.core.datasets import LmdbDataset

# ================== 配置区（按需修改） ==================
# 输入 LMDB（可为目录或 .lmdb 文件）
SRC_LMDB = "/home/dell/autodl-tmp/lorafair/data/is2r/filtered_train2"
# 输出 LMDB 路径（会自动创建父目录）
DST_LMDB = "/home/dell/autodl-tmp/lorafair/data/is2r/bestTrain2/bestTrain.lmdb"
# 选取最低比例（0~1]，例如 0.05 = 最低 5%
RATIO = 0.05
# 吸附能字段名；若为 None 则自动尝试常见候选
ENERGY_KEY = "y_relaxed"  # 或 None
# 是否在构建数据集时开启 normalize_labels
NORMALIZE_LABELS = False
# =======================================================

"""筛选吸附能最低的 N% 样本并另存为 LMDB。

参考：save_se_data.py

功能：
- 从输入 LMDB（fairchem 的 LmdbDataset 可读取的 src）遍历样本
- 从样本中提取“吸附能”标量（可配置字段名）
- 选出吸附能最低的 ratio（默认 0.05，即 5%）样本
- 写入到输出 LMDB（单个 .lmdb 文件，subdir=False）

用法示例：
python filter_lowest_adsorption_energy_to_lmdb.py \
  --src /path/to/input_lmdb_or_dir \
  --dst /path/to/output_lowest_5pct.lmdb \
  --ratio 0.05 \
  --energy-key ads_energy

注意：
- ratio 是“最低 N%”，例如 0.05 表示取能量最小的 5%（数值更负也算更小）。
- 会跳过无法解析吸附能的样本（并计数）。
"""


def build_dataset(src_path: str, normalize_labels: bool = False) -> LmdbDataset:
    config = {
        "src": src_path,
        "normalize_labels": normalize_labels,
    }
    return LmdbDataset(config)


def _to_scalar(x) -> Optional[float]:
    """尽可能从任意常见类型中提取标量 float。失败返回 None。"""
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        pass

    if hasattr(x, "item"):
        try:
            return float(x.item())
        except Exception:
            pass

    if isinstance(x, (list, tuple)) and len(x) > 0:
        return _to_scalar(x[0])

    try:
        import numpy as np  # noqa: F401

        if "numpy" in str(type(x)):
            try:
                return float(x.reshape(-1)[0])
            except Exception:
                pass
    except Exception:
        pass

    return None


def extract_ads_energy(sample: dict, energy_key: Optional[str]) -> Optional[float]:
    """从样本中提取吸附能标量。

    - 若指定 energy_key：优先从该 key 读取
    - 否则：按常见候选 key 依次尝试
    """
    if energy_key:
        if energy_key in sample:
            return _to_scalar(sample.get(energy_key))
        return None

    for k in [
        "adsorption_energy",
        "ads_energy",
        "e_ads",
        "E_ads",
        "energy",
        "y_relaxed",
        "y",
        "target",
        "targets",
    ]:
        if k in sample:
            v = _to_scalar(sample.get(k))
            if v is not None:
                return v
    return None


def _mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    n = len(values)
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / n
    return mean, math.sqrt(var)


def select_lowest_percent(
    dataset: LmdbDataset,
    ratio: float,
    energy_key: Optional[str],
) -> Tuple[List[Tuple[int, bytes]], List[float], int]:
    """返回 (selected_items, selected_energies, skipped_count)。

    selected_items: [(orig_index, pickled_sample_bytes), ...]
    """
    if ratio <= 0 or ratio > 1:
        raise ValueError(f"ratio 必须在 (0, 1] 内，当前 ratio={ratio}")

    energies: List[float] = []
    packed: List[bytes] = []
    skipped = 0

    total = len(dataset)
    for i in tqdm(range(total), desc="遍历样本并提取吸附能", ncols=100):
        sample = dataset[i]
        e = extract_ads_energy(sample, energy_key)
        if e is None or (isinstance(e, float) and (math.isnan(e) or math.isinf(e))):
            skipped += 1
            continue
        energies.append(float(e))
        packed.append(pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL))

    if not energies:
        return [], [], skipped

    k = max(1, int(math.ceil(len(energies) * ratio)))

    # 取能量最小的 k 个索引
    sorted_idx = sorted(range(len(energies)), key=lambda j: energies[j])
    keep_idx = sorted_idx[:k]

    selected_items: List[Tuple[int, bytes]] = []
    selected_energies: List[float] = []
    for out_i, j in enumerate(keep_idx):
        selected_items.append((j, packed[j]))
        selected_energies.append(energies[j])

    return selected_items, selected_energies, skipped


def write_lmdb(items: List[Tuple[int, bytes]], dst_lmdb: str) -> None:
    """写入单个 LMDB 文件。

    key 使用连续的 0..n-1，保持与 save_se_data.py 一致。
    """
    if not items:
        raise ValueError("没有可写入的数据（items 为空）")

    os.makedirs(os.path.dirname(os.path.abspath(dst_lmdb)), exist_ok=True)

    total_bytes = sum(len(v) for _, v in items)
    map_size = int(total_bytes * 1.3) + 32 * 1024 * 1024

    env = lmdb.open(dst_lmdb, map_size=map_size, subdir=False)
    try:
        with env.begin(write=True) as txn:
            for idx, (_, val) in enumerate(tqdm(items, desc="写入输出 LMDB", ncols=100)):
                txn.put(str(idx).encode(), val)
        env.sync()
    finally:
        env.close()


def main():
    print("配置：")
    print(f"SRC_LMDB        = {SRC_LMDB}")
    print(f"DST_LMDB        = {DST_LMDB}")
    print(f"RATIO           = {RATIO}")
    print(f"ENERGY_KEY      = {ENERGY_KEY}")
    print(f"NORMALIZE_LABELS= {NORMALIZE_LABELS}\n")

    dataset = build_dataset(SRC_LMDB, normalize_labels=NORMALIZE_LABELS)
    print(f"输入数据集大小：{len(dataset)}")

    selected_items, selected_energies, skipped = select_lowest_percent(
        dataset=dataset,
        ratio=RATIO,
        energy_key=ENERGY_KEY,
    )

    print(f"成功解析吸附能的样本数：{len(selected_energies)}")
    print(f"跳过（无法解析/NaN/Inf）的样本数：{skipped}")
    if not selected_items:
        raise RuntimeError("筛选结果为空：请检查 --energy-key 是否正确，或数据里是否存在吸附能字段")

    mean, std = _mean_std(selected_energies)
    print(f"筛选出的最低 {RATIO*100:.2f}%（约 {len(selected_items)} 条）吸附能统计：")
    print(f"mean: {mean}")
    print(f"stdev: {std}")
    print(f"min: {min(selected_energies)}")
    print(f"max: {max(selected_energies)}")

    print(f"写入输出 LMDB：{DST_LMDB}")
    write_lmdb(selected_items, DST_LMDB)
    print("✅ 完成")


if __name__ == "__main__":
    main()

