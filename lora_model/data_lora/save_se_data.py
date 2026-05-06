import os
import lmdb
import pickle
from typing import Set, Tuple, List, Optional

import pandas as pd
from tqdm import tqdm
from fairchem.core.datasets import LmdbDataset
import math

"""
基于 Excel 中给定的 MOF 名称列表，从原始 LMDB 数据集中筛选样本，并将结果拆分写入多个较小的 LMDB 文件。

使用说明：
- 请按需修改下方的文件路径：
  TRAIN_SRC_LMDB：训练集原始 LMDB 数据目录（或单一 lmdb 文件路径，取决于 LmdbDataset 的使用方式）
  VAL_SRC_LMDB：验证集原始 LMDB 数据目录
  TRAIN_XLSX：包含训练集 MOF 名称的 Excel 文件（列名为 "Name"）
  VAL_XLSX：包含验证集 MOF 名称的 Excel 文件（列名为 "Name"）
  TRAIN_DST_ROOT：训练集筛选后 LMDB 输出目录
  VAL_DST_ROOT：验证集筛选后 LMDB 输出目录
  NUM_SPLITS：将结果拆分成多少个 lmdb 分片

- Excel 要求：包含列名 "Name"，值如 "ABEFUL", "ABESUX" 等。
- 名称匹配逻辑：对样本的 sample["name"] 取前缀（用下划线 "_" 切分，取第 0 段），若前缀属于名称集合则保留。
- 额外功能：在验证集筛选完成后，自动计算 energy 的 mean 和 stdev 并打印，便于粘贴到配置。
"""

# ========= 配置区（按需修改） =========
TRAIN_SRC_LMDB = "/home/dell/autodl-tmp/lorafair/data/s2ef/train"
VAL_SRC_LMDB = "/home/dell/autodl-tmp/lorafair/data/s2ef/val"  # 若验证数据与训练数据在同一 LMDB，可与上方相同

TRAIN_XLSX = "/home/dell/autodl-tmp/lorafair/data/MOF_embedding_train-check580.xlsx"
VAL_XLSX = "/home/dell/autodl-tmp/lorafair/data/Embedding feature- Validation- Manual Checked.xlsx"

TRAIN_DST_ROOT = "/home/dell/autodl-tmp/lorafair/data/s2ef/filtered_train2"
VAL_DST_ROOT = "/home/dell/autodl-tmp/lorafair/data/s2ef/filtered_val2"

NUM_SPLITS = 40
NORMALIZE_LABELS = False
# ====================================


def load_names_from_excel(path: str, column: str = "Name") -> Set[str]:
    """从 Excel 读入 MOF 名称集合。
    - 自动去除空白、转为字符串、剔除空值。
    - 常见场景中名称已为大写，若需可在此强制 upper()。
    """
    df = pd.read_excel(path)
    if column not in df.columns:
        raise KeyError(f"文件 {path} 中未找到列 '{column}'，现有列：{list(df.columns)}")
    names = set()
    for v in df[column].dropna().tolist():
        s = str(v).strip()
        if s:
            names.add(s)  # 如需强制大写可改为 s.upper()
    return names


def build_dataset(src_path: str) -> LmdbDataset:
    config = {
        "src": src_path,
        "normalize_labels": NORMALIZE_LABELS,
    }
    return LmdbDataset(config)


def _to_scalar(x) -> Optional[float]:
    """尽可能从任意常见类型中提取标量 float。失败返回 None。"""
    if x is None:
        return None
    # 直接转 float
    try:
        return float(x)
    except Exception:
        pass
    # 有 item() 的（numpy 标量、torch 张量等）
    if hasattr(x, "item"):
        try:
            return float(x.item())
        except Exception:
            pass
    # list/tuple 取首元素递归
    if isinstance(x, (list, tuple)) and len(x) > 0:
        return _to_scalar(x[0])
    # numpy 数组
    try:
        import numpy as np  # noqa: F401
        if 'numpy' in str(type(x)):
            try:
                # 0 维或可展平
                return float(x.reshape(-1)[0])
            except Exception:
                pass
    except Exception:
        pass
    return None


def _extract_energy(sample: dict) -> Optional[float]:
    """尽可能从样本中提取 energy 标量，用于统计。
    优先顺序：energy, y_relaxed, y, target(s)。
    """
    for k in ["energy", "y_relaxed", "y", "target", "targets"]:
        if k in sample:
            v = _to_scalar(sample[k])
            if v is not None:
                return v
    return None


def select_items(dataset: LmdbDataset, name_set: Set[str]) -> Tuple[List[Tuple[int, bytes]], List[float]]:
    """从数据集中筛选 MOF 前缀在 name_set 中的样本，返回：
    - selected: [(orig_key, pickled_bytes), ...]
    - energies: [energy_scalar, ...]（可用于统计 mean/std；无法解析时会跳过）
    """
    selected: List[Tuple[int, bytes]] = []
    energies: List[float] = []
    total = len(dataset)
    for i in tqdm(range(total), desc="遍历并筛选", ncols=100):
        sample = dataset[i]  # 已解包为 dict
        raw_name = sample.get("name", "")
        prefix = raw_name.split("_")[0]
        if prefix in name_set:
            e = _extract_energy(sample)
            if e is not None:
                energies.append(float(e))
            data_bytes = pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL)
            selected.append((sample.get("iid") or i, data_bytes))
    return selected, energies


def write_splits(selected_items: List[Tuple[int, bytes]], dst_root: str, num_splits: int) -> None:
    """将 selected_items 拆分并写入多个 LMDB 分片。"""
    if not selected_items:
        print(f"[跳过] 没有可写入的数据：{dst_root}")
        return

    os.makedirs(dst_root, exist_ok=True)

    n = len(selected_items)
    items_per_split = n // num_splits + (n % num_splits > 0)
    print(f"将数据拆分成 {num_splits} 个 LMDB 文件，每份约 {items_per_split} 条样本 → {dst_root}")

    for split_idx in range(num_splits):
        start = split_idx * items_per_split
        end = min((split_idx + 1) * items_per_split, n)
        chunk = selected_items[start:end]
        if not chunk:
            break

        split_bytes = sum(len(v) for _, v in chunk)
        map_size = int(split_bytes * 1.3) + 32 * 1024 * 1024

        dst_path = os.path.join(dst_root, f"filtered_part_{split_idx+1:02d}.lmdb")
        print(f"→ 写入 {dst_path}，共 {len(chunk)} 条数据")

        env = lmdb.open(dst_path, map_size=map_size, subdir=False)
        try:
            with env.begin(write=True) as txn:
                for idx, (orig_key, val) in enumerate(tqdm(chunk, desc=f"写入 LMDB {split_idx+1:02d}", leave=False, ncols=100)):
                    txn.put(str(idx).encode(), val)
            env.sync()
        finally:
            env.close()


def _mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    n = len(values)
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / n  # 使用总体方差
    std = math.sqrt(var)
    return mean, std


def process_one_split(src_lmdb: str, names_xlsx: str, dst_root: str, tag: str, report_stats: bool = False) -> None:
    print("====================")
    print(f"[{tag}] 读取名称：{names_xlsx}")
    name_set = load_names_from_excel(names_xlsx, column="Name")
    print(f"[{tag}] 名称数：{len(name_set)}")

    print(f"[{tag}] 构建数据集：{src_lmdb}")
    dataset = build_dataset(src_lmdb)
    print(f"[{tag}] 原数据大小：{len(dataset)}")

    print(f"[{tag}] 开始筛选...")
    selected, energies = select_items(dataset, name_set)
    print(f"[{tag}] 匹配条目：{len(selected)}")

    if report_stats:
        mean, std = _mean_std(energies)
        print(f"[{tag}] energy 统计（基于筛选结果，总体统计）：")
        print(f"[{tag}] mean: {mean}")
        print(f"[{tag}] stdev: {std}")
        print(f"[{tag}] 可粘贴片段：\nnormalizer:\n  energy:\n    mean: {mean}\n    stdev: {std}")

    if not selected:
        print(f"[{tag}] 没有匹配数据，跳过写入。")
        return

    print(f"[{tag}] 写入分片到：{dst_root}")
    write_splits(selected, dst_root, NUM_SPLITS)
    print(f"[{tag}] 完成。")


if __name__ == "__main__":
    # 训练集
    process_one_split(
        src_lmdb=TRAIN_SRC_LMDB,
        names_xlsx=TRAIN_XLSX,
        dst_root=TRAIN_DST_ROOT,
        tag="Train",
        report_stats=False,  # 如需训练集统计可改为 True
    )

    # 验证集（按需统计 mean/stdev 并打印）
    process_one_split(
        src_lmdb=VAL_SRC_LMDB,
        names_xlsx=VAL_XLSX,
        dst_root=VAL_DST_ROOT,
        tag="Val",
        report_stats=True,
    )

    print("✅ 全部完成！")
