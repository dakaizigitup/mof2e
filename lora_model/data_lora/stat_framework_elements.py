"""统计 LMDB 数据集中非吸附物（tags!=2）原子的元素类型。

使用方法：
1. 修改 `LMDB_SRC` 为你的 .lmdb 文件或目录路径。
2. 可选：修改 `MAX_SAMPLES` 限制遍历样本数（None 表示全部）。
3. 运行：

   python stat_framework_elements.py
"""

from collections import Counter
from pathlib import Path

import torch
from fairchem.core.common.utils import setup_logging
from fairchem.core.datasets import LmdbDataset

# ======== 用户可修改的变量 ======== #
LMDB_SRC = "/home/dell/autodl-tmp/lorafair/data/mof2e/train/mof_only_aggregate_yrelaxed.lmdb"   # <<< 指向你的 LMDB
MAX_SAMPLES = None                          # 限制遍历条数；None 表示全部
# =================================== #

# 常见原子序数 → 元素符号，可按需扩充
_Z2SYM = {
    1: "H", 2: "He",
    3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
    11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar",
    19: "K", 20: "Ca",
    22: "Ti", 23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn",
    35: "Br", 53: "I",
    46: "Pd", 47: "Ag", 48: "Cd",
    78: "Pt", 79: "Au",
}

setup_logging()

def _safe_int_atomic_numbers(x: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(x):
        return x
    if x.dtype.is_floating_point:
        return x.round().to(torch.long)
    return x.to(torch.long)

def main():
    cfg = {
        "src": LMDB_SRC,
        "normalize_labels": False,
        "target_mean": 0.0,
        "target_std": 1.0,
        "lin_ref": None,
    }

    if not Path(LMDB_SRC).exists():
        raise FileNotFoundError(f"LMDB 路径不存在: {LMDB_SRC}")

    print(f"🔄 正在加载数据集: {LMDB_SRC}")
    ds = LmdbDataset(cfg)
    print(f"✅ 数据集大小: {len(ds)}")

    n_scan = len(ds) if MAX_SAMPLES is None else min(MAX_SAMPLES, len(ds))
    print(f"➡️  将遍历样本数: {n_scan}")

    counter = Counter()
    missing_tag = 0
    for i in range(n_scan):
        s = ds[i]
        tags = None
        atomic_numbers = None
        # 兼容 dict & Data
        if isinstance(s, dict):
            tags = s.get("tags", None)
            atomic_numbers = s.get("atomic_numbers", None)
        else:
            atomic_numbers = getattr(s, "atomic_numbers", None)
            tags = getattr(s, "tags", None)

        if tags is None or atomic_numbers is None:
            missing_tag += 1
            continue

        if not torch.is_tensor(tags):
            tags = torch.tensor(tags)
        atomic_numbers = _safe_int_atomic_numbers(atomic_numbers)

        mask = tags != 2  # 非吸附物
        for z in atomic_numbers[mask].tolist():
            sym = _Z2SYM.get(int(z), f"Z{int(z)}")
            counter[sym] += 1

    print("\n--- 非吸附物元素统计 ---")
    if counter:
        for sym, cnt in counter.most_common():
            print(f"  {sym}: {cnt}")
        print(f"\n总元素种类: {len(counter)}")
    else:
        print("未统计到任何元素！")

    if missing_tag:
        print(f"⚠️ 有 {missing_tag} 条样本缺少 tags 或 atomic_numbers，已跳过")

if __name__ == "__main__":
    main()
