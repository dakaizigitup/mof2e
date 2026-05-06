import os
import csv
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
from tqdm import tqdm
from fairchem.core.datasets import LmdbDataset

"""
用途：
- 扫描 LMDB 数据集，输出每个样本的原子数（natoms）及其 MOF 名称前缀；
- 统计整体原子数分布，以及按 MOF 前缀聚合的统计（count/min/mean/p50/p95/max）；
- 方便定位是否存在极端大的样本导致 OOM。

使用：
- 直接运行：python analyze_natoms_distribution.py
- 如需改路径，修改下方 SRC 和 OUT_DIR；或用环境变量传入：
  环境变量 SRC=/path/to/lmdb_dir OUT_DIR=/path/to/out python analyze_natoms_distribution.py

说明：
- 以 sample['name'] 的下划线前缀作为 MOF 名称；若无 'name' 字段则回退到 'sid' 或 "unknown"。
- natoms 优先取 sample['natoms']，没有则取 len(sample['pos'])。
- 输出两份 CSV：
  - natoms_per_sample.csv：每条样本一行（name, natoms）
  - natoms_per_mof_summary.csv：按 MOF 前缀聚合的统计（count/min/mean/p50/p95/max）
"""

# 默认路径（可按需修改）
SRC = os.environ.get("SRC", "/home/dell/autodl-tmp/lorafair/data/is2r/filtered_train2")
OUT_DIR = os.environ.get("OUT_DIR", "/home/dell/autodl-tmp/lorafair/fairchem/lora_model/data_lora")


def _to_scalar(x) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        pass
    if hasattr(x, "item"):
        try:
            return int(x.item())
        except Exception:
            pass
    return None


def _get_name_prefix(sample: dict) -> str:
    name = sample.get("name")
    if not name:
        name = str(sample.get("sid", "unknown"))
    return str(name).split("_")[0]


def _get_natoms(sample: dict) -> Optional[int]:
    if "natoms" in sample:
        v = _to_scalar(sample["natoms"])
        if v is not None:
            return v
    if "pos" in sample and sample["pos"] is not None:
        try:
            return int(len(sample["pos"]))
        except Exception:
            pass
    return None


def describe(arr: List[int]) -> Dict[str, float]:
    a = np.array(arr, dtype=np.int64)
    return {
        "count": int(a.size),
        "min": float(a.min()) if a.size else float("nan"),
        "mean": float(a.mean()) if a.size else float("nan"),
        "p50": float(np.percentile(a, 50)) if a.size else float("nan"),
        "p90": float(np.percentile(a, 90)) if a.size else float("nan"),
        "p95": float(np.percentile(a, 95)) if a.size else float("nan"),
        "p99": float(np.percentile(a, 99)) if a.size else float("nan"),
        "max": float(a.max()) if a.size else float("nan"),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"构建数据集：{SRC}")
    dataset = LmdbDataset({"src": SRC, "normalize_labels": False})
    print(f"样本总数：{len(dataset)}")

    per_sample: List[Tuple[str, int]] = []  # (name, natoms)
    per_mof: Dict[str, List[int]] = defaultdict(list)

    missing_natoms = 0

    for i in tqdm(range(len(dataset)), desc="扫描样本"):
        sample = dataset[i]
        name = str(sample.get("name", sample.get("sid", f"id{i}")))
        prefix = _get_name_prefix(sample)
        n = _get_natoms(sample)
        if n is None:
            missing_natoms += 1
            continue
        per_sample.append((name, n))
        per_mof[prefix].append(n)

    if missing_natoms:
        print(f"警告：有 {missing_natoms} 条样本未能解析 natoms（已跳过）")

    # 整体统计
    all_n = [n for _, n in per_sample]
    stats_all = describe(all_n)
    print("整体原子数统计：")
    for k in ["count", "min", "mean", "p50", "p90", "p95", "p99", "max"]:
        print(f"  {k}: {stats_all[k]}")

    # 打印 TOP-K 最大样本
    topk = 20
    print(f"原子数最大的 {topk} 个样本：")
    for name, n in sorted(per_sample, key=lambda x: x[1], reverse=True)[:topk]:
        print(f"  {name}: {n}")

    # 按 MOF 聚合统计
    mof_summary_rows = []
    for prefix, arr in per_mof.items():
        s = describe(arr)
        mof_summary_rows.append(
            {
                "mof": prefix,
                "count": s["count"],
                "min": s["min"],
                "mean": s["mean"],
                "p50": s["p50"],
                "p95": s["p95"],
                "max": s["max"],
            }
        )

    # 排序：优先按 max desc，次序按 count desc
    mof_summary_rows.sort(key=lambda r: (r["max"], r["count"]), reverse=True)

    # 写 CSV：每样本
    per_sample_csv = os.path.join(OUT_DIR, "natoms_per_sample.csv")
    with open(per_sample_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "natoms"])
        for name, n in per_sample:
            w.writerow([name, n])
    print(f"已写出：{per_sample_csv}")

    # 写 CSV：按 MOF 聚合
    per_mof_csv = os.path.join(OUT_DIR, "natoms_per_mof_summary.csv")
    with open(per_mof_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mof", "count", "min", "mean", "p50", "p95", "max"])
        for r in mof_summary_rows:
            w.writerow([r["mof"], r["count"], r["min"], r["mean"], r["p50"], r["p95"], r["max"]])
    print(f"已写出：{per_mof_csv}")

    # 友好提示：如何快速定位可疑大样本
    print("\n可疑大样本排查建议：")
    print("- 查看 natoms_per_sample.csv，按 natoms 降序筛选，定位前若干条记录；")
    print("- 查看 natoms_per_mof_summary.csv，按 max 或 p95 降序排序，找出异常 MOF；")
    print("- 训练时可对 natoms 设置上限过滤，或对大样本采用更小的 batch_size/梯度累积。")


if __name__ == "__main__":
    main()

