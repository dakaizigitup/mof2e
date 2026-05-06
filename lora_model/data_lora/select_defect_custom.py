"""可自定义目录与 MOF 名称的缺陷筛选脚本。

使用方法：
1. 修改下方 `LMDB_SRC` 变量为你的 LMDB 数据目录（可指向单个 .lmdb 文件或文件夹）。
2. 在同目录或任意位置准备一个包含 MOF 基名列表的纯文本文件，每行一个，如：

   ABOCUX
   IJUTEM
   ...

   将文件路径填入 `MOF_LIST_PATH`。
3. 运行脚本：

   python select_defect_custom.py

脚本会在当前目录生成 `defect_list.txt`，保存所有匹配样本的完整 name。
"""

import os
from collections import Counter
from pathlib import Path

import torch
from fairchem.core.datasets import LmdbDataset
from fairchem.core.common.utils import setup_logging

# ===== 用户需要修改的变量 ===== #
LMDB_SRC = "/path/to/your/lmdb_dir_or_file.lmdb"  # <<< 把这里改成你的 LMDB 路径
MOF_LIST_PATH = "mof_name_list.txt"                # <<< 把这里改成你的 MOF 名称列表文件
OUTPUT_TXT = "defect_list.txt"                     # 输出文件名，可自行调整
# ================================= #

setup_logging()


def _load_mof_name_list(txt_path: str):
    txt_path = Path(txt_path)
    if not txt_path.is_file():
        raise FileNotFoundError(f"未找到 MOF 列表文件: {txt_path}")
    names = []
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            nm = line.strip()
            if nm:
                names.append(nm)
    if not names:
        raise ValueError("MOF 名称列表文件为空！")
    return names


def main():
    mof_name_list = _load_mof_name_list(MOF_LIST_PATH)

    config = {
        "src": LMDB_SRC,
        "normalize_labels": False,
        "target_mean": 0.0,
        "target_std": 1.0,
        "lin_ref": None,
        "filter_mof_names": mof_name_list,
    }

    try:
        dataset = LmdbDataset(config)
        print(f"✅ 数据集加载成功，大小: {len(dataset)}")
    except Exception as e:
        print(f"❌ LmdbDataset 加载失败: {e}")
        return

    defect_list = []
    for i in range(len(dataset)):
        sample = dataset[i]
        name_full = sample["name"] if isinstance(sample, dict) else getattr(sample, "name", None)
        if not name_full:
            continue
        mof_base = str(name_full).split("_")[0]
        if mof_base in mof_name_list:
            defect_list.append(str(name_full))

    # 保存结果
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        for nm in defect_list:
            f.write(nm + "\n")
    print(f"✅ 已保存 {len(defect_list)} 条缺陷记录到 {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
