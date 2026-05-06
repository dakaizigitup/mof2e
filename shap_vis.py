from fairchem.core.datasets import create_dataset
import torch
from torch.utils.data import DataLoader
import os


def load_odac23_lmdb(lmdb_path, split="train"):
    """正确加载LMDB格式的ODAC23数据集"""

    config = {
        # 数据源 - LMDB文件夹路径
        "src": lmdb_path,

        # 指定数据格式为LMDB
        "format": "lmdb",

        # AtomsToGraphs参数
        "a2g_args": {
            "r_energy": True,  # 读取能量（CO2吸附能）
            "r_forces": False,  # ODAC23不包含力
            "r_stress": False,  # 不包含应力
            "r_distances": True,  # 计算原子间距离
            "r_edges": True,  # 构建图的边
            "r_pbc": True,  # 周期性边界条件
            "r_fixed": False,  # 固定原子信息
        },

        # 数据变换和归一化
        "transforms": {
            "normalizer": {
                "energy": {
                    "mean": 0.0,
                    "std": 1.0,
                }
            }
        },

        # 内存管理
        "keep_in_memory": False,  # 大数据集建议False

        # 其他选项
        "lin_ref": False,
        "otf_graph": True,  # 在线构建图
    }

    # 使用create_dataset创建LMDB数据集
    dataset = create_dataset(config,split)

    return dataset


# 使用示例
def load_all_odac23_splits(data_dir):
    """加载所有ODAC23数据分割"""

    datasets = {}
    splits = ['train', 'val']

    for split in splits:
        lmdb_path = os.path.join(data_dir, split)

        if os.path.exists(lmdb_path):
            try:
                dataset = load_odac23_lmdb(lmdb_path, split)
                datasets[split] = dataset
                print(f" 成功加载 {split}: {len(dataset)} 样本")

                # 检查LMDB文件夹内容
                lmdb_files = os.listdir(lmdb_path)
                print(f"  LMDB文件: {lmdb_files}")

            except Exception as e:
                print(f" 加载 {split} 失败: {e}")
        else:
            print(f" 路径不存在: {lmdb_path}")

    return datasets

dataset = load_all_odac23_splits('/data/wyx/ODAC23/s2ef/')
mm = 0
