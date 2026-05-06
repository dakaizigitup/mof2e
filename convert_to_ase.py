import torch
import ase.db
from ase import Atoms
import numpy as np
from pathlib import Path

# ！！！关键步骤：确保你已经安装了 fairchem v1
# 并且你的 Python 环境可以找到 LmdbDataset 类。
from fairchem.core.datasets.lmdb_dataset import LmdbDataset

# --- 1. 配置输入和输出路径 ---

# [需要你修改] 你旧的 LMDB 数据集所在的文件夹路径
# 注意：这里应该指向包含 .lmdb 文件和 metadata.npz 的文件夹
old_lmdb_path = "/data/wyx/ODAC23/is2r/train"

# [需要你修改] 你希望生成的新 aselmdb 文件的路径和名字
new_aselmdb_path = "/data/wyx/new_dataset.aselmdb"

# --- 2. 加载旧的 LMDB 数据集 ---

# 根据 LmdbDataset 的 __init__ 方法，我们需要传入一个 config 字典
config = {"src": old_lmdb_path}
old_dataset = LmdbDataset(config)

print(f"成功加载旧数据集，路径: {old_lmdb_path}")
print(f"共找到 {len(old_dataset)} 条数据。")

# --- 3. 加载元数据 (metadata.npz) ---

metadata_path = Path(old_lmdb_path) / "metadata.npz"
metadata = {}
if metadata_path.is_file():
    print(f"找到并加载元数据文件: {metadata_path}")
    # old_dataset._metadata 会自动加载它，我们可以直接使用
    metadata = old_dataset._metadata
    print(f"元数据包含键: {list(metadata.keys())}")
else:
    print("警告: 在数据目录中未找到 metadata.npz 文件。")

# --- 4. 连接到新的 aselmdb 数据库并开始转换 ---

with ase.db.connect(new_aselmdb_path) as new_db:
    print(f"\n开始写入新的数据库: {new_aselmdb_path}")

    # 遍历旧数据集中的每一条数据
    for i in range(len(old_dataset)):
        # data 是一个 PyG Data 对象
        data = old_dataset[i]
        # a. 从 PyG Data 对象创建 ASE Atoms 对象
        positions = data.pos.numpy()
        atomic_numbers = data.atomic_numbers.long().numpy()
        pbc = data.get("pbc", False)
        cell = data.cell.numpy() if pbc else None

        atoms = Atoms(
            numbers=atomic_numbers,
            positions=positions,
            cell=cell,
            pbc=pbc,
        )
        mm = 0
        # b. 准备要存储的键值对 (key_value_pairs)
        key_value_pairs = {
            # 首先从 data 对象中获取核心数据
            'energy': float(data.y),
            'forces': data.force.numpy(),
        }

        # c. 从 metadata 中提取该样本的附加数据
        for key, value_array in metadata.items():
            # value_array 是一个包含整个数据集元数据的长数组
            # 我们需要取出当前索引 i 对应的值
            # np.generic 用于处理 NumPy 数据类型到 Python 原生类型的转换
            sample_metadata_value = value_array[i]
            if isinstance(sample_metadata_value, np.generic):
                key_value_pairs[key] = sample_metadata_value.item()
            else:
                key_value_pairs[key] = sample_metadata_value

        # d. 将 Atoms 对象和所有键值对信息写入新数据库
        new_db.write(atoms, key_value_pairs=key_value_pairs)

        if (i + 1) % 500 == 0:
            print(f"  已转换 {i + 1} / {len(old_dataset)} 条数据...")

print("\n转换完成！")
print(f"新的 aselmdb 文件已保存在: {new_aselmdb_path}")

# --- 5. (可选) 验证新数据库 ---
print("\n正在验证新数据库中的第一条数据...")
with ase.db.connect(new_aselmdb_path) as db:
    if len(db) > 0:
        row = db.get(1)  # 读取第一条记录 (ASE DB 索引从 1 开始)
        print("成功读取第一条记录。")
        print(f"包含的键: {list(row.key_value_pairs.keys())}")
        print(f"原子数: {row.natoms}")
        print(f"能量: {row.energy}")
    else:
        print("数据库为空，无法验证。")