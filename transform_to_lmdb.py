import os
import lmdb
import torch
import pickle
import random
import numpy as np
from glob import glob
from tqdm import tqdm
from pathlib import Path
from fairchem.core.preprocessing import AtomsToGraphs
from fairchem.core.datasets import LmdbDataset


def set_seed(seed=42):
    """设置随机种子以确保可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_is2re_lmdb(pt_files, output_path, split_name):
    """
    创建IS2RE的LMDB数据集
    Args:
        pt_files: PT文件路径列表
        output_path: 输出LMDB文件路径
        split_name: 数据集分割名称（train/val/test）
    """
    # 创建LMDB环境
    db = lmdb.open(
        output_path,
        map_size=1099511627776 * 2,  # 2TB空间
        subdir=False,
        meminit=False,
        map_async=True,
    )

    idx = 0
    for file_path in tqdm(pt_files, desc=f"创建{split_name}数据集"):
        # 加载PT文件
        data = torch.load(file_path)

            # 对于测试集，移除能量和力的信息
            # if split_name == "test":
            #     if hasattr(data, "y"):
            #         # 保存原始能量作为ground truth（仅用于评估）
            #         data.y_true = data.y.clone()
            #         # 移除能量信息
            #         del data.y
            #     if hasattr(data, "forces"):
            #         # 保存原始力作为ground truth（仅用于评估）
            #         data.forces_true = data.forces.clone()
            #         # 移除力信息
            #         del data.forces

            # 添加IS2RE任务所需的属性
            # data.sid = idx  # 唯一系统标识符

            # 检查边索引是否存在（防止索引错误）
        # if not hasattr(data, "edge_index") or data.edge_index.shape[1] == 0:
        #     print(f"警告: 文件 {file_path} 没有边索引或边索引为空，跳过")
        #     continue

        # 写入LMDB
        txn = db.begin(write=True)
        txn.put(f"{idx}".encode("ascii"), pickle.dumps(data, protocol=-1))
        txn.commit()
        db.sync()
        idx += 1

        # except Exception as e:
        #     print(f"处理文件 {file_path} 时出错: {e}")

    print(f"{split_name}数据集创建完成，共 {idx} 个样本")
    db.close()


def main():
    # 设置参数
    pt_files_dir = "/data/wyx/dataset_final"  # 处理后的PT文件目录
    output_dir_train = "/data/wyx/dataset_final_lmdb/train"  # LMDB数据集输出目录
    output_dir_val = "/data/wyx/dataset_final_lmdb/val"
    # output_dir_test = "/home/wyx/fairchem/is2re_lmdb_dataset/test"
    # 设置随机种子
    set_seed(42)

    # 创建输出目录
    os.makedirs(output_dir_train, exist_ok=True)
    os.makedirs(output_dir_val, exist_ok=True)
    # os.makedirs(output_dir_test, exist_ok=True)
    # 获取所有PT文件路径
    pt_files = glob(os.path.join(pt_files_dir, "*.pt"))
    total_files = len(pt_files)
    print(f"找到 {total_files} 个PT文件")

    if total_files == 0:
        print("错误: 未找到PT文件，请检查目录路径")
        return

    # 随机打乱文件列表
    random.shuffle(pt_files)

    # 按照70/20/10比例划分数据集
    train_size = int(total_files * 0.8)
    val_size = int(total_files * 0.2)
    # test_size = total_files - train_size - val_size

    train_files = pt_files[:train_size]
    val_files = pt_files[train_size:train_size + val_size]
    # test_files = pt_files[train_size + val_size:]

    print(f"训练集: {len(train_files)} 个文件")
    print(f"验证集: {len(val_files)} 个文件")
    # print(f"测试集: {len(test_files)} 个文件")

    # 创建训练集LMDB
    train_lmdb_path = os.path.join(output_dir_train, "train.lmdb")
    create_is2re_lmdb(train_files, train_lmdb_path, "train")

    # 创建验证集LMDB
    val_lmdb_path = os.path.join(output_dir_val, "val.lmdb")
    create_is2re_lmdb(val_files, val_lmdb_path, "val")

    # 创建测试集LMDB
    # test_lmdb_path = os.path.join(output_dir_test, "test.lmdb")
    # create_is2re_lmdb(test_files, test_lmdb_path, "test")

    # 测试加载数据集
    try:
        train_dataset = LmdbDataset({"src": train_lmdb_path})
        val_dataset = LmdbDataset({"src": val_lmdb_path})
        # test_dataset = LmdbDataset({"src": test_lmdb_path})

        print(f"成功加载数据集:")
        print(f"训练集: {len(train_dataset)} 个样本")
        print(f"验证集: {len(val_dataset)} 个样本")
        # print(f"测试集: {len(test_dataset)} 个样本")

        # 打印第一个样本的属性
        print("\n训练集第一个样本属性:")
        sample = train_dataset[0]
        for key in sample.__dict__.keys():
            if not key.startswith('_'):
                print(f"  {key}: {type(getattr(sample, key))}")

    except Exception as e:
        print(f"加载数据集时出错: {e}")


if __name__ == "__main__":
    main()
