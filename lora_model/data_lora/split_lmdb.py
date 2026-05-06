import os
import lmdb
import pickle
import random
from tqdm import tqdm
from fairchem.core.datasets import LmdbDataset

# ========= 配置部分 =========
src_path = "/data/is2r/filtered_data"
dst_dir = "/data/is2r/filtered_val"
split_ratio = 0.8  # 80% 训练，20% 验证

# ========= 读取原始 LMDB =========
config = {
    "src": src_path,
    "normalize_labels": False
}
dataset = LmdbDataset(config)
print(f"✅ 原数据条目总数: {len(dataset)}")

# ========= 随机划分索引 =========
indices = list(range(len(dataset)))
random.shuffle(indices)
split_idx = int(len(indices) * split_ratio)
train_indices = indices[:split_idx]
val_indices = indices[split_idx:]

print(f"训练集: {len(train_indices)} 条, 验证集: {len(val_indices)} 条")

# ========= 准备保存路径 =========
os.makedirs(dst_dir, exist_ok=True)
train_path = os.path.join(dst_dir, "train.lmdb")
val_path = os.path.join(dst_dir, "val.lmdb")

def save_subset(indices, out_path):
    """将指定样本索引写入一个新的 LMDB 文件"""
    total_bytes = 0
    serialized = []
    for i in tqdm(indices, desc=f"预处理 {os.path.basename(out_path)}"):
        sample = dataset[i]
        data_bytes = pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL)
        serialized.append((i, data_bytes))
        total_bytes += len(data_bytes)
    map_size = int(total_bytes * 1.3) + 32 * 1024 * 1024

    env = lmdb.open(out_path, map_size=map_size, subdir=False)
    with env.begin(write=True) as txn:
        for new_i, (orig_i, data_bytes) in enumerate(tqdm(serialized, desc=f"写入 {os.path.basename(out_path)}")):
            txn.put(str(new_i).encode(), data_bytes)
    env.sync()
    env.close()
    print(f"✅ 保存完成: {out_path}")

# ========= 写入新的 LMDB 文件 =========
save_subset(train_indices, train_path)
save_subset(val_indices, val_path)

print("🎉 数据划分完成！")
print(f"训练集保存到: {train_path}")
print(f"验证集保存到: {val_path}")
