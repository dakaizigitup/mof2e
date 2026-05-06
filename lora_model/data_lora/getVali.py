import os
import lmdb
import pickle
from tqdm import tqdm
from fairchem.core.datasets import LmdbDataset

# ========= 1. 配置部分 (您需要修改这里) =========

# 输入：原始的、包含多个数据的 LMDB 文件路径
src_path = "/home/dell/autodl-tmp/lorafair/data/is2r/filtered_train"

# 输出：只包含一个数据的新 LMDB 文件的保存路径
# 注意：这里是一个文件路径，而不是目录
dst_path = "/home/dell/autodl-tmp/lorafair/data/is2r/validate_null/single_sample.lmdb"


# ========= 2. 读取原始 LMDB 并获取第一个样本 =========

print(f"➡️ 开始读取源 LMDB 文件: {src_path}")

# 使用 LmdbDataset 加载原始数据集
try:
    dataset = LmdbDataset({"src": src_path})
    print(f"✅ 成功加载源数据集，共 {len(dataset)} 个条目。")
except Exception as e:
    print(f"❌ 加载失败，请检查路径 '{src_path}' 是否正确。错误: {e}")
    exit()

# 检查数据集是否为空
if len(dataset) == 0:
    print("❌ 源数据集为空，无法提取样本。")
    exit()

# 获取第一个数据样本 (索引为 0)
first_sample = dataset[0]
print("✅ 已成功读取第一个数据样本。")


# ========= 3. 将单个样本写入新的 LMDB 文件 =========

print(f"➡️ 准备写入到新的 LMDB 文件: {dst_path}")

# 确保目标目录存在
dst_dir = os.path.dirname(dst_path)
if dst_dir:
    os.makedirs(dst_dir, exist_ok=True)

# 步骤 1: 将样本序列化为字节流
# 我们使用 pickle 来完成，这和 fairchem 内部的做法一致
data_bytes = pickle.dumps(first_sample, protocol=pickle.HIGHEST_PROTOCOL)
print(f"样本序列化后的字节大小: {len(data_bytes)}")

# 步骤 2: 估算新 LMDB 的 map_size
# map_size 必须比要写入的数据大。我们给它一些富余空间，比如2倍大小再加上1MB，以确保万无一失。
map_size = len(data_bytes) * 2 + 1024 * 1024 # 1MB buffer

# 步骤 3: 打开新的 LMDB 环境并写入数据
# `subdir=False` 表示 dst_path 是一个文件，而不是一个目录。
env = lmdb.open(dst_path, map_size=map_size, subdir=False)

try:
    with env.begin(write=True) as txn:
        # 在新的 LMDB 中，这个样本的键 (key) 是 '0'
        # lmdb 的键和值都必须是字节类型，所以我们需要 .encode()
        key = '0'.encode()
        txn.put(key, data_bytes)
    print(f"✅ 数据成功写入！")
finally:
    # 确保数据库被正确关闭
    env.sync()
    env.close()

print("\n" + "="*30)
print(f"🎉 操作完成！")
print(f"第一个样本已从 '{os.path.basename(src_path)}' 提取并保存到:")
print(f"{dst_path}")
print("="*30)