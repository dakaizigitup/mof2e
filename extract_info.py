from fairchem.core.datasets import LmdbDataset
import csv
import os
import tqdm
import pandas as pd
import collections
import re
import json
import numpy as np

# 加载数据集
dataset = LmdbDataset(config=dict(src="/data/wyx/ODAC23/is2r/train", r_energy=True))

# 创建一个字典来存储数据
data_dict = collections.defaultdict(lambda: collections.defaultdict(dict))
# 创建一个字典来存储包含FASFUD的MOF信息
fasfud_info = []

# 使用tqdm显示进度条
print(f"正在处理数据集，共{len(dataset)}个样本...")
for i in tqdm.tqdm(range(len(dataset))):
    sample = dataset[i]
    full_name = sample.name

    # 检查名称是否包含FASFUD
    if "FASFUD" in full_name:
        # 保存该MOF的所有详细信息
        fasfud_sample = {
            "name": full_name,
            "pos": sample.pos.tolist() if hasattr(sample, 'pos') else None,
            "cell": sample.cell.tolist() if hasattr(sample, 'cell') else None,
            "atomic_numbers": sample.atomic_numbers.tolist() if hasattr(sample, 'atomic_numbers') else None,
            "natoms": sample.natoms if hasattr(sample, 'natoms') else None,
            "tags": sample.tags.tolist() if hasattr(sample, 'tags') else None,
            "fixed": sample.fixed.tolist() if hasattr(sample, 'fixed') else None,
            "raw_y": float(sample.raw_y) if hasattr(sample, 'raw_y') else None,
            "y_init": float(sample.y_init),
            "y_relaxed": float(sample.y_relaxed),
            "pos_relaxed": sample.pos_relaxed.tolist() if hasattr(sample, 'pos_relaxed') else None,
            "nco2": sample.nco2,
            "nh2o": sample.nh2o,
            "nads": sample.nads if hasattr(sample, 'nads') else None,
            "sid": sample.sid.tolist() if hasattr(sample, 'sid') else None,
            "fid": sample.fid.tolist() if hasattr(sample, 'fid') else None,
            "supercell": sample.supercell.tolist() if hasattr(sample, 'supercell') else None,
            "oms": sample.oms if hasattr(sample, 'oms') else None,
            "defective": sample.defective if hasattr(sample, 'defective') else None,
            "id": sample.id if hasattr(sample, 'id') else None
        }

        # 移除None值，只保留实际存在的数据
        fasfud_sample = {k: v for k, v in fasfud_sample.items() if v is not None}
        fasfud_info.append(fasfud_sample)

    # 原有的数据处理代码保持不变
    if "_w" in full_name:
        base_name = full_name.split("_w")[0]
        match = re.search(r'_(\d+)$', full_name)
        if match:
            index = int(match.group(1))
        else:
            index = 0
    else:
        base_name = full_name
        index = 0

    if "random" in full_name.lower():
        name_type = f"{base_name}_random"
    else:
        name_type = base_name

    nco2 = sample.nco2
    nh2o = sample.nh2o

    initial_column = f"{nco2}CO2+{nh2o}H2O-initial"
    relaxed_column = f"{nco2}CO2+{nh2o}H2O-relaxed"

    data_dict[name_type][index][initial_column] = sample.y_init
    data_dict[name_type][index][relaxed_column] = sample.y_relaxed

# 将FASFUD信息保存为JSON文件
if fasfud_info:
    fasfud_output_file = "fasfud_info.json"
    print(f"正在将FASFUD MOF信息写入{fasfud_output_file}...")


    # 自定义JSON编码器来处理numpy数组
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)


    with open(fasfud_output_file, 'w') as f:
        json.dump(fasfud_info, f, indent=2, cls=NumpyEncoder)
    print(f"FASFUD MOF信息已成功写入{os.path.abspath(fasfud_output_file)}")
    print(f"找到的FASFUD MOF样本数量: {len(fasfud_info)}")
else:
    print("未找到包含FASFUD的MOF样本")

# 创建DataFrame的代码保持不变
rows = []
columns = ["Name", "index"]
for name, indices in data_dict.items():
    for idx, values in indices.items():
        row = {"Name": name, "index": idx}
        for col, val in values.items():
            if col not in columns:
                columns.append(col)
            row[col] = val
        rows.append(row)

df = pd.DataFrame(rows, columns=columns)

# 按Name和index排序
df = df.sort_values(by=["Name", "index"])

# 重新排列列以匹配所需格式
ordered_columns = ["Name", "index"]
for nco2 in range(2):
    for nh2o in range(3):
        col_init = f"{nco2}CO2+{nh2o}H2O-initial"
        col_relaxed = f"{nco2}CO2+{nh2o}H2O-relaxed"
        if col_init in df.columns:
            ordered_columns.append(col_init)
        if col_relaxed in df.columns:
            ordered_columns.append(col_relaxed)

# 选择存在的列
final_columns = [col for col in ordered_columns if col in df.columns]
df = df[final_columns]

# 将数据写入CSV文件
output_file = "dataset_info.csv"
print(f"正在将数据写入{output_file}...")
df.to_csv(output_file, index=False)

print(f"数据已成功写入{os.path.abspath(output_file)}")
print(f"数据集大小: {len(dataset)}个样本")