import torch
from fairchem.core.datasets import LmdbDataset
import os
import tqdm


ss = torch.load("./dataset_torch_files/ABEFUL_w_CO2_6.pt")

# 创建输出目录
output_dir = "dataset_torch_files"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"创建输出目录: {output_dir}")

# 加载数据集
dataset = LmdbDataset(config=dict(src="/data/wyx/ODAC23/is2r/train", r_energy=True))

# 使用tqdm显示进度条
for i in tqdm.tqdm(range(len(dataset)), desc="处理数据集"):
    try:
        data = dataset[i]

        # 获取数据名称作为文件名
        name = data['name']
        safe_name = ''.join(c if c.isalnum() or c in ['-', '_'] else '_' for c in name)

        # 保存为PyTorch文件
        file_path = os.path.join(output_dir, f"{safe_name}.pt")
        torch.save(data, file_path)

    except Exception as e:
        print(f"处理索引 {i} 时出错: {str(e)}")

print(f"完成! 已将数据条目保存到 {output_dir} 目录")