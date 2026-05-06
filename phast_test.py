import os
import torch
from tqdm import tqdm
from phast.embedding import PhysEmbedding
import argparse
from pathlib import Path


def process_pt_files(input_dir, output_dir, batch_size=32):
    """
    处理指定文件夹中的所有PT文件，计算嵌入信息并保存到输出文件夹

    Args:
        input_dir: 输入文件夹路径，包含原始PT文件
        output_dir: 输出文件夹路径，将保存添加了嵌入信息的PT文件
        batch_size: 批处理大小，用于减少内存使用
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有PT文件路径
    pt_files = [f for f in os.listdir(input_dir) if f.endswith('.pt')]
    print(f"找到 {len(pt_files)} 个PT文件需要处理")

    # 创建PhysEmbedding模型
    phys_embedding = PhysEmbedding(
        z_emb_size=64,
        tag_emb_size=32,
        period_emb_size=32,
        group_emb_size=32,
        properties_proj_size=64,
        final_proj_size=128
    )

    # 将模型移至GPU（如果可用）
    phys_embedding = phys_embedding

    # 批量处理文件
    for i in tqdm(range(0, len(pt_files), batch_size), desc="处理批次"):
        batch_files = pt_files[i:i + batch_size]

        for file_name in batch_files:
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)

            # 如果输出文件已存在，跳过处理
            if os.path.exists(output_path):
                continue

            try:
                # 加载数据
                data = torch.load(input_path)

                # 将数据移至设备
                atomic_numbers = data.atomic_numbers.long()
                tags = data.tags if hasattr(data, 'tags') else None
                # 获取嵌入
                with torch.no_grad():  # 不需要梯度计算
                    # 获取中间嵌入（projection前）
                    pg = phys_embedding.phys_ref.period_and_group(atomic_numbers)

                    # 初始化一个字典来存储各种嵌入
                    embeddings_dict = {}

                    # 分别获取各种嵌入
                    for e, emb in phys_embedding.embeddings.items():
                        if e in pg:
                            embeddings_dict[e] = emb(pg[e]).cpu()
                        elif e in {"z", "properties"}:
                            if e == "properties" and isinstance(emb, torch.nn.Sequential):
                                # 获取PropertiesEmbedding的输出（未经过projection）
                                raw_properties = emb[0](atomic_numbers).cpu()
                                embeddings_dict[f"{e}_raw"] = raw_properties
                                # 也可以保存projection后的结果
                                embeddings_dict[e] = emb(atomic_numbers).cpu()
                            else:
                                embeddings_dict[e] = emb(atomic_numbers).cpu()
                        elif e == "tag" and tags is not None:
                            embeddings_dict[e] = emb(tags).cpu()

                    # 获取最终嵌入
                    if tags is not None:
                        final_embeddings = phys_embedding(atomic_numbers, tags).cpu()
                    else:
                        final_embeddings = phys_embedding(atomic_numbers).cpu()

                # 将嵌入信息添加到数据中
                data.embeddings_dict = embeddings_dict
                data.node_embeddings = final_embeddings

                # 保存增强后的数据
                torch.save(data, output_path)

            except Exception as e:
                print(f"处理文件 {file_name} 时出错: {e}")

    print(f"处理完成! 结果保存在 {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='处理PT文件并添加嵌入信息')
    parser.add_argument('--input_dir', type=str, default='/home/wyx/fairchem/dataset_torch_files_new')
    parser.add_argument('--output_dir', type=str, default='/home/wyx/fairchem/dataset_final')
    parser.add_argument('--batch_size', type=int, default=1, help='批处理大小')

    args = parser.parse_args()

    process_pt_files(args.input_dir, args.output_dir, args.batch_size)
    # ss = torch.load('./dataset_torch_files_new/BEYKEX_w_CO2_random_1.pt')
    # m = 0