import os
import torch
from tqdm import tqdm
from phast.embedding import PhysEmbedding
import argparse
from pathlib import Path


def process_pt_files(data):
    """
    处理指定文件夹中的所有PT文件，计算嵌入信息并保存到输出文件夹

    Args:
        input_dir: 输入文件夹路径，包含原始PT文件
        output_dir: 输出文件夹路径，将保存添加了嵌入信息的PT文件
        batch_size: 批处理大小，用于减少内存使用
    """

    # 创建PhysEmbedding模型
    phys_embedding = PhysEmbedding(
        z_emb_size=64,
        tag_emb_size=32,
        period_emb_size=32,
        group_emb_size=32,
        properties_proj_size=64,
        final_proj_size=128
    )
    atomic_numbers = data.atomic_numbers.long()
    tags = data.tags if hasattr(data, 'tags') else None
    with torch.no_grad():  # 不需要梯度计算
        pg = phys_embedding.phys_ref.period_and_group(atomic_numbers)
        embeddings_dict = {}
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
            # torch.save(data, output_path)
            #
            # except Exception as e:
            #     print(f"处理文件 {file_name} 时出错: {e}")
    #
    # print(f"处理完成! 结果保存在 {output_dir}")

ss = torch.load('./dataset_torch_files_new/BEYKEX_w_CO2_random_1.pt')
process_pt_files(ss)