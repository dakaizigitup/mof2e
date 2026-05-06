import torch
import yaml
from pathlib import Path

# # 加载checkpoint
# ckpt = torch.load("/home/wyx/AI4sci/fairchem/configs/odac/s2ef/ckpt_ocp/Equiformer_V2_Large.pt")
# state = ckpt['state_dict']
# print(ckpt.keys()) # dict_keys(['state_dict', 'normalizers', 'config', 'val_metrics', 'amp'])
# print(ckpt['val_metrics'])
import torch

# 加载checkpoint
ckpt = torch.load("/home/wyx/AI4sci/fairchem/configs/odac/s2ef/ckpt_ocp/Equiformer_V2_Large.pt")
state = ckpt['state_dict']

# 保存为txt文件
output_path = "/home/wyx/AI4sci/fairchem/lora_model/model_lora/Equiformer_V2_Large_state_dict.txt"

with open(output_path, 'w', encoding='utf-8') as f:
    f.write("Equiformer V2 Large - State Dictionary\n")
    f.write("=" * 50 + "\n\n")
    
    for key, value in state.items():
        f.write(f"Key: {key}\n")
        f.write(f"Shape: {value.shape}\n")
        f.write(f"Type: {value.dtype}\n")
        f.write(f"Device: {value.device}\n")
        f.write("-" * 30 + "\n")

print(f"State dict已保存到: {output_path}")

# # 打印配置（可选）
# print("Configuration:")
# print(config)

# # 保存为YAML文件
# output_path = "/home/wyx/AI4sci/fairchem/lora_model/model_lora/Equiformer_V2_Large_config.yaml"

# # 确保输出目录存在
# Path(output_path).parent.mkdir(parents=True, exist_ok=True)

# # 保存配置到YAML文件
# with open(output_path, 'w', encoding='utf-8') as f:
#     yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)

# print(f"\n配置已保存到: {output_path}")

# # 验证保存的文件（可选）
# print("\n验证保存的YAML文件:")
# with open(output_path, 'r', encoding='utf-8') as f:
#     loaded_config = yaml.safe_load(f)
#     print("成功加载YAML文件，配置项数量:", len(loaded_config) if loaded_config else 0)
