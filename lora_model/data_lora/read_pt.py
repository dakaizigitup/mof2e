import torch

# 加载checkpoint文件
checkpoint_path = "/data0/wfz/code/fairchem/checkpoints/2025-08-26-16-55-28/checkpoint.pt"  # 替换为你的文件路径
checkpoint = torch.load(checkpoint_path, map_location='cpu')
print(checkpoint['step'])
# # 查看所有键
# print("=== Checkpoint Keys ===")
# for key in checkpoint.keys():
#     print(f"Key: {key}")
#     print(f"Type: {type(checkpoint[key])}")
#     if hasattr(checkpoint[key], 'shape'):
#         print(f"Shape: {checkpoint[key].shape}")
#     elif isinstance(checkpoint[key], (list, tuple)):
#         print(f"Length: {len(checkpoint[key])}")
#     elif isinstance(checkpoint[key], dict):
#         print(f"Dict with {len(checkpoint[key])} keys")
#     print("-" * 40)
