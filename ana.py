import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.util import view_as_windows

# --- 1. 参数设置 (您可以调整这些参数) ---
IMAGE_PATH = "result_from_json.png"  # <--- 修改这里！请填入您的图片文件名
PATCH_SIZE = 64              # 图像块的大小 (建议为2的幂，如32, 64)
STEP = 16                    # 提取图像块的步长 (步长越小，块越多，重叠越大)
BATCH_SIZE = 128             # 训练时的批处理大小
LEARNING_RATE = 1e-3         # 学习率
NUM_EPOCHS = 20              # 训练轮数 (可以根据需要增加)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 自动选择GPU或CPU

print(f"Using device: {DEVICE}")

# --- 2. 定义卷积自编码器模型 ---
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # -> 16 x 64 x 64
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),                  # -> 16 x 32 x 32
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # -> 32 x 32 x 32
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),                  # -> 32 x 16 x 16
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # -> 64 x 16 x 16
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)                   # -> 64 x 8 x 8 (潜在空间)
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), # -> 32 x 16 x 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2), # -> 16 x 32 x 32
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),  # -> 1 x 64 x 64
            nn.Sigmoid() # 将输出压缩到0-1范围
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# --- 3. 数据准备 ---
# 自定义数据集类
class PatchDataset(Dataset):
    def __init__(self, patches, transform=None):
        self.patches = patches
        self.transform = transform

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        patch = Image.fromarray(patch) # 转为PIL Image
        if self.transform:
            patch = self.transform(patch)
        return patch

# 加载并处理图像
try:
    img = Image.open(IMAGE_PATH).convert('L') # 打开图像并转为灰度图
except FileNotFoundError:
    print(f"错误：找不到图像文件 '{IMAGE_PATH}'。请确保文件名正确且文件与脚本在同一目录下。")
    exit()

img_np = np.array(img, dtype=np.float32) / 255.0 # 归一化到 [0, 1]

# 使用 view_as_windows 高效提取重叠的图像块
patches = view_as_windows(img_np, (PATCH_SIZE, PATCH_SIZE), step=STEP)
# 将多维数组重塑为 (N, patch_height, patch_width)
patches_flat = patches.reshape(-1, PATCH_SIZE, PATCH_SIZE)
print(f"从图像中提取了 {len(patches_flat)} 个图像块。")

# 创建数据加载器
transform = transforms.Compose([transforms.ToTensor()])
dataset = PatchDataset(patches_flat, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- 4. 训练模型 ---
model = ConvAutoencoder().to(DEVICE)
criterion = nn.MSELoss() # 损失函数：均方误差
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("\n开始训练模型...")
for epoch in range(NUM_EPOCHS):
    total_loss = 0
    for data in dataloader:
        inputs = data.to(DEVICE)
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], 平均损失: {avg_loss:.6f}')
print("模型训练完成！\n")

# --- 5. 计算重建误差并可视化 ---
print("正在计算重建误差热力图...")
model.eval() # 将模型设置为评估模式

# 初始化误差图和计数图，用于处理重叠区域
error_map = np.zeros_like(img_np, dtype=np.float32)
count_map = np.zeros_like(img_np, dtype=np.float32)
reconstructed_img = np.zeros_like(img_np, dtype=np.float32)

with torch.no_grad():
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            patch = patches[i, j]
            patch_tensor = transform(Image.fromarray(patch)).unsqueeze(0).to(DEVICE)
            
            # 获取重建块
            reconstructed_patch = model(patch_tensor)
            reconstructed_patch_np = reconstructed_patch.squeeze().cpu().numpy()
            
            # 计算原始块和重建块之间的均方误差
            error = np.mean((patch - reconstructed_patch_np)**2)
            
            # 将误差和重建块累加到对应位置
            row_start, col_start = i * STEP, j * STEP
            error_map[row_start:row_start+PATCH_SIZE, col_start:col_start+PATCH_SIZE] += error
            reconstructed_img[row_start:row_start+PATCH_SIZE, col_start:col_start+PATCH_SIZE] += reconstructed_patch_np
            count_map[row_start:row_start+PATCH_SIZE, col_start:col_start+PATCH_SIZE] += 1

# 对重叠区域取平均值
error_map /= count_map
reconstructed_img /= count_map

# 归一化误差图以便更好地可视化
error_map_normalized = (error_map - np.min(error_map)) / (np.max(error_map) - np.min(error_map))

print("正在生成结果图像...")
# --- 6. 显示结果 ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 原始图像
axes[0].imshow(img, cmap='gray')
axes[0].set_title('原始图像 (Original Image)')
axes[0].axis('off')

# 重建图像
axes[1].imshow(reconstructed_img, cmap='gray')
axes[1].set_title('重建图像 (Reconstructed Image)')
axes[1].axis('off')

# 误差热力图
axes[2].imshow(img, cmap='gray') # 先显示原始图像作为背景
heatmap = axes[2].imshow(error_map_normalized, cmap='jet', alpha=0.5) # 在上面叠加半透明的热力图
axes[2].set_title('缺陷热力图 (Defect Heatmap)')
axes[2].axis('off')
fig.colorbar(heatmap, ax=axes[2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()