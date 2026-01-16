# import torch
# import torch.nn as nn
# import torchvision.transforms as T
# from PIL import Image
# import matplotlib.pyplot as plt
# import cv2
# import numpy as np
# from sklearn.decomposition import PCA

# # 1. 加载 DINOv2 模型 (选择最小的 ViT-S 版本，适合快速实验)
# # 也可以选择 'dinov2_vitb14' (中等) 或 'dinov2_vitl14' (大型)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
# model.eval()

# # 2. 预处理函数
# # DINOv2 要求输入是 14 的倍数。即使你的原图是 64，我们将其放缩到 224 或 448 以获得更精细的特征
# patch_size = 14
# img_size = 448  # 448 / 14 = 32 patches 每行，能提供较好的分辨率

# transform = T.Compose([
#     T.Resize((img_size, img_size)),
#     T.ToTensor(),
#     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# def visualize_attention(img_path, layer_idx=None):
#     """
#     可视化 DINOv2 特征，支持选择不同层
#     Args:
#         img_path: 图像路径
#         layer_idx: 要提取的层索引 (0-11，共12层)
#                    None 时使用最后一层 (11)
#                    可以使用负数从后往前数，如 -1 表示最后一层，-6 表示中间层
#     """
#     # 加载图像
#     img_raw = Image.open(img_path).convert('RGB')
#     img_tensor = transform(img_raw).unsqueeze(0).to(device)

#     # 3. 提取特征并使用 PCA 降维
#     with torch.no_grad():
#         # DINOv2 ViT-S 共 12 层 (0-11)
#         # 获取所有中间层特征
#         all_features = model.get_intermediate_layers(img_tensor, n=12)  # 获取所有 12 层
        
#         # 选择要可视化的层
#         if layer_idx is None:
#             # 默认使用最后一层
#             features = all_features[-1]
#             layer_desc = "Last layer (11/12)"
#         else:
#             # 支持负数索引
#             features = all_features[layer_idx]
#             layer_desc = f"Layer {layer_idx} / 12"
        
#         # 提取网格尺寸
#         w, h = img_size // patch_size, img_size // patch_size
#         num_patches = w * h
        
#         # 转为 numpy 并展平为 (num_patches, feature_dim)
#         features_np = features.squeeze(0).cpu().numpy()  # (num_patches, dim)
        
#         # 应用 PCA 降维到 3 维（对应 RGB 三通道）
#         pca = PCA(n_components=3)
#         features_pca = pca.fit_transform(features_np)  # (num_patches, 3)
        
#         # 归一化到 [0, 1]
#         features_pca_min = features_pca.min(axis=0, keepdims=True)
#         features_pca_max = features_pca.max(axis=0, keepdims=True)
#         features_pca_norm = (features_pca - features_pca_min) / (features_pca_max - features_pca_min + 1e-8)
        
#         # reshape 回空间维度 (h, w, 3) 作为 RGB 图像
#         heatmap = features_pca_norm.reshape(h, w, 3)

#     # 4. 后处理：将 PCA 结果调整为原图尺寸
#     # heatmap 已经是 (h, w, 3) 的 RGB 形式
#     heatmap_resized = cv2.resize(heatmap, (img_raw.size[0], img_raw.size[1]))
#     heatmap_rgb = np.uint8(255 * heatmap_resized)  # 转为 uint8 RGB 格式

#     # 绘图：并排显示原图和 PCA 结果
#     plt.figure(figsize=(16, 7))
    
#     plt.subplot(1, 2, 1)
#     plt.imshow(img_raw)
#     plt.title("Original Image")
#     plt.axis('off')
    
#     plt.subplot(1, 2, 2)
#     plt.imshow(heatmap_rgb)
#     plt.title(f"DINOv2 Feature (PCA to RGB) - {layer_desc}")
#     plt.axis('off')
    
#     plt.tight_layout()
#     plt.show()

# 使用你的图片路径运行
# 可选择不同层来对比效果：
# layer_idx=None  -> 最后一层 (11/12)，语义性最强
# layer_idx=-6    -> 中间层 (6/12)，细节与语义平衡
# layer_idx=0     -> 第一层 (0/12)，细节最丰富
# visualize_attention('./dataset_yolo/images/val/5AQH050006A0EA_TDI_D0926G0527_S1_NON_PI800.jpg', layer_idx=6)  # 推荐用中间层


# import torch
# import torchvision.transforms as transforms
# import torch.nn.functional as F
# from PIL import Image

# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# def compute_patch_similarity_heatmap(patch_features, H, W, target_patch_coord):
#     """
#     计算指定patch与其他所有patch的余弦相似性，并生成热力图
    
#     Args:
#         patch_features: patch特征张量, shape (1, num_patches, feature_dim)
#         H: patch网格高度
#         W: patch网格宽度  
#         target_patch_coord: 目标patch坐标 (h_idx, w_idx)
    
#     Returns:
#         heatmap: 相似性热力图, shape (H, W)
#     """

#     assert patch_features.shape[1] == H * W, f"特征数量{H*W}与网格大小{H}x{W}不匹配"
    
#     # 提取目标patch的特征
#     target_idx = target_patch_coord[0] * W + target_patch_coord[1]
#     target_feature = patch_features[0, target_idx]  # shape (feature_dim,)
    
#     # 计算余弦相似性
#     similarities = F.cosine_similarity(
#         target_feature.unsqueeze(0),  # shape (1, feature_dim)
#         patch_features[0],            # shape (num_patches, feature_dim)
#         dim=1
#     )
    
#     # 重塑为2D热力图
#     heatmap = similarities.reshape(H, W).cpu().numpy()
    
#     return heatmap

# def plot_similarity_heatmap(heatmap, target_patch_coord):
#     """
#     绘制相似性热力图，并在目标patch位置显示红点
    
#     Args:
#         heatmap: 相似性热力图, shape (H, W)
#         target_patch_coord: 目标patch坐标 (h_idx, w_idx)
#     """
#     H, W = heatmap.shape
    
#     fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
#     # 显示热力图
#     im = ax.imshow(heatmap, cmap='viridis', aspect='equal')
    
#     # 在目标patch位置添加红点
#     target_h, target_w = target_patch_coord
#     ax.plot(target_w, target_h, 'ro', markersize=10, markeredgecolor='white', markeredgewidth=2)
    
#     # 添加颜色条
#     plt.colorbar(im, ax=ax, label='Cosine Similarity')
    
#     # 设置坐标轴标签
#     ax.set_xlabel('Width (patch index)')
#     ax.set_ylabel('Height (patch index)')
#     ax.set_title(f'Cosine Similarity to Patch at ({target_h}, {target_w})')
    
#     # 设置网格线
#     ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
#     ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
#     ax.grid(which="minor", color="white", linestyle='-', linewidth=0.5)
#     ax.tick_params(which="minor", size=0)
    
#     # 设置主刻度
#     ax.set_xticks(np.arange(0, W, max(1, W//10)))
#     ax.set_yticks(np.arange(0, H, max(1, H//10)))
    
#     plt.tight_layout()
#     plt.show()
    
#     return fig, ax

# # 加载 DINOv2 模型
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
# model.eval()

# # 预处理函数
# patch_size = 14
# img_size = 448  # 448 / 14 = 32 patches 每行

# transform = transforms.Compose([
#     transforms.Resize((img_size, img_size)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# # 你的图像的位置
# url = './dataset_yolo/images/train/5ACC290026B0DJ_TDI_D3067G0455_S1_PXL9_PI800_NON_PI800.jpg'
# image = Image.open(url).convert('RGB')
# print(image)

# tensor_image = transform(image).unsqueeze(0).to(device)

# with torch.no_grad():
#     # 获取 patch 分辨率
#     H = W = img_size // patch_size  # 32
#     print(f"Patch 分辨率: 高度 {H} patches, 宽度 {W} patches")
#     print(f"总共 {H * W} 个 patches")
    
#     # 提取特征
#     features_dict = model.forward_features(tensor_image)
#     patch_features = features_dict["x_norm_patchtokens"]
#     print("patch_features.shape: ", patch_features.shape)
    
#     # 你选中的目标patch的坐标，注意要在[0,H-1]和[0,W-1]范围内。
#     target_patch_coord = (15, 15)  # 示例坐标，调整为合适值
#     heatmap = compute_patch_similarity_heatmap(patch_features, H, W, target_patch_coord)
    
#     plot_similarity_heatmap(heatmap, target_patch_coord)


import torch
from modelscope import AutoImageProcessor, AutoModel
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.decomposition import PCA

# 1. 加载模型和处理器
pretrained_model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
# 如果想得到更多 patch (如 448x448 -> 28x28 patches)，在此指定 size
processor = AutoImageProcessor.from_pretrained(pretrained_model_name, size={"height": 1024, "width": 1024})
model = AutoModel.from_pretrained(pretrained_model_name, device_map="auto")

# 2. 加载图片并推理
url = "./Data/dataset_yolo/images/val/5AQH050006B2HA_TDI_D1112G0398_S1_NON_PI800.jpg"
image = Image.open(url).convert('RGB')
inputs = processor(images=image, return_tensors="pt").to(model.device)

with torch.inference_mode():
    outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state  # [1, N, 1024]

# 3. 特征提取逻辑修改
features_all = last_hidden_state.squeeze(0).cpu().numpy()

# DINOv3 结构通常是: [CLS(1)] + [Registers(4)] + [Patch Tokens(N-5)]
# 因此我们要从索引 5 开始取
num_registers = 4 
spatial_features = features_all[1 + num_registers:] 
num_patches = spatial_features.shape[0]

# 计算网格尺寸 (196 -> 14x14, 784 -> 28x28)
h = w = int(np.sqrt(num_patches))
print(f"Total tokens: {features_all.shape[0]}, Spatial patches: {num_patches}, Grid: {h}x{w}")

# 4. PCA 降维
pca = PCA(n_components=3)
pca_features = pca.fit_transform(spatial_features) # [196, 3]

# 归一化
pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min() + 1e-8)

# Reshape 成图像格式
heatmap = pca_features.reshape(h, w, 3)

# 5. 可视化
# 将原图缩放到热力图的尺寸
image_resized = image.resize((w, h), Image.Resampling.NEAREST)
image_array = np.array(image_resized) / 255.0  # 转为 numpy 数组并归一化到 [0, 1]

# 将热力图转为 uint8 格式显示
heatmap_rgb = np.uint8(255 * heatmap)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_array)
plt.title(f"Original Image (resized to {h}x{w})")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(heatmap_rgb)
plt.title(f"DINOv3 PCA Feature ({h}x{w} Grid)")
plt.axis('off')
plt.show()