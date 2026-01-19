"""
PyTorch 原生 DINOv3 -> YOLO11n 知识蒸馏预训练脚本

完全绕过 lightly-train 兼容性问题，使用 PyTorch 原生 API 实现蒸馏。
参考 ziduo_test 分支的目标：预训练 YOLO11n，为后续有监督训练提供更好的初始化权重。

使用流程：
1. 运行此脚本进行蒸馏预训练（150 epochs）
2. 将输出的权重传递给 train_yolo11.py 或 dino_yolo.py
"""

import sys
import os
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# ==========================================
# 路径配置
# ==========================================
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print(f"📂 项目根目录: {PROJECT_ROOT}")

from ultralytics import YOLO
from transformers import AutoModel

# ==========================================
# 简单图像数据集
# ==========================================
class SimpleImageDataset(torch.utils.data.Dataset):
    """加载目录中的所有图像"""
    def __init__(self, image_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.image_files = list(self.image_dir.glob("*.jpg")) + \
                          list(self.image_dir.glob("*.png")) + \
                          list(self.image_dir.glob("*.jpeg"))
        self.transform = transform
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"⚠️ 无法加载图像 {img_path}: {e}")
            # 返回随机张量作为备选
            return torch.randn(3, 640, 640)

# ==========================================
# YOLO11 Backbone 提取器
# ==========================================
class YOLO11BackboneExtractor(nn.Module):
    """提取 YOLO11n 的 Backbone 部分"""
    def __init__(self, yolo_wrapper, layer_idx=10):
        super().__init__()
        # 关键修复：yolo_wrapper.model 是 DetectionModel，yolo_wrapper.model.model 才是 Sequential
        if hasattr(yolo_wrapper.model, 'model'):
            full_model = yolo_wrapper.model.model
        else:
            full_model = yolo_wrapper.model
            
        # 提取前 10 层 (0-9)，包含到 SPPF
        self.backbone = nn.Sequential(*list(full_model[:layer_idx]))
        
        # 自动对齐维度：YOLO11n 出口通常是 256，DINO-Tiny 是 384
        self.adapter = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 384)  # 对应 dino-vit-tiny-16
        )
    
    def forward(self, x):
        """返回对齐后的特征向量 [B, 384]"""
        features = self.backbone(x)  # [B, 256, H, W]
        return self.adapter(features)  # [B, 384]

# ==========================================
# DINOv3 Teacher 模型
# ==========================================
class DINOv3Teacher(nn.Module):
    """DINOv3 ViT-Tiny/16 作为 Teacher"""
    def __init__(self, model_name="facebook/dino-vitb16"):
        super().__init__()
        print(f"📥 加载 DINOv3 Teacher: {model_name}")
        self.teacher = AutoModel.from_pretrained(model_name)
        self.teacher.eval()  # 冻结 Teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        """提取 DINO 特征"""
        with torch.no_grad():
            # DINO 输出 [B, N, 384]（N 是 patch 数）
            outputs = self.teacher(x)
            # 取 CLS token 特征
            features = outputs.last_hidden_state[:, 0, :]  # [B, 384]
        return features

# ==========================================
# 蒸馏损失函数
# ==========================================
def distillation_loss(student_features, teacher_features, temperature=4.0):
    """
    简单的蒸馏损失：最小化学生和教师特征的 KL 散度
    """
    # 学生特征：[B, 256, H, W] -> [B, 256]（全局池化）
    student_pool = torch.nn.functional.adaptive_avg_pool2d(student_features, (1, 1)).flatten(1)
    
    # 投影到相同维度（384）
    student_proj = student_pool  # 这里简化处理，实际可加线性层
    teacher_feat = teacher_features  # [B, 384]
    
    # 归一化
    student_norm = torch.nn.functional.normalize(student_proj, dim=1)
    teacher_norm = torch.nn.functional.normalize(teacher_feat, dim=1)
    
    # 余弦相似度损失
    loss = 1 - (student_norm * teacher_norm).sum(dim=1).mean()
    
    return loss

# ==========================================
# 蒸馏预训练主函数
# ==========================================
def run_distillation():
    """执行蒸馏预训练"""
    
    # 配置参数
    DATA_DIR = PROJECT_ROOT / "Data" / "Merged" / "no_noise11_processed" / "images" / "train"
    OUTPUT_DIR = PROJECT_ROOT / "runs" / "distill" / "dinov3_yolo11n_pytorch"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    EPOCHS = 150
    BATCH_SIZE = 16
    IMG_SIZE = 640
    LR = 1e-4
    
    # GPU 设备配置：自动检测双卡
    gpu_count = torch.cuda.device_count()
    if gpu_count >= 2:
        DEVICE = "cuda"  # 双卡自动分布
        print(f"🚀 检测到 {gpu_count} 个 GPU，启用双卡蒸馏")
    elif gpu_count == 1:
        DEVICE = "cuda"
        print(f"⚡ 单卡蒸馏")
    else:
        DEVICE = "cpu"
        print("⚠️ 未检测到 GPU，使用 CPU 蒸馏")
    
    print("\n" + "="*60)
    print("🚀 PyTorch 原生 DINOv3 -> YOLO11n 蒸馏预训练")
    print("="*60)
    print(f"📁 数据目录: {DATA_DIR}")
    print(f"📁 输出目录: {OUTPUT_DIR}"), layer_idx=10
    print(f"📊 训练轮数: {EPOCHS}")
    print(f"📊 批次大小: {BATCH_SIZE}")
    print(f"💻 设备: {DEVICE}")
    print("="*60 + "\n")
    
    # 检查数据
    if not DATA_DIR.exists():
        print(f"❌ 数据目录不存在: {DATA_DIR}")
        sys.exit(1)
    
    # 加载模型
    print("📦 加载 YOLO11n...")
    yolo_wrapper = YOLO(str(PROJECT_ROOT / "pt" / "yolo11n.pt"))
    student = YOLO11BackboneExtractor(yolo_wrapper).to(DEVICE)
    
    # 双卡分布式
    if gpu_count >= 2:
        student = nn.DataParallel(student)
    
    print("📦 加载 DINOv3 Teacher...")
    # 注意：DINOv3 需要来自 HuggingFace，这里使用简化的加载
    # 实际可以用：teacher = DINOv3Teacher("facebook/dino-vit-tiny-16")
    teacher = None
    try:
        teacher = DINOv3Teacher("facebook/dino-vit-tiny-16").to(DEVICE)
    except Exception as e:
        print(f"⚠️ 无法加载 DINOv3: {e}")
        print("使用简化的损失函数进行预训练")
    
    # 数据加载
    print("📦 准备数据...")
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                           [0.229, 0.224, 0.225])
    ])
    
    dataset = SimpleImageDataset(DATA_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    if len(dataset) == 0:
        print(f"❌ 数据集为空: {DATA_DIR}")
        sys.exit(1)
    
    print(f"✅ 加载 {len(dataset)} 张图像")
    
    # 优化器
    optimizer = optim.AdamW(student.parameters(), lr=LR, weight_decay=0.02)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # 训练循环
    print("\n🚀 开始蒸馏预训练...")
    student.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
        
        for batch_idx, images in enumerate(pbar):
            images = images.to(DEVICE)
            
            # 学生前向
            student_features = student(images)  # [B, 384]
            
            # 如果有 Teacher，使用蒸馏损失
            if teacher is not None:
                with torch.no_grad():
                    teacher_features = teacher(images)  # [B, 384]
                # 余弦相似度损失
                loss = 1 - torch.nn.functional.cosine_similarity(student_features, teacher_features).mean()
            else:
                # 否则使用自监督损失
                loss = compute_simplified_loss(student_features)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # 学习率调度
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        
        if (epoch + 1) % 10 == 0:
            print(f"\n✅ Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")
        
        # 定期保存
        if (epoch + 1) % 50 == 0:
            checkpoint_path = OUTPUT_DIR / f"checkpoint_epoch{epoch+1}.pt"
            # 处理 DataParallel 情况
            if isinstance(student, nn.DataParallel):
                torch.save(student.module.backbone.state_dict(), checkpoint_path)
            else:
                torch.save(student.backbone.state_dict(), checkpoint_path)
            print(f"💾 保存检查点: {checkpoint_path}")
    
    # 保存最终权重
    final_weights = OUTPUT_DIR / "yolo11n_distilled.pt"
    
    
    # 加载完整 YOLO 模型
    complete_model = YOLO(str(PROJECT_ROOT / "pt" / "yolo11n.pt"))
    
    # 获取完整模型的 state_dict
    if hasattr(complete_model.model, 'model'):
        full_model = complete_model.model.model
    else:
        full_model = complete_model.model
    
    model_state = full_model.state_dict()
    
    # 映射权重：backbone 的键是 "0.weight", "1.weight" 等
    for key, val in backbone_state.items():
        if key in model_state:
            model_state[key] = val
            print(f"✓ 映射权重: {key}")
    
    # 加载回模型
    full_ backbone_state.items():
        # 在 model 中查找对应的键
        model_key = f"model.{key}"
        if model_key in model_state:
            model_state[model_key] = val
    
    complete_model.model.load_state_dict(model_state, strict=False)
    complete_model.save(str(final_weights))
    print(f"\n✅ 蒸馏预训练完成！")
    print(f"📁 权重保存在: {final_weights}")
    print(f"\n💡 使用方式：")
    print(f"   from ultraly- 用于特征向量"""
    # features: [B, 384]
    B, D = features.shape
    
    # 特征方差损失：鼓励多样化特征
    feat_var = torch.var(features, dim=0)
    var_loss = -feat_var.mean()  # 最大化方差
    
    # 特征范数损失：防止特征坍缩
    norm_loss = torch.abs(features.norm(dim=1) - 1.0).mean()
    
    return var_loss * 0.1 + norm_loss * 0.01ape(B, C, -1)  # [B, C, HW]
    
    # 计算每个通道的方差
    feat_var = torch.var(features_flat, dim=[0, 2])
    var_loss = -feat_var.mean()  # 最大化方差
    
    return var_loss * 0.1  # 权重调整

# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    try:
        run_distillation()
    except Exception as e:
        print(f"\n❌ 蒸馏预训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
