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
import tarfile
import tarfile
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image

# ==========================================
# 路径配置
# ==========================================
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print(f"📂 项目根目录: {PROJECT_ROOT}")

from ultralytics import YOLO
from modelscope import AutoModel
import os

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
        
        # 自动对齐维度：YOLO11n 出口通常是 256，DINO-vitl16 是 1024
        self.adapter = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 1024)  # 对应 dino-vitl16  # 对应 dino-vitl16
            nn.Linear(256, 1024)  # 对应 dino-vitl16
        )
    
    def forward(self, x):
        """返回特征图和对齐后的特征向量"""
        feat_map = self.backbone(x)  # [B, 256, H, W]
        feat_vec = self.adapter(feat_map)  # [B, 1024]
        return feat_map, feat_vec

# ==========================================
# DINOv3 Teacher 模型
# ==========================================
class DINOv3Teacher(nn.Module):
    """DINOv3 ViT-L/16 作为 Teacher"""
    def __init__(self, model_path=None):
        super().__init__()
        # 智能路径检测
        if model_path is None:
            # Kaggle vitl16 路径
            kaggle_path = '/kaggle/input/dinov3-vitl16/pytorch/default/1/dinov3-vitl16/facebook/dinov3-vitl16-pretrain-lvd1689m'
            if os.path.exists(kaggle_path):
                model_path = kaggle_path
                print(f"📥 加载 Kaggle DINOv3 Teacher: {model_path}")
            else:
                # 备选路径
                model_path = '/kaggle/input/dinov3-vitl16/facebook/dinov3-vitl16'
                print(f"📥 加载 DINOv3 Teacher (备选): {model_path}")
        else:
            print(f"📥 加载自定义路径 DINOv3 Teacher: {model_path}")
        
        from modelscope import AutoModel
        self.teacher = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
    1024] 学生特征向量
    teacher_vec: [B, 102):
        """提取 DINO 特征"""
        with torch.no_grad():
            outputs = self.teacher(pixel_values=x, output_hidden_states=True)
            features = outputs.hidden_states[-1][:, 0, :]  # [B, 1024] CLS token
        return features

# ==========================================
# 蒸馏损失函数
# ==========================================
def compute_distill_loss(student_vec, teacher_vec, student_map):
    """
    计算蒸馏损失
    student_vec: [B, 384] 学生特征向量
    teacher_vec: [B, 384] 教师特征向量
    student_map: [B, 256, H, W] 学生特征图
    """
    # 1. 余弦相似度损失（主要蒸馏项）
    cos_sim = torch.nn.functional.cosine_similarity(student_vec, teacher_vec).mean()
    distill_loss = 1 - cos_sim
    
    # 2. 特征多样性损失（防止特征坍缩）
    B, C, H, W = student_map.shape
    feat_flat = student_map.reshape(B, C, -1)
    var_loss = -torch.var(feat_flat, dim=[0, 2]).mean()
    
    return distill_loss + 0.1 * var_loss

def compute_simplified_loss(student_vec, student_map):
    """简化的自监督损失（无需 Teacher）"""
    # 特征向量的方差损失
    B, D = student_vec.shape
    vec_var = torch.var(student_vec, dim=0).mean()
    var_loss = -vec_var
    
    # 特征范数损失（防止特征坍塌）
    norm_loss = torch.abs(student_vec.norm(dim=1) - 1.0).mean()
    
    return var_loss * 0.1 + norm_loss * 0.01

# ==========================================
# 蒸馏预训练主函数
# ==========================================
def run_distillation():
    """执行蒸馏预训练"""
    
    # 配置参数
    DATA_DIR = PROJECT_ROOT / "Data" / "Merged" / "no_noise11_processed" / "images" / "train"
    OUTPUT_DIR = PROJECT_ROOT / "runs" / "distill" / "dinov3_yolo11n_pytorch"
    YOLO11N_PATH = PROJECT_ROOT / "pt" / "yolo11n.pt"  # YOLO权重依然在代码仓内
    YOLO11N_PATH = PROJECT_ROOT / "pt" / "yolo11n.pt"  # YOLO权重依然在代码仓内
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    EPOCHS = 150
    BATCH_SIZE = 16
    IMG_SIZE = 640
    LR = 1e-4
    
    # GPU 设备配置：自动检测双卡
    gpu_count = torch.cuda.device_count()
    if gpu_count >= 2:
        DEVICE = "cuda"
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
    print(f"📁 数据目录（代码仓内）: {DATA_DIR}")
    print(f"📁 输出目录（代码仓内）: {OUTPUT_DIR}")
    print(f"📁 数据目录（代码仓内）: {DATA_DIR}")
    print(f"📁 输出目录（代码仓内）: {OUTPUT_DIR}")
    print(f"📊 训练轮数: {EPOCHS}")
    print(f"📊 批次大小: {BATCH_SIZE}")
    print(f"💻 设备: {DEVICE}")
    print(f"📦 YOLO11n路径（代码仓内）: {YOLO11N_PATH}")
    print(f"📦 ViT-L/16解压目录（独立）: {DINO_EXTRACT_DIR}")
    print(f"📦 ViT-L/16模型目录（深层）: {DINO_MODEL_DIR}")
    print(f"📦 YOLO11n路径（代码仓内）: {YOLO11N_PATH}")
    print(f"📦 ViT-L/16解压目录（独立）: {DINO_EXTRACT_DIR}")
    print(f"📦 ViT-L/16模型目录（深层）: {DINO_MODEL_DIR}")
    print("="*60 + "\n")
    
    # 检查数据
    if not DATA_DIR.exists():
        print(f"❌ 数据目录（代码仓内）不存在: {DATA_DIR}")
        print("💡 请确保数据存放路径正确，或创建对应目录")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"❌ 数据目录（代码仓内）不存在: {DATA_DIR}")
        print("💡 请确保数据存放路径正确，或创建对应目录")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        sys.exit(1)
    
    # 加载模型
    print("📦 加载 YOLO11n...")
    yolo_wrapper = YOLO(str(PROJECT_ROOT / "pt" / "yolo11n.pt"))
    student = YOLO11BackboneExtractor(yolo_wrapper, layer_idx=10).to(DEVICE)
    
    # 双卡分布式
    if gpu_count >= 2:
        student = nn.DataParallel(student)
    
    print("📦 加载 DINOv3 Teacher（独立ViT模型）...")
    print("📦 加载 DINOv3 Teacher（独立ViT模型）...")
    teacher = None
    try:
        # 加载独立路径的ViT模型，不影响代码仓其他逻辑
        teacher = DINOv3Teacher().to(DEVICE)
        print("✅ DINOv3 vitl16 Teacher 加载成功")
        # 加载独立路径的ViT模型，不影响代码仓其他逻辑
        teacher = DINOv3Teacher().to(DEVICE)
        print("✅ DINOv3 vitl16 Teacher 加载成功")
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
    # Jupyter环境适配：num_workers=0，避免多进程报错
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    # Jupyter环境适配：num_workers=0，避免多进程报错
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    if len(dataset) == 0:
        print(f"❌ 数据集（代码仓内）为空: {DATA_DIR}")
        print("💡 请放入图像数据后再运行")
        print(f"❌ 数据集（代码仓内）为空: {DATA_DIR}")
        print("💡 请放入图像数据后再运行")
        sys.exit(1)
    
    print(f"✅ 加载 {len(dataset)} 张图像（代码仓内数据）")
    print(f"✅ 加载 {len(dataset)} 张图像（代码仓内数据）")
    
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
            student_map, student_vec = student(images)
            
            # 如果有 Teacher，使用蒸馏损失
            if teacher is not None:
                with torch.no_grad():
                    teacher_vec = teacher(images)
                loss = compute_distill_loss(student_vec, teacher_vec, student_map)
            else:
                # 否则使用自监督损失
                loss = compute_simplified_loss(student_vec, student_map)
            
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
            if isinstance(student, nn.DataParallel):
                torch.save(student.module.backbone.state_dict(), checkpoint_path)
            else:
                torch.save(student.backbone.state_dict(), checkpoint_path)
            print(f"💾 保存检查点（代码仓内）: {checkpoint_path}")
            print(f"💾 保存检查点（代码仓内）: {checkpoint_path}")
    
    # 保存最终权重
    final_weights = OUTPUT_DIR / "yolo11n_distilled.pt"
    
    # 获取 backbone 权重
    if isinstance(student, nn.DataParallel):
        backbone_state = student.module.backbone.state_dict()
    else:
        backbone_state = student.backbone.state_dict()
    
    # 加载完整 YOLO 模型
    complete_model = YOLO(str(PROJECT_ROOT / "pt" / "yolo11n.pt"))
    
    # 获取完整模型的 state_dict
    if hasattr(complete_model.model, 'model'):
        full_model = complete_model.model.model
    else:
        full_model = complete_model.model
    
    model_state = full_model.state_dict()
    
    # 映射权重：backbone 的键是 "0.weight", "1.weight" 等
    print("\n🔄 映射蒸馏权重到完整模型...")
    for key, val in backbone_state.items():
        if key in model_state:
            model_state[key] = val
            print(f"✓ 映射权重: {key}")
    
    # 加载回模型
    full_model.load_state_dict(model_state, strict=False)
    complete_model.save(str(final_weights))
    
    print(f"\n✅ 蒸馏预训练完成！")
    print(f"📁 权重保存在（代码仓内）: {final_weights}")
    print(f"📁 权重保存在（代码仓内）: {final_weights}")
    print(f"\n💡 使用方式：")
    print(f"   python Code/train_yolo11.py")
    print(f"   (自动检测并加载蒸馏权重)")

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
