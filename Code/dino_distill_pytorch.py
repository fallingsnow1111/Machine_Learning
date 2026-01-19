"""
PyTorch 原生 DINOv3 -> YOLO11n 知识蒸馏预训练脚本
使用 Kaggle vitl16 作为教师模型
"""

import sys
import os
import tarfile
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image

# ===================== 核心：还原项目根目录（代码仓目录，其他路径均基于此） =====================
# 兼容Jupyter环境（__file__不存在）和普通Python脚本运行
try:
    # 普通Python脚本：获取当前文件的父级父级（代码仓的路径结构）
    PROJECT_ROOT = Path(__file__).parent.parent
except NameError:
    # Jupyter环境：使用当前工作目录作为代码仓根目录
    PROJECT_ROOT = Path.cwd()

sys.path.insert(0, str(PROJECT_ROOT))
print(f"📂 项目代码仓根目录: {PROJECT_ROOT}")

# ===================== 仅修改这里：ViT-L/16模型路径配置（独立于代码仓） =====================
DINO_TAR_PATH = Path("/mnt/workspace/dinov3-vitl16.tar.gz")  # 上传的模型压缩包路径
DINO_EXTRACT_DIR = Path("/mnt/workspace/dinov3-vitl16")  # 模型解压目录

# ===================== 解压函数（仅用于ViT模型） =====================
def extract_tar_gz(tar_path, extract_dir):
    """
    解压.tar.gz文件到指定目录（仅用于ViT模型，不影响代码仓其他文件）
    :param tar_path: .tar.gz压缩包路径
    :param extract_dir: 解压目标目录
    """
    extract_dir = Path(extract_dir)
    if extract_dir.exists():
        print(f"✅ ViT模型解压目录已存在，无需重复解压: {extract_dir}")
        return True
    
    # 检查压缩包是否存在
    if not tar_path.exists():
        print(f"❌ ViT模型压缩包不存在: {tar_path}")
        return False
    
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            # 显示解压进度
            members = tar.getmembers()
            for member in tqdm(members, desc=f"解压 ViT模型 {tar_path.name}"):
                tar.extract(member, extract_dir)
        print(f"✅ ViT模型解压完成: {extract_dir}")
        return True
    except Exception as e:
        print(f"❌ ViT模型解压失败: {e}")
        return False

from ultralytics import YOLO
from modelscope import AutoModel

class SimpleImageDataset(torch.utils.data.Dataset):
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
        except:
            return torch.randn(3, 640, 640)

class YOLO11BackboneExtractor(nn.Module):
    def __init__(self, yolo_wrapper, layer_idx=10):
        super().__init__()
        if hasattr(yolo_wrapper.model, 'model'):
            full_model = yolo_wrapper.model.model
        else:
            full_model = yolo_wrapper.model
        
        self.backbone = nn.Sequential(*list(full_model[:layer_idx]))
        self.adapter = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 1024)
        )
    
    def forward(self, x):
        feat_map = self.backbone(x)
        feat_vec = self.adapter(feat_map)
        return feat_map, feat_vec

class DINOv3Teacher(nn.Module):
    def __init__(self, tar_path=DINO_TAR_PATH, extract_dir=DINO_EXTRACT_DIR):
        super().__init__()
        # ===================== 仅修改：加载独立配置的ViT模型路径 =====================
        print(f"📥 准备加载ViT-L/16模型，压缩包路径: {tar_path}")
        
        # 先解压ViT模型（解压到独立目录，不放入代码仓）
        if not extract_tar_gz(tar_path, extract_dir):
            raise Exception("ViT模型压缩包解压失败，无法加载DINOv3 Teacher")
        
        # 解压后的模型路径（用于AutoModel加载）
        model_path = extract_dir
        print(f"📥 开始加载 DINOv3 Teacher: {model_path}")
        
        self.teacher = AutoModel.from_pretrained(str(model_path), trust_remote_code=True)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        with torch.no_grad():
            outputs = self.teacher(pixel_values=x, output_hidden_states=True)
            features = outputs.hidden_states[-1][:, 0, :]
        return features

def compute_distill_loss(student_vec, teacher_vec, student_map):
    cos_sim = torch.nn.functional.cosine_similarity(student_vec, teacher_vec).mean()
    distill_loss = 1 - cos_sim
    
    B, C, H, W = student_map.shape
    feat_flat = student_map.reshape(B, C, -1)
    var_loss = -torch.var(feat_flat, dim=[0, 2]).mean()
    
    return distill_loss + 0.1 * var_loss

def compute_simplified_loss(student_vec, student_map):
    B, D = student_vec.shape
    vec_var = torch.var(student_vec, dim=0).mean()
    var_loss = -vec_var
    norm_loss = torch.abs(student_vec.norm(dim=1) - 1.0).mean()
    return var_loss * 0.1 + norm_loss * 0.01

def run_distillation():
    # ===================== 完全保留：代码仓内的原有路径逻辑 =====================
    DATA_DIR = PROJECT_ROOT / "Data" / "Merged" / "no_noise11_processed" / "images" / "train"
    OUTPUT_DIR = PROJECT_ROOT / "runs" / "distill" / "dinov3_yolo11n_pytorch"
    YOLO11N_PATH = PROJECT_ROOT / "pt" / "yolo11n.pt"  # YOLO权重依然在代码仓内
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    EPOCHS = 150
    BATCH_SIZE = 16
    IMG_SIZE = 640
    LR = 1e-4
    
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
    print(f"📊 训练轮数: {EPOCHS}")
    print(f"📊 批次大小: {BATCH_SIZE}")
    print(f"💻 设备: {DEVICE}")
    print(f"📦 YOLO11n路径（代码仓内）: {YOLO11N_PATH}")
    print(f"📦 ViT-L/16模型路径（独立）: {DINO_EXTRACT_DIR}")
    print("="*60 + "\n")
    
    # 检查代码仓内的数据目录是否存在
    if not DATA_DIR.exists():
        print(f"❌ 数据目录（代码仓内）不存在: {DATA_DIR}")
        print("💡 请确保数据存放路径正确，或创建对应目录")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        sys.exit(1)
    
    # 检查代码仓内的YOLO11n权重是否存在
    if not YOLO11N_PATH.exists():
        print(f"⚠️ YOLO11n权重（代码仓内）不存在: {YOLO11N_PATH}")
        print("💡 正在自动下载yolo11n.pt到代码仓pt目录...")
        YOLO11N_PATH.parent.mkdir(parents=True, exist_ok=True)
        yolo_temp = YOLO("yolo11n.pt")
        yolo_temp.save(str(YOLO11N_PATH))
    
    print("📦 加载 YOLO11n（代码仓内权重）...")
    yolo_wrapper = YOLO(str(YOLO11N_PATH))
    student = YOLO11BackboneExtractor(yolo_wrapper, layer_idx=10).to(DEVICE)
    
    if gpu_count >= 2:
        student = nn.DataParallel(student)
    
    print("📦 加载 DINOv3 Teacher（独立ViT模型）...")
    teacher = None
    try:
        # 加载独立路径的ViT模型，不影响代码仓其他逻辑
        teacher = DINOv3Teacher().to(DEVICE)
        print("✅ DINOv3 vitl16 Teacher 加载成功")
    except Exception as e:
        print(f"⚠️ 无法加载 DINOv3: {e}")
        print("使用简化的损失函数进行预训练")
    
    print("📦 准备数据（代码仓内数据）...")
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = SimpleImageDataset(DATA_DIR, transform=transform)
    # Jupyter环境适配：num_workers=0，避免多进程报错
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    if len(dataset) == 0:
        print(f"❌ 数据集（代码仓内）为空: {DATA_DIR}")
        print("💡 请放入图像数据后再运行")
        sys.exit(1)
    
    print(f"✅ 加载 {len(dataset)} 张图像（代码仓内数据）")
    
    optimizer = optim.AdamW(student.parameters(), lr=LR, weight_decay=0.02)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    print("\n🚀 开始蒸馏预训练...")
    student.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
        
        for batch_idx, images in enumerate(pbar):
            images = images.to(DEVICE)
            
            student_map, student_vec = student(images)
            
            if teacher is not None:
                with torch.no_grad():
                    teacher_vec = teacher(images)
                loss = compute_distill_loss(student_vec, teacher_vec, student_map)
            else:
                loss = compute_simplified_loss(student_vec, student_map)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        
        if (epoch + 1) % 10 == 0:
            print(f"\n✅ Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")
        
        if (epoch + 1) % 50 == 0:
            checkpoint_path = OUTPUT_DIR / f"checkpoint_epoch{epoch+1}.pt"
            if isinstance(student, nn.DataParallel):
                torch.save(student.module.backbone.state_dict(), checkpoint_path)
            else:
                torch.save(student.backbone.state_dict(), checkpoint_path)
            print(f"💾 保存检查点（代码仓内）: {checkpoint_path}")
    
    final_weights = OUTPUT_DIR / "yolo11n_distilled.pt"
    
    if isinstance(student, nn.DataParallel):
        backbone_state = student.module.backbone.state_dict()
    else:
        backbone_state = student.backbone.state_dict()
    
    complete_model = YOLO(str(YOLO11N_PATH))
    
    if hasattr(complete_model.model, 'model'):
        full_model = complete_model.model.model
    else:
        full_model = complete_model.model
    
    model_state = full_model.state_dict()
    
    print("\n🔄 映射蒸馏权重到完整模型（代码仓内）...")
    for key, val in backbone_state.items():
        if key in model_state:
            model_state[key] = val
            print(f"✓ 映射权重: {key}")
    
    full_model.load_state_dict(model_state, strict=False)
    complete_model.save(str(final_weights))
    
    print(f"\n✅ 蒸馏预训练完成！")
    print(f"📁 权重保存在（代码仓内）: {final_weights}")
    print(f"\n💡 使用方式：")
    print(f"   python Code/train_yolo11.py")
    print(f"   (自动检测并加载代码仓内的蒸馏权重)")

if __name__ == "__main__":
    try:
        run_distillation()
    except Exception as e:
        print(f"\n❌ 蒸馏预训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
