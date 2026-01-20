"""
DINO蒸馏预训练主脚本
功能：使用DINOv3作为教师模型，对YOLO11n骨干网络进行知识蒸馏
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
from modelscope import AutoModel

# 导入配置文件
from distill_config import *

# ===================== 项目根目录配置 =====================
try:
    PROJECT_ROOT = Path(__file__).parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd()

sys.path.insert(0, str(PROJECT_ROOT))
print(f"📂 项目根目录: {PROJECT_ROOT}")

# ===================== 工具函数 =====================
def find_model_config_dir(base_dir):
    """递归查找包含config.json的模型目录"""
    base_dir = Path(base_dir)
    print(f"🔍 搜索模型核心文件: {base_dir}")
    
    for config_path in base_dir.rglob("config.json"):
        model_dir = config_path.parent
        has_safetensors = (model_dir / "model.safetensors").exists()
        has_bin = (model_dir / "pytorch_model.bin").exists()
        
        if has_safetensors or has_bin:
            print(f"✅ 找到模型目录: {model_dir}")
            return model_dir
    
    print(f"❌ 未找到有效模型目录")
    return None

# ===================== 数据集类 =====================
class SimpleImageDataset(torch.utils.data.Dataset):
    """简单的图像数据集（无标签）"""
    def __init__(self, image_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.image_files = list(self.image_dir.glob("*.jpg")) + \
                          list(self.image_dir.glob("*.png")) + \
                          list(self.image_dir.glob("*.jpeg"))
        self.transform = transform
        print(f"📦 加载 {len(self.image_files)} 张图像")
    
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
            print(f"⚠️ 加载图像失败: {img_path}, 使用随机张量替代")
            return torch.randn(3, IMG_SIZE, IMG_SIZE)

# ===================== YOLO骨干网络提取器 =====================
class YOLO11BackboneExtractor(nn.Module):
    """从YOLO11完整模型中提取骨干网络并添加适配器"""
    def __init__(self, yolo_wrapper, layer_idx=BACKBONE_LAYER_IDX):
        super().__init__()
        
        # 提取YOLO11的骨干网络
        if hasattr(yolo_wrapper.model, 'model'):
            full_model = yolo_wrapper.model.model
        else:
            full_model = yolo_wrapper.model
        
        self.backbone = nn.Sequential(*list(full_model[:layer_idx]))
        
        # 添加适配器（特征图 -> 特征向量）
        self.adapter = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, ADAPTER_HIDDEN_DIM)
        )
        
        print(f"✅ 骨干网络提取成功（前{layer_idx}层）")
    
    def forward(self, x):
        feat_map = self.backbone(x)        # 特征图
        feat_vec = self.adapter(feat_map)  # 特征向量
        return feat_map, feat_vec

# ===================== DINOv3教师模型 =====================
class DINOv3Teacher(nn.Module):
    """DINOv3教师模型（ViT-L/16）- 从ModelScope直接下载"""
    def __init__(self, model_name="facebook/dinov3-vitl16-pretrain-lvd1689m"):
        super().__init__()
        
        print(f"📥 准备加载DINOv3 Teacher: {model_name}")
        
        try:
            # 直接从ModelScope下载模型，无需解压tar.gz
            self.teacher = AutoModel.from_pretrained(
                model_name, 
                trust_remote_code=True,
                device_map="auto"  # 自动分配到可用设备
            )
            self.teacher.eval()
            
            # 冻结教师模型
            for param in self.teacher.parameters():
                param.requires_grad = False
            
            print(f"✅ DINOv3 Teacher加载成功")
            
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            raise Exception(f"无法从ModelScope加载DINOv3: {e}")
    
    def forward(self, x):
        """
        前向传播 - 提取完整的 patch tokens 特征
        Args:
            x: 输入图像 [B, 3, H, W]
        Returns:
            feat_map: 特征图 [B, D, H', W']（重塑后的 patch tokens）
            feat_vec: 特征向量 [B, D]（全局平均池化）
        """
        B = x.shape[0]
        
        with torch.no_grad():
            outputs = self.teacher(pixel_values=x, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-2]  # [B, num_tokens, D]
            
            # DINOv3结构: [CLS(1)] + [Registers(4)] + [Patch Tokens(N-5)]
            # 跳过 CLS 和 registers，提取空间 patch tokens
            num_registers = 4
            spatial_tokens = last_hidden[:, 1 + num_registers:, :]  # [B, num_patches, D]
            
            # 获取特征维度
            D = spatial_tokens.shape[-1]
            num_patches = spatial_tokens.shape[1]
            
            # 计算特征图的空间尺寸 (num_patches = H' * W')
            # DINOv3 使用 14x14 patch，所以对于 640x640 输入会得到 ~45x45
            H_prime = int(num_patches ** 0.5)
            W_prime = H_prime if (H_prime * H_prime == num_patches) else int(num_patches ** 0.5) + 1
            
            # 将 tokens 重塑为特征图 [B, D, H', W']
            feat_map = spatial_tokens.reshape(B, H_prime, W_prime, D).permute(0, 3, 1, 2)  
            
            # 全局平均池化得到特征向量 [B, D]
            feat_vec = spatial_tokens.mean(dim=1)  # [B, D]
            
            # 为了与学生模型维度匹配，可能需要适配
            # 学生模型的 feat_vec 经过线性层后是 [B, 1024]
            # 这里我们保持原始维度，在损失计算时处理维度对齐
        
        return feat_map, feat_vec

# ===================== 损失函数 =====================
def compute_distill_loss(student_vec, teacher_vec, student_map, teacher_map=None):
    """
    计算蒸馏损失（同时考虑特征向量和特征图）
    student_vec: 学生模型特征向量 [B, D_s]
    teacher_vec: 教师模型特征向量 [B, D_t]
    student_map: 学生模型特征图 [B, C_s, H_s, W_s]
    teacher_map: 教师模型特征图 [B, C_t, H_t, W_t]（可选）
    """
    
    # 1. 特征向量相似度损失
    # 处理维度不匹配：将高维投影到低维
    D_s = student_vec.shape[-1]
    D_t = teacher_vec.shape[-1]
    
    if D_s != D_t:
        # 简单的投影方案：截断或填充
        if D_s > D_t:
            student_vec_aligned = student_vec[:, :D_t]
        else:
            # 学生维度更小，用均值填充
            padding = torch.zeros(student_vec.shape[0], D_t - D_s, device=student_vec.device)
            student_vec_aligned = torch.cat([student_vec, padding], dim=1)
    else:
        student_vec_aligned = student_vec
    
    # 计算余弦相似度
    cos_sim = torch.nn.functional.cosine_similarity(student_vec_aligned, teacher_vec, dim=1).mean()
    vec_loss = (1 - cos_sim) * DISTILL_LOSS_WEIGHT
    
    # 2. 特征图相似度损失（可选）
    if teacher_map is not None:
        # 将特征图拉平为向量进行比较
        B, C_s, H_s, W_s = student_map.shape
        _, C_t, H_t, W_t = teacher_map.shape
        
        # 调整到相同尺寸
        if (H_s, W_s) != (H_t, W_t):
            teacher_map_resized = torch.nn.functional.interpolate(
                teacher_map, size=(H_s, W_s), mode='bilinear', align_corners=False
            )
        else:
            teacher_map_resized = teacher_map
        
        # 特征图匹配损失：计算特征图的相似度
        student_map_flat = student_map.reshape(B, C_s, -1)  # [B, C, HW]
        teacher_map_flat = teacher_map_resized.reshape(B, C_t, -1)  # [B, C, HW]
        
        # 计算每个像素点的特征相似度
        sim = torch.nn.functional.cosine_similarity(student_map_flat, teacher_map_flat, dim=1)  # [B, HW]
        map_loss = (1 - sim.mean()) * 0.5 * DISTILL_LOSS_WEIGHT
    else:
        map_loss = 0
    
    # 3. 特征多样性损失（鼓励特征多样性）
    feat_flat = student_map.reshape(B, student_map.shape[1], -1)
    var_loss = -torch.var(feat_flat, dim=[0, 2]).mean() * VAR_LOSS_WEIGHT
    
    total_loss = vec_loss + map_loss + var_loss
    
    return total_loss

def compute_simplified_loss(student_vec, student_map):
    """简化损失（无教师模型时使用）"""
    B, D = student_vec.shape
    
    # 特征方差损失
    vec_var = torch.var(student_vec, dim=0).mean()
    var_loss = -vec_var * VAR_LOSS_WEIGHT
    
    # 归一化损失
    norm_loss = torch.abs(student_vec.norm(dim=1) - 1.0).mean() * NORM_LOSS_WEIGHT
    
    return var_loss + norm_loss

# ===================== 主训练流程 =====================
def run_distillation():
    """执行蒸馏预训练"""
    
    # 路径配置
    data_dir = Path(DISTILL_DATA_DIR)
    output_dir = Path(OUTPUT_DIR)
    yolo_weights = Path(YOLO11N_WEIGHTS)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("🚀 DINO → YOLO11n 蒸馏预训练")
    print("="*60)
    print(f"📁 数据目录: {data_dir}")
    print(f"📁 输出目录: {output_dir}")
    print(f"📊 训练轮数: {EPOCHS}")
    print(f"📊 批次大小: {BATCH_SIZE}")
    print(f"💻 设备: {DEVICE}")
    print(f"📦 YOLO权重: {yolo_weights}")
    print("="*60 + "\n")
    
    # 检查数据目录
    if not data_dir.exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        print(f"💡 请先运行: python prepare_distill_data.py")
        sys.exit(1)
    
    # 检查YOLO权重
    if not yolo_weights.exists():
        print(f"⚠️ YOLO权重不存在，正在下载...")
        from ultralytics import YOLO
        yolo_temp = YOLO("yolo11n.pt")
        yolo_weights.parent.mkdir(parents=True, exist_ok=True)
        yolo_temp.save(str(yolo_weights))
    
    # 加载YOLO骨干网络
    print("📦 加载YOLO11n骨干网络...")
    from ultralytics import YOLO
    yolo_wrapper = YOLO(str(yolo_weights))
    student = YOLO11BackboneExtractor(yolo_wrapper, layer_idx=BACKBONE_LAYER_IDX).to(DEVICE)
    
    # 多GPU并行
    gpu_count = torch.cuda.device_count()
    if gpu_count >= 2 and USE_DATAPARALLEL:
        student = nn.DataParallel(student)
        print(f"🚀 启用双卡/多卡并行 ({gpu_count} GPUs)")
    
    # 加载教师模型
    print("📦 加载DINOv3 Teacher...")
    teacher = None
    try:
        teacher = DINOv3Teacher().to(DEVICE)
        print("✅ 教师模型加载成功")
    except Exception as e:
        print(f"⚠️ 教师模型加载失败: {e}")
        print("将使用简化损失函数进行预训练")
    
    # 准备数据
    print("📦 准备数据...")
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
    ])
    
    dataset = SimpleImageDataset(data_dir, transform=transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE == "cuda" else False
    )
    
    if len(dataset) == 0:
        print(f"❌ 数据集为空: {data_dir}")
        sys.exit(1)
    
    print(f"✅ 加载 {len(dataset)} 张图像")
    
    # 优化器和学习率调度器
    optimizer = optim.AdamW(student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # 开始训练
    print("\n🚀 开始蒸馏训练...")
    student.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
        
        for batch_idx, images in enumerate(pbar):
            # 调试模式：只训练少量batch
            if DEBUG_MODE and batch_idx >= DEBUG_MAX_BATCHES:
                break
            
            images = images.to(DEVICE)
            
            # 前向传播
            student_map, student_vec = student(images)
            
            # 计算损失
            if teacher is not None:
                with torch.no_grad():
                    teacher_map, teacher_vec = teacher(images)
                loss = compute_distill_loss(student_vec, teacher_vec, student_map, teacher_map)
            else:
                loss = compute_simplified_loss(student_vec, student_map)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        
        # 每10轮输出一次
        if (epoch + 1) % 10 == 0:
            print(f"\n✅ Epoch {epoch+1}/{EPOCHS} - Avg Loss: {avg_loss:.4f}")
        
        # 保存检查点
        if (epoch + 1) % SAVE_CHECKPOINT_INTERVAL == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch{epoch+1}.pt"
            if isinstance(student, nn.DataParallel):
                torch.save(student.module.backbone.state_dict(), checkpoint_path)
            else:
                torch.save(student.backbone.state_dict(), checkpoint_path)
            print(f"💾 保存检查点: {checkpoint_path}")
    
    # 保存最终权重
    if SAVE_FINAL_WEIGHTS:
        print("\n🔄 合并蒸馏权重到完整YOLO模型...")
        final_weights = output_dir / "yolo11n_distilled.pt"
        
        # 提取骨干网络权重
        if isinstance(student, nn.DataParallel):
            backbone_state = student.module.backbone.state_dict()
        else:
            backbone_state = student.backbone.state_dict()
        
        # 加载完整YOLO模型
        complete_model = YOLO(str(yolo_weights))
        if hasattr(complete_model.model, 'model'):
            full_model = complete_model.model.model
        else:
            full_model = complete_model.model
        
        model_state = full_model.state_dict()
        
        # 映射权重
        print("🔄 映射蒸馏权重...")
        mapped_count = 0
        for key, val in backbone_state.items():
            if key in model_state:
                model_state[key] = val
                mapped_count += 1
        
        print(f"✓ 成功映射 {mapped_count} 个权重层")
        
        # 保存
        full_model.load_state_dict(model_state, strict=False)
        complete_model.save(str(final_weights))
        
        print(f"\n✅ 蒸馏预训练完成！")
        print(f"📁 完整权重保存: {final_weights}")
        print(f"\n💡 下一步：")
        print(f"   python train.py  # 自动加载蒸馏权重进行检测训练")

if __name__ == "__main__":
    try:
        run_distillation()
    except Exception as e:
        print(f"\n❌ 蒸馏失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
