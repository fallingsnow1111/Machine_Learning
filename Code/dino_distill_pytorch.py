import sys
import os
import tarfile
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO
from modelscope import AutoModel

# ===================== 📡 自动平台检测 =====================
def detect_platform():
    if os.path.exists('/kaggle/working'): return "KAGGLE"
    if os.path.exists('/mnt/workspace'): return "ALIYUN"
    return "LOCAL"

PLATFORM = detect_platform()

# ===================== ⚙️ 配置类 =====================
class Config:
    def __init__(self, mode):
        self.mode = mode
        try:
            self.project_root = Path(__file__).parent.parent
        except NameError:
            self.project_root = Path.cwd()
        
        self.relative_data_path = "Data/Merged/mixed_processed"
        self.data_dir = self.project_root / self.relative_data_path
        
        self.epochs = 150
        self.batch_size = 8  # 🚀 建议调小一点，防止中层特征蒸馏 OOM
        self.img_size = 640
        self.lr = 1e-4
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 权重配比：空间特征（细节）对检测更重要
        self.lambda_global = 1.0  
        self.lambda_spatial = 5.0 # 🚀 MSE 通常数值较小，适当拉高权重

        if self.mode == "KAGGLE":
            self.output_dir = Path("/kaggle/working/runs/distill")
            self.yolo_pt_path = Path("/kaggle/working/yolo11n.pt")
            self.dino_model_path = Path("/kaggle/input/dinov3-vitl16/pytorch/default/1/dinov3-vitl16/facebook/dinov3-vitl16-pretrain-lvd1689m")
            self.dino_needs_extract = False
        else:
            self.output_dir = self.project_root / "runs/distill"
            self.yolo_pt_path = self.project_root / "pt/yolo11n.pt"
            self.dino_needs_extract = True
            self.dino_tar_path = Path("/mnt/workspace/dinov3-vitl16.tar.gz")
            self.dino_extract_dir = Path("/mnt/workspace/dinov3-vitl16")
            self.dino_model_path = None

    def check_env(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.yolo_pt_path.exists():
            YOLO("yolo11n.pt").save(str(self.yolo_pt_path))

cfg = Config(PLATFORM)

# ===================== 🧩 模型定义 =====================

class YOLO11Distiller(nn.Module):
    def __init__(self, yolo_path, layer_idx=10):
        super().__init__()
        yolo = YOLO(str(yolo_path))
        model_obj = yolo.model.model if hasattr(yolo.model, 'model') else yolo.model
        
        self.backbone = nn.Sequential(*list(model_obj[:layer_idx]))
        
        # 🚀 重点：适配器用于将 YOLO 的 256 通道“翻译”给 DINO 看
        self.spatial_adapter = nn.Conv2d(256, 1024, kernel_size=1)
        self.global_adapter = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 1024)
        )
    
    def forward(self, x):
        feat_map = self.backbone(x) 
        spatial_feat = self.spatial_adapter(feat_map)
        # 🚀 归一化特征，防止 MSE 损失炸开
        spatial_feat = F.normalize(spatial_feat, p=2, dim=1)
        global_feat = self.global_adapter(feat_map)
        global_feat = F.normalize(global_feat, p=2, dim=1)
        return spatial_feat, global_feat

class DINOv3Teacher(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 此处省略你之前的 extract_tar_gz 和 find_config_dir 函数逻辑
        path = config.dino_model_path if config.mode == "KAGGLE" else config.dino_extract_dir
        
        print(f"📥 正在加载教师模型: {path}")
        self.teacher = AutoModel.from_pretrained(str(path), trust_remote_code=True)
        self.teacher.eval()
        for p in self.teacher.parameters(): p.requires_grad = False
    
    def forward(self, x):
        with torch.no_grad():
            outputs = self.teacher(pixel_values=x, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1] 
            
            # 全局特征 (CLS)
            global_feat = F.normalize(last_hidden_state[:, 0, :], p=2, dim=1)
            
            # 空间特征 (Patch Tokens)
            patch_tokens = last_hidden_state[:, 1:, :] 
            b, n, c = patch_tokens.shape
            grid_size = int(n**0.5)
            spatial_feat = patch_tokens.transpose(1, 2).reshape(b, c, grid_size, grid_size)
            spatial_feat = F.normalize(spatial_feat, p=2, dim=1)
            
            return spatial_feat, global_feat

# ===================== 🚀 训练逻辑 =====================

def run():
    cfg.check_env()
    
    teacher = DINOv3Teacher(cfg).to(cfg.device)
    student = YOLO11Distiller(cfg.yolo_pt_path).to(cfg.device)
    
    # 🚀 使用 AdamW 并对适配器和 Backbone 统一优化
    optimizer = optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=0.01)
    
    dataset = DataLoader(SimpleImageDataset(cfg.data_dir, transform=... ), batch_size=cfg.batch_size, shuffle=True)

    print("\n🔥 开始中层特征对齐蒸馏...")
    student.train()
    for epoch in range(cfg.epochs):
        loop = tqdm(dataset, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        for img in loop:
            img = img.to(cfg.device)
            
            # 🚀 建议使用混合精度训练 (AMP) 节省显存
            with torch.cuda.amp.autocast():
                s_spatial, s_global = student(img)
                t_spatial, t_global = teacher(img)
                
                # 确保尺寸一致
                if s_spatial.shape[-2:] != t_spatial.shape[-2:]:
                    s_spatial = F.interpolate(s_spatial, size=t_spatial.shape[-2:], mode='bilinear')
                
                loss_g = 1 - F.cosine_similarity(s_global, t_global).mean()
                loss_s = F.mse_loss(s_spatial, t_spatial)
                loss = (cfg.lambda_global * loss_g) + (cfg.lambda_spatial * loss_s)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loop.set_postfix(loss=f"{loss.item():.4f}", s_loss=f"{loss_s.item():.4f}")

    # ===================== 💾 关键：保存逻辑修正 =====================
    final_path = cfg.output_dir / "yolo11n_distilled.pt"
    # 我们只提取 backbone 的权重，忽略适配器
    pure_backbone_state = student.backbone.state_dict()
    
    full_yolo = YOLO(str(cfg.yolo_pt_path))
    # 注入权重
    full_yolo.model.model[:10].load_state_dict(pure_backbone_state)
    full_yolo.save(str(final_path))
    print(f"🎉 蒸馏后的骨干网络已成功注入并保存至: {final_path}")

if __name__ == "__main__":
    run()