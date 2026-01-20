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

# ===================== 🛠️ 核心：手动挂载本地源码 =====================
try:
    project_root = Path(__file__).parent.parent.absolute()
except NameError:
    project_root = Path("/mnt/workspace/Machine_Learning")

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from ultralytics import YOLO
    print("✅ 成功加载本地 ultralytics 模块")
except ImportError:
    print(f"❌ 仍然找不到 ultralytics。请确认路径: {project_root}")
    sys.exit(1)

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
        self.project_root = project_root
        
        # 修正：将路径指向图片所在的 train 文件夹
        self.data_dir = self.project_root / "Data/Raw/mixed_processed"
        
        self.epochs = 50  # 特征蒸馏通常不需要150轮，50轮效果就很好了
        self.batch_size = 8 
        self.img_size = 640  # 保持16的整数倍，适配DINOv3 patch size=16
        self.lr = 1e-4
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.lambda_global = 1.0  
        self.lambda_spatial = 5.0 

        if self.mode == "KAGGLE":
            self.output_dir = Path("/kaggle/working/runs/distill")
            self.yolo_pt_path = Path("/kaggle/working/yolo11n.pt")
            self.dino_path = Path("/kaggle/input/dinov3-vitl16/pytorch/default/1/dinov3-vitl16/facebook/dinov3-vitl16-pretrain-lvd1689m")
            self.dino_needs_extract = False
        else:
            self.output_dir = self.project_root / "runs/distill"
            self.yolo_pt_path = self.project_root / "pt/yolo11n.pt"
            self.dino_needs_extract = True
            self.dino_tar_path = Path("/mnt/workspace/dinov3-vitl16.tar.gz")
            self.dino_path = Path("/mnt/workspace/dinov3-vitl16")

    def check_env(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.yolo_pt_path.exists():
            YOLO("yolo11n.pt").save(str(self.yolo_pt_path))
        
        # 阿里云环境下自动解压
        if self.dino_needs_extract and not self.dino_path.exists():
            print(f"⏳ 正在解压 DINOv3 到 {self.dino_path}...")
            with tarfile.open(self.dino_tar_path, 'r:gz') as tar:
                tar.extractall(path="/mnt/workspace/")

cfg = Config(PLATFORM)

# ===================== 🖼️ 数据集类 =====================
class SimpleImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, transform=None):
        self.files = sorted(list(Path(image_dir).rglob("*.jpg")) + 
                            list(Path(image_dir).rglob("*.png")))
        self.transform = transform
        if len(self.files) == 0:
            print(f"⚠️ 警告：目录 {image_dir} 下没发现图片！")

    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        return self.transform(img) if self.transform else img

# ===================== 🧩 模型定义 =====================

class YOLO11Distiller(nn.Module):
    def __init__(self, yolo_path, layer_idx=10):
        super().__init__()
        yolo = YOLO(str(yolo_path))
        model_obj = yolo.model.model if hasattr(yolo.model, 'model') else yolo.model
        self.backbone = nn.Sequential(*list(model_obj[:layer_idx]))
        self.spatial_adapter = nn.Conv2d(256, 1024, kernel_size=1)
        self.global_adapter = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(256, 1024)
        )
    
    def forward(self, x):
        feat_map = self.backbone(x) 
        spatial_feat = F.normalize(self.spatial_adapter(feat_map), p=2, dim=1)
        global_feat = F.normalize(self.global_adapter(feat_map), p=2, dim=1)
        return spatial_feat, global_feat

class DINOv3Teacher(nn.Module):
    def __init__(self, config, student_output_size=(40, 40)):
        """
        直接与学生模型特征尺寸对齐的教师模型
        :param config: 配置实例
        :param student_output_size: 学生模型输出的空间特征尺寸 (H, W)，默认(40,40)
        """
        super().__init__()
        search_root = Path(config.dino_path)
        
        # 自动定位 config.json
        real_path = None
        for p in search_root.rglob("config.json"):
            if p.name == "config.json":
                real_path = p.parent
                break
        
        if real_path is None:
            raise FileNotFoundError(f"❌ 没找到模型权重，请检查解压路径: {search_root}")

        print(f"✅ 找到 DINOv3 路径: {real_path}")
        self.teacher = AutoModel.from_pretrained(str(real_path), trust_remote_code=True, local_files_only=True)
        self.teacher.eval()
        for p in self.teacher.parameters(): p.requires_grad = False
        
        # 核心：记录学生模型输出尺寸，用于后续对齐
        self.student_output_size = student_output_size

    def forward(self, x):
        with torch.no_grad():
            outputs = self.teacher(pixel_values=x, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1] 
            global_feat = F.normalize(last_hidden_state[:, 0, :], p=2, dim=1)
            
            patch_tokens = last_hidden_state[:, 1:, :]  # 去掉cls token，形状[B, n, C]
            b, n, c = patch_tokens.shape
            
            # 核心：直接将token序列转为2D特征图，插值对齐到学生模型尺寸（无需假设n是完全平方数）
            # 步骤1：将 (B, n, C) 转为 (B, C, n)，再扩展为 (B, C, 1, n) 伪2D特征图
            patch_feat_1d = patch_tokens.transpose(1, 2)  # (B, C, n)
            patch_feat_pseudo_2d = patch_feat_1d.unsqueeze(2)  # (B, C, 1, n)
            
            # 步骤2：双线性插值，直接对齐到学生模型的输出尺寸 (H, W)
            spatial_feat = F.interpolate(
                input=patch_feat_pseudo_2d,
                size=self.student_output_size,
                mode='bilinear',
                align_corners=False
            )  # 输出形状 (B, C, H, W)，与学生模型完全对齐
            
            # 步骤3：归一化，保持与学生模型输出格式一致
            spatial_feat = F.normalize(spatial_feat, p=2, dim=1)
            
            return spatial_feat, global_feat

# ===================== 🚀 训练逻辑 =====================

def run():
    cfg.check_env()
    
    # 初始化教师模型（指定学生模型输出尺寸，直接对齐）
    teacher = DINOv3Teacher(cfg, student_output_size=(40, 40)).to(cfg.device)
    student = YOLO11Distiller(cfg.yolo_pt_path).to(cfg.device)
    optimizer = optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=0.01)
    
    # 图像预处理（匹配DINOv3预训练配置，保持16倍尺寸）
    transform = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = SimpleImageDataset(cfg.data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    print(f"\n🔥 蒸馏启动 | 数据量: {len(dataset)} | 设备: {cfg.device}")
    
    student.train()
    # 修复：新版torch.amp.GradScaler（解决弃用警告）
    scaler = torch.amp.GradScaler('cuda') if cfg.device == "cuda" else torch.amp.GradScaler('cpu')

    for epoch in range(cfg.epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        for img in loop:
            img = img.to(cfg.device)
            
            # 修复：新版torch.amp.autocast（解决弃用警告）
            with torch.amp.autocast('cuda', enabled=cfg.device=="cuda"):
                s_spatial, s_global = student(img)
                t_spatial, t_global = teacher(img)
                
                # 额外兜底：若尺寸仍有差异，再次插值（实际已通过教师模型对齐，可注释）
                if s_spatial.shape[-2:] != t_spatial.shape[-2:]:
                    s_spatial = F.interpolate(s_spatial, size=t_spatial.shape[-2:], mode='bilinear')
                
                loss_g = 1 - F.cosine_similarity(s_global, t_global).mean()
                loss_s = F.mse_loss(s_spatial, t_spatial)
                loss = (cfg.lambda_global * loss_g) + (cfg.lambda_spatial * loss_s)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            loop.set_postfix(loss=f"{loss.item():.4f}", spatial=f"{loss_s.item():.4f}")

    # 保存产物
    final_path = cfg.output_dir / "yolo11n_distilled.pt"
    full_yolo = YOLO(str(cfg.yolo_pt_path))
    full_yolo.model.model[:10].load_state_dict(student.backbone.state_dict())
    full_yolo.save(str(final_path))
    print(f"🎉 蒸馏成功！模型已保存至: {final_path}")

if __name__ == "__main__":
    run()