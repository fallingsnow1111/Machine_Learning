"""
DINOv3 -> YOLO11n 知识蒸馏预训练脚本
参考 ziduo_test 分支的成功经验

功能：
1. 从 DINOv3 蒸馏视觉特征到 YOLO11n backbone
2. 无监督预训练，不需要标签
3. 为后续的 DINO-YOLO 有监督训练提供更好的初始化权重

使用流程：
1. 运行此脚本进行蒸馏预训练（150 epochs）
2. 修改 dino_yolo.py 中的 PRETRAINED_WEIGHTS 路径
3. 运行 dino_yolo.py 进行有监督微调（50 epochs）
"""

import subprocess
import sys
from pathlib import Path
import torch
import torch.nn as nn

# ==========================================
# 路径配置
# ==========================================
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print(f"📂 项目根目录: {PROJECT_ROOT}")

# ==========================================
# YOLO11 适配器类（满足 lightly-train 接口要求）
# ==========================================
class YOLO11BackboneWrapper(nn.Module):
    """
    YOLO11 Backbone 适配器，实现 lightly-train 要求的接口
    """
    def __init__(self, backbone_model, feature_dim=256):
        super().__init__()
        self.backbone = backbone_model
        # YOLO11n SPPF 层（第9层）的输出通道数通常是 256
        self._feature_dim = feature_dim

    def feature_dim(self) -> int:
        """返回特征向量的维度"""
        return self._feature_dim

    def forward_features(self, x):
        """执行前向传播，提取特征图"""
        return self.backbone(x)

    def forward_pool(self, x):
        """执行全局平均池化，将特征图转为 1D 向量"""
        # x 形状通常是 [B, 256, H, W]，转为 [B, 256]
        return torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).flatten(1)

    def forward(self, x):
        """默认前向传播直接返回池化后的特征"""
        x = self.forward_features(x)
        return self.forward_pool(x)

    def get_model(self):
        """返回原始骨干网络"""
        return self.backbone

# ==========================================
# 安装依赖
# ==========================================
def install_lightly():
    """安装 lightly-train 库"""
    print("\n" + "="*60)
    print("📦 检查依赖...")
    print("="*60)
    
    try:
        import lightly_train
        print("✅ lightly-train 已安装")
    except ImportError:
        print("📥 正在安装 lightly-train...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "lightly-train"
        ])
        print("✅ lightly-train 安装完成")
    
    print("="*60 + "\n")


# ==========================================
# 配置参数
# ==========================================
# 数据路径（只需要图像，不需要标签）
DATA_DIR = PROJECT_ROOT / "Data" / "Merged" / "no_noise11_processed" / "images" / "train"

# 输出路径
OUTPUT_DIR = PROJECT_ROOT / "runs" / "distill" / "dinov3_yolo11n"

# 训练超参数
EPOCHS = 150              # 蒸馏预训练轮数（参考ziduo_test）
BATCH_SIZE = 16           # 批次大小
IMAGE_SIZE = 640          # 图像尺寸（与预处理和主训练保持一致）
DEVICES = 2               # GPU数量（双卡）
SEED = 42                 # 随机种子

# Teacher/Student 模型
TEACHER_MODEL = "dinov3/vitt16"       # DINOv3 ViT-Tiny/16
STUDENT_MODEL = "ultralytics/yolo11n"  # YOLO11n


# ==========================================
# 蒸馏预训练主函数
# ==========================================
def run_distillation():
    """执行知识蒸馏预训练"""
    import lightly_train
    from ultralytics import YOLO
    import torch
    
    # 检查数据目录
    if not DATA_DIR.exists():
        print(f"❌ 错误：数据目录不存在: {DATA_DIR}")
        print("请检查路径配置！")
        sys.exit(1)
    
    # 打印配置信息
    print("\n" + "="*60)
    print("🚀 DINOv3 -> YOLO11n 知识蒸馏预训练")
    print("="*60)
    print(f"📁 数据目录: {DATA_DIR}")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print(f"👨‍🏫 Teacher: {TEACHER_MODEL}")
    print(f"👨‍🎓 Student: YOLO11n (提取内部模型)")
    print(f"📊 训练轮数: {EPOCHS}")
    print(f"📊 批次大小: {BATCH_SIZE}")
    print(f"📊 图像尺寸: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"💻 GPU数量: {DEVICES}")
    print("="*60 + "\n")
    
    # 【关键修复】加载 YOLO 并提取内部 PyTorch 模型
    print("📦 正在加载 YOLO11n 模型...")
    yolo11n_weights = PROJECT_ROOT / "pt" / "yolo11n.pt"
    
    if yolo11n_weights.exists():
        print(f"✅ 找到权重文件: {yolo11n_weights}")
        yolo_manager = YOLO(str(yolo11n_weights))
    else:
        print(f"⚠️ 未找到 {yolo11n_weights}，使用架构配置初始化")
        yolo_manager = YOLO("yolo11n.yaml")
    
    # 提取内部的 torch.nn.Module（绕过 Ultralytics 包装类）
    inner_model = yolo_manager.model
    print(f"✅ 成功提取内部模型: {type(inner_model)}")
    print(f"   模型参数量: {sum(p.numel() for p in inner_model.parameters()):,}")
    
    # 【关键修复】手动提取 Backbone（YOLO11n 的前 0-9 层）
    # model[0-9] 是 backbone, model[9] 是 SPPF, model[10] 是 C2PSA
    # 提取到 SPPF 结束（索引 0-9，共 10 层）
    try:
        backbone_layers = list(inner_model.model[:10])  # 0-9 层
        raw_backbone = nn.Sequential(*backbone_layers)
        print(f"✅ 成功提取 Backbone: {len(backbone_layers)} 层")
        
        # 【核心修改】使用适配器包装，满足 lightly-train 接口要求
        # YOLO11n 的 SPPF 层（第9层）输出通道数是 256
        student_model = YOLO11BackboneWrapper(raw_backbone, feature_dim=256)
        print(f"✅ 适配器封装完成，特征维度: {student_model.feature_dim()}")
        
    except Exception as e:
        print(f"❌ Backbone 提取失败: {e}")
        print("无法继续执行蒸馏预训练")
        sys.exit(1)
    
    # 执行蒸馏预训练
    print("\n🚀 开始蒸馏预训练...")
    lightly_train.pretrain(
        # 输出目录
        out=str(OUTPUT_DIR),
        
        # 数据集路径（只需要图像，不需要标签）
        data=str(DATA_DIR),
        
        # 【核心修改】传入提取后的 PyTorch 模型，而不是字符串
        model=student_model,
        
        # 蒸馏方法
        method="distillation",
        
        # Teacher 模型配置
        method_args={
            "teacher": TEACHER_MODEL,
        },
        
        # 训练超参数
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        
        # 数据增强设置（参考ziduo_test，针对灰度工业图像优化）
        transform_args={
            # 图像尺寸
            "image_size": (IMAGE_SIZE, IMAGE_SIZE),
            
            # 颜色抖动（保守设置，适合灰度图）
            "color_jitter": {
                "prob": 0.3,           # 降低概率
                "brightness": 0.2,     # 适度亮度调整
                "contrast": 0.2,       # 适度对比度调整
                "saturation": 0.0,     # 灰度图不需要饱和度
                "hue": 0.0,            # 灰度图不需要色调
            },
            
            # 随机翻转（灰尘方向不固定）
            "random_flip": {
                "horizontal_prob": 0.5,
                "vertical_prob": 0.5,
            },
            
            # 随机旋转（工业检测场景）
            "random_rotation": {
                "degrees": 90,         # 90度旋转
                "prob": 0.5,
            },
            
            # 高斯模糊（模拟不同对焦状态）
            "gaussian_blur": {
                "prob": 0.2,
            },
        },
        
        # 设备设置
        devices=DEVICES,
        seed=SEED,
    )
    
    # 输出结果信息
    print("\n" + "="*60)
    print("✅ 蒸馏预训练完成！")
    print("="*60)
    print(f"📁 模型保存在: {OUTPUT_DIR / 'exported_models'}")
    print(f"📄 权重文件: exported_last.pt")
    print("\n💡 下一步操作：")
    print("   1. 编辑 Code/dino_yolo.py")
    print("   2. 修改 PRETRAINED_WEIGHTS 为：")
    print(f"      {OUTPUT_DIR / 'exported_models' / 'exported_last.pt'}")
    print("   3. 运行 python Code/dino_yolo.py 进行有监督微调")
    print("="*60 + "\n")


# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    # 安装依赖
    install_lightly()
    
    # 执行蒸馏预训练
    try:
        run_distillation()
    except Exception as e:
        print(f"\n❌ 蒸馏预训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
