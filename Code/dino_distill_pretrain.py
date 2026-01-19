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

# ==========================================
# 路径配置
# ==========================================
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print(f"📂 项目根目录: {PROJECT_ROOT}")

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
    # 确保 ultralytics 可被导入，以便 lightly-train 内部使用
    import ultralytics
    
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
    print(f"👨‍🎓 Student: {STUDENT_MODEL} (标准模式)")
    print(f"📊 训练轮数: {EPOCHS}")
    print(f"📊 批次大小: {BATCH_SIZE}")
    print(f"📊 图像尺寸: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"💻 GPU数量: {DEVICES}")
    print("="*60 + "\n")
    
    # 执行蒸馏预训练
    # 之前报错 "gaussian_blur.kernel_size" 已修复
    # 回退到使用 model="ultralytics/yolo11n" 字符串，这是 ziduo_test 分支验证过可行的方案
    lightly_train.pretrain(
        # 输出目录
        out=str(OUTPUT_DIR),
        
        # 数据集路径（只需要图像，不需要标签）
        data=str(DATA_DIR),
        
        # 使用字符串标识符，让 lightly-train 自动处理加载
        model=STUDENT_MODEL,
        
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
