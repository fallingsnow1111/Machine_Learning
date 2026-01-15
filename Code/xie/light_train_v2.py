from pathlib import Path
import sys
import subprocess

# 添加本地ultralytics路径
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent  # 返回到 Machine_Learning 目录
sys.path.insert(0, str(project_root))

# 自动安装必要的包
def install_package(package_name):
    """自动安装Python包"""
    try:
        __import__(package_name.split('[')[0])
        print(f"✓ {package_name} 已安装")
    except ImportError:
        print(f"正在安装 {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✓ {package_name} 安装完成")

# 检查并安装依赖
print("检查依赖包...")
required_packages = [
    'lightly-train',
    'torch',
    'torchvision', 
    'timm',
    'pyyaml',
    'tqdm'
]

for package in required_packages:
    install_package(package)

print("\n所有依赖已准备就绪！\n")

import lightly_train
from ultralytics import YOLO

print(f"Using ultralytics from: {project_root / 'ultralytics'}")

if __name__ == "__main__":
    # 使用 DINO v3 教师模型蒸馏到 YOLO11
    lightly_train.pretrain(
        out="runs/distillation/dinov3_to_yolo11",
        data="Data/Raw/dust",  # 数据目录
        model="yolo11n.pt",  # YOLO11 nano 模型
        method="distillationv1",
        method_args={
            "teacher":  "dinov3/vitb16",  # DINO v3 small 教师模型
        },
        epochs=100,
        batch_size=16
    )

    # 加载蒸馏后的模型进行微调
    model = YOLO("runs/distillation/dinov3_to_yolo11/exported_models/exported_last.pt")
    
    # 在数据集上微调
    model.train(
        data="Data/Raw/dust/dataset.yaml",
        epochs=50,
        imgsz=64,
        batch=16,
        device='cuda'
    )
    
    # 在测试集上评估
    results = model.val(
        data="Data/Raw/dust/dataset.yaml",
        split='test',
        imgsz=64
    )
    
    print("\n" + "=" * 50)
    print("Distillation and fine-tuning completed!")
    print(f"Final model saved")
    print("=" * 50)