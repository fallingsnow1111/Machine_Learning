from pathlib import Path
import sys
import subprocess

def install_package(package_name):
    """自动安装Python包"""
    try:
        __import__(package_name.split('[')[0].replace('-', '_'))
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
    'ultralytics',
    'timm',
    'pyyaml',
    'tqdm',
    'opencv-python',
    'matplotlib',
    'numpy'
]

for package in required_packages:
    install_package(package)

import lightly_train

if __name__ == "__main__":
    lightly_train.train(
        out="runs/distillation/dinov3_to_yolo11_640",
        data="Data/Processed/dust_640x640_enhanced",                    # 有标注的数据集
        model="ultralytics/yolo11s",
        task="object_detection",
        
        # 关键：设置蒸馏参数
        method="distillation",                      # 使用蒸馏方法
        method_args={
            "teacher": "dinov3/vits16",            # DINOv3 教师模型
        },
        
        # 类别定义
        classes={
            0: "dust",
        },
        
        # 训练参数
        epochs=300,
        batch_size=16,
        learning_rate=0.001,
    )