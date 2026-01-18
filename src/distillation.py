import subprocess
import sys

def install_dependencies(verbose: bool = False):
    """安装所需的依赖包"""
    dependencies = [
        "lightly-train",           # Lightly库用于自监督学习
        "ultralytics",       # YOLO11
        "torch",             # PyTorch
        "torchvision",       # 视觉工具
        "pillow",            # 图像处理
        "opencv-python",     # OpenCV
        "matplotlib",        # 可视化
        "numpy",             # 数值计算
        "pyyaml",            # YAML配置文件
        "tqdm",              # 进度条
    ]
    for package in dependencies:
        try:
            __import__(package.replace("-", "_").split("[")[0])
        except ImportError:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ])
    
    print("="*60)
    print("所有依赖已就绪！\n")

# 在导入其他模块前先安装依赖
if __name__ == "__main__":
    install_dependencies()

import lightly_train

if __name__ == "__main__": 
    # 从 DINOv3 蒸馏到 YOLO11n 用于 OLED 灰尘检测
    lightly_train.pretrain(
        # 输出目录
        out="runs/out/dinov3_yolo11n",
        
        # 数据集路径
        # 可以直接指向图片文件夹，不需要标签
        data="Data/dataset_yolo_processed",
        
        # 学生模型：YOLO11n
        model="ultralytics/yolo11n",
        
        # 蒸馏方法
        method="distillation",
        
        # 方法参数
        method_args={
            "teacher":  "dinov3/vits16",
        },
        
        # 训练超参数
        epochs=150,              # 小数据集需要更多epochs
        batch_size=16,           # 小batch size适合500张图片
        
        # 数据增强设置
        transform_args={
            # 图像尺寸
            "image_size": (640, 640),
            
            # 数据增强参数（针对工业检测场景）
            "color_jitter": {
                "prob": 0.5,      # 降低颜色抖动概率
                "brightness": 0.2,
                "contrast": 0.2,
                "saturation": 0.0, # 灰度图不需要饱和度调整
                "hue": 0.0,        # 灰度图不需要色调调整
            },

            # 随机翻转（适合灰尘检测）
            "random_flip": {
                "horizontal_prob": 0.5,
                "vertical_prob": 0.5,
            },
            
            # 随机旋转（灰尘方向不固定）
            "random_rotation": {
                "degrees": 90,
                "prob": 0.5,
            },
        },
        
        
        # 设备设置
        devices=2,                 # 使用2个GPU
        seed=42,                   # 固定随机种子保证可重复性
    )
    
    print("✅ 蒸馏训练完成！")
    print(f"模型保存在:  runs/out/dinov3_yolo11n/exported_models/")
    print(f"可以使用该模型进行后续的目标检测微调")

    