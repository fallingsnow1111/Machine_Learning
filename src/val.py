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

from ultralytics import YOLO

if __name__ == "__main__": 
     # 加载训练好的模型权重
    model = YOLO("pt/best.pt")

    val_results = model.val(
        data="Data/dataset_yolo_processed/dataset.yaml",
        split="test",
        imgsz=640,
        batch=32,
        conf=0.01,               # 置信度阈值
        iou=0.6,                 # NMS的IoU阈值
        plots=True,              # 生成验证图表
        save_json=True,          # 保存结果为JSON
    )
    
    # 打印验证结果
    print("\n" + "="*50)
    print("📊 验证结果:")
    print("="*50)
    print(f"mAP50: {val_results.box.map50:.4f}")
    print(f"mAP50-95: {val_results.box.map:.4f}")
    print(f"Precision: {val_results.box.mp:.4f}")
    print(f"Recall: {val_results.box.mr:.4f}")
    print("="*50 + "\n")