import sys
import subprocess
# 移除当前目录,避免导入本地 ultralytics
if '' in sys.path:
    sys.path.remove('')
if '.' in sys.path:
    sys.path.remove('.')

def install_dependencies(verbose: bool = False):
    """安装所需的依赖包"""
    dependencies = [
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

import os
from ultralytics import YOLO

def run_experiment():
    # --- 第一步：初始化并加载模型 ---
    # 使用更小的模型 nano 版本
    model = YOLO("./yaml/yolo26.yaml").load("./pt/yolo26n.pt") 

    # --- 第二步：开始训练（针对小数据集优化）---
    results = model.train(
        data="Data/Raw/dust/dataset.yaml",
        # 基础参数
        # 基础参数
        epochs=200,
        imgsz=640,            
        batch=16,         
        
       # 颜色增强（灰度，基本关掉）
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.05,

        # 几何增强适度
        degrees=8,
        translate=0.08,
        scale=0.3,
        shear=0.0,
        perspective=0.0,
        flipud=0.2,
        fliplr=0.5,
        mosaic=0.3,
        mixup=0.0,
        copy_paste=0.02,

        # 正则化
        dropout=0.35,
        weight_decay=0.01,

        # 早停与优化器
        patience=0,
        optimizer='AdamW',
        lr0=0.0005,
        lrf=0.01,

        device=[0, 1],
        plots=True
    )

    # --- 第三步：自动加载本次训练的最佳模型进行验证 ---
    trainer = getattr(model, "trainer", None)
    save_dir = trainer.save_dir if trainer else getattr(model, "save_dir", None)
    if save_dir is None:
        raise RuntimeError("训练未成功完成，未找到保存目录，请检查上方训练日志。")

    best_model_path = os.path.join(str(save_dir), 'weights', 'best.pt')
    best_model = YOLO(best_model_path)

    metrics = best_model.val(
        data="Data/Raw/dust/dataset.yaml",
        split="test", 
        imgsz=640,
        batch=16,
        device=[0, 1]
    )

    # --- 第四步：输出核心指标 ---
    print("\n" + "="*50)
    print("最终测试集评估结果:")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.p.mean():.4f}")
    print(f"Recall:    {metrics.box.r.mean():.4f}")
    print("="*50)

if __name__ == "__main__":
    run_experiment()