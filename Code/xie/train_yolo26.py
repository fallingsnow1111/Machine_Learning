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
        epochs=200,           # 增加epoch让早停起作用
        imgsz=640,            
        batch=32,            
        
        # 数据增强
        hsv_h=0.015,          # 灰度图适当减小
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15,           # 旋转增强
        translate=0.2,        # 平移增强
        scale=0.5,            # 缩放增强
        shear=0.0,            # 剪切增强
        perspective=0.0,      # 透视增强
        flipud=0.5,           # 上下翻转
        fliplr=0.5,           # 左右翻转
        mosaic=1.0,           # 马赛克增强
        mixup=0.15,           # Mixup增强
        copy_paste=0.1,       # 复制粘贴增强
        
        # 正则化
        dropout=0.2,          # Dropout防过拟合
        weight_decay=0.001,   # L2正则化
        
        # 早停
        patience=30,          # 30轮无改善则停止
        
        # 优化器
        optimizer='AdamW',    # AdamW对小数据集效果好
        lr0=0.001,            # 初始学习率
        lrf=0.01,             # 最终学习率因子
        
        # 其他
        device=[0, 1],
        plots=True,
    )

    # --- 第三步：自动加载本次训练的最佳模型进行验证 ---
    best_model_path = os.path.join(results.save_dir, 'weights', 'best.pt')
    best_model = YOLO(best_model_path)

    metrics = best_model.val(
        data="Data/Raw/dust/dataset.yaml",
        split="test", 
        imgsz=640,
        batch=32,
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