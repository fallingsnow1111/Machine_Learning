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
import torch
import ultralytics
from ultralytics import YOLO

def run_experiment():
    # --- 第一步：初始化并加载模型 ---
    # 使用更小的模型 nano 版本
    model = YOLO("./yaml/yolo26.yaml").load("./pt/yolo26n.pt") 

    # --- 第二步：开始训练（针对小数据集优化）---
    results = model.train(
        data="Data/Raw/dust/dataset.yaml",
        
        # 基础参数
        epochs=100,           # 增加epochs,让早停机制起作用
        imgsz=64,            # 匹配您的数据集尺寸
        batch=8,             # 减小batch size (小数据集用小batch)
        
        # 早停和正则化
        patience=20,         # 启用早停,20个epoch不改善就停止
        dropout=0.3,         # 增加dropout (0.2->0.3)
        
        # 优化器设置
        optimizer='SGD',     # SGD比AdamW更不容易过拟合
        lr0=0.001,          # 降低初始学习率
        lrf=0.01,           # 最终学习率衰减
        momentum=0.937,     # SGD动量
        weight_decay=0.001, # 增加权重衰减 (L2正则化)
        warmup_epochs=3,    # 减少warmup
        
        # 数据增强 (关键! 扩充小数据集)
        hsv_h=0.015,        # 色调增强
        hsv_s=0.7,          # 饱和度增强
        hsv_v=0.4,          # 亮度增强
        degrees=15,         # 随机旋转 ±15度
        translate=0.2,      # 增加平移增强
        scale=0.5,          # 增加缩放增强
        shear=5,            # 剪切变换
        perspective=0.001,  # 透视变换
        flipud=0.5,         # 上下翻转
        fliplr=0.5,         # 左右翻转
        mosaic=1.0,         # mosaic增强
        mixup=0.3,          # mixup增强
        copy_paste=0.3,     # copy-paste增强
        
        # 其他设置
        device=[0, 1],
        plots=True,
        cache=True,         # 缓存数据到内存 (小数据集可以)
        workers=4,          # 减少worker数量
        
        # 验证频率
        val=True,           # 每个epoch都验证
    )

    # --- 第三步：自动加载本次训练的最佳模型进行验证 ---
    best_model_path = os.path.join(results.save_dir, 'weights', 'best.pt')
    best_model = YOLO(best_model_path)

    metrics = best_model.val(
        data="Data/Raw/dust/dataset.yaml",
        split="test", 
        imgsz=64,           # 匹配训练尺寸
        batch=8,
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
    print("\n💡 小数据集训练建议:")
    print("1. 观察训练/验证损失曲线,若验证损失上升则提前停止")
    print("2. 考虑收集更多数据或使用预训练模型微调")
    print("3. 如果是灰度图,确保 dataset.yaml 中 nc=1")

if __name__ == "__main__":
    run_experiment()