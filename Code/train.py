# 引入上级目录以访问自定义模块
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ultralytics import YOLO
import torch.nn as nn

if __name__ == '__main__':
    # Load pretrained YOLO11n model
    model = YOLO("./yolo11P.yaml")

    # try to load pretrain parameters
    try:
        model.load("./pt/yolo11n.pt") 
        print("成功加载预训练权重！")
    except Exception as e:
        print(f"加载权重跳过或出错 (正常现象，因为结构变了): {e}")

    # Optimized training for small objects (dust points)
    results = model.train(
        data="./Data/Merged/no_dust11/dataset_merged.yaml",
        epochs=50,
        imgsz=1024,
        batch=-1,
        close_mosaic=0,      # 完全关闭 Mosaic（防止微小目标被过度缩小）
        mosaic=0.0,          # 强制禁用 Mosaic 增强
        patience=20, 
        optimizer='AdamW',
        lr0=0.001,           # 提高初始学习率（合成数据量大时）
        lrf=0.01,
        warmup_epochs=5.0,
        warmup_momentum=0.5, # 添加 warmup momentum
        degrees=5.0,
        translate=0.05,
        scale=0.2,
        copy_paste=0.3,
        mixup=0.0,           # 关闭 mixup（避免特征混淆）
        device=0,
        plots=True,
        dropout=0.2,
        box=7.5,             # 提高 box 损失权重（改善定位精度）
    )

    print("\n训练完成，开始在测试集(Test Set)上进行评估...")
    # 在测试集上验证 (使用训练好的最佳模型)
    try:
        metrics = model.val(split='test')
        print(f"Test Set mAP50: {metrics.box.map50}")
        print(f"Test Set mAP50-95: {metrics.box.map}")
    except Exception as e:
        print(f"测试集评估出错: {e}")


