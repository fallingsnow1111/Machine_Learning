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
        data="Data/merged_dataset/dataset_merged.yaml",
        epochs=50,
        imgsz=1024,
        batch=8,
        patience=20, 
        optimizer='AdamW',
        lr0=0.0005,  # 更低lr，稳定灰度特征
        lrf=0.01,
        warmup_epochs=5.0,
        degrees=5.0,
        translate=0.05,
        scale=0.2,
        copy_paste=0.4,
        mixup=0.0,
        device=0,
        plots=True,
        dropout=0.2,
    )

    print("\n训练完成，开始在测试集(Test Set)上进行评估...")
    # 在测试集上验证 (使用训练好的最佳模型)
    try:
        metrics = model.val(split='test')
        print(f"Test Set mAP50: {metrics.box.map50}")
        print(f"Test Set mAP50-95: {metrics.box.map}")
    except Exception as e:
        print(f"测试集评估出错: {e}")


