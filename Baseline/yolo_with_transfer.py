from ultralytics import YOLO
import torch.nn as nn

# Load pretrained YOLO11n model
model = YOLO("/root/autodl-tmp/DustDetection/Baseline/yolo11n.pt")

# Optimized training for small objects (dust points)
results = model.train(
    data="Data/dataset_yolo/dataset.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    patience=10,  # 延长早停，灰度学习慢
    optimizer='AdamW',
    lr0=0.0005,  # 更低lr，稳定灰度特征
    lrf=0.01,
    warmup_epochs=5.0,
    # box=10.0,  # 强强调框回归，提高小点定位
    # cls=0.3,   # 降分类权重（单类任务）
    degrees=5.0,
    translate=0.05,
    scale=0.2,
    mosaic=1.0,
    mixup=0.0,
    # perspective=0.0001,
    device=0,
    plots=True
)