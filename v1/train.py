from ultralytics import YOLO
import torch.nn as nn

# Load pretrained YOLO11n model
model = YOLO("./yolo11P.yaml")

# try to load pretrain parameters
try:
    model.load("./yolo11n.pt") 
    print("成功加载预训练权重！")
except Exception as e:
    print(f"加载权重跳过或出错 (正常现象，因为结构变了): {e}")

# Optimized training for small objects (dust points)
results = model.train(
    data="Data/dataset.yaml",
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


