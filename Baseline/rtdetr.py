from ultralytics import RTDETR

# Load a COCO-pretrained RT-DETR-l model
model = RTDETR("rtdetr-l.pt")

# Display model information (optional)
model.info()

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(
    data="Data/dataset.yaml",
    epochs=150,
    imgsz=640,
    batch=16,
    patience=50,  # 延长早停，灰度学习慢
    optimizer='AdamW',
    lr0=0.0005,  # 更低lr，稳定灰度特征
    lrf=0.01,
    warmup_epochs=5.0,
    box=10.0,  # 强调框回归，提高小点定位
    cls=0.3,   # 降分类权重（单类任务）
    degrees=5.0,
    translate=0.05,
    scale=0.2,
    mosaic=1.0,
    mixup=0.0,
    perspective=0.0001,
    device=0,
    plots=True
)