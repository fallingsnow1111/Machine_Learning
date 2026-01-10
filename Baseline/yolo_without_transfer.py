from ultralytics import YOLO

# 初始化YOLOv11n模型，从零训练（无预训练权重）
model = YOLO("yolo11n.yaml")  # 用YAML配置文件初始化结构，参数随机

# 训练参数：针对小数据集scratch，强调正则化和增强
results = model.train(
    data="Data/dataset_yolo/dataset.yaml",  # 你的数据集YAML
    epochs=200,                # 需更多epochs收敛
    imgsz=640,                 # 保持放大
    batch=8,                   # 小batch减过拟合（你的GPU够用）
    patience=50,               # 早停监控val
    optimizer='AdamW',         # 适合scratch的稳定优化器
    lr0=0.0001,                # 极低初始lr，防梯度问题
    lrf=0.01,                  # 最终lr = lr0 * lrf
    warmup_epochs=10.0,        # 长warmup，帮助随机初始化稳定
    weight_decay=0.001,        # 强正则化，防过拟合
    momentum=0.95,
    box=10.0,                  # 强调框回归（小目标）
    cls=0.5,                   # 单类，适中
    dfl=1.5,
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,  # 灰度禁用颜色
    degrees=5.0,               # 轻几何增强
    translate=0.05,
    scale=0.2,
    shear=0.0,
    mosaic=1.0,                # 强mosaic扩充有效数据
    mixup=0.0,                 # 关mixup，防模糊小点
    copy_paste=0.3,            # 轻复制尘点，增加小目标密度
    flipud=0.5, fliplr=0.5,
    label_smoothing=0.1,       # 平滑标签，减过拟合
    dropout=0.1,               # 轻dropout
    device=0,                  # GPU
    plots=True,                # 监控曲线
    verbose=True
)