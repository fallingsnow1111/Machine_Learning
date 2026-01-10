from ultralytics import YOLO

# 1. 加载训练好的模型
model = YOLO("/root/autodl-tmp/DustDetection/Baseline/runs/detect/train7/weights/best.pt")

# 2. 在 val 集上评估
metrics = model.val(
    data="../Data/dataset_yolo/dataset.yaml",  # 和训练时一样
    split="test",
    imgsz=640,
    batch=16,
    device=0
)

# 3. 查看核心指标
print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)
print("Precision:", metrics.box.p)
print("Recall:", metrics.box.r)
