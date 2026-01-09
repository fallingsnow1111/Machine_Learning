from ultralytics import YOLO

# 请将你训练好的 best.pt 文件放到项目根目录下的 pt 文件夹中
# 1. 加载训练好的模型
model = YOLO("pt/best.pt")

# 2. 在 val 集上评估
metrics = model.val(
    data="./Data/dataset.yaml",  # 和训练时一样
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
