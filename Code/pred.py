import sys
import os
# Add the project root directory to the python path so that local ultralytics can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 加载训练好的模型
    model = YOLO("pt/baseline.pt")

    # 2. 在 test 集上评估
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
