from pathlib import Path
import sys

# 添加本地ultralytics路径
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent  # 返回到 Machine_Learning 目录
sys.path.insert(0, str(project_root))

import lightly_train
from ultralytics import YOLO

print(f"Using ultralytics from: {project_root / 'ultralytics'}")

if __name__ == "__main__":
    # 使用 DINO v2 教师模型蒸馏到 YOLO11
    lightly_train.pretrain(
        out="runs/distillation/dinov2_to_yolo11",
        data="Data/Raw/dust",  # 数据目录
        model="yolo11n.pt",  # YOLO11 nano 模型
        method="distillation",
        method_args={
            "teacher": "dinov2_vits14",  # DINO v2 small 教师模型
        },
        epochs=100,
        batch_size=16,
        input_size=64,  # 64x64 图像尺寸
    )

    # 加载蒸馏后的模型进行微调
    model = YOLO("runs/distillation/dinov2_to_yolo11/exported_models/exported_last.pt")
    
    # 在数据集上微调
    model.train(
        data="Data/Raw/dust/dataset.yaml",
        epochs=50,
        imgsz=64,
        batch=16,
        device='cuda'
    )
    
    # 在测试集上评估
    results = model.val(
        data="Data/Raw/dust/dataset.yaml",
        split='test',
        imgsz=64
    )
    
    print("\n" + "=" * 50)
    print("Distillation and fine-tuning completed!")
    print(f"Final model saved")
    print("=" * 50)