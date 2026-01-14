# from ultralytics import YOLO
# import torch.nn as nn

# # Load pretrained YOLO11n model
# model = YOLO("/root/autodl-tmp/DustDetection/Baseline/yolo11n.pt")

# # Optimized training for small objects (dust points)
# results = model.train(
#     data="Data/dataset_yolo/dataset.yaml",
#     epochs=50,
#     imgsz=640,
#     batch=8,
#     patience=10,  # 延长早停，灰度学习慢
#     optimizer='AdamW',
#     lr0=0.0005,  # 更低lr，稳定灰度特征
#     lrf=0.01,
#     warmup_epochs=5.0,
#     # box=10.0,  # 强强调框回归，提高小点定位
#     # cls=0.3,   # 降分类权重（单类任务）
#     degrees=5.0,
#     translate=0.05,
#     scale=0.2,
#     mosaic=1.0,
#     mixup=0.0,
#     # perspective=0.0001,
#     device=0,
#     plots=True
# )


import os
import torch
from ultralytics import YOLO

# ==========================================
# 1. 配置参数
# ==========================================
TRAIN_DATA = "./Data/dataset_yolo_processed/dataset.yaml"
VAL_DATA = "./Data/dataset_yolo_processed/dataset.yaml" 
PRETRAINED_WEIGHTS = "yolo11n.pt"
DEVICE = '0' if torch.cuda.is_available() else 'cpu'

def run_experiment():
    # --- 第一步：初始化并加载模型 ---
    model = YOLO(PRETRAINED_WEIGHTS)

    # --- 第二步：开始训练 ---
    print("\n🚀 开始训练阶段...")
    results = model.train(
        data=TRAIN_DATA,
        epochs=50,
        imgsz=640,
        batch=16,
        patience=0, 
        optimizer='AdamW',
        lr0=0.0005,     
        lrf=0.01,
        warmup_epochs=5.0,
        translate=0.05,
        scale=0.1,
        copy_paste=0.4,
        device=DEVICE,
        plots=True,
        dropout=0.2,
    )

    # --- 第三步：自动加载本次训练的最佳模型进行验证 ---
    print("\n🔍 开始验证阶段 (使用本次训练的最佳权重)...")
    
    # 训练完成后，best.pt 的路径会自动保存在 results.save_dir 中
    best_model_path = os.path.join(results.save_dir, 'weights', 'best.pt')
    best_model = YOLO(best_model_path)

    metrics = best_model.val(
        data=VAL_DATA,
        split="test", 
        imgsz=640,
        batch=16,
        device=DEVICE
    )

    # --- 第四步：输出核心指标 ---
    print("\n" + "="*30)
    print("最终测试集评估结果:")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.p.mean():.4f}")
    print(f"Recall:    {metrics.box.r.mean():.4f}")
    print("="*30)

if __name__ == "__main__":
    run_experiment()