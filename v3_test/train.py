import os
import sys
from pathlib import Path
import torch
from ultralytics import YOLO

# ==========================================
# 1. 配置参数
# ==========================================
TRAIN_DATA = "./Data/dataset_yolo_processed/dataset.yaml"
VAL_DATA = "./Data/dataset_yolo_processed/dataset.yaml"
MODEL_CONFIG = "./yolo11P.yaml"
PRETRAINED_WEIGHTS = "./v3_test/exported_last.pt"
DEVICE = [0, 1] if torch.cuda.is_available() else "cpu"  # 使用两张显卡

def run_experiment():
    # --- 第一步：初始化并加载模型 ---
    model = YOLO(MODEL_CONFIG)

    try:
        model.load(PRETRAINED_WEIGHTS)
        print("✅ 成功加载预训练权重！")
    except Exception as e:
        print(f"⚠️ 加载权重跳过或出错 (若结构已修改则属于正常现象): {e}")

    def freeze_dino_callback(trainer):
        print("🔧 [Callback] 正在执行：强制锁定 DINO 相关参数...")
        frozen_count = 0
        for name, param in trainer.model.named_parameters():
            if "dino" in name:
                param.requires_grad = False
                frozen_count += 1
        print(f"✅ 已成功冻结 {frozen_count} 个 DINO 参数分支。")
    model.add_callback("on_train_start", freeze_dino_callback)

    print("\n🚀 开始训练阶段...")
    results = model.train(
        data=TRAIN_DATA,
        epochs=50,
        imgsz=640,
        batch=32,
        patience=0,
        optimizer="AdamW",
        cos_lr=True,
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

    print("\n🔍 开始验证阶段 (使用本次训练的最佳权重)...")
    best_model_path = os.path.join(results.save_dir, "weights", "best.pt")
    best_model = YOLO(best_model_path)

    metrics = best_model.val(
        data=VAL_DATA,
        split="test",
        imgsz=640,
        batch=16,
        device=DEVICE
    )

    print("\n" + "=" * 30)
    print("最终测试集评估结果:")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.p.mean():.4f}")
    print(f"Recall:    {metrics.box.r.mean():.4f}")
    print("=" * 30)

if __name__ == "__main__":
    run_experiment()