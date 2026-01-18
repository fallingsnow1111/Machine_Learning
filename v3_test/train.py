import os
import sys
from pathlib import Path

def fix_ddp_paths():
    """
    修复 DDP 训练时的路径问题
    - 确保项目根目录与 custom_modules 在 sys.path 中
    - 设置 PYTHONPATH 环境变量（子进程会继承）
    """
    # 以 Machine_Learning 作为项目根目录：v3_test 的上一级
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    custom_modules_path = PROJECT_ROOT / "custom_modules"

    # 1) sys.path（当前进程导入使用）
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    if custom_modules_path.exists() and str(custom_modules_path) not in sys.path:
        sys.path.insert(0, str(custom_modules_path))

    # 2) PYTHONPATH（DDP 子进程继承）
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    paths_to_add = [str(PROJECT_ROOT)]
    if custom_modules_path.exists():
        paths_to_add.append(str(custom_modules_path))

    if current_pythonpath:
        new_paths = [p for p in paths_to_add if p not in current_pythonpath.split(os.pathsep)]
        if new_paths:
            os.environ["PYTHONPATH"] = os.pathsep.join([current_pythonpath] + new_paths)
    else:
        os.environ["PYTHONPATH"] = os.pathsep.join(paths_to_add)

    # 避免 DDP 每个 rank 都刷屏
    rank = int(os.environ.get("RANK", "-1"))
    if rank in (-1, 0):
        print("[fix_ddp_paths] sys.path[0:3] =", sys.path[0:3])
        print("[fix_ddp_paths] PYTHONPATH =", os.environ.get("PYTHONPATH", ""))

# 必须在导入 ultralytics 之前执行
fix_ddp_paths()

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