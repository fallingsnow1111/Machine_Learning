import os
import sys
from pathlib import Path

# ==========================================
# 1. 环境修复与路径处理 (完全保留你的逻辑)
# ==========================================
BASE_DIR = Path(__file__).resolve().parent

def fix_ddp_paths():
    """
    修复 DDP 训练时的路径问题
    - 确保本地 ultralytics/custom_modules 在 sys.path 和 PYTHONPATH 中
    """
    paths_to_add = [BASE_DIR]

    for p in paths_to_add:
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)

    current_pythonpath = os.environ.get("PYTHONPATH", "")
    current_parts = [p for p in current_pythonpath.split(os.pathsep) if p] if current_pythonpath else []
    new_parts = [str(p) for p in paths_to_add if str(p) not in current_parts]

    if new_parts:
        os.environ["PYTHONPATH"] = os.pathsep.join(current_parts + new_parts) if current_parts else os.pathsep.join(new_parts)

fix_ddp_paths()

import torch
from ultralytics import YOLO

# ==========================================
# 2. 核心参数与 bf16 配置 (完全恢复你的逻辑)
# ==========================================
# 选择使用 bf16 的 AMP 精度以提升速度同时避免 fp16/amp 带来的不稳定
os.environ.setdefault("ULTRALYTICS_AMP_DTYPE", "bf16")

# 避免多卡显存碎片化导致的 OOM
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# DDP 相关优化
os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("NCCL_TIMEOUT", "600")  # 增加到 600s，Kaggle 网络可能较慢
os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "DETAIL")  # 更详细的调试信息

# --- [关键：解决 Kaggle 双卡卡死的必要修复] ---
os.environ["NCCL_P2P_DISABLE"] = "1"  # 禁用不支持的 P2P 通信
os.environ["NCCL_IB_DISABLE"] = "1"   # 禁用 InfiniBand
os.environ["NCCL_SOCKET_IFNAME"] = "lo"  # 强制使用 loopback 接口
# --------------------------------------------

def run_experiment():
    # 配置参数
    TRAIN_DATA = "./Data/dataset_yolo_augmented/dataset.yaml"
    VAL_DATA = "./Data/dataset_yolo_augmented/dataset.yaml" 
    MODEL_CONFIG = "./yolo11P.yaml"
    PRETRAINED_WEIGHTS = "./yolo11n.pt"

    # 多GPU配置逻辑
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            DEVICE = ','.join([str(i) for i in range(gpu_count)])
            print(f"🔥 检测到 {gpu_count} 个GPU，将使用多GPU训练: {DEVICE}")
        else:
            DEVICE = '0'
            print(f"🔥 检测到 1 个GPU，将使用单GPU训练: {DEVICE}")
    else:
        DEVICE = 'cpu'
        print("⚠️ 未检测到GPU，将使用CPU训练")

    # --- 第一步：初始化并加载模型 ---
    model = YOLO(MODEL_CONFIG)
    try:
        model.load(PRETRAINED_WEIGHTS)
        print("✅ 成功加载预训练权重！")
    except Exception as e:
        print(f"⚠️ 加载权重跳过或出错: {e}")

    # 冻结DINO参数逻辑 (完全恢复你的原始细节)
    def freeze_dino_on_train_start(trainer):
        print("🔧 [Callback on_train_start] 冻结 DINO 参数...")
        frozen_count = 0
        unfrozen_count = 0
        for name, param in trainer.model.named_parameters():
            if ".dino." in name and param.requires_grad:
                param.requires_grad = False
                frozen_count += 1
            elif any(x in name for x in ['input_projection', 'fusion_layer', 'feature_adapter', 'spatial_projection']):
                if not param.requires_grad:
                    param.requires_grad = True
                unfrozen_count += 1
        print(f"✅ 已冻结 {frozen_count} 个 DINO 模型参数")
        print(f"✅ 保持 {unfrozen_count} 个融合层参数可训练")
    
    model.add_callback("on_train_start", freeze_dino_on_train_start)

    # --- 第二步：开始训练 ---
    print("\n🚀 开始训练阶段...")
    results = model.train(
        data=TRAIN_DATA,
        epochs=60,
        imgsz=640,
        batch=8, 
        device=DEVICE,
        
        # --- [关键修改：DDP 稳定性] ---
        workers=2,         # 必须 > 0，防止 DDP 通信中因数据 IO 死锁
        close_mosaic=10,   # 最后 10 个 epoch 关闭 mosaic 增强，防止卡住
        # ----------------------------
        
        optimizer='AdamW',
        lr0=0.0005,
        lrf=0.01,
        warmup_epochs=3.0,
        warmup_bias_lr=0.1,
        translate=0.05,
        scale=0.1,
        dropout=0.5,
        weight_decay=0.005,
        plots=True,
        amp=True,          # 结合上面的环境变量，强制使用 bf16
        patience=20,
    )

    # --- 第三步：自动加载最佳模型进行验证 (恢复你的逻辑) ---
    print("\n🔍 开始验证阶段 (使用本次训练的最佳权重)...")
    best_model_path = os.path.join(results.save_dir, 'weights', 'best.pt')
    best_model = YOLO(best_model_path)

    metrics = best_model.val(
        data=VAL_DATA,
        split="test", 
        imgsz=640,
        batch=16,  # 降低 val batch，多卡时避免 OOM
        device=DEVICE
    )

    # --- 第四步：输出核心指标 (恢复你的逻辑) ---
    print("\n" + "="*30)
    print("最终测试集评估结果:")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.p.mean():.4f}")
    print(f"Recall:    {metrics.box.r.mean():.4f}")
    print("="*30)

if __name__ == "__main__":
    run_experiment()