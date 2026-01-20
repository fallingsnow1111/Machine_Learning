import os
import sys
from pathlib import Path


# 修复 DDP 进程的导入路径问题，确保子进程能找到本地 ultralytics 包
BASE_DIR = Path(__file__).resolve().parent


def fix_ddp_paths():
    """
    修复 DDP 训练时的路径问题
    - 确保本地 ultralytics/custom_modules 在 sys.path 和 PYTHONPATH 中
    """

    custom_modules_path = BASE_DIR / "custom_modules"

    paths_to_add = [BASE_DIR]
    if custom_modules_path.exists():
        paths_to_add.append(custom_modules_path)

    # 将路径添加到 sys.path（当前进程生效）
    for p in paths_to_add:
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)

    # 设置 PYTHONPATH 环境变量（子进程会继承）
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    current_parts = [p for p in current_pythonpath.split(os.pathsep) if p] if current_pythonpath else []
    new_parts = [str(p) for p in paths_to_add if str(p) not in current_parts]

    if new_parts:
        os.environ["PYTHONPATH"] = os.pathsep.join(current_parts + new_parts) if current_parts else os.pathsep.join(new_parts)


fix_ddp_paths()

import torch
from ultralytics import YOLO

# ==========================================
# 1. 配置参数
# ==========================================
TRAIN_DATA = "./Data/dataset_yolo_augmented/dataset.yaml"
VAL_DATA = "./Data/dataset_yolo_augmented/dataset.yaml" 
MODEL_CONFIG = "./yolo11P.yaml"
PRETRAINED_WEIGHTS = "./yolo11n.pt"

# 多GPU配置
# 自动检测可用的GPU数量，并配置使用所有可用GPU
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    if gpu_count > 1:
        # 多GPU: 使用 '0,1,2,3' 格式
        DEVICE = ','.join([str(i) for i in range(gpu_count)])
        print(f"🔥 检测到 {gpu_count} 个GPU，将使用多GPU训练: {DEVICE}")
    else:
        # 单GPU
        DEVICE = '0'
        print(f"🔥 检测到 1 个GPU，将使用单GPU训练: {DEVICE}")
else:
    DEVICE = 'cpu'
    print("⚠️ 未检测到GPU，将使用CPU训练")

# 选择使用 bf16 的 AMP 精度以提升速度同时避免 fp16/amp 带来的不稳定
# 可通过环境变量覆盖：ULTRALYTICS_AMP_DTYPE=bfloat16 或 bf16 / fp16
os.environ.setdefault("ULTRALYTICS_AMP_DTYPE", "bf16")

# 避免多卡显存碎片化导致的 OOM（PyTorch 官方建议）
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def run_experiment():
    # --- 第一步：初始化并加载模型 ---
    # 加载结构配置
    model = YOLO(MODEL_CONFIG)

    # 尝试加载预训练权重
    try:
        model.load(PRETRAINED_WEIGHTS)
        print("✅ 成功加载预训练权重！")
    except Exception as e:
        print(f"⚠️ 加载权重跳过或出错 (若结构已修改则属于正常现象): {e}")

    # 冻结DINO参数（只冻结DINO模型本身，不冻结融合层）
    def freeze_dino_on_train_start(trainer):
        """训练开始时冻结DINO参数"""
        print("🔧 [Callback on_train_start] 冻结 DINO 参数...")
        frozen_count = 0
        unfrozen_count = 0
        
        for name, param in trainer.model.named_parameters():
            # 只冻结 .dino. 路径下的参数（DINO模型本身）
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

        # 降低单卡显存占用：小 batch + 梯度累积
        batch=8,           # 全局 batch；多卡会自动拆分到每卡（2 卡则每卡 4）
        accumulate=2,      # 累积 2 个小 batch 相当于有效 batch=16
        device=DEVICE,

        # 优化器配置
        optimizer='AdamW',
        lr0=0.0005,     
        lrf=0.01,
        
        # Warmup配置
        warmup_epochs=3.0,   
        warmup_bias_lr=0.1,

        # 数据增强
        translate=0.05,
        scale=0.1,
        # copy_paste=0.4,
        
        # 正则化
        dropout=0.5,
        weight_decay=0.005,

        # 其他
        plots=True,
        amp=True,   # 启用AMP，但在内部强制使用bf16
        patience=20,
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