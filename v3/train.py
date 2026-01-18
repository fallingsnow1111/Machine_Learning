import os
import torch
from ultralytics import YOLO

# ==========================================
# 1. 配置参数
# ==========================================
TRAIN_DATA = "./Data/dataset_merged_no_noise/dataset.yaml"
VAL_DATA = "./Data/dataset_merged_no_noise/dataset.yaml" 
MODEL_CONFIG = "./yolo11P.yaml"
PRETRAINED_WEIGHTS = "./yolo11n.pt"
DEVICE = '0' if torch.cuda.is_available() else 'cpu'

# 选择使用 bf16 的 AMP 精度以提升速度同时避免 fp16/amp 带来的不稳定
# 可通过环境变量覆盖：ULTRALYTICS_AMP_DTYPE=bfloat16 或 bf16 / fp16
os.environ.setdefault("ULTRALYTICS_AMP_DTYPE", "bf16")

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
        batch=32,
        device=DEVICE,

        # 优化器配置
        optimizer='AdamW',
        lr0=0.0005,     
        lrf=0.01,
        
        # Warmup配置
        warmup_epochs=3.0,   
        warmup_momentum=0.8, 
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