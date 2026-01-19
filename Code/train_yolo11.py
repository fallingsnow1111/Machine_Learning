"""
YOLO11 标准架构双卡训练脚本
- 使用官方 YOLO11 架构（无自定义模块）
- 支持 DINOv3 蒸馏预训练权重
- 双卡 GPU 训练 (device='0,1')
"""

import sys
import os
from pathlib import Path

# ==========================================
# 路径配置
# ==========================================
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print(f"📂 项目根目录: {PROJECT_ROOT}")

import torch
from ultralytics import YOLO

# ==========================================
# 环境检测与路径配置
# ==========================================
IS_KAGGLE = os.path.exists('/kaggle/working')

BASE_DIR = PROJECT_ROOT
DATA_YAML = BASE_DIR / "Data" / "Merged" / "no_noise11_processed" / "dataset.yaml"

# 预训练权重配置：
# 优先使用 DINOv3 蒸馏预训练权重（如果存在）
DISTILL_WEIGHTS = BASE_DIR / "runs" / "distill" / "dinov3_yolo11n_pytorch" / "yolo11n_distilled.pt"
YOLO_WEIGHTS = BASE_DIR / "pt" / "yolo11n.pt"

# 打印路径信息
if IS_KAGGLE:
    print(f"✅ 检测到 Kaggle 环境")
print(f"   项目根目录: {BASE_DIR}")
print(f"   数据配置: {DATA_YAML}")

# ==========================================
# 训练参数
# ==========================================
# GPU 配置: 自动检测双卡
gpu_count = torch.cuda.device_count()
if gpu_count >= 2:
    DEVICE = '0,1'  # 双卡训练
    BATCH_SIZE = 8  # 双卡每卡 batch=8
    print(f"🚀 检测到 {gpu_count} 个 GPU，启用双卡训练 (device={DEVICE})")
elif gpu_count == 1:
    DEVICE = '0'
    BATCH_SIZE = 16
    print(f"⚡ 单卡训练 (device={DEVICE})")
else:
    DEVICE = 'cpu'
    BATCH_SIZE = 4
    print("⚠️ 未检测到 GPU，使用 CPU 训练")

# 环境变量覆盖
DEVICE = os.getenv('DEVICES', DEVICE)
BATCH_SIZE = int(os.getenv('BATCH_SIZE', BATCH_SIZE))

# ==========================================
# 训练超参数（完全按照 ziduo_test 分支配置）
# ==========================================
EPOCHS = 300              # ziduo 使用 300 轮
IMG_SIZE = 640            # ziduo 使用 640
BATCH_SIZE = 16           # ziduo 使用 batch=16
OPTIMIZER = 'AdamW'
LR0 = 0.0008              # ziduo: lr0=0.0008
LRF = 0.01                # ziduo: lrf=0.01
MOMENTUM = 0.937          # ziduo: momentum=0.937
WEIGHT_DECAY = 0.02       # ziduo: weight_decay=0.02
WARMUP_EPOCHS = 3         # ziduo: warmup_epochs=3
WARMUP_MOMENTUM = 0.8     # ziduo: warmup_momentum=0.8
WARMUP_BIAS_LR = 0.1      # ziduo: warmup_bias_lr=0.1
PATIENCE = 50             # ziduo: patience=50
CLOSE_MOSAIC = 20         # ziduo: close_mosaic=20

# Loss 权重配置（ziduo 使用默认值）
BOX_LOSS = 7.5            # ziduo: box=7.5
CLS_LOSS = 0.5            # ziduo: cls=0.5
DFL_LOSS = 1.5            # ziduo: dfl=1.5

# 数据增强参数（完全按照 ziduo_test）
HSV_H = 0.0               # ziduo: hsv_h=0.0（灰度图不需要色调）
HSV_S = 0.0               # ziduo: hsv_s=0.0（灰度图不需要饱和度）
HSV_V = 0.2               # ziduo: hsv_v=0.2（亮度调整）
DEGREES = 5.0             # ziduo: degrees=5.0（旋转角度）
TRANSLATE = 0.08          # ziduo: translate=0.08
SCALE = 0.15              # ziduo: scale=0.15
FLIPUD = 0.3              # ziduo: flipud=0.3（上下翻转）
FLIPLR = 0.5              # ziduo: fliplr=0.5（左右翻转）
MOSAIC = 0.25             # ziduo: mosaic=0.25
MIXUP = 0.0               # ziduo: mixup=0.0
COPY_PASTE = 0.0          # ziduo: copy_paste=0.0

# ==========================================
# 修复 DDP 路径问题
# ==========================================
def fix_ddp_paths():
    """
    修复 DDP 训练时的路径问题
    - 确保 ultralytics 在 sys.path 中
    - 设置 PYTHONPATH 环境变量
    """
    # 将项目根目录添加到 sys.path
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))
    
    # 设置 PYTHONPATH 环境变量（子进程会继承）
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    paths_to_add = [str(BASE_DIR)]
    
    if current_pythonpath:
        new_paths = [p for p in paths_to_add if p not in current_pythonpath]
        if new_paths:
            os.environ['PYTHONPATH'] = os.pathsep.join([current_pythonpath] + new_paths)
    else:
        os.environ['PYTHONPATH'] = os.pathsep.join(paths_to_add)
    
    print(f"✅ DDP 路径配置完成")
    print(f"   BASE_DIR: {BASE_DIR}")
    print(f"   PYTHONPATH: {os.environ['PYTHONPATH']}")

# ==========================================
# 主训练流程
# ==========================================
def run_experiment():
    """完整的训练 + 验证流程"""
    
    print("\n" + "="*60)
    print("🚀 YOLO11 标准架构训练配置")
    print("="*60)
    print(f"环境: {'Kaggle' if IS_KAGGLE else '本地'}")
    print(f"数据配置: {DATA_YAML}")
    print(f"架构: YOLO11n (官方标准)")
    print(f"设备: {DEVICE}")
    print(f"图像大小: {IMG_SIZE}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"训练轮数: {EPOCHS}")
    print(f"学习率: {LR0} -> {LR0 * LRF} (余弦衰减)")
    print(f"Warmup: {WARMUP_EPOCHS} 轮, momentum={WARMUP_MOMENTUM}, bias_lr={WARMUP_BIAS_LR}")
    print(f"优化器: {OPTIMIZER}, momentum={MOMENTUM}, weight_decay={WEIGHT_DECAY}")
    print(f"Loss 权重: box={BOX_LOSS}, cls={CLS_LOSS}, dfl={DFL_LOSS}")
    print(f"Mosaic 增强: {MOSAIC}, 最后 {CLOSE_MOSAIC} 轮关闭")
    print(f"早停: patience={PATIENCE}")
    print("="*60 + "\n")
    
    # 修复 DDP 路径（必须在训练前调用）
    fix_ddp_paths()
    
    # --- 第一步：初始化并加载模型 ---
    print("📦 初始化模型...")
    
    # 优先使用蒸馏预训练权重
    if DISTILL_WEIGHTS.exists():
        print(f"✅ 检测到 DINOv3 蒸馏预训练权重: {DISTILL_WEIGHTS}")
        model = YOLO(str(DISTILL_WEIGHTS))
        print("✅ 成功加载蒸馏预训练权重！")
    elif YOLO_WEIGHTS.exists():
        print(f"⚠️ 蒸馏权重不存在，使用官方预训练权重: {YOLO_WEIGHTS}")
        model = YOLO(str(YOLO_WEIGHTS))
    else:
        print("⚠️ 未找到预训练权重，从头开始训练")
        model = YOLO("yolo11n.yaml")

    # --- 第二步：开始训练 ---
    print("\n🚀 开始训练阶段...")
    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE, 
        save=True,
        cache=True,              # ziduo: cache=True
        
        # 优化器配置
        optimizer=OPTIMIZER,
        lr0=LR0,     
        lrf=LRF,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        warmup_epochs=WARMUP_EPOCHS,
        warmup_momentum=WARMUP_MOMENTUM,
        warmup_bias_lr=WARMUP_BIAS_LR,
        
        # 数据增强（完全按照 ziduo_test）
        augment=True,
        hsv_h=HSV_H,
        hsv_s=HSV_S,
        hsv_v=HSV_V,
        degrees=DEGREES,
        translate=TRANSLATE,
        scale=SCALE,
        flipud=FLIPUD,
        fliplr=FLIPLR,
        mosaic=MOSAIC,
        mixup=MIXUP,
        copy_paste=COPY_PASTE,
        
        # 其他配置
        device=DEVICE,
        plots=True,
        amp=True,                # ziduo: amp=True
        close_mosaic=CLOSE_MOSAIC,
        fraction=1.0,            # ziduo: fraction=1.0
        rect=False,              # ziduo: rect=False
        multi_scale=True,        # ziduo: multi_scale=True
        
        # Loss 权重配置
        box=BOX_LOSS,
        cls=CLS_LOSS,
        dfl=DFL_LOSS,
    )

    # --- 第三步：自动加载本次训练的最佳模型进行验证 ---
    print("\n🔍 开始验证阶段 (使用本次训练的最佳权重)...")
    
    try:
        best_model_path = Path(model.trainer.save_dir) / 'weights' / 'best.pt'
    except AttributeError:
        if IS_KAGGLE:
            best_model_path = BASE_DIR / 'runs' / 'detect' / 'train' / 'weights' / 'best.pt'
        else:
            best_model_path = BASE_DIR / 'runs' / 'detect' / 'train' / 'weights' / 'best.pt'
    
    if not best_model_path.exists():
        print(f"⚠️ 最佳权重不存在: {best_model_path}")
        return
    
    print(f"📂 加载最佳权重: {best_model_path}")
    best_model = YOLO(str(best_model_path))

    metrics = best_model.val(
        data=str(DATA_YAML),
        split="test", 
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE // 2,
        device=DEVICE
    )

    # --- 第四步：输出核心指标 ---
    print("\n" + "="*60)
    print("📊 最终测试集评估结果:")
    print("="*60)
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.p.mean():.4f}")
    print(f"Recall:    {metrics.box.r.mean():.4f}")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_experiment()
