# ==========================================
# 第一步：设置环境变量，规避安全校验与自动下载
# ==========================================
import os
import sys
from pathlib import Path

# 必须在 import torch 之前或紧随其后设置
os.environ['TORCH_ALLOW_WEIGHTS_ONLY_LOAD'] = '0'  # 关键修复：允许加载复杂对象
os.environ["ULTRALYTICS_DISABLE_AUTO_DOWNLOAD"] = "1"
os.environ["ULTRALYTICS_AMP_CHECK"] = "0"

import torch
# 也可以用代码方式强制设置
try:
    torch.serialization.add_safe_globals([Path]) # 允许 Path 对象
except:
    pass

PROJECT_ROOT = Path(__file__).parent.parent if '__file__' in locals() else Path("/mnt/workspace/Machine_Learning")
sys.path.insert(0, str(PROJECT_ROOT))

print(f"📂 项目根目录: {PROJECT_ROOT}")

from ultralytics import YOLO

# ==========================================
# 环境检测与路径配置
# ==========================================
IS_KAGGLE = os.path.exists('/kaggle/working')

BASE_DIR = PROJECT_ROOT
DATA_YAML = BASE_DIR / "Data" / "Raw" / "dust" / "dataset.yaml"

# 修正：蒸馏权重路径（匹配蒸馏代码输出）
DISTILL_WEIGHTS = BASE_DIR / "runs" / "distill" / "yolo11n_distilled.pt"
YOLO_WEIGHTS = BASE_DIR / "pt" / "yolo11n.pt"

# 打印路径信息
if IS_KAGGLE:
    print(f"✅ 检测到 Kaggle 环境")
print(f"   项目根目录: {BASE_DIR}")
print(f"   数据配置: {DATA_YAML}")
print(f"   蒸馏权重路径: {DISTILL_WEIGHTS}")
print(f"   官方权重路径: {YOLO_WEIGHTS}")

# ==========================================
# 训练参数
# ==========================================
# GPU 配置: 自动检测双卡
gpu_count = torch.cuda.device_count()
if gpu_count >= 2:
    DEVICE = '0,1'  # 双卡训练
    BATCH_SIZE = 16  # 双卡总batch=16（每卡8）
    print(f"🚀 检测到 {gpu_count} 个 GPU，启用双卡训练 (device={DEVICE})")
elif gpu_count == 1:
    DEVICE = '0'
    BATCH_SIZE = 16  # 单卡batch=16
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

# Loss 权重配置
BOX_LOSS = 7.5            # ziduo: box=7.5
CLS_LOSS = 0.5            # ziduo: cls=0.5
DFL_LOSS = 1.5            # ziduo: dfl=1.5

# 数据增强参数
HSV_H = 0.0               # ziduo: hsv_h=0.0
HSV_S = 0.0               # ziduo: hsv_s=0.0
HSV_V = 0.2               # ziduo: hsv_v=0.2
DEGREES = 5.0             # ziduo: degrees=5.0
TRANSLATE = 0.08          # ziduo: translate=0.08
SCALE = 0.15              # ziduo: scale=0.15
FLIPUD = 0.3              # ziduo: flipud=0.3
FLIPLR = 0.5              # ziduo: fliplr=0.5
MOSAIC = 0.25             # ziduo: mosaic=0.25
MIXUP = 0.0               # ziduo: mixup=0.0
COPY_PASTE = 0.0          # ziduo: copy_paste=0.0

# ==========================================
# 修复 DDP 路径问题
# ==========================================
def fix_ddp_paths():
    """修复 DDP 训练时的路径问题"""
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))
    
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
# 验证权重文件有效性
# ==========================================
def validate_weight_file(weight_path):
    """验证权重文件是否存在且完整"""
    if not weight_path.exists():
        return False, f"文件不存在: {weight_path}"
    if os.path.getsize(weight_path) < 1024 * 1024:  # 小于1MB，认为损坏
        return False, f"文件过小（可能损坏）: {weight_path}，大小：{os.path.getsize(weight_path)/1024:.2f} KB"
    try:
        torch.load(weight_path, map_location="cpu")
        return True, f"权重文件有效: {weight_path}"
    except Exception as e:
        return False, f"权重文件损坏，加载失败: {weight_path}，错误：{str(e)[:100]}"

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
    print(f"早停: patience={PATIENCE}")
    print("="*60 + "\n")
    
    # 修复 DDP 路径
    fix_ddp_paths()
    
    # --- 初始化并加载模型 ---
    print("📦 初始化模型...")
    
    weight_path = None
    # 优先使用蒸馏预训练权重
    if DISTILL_WEIGHTS.exists():
        is_valid, msg = validate_weight_file(DISTILL_WEIGHTS)
        if is_valid:
            print(f"✅ {msg}")
            weight_path = str(DISTILL_WEIGHTS)
        else:
            print(f"⚠️ {msg}")
    
    # 蒸馏权重无效，使用官方权重
    if weight_path is None and YOLO_WEIGHTS.exists():
        is_valid, msg = validate_weight_file(YOLO_WEIGHTS)
        if is_valid:
            print(f"✅ {msg}")
            weight_path = str(YOLO_WEIGHTS)
        else:
            print(f"⚠️ {msg}")
    
    # 加载模型
    if weight_path is not None:
        try:
            model = YOLO(weight_path)
            print("✅ 成功加载预训练权重！")
        except Exception as e:
            print(f"❌ 权重加载失败，将从头开始训练：{e}")
            model = YOLO("yolo11n.yaml")
    else:
        print("⚠️ 未找到有效预训练权重，从头开始训练")
        model = YOLO("yolo11n.yaml")

    # --- 开始训练 ---
    print("\n🚀 开始训练阶段...")
    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE, 
        save=True,
        cache=True,
        optimizer=OPTIMIZER,
        lr0=LR0,     
        lrf=LRF,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        warmup_epochs=WARMUP_EPOCHS,
        warmup_momentum=WARMUP_MOMENTUM,
        warmup_bias_lr=WARMUP_BIAS_LR,
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
        device=DEVICE,
        plots=True,
        amp=False,  # 彻底禁用AMP，避免校验报错
        close_mosaic=CLOSE_MOSAIC,
        fraction=1.0,
        rect=False,
        multi_scale=True,
        box=BOX_LOSS,
        cls=CLS_LOSS,
        dfl=DFL_LOSS,
    )

    # --- 验证最佳模型 ---
    print("\n🔍 开始验证阶段 (使用本次训练的最佳权重)...")
    
    try:
        best_model_path = Path(model.trainer.save_dir) / 'weights' / 'best.pt'
    except AttributeError:
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

    # --- 输出核心指标 ---
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