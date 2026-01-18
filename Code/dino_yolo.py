"""
DINO-YOLO 双卡训练脚本
- 基于 Linking 分支 v3_test/train.py 的工作版本
- 支持自动检测 Kaggle 环境和本地环境
- 双卡 GPU 训练 (device='0,1')
- 使用 custom_modules.dino 模块
"""

import sys
import os
from pathlib import Path

# ==========================================
# 路径配置（必须在导入 ultralytics 之前）
# ==========================================
# 获取项目根目录（dino_yolo.py 在 Code/ 子目录下，需要回到上级）
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 打印调试信息
print(f"📂 项目根目录: {PROJECT_ROOT}")
print(f"📂 Python 搜索路径已添加: {PROJECT_ROOT}")

import torch
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks

# 导入 DINO 模块
from custom_modules.dino import DINO3Preprocessor, DINO3Backbone


def register_custom_layers():
    """注册 DINO 模块到 YOLO 构建系统"""
    setattr(tasks, "DINO3Preprocessor", DINO3Preprocessor)
    setattr(tasks, "DINO3Backbone", DINO3Backbone)
    print("✅ 模块注册完成：DINO3Preprocessor, DINO3Backbone")


# ==========================================
# 环境检测与路径配置
# ==========================================
IS_KAGGLE = os.path.exists('/kaggle/working')

# BASE_DIR 现在是项目根目录（已在文件开头设置）
BASE_DIR = PROJECT_ROOT
DATA_YAML = BASE_DIR / "Data" / "Merged" / "no_noise11_processed" / "dataset.yaml"
MODEL_CONFIG = BASE_DIR / "YAML" / "dino_yolo_ema.yaml"
PRETRAINED_WEIGHTS = BASE_DIR / "pt" / "yolo11n.pt"

# 打印路径信息用于调试
if IS_KAGGLE:
    print(f"✅ 检测到 Kaggle 环境")
print(f"   项目根目录: {BASE_DIR}")
print(f"   模型配置: {MODEL_CONFIG}")
print(f"   数据配置: {DATA_YAML}")

# ==========================================
# 训练参数
# ==========================================
# GPU 配置: 自动检测双卡
gpu_count = torch.cuda.device_count()
if gpu_count >= 2:
    DEVICE = '0,1'  # 双卡训练
    BATCH_SIZE = 8  # 双卡可以用更大的batch
    print(f"🚀 检测到 {gpu_count} 个 GPU，启用双卡训练 (device={DEVICE})")
elif gpu_count == 1:
    DEVICE = '0'
    BATCH_SIZE = 8
    print(f"⚡ 单卡训练 (device={DEVICE})")
else:
    DEVICE = 'cpu'
    BATCH_SIZE = 4
    print("⚠️ 未检测到 GPU，使用 CPU 训练")

# 环境变量覆盖（方便 Kaggle Notebook 调试）
DEVICE = os.getenv('DEVICES', DEVICE)
BATCH_SIZE = int(os.getenv('BATCH_SIZE', BATCH_SIZE))

# 训练超参数
EPOCHS = 50
IMG_SIZE = 1024  # DINO 模型建议使用 1024
OPTIMIZER = 'AdamW'
LR0 = 0.0005  # 初始学习率
LRF = 0.01  # 最终学习率 = LR0 * LRF
WARMUP_EPOCHS = 5.0  # 10% 的 epoch 用于 warmup
PATIENCE = 0  # 不使用早停
CLOSE_MOSAIC = 10  # 最后 10 轮关闭 Mosaic 增强（占总轮数的 20%）

# 数据增强参数
TRANSLATE = 0.1  # 图像平移范围 ±10%
SCALE = 0.2  # 图像缩放范围 ±20%
COPY_PASTE = 0.4  # Copy-Paste 增强概率 40%
DROPOUT = 0.2  # Dropout 比例（应用于检测头）

# ==========================================
# 修复 DDP 路径问题
# ==========================================
def fix_ddp_paths():
    """
    修复 DDP 训练时的路径问题
    - 确保 custom_modules 在 sys.path 中
    - 设置 PYTHONPATH 环境变量
    """
    custom_modules_path = str(BASE_DIR / "custom_modules")
    if custom_modules_path not in sys.path:
        sys.path.insert(0, custom_modules_path)
    
    # 将项目根目录添加到 sys.path
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))
    
    # 设置 PYTHONPATH 环境变量（子进程会继承）
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    paths_to_add = [str(BASE_DIR), custom_modules_path]
    
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
    print("🚀 DINO-YOLO 训练配置")
    print("="*60)
    print(f"环境: {'Kaggle' if IS_KAGGLE else '本地'}")
    print(f"数据配置: {DATA_YAML}")
    print(f"模型配置: {MODEL_CONFIG}")
    print(f"预训练权重: {PRETRAINED_WEIGHTS}")
    print(f"设备: {DEVICE}")
    print(f"图像大小: {IMG_SIZE}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"训练轮数: {EPOCHS}")
    print(f"学习率: {LR0} -> {LR0 * LRF} (余弦衰减 + Warmup {WARMUP_EPOCHS} 轮)")
    print(f"Mosaic 增强: 前 {EPOCHS - CLOSE_MOSAIC} 轮启用, 后 {CLOSE_MOSAIC} 轮关闭")
    print("="*60 + "\n")
    
    # 修复 DDP 路径
    fix_ddp_paths()
    
    # 注册自定义模块
    register_custom_layers()
    
    # --- 第一步：初始化并加载模型 ---
    print("📦 初始化模型...")
    model = YOLO(str(MODEL_CONFIG))

    # 尝试加载预训练权重
    if PRETRAINED_WEIGHTS.exists():
        try:
            model.load(str(PRETRAINED_WEIGHTS))
            print("✅ 成功加载预训练权重！")
        except Exception as e:
            print(f"⚠️ 加载权重跳过或出错 (若结构已修改则属于正常现象): {e}")
    else:
        print(f"⚠️ 预训练权重不存在: {PRETRAINED_WEIGHTS}")

    # 冻结 DINO 参数
    def freeze_dino_callback(trainer):
        print("🔧 [Callback] 正在执行：强制锁定 DINO 相关参数...")
        frozen_count = 0
        for name, param in trainer.model.named_parameters():
            if "dino" in name.lower():
                param.requires_grad = False
                frozen_count += 1
        print(f"✅ 已成功冻结 {frozen_count} 个 DINO 参数分支。")
    
    model.add_callback("on_train_start", freeze_dino_callback)

    # --- 第二步：开始训练 ---
    print("\n🚀 开始训练阶段...")
    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE, 
        optimizer=OPTIMIZER,
        cos_lr=True,  # 使用余弦学习率衰减
        lr0=LR0,     
        lrf=LRF,
        warmup_epochs=WARMUP_EPOCHS,
        translate=TRANSLATE,
        scale=SCALE,
        copy_paste=COPY_PASTE,
        device=DEVICE,
        plots=True,
        dropout=DROPOUT,
        amp=False,  # 关闭混合精度（DINO 模型可能不兼容 AMP）
        close_mosaic=CLOSE_MOSAIC,  # 训练后期关闭 Mosaic 有助于模型收敛
    )

    # --- 第三步：自动加载本次训练的最佳模型进行验证 ---
    print("\n🔍 开始验证阶段 (使用本次训练的最佳权重)...")
    
    # 使用模型的 trainer.save_dir 获取保存路径
    try:
        best_model_path = Path(model.trainer.save_dir) / 'weights' / 'best.pt'
    except AttributeError:
        # 如果无法获取，使用默认路径（Kaggle 环境）
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
        batch=BATCH_SIZE // 2,  # 验证时使用较小的 batch
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
