"""
改进的训练脚本 - 自动检测并加载蒸馏权重
"""

import os

# 强制离线模式，阻断一切权重下载尝试
os.environ.setdefault("YOLO_OFFLINE", "1")
os.environ.setdefault("YOLO_CHECKS", "False")
os.environ.setdefault("ULTRALYTICS_HUB", "0")
# 禁用 ray tune 回调以避免不兼容的 API 调用
os.environ.setdefault("RAY_TUNE_DISABLE", "1")

import torch
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils import downloads


def _block_download(path, *args, **kwargs):
    """拒绝任何权重下载，仅允许已有本地文件。"""
    p = Path(path)
    if p.exists():
        return str(p)
    raise RuntimeError(f"Download blocked: {path}")


# 拦截 Ultralytics 的下载函数
downloads.attempt_download = _block_download
downloads.attempt_download_asset = _block_download

# ==========================================
# 1. 配置参数
# ==========================================
TRAIN_DATA = "./Data/dataset_yolo_processed/dataset.yaml"
VAL_DATA = "./Data/dataset_yolo_processed/dataset.yaml"

# 蒸馏权重路径（优先级从高到低，仅接受蒸馏权重）
DISTILLED_WEIGHTS_PATHS = [
    "./runs/distill/yolo11n_distilled.pt",          # 本地蒸馏权重
    "./v4/runs/distill/yolo11n_distilled.pt",      # v4目录蒸馏权重
    "./yolo11n_distilled.pt",                      # 当前目录
]

DEVICE = '0' if torch.cuda.is_available() else 'cpu'

# ==========================================
# 2. 智能权重加载
# ==========================================
def find_best_weights():
    """按优先级查找蒸馏权重，未找到则抛出错误。"""
    print("\n🔍 搜索蒸馏权重...")
    for distill_path in DISTILLED_WEIGHTS_PATHS:
        p = Path(distill_path)
        if p.exists():
            print(f"✅ 找到蒸馏权重: {p.absolute()}")
            return str(p)
    raise FileNotFoundError("未找到蒸馏权重，请先运行蒸馏预训练生成 yolo11n_distilled.pt")

# ==========================================
# 3. 主训练流程
# ==========================================
def run_experiment():
    """执行完整训练流程"""
    
    print("="*60)
    print("🚀 YOLO11 目标检测训练")
    print("="*60)
    
    # --- 第一步：加载蒸馏权重 ---
    try:
        weights_path = find_best_weights()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("💡 请先执行：python v4/distill_pretrain.py 生成蒸馏权重")
        raise

    # 加载模型（仅使用蒸馏权重初始化，不再加载模型yaml或官方权重）
    try:
        model = YOLO(weights_path)
        # 强制禁用再次下载官方权重
        model.overrides['pretrained'] = False
        # 移除 ray tune 回调，防止旧版 ray API 报错
        if hasattr(model, "callbacks"):
            model.callbacks = {
                k: [cb for cb in v if cb.__module__ != "ultralytics.utils.callbacks.raytune"]
                for k, v in model.callbacks.items()
            }
        print("🎉 成功加载蒸馏预训练权重！")
    except Exception as e:
        print(f"⚠️ 模型初始化失败: {e}")
        raise
    
    # --- 第二步：开始训练 ---
    print("\n🚀 开始训练阶段...")
    print(f"📊 训练数据: {TRAIN_DATA}")
    print(f"📊 验证数据: {VAL_DATA}")
    print(f"💻 设备: {DEVICE}")
    print()
    
    results = model.train(
        data=TRAIN_DATA,
        epochs=50,
        imgsz=640,
        batch=32,
        pretrained=False,  # 双保险：训练阶段也禁用下载
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
        save=True,
        save_period=10,  # 每10轮保存一次
    )
    
    # --- 第三步：验证 ---
    print("\n🔍 开始验证阶段...")
    
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
    print("\n" + "="*60)
    print("📊 最终测试集评估结果")
    print("="*60)
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.p.mean():.4f}")
    print(f"Recall:    {metrics.box.r.mean():.4f}")
    print("="*60)
    
    # 保存最佳权重到当前目录
    final_weights = Path("./best.pt")
    import shutil
    shutil.copy2(best_model_path, final_weights)
    print(f"\n💾 最佳权重已保存: {final_weights.absolute()}")
    
    return results, metrics

# ==========================================
# 4. 辅助功能
# ==========================================
def check_distilled_weights():
    """检查蒸馏权重是否存在"""
    for path in DISTILLED_WEIGHTS_PATHS:
        p = Path(path)
        if p.exists():
            print(f"✅ 蒸馏权重存在: {p.absolute()}")
            
            # 获取文件大小
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"   文件大小: {size_mb:.2f} MB")
            
            return True
    
    print("⚠️  未找到蒸馏权重")
    print("💡 运行以下命令进行蒸馏预训练：")
    print("   python prepare_distill_data.py")
    print("   python distill_pretrain.py")
    return False

if __name__ == "__main__":
    import sys
    
    # 支持命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == "--check":
            # 仅检查蒸馏权重
            check_distilled_weights()
            sys.exit(0)
        elif sys.argv[1] == "--help":
            print("使用方法:")
            print("  python train.py           # 开始训练")
            print("  python train.py --check   # 检查蒸馏权重")
            print("  python train.py --help    # 显示帮助")
            sys.exit(0)
    
    # 正常训练
    run_experiment()
