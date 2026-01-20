# ==========================================
# 第一步：设置环境变量，规避安全校验与自动下载
# ==========================================
import os
import sys
from pathlib import Path

# 尽可能在所有导入前设置
os.environ['TORCH_ALLOW_WEIGHTS_ONLY_LOAD'] = '0'
os.environ["ULTRALYTICS_DISABLE_AUTO_DOWNLOAD"] = "1"
os.environ["ULTRALYTICS_AMP_CHECK"] = "0"

import torch
# 显式允许关键对象
try:
    from pathlib import PosixPath, WindowsPath
    torch.serialization.add_safe_globals([Path, PosixPath, WindowsPath])
except:
    pass

# 确保项目根目录在路径中
PROJECT_ROOT = Path(__file__).parent.parent if '__file__' in locals() else Path("/mnt/workspace/Machine_Learning")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print(f"📂 项目根目录: {PROJECT_ROOT}")

from ultralytics import YOLO

# ==========================================
# 环境检测与路径配置
# ==========================================
IS_KAGGLE = os.path.exists('/kaggle/working')
BASE_DIR = PROJECT_ROOT
DATA_YAML = BASE_DIR / "Data" / "Merged" / "dust_processed" / "dataset.yaml"

# 匹配蒸馏代码输出路径
DISTILL_WEIGHTS = BASE_DIR / "runs" / "distill" / "yolo11n_distilled.pt"
YOLO_WEIGHTS = BASE_DIR / "pt" / "yolo11n.pt"

print(f"   数据配置: {DATA_YAML}")
print(f"   蒸馏权重路径: {DISTILL_WEIGHTS}")
print(f"   官方权重路径: {YOLO_WEIGHTS}")

# ==========================================
# 验证权重文件有效性（核心修复点）
# ==========================================
def validate_weight_file(weight_path):
    """验证权重文件是否存在且完整，并强制关闭安全检查"""
    if not weight_path.exists():
        return False, f"文件不存在: {weight_path}"
    if os.path.getsize(weight_path) < 1024 * 1024:
        return False, f"文件过小: {weight_path}"
    
    try:
        # 【核心修复】：显式添加 weights_only=False
        torch.load(weight_path, map_location="cpu", weights_only=False)
        return True, f"权重文件有效: {weight_path}"
    except Exception as e:
        return False, f"权重文件安全校验拦截: {str(e)[:80]}..."

# ==========================================
# 训练超参数配置
# ==========================================
gpu_count = torch.cuda.device_count()
DEVICE = '0,1' if gpu_count >= 2 else '0' if gpu_count == 1 else 'cpu'
BATCH_SIZE = 16

# 训练参数（对齐 ziduo_test 分支）
TRAIN_ARGS = {
    'epochs': 300,
    'imgsz': 640,
    'batch': BATCH_SIZE,
    'optimizer': 'AdamW',
    'lr0': 0.0008,
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.02,
    'warmup_epochs': 3,
    'patience': 50,
    'close_mosaic': 20,
    'box': 7.5,
    'cls': 0.5,
    'dfl': 1.5,
    'hsv_v': 0.2,
    'degrees': 5.0,
    'translate': 0.08,
    'scale': 0.15,
    'flipud': 0.3,
    'fliplr': 0.5,
    'mosaic': 0.25,
    'device': DEVICE,
    'amp': False,  # 彻底关闭AMP防止校验干扰
    'plots': True,
    'cache': True
}

# ==========================================
# 主训练流程
# ==========================================
def run_experiment():
    print("\n" + "="*60)
    print("🚀 YOLO11 蒸馏后微调启动")
    print("="*60)
    
    # 修复路径环境变量
    if str(BASE_DIR) not in sys.path: sys.path.insert(0, str(BASE_DIR))
    os.environ['PYTHONPATH'] = str(BASE_DIR)

    # --- 初始化并加载模型 ---
    print("📦 初始化模型...")
    
    weight_path = None
    # 1. 尝试蒸馏权重
    if DISTILL_WEIGHTS.exists():
        is_valid, msg = validate_weight_file(DISTILL_WEIGHTS)
        print(f"{'✅' if is_valid else '⚠️'} {msg}")
        if is_valid: weight_path = str(DISTILL_WEIGHTS)

    # 2. 备选官方权重
    if weight_path is None and YOLO_WEIGHTS.exists():
        is_valid, msg = validate_weight_file(YOLO_WEIGHTS)
        print(f"{'✅' if is_valid else '⚠️'} {msg}")
        if is_valid: weight_path = str(YOLO_WEIGHTS)

    # 3. 加载逻辑
    try:
        if weight_path:
            # 这里的 YOLO 内部载入可能还会触发校验，
            # 我们直接通过修改 torch 全局函数来“降维打击”
            original_torch_load = torch.load
            def safe_torch_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return original_torch_load(*args, **kwargs)
            
            torch.load = safe_torch_load # 临时猴子补丁
            model = YOLO(weight_path)
            torch.load = original_torch_load # 还原
            print(f"🎉 成功载入权重: {weight_path}")
        else:
            print("⚠️ 未找到有效权重，使用空架构初始化")
            model = YOLO("yolo11n.yaml")
    except Exception as e:
        print(f"❌ 加载失败，回退至空架构: {e}")
        model = YOLO("yolo11n.yaml")

    # --- 开始训练 ---
    print("\n🚀 开始训练阶段...")
    model.train(data=str(DATA_YAML), **TRAIN_ARGS)

    # --- 最终验证 ---
    print("\n🔍 开始测试集最终评估...")
    try:
        best_path = Path(model.trainer.save_dir) / 'weights' / 'best.pt'
        best_model = YOLO(str(best_path))
        metrics = best_model.val(data=str(DATA_YAML), split="test", imgsz=640)
        print("\n" + "="*60)
        print(f"📊 最终测试集 mAP50: {metrics.box.map50:.4f}")
        print("="*60)
    except Exception as e:
        print(f"⚠️ 评估过程出错: {e}")

if __name__ == "__main__":
    run_experiment()