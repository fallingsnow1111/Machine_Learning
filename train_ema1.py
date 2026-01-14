import sys
import os
import torch
from pathlib import Path
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
# 导入你的自定义模块
from custom_modules import ASPP, EMA

# --- 1. 注册自定义模块 (这是必须保留的硬核心) ---
def register_custom_layers():
    setattr(tasks, "ASPP", ASPP)
    setattr(tasks, "EMA", EMA)
    print("✅ 已成功注册 ASPP 和 EMA 模块")

# --- 2. 手动指定你的项目根目录 ---
# train_ema1.py 在项目根目录，所以 parent 就是根目录
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)
print(f"📍 当前工作目录已锁定为: {os.getcwd()}")

# ==========================================
# 3. 直观配置区 (在这里改地址，最直接)
# ==========================================
# 数据集地址：直接写你确认存在的那个路径
DATA_YAML = "Data/Raw/dust/dataset.yaml"

# 模型配置：确保文件名对得上
MODEL_CONFIG = "yolo_ema.yaml" 
PRETRAINED_WEIGHTS = "pt/yolo11n.pt"

DEVICE = '0' if torch.cuda.is_available() else 'cpu'

def run_experiment():
    register_custom_layers()

    # 检查文件是否存在，不存在直接报错，不搞“自动寻找”
    if not os.path.exists(DATA_YAML):
        print(f"❌ 错误：找不到数据集文件 {DATA_YAML}")
        return

    # --- 第一步：加载模型 ---
    model = YOLO(MODEL_CONFIG)
    try:
        model.load(PRETRAINED_WEIGHTS)
        print("✅ 预训练权重加载尝试完成")
    except Exception as e:
        print(f"⚠️ 权重加载提示: {e}")

    # --- 第二步：开始训练 (参数完全对齐你的 Baseline) ---
    results = model.train(
        data=DATA_YAML,
        epochs=50,
        imgsz=640,       # 如果你想严格对标 Baseline，就用 640
        batch=16,
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
    )

    # --- 第三步：验证 ---
    # 训练完后结果在 results.save_dir
    best_model_path = os.path.join(results.save_dir, 'weights', 'best.pt')
    best_model = YOLO(best_model_path)
    
    metrics = best_model.val(
        data=DATA_YAML,
        split="test", 
        imgsz=640, 
        batch=16,
        device=DEVICE
    )
    print(f"🚀 实验完成！测试集 mAP50: {metrics.box.map50:.4f}")

if __name__ == "__main__":
    run_experiment()