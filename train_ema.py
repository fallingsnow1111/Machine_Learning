import sys
import os
import torch
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
# 导入你的自定义模块
from custom_modules import ASPP, EMA

# --- 核心：注册自定义模块 ---
def register_custom_layers():
    tasks.ASPP = ASPP
    tasks.EMA = EMA
    print("✅ 已成功注册 ASPP 和 EMA 模块")

# 1. 路径处理
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    os.chdir(project_root)
    print(f"📍 工作目录: {os.getcwd()}")
except Exception as e:
    print(f"⚠️ 切换目录失败: {e}")

# ==========================================
# 2. 对齐 Baseline 配置参数
# ==========================================
# 数据集地址在这里传入！
TRAIN_DATA = "./Data/Merged/no_noise11_processed/dataset.yaml"
VAL_DATA = "./Data/Merged/no_noise11_processed/dataset.yaml" 
# 指向你那个带 ASPP/EMA/P2 的新 YAML
MODEL_CONFIG = "./yolo_ema.yaml" 
PRETRAINED_WEIGHTS = "./pt/yolo11n.pt"
DEVICE = '0' if torch.cuda.is_available() else 'cpu'

def run_experiment():
    # 必须在初始化 YOLO 前注册模块
    register_custom_layers()

    # --- 第一步：加载新结构模型 ---
    model = YOLO(MODEL_CONFIG)

    # 尝试加载预训练权重 
    # 注意：因为你改了结构（多了P2和ASPP），预训练权重只能加载骨干网部分，这是正常的
    try:
        model.load(PRETRAINED_WEIGHTS)
        print("✅ 成功加载预训练权重（部分匹配）")
    except Exception as e:
        print(f"⚠️ 权重加载提示: {e}")

    # --- 第二步：开始训练 (参数完全同步 Baseline) ---
    print("\n🚀 开始训练阶段...")
    results = model.train(
        data=TRAIN_DATA,
        epochs=50,
        imgsz=1024,      # 注意：你之前提到用1024，建议这里改为1024以匹配小目标需求
        batch=16,        # 如果显存不够，请调回 16
        patience=0, 
        optimizer='AdamW',
        lr0=0.0005,      # 保持你的 Baseline 参数
        lrf=0.01,
        warmup_epochs=5.0,
        translate=0.05,
        scale=0.1,
        copy_paste=0.4,
        device=DEVICE,
        plots=True,
        dropout=0.2,
    )

    # --- 第三步：自动验证 ---
    print("\n🔍 开始验证阶段...")
    best_model_path = os.path.join(results.save_dir, 'weights', 'best.pt')
    best_model = YOLO(best_model_path)

    metrics = best_model.val(
        data=VAL_DATA,
        split="test", 
        imgsz=1024,     # 验证尺寸也要和训练保持一致
        batch=16,
        device=DEVICE
    )

    print(f"\n最终测试集结果 (mAP50): {metrics.box.map50:.4f}")

if __name__ == "__main__":
    run_experiment()