import sys
import os
from pathlib import Path
from typing import Any, Optional
import torch
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
# 导入你的自定义模块
from custom_modules import ASPP, EMA

# --- 核心：注册自定义模块 ---
def register_custom_layers():
    # ultralytics.nn.tasks 里是动态查找模块，这里用 setattr 注册
    setattr(tasks, "ASPP", ASPP)
    setattr(tasks, "EMA", EMA)
    print("✅ 已成功注册 ASPP 和 EMA 模块")

def _find_project_root(start: Path) -> Path:
    """从脚本位置向上找项目根目录（包含 Data/ 和 ultralytics/ 或 *.yaml）。"""
    current = start
    for _ in range(10):
        if (current / "Data").exists() and ((current / "ultralytics").exists() or list(current.glob("*.yaml"))):
            return current
        if current.parent == current:
            break
        current = current.parent
    return start


def _pick_dataset_yaml(project_root: Path) -> Path:
    """优先使用环境变量 DATA_YAML，否则从常见位置挑一个存在的。"""
    env_path = os.getenv("DATA_YAML")
    if env_path:
        return Path(env_path).expanduser().resolve()

    candidates = [
        project_root / "Data" / "Merged" / "no_noise11_processed" / "dataset.yaml",
        project_root / "Data" / "Merged" / "noise11_processed" / "dataset.yaml",
        project_root / "Data" / "Merged" / "no_noise11" / "dataset_merged.yaml",
        project_root / "Data" / "Merged" / "noise11" / "dataset_merged.yaml",
        project_root / "Data" / "dataset.yaml",
    ]
    for p in candidates:
        if p.exists():
            return p

    merged_dir = project_root / "Data" / "Merged"
    if merged_dir.exists():
        # 兜底：自动选一个能找到的 dataset*.yaml
        for p in merged_dir.rglob("dataset*.yaml"):
            return p.resolve()
    # 默认返回第一个，后续会报更明确的错
    return candidates[0]


# 1. 路径处理（本地/Colab/Kaggle 都尽量稳）
script_dir = Path(__file__).resolve().parent
PROJECT_ROOT = _find_project_root(script_dir)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    os.chdir(PROJECT_ROOT)
    print(f"📍 工作目录: {os.getcwd()}")
except Exception as e:
    print(f"⚠️ 切换目录失败: {e}")

# ==========================================
# 2. 对齐 Baseline 配置参数
# ==========================================
# 数据集地址在这里传入！
DATA_YAML_PATH = _pick_dataset_yaml(PROJECT_ROOT)
TRAIN_DATA = str(DATA_YAML_PATH)
VAL_DATA = str(DATA_YAML_PATH)

# 指向你那个带 ASPP/EMA/P2 的新 YAML
MODEL_CONFIG = str((PROJECT_ROOT / "yolo_ema.yaml").resolve())
PRETRAINED_WEIGHTS = str((PROJECT_ROOT / "pt" / "yolo11n.pt").resolve())
DEVICE = '0' if torch.cuda.is_available() else 'cpu'

def run_experiment():
    # 必须在初始化 YOLO 前注册模块
    register_custom_layers()

    if not Path(TRAIN_DATA).exists():
        raise FileNotFoundError(
            "找不到数据集配置文件(dataset.yaml)。\n"
            f"当前 TRAIN_DATA: {TRAIN_DATA}\n"
            "你可以：\n"
            "1) 把正确的 dataset.yaml 放到项目内；或\n"
            "2) 在运行前设置环境变量 DATA_YAML=/绝对路径/到/dataset.yaml\n"
            "Kaggle 上也要确保 images/train 与 images/val 目录真实存在。"
        )

    if not Path(MODEL_CONFIG).exists():
        raise FileNotFoundError(f"找不到模型结构 YAML: {MODEL_CONFIG}")

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
    save_dir: Optional[str] = getattr(results, "save_dir", None)
    if not save_dir:
        raise RuntimeError("训练未返回 save_dir，无法定位 best.pt")
    best_model_path = os.path.join(save_dir, 'weights', 'best.pt')
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