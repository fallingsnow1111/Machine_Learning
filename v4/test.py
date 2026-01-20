"""
测试脚本 - 在测试集上评估模型性能
"""

import torch
from pathlib import Path
from ultralytics import YOLO

# ==========================================
# 配置参数
# ==========================================
# 模型权重路径（按优先级）
MODEL_WEIGHTS_PATHS = [
    "./best.pt",                                    # 当前目录最佳权重
    "./runs/detect/train/weights/best.pt",          # 训练输出最佳权重
]

# 测试数据配置
TEST_DATA = "../Data/dataset_yolo/dataset.yaml"

DEVICE = '0' if torch.cuda.is_available() else 'cpu'

# ==========================================
# 主函数
# ==========================================
def run_test():
    """在测试集上评估模型"""
    
    print("="*60)
    print("🔍 YOLO11 模型测试")
    print("="*60)
    
    # 查找模型权重
    model_path = None
    for path in MODEL_WEIGHTS_PATHS:
        p = Path(path)
        if p.exists():
            model_path = p
            break
    
    if model_path is None:
        print("❌ 未找到模型权重文件")
        print("💡 请先运行训练：python train.py")
        return
    
    print(f"📦 加载模型: {model_path.absolute()}")
    model = YOLO(str(model_path))
    
    # 检查测试数据
    test_data_path = Path(TEST_DATA)
    if not test_data_path.exists():
        print(f"❌ 测试数据不存在: {TEST_DATA}")
        return
    
    print(f"📊 测试数据: {test_data_path.absolute()}")
    print(f"💻 设备: {DEVICE}")
    print()
    
    # 运行测试
    print("🚀 开始测试...")
    metrics = model.val(
        data=str(test_data_path),
        split="test",
        imgsz=640,
        batch=16,
        device=DEVICE,
        plots=True,  # 生成可视化图表
    )
    
    # 输出结果
    print("\n" + "="*60)
    print("📊 测试结果")
    print("="*60)
    print(f"mAP50:       {metrics.box.map50:.4f}")
    print(f"mAP50-95:    {metrics.box.map:.4f}")
    print(f"Precision:   {metrics.box.p.mean():.4f}")
    print(f"Recall:      {metrics.box.r.mean():.4f}")
    
    # 各类别性能
    if hasattr(metrics.box, 'ap_class_index'):
        print("\n📋 各类别 mAP50:")
        for i, class_idx in enumerate(metrics.box.ap_class_index):
            class_map = metrics.box.ap50[i]
            print(f"   Class {class_idx}: {class_map:.4f}")
    
    print("="*60)
    
    return metrics

if __name__ == "__main__":
    run_test()
