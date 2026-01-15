from pathlib import Path
import sys
import subprocess

# 自动安装必要的包
def install_package(package_name):
    """自动安装Python包"""
    try:
        __import__(package_name.split('[')[0].replace('-', '_'))
        print(f"✓ {package_name} 已安装")
    except ImportError:
        print(f"正在安装 {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✓ {package_name} 安装完成")

# 检查并安装依赖
print("检查依赖包...")
required_packages = [
    'lightly-train',
    'torch',
    'torchvision', 
    'ultralytics',
    'timm',
    'pyyaml',
    'tqdm'
]

for package in required_packages:
    install_package(package)

print("\n所有依赖已准备就绪！\n")

import lightly_train
from ultralytics import YOLO

if __name__ == "__main__":
    # 设置项目根目录
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    
    # 数据路径（相对于项目根目录）
    DATA_DIR = PROJECT_ROOT / "Data/Raw/dust"
    DATASET_YAML = DATA_DIR / "dataset.yaml"
    
    # 输出路径
    OUT_DIR = PROJECT_ROOT / "runs/distillation/dinov3_to_yolo11"
    
    print("="*60)
    print("🚀 开始 DINO v3 -> YOLO11 知识蒸馏")
    print("="*60)
    print(f"📂 数据目录: {DATA_DIR}")
    print(f"📂 输出目录: {OUT_DIR}")
    print("="*60 + "\n")
    
    # 第一步：使用 lightly-train 进行预训练/蒸馏
    print("步骤 1/3: 知识蒸馏...")
    lightly_train.pretrain(
        out=str(OUT_DIR),
        data=str(DATA_DIR),  # 数据目录（包含 images/ 文件夹）
        model="ultralytics/yolo11n",  # YOLO11 nano 作为学生模型
        method="distillation",
        method_args={
            "teacher": "dinov3/vitb16",  # DINO v3 base 作为教师模型
            # 可选：调整蒸馏温度和损失权重
            # "temperature": 0.07,
            # "distillation_weight": 0.5,
        },
        epochs=100,
        batch_size=16,
        # 可选：添加更多训练参数
        # learning_rate=1e-4,
        # weight_decay=0.05,
    )
    
    print("\n" + "="*60)
    print("✅ 蒸馏完成！")
    print("="*60 + "\n")
    
    # 第二步：加载蒸馏后的模型
    print("步骤 2/3: 加载蒸馏模型并微调...")
    exported_model_path = OUT_DIR / "exported_models/exported_last.pt"
    
    if not exported_model_path.exists():
        print(f"⚠️ 找不到导出的模型: {exported_model_path}")
        print("请检查蒸馏是否成功完成")
        sys.exit(1)
    
    model = YOLO(str(exported_model_path))
    print(f"✅ 已加载模型: {exported_model_path}")
    
    # 第三步：在目标检测任务上微调
    print("\n开始微调...")
    results = model.train(
        data=str(DATASET_YAML),
        epochs=50,
        imgsz=640,  # 建议使用 640，而不是 64
        batch=16,
        device='0' if __import__('torch').cuda.is_available() else 'cpu',
        # 微调时使用较小的学习率
        lr0=0.0001,
        lrf=0.01,
        warmup_epochs=3,
        # 优化器设置
        optimizer='AdamW',
        weight_decay=0.0001,
        # 项目名称
        project=str(PROJECT_ROOT / "runs/detect"),
        name="distilled_yolo11",
        # 保存设置
        patience=10,
        save=True,
        plots=True,
    )
    
    print("\n" + "="*60)
    print("✅ 微调完成！")
    print("="*60 + "\n")
    
    # 第四步：在测试集上评估
    print("步骤 3/3: 评估模型性能...")
    
    # 加载最佳模型
    best_model_path = results.save_dir / "weights/best.pt"
    best_model = YOLO(str(best_model_path))
    
    # 在测试集上验证
    val_results = best_model.val(
        data=str(DATASET_YAML),
        split='test',
        imgsz=640,
        batch=16,
        device='0' if __import__('torch').cuda.is_available() else 'cpu',
    )
    
    print("\n" + "="*60)
    print("📊 最终测试集结果")
    print("="*60)
    print(f"mAP50: {val_results.box.map50:.4f}")
    print(f"mAP50-95: {val_results.box.map:.4f}")
    print(f"Precision: {val_results.box.p:.4f}")
    print(f"Recall: {val_results.box.r:.4f}")
    print("="*60)
    print(f"✅ 最佳模型已保存至: {best_model_path}")
    print("="*60)