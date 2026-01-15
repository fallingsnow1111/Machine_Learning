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
import torch

if __name__ == "__main__":
    # 设置项目根目录
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    
    # 数据路径（相对于项目根目录）
    DATA_DIR = PROJECT_ROOT / "Data/Raw/dust"
    DATASET_YAML = DATA_DIR / "dataset.yaml"
    
    # 输出路径
    OUT_DIR = PROJECT_ROOT / "runs/distillation/dinov3_to_yolo11_64x64_gray"
    
    print("="*60)
    print("🚀 开始 DINO v3 -> YOLO11 知识蒸馏")
    print("   专用于 64×64 灰度图像的灰尘检测")
    print("="*60)
    print(f"📂 数据目录: {DATA_DIR}")
    print(f"📂 输出目录: {OUT_DIR}")
    print(f"🖥️  设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print("="*60 + "\n")
    
    # 第一步：使用 lightly-train 进行预训练/蒸馏
    print("步骤 1/3: 知识蒸馏（针对小尺寸图像优化）...")
    lightly_train.pretrain(
        out=str(OUT_DIR),
        data=str(DATA_DIR),
        model="ultralytics/yolo11n",  # 使用 nano 模型（最小）
        method="distillation",
        method_args={
            "teacher": "dinov3/vits16",  # 使用小版本的 DINO（更适合小图）
            "temperature": 0.1,  # 降低温度，增强特征学习
            "distillation_weight": 0.7,  # 增加蒸馏权重
        },
        epochs=200,  # 小数据集需要更多轮次
        batch_size=32,  # 小图像可以用更大 batch
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
    
    # 第三步：在目标检测任务上微调（专门针对小目标优化）
    print("\n开始微调（小目标检测优化）...")
    results = model.train(
        data=str(DATASET_YAML),
        epochs=300,  # 小图像需要更多训练轮次
        imgsz=64,  # 保持原始 64×64 尺寸
        batch=32,  # 小图像可以用更大 batch
        device='0' if torch.cuda.is_available() else 'cpu',
        
        # 学习率设置（小目标需要更细致的学习）
        lr0=0.001,  # 初始学习率稍高
        lrf=0.001,  # 最终学习率降低
        warmup_epochs=10,
        warmup_momentum=0.5,
        
        # 优化器设置
        optimizer='AdamW',
        weight_decay=0.001,
        momentum=0.9,
        
        # 数据增强（针对灰尘检测优化）
        hsv_h=0.0,  # 灰度图不需要色调增强
        hsv_s=0.0,  # 灰度图不需要饱和度增强
        hsv_v=0.3,  # 适度的亮度增强
        degrees=15,  # 旋转增强
        translate=0.1,  # 平移增强
        scale=0.3,  # 缩放增强
        shear=0.0,  # 灰尘检测不需要剪切
        perspective=0.0,  # 64×64 太小，不需要透视
        flipud=0.5,  # 上下翻转
        fliplr=0.5,  # 左右翻转
        mosaic=0.5,  # 适度的 mosaic 增强
        mixup=0.1,  # 轻微的 mixup
        copy_paste=0.0,  # 不使用复制粘贴
        
        # 小目标检测优化
        close_mosaic=100,  # 最后 100 轮关闭 mosaic
        
        # 损失函数权重（针对小目标）
        box=7.5,  # 增加边界框损失权重
        cls=0.5,  # 分类损失（如果只有灰尘一类，可以降低）
        dfl=1.5,  # DFL 损失
        
        # IoU 设置
        iou=0.7,  # IoU 训练阈值
        
        # 项目名称
        project=str(PROJECT_ROOT / "runs/detect"),
        name="distilled_yolo11_dust_64x64",
        
        # 保存设置
        patience=50,  # 增加耐心值
        save=True,
        save_period=10,  # 每 10 轮保存一次
        plots=True,
        
        # 验证设置
        val=True,
        
        # 其他优化
        amp=True,  # 混合精度训练（如果 GPU 支持）
        fraction=1.0,  # 使用全部数据
        
        # 小目标特定设置
        overlap_mask=True,  # 允许重叠
        mask_ratio=4,  # mask 比例
    )
    
    print("\n" + "="*60)
    print("✅ 微调完成！")
    print("="*60 + "\n")
    
    # 第四步：在测试集上评估
    print("步骤 3/3: 评估模型性能...")
    
    # 加载最佳模型
    best_model_path = results.save_dir / "weights/best.pt"
    best_model = YOLO(str(best_model_path))
    
    # 在测试集上验证（针对小目标优化）
    val_results = best_model.val(
        data=str(DATASET_YAML),
        split='test',
        imgsz=64,  # 保持 64×64
        batch=32,
        device='0' if torch.cuda.is_available() else 'cpu',
        conf=0.001,  # 降低置信度阈值（小目标容易漏检）
        iou=0.5,  # IoU 阈值
        max_det=100,  # 每张图最多检测数（灰尘可能很多）
        plots=True,
    )
    
    print("\n" + "="*60)
    print("📊 最终测试集结果（64×64 灰度图像）")
    print("="*60)
    print(f"mAP50: {val_results.box.map50:.4f}")
    print(f"mAP50-95: {val_results.box.map:.4f}")
    print(f"Precision: {val_results.box.mp:.4f}")
    print(f"Recall: {val_results.box.mr:.4f}")
    print("="*60)
    print(f"✅ 最佳模型已保存至: {best_model_path}")
    print("="*60)
    
    # 额外建议
    print("\n💡 针对 OLED 灰尘检测的建议:")
    print("1. 如果效果仍不理想，考虑:")
    print("   - 使用异常检测方法（PaDiM/PatchCore）")
    print("   - 尝试分割模型: YOLO11n-seg")
    print("   - 增加正样本（含灰尘）的数量")
    print("2. 灰度图像处理:")
    print("   - 使用 CLAHE 对比度增强预处理")
    print("   - 确保数据集图像质量一致")
    print("3. 小目标检测:")
    print("   - 降低推理时的 conf 阈值到 0.001-0.01")
    print("   - 使用 TTA (Test Time Augmentation)")
