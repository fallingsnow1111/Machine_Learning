from pathlib import Path
import sys
import subprocess
import os
import shutil

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
    'tqdm',
    'opencv-python',
    'matplotlib',
    'numpy'
]

for package in required_packages:
    install_package(package)

print("\n所有依赖已准备就绪！\n")

import cv2
import numpy as np
from tqdm import tqdm
import lightly_train
from ultralytics import YOLO
import torch

# ==================== 预处理模块 ====================
def process_image_channels(img_path_str, target_size=(640, 640)):
    """
    处理单张图像：64x64灰度图 -> 640x640三通道图
    Ch0=原图Lanczos放大, Ch1=双边滤波, Ch2=CLAHE
    """
    img_gray = cv2.imread(img_path_str, 0)
    if img_gray is None:
        return None

    # 1. Lanczos 插值放大 (64 -> 640)
    img_upscaled = cv2.resize(img_gray, target_size, interpolation=cv2.INTER_LANCZOS4)

    # 2. 构建三个通道
    # Ch0: 原始放大
    c0 = img_upscaled
    
    # Ch1: 双边滤波 (降噪保边)
    c1 = cv2.bilateralFilter(img_upscaled, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Ch2: CLAHE (对比度增强)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    c2 = clahe.apply(img_upscaled)

    # 3. 合并为三通道 (BGR顺序)
    merged_img = cv2.merge([c0, c1, c2])
    return merged_img

def preprocess_dataset(input_dir, output_dir, target_size=(640, 640)):
    """预处理整个数据集"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 清空输出目录（如果存在）
    if output_path.exists():
        print(f"清空现有输出目录: {output_path}")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    files = [f for f in input_path.rglob('*') if f.is_file()]
    processed_count = 0

    print(f"\n开始预处理数据集: {input_dir}")
    print(f"目标尺寸: {target_size}")
    print(f"找到 {len(files)} 个文件\n")
    
    for file_path in tqdm(files, desc="预处理进度"):
        rel_path = file_path.relative_to(input_path)
        target_path = output_path / rel_path
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif']:
            img = process_image_channels(str(file_path), target_size)
            if img is not None:
                # 保存为 PNG 以保留质量
                save_path = target_path.with_suffix('.png')
                cv2.imwrite(str(save_path), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                processed_count += 1
        else:
            # 标签文件直接复制
            shutil.copy2(file_path, target_path)
    
    print(f"\n预处理完成！共处理 {processed_count} 张图像")
    return processed_count

def create_dataset_yaml(output_dir, classes=['dust']):
    """生成 dataset.yaml"""
    yaml_path = Path(output_dir) / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(f"path: {output_dir}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n\n")
        f.write(f"nc: {len(classes)}\n")
        f.write("names: " + str(classes) + "\n")
    print(f"✅ 已生成配置文件: {yaml_path}")
    return yaml_path

# ==================== 主训练流程 ====================
if __name__ == "__main__":
    # 设置项目根目录
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    
    # 原始数据路径（64×64灰度图）
    RAW_DATA_DIR = PROJECT_ROOT / "Data/Raw/dust"
    
    # 预处理后数据路径（640×640三通道图）
    PROCESSED_DATA_DIR = PROJECT_ROOT / "Data/Processed/dust_640x640"
    
    # 蒸馏输出路径
    DISTILL_OUT_DIR = PROJECT_ROOT / "runs/distillation/dinov3_to_yolo11_640"
    
    print("="*70)
    print("🚀 完整工作流程：预处理 -> 蒸馏 -> 微调")
    print("="*70)
    print(f"📂 原始数据: {RAW_DATA_DIR}")
    print(f"📂 处理后数据: {PROCESSED_DATA_DIR}")
    print(f"📂 蒸馏输出: {DISTILL_OUT_DIR}")
    print(f"🖥️  设备: {'GPU (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
    print("="*70 + "\n")
    
    # ==================== 步骤 1: 数据预处理 ====================
    print("\n" + "="*70)
    print("步骤 1/4: 数据预处理 (64x64灰度 -> 640x640三通道)")
    print("="*70)
    
    processed_count = preprocess_dataset(
        input_dir=str(RAW_DATA_DIR),
        output_dir=str(PROCESSED_DATA_DIR),
        target_size=(640, 640)
    )
    
    if processed_count == 0:
        print("❌ 预处理失败，没有处理任何图像！")
        sys.exit(1)
    
    # 生成 dataset.yaml
    DATASET_YAML = create_dataset_yaml(PROCESSED_DATA_DIR)
    
    # ==================== 步骤 2: 知识蒸馏 ====================
    print("\n" + "="*70)
    print("步骤 2/4: DINO v3 -> YOLO11 知识蒸馏")
    print("="*70)
    
    try:
        lightly_train.pretrain(
            out=str(DISTILL_OUT_DIR),
            data=str(PROCESSED_DATA_DIR),
            model="ultralytics/yolo11n",
            method="distillation",
            method_args={
                "teacher": "dinov3/vits16",  # 使用小版本DINO
            },
            epochs=100,  # 640尺寸可以少训练些轮次
            batch_size=8,  # 640尺寸需要减小batch
        )
        print("\n✅ 蒸馏完成！")
    except Exception as e:
        print(f"\n❌ 蒸馏失败: {e}")
        print("尝试跳过蒸馏，直接使用预训练模型...")
        DISTILL_OUT_DIR = None
    
    # ==================== 步骤 3: 加载模型 ====================
    print("\n" + "="*70)
    print("步骤 3/4: 加载模型并准备微调")
    print("="*70)
    
    if DISTILL_OUT_DIR and (DISTILL_OUT_DIR / "exported_models/exported_last.pt").exists():
        exported_model_path = DISTILL_OUT_DIR / "exported_models/exported_last.pt"
        print(f"✅ 使用蒸馏模型: {exported_model_path}")
        model = YOLO(str(exported_model_path))
    else:
        print("⚠️ 未找到蒸馏模型，使用官方预训练模型")
        model = YOLO('yolo11n.pt')
    
    # ==================== 步骤 4: 微调训练 ====================
    print("\n" + "="*70)
    print("步骤 4/4: 微调训练（针对640×640灰尘检测优化）")
    print("="*70)
    
    results = model.train(
        data=str(DATASET_YAML),
        epochs=200,
        imgsz=640,  # 使用处理后的 640×640
        batch=8,    # 640尺寸需要小batch
        device='0' if torch.cuda.is_available() else 'cpu',
        
        # 学习率设置
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=5,
        
        # 优化器
        optimizer='AdamW',
        weight_decay=0.0005,
        
        # 数据增强（针对三通道处理后的图像）
        hsv_h=0.0,    # 不是真彩色，关闭色调
        hsv_s=0.0,    # 不是真彩色，关闭饱和度
        hsv_v=0.2,    # 轻微亮度增强
        degrees=10,   # 旋转
        translate=0.1,
        scale=0.2,
        shear=0.0,
        perspective=0.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.3,   # 减少mosaic（已经放大过了）
        mixup=0.0,    # 不用mixup
        copy_paste=0.0,
        
        # 小目标优化
        close_mosaic=50,
        
        # 损失权重（针对小目标灰尘）
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        # IoU设置
        iou=0.6,
        
        # 保存设置
        project=str(PROJECT_ROOT / "runs/detect"),
        name="yolo11_dust_640_distilled",
        patience=30,
        save=True,
        save_period=10,
        plots=True,
        val=True,
        
        # 其他
        amp=True,
        fraction=1.0,
        overlap_mask=True,
        mask_ratio=4,
    )
    
    # ==================== 步骤 5: 评估 ====================
    print("\n" + "="*70)
    print("最终评估")
    print("="*70)
    
    best_model_path = results.save_dir / "weights/best.pt"
    best_model = YOLO(str(best_model_path))
    
    val_results = best_model.val(
        data=str(DATASET_YAML),
        split='test',
        imgsz=640,
        batch=8,
        device='0' if torch.cuda.is_available() else 'cpu',
        conf=0.001,  # 低置信度阈值
        iou=0.5,
        max_det=100,
        plots=True,
    )
    
    print("\n" + "="*70)
    print("📊 最终测试结果")
    print("="*70)
    print(f"mAP50:      {val_results.box.map50:.4f}")
    print(f"mAP50-95:   {val_results.box.map:.4f}")
    print(f"Precision:  {val_results.box.mp:.4f}")
    print(f"Recall:     {val_results.box.mr:.4f}")
    print("="*70)
    print(f"✅ 最佳模型: {best_model_path}")
    print(f"✅ 处理后数据: {PROCESSED_DATA_DIR}")
    print("="*70)
    
    print("\n💡 使用建议:")
    print("1. 推理时使用: best.pt 模型")
    print("2. 输入图像: 需要经过相同的预处理 (64->640, 三通道)")
    print("3. 置信度阈值: 建议从 0.001 开始调整")
    print("4. 如果效果不好，可以:")
    print("   - 增加训练数据（特别是正样本）")
    print("   - 调整预处理参数（CLAHE、双边滤波）")
    print("   - 尝试 yolo11n-seg 分割模型")
    print("   - 使用异常检测方法（PaDiM/PatchCore）")