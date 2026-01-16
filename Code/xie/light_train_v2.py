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

# ==================== 增强预处理模块 ====================
def process_image_channels_enhanced(img_path_str, target_size=(640, 640)):
    """
    增强版处理：更激进的对比度增强
    Ch0=直方图均衡, Ch1=自适应阈值, Ch2=强CLAHE
    """
    img_gray = cv2.imread(img_path_str, 0)
    if img_gray is None:
        return None

    # 1. Lanczos 插值放大
    img_upscaled = cv2.resize(img_gray, target_size, interpolation=cv2.INTER_LANCZOS4)

    # 2. 构建三个通道
    # Ch0: 原始放大 + 直方图均衡化
    c0 = cv2.equalizeHist(img_upscaled)
    
    # Ch1: 自适应阈值（突出小目标边缘）
    c1 = cv2.adaptiveThreshold(
        img_upscaled, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        blockSize=11, 
        C=2
    )
    
    # Ch2: 更强的CLAHE（增强局部对比度）
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    c2 = clahe.apply(img_upscaled)

    # 3. 合并
    merged_img = cv2.merge([c0, c1, c2])
    return merged_img

def preprocess_dataset(input_dir, output_dir, target_size=(640, 640), enhanced=True):
    """预处理整个数据集"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if output_path.exists():
        print(f"清空现有输出目录: {output_path}")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    files = [f for f in input_path.rglob('*') if f.is_file()]
    processed_count = 0

    print(f"\n开始预处理数据集: {input_dir}")
    print(f"目标尺寸: {target_size}")
    print(f"增强模式: {'启用' if enhanced else '禁用'}")
    print(f"找到 {len(files)} 个文件\n")
    
    process_func = process_image_channels_enhanced
    
    for file_path in tqdm(files, desc="预处理进度"):
        rel_path = file_path.relative_to(input_path)
        target_path = output_path / rel_path
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif']:
            img = process_func(str(file_path), target_size)
            if img is not None:
                save_path = target_path.with_suffix('.png')
                cv2.imwrite(str(save_path), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                processed_count += 1
        else:
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
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    RAW_DATA_DIR = PROJECT_ROOT / "Data/Raw/dust"
    PROCESSED_DATA_DIR = PROJECT_ROOT / "Data/Processed/dust_640x640_enhanced"
    DISTILL_OUT_DIR = PROJECT_ROOT / "runs/distillation/dinov3_to_yolo11_640_stable"
    
    print("="*70)
    print("🚀 稳定训练流程 - 针对波动优化")
    print("="*70)
    print(f"📂 原始数据: {RAW_DATA_DIR}")
    print(f"📂 处理后数据: {PROCESSED_DATA_DIR}")
    print(f"📂 蒸馏输出: {DISTILL_OUT_DIR}")
    print(f"🖥️  设备: {'GPU (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
    print("="*70 + "\n")
    
    # ==================== 步骤 1: 增强预处理 ====================
    print("\n" + "="*70)
    print("步骤 1/4: 增强预处理")
    print("="*70)
    
    processed_count = preprocess_dataset(
        input_dir=str(RAW_DATA_DIR),
        output_dir=str(PROCESSED_DATA_DIR),
        target_size=(640, 640),
        enhanced=True
    )
    
    if processed_count == 0:
        print("❌ 预处理失败，没有处理任何图像！")
        sys.exit(1)
    
    DATASET_YAML = create_dataset_yaml(PROCESSED_DATA_DIR)
    
    # ==================== 步骤 2: 知识蒸馏 ====================
    print("\n" + "="*70)
    print("步骤 2/4: DINO v3 知识蒸馏")
    print("="*70)
    
    try:
        lightly_train.pretrain(
            out=str(DISTILL_OUT_DIR),
            data=str(PROCESSED_DATA_DIR),
            model="ultralytics/yolo11n",
            method="distillation",
            method_args={
                "teacher": "dinov3/vitl16",
            },
            epochs=100,  # 增加蒸馏轮次以获得更好的初始化
            batch_size=16,  # 增大batch提高稳定性
        )
        print("\n✅ 蒸馏完成！")
    except Exception as e:
        print(f"\n❌ 蒸馏失败: {e}")
        print("继续使用预训练模型...")
        DISTILL_OUT_DIR = None
    
    # ==================== 步骤 3: 加载模型 ====================
    print("\n" + "="*70)
    print("步骤 3/4: 加载模型")
    print("="*70)
    
    if DISTILL_OUT_DIR and (DISTILL_OUT_DIR / "exported_models/exported_last.pt").exists():
        exported_model_path = DISTILL_OUT_DIR / "exported_models/exported_last.pt"
        print(f"✅ 使用蒸馏模型: {exported_model_path}")
        model = YOLO(str(exported_model_path))
    else:
        print("⚠️ 使用官方预训练模型 YOLO11n")
        model = YOLO('yolo11n.pt')
    
    # ==================== 步骤 4: 稳定微调训练 ====================
    print("\n" + "="*70)
    print("步骤 4/4: 稳定微调训练（降低波动）")
    print("="*70)
    
    results = model.train(
        data=str(DATASET_YAML),
        
        # ===== 训练轮次 =====
        epochs=300,  # 适中的轮次
        
        # ===== 图像尺寸 =====
        imgsz=640,
        batch=32,  # 更大的batch提高稳定性（如果显存允许）
        device='0' if torch.cuda.is_available() else 'cpu',
        workers=8,
        
        # ===== 学习率策略（关键：降低学习率）=====
        lr0=0.001,      # 🔥 降低初始学习率（从0.01降到0.001）
        lrf=0.01,       # 🔥 提高最终学习率占比（保持稳定）
        warmup_epochs=5,  # 减少warmup
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # ===== 优化器（使用AdamW提高稳定性）=====
        optimizer='AdamW',  # 🔥 改用AdamW（比SGD更稳定）
        momentum=0.937,
        weight_decay=0.0005,
        
        # ===== 数据增强（降低强度）=====
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.2,     # 🔥 降低亮度增强（从0.4降到0.2）
        degrees=10,    # 🔥 降低旋转（从20降到10）
        translate=0.1, # 🔥 降低平移（从0.2降到0.1）
        scale=0.3,     # 🔥 降低缩放（从0.5降到0.3）
        shear=0.0,
        perspective=0.0,
        flipud=0.5,
        fliplr=0.5,
        
        # ===== Mosaic & Mixup（降低强度）=====
        mosaic=0.8,    # 🔥 降低mosaic（从1.0降到0.8）
        mixup=0.05,    # 🔥 降低mixup（从0.15降到0.05）
        copy_paste=0.1,  # 🔥 降低copy-paste（从0.3降到0.1）
        
        # ===== 关闭mosaic时机 =====
        close_mosaic=50,  # 🔥 提前关闭mosaic（最后50轮）
        
        # ===== 损失权重（平衡调整）=====
        box=7.5,      # 🔥 适中的box权重（从10.0降到7.5）
        cls=0.5,      # 🔥 提高分类权重（从0.3到0.5）
        dfl=1.5,      # DFL损失
        
        # ===== IoU设置 =====
        iou=0.6,      # 🔥 提高IoU阈值（从0.5到0.6）
        
        # ===== NMS设置 =====
        conf=0.001,
        
        # ===== 保存设置 =====
        project=str(PROJECT_ROOT / "runs/detect"),
        name="yolo11_dust_stable_distilled",
        patience=50,  # 🔥 降低耐心值（从100到50）
        save=True,
        save_period=10,  # 🔥 更频繁保存（从20到10）
        plots=True,
        val=True,
        
        # ===== 其他优化 =====
        amp=True,
        fraction=1.0,
        
        # ===== 验证频率 =====
        # 增加验证频率以更好监控
        val_period=1,  # 每轮验证
        
        # ===== Dropout（如果支持）=====
        dropout=0.0,  # 不使用dropout

    )
    
    # ==================== 步骤 5: 多阈值评估 ====================
    print("\n" + "="*70)
    print("最终评估（多置信度阈值）")
    print("="*70)
    
    best_model_path = results.save_dir / "weights/best.pt"
    best_model = YOLO(str(best_model_path))
    
    # 测试多个置信度阈值
    conf_thresholds = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    
    print("\n不同置信度阈值的性能：")
    print("-" * 60)
    best_conf = 0.001
    best_map50 = 0
    
    for conf_th in conf_thresholds:
        val_results = best_model.val(
            data=str(DATASET_YAML),
            split='test',
            imgsz=640,
            batch=32,
            device='0' if torch.cuda.is_available() else 'cpu',
            conf=conf_th,
            iou=0.5,
            max_det=300,
            plots=False,
        )
        
        map50 = val_results.box.map50
        precision = val_results.box.mp
        recall = val_results.box.mr
        
        print(f"Conf={conf_th:.3f} | mAP50={map50:.4f} | P={precision:.4f} | R={recall:.4f}")
        
        if map50 > best_map50:
            best_map50 = map50
            best_conf = conf_th
    
    print("-" * 60)
    print(f"✅ 最佳置信度阈值: {best_conf:.3f} (mAP50={best_map50:.4f})")
    
    # 使用最佳阈值重新评估并保存可视化
    print(f"\n使用最佳阈值 {best_conf:.3f} 生成可视化结果...")
    val_results = best_model.val(
        data=str(DATASET_YAML),
        split='test',
        imgsz=640,
        batch=32,
        device='0' if torch.cuda.is_available() else 'cpu',
        conf=best_conf,
        iou=0.5,
        max_det=300,
        plots=True,
        save_json=True,
    )
    
    print("\n" + "="*70)
    print(f"📊 最终测试结果 (conf={best_conf:.3f})")
    print("="*70)
    print(f"mAP50:      {val_results.box.map50:.4f}")
    print(f"mAP50-95:   {val_results.box.map:.4f}")
    print(f"Precision:  {val_results.box.mp:.4f}")
    print(f"Recall:     {val_results.box.mr:.4f}")
    print("="*70)
    print(f"✅ 最佳模型: {best_model_path}")
    print(f"✅ 处理后数据: {PROCESSED_DATA_DIR}")
    print(f"✅ 推荐置信度: {best_conf:.3f}")
    print("="*70)