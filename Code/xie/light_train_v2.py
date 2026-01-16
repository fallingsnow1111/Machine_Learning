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
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))  # 增加clipLimit，减小tileSize
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
    
    process_func = process_image_channels_enhanced if enhanced else process_image_channels
    
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
    # 设置项目根目录
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    
    # 原始数据路径（64×64灰度图）
    RAW_DATA_DIR = PROJECT_ROOT / "Data/Raw/dust"
    
    # 预处理后数据路径（640×640三通道图）
    PROCESSED_DATA_DIR = PROJECT_ROOT / "Data/Processed/dust_640x640_enhanced"
    
    # 蒸馏输出路径
    DISTILL_OUT_DIR = PROJECT_ROOT / "runs/distillation/dinov3_to_yolo11_640_v2"
    
    # ==================== 步骤 1: 增强预处理 ====================
    print("\n" + "="*70)
    print("步骤 1/4: 增强预处理 (更强的对比度增强)")
    print("="*70)
    
    processed_count = preprocess_dataset(
        input_dir=str(RAW_DATA_DIR),
        output_dir=str(PROCESSED_DATA_DIR),
        target_size=(640, 640),
        enhanced=True  # 启用增强模式
    )
    
    if processed_count == 0:
        print("❌ 预处理失败，没有处理任何图像！")
        sys.exit(1)
    
    # 生成 dataset.yaml
    DATASET_YAML = create_dataset_yaml(PROCESSED_DATA_DIR)
    
    # ==================== 步骤 2: 知识蒸馏 ====================
    print("\n" + "="*70)
    print("步骤 2/4: 知识蒸馏")
    print("="*70)
    
    SKIP_DISTILLATION = False  # 设为 False 启用蒸馏
    
    if not SKIP_DISTILLATION:
        try:
            lightly_train.pretrain(
                out=str(DISTILL_OUT_DIR),
                data=str(PROCESSED_DATA_DIR),
                model="ultralytics/yolo11n",
                method="distillation",
                method_args={
                    "teacher": "dinov3/vitl16",
                },
                epochs=100,  # 减少蒸馏轮次，更多时间用于微调
                batch_size=16,
            )
            print("\n✅ 蒸馏完成！")
        except Exception as e:
            print(f"\n❌ 蒸馏失败: {e}")
            DISTILL_OUT_DIR = None
    else:
        print("⚠️ 跳过蒸馏步骤，直接使用预训练模型")
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
        model = YOLO('yolo11n.pt')
    
    # ==================== 步骤 4: 优化的微调训练 ====================
    print("\n" + "="*70)
    print("步骤 4/4: 优化微调（针对小目标&灰尘检测）")
    print("="*70)
    
    results = model.train(
        data=str(DATASET_YAML),
        
        # ===== 训练轮次 =====
        epochs=500,  # 增加训练轮次
        
        # ===== 图像尺寸 =====
        imgsz=640,
        batch=16,
        device='0' if torch.cuda.is_available() else 'cpu',
        workers=8,  # 增加数据加载线程
        
        # ===== 学习率策略=====
        lr0=0.001,      # 提高初始学习率
        lrf=0.01,    # 降低最终学习率
        warmup_epochs=10,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # ===== 优化器 =====
        optimizer='AdamW', 
        momentum=0.937,
        weight_decay=0.0005,
        
        # ===== 数据增强=====
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.2,     # 增加亮度增强
        degrees=10,    # 增加旋转
        translate=0.2, # 增加平移
        scale=0.5,     # 增加缩放
        shear=0.0,
        perspective=0.0,
        flipud=0.5,
        fliplr=0.5,
        
        # ===== Mosaic & Mixup=====
        mosaic=1.0,    # 全程使用mosaic
        mixup=0.15,    # 增加mixup
        copy_paste=0.3,  # 启用copy-paste增强
        
        # ===== 关闭mosaic时机 =====
        close_mosaic=0,  # 不提前关闭mosaic
        
        # ===== 损失权重=====
        box=10.0,      # 大幅增加box损失权重
        cls=0.3,       # 降低分类损失（单类）
        dfl=2.0,       # 增加DFL损失
        
        # ===== IoU设置 =====
        iou=0.5,       # 降低IoU阈值
        
        # ===== Anchor优化 =====
        # YOLO11没有anchor，但可以调整stride
        
        # ===== NMS设置====
        conf=0.001,    # 训练时的置信度阈值
        
        # ===== 保存设置 =====
        project=str(PROJECT_ROOT / "runs/detect"),
        name="yolo11_dust_optimized_v2",
        patience=100,  # 增加耐心值
        save=True,
        save_period=20,
        plots=True,
        val=True,
        
        # ===== EMA（指数移动平均）=====
        # YOLO11默认启用
        
        # ===== 其他优化 =====
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
    
    # 多个置信度阈值测试
    conf_thresholds = [0.001, 0.01, 0.05, 0.1]
    
    print("\n不同置信度阈值的性能：")
    for conf_th in conf_thresholds:
        val_results = best_model.val(
            data=str(DATASET_YAML),
            split='test',
            imgsz=640,
            batch=16,
            device='0' if torch.cuda.is_available() else 'cpu',
            conf=conf_th,
            iou=0.5,
            max_det=300,  # 增加最大检测数
            plots=False,
        )
        
        print(f"\nConf={conf_th:.3f} | mAP50={val_results.box.map50:.4f} | "
              f"P={val_results.box.mp:.4f} | R={val_results.box.mr:.4f}")
    
    # 使用最佳阈值重新评估并保存图像
    val_results = best_model.val(
        data=str(DATASET_YAML),
        split='test',
        imgsz=640,
        batch=16,
        device='0' if torch.cuda.is_available() else 'cpu',
        conf=0.001,
        iou=0.5,
        max_det=300,
        plots=True,
        save_json=True,
    )
    
    print("\n" + "="*70)
    print("📊 最终测试结果 (conf=0.001)")
    print("="*70)
    print(f"mAP50:      {val_results.box.map50:.4f}")
    print(f"mAP50-95:   {val_results.box.map:.4f}")
    print(f"Precision:  {val_results.box.mp:.4f}")
    print(f"Recall:     {val_results.box.mr:.4f}")
    print("="*70)
    print(f"✅ 最佳模型: {best_model_path}")
    print(f"✅ 处理后数据: {PROCESSED_DATA_DIR}")
    print("="*70)