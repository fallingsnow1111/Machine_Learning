import cv2
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# ================= 配置区域 =================
# 全部使用相对路径
INPUT_ROOT = r"./Data/Raw/dust"         # 输入根目录
OUTPUT_ROOT = r"./Data/Raw/dust_processed"  # 输出根目录
TARGET_SIZE = (640, 640)                             # 目标大小

# 算法参数
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)
BILATERAL_D = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75

# ================= 核心处理逻辑 =================

def process_image_channels(img_path_str):
    """生成的图片通道顺序：Ch0=原图, Ch1=双边滤波, Ch2=CLAHE"""
    img_gray = cv2.imread(img_path_str, 0)
    if img_gray is None: return None

    # 1. Lanczos 插值放大 (64 -> 640)
    img_upscaled = cv2.resize(img_gray, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)

    # 2. 构建通道
    c0 = img_upscaled
    c1 = cv2.bilateralFilter(img_upscaled, d=BILATERAL_D, 
                             sigmaColor=BILATERAL_SIGMA_COLOR, 
                             sigmaSpace=BILATERAL_SIGMA_SPACE)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_GRID_SIZE)
    c2 = clahe.apply(img_upscaled)

    # 3. 合并 (BGR 顺序)
    merged_img = cv2.merge([c0, c1, c2])
    return merged_img

def process_dataset(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not output_path.exists():
        os.makedirs(output_path)

    # 递归获取所有文件
    files = [f for f in input_path.rglob('*') if f.is_file()]
    processed_count = 0

    print(f"🚀 开始处理数据集，目标尺寸: {TARGET_SIZE}...")
    for file_path in tqdm(files, desc="Processing"):
        rel_path = file_path.relative_to(input_path)
        target_path = output_path / rel_path
        
        # 排除已有的 yaml 文件
        if file_path.suffix.lower() == '.yaml':
            continue

        target_path.parent.mkdir(parents=True, exist_ok=True)

        # 处理图片
        if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif']:
            img = process_image_channels(str(file_path))
            if img is not None:
                save_path = target_path.with_suffix('.jpg') 
                cv2.imwrite(str(save_path), img)
                processed_count += 1
        
        # 复制标签文件 (.txt)
        elif file_path.suffix.lower() == '.txt':
            shutil.copy2(file_path, target_path)
    
    print(f"✅ 处理完成，共生成 {processed_count} 张图像。")

# ================= 运行入口 =================
if __name__ == '__main__':
    if not os.path.exists(INPUT_ROOT):
        print(f"❌ 错误：未找到输入目录 {INPUT_ROOT}")
    else:
        # 1. 运行处理流程
        process_dataset(INPUT_ROOT, OUTPUT_ROOT)

        # 2. 生成 dataset.yaml (使用相对路径)
        classes = ['dust']
        yaml_path = os.path.join(OUTPUT_ROOT, 'dataset.yaml')
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            # 这里的 path 使用相对于项目根目录的相对路径
            # 注意：YOLO 训练时，path 是相对于执行 train 命令的目录
            f.write(f"path: {OUTPUT_ROOT}  # 相对路径\n")
            f.write(f"train: images/train\n")
            f.write(f"val: images/val\n")
            f.write(f"test: images/test\n\n")
            
            f.write(f"nc: {len(classes)}\n")
            f.write(f"names: {str(classes)}\n")
        
        print(f'\n[DONE] 预处理完成！')
        print(f'📝 配置文件已生成: {yaml_path}')
        print(f'💡 请确保在训练脚本中引用此相对路径。')