# 三通道转化，进行滤波和直方图均衡化
import cv2
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# ================= 配置区域 =================
INPUT_ROOT = r"./Data/mix"         # 输入根目录
OUTPUT_ROOT = r"./Data/mix_processed"  # 输出根目录
TARGET_SIZE = (640, 640)              # 目标大小

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
    # Ch0: 原始放大
    c0 = img_upscaled
    # Ch1: 双边滤波 (降噪保边)
    c1 = cv2.bilateralFilter(img_upscaled, d=BILATERAL_D, 
                             sigmaColor=BILATERAL_SIGMA_COLOR, 
                             sigmaSpace=BILATERAL_SIGMA_SPACE)
    # Ch2: CLAHE (特征增强)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_GRID_SIZE)
    c2 = clahe.apply(img_upscaled)

    # 3. 合并 (注意: OpenCV内存中是 BGR 顺序，对应 c0, c1, c2)
    # 也就是保存后: B通道=c0, G通道=c1, R通道=c2
    merged_img = cv2.merge([c0, c1, c2])
    return merged_img

def process_dataset(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    if not output_path.exists(): os.makedirs(output_path)

    files = [f for f in input_path.rglob('*') if f.is_file()]
    processed_count = 0

    print(f"开始处理数据集，目标尺寸: {TARGET_SIZE}...")
    for file_path in tqdm(files, desc="Processing"):
        rel_path = file_path.relative_to(input_path)
        target_path = output_path / rel_path
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif']:
            img = process_image_channels(str(file_path))
            if img is not None:
                # 统一保存为 jpg 以节省空间，也可以改为 png
                save_path = target_path.with_suffix('.jpg') 
                cv2.imwrite(str(save_path), img)
                processed_count += 1
        else:
            # 标签文件和其他文件直接复制
            shutil.copy2(file_path, target_path)
    
    print(f"处理完成，共生成 {processed_count} 张增强图像。")

# ================= 运行入口 =================
if __name__ == '__main__':
    # 1. 模拟数据生成 (如果你没有数据，这段代码会生成测试数据)
    if not os.path.exists(INPUT_ROOT):
        print(f"未找到输入目录，请确认输入目录{INPUT_ROOT}")

    # 2. 运行处理流程
    process_dataset(INPUT_ROOT, OUTPUT_ROOT)

