import cv2
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# ================= 配置区域 =================
# 建议使用绝对路径，或者确保当前工作目录正确
INPUT_ROOT = r"./Data/Raw/dust"         # 输入根目录
OUTPUT_ROOT = r"./Data/Raw/dust_processed"  # 输出根目录
TARGET_SIZE = (640, 640)                             # 目标大小

# 算法参数
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)
BILATERAL_D = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75

# 绘图设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

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

    # 3. 合并 (OpenCV BGR顺序保存后: B=c0, G=c1, R=c2)
    merged_img = cv2.merge([c0, c1, c2])
    return merged_img

def process_dataset(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 每次运行前建议清理旧输出，防止文件混乱
    if output_path.exists():
        print(f"♻️ 清理旧输出目录: {output_dir}")
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    files = [f for f in input_path.rglob('*') if f.is_file()]
    processed_count = 0

    print(f"🚀 开始处理数据集，目标尺寸: {TARGET_SIZE}...")
    for file_path in tqdm(files, desc="Processing"):
        rel_path = file_path.relative_to(input_path)
        target_path = output_path / rel_path
        
        # 排除已有的 yaml 文件，避免重复和冲突
        if file_path.suffix.lower() == '.yaml':
            continue

        target_path.parent.mkdir(parents=True, exist_ok=True)

        # 处理图片
        if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif']:
            img = process_image_channels(str(file_path))
            if img is not None:
                # 统一保存为 jpg
                save_path = target_path.with_suffix('.jpg') 
                cv2.imwrite(str(save_path), img)
                processed_count += 1
        
        # 复制标签文件 (.txt)
        elif file_path.suffix.lower() == '.txt':
            shutil.copy2(file_path, target_path)
        
        # 忽略其他无关文件（如 .zip, .DS_Store 等）
        else:
            continue
    
    print(f"✅ 处理完成，共生成 {processed_count} 张三通道增强图像。")

# ================= 运行入口 =================
if __name__ == '__main__':
    # 1. 检查输入目录
    if not os.path.exists(INPUT_ROOT):
        print(f"❌ 错误：未找到输入目录 {INPUT_ROOT}")
    else:
        # 2. 运行处理流程
        process_dataset(INPUT_ROOT, OUTPUT_ROOT)

        # 3. 生成 dataset.yaml (使用绝对路径，防止训练报错)
        classes = ['dust']
        abs_output_root = os.path.abspath(OUTPUT_ROOT)
        yaml_path = os.path.join(abs_output_root, 'dataset.yaml')
        
        # 检查子文件夹是否存在，确保 YAML 路径正确
        has_train = os.path.exists(os.path.join(abs_output_root, "images/train"))
        has_val = os.path.exists(os.path.join(abs_output_root, "images/val"))
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(f"path: {abs_output_root}  # 数据集根目录绝对路径\n")
            f.write(f"train: images/train\n")
            f.write(f"val: images/val\n")
            # 如果没有 test 文件夹，可以注释掉下面这行
            f.write(f"test: images/test\n\n")
            
            f.write(f"nc: {len(classes)}\n")
            f.write(f"names: {str(classes)}\n")
        
        print(f'\n[DONE] 预处理完成！')
        print(f'📍 增强后的数据集位于: {abs_output_root}')
        print(f'📝 配置文件已生成: {yaml_path}')
        if not has_val:
            print(f'⚠️ 警告：在输出目录中未发现 images/val 文件夹，请确保原始数据已分好类。')