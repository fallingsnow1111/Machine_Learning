"""
数据准备工具 - 合并有标签和无标签数据用于蒸馏
"""

import os
import sys
import shutil
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np

# ===================== 配置 =====================
# 数据源路径
SOURCE_DIRS = [
    "./Data/dataset_merged_no_noise/images/train",   
    "./Data/dataset_merged_no_noise/images/val",
    "./Data/no_dust"          
]

# 输出目录
OUTPUT_DIR = "./Data/distill_images"

# 支持的图像格式
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}

# ===================== 主函数 =====================
def prepare_distill_data():
    """合并所有图像到蒸馏数据目录"""
    
    output_path = Path(OUTPUT_DIR)
    
    # 清空已存在的目录（避免重复累积）
    if output_path.exists():
        print("🗑️  清空旧数据...")
        for item in output_path.iterdir():
            if item.is_file():
                item.unlink()
    else:
        output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("📦 准备蒸馏数据集")
    print("="*60)
    print(f"📁 输出目录: {output_path.absolute()}")
    print()
    
    # 统计信息
    total_copied = 0
    total_skipped = 0
    source_stats = {}
    
    # 遍历所有源目录
    for source_dir in SOURCE_DIRS:
        source_path = Path(source_dir)
        
        if not source_path.exists():
            print(f"⚠️  跳过不存在的目录: {source_dir}")
            continue
        
        print(f"📂 扫描: {source_dir}")
        
        # 收集图像文件（同时搜索大小写扩展名以兼容 Linux）
        image_files = []
        for ext in IMAGE_EXTENSIONS:
            image_files.extend(source_path.glob(f"*{ext}"))
            image_files.extend(source_path.glob(f"*{ext.upper()}"))
        
        if len(image_files) == 0:
            print(f"   ⚠️  没有找到图像文件")
            continue
        
        # 复制文件
        copied = 0
        skipped = 0
        
        for img_file in tqdm(image_files, desc=f"   复制", unit="file"):
            dest_file = output_path / img_file.name
            
            # 处理文件名冲突
            if dest_file.exists():
                # 添加父目录名前缀避免冲突
                parent_name = source_path.parent.name
                new_name = f"{parent_name}_{img_file.name}"
                dest_file = output_path / new_name
            
            try:
                shutil.copy2(img_file, dest_file)
                copied += 1
            except Exception as e:
                print(f"   ❌ 复制失败 {img_file.name}: {e}")
                skipped += 1
        
        source_stats[source_dir] = copied
        total_copied += copied
        total_skipped += skipped
        
        print(f"   ✅ 复制 {copied} 张图像\n")
    
    # 输出统计信息
    print("="*60)
    print("📊 数据准备完成")
    print("="*60)
    print(f"✅ 总计复制: {total_copied} 张图像")
    if total_skipped > 0:
        print(f"⚠️  跳过: {total_skipped} 张")
    print()
    
    print("📂 各源目录统计:")
    for source, count in source_stats.items():
        if count > 0:
            print(f"   {source}: {count} 张")
    
    print()
    print(f"📁 蒸馏数据目录: {output_path.absolute()}")
    print(f"💡 下一步运行: python distill_pretrain.py")
    print("="*60)
    
    # 数据量建议
    if total_copied < 1000:
        print("\n⚠️  数据量较少 (<1000张)")
        print("   建议：增加更多图像以获得更好的蒸馏效果")
    elif total_copied < 5000:
        print(f"\n✅ 数据量合适 ({total_copied}张)")
        print("   预期：基础蒸馏效果")
    else:
        print(f"\n🎉 数据量充足 ({total_copied}张)")
        print("   预期：较好蒸馏效果")
    
    # 自动进行三通道预处理
    preprocess_distill_images()
    
    return total_copied

# ===================== 三通道预处理 =====================
def apply_preprocess_channels(image_path_str):
    """
    三通道预处理：
    Ch0 = 原图放大
    Ch1 = 双边滤波（去噪保边）
    Ch2 = CLAHE（局部对比度增强）
    """
    img_gray = cv2.imread(image_path_str, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        return None
    
    # Lanczos插值放大到640x640
    img_upscaled = cv2.resize(img_gray, (640, 640), interpolation=cv2.INTER_LANCZOS4)
    
    # Ch0: 原始放大
    c0 = img_upscaled
    
    # Ch1: 双边滤波（去噪保边）
    c1 = cv2.bilateralFilter(img_upscaled, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Ch2: CLAHE（局部对比度增强）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    c2 = clahe.apply(img_upscaled)
    
    # 合并为三通道
    merged_img = cv2.merge([c0, c1, c2])
    return merged_img

def preprocess_distill_images():
    """对蒸馏图像目录进行三通道预处理"""
    
    output_path = Path(OUTPUT_DIR)
    
    if not output_path.exists():
        print(f"❌ 蒸馏数据目录不存在: {output_path}")
        return
    
    # 收集所有图像文件（同时搜索大小写扩展名以兼容 Linux）
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(output_path.glob(f"*{ext}"))
        image_files.extend(output_path.glob(f"*{ext.upper()}"))
    
    if len(image_files) == 0:
        print(f"⚠️  蒸馏数据目录中无图像文件")
        return
    
    print(f"\n" + "="*60)
    print("🔄 开始三通道预处理")
    print("="*60)
    print(f"📁 处理目录: {output_path.absolute()}")
    print(f"📊 需处理图像: {len(image_files)} 张")
    print()
    
    processed_count = 0
    failed_count = 0
    
    for img_file in tqdm(image_files, desc="预处理", unit="file"):
        try:
            # 进行三通道处理
            processed_img = apply_preprocess_channels(str(img_file))
            
            if processed_img is not None:
                # 保存处理后的图像（覆盖原文件）
                cv2.imwrite(str(img_file), processed_img)
                processed_count += 1
            else:
                failed_count += 1
        except Exception as e:
            print(f"\n❌ 处理失败 {img_file.name}: {e}")
            failed_count += 1
    
    print()
    print("="*60)
    print("✅ 三通道预处理完成")
    print("="*60)
    print(f"✅ 成功处理: {processed_count} 张")
    if failed_count > 0:
        print(f"❌ 处理失败: {failed_count} 张")
    print(f"📁 输出目录: {output_path.absolute()}")
    print()

def clean_distill_data():
    """清空蒸馏数据目录"""
    output_path = Path(OUTPUT_DIR)
    
    if not output_path.exists():
        print(f"✅ 目录不存在，无需清理: {OUTPUT_DIR}")
        return
    
    print(f"🗑️  清空目录: {output_path.absolute()}")
    
    count = 0
    for item in output_path.iterdir():
        if item.is_file():
            item.unlink()
            count += 1
    
    print(f"✅ 删除 {count} 个文件")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--clean":
        clean_distill_data()
    else:
        prepare_distill_data()
