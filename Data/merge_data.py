import os
import shutil
import random
import glob
import yaml
from tqdm import tqdm
from pathlib import Path

# Paths
ORIGINAL_DATA_DIR = Path("Data/Raw/dust")
SYNTHETIC_DATA_DIR = Path("Data/Synthetic/no_noise")
OUTPUT_DIR = Path("Data/Merged/no_noise41")
DATASET_YAML = "Data/Raw/dust/dataset.yaml"

# 🎛️ 数据融合比例 (可调参数)
# Train 集中原始数据的占比
ORIGINAL_DATA_RATIO = 0.80  # 80% 原始数据
SYNTHETIC_DATA_RATIO = 0.20  # 20% 合成数据
# Val 和 Test 集始终使用纯净原始数据

def collect_data_pairs(base_dir, is_synthetic=False, split_name=None):
    """
    Collects image and label file pairs. 
    Returns a list of tuples: (image_path, label_path)
    """
    pairs = []
    
    # Define image extensions to look for
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.PNG'}
    
    if is_synthetic:
        # Synthetic data structure: images/*.jpg, labels/*.txt
        img_dir = base_dir / "images"
        lbl_dir = base_dir / "labels"
        search_dirs = [(img_dir, lbl_dir)]
    else:
        if split_name:
            # Collect from specific split folder
            search_dirs = [
                (base_dir / "images" / split_name, base_dir / "labels" / split_name)
            ]
        else:
            # Original data structure: images/{train,val,test}, labels/{train,val,test}
            search_dirs = [
                (base_dir / "images/train", base_dir / "labels/train"),
                (base_dir / "images/val", base_dir / "labels/val"),
                (base_dir / "images/test", base_dir / "labels/test")
            ]
    

    for img_folder, lbl_folder in search_dirs:
        # Resolve absolute paths to avoid ambiguity
        img_folder = img_folder.resolve() if img_folder.exists() else img_folder
        lbl_folder = lbl_folder.resolve() if lbl_folder.exists() else lbl_folder
        
        if not img_folder.exists():
            print(f"Skipping {img_folder} (Directory not found)")
            continue
            
        print(f"Scanning {img_folder}...")
        for img_path in img_folder.iterdir():
            if img_path.suffix.lower() in valid_exts: # Case insensitive check
                # Construct corresponding label path
                lbl_name = img_path.stem + ".txt"
                lbl_path = lbl_folder / lbl_name
                
                if lbl_path.exists():
                    pairs.append((img_path, lbl_path))
                else:
                     # Warn if label is missing for synthetic data (shouldn't happen)
                     if is_synthetic:
                         print(f"Warning: Missing label for {img_path.name} at {lbl_path}")

                    
    return pairs

def merge_and_split():
    # 1. Prepare Output Directories
    if OUTPUT_DIR.exists():
        print(f"Warning: {OUTPUT_DIR} already exists. Merging into it...")
    
    for split in ['train', 'val', 'test']:
        (OUTPUT_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # 2. Collect Data (分开收集原始和合成数据)
    print("Collecting original data...")
    
    # Explicitly collect per split to preserve raw dataset structure
    original_train = collect_data_pairs(ORIGINAL_DATA_DIR, is_synthetic=False, split_name='train')
    original_val = collect_data_pairs(ORIGINAL_DATA_DIR, is_synthetic=False, split_name='val')
    original_test = collect_data_pairs(ORIGINAL_DATA_DIR, is_synthetic=False, split_name='test')
    
    print(f"Original pairs found:")
    print(f"  Train: {len(original_train)}")
    print(f"  Val: {len(original_val)}")
    print(f"  Test: {len(original_test)}")
    
    print("Collecting synthetic data...")
    synthetic_pairs = collect_data_pairs(SYNTHETIC_DATA_DIR, is_synthetic=True)
    print(f"Synthetic pairs found: {len(synthetic_pairs)}")
    
    if len(original_train) == 0:
        print("No original train data found. Exiting.")
        return
    
    # 3. Use original splits directly (Previous random split logic removed to prevent data leakage)
    # Val and test sets are kept exactly as they are in the raw data
    
    # 4. 计算合成数据需求量
    # 根据比例计算：synthetic / (original + synthetic) = SYNTHETIC_DATA_RATIO
    # => synthetic = original * SYNTHETIC_DATA_RATIO / ORIGINAL_DATA_RATIO
    num_synthetic_needed = int(len(original_train) * (SYNTHETIC_DATA_RATIO / ORIGINAL_DATA_RATIO))
    
    # 从合成数据中随机抽取
    random.shuffle(synthetic_pairs)
    synthetic_train = synthetic_pairs[:min(num_synthetic_needed, len(synthetic_pairs))]
    
    # 合并 train 集
    train_pairs = original_train + synthetic_train
    random.shuffle(train_pairs)
    
    # val 和 test 集保持纯净（只用原始数据）
    val_pairs = original_val
    test_pairs = original_test
    
    # 计算实际比例
    actual_original_ratio = len(original_train) / len(train_pairs) * 100
    actual_synthetic_ratio = len(synthetic_train) / len(train_pairs) * 100
    
    print(f"\n{'='*70}")
    print(f"📊 数据融合完成 - Train 集成分")
    print(f"{'='*70}")
    print(f"  原始数据:  {len(original_train):4d} 张  ({actual_original_ratio:.1f}%)")
    print(f"  合成数据:  {len(synthetic_train):4d} 张  ({actual_synthetic_ratio:.1f}%)")
    print(f"  Train 总计: {len(train_pairs):4d} 张")
    print(f"\n  Val 总计:   {len(val_pairs):4d} 张 (纯净原始)")
    print(f"  Test 总计:  {len(test_pairs):4d} 张 (纯净原始)")
    print(f"{'='*70}")

    # 5. Copy Files
    def copy_set(pairs, split_name):
        print(f"Copying {split_name} data...")
        for img_src, lbl_src in tqdm(pairs):
            # Define destination paths
            img_dst = OUTPUT_DIR / 'images' / split_name / img_src.name
            lbl_dst = OUTPUT_DIR / 'labels' / split_name / lbl_src.name
            
            # Handle duplicates renaming if necessary (unlikely with random names but possible)
            if img_dst.exists():
                stem = img_src.stem
                suffix = img_src.suffix
                new_name = f"{stem}_dup{suffix}"
                new_lbl_name = f"{stem}_dup.txt"
                img_dst = OUTPUT_DIR / 'images' / split_name / new_name
                lbl_dst = OUTPUT_DIR / 'labels' / split_name / new_lbl_name
            
            shutil.copy2(img_src, img_dst)
            shutil.copy2(lbl_src, lbl_dst)

    copy_set(train_pairs, 'train')
    copy_set(val_pairs, 'val')
    copy_set(test_pairs, 'test')

    # 6. Create new dataset.yaml
    create_yaml()
    
    print(f"\n{'='*70}")
    print(f"✅ 数据融合完成！")
    print(f"{'='*70}")
    print(f"📂 输出目录: {OUTPUT_DIR.absolute()}")
    print(f"📋 配置文件: {OUTPUT_DIR / 'dataset_merged.yaml'}")
    print(f"{'='*70}")

def create_yaml():
    # Read original yaml to get names and nc
    with open(DATASET_YAML, 'r') as f:
        data = yaml.safe_load(f)
    
    # Update paths
    # Note: Ultralytics YOLO supports absolute paths or relative to the execution directory.
    # It's safest to put the relative path from where user runs train.py
    
    new_yaml = {
        'path': str(OUTPUT_DIR),  # Use OUTPUT_DIR variable instead of hardcoding
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': data.get('nc', 1),
        'names': data.get('names', ['object'])
    }
    
    with open(OUTPUT_DIR / 'dataset_merged.yaml', 'w') as f:
        yaml.dump(new_yaml, f, sort_keys=False)

if __name__ == "__main__":
    merge_and_split()
