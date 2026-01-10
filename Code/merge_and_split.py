import os
import shutil
import random
import glob
import yaml
from tqdm import tqdm
from pathlib import Path

# Paths
ORIGINAL_DATA_DIR = Path("Data/raw")
SYNTHETIC_DATA_DIR = Path("Data/synthetic_data")
OUTPUT_DIR = Path("Data/merged_dataset")
DATASET_YAML = "Data/dataset.yaml"

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

def collect_data_pairs(base_dir, is_synthetic=False):
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
    original_pairs = collect_data_pairs(ORIGINAL_DATA_DIR, is_synthetic=False)
    print(f"Original pairs found: {len(original_pairs)}")
    
    print("Collecting synthetic data...")
    synthetic_pairs = collect_data_pairs(SYNTHETIC_DATA_DIR, is_synthetic=True)
    print(f"Synthetic pairs found: {len(synthetic_pairs)}")
    
    if len(original_pairs) == 0:
        print("No original data found. Exiting.")
        return
    
    # 3. 分离原始数据的 train/val/test（保持 test 集纯净）
    random.shuffle(original_pairs)
    total_original = len(original_pairs)
    
    # 原始数据按 7:2:1 划分
    original_train_end = int(total_original * TRAIN_RATIO)
    original_val_end = original_train_end + int(total_original * VAL_RATIO)
    
    original_train = original_pairs[:original_train_end]
    original_val = original_pairs[original_train_end:original_val_end]
    original_test = original_pairs[original_val_end:]  # 纯净的 test 集
    
    print(f"\nOriginal data split:")
    print(f"  Train: {len(original_train)}")
    print(f"  Val: {len(original_val)}")
    print(f"  Test: {len(original_test)} (纯净)")
    
    # 4. 混合策略：train 集中原始 90% + 合成 10%
    # 计算需要多少合成数据
    target_synthetic_ratio = 0.1  # 合成数据占 10%
    target_original_ratio = 0.9   # 原始数据占 90%
    
    # 根据原始 train 数据量计算需要的合成数据量
    # synthetic / (original + synthetic) = 0.1
    # synthetic = 0.1 * (original + synthetic)
    # 0.9 * synthetic = 0.1 * original
    # synthetic = (0.1 / 0.9) * original ≈ 0.111 * original
    num_synthetic_needed = int(len(original_train) * (target_synthetic_ratio / target_original_ratio))
    
    # 从合成数据中随机抽取
    random.shuffle(synthetic_pairs)
    synthetic_train = synthetic_pairs[:min(num_synthetic_needed, len(synthetic_pairs))]
    
    # 合并 train 集
    train_pairs = original_train + synthetic_train
    random.shuffle(train_pairs)
    
    # val 和 test 集保持纯净（只用原始数据）
    val_pairs = original_val
    test_pairs = original_test
    
    print(f"\nFinal split (原始:合成 = 9:1 in train):")
    print(f"  Train: {len(train_pairs)} (原始 {len(original_train)} + 合成 {len(synthetic_train)})")
    print(f"  Val: {len(val_pairs)} (纯净)")
    print(f"  Test: {len(test_pairs)} (纯净)")
    
    print(f"\nSplit counts: Train={len(train_pairs)}, Val={len(val_pairs)}, Test={len(test_pairs)}")

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
    print(f"\nCompleted! Merged dataset created at: {OUTPUT_DIR.absolute()}")
    print(f"New yaml file created: {OUTPUT_DIR / 'dataset_merged.yaml'}")

def create_yaml():
    # Read original yaml to get names and nc
    with open(DATASET_YAML, 'r') as f:
        data = yaml.safe_load(f)
    
    # Update paths
    # Note: Ultralytics YOLO supports absolute paths or relative to the execution directory.
    # It's safest to put the relative path from where user runs train.py
    
    new_yaml = {
        'path': "Data/merged_dataset", # Use relative path for cross-platform compatibility
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
