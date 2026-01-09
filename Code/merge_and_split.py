import os
import shutil
import random
import glob
import yaml
from tqdm import tqdm
from pathlib import Path

# Paths
ORIGINAL_DATA_DIR = Path("Data")
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

    # 2. Collect Data
    print("Collecting original data...")
    original_pairs = collect_data_pairs(ORIGINAL_DATA_DIR, is_synthetic=False)
    print(f"Original pairs found: {len(original_pairs)}")
    
    print("Collecting synthetic data...")
    synthetic_pairs = collect_data_pairs(SYNTHETIC_DATA_DIR, is_synthetic=True)
    print(f"Synthetic pairs found: {len(synthetic_pairs)}")
    
    all_pairs = original_pairs + synthetic_pairs
    random.shuffle(all_pairs)
    
    total_files = len(all_pairs)
    print(f"Total files to merge: {total_files}")
    
    if total_files == 0:
        print("No files found. Exiting.")
        return

    # 3. Calculate Split Indices
    train_end = int(total_files * TRAIN_RATIO)
    val_end = train_end + int(total_files * VAL_RATIO)
    
    train_pairs = all_pairs[:train_end]
    val_pairs = all_pairs[train_end:val_end]
    test_pairs = all_pairs[val_end:]
    
    print(f"Split counts: Train={len(train_pairs)}, Val={len(val_pairs)}, Test={len(test_pairs)}")

    # 4. Copy Files
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

    # 5. Create new dataset.yaml
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
