import cv2
import os
import numpy as np
import random
import glob
from tqdm import tqdm

#Path settings
TRAIN_IMAGES_DIR = "Data/images/train"
TRAIN_LABELS_DIR = "Data/labels/train"
NO_DUST_IMAGES_DIR = "Data/no_dust"
OUTPUT_DIR = "Data/synthetic_data"

# Create output directories
OUTPUT_IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
OUTPUT_LABELS_DIR = os.path.join(OUTPUT_DIR, "labels")
os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)

def xywhn2xyxy(x, y, w, h, W, H):
    """Convert normalized xywh center format to pixel xyxy format"""
    x1 = int((x - w / 2) * W)
    y1 = int((y - h / 2) * H)
    x2 = int((x + w / 2) * W)
    y2 = int((y + h / 2) * H)
    
    # Clip coordinates
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(W, x2)
    y2 = min(H, y2)
    
    return x1, y1, x2, y2

def load_dust_samples():
    """Extract all dust patches from the training set"""
    dust_patches = []
    print("Extracting dust samples from training data...")
    
    label_files = glob.glob(os.path.join(TRAIN_LABELS_DIR, "*.txt"))
    
    for label_file in tqdm(label_files):
        # Find corresponding image
        basename = os.path.splitext(os.path.basename(label_file))[0]
        # Try jpg and png
        img_path = os.path.join(TRAIN_IMAGES_DIR, basename + ".jpg")
        if not os.path.exists(img_path):
             img_path = os.path.join(TRAIN_IMAGES_DIR, basename + ".png")
        if not os.path.exists(img_path):
             img_path = os.path.join(TRAIN_IMAGES_DIR, basename + ".JPG")
        
        if not os.path.exists(img_path):
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        H, W = img.shape[:2]
        
        with open(label_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            cls_id = int(parts[0])
            # Assume all classes 0 are dust, or if you have multiple classes, adjust here
            # Here we extract everything as dust source
            
            x, y, w, h = map(float, parts[1:])
            x1, y1, x2, y2 = xywhn2xyxy(x, y, w, h, W, H)
            
            # Skip invalid small boxes
            if x2 - x1 < 5 or y2 - y1 < 5:
                continue
                
            patch = img[y1:y2, x1:x2]
            dust_patches.append(patch)
            
    print(f"Extracted {len(dust_patches)} dust patches.")
    return dust_patches

def generate_synthetic():
    dust_samples = load_dust_samples()
    if not dust_samples:
        print("No dust samples found! Check your paths.")
        return

    bg_images = glob.glob(os.path.join(NO_DUST_IMAGES_DIR, "*.*"))
    # Filter for image extensions
    bg_images = [f for f in bg_images if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    print(f"Found {len(bg_images)} background images. Starting synthesis...")
    
    for bg_path in tqdm(bg_images):
        bg_img = cv2.imread(bg_path)
        if bg_img is None:
            continue
            
        H, W = bg_img.shape[:2]
        new_labels = []
        
        # Clone background to avoid modifying original
        synthetic_img = bg_img.copy()
        
        # Randomly decide how many dusts to paste (e.g., 3 to 8)
        num_dusts = random.randint(3, 8)
        
        for _ in range(num_dusts):
            patch = random.choice(dust_samples)
            ph, pw = patch.shape[:2]
            
            # Random rotation (optional, 0, 90, 180, 270)
            k = random.randint(0, 3)
            patch = np.rot90(patch, k)
            ph, pw = patch.shape[:2] # Update shape after rotation
            
            # Ensure patch is smaller than background
            if ph >= H or pw >= W:
                continue
                
            # Random position (top-left corner)
            # Avoid edges
            px = random.randint(10, W - pw - 10)
            py = random.randint(10, H - ph - 10)
            
            # Center for seamlessClone (center of the patch in dst coordinates)
            center = (px + pw // 2, py + ph // 2)
            
            # Mask (full patch)
            mask = 255 * np.ones(patch.shape, patch.dtype)
            
            try:
                # Poisson blending
                # MIXED_CLONE often looks good for transparency-like effects, 
                # NORMAL_CLONE is standard pasting
                synthetic_img = cv2.seamlessClone(patch, synthetic_img, mask, center, cv2.MIXED_CLONE)
                
                # Calculate new label (normalized xywh)
                cx = (px + pw / 2) / W
                cy = (py + ph / 2) / H
                nw = pw / W
                nh = ph / H
                
                # Class 0 for dust
                new_labels.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                
            except Exception as e:
                # If seamlessClone fails (e.g. patch touching edge), skip
                pass
                
        # Save new image and label
        basename = os.path.basename(bg_path)
        name, _ = os.path.splitext(basename)
        new_filename = f"syn_{name}.jpg"
        
        cv2.imwrite(os.path.join(OUTPUT_IMAGES_DIR, new_filename), synthetic_img)
        
        with open(os.path.join(OUTPUT_LABELS_DIR, f"syn_{name}.txt"), 'w') as f:
            f.write('\n'.join(new_labels))
            
    print(f"Done! Synthetic data saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_synthetic()
