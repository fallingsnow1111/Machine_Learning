import cv2
import os
import numpy as np
import random
import glob
import shutil
from tqdm import tqdm

#Path settings
DATA_ROOT = "Data"
NO_DUST_IMAGES_DIR = "Data/no_dust"
OUTPUT_DIR = "Data/synthetic_data"

# Clean and recreate output directories to avoid mixing old and new data
OUTPUT_IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
OUTPUT_LABELS_DIR = os.path.join(OUTPUT_DIR, "labels")

if os.path.exists(OUTPUT_IMAGES_DIR):
    print(f"Cleaning existing synthetic images in {OUTPUT_IMAGES_DIR}...")
    shutil.rmtree(OUTPUT_IMAGES_DIR)
if os.path.exists(OUTPUT_LABELS_DIR):
    print(f"Cleaning existing synthetic labels in {OUTPUT_LABELS_DIR}...")
    shutil.rmtree(OUTPUT_LABELS_DIR)

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


def cv_imread(file_path):
    """Read image with unicode path support, forcing BGR"""
    try:
        # cv2.IMREAD_COLOR is 1
        cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), 1)
        return cv_img
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def cv_imwrite(file_path, img):
    """Write image with unicode path support"""
    try:
        cv2.imencode(os.path.splitext(file_path)[1], img)[1].tofile(file_path)
        return True
    except Exception as e:
        print(f"Error writing {file_path}: {e}")
        return False

def load_dust_samples_and_stats():
    """Extract all dust patches and their normalized positions from train/val/test sets"""
    dust_patches = []
    dust_positions = []  # Store normalized (x, y) center positions
    
    print("Extracting dust samples and position statistics from train/val/test...")
    
    # Save debug patches to verify what we are cropping
    debug_patch_dir = os.path.join(OUTPUT_DIR, "debug_patches")
    os.makedirs(debug_patch_dir, exist_ok=True)
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        images_dir = os.path.join(DATA_ROOT, "images", split)
        labels_dir = os.path.join(DATA_ROOT, "labels", split)
        
        if not os.path.exists(labels_dir):
            print(f"Skipping {split}: labels not found in {labels_dir}")
            continue
            
        label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
        print(f"Processing {split}: found {len(label_files)} label files")
        
        for label_file in tqdm(label_files, desc=f"Loading {split}"):
            # Find corresponding image
            basename = os.path.splitext(os.path.basename(label_file))[0]
            # Try multiple extensions
            img_found = False
            img_path = ""
            for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.bmp']:
                temp_path = os.path.join(images_dir, basename + ext)
                if os.path.exists(temp_path):
                    img_path = temp_path
                    img_found = True
                    break
            
            if not img_found:
                continue
                
            img = cv_imread(img_path)
            if img is None:
                continue
                
            H, W = img.shape[:2]
            
            with open(label_file, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                # Assume class 0 is dust (or take all if single-class)
                x, y, w, h = map(float, parts[1:])
                x1, y1, x2, y2 = xywhn2xyxy(x, y, w, h, W, H)
                
                # Skip invalid small boxes
                if x2 - x1 < 5 or y2 - y1 < 5:
                    continue
                    
                patch = img[y1:y2, x1:x2]
                if patch.size == 0:
                    continue
                
                dust_patches.append(patch)
                dust_positions.append((x, y))  # Store normalized center position
                
                # Save sample debug patches
                if len(dust_patches) <= 100 and len(dust_patches) % 10 == 0:
                    cv_imwrite(os.path.join(debug_patch_dir, f"patch_{split}_{len(dust_patches)}.jpg"), patch)
            
    print(f"Extracted {len(dust_patches)} dust patches with position stats.")
    print(f"Debug patches saved to {debug_patch_dir}")
    return dust_patches, dust_positions

def create_soft_mask(patch_h, patch_w, feather_size=5):
    """Create a feathered alpha mask for smooth blending"""
    mask = np.zeros((patch_h, patch_w), dtype=np.float32)
    
    # Ensure feather size doesn't exceed patch dimensions
    feather_size = min(feather_size, patch_h // 2, patch_w // 2)
    
    if feather_size <= 0:
        return np.ones((patch_h, patch_w), dtype=np.float32)
    
    # Inner region is 1.0 (fully opaque)
    mask[feather_size:patch_h-feather_size, feather_size:patch_w-feather_size] = 1.0
    
    # Apply Gaussian blur to create soft edges
    ksize = 2 * feather_size + 1
    if ksize % 2 == 0:
        ksize += 1
    mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
    
    # Normalize to [0, 1]
    if mask.max() > 0:
        mask = mask / mask.max()
    
    return mask

def generate_synthetic():
    dust_samples, dust_positions = load_dust_samples_and_stats()
    if not dust_samples:
        print("No dust samples found! Check your paths.")
        return

    bg_images = glob.glob(os.path.join(NO_DUST_IMAGES_DIR, "*.*"))
    # Filter for image extensions
    bg_images = [f for f in bg_images if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    print(f"Found {len(bg_images)} background images. Starting synthesis...")
    print("Using position-constrained random placement + Alpha Blending strategy.")
    
    for bg_path in tqdm(bg_images):
        bg_img = cv_imread(bg_path)
        if bg_img is None:
            continue
            
        H, W = bg_img.shape[:2]
        new_labels = []
        
        # Clone background for blending (use float32 for precision)
        synthetic_img = bg_img.astype(np.float32)
        
        # Randomly decide how many dusts to paste
        num_dusts = random.randint(1, 2)

        
        success_count = 0
        attempts_total = 0 # To avoid infinite loops across the whole image processing
        MAX_TOTAL_ATTEMPTS = 50 # Max tries per image to place ANY dust

        while success_count < num_dusts and attempts_total < MAX_TOTAL_ATTEMPTS:
            attempts_total += 1
            
            patch = random.choice(dust_samples)
            
            # --- Augmentation: Random Scaling ---
            scale = random.uniform(0.7, 1.3)
            ph, pw = patch.shape[:2]
            new_h, new_w = int(ph * scale), int(pw * scale)
            
            # Constrain to max 1/4 of background
            if new_h > H // 4 or new_w > W // 4:
                scale_limit = min((H // 4) / ph, (W // 4) / pw)
                new_h, new_w = int(ph * scale_limit), int(pw * scale_limit)
            
            # Skip if too small
            if new_h < 5 or new_w < 5:
                continue
            
            patch_resized = cv2.resize(patch, (new_w, new_h))
            ph, pw = patch_resized.shape[:2]
            
            # --- Augmentation: Random Rotation ---
            k = random.randint(0, 3)
            patch_resized = np.rot90(patch_resized, k).copy()
            ph, pw = patch_resized.shape[:2]
            
            # --- Position Strategy: Prior Distribution + Gaussian Jitter ---
            if dust_positions:
                # Sample from historical position distribution
                ref_x, ref_y = random.choice(dust_positions)
                
                # Add Gaussian jitter (std=10% allows exploration)
                jitter_std = 0.1
                norm_cx = ref_x + random.gauss(0, jitter_std)
                norm_cy = ref_y + random.gauss(0, jitter_std)
                
                # Clamp to [0, 1]
                norm_cx = max(0.0, min(1.0, norm_cx))
                norm_cy = max(0.0, min(1.0, norm_cy))
                
                center_x = int(norm_cx * W)
                center_y = int(norm_cy * H)
            else:
                # Fallback to uniform random
                center_x = random.randint(pw//2, W - pw//2)
                center_y = random.randint(ph//2, H - ph//2)
            
            # Calculate top-left from center
            x1 = center_x - pw // 2
            y1 = center_y - ph // 2
            x2 = x1 + pw
            y2 = y1 + ph
            
            # Boundary check and adjustment
            if x1 < 0:
                x1 = 0
                x2 = pw
            if y1 < 0:
                y1 = 0
                y2 = ph
            if x2 > W:
                x2 = W
                x1 = W - pw
            if y2 > H:
                y2 = H
                y1 = H - ph
            
            # Final validation
            if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
                continue
            
            # Recalculate final center
            final_cx = x1 + pw // 2
            final_cy = y1 + ph // 2
            
            # --- Blending Strategy: Alpha Blending with Feathering ---
            feather_size = max(2, int(min(ph, pw) * 0.1))  # 10% edge softness
            alpha_mask = create_soft_mask(ph, pw, feather_size)
            alpha_mask_3c = np.stack([alpha_mask] * 3, axis=-1)
            
            # Get ROI from background
            roi = synthetic_img[y1:y2, x1:x2]
            
            # Alpha blend: result = patch * alpha + background * (1 - alpha)
            patch_float = patch_resized.astype(np.float32)
            blended = patch_float * alpha_mask_3c + roi * (1.0 - alpha_mask_3c)
            
            # Paste blended result
            synthetic_img[y1:y2, x1:x2] = blended
            
            # --- Generate Label ---
            cx = final_cx / W
            cy = final_cy / H
            nw = pw / W
            nh = ph / H
            
            new_labels.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            success_count += 1
        
        # Check if we failed to paste any dust after all attempts
        if success_count == 0:
            continue
                
        # Save new image and label
        basename = os.path.basename(bg_path)
        name, _ = os.path.splitext(basename)
        new_filename = f"syn_{name}.jpg"
        
        # Convert back to uint8
        final_img = synthetic_img.clip(0, 255).astype(np.uint8)
        
        cv_imwrite(os.path.join(OUTPUT_IMAGES_DIR, new_filename), final_img)
        
        with open(os.path.join(OUTPUT_LABELS_DIR, f"syn_{name}.txt"), 'w') as f:
            f.write('\n'.join(new_labels))
            
    print(f"Done! Synthetic data saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_synthetic()
