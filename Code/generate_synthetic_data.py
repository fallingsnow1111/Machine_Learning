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

def load_dust_samples():
    """Extract all dust patches from the training set"""
    dust_patches = []
    print("Extracting dust samples from training data...")
    
    # Save debug patches to verify what we are cropping
    debug_patch_dir = os.path.join(OUTPUT_DIR, "debug_patches")
    os.makedirs(debug_patch_dir, exist_ok=True)
    
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
            
        img = cv_imread(img_path)
        if img is None:
            continue
            
        H, W = img.shape[:2]
        
        with open(label_file, 'r') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            parts = line.strip().split()
            cls_id = int(parts[0])
            
            x, y, w, h = map(float, parts[1:])
            x1, y1, x2, y2 = xywhn2xyxy(x, y, w, h, W, H)
            
            # Skip invalid small boxes
            if x2 - x1 < 5 or y2 - y1 < 5:
                continue
                
            patch = img[y1:y2, x1:x2]
            if patch.size == 0: continue
            
            dust_patches.append(patch)
            
            # Save first 50 patches for debugging
            if len(dust_patches) <= 50:
                cv_imwrite(os.path.join(debug_patch_dir, f"patch_{len(dust_patches)}.jpg"), patch)
            
    print(f"Extracted {len(dust_patches)} dust patches. (Debug patches saved to {debug_patch_dir})")
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
        bg_img = cv_imread(bg_path)
        if bg_img is None:
            continue
            
        H, W = bg_img.shape[:2]
        new_labels = []
        
        # Clone background to avoid modifying original
        synthetic_img = bg_img.copy()
        
        # Randomly decide how many dusts to paste (e.g., 1 to 3, to match single dust characteristic better)
        num_dusts = random.randint(1, 3)

        
        success_count = 0
        attempts_total = 0 # To avoid infinite loops across the whole image processing
        MAX_TOTAL_ATTEMPTS = 50 # Max tries per image to place ANY dust

        while success_count < num_dusts and attempts_total < MAX_TOTAL_ATTEMPTS:
            attempts_total += 1
            
            patch = random.choice(dust_samples)
            ph, pw = patch.shape[:2]
            
            # Random rotation (optional, 0, 90, 180, 270)
            k = random.randint(0, 3)
            # Make a copy before rotation to avoid modifying shared buffer (though rot90 usually returns view or copy)
            patch = np.rot90(patch.copy(), k)
            ph, pw = patch.shape[:2] # Update shape after rotation
            
            # Ensure patch is smaller than background
            # If patch is too large (larger than 1/4 of bg), resize it down
            if ph > H // 4 or pw > W // 4:
                scale = min((H // 4) / ph, (W // 4) / pw)
                # Ensure minimum scale doesn't make it 0
                if scale > 0:
                    patch = cv2.resize(patch, (0, 0), fx=scale, fy=scale)
                    ph, pw = patch.shape[:2]
                else:
                     continue

            # Verify size again after resize
            if ph >= H or pw >= W or ph <= 0 or pw <= 0:
                 continue
                
            # Random position (top-left corner)
            # Avoid edges more aggressively for seamlessClone stability
            # seamlessClone requires center point. If center is too close to edge, patch exceeds boundary.
            # Patch range must be completely inside.
            # Safe margin for center x: pw//2 to W - pw//2
            # Safe margin for center y: ph//2 to H - ph//2
            
            try:
                min_x = pw // 2 + 5
                max_x = W - pw // 2 - 5
                min_y = ph // 2 + 5
                max_y = H - ph // 2 - 5
                
                if min_x >= max_x or min_y >= max_y:
                    continue

                center_x = random.randint(min_x, max_x)
                center_y = random.randint(min_y, max_y)
                center = (center_x, center_y)

            except ValueError:
                # Range empty (image too small for margins)
                continue
            
            # Mask (full patch)
            mask = 255 * np.ones(patch.shape, patch.dtype)
            
            pasted_this_round = False
            try:
                # Poisson blending
                synthetic_img = cv2.seamlessClone(patch, synthetic_img, mask, center, cv2.NORMAL_CLONE)
                pasted_this_round = True
            except Exception as e:
                # Fallback: Direct Copy if seamlessClone fails
                # Calculate ROI for direct copy
                y1 = center_y - ph // 2
                x1 = center_x - pw // 2
                y2 = y1 + ph
                x2 = x1 + pw
                
                # Check if completely inside
                if y1 >= 0 and x1 >= 0 and y2 <= H and x2 <= W:
                     try:
                         synthetic_img[y1:y2, x1:x2] = patch
                         pasted_this_round = True
                     except ValueError as ve:
                         # Handle explicit channel mismatch if it still occurs
                         pass
                else:
                    pass
            
            if pasted_this_round:
                # Calculate new label (normalized xywh)
                # center is (center_x, center_y) in pixels
                cx = center_x / W
                cy = center_y / H
                nw = pw / W
                nh = ph / H
                
                # Class 0 for dust
                new_labels.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                
                success_count += 1


            # If we failed to paste ANY dust after all attempts, skip saving this file
            # or save it as negative sample (optional). Here we skip to ensure "all have dust".
            # print(f"Warning: Failed to paste any dust on {os.path.basename(bg_path)}, skipping.")
            continue
                
        # Save new image and label
        basename = os.path.basename(bg_path)
        name, _ = os.path.splitext(basename)
        new_filename = f"syn_{name}.jpg"
        
        cv_imwrite(os.path.join(OUTPUT_IMAGES_DIR, new_filename), synthetic_img)
        
        with open(os.path.join(OUTPUT_LABELS_DIR, f"syn_{name}.txt"), 'w') as f:
            f.write('\n'.join(new_labels))
            
    print(f"Done! Synthetic data saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_synthetic()
