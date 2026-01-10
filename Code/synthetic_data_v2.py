import cv2
import os
import numpy as np
import random
import glob
import shutil
from tqdm import tqdm
from sklearn.cluster import KMeans
from collections import defaultdict

#Path settings
DATA_ROOT = "Data"
NO_DUST_IMAGES_DIR = "Data/no_dust"
OUTPUT_DIR = "Data/synthetic_data"

# ========== 优化参数配置 ==========
# 策略选择
USE_BRIGHTNESS_MATCHING = True      # 启用亮度域匹配
USE_SPATIAL_ANCHORING = True        # 启用空间位置锚定
USE_BACKGROUND_CLUSTERING = True    # 启用背景聚类
USE_RESIDUAL_FUSION = False         # 启用残差融合模式（需要模板图）

# 参数设置
BRIGHTNESS_THRESHOLD = 15           # 亮度差异容忍阈值（0-255）
SPATIAL_RADIUS_RATIO = 0.3          # 空间锚定半径（相对图像宽度）
NUM_BG_CLUSTERS = 3                 # 背景聚类数量（亮、中、暗）
RESIDUAL_ALPHA = 0.7                # 残差融合强度

# Clean and recreate output directories to avoid mixing old and new data
OUTPUT_IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
OUTPUT_LABELS_DIR = os.path.join(OUTPUT_DIR, "labels")

if os.path.exists(OUTPUT_IMAGES_DIR):
    print(f"清理旧数据: {OUTPUT_IMAGES_DIR}...")
    shutil.rmtree(OUTPUT_IMAGES_DIR)
if os.path.exists(OUTPUT_LABELS_DIR):
    print(f"清理旧数据: {OUTPUT_LABELS_DIR}...")
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

def compute_brightness(img):
    """计算图像的平均亮度"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    return np.mean(gray)

def extract_background_features(img, x1, y1, x2, y2, border=5):
    """提取缺陷周围背景区域的特征（用于亮度匹配）"""
    H, W = img.shape[:2]
    # 扩展边界以获取周围背景
    bg_x1 = max(0, x1 - border)
    bg_y1 = max(0, y1 - border)
    bg_x2 = min(W, x2 + border)
    bg_y2 = min(H, y2 + border)
    
    bg_region = img[bg_y1:bg_y2, bg_x1:bg_x2]
    
    # 计算背景亮度（排除缺陷本身）
    mask = np.ones(bg_region.shape[:2], dtype=bool)
    inner_y1 = y1 - bg_y1
    inner_x1 = x1 - bg_x1
    inner_y2 = inner_y1 + (y2 - y1)
    inner_x2 = inner_x1 + (x2 - x1)
    mask[inner_y1:inner_y2, inner_x1:inner_x2] = False
    
    if len(bg_region.shape) == 3:
        gray_bg = cv2.cvtColor(bg_region, cv2.COLOR_BGR2GRAY)
    else:
        gray_bg = bg_region
    
    if mask.sum() > 0:
        brightness = np.mean(gray_bg[mask])
    else:
        brightness = np.mean(gray_bg)
    
    return brightness

def cv_imread(file_path):
    """Read image with unicode path support, forcing BGR"""
    try:
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
    """提取所有灰尘样本，并记录位置、亮度、背景特征等元信息"""
    dust_metadata = []
    
    print("[优化模式] 提取灰尘样本及上下文特征...")
    
    # Save debug patches to verify what we are cropping
    debug_patch_dir = os.path.join(OUTPUT_DIR, "debug_patches")
    os.makedirs(debug_patch_dir, exist_ok=True)
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        images_dir = os.path.join(DATA_ROOT, "images", split)
        labels_dir = os.path.join(DATA_ROOT, "labels", split)
        
        if not os.path.exists(labels_dir):
            print(f"跳过 {split}: 未找到标签目录 {labels_dir}")
            continue
            
        label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
        print(f"处理 {split}: 发现 {len(label_files)} 个标签文件")
        
        for label_file in tqdm(label_files, desc=f"加载 {split}"):
            # Find corresponding image
            basename = os.path.splitext(os.path.basename(label_file))[0]
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
                
                # Assume class 0 is dust
                x, y, w, h = map(float, parts[1:])
                x1, y1, x2, y2 = xywhn2xyxy(x, y, w, h, W, H)
                
                # Skip invalid small boxes
                if x2 - x1 < 5 or y2 - y1 < 5:
                    continue
                    
                patch = img[y1:y2, x1:x2]
                if patch.size == 0:
                    continue
                
                # 提取背景亮度特征
                bg_brightness = extract_background_features(img, x1, y1, x2, y2, border=5)
                
                # 保存完整元数据
                dust_metadata.append({
                    'patch': patch.copy(),
                    'position': (x, y),
                    'brightness': bg_brightness,
                    'abs_position': (x1, y1, x2, y2),
                    'img_shape': (H, W)
                })
                
                # Save first 150 patches for debugging
                if len(dust_metadata) <= 150:
                    cv_imwrite(os.path.join(debug_patch_dir, 
                                           f"patch_{split}_{len(dust_metadata)}_br{int(bg_brightness)}.jpg"), 
                              patch)
            
    dust_brightness_list = [m['brightness'] for m in dust_metadata]
    if dust_brightness_list:
        print(f"✓ 提取 {len(dust_metadata)} 个灰尘样本")
        print(f"  亮度范围: [{min(dust_brightness_list):.1f}, {max(dust_brightness_list):.1f}]")
        print(f"  调试图像: {debug_patch_dir}")
    return dust_metadata

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

def cluster_backgrounds(bg_images, n_clusters=3):
    """对背景图像进行聚类（基于亮度和纹理特征）"""
    print(f"\n[背景聚类] 分析 {len(bg_images)} 张背景图像...")
    
    features = []
    valid_images = []
    
    for bg_path in tqdm(bg_images, desc="提取背景特征"):
        img = cv_imread(bg_path)
        if img is None:
            continue
            
        # 特征1: 平均亮度
        brightness = compute_brightness(img)
        
        # 特征2: 亮度标准差（纹理复杂度）
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        std_dev = np.std(gray)
        
        # 特征3: 边缘密度（使用Canny）
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        features.append([brightness, std_dev, edge_density * 100])
        valid_images.append(bg_path)
    
    features = np.array(features)
    
    # K-means 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    
    # 按类别组织
    bg_clusters = defaultdict(list)
    for img_path, label in zip(valid_images, labels):
        bg_clusters[label].append(img_path)
    
    # 打印聚类结果
    for cluster_id, images in bg_clusters.items():
        cluster_features = features[labels == cluster_id]
        avg_brightness = np.mean(cluster_features[:, 0])
        print(f"  类别 {cluster_id}: {len(images)} 张图像, 平均亮度={avg_brightness:.1f}")
    
    return dict(bg_clusters), features, labels

def check_brightness_compatibility(target_region, dust_brightness, threshold=15):
    """检查目标区域亮度是否与灰尘原背景兼容"""
    target_brightness = compute_brightness(target_region)
    return abs(target_brightness - dust_brightness) < threshold

def generate_synthetic():
    dust_metadata = load_dust_samples_and_stats()
    if not dust_metadata:
        print("❌ 未找到灰尘样本！请检查路径。")
        return

    bg_images = glob.glob(os.path.join(NO_DUST_IMAGES_DIR, "*.*"))
    bg_images = [f for f in bg_images if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if len(bg_images) == 0:
        print(f"❌ 未找到背景图像！路径: {NO_DUST_IMAGES_DIR}")
        return
    
    print(f"\n{'='*60}")
    print(f"🚀 开始合成 - 工业级数据增强")
    print(f"{'='*60}")
    print(f"背景图像: {len(bg_images)} 张")
    print(f"灰尘样本: {len(dust_metadata)} 个")
    print(f"\n启用的优化策略:")
    print(f"  ✓ 亮度域匹配: {USE_BRIGHTNESS_MATCHING} (阈值={BRIGHTNESS_THRESHOLD})")
    print(f"  ✓ 空间位置锚定: {USE_SPATIAL_ANCHORING} (半径={SPATIAL_RADIUS_RATIO})")
    print(f"  ✓ 背景聚类: {USE_BACKGROUND_CLUSTERING} (类别数={NUM_BG_CLUSTERS})")
    print(f"  ✓ 残差融合: {USE_RESIDUAL_FUSION}")
    
    # 背景聚类预处理
    bg_clusters = None
    bg_features = None
    bg_labels = None
    dust_cluster_map = None
    
    if USE_BACKGROUND_CLUSTERING:
        bg_clusters, bg_features, bg_labels = cluster_backgrounds(bg_images, n_clusters=NUM_BG_CLUSTERS)
        # 计算每个类别的平均亮度作为匹配索引
        dust_cluster_map = {}
        for cluster_id in range(NUM_BG_CLUSTERS):
            cluster_features = bg_features[bg_labels == cluster_id]
            dust_cluster_map[cluster_id] = np.mean(cluster_features[:, 0])
        bg_images_to_use = bg_clusters
    else:
        bg_images_to_use = bg_images
    
    print(f"\n开始合成...\n")
    
    # 统计计数器
    total_generated = 0
    brightness_rejected = 0
    position_adjusted = 0
    
    # 遍历所有背景图像
    all_bg_paths = []
    if isinstance(bg_images_to_use, dict):
        for imgs in bg_images_to_use.values():
            all_bg_paths.extend(imgs)
    else:
        all_bg_paths = bg_images_to_use
    
    for bg_path in tqdm(all_bg_paths, desc="合成进度"):
        bg_img = cv_imread(bg_path)
        if bg_img is None:
            continue
            
        H, W = bg_img.shape[:2]
        new_labels = []
        
        # Clone background for blending
        synthetic_img = bg_img.astype(np.float32)
        
        # Randomly decide how many dusts to paste
        num_dusts = random.randint(1, 2)
        
        success_count = 0
        attempts_total = 0
        MAX_TOTAL_ATTEMPTS = 100

        while success_count < num_dusts and attempts_total < MAX_TOTAL_ATTEMPTS:
            attempts_total += 1
            
            # 从元数据中选择灰尘样本
            dust_meta = random.choice(dust_metadata)
            patch = dust_meta['patch'].copy()
            
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
            
            # --- 空间位置锚定策略 ---
            if USE_SPATIAL_ANCHORING:
                # 使用原始位置作为参考点
                ref_x, ref_y = dust_meta['position']
                
                # 计算允许的偏移半径（像素）
                spatial_radius = int(W * SPATIAL_RADIUS_RATIO)
                
                # 在半径内随机偏移
                offset_x = random.randint(-spatial_radius, spatial_radius)
                offset_y = random.randint(-spatial_radius, spatial_radius)
                
                center_x = int(ref_x * W) + offset_x
                center_y = int(ref_y * H) + offset_y
                
                position_adjusted += 1
            else:
                # 回退到高斯抖动策略
                ref_x, ref_y = dust_meta['position']
                jitter_std = 0.1
                norm_cx = ref_x + random.gauss(0, jitter_std)
                norm_cy = ref_y + random.gauss(0, jitter_std)
                norm_cx = max(0.0, min(1.0, norm_cx))
                norm_cy = max(0.0, min(1.0, norm_cy))
                center_x = int(norm_cx * W)
                center_y = int(norm_cy * H)
            
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
            
            # --- 亮度域匹配检查 ---
            if USE_BRIGHTNESS_MATCHING:
                target_region = bg_img[y1:y2, x1:x2]
                if not check_brightness_compatibility(
                    target_region, 
                    dust_meta['brightness'], 
                    threshold=BRIGHTNESS_THRESHOLD
                ):
                    brightness_rejected += 1
                    continue
            
            # Recalculate final center
            final_cx = x1 + pw // 2
            final_cy = y1 + ph // 2
            
            # --- Blending Strategy: Alpha Blending ---
            feather_size = max(2, int(min(ph, pw) * 0.1))
            alpha_mask = create_soft_mask(ph, pw, feather_size)
            alpha_mask_3c = np.stack([alpha_mask] * 3, axis=-1)
            
            # Get ROI from background
            roi = synthetic_img[y1:y2, x1:x2]
            
            # Alpha blend
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
        
        # Check if we failed to paste any dust
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
        
        total_generated += 1
    
    print(f"\n{'='*60}")
    print(f"✅ 合成完成！")
    print(f"{'='*60}")
    print(f"生成图像: {total_generated} 张")
    print(f"输出路径: {OUTPUT_DIR}")
    if USE_BRIGHTNESS_MATCHING:
        print(f"亮度匹配拒绝: {brightness_rejected} 次")
    if USE_SPATIAL_ANCHORING:
        print(f"位置锚定应用: {position_adjusted} 次")
    print(f"\n💡 提示: 检查 {os.path.join(OUTPUT_DIR, 'debug_patches')} 查看提取的样本")

if __name__ == "__main__":
    generate_synthetic()
