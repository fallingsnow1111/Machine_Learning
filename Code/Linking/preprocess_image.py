import cv2
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# ================= 配置区域 =================
INPUT_ROOT = r"./Data/Merged/noise11"         # 输入根目录
OUTPUT_ROOT = r"./Data/Merged/noise11_processed"  # 输出根目录
TARGET_SIZE = (640, 640)              # 目标大小

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

# ================= 新增: 可视化验证模块 =================

def visualize_verification(dataset_root):
    """
    随机抽取一张处理后的图片，展示其三个通道，
    并读取对应的 Label 绘制 Bounding Box，进行对齐检查。
    """
    import random
    
    # 1. 寻找一张有对应 label 的图片
    img_dir = Path(dataset_root) / "images"
    # 假设结构是 images/train/xxx.jpg，如果不是标准结构会递归查找
    all_imgs = list(img_dir.rglob("*.jpg"))
    
    if not all_imgs:
        print("未找到处理后的图片，无法可视化。")
        return

    # 随机选择一张图片
    found = False
    for _ in range(10): # 尝试10次寻找有对应label的图片
        img_path = random.choice(all_imgs)
        # 推断 label 路径: 假设 dataset/images/train/x.jpg -> dataset/labels/train/x.txt
        # 这里做一个通用的路径替换逻辑
        label_path_str = str(img_path).replace("images", "labels").replace(img_path.suffix, ".txt")
        if os.path.exists(label_path_str):
            found = True
            break
    
    if not found:
        print("警告：找到了图片，但没找到对应的标签文件，无法画框，仅显示通道。")
        label_path_str = None

    print(f"\n正在可视化验证文件: {img_path.name}")
    
    # 2. 读取图片 (BGR)
    img_bgr = cv2.imread(str(img_path))
    h, w = img_bgr.shape[:2]
    
    # 拆分通道 (注意我们之前的合并顺序: B=Raw, G=Bilateral, R=CLAHE)
    ch_raw = img_bgr[:, :, 0]
    ch_bilateral = img_bgr[:, :, 1]
    ch_clahe = img_bgr[:, :, 2]

    # 3. 绘制 YOLO 标签 (画在 原图 和 融合图 上)
    # 融合图用于展示最终效果 (转为 RGB 以便 matplotlib 显示正常颜色)
    img_vis_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 在 Raw 通道上也画个框，方便看最原始的对齐情况 (转为 BGR 方便画图)
    img_raw_vis = cv2.cvtColor(ch_raw, cv2.COLOR_GRAY2BGR)

    if label_path_str:
        with open(label_path_str, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = list(map(float, line.strip().split()))
                cls_id = int(parts[0])
                cx, cy, bw, bh = parts[1:]
                
                # YOLO 归一化坐标转像素坐标
                x_center = cx * w
                y_center = cy * h
                box_w = bw * w
                box_h = bh * h
                
                x1 = int(x_center - box_w / 2)
                y1 = int(y_center - box_h / 2)
                x2 = int(x_center + box_w / 2)
                y2 = int(y_center + box_h / 2)
                
                # 画绿色框，线宽 2
                cv2.rectangle(img_vis_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img_raw_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 4. Matplotlib 绘图
    plt.figure(figsize=(16, 8))

    # 子图1: 通道0 (放大后的原图 + 框)
    plt.subplot(2, 3, 1)
    plt.title(f"Channel 0: 原始放大 (Raw)\n主要特征来源")
    plt.imshow(img_raw_vis) # 已经画了框
    plt.axis('off')

    # 子图2: 通道1 (双边滤波)
    plt.subplot(2, 3, 2)
    plt.title(f"Channel 1: 双边滤波 (Bilateral)\n去噪 & 保边")
    plt.imshow(ch_bilateral, cmap='gray')
    plt.axis('off')

    # 子图3: 通道2 (CLAHE)
    plt.subplot(2, 3, 3)
    plt.title(f"Channel 2: CLAHE\n局部对比度增强")
    plt.imshow(ch_clahe, cmap='gray')
    plt.axis('off')

    # 子图4: 最终合成输入 (RGB显示 + 框)
    plt.subplot(2, 3, 4)
    plt.title(f"Model Input (RGB Visualization)\n最终输入模型的 3 通道")
    plt.imshow(img_vis_rgb)
    plt.axis('off')

    # 子图5: 局部放大检查 (Zoom In)
    # 我们裁剪第一个框的中心区域来看看细节
    if label_path_str and 'x1' in locals():
        # 裁剪框周围 50 像素
        crop_x1 = max(0, x1 - 20)
        crop_y1 = max(0, y1 - 20)
        crop_x2 = min(w, x2 + 20)
        crop_y2 = min(h, y2 + 20)
        crop_img = img_vis_rgb[crop_y1:crop_y2, crop_x1:crop_x2]
        
        plt.subplot(2, 3, 5)
        plt.title(f"Zoom In (局部细节检查)\n请确认绿框中心是否有灰尘")
        plt.imshow(crop_img)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# ================= 运行入口 =================
if __name__ == '__main__':
    # 1. 模拟数据生成 (如果你没有数据，这段代码会生成测试数据)
    if not os.path.exists(INPUT_ROOT):
        print(f"未找到输入目录，创建测试数据...")
        img_dir = Path(INPUT_ROOT) / "images" / "train"
        lbl_dir = Path(INPUT_ROOT) / "labels" / "train"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        
        # 造一张 64x64 的图，灰尘只有 2 个像素大
        dummy_img = np.zeros((64, 64), dtype=np.uint8) + 100 # 背景灰度 100
        # 灰尘在中心 (32,32)，颜色更深
        dummy_img[31:33, 31:33] = 30 
        
        cv2.imwrite(str(img_dir / "test_dust.jpg"), dummy_img)
        
        # 对应的 label (归一化)
        # 中心 0.5, 0.5, 宽 2/64, 高 2/64
        with open(lbl_dir / "test_dust.txt", "w") as f:
            f.write(f"0 0.5 0.5 {2/64} {2/64}")

    # 2. 运行处理流程
    process_dataset(INPUT_ROOT, OUTPUT_ROOT)

    # 3. 运行可视化验证 (YOLO Checks)
    print("\n正在启动可视化验证窗口...")
    visualize_verification(OUTPUT_ROOT)

    # 4. 生成 dataset.yaml
    classes = ['dust']
    yaml_path = os.path.join(OUTPUT_ROOT, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"path: {OUTPUT_ROOT}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n\n")
        f.write(f"nc: {len(classes)}\n")
        f.write("names: " + str(classes) + "\n")
    
    print(f'[DONE] 数据集预处理完成，已生成 {yaml_path}')
