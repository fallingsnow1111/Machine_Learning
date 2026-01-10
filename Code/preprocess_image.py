# import cv2
# import numpy as np
# import os
# import shutil
# import matplotlib.pyplot as plt
# from pathlib import Path
# from tqdm import tqdm

# # ================= 配置区域 =================
# INPUT_ROOT = r"../Data/dataset_yolo"         # 输入根目录
# OUTPUT_ROOT = r"../Data/dataset_yolo_processed"  # 输出根目录
# TARGET_SIZE = (640, 640)              # 目标大小

# # 算法参数
# CLAHE_CLIP_LIMIT = 2.0
# CLAHE_GRID_SIZE = (8, 8)
# BILATERAL_D = 9
# BILATERAL_SIGMA_COLOR = 75
# BILATERAL_SIGMA_SPACE = 75

# # 绘图设置
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# # ================= 核心处理逻辑 =================

# def process_image_channels(img_path_str):
#     """生成的图片通道顺序：Ch0=原图, Ch1=双边滤波, Ch2=CLAHE"""
#     img_gray = cv2.imread(img_path_str, 0)
#     if img_gray is None: return None

#     # 1. Lanczos 插值放大 (64 -> 640)
#     img_upscaled = cv2.resize(img_gray, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)

#     # 2. 构建通道
#     # Ch0: 原始放大
#     c0 = img_upscaled
#     # Ch1: 双边滤波 (降噪保边)
#     c1 = cv2.bilateralFilter(img_upscaled, d=BILATERAL_D, 
#                              sigmaColor=BILATERAL_SIGMA_COLOR, 
#                              sigmaSpace=BILATERAL_SIGMA_SPACE)
#     # Ch2: CLAHE (特征增强)
#     clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_GRID_SIZE)
#     c2 = clahe.apply(img_upscaled)

#     # 3. 合并 (注意: OpenCV内存中是 BGR 顺序，对应 c0, c1, c2)
#     # 也就是保存后: B通道=c0, G通道=c1, R通道=c2
#     merged_img = cv2.merge([c0, c1, c2])
#     return merged_img

# def process_dataset(input_dir, output_dir):
#     input_path = Path(input_dir)
#     output_path = Path(output_dir)
#     if not output_path.exists(): os.makedirs(output_path)

#     files = [f for f in input_path.rglob('*') if f.is_file()]
#     processed_count = 0

#     print(f"开始处理数据集，目标尺寸: {TARGET_SIZE}...")
#     for file_path in tqdm(files, desc="Processing"):
#         rel_path = file_path.relative_to(input_path)
#         target_path = output_path / rel_path
#         target_path.parent.mkdir(parents=True, exist_ok=True)

#         if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif']:
#             img = process_image_channels(str(file_path))
#             if img is not None:
#                 # 统一保存为 jpg 以节省空间，也可以改为 png
#                 save_path = target_path.with_suffix('.jpg') 
#                 cv2.imwrite(str(save_path), img)
#                 processed_count += 1
#         else:
#             # 标签文件和其他文件直接复制
#             shutil.copy2(file_path, target_path)
    
#     print(f"处理完成，共生成 {processed_count} 张增强图像。")

# # ================= 新增: 可视化验证模块 =================

# def visualize_verification(dataset_root):
#     """
#     随机抽取一张处理后的图片，展示其三个通道，
#     并读取对应的 Label 绘制 Bounding Box，进行对齐检查。
#     """
#     import random
    
#     # 1. 寻找一张有对应 label 的图片
#     img_dir = Path(dataset_root) / "images"
#     # 假设结构是 images/train/xxx.jpg，如果不是标准结构会递归查找
#     all_imgs = list(img_dir.rglob("*.jpg"))
    
#     if not all_imgs:
#         print("未找到处理后的图片，无法可视化。")
#         return

#     # 随机选择一张图片
#     found = False
#     for _ in range(10): # 尝试10次寻找有对应label的图片
#         img_path = random.choice(all_imgs)
#         # 推断 label 路径: 假设 dataset/images/train/x.jpg -> dataset/labels/train/x.txt
#         # 这里做一个通用的路径替换逻辑
#         label_path_str = str(img_path).replace("images", "labels").replace(img_path.suffix, ".txt")
#         if os.path.exists(label_path_str):
#             found = True
#             break
    
#     if not found:
#         print("警告：找到了图片，但没找到对应的标签文件，无法画框，仅显示通道。")
#         label_path_str = None

#     print(f"\n正在可视化验证文件: {img_path.name}")
    
#     # 2. 读取图片 (BGR)
#     img_bgr = cv2.imread(str(img_path))
#     h, w = img_bgr.shape[:2]
    
#     # 拆分通道 (注意我们之前的合并顺序: B=Raw, G=Bilateral, R=CLAHE)
#     ch_raw = img_bgr[:, :, 0]
#     ch_bilateral = img_bgr[:, :, 1]
#     ch_clahe = img_bgr[:, :, 2]

#     # 3. 绘制 YOLO 标签 (画在 原图 和 融合图 上)
#     # 融合图用于展示最终效果 (转为 RGB 以便 matplotlib 显示正常颜色)
#     img_vis_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
#     # 在 Raw 通道上也画个框，方便看最原始的对齐情况 (转为 BGR 方便画图)
#     img_raw_vis = cv2.cvtColor(ch_raw, cv2.COLOR_GRAY2BGR)

#     if label_path_str:
#         with open(label_path_str, 'r') as f:
#             lines = f.readlines()
#             for line in lines:
#                 parts = list(map(float, line.strip().split()))
#                 cls_id = int(parts[0])
#                 cx, cy, bw, bh = parts[1:]
                
#                 # YOLO 归一化坐标转像素坐标
#                 x_center = cx * w
#                 y_center = cy * h
#                 box_w = bw * w
#                 box_h = bh * h
                
#                 x1 = int(x_center - box_w / 2)
#                 y1 = int(y_center - box_h / 2)
#                 x2 = int(x_center + box_w / 2)
#                 y2 = int(y_center + box_h / 2)
                
#                 # 画绿色框，线宽 2
#                 cv2.rectangle(img_vis_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.rectangle(img_raw_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

#     # 4. Matplotlib 绘图
#     plt.figure(figsize=(16, 8))

#     # 子图1: 通道0 (放大后的原图 + 框)
#     plt.subplot(2, 3, 1)
#     plt.title(f"Channel 0: 原始放大 (Raw)\n主要特征来源")
#     plt.imshow(img_raw_vis) # 已经画了框
#     plt.axis('off')

#     # 子图2: 通道1 (双边滤波)
#     plt.subplot(2, 3, 2)
#     plt.title(f"Channel 1: 双边滤波 (Bilateral)\n去噪 & 保边")
#     plt.imshow(ch_bilateral, cmap='gray')
#     plt.axis('off')

#     # 子图3: 通道2 (CLAHE)
#     plt.subplot(2, 3, 3)
#     plt.title(f"Channel 2: CLAHE\n局部对比度增强")
#     plt.imshow(ch_clahe, cmap='gray')
#     plt.axis('off')

#     # 子图4: 最终合成输入 (RGB显示 + 框)
#     plt.subplot(2, 3, 4)
#     plt.title(f"Model Input (RGB Visualization)\n最终输入模型的 3 通道")
#     plt.imshow(img_vis_rgb)
#     plt.axis('off')

#     # 子图5: 局部放大检查 (Zoom In)
#     # 我们裁剪第一个框的中心区域来看看细节
#     if label_path_str and 'x1' in locals():
#         # 裁剪框周围 50 像素
#         crop_x1 = max(0, x1 - 20)
#         crop_y1 = max(0, y1 - 20)
#         crop_x2 = min(w, x2 + 20)
#         crop_y2 = min(h, y2 + 20)
#         crop_img = img_vis_rgb[crop_y1:crop_y2, crop_x1:crop_x2]
        
#         plt.subplot(2, 3, 5)
#         plt.title(f"Zoom In (局部细节检查)\n请确认绿框中心是否有灰尘")
#         plt.imshow(crop_img)
#         plt.axis('off')
    
#     plt.tight_layout()
#     plt.show()

# # ================= 运行入口 =================
# if __name__ == '__main__':
#     # 1. 模拟数据生成 (如果你没有数据，这段代码会生成测试数据)
#     if not os.path.exists(INPUT_ROOT):
#         print(f"未找到输入目录，创建测试数据...")
#         img_dir = Path(INPUT_ROOT) / "images" / "train"
#         lbl_dir = Path(INPUT_ROOT) / "labels" / "train"
#         img_dir.mkdir(parents=True, exist_ok=True)
#         lbl_dir.mkdir(parents=True, exist_ok=True)
        
#         # 造一张 64x64 的图，灰尘只有 2 个像素大
#         dummy_img = np.zeros((64, 64), dtype=np.uint8) + 100 # 背景灰度 100
#         # 灰尘在中心 (32,32)，颜色更深
#         dummy_img[31:33, 31:33] = 30 
        
#         cv2.imwrite(str(img_dir / "test_dust.jpg"), dummy_img)
        
#         # 对应的 label (归一化)
#         # 中心 0.5, 0.5, 宽 2/64, 高 2/64
#         with open(lbl_dir / "test_dust.txt", "w") as f:
#             f.write(f"0 0.5 0.5 {2/64} {2/64}")

#     # 2. 运行处理流程
#     process_dataset(INPUT_ROOT, OUTPUT_ROOT)

#     # 3. 运行可视化验证 (YOLO Checks)
#     print("\n正在启动可视化验证窗口...")
#     visualize_verification(OUTPUT_ROOT)


# import cv2
# import numpy as np
# import os
# import shutil
# import matplotlib.pyplot as plt
# from pathlib import Path
# from tqdm import tqdm

# # ================= 配置区域 =================
# INPUT_ROOT = r"./dataset_yolo"        
# OUTPUT_ROOT = r"./dataset_yolo_processed2"  
# TARGET_SIZE = (640, 640)              

# # 算法参数
# CLAHE_CLIP_LIMIT = 2.0
# CLAHE_GRID_SIZE = (8, 8)

# # 黑帽运算的核心参数：核大小
# # 灰尘放大后约 10-20 像素，核必须比灰尘大，建议 25-35
# MORPH_KERNEL_SIZE = (25, 25) 

# # 绘图设置
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# # ================= 新增算法: 形态学黑帽 (Black-Hat) =================
# def get_blackhat(img):
#     """
#     方法: Morphological Black-Hat (黑帽运算)
#     原理: 闭运算图 - 原图。
#     作用: 专门用于提取亮背景下的暗色微小物体，能有效消除复杂背景纹理。
#     """
#     # 1. 定义结构元素 (核)
#     # 使用椭圆形核 (MORPH_ELLIPSE) 比矩形核更贴合灰尘的形状
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)
    
#     # 2. 执行黑帽运算
#     # 这一步会自动：(背景+填平灰尘) - (背景+灰尘) = 灰尘
#     blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    
#     # 3. 二值化或阈值处理 (可选，但推荐)
#     # 为了让特征更纯净，可以过滤掉极微弱的背景残留
#     # 这里我们做一个简单的线性拉伸，让灰尘更亮
#     blackhat = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)
    
#     # [进阶技巧] 如果背景依然有残留线条，可以加一个阈值操作，只保留最亮的部分
#     ret, blackhat = cv2.threshold(blackhat, 50, 255, cv2.THRESH_TOZERO)
    
#     return blackhat

# # ================= 核心处理逻辑 =================

# def process_image_channels(img_path_str):
#     """生成的图片通道顺序：Ch0=原图, Ch1=黑帽特征, Ch2=CLAHE"""
#     img_gray = cv2.imread(img_path_str, 0)
#     if img_gray is None: return None

#     # 1. Lanczos 插值放大 (64 -> 640)
#     img_upscaled = cv2.resize(img_gray, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)

#     # 2. 构建通道
    
#     # --- Ch0: 原始放大 (基准信息) ---
#     c0 = img_upscaled
    
#     # --- Ch1: 黑帽形态学 (几何特征) ---
#     # [修改点] 这里替换了 DoG，专门针对复杂背景下的暗斑
#     c1 = get_blackhat(img_upscaled)
    
#     # --- Ch2: CLAHE (纹理/对比度特征) ---
#     clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_GRID_SIZE)
#     c2 = clahe.apply(img_upscaled)

#     # 3. 合并
#     merged_img = cv2.merge([c0, c1, c2])
#     return merged_img

# def process_dataset(input_dir, output_dir):
#     input_path = Path(input_dir)
#     output_path = Path(output_dir)
#     if not output_path.exists(): os.makedirs(output_path)

#     files = [f for f in input_path.rglob('*') if f.is_file()]
#     processed_count = 0

#     print(f"开始处理数据集，目标尺寸: {TARGET_SIZE}...")
#     for file_path in tqdm(files, desc="Processing"):
#         rel_path = file_path.relative_to(input_path)
#         target_path = output_path / rel_path
#         target_path.parent.mkdir(parents=True, exist_ok=True)

#         if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif']:
#             img = process_image_channels(str(file_path))
#             if img is not None:
#                 save_path = target_path.with_suffix('.jpg') 
#                 cv2.imwrite(str(save_path), img)
#                 processed_count += 1
#         else:
#             shutil.copy2(file_path, target_path)
    
#     print(f"处理完成，共生成 {processed_count} 张增强图像。")

# # ================= 可视化验证模块 (更新标题) =================

# def visualize_verification(dataset_root):
#     import random
#     img_dir = Path(dataset_root) / "images"
#     all_imgs = list(img_dir.rglob("*.jpg"))
#     if not all_imgs: return

#     # 寻找有 label 的图片
#     found = False
#     img_path = None
#     for _ in range(20): 
#         img_path = random.choice(all_imgs)
#         label_path_str = str(img_path).replace("images", "labels").replace(img_path.suffix, ".txt")
#         if os.path.exists(label_path_str):
#             found = True
#             break
    
#     if not found:
#         print("警告：未找到对应的标签文件。")
#         label_path_str = None

#     print(f"\n正在可视化验证文件: {img_path.name}")
    
#     img_bgr = cv2.imread(str(img_path))
#     h, w = img_bgr.shape[:2]
    
#     ch_raw = img_bgr[:, :, 0]
#     ch_blackhat = img_bgr[:, :, 1]  # Channel 1 现在是黑帽
#     ch_clahe = img_bgr[:, :, 2]

#     img_vis_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#     img_raw_vis = cv2.cvtColor(ch_raw, cv2.COLOR_GRAY2BGR)

#     if label_path_str:
#         with open(label_path_str, 'r') as f:
#             lines = f.readlines()
#             for line in lines:
#                 parts = list(map(float, line.strip().split()))
#                 cx, cy, bw, bh = parts[1:]
#                 x_center, y_center = cx * w, cy * h
#                 box_w, box_h = bw * w, bh * h
#                 x1, y1 = int(x_center - box_w / 2), int(y_center - box_h / 2)
#                 x2, y2 = int(x_center + box_w / 2), int(y_center + box_h / 2)
#                 cv2.rectangle(img_vis_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.rectangle(img_raw_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

#     plt.figure(figsize=(16, 8))

#     plt.subplot(2, 3, 1)
#     plt.title(f"Channel 0: 原始放大 (Raw)")
#     plt.imshow(img_raw_vis)
#     plt.axis('off')

#     plt.subplot(2, 3, 2)
#     plt.title(f"Channel 1: 黑帽运算 (Black-Hat)\n(消除背景，只留暗斑)")
#     plt.imshow(ch_blackhat, cmap='gray')
#     plt.axis('off')

#     plt.subplot(2, 3, 3)
#     plt.title(f"Channel 2: CLAHE")
#     plt.imshow(ch_clahe, cmap='gray')
#     plt.axis('off')

#     plt.subplot(2, 3, 4)
#     plt.title(f"Model Input (RGB)")
#     plt.imshow(img_vis_rgb)
#     plt.axis('off')

#     if label_path_str and 'x1' in locals():
#         crop_x1 = max(0, x1 - 20)
#         crop_y1 = max(0, y1 - 20)
#         crop_x2 = min(w, x2 + 20)
#         crop_y2 = min(h, y2 + 20)
#         crop_img = img_vis_rgb[crop_y1:crop_y2, crop_x1:crop_x2]
#         plt.subplot(2, 3, 5)
#         plt.title(f"Zoom In (局部检查)")
#         plt.imshow(crop_img)
#         plt.axis('off')
    
#     plt.tight_layout()
#     plt.show()

# if __name__ == '__main__':
#     # 自动生成测试数据逻辑省略...
#     if not os.path.exists(INPUT_ROOT):
#         # ... (可以使用之前的测试数据生成代码) ...
#         pass
        
#     process_dataset(INPUT_ROOT, OUTPUT_ROOT)
#     visualize_verification(OUTPUT_ROOT)


import cv2
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# ================= 配置区域 =================
INPUT_ROOT = r"./Data/dataset_yolo"         # 输入根目录
OUTPUT_ROOT = r"./Data/dataset_yolo_processed"  # 输出根目录 
TARGET_SIZE = (640, 640)              

# 算法参数
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)

# [关键参数] 中值滤波核大小
# 必须是一个大于灰尘直径的奇数。
# 假设放大后灰尘约 15-20 像素，建议设为 31 或 41
# 越大越能抹除灰尘，但太大会引入光照不均
MEDIAN_KERNEL_SIZE = 35

# 绘图设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ================= 新增算法: 中值差分 (Difference of Median) =================
def get_difference_of_median(img):
    """
    方法: Difference of Median (DoM)
    原理: |原图 - 中值滤波图|
    优势: 中值滤波能完美抹除"斑点"（灰尘），但保留"线条"（导线）。
          两者相减，导线被抵消，只剩下灰尘。
    """
    # 1. 中值滤波：估算"没有灰尘的背景"
    # 这一步会将灰尘这种"孤立噪点"直接抹平为背景色，但保留导线边缘
    bg_estimation = cv2.medianBlur(img, MEDIAN_KERNEL_SIZE)
    
    # 2. 差分运算：背景 - 原图
    # 在灰尘位置：背景(亮) - 原图(暗) = 差值(大) -> 亮斑
    # 在导线位置：背景(线) - 原图(线) = 差值(小) -> 黑
    diff = cv2.absdiff(img, bg_estimation)
    
    # 3. 增强：归一化拉伸，让灰尘更亮
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    
    # [可选] 阈值清理：过滤掉微小的背景噪声，只留最明显的斑点
    # _, diff = cv2.threshold(diff, 30, 255, cv2.THRESH_TOZERO)
    
    return diff

# ================= 核心处理逻辑 =================

def process_image_channels(img_path_str):
    """生成的图片通道顺序：Ch0=原图, Ch1=中值差分, Ch2=CLAHE"""
    img_gray = cv2.imread(img_path_str, 0)
    if img_gray is None: return None

    # 1. Lanczos 插值放大 (64 -> 640)
    img_upscaled = cv2.resize(img_gray, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)

    # 2. 构建通道
    
    # --- Ch0: 原始放大 ---
    c0 = img_upscaled
    
    # --- Ch1: 中值差分 (DoM) ---
    # [修改点] 替换之前的黑帽运算，专门对抗线条干扰
    c1 = get_difference_of_median(img_upscaled)
    
    # --- Ch2: CLAHE ---
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_GRID_SIZE)
    c2 = clahe.apply(img_upscaled)

    # 3. 合并
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
                save_path = target_path.with_suffix('.jpg') 
                cv2.imwrite(str(save_path), img)
                processed_count += 1
        else:
            shutil.copy2(file_path, target_path)
    
    print(f"处理完成，共生成 {processed_count} 张增强图像。")

# ================= 可视化验证模块 =================

def visualize_verification(dataset_root):
    import random
    img_dir = Path(dataset_root) / "images"
    all_imgs = list(img_dir.rglob("*.jpg"))
    if not all_imgs: return

    # 寻找有 label 的图片
    found = False
    img_path = None
    for _ in range(20): 
        img_path = random.choice(all_imgs)
        label_path_str = str(img_path).replace("images", "labels").replace(img_path.suffix, ".txt")
        if os.path.exists(label_path_str):
            found = True
            break
    
    if not found:
        print("警告：未找到对应的标签文件。")
        label_path_str = None

    print(f"\n正在可视化验证文件: {img_path.name}")
    
    img_bgr = cv2.imread(str(img_path))
    h, w = img_bgr.shape[:2]
    
    ch_raw = img_bgr[:, :, 0]
    ch_dom = img_bgr[:, :, 1]  # Channel 1 现在是 DoM
    ch_clahe = img_bgr[:, :, 2]

    img_vis_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_raw_vis = cv2.cvtColor(ch_raw, cv2.COLOR_GRAY2BGR)

    if label_path_str:
        with open(label_path_str, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = list(map(float, line.strip().split()))
                cx, cy, bw, bh = parts[1:]
                x_center, y_center = cx * w, cy * h
                box_w, box_h = bw * w, bh * h
                x1, y1 = int(x_center - box_w / 2), int(y_center - box_h / 2)
                x2, y2 = int(x_center + box_w / 2), int(y_center + box_h / 2)
                cv2.rectangle(img_vis_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img_raw_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    plt.figure(figsize=(16, 8))

    plt.subplot(2, 3, 1)
    plt.title(f"Channel 0: 原始放大 (Raw)")
    plt.imshow(img_raw_vis)
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title(f"Channel 1: 中值差分 (DoM)\n(过滤线条，保留斑点)")
    plt.imshow(ch_dom, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title(f"Channel 2: CLAHE")
    plt.imshow(ch_clahe, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.title(f"Model Input (RGB)")
    plt.imshow(img_vis_rgb)
    plt.axis('off')

    if label_path_str and 'x1' in locals():
        crop_x1 = max(0, x1 - 20)
        crop_y1 = max(0, y1 - 20)
        crop_x2 = min(w, x2 + 20)
        crop_y2 = min(h, y2 + 20)
        crop_img = img_vis_rgb[crop_y1:crop_y2, crop_x1:crop_x2]
        plt.subplot(2, 3, 5)
        plt.title(f"Zoom In (局部检查)")
        plt.imshow(crop_img)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # 自动生成测试数据逻辑省略...
    if not os.path.exists(INPUT_ROOT):
        print(INPUT_ROOT) 
        pass
        
    process_dataset(INPUT_ROOT, OUTPUT_ROOT)
    visualize_verification(OUTPUT_ROOT)
    
    # 生成 dataset.yaml
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