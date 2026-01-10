import cv2
import numpy as np
import os
import shutil
import yaml
from pathlib import Path
from tqdm import tqdm

# ================= 配置 =================
INPUT_DATASET_DIR = r"../Data/dataset_yolo"  # 输入数据集所在的文件夹
OUTPUT_DATASET_DIR = r"../Data/dataset_yolo_augmented"  # 输出数据集文件夹
DATASET_YAML = "dataset.yaml"  # 数据集配置文件名

# ================= 核心变换逻辑 =================

def rotate_bbox(norm_bbox, angle_deg):
    """
    旋转 YOLO 归一化坐标 (class, x_center, y_center, w, h)
    angle_deg 支持: 90, 180, 270
    """
    c, x, y, w, h = norm_bbox
    
    if angle_deg == 90:
        # 顺时针 90度: new_x = 1-y, new_y = x, new_w = h, new_h = w
        return [c, 1.0 - y, x, h, w]
    elif angle_deg == 180:
        # 180度: new_x = 1-x, new_y = 1-y
        return [c, 1.0 - x, 1.0 - y, w, h]
    elif angle_deg == 270:
        # 顺时针 270度 (逆时针90): new_x = y, new_y = 1-x, new_w = h, new_h = w
        return [c, y, 1.0 - x, h, w]
    return norm_bbox


def augment_image_and_label(img_path, label_path, output_img_dir, output_label_dir):
    """
    对单张图片及其对应标签进行数据增强
    生成 5 倍的数据（原图、翻转、3个旋转）
    """
    # 创建输出目录
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    # 1. 读取图像
    img = cv2.imread(str(img_path))
    if img is None:
        return 0
    
    # 2. 读取对应的 Label
    bboxes = []
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    bboxes.append([float(x) for x in line.split()])
    
    # 定义变换列表: (后缀名, 旋转角度, 是否水平翻转)
    # 动作: 原图, 翻转, 转90, 转180, 转270
    transforms = [
        ("_orig", 0, False),
        ("_flipH", 0, True),     # 水平翻转
        ("_rot90", 90, False),
        ("_rot180", 180, False),
        ("_rot270", 270, False) 
    ]

    count = 0
    for suffix, angle, flip_h in transforms:
        # --- 处理图像 ---
        new_img = img.copy()
        
        # 先旋转
        if angle == 90:
            new_img = cv2.rotate(new_img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            new_img = cv2.rotate(new_img, cv2.ROTATE_180)
        elif angle == 270:
            new_img = cv2.rotate(new_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # 后翻转
        if flip_h:
            new_img = cv2.flip(new_img, 1) # 1 代表水平翻转

        # --- 处理标签 ---
        new_bboxes = []
        for box in bboxes:
            c, x, y, w, h = box
            
            # 先旋转坐标
            if angle != 0:
                c, x, y, w, h = rotate_bbox([c, x, y, w, h], angle)
            
            # 后翻转坐标 (水平翻转: x 变成 1-x)
            if flip_h:
                x = 1.0 - x
            
            new_bboxes.append(f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

        # --- 保存结果 ---
        save_name = img_path.stem + suffix
        
        # 保存图片
        cv2.imwrite(os.path.join(output_img_dir, save_name + ".jpg"), new_img)
        
        # 保存标签
        with open(os.path.join(output_label_dir, save_name + ".txt"), "w") as f:
            f.write("\n".join(new_bboxes))
        
        count += 1
    
    return count


def copy_dataset_without_augment(src_img_dir, src_label_dir, output_img_dir, output_label_dir):
    """
    将数据集直接复制到输出目录，不进行增强
    """
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    count = 0
    
    # 复制图片
    if Path(src_img_dir).exists():
        for img_file in Path(src_img_dir).glob("*.jpg"):
            shutil.copy(str(img_file), os.path.join(output_img_dir, img_file.name))
            count += 1
        for img_file in Path(src_img_dir).glob("*.png"):
            shutil.copy(str(img_file), os.path.join(output_img_dir, img_file.name))
            count += 1
    
    # 复制标签
    if Path(src_label_dir).exists():
        for label_file in Path(src_label_dir).glob("*.txt"):
            shutil.copy(str(label_file), os.path.join(output_label_dir, label_file.name))
    
    return count


def process_yolo_dataset():
    """
    主函数：处理整个 YOLO 数据集
    只对训练集进行增强，验证集和测试集直接复制
    """
    input_base_path = Path(INPUT_DATASET_DIR)
    output_base_path = Path(OUTPUT_DATASET_DIR)
    
    # 读取原始 dataset.yaml
    yaml_path = input_base_path / DATASET_YAML
    if not yaml_path.exists():
        print(f"❌ 错误: 找不到 {yaml_path}")
        return
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        dataset_config = yaml.safe_load(f)
    
    print(f"📂 读取数据集配置: {yaml_path}")
    print(f"   - 训练集: {dataset_config.get('train', 'N/A')}")
    print(f"   - 验证集: {dataset_config.get('val', 'N/A')}")
    print(f"   - 测试集: {dataset_config.get('test', 'N/A')}")
    
    # 创建输出目录
    output_base_path.mkdir(exist_ok=True)
    
    # ===== 处理训练集 (增强) =====
    train_img_src = input_base_path / dataset_config.get('train', 'images/train')
    train_label_src = input_base_path / dataset_config.get('train').replace('images', 'labels')
    train_img_out = output_base_path / dataset_config.get('train', 'images/train')
    train_label_out = output_base_path / dataset_config.get('train').replace('images', 'labels')
    
    train_img_out.parent.mkdir(parents=True, exist_ok=True)
    train_label_out.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🔄 处理训练集（数据增强）...")
    train_count = 0
    if train_img_src.exists():
        img_files = list(train_img_src.glob("*.jpg")) + list(train_img_src.glob("*.png"))
        for img_path in tqdm(img_files, desc="训练集"):
            label_path = train_label_src / img_path.with_suffix(".txt").name
            aug_count = augment_image_and_label(img_path, label_path, str(train_img_out), str(train_label_out))
            train_count += aug_count
    
    print(f"✅ 训练集完成: {len(img_files)} 张原图 -> {train_count} 张增强图")
    
    # ===== 处理验证集 (不增强) =====
    val_img_src = input_base_path / dataset_config.get('val', 'images/val')
    val_label_src = input_base_path / dataset_config.get('val').replace('images', 'labels')
    val_img_out = output_base_path / dataset_config.get('val', 'images/val')
    val_label_out = output_base_path / dataset_config.get('val').replace('images', 'labels')
    
    print(f"\n📋 处理验证集（直接复制，无增强）...")
    val_count = copy_dataset_without_augment(str(val_img_src), str(val_label_src), str(val_img_out), str(val_label_out))
    print(f"✅ 验证集完成: {val_count} 张图片")
    
    # ===== 处理测试集 (不增强) =====
    test_img_src = input_base_path / dataset_config.get('test', 'images/test')
    test_label_src = input_base_path / dataset_config.get('test').replace('images', 'labels')
    test_img_out = output_base_path / dataset_config.get('test', 'images/test')
    test_label_out = output_base_path / dataset_config.get('test').replace('images', 'labels')
    
    print(f"\n📋 处理测试集（直接复制，无增强）...")
    test_count = copy_dataset_without_augment(str(test_img_src), str(test_label_src), str(test_img_out), str(test_label_out))
    print(f"✅ 测试集完成: {test_count} 张图片")
    
    # ===== 生成新的 dataset.yaml =====
    print(f"\n📝 生成新的 dataset.yaml...")
    
    # 更新路径为相对路径
    new_config = dataset_config.copy()
    new_config['path'] = '.'
    
    output_yaml_path = output_base_path / DATASET_YAML
    with open(output_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(new_config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✅ 已生成: {output_yaml_path}")
    
    # ===== 总结 =====
    print(f"\n" + "="*60)
    print(f"🎉 数据集处理完成！")
    print(f"="*60)
    print(f"输出位置: {output_base_path}")
    print(f"  - 训练集: {train_img_out} (原 {len(img_files)} 张 -> 增强后 {train_count} 张)")
    print(f"  - 验证集: {val_img_out} ({val_count} 张)")
    print(f"  - 测试集: {test_img_out} ({test_count} 张)")
    print(f"  - 配置文件: {output_yaml_path}")
    print(f"="*60)


if __name__ == "__main__":
    process_yolo_dataset()
