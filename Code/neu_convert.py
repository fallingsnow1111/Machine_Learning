import xml.etree.ElementTree as ET
import os
import shutil
import glob
from typing import Iterable, Optional

# Try to import tqdm
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=""):
        print(f"Processing: {desc}")
        return iterable

# === 配置区域 ===
RAW_DATA_PATH = "Data/external/NEU-DET"
OUTPUT_PATH = "Data/NEU_YOLO"
CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
SUPPORTED_EXTS = (".jpg", ".bmp", ".png", ".jpeg")


def convert(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)


def convert_annotation(xml_path, output_txt_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find("size")
        if size is None:
            return
            
        w_node = size.find("width")
        h_node = size.find("height")
        if w_node is None or w_node.text is None or h_node is None or h_node.text is None:
            return

        w = int(w_node.text)
        h = int(h_node.text)

        with open(output_txt_path, "w", encoding="utf-8") as out_file:
            for obj in root.iter("object"):
                cls_node = obj.find("name")
                if cls_node is None or not cls_node.text:
                    continue
                cls = cls_node.text
                
                if cls not in CLASSES:
                    continue
                cls_id = CLASSES.index(cls)
                xmlbox = obj.find("bndbox")

                if xmlbox is None:
                    continue

                def get_val(tag):
                    node = xmlbox.find(tag)
                    return float(node.text) if node is not None and node.text else 0.0

                b = (
                    get_val("xmin"),
                    get_val("xmax"),
                    get_val("ymin"),
                    get_val("ymax"),
                )

                if sum(b) == 0:
                     continue

                bb = convert((w, h), b)
                out_file.write(f"{cls_id} {bb[0]} {bb[1]} {bb[2]} {bb[3]}\n")
    except Exception as e:
        print(f"Error converting {xml_path}: {e}")


def first_existing(paths: Iterable[str]) -> Optional[str]:
    for path in paths:
        if os.path.exists(path) and os.path.isdir(path):
            return path
    return None


def get_all_images(root_dir: str):
    images = []
    # Walk through directory to find all images including subdirectories
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(SUPPORTED_EXTS):
                images.append(os.path.join(root, file))
    return images


def process_split(raw_split: str, out_split: str):
    imgs_path = first_existing(
        [
            os.path.join(RAW_DATA_PATH, raw_split, "images"),
            os.path.join(RAW_DATA_PATH, raw_split, "IMAGES"),
        ]
    )
    anns_path = first_existing(
        [
            os.path.join(RAW_DATA_PATH, raw_split, "annotations"),
            os.path.join(RAW_DATA_PATH, raw_split, "ANNOTATIONS"),
        ]
    )

    if not imgs_path or not anns_path:
        print(f"Skipping {raw_split}: 找不到 images/annotations 目录")
        return

    os.makedirs(os.path.join(OUTPUT_PATH, "images", out_split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, "labels", out_split), exist_ok=True)

    all_imgs = get_all_images(imgs_path)
    print(f"{raw_split}: 发现 {len(all_imgs)} 张图片")

    for img_abs_path in tqdm(all_imgs, desc=f"{raw_split}->{out_split}"):
        file_name = os.path.splitext(os.path.basename(img_abs_path))[0]
        ext = os.path.splitext(img_abs_path)[1]

        # Copy image to flat output directory
        dst_img = os.path.join(OUTPUT_PATH, "images", out_split, file_name + ext)
        shutil.copy2(img_abs_path, dst_img)

        # Find corresponding XML in annotations folder (flat)
        xml_file = os.path.join(anns_path, file_name + ".xml")
        txt_file = os.path.join(OUTPUT_PATH, "labels", out_split, file_name + ".txt")
        
        if os.path.exists(xml_file):
            convert_annotation(xml_file, txt_file)


def process_dataset():
    # Map raw folder names to target folder names
    # Adjust "validation" to "val"
    split_map = {"train": "train", "validation": "val"}
    
    if not os.path.exists(RAW_DATA_PATH):
         print(f"Error: {RAW_DATA_PATH} does not exist.")
         return

    for raw_split, out_split in split_map.items():
        process_split(raw_split, out_split)


if __name__ == "__main__":
    process_dataset()