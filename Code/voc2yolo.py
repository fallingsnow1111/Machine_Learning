import os
import shutil
import xml.etree.ElementTree as ET

voc_root = '../Data/dataset'
yolo_root = '../Data/dataset_yolo'

classes = ['dust']
splits = ['train', 'val', 'test']

def convert_bbox(size, box):
    w, h = size
    xmin, ymin, xmax, ymax = box
    x = (xmin + xmax) / 2.0 / w
    y = (ymin + ymax) / 2.0 / h
    w = (xmax - xmin) / w
    h = (ymax - ymin) / h
    return x, y, w, h

def process_split(split):
    imgset_file = os.path.join(voc_root, 'ImageSets', 'Main', f'{split}.txt')
    if not os.path.exists(imgset_file):
        print(f'[WARN] {split}.txt 不存在，跳过')
        return

    img_out_dir = os.path.join(yolo_root, 'images', split)
    label_out_dir = os.path.join(yolo_root, 'labels', split)
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(label_out_dir, exist_ok=True)

    with open(imgset_file, encoding='utf-8') as f:
        image_ids = f.read().strip().split()

    for img_id in image_ids:
        # ---------- 复制图片 ----------
        src_img = os.path.join(voc_root, 'JPEGImages', f'{img_id}.jpg')
        dst_img = os.path.join(img_out_dir, f'{img_id}.jpg')
        if os.path.exists(src_img):
            shutil.copy(src_img, dst_img)
        else:
            print(f'[WARN] 图片不存在: {src_img}')
            continue

        # ---------- 转换标注 ----------
        xml_path = os.path.join(voc_root, 'Annotations', f'{img_id}.xml')
        label_path = os.path.join(label_out_dir, f'{img_id}.txt')

        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find('size')
        W = int(size.find('width').text)
        H = int(size.find('height').text)

        with open(label_path, 'w') as out:
            for obj in root.iter('object'):
                cls_name = obj.find('name').text
                if cls_name not in classes:
                    continue

                cls_id = classes.index(cls_name)
                bndbox = obj.find('bndbox')
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)

                x, y, w, h = convert_bbox((W, H), (xmin, ymin, xmax, ymax))
                out.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    print(f'[OK] {split} 集处理完成，共 {len(image_ids)} 张')


if __name__ == '__main__':
    for split in splits:
        process_split(split)

    # 生成 dataset.yaml
    yaml_path = os.path.join(yolo_root, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"path: {yolo_root}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write(f"nc: {len(classes)}\n")
        f.write("names: " + str(classes) + "\n")

    print('[DONE] VOC → YOLO 数据集构建完成')