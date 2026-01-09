import cv2
import numpy as np
import os
import yaml
from pathlib import Path
import argparse

# ================= 配置 =================
DEFAULT_DATASET_DIR = r"./Data/merged_dataset"
DEFAULT_SPLIT = "train"  # 可选: train, val, test
DEFAULT_CLASS_NAMES = ['dust']  # 默认类别名称，会从yaml中读取

# ================= 可视化逻辑 =================

class YOLOVisualizer:
    def __init__(self, dataset_dir, split="train", class_names=None, resize_size=None):
        """
        初始化可视化器
        
        Args:
            dataset_dir: 数据集根目录
            split: 数据集分割 (train/val/test)
            class_names: 类别名称列表
            resize_size: 图片resize大小，如 (640, 640)，None 表示不resize
        """
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.class_names = class_names or DEFAULT_CLASS_NAMES
        self.resize_size = resize_size
        
        # 读取 dataset.yaml
        yaml_path = self.dataset_dir / "dataset.yaml"
        if yaml_path.exists():
            with open(yaml_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 获取类别名称
            if 'names' in config:
                self.class_names = config['names']
            
            # 获取数据集路径
            if split in config:
                self.split_path = config[split]
            else:
                self.split_path = f"images/{split}"
        else:
            self.split_path = f"images/{split}"
        
        # 构建完整路径
        if Path(self.split_path).is_absolute():
             self.img_dir = Path(self.split_path)
        else:
             self.img_dir = self.dataset_dir / self.split_path
             
        self.label_dir = self.dataset_dir / self.split_path.replace("images", "labels")
        
        # Verify directory exists
        if not self.img_dir.exists():
            print(f"❌ 图片目录不存在: {self.img_dir.resolve()}")
            
        # 获取所有图片 (Case insensitive search for extensions)
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
        self.img_files = []
        if self.img_dir.exists():
            for ext in extensions:
                 self.img_files.extend(list(self.img_dir.glob(ext.lower())))
                 self.img_files.extend(list(self.img_dir.glob(ext.upper())))
        
        # Remove duplicates and sort
        self.img_files = sorted(list(set(self.img_files)))
        
        self.current_idx = 0
        self.total_imgs = len(self.img_files)
        
        print(f"✅ 已加载 {self.total_imgs} 张图片")
        print(f"📁 图片目录: {self.img_dir}")
        print(f"📁 标签目录: {self.label_dir}")
        print(f"🏷️  类别: {self.class_names}")
        if self.resize_size:
            print(f"📖 Resize 大小: {self.resize_size}")
    
    def get_bboxes(self, img_path):
        """
        读取图片对应的 YOLO 格式标签
        返回: [(class_id, x_center, y_center, w, h), ...]
        """
        label_path = self.label_dir / img_path.with_suffix(".txt").name
        bboxes = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = [float(x) for x in line.split()]
                        if len(parts) >= 5:
                            bboxes.append(parts[:5])  # [class_id, x_c, y_c, w, h]
        
        return bboxes
    
    def resize_image(self, img):
        """
        将图片 resize 到指定大小（使用填充保持宽高比）
        """
        if self.resize_size is None:
            return img
        
        target_h, target_w = self.resize_size
        h, w = img.shape[:2]
        
        # 计算缩放比例
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 缩放图片
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 创建目标大小的图片（灰色背景）
        canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 128
        
        # 计算放置位置（居中）
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        
        # 将缩放后的图片粘贴到画布
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas
    
    def draw_bboxes(self, img, bboxes):
        """
        在图片上绘制边界框 (优化版：针对小目标优化显示)
        """
        img_copy = img.copy()
        h, w, _ = img.shape
        
        # 定义颜色
        colors = [
            (0, 255, 0),    # 绿色
            (0, 0, 255),    # 红色
            (255, 0, 0),    # 蓝色
            (0, 255, 255),  # 黄色
            (255, 0, 255),  # 品红
            (255, 255, 0),  # 青色
        ]
        
        for idx, bbox in enumerate(bboxes):
            class_id = int(bbox[0])
            x_c, y_c, box_w, box_h = bbox[1:]
            
            # 归一化坐标转像素坐标
            x_center = x_c * w
            y_center = y_c * h
            box_width = box_w * w
            box_height = box_h * h
            
            # 计算左上和右下角
            x1 = int(x_center - box_width / 2)
            y1 = int(y_center - box_height / 2)
            x2 = int(x_center + box_width / 2)
            y2 = int(y_center + box_height / 2)
            
            # 确保坐标在图片范围内
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)
            
            # 选择颜色
            color = colors[class_id % len(colors)]
            
            # -----------------------------------------------------------
            # 【修改点 1】: 框的线条粗细改为 1 (原来是 2)
            # -----------------------------------------------------------
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 1)
            
            # 绘制类别标签
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class {class_id}"
            label_text = f"{class_name}" # 去掉了 id 显示，让标签更短
            
            # -----------------------------------------------------------
            # 【修改点 2】: 字体大小计算逻辑 (让字变得更小)
            # -----------------------------------------------------------
            font = cv2.FONT_HERSHEY_SIMPLEX
            # 原来是 / 1000，现在改为 / 2000 (字缩小一倍)，最小 0.35
            font_scale = max(0.35, min(w, h) / 2000)
            # 字体粗细固定为 1 (原来是动态计算，可能会太粗)
            thickness = 1
            
            text_size = cv2.getTextSize(label_text, font, font_scale, thickness)[0]
            
            # -----------------------------------------------------------
            # 【修改点 3】: 不显示标签文字，避免挡住小目标
            # -----------------------------------------------------------
            
            # 已注释：不绘制标签背景和文字
            # bg_x1 = x1
            # bg_y1 = max(y1 - text_size[1] - 4, 0)
            # bg_x2 = x1 + text_size[0] + 2
            # bg_y2 = y1
            # cv2.rectangle(img_copy, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
            # cv2.putText(img_copy, label_text, (x1 + 1, y1 - 2), 
            #             font, font_scale, (255, 255, 255), thickness)
        
        return img_copy
    
    def show_image(self, idx):
        """
        显示指定索引的图片及其标注
        """
        if idx < 0 or idx >= self.total_imgs:
            print(f"❌ 索引超出范围: {idx}")
            return False
        
        img_path = self.img_files[idx]
        img = cv2.imread(str(img_path))
        
        if img is None:
            print(f"❌ 无法读取图片: {img_path}")
            return False
        
        # 读取标签
        bboxes = self.get_bboxes(img_path)
        
        # === 修复点：先在原图上画框，再 Resize ===
        # 1. 绘制边界框（使用原图坐标系，绝对准确）
        img_with_boxes = self.draw_bboxes(img, bboxes)
        
        # 2. 缩放带框的图片用于显示
        final_img = self.resize_image(img_with_boxes)
        # ======================================
        
        # 创建信息文字
        img_shape = img.shape
        resize_info = f" [Resized: {final_img.shape[1]}x{final_img.shape[0]}]" if self.resize_size else ""
        info_text = f"Img {idx + 1}/{self.total_imgs} | Obj: {len(bboxes)} | {img_path.name}{resize_info}"
        
        # 在图片上添加信息
        cv2.putText(final_img, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 显示图片
        window_name = f"YOLO Dataset Vis - {self.split}"
        cv2.imshow(window_name, final_img)
        
        return True
    
    def interactive_view(self):
        """交互式浏览数据集"""
        print("\n" + "="*60)
        print("📖 交互式浏览模式")
        print("="*60)
        print("按键说明:")
        print("  [→] 或 [d] : 下一张图片")
        print("  [←] 或 [a] : 上一张图片")
        print("  [q]       : 退出")
        print("  [s]       : 保存当前图片")
        print("  [g]       : 跳转到指定图片编号")
        print("  [r]       : 切换 Resize 模式")
        print("="*60 + "\n")
        
        # 第一次显示
        self.show_image(self.current_idx)
        
        while True:
            # 持续刷新显示（适配 OpenCV 的窗口机制）
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q'):
                print("✅ 退出")
                break
            elif key == ord('d') or key == 83:  # → 右箭头
                self.current_idx = min(self.current_idx + 1, self.total_imgs - 1)
                self.show_image(self.current_idx)
            elif key == ord('a') or key == 81:  # ← 左箭头
                self.current_idx = max(self.current_idx - 1, 0)
                self.show_image(self.current_idx)
            elif key == ord('s'):
                self._save_current_image()
            elif key == ord('g'):
                self._goto_image()
                self.show_image(self.current_idx)
            elif key == ord('r'):
                self._toggle_resize()
                self.show_image(self.current_idx)
        
        cv2.destroyAllWindows()
    
    def _save_current_image(self):
        """保存当前显示的图片"""
        img_path = self.img_files[self.current_idx]
        save_path = f"vis_{img_path.stem}_annotated.jpg"
        
        img = cv2.imread(str(img_path))
        bboxes = self.get_bboxes(img_path)
        img_with_boxes = self.draw_bboxes(img, bboxes)
        
        cv2.imwrite(save_path, img_with_boxes)
        print(f"✅ 已保存到: {save_path}")
    
    def _goto_image(self):
        """跳转到指定图片"""
        try:
            idx = int(input(f"请输入图片编号 (1-{self.total_imgs}): ")) - 1
            if 0 <= idx < self.total_imgs:
                self.current_idx = idx
            else:
                print(f"❌ 无效的图片编号")
        except ValueError:
            print("❌ 输入无效")
    
    def _toggle_resize(self):
        """切换 Resize 模式"""
        if self.resize_size:
            self.resize_size = None
            print("✅ 已关闭 Resize 模式，显示原始图片")
        else:
            self.resize_size = (640, 640)
            print("✅ 已启用 Resize 模式，图片将调整为 640x640")
    
    def batch_export(self, output_dir="./yolo_visualized"):
        """
        批量导出所有图片的标注结果
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\n📤 正在批量导出标注图片到 {output_dir}...")
        resize_info = f" (Resize to {self.resize_size[0]}x{self.resize_size[1]})" if self.resize_size else ""
        print(f"   {resize_info}")
        
        for idx, img_path in enumerate(self.img_files):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # === 修复点：先画框，再 Resize ===
            bboxes = self.get_bboxes(img_path)
            img_with_boxes = self.draw_bboxes(img, bboxes)
            final_img = self.resize_image(img_with_boxes)
            # ===============================
            
            save_path = output_path / img_path.name
            cv2.imwrite(str(save_path), final_img)
            
            if (idx + 1) % 10 == 0:
                print(f"   已处理: {idx + 1}/{self.total_imgs}")
        
        print(f"✅ 批量导出完成！共 {self.total_imgs} 张图片")


# ================= 主程序 =================

def main():
    parser = argparse.ArgumentParser(
        description="YOLO 数据集标注可视化工具 (修复版)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python vis_yolo.py                                   # 使用默认配置
  python vis_yolo.py --dataset ./dataset_yolo          # 指定数据集目录
  python vis_yolo.py --split val                       # 可视化验证集
  python vis_yolo.py --resize 640                      # 显示时 resize 到 640x640
  python vis_yolo.py --export                          # 批量导出所有标注
        """
    )
    
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_DIR,
                        help="数据集根目录路径")
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT, 
                        help="数据集分割 (train/val/test) 或者 '.' 用于无子目录结构")
    parser.add_argument("--export", action="store_true",
                        help="批量导出所有图片的标注结果")
    parser.add_argument("--output", type=str, default="./yolo_visualized",
                        help="导出目录")
    parser.add_argument("--classes", type=str, default=None,
                        help="类别名称，逗号分隔 (如: dust,defect,scratch)")
    parser.add_argument("--resize", type=int, default=None,
                        help="将图片 resize 到指定大小 (如: 640 表示 640x640)")
    
    args = parser.parse_args()
    
    # 解析类别名称
    class_names = None
    if args.classes:
        class_names = [c.strip() for c in args.classes.split(",")]
    
    # 解析 resize 大小
    resize_size = None
    if args.resize:
        resize_size = (args.resize, args.resize)
    
    # 创建可视化器
    viz = YOLOVisualizer(args.dataset, args.split, class_names, resize_size)
    
    # 执行操作
    if args.export:
        viz.batch_export(args.output)
    else:
        viz.interactive_view()


if __name__ == "__main__":
    main()