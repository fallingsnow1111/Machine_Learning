import subprocess
import sys

def install_dependencies(verbose: bool = False):
    """安装所需的依赖包"""
    dependencies = [
        "lightly-train",           # Lightly库用于自监督学习
        "ultralytics",       # YOLO11
        "torch",             # PyTorch
        "torchvision",       # 视觉工具
        "pillow",            # 图像处理
        "opencv-python",     # OpenCV
        "matplotlib",        # 可视化
        "numpy",             # 数值计算
        "pyyaml",            # YAML配置文件
        "tqdm",              # 进度条
    ]
    for package in dependencies:
        try:
            __import__(package.replace("-", "_").split("[")[0])
        except ImportError:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ])
    
    print("="*60)
    print("所有依赖已就绪！\n")

# 在导入其他模块前先安装依赖
if __name__ == "__main__":
    install_dependencies()

import lightly_train
from lightly_train import pretrain

if __name__ == "__main__": 
    # 从 DINOv3 蒸馏到 YOLO11n 用于 OLED 灰尘检测
    lightly_train.pretrain(
        # 输出目录
        out="runs/out/oled_dust_dinov3_yolo11n",
        
        # 数据集路径（包含您的500张灰度图）
        # 可以直接指向图片文件夹，不需要标签
        data="Data/Processed/dust_processed",
        
        # 学生模型：YOLO11n（最小的YOLO11模型）
        model="ultralytics/yolo11n",
        
        # 蒸馏方法
        method="distillation",
        
        # 方法参数
        method_args={
            # 教师模型：使用 DINOv3 的最小模型以适配小数据集
            "teacher":  "dinov3/vitt16",  # tiny 模型，适合小数据集
            
            # 温度参数（控制软标签的平滑度）
            "temperature": 0.1,
        },
        
        # 训练超参数（针对小数据集优化）
        epochs=200,              # 小数据集需要更多epochs
        batch_size=16,           # 小batch size适合500张图片
        
        # 数据增强设置
        transform_args={
            # 图像尺寸
            "image_size": (640, 640),
            
            # 针对灰度图像的通道设置
            "num_channels": 1,    # 灰度图为1通道
            
            # 数据增强参数（针对工业检测场景）
            "color_jitter": {
                "prob": 0.5,      # 降低颜色抖动概率
                "brightness": 0.2,
                "contrast": 0.2,
                "saturation": 0.0, # 灰度图不需要饱和度调整
                "hue": 0.0,        # 灰度图不需要色调调整
            },

            # 随机翻转（适合灰尘检测）
            "random_flip": {
                "horizontal_prob": 0.5,
                "vertical_prob": 0.5,
            },
            
            # 随机旋转（灰尘方向不固定）
            "random_rotation": {
                "degrees": 90,
                "prob": 0.5,
            },
        },
        
        
        # 设备设置
        devices=2,                 # 使用2个GPU
        seed=42,                   # 固定随机种子保证可重复性
    )
    
    print("✅ 蒸馏训练完成！")
    print(f"模型保存在:  runs/out/oled_dust_dinov3_yolo11n/exported_models/")
    print(f"可以使用该模型进行后续的目标检测微调")

from ultralytics import YOLO

if __name__ == "__main__": 
    # 加载蒸馏预训练的模型
    model = YOLO("runs/out/oled_dust_dinov3_yolo11n/exported_models/exported_last.pt")
    
    # 使用您的YOLO格式标签进行微调
    results = model.train(
        data="Data/Processed/dust_processed/dataset.yaml",   # 您的数据集配置文件
        epochs=300,              # 小数据集需要更多epochs
        imgsz=640,
        batch=8,
        patience=50,             # 早停耐心值
        save=True,
        cache=True,              # 缓存图像加速训练
        
        # 小数据集优化设置
        lr0=0.001,               # 初始学习率
        lrf=0.01,                # 最终学习率系数
        momentum=0.9,
        weight_decay=0.0005,
        warmup_epochs=3,
        
        # 数据增强（针对灰度图和小数据集）
        augment=True,
        hsv_h=0.0,               # 灰度图不需要色调调整
        hsv_s=0.0,               # 灰度图不需要饱和度调整
        hsv_v=0.2,               # 适度亮度调整
        degrees=10.0,            # 旋转角度
        translate=0.1,           # 平移
        scale=0.2,               # 缩放
        flipud=0.5,              # 上下翻转
        fliplr=0.5,              # 左右翻转
        mosaic=0.5,              # 马赛克增强概率
        mixup=0.1,               # mixup增强
    )

    print("\n" + "="*50)
    print("✅ 微调训练完成！开始验证...")
    print("="*50 + "\n")
    
    # 在验证集上评估模型
    val_results = model.val(
        data="Data/Processed/dust_processed/dataset.yaml",
        imgsz=640,
        batch=8,
        conf=0.25,               # 置信度阈值
        iou=0.6,                 # NMS的IoU阈值
        plots=True,              # 生成验证图表
        save_json=True,          # 保存结果为JSON
    )
    
    # 打印验证结果
    print("\n" + "="*50)
    print("📊 验证结果:")
    print("="*50)
    print(f"mAP50: {val_results.box.map50:.4f}")
    print(f"mAP50-95: {val_results.box.map:.4f}")
    print(f"Precision: {val_results.box.mp:.4f}")
    print(f"Recall: {val_results.box.mr:.4f}")
    print("="*50 + "\n")
    
    print(f"✅ 最终模型保存在: {model.ckpt_path}")
    print(f"📈 验证图表保存在: runs/detect/val/")