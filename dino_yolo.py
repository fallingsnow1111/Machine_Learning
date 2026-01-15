"""
DINO-YOLO 融合训练脚本
结合了 DINOv2 双注入架构 (P0/P3) 与灰尘检测优化方案 (P2 + ASPP + EMA)
"""

import sys
import os
import torch
from pathlib import Path
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks

# 导入所有自定义模块
from custom_modules import ASPP, EMA
from custom_modules.dino import DINOInputAdapter, DINOMidAdapter


def register_custom_layers():
    """注册所有自定义模块到 YOLO 构建系统"""
    setattr(tasks, "ASPP", ASPP)
    setattr(tasks, "EMA", EMA)
    setattr(tasks, "DINOInputAdapter", DINOInputAdapter)
    setattr(tasks, "DINOMidAdapter", DINOMidAdapter)
    print("✅ 模块注册完成：ASPP, EMA, DINOInputAdapter, DINOMidAdapter")


# ==================== 配置区 ====================
# 请根据你的实际路径修改以下变量

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent

# 数据集配置文件路径（修改为你的实际路径）
DATA_YAML = str(PROJECT_ROOT / "Data/Raw/dust/dataset.yaml")

# 模型配置文件
MODEL_YAML = str(PROJECT_ROOT / "dino_yolo.yaml")

# 预训练权重（只用来初始化骨干网络）
WEIGHTS = str(PROJECT_ROOT / "pt/yolo11n.pt")

# 训练参数
TRAIN_CONFIG = {
    "data": DATA_YAML,
    "epochs": 50,
    "imgsz": 1024,  # 建议用 1024，DINO 在大图上对小目标纹理提取更准
    "batch": 4,     # 如果 OOM，降到 4 或 2，并开启 accumulate
    "device": "0",
    "optimizer": "AdamW",
    "lr0": 0.0005,
    "weight_decay": 0.0001,
    "warmup_epochs": 3,
    "project": "dust_detection",
    "name": "dino_p2_aspp_ema",
    "patience": 15,
    "save": True,
    "save_period": 5,  # 每 5 个 epoch 保存一次
    "cache": False,    # 如果数据集不大可以设为 True 加速
    "workers": 4,
    "amp": True,       # 混合精度训练，节省显存
    # "accumulate": 4,  # 如果显存不足，取消注释这行（梯度累加）
}


def check_environment():
    """检查训练环境"""
    print("\n" + "="*60)
    print("🔍 环境检查")
    print("="*60)
    
    # 检查 CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用！请检查 GPU 驱动")
        return False
    
    print(f"✅ CUDA 可用：{torch.cuda.get_device_name(0)}")
    print(f"   显存：{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 检查数据集
    if not Path(DATA_YAML).exists():
        print(f"❌ 数据集配置文件不存在：{DATA_YAML}")
        print(f"   请修改脚本中的 DATA_YAML 变量")
        return False
    
    print(f"✅ 数据集配置：{DATA_YAML}")
    
    # 检查模型配置
    if not Path(MODEL_YAML).exists():
        print(f"❌ 模型配置文件不存在：{MODEL_YAML}")
        return False
    
    print(f"✅ 模型配置：{MODEL_YAML}")
    
    # 检查预训练权重
    if not Path(WEIGHTS).exists():
        print(f"⚠️  预训练权重不存在：{WEIGHTS}")
        print(f"   将从随机权重开始训练")
    else:
        print(f"✅ 预训练权重：{WEIGHTS}")
    
    print("="*60 + "\n")
    return True


def main():
    """主训练流程"""
    print("\n" + "="*60)
    print("🚀 DINO-YOLO 融合模型训练")
    print("="*60)
    
    # 环境检查
    if not check_environment():
        print("❌ 环境检查失败，请修复上述问题后重试")
        return
    
    # 注册自定义模块
    register_custom_layers()
    
    # 建立模型
    print("\n📦 加载模型...")
    print("⚠️  首次运行会自动下载 DINOv2 权重（约 100MB），请耐心等待")
    
    try:
        model = YOLO(MODEL_YAML)
        print("✅ 模型结构创建成功")
        
        # 尝试加载预训练权重
        if Path(WEIGHTS).exists():
            try:
                model.load(WEIGHTS)
                print("✅ YOLO 预训练权重加载成功")
            except Exception as e:
                print(f"⚠️  权重部分加载失败（这是正常的，因为结构大改）：{e}")
                print("   将使用可用的权重继续训练")
    
    except Exception as e:
        print(f"❌ 模型创建失败：{e}")
        return
    
    # 开始训练
    print("\n" + "="*60)
    print("🎯 开始训练")
    print("="*60)
    print(f"📊 训练配置：")
    for k, v in TRAIN_CONFIG.items():
        print(f"   {k}: {v}")
    print("="*60 + "\n")
    
    try:
        results = model.train(**TRAIN_CONFIG)
        
        print("\n" + "="*60)
        print("✅ 训练完成！")
        print("="*60)
        print(f"📁 结果保存在：{PROJECT_ROOT / TRAIN_CONFIG['project'] / TRAIN_CONFIG['name']}")
        print("="*60 + "\n")
        
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练出错：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
