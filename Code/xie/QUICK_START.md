# 🚀 DINOv3 到 YOLO11 知识蒸馏 - 快速使用指南

## 📝 项目文件说明

已为你创建以下文件：

```
Code/xie/
├── light_train_v2.py              # 主训练脚本（核心类）
├── quick_start_distillation.py   # 快速开始示例
├── train_with_config.py           # 配置文件驱动的训练脚本
├── distillation_config.yaml       # 训练配置文件
├── run_training.sh                # Bash 训练脚本（推荐）
├── README_distillation.md         # 详细文档
└── QUICK_START.md                 # 本文件
```

## ⚡ 快速开始（3步）

### 方法1: 使用 Bash 脚本（推荐）

```bash
# 进入项目目录
cd /home/xie/Others/Project/deeplearning/Machine_Learning

# 快速测试（10轮，验证流程）
./Code/xie/run_training.sh quick-test

# 标准训练（推荐）
./Code/xie/run_training.sh standard

# 高性能训练
./Code/xie/run_training.sh high-performance
```

### 方法2: 使用 Python 脚本

```bash
cd /home/xie/Others/Project/deeplearning/Machine_Learning

# 快速测试
python Code/xie/quick_start_distillation.py

# 或使用配置文件
python Code/xie/train_with_config.py --config Code/xie/distillation_config.yaml --template quick_test
```

### 方法3: 直接运行主脚本

```bash
cd /home/xie/Others/Project/deeplearning/Machine_Learning
python Code/xie/light_train_v2.py
```

## 🎯 训练模式选择

### 1️⃣ 快速测试（首次运行推荐）

**用途**: 验证环境和数据，快速看到结果  
**时间**: ~30分钟  
**配置**: 10轮蒸馏 + 10轮微调

```bash
./Code/xie/run_training.sh quick-test
```

### 2️⃣ 标准训练（日常使用推荐）

**用途**: 正常训练，获得良好效果  
**时间**: ~8-12小时  
**配置**: 100轮蒸馏 + 50轮微调

```bash
./Code/xie/run_training.sh standard
```

### 3️⃣ 高性能训练（追求最佳效果）

**用途**: 追求最高精度  
**时间**: ~24-36小时  
**配置**: 200轮蒸馏 + 100轮微调，使用更大的教师模型

```bash
./Code/xie/run_training.sh high-performance
```

### 4️⃣ 低资源训练（小GPU）

**用途**: GPU 内存不足时使用  
**配置**: 小批量 + 小图像尺寸

```bash
./Code/xie/run_training.sh low-resource
```

## 📊 查看训练结果

训练完成后，结果保存在：

```
runs/distillation/{experiment_name}/
├── stage1_distillation/
│   ├── exported_models/
│   │   └── exported_last.pt          # 预训练权重
│   └── events.out.tfevents.*         # TensorBoard 日志
└── stage2_finetune/
    └── train/
        ├── weights/
        │   ├── best.pt                # 最佳模型 ⭐
        │   └── last.pt                # 最后一轮模型
        ├── results.png                # 训练曲线
        └── confusion_matrix.png       # 混淆矩阵
```

### 查看训练日志

```bash
# TensorBoard (阶段1蒸馏)
tensorboard --logdir runs/distillation/{experiment_name}/stage1_distillation

# 查看训练曲线 (阶段2微调)
eog runs/distillation/{experiment_name}/stage2_finetune/train/results.png
```

## 🔧 自定义配置

编辑 `Code/xie/distillation_config.yaml` 文件：

```yaml
# 修改教师模型（更大 = 更好效果，更慢）
model:
  teacher: "dinov3/vitb16"  # vits16 < vitb16 < vitl16

# 修改学生模型（更大 = 更高精度，更慢）
model:
  student: "ultralytics/yolo11s"  # n < s < m < l < x

# 修改训练轮数
stage1_distillation:
  epochs: 100  # 增加以提高效果

stage2_finetune:
  epochs: 50   # 增加以提高效果

# 调整批量大小（根据GPU内存）
stage1_distillation:
  batch_size: 16  # 减小以节省内存

stage2_finetune:
  batch_size: 16  # 减小以节省内存
```

然后运行：

```bash
python Code/xie/train_with_config.py --config Code/xie/distillation_config.yaml
```

## 📈 模型推理

训练完成后，使用最佳模型进行推理：

```python
from ultralytics import YOLO

# 加载最佳模型
model = YOLO("runs/distillation/{experiment_name}/stage2_finetune/train/weights/best.pt")

# 单张图像预测
results = model.predict("path/to/image.jpg")

# 批量预测
results = model.predict("path/to/images/", save=True)

# 视频预测
results = model.predict("path/to/video.mp4", save=True)
```

## 💡 常见问题

### Q: CUDA out of memory 错误？

A: 减少批量大小
```bash
# 编辑 distillation_config.yaml
stage1_distillation:
  batch_size: 8  # 从 16 减到 8

stage2_finetune:
  batch_size: 8  # 从 16 减到 8
```

### Q: 训练太慢？

A: 使用更小的模型
```bash
# 编辑 distillation_config.yaml
model:
  teacher: "dinov3/vits16"  # 最小的教师模型
  student: "ultralytics/yolo11n"  # 最小的学生模型
```

### Q: 效果不够好？

A: 增加训练轮数或使用更大的教师模型
```bash
# 编辑 distillation_config.yaml
model:
  teacher: "dinov3/vitb16"  # 或 vitl16

stage1_distillation:
  epochs: 200  # 增加蒸馏轮数
```

### Q: 只想运行蒸馏预训练？

```bash
./Code/xie/run_training.sh stage1-only
```

### Q: 只想运行微调？

```bash
./Code/xie/run_training.sh stage2-only <预训练权重路径>
```

## 🎓 推荐训练流程

### 第一次使用：

1. **快速测试** (10-30分钟)
   ```bash
   ./Code/xie/run_training.sh quick-test
   ```
   验证环境和数据没问题

2. **标准训练** (8-12小时)
   ```bash
   ./Code/xie/run_training.sh standard
   ```
   获得基准结果

3. **调优** (根据需要)
   - 如果效果好：完成！
   - 如果效果不够：尝试高性能配置
   - 如果内存不足：使用低资源配置

## 📚 更多信息

- 详细文档: [README_distillation.md](README_distillation.md)
- lightly-train 文档: https://docs.lightly.ai/train/
- YOLO11 文档: https://docs.ultralytics.com/

## 🚀 现在就开始！

```bash
cd /home/xie/Others/Project/deeplearning/Machine_Learning
./Code/xie/run_training.sh quick-test
```

祝训练顺利！🎉
