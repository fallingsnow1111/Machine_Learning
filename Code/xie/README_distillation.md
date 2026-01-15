# DINOv3 到 YOLO11 知识蒸馏训练

这个项目使用 [lightly-train](https://github.com/lightly-ai/lightly-train) 将 DINOv3 的知识蒸馏到 YOLO11 模型中。

## 📋 目录

- [简介](#简介)
- [安装](#安装)
- [使用方法](#使用方法)
- [配置选项](#配置选项)
- [训练流程](#训练流程)
- [结果](#结果)

## 🎯 简介

知识蒸馏是一种模型压缩技术，通过让小模型（学生）学习大模型（教师）的知识，从而提高小模型的性能。

**训练流程：**
1. **阶段1 - 蒸馏预训练**：使用 DINOv3 教师模型在无标签数据上预训练 YOLO11 骨干网络
2. **阶段2 - 目标检测微调**：在标注的检测数据上微调预训练的模型

**优势：**
- 🚀 提升小模型性能（在 COCO 数据集上提升 2-3% mAP）
- 📦 保持模型体积小巧，适合部署
- 💡 充分利用无标签数据

## 📦 安装

### 1. 安装 lightly-train

```bash
pip install lightly-train
```

### 2. 安装其他依赖

```bash
pip install ultralytics torch torchvision
```

## 🚀 使用方法

### 快速开始

```python
from Code.xie.light_train_v2 import DINOv3ToYOLO11Distillation

# 创建训练器
trainer = DINOv3ToYOLO11Distillation(
    data_dir="Data/Raw/dust",
    output_dir="runs/distillation",
)

# 运行完整训练流程
trainer.run_full_pipeline(
    teacher_model="dinov3/vits16",      # DINOv3 教师模型
    student_model="ultralytics/yolo11n", # YOLO11 学生模型
    distillation_epochs=100,             # 蒸馏轮数
    finetune_epochs=50,                  # 微调轮数
    batch_size=16,
    image_size=640,
)
```

### 分步训练

如果你想分别控制两个阶段：

```python
from Code.xie.light_train_v2 import DINOv3ToYOLO11Distillation

trainer = DINOv3ToYOLO11Distillation(
    data_dir="Data/Raw/dust",
    output_dir="runs/distillation",
)

# 阶段1: 蒸馏预训练
pretrained_weights = trainer.stage1_distillation(
    teacher_model="dinov3/vits16",
    student_model="ultralytics/yolo11n",
    epochs=100,
    batch_size=32,
)

# 阶段2: 微调
trainer.stage2_finetune(
    pretrained_weights=pretrained_weights,
    epochs=50,
    batch_size=16,
)

# 验证
trainer.validate()
```

### 命令行运行

```bash
cd /home/xie/Others/Project/deeplearning/Machine_Learning
python Code/xie/light_train_v2.py
```

## ⚙️ 配置选项

### 教师模型选项

DINOv3 模型（推荐用于蒸馏）：
- `dinov3/vits16` - Small (22M 参数) - 快速训练
- `dinov3/vitb16` - Base (86M 参数) - 平衡性能和速度
- `dinov3/vitl16` - Large (304M 参数) - 最佳性能

### 学生模型选项

YOLO11 模型：
- `ultralytics/yolo11n` - Nano (2.6M 参数) - 最快速度
- `ultralytics/yolo11s` - Small (9.4M 参数) - 平衡
- `ultralytics/yolo11m` - Medium (20.1M 参数) - 更高精度
- `ultralytics/yolo11l` - Large (25.3M 参数) - 高精度
- `ultralytics/yolo11x` - Extra Large (56.9M 参数) - 最高精度

### 训练参数

```python
trainer.run_full_pipeline(
    teacher_model="dinov3/vits16",       # 教师模型
    student_model="ultralytics/yolo11n", # 学生模型
    distillation_epochs=100,              # 蒸馏预训练轮数 (建议 100-300)
    finetune_epochs=50,                   # 微调轮数 (建议 50-100)
    batch_size=16,                        # 批量大小 (根据 GPU 内存调整)
    image_size=640,                       # 图像大小 (640/1280)
)
```

## 📊 训练流程详解

### 阶段1: 蒸馏预训练

**目的**：在无标签图像上学习特征表示

**输入**：
- 无标签图像 (Data/Raw/dust/images/train/)
- DINOv3 教师模型

**输出**：
- 预训练的 YOLO11 骨干网络
- 保存位置: `runs/distillation/{experiment_name}/stage1_distillation/exported_models/exported_last.pt`

**训练过程**：
1. 教师模型（DINOv3）提取图像特征
2. 学生模型（YOLO11）学习模仿教师的特征
3. 使用蒸馏损失优化学生模型

### 阶段2: 目标检测微调

**目的**：在标注数据上训练检测头

**输入**：
- 标注的检测数据 (Data/Raw/dust/)
- 预训练的骨干网络

**输出**：
- 最终的检测模型
- 保存位置: `runs/distillation/{experiment_name}/stage2_finetune/train/weights/best.pt`

**训练过程**：
1. 加载预训练的骨干网络
2. 添加 YOLO 检测头
3. 在标注数据上微调整个模型
4. 使用检测损失（分类 + 定位 + 置信度）

## 📈 结果

训练完成后，结果保存在：

```
runs/distillation/{experiment_name}/
├── stage1_distillation/
│   ├── checkpoints/
│   │   └── last.ckpt
│   ├── exported_models/
│   │   └── exported_last.pt          # 预训练权重
│   └── events.out.tfevents.*         # TensorBoard 日志
└── stage2_finetune/
    └── train/
        ├── weights/
        │   ├── best.pt                # 最佳模型
        │   └── last.pt                # 最后一轮模型
        ├── results.png                # 训练曲线
        ├── confusion_matrix.png       # 混淆矩阵
        └── val_batch*.jpg             # 验证可视化
```

### 查看训练日志

```bash
# TensorBoard (阶段1)
tensorboard --logdir runs/distillation/{experiment_name}/stage1_distillation

# YOLO 训练日志 (阶段2)
# 查看 runs/distillation/{experiment_name}/stage2_finetune/train/results.png
```

## 🔧 常见问题

### 1. 内存不足

如果遇到 CUDA OOM 错误，尝试：
- 减少 `batch_size` (例如从 16 降到 8)
- 使用更小的教师模型 (`dinov3/vits16`)
- 减少 `image_size` (例如从 640 降到 416)

### 2. 训练太慢

- 使用更小的教师模型
- 减少 `distillation_epochs`
- 增加 `num_workers`

### 3. 效果不好

- 增加 `distillation_epochs` (建议至少 100 轮)
- 使用更大的教师模型 (`dinov3/vitb16` 或 `vitl16`)
- 确保有足够的无标签数据 (建议 >= 10,000 张图像)
- 增加 `finetune_epochs`

## 📚 参考资料

- [lightly-train 文档](https://docs.lightly.ai/train/)
- [lightly-train GitHub](https://github.com/lightly-ai/lightly-train)
- [DINOv3 论文](https://arxiv.org/abs/2304.07193)
- [YOLO11 文档](https://docs.ultralytics.com/)

## 🤝 贡献

欢迎提出问题和改进建议！

## 📄 许可证

- lightly-train: Apache 2.0
- DINOv3: DINOv3 License
- Ultralytics YOLO: AGPL-3.0
