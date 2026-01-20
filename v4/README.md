# V4: DINO蒸馏到YOLO11骨干网络

## 📚 项目说明

本版本实现了使用DINOv3作为教师模型，对YOLO11n骨干网络进行知识蒸馏预训练。

## 🎯 核心特性

1. **无监督预训练**：使用无标签图像即可进行蒸馏
2. **数据利用最大化**：
   - 有标签的灰尘图像（仅用图像，不用标签）
   - 无标签的无灰尘图像
   - 所有同场景图像都可使用
3. **自动权重融合**：蒸馏后的骨干网络权重自动合并到完整YOLO模型

## 📂 文件说明

### 核心脚本

- `prepare_distill_data.py` - 数据准备工具（合并有标签和无标签数据）
- `distill_pretrain.py` - DINO蒸馏预训练主脚本
- `train.py` - 改进的训练脚本（自动加载蒸馏权重）
- `test.py` - 测试脚本
- `pred.py` - 预测脚本

### 配置文件

- `yolo11P.yaml` - YOLO11模型配置
- `distill_config.py` - 蒸馏超参数配置

## 🚀 使用流程

### 步骤1: 准备蒸馏数据

```bash
python prepare_distill_data.py
```

此步骤会：

- 合并Data/dataset_no_noise_all的无灰尘图像
- 合并Data/dataset_yolo的有灰尘图像
- 输出到v4/distill_images/目录

### 步骤2: DINO蒸馏预训练（需要GPU）

```bash
python distill_pretrain.py
```

配置说明：

- 默认150轮训练（约2-4小时，视GPU性能）
- 批次大小16（可根据显存调整）
- 输出：runs/distill/yolo11n_distilled.pt

### 步骤3: 使用蒸馏权重训练检测模型

```bash
python train.py
```

脚本会自动检测并加载蒸馏权重

### 步骤4: 测试与预测

```bash
python test.py  # 测试
python pred.py  # 预测
```

## 📊 数据要求

### 最低要求

- 图像数量：≥1000张
- 图像类型：jpg/png
- 分辨率：不限（会自动resize到640×640）

### 推荐配置

- 图像数量：5000-10000张
- 混合使用有灰尘+无灰尘图像
- 覆盖不同光照、角度的场景

## ⚙️ 关键参数调整

### distill_config.py

```python
EPOCHS = 150          # 蒸馏轮数（越多效果越好，但耗时更长）
BATCH_SIZE = 16       # 批次大小（根据显存调整：8/16/32）
IMG_SIZE = 640        # 图像尺寸
LR = 1e-4            # 学习率
```

### train.py

```python
epochs = 50           # 检测训练轮数
batch = 32           # 检测训练批次
```

## 🔧 故障排查

### DINO模型加载失败

- 检查dinov3-vitl16.tar.gz是否存在
- 确认解压目录有config.json和model.safetensors

### 显存不足

- 减小BATCH_SIZE（16→8→4）
- 减小IMG_SIZE（640→512）

### 数据集为空

- 检查Data/dataset_no_noise_all/images/是否有图像
- 运行prepare_distill_data.py检查输出

## 📈 预期效果

使用蒸馏预训练后：

- ✅ 收敛速度提升20-30%
- ✅ 最终mAP提升2-5%
- ✅ 小目标检测精度提升
- ✅ 泛化能力增强
