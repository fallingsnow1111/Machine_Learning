# DINO-YOLO 训练指南

## 📋 两种训练方案

### 方案A：直接训练（快速实验）
直接使用YOLO11n预训练权重 + DINO模块增强

```bash
# 确保在 main 分支
git checkout main

# 激活环境
conda activate yolov11

# 直接训练（默认配置）
python Code/dino_yolo.py
```

---

### 方案B：蒸馏+微调（推荐，追求性能）
先用DINOv3蒸馏预训练，再进行有监督微调

#### 步骤1：蒸馏预训练（150 epochs，约8-10小时）
```bash
python Code/dino_distill_pretrain.py
```

#### 步骤2：修改权重路径
编辑 `Code/dino_yolo.py`，找到第60-68行：
```python
# 注释掉这行
# PRETRAINED_WEIGHTS = BASE_DIR / "pt" / "yolo11n.pt"

# 取消这行的注释
PRETRAINED_WEIGHTS = BASE_DIR / "runs" / "distill" / "dinov3_yolo11n" / "exported_models" / "exported_last.pt"
```

#### 步骤3：有监督微调（50 epochs，约2-3小时）
```bash
python Code/dino_yolo.py
```

---

## 🔧 当前配置（已优化）

### 模型架构
- **YAML配置**: `dino_yolo_SeNet.yaml`
- **特点**: SPPELAN + SeNet + P2/P3/P4/P5 四层检测头
- **检测头通道**: [32, 64, 128, 256] - 轻量化，防止显存溢出

### 训练参数
```python
EPOCHS = 50
IMG_SIZE = 1024
BATCH_SIZE = 8              # 双卡训练
OPTIMIZER = 'AdamW'
LR0 = 0.0005
CLOSE_MOSAIC = 15

# Loss权重（已调优）
box = 9.0                   # 提升定位精度
dfl = 2.0                   # 增强微小目标边界
```

### 数据配置
```python
DATA_YAML = "Data/Merged/no_noise11_processed/dataset.yaml"
```

---

## 📊 预期效果对比

| 方案 | mAP50 | Recall | 训练时间 | 推荐场景 |
|------|-------|--------|---------|---------|
| 方案A（直接训练） | ~79-80% | ~75-77% | 2-3小时 | 快速验证/资源受限 |
| 方案B（蒸馏+微调） | ~81-83%↑ | ~78-80%↑ | 10-13小时 | 追求极致性能 |

---

## ⚠️ 常见问题

### Q1: 显存不足 (OOM)
**解决方案**：
```python
# 修改 dino_yolo.py
BATCH_SIZE = 4              # 从8降到4
IMG_SIZE = 640              # 从1024降到640（不推荐）
```

### Q2: 蒸馏预训练失败
**原因**：lightly库未安装或版本不兼容
**解决方案**：
```bash
pip install lightly-train --upgrade
```

### Q3: 权重加载失败
**原因**：蒸馏权重路径不存在
**检查**：
```bash
ls runs/distill/dinov3_yolo11n/exported_models/exported_last.pt
```

---

## 📁 文件结构
```
Machine_Learning/
├── Code/
│   ├── dino_yolo.py                # 主训练脚本（已优化）
│   ├── dino_distill_pretrain.py    # 蒸馏预训练脚本（新增）
│   └── README_TRAINING.md          # 本文档
├── YAML/
│   ├── dino_yolo_SeNet.yaml        # ⭐ 当前使用的配置
│   ├── dino_yolo.yaml              # 备选配置
│   └── ...
├── Data/
│   └── Merged/
│       └── no_noise11_processed/   # 数据集
├── runs/
│   ├── distill/                    # 蒸馏输出（新）
│   └── detect/                     # 检测训练输出
└── pt/
    └── yolo11n.pt                  # 原始YOLO权重
```

---

## 🎯 推荐实验流程

### 第一轮：建立Baseline
```bash
# 使用方案A，记录指标
python Code/dino_yolo.py
# 记录: mAP50, Recall, 训练时间
```

### 第二轮：验证蒸馏效果
```bash
# 1. 蒸馏预训练
python Code/dino_distill_pretrain.py

# 2. 修改权重路径（见上文）
# 3. 微调训练
python Code/dino_yolo.py
# 对比: mAP50, Recall 是否提升
```

---

## 💡 参考

- **蒸馏策略来源**: `ziduo_test` 分支的成功经验
- **核心改进**: YOLO backbone学习DINOv3特征 + DINO模块增强
- **技术栈**: lightly-train + ultralytics + DINOv3
