"""
蒸馏配置文件 - 所有可调参数集中管理
"""

# ===================== 路径配置 =====================
# DINO模型配置（从ModelScope直接下载，无需本地文件）
DINO_MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"

# 项目内路径配置
DISTILL_DATA_DIR = "./Data/distill_images"      # 蒸馏数据目录（由prepare_distill_data.py生成）
OUTPUT_DIR = "./runs/distill"              # 蒸馏输出目录
YOLO11N_WEIGHTS = "./yolo11n.pt"          # YOLO11n预训练权重

# ===================== 训练超参数 =====================
EPOCHS = 50              # 蒸馏训练轮数（推荐：100-200）
BATCH_SIZE = 16           # 批次大小（根据显存调整：4/8/16/32）
IMG_SIZE = 640            # 图像尺寸
LR = 1e-4                # 学习率
WEIGHT_DECAY = 0.02       # 权重衰减

# ===================== 蒸馏损失权重 =====================
DISTILL_LOSS_WEIGHT = 1.0     # 蒸馏损失权重
VAR_LOSS_WEIGHT = 0.1         # 方差损失权重（鼓励特征多样性）
NORM_LOSS_WEIGHT = 0.01       # 归一化损失权重

# ===================== 骨干网络配置 =====================
BACKBONE_LAYER_IDX = 10       # YOLO11n骨干网络截取层数（默认10层）
ADAPTER_HIDDEN_DIM = 1024     # 适配器隐藏层维度

# ===================== 数据加载配置 =====================
NUM_WORKERS = 0              # DataLoader工作进程数（Jupyter环境必须为0）
PREFETCH_FACTOR = 2          # 预取批次数

# ===================== 检查点保存 =====================
SAVE_CHECKPOINT_INTERVAL = 50  # 每N轮保存一次检查点
SAVE_FINAL_WEIGHTS = True      # 是否保存最终完整模型权重

# ===================== 设备配置 =====================
# 自动检测GPU，也可手动指定 "cuda" 或 "cpu"
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_DATAPARALLEL = True  # 是否使用双卡/多卡并行（检测到多GPU时自动启用）

# ===================== 数据增强 =====================
NORMALIZE_MEAN = [0.485, 0.456, 0.406]  # ImageNet标准均值
NORMALIZE_STD = [0.229, 0.224, 0.225]   # ImageNet标准方差

# ===================== 调试模式 =====================
DEBUG_MODE = False            # 调试模式（只训练少量batch）
DEBUG_MAX_BATCHES = 10        # 调试模式最大batch数
