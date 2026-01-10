import os
from ultralytics import YOLO

# ================= 配置区域 =================
# 你的自定义模型配置文件路径
MODEL_YAML = "./yolo11P.yaml" 
# 你的数据集配置文件路径
DATA_YAML = "Data/dataset_yolo/dataset.yaml"   

# 训练参数设置
IMG_SIZE = 640
BATCH_SIZE = 16      
PROJECT_NAME = "runs/detect/dust_detection" # 项目名称，结果会保存在 runs/detect/dust_detection 下

# 阶段一配置 
STAGE1_EPOCHS = 15   # 跑 15 轮让 Head 适应
STAGE1_LR = 0.01     # 初始学习率 (默认值)

# 阶段二配置 
STAGE2_EPOCHS = 85   # 剩余轮次 (总共 100 轮)
STAGE2_LR = 0.001    # ⚠️ 关键：降低 10 倍学习率，防止破坏 Backbone 特征


def train(MODEL_YAML, DATA_YAML, IMG_SIZE, BATCH_SIZE, PROJECT_NAME, STAGE1_EPOCHS, STAGE1_LR, STAGE2_EPOCHS, STAGE2_LR):
    # ================= 阶段一：冻结骨干训练 =================
    print("\n" + "="*40)
    print("🚀 开始阶段一：冻结 Backbone (前10层) 训练 Head...")
    print("="*40 + "\n")
    
    # 1. 初始化模型 (从 YAML 构建新结构)
    model = YOLO(MODEL_YAML)
    
    # try to load pretrain parameters
    try:
        model.load("./yolo11n.pt") 
        print("成功加载预训练权重！")
    except Exception as e:
        print(f"加载权重跳过或出错: {e}")
    
    # 2. 开始训练
    # 注意：我们设置 name='stage1'，结果会保存在 runs/detect/dust_detection/stage1
    results_stage1 = model.train(
        data=DATA_YAML,
        epochs=STAGE1_EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        freeze=10,           # <--- 核心：冻结前10层
        project=PROJECT_NAME,
        name="stage1_freeze",
        patience=0,          # 阶段一不要早停，强制跑完让 Head 充分初始化
        lr0=STAGE1_LR,
        degrees=5.0,
        translate=0.05,
        scale=0.1,
        copy_paste=0.4,
        device=0,
        warmup_epochs=5 , 
    )
    
    # 3. 获取阶段一的最佳权重路径
    # results_stage1.save_dir 会自动指向 runs/detect/dust_detection/stage1_freeze
    stage1_weight_path = os.path.join(results_stage1.save_dir, "weights", "best.pt")
    
    print(f"\n✅ 阶段一完成！最佳权重已保存至: {stage1_weight_path}")
    
    
    # ================= 阶段二：全参微调 =================
    print("\n" + "="*40)
    print("🚀 开始阶段二：加载最佳权重，解冻所有层，低李率微调...")
    print("="*40 + "\n")
    
    # 1. 检查权重文件是否存在
    if not os.path.exists(stage1_weight_path):
        raise FileNotFoundError(f"未找到阶段一的权重文件: {stage1_weight_path}，请检查训练是否报错。")
    
    # 2. 加载阶段一训练好的权重
    # 注意：这里直接加载 .pt，它里面已经包含了你修改过的 P2 结构，不需要再指定 YAML
    model_finetune = YOLO(stage1_weight_path)
    
    # 3. 开始微调训练
    model_finetune.train(
        data=DATA_YAML,
        epochs=STAGE2_EPOCHS, # 训练剩余的轮数
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        freeze=None,          # <--- 核心：设置为 None 表示不冻结任何层
        project=PROJECT_NAME,
        name="stage2_unfreeze",
        lr0=STAGE2_LR,        # <--- 核心：使用更小的学习率
        optimizer='AdamW',    # 对于灰度微小目标，AdamW 通常比 SGD 更稳，推荐加上
        close_mosaic=10,      # 最后 10 轮关闭 Mosaic 增强，有助于精细定位
        warmup_epochs=0 ,      # 既然是微调，不需要太长的热身
        degrees=5.0,
        translate=0.05,
        scale=0.1,
        copy_paste=0.4,
        device=0,
        dropout=0.2,
        weight_decay=0.005
    )
    
    print(f"\n🎉 所有训练阶段完成！最终模型位于{PROJECT_NAME}/stage2_unfreeze/weights/best.pt")


# 开始训练
train(MODEL_YAML, DATA_YAML, IMG_SIZE, BATCH_SIZE, PROJECT_NAME, STAGE1_EPOCHS, STAGE1_LR, STAGE2_EPOCHS, STAGE2_LR)