from ultralytics import YOLO
import os

def main():
    # === Step 1: Pre-train on NEU Dataset ===
    # 这是一个外部开源数据集，用来教会模型什么是"工业缺陷"
    print("\n" + "="*50)
    print("🚀 Step 1: 在 NEU-DET 大型数据集上预训练...")
    print("="*50)
    
    # 第一次使用通用权重 (yolo11n.pt)
    model = YOLO("pt/yolo11n.pt") 
    
    # 开始训练 NEU
    results_neu = model.train(
        data="Data/dataset_neu.yaml",
        epochs=50,       # 50轮足够提取通用特征
        imgsz=640,
        batch=16,
        project="runs/detect", # 统一保存路径
        name="neu_pretrain",   # 实验名称
        exist_ok=True,    # 覆盖已有结果
        device=0          # 使用第一个GPU
    )
    
    print(f"Step 1 完成! 最佳模型保存在: {results_neu.save_dir}")

    # === Step 2: Fine-tune on Your Data ===
    # 加载刚才训练好的 NEU 最佳模型，迁移到你的小样本任务上
    print("\n" + "="*50)
    print("🚀 Step 2: 在目标数据集 (Dust) 上微调...")
    print("="*50)
    
    best_neu_model_path = os.path.join(results_neu.save_dir, "weights", "best.pt")
    
    # 加载训练好的权重
    # 注意：YOLO检测到类别数量不一致(NEU是6类，你的是1类)时，
    # 会自动重置最后的输出层(Head)，这正是我们想要的
    model_finetune = YOLO(best_neu_model_path)
    
    # 微调训练
    model_finetune.train(
        data="Data/dataset.yaml", # 指向你的 dataset.yaml
        epochs=100,               # 在你的数据上多跑一些轮次
        imgsz=640,
        batch=16,
        project="runs/detect",
        name="dust_finetune_from_neu",
        lr0=0.005,                # 初始学习率稍微调低一点点(默认是0.01)，保护特征不被破坏太快
        device=0
    )

    print("\n✅ 所有训练完成！")

if __name__ == "__main__":
    main()
