import os
import shutil
import random
from pathlib import Path
import torch
import yaml
from sklearn.model_selection import KFold
from ultralytics import YOLO
import numpy as np


# ==========================================
# 1. 配置参数
# ==========================================
# 原始数据集路径
ORIGINAL_DATASET = r"f:\课设\Machine_Learning\Data\dataset_yolo"
# K折工作目录
KFOLD_DATASET_ROOT = r"f:\课设\Machine_Learning\Data\dataset_kfold"
# 模型配置
MODEL_CONFIG = "./yolo11P.yaml"
PRETRAINED_WEIGHTS = "./yolo11n.pt"
# 设备
DEVICE = '0' if torch.cuda.is_available() else 'cpu'
# K折数
K_FOLDS = 5
# 随机种子
RANDOM_SEED = 42


# ==========================================
# 2. 数据集准备函数
# ==========================================
def prepare_kfold_dataset():
    """
    准备K折交叉验证的数据集结构
    
    ⚠️ 重要说明：
    - K折交叉验证只使用 train + val 数据
    - test数据集完全独立，不参与K折划分  
    - test数据集只在最后用于评估最终模型性能
    
    将原始train/val数据合并，用于K折划分
    """
    print("\n" + "="*50)
    print("📂 准备K折交叉验证数据集...")
    print("="*50)
    print("⚠️  注意：只合并 train + val，test集保持独立用于最终评估！")
    
    # 收集训练、验证和测试图像
    train_images = list(Path(ORIGINAL_DATASET).glob("images/train/*"))
    val_images = list(Path(ORIGINAL_DATASET).glob("images/val/*"))
    test_images = list(Path(ORIGINAL_DATASET).glob("images/test/*"))
    
    # ✅ K折只使用train+val合并的数据
    all_images = train_images + val_images
    
    print(f"✅ Train集: {len(train_images)} 张训练图像")
    print(f"✅ Val集: {len(val_images)} 张验证图像")
    print(f"✅ 用于K折: {len(all_images)} 张图像 (train+val合并)")
    print(f"🔒 Test集: {len(test_images)} 张图像 (保留，不参与K折)")
    
    # 提取图像路径（相对于images目录）
    image_files = []
    for img_path in all_images:
        # 获取文件名（不含扩展名）
        img_stem = img_path.stem
        image_files.append(img_stem)
    
    return image_files


def create_fold_dataset(image_files, train_indices, val_indices, fold_num):
    """
    为特定fold创建数据集
    
    Args:
        image_files: 所有图像文件名列表
        train_indices: 训练集索引
        val_indices: 验证集索引
        fold_num: 当前fold编号
    """
    fold_dir = Path(KFOLD_DATASET_ROOT) / f"fold_{fold_num}"
    
    # 创建目录结构
    for split in ['train', 'val']:
        (fold_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (fold_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # 复制文件
    def copy_files(indices, split):
        for idx in indices:
            img_name = image_files[idx]
            
            # 查找源文件（可能在train或val目录）
            src_img = None
            src_label = None
            
            for source_split in ['train', 'val']:
                # 尝试不同的图像扩展名
                for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG']:
                    img_path = Path(ORIGINAL_DATASET) / 'images' / source_split / f"{img_name}{ext}"
                    if img_path.exists():
                        src_img = img_path
                        break
                if src_img:
                    # 找到对应的标签
                    label_path = Path(ORIGINAL_DATASET) / 'labels' / source_split / f"{img_name}.txt"
                    if label_path.exists():
                        src_label = label_path
                    break
            
            if src_img:
                # 复制图像
                dst_img = fold_dir / 'images' / split / src_img.name
                shutil.copy2(src_img, dst_img)
                
                # 复制标签（如果存在）
                if src_label:
                    dst_label = fold_dir / 'labels' / split / f"{img_name}.txt"
                    shutil.copy2(src_label, dst_label)
    
    print(f"\n📁 创建 Fold {fold_num} 数据集...")
    copy_files(train_indices, 'train')
    print(f"  ✅ 复制 {len(train_indices)} 张训练图像")
    copy_files(val_indices, 'val')
    print(f"  ✅ 复制 {len(val_indices)} 张验证图像")
    
    # 创建dataset.yaml
    dataset_yaml = {
        'path': str(fold_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,  # 根据您的实际类别数修改
        'names': []  # 会从原始dataset.yaml读取
    }
    
    # 读取原始dataset.yaml获取类别名称
    original_yaml_path = Path(ORIGINAL_DATASET) / 'dataset.yaml'
    if original_yaml_path.exists():
        with open(original_yaml_path, 'r', encoding='utf-8') as f:
            original_config = yaml.safe_load(f)
            if 'names' in original_config:
                dataset_yaml['names'] = original_config['names']
            if 'nc' in original_config:
                dataset_yaml['nc'] = original_config['nc']
    
    yaml_path = fold_dir / 'dataset.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(dataset_yaml, f, sort_keys=False)
    
    return str(yaml_path)


# ==========================================
# 3. 训练函数
# ==========================================
def train_single_fold(fold_num, dataset_yaml, results_dir):
    """
    训练单个fold
    
    Args:
        fold_num: fold编号
        dataset_yaml: 数据集配置文件路径
        results_dir: 结果保存目录
    """
    print("\n" + "="*50)
    print(f"🚀 开始训练 Fold {fold_num}")
    print("="*50)
    
    # 初始化模型
    model = YOLO(MODEL_CONFIG)
    
    # 加载预训练权重
    try:
        model.load(PRETRAINED_WEIGHTS)
        print("✅ 成功加载预训练权重！")
    except Exception as e:
        print(f"⚠️ 加载权重跳过或出错: {e}")
    
    # 冻结DINO参数的回调
    def freeze_dino_on_train_start(trainer):
        """训练开始时冻结DINO参数"""
        print("🔧 [Callback] 冻结 DINO 参数...")
        frozen_count = 0
        unfrozen_count = 0
        
        for name, param in trainer.model.named_parameters():
            if ".dino." in name and param.requires_grad:
                param.requires_grad = False
                frozen_count += 1
            elif any(x in name for x in ['input_projection', 'fusion_layer', 
                                         'feature_adapter', 'spatial_projection']):
                if not param.requires_grad:
                    param.requires_grad = True
                unfrozen_count += 1
        
        print(f"✅ 已冻结 {frozen_count} 个 DINO 模型参数")
        print(f"✅ 保持 {unfrozen_count} 个融合层参数可训练")
    
    model.add_callback("on_train_start", freeze_dino_on_train_start)
    
    # 训练
    results = model.train(
        data=dataset_yaml,
        epochs=50,
        imgsz=640,
        batch=32,
        patience=0,
        optimizer='AdamW',
        amp=False,
        cos_lr=True,
        lr0=0.0005,
        lrf=0.01,
        warmup_epochs=5.0,
        translate=0.05,
        scale=0.1,
        copy_paste=0.4,
        device=DEVICE,
        plots=True,
        dropout=0.2,
        project=results_dir,
        name=f'fold_{fold_num}',
        exist_ok=True,
    )
    
    # 验证
    print(f"\n🔍 验证 Fold {fold_num}...")
    best_model_path = Path(results.save_dir) / 'weights' / 'best.pt'
    best_model = YOLO(best_model_path)
    
    metrics = best_model.val(
        data=dataset_yaml,
        split='val',
        imgsz=640,
        batch=16,
        device=DEVICE
    )
    
    # 返回关键指标
    return {
        'fold': fold_num,
        'mAP50': float(metrics.box.map50),
        'mAP50-95': float(metrics.box.map),
        'precision': float(metrics.box.p.mean()),
        'recall': float(metrics.box.r.mean()),
        'best_weights': str(best_model_path)
    }


# ==========================================
# 4. 主函数：K折交叉验证
# ==========================================
def run_kfold_cross_validation():
    """
    执行K折交叉验证
    """
    # 设置随机种子
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    print("\n" + "="*70)
    print(f"🎯 开始 {K_FOLDS} 折交叉验证")
    print("="*70)
    
    # 准备数据集
    image_files = prepare_kfold_dataset()
    
    # K折划分
    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    # 存储每个fold的结果
    all_results = []
    
    # 结果保存目录
    results_base_dir = Path("runs") / "kfold_experiments"
    results_base_dir.mkdir(parents=True, exist_ok=True)
    
    # 遍历每个fold
    for fold_num, (train_idx, val_idx) in enumerate(kfold.split(image_files), 1):
        print(f"\n{'='*70}")
        print(f"📊 处理 Fold {fold_num}/{K_FOLDS}")
        print(f"{'='*70}")
        
        # 创建fold数据集
        dataset_yaml = create_fold_dataset(
            image_files, 
            train_idx, 
            val_idx, 
            fold_num
        )
        
        # 训练
        fold_result = train_single_fold(
            fold_num, 
            dataset_yaml, 
            str(results_base_dir)
        )
        
        all_results.append(fold_result)
        
        # 打印当前fold结果
        print(f"\n{'='*50}")
        print(f"Fold {fold_num} 结果:")
        print(f"  mAP50:     {fold_result['mAP50']:.4f}")
        print(f"  mAP50-95:  {fold_result['mAP50-95']:.4f}")
        print(f"  Precision: {fold_result['precision']:.4f}")
        print(f"  Recall:    {fold_result['recall']:.4f}")
        print(f"{'='*50}")
    
    # ==========================================
    # 5. 汇总结果
    # ==========================================
    print("\n\n" + "="*70)
    print("📈 K折交叉验证汇总结果")
    print("="*70)
    
    # 计算平均值和标准差
    metrics_names = ['mAP50', 'mAP50-95', 'precision', 'recall']
    
    print(f"\n{'Metric':<15} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 60)
    
    summary = {}
    for metric in metrics_names:
        values = [r[metric] for r in all_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        
        summary[metric] = {
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val
        }
        
        print(f"{metric:<15} {mean_val:<10.4f} {std_val:<10.4f} {min_val:<10.4f} {max_val:<10.4f}")
    
    # 逐fold详细结果
    print("\n" + "="*70)
    print("各Fold详细结果 (在各自验证集上):")
    print("="*70)
    for result in all_results:
        print(f"\nFold {result['fold']}:")
        for metric in metrics_names:
            print(f"  {metric:<12}: {result[metric]:.4f}")
    
    # 保存汇总结果到文件
    summary_file = results_base_dir / "kfold_summary.yaml"
    with open(summary_file, 'w', encoding='utf-8') as f:
        yaml.dump({
            'k_folds': K_FOLDS,
            'summary': summary,
            'fold_results': all_results
        }, f, sort_keys=False)
    
    print(f"\n✅ K折交叉验证汇总已保存到: {summary_file}")
    
    # 找出最佳fold
    best_fold = max(all_results, key=lambda x: x['mAP50-95'])
    print(f"\n🏆 最佳 Fold: Fold {best_fold['fold']}")
    print(f"   mAP50-95: {best_fold['mAP50-95']:.4f}")
    print(f"   权重路径: {best_fold['best_weights']}")
    
    # ==========================================
    # 6. 最终测试集评估（重要！）
    # ==========================================
    print("\n\n" + "="*70)
    print("🎯 最终评估：在独立测试集上评估最佳模型")
    print("="*70)
    print("⚠️  这是模型的最终性能，用于论文报告！")
    print("📌  测试集从未参与K折划分和训练过程")
    
    # 使用最佳fold的模型在原始独立测试集上评估
    best_model = YOLO(best_fold['best_weights'])
    test_metrics = best_model.val(
        data=str(Path(ORIGINAL_DATASET) / 'dataset.yaml'),
        split='test',  # 使用完全独立的test集
        imgsz=640,
        batch=16,
        device=DEVICE
    )
    
    # 打印最终测试集结果
    print("\n" + "="*70)
    print("📊 最终测试集结果 (用于论文报告):")
    print("="*70)
    print(f"mAP50:        {test_metrics.box.map50:.4f}")
    print(f"mAP50-95:     {test_metrics.box.map:.4f}")
    print(f"Precision:    {test_metrics.box.p.mean():.4f}")
    print(f"Recall:       {test_metrics.box.r.mean():.4f}")
    print("="*70)
    
    # 保存测试集结果
    test_summary = {
        'test_set_results': {
            'mAP50': float(test_metrics.box.map50),
            'mAP50-95': float(test_metrics.box.map),
            'precision': float(test_metrics.box.p.mean()),
            'recall': float(test_metrics.box.r.mean()),
        },
        'best_model': best_fold['best_weights'],
        'best_fold': best_fold['fold']
    }
    
    test_summary_file = results_base_dir / "final_test_results.yaml"
    with open(test_summary_file, 'w', encoding='utf-8') as f:
        yaml.dump(test_summary, f, sort_keys=False)
    
    print(f"\n✅ 测试集结果已保存到: {test_summary_file}")
    
    print("\n💡 如何理解这些结果:")
    print("  1️⃣  K折交叉验证结果 (上方表格):")
    print("     - 用途: 选择最佳模型、调整超参数、评估模型稳定性")
    print("     - 数据: train+val 合并后K折划分")
    print("  2️⃣  最终测试集结果 (本节):")
    print("     - 用途: 报告模型的真实泛化能力")
    print("     - 数据: 完全独立的测试集，从未参与训练")
    print("     - 👉 在论文中报告这个结果！")
    
    return all_results, summary, test_metrics


# ==========================================
# 7. 主入口
# ==========================================
if __name__ == "__main__":
    all_results, summary, final_test_metrics = run_kfold_cross_validation()
    print("\n✅ 完整K折交叉验证流程完成！")
