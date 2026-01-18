"""
K折交叉验证 - 快速测试版本
用于验证流程是否正确，使用较少的epochs
"""
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
ORIGINAL_DATASET = r".\Data\dataset_merged_no_noise"
KFOLD_DATASET_ROOT = r".\Data\dataset_kfold_quick"
MODEL_CONFIG = "./yolo11P.yaml"
PRETRAINED_WEIGHTS = "./yolo11n.pt"
DEVICE = '0' if torch.cuda.is_available() else 'cpu'
K_FOLDS = 3  # 快速测试使用3折
RANDOM_SEED = 42
EPOCHS = 10  # 快速测试使用10个epochs


def prepare_kfold_dataset():
    """
    准备K折交叉验证的数据集
    
    ⚠️ 重要说明：
    - K折交叉验证只使用 train + val 数据
    - test数据集完全独立，不参与K折划分
    - test数据集只在最后用于评估最终模型性能
    """
    print("\n📂 准备K折交叉验证数据集...")
    print("⚠️  注意：只合并 train + val，test集保持独立！")
    
    train_images = list(Path(ORIGINAL_DATASET).glob("images/train/*"))
    val_images = list(Path(ORIGINAL_DATASET).glob("images/val/*"))
    test_images = list(Path(ORIGINAL_DATASET).glob("images/test/*"))
    
    # ✅ K折只使用train+val
    all_images = train_images + val_images
    
    print(f"✅ Train集: {len(train_images)} 张")
    print(f"✅ Val集: {len(val_images)} 张")
    print(f"✅ 用于K折: {len(all_images)} 张 (train+val合并)")
    print(f"🔒 Test集: {len(test_images)} 张 (保留，不参与K折)")
    
    image_files = [img_path.stem for img_path in all_images]
    return image_files


def create_fold_dataset(image_files, train_indices, val_indices, fold_num):
    """为特定fold创建数据集"""
    fold_dir = Path(KFOLD_DATASET_ROOT) / f"fold_{fold_num}"
    
    # 创建目录
    for split in ['train', 'val']:
        (fold_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (fold_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # 复制文件
    def copy_files(indices, split):
        for idx in indices:
            img_name = image_files[idx]
            src_img = None
            src_label = None
            
            for source_split in ['train', 'val']:
                for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG']:
                    img_path = Path(ORIGINAL_DATASET) / 'images' / source_split / f"{img_name}{ext}"
                    if img_path.exists():
                        src_img = img_path
                        break
                if src_img:
                    label_path = Path(ORIGINAL_DATASET) / 'labels' / source_split / f"{img_name}.txt"
                    if label_path.exists():
                        src_label = label_path
                    break
            
            if src_img:
                dst_img = fold_dir / 'images' / split / src_img.name
                shutil.copy2(src_img, dst_img)
                if src_label:
                    dst_label = fold_dir / 'labels' / split / f"{img_name}.txt"
                    shutil.copy2(src_label, dst_label)
    
    copy_files(train_indices, 'train')
    copy_files(val_indices, 'val')
    
    # 创建dataset.yaml
    dataset_yaml = {'path': str(fold_dir.absolute()), 'train': 'images/train', 'val': 'images/val'}
    
    original_yaml_path = Path(ORIGINAL_DATASET) / 'dataset.yaml'
    if original_yaml_path.exists():
        with open(original_yaml_path, 'r', encoding='utf-8') as f:
            original_config = yaml.safe_load(f)
            dataset_yaml.update({k: original_config[k] for k in ['names', 'nc'] if k in original_config})
    
    yaml_path = fold_dir / 'dataset.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(dataset_yaml, f, sort_keys=False)
    
    return str(yaml_path)


def train_single_fold(fold_num, dataset_yaml, results_dir):
    """训练单个fold"""
    print(f"\n🚀 开始训练 Fold {fold_num}")
    
    model = YOLO(MODEL_CONFIG)
    
    try:
        model.load(PRETRAINED_WEIGHTS)
        print("✅ 加载预训练权重")
    except Exception as e:
        print(f"⚠️ 权重加载跳过: {e}")
    
    def freeze_dino_on_train_start(trainer):
        frozen_count = 0
        for name, param in trainer.model.named_parameters():
            if ".dino." in name and param.requires_grad:
                param.requires_grad = False
                frozen_count += 1
        print(f"✅ 已冻结 {frozen_count} 个 DINO 参数")
    
    model.add_callback("on_train_start", freeze_dino_on_train_start)
    
    # 训练（使用较少的epochs）
    results = model.train(
        data=dataset_yaml,
        epochs=EPOCHS,  # 快速测试
        imgsz=640,
        batch=16,  # 较小batch
        patience=0,
        optimizer='AdamW',
        amp=False,
        cos_lr=True,
        lr0=0.0005,
        lrf=0.01,
        warmup_epochs=2.0,  # 减少warmup
        device=DEVICE,
        plots=False,  # 不生成图表以节省时间
        project=results_dir,
        name=f'fold_{fold_num}',
        exist_ok=True,
    )
    
    # 验证（在当前fold的验证集上）
    best_model_path = Path(results.save_dir) / 'weights' / 'best.pt'
    best_model = YOLO(best_model_path)
    metrics = best_model.val(
        data=dataset_yaml,
        split='val',
        imgsz=640,
        batch=16,
        device=DEVICE,
        project=results_dir,
        name=f'fold_{fold_num}_val',
        exist_ok=True
    )
    
    return {
        'fold': fold_num,
        'mAP50': float(metrics.box.map50),
        'mAP50-95': float(metrics.box.map),
        'precision': float(metrics.box.p.mean()),
        'recall': float(metrics.box.r.mean()),
        'best_weights': str(best_model_path),  # 保存权重路径
    }


def run_kfold_cross_validation():
    """执行K折交叉验证"""
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    print(f"\n🎯 开始 {K_FOLDS} 折交叉验证（快速测试版 - {EPOCHS} epochs）")
    
    image_files = prepare_kfold_dataset()
    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    all_results = []
    results_base_dir = Path("runs") / "kfold_quick"
    results_base_dir.mkdir(parents=True, exist_ok=True)
    
    for fold_num, (train_idx, val_idx) in enumerate(kfold.split(image_files), 1):
        print(f"\n📊 Fold {fold_num}/{K_FOLDS}")
        dataset_yaml = create_fold_dataset(image_files, train_idx, val_idx, fold_num)
        fold_result = train_single_fold(fold_num, dataset_yaml, str(results_base_dir))
        all_results.append(fold_result)
        
        print(f"  mAP50: {fold_result['mAP50']:.4f} | mAP50-95: {fold_result['mAP50-95']:.4f}")
    
    # ==========================================
    # K折交叉验证汇总
    # ==========================================
    print("\n" + "="*70)
    print("📈 K折交叉验证汇总结果 (在各fold的验证集上)")
    print("="*70)
    for metric in ['mAP50', 'mAP50-95', 'precision', 'recall']:
        values = [r[metric] for r in all_results]
        print(f"{metric:<12}: {np.mean(values):.4f} ± {np.std(values):.4f}")
    
    # 选择最佳fold
    best_fold = max(all_results, key=lambda x: x['mAP50-95'])
    print(f"\n🏆 最佳 Fold: Fold {best_fold['fold']} (mAP50-95: {best_fold['mAP50-95']:.4f})")
    
    # ==========================================
    # 最终测试集评估（重要！）
    # ==========================================
    print("\n" + "="*70)
    print("🎯 最终评估：在独立测试集上评估最佳模型")
    print("="*70)
    print("⚠️  这是模型的最终性能，用于论文报告！")
    
    # 使用最佳fold的模型在原始测试集上评估
    best_model = YOLO(best_fold['best_weights'])
    test_metrics = best_model.val(
        data=str(Path(ORIGINAL_DATASET) / 'dataset.yaml'),
        split='test',  # 使用独立的test集
        imgsz=640,
        batch=16,
        device=DEVICE,
        project=str(results_base_dir),
        name='final_test',
        exist_ok=True
    )
    
    print("\n" + "="*70)
    print("📊 最终测试集结果 (用于论文报告):")
    print("="*70)
    print(f"mAP50:        {test_metrics.box.map50:.4f}")
    print(f"mAP50-95:     {test_metrics.box.map:.4f}")
    print(f"Precision:    {test_metrics.box.p.mean():.4f}")
    print(f"Recall:       {test_metrics.box.r.mean():.4f}")
    print("="*70)
    
    print("\n💡 理解这些结果:")
    print("  - K折交叉验证结果: 用于选择最佳模型/超参数")
    print("  - 测试集结果: 模型的真实泛化能力，报告这个！")
    
    return all_results, test_metrics


if __name__ == "__main__":
    kfold_results, final_test_metrics = run_kfold_cross_validation()
    
    print("\n✅ K折交叉验证完成！")
    print(f"📁 结果保存在: runs/kfold_quick/")
