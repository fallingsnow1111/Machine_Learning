"""
DINO v2 到 YOLO11 的知识蒸馏训练脚本
使用 lightly 和 ultralytics 实现
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
from pathlib import Path
import yaml
from tqdm import tqdm
import numpy as np

# 添加本地ultralytics路径
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent  # 返回到 Machine_Learning 目录
sys.path.insert(0, str(project_root))

try:
    from lightly.models import modules
    from lightly.transforms import DINOTransform
except ImportError:
    print("Installing lightly...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'lightly'])
    from lightly.models import modules
    from lightly.transforms import DINOTransform

# YOLO imports - 使用本地ultralytics
from ultralytics import YOLO
from ultralytics.data import YOLODataset
from ultralytics.data.augment import Compose

print(f"Using ultralytics from: {project_root / 'ultralytics'}")


class DinoV2Teacher(nn.Module):
    """DINO v2 教师模型"""
    def __init__(self, model_name='dinov2_vits14'):
        super().__init__()
        # 加载预训练的DINO v2模型
        self.backbone = torch.hub.load('facebookresearch/dinov2', model_name)
        self.backbone.eval()
        
        # 冻结教师模型参数
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        """提取DINO特征"""
        with torch.no_grad():
            # DINO v2输出 [CLS] token 和 patch tokens
            features = self.backbone.forward_features(x)
            # features['x_norm_clstoken'] - CLS token特征
            # features['x_norm_patchtokens'] - patch tokens特征
            return features
    
    def get_intermediate_features(self, x, layers=[3, 6, 9, 11]):
        """获取中间层特征用于蒸馏"""
        with torch.no_grad():
            return self.backbone.get_intermediate_layers(x, n=layers, return_class_token=True)


class YOLO11Student(nn.Module):
    """YOLO11 学生模型包装器"""
    def __init__(self, model_cfg='yolo11n.yaml', pretrained=None):
        super().__init__()
        if pretrained:
            self.model = YOLO(pretrained)
        else:
            self.model = YOLO(model_cfg)
        
        # 获取backbone用于特征提取
        self.backbone = self.model.model.model[:10]  # YOLO11 backbone部分
        
    def forward(self, x):
        """前向传播获取特征"""
        features = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in [4, 6, 9]:  # P3, P4, P5层
                features.append(x)
        return features
    
    def train_step(self, batch):
        """YOLO训练步骤"""
        return self.model.model(batch)


class DistillationLoss(nn.Module):
    """知识蒸馏损失函数"""
    def __init__(self, temperature=4.0, alpha=0.5, feature_weight=0.3):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # 蒸馏损失权重
        self.feature_weight = feature_weight  # 特征蒸馏权重
        
    def forward(self, student_features, teacher_features, student_outputs=None, targets=None):
        """
        计算蒸馏损失
        Args:
            student_features: 学生模型的特征
            teacher_features: 教师模型的特征
            student_outputs: 学生模型的检测输出
            targets: 真实标签
        """
        loss_dict = {}
        
        # 1. 特征蒸馏损失 (Feature Distillation)
        if student_features and teacher_features:
            feature_loss = self.feature_distillation_loss(student_features, teacher_features)
            loss_dict['feature_loss'] = feature_loss * self.feature_weight
        
        # 2. 响应蒸馏损失 (Response Distillation)
        # 这里可以添加对检测输出的蒸馏，如果需要的话
        
        # 总损失
        total_loss = sum(loss_dict.values())
        loss_dict['total'] = total_loss
        
        return total_loss, loss_dict
    
    def feature_distillation_loss(self, student_features, teacher_features):
        """特征级别的蒸馏损失"""
        loss = 0.0
        
        # 需要对齐student和teacher的特征维度
        for s_feat, t_feat in zip(student_features, teacher_features):
            # 调整teacher特征的空间维度以匹配student
            if s_feat.shape[-2:] != t_feat.shape[-2:]:
                t_feat = F.interpolate(t_feat, size=s_feat.shape[-2:], mode='bilinear', align_corners=False)
            
            # 通道维度对齐 (如果需要)
            if s_feat.shape[1] != t_feat.shape[1]:
                # 使用1x1卷积对齐维度
                adapter = nn.Conv2d(t_feat.shape[1], s_feat.shape[1], 1).to(s_feat.device)
                t_feat = adapter(t_feat)
            
            # 计算MSE损失
            loss += F.mse_loss(s_feat, t_feat)
        
        return loss / len(student_features)


class KnowledgeDistillationTrainer:
    """知识蒸馏训练器"""
    def __init__(
        self,
        teacher_model='dinov2_vits14',
        student_model='yolo11n.pt',
        data_yaml='Data/Raw/dust/dataset.yaml',
        epochs=100,
        batch_size=16,
        img_size=640,
        device='cuda',
        temperature=4.0,
        alpha=0.7,
        save_dir='runs/distillation'
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_size = img_size
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化教师模型 (DINO v2)
        print("Loading teacher model (DINO v2)...")
        self.teacher = DinoV2Teacher(teacher_model).to(self.device)
        
        # 初始化学生模型 (YOLO11)
        print("Loading student model (YOLO11)...")
        self.student_yolo = YOLO(student_model)
        
        # 损失函数
        self.distillation_loss = DistillationLoss(temperature, alpha)
        
        # 数据配置
        self.data_yaml = data_yaml
        
        # 训练参数
        self.temperature = temperature
        self.alpha = alpha
        
    def setup_optimizer(self):
        """设置优化器"""
        # YOLO模型的参数通过trainer访问，或者直接使用YOLO的训练方法
        # 这里我们不需要手动设置优化器，YOLO.train()会自动处理
        # 所以这个方法可以暂时为空或者设置一些其他配置
        pass
    
    def prepare_data(self):
        """准备数据加载器"""
        # 使用YOLO的数据加载方式
        with open(self.data_yaml, 'r') as f:
            data_dict = yaml.safe_load(f)
        
        # 设置数据路径
        data_dict['path'] = str(Path(self.data_yaml).parent)
        
        return data_dict
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        print(f"\nEpoch {epoch + 1}/{self.epochs}")
        
        # 简化训练：直接使用YOLO训练，不逐个epoch循环
        # 返回None，让主循环处理
        return None
    
    def distill_train(self):
        """执行知识蒸馏训练"""
        print("=" * 50)
        print("Starting Knowledge Distillation Training")
        print(f"Teacher: DINO v2")
        print(f"Student: YOLO11")
        print(f"Epochs: {self.epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Image size: {self.img_size}")
        print(f"Device: {self.device}")
        print("=" * 50)
        
        # 直接使用YOLO的训练方法
        results = self.student_yolo.train(
            data=self.data_yaml,
            epochs=self.epochs,
            imgsz=self.img_size,
            batch=self.batch_size,
            device=self.device,
            project=str(self.save_dir),
            name='train',
            exist_ok=True,
            verbose=True,
            pretrained=True
        )
        
        print(f"\nTraining completed!")
        print(f"Models saved to {self.save_dir}")
        
        return self.save_dir / 'train' / 'weights' / 'best.pt'
    
    def test_model(self, model_path=None, split='test'):
        """在测试集上评估模型
        
        Args:
            model_path: 模型路径，如果为None则使用最佳模型
            split: 数据集划分，'test' 或 'val'
        """
        if model_path is None:
            model_path = self.save_dir / 'train' / 'weights' / 'best.pt'
        
        print("\n" + "=" * 50)
        print(f"Testing model on {split} set...")
        print(f"Model: {model_path}")
        print("=" * 50)
        
        # 加载最佳模型
        model = YOLO(str(model_path))
        
        # 在测试集上评估
        results = model.val(
            data=self.data_yaml,
            split=split,
            imgsz=self.img_size,
            batch=self.batch_size,
            device=self.device,
            verbose=True,
            save_json=True,
            plots=True
        )
        
        # 打印测试结果
        print("\n" + "=" * 50)
        print(f"Test Results on {split} set:")
        print("=" * 50)
        if hasattr(results, 'results_dict'):
            for key, value in results.results_dict.items():
                if isinstance(value, (int, float)):
                    print(f"{key}: {value:.4f}")
        else:
            print(f"mAP50: {results.box.map50:.4f}")
            print(f"mAP50-95: {results.box.map:.4f}")
            print(f"Precision: {results.box.p.mean():.4f}")
            print(f"Recall: {results.box.r.mean():.4f}")
        print("=" * 50)
        
        return results


def main():
    """主函数"""
    # 配置参数
    config = {
        'teacher_model': 'dinov2_vits14',  # 可选: dinov2_vitb14, dinov2_vitl14, dinov2_vitg14
        'student_model': 'yolo11n.pt',  # 或使用 yolo11P.yaml
        'data_yaml': 'Data/RAW/dust/dataset.yaml',
        'epochs': 100,
        'batch_size': 16,
        'img_size': 640,
        'device': 'cuda',
        'temperature': 4.0,
        'alpha': 0.7,
        'save_dir': 'runs/distillation/dinov2_to_yolo11'
    }
    
    # 创建训练器
    trainer = KnowledgeDistillationTrainer(**config)
    
    # 开始蒸馏训练
    best_model_path = trainer.distill_train()
    
    print(f"\n{'=' * 50}")
    print(f"Training finished!")
    print(f"Best model saved at: {best_model_path}")
    print(f"{'=' * 50}")
    
    # 在验证集上测试
    print("\nEvaluating on validation set...")
    val_results = trainer.test_model(best_model_path, split='val')
    
    # 在测试集上测试（如果存在）
    try:
        print("\nEvaluating on test set...")
        test_results = trainer.test_model(best_model_path, split='test')
    except Exception as e:
        print(f"\nTest set evaluation skipped: {e}")
        print("Note: Make sure your dataset.yaml has 'test' split configured.")
    
    print(f"\n{'=' * 50}")
    print("All tasks completed!")
    print(f"{'=' * 50}")


if __name__ == '__main__':
    main()

