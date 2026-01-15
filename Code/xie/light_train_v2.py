"""
DINOv3 to YOLO11 Knowledge Distillation Training Script

This script implements a two-stage training process:
1. Stage 1: Pretrain YOLO11 backbone using DINOv3 teacher (distillation)
2. Stage 2: Fine-tune the pretrained model on labeled detection data

Based on lightly-ai/lightly-train project
"""

from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import lightly_train
from ultralytics import YOLO
import torch
from datetime import datetime


class DINOv3ToYOLO11Distillation:
    """
    DINOv3 蒸馏到 YOLO11 的训练管理器
    """
    
    def __init__(
        self,
        data_dir: str = "Data/Raw/dust",
        output_dir: str = "runs/distillation",
        experiment_name: str = None
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        
        # 生成实验名称
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"dinov3_yolo11_{timestamp}"
        
        self.experiment_name = experiment_name
        self.distillation_output = self.output_dir / experiment_name / "stage1_distillation"
        self.finetune_output = self.output_dir / experiment_name / "stage2_finetune"
        
        # 确保目录存在
        self.distillation_output.mkdir(parents=True, exist_ok=True)
        self.finetune_output.mkdir(parents=True, exist_ok=True)
        
        print("=" * 80)
        print("DINOv3 → YOLO11 Knowledge Distillation Training")
        print("=" * 80)
        print(f"数据目录: {self.data_dir}")
        print(f"输出目录: {self.output_dir / experiment_name}")
        print(f"设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        print("=" * 80 + "\n")
    
    def stage1_distillation(
        self,
        teacher_model: str = "dinov3/vits16",
        student_model: str = "ultralytics/yolo11n",
        epochs: int = 100,
        batch_size: int = 32,
        image_size: int = 640,
    ):
        """
        阶段1: 使用 DINOv3 教师模型蒸馏预训练 YOLO11
        
        Args:
            teacher_model: DINOv3 教师模型名称
            student_model: YOLO11 学生模型名称
            epochs: 训练轮数
            batch_size: 批量大小
            image_size: 图像大小
        """
        print("\n" + "=" * 80)
        print("阶段 1: DINOv3 知识蒸馏预训练")
        print("=" * 80)
        print(f"教师模型: {teacher_model}")
        print(f"学生模型: {student_model}")
        print(f"训练轮数: {epochs}")
        print(f"批量大小: {batch_size}")
        print(f"图像大小: {image_size}")
        print("=" * 80 + "\n")
        
        # 准备无标签数据目录（只需要图像）
        unlabeled_data_dir = self.data_dir / "images" / "train"
        
        if not unlabeled_data_dir.exists():
            raise ValueError(f"数据目录不存在: {unlabeled_data_dir}")
        
        try:
            # 使用 lightly-train 进行蒸馏预训练
            lightly_train.pretrain(
                out=str(self.distillation_output),
                data=str(unlabeled_data_dir),  # 只需要图像目录
                model=student_model,
                method="distillation",  # 使用 distillation 方法
                method_args={
                    "teacher": teacher_model,  # DINOv3 教师模型
                    "temperature": 0.07,       # 温度参数
                },
                epochs=epochs,
                batch_size=batch_size,
                image_size=image_size,
                num_workers="auto",
            )
            
            exported_model = self.distillation_output / "exported_models" / "exported_last.pt"
            
            if exported_model.exists():
                print("\n" + "=" * 80)
                print("✓ 阶段1完成: 蒸馏预训练成功!")
                print(f"✓ 预训练模型已保存: {exported_model}")
                print("=" * 80 + "\n")
                return exported_model
            else:
                raise FileNotFoundError(f"未找到导出的模型: {exported_model}")
                
        except Exception as e:
            print(f"\n✗ 阶段1失败: {str(e)}")
            raise
    
    def stage2_finetune(
        self,
        pretrained_weights: Path,
        epochs: int = 50,
        batch_size: int = 16,
        image_size: int = 640,
        device: str = None,
    ):
        """
        阶段2: 在标注数据上微调预训练模型
        
        Args:
            pretrained_weights: 预训练权重路径
            epochs: 微调轮数
            batch_size: 批量大小
            image_size: 图像大小
            device: 设备 (cuda/cpu)
        """
        print("\n" + "=" * 80)
        print("阶段 2: 目标检测微调")
        print("=" * 80)
        print(f"预训练权重: {pretrained_weights}")
        print(f"微调轮数: {epochs}")
        print(f"批量大小: {batch_size}")
        print(f"图像大小: {image_size}")
        print("=" * 80 + "\n")
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 数据集配置文件
        dataset_yaml = self.data_dir / "dataset.yaml"
        
        if not dataset_yaml.exists():
            raise ValueError(f"数据集配置文件不存在: {dataset_yaml}")
        
        try:
            # 加载预训练模型
            model = YOLO(str(pretrained_weights))
            
            # 在标注数据上微调
            results = model.train(
                data=str(dataset_yaml),
                epochs=epochs,
                imgsz=image_size,
                batch=batch_size,
                device=device,
                project=str(self.finetune_output),
                name="train",
                patience=20,           # 早停耐心值
                save=True,            # 保存检查点
                save_period=10,       # 每10轮保存一次
                plots=True,           # 生成训练图表
                val=True,             # 每轮验证
                cache=True,           # 缓存图像以加快训练
            )
            
            print("\n" + "=" * 80)
            print("✓ 阶段2完成: 微调训练成功!")
            print(f"✓ 最终模型已保存: {self.finetune_output}/train/weights/best.pt")
            print("=" * 80 + "\n")
            
            return results
            
        except Exception as e:
            print(f"\n✗ 阶段2失败: {str(e)}")
            raise
    
    def validate(self, model_path: Path = None):
        """
        在测试集上验证模型
        
        Args:
            model_path: 模型权重路径，默认使用最佳模型
        """
        if model_path is None:
            model_path = self.finetune_output / "train" / "weights" / "best.pt"
        
        if not model_path.exists():
            raise ValueError(f"模型文件不存在: {model_path}")
        
        print("\n" + "=" * 80)
        print("模型验证")
        print("=" * 80)
        print(f"模型路径: {model_path}")
        print("=" * 80 + "\n")
        
        dataset_yaml = self.data_dir / "dataset.yaml"
        model = YOLO(str(model_path))
        
        # 在测试集上评估
        results = model.val(
            data=str(dataset_yaml),
            split='test',
        )
        
        print("\n" + "=" * 80)
        print("验证结果:")
        print(f"mAP50: {results.box.map50:.4f}")
        print(f"mAP50-95: {results.box.map:.4f}")
        print("=" * 80 + "\n")
        
        return results
    
    def run_full_pipeline(
        self,
        teacher_model: str = "dinov3/vits16",
        student_model: str = "ultralytics/yolo11n",
        distillation_epochs: int = 100,
        finetune_epochs: int = 50,
        batch_size: int = 16,
        image_size: int = 640,
    ):
        """
        运行完整的训练流程
        
        Args:
            teacher_model: DINOv3 教师模型
            student_model: YOLO11 学生模型
            distillation_epochs: 蒸馏预训练轮数
            finetune_epochs: 微调轮数
            batch_size: 批量大小
            image_size: 图像大小
        """
        print("\n" + "=" * 80)
        print("开始完整训练流程")
        print("=" * 80 + "\n")
        
        # 阶段1: 蒸馏预训练
        pretrained_weights = self.stage1_distillation(
            teacher_model=teacher_model,
            student_model=student_model,
            epochs=distillation_epochs,
            batch_size=batch_size,
            image_size=image_size,
        )
        
        # 阶段2: 微调
        self.stage2_finetune(
            pretrained_weights=pretrained_weights,
            epochs=finetune_epochs,
            batch_size=batch_size,
            image_size=image_size,
        )
        
        # 验证
        self.validate()
        
        print("\n" + "=" * 80)
        print("✓ 完整训练流程完成!")
        print(f"✓ 结果保存在: {self.output_dir / self.experiment_name}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    # 创建训练器实例
    trainer = DINOv3ToYOLO11Distillation(
        data_dir="Data/Raw/dust",
        output_dir="runs/distillation",
    )
    
    # 运行完整训练流程
    trainer.run_full_pipeline(
        teacher_model="dinov3/vits16",      # DINOv3 small 教师模型 (可选: vitb16, vitl16)
        student_model="ultralytics/yolo11n", # YOLO11 nano 学生模型
        distillation_epochs=100,             # 蒸馏预训练轮数
        finetune_epochs=50,                  # 微调轮数
        batch_size=16,                       # 批量大小
        image_size=64,                      # 图像大小
    )