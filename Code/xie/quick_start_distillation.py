"""
Quick Start: DINOv3 到 YOLO11 知识蒸馏

这是一个简化的快速开始脚本，用于快速测试蒸馏训练流程。
使用较少的训练轮数和较小的模型以快速验证流程。
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from light_train_v2 import DINOv3ToYOLO11Distillation


def quick_test():
    """
    快速测试配置 - 用于验证流程
    使用较小的模型和较少的训练轮数
    """
    print("\n" + "=" * 80)
    print("快速测试模式")
    print("=" * 80 + "\n")
    
    trainer = DINOv3ToYOLO11Distillation(
        data_dir="Data/Raw/dust",
        output_dir="runs/distillation",
        experiment_name="quick_test"
    )
    
    trainer.run_full_pipeline(
        teacher_model="dinov3/vits16",       # 最小的 DINOv3 模型
        student_model="ultralytics/yolo11n", # 最小的 YOLO11 模型
        distillation_epochs=10,              # 快速测试：10 轮
        finetune_epochs=10,                  # 快速测试：10 轮
        batch_size=8,                        # 小批量
        image_size=64,
    )


def standard_training():
    """
    标准训练配置 - 推荐用于实际训练
    """
    print("\n" + "=" * 80)
    print("标准训练模式")
    print("=" * 80 + "\n")
    
    trainer = DINOv3ToYOLO11Distillation(
        data_dir="Data/Raw/dust",
        output_dir="runs/distillation",
        experiment_name="standard_training"
    )
    
    trainer.run_full_pipeline(
        teacher_model="dinov3/vits16",       # Small teacher
        student_model="ultralytics/yolo11s", # Small student
        distillation_epochs=100,             # 完整蒸馏训练
        finetune_epochs=50,                  # 充分微调
        batch_size=16,
        image_size=64,
    )


def high_performance_training():
    """
    高性能配置 - 追求最佳精度
    需要更多计算资源和时间
    """
    print("\n" + "=" * 80)
    print("高性能训练模式")
    print("=" * 80 + "\n")
    
    trainer = DINOv3ToYOLO11Distillation(
        data_dir="Data/Raw/dust",
        output_dir="runs/distillation",
        experiment_name="high_performance"
    )
    
    trainer.run_full_pipeline(
        teacher_model="dinov3/vitb16",       # Base teacher (更大更强)
        student_model="ultralytics/yolo11m", # Medium student
        distillation_epochs=200,             # 更长的预训练
        finetune_epochs=100,                 # 更长的微调
        batch_size=16,
        image_size=64,
    )


def custom_training():
    """
    自定义训练 - 分步控制训练流程
    """
    print("\n" + "=" * 80)
    print("自定义训练模式")
    print("=" * 80 + "\n")
    
    trainer = DINOv3ToYOLO11Distillation(
        data_dir="Data/Raw/dust",
        output_dir="runs/distillation",
        experiment_name="custom_training"
    )
    
    # 阶段1: 蒸馏预训练
    print("执行阶段1: 蒸馏预训练...")
    pretrained_weights = trainer.stage1_distillation(
        teacher_model="dinov3/vits16",
        student_model="ultralytics/yolo11n",
        epochs=50,
        batch_size=32,
        image_size=64,
    )
    
    # 阶段2: 微调
    print("执行阶段2: 微调...")
    trainer.stage2_finetune(
        pretrained_weights=pretrained_weights,
        epochs=30,
        batch_size=16,
        image_size=64,
    )
    
    # 验证
    print("执行验证...")
    trainer.validate()


def only_distillation():
    """
    仅执行蒸馏预训练
    """
    print("\n" + "=" * 80)
    print("仅蒸馏预训练模式")
    print("=" * 80 + "\n")
    
    trainer = DINOv3ToYOLO11Distillation(
        data_dir="Data/Raw/dust",
        output_dir="runs/distillation",
        experiment_name="only_distillation"
    )
    
    pretrained_weights = trainer.stage1_distillation(
        teacher_model="dinov3/vits16",
        student_model="ultralytics/yolo11n",
        epochs=100,
        batch_size=32,
        image_size=64,
    )
    
    print(f"\n预训练完成！权重保存在: {pretrained_weights}")


def only_finetune(pretrained_weights_path: str):
    """
    仅执行微调（需要已有的预训练权重）
    
    Args:
        pretrained_weights_path: 预训练权重路径
    """
    print("\n" + "=" * 80)
    print("仅微调模式")
    print("=" * 80 + "\n")
    
    trainer = DINOv3ToYOLO11Distillation(
        data_dir="Data/Raw/dust",
        output_dir="runs/distillation",
        experiment_name="only_finetune"
    )
    
    trainer.stage2_finetune(
        pretrained_weights=Path(pretrained_weights_path),
        epochs=50,
        batch_size=16,
        image_size=64,
    )
    
    # 验证
    trainer.validate()


if __name__ == "__main__":
    # 选择一个训练模式
    
    # 1. 快速测试（推荐首次运行）
    quick_test()
    
    # 2. 标准训练（推荐日常使用）
    # standard_training()
    
    # 3. 高性能训练（追求最佳效果）
    # high_performance_training()
    
    # 4. 自定义训练（完全控制）
    # custom_training()
    
    # 5. 仅蒸馏预训练
    # only_distillation()
    
    # 6. 仅微调（需要预训练权重）
    # only_finetune("runs/distillation/xxx/stage1_distillation/exported_models/exported_last.pt")
