"""
测试 DINO-YOLO 模型构建
用于验证模型结构是否正确，不进行训练
"""

import torch
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
from custom_modules import ASPP, EMA
from custom_modules.dino import DINOInputAdapter, DINOMidAdapter


def register_custom_layers():
    """注册所有自定义模块"""
    setattr(tasks, "ASPP", ASPP)
    setattr(tasks, "EMA", EMA)
    setattr(tasks, "DINOInputAdapter", DINOInputAdapter)
    setattr(tasks, "DINOMidAdapter", DINOMidAdapter)
    print("✅ 模块注册完成\n")


def test_model_build():
    """测试模型构建"""
    print("="*60)
    print("🔍 测试 DINO-YOLO 模型构建")
    print("="*60)
    
    # 注册模块
    register_custom_layers()
    
    # 尝试构建模型
    try:
        print("\n📦 构建模型...")
        model = YOLO("dino_yolo.yaml")
        print("✅ 模型构建成功！\n")
        
        # 打印模型信息
        print("="*60)
        print("📊 模型信息")
        print("="*60)
        model.info(detailed=False)
        
        # 测试前向传播
        print("\n" + "="*60)
        print("🧪 测试前向传播")
        print("="*60)
        
        # 创建随机输入（灰度图）
        x = torch.randn(1, 1, 640, 640).cuda()  # 1 通道灰度图
        print(f"输入形状: {x.shape}")
        
        model.model.cuda()
        with torch.no_grad():
            output = model.model(x)
        
        print(f"✅ 前向传播成功！")
        print(f"输出数量: {len(output) if isinstance(output, (list, tuple)) else 1}")
        
        # 统计参数量
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        
        print("\n" + "="*60)
        print("📈 模型统计")
        print("="*60)
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        print(f"冻结参数: {total_params - trainable_params:,}")
        print("="*60)
        
        print("\n✅ 所有测试通过！模型可以正常使用。")
        
    except Exception as e:
        print(f"\n❌ 模型构建失败：{e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = test_model_build()
    if not success:
        print("\n💡 提示：请检查以上错误信息，修复后重试")
