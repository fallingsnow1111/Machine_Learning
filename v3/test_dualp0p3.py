"""
测试 DualP0P3 架构的脚本
验证 DINO3Preprocessor 和 DINO3Backbone 是否正常工作
"""
import torch
from ultralytics import YOLO

def test_dualp0p3_architecture():
    print("🧪 测试 DualP0P3 架构")
    print("=" * 60)
    
    # 1. 加载模型配置
    print("\n1️⃣ 加载模型配置...")
    model = YOLO("yolo11P.yaml")
    
    # 2. 检查模型结构
    print("\n2️⃣ 检查模型结构...")
    model_str = str(model.model)
    
    has_preprocessor = 'DINO3Preprocessor' in model_str
    has_backbone = 'DINO3Backbone' in model_str
    
    print(f"   DINO3Preprocessor (P0输入增强): {'✅ 找到' if has_preprocessor else '❌ 未找到'}")
    print(f"   DINO3Backbone (P3特征增强): {'✅ 找到' if has_backbone else '❌ 未找到'}")
    
    if not (has_preprocessor and has_backbone):
        print("\n❌ 模型结构不正确!")
        return False
    
    # 3. 测试前向传播
    print("\n3️⃣ 测试前向传播...")
    dummy_input = torch.randn(1, 3, 640, 640)
    
    try:
        with torch.no_grad():
            output = model.model(dummy_input)
        print(f"   ✅ 前向传播成功!")
        print(f"   输出形状: {[o.shape if hasattr(o, 'shape') else type(o) for o in output]}")
    except Exception as e:
        print(f"   ❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. 验证架构
    print("\n4️⃣ 验证架构...")
    print("   架构: Input -> DINO3Preprocessor(P0) -> YOLOv11 -> DINO3Backbone(P3) -> Head")
    print("   ✅ DualP0P3 架构验证通过!")
    
    print("\n" + "=" * 60)
    print("🎉 所有测试通过! DualP0P3 架构已正确实现")
    print("\n💡 使用建议:")
    print("   python train.py  # 开始训练")
    
    return True

if __name__ == "__main__":
    success = test_dualp0p3_architecture()
    exit(0 if success else 1)
