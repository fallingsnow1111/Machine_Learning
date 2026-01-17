"""
测试DINO模型的输出格式
"""

import torch
from modelscope import AutoModel

print("=" * 80)
print("测试DINO模型输出格式")
print("=" * 80)

# 加载DINO模型
model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
print(f"\n加载模型: {model_name}")

try:
    dino = AutoModel.from_pretrained(model_name)
    print("✅ 模型加载成功")
    
    # 创建测试输入
    x = torch.randn(1, 3, 224, 224)
    print(f"\n测试输入: {x.shape}")
    
    # 测试1: 不传output_hidden_states
    print("\n【测试1】默认调用（不传output_hidden_states）")
    print("-" * 80)
    with torch.no_grad():
        outputs1 = dino(pixel_values=x)
    
    print(f"输出类型: {type(outputs1)}")
    print(f"输出属性: {[attr for attr in dir(outputs1) if not attr.startswith('_')]}")
    
    if hasattr(outputs1, 'keys'):
        print(f"输出keys: {outputs1.keys()}")
    
    # 测试2: 传output_hidden_states=True
    print("\n【测试2】传output_hidden_states=True")
    print("-" * 80)
    with torch.no_grad():
        outputs2 = dino(pixel_values=x, output_hidden_states=True)
    
    print(f"输出类型: {type(outputs2)}")
    print(f"输出属性: {[attr for attr in dir(outputs2) if not attr.startswith('_')]}")
    
    if hasattr(outputs2, 'keys'):
        print(f"输出keys: {outputs2.keys()}")
    
    # 检查hidden_states
    if hasattr(outputs2, 'hidden_states'):
        if outputs2.hidden_states is not None:
            print(f"\n✅ hidden_states存在!")
            print(f"  类型: {type(outputs2.hidden_states)}")
            if isinstance(outputs2.hidden_states, (list, tuple)):
                print(f"  长度: {len(outputs2.hidden_states)}")
                print(f"  第一层shape: {outputs2.hidden_states[0].shape}")
                print(f"  最后一层shape: {outputs2.hidden_states[-1].shape}")
        else:
            print(f"\n❌ hidden_states是None!")
    else:
        print(f"\n❌ 没有hidden_states属性!")
    
    # 检查last_hidden_state
    if hasattr(outputs2, 'last_hidden_state'):
        print(f"\n✅ last_hidden_state存在!")
        print(f"  shape: {outputs2.last_hidden_state.shape}")
    else:
        print(f"\n❌ 没有last_hidden_state属性!")
    
    # 测试3: 检查配置
    print("\n【测试3】DINO模型配置")
    print("-" * 80)
    if hasattr(dino, 'config'):
        config = dino.config
        print(f"配置类型: {type(config)}")
        print(f"hidden_size: {getattr(config, 'hidden_size', 'N/A')}")
        print(f"num_hidden_layers: {getattr(config, 'num_hidden_layers', 'N/A')}")
        print(f"patch_size: {getattr(config, 'patch_size', 'N/A')}")
        print(f"output_hidden_states: {getattr(config, 'output_hidden_states', 'N/A')}")
    
    # 测试4: 尝试强制设置output_hidden_states
    print("\n【测试4】修改配置并测试")
    print("-" * 80)
    if hasattr(dino, 'config'):
        dino.config.output_hidden_states = True
        print("已设置 config.output_hidden_states = True")
        
        with torch.no_grad():
            outputs3 = dino(pixel_values=x)
        
        if hasattr(outputs3, 'hidden_states') and outputs3.hidden_states is not None:
            print(f"✅ 现在hidden_states可用!")
            print(f"  长度: {len(outputs3.hidden_states)}")
        else:
            print(f"❌ hidden_states仍然不可用")
    
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("测试完成")
print("=" * 80)
