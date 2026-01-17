"""
关键问题诊断：DINO模块冻结导致的BatchNorm问题

问题假设：
1. DINO模块被冻结（requires_grad=False）
2. 但是input_projection和fusion_layer中的BatchNorm没有被正确设置为eval模式
3. 导致验证时BatchNorm的running_mean/running_var未正确初始化或更新
4. 最终导致输出异常

解决方案：
1. 在冻结DINO时，同时将相关的BatchNorm设置为eval模式
2. 或者在训练开始前先warmup初始化BatchNorm统计量
"""

import torch
from ultralytics import YOLO

print("=" * 80)
print("诊断：DINO冻结与BatchNorm问题")
print("=" * 80)

# 加载模型配置
model = YOLO("./yolo11P.yaml")

print("\n1. 检查DINO相关模块的结构")
print("-" * 80)

dino_modules = {}
for name, module in model.model.named_modules():
    if 'DINO' in str(type(module)) or 'dino' in name.lower():
        dino_modules[name] = module
        print(f"\n模块: {name}")
        print(f"  类型: {type(module).__name__}")
        
        # 检查子模块
        for sub_name, sub_module in module.named_children():
            print(f"  └─ {sub_name}: {type(sub_module).__name__}")
            
            # 如果是Sequential，展开看
            if isinstance(sub_module, torch.nn.Sequential):
                for i, layer in enumerate(sub_module):
                    layer_type = type(layer).__name__
                    print(f"      [{i}] {layer_type}", end="")
                    if isinstance(layer, torch.nn.BatchNorm2d):
                        print(" ⚠️ BatchNorm层!")
                    else:
                        print()

print("\n" + "=" * 80)
print("2. 模拟训练时的冻结操作")
print("-" * 80)

print("\n冻结前:")
for name, param in model.model.named_parameters():
    if 'dino' in name.lower() or 'input_projection' in name or 'fusion_layer' in name:
        print(f"  {name}: requires_grad={param.requires_grad}")
        if len(list(param.shape)) > 0:
            break  # 只显示几个

# 模拟冻结
print("\n执行冻结...")
frozen_count = 0
for name, param in model.model.named_parameters():
    if "dino" in name:
        param.requires_grad = False
        frozen_count += 1

print(f"✅ 冻结了 {frozen_count} 个参数")

print("\n冻结后:")
for name, param in model.model.named_parameters():
    if 'dino' in name.lower() or 'input_projection' in name or 'fusion_layer' in name:
        print(f"  {name}: requires_grad={param.requires_grad}")
        if len(list(param.shape)) > 0:
            break

print("\n" + "=" * 80)
print("3. 检查BatchNorm的训练状态")
print("-" * 80)

print("\n设置为train()模式后的BatchNorm状态:")
model.model.train()

for name, module in model.model.named_modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        if 'input_projection' in name or 'fusion_layer' in name or 'dino' in name.lower():
            print(f"  {name}:")
            print(f"    training: {module.training}")
            print(f"    track_running_stats: {module.track_running_stats}")

print("\n设置为eval()模式后的BatchNorm状态:")
model.model.eval()

for name, module in model.model.named_modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        if 'input_projection' in name or 'fusion_layer' in name or 'dino' in name.lower():
            print(f"  {name}:")
            print(f"    training: {module.training}")

print("\n" + "=" * 80)
print("4. 测试：冻结后BatchNorm统计量的更新")
print("-" * 80)

# 重新初始化模型
model = YOLO("./yolo11P.yaml")

# 冻结DINO
for name, param in model.model.named_parameters():
    if "dino" in name:
        param.requires_grad = False

# 设置为训练模式
model.model.train()

# 记录初始的BatchNorm统计
initial_stats = {}
for name, module in model.model.named_modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        if 'input_projection' in name or 'fusion_layer' in name:
            if module.running_mean is not None:
                initial_stats[name] = {
                    'mean': module.running_mean.clone(),
                    'var': module.running_var.clone(),
                    'tracked': module.num_batches_tracked.item()
                }

print(f"找到 {len(initial_stats)} 个需要监控的BatchNorm层")

# 运行几次forward
print("\n运行5次forward...")
x = torch.randn(4, 3, 640, 640)
for i in range(5):
    with torch.no_grad():
        _ = model.model(x)
    print(f"  Forward {i+1}/5 完成")

# 检查统计量是否更新
print("\nBatchNorm统计量变化:")
for name, module in model.model.named_modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        if name in initial_stats:
            if module.running_mean is not None:
                mean_changed = not torch.equal(module.running_mean, initial_stats[name]['mean'])
                var_changed = not torch.equal(module.running_var, initial_stats[name]['var'])
                tracked_changed = module.num_batches_tracked.item() != initial_stats[name]['tracked']
                
                print(f"\n  {name}:")
                print(f"    mean变化: {mean_changed}")
                print(f"    var变化: {var_changed}")
                print(f"    tracked: {initial_stats[name]['tracked']} -> {module.num_batches_tracked.item()}")
                
                if not tracked_changed:
                    print(f"    ⚠️ BatchNorm统计量未更新!")

print("\n" + "=" * 80)
print("诊断完成")
print("=" * 80)

print("\n关键发现:")
print("1. 如果BatchNorm在training=True时统计量未更新")
print("   -> 说明这些层没有参与前向传播，或者被错误地冻结")
print("2. 如果input_projection/fusion_layer的BatchNorm被冻结")
print("   -> 需要修改冻结逻辑，只冻结DINO模型本身")
print("3. 如果统计量为0或未初始化")
print("   -> 需要在训练前进行warmup")
