"""
关键测试：对比训练中的验证 vs 独立测试的区别
重点检查：模型状态、输出格式、NMS参数
"""

import torch
from pathlib import Path

print("=" * 80)
print("关键测试：找出验证mAP=0的根本原因")
print("=" * 80)

# 1. 检查最近训练的模型
train_dir = Path("./runs/detect/train")
if train_dir.exists():
    last_pt = train_dir / "weights" / "last.pt"
    if last_pt.exists():
        print(f"\n找到训练模型: {last_pt}")
        
        # 加载模型
        from ultralytics import YOLO
        model = YOLO(str(last_pt))
        
        print("\n" + "=" * 80)
        print("测试：检查模型在不同模式下的输出差异")
        print("=" * 80)
        
        # 创建测试数据
        test_img = torch.randn(1, 3, 640, 640)
        
        # ===== 关键测试1: 检查Detect head的anchors和strides =====
        print("\n【测试1】检查Detect head的anchors和strides初始化")
        detect = model.model.model[-1]
        
        print(f"初始状态:")
        print(f"  anchors.shape: {detect.anchors.shape}")
        print(f"  strides.shape: {detect.strides.shape}")
        print(f"  dynamic: {detect.dynamic}")
        print(f"  shape: {detect.shape}")
        
        # 运行一次forward看anchors是否正确生成
        model.model.eval()
        with torch.no_grad():
            _ = model.model(test_img)
        
        print(f"\nforward后:")
        print(f"  anchors.shape: {detect.anchors.shape}")
        print(f"  strides: {detect.strides}")
        print(f"  shape: {detect.shape}")
        
        # ===== 关键测试2: 逐步检查推理流程 =====
        print("\n" + "=" * 80)
        print("【测试2】逐步追踪推理流程")
        print("=" * 80)
        
        model.model.eval()
        with torch.no_grad():
            # 获取原始输出
            output = model.model(test_img)
            
            if isinstance(output, tuple) and len(output) == 2:
                y, x_raw = output
                
                print(f"\n1. 原始输出:")
                print(f"   y.shape: {y.shape}  (应该是 [B, num_boxes, 4+1+nc])")
                print(f"   x_raw类型: {type(x_raw)}")
                
                # 检查y的内容
                print(f"\n2. y的统计:")
                print(f"   范围: [{y.min().item():.4f}, {y.max().item():.4f}]")
                print(f"   均值: {y.mean().item():.4f}")
                print(f"   标准差: {y.std().item():.4f}")
                
                # 检查bbox部分 (前4列)
                bbox = y[0, :, :4]
                print(f"\n3. BBox部分 (前4列):")
                print(f"   范围: [{bbox.min().item():.4f}, {bbox.max().item():.4f}]")
                print(f"   是否有负值: {(bbox < 0).any().item()}")
                print(f"   是否超出图像: {(bbox > 640).any().item()}")
                
                # 检查置信度 (第5列, index=4)
                if y.shape[2] >= 5:
                    conf = y[0, :, 4]
                    print(f"\n4. 置信度部分 (第5列):")
                    print(f"   范围: [{conf.min().item():.6f}, {conf.max().item():.6f}]")
                    print(f"   平均: {conf.mean().item():.6f}")
                    print(f"   中位数: {conf.median().item():.6f}")
                    
                    # 统计不同阈值下的框数
                    print(f"\n   各阈值下的框数:")
                    for thresh in [0.0001, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5]:
                        count = (conf > thresh).sum().item()
                        print(f"     > {thresh:.4f}: {count}")
                
                # 检查类别概率 (第6列开始)
                if y.shape[2] >= 6:
                    cls_probs = y[0, :, 5:]
                    print(f"\n5. 类别概率部分 (第6列起):")
                    print(f"   形状: {cls_probs.shape}")
                    print(f"   范围: [{cls_probs.min().item():.6f}, {cls_probs.max().item():.6f}]")
                    print(f"   平均: {cls_probs.mean().item():.6f}")
                
        # ===== 关键测试3: 模拟NMS流程 =====
        print("\n" + "=" * 80)
        print("【测试3】模拟NMS流程，找出过滤原因")
        print("=" * 80)
        
        from ultralytics.utils.ops import non_max_suppression
        
        # 使用极低的阈值测试
        print("\n使用极低阈值测试NMS:")
        for conf_t in [0.0001, 0.001, 0.01]:
            nms_out = non_max_suppression(
                y,
                conf_thres=conf_t,
                iou_thres=0.7,
                max_det=300
            )
            print(f"  conf={conf_t}: 检测到 {len(nms_out[0])} 个框")
            if len(nms_out[0]) > 0:
                print(f"    前3个框的置信度: {nms_out[0][:3, 4].tolist()}")
        
        # ===== 关键测试4: 检查LazyConv2d初始化状态 =====
        print("\n" + "=" * 80)
        print("【测试4】检查LazyConv2d初始化状态")
        print("=" * 80)
        
        lazy_layers = []
        for name, module in model.model.named_modules():
            if isinstance(module, torch.nn.modules.lazy.LazyConv2d):
                lazy_layers.append((name, module))
                has_weight = hasattr(module, 'weight') and module.weight is not None
                print(f"  {name}: 已初始化={has_weight}")
                if has_weight:
                    print(f"    weight.shape: {module.weight.shape}")
        
        if not lazy_layers:
            print("  ✅ 未发现LazyConv2d层")
        
        # ===== 关键测试5: 检查BatchNorm的running stats =====
        print("\n" + "=" * 80)
        print("【测试5】检查BatchNorm统计量")
        print("=" * 80)
        
        bn_stats = []
        for name, module in model.model.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                if 'dino' in name.lower() or 'fusion' in name.lower():
                    bn_stats.append(name)
                    print(f"\n  {name}:")
                    print(f"    training: {module.training}")
                    print(f"    track_running_stats: {module.track_running_stats}")
                    if module.running_mean is not None:
                        print(f"    running_mean: 均值={module.running_mean.mean().item():.6f}, "
                              f"标准差={module.running_mean.std().item():.6f}")
                        print(f"    running_var: 均值={module.running_var.mean().item():.6f}, "
                              f"标准差={module.running_var.std().item():.6f}")
                        print(f"    num_batches_tracked: {module.num_batches_tracked.item()}")
        
        if not bn_stats:
            print("  未发现DINO/fusion相关的BatchNorm")
        
    else:
        print("未找到训练权重")
else:
    print("未找到训练目录")

print("\n" + "=" * 80)
print("测试完成")
print("=" * 80)
print("\n关键检查点:")
print("1. 如果置信度全部<0.001 -> 模型输出有问题，可能是DINO模块导致")
print("2. 如果bbox超出图像范围 -> decode_bboxes有问题")
print("3. 如果LazyConv2d未初始化 -> 需要先warmup")
print("4. 如果BatchNorm的num_batches_tracked=0 -> 训练时没有更新统计量")
print("5. 如果NMS前有框但NMS后无框 -> NMS阈值或格式问题")
