"""
诊断验证时mAP=0但测试正常的问题
关键检查点：
1. 模型在train()和eval()模式下的输出
2. NMS前后的预测数量
3. 置信度分布
4. 验证参数设置
"""

import torch
import numpy as np
from ultralytics import YOLO
from ultralytics.utils import ops

print("=" * 80)
print("诊断：验证时mAP=0但测试正常的问题")
print("=" * 80)

# 加载模型 - 假设已经训练过
model_path = "./runs/detect/train/weights/last.pt"  # 修改为你的模型路径
try:
    model = YOLO(model_path)
    print(f"✅ 加载模型: {model_path}")
except:
    print(f"⚠️ 无法加载训练模型，使用随机初始化的模型")
    model = YOLO("./yolo11P.yaml")

# 创建测试输入
batch_size = 2
img_size = 640
x = torch.randn(batch_size, 3, img_size, img_size)
print(f"\n输入: {x.shape}")

# ============================================================================
# 测试1: 检查训练模式输出
# ============================================================================
print("\n" + "=" * 80)
print("测试1: 训练模式输出")
print("=" * 80)
model.model.train()
with torch.no_grad():
    try:
        output_train = model.model(x)
        print(f"✅ 训练模式forward成功")
        print(f"输出类型: {type(output_train)}")
        
        if isinstance(output_train, list):
            print(f"输出列表长度: {len(output_train)}")
            for i, o in enumerate(output_train):
                if isinstance(o, torch.Tensor):
                    print(f"  层{i}: shape={o.shape}, dtype={o.dtype}")
                    # 检查是否包含NaN或Inf
                    print(f"       NaN: {torch.isnan(o).any().item()}, Inf: {torch.isinf(o).any().item()}")
                    print(f"       范围: [{o.min().item():.4f}, {o.max().item():.4f}]")
    except Exception as e:
        print(f"❌ 训练模式失败: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# 测试2: 检查验证模式输出（关键！）
# ============================================================================
print("\n" + "=" * 80)
print("测试2: 验证模式输出（eval模式）")
print("=" * 80)
model.model.eval()
with torch.no_grad():
    try:
        output_eval = model.model(x)
        print(f"✅ 验证模式forward成功")
        print(f"输出类型: {type(output_eval)}")
        
        if isinstance(output_eval, tuple) and len(output_eval) == 2:
            y, x_raw = output_eval
            print(f"\n返回: (y, x_raw)")
            print(f"  y.shape: {y.shape}")
            print(f"  y.dtype: {y.dtype}")
            print(f"  NaN: {torch.isnan(y).any().item()}, Inf: {torch.isinf(y).any().item()}")
            print(f"  范围: [{y.min().item():.4f}, {y.max().item():.4f}]")
            
            # 分析预测内容
            print(f"\n  预测格式分析 [B, N, C]:")
            print(f"    batch_size: {y.shape[0]}")
            print(f"    max_boxes: {y.shape[1]}")
            print(f"    info_dim: {y.shape[2]} (应该是 4+1+nc)")
            
            # 检查每个维度的统计
            print(f"\n  各列统计（第一个batch）:")
            for i in range(min(10, y.shape[2])):
                col_data = y[0, :, i]
                print(f"    列{i}: 范围=[{col_data.min().item():.4f}, {col_data.max().item():.4f}], "
                      f"均值={col_data.mean().item():.4f}")
            
            # 检查置信度（通常在第5列，即index=4）
            if y.shape[2] >= 5:
                print(f"\n  置信度分析（假设在第5列）:")
                conf = y[:, :, 4]
                for thresh in [0.001, 0.01, 0.05, 0.1, 0.25, 0.5]:
                    count = (conf > thresh).sum(dim=1)
                    print(f"    置信度 > {thresh}: {count.tolist()}")
                
                # 显示最高置信度的几个预测
                print(f"\n  最高置信度的5个预测:")
                top_conf_idx = conf[0].topk(5).indices
                for idx in top_conf_idx:
                    print(f"    Box {idx.item()}: {y[0, idx, :8].tolist()}")
                    
        elif isinstance(output_eval, torch.Tensor):
            print(f"输出shape: {output_eval.shape}")
            
    except Exception as e:
        print(f"❌ 验证模式失败: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# 测试3: 模拟验证流程的NMS
# ============================================================================
print("\n" + "=" * 80)
print("测试3: 模拟验证流程的NMS")
print("=" * 80)

if isinstance(output_eval, tuple) and len(output_eval) == 2:
    y_pred = output_eval[0]
    
    # 使用不同的NMS参数测试
    for conf_thresh in [0.001, 0.01, 0.25]:
        for iou_thresh in [0.45, 0.7]:
            print(f"\n  NMS参数: conf={conf_thresh}, iou={iou_thresh}")
            
            try:
                from ultralytics.utils.ops import non_max_suppression
                
                # 注意：y_pred可能需要转换格式
                # YOLO的输出格式通常是 [x, y, w, h, conf, cls...]
                nms_output = non_max_suppression(
                    y_pred,
                    conf_thres=conf_thresh,
                    iou_thres=iou_thresh,
                    max_det=300
                )
                
                print(f"    NMS输出框数: {[len(o) for o in nms_output]}")
                
                if len(nms_output[0]) > 0:
                    print(f"    第一张图的前3个框:")
                    print(f"      {nms_output[0][:3]}")
                    
            except Exception as e:
                print(f"    NMS失败: {e}")

# ============================================================================
# 测试4: 检查Detect Head的配置
# ============================================================================
print("\n" + "=" * 80)
print("测试4: Detect Head配置")
print("=" * 80)

detect = model.model.model[-1]
print(f"Detect类型: {type(detect).__name__}")
print(f"  nc (类别数): {detect.nc}")
print(f"  nl (检测层数): {detect.nl}")
print(f"  reg_max: {detect.reg_max}")
print(f"  no (每anchor输出): {detect.no} (应该是 {detect.reg_max * 4 + detect.nc})")
print(f"  stride: {detect.stride}")
print(f"  training: {detect.training}")
print(f"  export: {detect.export}")

# 检查cv2和cv3的输出通道
for i in range(detect.nl):
    cv2_out = detect.cv2[i][-1].out_channels
    cv3_out = detect.cv3[i][-1].out_channels
    print(f"  层{i}: cv2输出={cv2_out}, cv3输出={cv3_out}, "
          f"总计={cv2_out + cv3_out}, 期望={detect.no}")

# ============================================================================
# 测试5: 检查DINO模块的影响
# ============================================================================
print("\n" + "=" * 80)
print("测试5: DINO模块状态")
print("=" * 80)

dino_modules_found = False
for name, module in model.model.named_modules():
    if 'dino' in name.lower() or 'DINO' in str(type(module)):
        dino_modules_found = True
        print(f"\n模块: {name}")
        print(f"  类型: {type(module).__name__}")
        print(f"  training: {module.training}")
        
        # 检查是否有LazyConv2d
        for sub_name, sub_module in module.named_modules():
            if isinstance(sub_module, torch.nn.modules.lazy.LazyConv2d):
                print(f"  发现LazyConv2d: {sub_name}")
                # 检查是否已初始化
                if hasattr(sub_module, 'weight') and sub_module.weight is not None:
                    print(f"    已初始化: weight.shape={sub_module.weight.shape}")
                else:
                    print(f"    ⚠️ 未初始化!")

if not dino_modules_found:
    print("未找到DINO模块")

# ============================================================================
# 测试6: 对比test和val的参数
# ============================================================================
print("\n" + "=" * 80)
print("测试6: 验证器参数")
print("=" * 80)

# 检查默认验证参数
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG

val_cfg = get_cfg(DEFAULT_CFG)
print(f"默认验证参数:")
print(f"  conf: {val_cfg.conf}")
print(f"  iou: {val_cfg.iou}")
print(f"  max_det: {val_cfg.max_det}")
print(f"  half: {val_cfg.half}")
print(f"  batch: {val_cfg.batch}")

print("\n" + "=" * 80)
print("诊断完成")
print("=" * 80)

print("\n可能的问题:")
print("1. 如果NMS前有框，NMS后没有 -> NMS阈值问题或框格式问题")
print("2. 如果NMS前就没框 -> 模型输出问题或置信度过低")
print("3. 如果置信度都很低 -> 模型未训练好或DINO模块有问题")
print("4. 如果有LazyConv2d未初始化 -> 需要先运行一次forward初始化")
