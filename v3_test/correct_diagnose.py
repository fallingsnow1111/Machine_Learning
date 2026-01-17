"""
正确诊断验证mAP=0的问题
"""

import torch
from pathlib import Path

print("=" * 80)
print("正确诊断：验证mAP=0问题")
print("=" * 80)

# 加载模型
from ultralytics import YOLO

model_path = "./runs/detect/train7/weights/last.pt"
if not Path(model_path).exists():
    print(f"❌ 模型不存在: {model_path}")
    exit()

model = YOLO(model_path)
print(f"✅ 加载模型: {model_path}")

# 创建测试输入
x = torch.randn(2, 3, 640, 640)
print(f"输入: {x.shape}\n")

# 测试验证模式
model.model.eval()
with torch.no_grad():
    output = model.model(x)

if isinstance(output, tuple):
    y = output[0]
    print(f"模型输出: {y.shape}")
    print(f"  正确理解: [batch={y.shape[0]}, channels={y.shape[1]}, num_boxes={y.shape[2]}]")
    print(f"  即: {y.shape[2]}个框，每个框{y.shape[1]}个值\n")
    
    # 转置为[B, N, C]格式便于分析
    y_transposed = y.transpose(1, 2)  # [2, 8400, 5]
    print(f"转置后: {y_transposed.shape} [batch, num_boxes, info_per_box]")
    
    # 分析第一张图的数据
    first_img = y_transposed[0]  # [8400, 5]
    print(f"\n第一张图的预测:")
    print(f"  总框数: {first_img.shape[0]}")
    print(f"  每框信息: {first_img.shape[1]} (应该是 4bbox + 1conf，因为nc=1)")
    
    # 分析bbox (前4列)
    bbox = first_img[:, :4]
    print(f"\nBBox坐标 (前4列):")
    print(f"  范围: [{bbox.min().item():.2f}, {bbox.max().item():.2f}]")
    print(f"  均值: {bbox.mean().item():.2f}")
    
    # 分析置信度 (第5列，index=4)
    conf = first_img[:, 4]
    print(f"\n置信度 (第5列):")
    print(f"  范围: [{conf.min().item():.6f}, {conf.max().item():.6f}]")
    print(f"  均值: {conf.mean().item():.6f}")
    print(f"  中位数: {conf.median().item():.6f}")
    
    # 统计不同阈值下的框数
    print(f"\n不同置信度阈值下的框数:")
    for thresh in [0.001, 0.01, 0.05, 0.1, 0.25, 0.5]:
        count = (conf > thresh).sum().item()
        print(f"  > {thresh:.3f}: {count}")
    
    # 显示最高置信度的几个框
    print(f"\n最高置信度的5个框:")
    top_indices = conf.topk(5).indices
    for idx in top_indices:
        box_info = first_img[idx]
        print(f"  框{idx.item()}: bbox=[{box_info[:4].tolist()}], conf={box_info[4].item():.6f}")
    
    # 测试NMS
    print("\n" + "=" * 80)
    print("测试NMS")
    print("=" * 80)
    
    from ultralytics.utils import nms
    
    for conf_t in [0.001, 0.01, 0.25]:
        nms_out = nms.non_max_suppression(
            y,  # [B, C, N]格式
            conf_thres=conf_t,
            iou_thres=0.7,
            max_det=300
        )
        print(f"conf={conf_t}: 第一张图检测到 {len(nms_out[0])} 个框")
        if len(nms_out[0]) > 0:
            print(f"  置信度范围: [{nms_out[0][:, 4].min().item():.6f}, {nms_out[0][:, 4].max().item():.6f}]")

print("\n" + "=" * 80)
print("诊断结论")
print("=" * 80)

print("""
关键发现：
1. 如果置信度都很低（<0.001），说明模型未训练好或有问题
2. 如果NMS前有框但NMS后没框，说明NMS参数问题或bbox格式问题  
3. 如果bbox超出图像范围很多，说明decode有问题
4. 如果置信度值异常（不在0-1之间），说明激活函数有问题
""")
