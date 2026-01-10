import cv2
import numpy as np

# 替换为你觉得“看起来是灰色”的图片路径
# 比如: runs/detect/dust_detection/stage2_unfreeze/val_batch0_pred.jpg
# 或者: dataset_processed/images/train/某张图片.jpg
image_path = "/root/autodl-tmp/DustDetection/v2/runs/detect/train8/val_batch2_pred.jpg" 

img = cv2.imread(image_path)

if img is None:
    print("❌ 无法读取图片，请检查路径。")
else:
    print(f"1. 图像形状: {img.shape}")
    if len(img.shape) == 3 and img.shape[2] == 3:
        print("✅ 确认：YOLO 正在使用 3 通道图像。")
    else:
        print("⚠️ 警告：图像确实是单通道的！")

    # 计算通道间的差异 (数学证明)
    # 分离通道 (OpenCV 默认 BGR)
    b, g, r = cv2.split(img)

    # 计算绝对差值
    diff_bg = np.mean(np.abs(b.astype(int) - g.astype(int)))
    diff_br = np.mean(np.abs(b.astype(int) - r.astype(int)))

    print(f"\n2. 通道差异分析:")
    print(f"   B通道(原图) vs G通道(滤波) 平均像素差: {diff_bg:.4f}")
    print(f"   B通道(原图) vs R通道(CLAHE) 平均像素差: {diff_br:.4f}")

    if diff_br < 1.0 and diff_bg < 1.0:
        print("\n🧐 结论: 三个通道几乎一模一样，所以看起来是纯灰色的。")
    else:
        print("\n🎉 结论: 通道间存在数值差异！虽然看起来像灰色，但计算机能看到它是彩色的。")
        print("   (YOLO 模型完全可以利用这些差异进行特征提取)")