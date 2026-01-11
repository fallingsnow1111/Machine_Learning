# Soft NMS 修改说明

## 修改概述

为了便于区分原始 NMS 和 Soft NMS，我在 v3 文件夹下的 ultralytics 代码中添加了 Soft NMS 功能。Soft NMS 是一种改进的非极大值抑制方法，它通过降低重叠框的置信度而不是完全抑制它们，来保留更多潜在正确的检测结果。

## 新增功能

- **TorchNMS.soft_nms**: 在 `ultralytics/utils/nms.py` 的 `TorchNMS` 类中新增的静态方法，实现 Soft NMS 算法。

  - 支持线性衰减和 Gaussian 衰减两种方法。
  - 参数包括 IoU 阈值、衰减方法和 sigma 值。

- **non_max_suppression 函数更新**: 在 `non_max_suppression` 函数中添加了 `soft_nms` 参数，默认为 False。
  - 当 `soft_nms=True` 时，使用 Soft NMS 而不是硬 NMS。
  - 新增参数：
    - `soft_nms_method`: 衰减方法 ('linear' 或 'gaussian')。
    - `soft_nms_sigma`: Gaussian 方法的 sigma 值。

## 修改的文件和位置

1. **`ultralytics/utils/nms.py`**:

   - 添加 `TorchNMS.soft_nms` 方法。
   - 修改 `non_max_suppression` 函数签名和内部逻辑。

2. **`ultralytics/models/yolo/detect/predict.py`** (第 54 行附近):

   - 在 `postprocess` 方法中调用 `non_max_suppression` 时，添加 `soft_nms=True` 等参数。

3. **`ultralytics/models/yolo/detect/val.py`** (第 115 行附近):
   - 在 `postprocess` 方法中调用 `non_max_suppression` 时，添加 `soft_nms=True` 等参数。

## 使用说明

- 默认情况下，代码仍使用原始 NMS。要启用 Soft NMS，请在调用 `non_max_suppression` 时设置 `soft_nms=True`。
- Soft NMS 可能返回更多检测框，因为它不会完全抑制重叠框。建议根据任务调整 IoU 阈值和衰减参数。
- 测试时，请验证性能和准确性。

## 注意事项

- Soft NMS 会改变检测结果的分布，可能需要重新调整置信度阈值。
- 如果需要恢复原始行为，将 `soft_nms` 参数设为 False。
