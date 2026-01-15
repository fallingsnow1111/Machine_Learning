from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from ultralytics.nn.modules import Conv


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (lightweight).

    **符合 YOLO 参数契约**:
        c1: 输入通道数 (第一个参数，YOLO 自动传入)
        c2: 输出通道数 (第二个参数，YAML 指定，自动缩放)
        rates: 膨胀率列表 (可选参数)
        
    YAML 示例: [-1, 1, ASPP, [256, [1, 6, 12, 18]]]
    解析结果: c1=128 (自动), c2=64 (256*0.25), rates=[1,6,12,18]
    """

    def __init__(self, c1, c2, rates=(1, 6, 12, 18)):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.rates = tuple(int(r) for r in rates)
        if len(self.rates) < 2:
            raise ValueError("ASPP rates must contain at least 2 values")

        # 立即构建层（c1 和 c2 已知）
        self.branches = nn.ModuleList()
        self.branches.append(Conv(c1, c2, k=1, s=1))
        for r in self.rates[1:]:
            self.branches.append(Conv(c1, c2, k=3, s=1, d=r))

        self.project = Conv(c2 * len(self.branches), c2, k=1, s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [branch(x) for branch in self.branches]
        return self.project(torch.cat(outs, dim=1))
        ys = [b(x) for b in self.branches]
        return self.project(torch.cat(ys, dim=1))
