from __future__ import annotations

import torch
import torch.nn as nn


class EMA(nn.Module):
    """Efficient Multi-scale Attention (group-wise attention).

    **符合 YOLO 参数契约**:
        c1: 输入通道数 (第一个参数，YOLO 自动传入)
        c2: 输出通道数 (第二个参数，通常等于 c1)
        factor: 分组因子 (可选参数)

    YAML 示例: [-1, 1, EMA, [256, 8]]
    解析结果: c1=128 (自动), c2=64 (256*0.25), factor=8
    """

    def __init__(self, c1, c2, factor=8):
        super().__init__()
        if factor <= 0:
            raise ValueError("EMA factor must be positive")
        if c1 != c2:
            raise ValueError(f"EMA requires c1==c2 (got c1={c1}, c2={c2})")

        self.c = c1  # c1 == c2
        self.factor = factor
        self.groups = self._pick_groups(c1)
        gc = c1 // self.groups

        # 立即构建层
        self.conv1 = nn.Conv2d(gc, gc, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(gc, gc, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(gc)
        self.act = nn.SiLU()
        self.gate = nn.Conv2d(gc, gc, kernel_size=1)

    def _pick_groups(self, c: int) -> int:
        g = min(self.factor, c)
        while g > 1 and (c % g) != 0:
            g -= 1
        return max(g, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        g = self.groups
        gc = c // g

        # (B*G, Cg, H, W)
        xg = x.view(b * g, gc, h, w)

        # multi-directional context
        ctx_h = xg.mean(dim=3, keepdim=True)  # (B*G, Cg, H, 1)
        ctx_w = xg.mean(dim=2, keepdim=True)  # (B*G, Cg, 1, W)
        ctx = ctx_h + ctx_w

        y = self.conv1(ctx)
        y = self.conv3(y)
        y = self.act(self.bn(y))

        attn = torch.sigmoid(self.gate(y))
        out = xg * attn

        return out.view(b, c, h, w)
