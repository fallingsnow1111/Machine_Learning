from __future__ import annotations

import torch
import torch.nn as nn


class EMA(nn.Module):
    """Efficient Multi-scale Attention (a lightweight, group-wise attention).

    This is a practical implementation for detection backbones/necks:
    - split channels into groups
    - compute spatial context along H and W
    - generate a sigmoid gate and reweight features

    Args:
        channels: input channels
        factor: number of groups (must divide channels)

    Output shape is the same as input.
    """

    def __init__(self, channels: int | None = None, factor: int = 8):
        super().__init__()
        if factor <= 0:
            raise ValueError("EMA factor must be positive")

        self._requested_channels = channels
        self._requested_factor = factor

        self._built = False
        self.groups = 1
        self.conv1: nn.Conv2d | None = None
        self.conv3: nn.Conv2d | None = None
        self.bn: nn.BatchNorm2d | None = None
        self.act = nn.SiLU()
        self.gate: nn.Conv2d | None = None

    def _pick_groups(self, c: int) -> int:
        g = min(self._requested_factor, c)
        while g > 1 and (c % g) != 0:
            g -= 1
        return max(g, 1)

    def _build(self, c: int):
        g = self._pick_groups(c)
        gc = c // g

        self.groups = g
        self.conv1 = nn.Conv2d(gc, gc, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(gc, gc, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(gc)
        self.gate = nn.Conv2d(gc, gc, kernel_size=1, bias=True)
        self._built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if not self._built:
            self._build(int(c))
            assert self.conv1 is not None and self.conv3 is not None and self.bn is not None and self.gate is not None

        g = self.groups
        gc = c // g

        # (B*G, Cg, H, W)
        xg = x.view(b * g, gc, h, w)

        # multi-directional context
        ctx_h = xg.mean(dim=3, keepdim=True)  # (B*G, Cg, H, 1)
        ctx_w = xg.mean(dim=2, keepdim=True)  # (B*G, Cg, 1, W)
        ctx = ctx_h + ctx_w

        y = self.conv1(ctx)  # type: ignore[operator]
        y = self.conv3(y)  # type: ignore[operator]
        y = self.act(self.bn(y))  # type: ignore[arg-type]

        attn = torch.sigmoid(self.gate(y))  # type: ignore[operator]
        out = xg * attn

        return out.view(b, c, h, w)
