from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from ultralytics.nn.modules import Conv


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (lightweight, lazy-build).

    Important:
        We intentionally *do not* require c1 at __init__ time.
        Ultralytics YAML parser only auto-injects c1 for a fixed set of built-in modules.
        To avoid modifying ultralytics/, this module initializes its internal layers on the first forward.

    Args:
        c2: output channels for each branch. If None, keep same as input channels.
        rates: dilation rates for 3x3 branches
    """

    def __init__(self, c2: int | None = None, rates: Iterable[int] = (1, 6, 12, 18)):
        super().__init__()
        self.c2 = c2
        self.rates = tuple(int(r) for r in rates)
        if len(self.rates) < 2:
            raise ValueError("ASPP rates must contain at least 2 values")

        self._built = False
        self.branches = nn.ModuleList()
        self.project: nn.Module | None = None

    def _build(self, c1: int):
        c2 = int(c1 if self.c2 is None else self.c2)

        branches = nn.ModuleList()
        branches.append(Conv(c1, c2, k=1, s=1))
        for r in self.rates[1:]:
            branches.append(Conv(c1, c2, k=3, s=1, d=r))

        self.branches = branches
        self.project = Conv(c2 * len(self.branches), c2, k=1, s=1)
        self._built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._built:
            self._build(int(x.shape[1]))
            assert self.project is not None
        ys = [b(x) for b in self.branches]
        return self.project(torch.cat(ys, dim=1))
