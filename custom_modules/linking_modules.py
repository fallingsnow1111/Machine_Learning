"""
Linking 分支的改进模块
包含 SPPELAN, C3k2PC, SeNet, PConv 等优化组件
"""

import torch
from torch import nn
import torch.nn.functional as F


# ==========================================
# 1. PConv - Partial Convolution (核心加速模块)
# ==========================================
class PConv(nn.Module):
    """
    Partial Convolution - 只对部分通道做卷积，其余直接传递
    论文: Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks
    
    优势:
    - 减少计算量 25-30%
    - 保持精度
    - 适合边缘模糊的小目标
    """
    def __init__(self, dim, n_div=4, forward="split_cat"):
        """
        Args:
            dim: 输入通道数
            n_div: 分割比例，dim//n_div 的通道做卷积
            forward: 前向模式 ("slicing" 用于推理, "split_cat" 用于训练)
        """
        super(PConv, self).__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == "slicing":
            self.forward = self.forward_slicing
        elif forward == "split_cat":
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        """推理模式：原地修改"""
        x = x.clone()
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        return x

    def forward_split_cat(self, x):
        """训练模式：分割-卷积-拼接"""
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x


# ==========================================
# 2. SPPELAN - 改进的空间金字塔池化
# ==========================================
class SPPELAN(nn.Module):
    """
    SPP-ELAN - 高效的空间金字塔池化
    
    与 SPPF 的区别:
    - 串行池化 (更高效)
    - 中间通道数可控
    - 计算量更小
    """
    def __init__(self, c1, c2, c3=None, k=5):
        """
        Args:
            c1: 输入通道
            c2: 输出通道
            c3: 中间通道（默认为 c2//2）
            k: 池化核大小
        """
        super().__init__()
        c3 = c3 or c2 // 2
        self.c = c3
        
        # 导入 Conv（从 ultralytics）
        try:
            from ultralytics.nn.modules import Conv
        except ImportError:
            # 如果导入失败，使用简单的 Conv 替代
            class Conv(nn.Module):
                def __init__(self, c1, c2, k=1, s=1):
                    super().__init__()
                    self.conv = nn.Conv2d(c1, c2, k, s, k//2, bias=False)
                    self.bn = nn.BatchNorm2d(c2)
                    self.act = nn.SiLU(inplace=True)
                def forward(self, x):
                    return self.act(self.bn(self.conv(x)))
        
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        """串行多尺度池化"""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


# ==========================================
# 3. SeNet - 通道注意力机制
# ==========================================
class SeNet(nn.Module):
    """
    Squeeze-and-Excitation Network
    
    轻量级通道注意力:
    - 全局平均池化 -> 两次全连接 -> Sigmoid
    - 自适应重校准通道权重
    """
    def __init__(self, channel, ratio=16):
        """
        Args:
            channel: 输入通道数
            ratio: 压缩比例（默认 16）
        """
        super(SeNet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, h, w = x.size()
        # 全局平均池化: [B, C, H, W] -> [B, C, 1, 1] -> [B, C]
        avg = self.avg_pool(x).view([b, c])
        # 通道注意力: [B, C] -> [B, C, 1, 1]
        fc = self.fc(avg).view([b, c, 1, 1])
        # 加权: 原特征图 * 注意力权重
        return x * fc


# ==========================================
# 4. BottleneckPC - 使用 PConv 的 Bottleneck
# ==========================================
class BottleneckPC(nn.Module):
    """
    带 PConv 的 Bottleneck 模块
    
    结构:
    - 1x1 Conv 降维
    - PConv 提取特征 + 1x1 Conv 升维
    - 残差连接
    """
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """
        Args:
            c1: 输入通道
            c2: 输出通道
            shortcut: 是否使用残差连接
            g: 分组卷积数
            k: 卷积核大小
            e: 扩展比例
        """
        super().__init__()
        c_ = int(c2 * e)  # 中间通道
        
        # 导入 Conv
        try:
            from ultralytics.nn.modules import Conv
        except ImportError:
            class Conv(nn.Module):
                def __init__(self, c1, c2, k=1, s=1):
                    super().__init__()
                    self.conv = nn.Conv2d(c1, c2, k, s, k//2, bias=False)
                    self.bn = nn.BatchNorm2d(c2)
                    self.act = nn.SiLU(inplace=True)
                def forward(self, x):
                    return self.act(self.bn(self.conv(x)))
        
        self.cv1 = Conv(c1, c_, k[0], 1)
        # 核心改进: PConv + 1x1 Conv
        self.cv2 = nn.Sequential(
            PConv(c_, n_div=4, forward="split_cat"),
            Conv(c_, c2, 1, 1)
        )
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


# ==========================================
# 5. C3kPC - 使用 PConv 的 C3k 模块
# ==========================================
class C3kPC(nn.Module):
    """C3k block with PConv"""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.m = nn.Sequential(*(BottleneckPC(c1, c2, shortcut, g, e=e) for _ in range(n)))

    def forward(self, x):
        return self.m(x)


# ==========================================
# 6. C3k2PC - 主力模块
# ==========================================
class C3k2PC(nn.Module):
    """
    CSP Bottleneck with 2 convolutions and Partial Conv
    
    YOLO11 中的 C2f 改进版，使用 PConv 加速
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """
        Args:
            c1: 输入通道
            c2: 输出通道
            n: Bottleneck 数量
            c3k: 是否使用 C3k 模式
            e: 扩展比例
            g: 分组数
            shortcut: 残差连接
        """
        super().__init__()
        self.c = int(c2 * e)  # 中间通道
        
        # 导入 Conv
        try:
            from ultralytics.nn.modules import Conv
        except ImportError:
            class Conv(nn.Module):
                def __init__(self, c1, c2, k=1, s=1):
                    super().__init__()
                    self.conv = nn.Conv2d(c1, c2, k, s, k//2, bias=False)
                    self.bn = nn.BatchNorm2d(c2)
                    self.act = nn.SiLU(inplace=True)
                def forward(self, x):
                    return self.act(self.bn(self.conv(x)))
        
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)
        
        # 使用 PConv 版本的 Bottleneck
        self.m = nn.ModuleList(
            C3kPC(self.c, self.c, 2, shortcut, g) if c3k 
            else BottleneckPC(self.c, self.c, shortcut, g) 
            for _ in range(n)
        )

    def forward(self, x):
        """CSP 前向传播"""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """分割模式前向传播（用于推理优化）"""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ==========================================
# 模块导出
# ==========================================
__all__ = [
    'PConv',
    'SPPELAN', 
    'SeNet',
    'BottleneckPC',
    'C3kPC',
    'C3k2PC',
]
