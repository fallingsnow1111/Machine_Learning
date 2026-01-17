import torch
from torch import nn
import torch.nn.functional as F
from modelscope import AutoModel
import numpy as np


class DINO3Preprocessor(nn.Module):
    """
    DINO3 Preprocessor - 在P0输入阶段增强图像
    
    架构: Input Image (3ch) -> DINO3特征提取 -> 卷积网络 -> Enhanced Image (3ch)
    输出增强的RGB图像，而非特征向量
    
    Args:
        c1: 输入通道数（YOLO 自动传入，通常是 3）
        model_name_or_path: DINO 模型路径或名称
        output_channels: 输出通道数（默认 3）
    """
    def __init__(self, c1, model_name_or_path='facebook/dinov3-vitl16-pretrain-lvd1689m', output_channels=3):
        super().__init__()
        self.c1 = c1
        self.model_name = model_name_or_path
        self.output_channels = output_channels
        
        # 从 modelscope 加载 DINO 模型
        print(f"📥 DINO3Preprocessor 正在从路径加载模型: {model_name_or_path}")
        print(f"   输入通道: {c1}, 输出通道: {output_channels}")
        self.dino = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.embed_dim = self.dino.config.hidden_size  # 1024 for vitl16
        self.patch_size = self.dino.config.patch_size  # 16
        
        # 特征处理网络: DINO特征 -> 3通道增强图像
        # 参考仓库: 通过卷积网络将高维特征转换为3通道图像
        self.feature_processor = nn.Sequential(
            nn.Conv2d(self.embed_dim, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.SiLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.Conv2d(256, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, self.output_channels, 3, padding=1),
            nn.Tanh()  # 输出归一化到 [-1, 1]
        )
        
        # 残差连接权重
        self.gamma = nn.Parameter(torch.zeros(1))
        
        print(f"✅ DINO3Preprocessor 初始化完成")
        print(f"   特征维度: {self.embed_dim}, 输出通道: {self.output_channels}")
    
    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] 输入图像
        Returns:
            enhanced_image: [B, 3, H, W] 增强后的图像
        """
        B, C, H, W = x.shape
        device = x.device
        original_input = x
        
        # DINO 期望输入: [B, 3, 1024, 1024]
        x_resized = F.interpolate(x, size=(1024, 1024), mode='bilinear', align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x_normalized = (x_resized - mean) / std
        
        # 提取 DINO 特征
        with torch.no_grad():
            outputs = self.dino(pixel_values=x_normalized, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1]  # [B, num_tokens, embed_dim]
        
        # 去掉 [CLS] token 和 register tokens
        num_registers = 4
        spatial_features = last_hidden_state[:, 1 + num_registers:, :]  # [B, num_patches, embed_dim]
        
        # 重塑为空间特征图
        _, num_patches, _ = spatial_features.shape
        h = w = int(np.sqrt(num_patches))
        dino_features = spatial_features.permute(0, 2, 1).reshape(B, self.embed_dim, h, w)
        
        # 通过特征处理网络转换为3通道图像
        enhanced_features = self.feature_processor(dino_features)
        
        # 上采样到原始尺寸
        enhanced_features = F.interpolate(
            enhanced_features, size=(H, W), mode='bilinear', align_corners=False
        )
        
        # Tanh输出是 [-1, 1]，归一化到 [0, 1]
        enhanced_features = (enhanced_features + 1) / 2
        
        # 与原图加权残差连接
        enhanced_image = (
            original_input * (1 - self.gamma) + 
            enhanced_features * self.gamma
        )
        
        return enhanced_image


class DINO3Backbone(nn.Module):
    """
    DINO3 Backbone - 在P3阶段增强CNN特征
    
    架构: CNN Features -> 投影为伪RGB -> DINO3特征提取 -> 与原CNN特征融合
    
    Args:
        c1: 输入通道数（YOLO 自动传入，如 P3 层的 512 通道）
        model_name_or_path: DINO 模型路径或名称
        output_channels: 输出通道数（如 128）
    """
    def __init__(self, c1, model_name_or_path='facebook/dinov3-vits16-pretrain-lvd1689m', 
                 output_channels=512):
        super().__init__()
        self.c1 = c1  # 保存输入通道数
        self.model_name = model_name_or_path
        self.output_channels = output_channels
        
        # 从 modelscope 加载 DINO 模型
        print(f"📥 DINO3Backbone 正在从路径加载模型: {model_name_or_path}")
        print(f"   输入通道: {c1}, 输出通道: {output_channels}")
        self.dino = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.embed_dim = self.dino.config.hidden_size  # 1024 for vitl16
        self.patch_size = self.dino.config.patch_size  # 16

        
        # 投影层将在第一次forward时动态创建（因为input_channels可能未知）
        self.input_projection = None
        self.fusion_layer = None
        self.feature_adapter = None
        self.spatial_projection = None
        
        print(f"✅ DINO3Backbone 初始化完成")
        print(f"   特征维度: {self.embed_dim}, 输出通道: {self.output_channels}")
    
    def _create_projection_layers(self, input_channels=None):
        """根据实际输入通道数创建投影层"""
        # 如果没有传入 input_channels，使用 self.c1
        if input_channels is None:
            input_channels = self.c1
        
        # CNN特征 -> 伪RGB (用于送入DINO)
        self.input_projection = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 1, 1),
            nn.Tanh()
        )
        
        # DINO特征适配器: embed_dim -> output_channels
        self.feature_adapter = nn.Sequential(
            nn.Linear(self.embed_dim, self.output_channels),
            nn.LayerNorm(self.output_channels),
            nn.GELU()
        )
        
        # 空间投影: 调整特征图分辨率
        self.spatial_projection = nn.Sequential(
            nn.Conv2d(self.output_channels, self.output_channels, 3, 1, 1),
            nn.BatchNorm2d(self.output_channels),
            nn.ReLU(inplace=True)
        )
        
        # 融合层: CNN特征 + DINO特征
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(input_channels + self.output_channels, self.output_channels, 3, 1, 1),
            nn.BatchNorm2d(self.output_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] CNN特征 (如P3层的256通道特征)
        Returns:
            enhanced_features: [B, output_channels, H, W] 增强后的特征
        """
        B, C, H, W = x.shape
        device = x.device
        
        # 第一次forward时创建投影层
        if self.input_projection is None:
            self.input_channels = C
            self._create_projection_layers(C)
            # 移动到与输入相同的设备
            self.input_projection = self.input_projection.to(device)
            self.feature_adapter = self.feature_adapter.to(device)
            self.spatial_projection = self.spatial_projection.to(device)
            self.fusion_layer = self.fusion_layer.to(device)
        
        # 1. 将CNN特征投影为伪RGB图像
        pseudo_rgb = self.input_projection(x)  # [B, 3, H, W]
        
        # 2. 调整到DINO期望的尺寸
        dino_size = 224  # DINO训练时的标准尺寸
        pseudo_rgb_resized = F.interpolate(
            pseudo_rgb, size=(dino_size, dino_size), 
            mode='bilinear', align_corners=False
        )
        
        # ImageNet 归一化
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        pseudo_rgb_normalized = (pseudo_rgb_resized - mean) / std
        
        # 3. 通过DINO提取特征
        with torch.no_grad():
            outputs = self.dino(pixel_values=pseudo_rgb_normalized, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1]  # [B, num_tokens, embed_dim]
        
        # 去掉 [CLS] token 和 register tokens
        num_registers = 4
        spatial_features = last_hidden_state[:, 1 + num_registers:, :]  # [B, num_patches, embed_dim]
        
        # 重塑为空间特征图
        _, num_patches, _ = spatial_features.shape
        h = w = int(np.sqrt(num_patches))
        
        # 4. 适配通道维度
        # [B, num_patches, embed_dim] -> [B, h, w, embed_dim]
        features_2d = spatial_features.view(B, h, w, self.embed_dim)
        # 通过线性层适配: embed_dim -> output_channels
        adapted_features = self.feature_adapter(features_2d)  # [B, h, w, output_channels]
        # 转换为 [B, output_channels, h, w]
        adapted_features = adapted_features.permute(0, 3, 1, 2)
        
        # 5. 空间投影和上采样到原始尺寸
        dino_features = self.spatial_projection(adapted_features)
        dino_features_resized = F.interpolate(
            dino_features, size=(H, W), 
            mode='bilinear', align_corners=False
        )
        
        # 6. 与原CNN特征融合
        combined_features = torch.cat([x, dino_features_resized], dim=1)
        enhanced_features = self.fusion_layer(combined_features)
        
        return enhanced_features