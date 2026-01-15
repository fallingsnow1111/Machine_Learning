import torch
from torch import nn
import torch.nn.functional as F
from modelscope import AutoModel
import numpy as np
from sklearn.decomposition import PCA


class IdentityLayer(nn.Module):
    """恒等层，直接传递输入，不做任何变换"""
    def __init__(self):
        super().__init__()
        self.out_channels = 3  # 固定输出3通道
        
    def forward(self, x):
        return x


class DINOFeatureExtractor(nn.Module):
    """DINOv3特征提取器"""
    def __init__(self, model_name='facebook/dinov3-vitl16-pretrain-lvd1689m', freeze=True, pca_components=None):
        super().__init__()
        self.pretrained_model_name = model_name
        self.dino = AutoModel.from_pretrained(self.pretrained_model_name, device_map="auto")
        self.patch_size = self.dino.config.patch_size  # 通常是16
        self.embed_dim = self.dino.config.hidden_size  # large:1024
        self.pca_components = pca_components
        
        if freeze:
            for param in self.dino.parameters():
                param.requires_grad = False
            self.dino.eval()
        
    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] 输入图像 (假设已归一化到 [0, 1])
        Returns:
            features: [B, c, h, w] where c is pca_components if set, else embed_dim;
                      h = w = input_size / patch_size (after processor resize)
        """
        B, C, H, W = x.shape
        device = x.device
        
        # 检查是否是全0张量（用于模型初始化）
        if torch.all(x == 0):
            h_out = w_out = 64
            if self.pca_components:
                return torch.zeros(B, self.pca_components, h_out, w_out, device=device)
            else:
                return torch.zeros(B, self.embed_dim, h_out, w_out, device=device)
        
        # DINO期望输入: [B, 3, 1024, 1024], 归一化到ImageNet均值/方差
        x_resized = F.interpolate(x, size=(1024, 1024), mode='bilinear', align_corners=False)
        
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        x_normalized = (x_resized - mean) / std
        
        # 直接传给DINO模型（假设模型在GPU上）
        outputs = self.dino(pixel_values=x_normalized, output_hidden_states=True)
        
        # 获取最后一层的hidden states
        last_hidden_state = outputs.hidden_states[-1]  # [B, num_tokens, embed_dim]
        
        # 去掉[cls]和registers
        num_registers = 4 
        spatial_features = last_hidden_state[:, 1 + num_registers:, :]  # [B, num_patches, embed_dim]
        _, num_patches, _ = spatial_features.shape
        h = w = int(np.sqrt(num_patches))
        assert h * w == num_patches, "Number of patches must form a square grid"
        
        if self.pca_components:
            # 使用PyTorch在GPU上实现PCA降维
            features_list = []
            for i in range(B):
                sf = spatial_features[i]  # [num_patches, embed_dim]，保持在GPU
                
                # 保存原始数据类型，SVD需要FP32
                original_dtype = sf.dtype
                sf = sf.float()
                
                # 中心化
                sf_mean = sf.mean(dim=0, keepdim=True)
                sf_centered = sf - sf_mean
                
                # SVD分解（PyTorch GPU版本，需要FP32）
                U, S, V = torch.svd(sf_centered)
                
                # 取前pca_components个主成分
                pca_f = U[:, :self.pca_components] @ torch.diag(S[:self.pca_components])
                
                # 归一化到[0, 1]
                pca_f = (pca_f - pca_f.min()) / (pca_f.max() - pca_f.min() + 1e-8)
                
                # 转回原始精度
                pca_f = pca_f.to(original_dtype)
                features_list.append(pca_f)
            
            pca_features = torch.stack(features_list, dim=0)  # [B, num_patches, pca_components]
            heatmap = pca_features.reshape(B, h, w, self.pca_components).permute(0, 3, 1, 2)  # [B, pca_components, h, w]
            return heatmap
        else:
            # 不做PCA (针对P3阶段，保留全部通道)
            features = spatial_features.permute(0, 2, 1).reshape(B, self.embed_dim, h, w)  # [B, embed_dim, h, w]
            return features


class DINOYOLOFusion(nn.Module):
    """DINO和YOLO特征融合模块"""
    def __init__(self, dino_dim, yolo_dim, out_dim, fusion_type='concat'):
        super().__init__()
        self.fusion_type = fusion_type
        self.out_channels = out_dim  
        self.out_dim = out_dim // 2
        
        if fusion_type == 'concat':
            self.dino_proj = nn.Sequential(
                nn.Conv2d(dino_dim, self.out_dim, 1, bias=False),
                nn.BatchNorm2d(self.out_dim),
                nn.SiLU(inplace=True)
            )
            self.yolo_proj = nn.Sequential(
                nn.Conv2d(yolo_dim, self.out_dim, 1, bias=False),
                nn.BatchNorm2d(self.out_dim),
                nn.SiLU(inplace=True)
            )
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(self.out_dim * 2, out_dim, 1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.SiLU(inplace=True)
            )
        elif fusion_type == 'add':
            self.dino_proj = nn.Conv2d(dino_dim, out_dim, 1)
            self.yolo_proj = nn.Conv2d(yolo_dim, out_dim, 1)
        
    def forward(self, x):
        """
        Args:
            x: list of [dino_feat, yolo_feat] 或单独的特征
        """
        # 如果输入是列表，解包
        if isinstance(x, list):
            dino_feat, yolo_feat = x
        else:
            raise ValueError("DINOYOLOFusion expects a list of [dino_feat, yolo_feat]")
        
        # 投影到相同通道数
        dino_feat = self.dino_proj(dino_feat)
        yolo_feat = self.yolo_proj(yolo_feat)
        
        # 调整空间尺寸匹配
        if dino_feat.shape[2:] != yolo_feat.shape[2:]:
            dino_feat = F.interpolate(
                dino_feat, 
                size=yolo_feat.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        if self.fusion_type == 'concat':
            fused = torch.cat([dino_feat, yolo_feat], dim=1)
            return self.fusion_conv(fused)
        elif self.fusion_type == 'add':
            return dino_feat + yolo_feat