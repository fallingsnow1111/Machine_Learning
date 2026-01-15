import torch
from torch import nn
import torch.nn.functional as F
from modelscope import AutoImageProcessor, AutoModel
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
        self.processor = AutoImageProcessor.from_pretrained(self.pretrained_model_name, size={"height": 1024, "width": 1024})
        self.dino = AutoModel.from_pretrained(self.pretrained_model_name, device_map="auto")
        self.patch_size = self.dino.config.patch_size  # 通常是16
        self.embed_dim = self.dino.config.hidden_size  # large:1024
        self.pca_components = pca_components
        
        if freeze:
            for param in self.dino.parameters():
                param.requires_grad = False
            self.dino.eval()
        
    @torch.no_grad()
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
            # 返回假的特征图用于初始化 stride 计算
            # processor 默认将图像 resize 到 1024x1024，patch_size=16
            # 所以输出空间尺寸是 1024/16 = 64
            h_out = w_out = 64
            if self.pca_components:
                return torch.zeros(B, self.pca_components, h_out, w_out, device=device)
            else:
                return torch.zeros(B, self.embed_dim, h_out, w_out, device=device)
        
        # 将 tensor 转换为 numpy array (processor 期望的格式)
        # Clip 到 [0, 1] 并转为 [0, 255] uint8
        x_clamped = torch.clamp(x, 0, 1)
        x_np = (x_clamped.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        images_list = [x_np[i] for i in range(B)]
        
        inputs = self.processor(images=images_list, return_tensors="pt").to(next(self.dino.parameters()).device)
        outputs = self.dino(**inputs, output_hidden_states=True)
        
        # 获取最后一层的hidden states
        last_hidden_state = outputs.hidden_states[-1]  # [B, num_tokens, embed_dim]
        
        # 去掉[cls]和registers
        num_registers = 4 
        spatial_features = last_hidden_state[:, 1 + num_registers:, :]  # [B, num_patches, embed_dim]
        _, num_patches, _ = spatial_features.shape
        h = w = int(np.sqrt(num_patches))
        assert h * w == num_patches, "Number of patches must form a square grid"
        
        if self.pca_components:
            # PCA 降维 (针对P0阶段，只保留3个通道)
            features_list = []
            for i in range(B):
                sf = spatial_features[i].cpu().numpy()
                pca = PCA(n_components=self.pca_components)
                pca_f = pca.fit_transform(sf)
                pca_f = (pca_f - pca_f.min()) / (pca_f.max() - pca_f.min() + 1e-8)
                features_list.append(pca_f)
            pca_features = np.stack(features_list, axis=0)  # [B, num_patches, pca_components]
            heatmap = torch.from_numpy(pca_features.reshape(B, h, w, self.pca_components)).to(device).float().permute(0, 3, 1, 2)  # [B, pca_components, h, w]
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