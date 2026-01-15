"""
DINO-YOLO 融合模块：适配灰度图像的双注入架构
- DINOInputAdapter: P0 层预处理注入，增强输入图像的语义信息
- DINOMidAdapter: P3 层中间特征注入，提升特征提取质量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOBase(nn.Module):
    """基础 DINO 加载器：负责加载权重、冻结参数、解决尺寸不匹配"""
    _dino_model = None

    def __init__(self, model_name='dinov2_vits14'):  # 用 vits14 更快，vitb14 更准
        super().__init__()
        if DINOBase._dino_model is None:
            print(f"🏗️ [DINO] Loading {model_name} (Frozen)...")
            # 自动下载并加载 DINOv2
            DINOBase._dino_model = torch.hub.load('facebookresearch/dinov2', model_name).cuda()
            # 冻结所有参数（我们只用它提取特征，不训练它）
            for p in DINOBase._dino_model.parameters():
                p.requires_grad = False
            DINOBase._dino_model.eval()
        
        self.dino = DINOBase._dino_model
        # ViT-S=384, ViT-B=768
        self.embed_dim = 384 if 'vits' in model_name else 768 

    def extract_feat(self, x):
        """
        提取 DINO 特征，自动处理灰度图和尺寸对齐
        Args:
            x: (B, C, H, W) - 输入特征图，可以是 1 或 3 通道
        Returns:
            out: (B, embed_dim, h_patch, w_patch) - DINO 特征图
        """
        B, C, H, W = x.shape
        
        # 1. 灰度图适配：如果是 1 通道，复制成 3 通道喂给 DINO
        if C == 1:
            x_in = x.repeat(1, 3, 1, 1)
        else:
            x_in = x

        # 2. 尺寸适配：DINO 需要 H, W 是 14 的倍数
        patch_size = 14
        h_new = (H // patch_size) * patch_size
        w_new = (W // patch_size) * patch_size
        
        # 如果尺寸不对，临时缩放一下喂给 DINO
        if h_new != H or w_new != W:
            x_in = F.interpolate(x_in, size=(h_new, w_new), mode='bilinear', align_corners=False)
            
        with torch.no_grad():
            # 获取 Patch Tokens
            out = self.dino.forward_features(x_in)["x_norm_patchtokens"]
            
        # 3. Reshape 回特征图格式 (B, embed_dim, h_patch, w_patch)
        out = out.permute(0, 2, 1).reshape(B, self.embed_dim, h_new // patch_size, w_new // patch_size)
        return out


class DINOInputAdapter(DINOBase):
    """
    P0 层注入：预处理增强
    输入：灰度原图 (B, 1, H, W)
    过程：DINO 提取语义特征 -> 投影 -> 融合
    输出：增强后的伪彩色图 (B, 3, H, W) -> 给 YOLO Backbone 吃
    """
    def __init__(self, c1, c2):  # c1=1 (灰度), c2=3 (YOLO第一层通常需要3)
        super().__init__()
        self.projector = nn.Sequential(
            nn.Conv2d(self.embed_dim, c2, kernel_size=1),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )
        # 如果输入本来就是3通道，这里要改一下适配
        self.input_proj = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()

    def forward(self, x):
        # 1. DINO 提取特征
        dino_feat = self.extract_feat(x)  # (B, 384, H/14, W/14)
        
        # 2. 恢复到原图尺寸
        dino_feat = F.interpolate(dino_feat, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # 3. 投影为 3 通道
        semantic_map = self.projector(dino_feat)  # (B, 3, H, W)
        
        # 4. 融合：原图信息 + DINO 语义信息
        # 即使原图是灰度，这里也输出 3 通道，相当于给灰度图"上色"，标记出重点区域
        return self.input_proj(x) + semantic_map


class DINOMidAdapter(DINOBase):
    """
    P3 层注入：中层特征融合
    输入：YOLO P3 特征 (B, C_in, H/8, W/8)
    过程：特征转图 -> DINO -> 门控融合
    输出：增强后的 P3 特征
    """
    def __init__(self, c1, c2):
        super().__init__()
        # 把 YOLO 特征图伪装成 3 通道图像喂给 DINO
        self.feat_to_img = nn.Conv2d(c1, 3, 1)
        
        # 融合门控 (可学习参数，初始为 0，防止破坏原有特征)
        self.gamma = nn.Parameter(torch.zeros(1)) 
        
        # 把 DINO 特征投影回 YOLO 通道数
        self.back_proj = nn.Sequential(
            nn.Conv2d(self.embed_dim, c2, 1),
            nn.BatchNorm2d(c2)
        )

    def forward(self, x):
        # x 是 YOLO 的中间特征 (例如 128 或 256 通道)
        
        # 1. 伪装成图片 (B, 3, H', W')
        x_fake_img = self.feat_to_img(x)
        
        # 2. DINO 提取
        dino_out = self.extract_feat(x_fake_img)
        
        # 3. 对齐尺寸 (防止 DINO patch 导致的细微尺寸差异)
        dino_out = F.interpolate(dino_out, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # 4. 投影回 YOLO 通道
        feat_refined = self.back_proj(dino_out)
        
        # 5. 门控残差连接：Original + alpha * DINO
        return x + self.gamma * feat_refined
