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
            # 自动下载并加载 DINOv2（先在 CPU 上加载）
            DINOBase._dino_model = torch.hub.load('facebookresearch/dinov2', model_name)
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
        device = x.device  # 获取输入设备
        
        # 确保 DINO 模型在正确的设备上
        if next(self.dino.parameters()).device != device:
            self.dino = self.dino.to(device)
        
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
    输入：灰度原图 (B, 1, H, W) 或 RGB (B, 3, H, W)
    过程：DINO 提取语义特征 -> 投影 -> 融合
    输出：增强后的伪彩色图 (B, 3, H, W) -> 给 YOLO Backbone 吃
    
    Args:
        c1: 输入通道数（YOLO 自动推断，通常是 1 或 3）
    注意：输出通道固定为 3 (RGB)，不受 width_multiple 影响
    """
    def __init__(self, c1):  # 只接收 c1，c2 固定为 3
        super().__init__()
        self.c1 = c1
        self.c2 = 3  # 固定输出 RGB
        
        self.projector = nn.Sequential(
            nn.Conv2d(self.embed_dim, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.SiLU()
        )
        # 如果输入本来就是3通道，这里要改一下适配
        self.input_proj = nn.Conv2d(c1, 3, 1) if c1 != 3 else nn.Identity()
        
        print(f"✅ [DINOInputAdapter] 初始化：输入通道={c1}, 输出通道=3 (固定)")

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
    
    **关键设计 - 符合 YOLO 参数契约**:
    - c1, c2 必须是前两个参数（YOLO 自动处理通道缩放）
    - c2 会自动应用 width_scale（如 256 * 0.25 = 64）
    - 动态创建：涉及输入通道数的层在 forward 首次调用时创建
    
    YAML 示例: [-1, 1, DINOMidAdapter, [256, 'dinov2_vits14', True]]
    解析结果: c1=128 (自动), c2=64 (256*0.25), model_name='dinov2_vits14', freeze=True
    """
    def __init__(self, c1, c2, model_name="dinov2_vits14", freeze=True):
        super().__init__(model_name)  # DINOBase 只接收 model_name
        self.c1 = c1  # 输入通道数（YOLO 自动传入）
        self.c2 = c2  # 输出通道数（已应用 width_scale）
        self.freeze = freeze
        
        # 延迟创建的层（首次 forward 时创建）
        self.feat_to_img = None
        self.dino_proj = None
        self.fusion_conv = None
        
        print(f"✅ [DINOMidAdapter] 初始化：c1={c1}, c2={c2}, model={model_name}, freeze={freeze}")
        print(f"   💡 投影层将在首次 forward 时动态创建")

    def _create_projection_layers(self, device):
        """首次调用时创建投影层，使用 self.c1 和 self.c2"""
        # 1. YOLO特征 -> 伪RGB图像 (用于DINO输入)
        self.feat_to_img = nn.Sequential(
            nn.Conv2d(self.c1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 1, 1),
            nn.Tanh()
        ).to(device)
        
        # 2. DINO特征 -> 目标通道数
        self.dino_proj = nn.Conv2d(
            self.embed_dim, self.c2, 1
        ).to(device)
        
        # 3. 融合层 (YOLO原始 + DINO增强 -> 输出)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(self.c1 + self.c2, self.c2, 3, 1, 1),
            nn.BatchNorm2d(self.c2),
            nn.ReLU(inplace=True)
        ).to(device)
        
        print(f"   🔧 [DINOMidAdapter] 动态创建层：{self.c1} -> {self.c2} (device={device})")

    def forward(self, x):
        """
        x: [B, c1, H, W] - YOLO的P3特征
        返回: [B, c2, H, W] - 融合后的特征
        
        流程:
        1. 首次调用：创建投影层
        2. YOLO特征 -> 伪RGB -> DINO -> 提取特征
        3. 融合 YOLO 原始特征和 DINO 增强特征
        """
        B, C_in, H, W = x.shape
        
        # 首次调用：创建所有投影层
        if self.feat_to_img is None:
            self._create_projection_layers(x.device)
        
        # 1. 将YOLO特征转换为伪RGB图像
        pseudo_img = self.feat_to_img(x)  # [B, 3, H, W]
        
        # 2. 提取DINO特征
        dino_feat = self.extract_feat(pseudo_img)  # [B, embed_dim, H', W']
        
        # 3. 调整DINO特征尺寸到与输入相同
        dino_feat = F.interpolate(dino_feat, size=(H, W), mode='bilinear', align_corners=False)
        
        # 4. 调整DINO特征通道数
        adapted_dino = self.dino_proj(dino_feat)  # [B, c2, H, W]
        
        # 5. 融合原始特征和DINO特征
        fused = torch.cat([x, adapted_dino], dim=1)  # [B, c1+c2, H, W]
        out = self.fusion_conv(fused)  # [B, c2, H, W]
        
        return out
