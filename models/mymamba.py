import torch
import torch.nn as nn
from mamba_ssm import Mamba

class ChannelAttention(nn.Module):
    """简单的通道注意力，增强光谱维度的特征表达"""
    def __init__(self, channel, reduction=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class EnhancedMambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm_h = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        
        # 1. 局部特征提取分支 (Depth-wise Conv 节省计算量)
        self.local_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1)
        )
        
        # 2. 水平方向 Mamba (处理左->右关系)
        self.mamba_h = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        
        # 3. 垂直方向 Mamba (处理上->下关系)
        self.mamba_v = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        
        # 4. 光谱/通道注意力
        self.ca = ChannelAttention(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # --- 分支 1：局部细节提取 ---
        local_feat = self.local_conv(x)
        
        # --- 分支 2：水平方向全局建模 (Horizontal) ---
        # 展平: [B, C, H, W] -> [B, H*W, C]
        x_h = x.flatten(2).transpose(1, 2)
        x_h = self.norm_h(x_h)
        out_h = self.mamba_h(x_h)
        out_h = out_h.transpose(1, 2).reshape(B, C, H, W)
        
        # --- 分支 3：垂直方向全局建模 (Vertical) ---
        # 翻转空间维度并展平: [B, C, H, W] -> [B, C, W, H] -> [B, W*H, C]
        x_v = x.transpose(2, 3).flatten(2).transpose(1, 2)
        x_v = self.norm_v(x_v)
        out_v = self.mamba_v(x_v)
        out_v = out_v.transpose(1, 2).reshape(B, C, W, H).transpose(2, 3)
        
        # --- 特征融合与光谱强化 ---
        # 融合: 水平全局 + 垂直全局 + 局部细节
        fused_feat = out_h + out_v + local_feat
        
        # 经过通道光谱注意力
        out = self.ca(fused_feat)
        
        return out + x # 最终残差连接

class MambaSSR(nn.Module):
    def __init__(self, in_channels=3, out_channels=31, dim=64, num_blocks=6):
        super(MambaSSR, self).__init__()
        
        # 浅层特征提取
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 深层特征提取 (堆叠 Enhanced Mamba Blocks)
        self.body = nn.Sequential(
            *[EnhancedMambaBlock(dim=dim) for _ in range(num_blocks)]
        )
        
        # 重建层 (增加一层提升映射能力)
        self.tail = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        feat = self.head(x)
        res = self.body(feat)
        out = self.tail(res + feat) # 全局残差
        return out

if __name__ == '__main__':
    # 简单的测试代码
    model = MambaSSR().cuda()
    x = torch.randn(1, 3, 64, 64).cuda()
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
