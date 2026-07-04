import torch
from torch import nn
import math

class VisionTransformerBlock(nn.Module):
    """标准 Transformer 块，使用 Pre-LN 结构"""
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: (B, L, D)
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

class VisionTransformer(nn.Module):
    """基于 Transformer 的历史帧编码器，输出序列特征"""
    def __init__(self,
                 img_size=384,
                 patch_size=32,
                 in_chans=3,
                 embed_dim=768,
                 depth=4,
                 nhead=8,
                 mlp_ratio=4.0,
                 dropout=0.0,
                 dtype=torch.float32):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches_per_frame = (img_size // patch_size) ** 2

        # Patch embedding (共享于每一帧)
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size, stride=patch_size,
            dtype=dtype
        )

        # 可学习的位置编码：空间位置 + 时序位置
        # 为简单起见，直接为总序列长度 (T * num_patches_per_frame) 分配一个位置编码
        # 实际使用时需要知道最大帧数，这里先预留一个较大长度
        max_frames = 8  # 可根据需要调整
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_frames * self.num_patches_per_frame, embed_dim, dtype=dtype)
        )

        # Transformer 编码层
        self.blocks = nn.ModuleList([
            VisionTransformerBlock(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout
            ) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        x: (B, T, C, H, W) 或 (B, C, H, W) 兼容单帧
        返回: (B, total_patches, embed_dim)
        """
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            # 逐帧 patch embedding
            frame_tokens = []
            for t in range(T):
                frame = x[:, t, :, :, :]          # (B, C, H, W)
                patches = self.patch_embed(frame)  # (B, D, Hp, Wp)
                patches = patches.flatten(2).transpose(1, 2)  # (B, N, D)
                frame_tokens.append(patches)
            x = torch.cat(frame_tokens, dim=1)    # (B, T*N, D)
            # 添加位置编码（如果序列长度超过预设则截断或插值）
            if x.size(1) > self.pos_embed.size(1):
                # 简单重复或插值，实际可学习位置编码通常固定长度，这里仅作示意
                pos = self.pos_embed[:, :x.size(1), :]
            else:
                pos = self.pos_embed[:, :x.size(1), :]
            x = x + pos
        else:
            # 单帧处理
            x = self.patch_embed(x)
            x = x.flatten(2).transpose(1, 2)
            x = x + self.pos_embed[:, :x.size(1), :]

        # 通过 Transformer 块
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x

    def reset_state(self):
        # Transformer 无状态，提供空方法以兼容原接口
        pass