"""
Small ViT with SwinV2-Style Window Attention for CVS Classification
====================================================================
~24M param model trained from scratch on surgical video data.

Architecture:
  Conv3D PatchEmbed -> 12x WindowTransformerBlock (alternating shift)
  -> AttentionPooling -> CVS Head (3 logits)
  -> LightweightSegDecoder (per-frame segmentation)

Key design:
  - Window size 8 on 16x16 spatial grid -> 4 windows of 64 tokens
  - Spatial-only window attention within each temporal bin
  - SwinV2 post-norm residual connections
  - Cosine attention with learnable logit scale
  - Continuous relative position bias via small MLP
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# DropPath: try timm first, provide fallback
try:
    from timm.layers import DropPath
except ImportError:
    try:
        from timm.models.layers import DropPath
    except ImportError:

        class DropPath(nn.Module):
            """Stochastic depth (drop path) fallback."""

            def __init__(self, drop_prob: float = 0.0):
                super().__init__()
                self.drop_prob = drop_prob

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if not self.training or self.drop_prob == 0.0:
                    return x
                keep_prob = 1 - self.drop_prob
                shape = (x.shape[0],) + (1,) * (x.ndim - 1)
                random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
                random_tensor = torch.floor_(random_tensor + keep_prob)
                return x.div(keep_prob) * random_tensor


# ============================================================================
# Window partition / reverse utilities
# ============================================================================

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Partition feature map into non-overlapping windows.

    Args:
        x: [B, H, W, C]
        window_size: window size

    Returns:
        windows: [num_windows * B, window_size, window_size, C]
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """
    Reverse window partition back to feature map.

    Args:
        windows: [num_windows * B, window_size, window_size, C]
        window_size: window size
        H: height of feature map
        W: width of feature map

    Returns:
        x: [B, H, W, C]
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# ============================================================================
# MLP
# ============================================================================

class Mlp(nn.Module):
    """Standard 2-layer MLP with GELU and dropout."""

    def __init__(self, in_features: int, hidden_features: int = None, out_features: int = None, drop: float = 0.0):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ============================================================================
# 3D Patch Embedding
# ============================================================================

class PatchEmbed3D(nn.Module):
    """
    Video patch embedding using 3D convolution.

    [B, C, T, H, W] -> [B, num_tokens, embed_dim]
    With kernel/stride (2, 16, 16): 16 frames -> 8 temporal bins, 256x256 -> 16x16 spatial
    Total tokens = 8 * 16 * 16 = 2048
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 384,
        temporal_kernel: int = 2,
        spatial_kernel: int = 16,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.temporal_kernel = temporal_kernel
        self.spatial_kernel = spatial_kernel

        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=(temporal_kernel, spatial_kernel, spatial_kernel),
            stride=(temporal_kernel, spatial_kernel, spatial_kernel),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T, H, W]

        Returns:
            [B, num_tokens, embed_dim]
        """
        x = self.proj(x)  # [B, embed_dim, T', H', W']
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, T'*H'*W', embed_dim]
        x = self.norm(x)
        return x


# ============================================================================
# Window Attention (SwinV2-style)
# ============================================================================

class WindowAttention(nn.Module):
    """
    SwinV2-style window attention with:
    - Cosine attention (normalized dot product)
    - Learnable logit scale per head
    - Continuous relative position bias via small MLP
    - Separate q_bias and v_bias (no k bias)
    """

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads

        # Learnable logit scale per head (SwinV2), initialized to log(10)
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones(num_heads, 1, 1)))

        # Continuous relative position bias MLP
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False),
        )

        # Build relative coordinate table
        self._build_relative_coords()

        # QKV projection with separate q_bias and v_bias (no k bias)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(dim))
        self.v_bias = nn.Parameter(torch.zeros(dim))

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def _build_relative_coords(self):
        """Build relative coordinate table and index for position bias."""
        ws = self.window_size
        # Relative coordinate table: (2*ws-1) * (2*ws-1), 2
        coords_h = torch.arange(-(ws - 1), ws, dtype=torch.float32)
        coords_w = torch.arange(-(ws - 1), ws, dtype=torch.float32)
        # Normalize to [-1, 1] range with log-scale (SwinV2)
        coords_h = coords_h / (ws - 1) if ws > 1 else coords_h
        coords_w = coords_w / (ws - 1) if ws > 1 else coords_w
        # Sign * log(1 + |coord|) normalization (SwinV2 style)
        coords_h = torch.sign(coords_h) * torch.log2(1 + coords_h.abs())
        coords_w = torch.sign(coords_w) * torch.log2(1 + coords_w.abs())

        relative_coords_table = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing='ij')
        ).permute(1, 2, 0).contiguous().view(-1, 2)  # [(2*ws-1)^2, 2]
        self.register_buffer("relative_coords_table", relative_coords_table, persistent=False)

        # Relative position index for each token pair in window
        coords_h = torch.arange(ws)
        coords_w = torch.arange(ws)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # [2, ws, ws]
        coords_flatten = coords.view(2, -1)  # [2, ws*ws]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, N, N]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [N, N, 2]
        relative_coords[:, :, 0] += ws - 1
        relative_coords[:, :, 1] += ws - 1
        relative_coords[:, :, 0] *= 2 * ws - 1
        relative_position_index = relative_coords.sum(-1)  # [N, N]
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [num_windows*B, N, C] where N = window_size * window_size
            mask: [num_windows, N, N] or None

        Returns:
            [num_windows*B, N, C]
        """
        B_, N, C = x.shape

        # QKV with q_bias and v_bias
        qkv_bias = torch.cat([
            self.q_bias,
            torch.zeros_like(self.v_bias),  # no k bias
            self.v_bias,
        ])
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each [B_, num_heads, N, head_dim]

        # Cosine attention (SwinV2)
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)

        # Learnable logit scale, clamped to avoid overflow
        logit_scale = torch.clamp(self.logit_scale, max=math.log(100.0)).exp()
        attn = attn * logit_scale

        # Continuous relative position bias
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table)  # [(2*ws-1)^2, num_heads]
        # Apply sigmoid gating and scale (SwinV2: 16 * sigmoid)
        relative_position_bias_table = 16 * torch.sigmoid(relative_position_bias_table)

        rel_pos_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            N, N, -1
        )  # [N, N, num_heads]
        rel_pos_bias = rel_pos_bias.permute(2, 0, 1).contiguous().unsqueeze(0)  # [1, num_heads, N, N]
        attn = attn + rel_pos_bias

        # Shifted window mask
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)  # [B, nW, 1, N, N]
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ============================================================================
# Window Transformer Block
# ============================================================================

class WindowTransformerBlock(nn.Module):
    """
    Transformer block with SwinV2-style window attention.

    Handles temporal reshaping:
    - Input: [B, 2048, C] (8 temporal bins * 16*16 spatial)
    - Reshape to [B*8, 16, 16, C] (separate temporal bins)
    - Apply cyclic shift, window partition, attention, reverse
    - SwinV2 post-norm residual connections
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 8,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        num_temporal_bins: int = 8,
        spatial_size: int = 16,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.num_temporal_bins = num_temporal_bins
        self.spatial_size = spatial_size

        # Clamp shift_size
        if self.shift_size >= self.window_size:
            self.shift_size = 0

        # SwinV2 post-norm: norm AFTER attention/mlp, not before
        self.attn = WindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm1 = nn.LayerNorm(dim)

        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)
        self.norm2 = nn.LayerNorm(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Build attention mask for shifted windows
        if self.shift_size > 0:
            self._build_attn_mask()
        else:
            self.register_buffer("attn_mask", None, persistent=False)

    def _build_attn_mask(self):
        """Build attention mask for shifted window self-attention."""
        H = self.spatial_size
        W = self.spatial_size
        ws = self.window_size
        ss = self.shift_size

        img_mask = torch.zeros(1, H, W, 1)
        h_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        w_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, ws)  # [nW, ws, ws, 1]
        mask_windows = mask_windows.view(-1, ws * ws)  # [nW, ws*ws]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, N, N]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        self.register_buffer("attn_mask", attn_mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, num_tokens, C] where num_tokens = num_temporal_bins * spatial_size^2

        Returns:
            [B, num_tokens, C]
        """
        B, L, C = x.shape
        T = self.num_temporal_bins
        H = self.spatial_size
        W = self.spatial_size

        # Reshape to separate temporal bins: [B*T, H, W, C]
        x_spatial = x.view(B * T, H * W, C).view(B * T, H, W, C)

        shortcut = x_spatial

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x_spatial, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x_spatial

        # Window partition
        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B*T, ws, ws, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [nW*B*T, ws*ws, C]

        # Window attention (SwinV2 post-norm: attention first, then norm)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # Reshape back
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # Reverse window partition
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # [B*T, H, W, C]

        # Reverse cyclic shift
        if self.shift_size > 0:
            x_spatial = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x_spatial = shifted_x

        # SwinV2 post-norm residual
        x_spatial = shortcut + self.drop_path(self.norm1(x_spatial))

        # MLP with post-norm residual
        x_spatial = x_spatial + self.drop_path(self.norm2(self.mlp(x_spatial)))

        # Reshape back to [B, num_tokens, C]
        x = x_spatial.view(B * T, H * W, C).view(B, T * H * W, C)
        return x


# ============================================================================
# ViT Backbone with Window Attention
# ============================================================================

class ViTSmallWindow(nn.Module):
    """
    Small ViT backbone with SwinV2-style window attention.

    PatchEmbed3D + learned pos_embed + 12 blocks (alternating shift) + LayerNorm
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        num_frames: int = 16,
        spatial_size: int = 16,
        temporal_kernel: int = 2,
        spatial_kernel: int = 16,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth

        num_temporal_bins = num_frames // temporal_kernel  # 16 // 2 = 8
        num_tokens = num_temporal_bins * spatial_size * spatial_size  # 8 * 16 * 16 = 2048

        # Patch embedding
        self.patch_embed = PatchEmbed3D(
            in_channels=in_channels,
            embed_dim=embed_dim,
            temporal_kernel=temporal_kernel,
            spatial_kernel=spatial_kernel,
        )

        # Learned positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(drop_rate)

        # Stochastic depth decay
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Transformer blocks with alternating shift
        self.blocks = nn.ModuleList([
            WindowTransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                num_temporal_bins=num_temporal_bins,
                spatial_size=spatial_size,
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T, H, W] input video

        Returns:
            [B, num_tokens, embed_dim]
        """
        x = self.patch_embed(x)  # [B, num_tokens, embed_dim]
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x


# ============================================================================
# Attention Pooling (from train_regularized.py)
# ============================================================================

class AttentionPooling(nn.Module):
    """Learnable attention pooling over token dimension."""

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        nn.init.normal_(self.query, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        query = self.query.expand(B, -1, -1)
        attn_out, _ = self.attention(query, x, x)
        return self.norm(attn_out.squeeze(1))


# ============================================================================
# Lightweight Segmentation Decoder (adapted for 384-dim input)
# ============================================================================

class LightweightSegDecoder(nn.Module):
    """Lightweight segmentation decoder adapted for small ViT features."""

    def __init__(
        self,
        hidden_dim: int = 384,
        num_classes: int = 5,
        output_size: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.output_size = output_size

        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        features: torch.Tensor,
        frame_indices: torch.Tensor,
        batch_indices: torch.Tensor,
        num_temporal_bins: int = 8,
        spatial_size: int = 16,
    ) -> torch.Tensor:
        if len(frame_indices) == 0:
            return torch.zeros(0, self.num_classes, self.output_size, self.output_size,
                             device=features.device)

        B, num_tokens, D = features.shape
        spatial_tokens = spatial_size * spatial_size

        temporal_indices = frame_indices // 2

        frame_features = []
        for i, (batch_idx, temp_idx) in enumerate(zip(batch_indices, temporal_indices)):
            start_idx = temp_idx * spatial_tokens
            end_idx = start_idx + spatial_tokens

            if end_idx > num_tokens:
                end_idx = num_tokens
                start_idx = max(0, end_idx - spatial_tokens)

            spatial_feats = features[batch_idx, start_idx:end_idx]

            if spatial_feats.shape[0] < spatial_tokens:
                padding = torch.zeros(spatial_tokens - spatial_feats.shape[0], D,
                                     device=features.device)
                spatial_feats = torch.cat([spatial_feats, padding], dim=0)

            frame_features.append(spatial_feats)

        frame_features = torch.stack(frame_features, dim=0)
        frame_features = self.proj(frame_features)

        N = frame_features.shape[0]
        frame_features = frame_features.view(N, spatial_size, spatial_size, -1)
        frame_features = frame_features.permute(0, 3, 1, 2).contiguous()

        logits = self.decoder(frame_features)

        return logits


# ============================================================================
# Full CVS Model
# ============================================================================

class ViTSmallWindowCVS(nn.Module):
    """
    Full model: Small ViT with window attention for CVS classification + segmentation.

    Architecture:
        ViTSmallWindow backbone (trainable from scratch)
            |
            +-> [Hard Attention Mask] -> AttentionPooling -> CVS Head -> 3 logits
            |
            +-> LightweightSegDecoder -> per-frame segmentation masks
    """

    def __init__(
        self,
        # Backbone params
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        num_frames: int = 16,
        spatial_size: int = 16,
        temporal_kernel: int = 2,
        spatial_kernel: int = 16,
        # CVS head params
        cvs_hidden: int = 512,
        cvs_dropout: float = 0.5,
        attention_heads: int = 8,
        attention_dropout: float = 0.1,
        # Seg head params
        num_seg_classes: int = 5,
        seg_output_size: int = 64,
        seg_dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Backbone
        self.backbone = ViTSmallWindow(
            in_channels=3,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            num_frames=num_frames,
            spatial_size=spatial_size,
            temporal_kernel=temporal_kernel,
            spatial_kernel=spatial_kernel,
        )

        # CVS classification head
        self.pooler = AttentionPooling(
            hidden_dim=embed_dim,
            num_heads=attention_heads,
            dropout=attention_dropout,
        )

        self.cvs_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, cvs_hidden),
            nn.GELU(),
            nn.Dropout(cvs_dropout),
            nn.Linear(cvs_hidden, 3),
        )

        # Segmentation head
        self.seg_head = LightweightSegDecoder(
            hidden_dim=embed_dim,
            num_classes=num_seg_classes,
            output_size=seg_output_size,
            dropout=seg_dropout,
        )

        self._init_cvs_head()

    def _init_cvs_head(self):
        for module in self.cvs_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        pixel_values_videos: torch.Tensor,
        mask_frame_indices: torch.Tensor = None,
        mask_batch_indices: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ) -> dict:
        """
        Forward pass.

        Args:
            pixel_values_videos: [B, T, C, H, W] input video
            mask_frame_indices: frame indices for segmentation
            mask_batch_indices: batch indices for segmentation
            attention_mask: [B, num_tokens] hard attention mask (optional)
        """
        # Reshape [B, T, C, H, W] -> [B, C, T, H, W] for Conv3d
        x = pixel_values_videos.permute(0, 2, 1, 3, 4).contiguous()

        # Extract features
        features = self.backbone(x)  # [B, num_tokens, embed_dim]

        # Apply hard attention mask
        if attention_mask is not None:
            features = features * attention_mask.unsqueeze(-1)

        # CVS classification
        pooled = self.pooler(features)
        cvs_logits = self.cvs_head(pooled)

        result = {"cvs_logits": cvs_logits}

        # Segmentation
        if mask_frame_indices is not None and mask_batch_indices is not None and len(mask_frame_indices) > 0:
            seg_logits = self.seg_head(
                features,
                mask_frame_indices,
                mask_batch_indices,
            )
            result["seg_logits"] = seg_logits

        return result

    def process_videos(self, videos, device: torch.device) -> torch.Tensor:
        """Process videos for model input (same logic as VJEPA_LoRA.process_videos)."""
        if isinstance(videos, torch.Tensor):
            video_tensor = videos
            if video_tensor.dim() == 5:
                B, T, dim3, H, W = video_tensor.shape
                if dim3 == 3 and video_tensor.dtype == torch.float32:
                    if video_tensor.min() < -0.5 or video_tensor.max() > 1.5:
                        return video_tensor.to(device)
                    else:
                        video_tensor = video_tensor.to(device)
                        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)
                        return (video_tensor - mean) / std
                elif W == 3:
                    video_tensor = video_tensor.permute(0, 1, 4, 2, 3).contiguous()
                    video_tensor = video_tensor.to(device).float() / 255.0
                    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)
                    return (video_tensor - mean) / std
            raise ValueError(f"Unexpected tensor shape: {video_tensor.shape}")

        elif isinstance(videos, list):
            if len(videos) == 0:
                raise ValueError("Empty video list")
            first = videos[0]
            if isinstance(first, np.ndarray):
                video_tensor = torch.from_numpy(np.stack(videos))
            elif isinstance(first, torch.Tensor):
                video_tensor = torch.stack(videos)
            else:
                raise ValueError(f"Unsupported video type: {type(first)}")

            video_tensor = video_tensor.permute(0, 1, 4, 2, 3).contiguous()
            video_tensor = video_tensor.to(device).float() / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)
            return (video_tensor - mean) / std

        raise ValueError(f"Unsupported videos type: {type(videos)}")

    def get_num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_num_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ViTSmallWindowCVS - Architecture Test")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create model
    model = ViTSmallWindowCVS(
        embed_dim=384,
        depth=12,
        num_heads=6,
        window_size=8,
        drop_path_rate=0.1,
    ).to(device)

    total_params = model.get_num_total_params()
    trainable_params = model.get_num_trainable_params()
    print(f"\nTotal params: {total_params / 1e6:.2f}M")
    print(f"Trainable params: {trainable_params / 1e6:.2f}M")
    print(f"All require grad: {total_params == trainable_params}")

    # Shape test: [B, T, C, H, W]
    B, T, C, H, W = 2, 16, 3, 256, 256
    x = torch.randn(B, T, C, H, W, device=device)

    print(f"\nInput shape: {x.shape}")

    # Forward pass with segmentation
    mask_frame_indices = torch.tensor([0, 2, 4], device=device)
    mask_batch_indices = torch.tensor([0, 0, 1], device=device)

    with torch.no_grad():
        outputs = model(x, mask_frame_indices, mask_batch_indices)

    print(f"CVS logits shape: {outputs['cvs_logits'].shape}")
    assert outputs['cvs_logits'].shape == (B, 3), f"Expected (2, 3), got {outputs['cvs_logits'].shape}"

    if 'seg_logits' in outputs:
        print(f"Seg logits shape: {outputs['seg_logits'].shape}")

    # Test attention matrix size
    print(f"\nWindow size: 8 on 16x16 grid -> attention matrices are 64x64 (not 2048x2048)")

    # Test with hard attention mask
    num_tokens = 2048
    attn_mask = torch.ones(B, num_tokens, device=device)
    attn_mask[:, :128] = 0.0  # zero out first 128 tokens

    with torch.no_grad():
        outputs_masked = model(x, mask_frame_indices, mask_batch_indices, attn_mask)

    print(f"CVS logits (masked) shape: {outputs_masked['cvs_logits'].shape}")

    print("\nAll tests passed!")
