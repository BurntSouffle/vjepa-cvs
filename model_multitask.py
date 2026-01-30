"""
Multi-task V-JEPA Model for CVS Classification + Segmentation.

Architecture:
    V-JEPA Encoder (last 2 layers unfrozen)
        -> CVS Head (attention pooling + MLP)
        -> Seg Head (lightweight decoder)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


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


class LightweightSegDecoder(nn.Module):
    """
    Lightweight segmentation decoder for V-JEPA features.

    Takes per-frame features and upsamples to segmentation masks.
    Designed to be efficient while leveraging V-JEPA's rich representations.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        num_classes: int = 5,
        output_size: int = 64,
        dropout: float = 0.1,
    ):
        """
        Args:
            hidden_dim: V-JEPA hidden dimension (1024 for ViT-L)
            num_classes: Number of segmentation classes
            output_size: Output mask resolution
            dropout: Dropout rate
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.output_size = output_size

        # V-JEPA ViT-L with 256x256 input has 16x16 spatial patches
        # With 16 frames, we get 8 temporal bins (2 frames per bin)
        # Total tokens = 8 * 256 = 2048 (or similar)
        # Spatial dim = 16x16 = 256 tokens per temporal bin

        # Feature projection
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Lightweight convolutional upsampler
        # From 16x16 -> 64x64 (4x upscale)
        self.decoder = nn.Sequential(
            # 16x16 -> 32x32
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            # 32x32 -> 64x64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            # Final conv
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
        """
        Decode features for specific frames.

        Args:
            features: (B, num_tokens, hidden_dim) - V-JEPA output
            frame_indices: (N,) - which frames to decode (0-15)
            batch_indices: (N,) - which batch items
            num_temporal_bins: Number of temporal bins in V-JEPA output
            spatial_size: Spatial dimension (16 for 256x256 input)

        Returns:
            logits: (N, num_classes, H, W) segmentation logits
        """
        if len(frame_indices) == 0:
            return torch.zeros(0, self.num_classes, self.output_size, self.output_size,
                             device=features.device)

        B, num_tokens, D = features.shape
        spatial_tokens = spatial_size * spatial_size  # 256

        # Map frame indices (0-15) to temporal bin indices (0-7)
        # Assuming 2 frames per temporal bin
        temporal_indices = frame_indices // 2  # Map 16 frames to 8 bins

        # Extract features for each requested frame
        frame_features = []
        for i, (batch_idx, temp_idx) in enumerate(zip(batch_indices, temporal_indices)):
            # Get spatial tokens for this temporal bin
            start_idx = temp_idx * spatial_tokens
            end_idx = start_idx + spatial_tokens

            # Handle edge case where indices might be out of bounds
            if end_idx > num_tokens:
                end_idx = num_tokens
                start_idx = max(0, end_idx - spatial_tokens)

            spatial_feats = features[batch_idx, start_idx:end_idx]  # (spatial_tokens, D)

            # Ensure we have the right number of tokens
            if spatial_feats.shape[0] < spatial_tokens:
                # Pad if needed
                padding = torch.zeros(spatial_tokens - spatial_feats.shape[0], D,
                                     device=features.device)
                spatial_feats = torch.cat([spatial_feats, padding], dim=0)

            frame_features.append(spatial_feats)

        # Stack: (N, spatial_tokens, D)
        frame_features = torch.stack(frame_features, dim=0)

        # Project features
        frame_features = self.proj(frame_features)  # (N, spatial_tokens, 256)

        # Reshape to spatial grid: (N, 256, 16, 16)
        N = frame_features.shape[0]
        frame_features = frame_features.view(N, spatial_size, spatial_size, -1)
        frame_features = frame_features.permute(0, 3, 1, 2).contiguous()

        # Decode to segmentation logits
        logits = self.decoder(frame_features)  # (N, num_classes, 64, 64)

        return logits


class VJEPA_MultiTask(nn.Module):
    """
    Multi-task V-JEPA model for CVS classification and segmentation.

    Architecture:
        V-JEPA Encoder (last N layers unfrozen)
            |
            +-> Attention Pooling -> CVS Head -> 3 logits
            |
            +-> Seg Decoder -> per-frame segmentation masks
    """

    def __init__(
        self,
        model_name: str = "facebook/vjepa2-vitl-fpc16-256-ssv2",
        unfreeze_last_n_layers: int = 2,
        hidden_dim: int = 1024,
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

        self.model_name = model_name
        self.unfreeze_last_n_layers = unfreeze_last_n_layers
        self.hidden_dim = hidden_dim

        # Load V-JEPA backbone
        print(f"Loading V-JEPA model: {model_name}")
        self.backbone = AutoModel.from_pretrained(model_name)

        # Setup partial unfreezing
        self._setup_backbone_freezing(unfreeze_last_n_layers)

        # CVS classification head
        self.pooler = AttentionPooling(
            hidden_dim=hidden_dim,
            num_heads=attention_heads,
            dropout=attention_dropout,
        )

        self.cvs_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, cvs_hidden),
            nn.GELU(),
            nn.Dropout(cvs_dropout),
            nn.Linear(cvs_hidden, 3),  # C1, C2, C3
        )

        # Segmentation head
        self.seg_head = LightweightSegDecoder(
            hidden_dim=hidden_dim,
            num_classes=num_seg_classes,
            output_size=seg_output_size,
            dropout=seg_dropout,
        )

        self._init_heads()

    def _init_heads(self):
        """Initialize classification head weights."""
        for module in self.cvs_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _setup_backbone_freezing(self, unfreeze_last_n_layers: int):
        """Setup partial unfreezing of backbone."""
        # First freeze all
        for param in self.backbone.parameters():
            param.requires_grad = False

        total_backbone_params = sum(p.numel() for p in self.backbone.parameters())

        if unfreeze_last_n_layers > 0:
            total_layers = 24  # ViT-L
            start_unfreeze = total_layers - unfreeze_last_n_layers

            print(f"Unfreezing last {unfreeze_last_n_layers} transformer layers")

            unfrozen_params = 0
            for name, param in self.backbone.named_parameters():
                should_unfreeze = False

                for pattern in ['encoder.layer.', 'encoder.layers.', 'blocks.', 'layer.']:
                    if pattern in name:
                        try:
                            after_pattern = name.split(pattern)[1]
                            layer_num = int(after_pattern.split('.')[0])
                            if layer_num >= start_unfreeze:
                                should_unfreeze = True
                                break
                        except (ValueError, IndexError):
                            continue

                if 'encoder.norm' in name:
                    should_unfreeze = True

                if should_unfreeze:
                    param.requires_grad = True
                    unfrozen_params += param.numel()

            # Convert to float32 for stable fine-tuning
            self.backbone = self.backbone.float()
            print(f"Converted backbone to float32")
            print(f"Unfrozen params: {unfrozen_params / 1e6:.1f}M / {total_backbone_params / 1e6:.1f}M")
        else:
            print("Backbone fully frozen")

    def forward(
        self,
        pixel_values_videos: torch.Tensor,
        mask_frame_indices: torch.Tensor = None,
        mask_batch_indices: torch.Tensor = None,
    ) -> dict:
        """
        Forward pass for both tasks.

        Args:
            pixel_values_videos: (B, T, C, H, W) processed video tensor
            mask_frame_indices: (N,) frame indices for segmentation (optional)
            mask_batch_indices: (N,) batch indices for segmentation (optional)

        Returns:
            dict with:
                - cvs_logits: (B, 3) CVS classification logits
                - seg_logits: (N, num_classes, H, W) segmentation logits (if indices provided)
        """
        # Get V-JEPA features
        features = self.backbone.get_vision_features(
            pixel_values_videos=pixel_values_videos
        )  # (B, num_tokens, hidden_dim)

        # Convert to float32 for heads
        features = features.float()

        # CVS classification
        pooled = self.pooler(features)
        cvs_logits = self.cvs_head(pooled)

        result = {"cvs_logits": cvs_logits}

        # Segmentation (only if indices provided)
        if mask_frame_indices is not None and mask_batch_indices is not None and len(mask_frame_indices) > 0:
            seg_logits = self.seg_head(
                features,
                mask_frame_indices,
                mask_batch_indices,
            )
            result["seg_logits"] = seg_logits

        return result

    def process_videos(self, videos, device: torch.device) -> torch.Tensor:
        """Process videos for V-JEPA input (same as base model)."""
        import numpy as np

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


def create_multitask_model(config: dict) -> VJEPA_MultiTask:
    """Create multi-task model from config."""
    model_cfg = config["model"]

    model = VJEPA_MultiTask(
        model_name=model_cfg["name"],
        unfreeze_last_n_layers=model_cfg.get("unfreeze_last_n_layers", 2),
        hidden_dim=model_cfg.get("hidden_dim", 1024),
        cvs_hidden=model_cfg.get("cvs_hidden", 512),
        cvs_dropout=model_cfg.get("cvs_dropout", 0.5),
        attention_heads=model_cfg.get("attention_heads", 8),
        attention_dropout=model_cfg.get("attention_dropout", 0.1),
        num_seg_classes=model_cfg.get("num_seg_classes", 5),
        seg_output_size=model_cfg.get("seg_output_size", 64),
        seg_dropout=model_cfg.get("seg_dropout", 0.1),
    )

    return model


if __name__ == "__main__":
    print("=" * 60)
    print("Testing VJEPA_MultiTask Model")
    print("=" * 60)

    # Create model
    model = VJEPA_MultiTask(
        unfreeze_last_n_layers=2,
        num_seg_classes=5,
    )

    print(f"\nTotal params: {model.get_num_total_params() / 1e6:.1f}M")
    print(f"Trainable params: {model.get_num_trainable_params() / 1e6:.1f}M")

    # Test forward pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Dummy input
    B, T, C, H, W = 2, 16, 3, 256, 256
    videos = torch.randn(B, T, C, H, W, device=device)

    # Without segmentation
    print("\nTesting CVS-only forward:")
    with torch.no_grad():
        out = model(videos)
    print(f"  CVS logits: {out['cvs_logits'].shape}")

    # With segmentation
    print("\nTesting multi-task forward:")
    frame_indices = torch.tensor([0, 5, 10, 2, 8], device=device)  # 5 frames
    batch_indices = torch.tensor([0, 0, 0, 1, 1], device=device)    # from batches 0 and 1

    with torch.no_grad():
        out = model(videos, frame_indices, batch_indices)
    print(f"  CVS logits: {out['cvs_logits'].shape}")
    print(f"  Seg logits: {out['seg_logits'].shape}")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
